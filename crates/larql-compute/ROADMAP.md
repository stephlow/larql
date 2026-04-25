# Roadmap — larql-compute

## Current state (2026-04-25, M3 Max, real vindex)

| Engine | tok/s | ms/tok | Notes |
|---|---|---|---|
| **LARQL Metal** (gemma3-4b-q4k-v2, Q6_K down) | **68** | 14.7 | production extract; q6k_matvec 4-elem rewrite + min-heap top-k |
| **LARQL Metal** (gemma3-4b-q4k-downq4k, all-Q4_K) | **70.1** | 14.26 | all-Q4_K extract; q4k_geglu_silu_down fires |
| **Ollama** gemma3:4b | **100–105** | 9.5–10.0 | reference |
| **Gap** | LARQL is 1.48–1.53× slower | +5ms/tok | per-stage decomposition below |

Per-stage breakdown (larql-metal, gemma3-4b-q4k-v2, 100-token run):

| Stage | ms/tok | % |
|---|---|---|
| GPU fwd | 12.7 | 84.8% |
| lm_head | 2.3 | 15.1% |
| embed + norm + detok | ~0.01 | ~0% |

GPU fwd is 84% of decode time; FFN is ~87% of GPU fwd. The Q6_K down
projection (2560×10240 per layer × 34 layers) is the dominant kernel.

The "117 tok/s" historical number was synthetic-weight Q4_KF without
real vindex load. Production extracts use Q6_K down (Ollama
convention); the q4_KF fast-path doesn't apply to those.

---

## P0: Production gap closers (open)

These are the optimizations from the 2026-04-25 diagnostic — ranked
by leverage. Lands sequentially; #1 alone closes ~half the gap.

### #1 — Q6_K fused activation+down (closed — wrong fix, correct diagnosis)

**Status:** Benchmarked (2026-04-25). Not viable. Routing reverted.
Root cause of original regression identified and documented.

**What was tried:** Added threadgroup-memory caching of `gate`/`up`
per super-block so all 4 simdgroups in a TG share one device load
(128 threads × 2 values each). All 5 parity tests pass. But
`larql bench gemma3-4b-q4k-v2` showed 61–62 tok/s — identical to
the unfused-TG-cache attempt and identical to the regression without
TG caching. TG caching had zero effect.

**Root cause (corrected):** bandwidth was never the bottleneck.
gate/up = 80 KB total per dispatch — well within M3 Max GPU L2 cache.
All 640 TGs share the same gate/up data → L2 cache-hits from TG 2
onward. The real regression is GELU-tanh recomputation:

- Separated path: `geglu_gelu_tanh` kernel runs 10,240 threads,
  each computing one `tanh(gate[i])`. Total: 10,240 `tanh` calls.
- Fused path: inner loop computes `tanh(gate[i])` for every output
  row independently. At N=2560 output rows: 2,560 × 10,240 =
  **26.2 M `tanh` calls** — 2560× more than separated.

`tanh` is a transcendental function; GPU ALU cost dominates. The
saved dispatch + buffer round-trip (~0.2 ms) doesn't offset the
extra 2560× `tanh` work at production shape.

**Q4_K fusion wins for a different reason:** the all-Q4_K model
uses SiLU (`x/(1+exp(-x))`), not GELU-tanh. SiLU is cheaper than
`tanh`, so the recomputation overhead is smaller relative to the
heavier Q4_K dequant per cell.

**Remaining Q6_K opportunity:** optimise `q6k_matvec` throughput
directly (P0 #5 below) — currently 79 GE/s vs Q4_K 105 GE/s.
Alternatively: precompute `act[]` via a fast batch activation and
pass a float input to a future `q6k_matvec_f32in` kernel (avoids
the per-row `tanh` recomputation entirely while still fusing
dispatch). ~50 LOC new shader.

### #2 — Single encoder per token (done — was already implemented)

**Status:** The decode loop already uses ONE encoder for ALL 34 layers
(non-MoE path). The ROADMAP item was mislabelled — the actual overhead
is per-`dispatch_thread_groups` call (~5-8µs each), not per-encoder.
Current dispatch count: ~14 dispatches/layer × 34 = 476 dispatches/tok
= ~2.4-3.8ms of dispatch overhead. Reducing requires kernel fusion.

### #3 — Fused `rms_norm + QKV projection` for Q4_K/Q6_K path (open)

**Estimated gain: ~0.2ms/tok (1 saved dispatch × 34 layers × 5-8µs).**
Currently `encode_input_norm_and_qkv` runs two dispatches per layer:
`rms_norm_pipeline` → f32 norm_out buffer → `q4k_q6k_qkv_proj`.
The norm_out write/read is L2-cached (10 KB), so main saving is the
dispatch. A fused `rms_norm_q4k_q6k_qkv` shader:
- Phase 1 (all 128 threads cooperate): reduce `||h||²` / hidden
- Phase 2 (each simdgroup independently): matvec with inline `h[i] / rms * w[i]`
Effort: ~200 LOC MSL (cooperative reduction + two-format Q4K/Q6K paths).
The revised estimate is ~0.2ms (not 0.4ms — norm_out is L2-cached).

### #4 — LM head wrapper overhead (partial — heap done 2026-04-25)

**Remaining gain: ~0.1ms.** `backend_lm_head_topk`:
- ~~partial-sort 262k → top-k~~ → **min-heap done**: avoids 2MB Vec allocation,
  saves ~0.1ms (observed lm_head 2.38 → 2.27ms).
- GPU dispatch+commit+wait: ~200µs — reducible with async readback.
- Buffer readback (1 MB): ~150µs — async pipelining needed.
- Remaining overhead after heap: ~0.35ms.
The GPU kernel itself (1.55ms) is the irreducible floor.

### #5 — `q6k_matvec` 4-element batching (done 2026-04-25)

**Gain: ~1.7ms/tok GPU fwd / ~10% / +7 tok/s** (62→69 tok/s).

Root cause of prior slowness: the scalar inner loop computed `(i & 3u) << 1u`
as a runtime shift for hi2 extraction — the GPU can't hoist a lane-varying
shift amount. Restructured to process 4 consecutive elements per lane per pass
(2 passes × 32 lanes × 4 elements = 256 per superblock) so hi2 shifts are
compile-time constants (0, 2, 4, 6), reducing ops per element and enabling
4-way ILP within each lane. Also: preloaded 16 scale values into registers +
raised ROWS_PER_TG to 8 (256 threads/TG). All Q6_K parity tests pass.

---

## P0: Structural cleanup (open)

From the 2026-04-25 codebase review. Most ship in the same time
window as the perf wins above; some unblock cleaner perf work.

### #6 — Magic-string kernel names on non-tiled shaders (open)

`metal/mod.rs` has **27 raw `library.get_function("...")` calls**
for shaders without `KernelHandle`-style row-tiling (sgemm, geglu,
rope, rms_norm, layer_norm, kv_attention, etc.). They don't need
geometry tracking, but the *kernel name string* still drifts —
renaming a shader silently breaks runtime binding.

Add a `KernelName` trait (sibling of `TiledKernel`) that exports
`KERNEL_NAME` per shader file. Then `library.get_function(<shader>::NAME, …)`
reads the constant. ~30 LOC per shader file, mechanical.

### #7 — `QuantFormat` pattern-match spread (open)

14 files independently `match QuantFormat::*`. Adding FP4 / FP8 /
BF16 = 14 file edits.

Introduce a `FormatRoute` enum computed once per layer
(`F32Input { fused_down: Option<&KernelHandle> }`,
`Q8Input { norm_q8: …, qkv_q8: … }`, etc.) with the `match
QuantFormat::*` confined to one constructor in
`metal/stages/quant_matvec.rs`. Callers receive the opaque route.
Adding FP4 = one match arm.

### #8 — `Pipelines` struct asymmetry (open)

`metal/stages/quant_matvec.rs::Pipelines` mixes `&KernelHandle`
(only `q4_matvec`) with bare `&ComputePipelineState` (q4k_matvec,
q4kf_proj, q6k_matvec). Markers exist for all of them — migrate to
uniform `KernelHandle` storage. Mechanical, ~100 LOC across
callsites.

### #9 — `FullPipelineLayer` 63 pub fields (open)

Constructing one for tests is 30 lines of `field: junk`. Split into
`LayerWeights { wq, wk, wv, wo, gate, up, down }` +
`LayerNorms { input_norm, post_attn_norm, … }` +
`LayerArchParams { eps, attn_scale, head_dim, … }` + optional
`MoeBlock` (already exists). Tests construct just the relevant
subset. ~200 LOC of restructuring + caller updates.

### #10 — `dispatch_full_pipeline` 30+ params (open)

Even after stage extraction the signature is unreadable. Same
`Pipelines`-struct treatment as `stages/quant_matvec.rs` — bundle
the pipelines and norms into a `FullPipelineRefs<'_>` context.

### #11 — `compare_*.rs` examples consolidation (open)

5 `compare_*.rs` files (~1400 LOC) overlap heavily. Particularly
`compare_decode` (195) and `compare_pipeline` (240). Consolidate to
one with subcommand flags.

### #12 — `ProfileTimings` producer (open)

`ProfileTimings` struct + `format_summary` shipped (2026-04-25) but
no code populates `gate_up_ms` / `down_ms`. Wire commit/wait
boundaries through `decode_token_with_moe_fn` — completes the
diagnostic that replaced the deleted 567-LOC `decode_profile.rs`.

---

## P0: Exceed Ollama — DONE (2026-04-09)

### ✅ Full kernel + norm optimization
**Status**: Complete — 17% faster than Ollama

8.5ms / 117 tok/s vs Ollama 10.3ms / 98 tok/s. Key changes:
- Cooperative SIMD norm reduction (O(N²)→O(N)) — saved ~10ms alone
- Q4_KF (GGUF) FFN through llama.cpp-exact q4kf_proj kernel
- Fused gate+up kernels (q4k_ffn_gate_up + q4kf_ffn_gate_up)
- Q4_K matvec rewrite: uint4, 8 rows/TG, multi-row (nr0=2)
- Pre-allocated scratch buffers (550 allocs → 20)
- Batched RoPE + V-norm, SIMD KV attention
- Single cmd buffer + single global encoder

Previous: 29.2ms / 34 tok/s (2.84x Ollama).

### ✅ Dispatch merging
**Status**: Complete (but negligible impact — Apple Silicon dispatch overhead is ~0ms)

### Wire cached layers into decode path
**Impact**: ~4x speedup (compute 8 layers instead of 34)  
**Effort**: Low  
**Status**: Not started (infrastructure ready in larql-inference)

L0-12 are template-fixed (0.999 cosine similarity). At 0.25ms/layer × 8 layers = 2ms → ~500 tok/s.

### ✅ Optimize KV cache attend kernel
**Status**: Complete — simd_max/simd_sum reductions, float4 Q·K dot products, 1024-entry scores.

### ✅ Fix O(N²) norm kernels
**Status**: Complete — cooperative SIMD reduction in all norms. Saved ~10ms (the single biggest win).

## P0.5: Gemma 4 26B A4B correctness

### ✅ CPU MoE decode interleave — DONE (2026-04-20)
GPU dense FFN + CPU MoE per layer. See `metal/decode/moe_combine.rs`
for the outer combine math (HF Gemma 4 has three post-FFN norms per
MoE layer: `_1` on dense, `_2` on MoE, and un-suffixed outer on the
sum — only the un-suffixed one gets `layer_scalar` applied to the
whole layer output after residual add).

### ✅ Full end-to-end correctness — DONE (2026-04-24)
Four coordinated fixes were needed (earlier "working" claim was only
approximate — the Metal output was degenerate token repetition on a
cold vindex). All verified against HF bf16 via layer-by-layer
residual-cosine diff in `metal/decode/diag.rs::ResidualDump` +
`/tmp/hf_residuals.py` + `/tmp/diff_residuals.py`.

1. **Row-padded Q4_K / Q6_K storage** for matrices whose inner dim
   isn't a multiple of 256. 26B A4B's `intermediate_size=2112` gives
   8.25 super-blocks per row; old extraction stored contiguously and
   the matvec shader read wrong bytes for every `down_proj` row past
   row 0. `pad_rows_to_256` in `larql-vindex/format/weights/write.rs`
   per-row-pads to the next 256-multiple; runtime dispatches
   `down_proj` with `K = inter_padded` (see `metal/decode/mod.rs`).
   `act_buf` allocated to `inter_padded * 4` bytes and zero-inited so
   the trailing columns contribute nothing. Aligned models
   (`inter_padded == inter`) are unchanged.
2. **Parameter-free router RMSNorm** — HF `Gemma4TextRouter.norm` has
   `with_scale=False`; no weight tensor exists on disk. Trait method
   `moe_router_norm_parameter_free()` + `rms_norm_no_weight` branch
   in `cpu/ops/moe/forward.rs`. Also added `router.scale *
   hidden_size^-0.5` multiplier (HF's `scalar_root_size`).
3. **Outer `post_feedforward_layernorm.weight`** (un-suffixed) added
   to extraction + wired through `FullPipelineLayer.moe_outer_post_norm`.
   Distinct from the `_1` dense-branch norm that was previously being
   double-applied.
4. **`layer_scalar` applied to the whole layer output** after residual
   add (`new_h *= layer_scalar`) — matches HF's `hidden_states *=
   self.layer_scalar` at the end of `Gemma4TextDecoderLayer.forward`.
   Prior code folded it into the outer-norm scale (14× magnitude
   error, collapsed the model to degenerate output).

Artifacts for future regression checks:
- `crates/larql-cli/examples/patch_down_proj.rs` — surgical vindex
  patcher (re-quantises `down_proj` rows with per-row padding).
  Avoids re-extracting 42 GB when the extraction side is fixed.
- `crates/larql-compute/src/metal/decode/diag.rs::ResidualDump` —
  env-gated (`LARQL_DUMP_RESIDUALS=<path>`) binary dump of every
  layer's `layer_in` / `h_post_attn` / `layer_out` for HF-ref diff.
- `crates/larql-inference/tests/test_arch_golden.rs` — architecture
  regression suite with one `#[test]` per `(arch × backend)`,
  skip-if-missing for vindexes. Caught the broken output immediately
  and flagged which architecture-specific change broke it.

### Batched MoE prefill
**Effort**: Medium
**Status**: Workaround shipped (token-by-token decode loop in `prefill_q4`)

Current workaround is correct but serialises `seq_len` decode calls —
O(seq_len × num_layers) GPU command buffers for a prompt. The real fix
is a batched prefill that processes all positions in a single pass:
for each layer, dispatch GPU dense FFN over all positions, then CPU MoE
over all positions, then proceed to next layer. Requires restructuring
`dispatch_full_pipeline` to accept a per-layer CPU callback.

### Fix `dispatch_full_pipeline` layer_scalar
**Effort**: Low
**Status**: Not started — current models (Gemma 3 4B) not affected

`dispatch_full_pipeline` applies `layer_scalar` to `h_bufs[l+1]`
(full residual = `h_post_attn + ffn_delta`) instead of just the FFN
delta. Correct formula: `h_post_attn + scalar * ffn_delta`.

Fix: pass `(scale_pipeline, scalar)` into
`residual::encode_post_ffn`, apply scalar to the normed FFN buffer
before the residual add. Call sites: `full_pipeline.rs:844`,
`tests/test_metal_shaders.rs:2696,2748` — add `None` for non-scaling.

Not urgent: Gemma 3 4B has `layer_scalar = 0.0` (no scaling); Gemma 4
26B is all-MoE and bypasses `dispatch_full_pipeline` via the new
decode-loop prefill.

## P1: Production Hardening

### Streaming prefill
**Effort**: Medium  
**Status**: Prefill pipeline exists but uses CPU for KV cache population

The `prefill_q4` GPU pipeline runs the forward pass. KV cache is populated via CPU `prefill_with_kv` afterward. Integrate KV cache writes into the GPU pipeline to eliminate the CPU roundtrip.

### Dynamic KV cache sizing
**Effort**: Low  
**Status**: Fixed at 4096 max_seq

Current KV cache allocates for 4096 tokens at creation. Need dynamic growth or configurable max_seq for long-context inference.

---

## P1.5: Platform expansion

**Prerequisite: performance parity with Ollama on Metal first.**
These items are sequenced after the Metal gap closes (~1.0× vs Ollama),
so platform users start with a competitive baseline.

### Linux support
**Effort**: Medium  
**Status**: Not started

larql-compute is Metal-only. The `ComputeBackend` trait and CPU fallback
already compile on Linux (no Metal dependency at the trait level). Gaps:

- `larql-compute` feature-gates: `#[cfg(feature = "metal")]` guards the
  entire `metal::` module; the CPU path is the Linux baseline today.
- `larql-cli` / `larql-inference`: a small number of `metal`-feature
  entrypoints need `#[cfg(...)]` guards to build without Metal.
- No build-system CI: add a GitHub Actions Linux matrix that builds all
  crates without `--features metal` and runs the CPU test suite.

Expected result: `cargo build -p larql-cli` (no features) works on
Ubuntu 22.04 / 24.04 x86_64 and aarch64, with CPU-only decode.

### Windows support
**Effort**: Medium  
**Status**: Not started

Similar to Linux plus:
- Path handling: a small number of `std::fs::File::create` /
  `PathBuf::join` calls use `/tmp/` or Unix paths — audit and fix.
- Symbol visibility: `extern "C"` symbols from BLAS need checked on
  MSVC (MKL) and MinGW (OpenBLAS).
- CI: Windows matrix in GitHub Actions using `windows-2022`.

Expected result: `cargo build -p larql-cli` works on Windows 11
x86_64 (MSVC toolchain) with CPU-only decode.

### CUDA backend (re-land from earlier PR)
**Effort**: Large  
**Status**: Trait ready, implementation was in an earlier PR — needs
        cherry-pick + rebase onto current `ComputeBackend` trait.

An earlier PR implemented CUDA kernels but was not merged. Current
`ComputeBackend` trait supports the interface; the Metal decode loop
(`decode_token_with_moe_fn`) provides the implementation template.

Scope to re-land:
1. `cuda::` module gated on `--features cuda` (mirrors `metal::` module).
2. Buffer management via `cuMemAlloc` / `cuMemcpy` under unified-memory
   or explicit device buffers.
3. Kernel ports: `q4k_matvec`, `q6k_matvec`, fused attention (FlashAttention
   or a clean CUDA port of the Metal `kv_attention` kernel), `rms_norm`.
4. `DecodeBackend` impl wired into `decode_token_with_moe_fn`.
5. `larql bench --backends cuda` path in the CLI.

Target: competitive with llama.cpp on a single A100 / H100 for
Gemma 3 4B and Gemma 4 27B (the models already validated on Metal).

## P2: Research

### Q4_K FFN pipeline (end-to-end) — DONE
**Effort**: Medium  
**Status**: ✅ Complete (2026-04-07)

Vindex loader (`load_interleaved_q4k`), inference wiring (`predict_honest` prefers Q4_K FFN), and format tag propagation through `FullPipelineLayer` all wired. When `interleaved_q4k.bin` exists, Q4_K format flows through to compute shader dispatch.

### simdgroup_multiply_accumulate for tiled matmul
**Effort**: Large  
**Status**: Research

Apple Silicon has dedicated matrix hardware. For batch inference (seq>1), tiled Q4_K matmul using simdgroup_matrix operations could significantly speed up prefill. Not useful for seq=1 decode (matvec, not matmul).

### Fused layer kernel
**Effort**: Large  
**Status**: Research

Single kernel per layer: norm → QKV → attention → O → residual → norm → FFN → residual. Eliminates ALL inter-op dispatch overhead. Requires careful register management and threadgroup synchronization.

## Completed

| Item | Date | Impact |
|------|------|--------|
| ComputeBackend trait | 2026-04-03 | Foundation |
| Q4_0 v1-v5 kernels | 2026-04-05 | v4 at 61 GB/s |
| Multi-layer FFN batch | 2026-04-05 | 8.4ms/21L |
| Fused attention (RoPE+GQA+softcap) | 2026-04-06 | Correct output |
| Q8 fused QKV | 2026-04-06 | 2.2x vs separate |
| Full pipeline (attn+FFN, 1 cmd) | 2026-04-06 | 18.5ms/21L |
| Safe buffer reads | 2026-04-06 | 13 unsafe sites → 1 |
| CPU Q4_K/Q6_K reference | 2026-04-06 | Cross-backend tests |
| Cross-backend tests (11 tests) | 2026-04-06 | Metal vs CPU verified |
| Q4_K fused QKV | 2026-04-06 | 1.78x vs Q8 |
| Dual-path decode (Q4_K/Q8 auto) | 2026-04-06 | 59 tok/s |
| GPU prefill pipeline | 2026-04-06 | seq>1 on GPU |
| skip_rope flag | 2026-04-06 | Prefill KV cache |
| Sub-block lane assignment | 2026-04-07 | 83% utilization |
| llama.cpp kernel architecture port | 2026-04-07 | Register-based input |
| Component profiling | 2026-04-07 | Found real bottleneck |
| Zero warnings | 2026-04-07 | Clean build |
| ADR documentation | 2026-04-07 | 8 decisions recorded |
| Partial RoPE (rotary_dim) | 2026-04-07 | rope_apply + fused_attention, ADR-010 |
| Gemma 4 architecture support | 2026-04-07 | Per-layer head_dim, KV heads, K=V, layer_scalar |
| Shader documentation | 2026-04-07 | docs/shaders.md — all 44 kernels |
| Quantization format docs | 2026-04-07 | docs/quantization-formats.md |
| Decode pipeline docs | 2026-04-07 | docs/decode-pipeline.md |
| Example reorganization | 2026-04-07 | 25 examples: demo_, compare_, profile_, best_, test_ |
| PERFORMANCE.md refresh | 2026-04-07 | All numbers from fresh benchmark runs |
| ROADMAP.md | 2026-04-07 | P0/P1/P2 targets documented |
| Per-layer architecture params (ADR-011) | 2026-04-07 | 18 fields on FullPipelineLayer: eps, attn_scale, head_dim, num_q/kv_heads, rope_base, rotary_dim, sliding_window, v_norm, layer_scalar, norm_type, ffn_type, activation, biases |
| pipeline.rs extraction | 2026-04-07 | FullPipelineLayer + types moved from lib.rs to pipeline.rs |
| 7 new shader kernels | 2026-04-07 | silu, gelu_tanh, layer_norm (2), v_norm, scale_vector, rope_at_pos partial |
| Model-agnostic compute | 2026-04-07 | No hardcoded model assumptions — all behavior parameterized per-layer |
| Single cmd buffer decode | 2026-04-08 | All 34 layers in one cmd, single encoder per layer |
| Batched RoPE/V-norm | 2026-04-08 | rope_at_pos_batched, v_norm_batched — 16 dispatches → 3 |
| Q4_K FFN format routing | 2026-04-08 | Q4_K weights use q4k_matvec, skip Q8 quantize |
| Fused gate+up kernel | 2026-04-08 | q4k_ffn_gate_up — single dispatch, shared input |
| Q4_K matvec rewrite | 2026-04-08 | uint4 loads, 8 rows/TG, sub-block striping, nr0=2 |
| Q4_KF FFN routing | 2026-04-08 | llama.cpp-exact q4kf_proj for FFN gate/up/down |
| SIMD KV attention | 2026-04-08 | simd_max/simd_sum, float4 dot, 3 barriers (was 6) |
| Ollama parity | 2026-04-08 | 2.84x → ~1.25x at 34 layers, no caching |
| Q4_KF fused gate+up | 2026-04-09 | q4kf_ffn_gate_up — llama.cpp inner loop, shared input |
| Pre-allocated scratch buffers | 2026-04-09 | 550 allocs → 20, saved ~2ms |
| Single global encoder | 2026-04-09 | One encoder for all 34 layers (no per-layer create/end) |
| **Cooperative SIMD norms** | **2026-04-09** | **O(N²)→O(N) in rms_norm/residual_norm — saved ~10ms** |
| **Ollama EXCEEDED** | **2026-04-09** | **8.5ms / 117 tok/s = 0.83x Ollama (17% faster)** |
