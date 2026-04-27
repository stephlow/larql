# Roadmap — larql-compute

## Current state (2026-04-26, M3 Max, real vindex)

| Engine | tok/s | ms/tok | Notes |
|---|---|---|---|
| **LARQL Metal** (gemma3-4b-q4k-v2, Q6_K down) | **81–84** | ~12.0ms | q6k_matvec ROWS_PER_TG=4 + lm_head GPU top-K (2026-04-26) |
| **LARQL Metal** (gemma3-4b-q4k-downq4k, all-Q4_K) | **70.1** | 14.26 | all-Q4_K extract; q4k_geglu_silu_down fires |
| **Ollama** gemma3:4b | **98–103** | ~10ms | reference (same hardware, same prompt) |
| **Gap** | LARQL is **~1.22×** slower | ~2.2ms/tok | per-stage decomposition below |
| **LARQL Metal** (gemma4-26B-A4B, MoE Q4K GPU dispatch) | **5.1** | ~194ms | Phase 1 shipped; Phase 2 open — see P0 below |
| **LARQL Metal** (gemma4-26B-A4B, `SKIP_MOE=1` ceiling) | **56.8** | ~15ms | GPU-only baseline; expert dispatch accounts for ~179ms gap |

Per-stage (50-token decode after 3 warmup, typical):

| Stage | LARQL | Ollama (est.) | Gap |
|---|---|---|---|
| GPU fwd | ~11.2ms | ~8.5ms | ~2.7ms |
| lm_head | ~1.84ms | ~1.3ms | ~0.5ms |
| **Total** | **~12.3ms** | **~9.9ms** | **~2.4ms** |

**lm_head shipped 2026-04-26**: 2.28ms → 1.84ms (~0.44ms saved). Two
pieces — (1) `top_k_sorted` in `larql-vindex/index/storage/lm_head.rs` now
runs an argmax fast path for `k=1` and a size-K min-heap for `k>1` instead
of allocating a 2MB `Vec<(u32, f32)>` and `select_nth_unstable` over 262K
elements (~0.25ms saved). (2) New `f32_topk_partial` MSL shader emits
`K_TOPK = 8` (val, idx) pairs per TG via repeated simd_max + index-mask;
backend methods `f16_gemv_topk` / `q4_matvec_topk` route the bench's
`top_k = 5` lm_head call through GPU partial top-K + 64KB readback +
size-K CPU heap, avoiding the 1MB scores readback and the linear scan
over 262K floats (~0.2ms additional). Greedy-decode `f16_gemv_topk1` /
`q4_matvec_topk1` are also wired (no production caller yet — bench /
generate both use top_k=5).

**Gap analysis (2026-04-26, measured + per-kernel profiling):**

| Source | LARQL | Ollama (est.) | Gap |
|---|---|---|---|
| Dispatch overhead | ~1.87ms (374 × 5µs) | ~1.36ms (272 × 5µs) | **0.51ms** |
| Kernel compute | ~9.1ms | ~7.1ms | **~2.0ms** |
| lm_head overhead | ~1.84ms | ~1.30ms | **~0.5ms** |

**Per-kernel profiler results** (run `diag_profile_kernels`, see PERFORMANCE.md):

| Kernel | Batched GB/s | ms/tok | Bottleneck |
|---|---|---|---|
| q6k_matvec (down, K=10240) | ~315 GB/s | ~2.3ms | bandwidth-bound (LPDDR5X) |
| q4k_ffn_gate_up (gate+up, K=2560) | ~272 GB/s | ~3.7ms | **compute-bound** (Q4_K dequant) |
| f32_gemv (lm_head, 262K×2560) | ~370 GB/s | — | bandwidth-bound (near peak) |

Down + gate+up = **~6ms/tok** of the ~11ms GPU fwd. Gate+up is compute-bound
because Q4_K at K=2560 (0.5625 B/elem, lowest ratio) — the GPU spends more
cycles on nibble dequant arithmetic than waiting for LPDDR5X.

The "117 tok/s" historical number was synthetic-weight Q4_KF without
real vindex load. Production extracts use Q6_K down (Ollama
convention); the q4_KF fast-path doesn't apply to those.

---

## P0: Production gap closers

Remaining gap: **~1.30×** (~77 vs ~100 tok/s, ~3ms/tok).

### Prefill: per-position matvec → matmul (4-14× gap, biggest end-to-end win)

**Measured 2026-04-27** (gemma3-4b-q4k-v2.vindex). The gap **scales with prompt length**:

| prompt length | larql prefill | ollama prefill | gap |
|---|---|---|---|
| 18 tok (chat) | 196 ms (10.9 ms/tok) | 50 ms (2.8 ms/tok) | **3.9×** |
| 340 tok (long) | 2933 ms (8.6 ms/tok) | 210 ms (0.62 ms/tok) | **14×** |

The widening ratio is the smoking gun: larql is per-position linear (`prefill ≈ seq_len × decode_per_tok`); ollama is sublinear via gemm. Decode itself (seq=1) is only 1.30× behind.

**Root cause** (verified 2026-04-27 by reading `metal/ops/full_pipeline/dispatch.rs`): `prefill_q4 → dispatch_full_pipeline` IS already wired and IS allocating `[seq_len × hidden]` buffers, but every per-stage compute step issues per-position matvec dispatches. For an 18-token × 34-layer prefill that's ~600+ matvec calls vs ollama's ~34 gemm calls per stage.

**The earlier "wire dispatch_prefill" suggestion was wrong** — `metal/prefill.rs::dispatch_prefill` is dead code; production already goes through `prefill_q4`. Infrastructure isn't missing, the kernel approach is.

**Three actionable wins, ordered by effort × impact:**

1. **Encoder coalescing** — **SHIPPED 2026-04-27**, marginal impact.
   Hoisted `cmd.new_compute_command_encoder()` out of the per-position loops in `dispatch.rs::399` (O proj) and `stages.rs::97`, `:174` (input_norm + QKV). One encoder per stage instead of `seq_len` of them. **Measured: saves ~5% on long prompts, within noise on short prompts.** The 5 µs × seq_len savings is real but dwarfed by per-dispatch kernel compute time. No regression on decode (seq=1 path runs the loop once, identical semantics). 135 Metal tests still pass.

2. **Q4_K threadgroup memory reuse across positions** (M, 2-3 days, ~20-30% on long prompts — speculative)
   The current matvec loads the same Q4_K weight rows from device memory once per position dispatch. Cache one super-block of weights in threadgroup memory and run all `seq_len` positions through it before advancing rows. Same matvec primitive, reordered loops. Closes a chunk without writing new shaders. **Caveat**: the gate+up kernel is already compute-bound (272 GB/s, ALU-limited dequant), so weight-side caching may not help much; output-side caching across positions might.

3. **Q4_K matmul (gemm) kernel** — **SHIPPED 2026-04-27** (kernel + parity tests; not yet wired into prefill).
   `crates/larql-compute/src/metal/shaders/q4k_matmul.rs` — amortises Q4_K dequant across `COLS_PER_TG=4` positions per super-block. Same `ROWS_PER_TG=4` simdgroup geometry as `q4k_matvec`, plus a per-thread `acc[4]` accumulator array (16 bytes register footprint, fits comfortably). 5 parity tests in `tests/test_kernel_q4k_matmul.rs` assert bit-equivalence with stacked matvec calls across basic / seq_len=1 / ragged-tail / production shapes. Perf spot-check (`tests/test_kernel_q4k_matmul_perf.rs`, gated on `LARQL_PERF_SPOT_CHECK=1`) on N=2560, K=8192, M=18: **3.82× speedup** (4.99 ms stacked matvec → 1.31 ms matmul). At full closure that's ~196 ms → ~51 ms prefill on Gemma 3 4B (ollama parity).

   **Wiring status — partial 2026-04-27**: Wired into the O projection site (`dispatch.rs::5. O projection`). Added `q4k_matmul: Option<&KernelHandle>` to `quant_matvec::Pipelines`; threaded through `dispatch_full_pipeline` signature and all callers. Branches on `seq_len > 1 && format == Q4_K && pipeline.is_some()` and falls back to per-position matvec otherwise. Decode (seq=1) keeps the matvec path, decode tests (135 lib) all pass.

   **Measured impact of partial wiring**: WITHIN NOISE. Short prompt 196 → 203 ms; long prompt 2933 → 3006 ms; decode 13.78 → 13.45 ms/tok. O projection is only ~1/7 of the per-position Q4_K work in prefill — the 3.8× kernel speedup applied to one site saves ~2 ms on an 18-tok prompt, below the ±5% prefill noise floor. The kernel works, but a single call site doesn't show in the headline number.

   **Open — full wiring** (the actual perf delivery):
   - `metal/stages/ffn.rs::76,135,172`: FFN gate, up, and down matvec loops. Each is a clean per-position Q4_K matvec — direct matmul swap, no fused-kernel complications. Combined ~3× the work of O proj; should be the largest measurable win.
   - `metal/ops/full_pipeline/stages.rs::97` (QKV f32 path): fused `q4kf_qkv_proj` / `q4k_qkv_proj` kernels do Q+K+V in one dispatch per position. Either (a) write a fused Q+K+V matmul kernel (mirrors the per-position fused convention, biggest one-time effort), or (b) fall back to per-projection matmul (3 calls per layer, simpler but loses the per-position fusion win). Bench-test both to decide.
   - `metal/ops/full_pipeline/stages.rs::174` (Q8 path): same pattern; Q8 has its own fused QKV kernel.

   Once gate/up/down + QKV are all wired, total Q4_K per-position dispatches drop from ~7×seq_len per layer to ~5 per layer (matmul replaces gate/up/down/QKV; activation + residual stay per-position because they're not matmuls). At that point the 3.8× kernel speedup should translate to a ~3× prefill improvement, closing most of the 4-14× gap.

   For the long-haul (matching ollama on 340-token prompts): the current matmul uses simdgroup-sum reduction; a future step is `simdgroup_matrix` operations (the existing P2 entry below). The current kernel is "matvec amortised", not true gemm — but the perf headroom from amortisation alone is enough to close the short-prompt gap if all sites are wired.

**What landed in #1 (for future-me)**: encoder coalescing at three sites (`dispatch.rs::5. O projection`, `stages.rs::QKV f32 path`, `stages.rs::QKV Q8 path`). The FFN stage was already coalesced — `ffn::encode_gated/encode_standard` take a single encoder and iterate per-position dispatches inside. `residual::encode_post_attn/post_ffn` similarly. So the only remaining waste was at the dispatch.rs/stages.rs level.

**Bench reproduction**:
- Short: `larql bench <vindex> --backends metal --ollama gemma3:4b --tokens 100 --warmup 8`
- Long: same with `--prompt "<340+ token prompt>"` to surface the full gap.

### q6k_matvec ROWS_PER_TG shader/dispatch mismatch — **FIXED (2026-04-26)**

**Root cause of the "regression" to 68-70 tok/s:** the shader constant
`Q6K_ROWS_PER_TG` and the Rust dispatch constant `ROWS_PER_TG` were mismatched:

- **Shader:** `Q6K_ROWS_PER_TG = 2` → `row_idx = tg_id * 2 + sg_id` (sg_id 0..3 = 4 rows per TG)
- **Rust dispatch (HEAD):** `ROWS_PER_TG = 4` → dispatched ceil(N/4) = 640 TGs

With this mismatch, maximum covered row = 639 × 2 + 3 = **1281 of 2560**. Rows 1282–2559 received **zeros** — a silent correctness bug in the FFN down projection for dense models. Model output was degraded but simple prompts (e.g. "Paris") survived because the residual stream carried enough signal.

The stash that fixed the dispatch to `ROWS_PER_TG = 2` made the output correct but dispatched 1280 TGs — 2× more work than necessary (each row computed by two adjacent simdgroups due to the overlap in the formula).

**Fix:** set both constants to `4`: shader `Q6K_ROWS_PER_TG = 4` and Rust `ROWS_PER_TG = 4`. Each TG covers 4 non-overlapping rows (sg_id 0..3), dispatches 640 TGs, correct output, optimal throughput.

**Result:** 78.7 tok/s, GPU fwd 10.8ms — **correct and faster than the broken HEAD**.

### P0 correctness blockers — status (2026-04-26)

1. **✅ q6k_matvec ROWS_PER_TG mismatch** — FIXED. Shader and Rust constants both set
   to 4. All 2560 rows now covered; dense model back to 78.7 tok/s. See entry above.

2. **✅ Mixed Q4_K/Q6_K QKV fused V path** — resolved 2026-04-26 (stale entry).
   The named test `q4k_q6k_qkv_proj_normed_matches_separate_norm_and_proj`
   passes against `q6k_matvec` at the original 512-hidden test geometry
   AND at production hidden=2560 (10 super-blocks/row). Added
   `q4k_q6k_qkv_proj_normed_matches_at_production_hidden` regression
   test pinning the larger shape so any future drift is caught at
   production K, not via a model-output bug report.

3. **MoE GPU dispatch: activation scratch not padded to `inter_padded` (open).**
   `gpu_moe_dispatch` dispatches expert down with `K = inter_padded` but the activation
   buffer is sized and offset-indexed with `inter`. For `moe_intermediate_size=704`
   (`inter_padded=768`), the down projection reads 64 floats beyond each expert's
   activation slice. Fix: allocate `top_k × inter_padded × 4` bytes, zero-fill padded
   tail, offset per expert by `inter_padded` (not `inter`).

4. **MoE GPU parity test coverage thin (open).**
   Existing tests cover CPU routing and prefill shape/finiteness but not
   `gpu_moe_dispatch` correctness for Q4_K experts, padded intermediates, or
   `valid_count < top_k`.

| Source | Gap | Status |
|---|---|---|
| **Kernel compute** | **~2.0ms** | gate+up compute-bound (K=2560 ALU-limited); open |
| **lm_head overhead** | **~0.5ms** | GPU argmax_partial (top_k=1) + GPU topk_partial K_TOPK=8 (top_k=5) shipped 2026-04-26 (`f32_topk_partial` shader, `f16_gemv_topk` / `q4_matvec_topk` wired into `lm_head_knn_backend`) |
| **Dispatch overhead** | **~0.5ms** | Mostly closed (374 vs Ollama ~272 dispatches) |

**Achievable targets:**
- Close kernel compute gap → **~87 tok/s**
- Close lm_head gap → **~85 tok/s**
- Close all remaining → **~95 tok/s** (~Ollama parity)

**Key findings from per-kernel profiler (`diag_profile_kernels`):**
- Gate+up is **COMPUTE-BOUND** at 272 GB/s (K=2560, 0.5625 B/elem, dequant-limited).
  Float4 dual-sub-block approach was tried and regressed — complex addressing offsets
  gains from ILP. Format-compatible vectorization remains the unsolved problem.
- q6k_matvec (down) is **bandwidth-bound** at ~315 GB/s (K=10240, 0.82 B/elem).
  ROWS_PER_TG=2 (64 threads/TG) improved it by ~5% via better occupancy.
- lm_head f32_gemv is near peak at 370 GB/s — the overhead is CPU-side (readback,
  sort). `f32_gemv_topk1` GPU argmax ships the fix for top_k=1 callers.

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

### #5 — `q6k_matvec` full rewrite (done 2026-04-25)

**Total gain: ~3ms/tok / ~20% / +10 tok/s** (62→72 tok/s), in two phases:

**Phase A — 4-element batching** (+7 tok/s, 62→69):
Scalar inner loop used `(i & 3u) << 1u` — a runtime shift the GPU can't hoist.
Restructured to 4-element groups with compile-time hi2 shifts (0,2,4,6), 16
preloaded scales, and ROWS_PER_TG=8. All tests pass.

**Phase B — inter-superblock interleaving + X preload + deferred scale** (+3 tok/s, 69→72):
Adapted the llama.cpp `kernel_mul_mv_q6_K_f32_impl` strategy to LARQL's linear
Q6_K layout (GGUF's transposed layout can't be ported directly — different format):
- `ix = lane & 1` → adjacent lanes process alternate superblocks, letting DRAM
  serve two memory banks in parallel.
- `xl[16]` preloaded before weight reads → X fetches overlap weight byte loads.
- Deferred scale: `acc += d*sc * (unscaled_sum_4_elems)` — 4× fewer scale mults.
- ROWS_PER_TG dropped from 8→4 (128 threads/TG) → halved register pressure,
  2× more concurrent TGs, better latency hiding on LPDDR5X.
Effective Q6_K bandwidth: ~322 GB/s (up from ~294 GB/s).

### #5b — `q4k_matvec` llama.cpp-style rewrite (open — see #6)

Folded into #6 below with updated size estimate.

---

### q6k_matvec ROWS_PER_TG — correctness fix (2026-04-26)

**Corrected to ROWS_PER_TG=4** for both shader and Rust dispatch constant. See "P0
correctness blockers" entry above for full diagnosis. The previous ROWS_PER_TG=2
ship note was based on a mismatch that appeared to gain performance by skipping half
the rows — real performance at correct ROWS_PER_TG=4 is **78.7 tok/s, GPU fwd 10.8ms**,
better than any previous measurement.

### f32_gemv_topk1 GPU argmax (done 2026-04-26, infrastructure)

New `MatMul::f32_gemv_topk1` trait method: runs gemv + GPU argmax in one command
buffer, reads back only 8KB (1024 partial results) instead of 1MB (262K scores).
Saves ~0.33ms for top_k=1 callers. Implemented on MetalBackend. Main decode loop
uses the KNN lm_head path (top_k=5 → KNN fires first), so this doesn't yet
benefit the bench. Useful for non-KNN models and future greedy-decode APIs.

### Q4_K `sumy` precompute (2026-04-26, measured 2026-04-27 — no measurable gain)

Separated the X-sum used in the min-correction term from the FMA dot-product
loop in `q4k_matvec` and `q4k_ffn_gate_up`. Previously both shared one loop
(`dot_acc` and `sum_acc` accumulated together); now a dedicated `sumy` pass
runs first, leaving the dot loop as a pure FMA chain the compiler can
schedule without interleaved additions. Applied to both the standalone matvec
and the fused gate+up shader.

**Measured 2026-04-27 on the all-Q4_K extract (`gemma3-4b-q4k-downq4k`),
3 runs each, identical bench setup:**

| Shader form | Run 1 | Run 2 | Run 3 | GPU fwd |
|---|---|---|---|---|
| With `sumy` precompute (split loops) | 71.7 | 72.3 | 72.1 | 12.67–12.74 ms |
| Without (combined `dot_acc` / `sum_acc`) | 72.4 | 71.6 | 72.9 | 12.62–12.77 ms |

Difference is within run-to-run variance — the Apple Silicon shader compiler
schedules the combined loop just as well as the split form. Kept the split
version anyway since it's cleaner code for future readers; no perf regression
either direction. Worth flagging that this micro-optimisation didn't pan out
so future "split the FMA chain from the sum" attempts know the answer.

### #6 — Q4_K kernel optimization (explored 2026-04-26, blocked by ALU bound)

**Tried:** (a) inter-superblock interleaving (ix=lane&1 stride-2, already applied).
(b) 2 rows per simdgroup + 64 threads/TG (REGRESSED: halves total wavefronts,
  hurts more than X-sharing helps for K=2560).
(c) llama.cpp uint16 `float4` trick — INCOMPATIBLE: llama.cpp uses a
  transposed nibble layout (qs[b] lo=elem b, hi=elem b+32) while LARQL uses
  linear (qs[b] lo=elem 2b, hi=elem 2b+1). The uint16 accumulation trick only
  works for the transposed layout.

**Root cause unchanged:** K=2560 fits in GPU L1 cache (1440 bytes/row). The
weight read bottleneck is not the X reads but the ~89 MB/layer weight data,
and the main gap vs Ollama is in ALL-operations bandwidth (322 vs ~414 GB/s).

**Remaining Q4_K opportunity:** `sumy[]` precomputation (saves 16 additions
per superblock for the min correction term) and profiling to understand the
full ~2ms kernel gap. For K=8192 (Wo, 4608 bytes/row = DRAM-bound),
inter-superblock interleaving at stride 2 is already applied; stride-4
(ix=lane/8) would add more DRAM bank parallelism.

**Root cause of limited gain:** All Q4_K matvecs in Gemma 3 4B use K=2560 as
input dimension (hidden size). K=2560 → 10 superblocks × 144 bytes = 1440 bytes
per row — fits entirely in GPU L1 cache. The old lane-stride approach had 22/32
idle lanes for K=2560, but L1-cached superblock data hid that inefficiency. The
inter-superblock optimization helps primarily when K is large enough that
superblock data spills to DRAM — which is why Q6_K down (K=10240, 8400 bytes/row,
21.5 MB total) got a much larger gain.

**Potential remaining Q4_K gains:** The llama.cpp approach uses `yl[]/yh[]`
preloading + `float4 acc1/acc2` vectorized accumulation. For the output dimension
(N=10240 for gate/up), more TGs may help via better GPU saturation. But the
fundamental bottleneck for Q4_K with K=2560 is now something else.

**Estimated gain: ~1.0–1.5ms/tok.** The Q4_K kernel handles:
- Wq (8192×2560) + Wk (4096×2560) + Wv fused QKV: 26.3 MB/layer × 34 = 895 MB
- Wo (2560×8192): 11.8 MB/layer × 34 = 401 MB
- W gate+up (10240×2560 ×2, fused): 29.5 MB/layer × 34 = 1003 MB
- **Total Q4_K data: ~2300 MB/token** (vs Q6_K's 1023 MB — more than double)

The old sub-block-stride kernel hasn't been touched. Applying the same
inter-superblock + preload + deferred-scale treatment as Q6_K should
close a proportionally larger gap.

**llama.cpp Q4_K algorithm** (`kernel_mul_mv_q4_K_f32_impl`):
```
ix = tiisg / 8     → 0..3: which of 4 parallel superblock groups
it = tiisg % 8     → 0..7: position within the group
iq = it / 4        → 0 or 1: low or high sub-block
ir = it % 4        → 0..3: which of 4 groups within sub-block

for (ib = ix; ib < nb; ib += 4):   // stride 4, processes 4 superblocks at once
    yl[16], yh[16] = preload X values for this superblock
    sumy[4]        = precompute X sums (for the min correction term)
    for row in 0..nr0:             // nr0=2: 2 rows per simdgroup
        float4 acc1, acc2 = { 0 }  // vectorized accumulation
        FOR_UNROLL (i=0..3):
            acc1[0..3], acc2[0..3] += nibble × yl/yh
        sumf[row] += d × (acc1 scale corrections) - dmin × (sumy correction)
```

Key differences from LARQL's current `q4k_matvec`:
1. **4 parallel superblock groups** (ix=0..3): all 4 groups run simultaneously,
   4× as many concurrent DRAM reads vs LARQL's 1 per stride.
2. **`yl[16]/yh[16]` preloaded**: X reads issued before weight bytes.
3. **`sumy[4]` precomputed**: the `Σ x[i]` term for min correction is
   accumulated once per superblock per ix-group, not per nibble.
4. **`float4 acc1/acc2`**: 4-wide vectorized accumulation — compiler can emit
   packed FMAs for 4× instruction-level throughput.
5. **2 rows per simdgroup** (`nr0=2`): both rows share the same superblock
   reads, amortising preload cost across 2 outputs.

**LARQL's Q4_K format matches GGUF** (same 144-byte block structure: d/dmin
f16 + 12-byte packed scales/mins + 128 bytes of 4-bit nibbles). llama.cpp's
algorithm can be ported directly without format translation.

**Effort:** ~200 LOC MSL. Need to adapt the `yl[]/yh[]` preload pattern
for LARQL's block layout, handle the `fused_q4k_qkv` path (3 output
matrices), and update `q4k_ffn_gate_up` to use the same interleaving.

### #7 — Dispatch fusion: consolidate per-layer ops (open)

**Estimated gain: ~1.0ms/tok** (saves ~200 dispatches at ~5µs each).

Current per-layer dispatch count (~14 for Gemma 3 4B):
1. `rms_norm` (input norm)
2. `q4k_q6k_qkv_proj` (QKV projection)
3. `qk_norm` — Q heads
4. `qk_norm` — K heads
5. `rope_at_pos_batched` — Q heads
6. `rope_at_pos_batched` — K heads
7. `kv_append`
8. `kv_attend`
9. `o_proj` (O projection)
10. `residual_norm` (post-attention residual + FFN norm)
11. `q4k_ffn_gate_up` (fused gate+up)
12. `geglu_gelu_tanh` (activation)
13. `q6k_matvec` (FFN down)
14. `residual_add` (post-FFN)

Three fusions with clear wins (each saves 34 dispatches = ~0.17ms):

**7a — Fused QK-norm Q+K** ✅ done 2026-04-25 (+0.17ms recovered):
New `qk_norm_qk` shader dispatches total_heads = q_heads + kv_heads in one
call; TG index selects Q buffer + q_weight vs K buffer + k_weight.

**7b — Fused RoPE Q+K** ✅ done 2026-04-25 (+0.17ms recovered):
New `rope_at_pos_batched_qk` shader: grid `(rope_pairs, q_heads+kv_heads, 1)`;
thread `h < num_q` selects Q buffer, else K buffer.

**7c — Fused input norm + QKV projection** ✅ done 2026-04-25:
New `q4k_q6k_qkv_proj_normed` kernel: all 128 threads cooperatively reduce
`||h||²` in Phase 1 (barrier), then each simdgroup runs its matvec with inline
`h[i] * rms * (offset + norm_w[i])`. Fires when format is Q4_K Q/K + Q6_K V,
standard RMS norm, no bias (Gemma 3 4B production).

**7e — Fused residual_norm + residual_add** ✅ done 2026-04-25:
New `residual_norm_store` kernel writes both `ffn_norm_out` (normed FFN input)
and `h_post_attn` (raw sum for post-FFN add) in one pass. Replaces the
`residual_norm + residual_add` two-dispatch pair in the Q4_K hot path.

**7d — Fused GEGLU + down** (~0.17ms):
Dispatches 12+13 can be merged for Q4_K down (already done). For Q6_K down,
fusion was attempted but regressed due to GELU-tanh recomputation cost
(see #1 closed). Not viable unless activation is precomputed separately.

---

## P0: Diagnostic infrastructure (done 2026-04-26)

Diagnostics were previously scattered across three locations:
- `src/metal/decode/diag.rs` — NaN detection, residual dumps, per-layer bisect
- `src/metal/decode/profile.rs` — stage-level `ProfileTimings`
- `examples/debug_decode_pipeline.rs` — decode pipeline stage bisect entry point

Now consolidated under `src/metal/diag/`:
- `diag/mod.rs` — public API, re-exports `ProfileTimings`, documents all tools
- `diag/kernel_profile.rs` — `KernelResult` + `profile_all()` for per-kernel
  bandwidth measurement (isolated vs batched, GB/s, bottleneck classification)
- Examples renamed to `diag_*` prefix for clarity

**Key diagnostic commands:**
```bash
# Per-kernel bandwidth profiler (results go to PERFORMANCE.md)
cargo run --release --features metal -p larql-compute --example diag_profile_kernels

# Decode pipeline stage bisect (bisect CPU/Metal divergence)
LARQL_METAL_DUMP_LAYERS=/tmp/dump \
  cargo run --release --features metal -p larql-compute --example diag_decode_pipeline

# NaN/divergence bisect at specific layer (env-gated, zero binary overhead)
LARQL_DECODE_DIAG_LAYER=12 larql infer <vindex> "prompt"
```

---

## P0: Structural cleanup (open)

From the 2026-04-25 codebase review. Most ship in the same time
window as the perf wins above; some unblock cleaner perf work.

### #6 — Magic-string kernel names on non-tiled shaders (DONE)

Added `ShaderKernel` trait + `get_shader_pipeline::<T>()` to
`kernel/traits.rs`; 31 magic strings eliminated. Each shader now
exports a compile-time `NAME` constant — renaming a shader causes a
compile error rather than a silent runtime panic.

### #7 — `QuantFormat` pattern-match spread (partial — classifiers shipped 2026-04-27)

**Classifier helpers shipped:** `QuantFormat::is_q4k_family()` /
`is_q4kf()` / `is_legacy_q8()` on `pipeline.rs`. The most-duplicated
predicate (`format == Q4_K || == Q4_KF || == Q6_K`, repeated verbatim
in `decode/mod.rs` ×2 and `decode_hybrid.rs` ×1) collapses to a single
method call. Adding a future Q4_K-style format updates one classifier,
not 3+ OR-chains. Pinned by `quant_format_classifiers` test.

**Full `FormatRoute` enum DEFERRED.** The roadmap intent
(`F32Input { fused_down: Option<&KernelHandle> }` / `Q8Input { norm_q8,
qkv_q8 }` / etc., with the `match QuantFormat::*` confined to one
constructor in `metal/stages/quant_matvec.rs`) is a 49-file refactor —
every dispatch site that currently matches on `QuantFormat` would need
to switch to consuming a `FormatRoute`. Doing it concurrently with the
in-flight MoE struct refactor risks heavy merge conflicts. Defer until
MoE settles AND there's a concrete near-term need (e.g. an FP4 / FP8
format being added). The classifier helpers above absorb the immediate
duplication cost in the meantime.

### #8 — `Pipelines` struct asymmetry (DONE)

All fields in `metal/stages/quant_matvec.rs::Pipelines` now use
`&KernelHandle`; geometry drift is now a compile error rather than
a silent dispatch mismatch. ~100 LOC mechanical migration across
callsites.

### #9 — `FullPipelineLayer` 63 pub fields (partial — `Default` shipped 2026-04-27)

**Test ergonomics fix shipped:** `FullPipelineLayer` and `QuantWeight` now
implement `Default`, so test code uses
`FullPipelineLayer { wq, ..Default::default() }` instead of spelling out 30
fields. The pre-existing `minimal_layer` helper collapsed from 30 lines to
10. New `default_layer_accepts_local_borrows_via_spread` test pins the
pattern for future tests (verifies `..Default::default()` reborrows the
`'static` defaults at the caller's stack-local lifetime — typical Rust
HRTB territory but worth a test since it's a non-obvious property).

**Full sub-struct split DEFERRED.** The roadmap intent
(`LayerWeights` / `LayerNorms` / `LayerArchParams` / optional `MoeBlock`)
is a 30+ caller-file refactor. Doing it concurrently with the in-flight
MoE struct refactor (ongoing in this branch) risks merge conflicts on
`pipeline.rs`. Pick this back up once MoE work settles. The `Default`
impl removes the immediate test pain — that was the user-visible cost
of #9.

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

### Batched MoE prefill — **SHIPPED (2026-04-26)**

Replaced the O(seq_len × num_layers) token-by-token decode loop with a
batched approach: `dispatch_full_pipeline` now accepts an optional
`moe_fn: Option<&mut dyn FnMut(usize, &[f32], &mut [f32])>` callback.
When the callback is present and a layer has MoE, the function commits
the GPU command buffer after that layer's dense FFN, calls the closure
(which runs CPU experts for all seq_len positions and applies outer norm
+ layer_scalar), then restarts the command buffer for the next layer.

**Measured on Gemma 4 26B A4B (5-token prompt, 15 warmup / 30 tokens, M3 Max):**

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| Prefill | 1889ms | 1297ms | **−31%** |
| Decode GPU fwd | 334ms/tok | 246ms/tok | **−26%** |
| Decode tok/s | 2.9 | **3.9** | **+35%** |

**Why:** 5-token prefill now uses 26 GPU commits (one per layer) vs 130
(5 positions × 26 layers). Batching all positions per layer also improves
weight cache utilisation. GPU layer_scalar skipped for MoE layers in the
dispatch; the callback applies it correctly after combining dense + MoE.
`kv_copy::populate_kv_one_layer` added for per-layer KV cache population.

### GPU expert dispatch — Phase 2: pre-allocated staging buffers (ACTIVE 2026-04-26)

**Status**: ACTIVE — the single remaining fix to reach ~15–20 tok/s on Gemma 4 26B A4B  
**Measured**: Phase 1 shipped 5.1 tok/s. Phase 2 expected ~4× gain. GPU-only ceiling: 56.8 tok/s.

**Scope (single landing):**

1. **Pre-allocate persistent staging buffers** in `decode_token_q4k_moe`
   (`metal/moe_dispatch.rs`). Sizes are constants of `(top_k, inter_padded,
   hidden, row_bytes, down_row_bytes)` — known once per decode, not per layer.
   Buffers to pre-allocate (all `StorageModeShared` so CPU writes via
   `buffer.contents()`):
   - `gate_buf`: `top_k × inter × row_bytes`
   - `up_buf`: `top_k × inter × row_bytes`
   - `down_bufs`: `top_k` × `[hidden × down_row_bytes]` (per-expert; experts
     come from different mmap pages, so K independent staging buffers — not
     a single concatenated one).
   - `g_out`, `u_out`: `top_k × inter × 4`
   - `act_buf`: `top_k × inter_padded × 4`, zero-initialised once
   - `expert_outs`: `top_k × hidden × 4`

   `gpu_moe_dispatch` becomes `gpu_moe_dispatch_with_scratch(scratch, ...)`;
   the per-call body just memcpys expert bytes into the existing buffer
   contents and dispatches. No `self.bufs.output(...)` calls inside the
   per-layer hot path.

2. **Fix activation-stride bug** (P0 correctness blocker #3 in this file).
   Today: `act_buf` allocated at `valid_count × inter_padded × 4`, but the
   geglu kernel writes linearly at stride `inter`. For
   `moe_intermediate_size` not a multiple of 256 (e.g. Gemma 4 26B's 2112 →
   inter_padded 2304), expert `e>0` reads stale/garbage floats. Fix:
   dispatch `geglu_gelu_tanh` per expert with `g_out`/`u_out` linear offset
   `e × inter × 4` and `act_buf` strided offset `e × inter_padded × 4`. K
   extra dispatches per layer (top_k=8 → 8 small dispatches) but each is
   ~5µs — negligible vs the ~120ms allocation overhead this PR removes.
   Alternative: stride-aware kernel — defer if perf demands it post-bench.

3. **Borrow expert slices instead of `to_vec()`** (host-copy churn). Today
   `larql-inference::layer_graph::generate::gpu` allocates two
   `Vec<u8>` per expert per layer (~2.2 MB heap-copy × 30 layers × 8 experts
   per token). Change `get_expert: impl Fn(layer, expert) -> Option<(Vec<u8>,
   Vec<u8>)>` to return `Option<(&[u8], &[u8])>`. Lifetime-bound to the
   weights mmap — borrow lasts only across the `gpu_moe_dispatch` call.
   Updates `decode_token_q4k_moe` signature + the inference-side caller.

4. **Add parity test** `gpu_moe_dispatch` Q4_K experts with
   - aligned `inter` (e.g. 768),
   - misaligned `inter` requiring padding (e.g. 704),
   - `valid_count < top_k` (some experts return None),
   against CPU MoE reference.

**Acceptance criteria**:
- `cargo test -p larql-compute --features metal` green (existing + new parity).
- `larql bench gemma4-26b-a4b` ≥ 15 tok/s (3× from baseline 5.1).
- No regression on `larql bench gemma3-4b-q4k-v2` (dense path untouched).

**Out of scope for this PR**: dense kernel optimisation, fused
QKV V-path correctness blocker (#2), the expert-bytes-→-Metal-buffer copy
itself (already a single memcpy via `contents()` ptr; can't shrink further
without DMA-side weights, which is a larger refactor).


**Root cause of remaining gap.** `gpu_moe_dispatch` calls `self.bufs.output()` ~10 times per
MoE layer to allocate gate, up, per-expert-down, activation, and output Metal buffers.
With 30 MoE layers × ~10 allocations = 300 Metal buffer allocations per decode token,
each allocation of a 1–9 MB `StorageModeShared` buffer costs ~0.4ms on M3 Max.
**Total: ~120ms/token in allocation overhead** (measured: 194ms total − ~40ms compute − ~30ms syncs).

There is also avoidable host-copy churn before those Metal allocations:
`larql-inference::layer_graph::generate::gpu` calls
`weights.get_layer_entry_bytes(...)?` and immediately `to_vec()`s both
expert slices before `gpu_moe_dispatch` copies them into Metal staging.
For Gemma 4 26B A4B, this is 30 layers × top_k=8 × roughly 2.2MB of
heap copies per decode token. Phase 2 should change the API to pass
borrowed mmap slices (`&[u8]`) through the closure and copy exactly once
into reusable Metal buffers.

**Fix.** Pre-allocate all staging buffers once before the layer loop in
`decode_token_q4k_moe` (in `metal/moe_dispatch.rs`), identical to the pattern that
eliminated 550→20 allocations in `decode_token_with_moe_fn` (see ship log below):

```
Pre-allocated once:
  gate_buf:     [top_k × inter × row_bytes]  (gate Q4K staging)
  up_buf:       [top_k × inter × row_bytes]  (up Q4K staging)
  down_bufs:    [top_k] × [hidden × down_row_bytes]  (per-expert down Q4K staging)
  act_buf:      [top_k × inter × 4]  (f32 activations after GELU)
  expert_outs:  [top_k × hidden × 4]  (f32 expert outputs)
```

Sizes are constant per model (determined by `moe.intermediate_size`, `moe.top_k`,
`hidden`). The pre-allocated buffers are reused for all 30 layers via write-in-place
to `buffer.contents()` pointers.

**Effort**: ~1 session. No new shaders needed — just restructure the buffer lifecycle
in `decode_token_q4k_moe`.

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
26B uses the MoE callback path which applies layer_scalar correctly.

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
