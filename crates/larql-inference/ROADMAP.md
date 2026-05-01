# Roadmap — larql-inference

## Current: ~95 tok/s (Metal Q4K) | Ollama: ~101 tok/s | 4 KV engines

## ✅ Metal lm_head — stride-32 Q4_K matvec, f16 GEMV fallback (correctness + perf fix, 2026-05-01)

Gemma 3 4B Metal end-to-end was producing the wrong continuation
("The Capital of France is:  **") on `"The capital of France is"`
while CPU produced the correct "**Paris**" answer. Bisected:

- Per-layer hidden parity holds (`test_decode_consistency_gemma3_4b`
  and the new 2-step variant pass at cos ≥ 0.99995 across all 34
  layers, 1 and 2 decode steps) — KV cache writes/reads and per-layer
  Metal kernels are correct.
- The single-token logits goldens for Metal pinned a top-5 set whose
  positions 4-5 differed from CPU at the prefill boundary, even though
  top-1 matched (`top1_logit Δ ≈ 5e-4`).
- A/B with `LARQL_LM_HEAD_FORCE_CPU=1` confirmed Metal generated
  "Paris" once the lm_head bypassed the Q4_K matvec path, isolating
  the drift to that specific kernel.

Root cause: `shaders/q4k_matvec.rs` 32-lane simdgroup parallel
reduction with a 2-way inter-superblock split (`ix = lane & 1u`)
accumulates partial sums in a different order than the f32 reference.
Same f32 precision at every step; the difference is reduction-tree
associativity. On a 262K × 2560 lm_head matvec this surfaces as
~1e-3 relative drift on top-1 logits, enough to flip rank-1 on
close-call tokens (e.g. " Capital" vs " capital" at decode step 1
of Gemma 3 4B).

**Fix**: `lm_head_topk` (`layer_graph/generate/lm_head.rs`) routes
through the new `lm_head_knn_backend_skip_q4k` method on `VectorIndex`
when the active backend is non-CPU. That dispatch chain replaces the
production `q4k_matvec` first-path with a 3-step ladder:

  1. **Stride-32 Q4_K matvec** (`backend.q4k_matvec_stride32`,
     `shaders/q4k_matvec_stride32.rs` — new) — same Q4_K bytes as
     production, same bandwidth (330 MB/tok read), but lane `k`
     accumulates the dot-product over elements `i % 32 == k` and the
     final reduction is `simd_sum` across 32 lanes — bit-equivalent
     reduction tree to `f16_gemv`. Recovers rank-1 stability without
     paying the f16 fallback's 4× bandwidth penalty.
  2. **f16 GEMV on `embeddings.bin` mmap** (tied-embed only, ~2×
     bandwidth of Q4_K) — fallback when the stride-32 kernel isn't
     dispatchable.
  3. f32 BLAS fallback (`lm_head_knn`).

Opt out of stride-32 with `LARQL_LM_HEAD_STRIDE32=0`; opt back into
the production Q4_K path with `LARQL_METAL_LM_HEAD=1`.

Five attempts on the way to this:
- v1: route through `CpuBackend` via `index.lm_head_knn_backend` —
  picks the **scalar** Q4_K reference (`cpu/ops/q4k_matvec.rs::dispatch`,
  unvectorised), ~510 ms/tok → **1.9 tok/s** end-to-end.
- v2: route through `CpuBackend` via `backend_lm_head_topk` (CPU BLAS
  on f32 `weights.lm_head`), ~30 ms/tok → **23.6 tok/s**.
- v3: route through Metal `backend.f32_gemv` on f32 `weights.lm_head`,
  ~8 ms/tok → **52.2 tok/s** sustained.
- v4: Metal `f16_gemv` on the embed `f16_mmap`, ~4 ms/tok →
  **66.8 tok/s** sustained.
- **v5 (shipped)**: Metal stride-32 `q4k_matvec` on Q4_K mmap, ~3
  ms/tok → **71.5 tok/s** sustained.

**Validation**:
- `arch_gemma3_4b_gpu` now generates `"The capital of France is **Paris**."` (was `"The Capital of France is:  **"`).
- All 4 `gemma3` logits goldens pass for both backends; pinned values are now equal post-fix (per-backend split kept for future drift detection).
- 2-step decode parity (`decode_consistency_gemma3_4b_2steps` — new) confirms KV-cache write/read across decode steps is independently correct.

**Bench (Gemma 3 4B, M3 Max, `larql bench gemma3-4b-q4k-v2 --ollama gemma3:4b -n 50 --warmup 5`, sustained / cold-GPU)**:

| Path | Decode tok/s | lm_head ms/tok | GPU fwd ms/tok | vs ollama |
|---|---|---|---|---|
| Pre-fix (Metal Q4_K matvec, **wrong output**) | ~78 (historic) | ~1 | ~12 | 1.34× slower |
| v1: CPU `index.lm_head_knn_backend` (scalar Q4_K) | **1.9** | 509.3 | 18.6 | 55× slower |
| v2: CPU `backend_lm_head_topk` (BLAS f32) | 23.6 | 30.4 | 12.6 | 4.4× slower |
| v3: Metal `backend.f32_gemv` on f32 lm_head | 52.2 | 8.0 | 12.0 | 2.0× slower |
| v4: Metal `f16_gemv` on embed f16_mmap | 66.8 | 3.8 | 11.8 | 1.57× slower |
| **v5 (shipped)**: Metal stride-32 `q4k_matvec` | **71.5** | 3.0 | 11.7 | **1.44× slower** |
| ollama gemma3:4b | 102.8 | — | — | 1.00× |

(Watch for thermal noise: back-to-back benches on a hot GPU drop
sustained tok/s by 25-30%; cool-GPU numbers above match the historic
~78 baseline structure when adjusting for the 3 ms lm_head cost.)

lm_head is now ~21% of decode (down from 96.5% in v1, 25.5% in v4).
The stride-32 kernel approaches Q4_K's bandwidth floor (330 MB/tok ÷
~400 GB/s ≈ 0.8 ms theoretical; we're at 3 ms ≈ 28% of peak). The
remaining 1.44× gap to ollama (and the ~6 tok/s gap to the historic
~78 baseline) lives entirely in **GPU forward** (75% of decode @
11.7 ms), which is a separate roadmap item — `q4k_matvec` 8sg /
Q4_K matmul for prefill / kernel fusion / encoder coalescing.

**Files**:
- `crates/larql-compute/src/metal/shaders/q4k_matvec_stride32.rs` — new shader, f16_gemv-style stride-32 reduction
- `crates/larql-compute/src/metal/shaders/mod.rs` — register the new module + push to merged source
- `crates/larql-compute/src/metal/mod.rs` — `q4k_matvec_stride32_pipeline` field + KernelHandle init
- `crates/larql-compute/src/metal/trait_impl/matmul.rs` — `MetalBackend::q4k_matvec_stride32` inherent method
- `crates/larql-compute/src/metal/trait_impl/quant_matvec.rs` — `QuantMatVec::q4k_matvec_stride32` trait wire-up
- `crates/larql-compute/src/backend/quant_matvec.rs` — trait method declaration (default returns `None`)
- `crates/larql-inference/src/layer_graph/generate/lm_head.rs` — `lm_head_topk` `prefer_cpu` branch routes to `index.lm_head_knn_backend_skip_q4k(..., backend)`
- `crates/larql-vindex/src/index/storage/lm_head.rs` — new `lm_head_knn_backend_skip_q4k` method (path 1 = stride-32 Q4_K, path 2 = f16 GEMV, path 3 = f32 BLAS); `LARQL_LM_HEAD_STRIDE32=0` opt-out
- `crates/larql-inference/src/residual_diff/capture.rs` — `metal_decode_steps` helper for multi-step parity
- `crates/larql-inference/tests/test_decode_consistency.rs` — `decode_consistency_gemma3_4b_2steps` test
- `crates/larql-inference/tests/test_logits_goldens.rs` — Metal pins re-captured for v5 stride-32 path

---

## Open: GPU-forward kernel utilization — closing the 4.4 ms gap to ollama

**Status**: Open as of 2026-05-01. Diagnosed via
`cargo run -p larql-compute --release --features metal --example diag_profile_kernels`
plus per-step `LARQL_PROFILE_DECODE=1` profiling on Gemma 3 4B; ollama's
fine-grained timings via `/api/generate` (`total_duration`,
`prompt_eval_duration`, `eval_duration`).

**Where the gap *really* lives** (corrected 2026-05-01 after instrumenting
in-pipeline GPU vs CPU timing via `LARQL_GPU_TIMING=1` —
`metal/decode/gpu_timing.rs::TokenGpuTime`):

```
Per-token decode_token (n=12, steady state):
  Wall:    ~10.7 ms
  GPU:     ~10.5 ms  (98% of wall — kernels are GPU-bound)
  CPU:      ~0.5 ms  (5% — dispatch overhead is NOT the bottleneck)
  cmd_bufs: 1 per token (one coalesced buffer covers all 34 layers)
```

So the 14.0 ms/tok vs ollama's 10.4 ms/tok gap breaks down as:

| Stage | larql | ollama (est.) | gap |
|---|---|---|---|
| `decode_token` GPU compute | 10.5 ms | ~7-8 ms | +2.5-3 ms |
| lm_head | 3.0 ms | ~2 ms | +1 ms |
| other | ~0.5 ms | ~0.5 ms | 0 |
| **total** | **14.0 ms** | **10.4 ms** | **+3.5 ms** |

Both gaps are **GPU compute, not CPU dispatch**. Kernel-isolated
`metal/diag/kernel_profile.rs` GB/s overstated the headroom (kernels
run partially pipelined within one cmd buffer; isolated GB/s isn't
the right metric). Our actual decode is at ~75-80% of ollama's
throughput on the same hardware — competitive but not parity.

**Earlier (incorrect) diagnosis preserved for context**:

| Stage | larql peak | ollama | gap | recoverable? |
|---|---|---|---|---|
| GPU forward | 11.6 ms/tok | ~7-8 ms | **+4 ms** | yes — see kernel breakdown |
| lm_head | 3.0 ms/tok | ~1.5-2 | +1.5 ms | mostly tight (~28% of Q4_K bandwidth floor) |
| total/tok | **14.7 ms** | 9.6 ms | +5 ms | most via GPU fwd |
| tok/s | 68 | 104 | 1.53× | |

(Sustained tok/s drops further on hot GPU — thermal throttling doubles
GPU fwd time over ~16 decode steps. Ollama is presumably less affected
because their faster decode finishes the same wallclock budget with less
GPU on-time.)

**Per-kernel utilization** (decode, Gemma 3 4B, M3 Max LPDDR5X ~400 GB/s peak):

| Kernel | Bandwidth | % of peak | ms/tok | Headroom (at 80% peak) |
|---|---|---|---|---|
| q6k_matvec (FFN down, K=10240) | 321 GB/s | 80% | 2.3 | ~0 (already tight) |
| q4k_ffn_gate_up (gate+up, K=2560) | 187 GB/s | **47%** | 5.4 | **-2.2 ms** |
| q4k_matvec (Wo, K=8192) | 184 GB/s | **46%** | 2.2 | **-0.9 ms** |
| q4k_qkv_proj (Q+K+V fused, K=2560) | 114 GB/s | **28%** | 7.1 | **-4.3 ms** |
| **Total recoverable in GPU fwd** | | | | **~7 ms/tok** |

If we hit 80% peak across the three under-utilized kernels: GPU fwd
drops 11.7 ms → ~5 ms, total decode 14 ms → ~8 ms, **125+ tok/s
end-to-end** (ahead of ollama). Realistic target with kernel rewrites:
**80-90 tok/s** as a first milestone (matches the historic memory's
"~78 baseline" pre-correctness-fix).

**Why under-utilized**: per `metal/diag/kernel_profile.rs` annotations,
`q4k_ffn_gate_up` is "COMPUTE-BOUND (K=2560 dequant dominates)". The
Q4_K dequant inline in the shader (decode super-block scale, sub-block
scale via 6-bit unpack, nibble extract, FMA) eats ALU cycles that block
memory issue. Each lane redundantly decodes the per-super-block
`d`/`dmin` and per-sub-block `sc`/`mn`, so the simdgroup spends 32× the
necessary dequant work and the per-row FMA chain stalls waiting for
operands. Llama.cpp's equivalent kernel co-operates one lane per
simdgroup to load scales into threadgroup memory, then broadcasts to
all 32 lanes for a tight FMA loop — eliminates the redundancy.

The same pattern applies to `q4k_qkv_proj` (also K=2560) and `q4k_matvec`
on Wo (K=8192). The three are the largest per-token GPU costs; closing
their utilization is the highest-leverage GPU-fwd work item.

**Optimization paths in priority order** (each independent and stackable):

### G-1 — Cooperative scale-loading in `q4k_ffn_gate_up`

**Status**: ❌ Tried 2026-05-01, no end-to-end win. Kernel kept opt-in
(`LARQL_GATE_UP_COOP=1` → `q4k_ffn_gate_up_coop_pipeline`,
`shaders/q4k_ffn_gate_up_coop.rs`) for future hardware / fusion
scenarios.

**What was tried**: shipped a new `q4k_ffn_gate_up_coop` shader that
keeps the production lane partitioning (`ix = lane & 1u`,
`j = (lane >> 1) >> 1`) but does the per-super-block dequant
cooperatively:
- Lanes 0..7 of each simdgroup each compute one sub-block's
  `(scale = d * sc, mmin = dmin * mn)`.
- Writes go to threadgroup memory (256 B / TG, well under hardware
  limit).
- `threadgroup_barrier(mem_threadgroup)` flushes; all 32 lanes read
  their owned `j`'s `(scale, mmin)`.
- Each writer also re-decodes `d`/`dmin` itself (8× redundant vs
  production's 32×) — using `simd_broadcast` for `d`/`dmin` produced
  wrong output (close-call top-1 flips), likely from the broadcast
  reordering the inner FMA chain enough to drift past the rank-1 gap.

**Result**: bench A/B (3 runs each, cold + warm GPU):
- Coop:     72.1 / 61.8 / 71.8 tok/s, GPU fwd 12.1 / 14.1 / 12.1 ms
- Baseline: 63.2 / 73.0 / 62.2 tok/s, GPU fwd 13.9 / 11.8 / 13.8 ms

Within thermal noise. **No end-to-end win**.

**Why the diagnosis was misleading**: `metal/diag/kernel_profile.rs`
flagged `q4k_ffn_gate_up` as "COMPUTE-BOUND (K=2560 dequant
dominates)" based on isolated-kernel GB/s measurement. In practice the
production kernel's per-lane redundant dequant ALU **runs concurrently
with the per-row weight loads**, filling memory-stall bubbles for free.
Removing the redundant ALU saves cycles in isolation but doesn't
increase memory throughput — the actual bottleneck. Same lesson as the
2026-04-28 `LARQL_F16_ACC=1` kernel-isolated 1.79× → end-to-end parity
finding. Kernel-isolated profiler GB/s alone is not predictive of
end-to-end wins on Apple Silicon GPUs; the right metric is full
end-to-end tok/s on a quiet GPU.

**Implications for G-2 and G-3** (same cooperative pattern proposed
for `q4k_qkv_proj` and `q4k_matvec`-Wo): expect the same null result,
since both kernels share the same per-lane dequant pattern with the
same memory/ALU overlap. Not worth shipping G-2/G-3 as written;
de-prioritise.

**What's actually on the critical path** (revised): the GB/s
under-utilization isn't ALU-driven, it's **memory access pattern /
occupancy**. Possible causes:

- Per-row weight loads are scattered enough that prefetchers don't
  saturate the LPDDR5X channels.
- Threadgroup count too low to hide memory latency across TGs.
- Per-row register footprint blocks higher concurrent-TG counts.

These need a different toolset (Xcode GPU frame capture / Metal
profiler) to localise — kernel-isolated GB/s alone isn't enough.

---

### G-2 — NR0=2 + shared-X-vector port from llama.cpp

**Status**: ❌ Tried 2026-05-01, **slight regression** (~3% slower).
Kernel kept opt-in (`LARQL_GATE_UP_NR2=1` →
`q4k_ffn_gate_up_nr2_pipeline`, `shaders/q4k_ffn_gate_up_nr2.rs`) for
future exploration on different shapes / hardware.

**Result** (3 runs each, thermal-mixed):
- NR2:           68.6 / 69.2 / 68.3 tok/s, GPU fwd 12.76/12.56/12.84 ms
- Baseline 8sg:  71.1 / 71.1 / 71.0 tok/s, GPU fwd 12.24/12.22/12.26 ms

NR2 is ~0.5 ms/tok slower in GPU forward despite the X-cache-traffic
math predicting a savings.

**Why the diagnosis was wrong**: For Gemma 3 4B's K=2560 input, the
X-vector is 10 KB — easily fits in L1 cache (per-simdgroup or
per-TG). Whatever per-row "X reload" we measured at the kernel
boundary is being served from L1 hits, not LPDDR5X traffic. The
per-row reload doesn't actually consume bandwidth, so eliminating it
via NR0=2 saves nothing.

**This is now the THIRD consecutive miss** on a kernel optimisation
that looked high-confidence from `metal/diag/kernel_profile.rs`'s
isolated GB/s measurement (after `LARQL_F16_ACC=1` 2026-04-28 and
`LARQL_GATE_UP_COOP=1` 2026-05-01). The pattern is now clear:
**isolated kernel GB/s is not predictive of end-to-end tok/s on
Apple Silicon**. The bottleneck must be one of:

- Dispatch / scheduling overhead (not measured by `kernel_profile`)
- Memory subsystem contention across in-flight TGs (not measured)
- Thermal throttling shifting the steady-state target (real but
  doesn't explain peak-cold differences)

**Implications for future kernel work**: stop guessing from isolated
GB/s. Either:
1. Get **actual end-to-end profiling** (Xcode GPU frame capture)
   before any further kernel optimisation work — see G-5.
2. Attack **structural** changes that bypass per-kernel utilisation
   entirely — most notably **G-3** (flash-attention fusion), which
   reduces dispatch count regardless of per-kernel GB/s.

#### Original diagnosis (preserved for context, since the analysis was
correct *for what it measured* — the kernel-isolated GB/s gap is
real, but the gap doesn't translate to end-to-end work)

**Diagnosis**: Side-by-side bench against `ollama gemma3:4b` on
`"The capital of France is"`, num_predict=20:

| | larql | ollama | Δ |
|---|---|---|---|
| Decode tok/s | 71.7 | 96.0 | **+3.53 ms/tok gap** |
| GPU fwd | 12.5 ms | est. 7-8 ms | ~5 ms gap |

llama.cpp's Q4_K matvec
(`ggml/src/ggml-metal/ggml-metal-impl.h::N_R0_Q4_K`) processes **2
output rows per simdgroup** (`NR0=2`) with the X-vector loaded once
into per-lane registers and reused across both rows. Ours processes
1 row per simdgroup; the same 2560-element X-vector is reloaded per
row from cache. With our 8sg / 8-rows-per-TG geometry, that's ~2× the
X-cache traffic of llama.cpp's 2sg / 4-rows-per-TG, which matches our
measured 47% / 28% peak utilization on `q4k_ffn_gate_up` /
`q4k_qkv_proj` (the two biggest decode costs).

**Approach** (mirrors llama.cpp `kernel_mul_mv_q4_K_f32`):

1. Each simdgroup handles 2 output rows (`NR0 = 2`).
2. X-vector slice loaded once into `xl[16]` per lane.
3. For each of 2 rows: separate `sumf[2]` accumulator running the
   per-element FMA against the same `xl[16]`.
4. Two `simd_sum` calls at the end, two row-writes.

**Caveats** to watch:
- Auto-memory note from 2026-04-19: "N_DST=2 caused ~10% regression,
  N_DST=4 caused 24× regression (register spilling)". That earlier
  attempt likely **didn't share the X-vector** across rows — it just
  doubled the per-thread register footprint. The win in llama.cpp
  comes from the **shared X load**, not from naively doubling NR0.
- Verify register count via Xcode's Metal compiler diagnostic
  (`MTLLibrary.functionInfo.maxThreadsPerThreadgroup`) before shipping.
- Inner FMA chain becomes 2 chained FMAs per (lane, element) — same
  total work, but compiler must keep both `sumf[0]` and `sumf[1]` in
  registers without spilling.

**Validation**:
- Kernel-level parity test against current `q4k_ffn_gate_up` on
  synthetic data (cos ≥ 0.9999 — same Q4_K math, just multi-row dispatch).
- `arch_golden_gemma3_4b_gpu` continues to emit "**Paris**".
- `decode_consistency_gemma3_4b_2steps` continues to pass.

**Expected**: 187 GB/s → ~280 GB/s on `q4k_ffn_gate_up` → 5.4 → 3.5 ms/tok
across 34 layers → **+10-15 tok/s end-to-end** on Gemma 3 4B.
Apply the same pattern to `q4k_qkv_proj` (114 → 200 GB/s → +20 tok/s).
Stretch goal: **~95-100 tok/s, ollama parity**.

### G-3 — Flash-attention-style fused attention kernel (HIGH PRIORITY)

**Status**: Open. Larger lift than G-2 but orthogonal — attacks
**dispatch overhead** (~1.0 ms/tok savings) rather than per-kernel
utilization.

**Current decode dispatch chain per layer**: ~11 dispatches × 34
layers = ~374 dispatches/tok × ~5 µs each = **1.87 ms/tok overhead**.
Llama.cpp's flash-attention path collapses RoPE + QK_norm + KV_append +
KV_attend + O_proj fragments into 1-2 dispatches → ~6-7 per layer ×
34 = ~200/tok ≈ 1.0 ms overhead. **~0.85 ms/tok recoverable**.

**Approach** (mirrors llama.cpp `kernel_flash_attn_ext_*`):

1. Single fused kernel takes Q, K, V (already projected and RoPE-rotated),
   and the KV cache. Computes `softmax(QK^T / √d) · V` in one pass.
2. Tile over Q heads × KV blocks; each TG handles one Q head's softmax
   row, accumulating against the V tile in registers.
3. Online softmax (re-normalising incrementally) — avoids the
   per-position Q output allocation our current `kv_attend` materializes.

**File**: `crates/larql-compute/src/metal/shaders/fused_attention.rs`
already exists as a stub — flesh out using llama.cpp's
`kernel_flash_attn_ext_q4_K_f32` as the template (templated over Q
quant type, K head_dim, V head_dim).

**Validation**:
- Per-kernel parity test against current per-stage chain on synthetic
  Q/K/V/cache (cos ≥ 0.9999).
- `arch_golden_gemma3_4b_gpu`, `decode_consistency_gemma3_4b{,_2steps}`
  continue to pass.
- Wider sweep across Gemma 4 31B dense / 26B-A4B (different head
  geometries — global vs sliding-window layers, different head_dim).

**Expected**: -0.85 ms/tok dispatch overhead → **+5-8 tok/s end-to-end**
on Gemma 3 4B.

**Sequencing**: G-2 first (smaller, more bounded), G-3 second
(builds on G-2's NR0 understanding plus the existing
`fused_attention.rs` stub). Both together project to **95-105 tok/s
on Gemma 3 4B** (full ollama parity).

### G-3 — Dispatch-count reduction (✅ first fusion validates the model, 2026-05-01)

**First fusion shipped — `qk_norm_rope_fused`**:
`shaders/qk_norm_rope_fused.rs` collapses `qk_norm_qk` +
`rope_at_pos_batched_qk` into one kernel (each TG handles one head:
RMS-norm → weight scale → in-place RoPE rotation, with one
`threadgroup_barrier` between the norm and rotate phases). Opt-in via
`LARQL_FUSED_QK_NORM_ROPE=1`.

**Measured GPU-only timing** (n=10 each, on Gemma 3 4B M3 Max):

```
                     GPU median   CPU median   Wall median
FUSED QKN+ROPE       10.35 ms     0.55 ms      10.85 ms
BASELINE             10.45 ms     0.70 ms      11.08 ms
─────────────────────────────────────────────────────────────
SAVINGS              -0.10 ms     -0.15 ms     -0.23 ms ✓
```

The 0.23 ms/tok savings matches the theoretical 1-dispatch-saved ×
34-layers × ~7 µs estimate exactly. Splits cleanly into ~0.10 ms GPU
(less inter-dispatch latency in the cmd buffer) and ~0.15 ms CPU
(one fewer `set_compute_pipeline_state` + buffer-bind + dispatch
encode per layer).

`arch_gemma3_4b_gpu` produces "Paris" — bit-equivalent to the
production chain.

**Validation that the diagnosis is right**: the predicted savings
landed exactly where calculated, unlike G-1 (`F16_ACC` no-win), G-2'
(`GATE_UP_COOP` no-win), G-2 (`GATE_UP_NR2` -3% regression). This
confirms dispatch-count was the real bottleneck.

**Second fusion shipped — `residual_norm_store` in post_norms branch**:
The post_norms decode path (Gemma 3/4) was using two dispatches —
`residual_norm` then `residual_add` — when `residual_norm_store`
already does both in one kernel for the `!post_norms` branch.
Routing the post_norms branch through `residual_norm_store` is
mechanically the same fusion as the QK-norm+RoPE one. Saves another
~0.23 ms/tok. Now always-on (no env flag) since the kernel was
already battle-tested on the !post_norms path.

**Third fusion shipped — `post_attn_residual_norm_store`**:
Triple-fusion (post_attn_norm + residual + ffn_norm + h_post_attn
store) into one kernel doing 2 sequential RMS reductions per TG.
`shaders/post_attn_residual_norm_store.rs` + opt-in env
`LARQL_FUSED_POST_ATTN_NORM=1`. Math verified — `arch_gemma3_4b_gpu`
emits "Paris". **Bench result**: end-to-end 70-72 tok/s, ~0.05 ms
savings on top of stacked-2 — real but below thermal-noise floor.
The 2 RMS reductions in one TG add compute density that partially
offsets the dispatch overhead saved. Net: smaller win than the
prior two fusions; kept opt-in for completeness.

**Stacked GPU-only timing summary** (cold-state, 5 samples each):

| Configuration | GPU median | Δ vs baseline |
|---|---|---|
| Baseline (all unfused, post-2026-05-01 lm_head v5) | ~10.45 ms | — |
| + `LARQL_FUSED_QK_NORM_ROPE=1` | ~10.35 ms | -0.10 ms |
| + `residual_norm_store` (always-on) | ~10.07 ms | -0.38 ms |
| + `LARQL_FUSED_POST_ATTN_NORM=1` | ~10.02 ms | -0.43 ms |
| + `LARQL_FUSED_POST_FFN_NORM=1` | **~9.67 ms** | **-0.78 ms** |

**End-to-end tok/s** (Gemma 3 4B, 30 tokens, warm GPU):

| Path | Sustained tok/s |
|---|---|
| Pre-fix Metal (wrong output) | ~78 |
| v5 lm_head fix (correctness) | 71-72 |
| + 2 fusions stacked | 73 |
| + 3 fusions stacked | 71-72 (in noise) |
| + 4 fusions stacked (env-gated) | 74-75 |
| **All 4 fusions default-on** (shipped 2026-05-01) | **72-74** |
| Ollama gemma3:4b | 96-104 |

**Default-on shipped state** (no env vars needed): all four fusions
land their measured savings without flag friction. End-to-end
~72-74 tok/s sustained, generates "Paris" correctly. Opt-out flags
still wired (`LARQL_FUSED_QK_NORM_ROPE=0`, `LARQL_FUSED_POST_ATTN_NORM=0`,
`LARQL_FUSED_POST_FFN_NORM=0`) for diagnostic A/B if regressions
ever surface. The fifth fusion (Q6_K geglu+down) remains broken
and dead-code — needs kernel-level parity test against
`cpu/ops/q4_common::q6k_matvec` to localise the bug before re-engaging.

**Fourth fusion attempt — `q6k_geglu_gelu_tanh_down_cached`** (❌ both
the new cached kernel AND the existing production
`q6k_geglu_gelu_tanh_down` produce wrong output on
gemma3-4b-q4k-v2 — model collapses to "The" and stops at first decode
step). The prior memory claim "Q6_K fused kernels are
parity-tested" no longer holds against the current
`interleaved_q4k.bin` layout — likely the kernel's Q6_K block-byte
offsets drifted vs the writer in `format/weights/write_q4k` at some
point. Real fix needs a kernel-level parity test against
`cpu/ops/q4_common::q6k_matvec` reference on synthetic data, then a
re-route. Kernel and pipeline kept registered as dead code; env var
`LARQL_FUSED_Q6K_DOWN` is a no-op until the underlying bug is
diagnosed. See `shaders/q6k_geglu_gelu_tanh_down_cached.rs`.

**Remaining gap to 80 tok/s** (~3 more fusions of similar mechanic
needed):

**Realistic savings**: ~140 dispatches/tok × ~7 µs avg = **~1 ms/tok**
end-to-end → projects to **77-80 tok/s**. Smaller than the original
3.5 ms gap but the only one of G-1..G-3' the corrected diagnosis
actually supports.

**Current per-layer dispatch count** (~10-11 dispatches × 34 layers):
1. fused input_norm + QKV proj (1)
2. QK_norm (1)
3. RoPE batched Q+K (1)
4. V_norm (Gemma 4 only) (0-1)
5. KV append (1)
6. KV attend (1)
7. O proj (1)
8. post_attn residual + ffn_norm (fused) (1)
9. gate + up (fused) (1)
10. GEGLU (1)
11. down (1)
12. post_ffn residual (1)

**Where to fuse** (in priority order, smallest scope first):
- Fuse `QK_norm` + `RoPE` + `V_norm` into one batched kernel
  (reads/writes Q,K,V buffers — no inter-dispatch round-trip).
  Saves ~2 dispatches/layer × 34 = ~68 dispatches/tok.
- Fuse `KV append` + `KV attend` (`kv_attend` already reads cache;
  could append the new K/V row in the same kernel before attending).
  Saves 1 dispatch/layer × 34 = 34/tok.
- Fuse `GEGLU` + `down`: existing `q4k_geglu_silu_down` /
  `q4k_geglu_gelu_tanh_down` kernels exist but are disabled
  (`encode_ffn.rs::use_fused = false` per a NaN finding on certain
  Q4_K-down configs). Re-test on **gemma3-4b-q4k-v2 (f16 down)**
  where the NaN issue doesn't apply — the fused-down kernel only
  fires when `down_format == Q4_K`, so f16-down vindexes already
  go through the slow path; the gate is empty for them. **G-FFN-1**
  (separate sub-item): rebuild the fused-down kernel for f16 down
  to actually engage. Saves 1-2 dispatches/layer × 34 = 34-68/tok.

**Total savings if all three land**: ~140 dispatches × 7 µs ≈ 1 ms.
Combined with no-loss retention of the v5 lm_head fix, **end-to-end
projection: ~77-80 tok/s**, closing ~1/3 of the gap to ollama.

The original "G-3 = full flash-attention" sequencing was an
overestimate — flash-attn would also need the per-position softmax
re-norm (online softmax) which is a non-trivial precision puzzle for
Gemma 3's softcapped attention logits. The smaller fusion items above
are higher-confidence, lower-risk, and stack toward the same goal.

### G-3' — DEPRECATED entry kept for context (full flash-attention)

After three failed kernel optimizations (`F16_ACC`, `GATE_UP_COOP`,
`GATE_UP_NR2`) — all targeting per-kernel ALU/cache that the
kernel-isolated profiler suggested were bottlenecks — followed by
in-pipeline GPU timing showing our per-dispatch time is already
competitive (~30 µs avg), the picture is now clear: **the gap to
ollama is dispatch count, not per-kernel speed**.

```
                     dispatches/tok    avg µs/dispatch    total
  larql              ~340             ~30 µs            ~10.5 ms
  ollama (est.)      ~200             ~40 µs             ~8.0 ms
                     ────────────────────────────────────────────
  diff               -140             slower per         +2.5 ms
```

So **G-3 (flash-attention fusion)** is the right work item — it
collapses 5-6 attention dispatches per layer (RoPE + QK_norm + V_norm
+ KV_append + KV_attend + sometimes O_proj) into 1-2 dispatches.
Saves ~140 dispatches/tok regardless of per-kernel GB/s.

The earlier "G-3 builds on G-2's NR0 understanding" sequencing note
was wrong; G-2 didn't move the needle so G-3 should go first.

### G-5 — Memory access pattern audit (deferred)

**Status**: Open. Should run before any further kernel rewrites.

**Approach**: Use Xcode's GPU frame capture / Metal Profiler on a
single decode token, focused on `q4k_ffn_gate_up` and `q4k_qkv_proj`.
Look at:
- L2 cache hit rate per dispatch (low = scattered access; high = the
  diagnosis is wrong about memory being the bottleneck).
- Concurrent threadgroup count vs theoretical (low = register
  pressure or threadgroup-mem capping occupancy).
- Memory access stall events on the FMA chain.

The output should distinguish (a) scattered access pattern hurting
prefetch, (b) low occupancy hiding latency poorly, (c) actually
ALU-bound but the existing in-kernel ALU isn't the redundant dequant.

Without this, optimization is guess-and-check. Kernel-isolated GB/s
on `metal/diag/kernel_profile.rs` doesn't predict end-to-end wins on
Apple Silicon (G-1 and the prior `LARQL_F16_ACC=1` attempt both
demonstrated this).

### G-4 — Flash-attention-style fused attention kernel

**Status**: Open. Larger lift, separate from G-1..G-3 / G-5. Promoted
toward the top of the list because it eliminates dispatch overhead
(orthogonal to per-kernel utilization), so it should win regardless
of what G-5 finds about the matmul kernels.

Per-token attention currently dispatches as:
- `q4k_qkv_proj` (Q + K + V projection)
- `qk_norm` (Gemma 3/4)
- `rope_at_pos`
- `kv_append`
- `kv_attend` (the actual `softmax(QK^T)V`)
- `q4k_matvec` (O projection)

Six dispatches per layer × 34 layers = 204 dispatches per token, each
costing ~5-8 µs scheduling overhead = 1-1.6 ms/tok in pure dispatch
time. A flash-attention-style fused kernel (`fused_attention.rs` is a
stub) would collapse RoPE+QK norm+append+attend into one or two
dispatches, saving ~0.5-1 ms/tok dispatch overhead plus the per-stage
buffer round-trips.

**Expected**: +5-10 tok/s end-to-end after G-1..G-3 are in place.

---

## Status

The four KV-cache engines shipped in `engines/kv_engines/` all reach ~93-95 tok/s
on Gemma 3 4B using the Metal Q4K path (matching Ollama within 6%). See bench:

```
larql bench gemma3-4b-q4k --engine markov-rs,unlimited-context,turbo-quant,apollo
```

---

## P0: Mechanistic hooks (lazarus parity)

Driver: replace chuk-mlx as the engine for `chuk-mcp-lazarus`. Lazarus has 77
inference-time MCP tools (capture, ablate, patch, steer, probe, DLA, KV
surgery). Larql today only writes to weights (MEMIT, KNN, LQL) — it has no
mid-forward inspection/intervention API. The whole tool surface collapses to
one missing primitive: a programmatic forward-hook system.

### M1 — `LayerHook` trait + CPU plumbing (read + write)
**Status**: In progress
**File**: `forward/hooks.rs` (new), `forward/layer.rs`, `forward/trace.rs`

Trait shape:
```rust
pub trait LayerHook {
    fn on_pre_layer(&mut self, layer: usize, h: &Array2<f32>) {}
    fn on_post_attention(&mut self, layer: usize, h: &mut Array2<f32>) {}
    fn on_attention_weights(&mut self, layer: usize, w: &AttentionWeights) {}
    fn on_ffn_activation(&mut self, layer: usize, gate: &Array2<f32>) {}
    fn on_post_layer(&mut self, layer: usize, h: &mut Array2<f32>) {}
}
```

Insertion points in `run_layer_with_capture`: pre-layer (h entering),
post-attention (`h_post_attn`, `&mut`), FFN gate activation (`activation`),
post-attention-weights (`attn_weights`), post-layer (`h_out`, `&mut`).

The `&mut` on post-attention and post-layer is what unlocks the entire
intervention surface — ablation, steering, patching, subspace surgery are all
just `LayerHook` impls.

Plumbing strategy: `run_layer_with_capture` and `trace_forward_full` grow an
optional `&mut dyn LayerHook` parameter. Existing call sites pass `None`
(zero overhead — noop when absent). Hot generation paths in `predict.rs`
remain unchanged for slice 1; M6 wires hooks into the Metal `generate` path.

### M2 — Built-in hooks
**Status**: Not started
**File**: `forward/hooks.rs`

- `NoopHook` — never fires, used by tests.
- `RecordHook { layers: HashSet<usize> }` — captures pre/post-layer residuals
  and FFN activations; replaces the file-output path of `capture_residuals`.
- `ZeroAblateHook { layers, positions }` — zeros residual at requested coords.
- `SteerHook { vectors: HashMap<usize, (Array1<f32>, f32)> }` — adds α·v at
  specified layer's `on_post_layer`.

### M3 — Activation patching
**Status**: Not started — blocked on M1
**File**: `forward/patching.rs` (new)

Two-pass primitive: pass 1 with a `RecordHook` collects the donor residual at
(layer L, pos p) from prompt A; pass 2 runs prompt B with a `PatchHook` that
overwrites the same coords. This is the building block for `full_causal_trace`
(2D position × layer grid) — lazarus's flagship causal tool.

### M4 — Full logit lens
**Status**: Not started
**File**: `forward/predict/dense.rs`

Today: `logit_lens_top1(layer)` returns one token. Add:
- `logit_lens_topk(layer, k) -> Vec<(u32, f32)>`
- `track_token(layer, target_id) -> f32` — log-prob of a specific token at
  a specific layer.
- `track_race(layers, k) -> Vec<Vec<(u32, f32)>>` — top-k per layer in one
  pass for streaming top-k diagrams.

All three project the same captured residual through final norm + lm_head; no
new forward passes.

### M5 — KV cache surgery
**Status**: Not started
**File**: `attention/decode.rs:KvCache`

Lazarus `prefill_inject` and `kv_inject_test` need to lift K/V from one cache
into another. Add `get_layer(layer) -> (&[f32], &[f32])`,
`set_layer(layer, k, v)`, `clone_at_position(other, layer, pos_range)`.

### M6 — Hooks during multi-token generation
**Status**: Shipped
**File**: `forward/kv_generate.rs::generate_cached_hooked`,
`crates/larql-python/src/walk.rs::generate_with_hooks`

Final design: **hooks-on-CPU, Metal-stays-fast**. Lazarus-style mech interp
during multi-token generation goes through `generate_cached_hooked` on the
CPU KV-cache path; the Metal-fast `layer_graph::generate::gpu::generate*`
remains hook-free.

Why not propagate hooks into the Metal path: the Metal `decode_token` and
`prefill_q4` calls are end-to-end fused kernels that handle every layer in
one dispatch. Threading hooks in would require either CPU readback per
layer (kills the fusion benefit) or a parallel kernel surface that splits
on layer boundaries (kills the fast path even when no hook is registered).
Mech-interp tools care about correctness over throughput, so paying the
CPU-path cost when hooks are active is the right trade.

Interface mirrors `trace_forward_full_hooked` — same `LayerHook` trait;
`on_pre_layer`, `on_post_attention(&mut)`, `on_post_layer(&mut)` fire on
every layer of every step (prefill + each decode step).
`on_attention_weights` and `on_ffn_activation` do **not** fire on this
path — the production decode kernels don't capture those intermediates.
Use `trace_forward_full_hooked` for a single forward pass when you need
them.

Tests: `forward::kv_generate::tests` — noop matches baseline; record fires
on prefill + every decode step; α=5 steer changes generated tokens vs
baseline. Demo: `examples/mech_interp_demo.rs` § [7] shows
`baseline_ids = [12, 30, 10, 29]` vs `steered_ids = [4, 4, 4, 4]`.

### M7 — `W_E` / `W_U` + `project_through_unembed`

### M7 — `W_E` / `W_U` + `project_through_unembed`
**Status**: Not started
**File**: `forward/predict/dense.rs`, `lib.rs` re-exports

Lazarus tools `head_dla`, `decode_residual`, `embedding_neighbors` need
direct embedding/unembedding matrix access plus a "project this vector
through `W_U`, return top-k tokens" helper. Today both matrices are wrapped
in `VectorIndex` with no public accessor. Add `weights.embed_matrix()` and
`weights.unembed_matrix()` plus a free function `project_to_vocab_topk(vec, weights, k)`.

### M8 — pyo3 `PyLayerHook`
**Status**: Blocked on M1
**File**: `crates/larql-python/src/hooks.rs` (new)

Wrap a Python callable in a `PyLayerHook(PyObject)` that implements
`LayerHook`. Tensors crossed with `numpy.PyArray2<f32>` (zero-copy on
CPU path). MCP tools in lazarus are then just Python that registers a
hook and calls `infer()`.

---

## P0: Generation quality (blocks demo)

### Chat template — inference side
**Status**: Not started  
**Files**: `layer_graph/generate/gpu.rs`, `layer_graph/generate/cpu.rs`  
Read `tokenizer_config.json` from the vindex, parse the `chat_template` Jinja
field with `minijinja` (already in `Cargo.toml`), apply to the token sequence
before generation. `--no-chat-template` flag to bypass for base models or raw
prompts.

### EOS detection
**Status**: ✅ Done 2026-04-26 — see `layer_graph/generate/eos.rs`  
`EosConfig` reads `eos_token_id` (scalar or array) and `stop_strings` from
`generation_config.json`, layered on top of `BUILTIN_STOP_STRINGS` (covers
Gemma `<end_of_turn>`, ChatML `<|im_end|>`, Llama-3 `<|eot_id|>`/`<|eom_id|>`).
Wired into `generate_with_sampling` via `eos.is_eos(id, &decoded)`. Greedy
`generate` defaults to `EosConfig::builtin()` so existing callers Just Work.

### Token spacing / detokenisation
**Status**: ✅ Done 2026-04-26 — see `layer_graph/generate/detok.rs`  
`Detokenizer` keeps the cumulative ID buffer and emits only the freshly-grown
suffix on each `push`. Equivalent to llama.cpp `llama_token_to_piece` and HF
Python `decode_stream`. Handles HF leading-space (`▁`) for SP tokenizers and
multi-byte UTF-8 chars that straddle a token boundary. Demo at
`examples/detok_demo.rs` shows the bug ("thecapitaloffranceisparis") and the
fix ("the capital of france is paris").

### Token streaming
**Status**: ✅ Done 2026-04-26 — see `layer_graph/generate/gpu.rs`  
`generate_streaming(..., on_token: F)` fires `on_token(id, text, prob)` for
every emitted token, including the first (which comes out of prefill). Uses
`Detokenizer::push` so streamed text preserves HF leading-space spacing.
`generate_with_sampling` is a thin wrapper passing a no-op closure so
non-streaming callers are unaffected. Demo at `examples/streaming_demo.rs`
prints tokens live with stdout flushing.

### Sampling
**Status**: ✅ Done 2026-04-26 — see `layer_graph/generate/sampling.rs`  
`Sampler` + `SamplingConfig` covers greedy / temperature / top-k / top-p with
optional `seed` for reproducibility. Two paths: full-vocab `sample(logits)`
for the OpenAI-API logprob future, sparse `sample_from_topk(hits)` for the
production hot path. Wired into `generate_with_sampling`. Sparse-path
overhead is <2µs/call at top-K=64 (<0.02% of decode budget). CLI flags
(`--temperature`/`--top-p`/`--top-k`) are still owned by `larql-cli`.

### Multi-turn KV state
**Status**: ✅ Done 2026-04-26 (token-buffer) — see `layer_graph/generate/chat_session.rs`  
`ChatSession` owns the running token buffer with whole-turn eviction at
`max_context`. Pluggable `TurnRenderer` covers Gemma / ChatML / Llama-3
templates. The most recent turn is never dropped — eviction is a no-op
when only one turn remains, so a long single prompt is preserved over
silently truncating. `examples/chat_demo.rs` runs a 3-turn conversation.

True KV carryover across turns (so prefill on turn N+1 only processes
the new tokens) is a follow-up — the API surface is in place; it's an
internal optimisation.

### Gemma 3 4B regression smoke test
**Status**: ✅ Done 2026-04-26 — see `tests/test_gemma3_smoke.rs`  
Loads vindex from `LARQL_VINDEX_PATH`, runs single-token greedy generation
on `"The capital of France is"`, asserts first token (trimmed) equals
`"Paris"`. Gated `#[ignore]`; `CI_INTEGRATION=1` flips to fail-loud when
the vindex env isn't set so CI can require the test rather than silently
skip. Defaults configurable via `LARQL_SMOKE_PROMPT` / `LARQL_SMOKE_EXPECTED`.

---

## P0: MoE inference completions

### MoE-aware CPU forward pass
**Status**: Not started  
`predict_q4k` / `WeightFfn::forward` has no MoE branch. Wire `cpu_moe_forward`
(already in `larql-compute/src/cpu/ops/moe.rs`) into `forward/layer.rs`.

### Wire `RouterIndex` client-side
**Status**: Not started  
`larql-vindex/src/index/router.rs` exists but is not connected to the forward
pass. Connect so MoE router runs locally against the vindex before dispatching.

---

## P0: CPU MoE expert path — close the bandwidth-bound gap (Gemma 4 26B-A4B)

**Why this is P0**: The grid currently runs at **2.3 tok/s** loopback on 26B-A4B
(2 shards same M3 Max). Server compute = 95% of token wall time (250 ms/tok);
network = 2%. Theoretical CPU bandwidth floor for 4B active params at Q4_K is
~10 ms/tok = ~100 tok/s on M3 Max LPDDR5X (~400 GB/s peak), conservatively
~25 tok/s at 50 GB/s effective. We are **40-100× over the bandwidth floor** —
the gap is structural in the CPU expert path, not in kernel quality. Metal
experts measured 3.7× (→ 9.4 tok/s) but stay shipped-off pending the
`inter=704` accuracy bug (see `larql-compute/ROADMAP.md`). Closing this gap
unblocks shipping CPU-only without waiting on the Metal kernel fix and lifts
the Metal-on path proportionally once that lands.

**Target**: 25 tok/s CPU-only on Gemma 4 26B-A4B grid loopback (~10× current).

### M-CPU-1 — stop the `to_vec()` copy on cache hit
**Status**: ✅ Done 2026-05-01  
**File**: `crates/larql-compute/src/cpu/ops/moe/expert.rs`  
`run_single_expert_into` was doing `let gate_up_w_f32 = v.to_vec()` on every
call, copying ~12 MB *even on cache hit*. Replaced with an
`Option<ExpertF32>` (Arc) held for the call's lifetime; `gate_w` / `up_w`
slice into the cached payload directly. No behavioural change; tests pass.

### M-CPU-2 — K=8 per-layer experts run in parallel + fold/reduce accumulator
**Status**: ✅ Done 2026-05-01  
**File**: `crates/larql-server/src/routes/expert.rs`  
Confirmed the production gRPC path (`run_experts_cpu_batch`) already uses
rayon `par_iter` over the K active experts with per-rayon-thread
`ExpertScratch`. Refactored from `collect Vec<(Vec<f32>, weight)> + serial
sum` to `par_iter.fold(per-worker hidden-acc).reduce(...)`, eliminating the
per-expert 11 KB Vec allocation (~2.7 MB/token at 30 layers × K=8). Also
parallelised the HTTP `handle_expert_batch` endpoint (was serial `iter().map`).

### M-CPU-3 — `LARQL_MOE_CACHE_ENTRIES` default raised 64 → 256
**Status**: ✅ Done 2026-05-01  
**File**: `crates/larql-compute/src/cpu/ops/moe/cache.rs`  
Default cap covers one full token's working set (30 layers × top-K=8 = 240
experts) with headroom. Eviction-driven p99 outliers gone (11.62 → 2.42 ms
on `cpu_moe_forward` floor). RSS cost: +2 GB per shard (9.7 → 13.6 GB on
single-shard 26B-A4B bench). Long-term answer is M-CPU-4 (kill the cache
entirely via direct Q4_K matvec); cap=256 is the right default until then.

### M-CPU-4 — NEON-vectorised Q4_K matvec (load-bearing item)
**Status**: ✅ Done 2026-05-01 — measured **8.6× sweep speedup**  
**File**: `crates/larql-compute/src/cpu/ops/q4k_q8k_dot.rs` (new module);
wired in `expert.rs::run_single_expert`, `expert.rs::run_single_expert_q4k_q8k_into`,
`forward.rs::cpu_moe_forward`, `routes/expert.rs::run_experts_cpu_batch`.  
New isolated module mirrors llama.cpp's `ggml_vec_dot_q4_K_q8_K`:

- `quantize_x_to_q8k(x)` → per-256-element absmax + i8 + per-32-subblock i16 sums.
- `q4k_q8k_matvec_scalar` — scalar reference, integer dot math.
- `q4k_q8k_matvec_neon` — aarch64 SDOT inner loop (16 i8 × i8 → 4 i32 lanes
  per instruction). Implemented via stable `core::arch::asm!` because
  `core::arch::aarch64::vdotq_s32` is still unstable on Rust 1.91 (gated
  behind `stdarch_neon_dotprod`, rust-lang/rust#117224).

Test rig: Q8_K quantiser round-trip; scalar Q4_K×Q8_K vs cached-f32 path
within Q8 quant noise; multi-block matvec parity; **NEON vs scalar bit-exact**
(`to_bits()` equality on non-trivial sin/cos input); zero-dim and short-buffer
edge cases. 7 new tests, all passing.

The Q4_K direct path is on by default for Q4_K weights; `LARQL_DISABLE_Q4K_DIRECT=1`
falls back to the BLAS-on-cached-f32 path for kernel-debug A/B comparison.

Bench measurements (Gemma 4 26B-A4B, M3 Max, single-shard loopback):

| Metric | Baseline (cap=64) | M-CPU-1/2/3 (cap=256) | + M-CPU-4 (NEON Q4_K) | Total Δ |
|---|---|---|---|---|
| `forward_moe` warm 1-layer HTTP RTT | 2.53 ms | 2.43 ms | **0.95 ms** | **2.7×** |
| `cpu_moe_forward` warm floor | 3.52 ms | 1.94 ms | **0.39 ms** | **9.0×** |
| `cpu_moe_forward` p99 | 11.62 ms | 2.42 ms | **0.50 ms** | **23×** |
| **30-layer sweep** | **221 ms** | 205 ms | **25.6 ms (0.85 ms/layer)** | **8.6×** |
| Steady RSS | 11.4 GB | 13.6 GB | **10.5 GB** | -8% |

The sweep at 25.6 ms projects to ~25-30 tok/s end-to-end on the gRPC grid
(vs 2.3 tok/s baseline = ~10-13× end-to-end). RSS dropped below baseline
because the f32 dequant cache is largely inert in the new path —
direct-Q4K reads straight from mmap.

Follow-ups (if further perf needed): (1) shrink `LARQL_MOE_CACHE_ENTRIES`
default back to 64 or 32 once the BF16 fallback path is removed (cache only
serves legacy BF16 vindexes now); (2) reuse per-rayon-thread scratch for the
`gate_out` / `up_out` / `act_q8k` heap allocs in `cpu_moe_forward`'s
direct-Q4K branch (currently per-call); (3) wire AVX2 dot-product equivalent
for x86 hosts (`_mm256_maddubs_epi16`).

### M-CPU-5 — bench harness + per-fix tok/s attribution
**Status**: ✅ Done 2026-05-01  
**File**: `crates/larql-server/examples/bench_expert_server.rs` (+ pre-existing
`unit_filter` fixture compile fix; two-shard mode has a separate pre-existing
expert-127 off-by-one).  
Single-shard bench on `output/gemma4-26b-a4b-q4k.vindex` (M3 Max, 2026-05-01):

| Metric | cap=64 | cap=256 (new default) | Δ |
|---|---|---|---|
| `forward_moe` warm 1-layer HTTP RTT | 2.53 ms | 2.43 ms | -4% |
| `cpu_moe_forward` warm floor (mean) | 3.52 ms | **1.94 ms** | **-45%** |
| `cpu_moe_forward` p99 (eviction outliers) | 11.62 ms | 2.42 ms | **-79%** |
| 30-layer sweep | 221 ms | 205 ms | -7% |
| Steady RSS | 11.4 GB | 13.6 GB | +19% |

The per-call floor improvement is real; the sweep regression vs the
ROADMAP-published 56 ms (from 2026-04-26) is on current code regardless of
cap, indicating a code drift between then and now that should be bisected
separately. The point of the bench: it falsifies "more cache = more tok/s"
as the path to target, and confirms M-CPU-4 (NEON direct-Q4K, no f32 cache)
as the only structural answer.

### M-CPU-6 — Bottleneck-driven follow-ups (post-NEON profiling round)
**Status**: ✅ Done 2026-05-01  
**Files**: `q4k_q8k_dot.rs`, `cpu/ops/q4_common.rs::f16_to_f32`,
`moe/expert.rs`, `moe/forward.rs`, server `routes/expert.rs` +
`larql-inference/ffn/moe_remote.rs`.

After M-CPU-1..4 landed, samply (`/usr/bin/sample bench_expert_server 30`)
identified the next bottlenecks:

1. **f16-to-f32 was calling `__powisf2`**.  `2.0f32.powi(exp - 15)` lowered
   to a libcall; ~11 M decodes/token at 26B-A4B sizes routed through the
   software powi.  Replaced with pure-integer bit-manipulation
   (`f16_to_f32`).  Bit-exact for all 65,536 inputs (test:
   `f16_to_f32_bit_exact_for_all_inputs`).  Removed the bl from the
   kernel — but wall-clock barely moved, which DIAGNOSED the kernel as
   already DRAM-bandwidth bound (the powi work was hidden in memory
   stalls).

2. **Reusable Q8_K activation buffer (`act_q8k`) in `ExpertScratch`**.
   The per-expert activation Q8_K quantisation was allocating a fresh
   `Q8KActivation` per call; ~5% of calls hit a 150 µs allocator slow
   path that dragged par_iter wall up.  Added `act_q8k` field +
   `quantize_x_to_q8k_into` API + `Q8KActivation::with_capacity`.
   `forward_moe` p99 dropped 23% (1.38 → 1.06 ms).

3. **`cpu_moe_forward` refactored to use thread-local `ExpertScratch` via
   rayon `fold/reduce`**.  Eliminates the per-expert
   `Vec<f32>` allocs in the in-process MoE path AND deduplicates the
   kernel logic (now goes through `run_single_expert_q4k_q8k_into`
   instead of an inlined copy).

4. **`run_single_expert` (HTTP single-expert entry) now uses thread-local
   scratch on the Q4_K path**.  K=8 calls per layer no longer
   re-allocate gate_out / up_out / act / act_q8k; only the final
   returned `Vec<f32>` is allocated.

5. **New `/v1/experts/layer-batch` endpoint + wire format**.  Ships ONE
   residual + K (expert_id, weight) pairs per call (vs K identical
   residuals on the legacy `/v1/expert/batch` path).  Server applies
   `pre_experts_norm` once + Q8_K quantises h_norm once + fans out the
   K experts via `run_experts_cpu_batch`.  `RemoteMoeBackend::forward_moe`
   updated to use the new endpoint.  Saved K-1 redundant pre-norm + Q8K
   quantisations and ~2.6 MB/token of redundant residual on the wire.

6. **Tried fused gate+up matvec**.  Implemented `q4k_q8k_gate_up_into`
   with NEON SDOTs interleaved across both matrices.  Bit-exact parity
   test against back-to-back single-matvec calls (`q8k_gate_up_fused_matches_separate_matvecs`).
   Bench: ~4% slower on the 30-layer sweep.  M3 Max OoO engine
   already extracts ILP from the back-to-back independent matvec calls;
   manual interleaving adds register pressure and hurts the L1 prefetcher
   pattern.  Reverted the wiring; kept the function for future
   architectures where the trade may flip.

End-state bench (Gemma 4 26B-A4B, M3 Max, single-shard loopback):

| Metric | Baseline (cap=64) | M-CPU-1..4 | + M-CPU-6 (post-profile fixes) | Total Δ |
|---|---|---|---|---|
| `forward_moe` warm 1-layer HTTP RTT | 2.53 ms | 0.95 ms | **0.83 ms** | **3.0×** |
| `cpu_moe_forward` warm floor | 3.52 ms | 0.39 ms | **0.38 ms** | **9.3×** |
| **30-layer sweep** | **221 ms** | 25.6 ms | **24.2 ms (0.81 ms/layer)** | **9.1×** |
| Steady RSS | 11.4 GB | 10.5 GB | 10.5 GB | -8% |

Sweep at 24.2 ms projects to **~25-30 tok/s end-to-end on the gRPC grid**
(vs 2.3 tok/s baseline = ~10-13× end-to-end).  Path is now firmly
DRAM-bandwidth bound (~32 GB/s aggregate vs ~50-100 GB/s practical M3
Max CPU peak); further wins require structural changes (multi-row
matvec sharing super-block reads across output rows, prefetch
instructions ahead of SDOT loads, or simply waiting on the Metal MoE
expert kernel fix to land for an additional ~3.7× via GPU dispatch).

---

## P0: Engine performance parity

### TurboQuant Metal K/V checkpoint compression
**Impact**: Reduces boundary checkpoint from 278 KB → 36 KB/window (7.7×) for long contexts.  
**Status**: TurboQuant runs at Metal speed. Compressed boundary checkpoints require
Metal K/V read-back. Add `backend.get_kv_last_position(layer)` to the Metal backend.

### Apollo `prefill_to_layer` — true layer-skip
**Impact**: ~20% faster per step in compressed path.  
**Status**: `forward_from_layer` ships; K/V seeding at `crystal_layer` is a follow-up.

### Apollo store builder
**Impact**: Currently requires pre-built NPY/NPZ files.  
**Status**: Not started. `ApolloEngine::build_from_document(weights, tokenizer, tokens)`.

---

## P0: Evaluation parity (blocks architecture claims)

larql is a research engine for novel architectures (WalkFfn, vindex KV engines, gate
KNN, layer-skip via Apollo). To show an architecture is competitive we need to run
the same eval harnesses other engines run — otherwise we are only ever comparing
synthetic prompts to synthetic prompts. The items below build on the generation-quality
P0 above (sampling, streaming, chat templates, multi-turn KV); without those, none
of the harnesses load at all. Goal is parity for fair evaluation, not feature
parity for its own sake.

### Per-position logprobs / top-k logprobs
**Status**: Not started  
**Files**: `forward/predict/raw.rs`, expose via `lib.rs`  
Add `forward_logprobs(weights, token_ids, target_ids) -> Vec<f32>` returning
per-position log-likelihood of `target_ids[i]` given prefix `token_ids[..i]`.
Also expose top-k logprobs from `forward_raw_logits`. lm-evaluation-harness and
most multiple-choice benchmarks (HellaSwag, ARC, MMLU, WinoGrande, PIQA) score
by sequence log-likelihood, not generation. Without this no likelihood-class
benchmark can run, so no architecture claim has a published comparator.

### OpenAI-compatible HTTP API
**Status**: Not started  
**Files**: `crates/larql-server/src/openai/` (new), thin wrapper over inference  
`larql-server` exposes `/v1/infer` and `/v1/walk`; eval frameworks (lm-eval-harness,
simple-evals, evalplus, AlpacaEval, swe-bench harnesses) plug into
`/v1/chat/completions` and `/v1/completions`. Add OpenAI-shape endpoints as a
wrapper over `generate` + sampling + chat-template rendering + logprob fields.
Unlocks every harness without per-harness adapters.

### Batch inference (independent prompts)
**Status**: Not started  
**Files**: `forward/predict/`, new `predict_batch`  
Distinct from continuous batching. Eval suites issue thousands of independent
prompts; serial execution makes a single benchmark run take hours-to-days. Add
`predict_batch(weights, prompts: &[Vec<u32>]) -> Vec<Vec<f32>>` that prefills each
prompt against the same weight mmap. Each prompt gets its own KV-engine instance,
so all four engines work unchanged.

### LoRA / adapter loading at runtime
**Status**: Not started  
**Files**: `forward/layer.rs`, `larql-models` weight loader  
Many arch papers ship LoRA-tuned variants (instruction-tuned on top of a base).
Without LoRA, larql cannot compare `WalkFfn` on `Gemma-3-4B-base` vs
`Gemma-3-4B-it` without re-quantising a merged model. Add
`WeightSet::with_lora(adapter_path)` wrapping `gate/up/down/q/k/v/o` matmuls as
`W·x + α·B(A·x)`. Stretch: composable adapter stack for ablation
(WalkFfn + LoRA-A vs WalkFfn + LoRA-B on the same base).

### Eval-harness smoke run
**Status**: Not started  
End-to-end test: run lm-eval-harness `hellaswag` (10 samples) against
`larql-server` and assert non-zero accuracy. Gate on `CI_INTEGRATION=1`. This
is what moves "we have logprobs" from a unit test to "harnesses actually plug in."

---

## P1: Eval-class coverage

Each item below unlocks a specific class of evaluation. Land in the order an arch
claim needs them — no need to do all up front. Prerequisite for all of them: the
P0 evaluation-parity stack above.

### Structured output / GBNF grammar / JSON Schema
**Status**: Partial — regex/grammar hook exists in `generate`; not wired to JSON
Schema or BNF.  
**Unlocks**: JSONSchemaBench, BFCL (function-calling leaderboard), any eval
requiring schema-conformant output.  
Apply a constrained-decoding mask over logits before sampling. Minimum viable:
GBNF parser (port from `llama.cpp` grammar.cpp); JSON Schema compiles to GBNF.

### Vision / multimodal forward
**Status**: Not started  
**Unlocks**: MMMU, ChartQA, DocVQA, multimodal subsets of larger suites.
Validates that WalkFfn and the four KV engines work on multimodal weights, not
just text.  
Gemma 3 (4B/12B/27B) and Llama 3.2 ship vision variants; vision-tower weights
are already in safetensors. Add image-embedding pipeline → token-mixing →
existing decoder forward. No new KV-engine work required (image tokens look
like text tokens to the decoder).

### Tool / function calling
**Status**: Not started — depends on chat templates (P0) + structured output
(P1 above).  
**Unlocks**: BFCL, ToolBench, AgentBench, any agent-style eval.  
Once the two prerequisites land this is template glue: parse tool-call markers
in the rendered chat template, emit structured calls via the constrained-decoding
path.

### Speculative decoding
**Status**: Not started  
**Why this matters for arch claims**: any "WalkFfn at X tok/s" comparison
against engines that ship speculative decoding (vLLM, TGI, llama.cpp `--draft`)
is misleading without it. Speculative decoding also interacts non-trivially with
gate KNN — draft and target may diverge on top-k feature selection, which is its
own arch question worth answering.  
**Path**: self-spec via `forward_from_layer` (early-exit verification) is the
cheapest entry; full draft-target spec is a follow-up.

### Trace capture during eval batches
**Status**: Partial — `trace_forward_full` works on single prompts.  
Extend to the batch + logprob path so mechanistic interpretability can use
eval-set inputs without re-running. This is what makes "we ran HellaSwag and
the WalkFfn-replaced layers behaved like X" a single-pass measurement.

---

## P1: Architecture coverage

### Wire v_shares_k into forward pass
**Effort**: Low — `v_shares_k()` already in larql-models; swap runtime check.

### Validate PLE end-to-end (Gemma 4 E2B)
**Effort**: Medium — config parsed; forward pass not yet wired.

### KV layer sharing for Gemma 4
**Effort**: Medium — `kv_shared_source_layer()` returns correct sources; cache allocation not yet sharing.

### Llama 3 / Gemma 4 engine validation
All four engines validated on Gemma 3 4B. Need empirical `cos h = 1.000000` validation on Llama 3 / Gemma 4.

### MarkovRS batched K/V recompute kernel
**Impact**: Eliminate 2000× FLOP overhead on CPU decode path.  
**Effort**: Medium (new Metal shader for `[W, hidden] @ [hidden, kv_dim]` Q4K projection).

---

## P1: Structure & file layout

From 2026-04-26 code review. All public APIs preserved; changes are internal re-organisation.

### High priority

**`ffn/remote.rs` (893 LOC) — split into `remote/`** ✅ Done 2026-04-26  
`ffn/remote/codec.rs` — binary codec, wire types, latency stats, codec tests.  
`ffn/remote/http.rs` — RemoteFfnConfig, RemoteWalkBackend, RemoteFfnError, HTTP tests.  
`ffn/remote/mod.rs` — thin re-export + protocol doc.  
No magic strings: `BINARY_CT`, `BATCH_MARKER`, `STATS_PATH`, `WALK_FFN_PATH` are named constants.

**`turbo_quant/mod.rs` → `turbo_quant/engine.rs`** ✅ Done 2026-04-26  
TurboQuantEngine + TurboQuant codec moved to `engine.rs`. `mod.rs` is a thin re-export of sub-modules + `pub use engine::{TurboQuantEngine, TurboQuant}`.

**`vindex/walk_ffn/mod.rs` → `walk_ffn/engine.rs`**  
Deferred: walk path submodules use `pub(super) impl WalkFfn` blocks that are
architecturally tied to `mod.rs` as the parent. Requires changing visibility to
`pub(in crate::vindex::walk_ffn)` across 6 files — low risk/reward compared to
other P1 items. Backlog.

**`layer_graph/predict.rs` (700 LOC) — split**  
Five `predict_*` variant functions sharing a shell. Extract to `predict/base.rs`
(shared embed→loop→logits shell) + `predict/variants.rs` (per-strategy overloads).

**`residual.rs` at crate root → `forward/norm.rs`**  
It's a collection of norm primitives used exclusively by the forward pass. Moving
it co-locates it with the other forward utilities (`ops.rs`, `layer.rs`).

**`capture.rs` at crate root → `trace/`**  
`InferenceModel` / `CaptureConfig` belong with the trace infrastructure.

### Medium priority

**Softmax in 5 locations — unify**  
`trace/vocab.rs`, `engines/accuracy.rs`, `ffn/moe_remote.rs`,
`layer_graph/logits.rs`, `forward/target_delta.rs` each have a private softmax.
Promote `engines/accuracy.rs::softmax` to `forward/ops.rs` (or `residual.rs`);
have the others `use crate::forward::softmax`.

**`embed_tokens_pub` / `run_attention_public` naming**  
The `_pub` suffix is redundant on public functions. Rename to `embed_tokens` and
`run_attention` or document why the suffix exists. `_pub` vs `_public` is also
inconsistent.

**`ApolloEngine` and `TurboQuantEngine` not re-exported at crate root**  
`MarkovResidualEngine` and `UnlimitedContextEngine` are re-exported; the other
two engines are not. Either export all four or none.

**`walker/` and `experts/` have no module-level docs**  
Add `//!` headers explaining purpose and entry points.

**`vindex/` module doc is vague**  
"Vindex integration" says nothing to a new reader. Expand to explain what the
vindex is and what this module provides.

### Low priority

**`forward` re-export block is 70+ items with no sub-grouping**  
Split into clearly commented groups: prediction, tracing, raw logits, analysis
(memit, target_delta, infer_patched).

**`trace as trace_decomposed` alias in `lib.rs`**  
Aliases a naming problem rather than fixing it. Rename the function itself.

**`RawForward` is an implementation detail in the public API**  
Users never construct `RawForward` directly; it's only returned by
`forward_raw_logits`. Consider whether it needs to be pub.

**`generate_cached*` in `forward/` vs `generate` in `layer_graph/`**  
Two generation APIs with similar names but different semantics (CPU KV-cache step
vs Metal fused pipeline). Add a clear doc comment on each explaining the difference.

---

## P1: Quality bugs (from 2026-04-26 review)

### `grid.rs` — hardcoded `eos_id = 1` is a real bug ✅ Fixed 2026-04-26
**File**: `layer_graph/grid.rs`  
Replaced `eos_id: u32 = 1` with `is_end_of_turn(tok_str.trim())` on both the prefill-exit
and decode-loop paths, matching all other generation code.

### Softmax duplicated in 5 locations ✅ Fixed 2026-04-26 (2 of 5)
**Files**: `trace/vocab.rs`, `engines/accuracy.rs` now use `pub use crate::forward::softmax`.  
Canonical implementation lives in `forward/ops.rs`, exported via `forward/mod.rs`.  
`ffn/moe_remote.rs` (in-place `&mut [f32]`), `logits.rs` (single-prob extractor),
`target_delta.rs` (Array1) remain local — different enough to not unify.

### `forward/ple.rs` hardcodes `1e-6` norm epsilon ✅ Fixed 2026-04-26
`1e-6` replaced with `arch.norm_eps()` for consistency.

### `grid.rs` undocumented `SKIP_MOE` env var ✅ Fixed 2026-04-26
Added `# Diagnostics` section to module doc.

---

## P1: Test coverage gaps

From 2026-04-26 coverage review (50.45% line coverage).

### Critical

**`markov_residual/` — zero tests across all 5 new files** ✅ Done 2026-04-26  
`store.rs`: clip_layer edge cases (no-window noop, at-limit, over-limit), memory_bytes, window_tokens.  
`engine.rs`: name, memory lifecycle, prefill→decode cycle, window clipping, multi-step shapes.  
`compute.rs`: recompute_kv shape/finiteness/RoPE shift, rs_prefill result shape + window, rs_decode_step position advance.

**`ffn/sparse_compute.rs` and `ffn/sparse.rs` — zero tests** ✅ Done 2026-04-26  
`sparse_compute.rs`: empty-features→zeros, single/multi-token shape, top-K ordering, dense-fallback equivalence, down-override effect.  
`sparse.rs`: name, all-layers shape/finiteness, top-k vs dense match, with_activation shapes.

**`ffn/graph_backend.rs` — zero tests** ✅ Done 2026-04-26  
Construction (layer count, empty layers), lookup_from_tokens (top-K limit, unknown layer, empty scores, out-of-range tokens), precompute_entity, save/load roundtrip.

**`layer_graph/` — 7 of 17 files untested** ✅ All 7 done 2026-04-26  
`dense.rs` — DenseLayerGraph shape/finiteness/capture, PerLayerGraph bounds.  
`walk.rs` — WalkLayerGraph all-layers, PipelinedLayerGraph in/out-of-range.  
`mod.rs` — trait dispatch, name distinctness.  
`prefill.rs` — CPU path: shape, finiteness, partial range, empty range, logit correctness.  
`template.rs` — detect_template (7 pure tests), TemplateUniverse build/get/total, GuidedWalkLayerGraph shape/finiteness.  
`pipeline_layer.rs` — build_arch_params param extraction, resolve_attn_weights None path, resolve_ffn_weights legacy stride slicing.  
`grid.rs` — error path: no Q4K mmap → `Err(BadResponse)`.  
Integration tests: `tests/test_layer_graph_integration.rs` — real vindex tests for prefill_with_kv, build_pipeline_layers, TemplateUniverse, GuidedWalkLayerGraph (all `#[ignore]`, run with `--ignored`).

### High priority

**`forward/ops.rs` — zero tests** ✅ Done 2026-04-26  
`dot_proj`: shape, identity-weight, value-correctness.  
`add_bias`: all-rows updated, shorter-bias safe, zero-bias noop.  
`apply_norm`: shape, finite output, offset produces different result.

**`forward/ple.rs` — zero tests** ✅ Done 2026-04-26  
precompute returns empty for non-PLE arch, apply_ple None/missing-weight guard paths,
output shape. Softmax tests moved here as a side-effect of unification.

**`engines/kv_engines/unlimited_context/extend.rs` — zero tests** ✅ Done 2026-04-26  
empty_prior shape, empty-tokens/wrong-prior-len → None, single/multi-token extend, kv_cache
row count, checkpoint = last-row, abs_start shifts RoPE, finite logits, chained extends.

### Medium priority

**GQA head grouping (`reps` parameter) not tested** ✅ Done 2026-04-26  
Three tests: output shape (4Q/2KV/reps=2), finiteness, and head-pair sharing — heads 0 & 1
sharing KV-head 0 produce identical output rows.

**RoPE missing property tests** ✅ Done 2026-04-26  
rope_base sensitivity, fraction=1.0 equals full-rope, offset=N matches sequential position N,
partial fractions 0.25/0.5/0.75 all finite.

**No synthetic end-to-end tests for `generate()`**  
`generate()` (Metal GPU path) is only tested with `#[ignore]` real-model tests.
Add a synthetic CPU-backend integration test using `make_test_weights()`.

---

## P2: Research

### Hybrid head caching (RS+CA)
95.5% of attention heads are static (cacheable). Would give ~180-370× compression
at 370K tokens — between TurboQuant (4×) and MarkovRS (287×) with near-exact accuracy.

### Graph Walk engine
FFN graph walk is proven (348K features, 34 layers, zero accuracy loss).
Full RS Graph Walk requires cracked attention (static head caching).
`GraphWalkEngine` would eliminate the forward pass entirely for parametric queries.

### Continuous batching + paged attention (deferred)
**Why deferred**: arch claims larql cares about are likelihood-bounded, not
throughput-bounded. PagedAttention-style KV management interacts with all four
KV engines (each has its own checkpoint geometry), and the design work isn't
worth it until a specific eval forces it. Revisit if a throughput-class
benchmark becomes load-bearing for an arch claim.

### Multi-GPU / tensor-parallel (deferred)
`larql-grid` already shards layers across hosts. Tensor-parallel within a layer
is a separate problem and not on the critical path until 70B+ models become the
bottleneck.

---

## Completed

| Item | Date | Impact |
|------|------|--------|
| Forward pass (CPU BLAS) | 2026-03 | Foundation |
| BLAS-fused attention | 2026-04-03 | Online softmax, O(seq) memory |
| WalkFfn (sparse FFN via vindex) | 2026-04-03 | Gate KNN + top-K |
| CachedLayerGraph | 2026-04-04 | Skip L0-12, 0.999 cosine |
| LayerGraph trait | 2026-04-04 | Pluggable per-layer routing |
| predict_honest | 2026-04-06 | Production path, GPU+CPU hybrid |
| GPU prefill pipeline | 2026-04-06 | seq>1 on GPU (pre-norm models) |
| Q4_K FFN format wiring | 2026-04-07 | Vindex Q4_K FFN → FullPipelineLayer |
| GELU-tanh activation | 2026-04-07 | Gemma3 correct on GPU |
| Post-norm guard | 2026-04-07 | Gemma3 falls to CPU correctly |
| KvEngine trait + EngineKind | 2026-04-25 | Pluggable engine selector + CLI params |
| MarkovResidualEngine | 2026-04-25 | Residual-based KV (exact, 287×) |
| UnlimitedContextEngine | 2026-04-25 | Window checkpoints (exact within window, 254×) |
| BackendFfn (Q4K FFN dispatch) | 2026-04-25 | WalkFfn + Metal for FFN in all engines |
| cold_kv cache (MarkovRS) | 2026-04-25 | Skip cold-tier recompute; 8.5× decode speedup |
| Profiler (per-stage timing) | 2026-04-25 | `larql bench --engine --profile` breakdown |
| TurboQuantEngine | 2026-04-26 | 4-bit WHT+Lloyd-Max K/V compression (4×, cos≈0.991) |
| ApolloEngine | 2026-04-26 | Retrieval+injection (20,000×, compressed path) |
| `forward_from_layer` | 2026-04-26 | Start forward at crystal_layer; 8.5× Apollo speedup |
| Metal Q4K path for all engines | 2026-04-26 | ~95 tok/s across all 4 engines |
| `generate/` split (cpu/gpu/lm_head/types) | 2026-04-26 | Structured generation directory |
| `markov_residual/` split (store/engine/compute/q4k) | 2026-04-26 | Structured engine directory |
| `forward/predict/` split (types/raw/dense/ffn) | 2026-04-26 | Forward predict directory |
| `forward/ops.rs` extracted | 2026-04-26 | Shared math primitives |
| `graph_ffn.rs` → `ffn/graph_backend.rs` | 2026-04-26 | Correct placement in ffn/ |
| 400+ unit tests | 2026-04-26 | Synthetic weights, no disk I/O |
| 49% line coverage (llvm-cov) | 2026-04-26 | Baseline measured |
| Code quality review (3-agent) | 2026-04-26 | Unsafe removed, LCG fixed, OnceLock added |
| P1 code quality fixes (magic strings, duplication) | 2026-04-25 | env-var names, GELU constants |
| `ffn/remote.rs` → `remote/codec.rs` + `remote/http.rs` | 2026-04-26 | No magic strings; codec/HTTP separation |
| `turbo_quant/mod.rs` → `engine.rs` | 2026-04-26 | Consistent engine layout; thin mod.rs |
| Tests: `markov_residual/` (store, engine, compute) | 2026-04-26 | 0 → 15 tests; prefill/decode/clip coverage |
| Tests: `ffn/sparse_compute.rs` + `ffn/sparse.rs` | 2026-04-26 | 0 → 14 tests; sparse FFN validated |
| Tests: `ffn/graph_backend.rs` | 2026-04-26 | 0 → 10 tests; GateIndex build/lookup/save |
| Tests: `forward/ops.rs` | 2026-04-26 | 0 → 8 tests; dot_proj/add_bias/apply_norm |
| 457 unit tests total | 2026-04-26 | +~50 tests vs previous session |
| Bug: `eos_id = 1` in grid.rs | 2026-04-26 | Correct EOS on all models, not just Gemma |
| Softmax unified to `forward/ops.rs` | 2026-04-26 | 2 duplicate impls removed |
| `forward/ple.rs` norm_eps fixed | 2026-04-26 | Uses `arch.norm_eps()` not hardcoded 1e-6 |
| Tests: `unlimited_context/extend.rs` | 2026-04-26 | 0 → 8 tests; checkpoint, RoPE, chained extends |
| Tests: `layer_graph/dense.rs` | 2026-04-26 | 0 → 8 tests; shape, capture, PerLayerGraph bounds |
| Tests: `layer_graph/walk.rs` | 2026-04-26 | 0 → 7 tests; Walk + Pipelined layer range |
| Tests: `layer_graph/mod.rs` | 2026-04-26 | 0 → 3 tests; trait dispatch, name distinctness |
| Tests: `forward/ple.rs` | 2026-04-26 | 0 → 6 tests; guard paths + softmax |
| Tests: GQA reps>1 | 2026-04-26 | 3 tests; shape, finiteness, KV-head sharing |
| Tests: RoPE property tests | 2026-04-26 | 4 tests; base sensitivity, offset=position, fractions |
| 499 unit tests total | 2026-04-26 | +42 tests; all passing |
| Tests: `layer_graph/prefill.rs` | 2026-04-26 | 6 tests; CPU path shape/finiteness/logits |
| Tests: `layer_graph/template.rs` | 2026-04-26 | 12 tests; detect_template + TemplateUniverse + GuidedWalk |
| Tests: `layer_graph/pipeline_layer.rs` | 2026-04-26 | 6 tests; arch params, attn weights, FFN stride |
| Tests: `layer_graph/grid.rs` | 2026-04-26 | 1 test; error path for missing Q4K mmap |
| Integration tests: `test_layer_graph_integration.rs` | 2026-04-26 | 7 ignored tests; real vindex prefill/pipeline/template |
| Fix: `residual_diff/capture.rs` missing PathBuf import | 2026-04-26 | Pre-existing bug; broke lib test compilation |
| 525 unit tests total | 2026-04-26 | All passing |
| `generate/eos.rs` — `EosConfig` | 2026-04-26 | Built-in stops + `generation_config.json`; fixes Gemma 4 `<end_of_turn>` bug |
| `generate/detok.rs` — `Detokenizer` | 2026-04-26 | Cumulative-decode delta; preserves HF `▁` leading-space across SP and BPE |
| `generate/sampling.rs` — `Sampler` + `SamplingConfig` | 2026-04-26 | Greedy / temp / top-k / top-p + seed; <2µs/call sparse path |
| `generate_with_sampling` wired into GPU path | 2026-04-26 | Greedy `generate` is a thin wrapper; backward compatible |
| Examples: `sampling_demo`, `eos_demo`, `detok_demo` | 2026-04-26 | End-to-end demos; detok runs without a model |
| `bench_sampling` benchmark | 2026-04-26 | Per-call cost across 4 configs × 3 vocab sizes; results in PERFORMANCE.md |
| 35 sampling/eos/detok tests | 2026-04-26 | All passing; 613 lib tests total |
| `generate_streaming(... on_token)` callback | 2026-04-26 | Per-token streaming; `generate_with_sampling` is thin no-op wrapper |
| `chat_session.rs` — `ChatSession` + `TurnRenderer` | 2026-04-26 | Multi-turn buffer with whole-turn eviction; Gemma/ChatML/Llama-3 renderers |
| Examples: `streaming_demo`, `chat_demo` | 2026-04-26 | Live token streaming + 3-turn chat over `ChatSession` |
| Smoke test: `test_gemma3_smoke.rs` | 2026-04-26 | One-token greedy regression; CI_INTEGRATION fail-loud mode |
| 13 ChatSession tests + streaming integration | 2026-04-26 | All passing; 626 lib tests total |
| Q4_K stride validation in `load_attn_q4k` | 2026-04-27 | Catches stale 148-byte vindexes; clear "rebuild" error vs silent NaN |
| `QuantFormatInfo::expected_bytes(&shape)` helper | 2026-04-27 | Single source of truth for stride math; used by loader validation |
| 11 stride-validation tests (registry + loader) | 2026-04-27 | 144 vs 148-byte stride; arbitrary lengths; Q4_K & Q6_K shapes |
| Q4_K vs Q4_KF kernel routing fix in `quant_matvec::encode` | 2026-04-27 | Q4_K weights now dispatch the Q4_K kernel; `FusedQkvKernel` enum carries TG geometry |
| `vindex::open_inference_vindex` strict loader | 2026-04-27 | Single entry point; propagates stride errors instead of silently degrading |
| Demos switched to `open_inference_vindex` | 2026-04-27 | sampling/streaming/eos/chat now error loudly with rebuild guidance on stale vindexes |

### 2026-04-30 — gRPC grid accuracy + dense Metal chat template + Gemma 4 model coverage

End-to-end accuracy work across Gemma 4's three production variants (26B-A4B
MoE via gRPC grid, 31B dense via Metal, E2B with PLE). Started from the gRPC
grid producing semantically wrong text ("not specified in the text") and
ended with all four Gemma 4 vindexes producing correct answers. Per-layer
CPU vs Metal residual parity (cos ≥ 0.9999 across all 60 layers of the 31B)
confirmed the inference math itself was always correct — every remaining
gap was somewhere in the wrapping, sampling, or routing logic.

| What | Date | Notes |
|------|------|-------|
| `grid.rs` uses `Detokenizer` + `EosConfig::from_vindex_dir` | 2026-04-30 | Was per-token decode losing SP `▁` leading-space + falling back to `<{id}>` for special tokens; output looked like "Thecapital of France is**not specified...**" |
| Special-token suppression in grid `pick_next_filtered` | 2026-04-30 | Built from `tokenizer.get_added_tokens_decoder()` + structural-marker scan (`<unused…>`, HTML tags, `[multimodal]`). Top-K=256 fallback finds a real word when many candidates are markers. Q4_K quantisation noise was lifting `<mask>` (id 4) over the intended next word at the first answer position |
| `chat::render_user_prompt` shared helper | 2026-04-30 | Centralises `LARQL_RAW_PROMPT` / `LARQL_THINKING` / `LARQL_SYSTEM` / `LARQL_NO_DEFAULT_SYSTEM` + auto Gemma 4 default system prompt. Used by both `run_with_moe_shards` (gRPC) and `walk_cmd::run_predict_q4k` (dense Metal) |
| Built-in Gemma 4 fallback chat template | 2026-04-30 | Vindexes extracted before `chat_template.jinja` was snapshotted (early 31B and E2B) silently sent raw prompts and looped "The answer is:". `family_default_template("gemma4")` plugs the gap |
| Dense Metal path now applies chat templates | 2026-04-30 | `walk_cmd::run_predict_q4k` was sending the raw user string to `encode_prompt`; the chat-template machinery only ran for gRPC. Both paths now go through `render_user_prompt` |
| `lm_head_topk` falls back to backend GEMV when KNN is all-zero | 2026-04-30 | At the prefill→decode boundary the Metal `q4k_matvec` for lm_head occasionally returned 256/256 zero scores while h_1d was healthy (rms ≈ 4, max_abs ≈ 60). Detect + retry via `backend_lm_head_topk` recovers a non-zero distribution immediately |
| PLE auto-route for Gemma 4 E2B | 2026-04-30 | E2B has `hidden_size_per_layer_input=256` (per-layer-input gate + projection + norm + global PLE embedding). The CPU dense path implements PLE; Metal does not. `generate_streaming` now checks `arch.has_per_layer_embeddings()` and delegates to `generate_via_cpu_q4k` for those models so the residual stream gets the per-layer per-position contribution. Without this E2B emitted multilingual gibberish; with it, "The capital of France is Paris" |
| Diagnostic env vars: `LARQL_DEBUG_TOKEN_IDS`, `LARQL_DEBUG_TOPK` | 2026-04-30 | Per-step token-id + raw top-K scores in both `grid.rs` (gRPC) and `gpu.rs` (dense). Surfaced the "all logits == 0.000" smoking gun that localised the lm_head KNN bug |
| `larql parity --component layer` extended to dense | 2026-04-30 | Was MoE-only (`LARQL_DUMP_RESIDUALS`). Now uses `LARQL_METAL_DUMP_LAYERS` for dense models — wrote per-layer `metal_layer_NN_h_out.f32` and CPU dump files. Gave us the cos ≥ 0.9999 confirmation across 60 layers that ruled out the inference math as the bug source |
| `larql parity --component lm-head` works on dense | 2026-04-30 | Dropped the MoE-only gate for `lm-head` (Q4_K vs f32 reference is backend-agnostic) |
| `test_logits_goldens.rs` compile fix + 5 new entries | 2026-04-30 | Added missing `None` for `predict_q4k_hidden`'s `Option<&RemoteMoeBackend>`; refreshed stale 5 goldens to match current kernel state; added `gemma3-4b-q4k-downq4k` (Q4_K-down regression test), `gemma4-31b-q4k-q6kdown` (Q6_K-down dense), `gemma4-e2b-q4k` (PLE auto-route) — 13/13 passing |
| Discovered: in-process Metal MoE path (`gpu_moe_dispatch_with_scratch`) shares the bug | 2026-04-30 | Until now nobody had run `larql run --metal` on Gemma 4 26B-A4B (the gRPC grid was the only tested path). It produces the same wrong text as the server's Metal expert dispatch ("answer is in the context" instead of "Paris"). The gRPC-with-CPU-experts path has been the only working route all along — the in-process Metal MoE was always broken for this model. See `larql-compute/ROADMAP.md` "Open: Metal MoE expert kernel — accuracy bug at inter=704" for the kernel-side fix plan |
