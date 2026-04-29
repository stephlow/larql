# Performance — larql-compute

Machine: M3 Max, macOS 24.6.0, Gemma 3 4B (34 layers, hidden=2560, inter=10240, vocab=262K)
Vindex: `gemma3-4b-q4k-v2` (Q4_K attn/gate/up, Q6_K V/down — Ollama convention)

> **Note on the historical "81–84 tok/s"**: an earlier ROADMAP table cited
> 81–84 tok/s for this same vindex on 2026-04-26. Bisect (2026-04-28)
> traced that to a silent dispatch bug fixed in commit `077884b "working
> on performance"`: Q4_K weights were routed through the **Q4_KF kernel**
> with the wrong threadgroup geometry (4 rows/TG instead of 8), leaving
> ~75% of output rows unwritten. The 81–84 was real wall-clock
> throughput on broken (wrong-output) code. **78.7 tok/s is the correct
> baseline for valid output.** Reverting 077884b would re-introduce the
> bug.

> **Profiler note (2026-04-28)**: an earlier per-kernel diagnosis claimed
> q4k_ffn_gate_up was "ALU-limited at 103 GB/s, compute-bound on Q4_K
> dequant". That was a profiler bug — `measure_batched` was creating a
> fresh cmd buffer per kernel call (with commit+wait per call) instead
> of running `n_layers` dispatches in one cmd buffer, so per-call
> dispatch overhead dominated the measurement. Fixed via
> `measure_single_cmdbuf_batched`. Corrected numbers: q4k_ffn_gate_up at
> **274 GB/s = 74% of LPDDR5X peak (bandwidth-bound)**, not 103 GB/s
> compute-bound. Both big FFN kernels are at bandwidth saturation; the
> 1.30× decode gap to ollama is distributed across the pipeline, not
> concentrated in any single kernel.

---

## Current state (2026-04-28)

```
larql-metal  gemma3-4b-q4k-v2     80.3 tok/s   12.45ms/tok (gate+up 8sg + q4k_matvec 8sg, 2026-04-28)
larql-metal  gemma3-4b-q4k-v2     76.3 tok/s   13.11ms/tok (q4k_matvec 4sg, gate+up 8sg)
larql-metal  gemma3-4b-q4k-v2     78.9 tok/s   12.67ms/tok (gate+up 8sg, q4k_matvec 4sg)
Ollama       gemma3:4b            94–98 tok/s   ~10.5ms/tok
Gap          ~1.18×               ~1.95ms/tok

larql-metal  gemma4-26B-A4B         5.1 tok/s  ~194ms/tok  (Phase 1 GPU dispatch; Phase 2 open)
SKIP_MOE ceiling                   56.8 tok/s   ~15ms/tok  (attention + dense FFN only)
```

Per-stage (Gemma 3 4B, 100-token run, 8 warmup):

| Stage | ms/tok | % |
|---|---|---|
| GPU fwd | ~10.8ms | 83% |
| lm_head | ~2.2ms | 17% |
| embed + norm + detok | ~0.01ms | ~0% |

**Recent changes (2026-04-26 → 2026-04-28):**

| Change | Model | Effect | Notes |
|---|---|---|---|
| **lm_head Q4_K vs Q4_0 dispatch fix** | Gemma 3 4B v2 | correctness — output was gibberish | Writer produced Q4_K, reader dispatched Q4_0 (same byte rate so file size matched). Now dispatches q4k_matvec. |
| **MoE combine helper unification** (CPU + Metal share `outer_combine.rs`) | Gemma 4 26B-A4B | **correctness — was multilingual gibberish** | 4 silent divergences between CPU/Metal MoE combine logic (f32/f64 RMS, identity-scale-on-missing-norm, etc.) collapsed into one helper. Verified via `larql parity --component layer`: 30/30 layers cos=1.0. |
| **Q4_K dispatch correctness fix** (commit 077884b) | Gemma 3 4B | **−5 tok/s** (84 → 79) | Q4_K was routed through Q4_KF kernel, leaving 75% of output rows unwritten; 81-84 was on broken code, 79 is correct baseline |
| **`q6k_matvec` ROWS_PER_TG=4 correctness fix** | Gemma 3 4B | **78.7 tok/s, GPU fwd 10.8ms** | Silent bug: rows 1282-2559 were zeros; fixed to ROWS_PER_TG=4 everywhere |
| **Profiler harness fix** (`measure_single_cmdbuf_batched`) | profiling tool | corrects per-kernel GB/s by 2-4× | Old harness ran each kernel call in its own cmd buffer; per-call dispatch overhead dominated the measurement. Fixed numbers: q6k_matvec 311 GB/s (was 74), q4k_ffn_gate_up 274 GB/s (was 103). |
| **`q4k_matmul` Metal kernel** + parity tests | prefill | kernel 1.79× isolated; **end-to-end no win** | Wiring into O proj + FFN gate+up was attempted and reverted 2026-04-28: short-prompt prefill within noise, long-prompt prefill regressed ~10%. Same failure mode as f16 acc — kernel was bandwidth-near-peak and matmul's [seq_len × hidden] X working set thrashes L1 on long prompts. Kernel remains available via `MetalBackend::q4k_matmul` for callers that want it; not in production decode/prefill path. |
| **Encoder coalescing** in 3 dispatch sites (O proj, QKV f32, QKV Q8) | prefill | <5% on long prompts | Below noise on short prompts. Real win is the matmul kernel above; coalescing was the cheap risk-free first move. |
| **`q4k_ffn_gate_up_f16acc` shader** (opt-in, `LARQL_F16_ACC=1`) | Gemma 3 4B | kernel 1.79× isolated; **end-to-end at parity** | Numerical parity perfect (10-prompt greedy bit-identical), but kernel was already bandwidth-bound — freed ALU cycles get absorbed by surrounding kernels. Initial +23% measurement was thermal-throttle artifact. Kept as opt-in. |
| **`q4k_ffn_gate_up_8sg` shader** (now default; opt-out `LARQL_GATE_UP_8SG=0`) | Gemma 3 4B | **+2.1% end-to-end** (77.2 → 78.9 tok/s) | 8 simdgroups per TG (256 threads, 8 rows/TG) instead of 4/128/4. Same per-thread register footprint (`nr0=1`). Bit-identical output. First positive end-to-end perf this session. |
| **`q6k_matvec_8sg` shader** (opt-in only, `LARQL_Q6K_8SG=1`) | Gemma 3 4B | kernel **1.96× isolated**, end-to-end **at parity** | Q6_K was already at 84% of LPDDR5X peak — too little headroom for 8sg to recover; larger TGs cause schedule contention with 8sg gate+up. Kept opt-in. |
| **`q4k_matvec_8sg` shader** (now default; opt-out `LARQL_Q4K_MATVEC_8SG=0`) | Gemma 3 4B | **+5.2% end-to-end** (76.3 → 80.3 tok/s) | Profiler showed q4k_matvec at 220 GB/s = 55% of LPDDR5X peak (most under-utilised matvec). 8sg gives biggest single-shader win this session — touches Wo + QKV fallback + other call sites, gains compound. Bit-equal parity ✓. |
| **Pattern observation (2026-04-28)**: 8sg geometry helps proportionally to bandwidth headroom: 55% util (q4k_matvec) → +5.2%; 68% util (gate+up) → +2.1%; 84% util (q6k_matvec) → 0% (regressed). When considering 8sg for a new kernel, profile its production-batched GB/s first — only worth it if utilisation is below ~75% of LPDDR5X peak. | | | |
| `f32_gemv_topk1` GPU argmax | any | 0 in bench (KNN fires first) | Saves 0.33ms for top_k=1 non-KNN callers |
| Q4_K float4 dual-sub-block | Gemma 3 4B | **REGRESSED** (reverted) | K=2560 — added addressing overhead |
| Batched MoE prefill | Gemma 4 26B A4B | **+35% tok/s, −31% prefill** | 130 → 26 GPU commits for 5-token prompt |
| Q4_K `sumy` precompute | Gemma 3 4B | neutral (within noise) | Compiler already hoisting; FMA chain unchanged |
| Per-layer Q4K format + GPU expert dispatch | Gemma 4 26B A4B | **+75% overall (2.9 → 5.1 tok/s)** | Expert FFNs on GPU; see §26B A4B below |

### Per-kernel batched throughput (corrected 2026-04-28)

`diag_profile_kernels`, M3 Max, gemma3-4b-q4k-v2:

| Kernel | Batched ms/call | GB/s | Per-token (×34) | Bottleneck |
|---|---|---|---|---|
| q4k_ffn_gate_up (gate+up, K=2560) | 0.108 ms | **274 GB/s** | 3.7 ms | bandwidth-bound, 74% of LPDDR5X peak |
| q6k_matvec (down, K=10240) | 0.069 ms | **311 GB/s** | 2.3 ms | bandwidth-bound, 84% of peak |
| f32_gemv (lm_head, 262K×2560) | — | **374 GB/s** | 1.93 ms | at LPDDR5X peak |
| Wo + QKV + attention + 4× RMS norms | mixed | mixed | ~5.9 ms | mixed, presumed near-peak |

**No headroom in any single kernel.** The 1.30× decode gap to ollama is distributed across dispatch overhead + sustained-clock effects + the cumulative inefficiency of running fewer-fused kernels than llama.cpp.

---

## Gemma 4 26B A4B — MoE model (2026-04-26)

Machine: M3 Max, 5-token prompt, 15 warmup / 30 measured tokens  
Vindex: `gemma-4-26B-A4B-it.vindex` (30 layers, 128 experts/layer, top-K=8, inter=704, hidden=2816)

### Progress log

| Optimisation | Decode tok/s | GPU fwd | Δ |
|---|---|---|---|
| BF16 blob baseline | 2.9 | 334ms | — |
| Batched MoE prefill | 3.9 | 246ms | +35% |
| Q4K per-layer format + GPU expert dispatch | **5.1** | **~194ms** | **+75% from baseline** |
| GPU-only ceiling (`SKIP_MOE=1`) | 56.8 | 15ms | theoretical max |

### Current bottleneck: Metal buffer allocation overhead

GPU fwd 194ms breaks down as:
- Actual GPU compute (30 × attention + dense FFN + 8 expert dispatches): ~40ms
- 30 MoE layer syncs (commit + wait for h_post_attn routing): ~30ms
- **Metal buffer allocation: ~120ms** — root cause of remaining gap

Per decode token, `gpu_moe_dispatch` calls `self.bufs.output()` ~10 times per
layer (gate buf, up buf, 8 down bufs, act buf, outputs buf) = 300 allocations/token.
Each `MTLResourceOptions::StorageModeShared` allocation of 1–9 MB takes ~0.4ms
on M3 Max = ~120ms total.

### Phase 2: pre-allocated scratch buffers (open)

Pre-allocate fixed-size staging buffers once before the decode loop in
`decode_token_q4k_moe`, same pattern as `decode_token`'s scratch buffer
pre-allocation (which eliminated 550 allocations → 20 for the dense path).
Sizes are fixed for a given model — known at init time from `moe.intermediate_size`,
`moe.num_experts`, `moe.top_k`, `hidden`.

Expected result: ~15–20 tok/s (~4× current), closing most of the gap to the GPU ceiling.

---

## Per-kernel profiling (2026-04-26, M3 Max, Gemma 3 4B shapes)

Run: `cargo run --release --features metal -p larql-compute --example diag_profile_kernels`

Two measurement modes:
- **Isolated**: one commit+wait per call (includes ~20µs GPU spin-up overhead)
- **Batched**: 34 calls per command buffer, single commit+wait (matches real decode pipeline)

| Kernel | Data/layer | Batched GB/s | Batched ms/layer | ms/tok×34L | Bottleneck |
|---|---|---|---|---|---|
| q6k_matvec (FFN down, K=10240) | 21.5 MB | **312 GB/s** | 0.069ms | 2.34ms | bandwidth-bound |
| q4k_ffn_gate_up (gate+up, K=2560) | 29.5 MB | **272 GB/s** | 0.108ms | 3.68ms | **compute-bound** |
| f32_gemv (lm_head, 262K×2560) | 2680 MB | **370 GB/s** | — | 7.4ms | bandwidth-bound (near peak) |

**These two kernels (down + gate+up) account for 6.01ms of the ~11.7ms GPU fwd.**

### Why gate+up is compute-bound

Q4_K at K=2560 has the lowest bytes-per-element ratio (0.5625 B/elem) of any kernel.
The GPU spends more cycles on nibble dequant than waiting for LPDDR5X. Ollama closes
this gap via vectorized `float4` accumulation in their `kernel_mul_mv_q4_K_f32_impl`,
but that kernel assumes a transposed nibble layout (GGUF format: lo=elem b, hi=elem b+32)
incompatible with LARQL's linear layout (lo=elem 2b, hi=elem 2b+1).

### Projected impact of closing each gap

| Gap | Current | Target (Ollama est.) | Savings |
|---|---|---|---|
| q6k_matvec: 312→390 GB/s | 2.34ms | 1.87ms | 0.47ms |
| q4k_ffn_gate_up: 272→390 GB/s | 3.68ms | 2.57ms | 1.11ms |
| lm_head overhead | 2.45ms | ~1.3ms | 1.15ms |
| Dispatch overhead | ~1.87ms | ~1.36ms | 0.51ms |
| **Total projected savings** | | | **~3.24ms** → ~85 tok/s |

---

## llama.cpp / Ollama gap analysis (2026-04-25)

### Bandwidth budget

Gemma 3 4B weight data read per token (34 layers):

| Matrix | Format | Size/layer | Total 34L |
|---|---|---|---|
| Wq (8192×2560) | Q4_K | 11.8 MB | 401 MB |
| Wk (4096×2560) | Q4_K | 5.9 MB | 201 MB |
| Wv (4096×2560) | Q6_K | 8.6 MB | 292 MB |
| Wo (2560×8192) | Q4_K | 11.8 MB | 401 MB |
| W gate+up (10240×2560 ×2) | Q4_K | 29.5 MB | 1003 MB |
| W down (2560×10240) | Q6_K | 21.5 MB | 731 MB |
| **Total** | | **89.1 MB** | **3029 MB** |

Theoretical minimums at M3 Max GPU bandwidth:

| Bandwidth | Min time | Max tok/s |
|---|---|---|
| 400 GB/s (peak) | 7.6ms | 132 |
| 300 GB/s (practical) | 10.1ms | 99 |

Measured effective bandwidth (kernel time only, subtracting dispatch overhead):

| Engine | GPU fwd | Dispatch est. | Kernel time | Eff. BW |
|---|---|---|---|---|
| LARQL | 11.8ms | ~2.4ms (476 dispatches×5µs) | ~9.4ms | ~322 GB/s |
| Ollama | 10.1ms | ~1.4ms (272 dispatches×5µs) | ~8.7ms | ~348 GB/s |

LARQL kernels are at ~322 GB/s vs Ollama's ~348 GB/s — a 8% kernel efficiency
gap. The larger gap (1.33×) is dominated by dispatch overhead.

### Dispatch count gap

LARQL has ~14 dispatches per layer × 34 = **476 dispatches/token** = ~2.4ms overhead.
Ollama groups ops more aggressively: estimated ~8 dispatches/layer × 34 = ~272 dispatches.
Dispatch savings alone: **~1.0ms/token**.

### Three specific things llama.cpp does in Q6_K that we've now partially adopted

Comparing `kernel_mul_mv_q6_K_f32_impl` (llama.cpp) vs `q6k_matvec` (LARQL):

| Technique | llama.cpp | LARQL (post 2026-04-25) | Impact |
|---|---|---|---|
| Inter-superblock interleaving | `ix = tiisg%2` → 2 banks in parallel | ✅ `ix = lane & 1u` | Better DRAM utilization |
| X preloading | `yl[16]` loaded before compute loop | ✅ `xl[16]` preloaded | Hides L2 latency |
| Deferred scaling | `float4 sums` → scale once/group | ✅ `acc += d*sc*(...)` | 4× fewer multiplications |
| TG size | 64 threads (2 rows/TG) | 128 threads (4 rows/TG) | Lower register pressure |
| Block format | GGUF transposed layout | LARQL linear layout | Different algorithms needed |

The format mismatch (LARQL uses linear Q6_K, GGUF uses transposed) means
llama.cpp's exact inner loop can't be ported directly — the element ordering
is different. The inter-superblock interleaving + preload + deferred scale
improvements were adapted to the linear layout.

### What remains

1. **Dispatch overhead** (~1ms): 14→8 dispatches/layer through fusion
   - Fused input norm + QKV projection (saves 34 dispatches)
   - Combined QK-norm Q+K (saves 34 dispatches)
   - Combined RoPE Q+K dispatch (saves 34 dispatches)
   Together: ~102 fewer dispatches = ~0.5ms

2. **Q4_K kernel** (~0.5ms): gate+up (Q4_K, 29.5 MB/layer) runs the old sub-block
   stride kernel. llama.cpp's `kernel_mul_mv_q4_K_f32_impl` uses:
   - 4 parallel block groups (`ix=tiisg/8`, 4 groups at once)
   - `yl[]/yh[]` preloading of X values + `sumy[]` for the min correction
   - `float4 acc1/acc2` vectorized accumulation
   Adapting these to LARQL's GGUF-compatible Q4_K format should close another
   ~0.5ms.

3. **lm_head** (~0.5ms overhead over 1.55ms kernel): async readback + heap
   top-k already reduced the CPU-side cost; GPU-side quantize still CPU-bound.

---

## Optimization history

| Date | Change | Before | After | Delta |
|---|---|---|---|---|
| 2026-04-09 | Full kernel + norm rewrite, Q4_KF, fused ops | 29ms (34 tok/s) | 8.5ms (117 tok/s) | −20ms |
| 2026-04-19 | FFN Q4K + Q6K correctness, decode KV cache | — | 14.7ms (68 tok/s) | baseline |
| 2026-04-25 | `q6k_matvec` 4-element batching (compile-time hi2 shifts) | 14.7ms | 13.7ms | −1.0ms |
| 2026-04-25 | Q6K inter-superblock interleaving + X preload + deferred scale | 13.7ms | 11.8ms | −1.9ms |
| 2026-04-25 | lm_head min-heap top-k (avoids 2MB Vec allocation) | 2.40ms | 2.35ms | −0.05ms |
| 2026-04-25 | Dispatch fusions (QK-norm Q+K, RoPE Q+K, residual_norm_store, normed QKV) | 72ms | ~13ms | +1–2 tok/s |
| 2026-04-26 | `f32_gemv_topk1` GPU argmax (gemv + argmax, 8KB readback vs 1MB) | — | — | 0.33ms/tok for top_k=1 |
| 2026-04-26 | Diagnostic: `diag_profile_kernels` (per-kernel GB/s, isolated+batched) | — | — | tooling |
| 2026-04-26 | **q6k_matvec ROWS_PER_TG=4 correctness fix** (shader+dispatch mismatch; rows 1282-2559 were zeros) | 68-75 tok/s (wrong) | **78.7 tok/s, 10.8ms** | +0.2ms vs wrong fast path; correct output |
| 2026-04-26 | Batched MoE prefill (dispatch_full_pipeline moe_fn callback) | 2.9 tok/s, 334ms | 3.9 tok/s, 246ms | −31% prefill, +35% decode |
| 2026-04-26 | Per-layer Q4K expert format + GPU dispatch (Phase 1) | 3.9 tok/s | **5.1 tok/s, 194ms** | +31% decode; Phase 2 open |

---

## Historical context

```
2026-04-09 — synthetic Q4_KF (random weights):  8.5ms = 117 tok/s (17% FASTER than Ollama)
           The 117 tok/s number used synthetic weights; Q4_KF fast-path doesn't
           fire on production GGUF extracts which use Q6_K for down projection.

2026-04-19 — first real-vindex decode:  ~14.7ms = 67.9 tok/s  (Ollama ~100 tok/s)
           Real model uses Q4_K gate/up + Q6_K down (Ollama convention).
           Q6_K was the bottleneck: 79 GE/s effective vs Q4_K's 105 GE/s.

2026-04-25 — Q6_K rewrite session:  62 → 72 tok/s over three shader iterations.
           Root cause of original gap: runtime hi2 shift + sequential superblock
           access + register pressure from sc_f[16] preload (paradoxically hurt
           by occupancy reduction).
```

---

## Key data points for future work

- M3 Max GPU practical bandwidth: ~300-400 GB/s (system-shared LPDDR5X)
- Ollama effective bandwidth: ~390 GB/s (measured, not estimated — inferred from kernel gap)
- LARQL effective bandwidth: ~315-330 GB/s
- Metal dispatch overhead: ~5µs per `dispatch_thread_groups` call
- Current: 374 dispatches/tok ≈ 1.9ms overhead (vs Ollama ~272 = 1.4ms → 0.5ms gap)
- **Gate+up is ALU-limited at K=2560**: 272 GB/s despite L1-cached input; dequant ops dominate
- **q6k_matvec is bandwidth-limited at K=10240**: 315 GB/s; ROWS_PER_TG=4 (640 TGs × 128 threads, 4 rows/TG, no overlap) is both correct and fast (78.7 tok/s)
- `f32_gemv_topk1` GPU argmax: fires for top_k=1 callers; main decode uses KNN lm_head (top_k=5), so bench gain = 0. Value for non-KNN model paths.
- To close the kernel compute gap: need format-compatible vectorized Q4_K dequant (no solved approach yet)
