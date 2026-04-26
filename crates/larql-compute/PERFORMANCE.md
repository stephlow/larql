# Performance — larql-compute

Machine: M3 Max, macOS 24.6.0, Gemma 3 4B (34 layers, hidden=2560, inter=10240, vocab=262K)
Vindex: `gemma3-4b-q4k-v2` (Q4_K attn/gate/up, Q6_K V/down — Ollama convention)

---

## Current state (2026-04-26)

```
larql-metal  gemma3-4b-q4k-v2   75–77 tok/s   13.0ms/tok
Ollama       gemma3:4b          97–103 tok/s  10.0ms/tok
Gap          1.26–1.34×         +3ms/tok
```

Per-stage breakdown (100-token run, 8 warmup):

| Stage | ms/tok | % |
|---|---|---|
| GPU fwd | 11.7–11.9 | 83% |
| lm_head | 2.35 | 17% |
| embed + norm + detok | ~0.01 | ~0% |

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

- M3 Max GPU practical bandwidth: ~300-350 GB/s (system-shared LPDDR5X)
- Ollama reaches ~348 GB/s effective on weight reads
- LARQL currently at ~322 GB/s — gap is dispatch overhead, not kernel quality
- Metal dispatch overhead: ~5µs per `dispatch_thread_groups` call
- At 476 dispatches/tok: 2.4ms pure overhead (vs Ollama's ~1.4ms)
- Reducing to 200 dispatches/tok would save ~1.4ms → ~83 tok/s
- Q6_K linear-format kernel registers: ~20/thread × 128 threads = 2560/TG
- Q6_K ROWS_PER_TG=4: 640 TGs for N=2560 (adequate GPU saturation)
