# larql-compute

Hardware-accelerated compute backends for LARQL. CPU (BLAS + NEON Q4), Metal GPU, and future CUDA.

## What it does

Provides a `ComputeBackend` trait that abstracts all hardware-specific matrix operations. Every LARQL crate (inference, vindex) uses this trait — the caller never knows whether the operation runs on CPU or GPU.

## Backends

| Backend | Feature flag | f32 matmul | Quantized ops | Pipeline |
|---------|-------------|------------|---------------|---------|
| **CPU** | (always) | BLAS (Accelerate AMX) | C kernel (ARM vdotq_s32) | Sequential |
| **Metal** | `--features metal` | Tiled shaders | Simdgroup Q4/Q4_K/Q6_K/Q8 | One command buffer |
| **CUDA** | (planned) | — | — | — |

## Performance vs Ollama

Live `larql bench gemma3-4b-q4k-v2 --backends metal --tokens 50 --ollama gemma3:4b`
on M3 Max (2026-04-25):

```
  Backend                 prefill       ms/tok      tok/s  steps  notes
  larql-metal               72.1ms      15.13ms      66.1      49
  ollama gemma3:4b          49.3ms      10.26ms      97.5      23

  Per-stage average (larql-metal):
    embed      0.002ms   ( 0.0%)
    GPU fwd   13.637ms   (85.6%)    ← decode hot path
    final_norm 0.007ms   ( 0.0%)
    lm_head    2.285ms   (14.3%)
    detok      0.007ms   ( 0.0%)
```

Reproduce: `larql bench <vindex-shorthand> --backends metal --tokens 50
--ollama <ollama-tag>`. CPU + Ollama variants via `--backends cpu,metal`.

### Q4_KF route (llama.cpp-exact kernel)

The 2026-04-08 optimization burst on the Q4_KF route hit **117 tok/s**
on the same hardware (Gemma 3 4B Q4_KF vindex, decode-only, KV cached).
That's still the best-case number once a Q4_KF vindex is loaded —
`larql bench gemma3-4b-q4kf` reproduces it. The 66 tok/s number above
is the Q4_K path (current default extract format).

### Key optimisations

| Optimization | Date | Savings | Technique |
|-------------|------|---------|-----------|
| **Q4K_*_MAX_K shared-tile fix** | 2026-04-25 | (correctness) | Drop 4096-float threadgroup tile in q4k_matvec / q4k_ffn_gate_up; closed Gemma 4 31B parity gap (cos 0.997→1.000) |
| Cooperative SIMD norms | 2026-04-09 | ~10ms | O(N²)→O(N) reads in rms_norm / residual_norm |
| Q4_KF FFN routing | 2026-04-09 | ~8ms | llama.cpp-exact kernel (q4kf_proj) for FFN |
| Q4_K matvec rewrite | 2026-04-09 | ~3ms | uint4 loads, 8 rows/TG, multi-row (nr0=2) |
| Buffer pre-allocation | 2026-04-08 | ~2ms | Eliminate 550 Metal allocs per decode |
| Fused gate+up kernels | 2026-04-08 | ~1ms | q4k_ffn_gate_up + q4kf_ffn_gate_up |
| Batched RoPE/V-norm | 2026-04-08 | ~0.5ms | 16 per-head dispatches → 3 batched |
| SIMD KV attention | 2026-04-08 | ~1ms | simd_max/simd_sum, fewer barriers |

### Architecture

Single command buffer + single global encoder for all 34 layers. Pre-allocated scratch
buffers. Format-aware FFN: Q4_KF routes through llama.cpp kernel, Q4_K through fused
gate+up, Q4_0 through legacy Q8 path. All norms use cooperative SIMD reduction.

## Shaders

Production kernels are in **bold**; the rest are either dispatched only by
diagnostic / fallback paths or compiled-but-unwired (kept around because
the shader source is small and the bench harness still exercises them).

| Category | Kernels | Notes |
|----------|---------|-------|
| f32 matmul | sgemm, sgemm_transb | Tiled 32×32 |
| f32/f16 gemv | **f32_gemv**, **f16_gemv** | LM head (large vocab × hidden) |
| Q4_0 matvec | **q4_matvec_v4** (prod), q4_f32_matvec, q4_vecmat | v4: uint32 wide loads, 61 GB/s |
| Q4_K / Q4_KF | **q4k_matvec**, **q4k_qkv_proj**, **q4k_q6k_qkv_proj**, **q4kf_qkv_proj**, **q4kf_proj** | All read X directly from device memory (no shared-memory tile cap) |
| Q4_K fused FFN | **q4k_ffn_gate_up**, **q4kf_ffn_gate_up** | Fused gate+up, shared input |
| Q6_K | **q6k_matvec** | Used for V proj on Gemma 3 / 4 (Q4_K Q/K + Q6_K V) and Q6_K down |
| Q8 | **q8_matvec**, **q8_qkv_proj**, **quantize_q8** | Fused QKV, simdgroup reduction |
| Attention | **fused_attention** (RoPE+GQA+softcap), **kv_attention** (decode), **kv_cache_append** | SIMD reductions, float4 dot |
| Normalization | **rms_norm**, **layer_norm** / **layer_norm_no_bias**, **v_norm_batched**, **qk_norm** | Cooperative SIMD reduction |
| Activation | **geglu_silu**, **geglu_gelu_tanh**, **silu**, **gelu_tanh** | Gated + standalone |
| Element-wise | **residual_add**, **scale_vector** | |
| RoPE | **rope_apply** (prefill multi-pos), **rope_at_pos** (prefill stage), **rope_at_pos_batched** (decode) | All bit-equal at the production geometries |
| Fused ops | **rms_norm_q8**, **residual_norm**, **residual_norm_q8** | Multi-op fusion |
| Experimental / unwired | causal_attention, q4_matvec_v2/v3/v5, q4_sparse_matvec, q8_proj_rope, q4k_geglu_silu_down, q4k_geglu_gelu_tanh_down, v_norm (singleton), turboquant_encode/decode, graph_walk_knn | Kept compiled; not dispatched in production decode/prefill |

## Safe Buffer Access

All Metal buffer reads go through one audited function with null/size checks:

```rust
// Replaces 13 previous unsafe { from_raw_parts } sites
pub fn read_buffer_f32(buf: &metal::Buffer, len: usize) -> Vec<f32>
```

## Quick Start

```rust
use larql_compute::{ComputeBackend, default_backend};

let backend = default_backend();
println!("Using: {} ({})", backend.name(), backend.device_info());

// f32 matmul
let c = backend.matmul_transb(a.view(), b.view());

// Q4_K matvec (Ollama-compatible format)
let scores = backend.q4k_matvec(&q4k_data, &x, rows, hidden);

// KV-cached decode (one token through all layers)
let h = backend.decode_token(&layers, &x, hidden, inter, q_dim, kv_dim,
    num_q_heads, num_kv_heads, head_dim, rope_base);

// GPU prefill (seq>1, populates KV cache)
let h = backend.prefill_q4(&layers, &x, hidden, inter, q_dim, kv_dim,
    seq_len, num_q_heads, num_kv_heads, head_dim, rope_base, qk_norm, softcap);
```

## Linear algebra primitives (`cpu/ops/linalg.rs`)

Beyond the matmul/quantization backends, `larql-compute` ships a small set
of pure-CPU f64 linear algebra primitives used by the higher crates:

| Primitive | Signature | Used by |
|-----------|-----------|---------|
| `cholesky(a, ridge)` | `(N,N) → L (N,N)` lower-triangular factor with optional ridge | MEMIT covariance solve, vindex MEMIT |
| `cholesky_solve(L, B)` | solves `L L^T X = B` for any `(N,m)` RHS | as above |
| `cholesky_inverse(L)` | A⁻¹ = `cholesky_solve(L, I)` | covariance whitening |
| `ridge_decomposition_solve(K, T, λ)` | closed-form `ΔW = T^T (K K^T + λI)⁻¹ K`, returns `(d,d)` | `larql_vindex::memit_solve` (COMPACT MAJOR) |

The N×N Cholesky runs in f64 — `K K^T` becomes ill-conditioned in f32 when
keys share a dominant direction (canonical-form templates, exp 8). Inputs/
outputs of `ridge_decomposition_solve` are f32 for caller convenience; the
solve is f64 internally.

Bench: `cargo bench -p larql-compute --bench linalg`
Demo:  `cargo run --release -p larql-compute --example demo_ridge_solve`

> The MEMIT-flavoured wrapper (`memit_solve` returning `MemitSolveResult`
> with per-fact reconstruction quality) lives in `larql-vindex` next to
> `MemitStore`. The production weight-edit pipeline with covariance
> whitening is in `larql-inference/forward/memit.rs`.

## Architecture

```
src/
  lib.rs              Re-exports from pipeline.rs and backend.rs
  pipeline.rs         QuantFormat, QuantWeight, NormType, FfnType, Activation, FullPipelineLayer
  backend.rs          ComputeBackend trait (15 methods)

  cpu/
    mod.rs            CpuBackend (BLAS f32 + C Q4 + Q4_K/Q6_K reference)
    ops/              f32_matmul, q4_matvec, q4_vecmat, q4k_matvec, q6k_matvec,
                      q4_common (quantizers: Q4_0, Q4_K, Q4_KF, Q6_K, GGUF Q4_K),
                      q8_matvec, vector, attention, geglu

  metal/              (feature-gated: --features metal)
    mod.rs            MetalBackend (30+ pipeline states, KV cache)
    trait_impl.rs     ComputeBackend dispatch (Q4_K/Q8 dual-path)
    decode/           KV-cached decode (norm→QKV→attend→O→FFN per layer)
      mod.rs          decode_token + decode_token_with_moe_fn (top-level loop)
      encode_qkv.rs   Step 1 — input norm + format-aware fused QKV
      encode_ffn.rs   Step 6 — format-aware FFN (Q4_KF / Q4_K / Q4_0)
      moe_combine.rs  Hybrid-MoE outer combine (Gemma 4 26B A4B)
      diag.rs         Per-stage / residual / NaN dump helpers
    prefill.rs        GPU prefill for seq>1
    buffers.rs        GPU buffer cache + read_buffer_f32
    shaders/          Metal kernel sources (one file per shader)
    stages/           Reusable stage encoders (qkv_proj, rope, qk_norm,
                      ffn, residual, layer_scalar, quant_matvec, …)
    ops/              GPU dispatch helpers (full_pipeline, kv_cache, …)

  csrc/q4_dot.c       ARM NEON Q4 kernel
```

## Tests

```bash
# CPU only
cargo test -p larql-compute

# CPU + Metal (full kernel + cross-backend coverage)
cargo test -p larql-compute --features metal
```

~165 tests with `--features metal` across:

- `tests/test_metal_shaders.rs` — quantization round-trips, cross-backend
  correctness (Metal vs CPU with tolerance), shader compilation, fused
  attention, partial RoPE, KV cache, pipeline output verification,
  activations (SiLU, GELU-tanh, GEGLU), LayerNorm, V-norm, scale_vector.
- `tests/test_kernel_*.rs` — focused per-kernel suites pinning each
  production shader at every architecture geometry (Llama 2 / Mistral /
  Gemma 3 4B / Gemma 4 31B sliding+global). One file per shader family:
  `kv_attention`, `kv_cache_append`, `qk_norm`, `rope_at_pos`, `rope`
  (rope_at_pos_batched), `v_norm`, `q4k_ffn_gate_up`. Includes
  prefill→decode KV-cache hand-off and the regression for the previously
  silent `Q4K_GU_MAX_K=4096` shared-memory cap (now read X directly from
  device memory; see ROADMAP ship log 2026-04-25).
- `tests/test_correctness.rs` and `tests/test_q4_x86_correctness.rs` —
  CPU-only quantization round-trips.

The cross-backend / cross-stage parity layer lives in `larql-inference`:

- `larql-inference/tests/test_cpu_metal_parity.rs` — full prefill,
  CPU vs Metal at every layer, all four production architectures.
- `larql-inference/tests/test_decode_consistency.rs` — Metal decode
  vs CPU prefill at the same effective sequence length.
- `larql-inference/tests/test_decode_stage_bisect.rs` — per-stage L0
  divergence localiser (closed the Gemma 4 31B parity gap; ship log
  2026-04-25).
- `larql-inference/tests/test_logits_goldens.rs` — pinned top-5 +
  top-1 logit per (architecture × backend) on a fixed prompt. Catches
  *correlated* drift (CPU and Metal regressing in the same direction)
  that the parity tests can't detect.

## Examples

### Demos

```bash
# Architecture overview — guided tour of all major design decisions
cargo run --release --features metal -p larql-compute --example demo_architecture

# Basic usage — backend detection, matmul, Q4 dispatch
cargo run --release --features metal -p larql-compute --example demo_basic
```

### Benchmarks: Compare (us vs Ollama)

The headline number — production decode tok/s vs Ollama on the same
hardware — comes from the CLI's `bench` subcommand, which loads a
real vindex and timing-matches a live `ollama generate` round trip:

```bash
larql bench gemma3-4b-q4k-v2 --backends metal --tokens 50 --ollama gemma3:4b
```

The synthetic-weight comparisons under `--example` are kernel-level
microbenchmarks (no real model), useful for isolating one shader at a
time:

```bash
cargo run --release --features metal -p larql-compute --example compare_decode     # Q4_K vs Q8, KV cached
cargo run --release --features metal -p larql-compute --example compare_generation  # Prefill + decode
cargo run --release --features metal -p larql-compute --example compare_pipeline    # Attention + FFN breakdown
cargo run --release --features metal -p larql-compute --example compare_formats     # Q4_KF vs Q4_K vs GGUF
cargo run --release --features metal -p larql-compute --example compare_ollama      # Synthetic LARQL vs live Ollama
```

The synthetic-weight numbers run faster than real-vindex decode (no
weight-load / lm-head overhead). The real number is what `larql bench`
reports against a production vindex.

### Benchmarks: Profile (bottleneck analysis)

```bash
cargo run --release --features metal -p larql-compute --example profile_components   # Every op isolated over 34 layers
cargo run --release --features metal -p larql-compute --example profile_operations   # CPU vs Metal per-operation
cargo run --release --features metal -p larql-compute --example profile_kernels      # Q4 v1-v5, sparse, attention
cargo run --release --features metal -p larql-compute --example profile_raw_dispatch # Pure kernel, zero overhead
cargo run --release --features metal -p larql-compute --example profile_new_kernels  # New model-agnostic kernels
cargo run --release --features metal -p larql-compute --example profile_kv_cache     # Attention vs cache length
cargo run --release --features metal -p larql-compute --example profile_bandwidth    # Raw memory throughput
```

### Benchmarks: Best Run

```bash
cargo run --release --features metal -p larql-compute --example best_pipeline       # Full pipeline, 1 cmd buffer
cargo run --release --features metal -p larql-compute --example best_multi_layer     # Multi-layer batch
```

### Diagnostics: parity bisect

When a forward path drifts (CPU vs Metal, or Metal decode vs a fresh
prefill), the per-stage bisect tool localises the divergence to a
single sub-stage of a single layer. This is the diagnostic that
closed the open Gemma 4 31B parity gap (2026-04-25 ship log) — every
attention-side stage at L0 matched at `cos=1.0`, the first
divergence appeared at `ffn_out_raw` / `down_out`, pointing at the
`q4k_ffn_gate_up` shader.

```bash
# Per-layer end-of-layer diff: CPU prefill vs Metal prefill
cargo run --release --features metal -p larql-inference \
    --example residual_diff -- <vindex> "The capital of France is"

# Per-stage L0 diff: CPU prefill vs Metal KV-cached decode
cargo run --release --features metal -p larql-inference \
    --example stage_bisect -- <vindex> "The capital of France is" 0
```

`stage_bisect` exposes the public `larql_inference::residual_diff::stages`
API; the same calls back the regression suite at
`larql-inference/tests/test_decode_stage_bisect.rs`.

## Documentation

| Doc | Content |
|-----|---------|
| [PERFORMANCE.md](PERFORMANCE.md) | Benchmark data, component profiling, optimization history |
| [ROADMAP.md](ROADMAP.md) | Planned optimizations, performance targets |
| [docs/adr/](docs/adr/) | 12 architectural decision records (design choices, algorithm origins, per-layer params, encoder merging) |
| [docs/shaders.md](docs/shaders.md) | Metal kernels with origin, performance, parameters (may lag the source — see the Shaders table above for the current production set) |
| [docs/quantization-formats.md](docs/quantization-formats.md) | Q4_0, Q4_K, Q4_KF, Q6_K, Q8_0 format specs |
| [docs/decode-pipeline.md](docs/decode-pipeline.md) | Decode data flow, dual-path architecture, KV cache |

## Design Principles

1. **Trait-based dispatch** — callers use `ComputeBackend` exclusively
2. **One file per kernel family** — ~38 shader files under `src/metal/shaders/`, each containing related kernels
3. **Zero-copy mmap** — `newBufferWithBytesNoCopy` for weight buffers
4. **Safe by default** — `read_buffer_f32` with bounds checking
5. **Feature-gated** — Metal with `--features metal`, CPU always available
6. **Auto-calibration** — benchmarks CPU vs GPU at startup for routing threshold
7. **Dual-path decode** — auto-detects Q4_K vs Q8 weights, uses optimal pipeline
8. **GGUF-compatible** — Q4_K/Q6_K formats match Ollama's quantization

## License

Apache-2.0
