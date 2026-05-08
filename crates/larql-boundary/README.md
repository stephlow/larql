# larql-boundary

Confidence-gated BOUNDARY ref codec for [LARQL](https://github.com/chrishayuk/chuk-larql-rs).

Implements Phases 1–3 of the [BOUNDARY_REF_PROTOCOL spec](../../experiments/43_residual_stream_codec/BOUNDARY_REF_PROTOCOL.md):

| Phase | Module | What it does |
|-------|--------|--------------|
| 1 | `codec` | Compress / decompress residual vectors |
| 2 | `metadata` | Compute per-boundary confidence fields from logit slices |
| 3 | `gate` | Per-boundary compression decision |

## Quick start

```rust
use larql_boundary::{codec, gate, metadata};
use larql_boundary::gate::BoundaryGateConfig;

// Phase 1: encode a residual (2564 bytes for d=2560, vs 5120 for bf16)
let residual = vec![0.1f32; 2560];
let payload = codec::int8::encode(&residual);
let decoded = codec::int8::decode(&payload);

// Phase 2: compute metadata from logits (caller runs the forward pass)
let raw_logits = vec![0.0f32; 262_145]; // vocab size
let mut meta = metadata::compute(&raw_logits, None);

// Phase 3: gate decision (default config = calibration_mode=true → always bf16)
let config = BoundaryGateConfig::default();
let decision = gate::apply(&mut meta, &config);
```

## Codec schemes

| Scheme | Bytes (d=2560) | Ratio | Contract |
|--------|----------------|-------|----------|
| `codec::bf16` | 5 120 | 1× | Exact |
| `codec::int8` | 2 564 | 2× | D- (ArgmaxNearEquivalent) |

**Why σ-clipping?** Gemma-class residuals have absmax/σ ≈ 92×. Absmax
quantisation wastes 99% of int8's 256 levels on rare extreme values.
3σ-clipping concentrates all levels on the ±3σ band where the
prediction-relevant geometry lives.

See [Exp 43 SPEC](../../experiments/43_residual_stream_codec/SPEC.md) for
the full characterisation (30 prompts, layer 33, Gemma 3 4B).

## Model-agnostic design

This crate takes `&[f32]` slices only — no model weights, no inference
backends, no MLX dependency. The caller (typically `larql-inference`)
runs the forward pass and passes logit slices in.

## Calibration mode

`BoundaryGateConfig::default()` has `calibration_mode = true`, which
overrides all thresholds and always falls back to bf16.

**Exp 44 Track A result** (396 boundary positions from Frankenstein, Gemma 3 4B):

| Threshold (log-prob) | Accept rate | Early-div (≤5 steps) | Total div | System compression |
|---------------------:|------------:|---------------------:|----------:|-------------------:|
| 0.00 (all) | 100% | 10.0% | 21.1% | 2.00× |
| 1.35 | 82.2% | 8.1% | 21.1% | 1.82× |
| **2.16 (recommended)** | **68.9%** | **4.8%** | **19.8%** | **1.69×** |
| 4.05 | 47.8% | 4.7% | 21.3% | 1.48× |
| 6.75 | 14.4% | 0.0% | 5.8% | 1.14× |

Recommended threshold 2.16 log-prob units meets: accept ≥ 50% and early_div < 5%.

**What the contract actually guarantees at 2.16:**
`ArgmaxNearEquivalentHighMargin` (D-@high) means the first ~5 generated tokens are safe
at the 4.8% point-estimate level (95% CI ≈ 1.6%–10.7%, n=62 accepted boundaries). Total
divergence over 20 tokens is ~19.8% — the same as lower thresholds. The early-div filter
is what does the safety work; cascade compounds after the first wrong token. Use D-@high
for LARQL's boundary-to-fresh-decode use case where the decoder quickly accumulates its
own context; do not rely on it for long uninterrupted continuation from the boundary.

To use calibrated thresholds:

```rust
let config = BoundaryGateConfig {
    calibration_mode: false,
    min_log_prob_margin: 2.16,  // fitted by Exp 44 Track A (log-prob units)
    min_top1_prob: 0.5,
    ..Default::default()
};
```

See `experiments/44_boundary_gate_calibration/` for the full calibration run.

## Running examples

```sh
cargo run -p larql-boundary --example encode_decode
cargo run -p larql-boundary --example gate_decision
```

## Tests

```sh
cargo test -p larql-boundary
```

## Benchmarks

```sh
cargo bench -p larql-boundary
```

## Platform support

Pure Rust, no `unsafe`, no OS-specific code. Tested on macOS, Linux,
and Windows. Uses `to_le_bytes()` / `from_le_bytes()` throughout for
endian safety.

## Protocol spec

Full spec: [`experiments/43_residual_stream_codec/BOUNDARY_REF_PROTOCOL.md`](
../../experiments/43_residual_stream_codec/BOUNDARY_REF_PROTOCOL.md)
