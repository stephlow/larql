# Decode Pipeline — larql-compute

How `decode_token` processes one token through all layers with KV cache.

## Overview

```
Input: x[hidden] (embedded token)
Output: h[hidden] (final hidden state for logit projection)

Per layer (~11 dispatches, all in a SINGLE Metal encoder):
  1. Fused norm + QKV projection (q4k_q6k_qkv_proj_normed — 1 dispatch)
     OR: rms_norm (1) + q4k_q6k_qkv_proj (1) = 2 dispatches
  2. Fused QK-norm Q+K (qk_norm_qk — 1 dispatch, was 2)
  3. Fused RoPE Q+K (rope_at_pos_batched_qk — 1 dispatch, was 2)
  4. Batched V-norm (optional, Gemma 4)
  5. KV cache append + attend (SIMD reductions)
  6. O projection (q4k_matvec)
  7. Fused residual+norm (residual_norm_store — 1 dispatch, writes both
     ffn_norm_out and h_post_attn; was 2 dispatches)
  8. FFN gate+up fused (q4k_ffn_gate_up — 1 dispatch)
  9. GEGLU activation
 10. FFN down (q6k_matvec)
 11. Post-FFN residual add
```

All layers run in a **single Metal command buffer with a single global encoder**.
No per-layer encoder create/end overhead. Apple Silicon serialises compute
dispatches within an encoder so no explicit barriers are needed.

## Dispatch fusion history

Starting from ~14 dispatches/layer (476/token), 5 fusions land in 2026-04-25:

| Fusion | Dispatches saved | Technique |
|---|---|---|
| `qk_norm_qk` | 34/token | One dispatch for Q+K heads instead of two |
| `rope_at_pos_batched_qk` | 34/token | One dispatch for Q+K heads |
| `residual_norm_store` | 34/token | Writes normed + raw sum simultaneously |
| `q4k_q6k_qkv_proj_normed` | 34/token | Norm computed inline in QKV TGs |

Current: **~374 dispatches/token** (~1.9ms overhead at 5µs/dispatch).
Ollama estimate: ~272 dispatches (~1.4ms).

## Dual-Path Architecture

`decode_token` auto-detects the weight format from `FullPipelineLayer.wq.format`.

### Q4_K + Q6_K Path (production — Gemma 3 / 4 Ollama extracts)

```
h_buf [f32]
  → q4k_q6k_qkv_proj_normed (RMS norm inline + fused Q4_K Q/K + Q6_K V)
  → qk_norm_qk (fused Q+K norm)
  → rope_at_pos_batched_qk (fused Q+K RoPE)
  → v_norm_batched (optional, Gemma 4)
  → kv_cache_append + kv_attention
  → q4k_matvec (O projection)
  → residual_norm_store → ffn_norm_out [f32] + h_post_attn [f32]
  → q4k_ffn_gate_up → geglu_gelu_tanh → q6k_matvec (down)
  → residual_add → h_buf [f32]
```

### Q4_KF Path (fastest for Q4_KF vindexes)

```
h_buf [f32]
  → rms_norm → norm_f32 [f32]
  → q4kf_qkv_proj → Q, K, V [f32]
  → rope_at_pos_batched_qk + kv_attach
  → q4kf_proj (O) → residual_norm_store → FFN via q4kf_proj
```

### Q8 Path (legacy)

```
h_buf [f32]
  → rms_norm_q8 (fused) → q8_buf + q8s_buf
  → q8_qkv_proj → Q, K, V → kv_attend
  → quantize_q8 → q8_matvec (O)
  → residual_norm_q8 → FFN (same as Q4_K)
```

## KV Cache

```rust
pub struct KVCache {
    pub layers: Vec<LayerKVCache>,
}

pub struct LayerKVCache {
    pub k_cache: Buffer,    // [max_seq, num_kv_heads, head_dim] f32
    pub v_cache: Buffer,    // same
    pub current_len: usize,
    pub max_seq: usize,     // default 4096
}
```

Populated during prefill; extended by `kv_cache_append` each decode step.
`kv_attention` attends Q against all cached K/V (positions 0..current_len).

## Hybrid MoE — Batched Prefill Path (2026-04-26)

For hybrid MoE models (e.g. Gemma 4 26B A4B), each decoder layer has both
a dense FFN block (GPU) and a sparse expert block (CPU). `dispatch_full_pipeline`
accepts an optional `moe_fn` callback that fires after each MoE layer's dense FFN.

**Before (token-by-token loop):**
```
for pos in 0..seq_len:
    decode_token(layers, h[pos])   // ALL layers per token
```
O(seq_len × num_layers) GPU command buffer commits.

**After (batched per layer):**
```
for l in 0..num_layers:
    GPU: dispatch all seq_len positions through layer l's attention + dense FFN
    commit + wait
    if layer l has MoE:
        CPU: moe_fn(l, h_post_attn[0..seq_len], new_h[0..seq_len])
             ↳ experts for all positions + outer_norm + layer_scalar
```
O(num_layers) commits. For a 5-token prefill on 26 MoE layers: **26 commits vs 130**.

**Key invariant:** The GPU `layer_scalar` step (step 11) is skipped for MoE layers
when `moe_fn` is provided. The callback applies `layer_scalar` itself after
combining dense + MoE output — matching HF's `hidden_states *= layer_scalar`
placement at the end of `Gemma4TextDecoderLayer.forward`.

**Measured gain (Gemma 4 26B A4B, M3 Max, 15 warmup / 30 tokens):**

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| Prefill (5-token) | 1889ms | 1297ms | **−31%** |
| Decode GPU fwd | 334ms/tok | 246ms/tok | **−26%** |
| Decode tok/s | 2.9 | **3.9** | **+35%** |

**KV cache:** Per-layer variant `populate_kv_one_layer` (in `kv_copy.rs`)
copies one layer's K/V scratch immediately after each per-layer commit,
so the cache is current before the MoE callback reads `h_post_attn`.

## Performance (M3 Max, 2026-04-26)

### Gemma 3 4B (dense, 34 layers)

| Path | GPU fwd | tok/s | vs Ollama |
|---|---|---|---|
| **Q4_K+Q6_K decode (34L)** | **11.1ms** | **75–79** | **1.24–1.30×** |
| Ollama gemma3:4b | ~8.5ms | 97–103 | 1.0× |

Per-stage: GPU fwd 83%, lm_head 17%.

### Gemma 4 26B A4B (hybrid MoE, 26 layers, batched prefill)

| Metric | tok/s | GPU fwd/tok |
|---|---|---|
| **LARQL Metal** | **3.9** | **246ms** |

Effective bandwidth: LARQL ~329 GB/s, Ollama ~348 GB/s (Gemma 3).
Total weight data per token: 3029 MB (34 layers × 89.1 MB/layer).
See `PERFORMANCE.md` for the full bandwidth budget and gap analysis.
