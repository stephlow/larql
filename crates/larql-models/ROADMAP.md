# Roadmap — larql-models

## Current: 12 architectures, 221 tests, safetensors + GGUF loading

## P0: Code Quality (from 2026-04-26 review)

### Fix silent dtype skip in safetensors loader
**Impact**: Unsupported dtypes drop silently — no warning, no error  
**Effort**: Tiny  
**Status**: Done 2026-04-26

Added `skipped_tensors: Vec<(String, String)>` to `ModelWeights`. Both silent-skip sites in `loading/safetensors.rs` now pattern-match `UnsupportedDtype` explicitly (collecting key + dtype name) and bubble up any other error with `return Err(e)` rather than swallowing it. Callers can inspect `weights.skipped_tensors` to see which tensors were skipped and why (integer tensors like attention masks are benign; unexpected entries indicate a format gap).

### Tests for `q4k_row_scaled_add` / `q6k_row_scaled_add` / NEON vs scalar parity
**Impact**: NEON paths on hot decode path are untested  
**Effort**: Low  
**Status**: Done 2026-04-26 — 10 new tests added; `q4k_row_dot_scalar` exposed as `pub(super)` to match q6k pattern

Tests added:
- `q4k_row_dot_neon_matches_scalar_{single,multi}_block`
- `q4k_row_dot_matches_dequantized_dot`
- `q4_k_dequantize_known_nonzero_values` (verifies exact decoded values, not just shape)
- `q4k_row_scaled_add_matches_alpha_times_deq`
- `q6k_row_scaled_add_matches_alpha_times_deq`
- `q{4,6}k_row_scaled_add_rejects_misaligned`

### Constants for config field name variants
**Impact**: grep confusion when a new config alias appears  
**Effort**: Tiny  
**Status**: Done 2026-04-26 — `NUM_EXPERTS_KEYS`, `NUM_EXPERTS_PER_TOK_KEYS` consts + `field_u64` helper in `detect.rs`. Adding a new alias is a one-line change to the const.

### `normalize_key` / `normalize_key_pub` duplication
**Impact**: Dead indirection  
**Effort**: Tiny  
**Status**: Done 2026-04-26 — `normalize_key_pub` removed, `normalize_key` promoted to `pub(crate)`, `gguf.rs` call site updated.

### Consolidate MXFP4 dequant into `quant/mxfp4.rs`
**Impact**: Logical cohesion — MXFP4 decode is split between `loading/safetensors.rs:288–383` and `quant/mxfp4.rs`  
**Effort**: Low  
**Status**: Done 2026-04-26 — `split_gate_up_experts` added to `quant/mxfp4.rs` (GPT-OSS fused gate/up split logic + 2 tests). Loading function renamed `load_mxfp4_expert_tensors`, unused `_vectors` param removed, down projection loop uses `into_iter` to avoid `.clone()`.

### Note on quant/dequant crate split
**Decision**: `larql-models/quant/` is **format deserialization** (GGUF/safetensors → f32). `larql-compute` has **compute operations** (quantized matvec, Metal shaders). The split is correct. The `f16_to_f32` copies in `larql-compute/cpu/ops/q4k_matvec.rs` and `q6k_matvec.rs` are intentional — CPU reference impls for Metal shader testing, isolated by design. `larql-compute` is dev-only dep; don't flip that direction.

## P1: Architecture Coverage

### Phi-3 / Phi-4
**Effort**: Low  
**Status**: Not started

Similar to Llama with some attention differences (partial RoPE, SuRoPE). Most trait defaults apply.

### Command R / Cohere
**Effort**: Medium  
**Status**: Not started

Different attention key pattern, different norm placement.

### Mamba / state-space models
**Effort**: Large  
**Status**: Research

Would require extending the trait beyond transformer assumptions (no attention keys, no KV cache). May warrant a separate trait hierarchy.

## P2: Loading Improvements

### Streaming safetensors loading
**Effort**: Medium  
**Status**: Not started

Current loader reads all shards into memory. For 70B+ models, streaming with per-layer loading would reduce peak memory. Already have mmap infrastructure — extend to lazy loading with `Arc<Mmap>` references.

### GGUF quantized inference (skip dequant)
**Effort**: Large  
**Status**: Not started

Currently GGUF tensors are dequantized to f32 during loading. For Q4_K/Q6_K formats, keep data in quantized form and pass directly to `larql-compute` Q4_K shaders. Requires a `QuantizedWeights` variant alongside `ModelWeights`.

### MLX npz/safetensors hybrid
**Effort**: Low  
**Status**: Partial (MLX safetensors work, npz not yet)

Apple MLX models sometimes use `.npz` format. Add npz parsing alongside safetensors.

## P3: Trait Evolution

### Per-layer FFN type
**Effort**: Low  
**Status**: Not started

Some models (e.g., future MoE variants) may have different FFN types per layer (dense for early layers, MoE for later). Add `ffn_type_for_layer(layer)` method.

### Attention pattern abstraction
**Effort**: Medium  
**Status**: Research

Current sliding window is boolean per layer. Future models may have more complex patterns (local + global hybrid, dilated attention, prefix caching hints). Consider a richer `AttentionPattern` enum.

### Config validation
**Effort**: Low  
**Status**: Not started

Add a `validate()` method to `ModelArchitecture` that checks for inconsistencies (e.g., head_dim doesn't divide hidden_size, num_experts set but not num_experts_per_token). Currently these fail silently at inference time.

## Completed

| Item | Date | Impact |
|------|------|--------|
| ModelArchitecture trait | 2026-03 | Foundation — 82 methods with defaults |
| Gemma 2/3 support | 2026-03 | QK-norm, softcapping, sliding window |
| Llama/Mistral/Qwen/DeepSeek | 2026-03 | Core architecture coverage |
| Mixtral MoE (PerExpert) | 2026-03 | Expert key patterns |
| GPT-OSS (PackedMxfp4) | 2026-03 | MXFP4 dequantization, packed expert keys |
| Granite (scaling multipliers) | 2026-03 | Embedding/residual/attention/logits scaling |
| StarCoder2 | 2026-03 | LayerNorm, bias, GELU |
| GGUF loading | 2026-03 | Q4_0/Q4_1/Q8_0/F16/BF16 dequantization |
| Safetensors mmap + HF cache | 2026-03 | Zero-copy loading, cache resolution |
| drop_ffn_weights | 2026-04 | Walk-only mode saves ~13GB |
| Gemma 4 architecture | 2026-04 | Per-layer geometry, PLE, KV sharing, V-norm, layer scalars |
| Gemma 4 31B + E2B configs | 2026-04 | Both variants tested with real config.json |
| Gemma4Arch re-export | 2026-04-07 | Public API complete |
| v_shares_k from config | 2026-04-07 | Uses attention_k_eq_v flag instead of hardcoded false |
| Gemma 3 qk_norm_weight_offset | 2026-04-07 | Was missing (Gemma 2 had it, Gemma 3 didn't) |
| Full test coverage (130 tests) | 2026-04-07 | All 12 architectures tested: Gemma 2/3/4, Llama, Mistral, Mixtral, Qwen, DeepSeek, GPT-OSS, Granite, StarCoder2, Generic |
| GGML quant test gaps closed (51 tests) | 2026-04-26 | q4k_row_dot NEON≡scalar, q4k/q6k scaled_add correctness, Q4_K known nonzero values |
| Silent dtype skip fixed | 2026-04-26 | `skipped_tensors` field on ModelWeights; UnsupportedDtype collected, other errors bubbled |
| normalize_key_pub removed | 2026-04-26 | Dead wrapper gone; `normalize_key` is `pub(crate)` |
| Config alias constants | 2026-04-26 | `NUM_EXPERTS_KEYS`, `NUM_EXPERTS_PER_TOK_KEYS`, `field_u64` helper in `detect.rs` |
| MXFP4 consolidation | 2026-04-26 | `split_gate_up_experts` in `quant/mxfp4.rs`; loader thinned + renamed |
| Clippy clean (zero warnings) | 2026-04-07 | lib + examples + tests all pass `-D warnings` |
| Documentation suite | 2026-04-07 | README, ROADMAP, PERFORMANCE, 3 docs, 6 ADRs |
| Example suite (3 demos) | 2026-04-07 | architecture_demo (all 12), demo_tensor_keys (all 12), demo_loading |
