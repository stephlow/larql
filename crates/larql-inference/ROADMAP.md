# Roadmap — larql-inference

## Current: ~95 tok/s (Metal Q4K) | Ollama: ~101 tok/s | 4 KV engines

## Status

The four KV-cache engines shipped in `engines/kv_engines/` all reach ~93-95 tok/s
on Gemma 3 4B using the Metal Q4K path (matching Ollama within 6%). See bench:

```
larql bench gemma3-4b-q4k --engine markov-rs,unlimited-context,turbo-quant,apollo
```

---

## P0: Engine performance parity

### TurboQuant Metal K/V checkpoint compression
**Impact**: Reduces boundary checkpoint from 278 KB → 36 KB/window (7.7×) for long contexts.
**Effort**: Medium
**Status**: TurboQuant runs at Metal speed. Compressed boundary checkpoints require
Metal K/V read-back (saving last-position K/V to CPU after each window close).
Add `backend.get_kv_last_position(layer)` to the Metal backend.

### Apollo `prefill_to_layer` — true layer-skip
**Impact**: Apollo's compressed path currently starts `forward_from_layer` at
`crystal_layer=30` but still embeds query tokens from scratch. True skip would
start the forward pass with the boundary residual as the KV context, saving
another ~20% per step.
**Effort**: Low — `forward_from_layer` exists; need to pass prior K/V correctly.
**Status**: `forward_from_layer` ships; K/V seeding at crystal_layer is a follow-up.

### Apollo store builder
**Impact**: Currently requires pre-built NPY/NPZ store files. Add
`ApolloEngine::build_from_document(weights, tokenizer, document_tokens)` that
builds the store in memory without disk files.
**Effort**: Medium (needs residual capture at crystal_layer during prefill).
**Status**: Not started.

---

## P1: Architecture coverage

### Wire v_shares_k into forward pass
**Impact**: Correct K=V handling for Gemma 4 without runtime tensor probing  
**Effort**: Low  
**Status**: `v_shares_k()` trait method done in larql-models (returns `config.attention_k_eq_v`). Forward pass currently detects K=V by checking for a missing `v_proj` tensor at runtime — swap to use the config flag directly.

### Validate PLE (per-layer embeddings) end-to-end
**Impact**: Correct Gemma 4 E2B inference  
**Effort**: Medium  
**Status**: Keys and config parsed in larql-models (`per_layer_embed_key`, `per_layer_input_gate_key`, `per_layer_projection_key`, `post_per_layer_input_norm_key`). Forward pass not yet wired. Need to add the gated per-layer embedding lookup and verify against HuggingFace reference outputs.

### KV layer sharing for Gemma 4
**Impact**: 20 fewer KV caches for Gemma 4 (20 shared layers)  
**Effort**: Medium  
**Status**: `kv_shared_source_layer()` returns correct sources in larql-models. KV cache allocation and lookup not yet sharing across layers in the inference path.

### Llama 3 / Gemma 4 engine validation
All four engines are validated on Gemma 3 4B. Llama 3 and Gemma 4 E2B/E4B pass
the architecture preconditions (RoPE, deterministic norm) but need empirical
validation of the `cos h = 1.000000` contract for MarkovRS.

### MarkovRS batched K/V recompute kernel
**Impact**: `recompute_kv` currently uses f32 BLAS for `[W, hidden] @ [hidden, kv_dim]`.
A Metal kernel for batched Q4K projection would eliminate the 2000× FLOP overhead
and bring MarkovRS close to UnlimitedContext for CPU decode.
**Effort**: Medium (new Metal shader).

---

## P2: Research

### Hybrid head caching (RS+CA)
95.5% of attention heads are static (cacheable). Caching only those heads while
keeping 4.5% dynamic KV would give ~180-370× compression at 370K tokens —
between TurboQuant (4×) and MarkovRS (287×) but with near-exact accuracy.

### Graph Walk engine
FFN-only graph walk is proven (348K features, 34 layers, zero accuracy loss via
vindex). Full RS Graph Walk requires "cracked attention" (static head caching).
When that ships, `GraphWalkEngine` can eliminate the forward pass entirely for
parametric queries.

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
| Zero warnings | 2026-04-07 | Clean build |
| PERFORMANCE.md | 2026-04-07 | Benchmark data documented |
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
| kv_engines/ subfolder | 2026-04-26 | Organised engine hierarchy |
| 106 engine unit tests | 2026-04-26 | Codec quality, routing, compliance, construction |
| kv-cache-benchmark rewired | 2026-04-25 | turbo_quant/ + apollo/ re-export from larql-inference |
