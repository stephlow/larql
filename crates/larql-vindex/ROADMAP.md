# Roadmap — larql-vindex

## Current State

- 167 unit tests + 137 integration tests passing, 0 build warnings
- 3 storage formats: f32, Q8, Q4_K/Q6_K (Ollama-compatible)
- Mmap zero-copy with adaptive residency
- HNSW graph index wired into `gate_knn` (opt-in via `--hnsw`)
- Q4_K dequant cache LRU-bounded via `--max-q4k-cache-layers`
- Patch system for editable knowledge

## P0: Decode-path performance

Items raised by the 2026-04-25 perf audit (see PERFORMANCE.md and the
`gpu_forward_gap` memo). Vindex-side only — Metal kernel work lives in
larql-compute's roadmap.

### Bound the Q4_K dequant cache (LRU like gate cache) — DONE
**Impact**: Caps CPU-fallback RAM at a configurable budget (worst-case
today: 10.7 GB on 4B / ~110 GB on 31B if all layers cache fully)
**Effort**: Low
**Status**: ✅ Complete (2026-04-25)
- `set_q4k_ffn_cache_max_layers` API + LRU eviction in `walk.rs`
- `q4k_ffn_cache_stats` diagnostic, surfaced via `larql bench -v`
- `--max-q4k-cache-layers N` flag on `larql serve`
- Confirmed empirically: Metal full-K decode never populates the cache
  (`q4k_ffn_cache after larql-metal: 0 populated slots, 0.0 MB`)

**Finding from 2026-04-25 audit**: the Metal hot path never populates
`q4k_ffn_cache` (`larql bench --backends metal -v` reports
`q4k_ffn_cache after larql-metal: 0 populated slots, 0.0 MB`). The
full-K Metal branch in `walk_ffn/sparse.rs:84-117` streams Q4_K bytes
through `q4k_matmul_transb` and bypasses `q4k_ffn_layer` entirely. The
dequant cache only fires in the CPU per-position fallback at
`walk_ffn/sparse.rs:145` (`hits.len() >= 512 && down_native.is_none()`)
— and there it's a 30× win because one 614 ms layer-dequant is
amortised across thousands of feature reads per token.

So the cache is correct, not pathological. What's missing is an upper
bound: a long-running CPU-only server can grow it to all 34 layers ×
105 MB on Gemma 3 4B (10.7 GB) or 60 layers × 1.85 GB on 31B (~110 GB).
Mirror the existing gate-cache pattern (`gate_cache_max_layers`,
`gate_cache_lru` in `index/core.rs` / `gate.rs:80`) for the Q4_K FFN
cache:

1. Add `q4k_ffn_cache_max_layers` (atomic) + `q4k_ffn_cache_lru`
   (Mutex<VecDeque<usize>>) to `VectorIndex`.
2. On insert in `q4k_ffn_layer`, push the layer to the LRU and evict
   from the front when the cap is exceeded; clear the evicted layer's
   slot triple.
3. Expose `set_q4k_ffn_cache_max_layers(n)` + a `--max-q4k-cache-layers
   N` flag on `larql serve` and any other long-running CLI.
4. Default cap = 0 (unbounded — keeps current behaviour). Recommend 8
   for a CPU-only Gemma 3 4B server (≈ 840 MB ceiling for the down
   leg; gate/up dequant aren't on the hot path).

### Q4_K interleaved madvise + per-layer prefetch — DONE
**Impact**: Free win on cold-page first-token latency; small steady-state
**Effort**: Low
**Status**: ✅ Complete (2026-04-25)
- `prefetch_interleaved_q4k_layer` added to `walk.rs` (manifest-aware
  for mixed Q4_K/Q6_K layouts; uniform-stride fallback otherwise)
- Wired into `walk_ffn/sparse.rs` (hot path) and
  `walk_ffn/interleaved_q4k.rs` (dequant fallback)
- Trait surface: `GateIndex::prefetch_interleaved_q4k_layer`

### Audit `save_gate_vectors` 1.4 → 2.0 ms regression — DONE (false alarm)
**Status**: ✅ Resolved (2026-04-25) — not a regression
- Criterion's own change report flagged `p = 0.21 > 0.05` ("No change
  in performance detected"); the eyeballed 40% drift was inside the CI
- `git log` shows no functional changes to the save path since
  2026-04-07 (only sibling additions: `set_up_vector`, etc.)

### Lift gate KNN out of brute-force on the decode hot path — DONE
**Impact**: 64-expert MoE 230 → ~60 ms gate KNN/layer (search + re-rank)
**Effort**: Medium
**Status**: ✅ Complete (2026-04-25)
- `gate_knn_hnsw` was already routed in `gate_knn` behind
  `hnsw_enabled`. Two production fixes landed:
  1. **Zero-copy view** for f32-mmap layers — was cloning the entire
     gate matrix per query (~100 MB on Gemma 3 4B) defeating mmap
  2. **Abs-magnitude ranking parity** — brute uses `|dot|`, HNSW
     ranked by signed dot, systematically dropping large-negative
     features. Now oversamples 4× and re-ranks at the seam to match
- New end-to-end smoke test (`gate_knn_hnsw_smoke`) verifies
  enable/disable cycle restores brute results bit-for-bit
- `--hnsw` + `--hnsw-ef-search` flags on `larql serve`
- **Caveat**: HNSW is approximate (recall 80–95%). Default off; opt-in
  for high-feature MoE where brute gemv dominates

### Bench rig hygiene — fail fast under host contention — DONE
**Impact**: Makes regression detection meaningful again
**Effort**: Low
**Status**: ✅ Complete (2026-04-25)
- `vindex_scaling` calls `refuse_under_contention()` at every bench
  group entry; refuses with non-zero exit if `pgrep -fl
  'larql-(server|router)'` matches
- `LARQL_BENCH_ALLOW_DAEMONS=1` env override for intentional in-flight
  benching
- `make bench-vindex` (synthetic, safe) and `make bench-vindex-scaling`
  (production-dim, daemon-checked) split as separate targets

## P0: Support Cached Layer Decode

### Store pre-computed residuals for template-fixed layers (L0-12)
**Impact**: Enables 155+ tok/s decode (skip 13 of 21 layers)  
**Effort**: Medium  
**Status**: Not started (infrastructure ready — CachedLayerGraph in larql-inference)

The vindex needs to store cached residuals per template. During extraction, run one forward pass per template through L0-12 and save the output residual. At decode time, look up the cached residual instead of computing 13 layers.

### Wire Q4_K FFN consumption (interleaved_q4k.bin) — DONE
**Impact**: Match Ollama's exact FFN quantization  
**Effort**: Medium  
**Status**: ✅ Complete (2026-04-07)

Added `load_interleaved_q4k()`, `has_interleaved_q4k()`, `interleaved_q4k_mmap_ref()` to vindex.
Inference `predict_honest` now prefers Q4_K FFN (`interleaved_q4k.bin`) over Q4_0.
Format tag (`ffn_format`) passed through `FullPipelineLayer` to compute for shader dispatch.

### GGUF Q4_K format option (144 bytes vs 148 bytes)
**Impact**: Direct compatibility with llama.cpp weight files  
**Effort**: Low  
**Status**: Quantizer ready in larql-compute (`quantize_q4_k_gguf`)

Add option to store attention weights in GGUF-canonical 144-byte Q4_K format (packed scales+mins in 12 bytes) instead of our 148-byte format.

## P1: Production Hardening

### HuggingFace resolution in Vindexfile
**Effort**: Medium  
**Status**: TODO in `vindexfile/mod.rs:162`

FROM directive in Vindexfile should resolve `hf://user/repo` paths.

### Streaming extraction checkpoints
**Effort**: Medium  
**Status**: Not started

Save extraction progress between layers so interrupted builds can resume.

### Q4_K FFN in vindex
**Effort**: Low  
**Status**: Not started (Q4_0 interleaved exists)

Currently FFN gate/up/down stored as Q4_0. Switch to Q4_K (matching Ollama) for better precision at similar size.

## P2: Research

### Multi-model vindex
Store features from multiple models in one vindex. Compare representations across architectures.

### Incremental extraction
Add new layers/features to an existing vindex without full rebuild.

## Completed

| Item | Date | Impact |
|------|------|--------|
| Core VectorIndex with mmap | 2026-03 | Foundation |
| Gate KNN (brute-force + BLAS) | 2026-03 | Walk engine |
| Walk FFN (per-feature down/up vectors) | 2026-03 | Sparse inference |
| Binary down_meta format | 2026-03 | 5x compression vs JSONL |
| F16 storage + decode cache | 2026-03 | 2x smaller gate vectors |
| Interleaved layout (gate\|up\|down packed) | 2026-04 | Reduced TLB thrash |
| Q4_0 gate vectors + interleaved | 2026-04 | 7x smaller gates |
| HNSW graph index | 2026-04 | Sub-linear KNN |
| Adaptive residency (pin/evict) | 2026-04 | Memory budget management |
| Patch system (PatchedVindex) | 2026-04 | Editable knowledge |
| MoE expert routing | 2026-04 | Mixtral/DeepSeek support |
| Q4_K/Q6_K attention weights | 2026-04 | Ollama-compatible |
| Q8 attention weights | 2026-04 | Higher precision option |
| Streaming extraction (mmap, per-layer) | 2026-04 | ~2 GB peak RAM |
| Safety doc for mmap_optimized | 2026-04-07 | Clippy compliance |
| VindexPatch::is_empty() | 2026-04-07 | API completeness |
| Q4_K FFN loader + wiring | 2026-04-07 | `interleaved_q4k.bin` end-to-end |
| Quantizer single source of truth | 2026-04-07 | Builder uses larql-compute (ADR-008) |
| Example cleanup (13→11) | 2026-04-07 | Removed Q4_0 attn + Q4_0 interleaved |
| 8 ADRs documented | 2026-04-07 | All major decisions recorded |
| PERFORMANCE.md + format alignment | 2026-04-07 | Fresh benchmarks, verified pipeline |
