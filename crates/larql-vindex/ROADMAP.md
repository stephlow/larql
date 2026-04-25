# Roadmap — larql-vindex

## Current state (as of 2026-04-25)

- **328 tests passing** on `larql-vindex` (180 unit + 148 integration);
  211 on `larql-models`. Workspace builds clean.
- **Folder layout decomposed**:
  - `index/{storage,compute,mutate}/` — substores, KNN dispatch, mutation
  - `format/{huggingface,weights,filenames,fp4_codec,…}/`
  - `engine/` (was `storage/`) — StorageEngine + epoch + MEMIT
  - `config/{index,quantization,model,compliance,dtype}.rs` — was the
    624-line `types.rs` monolith
  - No `.rs` file > 750 lines (down from 1366 monolith)
- **Quant dispatch via `quant::registry`** — adding the next K-quant is
  one table entry plus codec functions; ~3-file edit.
- **Filename literals centralised** in `format::filenames` (252+
  occurrences → one constant module). Round-2 added 8 missed
  constants (LM_HEAD_BIN + FP4 family + attn_q4/q8 manifests).
- **`VectorIndex` god struct decomposed** into four typed substores
  (`GateStore`, `FfnStore`, `ProjectionStore`, `MetadataStore`). Adding
  a new field is one edit in the relevant store.
- **5 storage formats**: f32, f16, Q4_0, Q4_K/Q6_K (Ollama-compatible),
  Q8, FP4/FP8 (exp 26).
- Mmap zero-copy with adaptive residency.
- HNSW graph index wired into `gate_knn` (opt-in via `--hnsw`).
- Q4_K dequant cache LRU-bounded via `--max-q4k-cache-layers`.
- Patch system for editable knowledge (`PatchedVindex` overlay).
- **Vindexfile `FROM hf://...`** — HF resolution wired through the
  same resolver `larql run` and `larql extract` use.
- **Streaming extract checkpoints + auto-resume** — phase-level
  progress recorded to `.extract_checkpoint.json`; gate + down_meta
  phases auto-skip on a compatible checkpoint.
- **Stage labels centralised** in `extract::stage_labels` (15 labels;
  typo at any site is now a compile error).
- `make coverage` + `make coverage-summary` (cargo-llvm-cov).
- Bench rig daemon-aware (`make bench-vindex-scaling` refuses if
  `larql-server` / `larql-router` are running on the host).

---

## P0: Active

Nothing in P0 is currently blocking — all known critical-path issues
have landed.

## P1: Active

### Perf round-4 (2026-04-25): three concrete wins identified

End-to-end decode is 86.7 % GPU forward — vindex itself is a thin
mmap shim during real decode. But the bench survey found three
measurable vindex-side wins. All have benches already wired; record
before/after numbers in commit messages.

**Mmap design constraint** — keep the mmap zero-copy path the production
fast lane. MoE experts (Kimi K-series, DeepSeek-V3+) and multi-shard
grid servers (`larql-router` + per-layer-range `larql-server` shards)
depend on each shard mmaping its slice without paying for full-tensor
heap clones. Anything that adds heap-side caching on the hot path is a
regression for those workloads — wins below either delete heap caches
(W2) or live entirely outside the mmap lane (W1, W3).

#### W1. `top_k_from_scores` → bounded min-heap ✅ shipped 2026-04-25
**Impact**: 5.4 MB → 16 KB allocation per walk on Gemma 4B shape;
**-18 % gate_knn @ 4096×512**, **-62 % walk @ 14L×4096×512**;
flat at 10240×2560 (BLAS dominates)
**Effort**: 2 hours actual
**Bench**: `cargo bench -p larql-vindex --bench vindex_ops -- gate_knn_per_layer`
(also `walk_all_layers`)
**Status**: ✅ Shipped — `top_k_by_abs` free fn at `gate_knn.rs`,
inline copies in `gate_walk` and `gate_knn_top_per_position` routed
through it. Full 330-test suite green; clippy clean.

| Bench | Before | After | Δ |
|---|---|---|---|
| gate_knn 4096×512 | 425 µs | 352 µs | -18 % |
| walk 14L×4096×512 | 5.79 ms | 2.20 ms | -62 % |
| gate_knn 10240×2560 | 2.66 ms | 2.65 ms | flat |

`gate_knn.rs:181` allocates a `Vec<(usize, f32)>` of size N (full
score vector) and runs `select_nth_unstable_by` to get K. For walks
with K ≪ N, replace with a fixed-size min-heap (K = top_k) walked
once over the scores. Same comparator (`abs` order); allocation drops
from O(N) to O(K).

#### W2. Q4K down cache — investigate, don't blindly delete
**Impact**: Up to ~840 MB potential RSS removal, plus a hot-path
mutex — *if* a transposed-row alternative can be built. Premise of
the bench was wrong: `q4k_cache` measures `[intermediate, hidden]`
(gate/up shape) where row beats cache 230× at K=100. But the cache
*only* fires on down, which is `[hidden, intermediate]` on disk
(PyTorch `nn.Linear` orientation). There is no per-feature down
decode without either (a) a new transposed-block kernel, or (b) a
new on-disk feature-major Q4K down file.
**Effort**: 1–2 days for option (a); larger with format change for (b)
**Bench**: Need a new bench that decodes one feature's down vector
from `[hidden, intermediate]` Q4K bytes — both the cache path and
any new transposed-row path — to measure the actual trade-off
**Status**: Investigation. Don't delete the cache until the
replacement kernel exists.

Side findings — even without removing the cache, these are cheap
cleanups worth doing:
- `q4k_ffn_row_dot_via_cache` is documented as "currently unused";
  delete if grep confirms.
- `q4k_ffn_row_scaled_add` for `component == 2` uses
  `bytes_per_row(hidden)` which is wrong for the transposed layout.
  It's never called via `ffn_row_scaled_add` (the dispatch routes
  down to the cache path) but the dead branch is a footgun. Either
  delete it for `component == 2` or document the constraint.

#### W3. Parallelize HNSW warmup (across layers) ✅ shipped 2026-04-25
**Impact**: 8-layer dense HNSW warmup **3.6×** (395 → 109 ms); 4-layer
MoE warmup **2.8×** (785 → 276 ms). Estimated 34-layer Gemma 4B
warmup goes from ~2.6 s serial to ~700 ms.
**Effort**: half-day actual
**Bench**: `cargo bench -p larql-vindex --bench hnsw_decode -- hnsw_warmup`
(new bench shipped with this change)
**Status**: ✅ Shipped — added `warmup_hnsw_all_layers()` API:
parallel-builds across layers via rayon, with the cache lock held
only at the snapshot + install boundaries. Per-layer HNSW build
remains serial (algorithm requires it). Side-fix: `get_or_build_hnsw`
no longer holds the cache lock across the ~76 ms build, so concurrent
KNN queries on different layers don't block.

| Bench | Serial | Parallel | Speedup |
|---|---|---|---|
| dense-8L (10240×2560) | 395 ms | 109 ms | 3.6× |
| moe-4L (32768×2560) | 785 ms | 276 ms | 2.8× |

Speedup is sub-linear in cores because BLAS itself spawns threads
inside each parallel HNSW build (oversubscription). Future: bound
BLAS to 1 thread inside the warmup pool to recover the missing
factor.

### Cached layer decode for template-fixed layers (L0–12) — parked
**Impact**: 155+ tok/s decode (skip 13 of 21 layers)
**Effort**: Medium
**Status**: ⏸ Parked — depends on upstream work that isn't ready yet.
Don't start until the prerequisite lands. Keep `CachedLayerGraph` in
`larql-inference` as the integration point.

### Layer-level resume within an incomplete phase
**Impact**: A run interrupted at gate-layer-30-of-34 today re-runs
all 34 layers; layer-level resume would skip 30
**Effort**: Medium
**Status**: Forward-looking — phase-level resume now in place
(2026-04-25 round-3); the layer-level extension needs mid-phase file
truncation to the last clean layer boundary, which is more delicate
than the phase flag.

## P2: Forward-looking

### Parallelize gate KNN for batch inference
**Impact**: 2–4× prefill throughput on multi-token batches
**Effort**: Medium
**Status**: Forward-looking

`gate_matmul` already runs across all positions in one BLAS call but
the per-position top-K selection is sequential. Rayon-shard the
selection across rows (or fold into a single batched argpartial). Not
urgent — Metal kernel work (Q6_K dequant + 8-rows/TG) is the bigger
throughput lever.

### `VindexStorage` trait abstraction
**Impact**: Lets Redis / S3 / GPU-residency backends plug in
**Effort**: Medium
**Status**: Forward-looking

The substore extraction got most of the way there. Formalise a
sealed `VindexStorage` trait (mmap-agnostic row accessor) so Q4K row
reads can route through Redis-cached or S3-buffered backends without
walk-kernel changes.

### Expert-level sharding protocol
**Impact**: Unlocks > 256-expert MoE sharding-within-layer
**Effort**: Medium
**Status**: Forward-looking

Today `larql-router` shards by layer, not by expert ID within a
layer. For DeepSeek-V4-class models (1K+ experts) experts need to
shard across servers. Add an `ExpertRoute` message type to
`larql-router-protocol` and wire `GridState` dispatch.

### Q5_K / Q3_K / BF16 quant additions
**Effort**: Small per format (≈ 3 files thanks to the registry)
**Status**: Not yet needed — add when a target model demands it

Path: implement codec functions in `larql-models/src/quant/ggml/`,
add one entry to `QUANT_FORMATS` in `quant::registry`, add match arm
in `larql-compute::backend::quant_matvec`. Verified by the round-2
audit.

### Multi-model vindex
**Status**: Research

Store features from multiple models in one vindex. Compare
representations across architectures.

### Incremental extraction
**Status**: Research

Add new layers / features to an existing vindex without full rebuild.

---

## Won't fix

- **`detect.rs` (1391 L) split** in `larql-models` — cohesive single
  entry point dispatching to 12 architectures. Splitting fragments
  without modularity gain. Reconsider when a second detection system
  emerges (auto-discovery from model ID, multi-modal config).

---

## Completed

### 2026-04-25 — round-3 polish

| Item | Outcome |
|------|---------|
| Split `config/types.rs` (628 L) | → `config/{index,quantization,model,compliance}.rs` + back-compat `types` alias module |
| HuggingFace resolution in Vindexfile | `FROM hf://...` directives now resolve via `format::huggingface::resolve_hf_vindex` |
| Streaming extract phase checkpoints | `extract::checkpoint::Checkpoint` written to `.extract_checkpoint.json` after each phase; cleared on full success; 6 unit tests |
| Auto-resume from checkpoint | `gate_layer_infos` persisted in checkpoint; on resume the gate + down_meta phases are skipped and existing files reused; incompatible checkpoints discarded with warning |
| `extract::stage_labels` constants module | 15 callback labels (8 stages + 6 components + relation_clusters) extracted from 65+ literal sites — typo'd `on_stage_done("gate_vectro")` is now a compile error |
| GGUF Q4_K format check | No-op — 144-byte GGUF-canonical layout was already in use everywhere; only fixed a stale 148-byte comment in `larql-compute/src/pipeline.rs` |

### 2026-04-25 — second audit + round-2 cleanup

| Item | Outcome |
|------|---------|
| Add 8 missing filename constants | `LM_HEAD_BIN` (10×), `GATE_VECTORS_FP4_BIN` (7×), `DOWN_FEATURES_FP8_BIN` (5×), `UP_FEATURES_FP4_BIN` (4×), 4× attn manifests |
| Migrate ~20 unmigrated `Q4_K`/`Q6_K` dispatch sites | Most in `larql-inference` (q4k_forward, walk_ffn, pipeline_layer); routed through `quant::registry::lookup` |
| Replace 2× `unwrap_or("Q4_K")` silent fallbacks | `attn.rs`, `ffn_store.rs` — now error on missing/unknown format tags |
| `storage/` → `engine/` rename | Top-level lifecycle dir; back-compat alias `pub use engine as storage;` |
| Duplicate `fp4_storage.rs` rename | `format/fp4_codec.rs` (codec) + `index/storage/fp4_store.rs` (runtime store) |
| Merge `ffn_data.rs` into `ffn_store.rs` | Struct + impls + Clone in one file |
| Inline `gate_trait.rs` (198 L) | Block moved into `index/core.rs` |
| `accessors.rs` → `gate_accessors.rs` | Disambiguates the gate-specific accessors |

### 2026-04-25 — first audit + round-1 cleanup

| Item | Outcome |
|------|---------|
| `quant::registry` — single dispatch table | Q5_K addition drops from 8 files to 3; deletes ~12 silent-fallback `_ => None` arms |
| `format::filenames` — 19 (then 27) constants | 244 filename literals consolidated |
| Folder split: `index/{storage,compute,mutate}/` | 11 files moved; backwards-compat aliases |
| `gate.rs` (992) split | → `compute/gate_knn.rs` (615) + `storage/gate_store.rs` (446) |
| `walk.rs` (862) split | → `storage/ffn_store.rs` (720) + `compute/q4k_dispatch.rs` (168) |
| `VectorIndex` god struct → 4 substores | `GateStore` / `FfnStore` / `ProjectionStore` / `MetadataStore` |
| `format/huggingface.rs` (1366) split | → `huggingface/{mod,download,publish,discovery}.rs` |
| `format/weights/write.rs` (1249) split | → `weights/{write_f32,write_q4k}.rs` |
| `larql-models/src/quant/ggml.rs` (1352) split | → `quant/ggml/{mod,legacy,q4_k,q6_k,quantize}.rs` |
| Naming pass `Q4k` → `Q4K` | 8 occurrences across 24 files; serialised tags unchanged |
| Coverage tooling | `make coverage` + `make coverage-summary` (cargo-llvm-cov) |
| GGML round-trip tests | Q4_0 / Q4_K / Q6_K with frozen tolerance bounds |
| Golden save/load test | Deterministic save, KNN bit-exact across save/load, mmap zero-copy invariant, HNSW post-reload |
| HNSW + Q4K cache benches | `benches/hnsw_decode.rs` + `benches/q4k_cache.rs` |
| README + PERFORMANCE.md refresh | Test counts, end-to-end Q4K decode timings |

### 2026-04-25 — perf audit fixes

| Item | Outcome |
|------|---------|
| Bound the Q4_K dequant cache (LRU) | `set_q4k_ffn_cache_max_layers` + `--max-q4k-cache-layers N` flag on `larql serve` |
| Q4_K interleaved madvise + per-layer prefetch | `prefetch_interleaved_q4k_layer` mirrors the Q4_0 path; wired into `walk_ffn/sparse.rs` |
| HNSW on the decode hot path | Zero-copy view for f32-mmap layers (was cloning ~100 MB / query); abs-magnitude ranking parity (oversample 4× + re-rank); `--hnsw` + `--hnsw-ef-search` flags |
| Bench rig hygiene | Refuses if `larql-(server\|router)` daemons are alive; `LARQL_BENCH_ALLOW_DAEMONS=1` override; `make bench-vindex` vs `bench-vindex-scaling` split |
| `save_gate_vectors` regression check | False alarm — criterion p=0.21, no statistically detectable change |

### 2026-04-07 — first iteration

| Item | Outcome |
|------|---------|
| Q4_K FFN loader + wiring | `interleaved_q4k.bin` end-to-end; inference `predict_honest` prefers Q4_K over Q4_0 |
| Quantizer single source of truth | Builder uses `larql-compute` (ADR-008) |
| Example cleanup (13 → 11) | Removed Q4_0 attn + Q4_0 interleaved |
| 8 ADRs documented | All major decisions recorded |
| PERFORMANCE.md + format alignment | Fresh benchmarks, verified pipeline |
| Safety doc for `mmap_optimized` | Clippy compliance |
| `VindexPatch::is_empty()` | API completeness |

### 2026-03 / 2026-04 — foundation

| Item | Date | Impact |
|------|------|--------|
| Core `VectorIndex` with mmap | 2026-03 | Foundation |
| Gate KNN (brute-force + BLAS) | 2026-03 | Walk engine |
| Walk FFN (per-feature down/up vectors) | 2026-03 | Sparse inference |
| Binary down_meta format | 2026-03 | 5× compression vs JSONL |
| F16 storage + decode cache | 2026-03 | 2× smaller gate vectors |
| Interleaved layout (gate\|up\|down packed) | 2026-04 | Reduced TLB thrash |
| Q4_0 gate vectors + interleaved | 2026-04 | 7× smaller gates |
| HNSW graph index | 2026-04 | Sub-linear KNN |
| Adaptive residency (pin/evict) | 2026-04 | Memory budget management |
| Patch system (`PatchedVindex`) | 2026-04 | Editable knowledge |
| MoE expert routing | 2026-04 | Mixtral/DeepSeek support |
| Q4_K/Q6_K attention weights | 2026-04 | Ollama-compatible |
| Q8 attention weights | 2026-04 | Higher precision option |
| Streaming extraction (mmap, per-layer) | 2026-04 | ~2 GB peak RAM |
