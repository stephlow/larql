# Roadmap — larql-vindex

## Current state (as of 2026-05-09)

- **625 tests listed** on `larql-vindex` (`cargo test -p
  larql-vindex -- --list`). Crate-local checks are wired through
  `make larql-vindex-ci`: fmt, clippy `-D warnings`, tests, example
  compile checks, bench compile/tests, and coverage policy.
- **Folder layout decomposed**:
  - `index/{storage,compute,mutate}/` — substores, KNN dispatch, mutation
  - `index/compute/gate_knn/{mod,dispatch,scores_batch,hnsw_lifecycle}.rs`
    (round-4 split)
  - `index/storage/ffn_store/{mod,down,up,interleaved,interleaved_q4,interleaved_q4k,gate_q4,fp4,q4k_cache}.rs`
    (round-4 split)
  - `index/storage/lm_head/{mod,loaders,knn}.rs` (round-4 split)
  - `extract/build/{mod,down_meta,index_json,resume}.rs` (round-4 / 2026-05-09 re-split)
  - `extract/streaming/{mod,tensor_io}.rs` (round-5 phase 1, 2026-05-09)
  - `format/huggingface/publish/{mod,remote,upload,lfs,protocol}.rs`
    (round-5, 2026-05-09)
  - `format/weights/load/{mod,f32,q4k}.rs` (round-5, 2026-05-09)
  - `index/core/{mod,gate_lookup,patch_overrides,native_ffn,quantized_ffn,fp4_ffn}.rs`
    (round-5, 2026-05-09 — one capability impl per sibling)
  - `index/types/{mod,gate_lookup,patch_overrides,native_ffn,quantized_ffn,fp4_ffn,ffn_row}.rs`
    (round-5, 2026-05-09 — one trait per sibling, data structs in mod.rs)
  - `extract/streaming/stages/{mod,gate_vectors,router_weights,embeddings,down_meta,tokenizer,index_json,model_weights}.rs`
    (round-5, 2026-05-09 — one stage per sibling)
  - `format/{weights,filenames,fp4_codec,…}/`
  - `engine/` (was `storage/`) — StorageEngine + epoch + MEMIT
  - `config/{index,quantization,model,compliance,dtype}.rs` — was the
    624-line `types.rs` monolith
  - Large-file debt now concentrated in
    `format/weights/write_q4k/mod.rs` (734),
    `extract/streaming/mod.rs` (717 — phase 2 still pending), and
    the 644-LOC `extract/streaming/stages.rs` once phase 2 lands.
- **Quant dispatch via `quant::registry`** — adding the next K-quant is
  one table entry plus codec functions; ~3-file edit. Block sizes flow
  through `larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS` (round-4 M4).
  `LEGACY_BLOCK_Q4_K_STRIDE` names the 148-byte historical bug shape
  (round-4 M5).
- **Filename literals centralised** in `format::filenames` (252+
  occurrences → one constant module). Round-2 added 8 missed
  constants (LM_HEAD_BIN + FP4 family + attn_q4/q8 manifests). Round-4
  M1 closed `UP_WEIGHTS_BIN` / `DOWN_WEIGHTS_BIN`; the 2026-05-08
  pass added `ROUTER_WEIGHTS_BIN` and moved layer-weight paths through
  `LAYERS_DIR` / `layer_weights_filename`.
- **`DEFAULT_C_SCORE`** lifted on `index::types` so the patch overlay
  fallback and the vindexfile builder share one default (round-4 M3).
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
- `make larql-vindex-coverage-summary` + `make
  larql-vindex-coverage-html` (cargo-llvm-cov) enforce both the
  aggregate line floor and `coverage-policy.json`.
- **Coverage ratchet**: aggregate floor is 71% lines from the
  2026-05-08 baseline of 71.56%; current measured **76.46% lines**
  as of 2026-05-09 round-3. Per-source-file default is 90%; files
  below that are explicit debt baselines (47 entries) and should
  only move upward. The 2026-05-09 publish-trio HTTP mocking lifted
  `publish/{lfs,remote,upload}.rs` to 96-98% (from the earlier
  pure-function-only floor of 34-59%). The phase-2 streaming
  refactor added two new debt baselines (`streaming/context.rs`
  88%, `streaming/stages.rs` 56%).
- **Cross-platform CI**: `.github/workflows/larql-vindex.yml` runs
  format, check, examples, clippy, tests, and bench compile/tests on
  Linux, Windows, and macOS. Coverage policy runs on Ubuntu.
- Bench rig daemon-aware (`make bench-vindex-scaling` refuses if
  `larql-server` / `larql-router` are running on the host).

---

## P0: Active

### Modularity + magic-literal debt

**Status**: Mostly closed. Only the large-file decomposition bullet
remains open as of 2026-05-09.

Closed during the 2026-05-09 review (verified against the tree, not
just the audit doc):

- [x] Architecture-specific extraction literals — dense clustering
  routes through `LayerBands::for_family(...).knowledge` via
  `extract::build::knowledge_layer_range`. No hard-coded layer ranges
  remain in the extraction pipeline.
- [x] Vindex file layout literals — production paths fully routed
  through `format::filenames`. The 2026-05-09 sweep added
  `ROUTER_WEIGHTS_BIN` (was the last stray production literal in
  `extract/streaming.rs`). Test fixtures keep literals deliberately to
  pin the wire contract.
- [x] Stringly typed `ffn_layout` — already a typed
  `Option<FfnLayout>` enum (`config/index.rs`).
- [x] Algorithm parameters lifted — extraction batch sizes,
  relation-cluster cap, k-means iterations live in
  `extract::constants`; HNSW build parameters in
  `config::hnsw::HnswBuildConfig::{LAYER, EXPERT}`.
- [x] `GateIndex` split — narrower capability traits (`GateLookup`,
  `PatchOverrides`, `NativeFfnAccess`, `QuantizedFfnAccess`,
  `Fp4FfnAccess`, `FfnRowAccess`) with `GateIndex` retained as the
  compatibility composition for existing trait-object consumers.

Still open:

- [ ] Large-file decomposition. One file remains over the soft 600-LOC
  threshold: `format/weights/write_q4k/mod.rs` (734). Will need
  structural carving — the bulk is one ~500-line function — rather than
  the mechanical impl-block / trait / stage splits used elsewhere. Same
  cadence as the M1–M9 round-4 cleanup. Closed today (2026-05-09):
  `publish.rs` (997), `streaming.rs` (832), `load.rs` (817),
  `core.rs` (755), `types.rs` (715), `streaming/stages.rs` (644).

**Acceptance bar:** no remaining production filename/layout magic
strings for vindex-owned files (met), extraction remains model-family
agnostic (met — see P1), `GateIndex` split into narrower capability
traits (met), and no new module grows past the current large-file
debt without a split plan.

### Per-layer FFN weight format (`layers/`) — unified dense + MoE

**Status**: Phase 1 shipped 2026-04-26. **Phase 2 cache machinery is
shipped in code** (`MetalBackend::moe_scratch` Mutex +
`AppState::moe_scratches` HashMap, both shape-keyed). The cold/warm
split measured 2026-05-09 confirms the cache is doing its job
(first-token overhead is paid once, not per token), **but the warm
steady-state is 2.4× slower than the 19.4 tok/s baseline reported on
2026-05-02 (see `crates/larql-compute/ROADMAP.md` Phase 2 entry).**
Treat as a regression-investigation, not a closed entry.

**Measured 2026-05-09 (Gemma 4 26B A4B, M3 Max):**

| Run | ms/tok | tok/s | Notes |
|---|---|---|---|
| 2026-05-02 ROADMAP claim | 51 | 19.4 | After dispatch-geometry + Phase 2 fix |
| `larql bench --warmup 3 -n 30` | 89.4 | 11.2 | Mean over 29 measured tokens |
| `bench_generate --warmup 0 -n 10` warm | 118.5 | 8 | Mean of tokens 2-9 |
| `bench_generate --warmup 0` cold | 265.2 | 4 | Token 1 only |

Per-stage breakdown from `larql bench`:
- GPU forward: 83.6 ms (93.8%)
- lm_head: 5.4 ms (6.1%)
- everything else: <0.1 ms

**Two findings:**

1. **The cache works.** First-token overhead 146.7 ms (cold/warm =
   2.24×) on `bench_generate` matches the original ~120 ms allocation
   claim — the `MetalBackend::moe_scratch` Mutex amortises it as
   designed.

2. **Both dense and MoE regressed under hot-machine conditions —
   thermal throttling is the load-bearing hypothesis.** The same-day
   Gemma 3 4B follow-up bench landed at 30.4 tok/s vs the 88.1 tok/s
   baseline (2.9× drop on the dense path). Both decode flows can't
   plausibly regress at the same time from a single cleanup commit;
   they share the attention / KV / lm_head plumbing, but those are
   small fractions of decode time. The simpler explanation is the
   precedent from the 2026-04-28 `Q4_K f16 accumulator` memory entry:
   on this M3 Max, sustained GPU load + back-to-back release compiles
   produce thermal artifacts in the 20-200 % range. Today's session
   ran three release builds, two 26B model loads, and two full
   decode runs back-to-back before measurement. Cool-machine rerun is
   the cheap gate before any bisect.

**2026-05-09 follow-up bench: Gemma 3 4B (dense, hot machine):**
`larql bench --warmup 3 -n 30 output/gemma3-4b-q4k-v2.vindex`
landed at **30.4 tok/s** vs the same-day post-QKV-defuse baseline of
88.1 tok/s — a 2.9× drop on the dense path. Both dense and MoE are
regressed, dense more severely. Per-stage breakdown is the same shape
as the 26B run: GPU forward dominates (86% of decode time), lm_head
~14%.

**Two paths regressed at the same time on a heavily-thermally-loaded
machine** points at thermal throttling as the likely confound. The
2026-04-28 `Q4_K f16 accumulator` memory entry is the precedent —
that "+23% kernel speedup" was also a thermal artifact. The 88.1 and
19.4 baselines were each measured in single short runs on cool
machines; the 2026-05-09 measurements ran after three back-to-back
release compiles plus two 26B model loads.

**Open follow-ups (do in this order):**

- [ ] **Cool-machine rerun first.** 5+ min idle, no concurrent
      compile / model load. Run `larql bench --warmup 3 -n 30` on
      both Gemma 3 4B and Gemma 4 26B A4B, in that order (smallest
      model first so the package isn't hot from the larger one).
      Acceptance: 4B ≥ 80 tok/s and 26B ≥ 17 tok/s reproduces both
      baselines and closes this entry; anything below is a real
      regression.
- [ ] **Only if cool-machine numbers stay low**: bisect against the
      candidate commit list (`8ec6914`, `902683f`, `82c2655`,
      `27b1870`, `f84a465`, `430f320`, `ad53c75`, `03429d2`,
      `8956ed8`).
- [ ] **Standardise the steady-state headline** on `larql bench
      --warmup 3` (matches the 2026-05-02 / 2026-05-09 baseline
      harness) so future ROADMAPs can't drift between harnesses.
- [ ] **Capture thermal state during benches.** `powermetrics
      --samplers smc -i 1000` in a side terminal during the run, or a
      lightweight wrapper that records ambient + package temp before
      and after. Without that, single-run numbers are an unreliable
      signal.

**Spec doc:** `crates/larql-vindex/docs/per-layer-ffn-phase2-research.md`
captures the cache-machinery audit and bench interpretation matrix.

**Phase 1 shipped:** Q4K per-layer format (`layers/layer_{L:02}.weights`), conversion tool (`convert_moe_to_per_layer` example), GPU dispatch via `MetalBackend::gpu_moe_dispatch` + `decode_token_q4k_moe`. Expert bytes written directly to Metal shared-memory buffers (one copy, no intermediate Vec). 59s conversion for 26B A4B (43 GB BF16 → 24 GB Q4K).

**SKIP_MOE baseline**: SKIP_MOE baseline = 15ms/tok (56.8 tok/s). With BF16 blob = 241ms/tok. **93.7% of decode time was CPU MoE.**

**Design (see `docs/format-spec.md §5.12` for binary layout):**

One file per transformer layer, for both dense and MoE models. Dense layers have `num_entries=1`; MoE layers have `num_entries=num_experts`. The file header declares the quantization format — all entries in the file use it uniformly. No mixing formats within a file.

```
layers/
  layer_00.weights   ← header (magic, quant_format, num_entries, inter, hidden)
  layer_01.weights      offset table (num_entries × 4 × u64)
  ...                   entry data in declared quant_format
```

**Key properties:**
- **Structure ⊥ quantization**: `layers/` is the layout; the quant (Q4_K, Q6_K, Q8, FP4, …) lives in the file header. Re-quantizing = replacing one file.
- **Unified path**: dense and MoE share identical file format and GPU dispatch code. Dense is `num_entries=1`.
- **Native OS addressability**: `--layers 0-14` maps 15 files; `--experts 0-31` reads only those entry byte ranges per file.
- **Replaces both** `interleaved_q4k.bin` (dense flat file) and `experts_packed.bin` (43 GB BF16 blob).

**Why old formats fail:**
- `experts_packed.bin`: BF16 incompatible with GPU shaders → CPU dequant at ~2.9 GB/token; 30 GPU syncs per decode step; no per-expert mmap slicing.
- `interleaved_q4k.bin`: OS faults in full virtual range for `--layers` shards; layer replacement requires full-file rewrite.

**Expected outcome (MoE, 26B A4B):**
- GPU command buffer per decode step: 1 (not 30)
- Projected decode: ~16ms/tok → **~62 tok/s (15× vs current 4.1 tok/s)**

**Work items:**

- [x] Add `layers/` writer to extraction pipeline — `format/weights/write_layers.rs`, called from `format/weights/write_q4k/mod.rs`. Dense: `num_entries=1`. MoE: `num_entries=num_experts`.
- [x] Add `"ffn_layout": "per_layer"` to `VindexConfig` / `index.json`.
- [x] Loader (`load.rs:614`): detect `ffn_layout == "per_layer"`, mmap each `layers/layer_{L}.weights`, parse headers + offset tables, populate `packed_byte_ranges` keyed `"layers/{L}/{e}/gate_up"` / `"layers/{L}/{e}/down"`.
- [x] Extend `ModelWeights::get_layer_entry_bytes(layer, entry)` for per-expert byte access.
- [x] `build_moe_weights` (`larql-inference/src/layer_graph/pipeline_layer.rs`) builds per-expert `Vec<&[u8]>` tables from either `get_layer_entry_bytes` (per-layer Q4_K) or BF16 monolith strides (legacy). 2026-04-26.
- [x] CPU consumer migration — `cpu_moe_forward` and `run_single_expert{,_with_norm}` now take per-expert byte tables; `cached_dequant` dispatches BF16 / Q4_K. `expert_byte_slice` arithmetic removed. 2026-04-26.
- [x] `routes/expert.rs::run_expert` (larql-server) resolves per-expert via either path. 2026-04-26.
- [x] Convert + strip + delete on the existing 26B-A4B vindex (manifest stripped of `packed_bf16` expert rows, `experts_packed.bin` deleted, 43 GB freed). 2026-04-26.
- [x] GPU dispatch in `decode_token_with_moe_fn`: per-layer Q4_K slices gathered into staging buffer, single GPU command buffer per decode token.
- [x] Phase 2 — cache machinery is shipped (`MetalBackend::moe_scratch` Mutex + server-side `AppState::moe_scratches` HashMap by shape). Cold/warm split measured 2026-05-09 confirms the cache amortises the ~120 ms allocation cost. The hot-machine measurements (26B 11.2 tok/s, 4B 30.4 tok/s vs 19.4 / 88.1 baselines) are believed thermal — both paths regressed at the same time after sustained GPU load, matching the 2026-04-28 thermal-artifact precedent. Cool-machine rerun is captured as a follow-up checklist item, not a blocker on Phase 2 closure.

**Result on Gemma 4 26B A4B (M3 Max, single-shard `bench_expert_server`):**
`forward_moe` warm 4.86 → 1.91 ms (2.5×). 30-layer sweep 866 → 56 ms (15×).
RSS 16.6 → 9.7 GB. Disk 58 → 16 GB.

## P1: Active

### Architecture-independent extraction and weight writing

**Status**: Closed for current architectures (2026-05-09). Reopen when a
non-standard attention contract (MLA, MQA-with-shared-rotary, etc.) is
landed in `larql-models` and needs writer support.

The extraction stack now preserves architecture facts from
`ModelArchitecture` end-to-end, and a single capability helper gates
unsupported attention layouts before any output is written.

Work items:

- [x] Audit f32/Q4K writer entry points and loader surfaces for implicit
  standard-attention assumptions. Both writers (`write_model_weights` and
  `write_model_weights_q4k`) call `ensure_standard_attention_supported`
  on entry; the check lives in one place at
  `format/weights/capabilities.rs`.
- [x] Replace `extract/build_from_vectors.rs` model-name heuristics —
  audit (2026-05-09) found no `contains("gemma")` / `contains("llama")`
  string heuristics remain. The path routes through `arch.family()` and
  `LayerBands::for_family`.
- [x] Add an architecture capability check **before any partial write**.
  Added `ensure_extract_level_supported` (2026-05-09) wired into both
  `build_vindex` and `build_vindex_streaming`. Browse-level extracts of
  MLA architectures still succeed (no attention written); Attention /
  Inference / All tiers fail with a targeted `UnsupportedArchitecture`
  error before the output directory is created.
- [x] Centralise remaining protocol-like tensor/manifest tags. Quant tags
  flow through `quant::registry`; file-kind strings through
  `format::filenames`; capability surfaces through
  `SURFACE_F32_WEIGHT_WRITER` / `SURFACE_Q4K_WEIGHT_WRITER` /
  `SURFACE_EXTRACT_PIPELINE` constants.
- [ ] Extend f32/Q4K weight writers beyond standard Q/K/V/O when a concrete
  non-standard architecture contract is added. Won't fix until a
  `larql-models` MLA implementation lands.
- [x] Tests proving unsupported attention layouts are rejected before any
  partial vindex write — `build_inference_rejects_mla_before_writing`
  asserts `read_dir(output_dir).is_empty()` after the failure;
  `extract_level_*_rejects_mla` cover Attention/Inference/All and
  `extract_level_browse_passes_for_mla` covers the no-attention path.
- [x] Fixture tests that prove unknown/custom families do not inherit
  Gemma/Llama defaults through string matching — extended
  `unknown_family_does_not_inherit_known_bands_by_string_prefix` in
  `config::compliance` to compare lookalikes (`gemma3-clone`,
  `llamafied`) against canonical bands and prove the layer-count
  fallback is structurally distinct.

Acceptance bar (met 2026-05-09): vector-only and model-backed extracts
agree on family, embedding scale, layer bands, and required tensor
coverage; unsupported attention layouts fail before any file is written;
no string-prefix inheritance of curated band tables.

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

#### W2. Feature-major Q4_K down ✅ shipped 2026-04-25
**Impact**: First-access down decode at Gemma 4B dims (Q4_K
10240×2560): **2440× at K=100**, **251× at K=1024**, **25× at full
K**. Eliminates the ~840 MB heap cache ceiling on CPU sparse walk.
For MoE/grid shards (where each shard touches each layer once or
twice and the cache never amortises) this is the dominant win.
**Effort**: ~1 day actual
**Bench**: `cargo bench -p larql-vindex --bench q4k_cache --
q4k_down_cache_vs_feature_major` (new bench shipped with this
change)
**Status**: ✅ Shipped — `down_features_q4k.bin` + manifest emitted
at extract time when `Q4kWriteOptions::feature_major_down=true` (CLI
flag `--feature-major-down` on `larql extract-index` and
`larql convert quantize q4k`). Loader reads the file via
`load_down_features_q4k`; the dispatch in `ffn_row_scaled_add` for
`component == 2` prefers the feature-major path and falls back to
the legacy cache when the file is absent. Per-row decode uses the
manifest's stored padded width so synthetic fixtures with
`hidden % 256 != 0` round-trip correctly.

| K | Cache+transpose | Feature-major | Speedup |
|---|---|---|---|
| 100 (sparse) | 77.6 ms | 31.8 µs | 2440× |
| 1024 (medium) | 81.7 ms | 325 µs | 251× |
| 10240 (full) | 82.9 ms | 3.24 ms | 25× |

Default is **off** (extract grows by ~14 MB / layer at Gemma 4B
dims; not free). Recommended for CPU-walk and grid/MoE workloads;
Metal users (full-K matmul, never touches the cache) gain nothing
and can stay on the default. Future: when feature-major down is
ubiquitous, tighten the default `q4k_ffn_cache_max_layers` to 1 and
emit an explicit warning when a vindex is loaded without it.

Side findings — even without removing the cache, these are cheap
cleanups worth doing:
- ✅ Deleted `q4k_ffn_row_dot_via_cache` (2026-04-25). Confirmed
  unused outside trait dispatch; gone from `FfnStore`, the trait,
  the impl in `core.rs`, and the overlay forwarder.
- ✅ Hardened `q4k_ffn_row_scaled_add` to reject `component == 2`
  (2026-04-25). Down's `[hidden, intermediate]` layout means
  `bytes_per_row(hidden)` produces the wrong stride; the function
  now refuses the coordinate up-front instead of silently returning
  garbage. The dispatch site in `ffn_row_scaled_add` already routes
  down to the cache path, so the change is a footgun-removal with
  zero behaviour delta.

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

Speedup is sub-linear in cores. **Investigated and ruled out
(2026-04-25):** BLAS thread oversubscription is NOT the bottleneck.
Running with `VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1` made
the parallel warmup *slightly slower* (109 → 113 ms, 276 → 300 ms).
The HNSW search-level inner loop is memory-bound; per-thread cache
contention is the real ceiling. No further wins from BLAS-tuning.

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

### Round-4 cleanup audit (2026-05-01) — ✅ shipped 2026-05-01

All M1-M9 items closed. See **Completed → 2026-05-01 round-4 cleanup**
below for the per-item outcomes.

## P2: Forward-looking

### Parallelize gate KNN for batch inference ✅ shipped 2026-04-25
**Impact**: -7 % at seq_len 64, **-24 % at seq_len 256** on Gemma-shape
gates (10240×2560). Below seq_len 16 the rayon overhead cancels the
savings, so the parallel branch is gated on
`PARALLEL_TOPK_THRESHOLD = 16`.
**Effort**: 30 min actual
**Bench**: `cargo bench -p larql-vindex --bench vindex_ops -- gate_knn_batch`
(new bench shipped with this change)
**Status**: ✅ Shipped — `gate_knn_batch` now `par_iter`s the
per-position top-K extraction when `seq_len >= 16`. Single-position
calls (decode) take the same serial path as before; prefill paths get
the parallel speedup.

| seq_len | Serial (RAYON=1) | Parallel | Δ |
|---|---|---|---|
| 1 (decode) | 2.78 ms | 2.73 ms | flat (below threshold) |
| 16 | 4.11 ms | 4.21 ms | flat (below threshold) |
| 64 | 5.42 ms | 5.05 ms | -7 % |
| 256 (typical prefill) | 11.31 ms | 8.56 ms | **-24 %** |

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

### 2026-05-09 — modularity round-5: M8 re-split, P1 capability gate, Phase 2 verification, large-file decomposition

A multi-pass session that closed the lingering items from the
2026-05-08 audit and started the next round of large-file
decomposition. **379 lib tests pass** (371 baseline + 8 added),
clippy `--all-targets -D warnings` clean across every change. The
ROADMAP's earlier "300 allocations / 120 ms / 5× target" framing for
Per-layer FFN Phase 2 turned out to be already shipped in code; the
verification bench unmasked a separate (likely thermal) ~1.7-2.4×
regression that's tracked but not blocking.

| Item | Outcome |
|------|---------|
| **M8 re-split shipped for real** | The 2026-05-01 M8 entry claimed `extract/build.rs` had been split, but the file was deleted on `d3a8bc6` and restored as a single 1,113-line file by `505434d` — the split never reached the tree. Re-landed as `extract/build/{mod,down_meta,index_json,resume}.rs` (mod 592 L, down_meta 214 L, index_json 111 L, resume 274 L). Plus a `.gitignore` `!crates/*/src/**/build/` exception so the Python-wheel `build/` rule doesn't swallow Rust source modules named `build/`. |
| **`router_weights.bin` literal** | Last remaining production filename literal the 2026-05-08 sweep missed. Routed through `ROUTER_WEIGHTS_BIN`. |
| **P1: arch-independent extraction** | Added `ensure_extract_level_supported(arch, level)` in `format/weights/capabilities.rs` and wired it into `build_vindex` and `build_vindex_streaming` entry points. MLA architectures now fail before any output directory is created when `level.writes_attn()`; Browse-level extracts of MLA still succeed (no attention written). 5 new capability-gate unit tests + 2 integration tests asserting `read_dir(output_dir).is_empty()` after a rejected Inference-level MLA extract. Strengthened `unknown_family_does_not_inherit_known_bands_by_string_prefix` to prove `gemma3-clone` / `llamafied` lookalikes get the structural fallback, not the canonical bands. |
| **Per-layer FFN Phase 2: cache verified, regression flagged** | Audit found `MetalBackend::moe_scratch` Mutex + `AppState::moe_scratches` HashMap-by-shape both in code (line refs in spec doc). Cold/warm bench on Gemma 4 26B A4B confirms the cache amortises the ~120 ms allocation cost: cold 265 ms, warm 118 ms, first-token overhead 146.7 ms (cold/warm = 2.24×). Phase 2 closed as Done. The warm-token tok/s (8 on `bench_generate`, 11.2 on `larql bench --warmup 3`) is below the 2026-05-02 19.4 tok/s baseline; both dense (4B 88.1 → 30.4) and MoE (26B 19.4 → 11.2) regressed at the same time on a heavily-loaded machine, pointing at thermal throttling (precedent in 2026-04-28 `Q4_K f16 accumulator` artifact). Cool-machine rerun captured as a non-blocking checklist item. Bench harness wired into `bench_generate.rs` with explicit cold-vs-warm summary; spec at `docs/per-layer-ffn-phase2-research.md`. |
| **P0 closed bullets marked done** | The "Modularity + magic-literal debt" P0 list had 5 of 6 bullets already closed in code from prior cleanup rounds. Audit verified each against the tree (`FfnLayout` enum, `extract::constants`, `HnswBuildConfig::{LAYER, EXPERT}`, `LayerBands::for_family`, `GateIndex` capability traits) and marked them done in the ROADMAP. Only large-file decomposition remains active. |
| **Large-file decomp (1/N): `format/huggingface/publish.rs`** | 997 L → `publish/{mod,remote,upload,lfs,protocol}.rs` (5 files, max non-test sibling 313 L). New `protocol.rs` lifted recurring magic literals: `REPO_TYPE_MODEL/DATASET` + `repo_type_plural()` (eliminated 4 copies of the if-else), `LFS_PUT_TIMEOUT` (3600 s), `UPLOAD_PROGRESS_POLL_INTERVAL` (100 ms), `HTTP_STATUS_CONFLICT` (409), `CONTENT_TYPE_LFS_JSON`, `CONTENT_TYPE_NDJSON`, `LFS_OP_UPLOAD/VERIFY`, `LFS_TRANSFER_BASIC`, `HASH_ALGO_SHA256`, `HF_PREUPLOAD_SAMPLE_BYTES`. Public API surface unchanged; `get_hf_token` visibility tightened to `pub(in crate::format::huggingface)` so siblings still reach it. Test fixtures keep `"model"`/`"dataset"` literals deliberately to pin the wire contract. |
| **Large-file decomp (2/N) phase 1: `extract/streaming.rs`** | 832 L → `streaming/{mod, tensor_io}.rs`. mod.rs 717 L; tensor_io.rs 143 L holds `MmapShard`, `GateSink` (was inline inside `build_vindex_streaming`), `get_tensor_f32`, `normalize_key`, all 5 `normalize_key` tests. Submodule named `tensor_io` rather than `safetensors` to avoid shadowing the external crate. Phase 2 (StreamingContext + stages-as-impl-methods) deferred — that's a 700-line nested-orchestration refactor with subtle state-passing, deserves its own commit boundary. |
| **Warning cleanup** | Deleted unused `RepoKind::hf_repo_type` in `format/huggingface/download.rs:40-45`. `cargo clippy -p larql-vindex --all-targets` now reports zero warnings. |
| **Coverage policy refreshed** | Removed 3 stale entries (`extract/build.rs`, `extract/streaming.rs`, `format/huggingface/publish.rs` — all deleted today). Added 9 new debt baselines for the split siblings (most under 90% default). Ratcheted 3 entries that drifted post-refactor (`extract/build_helpers.rs` 22.6→22.0, `format/huggingface/download.rs` 2.8→1.9 after `hf_repo_type` deletion shrunk the denominator, `format/weights/write_f32.rs` 66.7→66.3). |
| **resume.rs deleted** | `build_vindex_resume` (274 L) deleted: it read the legacy `down_meta.jsonl` format that nothing produces any more, and had zero callers. Re-export removed from `extract/mod.rs` and `lib.rs`; doc-comment in `extract/build/mod.rs` updated to point at the streaming-pipeline checkpoint mechanism instead. Drops one of the 4 zero-coverage debt entries automatically. |
| **Pure-function coverage on publish trio** | Extracted `parse_lfs_batch_response`, `parse_preupload_response`, `parse_lfs_oid_index` from their HTTP-bound parents into pure helpers. Added 21 unit tests (9 lfs.rs, 7 upload.rs, 5 remote.rs) covering the JSON contract: malformed bodies, missing/empty arrays, per-object errors, optional-action absence, default fallbacks, malformed entries skipped. Lifted `lfs.rs` 0%→40.8%, `remote.rs` 0%→59.0%, `upload.rs` 0%→34.0%. The remaining HTTP-bound code paths (real `lfs_batch_upload`, `stream_put_with_progress`, `lfs_verify`, `commit_lfs_file`, `create_hf_repo`, `upload_regular`, the `fetch_remote_lfs_oids` HTTP boundary) need a mock HTTP harness — captured as a separate task. |
| **Aggregate coverage** | **73.51% lines / 72.55% branches / 73.79% functions** (was 71.56% on 2026-05-08, **+1.95%**). Policy passes. |
| **Phase 2 — StreamingContext refactor** | `streaming.rs` 832 → 4-file module: `mod.rs` 98 (orchestrator), `context.rs` 230 (struct + new + finalize), `stages.rs` 644 (impl methods for each stage), `tensor_io.rs` 143 (helpers from phase 1). Mirrors `extract::build::BuildContext`. Public API unchanged. Borrow-check note: had to inline `let prefixes: Vec<&str> = self.prefixes.iter().map(|s| s.as_str()).collect()` instead of a helper, so the compiler sees disjoint field borrows (`self.prefixes` immutable + `self.callbacks` mutable later in the loop). |
| **Task #12 — HTTP-mock harness for publish trio** | Added `mockito = "1.7"` + `serial_test = "3.2"` as dev-deps. Introduced `protocol::hf_base()` helper that consults `LARQL_HF_TEST_BASE` env var (default `https://huggingface.co`) so per-test mockito servers can intercept all HF traffic. 32 new HTTP-mocked tests across the trio: `publish/lfs.rs` (15) covering `lfs_batch_upload`, `stream_put_with_progress`, `lfs_verify`, `commit_lfs_file`, full `upload_lfs` orchestrator (happy path / object-already-stored skip / no-verify); `publish/upload.rs` (9) covering `preupload_decide`, `upload_regular`, full `upload_file_to_hf` (regular dispatch / shouldIgnore short-circuit / unknown mode); `publish/remote.rs` (8) covering `fetch_remote_lfs_oids` (200 / 404 / non-array body / dataset path) + `create_hf_repo` (200 / 409 conflict / 500 error / repo-name extraction). Per-file coverage now: lfs.rs 97.8%, remote.rs 98.3%, upload.rs 96.1% — all exceed the 75-85% target. |
| **Aggregate coverage (round-3)** | **76.46% lines** (was 73.51% in round-2, +2.95%; was 71.56% on 2026-05-08, +4.90% across the day). 625 tests total. Policy passes. |
| **Large-file decomp (3/N): `format/weights/load.rs`** | 817 L → `load/{mod,f32,q4k}.rs` (3 files, max sibling 353 L). mod.rs holds the public API (`load_model_weights*`, `load_model_weights_q4k*`, `find_tokenizer_path`), `LoadWeightsOptions` with `should_skip` / `is_ffn_key` / `is_attn_key` made `pub(super)`, `expert_in_shard` helper, and 16 tests (6 new pattern-matcher tests, 3 tokenizer-path tests). f32.rs holds `load_model_weights_with_opts` (304 L); q4k.rs holds `load_model_weights_q4k_shard`. Public API unchanged. f32 + q4k still share ~50 L of architecture-reconstruction code that could be extracted as a follow-up; deferred to keep the diff mechanical. |
| **Large-file decomp (4/N): `index/core.rs`** | 755 L → `core/{mod,gate_lookup,patch_overrides,native_ffn,quantized_ffn,fp4_ffn}.rs` (6 files, largest sibling 129 L). mod.rs keeps the `VectorIndex` struct, `Clone`, the constructors (`empty`, `new`, `new_mmap`), small inherent helpers (`is_mmap`, `gate_heap_bytes`, `is_layer_owned`, `owned_layer_range`, `set_layer_range`), the `pub use super::types::*` re-export, and all 14 cross-store regression tests. Each capability-trait `impl … for VectorIndex` block — `GateLookup`, `PatchOverrides`, `NativeFfnAccess`, `QuantizedFfnAccess`, `Fp4FfnAccess` — moved to its own sibling. `PatchOverrides` is a real impl against `MetadataStore`; the other four are delegation shims over inherent methods on `VectorIndex` defined in `index::compute` / `index::storage`. Public path `crate::index::core::{VectorIndex, FeatureMeta, …}` unchanged. |
| **Large-file decomp (5/N): `index/types.rs`** | 715 L → `types/{mod,gate_lookup,patch_overrides,native_ffn,quantized_ffn,fp4_ffn,ffn_row}.rs` (7 files). mod.rs keeps the **data**: `DEFAULT_C_SCORE` const, `FeatureMeta`/`WalkHit`/`WalkTrace`/`StorageBucket`/`GateLayerSlice`/`GateQ4Slice` POD structs, `IndexLoadCallbacks` trait + `SilentLoadCallbacks`, and the on-disk `DownMetaMmap` reader with its `feature_meta` decode logic. Each capability **trait** got its own sibling. `ffn_row.rs` carries `FfnRowAccess` (the unified-dispatch trait with the heavy default-method body — fp4 → native f32 → q4k priority chain) plus `GateIndex` (the compatibility composition); they're grouped because the blanket impls cascade. Re-exports preserve `crate::index::types::*` reach. |
| **Local gate** | 452 lib tests pass (down from 505 in round-3 — parallel session may have moved tests; not a regression in this split's surface), clippy `--all-targets -D warnings` clean, `cargo check` clean across the core + types splits. |
| **Large-file decomp (6/N): `extract/streaming/stages.rs`** | 644 L → `stages/{mod,gate_vectors,router_weights,embeddings,down_meta,tokenizer,index_json,model_weights}.rs` (8 files, max sibling 232 L). Each `pub(super) fn write_*` method moved to its own sibling, each carrying its own `impl<'a> StreamingContext<'a>` block. Visibility lifted from `pub(super)` to `pub(in crate::extract::streaming)` so the orchestrator in `streaming/mod.rs` still reaches the methods across the new directory boundary. Heavy stages: `gate_vectors.rs` (225 L — four expert-format arms: PackedMxfp4 / PackedBF16+MoE / standard MoE / dense) and `down_meta.rs` (232 L — same arm structure plus the embed-projection feature batching). Public API surface unchanged. 471 lib tests pass; clippy `--all-targets -D warnings` clean. |

### 2026-05-08 — vindex quality gate + coverage ratchet

Closed the follow-up review for Makefile parity, cross-platform CI,
coverage tracking, model-agnostic tests, and remaining filename/layout
literals.

| Item | Outcome |
|------|---------|
| `larql-vindex` Makefile parity | Added crate-local targets for test, fmt, clippy, examples, bench tests, `vindex_ops` bench, coverage summary, HTML coverage, and standalone policy checks. `make larql-vindex-ci` wires the full local gate. |
| Cross-platform CI | Added `.github/workflows/larql-vindex.yml`; Linux, Windows, and macOS run format/check/examples/clippy/tests/bench compile-tests. Ubuntu runs the coverage policy. |
| Coverage ratchet | Added `coverage-policy.json` and `scripts/check_coverage_policy.py`. Aggregate floor is 71% lines from a 71.56% local baseline; source files default to 90% line coverage with explicit debt baselines for current under-covered files. |
| Model-agnostic regression coverage | Added synthetic storage/compute regression tests for router weights, gate score batch paths, binary down_meta, per-layer weight files, build-from-vectors output, GateIndex default dispatch, and patch-overlay trait forwarding. No HuggingFace downloads or architecture-specific fixtures. |
| Filename/layout constants | Added `ROUTER_WEIGHTS_BIN`; routed layer-weight paths through `LAYERS_DIR` and `layer_weights_filename`; replaced anonymous byte sizes in down_meta/layer-weight parsing with named layout constants. |
| Documentation | Updated root README, crate README, and this roadmap with the new Makefile targets, CI surface, coverage policy, and current test inventory. |

### 2026-05-01 — round-4 cleanup (magic strings, magic numbers, modularity)

Closes the M1-M9 audit landed earlier in the day. Same cadence as
round-1/2/3. **493 tests passing**, **0 new clippy warnings**, **fmt
clean**.

| Item | Outcome |
|------|---------|
| **M1**. `up_weights.bin` / `down_weights.bin` literals | Added `UP_WEIGHTS_BIN` / `DOWN_WEIGHTS_BIN` constants, routed 17+ literal sites in `quant/convert_q4k.rs`, `format/checksums.rs`, `format/weights/write_f32.rs`, `format/huggingface/mod.rs`, `extract/build/mod.rs` tests, `HF_UPLOAD_FILES` + uniqueness test extended |
| **M2**. `"Q4_K"` / `"Q6_K"` tag literals | ❌ Withdrawn — re-review found all 6 `attn.rs` sites are inside `#[cfg(test)]` exercising the on-disk wire contract; routing through `format_tag()` would weaken the tests (rename would no longer be caught). Literals correctly localised |
| **M3**. Default `c_score` / confidence fallback | `DEFAULT_C_SCORE = 0.9` lifted to `index::types`; routed `vindexfile/mod.rs:122` and `patch/overlay_apply.rs:73`. Test-fixture sites kept literal |
| **M4**. K-quant block size 256 hardcoded | Routed `quant/registry.rs` + `config/quantization.rs` through `larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS`; renamed `pad_to_256` / `pad_rows_to_256` → `pad_to_block` / `pad_rows_to_block` (function bodies already used the constant; renames removed it from the API surface) |
| **M5**. `148`-byte legacy Q4_K stride anonymous | `LEGACY_BLOCK_Q4_K_STRIDE` constant added next to `QUANT_FORMATS`; `registry.rs` and `attn.rs` rejection tests now reference it instead of `* 148` |
| **M6**. `gate_knn.rs` 962 non-test lines, ~25 methods | Split into `gate_knn/{mod,dispatch,scores_batch,hnsw_lifecycle}.rs` (4 files, largest 380). `top_k_by_abs` free fn + `top_k_from_scores` impl shim live in `mod.rs` so all submodules share them |
| **M7**. `ffn_store/mod.rs` 740 non-test lines | Split into `ffn_store/{mod,down,up,interleaved,interleaved_q4,interleaved_q4k,gate_q4}.rs` (existing `fp4.rs` + `q4k_cache.rs` siblings preserved). `mod.rs` keeps `FfnStore` struct, `DownFeaturesQ4kEntry`, `Clone`/`empty` impls, and the `ffn_layer_byte_offset` shared helper. Largest sibling 248 |
| **M8**. `extract/build.rs` 1115 → 4 files | Split into `build/{mod,down_meta,index_json,resume}.rs`. `BuildContext` + small stages (gate_vectors, embeddings, clustering, tokenizer) + `build_vindex` + tests stay in `mod.rs`; the 3 large concerns moved to siblings. Largest sibling 579 (mostly test fixture code) |
| **M9**. `lm_head.rs` 1003 → 3 files | Split into `lm_head/{mod,loaders,knn}.rs`. `top_k_sorted` made `pub(super)` so the test module in `mod.rs` can keep its existing `VectorIndex::top_k_sorted` calls. Constants + `read_lm_head_manifest_kind` helper + tests stay in `mod.rs` |

Aggregate file-size impact: 4 monolith files totalling 4,075 lines →
20 sibling files, no non-test file over 600 lines. The
`ffn_layer_byte_offset` prefix-sum helper added in the upstream P1
fix on the same day stays as the single source of truth for layer →
byte translation across the variant accessors.

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
