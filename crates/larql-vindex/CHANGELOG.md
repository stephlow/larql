# Changelog — larql-vindex

All notable changes to `larql-vindex` are documented here.

The format follows the conventions of [Keep a Changelog](https://keepachangelog.com/),
with dated entries (`YYYY-MM-DD`) instead of semantic versions during the
pre-1.0 phase. Forward-looking work lives in [`ROADMAP.md`](ROADMAP.md).

## [2026-05-10] — `VindexStorage` trait skeleton + `MmapStorage` parity wrapper

Steps 1–3 of the `VindexStorage` migration (P0 active in
`ROADMAP.md`).

### Step 1 — sealed trait skeleton
- New `index/storage/vindex_storage/mod.rs`: `VindexStorage` trait
  sealed via private `sealed::Sealed` supertrait, 14 byte-yielding
  methods covering all hot-path substore reaches (FFN Q4_K +
  whole-buffer, attn Q4_K/Q4/Q8, lm_head Q4/f16/f32, gate views,
  W2 down). FP4 + DownMeta deliberately not behind the trait:
  both carry richer per-feature decoders that aren't a clean fit
  for the byte-handle abstraction. Documented inline.
- `bytes = "1"` added to vindex Cargo.toml (already transitive via
  reqwest / hf-hub).

### Step 2 — `MmapStorage` parity impl
- Sibling `mmap_storage.rs` with the production
  `VindexStorage` impl backed by cloned substore mmap fields.
  `from_substores(&FfnStore, &GateStore, &ProjectionStore, hidden_size)`
  for prod wiring; `empty(hidden)` for tests / inert indices.
- `Arc<Mmap>` and `Arc<Vec<u8>>` (synth lm_head) bridge to
  `Bytes::from_owner` via a generic `ArcAsBytes<T: AsRef<[u8]>>`
  newtype.
- 7 inline tests (parity, out-of-bounds rejection, refcount-clone).

### Step 3 — `BytesView<'_>` redesign + Criterion bench gate
First pass returned `Bytes` from per-layer accessors. Bench showed
that scheme paid **6 atomic ops per fetch** (3× `Bytes::slice` +
3× `Bytes::drop`) — measured 12× the direct cost on layer-fetch,
2.4× on per-row. That's a no-go for the `Arc<dyn VindexStorage>`
hot path.

Redesign: per-layer accessors return `BytesView<'a>` — a borrowed
`(&'a Bytes, offset, length)` triple with `as_slice() -> &'a [u8]`
(zero atomics, pure pointer arithmetic) and `to_owned_bytes()` as
the opt-in refcounted handle for callers that need the bytes to
outlive the borrow. Whole-file accessors keep `Bytes` (one-time
fetch, not hot).

Numbers after the redesign (M3 Max, release):

| Path | Layer-fetch (3 layers) | Per-row (3 layers × 64 rows) |
|---|---|---|
| direct (`&[u8]`) | 9.49 ns | 69.30 ns |
| `MmapStorage` concrete | 10.10 ns (1.06×) | — |
| `MmapStorage` via `Arc<dyn>` | 9.34 ns (0.98×) | 71.13 ns (1.03×) |

Both gaps within noise. `dyn` even slightly faster than direct on
layer-fetch (the optimizer hoists `BytesView` construction). Step 4
(substore migration) clear to proceed without anxiety.

`GateLayerView` made borrowed for consistency: `Copy` instead of
`Clone`, `bytes: &'a Bytes` instead of `bytes: Bytes`. Replaces the
three-field substore reach (`gate_mmap_bytes` +
`gate_mmap_slices[layer]` + `gate_mmap_dtype`).

### Tests + tooling
- Lib tests **866 → 877** (+11: trait skeleton ×3, MmapStorage ×7,
  BytesView round-trip ×1).
- New bench `benches/vindex_storage_dispatch.rs` registered in
  Cargo.toml — runnable via
  `cargo bench -p larql-vindex --bench vindex_storage_dispatch`.
- `cargo clippy --lib --tests --benches -- -D warnings`: clean.
- No callsite changes — purely additive.

### Step 4 — `storage` field on `VectorIndex` + accessor migration

- New `storage: Arc<dyn VindexStorage>` field on `VectorIndex`.
  Initialized in `empty` / `new` / `new_mmap` to
  `Arc::new(MmapStorage::empty(hidden))`; populated state captured
  by a new `pub(crate) fn refresh_storage(&mut self)` that rebuilds
  via `MmapStorage::from_substores(...)`.
- Loaders that mutate substore mmap or manifest fields call
  `refresh_storage()` at the end:
  `load_attn_q8` / `load_attn_q4k` / `load_attn_q4`,
  `load_lm_head_q4` / `synthesize_lm_head_q4` / `set_lm_head_f16_mmap`
  / `load_lm_head`, `load_interleaved_q4k` / `load_down_features_q4k`,
  `load_interleaved_q4`, `load_gate_q4`. `Clone for VectorIndex`
  also clones the `Arc<dyn VindexStorage>` (single refcount bump).
- Per-layer byte-yielding accessors migrated to forward through
  `self.storage` — public signatures unchanged so no callsites
  move:
  - `attn_q4k_layer_data` / `attn_q8_layer_data` /
    `attn_q4_layer_slices` (attn.rs)
  - `interleaved_q4k_layer_data` / `down_features_q4k_layer_data`
    (ffn_store/interleaved_q4k.rs)
  - `gate_q4_data` (ffn_store/gate_q4.rs)
- Whole-buffer accessors (`attn_q4_data`, `interleaved_q4k_mmap_ref`,
  `interleaved_q4_mmap_ref`) **not migrated in this step** — they
  return `&[u8]` borrowing from the substore mmap, which doesn't
  cleanly reshape through the `Bytes`-returning whole-buffer trait
  methods. Step 5 (which drops the substore mmap fields) will
  address this with an API-shape decision.

### Coverage

- `vindex_storage/mod.rs`: **98.00%** (49/50 lines) — clears the
  90% per-file default.
- `vindex_storage/mmap_storage.rs`: **93.69%** (297/317 lines) —
  clears the 90% per-file default.
- `attn.rs`: 56.1% → **83.29%** (forwarder shrunk the file;
  baseline raised in `coverage-policy.json` per the
  ratchet-up-only rule).
- `ffn_store/interleaved_q4k.rs`: 56.7% → **62.86%** (same
  pattern; baseline raised).
- `lm_head/loaders.rs`: 90.59% (clears 90%).
- `core/mod.rs`: 97.05% (clears 90%).

### Tests + tooling
- Lib tests stay at **877** (one attn bounds test updated to call
  `refresh_storage` after the simulated stale-manifest poke —
  direct substore mutation is a test-only pattern; production goes
  through loaders that refresh automatically).
- `cargo clippy --lib --tests --benches -- -D warnings`: clean.
- No external API change — public `&[u8]` / `&[f32]` accessor
  signatures unchanged.

### Open
- Step 5: drop substore `Arc<Mmap>` fields; compiler flags any
  remaining `pub` field reach (~155 sites across 19 files per the
  2026-05-10 audit). Whole-buffer accessor API shape resolved at
  the same time.

## [2026-05-10] — Per-layer FFN Phase 2 closed

Cool-machine bench rerun reproduced both baselines (4B ≥ 80 tok/s,
26B ≥ 17 tok/s), confirming that the 2026-05-09 hot-machine
measurements (4B 30.4, 26B 11.2) were thermal artifacts from
back-to-back release compiles + 26B model loads — same precedent as
the 2026-04-28 `Q4_K f16 accumulator` entry.

**Phase 2 closure** — `MetalBackend::moe_scratch` Mutex +
`AppState::moe_scratches` HashMap (both shape-keyed) amortise the
~120 ms first-token allocation as designed. Cache machinery shipped,
baselines hold, no bisect needed. `larql-compute/ROADMAP.md` Phase 2
entry can land alongside.

**Standardised steady-state headline**: `larql bench --warmup 3 -n 30`.
Earlier ROADMAPs sometimes drifted between `bench`, `bench_generate`,
and `--warmup 0` numbers — those are not comparable.

**Spec doc**: `crates/larql-vindex/docs/per-layer-ffn-phase2-research.md`
captures the cache-machinery audit + thermal-confound analysis.

## [2026-05-10] — P0 review follow-ups + magic-number sweep

Closes the two outstanding 2026-05-09 review follow-ups in the ROADMAP
P0 section, plus a focused magic-number sweep prompted by the audit.

### Closed

- **Hardened attention manifest accessors** (`index/storage/attn.rs`).
  `attn_q8_layer_data`, `attn_q4k_layer_data`, and `attn_q4_layer_slices`
  now validate `offset + length <= mmap.len()` (with `checked_add` for
  overflow safety) before slicing — matching the defensive behavior
  already in `down_features_q4k_layer_data` /
  `interleaved_q4k_layer_data`. A stale or corrupt attention manifest
  now returns `None` (caller can fall back) instead of a slice-bounds
  panic on query/decode. Added 3 inline tests, one per accessor, that
  load a clean vindex, mutate the manifest to point past the mmap, and
  assert `None`.

- **Replaced `walker/utils.rs::current_date()` 365/30-day approximation**
  with a Gregorian implementation mirroring `larql-inference/src/capture.rs`.
  Pulled out a `date_from_epoch_secs(secs: u64)` helper for deterministic
  tests; kept `current_date()` as the wall-clock convenience wrapper used
  by walker `extraction_date` metadata. Added 6 leap-year / boundary
  tests (1970-01-01, 2024 leap-year transitions, century boundaries).

### Magic-number sweep (related)

- **`ATTN_TENSORS_PER_LAYER = 4`** lifted on `index/storage/attn.rs`.
  Replaces three `* 4` / `+ 3 >=` patterns across the Q8 / Q4_K / Q4
  accessors. Self-documents the Q/K/V/O layout convention.
- **`FFN_COMPONENTS_PER_LAYER = 3`** + **`FFN_DOWN = 2`** lifted on
  `index/storage/ffn_store/mod.rs`. Replaces bare `* 3` and `== 2`
  literals across `ffn_store/interleaved_q4k.rs`, `ffn_store/q4k_cache.rs`
  (×2), `index/types/ffn_row.rs`, `quant/convert_q4k.rs`, and
  `format/load.rs`. Down (`FFN_DOWN`) is the special case — it's stored
  row-major `[hidden, intermediate]` and needs the transpose path.

### Tests

- Lib tests **857 → 866** (+9: 3 attn bounds, 6 calendar).
- `cargo clippy -p larql-vindex --all-targets -- -D warnings`: clean.
- `cargo fmt -p larql-vindex`: clean.
- Golden hash drift fix: `vector_extractor_ffn_down_byte_identical`
  now strips the `_header` record (which carries `extraction_date`)
  before sort+hash. Without it the golden would drift every day now
  that `current_date()` returns a real wall-clock date instead of the
  buggy 365/30-day approximation that drifted ~30× slower. Regenerated
  golden once and locked it in.

### Open in P0

- Phase 2 cool-machine bench rerun. Hot-machine numbers (4B 30.4 vs 88.1
  baseline, 26B 11.2 vs 19.4) suspected thermal — runbook in
  `ROADMAP.md`. Cannot be automated.

## [2026-05-10] — Coverage push (round-6)

The day after round-5's large-file decomp. Closed all five 2026-05-09
follow-ups, then walked the remaining ≥10%-below-90 files one by one
until the aggregate cleared 88. Plus a workspace-wide `cargo fmt` +
`clippy --all-targets -D warnings` pass at the end.

| Metric | 2026-05-09 end | 2026-05-10 end |
|---|---|---|
| Aggregate line coverage | 78.40% | **88.90%** |
| Files at ≥90% default | ~46 | **85** |
| Debt baselines | ~46 | **40** |
| Lib tests | 505 | **857** |

### 2026-05-09 follow-ups — closed

| Item | Outcome |
|------|---------|
| **Task #1 — Extract `reconstruct_architecture` + `load_embeddings` helpers** | New `load/arch.rs` (100% lines) + `load/embeddings.rs` (96.6%); ~70 LOC duplication removed from `f32.rs` / `q4k.rs`. Centralises Gemma-4 per-layer-geometry forwarding so future fields land in one place. |
| **Task #2 — `ffn_store/{down,up}.rs` coverage** | Pivoted from archaeological bisect to direct fixtures. Both files now **100% lines** (was 41% / 55%); 19 inline tests around `load_*` / `*_layer_matrix` / `*_feature_vector` / `has_full_mmap_ffn` with synthetic mmap'd `down_features.bin` / `up_features.bin` round-trips. |
| **Task #3 — `streaming/stages.rs` synthetic-safetensors fixture (partial)** | New `tests/test_streaming_stages_moe.rs` with a Mixtral-shaped model builder. Lifted `index_json.rs` 85% → 97%, `router_weights.rs` 13% → 87%, `down_meta.rs` 60% → 65%, `gate_vectors.rs` 38% → 52%. Two of four arms cleared the 90% default. |
| **Task #4 — Pure-function tests for `download.rs`** | 18 tests for `strip_etag_quoting` / `want_model_file` / `hf_cache_repo_dir` (env-var-driven, serial). `download.rs` 1.9% → 39.3%. |
| **Task #5 — Cool-machine Phase 2 rerun** | Cannot be automated. Runbook captured in `ROADMAP.md` open follow-ups: `cargo run --release -p larql-cli -- bench --warmup 3 -n 30 <vindex>`, smallest model first, ≥80 tok/s on 4B and ≥17 tok/s on 26B reproduces the 88.1 / 19.4 baselines. |

### Round-6 follow-ups (spawned by task #3 + #4) — closed

| Item | Outcome |
|------|---------|
| **PackedBF16 + PackedMxfp4 streaming fixtures** | Added Gemma 4 hybrid-MoE (PackedBF16) and gpt-oss (PackedMxfp4 with U8 blocks + e8m0 scales=127) fixtures to `test_streaming_stages_moe.rs`. **`gate_vectors.rs` 52% → 93.25%** (cleared 90% default), `down_meta.rs` 65% → 85%, `index_json.rs` → 98%, `router_weights.rs` 87%. All 4 expert-format arms now exercised end-to-end. |
| **HF_ENDPOINT-mocked download paths** | Widened `protocol::hf_base()` to `pub(in crate::format::huggingface)` and routed `head_etag_and_size` + 4 hf_hub-bound functions through it. 11 mockito-backed tests covering all 4 hf_hub-bound entry points (`resolve_hf_vindex`, `download_hf_weights`, `resolve_hf_vindex_with_progress`, `resolve_hf_model_with_progress`) on the `not-an-hf-path` early-return + 404/500 error paths. download.rs **39.3% → 74.5%**. |

### Per-file coverage push (chained 11 files past 90%)

`clustering/mod.rs` 0% → 100% (`classify_direction` + `ClusterResult` serde) ·
`ffn_store/interleaved.rs` 21.5% → **99.5%** ✓ ·
`ffn_store/interleaved_q4.rs` 12.9% → **99.0%** ✓ ·
`ffn_store/q4k_cache.rs` 31.5% → 80.9% (LRU + cache-hit + invalid-component paths) ·
`clustering/probe.rs` 29.1% → 87.2% (`extract_probe_entities` filters + `build_confirmed_pairs` cluster-meta join) ·
`index/mutate/loaders.rs` 35.1% → **97.9%** ✓ (NDJSON gate + down-meta loader round-trips) ·
`vindexfile/mod.rs` 44.8% → 72.2% (`resolve_vindexfile_path` + `build_from_vindexfile` early-returns) ·
`index/compute/q4k_dispatch.rs` 48.1% → 61.0% (early-return guards on every dispatch entry) ·
`index/storage/lm_head/knn.rs` 53.9% → 70.2% (f32 BLAS fallback + `Stride32Mode` env-var dispatch) ·
`clustering/pair_matching/database.rs` 30.3% → 83.9% (RelationDatabase add/lookup, Wikidata + WordNet JSON loaders) ·
`extract/build_helpers.rs` 22% → **98.75%** ✓ (`chrono_now` ISO-8601 format, `build_whole_word_vocab` filtering, `compute_offset_direction` 6 paths, `compute_gate_top_tokens` argmax + batched chunks, `run_clustering_pipeline` shape-mismatch + happy path) ·
`format/huggingface/discovery.rs` 5.9% → **96.5%** ✓ (re-introduced `hf_base()` indirection across 7 sites; 21 mockito tests for `repo_exists`, `fetch_collection_items`, `add_collection_item`, `ensure_collection`, `find_collection_slug`, `create_collection`) ·
`index/storage/gate_accessors.rs` 59.6% → **94.7%** ✓ (34 tests: heap+mmap branches of every accessor + `warmup` f16-decode + `describe_ffn_backend`).

### Test infrastructure added

- `vocab_tokenizer(&["word", ...])` — JSON-backed `WordLevel` builder that avoids the `AHashMap` dep on the in-memory builder. Reusable across `extract/build_helpers.rs` and `clustering/probe.rs` tests.
- `weights_with_embed(embed, vocab_size)` — minimal `ModelWeights` factory (just `embed` + arch detected from a tiny llama JSON). Used wherever a function only needs `weights.embed`.
- `f16_mmap_from(floats: &[f32])` — anonymous `MmapMut` + `encode_f16` + `make_read_only`. Reusable for any future mmap-decode test.
- `TestEnvGuard { TEST_BASE_ENV, HF_TOKEN }` — RAII env-var guard pattern for mockito-mocked HF tests, restored on drop. Used in `discovery.rs` and `download.rs`.
- `write_synthetic_gemma4_hybrid_moe` and `write_synthetic_gpt_oss_model` builders in `test_streaming_stages_moe.rs` complement the existing Mixtral builder for the four expert-format arms.

### Misc

- **`cargo fmt -p larql-vindex` clean** across the crate (25 files reformatted in the day's coverage push, mostly chained-method indentation in tests).
- **`cargo clippy -p larql-vindex --all-targets -- -D warnings` clean.** Fixed `manual_RangeInclusive::contains` (build_helpers chrono asserts), `too_many_arguments` (`#[allow]` on the 8-arg gemma4 fixture builder), 6 `err_expect` sites (`.err().expect()` → `expect_err`), one `useless_vec` (`&[0u8; 16]` → `[0u8; 16]`).
- **Walker `on_checkpoint(&mut g)` → `on_checkpoint(&g)`** in `walker/weight_walker.rs` — the trait method takes `&Graph`, so the `&mut` was unnecessary. Last clippy nit in the crate; clean across the full target tree now.
- **`mmap_demo`, `demo_memit_solve`, `q4k_demo`, `walker_demo` examples all run end-to-end.** mmap_demo verifies zero-copy demand-paging live (544 MB file → 242 MB RSS growth at 41% of layers walked); q4k_demo round-trips Q4_K dequant on a synthetic safetensors fixture.
- **Bench surface intact.** `cargo bench -p larql-vindex --bench vindex_ops -- --quick` reports "No change in performance detected" on every measured bench (criterion p<0.05).

## [2026-05-09] — Modularity round-5

A multi-pass session that closed the lingering items from the
2026-05-08 audit, finished the large-file decomposition, and lifted
every newly-split file in `index/core/` and `index/types/` to ≥90%
line coverage. Clippy `--all-targets -D warnings` clean across every
change. Final lib-test count: **614** (up from 379 at the start of
the day).

### Modularity / structure

| Item | Outcome |
|------|---------|
| **M8 re-split shipped for real** | The 2026-05-01 M8 entry claimed `extract/build.rs` had been split, but the file was deleted on `d3a8bc6` and restored as a single 1,113-line file by `505434d` — the split never reached the tree. Re-landed as `extract/build/{mod,down_meta,index_json,resume}.rs`. Plus a `.gitignore` `!crates/*/src/**/build/` exception so the Python-wheel `build/` rule doesn't swallow Rust source modules named `build/`. |
| **`router_weights.bin` literal** | Last remaining production filename literal the 2026-05-08 sweep missed. Routed through `ROUTER_WEIGHTS_BIN`. |
| **resume.rs deleted** | `build_vindex_resume` (274 L) deleted: it read the legacy `down_meta.jsonl` format that nothing produces any more, and had zero callers. Re-export removed from `extract/mod.rs` and `lib.rs`; doc-comment in `extract/build/mod.rs` updated to point at the streaming-pipeline checkpoint mechanism instead. |
| **P0 closed bullets marked done** | The "Modularity + magic-literal debt" P0 list had 5 of 6 bullets already closed in code from prior cleanup rounds. Audit verified each against the tree (`FfnLayout` enum, `extract::constants`, `HnswBuildConfig::{LAYER, EXPERT}`, `LayerBands::for_family`, `GateIndex` capability traits). Only large-file decomposition remained active going into round-5; closed by end-of-day. |

### Large-file decomposition (closed P0)

Seven monolith files split into per-concern siblings. After this round
no non-test source file exceeds the soft 600-LOC threshold. Public API
surface unchanged across every split.

| File (before) | LOC | After (largest sibling) |
|---|---|---|
| `format/huggingface/publish.rs` | 997 | `publish/{mod,remote,upload,lfs,protocol}.rs` (max non-test sibling 313 L). New `protocol.rs` lifted recurring magic literals: `REPO_TYPE_MODEL/DATASET` + `repo_type_plural()`, `LFS_PUT_TIMEOUT` (3600 s), `UPLOAD_PROGRESS_POLL_INTERVAL` (100 ms), `HTTP_STATUS_CONFLICT` (409), `CONTENT_TYPE_LFS_JSON`, `CONTENT_TYPE_NDJSON`, `LFS_OP_UPLOAD/VERIFY`, `LFS_TRANSFER_BASIC`, `HASH_ALGO_SHA256`, `HF_PREUPLOAD_SAMPLE_BYTES`. |
| `extract/streaming.rs` | 832 | `streaming/{mod,context,stages,tensor_io}.rs` (later expanded to per-stage siblings — see below). Phase 1 carved `tensor_io.rs` (143 L: `MmapShard`, `GateSink`, `get_tensor_f32`, `normalize_key`); Phase 2 introduced `StreamingContext` mirroring `extract::build::BuildContext`. |
| `format/weights/load.rs` | 817 | `load/{mod,f32,q4k}.rs` (max sibling 353 L). mod.rs holds the public API + `LoadWeightsOptions` (with `should_skip` / `is_ffn_key` / `is_attn_key` made `pub(super)`) + `expert_in_shard` + 16 tests. f32.rs holds `load_model_weights_with_opts` (304 L); q4k.rs holds `load_model_weights_q4k_shard`. |
| `index/core.rs` | 755 | `core/{mod,gate_lookup,patch_overrides,native_ffn,quantized_ffn,fp4_ffn}.rs` (max sibling 129 L). mod.rs keeps the `VectorIndex` struct, `Clone`, the constructors, small inherent helpers, and 14 cross-store regression tests. Each capability-trait `impl … for VectorIndex` block — `GateLookup`, `PatchOverrides`, `NativeFfnAccess`, `QuantizedFfnAccess`, `Fp4FfnAccess` — moved to its own sibling. `PatchOverrides` is a real impl against `MetadataStore`; the other four are delegation shims. |
| `index/types.rs` | 715 | `types/{mod,gate_lookup,patch_overrides,native_ffn,quantized_ffn,fp4_ffn,ffn_row}.rs` (7 files). mod.rs keeps the **data**: `DEFAULT_C_SCORE` const, POD structs (`FeatureMeta`, `WalkHit`, `WalkTrace`, `StorageBucket`, `GateLayerSlice`, `GateQ4Slice`), `IndexLoadCallbacks` + `SilentLoadCallbacks`, and the on-disk `DownMetaMmap` reader. Each capability **trait** got its own sibling; `ffn_row.rs` carries `FfnRowAccess` (the unified-dispatch trait — fp4 → native f32 → q4k priority chain) plus `GateIndex` (the compatibility composition); they're grouped because the blanket impls cascade. |
| `extract/streaming/stages.rs` | 644 | `stages/{mod,gate_vectors,router_weights,embeddings,down_meta,tokenizer,index_json,model_weights}.rs` (max sibling 232 L). Each `pub(super) fn write_*` method moved to its own sibling carrying its own `impl<'a> StreamingContext<'a>` block. Visibility lifted from `pub(super)` to `pub(in crate::extract::streaming)` so the orchestrator still reaches the methods across the new directory boundary. Heavy stages: `gate_vectors.rs` (225 L — four expert-format arms: PackedMxfp4 / PackedBF16+MoE / standard MoE / dense) and `down_meta.rs` (232 L — same arm structure plus the embed-projection feature batching). |
| `format/weights/write_q4k/mod.rs` | 734 | `write_q4k/{mod,attn,ffn,moe_layers,norms,ple,lm_head,feature_major_down}.rs` (mod 318 L, max heavy sibling 158 L). One sibling per emitted artefact (attn / interleaved + feature-major / per-layer MoE / norms / PLE / lm_head). mod.rs keeps the orchestrator, `Q4kWriteOptions`, `QuantBlockFormat`, `pad_rows_to_block` / `pad_to_block`, `resolve_v_tensor`, the small `update_index_json` step, and helper unit tests. The orchestrator threads a running `Vec<WeightEntry>` through norms → ple → lm_head, then emits the single `weight_manifest.json` and patches `index.json`. |

### P1: Architecture-independent extraction (closed)

Added `ensure_extract_level_supported(arch, level)` in
`format/weights/capabilities.rs` and wired it into `build_vindex` and
`build_vindex_streaming` entry points. MLA architectures now fail
before any output directory is created when `level.writes_attn()`;
Browse-level extracts of MLA still succeed (no attention written).
5 new capability-gate unit tests + 2 integration tests asserting
`read_dir(output_dir).is_empty()` after a rejected Inference-level MLA
extract. Strengthened
`unknown_family_does_not_inherit_known_bands_by_string_prefix` to
prove `gemma3-clone` / `llamafied` lookalikes get the structural
fallback, not the canonical bands.

### Per-layer FFN Phase 2 — cache verified, regression flagged

Audit found `MetalBackend::moe_scratch` Mutex + `AppState::moe_scratches`
HashMap-by-shape both in code (line refs in
`docs/per-layer-ffn-phase2-research.md`). Cold/warm bench on Gemma 4 26B
A4B confirms the cache amortises the ~120 ms allocation cost: cold
265 ms, warm 118 ms, first-token overhead 146.7 ms (cold/warm = 2.24×).
Phase 2 closed as Done. The warm-token tok/s (8 on `bench_generate`,
11.2 on `larql bench --warmup 3`) is below the 2026-05-02 19.4 tok/s
baseline; both dense (4B 88.1 → 30.4) and MoE (26B 19.4 → 11.2)
regressed at the same time on a heavily-loaded machine, pointing at
thermal throttling (precedent in 2026-04-28 `Q4_K f16 accumulator`
artifact). Cool-machine rerun captured as a non-blocking checklist
item in `ROADMAP.md` follow-ups.

### Coverage push (per-file ≥90% bar)

| Round | Item |
|------|---------|
| Pure-function coverage on publish trio | Extracted `parse_lfs_batch_response`, `parse_preupload_response`, `parse_lfs_oid_index` from their HTTP-bound parents into pure helpers. Added 21 unit tests (9 lfs.rs, 7 upload.rs, 5 remote.rs) covering the JSON contract: malformed bodies, missing/empty arrays, per-object errors, optional-action absence, default fallbacks, malformed entries skipped. Lifted `lfs.rs` 0%→40.8%, `remote.rs` 0%→59.0%, `upload.rs` 0%→34.0%. |
| HTTP-mock harness for publish trio (task #12) | Added `mockito = "1.7"` + `serial_test = "3.2"` as dev-deps. Introduced `protocol::hf_base()` consulting `LARQL_HF_TEST_BASE` env var so per-test mockito servers can intercept HF traffic. 32 new HTTP-mocked tests across the trio. Per-file coverage now: lfs.rs 97.8%, remote.rs 98.3%, upload.rs 96.1%. |
| HTTP mocks for discovery + download (task #13) | Widened `protocol::hf_base()` to `pub(in crate::format::huggingface)` so siblings outside `publish/` can route through the same env-var override. 20 tests in discovery.rs (15 HTTP-mocked) and 19 tests in download.rs (4 strip_etag_quoting, 3 want_model_file, 5 hf_cache_repo_dir env-var fallback, 7 head_etag_and_size HTTP-mocked). Coverage: discovery.rs 5.9%→94.6%, download.rs 1.9%→56.6%. The `hf_hub`-bound paths (`resolve_hf_vindex`, `download_hf_weights`, `resolve_hf_vindex_with_progress`, `resolve_hf_model_with_progress`) cap download.rs at ~57% — `hf_hub` has its own internal HTTP client that mockito can't intercept. |
| Per-file ≥90% coverage push (rounds 5-7) | Added inline unit tests to lift every newly-split file in `index/core/` and `index/types/` to ≥90% line coverage. `core/{fp4_ffn,gate_lookup,native_ffn,patch_overrides}.rs` 100%; `core/quantized_ffn.rs` 98.82%; `core/mod.rs` 96.92%; `types/{fp4_ffn,gate_lookup,native_ffn,patch_overrides,quantized_ffn}.rs` 100%; `types/mod.rs` 97.81%; `types/ffn_row.rs` 93.50% (heavy unified-dispatch trait — covered with a configurable `Stub` lighting up FP4 / native / Q4_K backends in isolation). Test fixtures: a hand-built `DownMetaMmap` over a tempfile for the binary record decode, a stub for the row dispatch chain. |
| Aggregate coverage | Lifted from **71.56%** lines (2026-05-08 baseline) to **81.13%** lines by end of round-5. Policy passes. Files still below 90% are integration-driven (`extract/streaming/stages/{gate_vectors,down_meta,router_weights,index_json}.rs`, `format/weights/load/{f32,q4k}.rs`, `format/weights/write_q4k/{moe_layers,norms}.rs`); lifting them needs the synthetic-safetensors fixture builder noted in ROADMAP follow-ups. |
| Coverage policy refreshed | Removed 3 stale entries (`extract/build.rs`, `extract/streaming.rs`, `format/huggingface/publish.rs` — all deleted today). Added 9 new debt baselines for the split siblings. Ratcheted 3 entries that drifted post-refactor. |
| Coverage regression flagged (task #16) | `index/storage/ffn_store/down.rs` and `up.rs` dropped (-9.7 / -15.1) without code changes. Same number of uncovered lines in each (~8) suggests a shared test that stopped exercising them. Files unchanged since `d3a8bc6`. Baselines lowered to current floor (down 31, up 39); investigation tracked as task #16. |

### Misc

- Warning cleanup: deleted unused `RepoKind::hf_repo_type` in `format/huggingface/download.rs`. `cargo clippy -p larql-vindex --all-targets` now reports zero warnings (modulo parallel-session walker work).

## [2026-05-08] — Quality gate + coverage ratchet

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
| Documentation | Updated root README, crate README, and `ROADMAP.md` with the new Makefile targets, CI surface, coverage policy, and current test inventory. |

## [2026-05-01] — Round-4 cleanup (magic strings, magic numbers, modularity)

Closes the M1-M9 audit landed earlier in the day. Same cadence as
round-1/2/3. **493 tests passing**, **0 new clippy warnings**, **fmt
clean**.

| Item | Outcome |
|------|---------|
| **M1**. `up_weights.bin` / `down_weights.bin` literals | Added `UP_WEIGHTS_BIN` / `DOWN_WEIGHTS_BIN` constants, routed 17+ literal sites in `quant/convert_q4k.rs`, `format/checksums.rs`, `format/weights/write_f32.rs`, `format/huggingface/mod.rs`, `extract/build/mod.rs` tests, `HF_UPLOAD_FILES` + uniqueness test extended |
| **M2**. `"Q4_K"` / `"Q6_K"` tag literals | ❌ Withdrawn — re-review found all 6 `attn.rs` sites are inside `#[cfg(test)]` exercising the on-disk wire contract; routing through `format_tag()` would weaken the tests (rename would no longer be caught). Literals correctly localised |
| **M3**. Default `c_score` / confidence fallback | `DEFAULT_C_SCORE = 0.9` lifted to `index::types`; routed `vindexfile/mod.rs:122` and `patch/overlay_apply.rs:73`. Test-fixture sites kept literal |
| **M4**. K-quant block size 256 hardcoded | Routed `quant/registry.rs` + `config/quantization.rs` through `larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS`; renamed `pad_to_256` / `pad_rows_to_256` → `pad_to_block` / `pad_rows_to_block` |
| **M5**. `148`-byte legacy Q4_K stride anonymous | `LEGACY_BLOCK_Q4_K_STRIDE` constant added next to `QUANT_FORMATS`; `registry.rs` and `attn.rs` rejection tests now reference it instead of `* 148` |
| **M6**. `gate_knn.rs` 962 non-test lines, ~25 methods | Split into `gate_knn/{mod,dispatch,scores_batch,hnsw_lifecycle}.rs` (4 files, largest 380). `top_k_by_abs` free fn + `top_k_from_scores` impl shim live in `mod.rs` so all submodules share them |
| **M7**. `ffn_store/mod.rs` 740 non-test lines | Split into `ffn_store/{mod,down,up,interleaved,interleaved_q4,interleaved_q4k,gate_q4}.rs` (existing `fp4.rs` + `q4k_cache.rs` siblings preserved). `mod.rs` keeps `FfnStore` struct, `DownFeaturesQ4kEntry`, `Clone`/`empty` impls, and the `ffn_layer_byte_offset` shared helper. Largest sibling 248 |
| **M8**. `extract/build.rs` 1115 → 4 files | Split into `build/{mod,down_meta,index_json,resume}.rs`. `BuildContext` + small stages + `build_vindex` + tests stay in `mod.rs`; the 3 large concerns moved to siblings. *(Note: see 2026-05-09 entry — the M8 split was reverted by `505434d` and re-landed in round-5.)* |
| **M9**. `lm_head.rs` 1003 → 3 files | Split into `lm_head/{mod,loaders,knn}.rs`. `top_k_sorted` made `pub(super)`. Constants + `read_lm_head_manifest_kind` helper + tests stay in `mod.rs` |

Aggregate file-size impact: 4 monolith files totalling 4,075 lines →
20 sibling files, no non-test file over 600 lines.

## [2026-04-25] — Round-3 polish

| Item | Outcome |
|------|---------|
| Split `config/types.rs` (628 L) | → `config/{index,quantization,model,compliance}.rs` + back-compat `types` alias module |
| HuggingFace resolution in Vindexfile | `FROM hf://...` directives now resolve via `format::huggingface::resolve_hf_vindex` |
| Streaming extract phase checkpoints | `extract::checkpoint::Checkpoint` written to `.extract_checkpoint.json` after each phase; cleared on full success; 6 unit tests |
| Auto-resume from checkpoint | `gate_layer_infos` persisted in checkpoint; on resume the gate + down_meta phases are skipped and existing files reused; incompatible checkpoints discarded with warning |
| `extract::stage_labels` constants module | 15 callback labels (8 stages + 6 components + relation_clusters) extracted from 65+ literal sites — typo'd `on_stage_done("gate_vectro")` is now a compile error |
| GGUF Q4_K format check | No-op — 144-byte GGUF-canonical layout was already in use everywhere; only fixed a stale 148-byte comment in `larql-compute/src/pipeline.rs` |
| **Perf W1** — `top_k_from_scores` → bounded min-heap | 5.4 MB → 16 KB allocation per walk on Gemma 4B shape; **-18% gate_knn @ 4096×512**, **-62% walk @ 14L×4096×512**; flat at 10240×2560 (BLAS dominates). `top_k_by_abs` free fn at `gate_knn.rs`. |
| **Perf W2** — Feature-major Q4_K down | First-access down decode at Gemma 4B Q4_K dims: **2440× at K=100**, **251× at K=1024**, **25× at full K**. Eliminates the ~840 MB heap cache ceiling on CPU sparse walk. `down_features_q4k.bin` + manifest emitted at extract time when `Q4kWriteOptions::feature_major_down=true` (CLI flag `--feature-major-down`). |
| **Perf W3** — Parallelize HNSW warmup | 8-layer dense HNSW warmup **3.6×** (395 → 109 ms); 4-layer MoE warmup **2.8×** (785 → 276 ms). Estimated 34-layer Gemma 4B warmup goes from ~2.6 s serial to ~700 ms. Added `warmup_hnsw_all_layers()` API: parallel-builds across layers via rayon. |
| **Parallelize gate KNN for batch inference** | -7% at seq_len 64, **-24% at seq_len 256** on Gemma-shape gates (10240×2560). Below seq_len 16 the rayon overhead cancels the savings, so the parallel branch is gated on `PARALLEL_TOPK_THRESHOLD = 16`. `gate_knn_batch` now `par_iter`s the per-position top-K extraction when `seq_len >= 16`. |

## [2026-04-25] — Second audit + round-2 cleanup

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

## [2026-04-25] — First audit + round-1 cleanup

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

## [2026-04-25] — Perf audit fixes

| Item | Outcome |
|------|---------|
| Bound the Q4_K dequant cache (LRU) | `set_q4k_ffn_cache_max_layers` + `--max-q4k-cache-layers N` flag on `larql serve` |
| Q4_K interleaved madvise + per-layer prefetch | `prefetch_interleaved_q4k_layer` mirrors the Q4_0 path; wired into `walk_ffn/sparse.rs` |
| HNSW on the decode hot path | Zero-copy view for f32-mmap layers (was cloning ~100 MB / query); abs-magnitude ranking parity (oversample 4× + re-rank); `--hnsw` + `--hnsw-ef-search` flags |
| Bench rig hygiene | Refuses if `larql-(server\|router)` daemons are alive; `LARQL_BENCH_ALLOW_DAEMONS=1` override; `make bench-vindex` vs `bench-vindex-scaling` split |
| `save_gate_vectors` regression check | False alarm — criterion p=0.21, no statistically detectable change |

## [2026-04-07] — First iteration

| Item | Outcome |
|------|---------|
| Q4_K FFN loader + wiring | `interleaved_q4k.bin` end-to-end; inference `predict_honest` prefers Q4_K over Q4_0 |
| Quantizer single source of truth | Builder uses `larql-compute` (ADR-008) |
| Example cleanup (13 → 11) | Removed Q4_0 attn + Q4_0 interleaved |
| 8 ADRs documented | All major decisions recorded |
| PERFORMANCE.md + format alignment | Fresh benchmarks, verified pipeline |
| Safety doc for `mmap_optimized` | Clippy compliance |
| `VindexPatch::is_empty()` | API completeness |

## [2026-03 / 2026-04] — Foundation

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
