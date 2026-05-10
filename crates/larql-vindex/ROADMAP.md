# Roadmap — larql-vindex

## Current state (as of 2026-05-10)

- **857 lib tests** on `larql-vindex` (`cargo test -p larql-vindex
  --lib`). Crate-local checks are wired through `make larql-vindex-ci`:
  fmt, clippy `-D warnings`, tests, example compile checks, bench
  compile/tests, and coverage policy. All four self-contained examples
  (`mmap_demo`, `demo_memit_solve`, `q4k_demo`, `walker_demo`) run
  end-to-end; the rest compile clean via `cargo check --examples`.
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
  - `format/weights/write_q4k/{mod,attn,ffn,moe_layers,norms,ple,lm_head,feature_major_down}.rs`
    (round-5, 2026-05-09 — one emitted artefact per sibling)
  - `format/{weights,filenames,fp4_codec,…}/`
  - `engine/` (was `storage/`) — StorageEngine + epoch + MEMIT
  - `config/{index,quantization,model,compliance,dtype}.rs` — was the
    624-line `types.rs` monolith
  - Large-file debt **closed 2026-05-10**: every non-test file in the
    crate is under the soft 600-LOC threshold. The four largest
    are now `format/weights/write_q4k/mod.rs` (318 L after split),
    `index/storage/lm_head/knn.rs` (~525), `index/storage/attn.rs`,
    and `quant/convert_q4k.rs` — all comfortably below the threshold.
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
  2026-05-08 baseline of 71.56%; current measured **88.90% lines**
  as of 2026-05-10 round-6. Per-source-file default is 90%; files
  below that are explicit debt baselines (40 entries) and should
  only move upward. **85 of 125 files at the 90% default**, up from
  41 on 2026-05-08. Today's round-6 push lifted 13 files past the
  default — see `CHANGELOG.md` for the per-file deltas. Aggregate
  gap to 90%: **1.1 points**.
- **Cross-platform CI**: `.github/workflows/larql-vindex.yml` runs
  format, check, examples, clippy, tests, and bench compile/tests on
  Linux, Windows, and macOS. Coverage policy runs on Ubuntu.
- Bench rig daemon-aware (`make bench-vindex-scaling` refuses if
  `larql-server` / `larql-router` are running on the host).

---

## P0: Active

### Review follow-ups from 2026-05-09

- [ ] Harden attention weight manifest accessors in
  `index/storage/attn.rs`. `attn_q8_layer_data`,
  `attn_q4k_layer_data`, and `attn_q4_layer_slices` should validate
  `offset + length <= mmap.len()` with checked arithmetic before slicing,
  matching the defensive behavior already used by FFN Q4K accessors. A stale
  or corrupt attention manifest should return `None` or a typed load error,
  not panic during query/decode.
- [ ] Replace `walker/utils.rs::current_date()` with a real calendar-date
  implementation or remove the date from walker metadata. The current helper
  approximates years as 365 days and months as 30 days, so extraction metadata
  can be wrong even though the format looks valid.

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

- [x] Large-file decomposition — **closed 2026-05-09**. Last file
  (`format/weights/write_q4k/mod.rs` 734 L) split into one sibling per
  emitted artefact (attn / ffn / moe_layers / norms / ple / lm_head),
  orchestrator down to 318 L. Closed today: `publish.rs` (997),
  `streaming.rs` (832), `load.rs` (817), `core.rs` (755), `types.rs`
  (715), `streaming/stages.rs` (644), `write_q4k/mod.rs` (734). No
  non-test file in vindex now exceeds the soft 600-LOC threshold.

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

## History

Completed entries previously kept here have been moved to
[`CHANGELOG.md`](CHANGELOG.md), reverse-chronological by date. Active
P0/P1/P2 items above; once a row lands it migrates to the changelog.
