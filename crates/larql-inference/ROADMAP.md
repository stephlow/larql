# Roadmap — larql-inference

## Current: ~95 tok/s (Metal Q4K) | Ollama: ~101 tok/s | 4 KV engines

## Status

The four KV-cache engines shipped in `engines/kv_engines/` all reach ~93-95 tok/s
on Gemma 3 4B using the Metal Q4K path (matching Ollama within 6%). See bench:

```
larql bench gemma3-4b-q4k --engine markov-rs,unlimited-context,turbo-quant,apollo
```

---

## P0: Generation quality (blocks demo)

### Chat template — inference side
**Status**: Not started  
**Files**: `layer_graph/generate/gpu.rs`, `layer_graph/generate/cpu.rs`  
Read `tokenizer_config.json` from the vindex, parse the `chat_template` Jinja
field with `minijinja` (already in `Cargo.toml`), apply to the token sequence
before generation. `--no-chat-template` flag to bypass for base models or raw
prompts.

### EOS detection
**Status**: Partial — checks `<eos>`, `</s>`, `<|endoftext|>` but missing Gemma 4 `<end_of_turn>`  
**Files**: `layer_graph/generate/gpu.rs`  
Read `eos_token_id` and `stop_strings` from `generation_config.json`. Gemma 4
lists `<end_of_turn>` in `stop_strings` but not in `eos_token_id`; without this
fix greedy decode runs to `--max-tokens`.

### Token spacing / detokenisation
**Status**: Not started  
Accumulate tokens before decoding; trim only the first token. HuggingFace
tokenizers use a leading-space convention (`▁Paris`) that is stripped incorrectly
when decoding single tokens.

### Token streaming
**Status**: Not started  
Change `generate` / `generate_cached` to accept `on_token: impl FnMut(&str, f64)`
callback. Currently the full token list is collected before returning.

### Sampling
**Status**: Not started  
Add temperature softmax, top-k, and top-p (nucleus) filtering after lm_head and
before argmax. Flags (`--temperature`, `--top-p`, `--top-k`) owned by `larql-cli`.

### Multi-turn KV state
**Status**: Not started — `larql chat` resets KV cache per turn today  
Maintain a running `token_ids` buffer across turns. `--max-context N` eviction:
drop oldest turns when the buffer exceeds `N`.

### Gemma 3 4B regression smoke test
**Status**: Not started  
Load `gemma3-4b-q4k-streaming`, run one-token generation, assert first token is
`"Paris"`. Gate on `CI_INTEGRATION=1`.

---

## P0: MoE inference completions

### MoE-aware CPU forward pass
**Status**: Not started  
`predict_q4k` / `WeightFfn::forward` has no MoE branch. Wire `cpu_moe_forward`
(already in `larql-compute/src/cpu/ops/moe.rs`) into `forward/layer.rs`.

### Wire `RouterIndex` client-side
**Status**: Not started  
`larql-vindex/src/index/router.rs` exists but is not connected to the forward
pass. Connect so MoE router runs locally against the vindex before dispatching.

---

## P0: Engine performance parity

### TurboQuant Metal K/V checkpoint compression
**Impact**: Reduces boundary checkpoint from 278 KB → 36 KB/window (7.7×) for long contexts.  
**Status**: TurboQuant runs at Metal speed. Compressed boundary checkpoints require
Metal K/V read-back. Add `backend.get_kv_last_position(layer)` to the Metal backend.

### Apollo `prefill_to_layer` — true layer-skip
**Impact**: ~20% faster per step in compressed path.  
**Status**: `forward_from_layer` ships; K/V seeding at `crystal_layer` is a follow-up.

### Apollo store builder
**Impact**: Currently requires pre-built NPY/NPZ files.  
**Status**: Not started. `ApolloEngine::build_from_document(weights, tokenizer, tokens)`.

---

## P1: Architecture coverage

### Wire v_shares_k into forward pass
**Effort**: Low — `v_shares_k()` already in larql-models; swap runtime check.

### Validate PLE end-to-end (Gemma 4 E2B)
**Effort**: Medium — config parsed; forward pass not yet wired.

### KV layer sharing for Gemma 4
**Effort**: Medium — `kv_shared_source_layer()` returns correct sources; cache allocation not yet sharing.

### Llama 3 / Gemma 4 engine validation
All four engines validated on Gemma 3 4B. Need empirical `cos h = 1.000000` validation on Llama 3 / Gemma 4.

### MarkovRS batched K/V recompute kernel
**Impact**: Eliminate 2000× FLOP overhead on CPU decode path.  
**Effort**: Medium (new Metal shader for `[W, hidden] @ [hidden, kv_dim]` Q4K projection).

---

## P1: Structure & file layout

From 2026-04-26 code review. All public APIs preserved; changes are internal re-organisation.

### High priority

**`ffn/remote.rs` (893 LOC) — split into `remote/`** ✅ Done 2026-04-26  
`ffn/remote/codec.rs` — binary codec, wire types, latency stats, codec tests.  
`ffn/remote/http.rs` — RemoteFfnConfig, RemoteWalkBackend, RemoteFfnError, HTTP tests.  
`ffn/remote/mod.rs` — thin re-export + protocol doc.  
No magic strings: `BINARY_CT`, `BATCH_MARKER`, `STATS_PATH`, `WALK_FFN_PATH` are named constants.

**`turbo_quant/mod.rs` → `turbo_quant/engine.rs`** ✅ Done 2026-04-26  
TurboQuantEngine + TurboQuant codec moved to `engine.rs`. `mod.rs` is a thin re-export of sub-modules + `pub use engine::{TurboQuantEngine, TurboQuant}`.

**`vindex/walk_ffn/mod.rs` → `walk_ffn/engine.rs`**  
Deferred: walk path submodules use `pub(super) impl WalkFfn` blocks that are
architecturally tied to `mod.rs` as the parent. Requires changing visibility to
`pub(in crate::vindex::walk_ffn)` across 6 files — low risk/reward compared to
other P1 items. Backlog.

**`layer_graph/predict.rs` (700 LOC) — split**  
Five `predict_*` variant functions sharing a shell. Extract to `predict/base.rs`
(shared embed→loop→logits shell) + `predict/variants.rs` (per-strategy overloads).

**`residual.rs` at crate root → `forward/norm.rs`**  
It's a collection of norm primitives used exclusively by the forward pass. Moving
it co-locates it with the other forward utilities (`ops.rs`, `layer.rs`).

**`capture.rs` at crate root → `trace/`**  
`InferenceModel` / `CaptureConfig` belong with the trace infrastructure.

### Medium priority

**Softmax in 5 locations — unify**  
`trace/vocab.rs`, `engines/accuracy.rs`, `ffn/moe_remote.rs`,
`layer_graph/logits.rs`, `forward/target_delta.rs` each have a private softmax.
Promote `engines/accuracy.rs::softmax` to `forward/ops.rs` (or `residual.rs`);
have the others `use crate::forward::softmax`.

**`embed_tokens_pub` / `run_attention_public` naming**  
The `_pub` suffix is redundant on public functions. Rename to `embed_tokens` and
`run_attention` or document why the suffix exists. `_pub` vs `_public` is also
inconsistent.

**`ApolloEngine` and `TurboQuantEngine` not re-exported at crate root**  
`MarkovResidualEngine` and `UnlimitedContextEngine` are re-exported; the other
two engines are not. Either export all four or none.

**`walker/` and `experts/` have no module-level docs**  
Add `//!` headers explaining purpose and entry points.

**`vindex/` module doc is vague**  
"Vindex integration" says nothing to a new reader. Expand to explain what the
vindex is and what this module provides.

### Low priority

**`forward` re-export block is 70+ items with no sub-grouping**  
Split into clearly commented groups: prediction, tracing, raw logits, analysis
(memit, target_delta, infer_patched).

**`trace as trace_decomposed` alias in `lib.rs`**  
Aliases a naming problem rather than fixing it. Rename the function itself.

**`RawForward` is an implementation detail in the public API**  
Users never construct `RawForward` directly; it's only returned by
`forward_raw_logits`. Consider whether it needs to be pub.

**`generate_cached*` in `forward/` vs `generate` in `layer_graph/`**  
Two generation APIs with similar names but different semantics (CPU KV-cache step
vs Metal fused pipeline). Add a clear doc comment on each explaining the difference.

---

## P1: Quality bugs (from 2026-04-26 review)

### `grid.rs` — hardcoded `eos_id = 1` is a real bug ✅ Fixed 2026-04-26
**File**: `layer_graph/grid.rs`  
Replaced `eos_id: u32 = 1` with `is_end_of_turn(tok_str.trim())` on both the prefill-exit
and decode-loop paths, matching all other generation code.

### Softmax duplicated in 5 locations ✅ Fixed 2026-04-26 (2 of 5)
**Files**: `trace/vocab.rs`, `engines/accuracy.rs` now use `pub use crate::forward::softmax`.  
Canonical implementation lives in `forward/ops.rs`, exported via `forward/mod.rs`.  
`ffn/moe_remote.rs` (in-place `&mut [f32]`), `logits.rs` (single-prob extractor),
`target_delta.rs` (Array1) remain local — different enough to not unify.

### `forward/ple.rs` hardcodes `1e-6` norm epsilon ✅ Fixed 2026-04-26
`1e-6` replaced with `arch.norm_eps()` for consistency.

### `grid.rs` undocumented `SKIP_MOE` env var ✅ Fixed 2026-04-26
Added `# Diagnostics` section to module doc.

---

## P1: Test coverage gaps

From 2026-04-26 coverage review (50.45% line coverage).

### Critical

**`markov_residual/` — zero tests across all 5 new files** ✅ Done 2026-04-26  
`store.rs`: clip_layer edge cases (no-window noop, at-limit, over-limit), memory_bytes, window_tokens.  
`engine.rs`: name, memory lifecycle, prefill→decode cycle, window clipping, multi-step shapes.  
`compute.rs`: recompute_kv shape/finiteness/RoPE shift, rs_prefill result shape + window, rs_decode_step position advance.

**`ffn/sparse_compute.rs` and `ffn/sparse.rs` — zero tests** ✅ Done 2026-04-26  
`sparse_compute.rs`: empty-features→zeros, single/multi-token shape, top-K ordering, dense-fallback equivalence, down-override effect.  
`sparse.rs`: name, all-layers shape/finiteness, top-k vs dense match, with_activation shapes.

**`ffn/graph_backend.rs` — zero tests** ✅ Done 2026-04-26  
Construction (layer count, empty layers), lookup_from_tokens (top-K limit, unknown layer, empty scores, out-of-range tokens), precompute_entity, save/load roundtrip.

**`layer_graph/` — 7 of 17 files untested** ✅ All 7 done 2026-04-26  
`dense.rs` — DenseLayerGraph shape/finiteness/capture, PerLayerGraph bounds.  
`walk.rs` — WalkLayerGraph all-layers, PipelinedLayerGraph in/out-of-range.  
`mod.rs` — trait dispatch, name distinctness.  
`prefill.rs` — CPU path: shape, finiteness, partial range, empty range, logit correctness.  
`template.rs` — detect_template (7 pure tests), TemplateUniverse build/get/total, GuidedWalkLayerGraph shape/finiteness.  
`pipeline_layer.rs` — build_arch_params param extraction, resolve_attn_weights None path, resolve_ffn_weights legacy stride slicing.  
`grid.rs` — error path: no Q4K mmap → `Err(BadResponse)`.  
Integration tests: `tests/test_layer_graph_integration.rs` — real vindex tests for prefill_with_kv, build_pipeline_layers, TemplateUniverse, GuidedWalkLayerGraph (all `#[ignore]`, run with `--ignored`).

### High priority

**`forward/ops.rs` — zero tests** ✅ Done 2026-04-26  
`dot_proj`: shape, identity-weight, value-correctness.  
`add_bias`: all-rows updated, shorter-bias safe, zero-bias noop.  
`apply_norm`: shape, finite output, offset produces different result.

**`forward/ple.rs` — zero tests** ✅ Done 2026-04-26  
precompute returns empty for non-PLE arch, apply_ple None/missing-weight guard paths,
output shape. Softmax tests moved here as a side-effect of unification.

**`engines/kv_engines/unlimited_context/extend.rs` — zero tests** ✅ Done 2026-04-26  
empty_prior shape, empty-tokens/wrong-prior-len → None, single/multi-token extend, kv_cache
row count, checkpoint = last-row, abs_start shifts RoPE, finite logits, chained extends.

### Medium priority

**GQA head grouping (`reps` parameter) not tested** ✅ Done 2026-04-26  
Three tests: output shape (4Q/2KV/reps=2), finiteness, and head-pair sharing — heads 0 & 1
sharing KV-head 0 produce identical output rows.

**RoPE missing property tests** ✅ Done 2026-04-26  
rope_base sensitivity, fraction=1.0 equals full-rope, offset=N matches sequential position N,
partial fractions 0.25/0.5/0.75 all finite.

**No synthetic end-to-end tests for `generate()`**  
`generate()` (Metal GPU path) is only tested with `#[ignore]` real-model tests.
Add a synthetic CPU-backend integration test using `make_test_weights()`.

---

## P2: Research

### Hybrid head caching (RS+CA)
95.5% of attention heads are static (cacheable). Would give ~180-370× compression
at 370K tokens — between TurboQuant (4×) and MarkovRS (287×) with near-exact accuracy.

### Graph Walk engine
FFN graph walk is proven (348K features, 34 layers, zero accuracy loss).
Full RS Graph Walk requires cracked attention (static head caching).
`GraphWalkEngine` would eliminate the forward pass entirely for parametric queries.

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
| `generate/` split (cpu/gpu/lm_head/types) | 2026-04-26 | Structured generation directory |
| `markov_residual/` split (store/engine/compute/q4k) | 2026-04-26 | Structured engine directory |
| `forward/predict/` split (types/raw/dense/ffn) | 2026-04-26 | Forward predict directory |
| `forward/ops.rs` extracted | 2026-04-26 | Shared math primitives |
| `graph_ffn.rs` → `ffn/graph_backend.rs` | 2026-04-26 | Correct placement in ffn/ |
| 400+ unit tests | 2026-04-26 | Synthetic weights, no disk I/O |
| 49% line coverage (llvm-cov) | 2026-04-26 | Baseline measured |
| Code quality review (3-agent) | 2026-04-26 | Unsafe removed, LCG fixed, OnceLock added |
| P1 code quality fixes (magic strings, duplication) | 2026-04-25 | env-var names, GELU constants |
| `ffn/remote.rs` → `remote/codec.rs` + `remote/http.rs` | 2026-04-26 | No magic strings; codec/HTTP separation |
| `turbo_quant/mod.rs` → `engine.rs` | 2026-04-26 | Consistent engine layout; thin mod.rs |
| Tests: `markov_residual/` (store, engine, compute) | 2026-04-26 | 0 → 15 tests; prefill/decode/clip coverage |
| Tests: `ffn/sparse_compute.rs` + `ffn/sparse.rs` | 2026-04-26 | 0 → 14 tests; sparse FFN validated |
| Tests: `ffn/graph_backend.rs` | 2026-04-26 | 0 → 10 tests; GateIndex build/lookup/save |
| Tests: `forward/ops.rs` | 2026-04-26 | 0 → 8 tests; dot_proj/add_bias/apply_norm |
| 457 unit tests total | 2026-04-26 | +~50 tests vs previous session |
| Bug: `eos_id = 1` in grid.rs | 2026-04-26 | Correct EOS on all models, not just Gemma |
| Softmax unified to `forward/ops.rs` | 2026-04-26 | 2 duplicate impls removed |
| `forward/ple.rs` norm_eps fixed | 2026-04-26 | Uses `arch.norm_eps()` not hardcoded 1e-6 |
| Tests: `unlimited_context/extend.rs` | 2026-04-26 | 0 → 8 tests; checkpoint, RoPE, chained extends |
| Tests: `layer_graph/dense.rs` | 2026-04-26 | 0 → 8 tests; shape, capture, PerLayerGraph bounds |
| Tests: `layer_graph/walk.rs` | 2026-04-26 | 0 → 7 tests; Walk + Pipelined layer range |
| Tests: `layer_graph/mod.rs` | 2026-04-26 | 0 → 3 tests; trait dispatch, name distinctness |
| Tests: `forward/ple.rs` | 2026-04-26 | 0 → 6 tests; guard paths + softmax |
| Tests: GQA reps>1 | 2026-04-26 | 3 tests; shape, finiteness, KV-head sharing |
| Tests: RoPE property tests | 2026-04-26 | 4 tests; base sensitivity, offset=position, fractions |
| 499 unit tests total | 2026-04-26 | +42 tests; all passing |
| Tests: `layer_graph/prefill.rs` | 2026-04-26 | 6 tests; CPU path shape/finiteness/logits |
| Tests: `layer_graph/template.rs` | 2026-04-26 | 12 tests; detect_template + TemplateUniverse + GuidedWalk |
| Tests: `layer_graph/pipeline_layer.rs` | 2026-04-26 | 6 tests; arch params, attn weights, FFN stride |
| Tests: `layer_graph/grid.rs` | 2026-04-26 | 1 test; error path for missing Q4K mmap |
| Integration tests: `test_layer_graph_integration.rs` | 2026-04-26 | 7 ignored tests; real vindex prefill/pipeline/template |
| Fix: `residual_diff/capture.rs` missing PathBuf import | 2026-04-26 | Pre-existing bug; broke lib test compilation |
| 525 unit tests total | 2026-04-26 | All passing |
