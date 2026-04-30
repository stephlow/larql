# Roadmap — larql-inference

## Current: ~95 tok/s (Metal Q4K) | Ollama: ~101 tok/s | 4 KV engines

## Status

The four KV-cache engines shipped in `engines/kv_engines/` all reach ~93-95 tok/s
on Gemma 3 4B using the Metal Q4K path (matching Ollama within 6%). See bench:

```
larql bench gemma3-4b-q4k --engine markov-rs,unlimited-context,turbo-quant,apollo
```

---

## P0: Mechanistic hooks (lazarus parity)

Driver: replace chuk-mlx as the engine for `chuk-mcp-lazarus`. Lazarus has 77
inference-time MCP tools (capture, ablate, patch, steer, probe, DLA, KV
surgery). Larql today only writes to weights (MEMIT, KNN, LQL) — it has no
mid-forward inspection/intervention API. The whole tool surface collapses to
one missing primitive: a programmatic forward-hook system.

### M1 — `LayerHook` trait + CPU plumbing (read + write)
**Status**: In progress
**File**: `forward/hooks.rs` (new), `forward/layer.rs`, `forward/trace.rs`

Trait shape:
```rust
pub trait LayerHook {
    fn on_pre_layer(&mut self, layer: usize, h: &Array2<f32>) {}
    fn on_post_attention(&mut self, layer: usize, h: &mut Array2<f32>) {}
    fn on_attention_weights(&mut self, layer: usize, w: &AttentionWeights) {}
    fn on_ffn_activation(&mut self, layer: usize, gate: &Array2<f32>) {}
    fn on_post_layer(&mut self, layer: usize, h: &mut Array2<f32>) {}
}
```

Insertion points in `run_layer_with_capture`: pre-layer (h entering),
post-attention (`h_post_attn`, `&mut`), FFN gate activation (`activation`),
post-attention-weights (`attn_weights`), post-layer (`h_out`, `&mut`).

The `&mut` on post-attention and post-layer is what unlocks the entire
intervention surface — ablation, steering, patching, subspace surgery are all
just `LayerHook` impls.

Plumbing strategy: `run_layer_with_capture` and `trace_forward_full` grow an
optional `&mut dyn LayerHook` parameter. Existing call sites pass `None`
(zero overhead — noop when absent). Hot generation paths in `predict.rs`
remain unchanged for slice 1; M6 wires hooks into the Metal `generate` path.

### M2 — Built-in hooks
**Status**: Not started
**File**: `forward/hooks.rs`

- `NoopHook` — never fires, used by tests.
- `RecordHook { layers: HashSet<usize> }` — captures pre/post-layer residuals
  and FFN activations; replaces the file-output path of `capture_residuals`.
- `ZeroAblateHook { layers, positions }` — zeros residual at requested coords.
- `SteerHook { vectors: HashMap<usize, (Array1<f32>, f32)> }` — adds α·v at
  specified layer's `on_post_layer`.

### M3 — Activation patching
**Status**: Not started — blocked on M1
**File**: `forward/patching.rs` (new)

Two-pass primitive: pass 1 with a `RecordHook` collects the donor residual at
(layer L, pos p) from prompt A; pass 2 runs prompt B with a `PatchHook` that
overwrites the same coords. This is the building block for `full_causal_trace`
(2D position × layer grid) — lazarus's flagship causal tool.

### M4 — Full logit lens
**Status**: Not started
**File**: `forward/predict/dense.rs`

Today: `logit_lens_top1(layer)` returns one token. Add:
- `logit_lens_topk(layer, k) -> Vec<(u32, f32)>`
- `track_token(layer, target_id) -> f32` — log-prob of a specific token at
  a specific layer.
- `track_race(layers, k) -> Vec<Vec<(u32, f32)>>` — top-k per layer in one
  pass for streaming top-k diagrams.

All three project the same captured residual through final norm + lm_head; no
new forward passes.

### M5 — KV cache surgery
**Status**: Not started
**File**: `attention/decode.rs:KvCache`

Lazarus `prefill_inject` and `kv_inject_test` need to lift K/V from one cache
into another. Add `get_layer(layer) -> (&[f32], &[f32])`,
`set_layer(layer, k, v)`, `clone_at_position(other, layer, pos_range)`.

### M6 — Hooks during multi-token generation
**Status**: Shipped
**File**: `forward/kv_generate.rs::generate_cached_hooked`,
`crates/larql-python/src/walk.rs::generate_with_hooks`

Final design: **hooks-on-CPU, Metal-stays-fast**. Lazarus-style mech interp
during multi-token generation goes through `generate_cached_hooked` on the
CPU KV-cache path; the Metal-fast `layer_graph::generate::gpu::generate*`
remains hook-free.

Why not propagate hooks into the Metal path: the Metal `decode_token` and
`prefill_q4` calls are end-to-end fused kernels that handle every layer in
one dispatch. Threading hooks in would require either CPU readback per
layer (kills the fusion benefit) or a parallel kernel surface that splits
on layer boundaries (kills the fast path even when no hook is registered).
Mech-interp tools care about correctness over throughput, so paying the
CPU-path cost when hooks are active is the right trade.

Interface mirrors `trace_forward_full_hooked` — same `LayerHook` trait;
`on_pre_layer`, `on_post_attention(&mut)`, `on_post_layer(&mut)` fire on
every layer of every step (prefill + each decode step).
`on_attention_weights` and `on_ffn_activation` do **not** fire on this
path — the production decode kernels don't capture those intermediates.
Use `trace_forward_full_hooked` for a single forward pass when you need
them.

Tests: `forward::kv_generate::tests` — noop matches baseline; record fires
on prefill + every decode step; α=5 steer changes generated tokens vs
baseline. Demo: `examples/mech_interp_demo.rs` § [7] shows
`baseline_ids = [12, 30, 10, 29]` vs `steered_ids = [4, 4, 4, 4]`.

### M7 — `W_E` / `W_U` + `project_through_unembed`

### M7 — `W_E` / `W_U` + `project_through_unembed`
**Status**: Not started
**File**: `forward/predict/dense.rs`, `lib.rs` re-exports

Lazarus tools `head_dla`, `decode_residual`, `embedding_neighbors` need
direct embedding/unembedding matrix access plus a "project this vector
through `W_U`, return top-k tokens" helper. Today both matrices are wrapped
in `VectorIndex` with no public accessor. Add `weights.embed_matrix()` and
`weights.unembed_matrix()` plus a free function `project_to_vocab_topk(vec, weights, k)`.

### M8 — pyo3 `PyLayerHook`
**Status**: Blocked on M1
**File**: `crates/larql-python/src/hooks.rs` (new)

Wrap a Python callable in a `PyLayerHook(PyObject)` that implements
`LayerHook`. Tensors crossed with `numpy.PyArray2<f32>` (zero-copy on
CPU path). MCP tools in lazarus are then just Python that registers a
hook and calls `infer()`.

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
**Status**: ✅ Done 2026-04-26 — see `layer_graph/generate/eos.rs`  
`EosConfig` reads `eos_token_id` (scalar or array) and `stop_strings` from
`generation_config.json`, layered on top of `BUILTIN_STOP_STRINGS` (covers
Gemma `<end_of_turn>`, ChatML `<|im_end|>`, Llama-3 `<|eot_id|>`/`<|eom_id|>`).
Wired into `generate_with_sampling` via `eos.is_eos(id, &decoded)`. Greedy
`generate` defaults to `EosConfig::builtin()` so existing callers Just Work.

### Token spacing / detokenisation
**Status**: ✅ Done 2026-04-26 — see `layer_graph/generate/detok.rs`  
`Detokenizer` keeps the cumulative ID buffer and emits only the freshly-grown
suffix on each `push`. Equivalent to llama.cpp `llama_token_to_piece` and HF
Python `decode_stream`. Handles HF leading-space (`▁`) for SP tokenizers and
multi-byte UTF-8 chars that straddle a token boundary. Demo at
`examples/detok_demo.rs` shows the bug ("thecapitaloffranceisparis") and the
fix ("the capital of france is paris").

### Token streaming
**Status**: ✅ Done 2026-04-26 — see `layer_graph/generate/gpu.rs`  
`generate_streaming(..., on_token: F)` fires `on_token(id, text, prob)` for
every emitted token, including the first (which comes out of prefill). Uses
`Detokenizer::push` so streamed text preserves HF leading-space spacing.
`generate_with_sampling` is a thin wrapper passing a no-op closure so
non-streaming callers are unaffected. Demo at `examples/streaming_demo.rs`
prints tokens live with stdout flushing.

### Sampling
**Status**: ✅ Done 2026-04-26 — see `layer_graph/generate/sampling.rs`  
`Sampler` + `SamplingConfig` covers greedy / temperature / top-k / top-p with
optional `seed` for reproducibility. Two paths: full-vocab `sample(logits)`
for the OpenAI-API logprob future, sparse `sample_from_topk(hits)` for the
production hot path. Wired into `generate_with_sampling`. Sparse-path
overhead is <2µs/call at top-K=64 (<0.02% of decode budget). CLI flags
(`--temperature`/`--top-p`/`--top-k`) are still owned by `larql-cli`.

### Multi-turn KV state
**Status**: ✅ Done 2026-04-26 (token-buffer) — see `layer_graph/generate/chat_session.rs`  
`ChatSession` owns the running token buffer with whole-turn eviction at
`max_context`. Pluggable `TurnRenderer` covers Gemma / ChatML / Llama-3
templates. The most recent turn is never dropped — eviction is a no-op
when only one turn remains, so a long single prompt is preserved over
silently truncating. `examples/chat_demo.rs` runs a 3-turn conversation.

True KV carryover across turns (so prefill on turn N+1 only processes
the new tokens) is a follow-up — the API surface is in place; it's an
internal optimisation.

### Gemma 3 4B regression smoke test
**Status**: ✅ Done 2026-04-26 — see `tests/test_gemma3_smoke.rs`  
Loads vindex from `LARQL_VINDEX_PATH`, runs single-token greedy generation
on `"The capital of France is"`, asserts first token (trimmed) equals
`"Paris"`. Gated `#[ignore]`; `CI_INTEGRATION=1` flips to fail-loud when
the vindex env isn't set so CI can require the test rather than silently
skip. Defaults configurable via `LARQL_SMOKE_PROMPT` / `LARQL_SMOKE_EXPECTED`.

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

## P0: Evaluation parity (blocks architecture claims)

larql is a research engine for novel architectures (WalkFfn, vindex KV engines, gate
KNN, layer-skip via Apollo). To show an architecture is competitive we need to run
the same eval harnesses other engines run — otherwise we are only ever comparing
synthetic prompts to synthetic prompts. The items below build on the generation-quality
P0 above (sampling, streaming, chat templates, multi-turn KV); without those, none
of the harnesses load at all. Goal is parity for fair evaluation, not feature
parity for its own sake.

### Per-position logprobs / top-k logprobs
**Status**: Not started  
**Files**: `forward/predict/raw.rs`, expose via `lib.rs`  
Add `forward_logprobs(weights, token_ids, target_ids) -> Vec<f32>` returning
per-position log-likelihood of `target_ids[i]` given prefix `token_ids[..i]`.
Also expose top-k logprobs from `forward_raw_logits`. lm-evaluation-harness and
most multiple-choice benchmarks (HellaSwag, ARC, MMLU, WinoGrande, PIQA) score
by sequence log-likelihood, not generation. Without this no likelihood-class
benchmark can run, so no architecture claim has a published comparator.

### OpenAI-compatible HTTP API
**Status**: Not started  
**Files**: `crates/larql-server/src/openai/` (new), thin wrapper over inference  
`larql-server` exposes `/v1/infer` and `/v1/walk`; eval frameworks (lm-eval-harness,
simple-evals, evalplus, AlpacaEval, swe-bench harnesses) plug into
`/v1/chat/completions` and `/v1/completions`. Add OpenAI-shape endpoints as a
wrapper over `generate` + sampling + chat-template rendering + logprob fields.
Unlocks every harness without per-harness adapters.

### Batch inference (independent prompts)
**Status**: Not started  
**Files**: `forward/predict/`, new `predict_batch`  
Distinct from continuous batching. Eval suites issue thousands of independent
prompts; serial execution makes a single benchmark run take hours-to-days. Add
`predict_batch(weights, prompts: &[Vec<u32>]) -> Vec<Vec<f32>>` that prefills each
prompt against the same weight mmap. Each prompt gets its own KV-engine instance,
so all four engines work unchanged.

### LoRA / adapter loading at runtime
**Status**: Not started  
**Files**: `forward/layer.rs`, `larql-models` weight loader  
Many arch papers ship LoRA-tuned variants (instruction-tuned on top of a base).
Without LoRA, larql cannot compare `WalkFfn` on `Gemma-3-4B-base` vs
`Gemma-3-4B-it` without re-quantising a merged model. Add
`WeightSet::with_lora(adapter_path)` wrapping `gate/up/down/q/k/v/o` matmuls as
`W·x + α·B(A·x)`. Stretch: composable adapter stack for ablation
(WalkFfn + LoRA-A vs WalkFfn + LoRA-B on the same base).

### Eval-harness smoke run
**Status**: Not started  
End-to-end test: run lm-eval-harness `hellaswag` (10 samples) against
`larql-server` and assert non-zero accuracy. Gate on `CI_INTEGRATION=1`. This
is what moves "we have logprobs" from a unit test to "harnesses actually plug in."

---

## P1: Eval-class coverage

Each item below unlocks a specific class of evaluation. Land in the order an arch
claim needs them — no need to do all up front. Prerequisite for all of them: the
P0 evaluation-parity stack above.

### Structured output / GBNF grammar / JSON Schema
**Status**: Partial — regex/grammar hook exists in `generate`; not wired to JSON
Schema or BNF.  
**Unlocks**: JSONSchemaBench, BFCL (function-calling leaderboard), any eval
requiring schema-conformant output.  
Apply a constrained-decoding mask over logits before sampling. Minimum viable:
GBNF parser (port from `llama.cpp` grammar.cpp); JSON Schema compiles to GBNF.

### Vision / multimodal forward
**Status**: Not started  
**Unlocks**: MMMU, ChartQA, DocVQA, multimodal subsets of larger suites.
Validates that WalkFfn and the four KV engines work on multimodal weights, not
just text.  
Gemma 3 (4B/12B/27B) and Llama 3.2 ship vision variants; vision-tower weights
are already in safetensors. Add image-embedding pipeline → token-mixing →
existing decoder forward. No new KV-engine work required (image tokens look
like text tokens to the decoder).

### Tool / function calling
**Status**: Not started — depends on chat templates (P0) + structured output
(P1 above).  
**Unlocks**: BFCL, ToolBench, AgentBench, any agent-style eval.  
Once the two prerequisites land this is template glue: parse tool-call markers
in the rendered chat template, emit structured calls via the constrained-decoding
path.

### Speculative decoding
**Status**: Not started  
**Why this matters for arch claims**: any "WalkFfn at X tok/s" comparison
against engines that ship speculative decoding (vLLM, TGI, llama.cpp `--draft`)
is misleading without it. Speculative decoding also interacts non-trivially with
gate KNN — draft and target may diverge on top-k feature selection, which is its
own arch question worth answering.  
**Path**: self-spec via `forward_from_layer` (early-exit verification) is the
cheapest entry; full draft-target spec is a follow-up.

### Trace capture during eval batches
**Status**: Partial — `trace_forward_full` works on single prompts.  
Extend to the batch + logprob path so mechanistic interpretability can use
eval-set inputs without re-running. This is what makes "we ran HellaSwag and
the WalkFfn-replaced layers behaved like X" a single-pass measurement.

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

### Continuous batching + paged attention (deferred)
**Why deferred**: arch claims larql cares about are likelihood-bounded, not
throughput-bounded. PagedAttention-style KV management interacts with all four
KV engines (each has its own checkpoint geometry), and the design work isn't
worth it until a specific eval forces it. Revisit if a throughput-class
benchmark becomes load-bearing for an arch claim.

### Multi-GPU / tensor-parallel (deferred)
`larql-grid` already shards layers across hosts. Tensor-parallel within a layer
is a separate problem and not on the critical path until 70B+ models become the
bottleneck.

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
| `generate/eos.rs` — `EosConfig` | 2026-04-26 | Built-in stops + `generation_config.json`; fixes Gemma 4 `<end_of_turn>` bug |
| `generate/detok.rs` — `Detokenizer` | 2026-04-26 | Cumulative-decode delta; preserves HF `▁` leading-space across SP and BPE |
| `generate/sampling.rs` — `Sampler` + `SamplingConfig` | 2026-04-26 | Greedy / temp / top-k / top-p + seed; <2µs/call sparse path |
| `generate_with_sampling` wired into GPU path | 2026-04-26 | Greedy `generate` is a thin wrapper; backward compatible |
| Examples: `sampling_demo`, `eos_demo`, `detok_demo` | 2026-04-26 | End-to-end demos; detok runs without a model |
| `bench_sampling` benchmark | 2026-04-26 | Per-call cost across 4 configs × 3 vocab sizes; results in PERFORMANCE.md |
| 35 sampling/eos/detok tests | 2026-04-26 | All passing; 613 lib tests total |
| `generate_streaming(... on_token)` callback | 2026-04-26 | Per-token streaming; `generate_with_sampling` is thin no-op wrapper |
| `chat_session.rs` — `ChatSession` + `TurnRenderer` | 2026-04-26 | Multi-turn buffer with whole-turn eviction; Gemma/ChatML/Llama-3 renderers |
| Examples: `streaming_demo`, `chat_demo` | 2026-04-26 | Live token streaming + 3-turn chat over `ChatSession` |
| Smoke test: `test_gemma3_smoke.rs` | 2026-04-26 | One-token greedy regression; CI_INTEGRATION fail-loud mode |
| 13 ChatSession tests + streaming integration | 2026-04-26 | All passing; 626 lib tests total |
| Q4_K stride validation in `load_attn_q4k` | 2026-04-27 | Catches stale 148-byte vindexes; clear "rebuild" error vs silent NaN |
| `QuantFormatInfo::expected_bytes(&shape)` helper | 2026-04-27 | Single source of truth for stride math; used by loader validation |
| 11 stride-validation tests (registry + loader) | 2026-04-27 | 144 vs 148-byte stride; arbitrary lengths; Q4_K & Q6_K shapes |
| Q4_K vs Q4_KF kernel routing fix in `quant_matvec::encode` | 2026-04-27 | Q4_K weights now dispatch the Q4_K kernel; `FusedQkvKernel` enum carries TG geometry |
| `vindex::open_inference_vindex` strict loader | 2026-04-27 | Single entry point; propagates stride errors instead of silently degrading |
| Demos switched to `open_inference_vindex` | 2026-04-27 | sampling/streaming/eos/chat now error loudly with rebuild guidance on stale vindexes |

### 2026-04-30 — gRPC grid accuracy + dense Metal chat template + Gemma 4 model coverage

End-to-end accuracy work across Gemma 4's three production variants (26B-A4B
MoE via gRPC grid, 31B dense via Metal, E2B with PLE). Started from the gRPC
grid producing semantically wrong text ("not specified in the text") and
ended with all four Gemma 4 vindexes producing correct answers. Per-layer
CPU vs Metal residual parity (cos ≥ 0.9999 across all 60 layers of the 31B)
confirmed the inference math itself was always correct — every remaining
gap was somewhere in the wrapping, sampling, or routing logic.

| What | Date | Notes |
|------|------|-------|
| `grid.rs` uses `Detokenizer` + `EosConfig::from_vindex_dir` | 2026-04-30 | Was per-token decode losing SP `▁` leading-space + falling back to `<{id}>` for special tokens; output looked like "Thecapital of France is**not specified...**" |
| Special-token suppression in grid `pick_next_filtered` | 2026-04-30 | Built from `tokenizer.get_added_tokens_decoder()` + structural-marker scan (`<unused…>`, HTML tags, `[multimodal]`). Top-K=256 fallback finds a real word when many candidates are markers. Q4_K quantisation noise was lifting `<mask>` (id 4) over the intended next word at the first answer position |
| `chat::render_user_prompt` shared helper | 2026-04-30 | Centralises `LARQL_RAW_PROMPT` / `LARQL_THINKING` / `LARQL_SYSTEM` / `LARQL_NO_DEFAULT_SYSTEM` + auto Gemma 4 default system prompt. Used by both `run_with_moe_shards` (gRPC) and `walk_cmd::run_predict_q4k` (dense Metal) |
| Built-in Gemma 4 fallback chat template | 2026-04-30 | Vindexes extracted before `chat_template.jinja` was snapshotted (early 31B and E2B) silently sent raw prompts and looped "The answer is:". `family_default_template("gemma4")` plugs the gap |
| Dense Metal path now applies chat templates | 2026-04-30 | `walk_cmd::run_predict_q4k` was sending the raw user string to `encode_prompt`; the chat-template machinery only ran for gRPC. Both paths now go through `render_user_prompt` |
| `lm_head_topk` falls back to backend GEMV when KNN is all-zero | 2026-04-30 | At the prefill→decode boundary the Metal `q4k_matvec` for lm_head occasionally returned 256/256 zero scores while h_1d was healthy (rms ≈ 4, max_abs ≈ 60). Detect + retry via `backend_lm_head_topk` recovers a non-zero distribution immediately |
| PLE auto-route for Gemma 4 E2B | 2026-04-30 | E2B has `hidden_size_per_layer_input=256` (per-layer-input gate + projection + norm + global PLE embedding). The CPU dense path implements PLE; Metal does not. `generate_streaming` now checks `arch.has_per_layer_embeddings()` and delegates to `generate_via_cpu_q4k` for those models so the residual stream gets the per-layer per-position contribution. Without this E2B emitted multilingual gibberish; with it, "The capital of France is Paris" |
| Diagnostic env vars: `LARQL_DEBUG_TOKEN_IDS`, `LARQL_DEBUG_TOPK` | 2026-04-30 | Per-step token-id + raw top-K scores in both `grid.rs` (gRPC) and `gpu.rs` (dense). Surfaced the "all logits == 0.000" smoking gun that localised the lm_head KNN bug |
| `larql parity --component layer` extended to dense | 2026-04-30 | Was MoE-only (`LARQL_DUMP_RESIDUALS`). Now uses `LARQL_METAL_DUMP_LAYERS` for dense models — wrote per-layer `metal_layer_NN_h_out.f32` and CPU dump files. Gave us the cos ≥ 0.9999 confirmation across 60 layers that ruled out the inference math as the bug source |
| `larql parity --component lm-head` works on dense | 2026-04-30 | Dropped the MoE-only gate for `lm-head` (Q4_K vs f32 reference is backend-agnostic) |
| `test_logits_goldens.rs` compile fix + 5 new entries | 2026-04-30 | Added missing `None` for `predict_q4k_hidden`'s `Option<&RemoteMoeBackend>`; refreshed stale 5 goldens to match current kernel state; added `gemma3-4b-q4k-downq4k` (Q4_K-down regression test), `gemma4-31b-q4k-q6kdown` (Q6_K-down dense), `gemma4-e2b-q4k` (PLE auto-route) — 13/13 passing |
