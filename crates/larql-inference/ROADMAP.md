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
**Files**: `src/forward/generate.rs`, `src/forward/generate_cached.rs`  
Read `tokenizer_config.json` from the vindex, parse the `chat_template` Jinja
field with `minijinja` (already in `Cargo.toml`), apply to the token sequence
before generation. `--no-chat-template` flag to bypass for base models or raw
prompts. `larql-cli` owns the flag; this crate owns the template application.

### EOS detection
**Status**: Partial — checks `<eos>`, `</s>`, `<|endoftext|>` but missing Gemma 4 `<end_of_turn>`  
**Files**: `src/forward/generate.rs`  
Read `eos_token_id` (and `eos_token_ids` list) from `config.json`; also read
`stop_strings` from `generation_config.json`. Check decoded token string + token
ID at every generate step. Gemma 4 lists `<end_of_turn>` in `stop_strings` but
not in `eos_token_id`; without this fix greedy decode runs to `--max-tokens`.

### Token spacing / detokenisation
**Status**: Not started  
**Files**: `src/forward/generate.rs`  
`tokenizer.decode` is called per-token; accumulate instead, trimming only the
very first token. HuggingFace tokenizers use a leading-space convention (`▁Paris`)
that is stripped incorrectly when decoding single tokens, causing "Parisatthe..."
output.

### Token streaming
**Status**: Not started  
**Files**: `src/forward/generate.rs`  
Change `generate` / `generate_cached` to accept `on_token: impl FnMut(&str, f64)`
callback. Caller (CLI) prints each token; server uses SSE chunks from the same
callback. Currently the full token list is collected before returning — the CLI
is silent for the entire `--max-tokens` run.

### Sampling
**Status**: Not started  
**Files**: `src/forward/generate.rs`  
Add temperature softmax, top-k filtering, and top-p (nucleus) filtering as
logit post-processing steps after lm_head and before argmax. No GPU changes
required. Flags (`--temperature`, `--top-p`, `--top-k`) are owned by `larql-cli`.

### Repetition penalty
**Status**: Not started  
**Files**: `src/forward/generate.rs`  
Before argmax / sampling, divide each logit by the repetition penalty if that
token appears in the recent generation window. Practical fix for greedy looping
on base models without a chat template. Flag (`--repetition-penalty`) owned by
`larql-cli`.

### Multi-turn KV state
**Status**: Not started — `larql chat` resets KV cache per turn today  
**Files**: `src/forward/generate.rs`, `src/forward/kv_generate.rs`  
Maintain a running `token_ids` buffer across turns. After each response, append
response token IDs before the next user turn so the KV cache grows across turns.
`--max-context N` eviction: drop oldest turns when the buffer exceeds `N`.

### Long context / dynamic KV
**Status**: Not started — hard-capped at 4096 tokens  
**Files**: `src/forward/generate.rs`  
Expose `--max-context N` (default 8192) threaded to `KVCache::new_per_layer`.
Dynamic Metal buffer growth or sliding-window fallback when `current_len` reaches
`max_seq`. Interim acceptable: warn and truncate, document the limit.

### Gemma 3 4B regression smoke test
**Status**: Not started  
Load `gemma3-4b-q4k-streaming`, run `larql run "The capital of France is" -n 1 --metal`,
assert first token is `"Paris"`. Gate on `CI_INTEGRATION=1` so it doesn't run
on every PR but does run before release branches.

---

## P0: MoE inference completions

### MoE-aware CPU forward pass
**Status**: Not started  
**Files**: `src/forward/layer.rs`  
`predict_q4k` / `WeightFfn::forward` has no MoE branch; the non-Metal CPU path
produces wrong output on Gemma 4 26B A4B. Wire `cpu_moe_forward` (already
implemented in `larql-compute/src/cpu/ops/moe.rs`) into `forward/layer.rs` for
the `predict_q4k` path.

### Wire `RouterIndex` client-side
**Status**: Not started  
**Files**: `src/forward/layer.rs`  
`crates/larql-vindex/src/index/router.rs` exists but is not connected to the
forward pass. Connect it so the MoE router runs locally against the vindex's
router index before dispatching to local or remote experts.

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

## P1: Code quality — modularity & magic strings

### High priority

**Centralise env-var names**
Inline string literals `"LARQL_CPU_STAGE_DUMP"` (`forward/layer.rs:63`),
`"LARQL_WALK_TRACE"` (`vindex/walk_ffn/mod.rs:131`), and others scattered
across modules. A typo is a silent no-op. Create an `env_config` module with
typed accessors (`fn stage_dump_dir() -> Option<PathBuf>`, etc.) as the single
source of truth.

**Deduplicate `current_date()`**
Identical implementation in `capture.rs:288` and `walker/utils.rs:55`, both
using the same approximate `days/365` arithmetic. Delete one, expose from a
shared utility.

**Magic batch size in `graph_ffn.rs`**
`let batch_size = 8192` appears at lines 82 and 166 with the memory rationale
only in an inline comment. Promote to `const GATE_INDEX_BATCH_SIZE: usize = 8192`
at module level with the doc.

**GELU approximation coefficients**
`ffn/mod.rs:86-87` has bare `0.797_884_6` and `0.044715`. Name them
`GELU_TANH_COEFF` / `GELU_TANH_CUBIC` with a source citation.

**Embedding layer −1 sentinel**
`trace/store.rs:43,150` and `trace/types.rs:10` special-case layer −1 inline.
`const EMBEDDING_LAYER: i32 = -1` plus a `fn is_embedding_layer(layer: i32) -> bool` helper.

---

### Medium priority — modularity

**Engine dispatch on string literals**
`engines/mod.rs:156-175` matches `"markov-rs"`, `"unlimited-context"`,
`"turbo-quant"`, `"apollo"` as bare strings. `EngineInfo.backend: String`
exposes the same problem in the public API. Define `BackendKind { Cpu, Metal }`
and `EngineKind { MarkovRs, UnlimitedContext, TurboQuant, Apollo }` enums as
the source of truth; derive `Display` to keep the string interface externally.

**Forward-pass loop duplicated 4+ times**
`predict_with_temperature`, `predict_with_ffn`, `predict_with_router`, and
`predict_with_strategy` all repeat the embed→loop-layers→lm_head shell with
minor per-layer variation. Extract a `predict_impl(weights, tokenizer, tokens,
layer_fn: impl Fn) -> PredictResult` that owns the shell; callers pass a
closure for per-layer logic.

**KV cache loop duplicated across engines**
`MarkovResidualEngine`, `UnlimitedContextEngine`, `TurboQuantEngine` each
re-implement the prefill→token→extend loop. Define a `KVCacheStrategy` trait
(or shared loop helper) to consolidate the common structure.

**`infer_patched.rs` hard-wires `WalkFfn` internals**
`forward/infer_patched.rs:67-91` calls `WalkFfn::new_unlimited_with_trace`
directly then extracts residuals, coupling the INFER pipeline to WalkFfn
internals. Expose residual capture via a callback/trait on `FfnBackend` instead.

**Chat template family-matching duplicated**
`"gemma"`, `"mistral"`, `"llama"` family strings matched independently in
`chat/fallback.rs:30` and `chat/source.rs`. Extract a single `FamilyMatcher`
type reused by both the HF-file path and the hardcoded fallback.

**Trace capture re-implements forward pass**
`trace/capture.rs` duplicates the embedding and layer computation from
`forward/embed.rs` / `forward/layer.rs` to intercept residuals, creating two
parallel implementations that drift on any attention/FFN change. Add a
`capture_residual` callback to the main forward loop instead.

---

### Low priority

**RoPE base constant in tests**
`attention/rope.rs` hard-codes `10000.0` in 7 test methods. Define
`const DEFAULT_ROPE_BASE: f64 = 10000.0` at module level and use it uniformly.

**Walker threshold table**
`walker/utils.rs:30-52` has 7 sequential `if` statements for threshold buckets
(0.01, 0.05, 0.10, …). Replace with a `const THRESHOLD_BUCKETS: &[(f64, &str)]`
slice iterated once.

**`head_dim` inferred from `kv_dim` in TurboQuant**
`engines/kv_engines/turbo_quant/mod.rs:99` guesses `head_dim` from `kv_dim`
instead of reading it from arch. Pass `head_dim` as a parameter from engine
init.

**`L1_DEFAULT_MAX_ENTRIES` unused at call sites**
`vindex/l1_cache.rs:12` defines the constant but call sites hard-code the same
value independently. Audit and use the constant everywhere.

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
