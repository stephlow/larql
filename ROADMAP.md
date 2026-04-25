# LARQL Roadmap

Top-level plan of record. Per-crate specifics live in
`crates/<crate>/ROADMAP.md`; this file tracks user-visible features,
the demo narrative, and cross-crate work.

## Current state

- **490 tests passing** across 14 suites, 0 build warnings.
- **Primary CLI verbs** in place: `run`, `chat`, `pull`, `list`, `show`,
  `rm`, `link`, `serve`. Legacy research commands under `larql dev
  <subcmd>` with argv trampoline for backwards-compat.
- **Dual cache** (HuggingFace hub + `~/.cache/larql/local/`) with
  shorthand resolution (`larql run gemma3-4b-it-vindex …`).
- **Remote FFN path (Phase 0 — dense):** `POST /v1/walk-ffn`
  `full_output: true` returns hidden-size output vectors per layer;
  `RemoteWalkBackend` in `larql-inference` drops into `predict_with_ffn`
  unchanged; `larql run --ffn URL` + `larql serve --ffn-only` wire it
  end-to-end. gRPC mirror also landed.
- **Vindex size reductions:** `--compact` (drops
  `up_weights.bin`/`down_weights.bin`), `--drop-gate-vectors` (rebuilds
  gate from `interleaved_q4k.bin` at load), `--quant q4k` implies f16
  on side-channel tensors. Combined: a new 31B q4k extract is **~22 GB
  vs 52 GB before** (~60% smaller).

---

## P0 — Act 2 of the demo: "The experts live elsewhere"

### Phase 1 — MoE inference path (blocks Act 2)

The whole Act 2 story is MoE-distributed.

- [x] **Gemma 4 MoE architecture hooks** in
  `crates/larql-models/src/architectures/gemma4.rs` — `is_hybrid_moe`,
  `num_experts`, `num_experts_per_token`, `moe_router_key`,
  `packed_experts_gate_up_key`, `packed_experts_down_key`, per-layer
  norms (`pre_feedforward_layernorm_2`, `post_feedforward_layernorm_2`),
  `moe_router_per_expert_scale_key`, `layer_scalar_key`.
- [x] **CPU MoE forward pass** (`crates/larql-compute/src/cpu/ops/moe.rs`):
  BF16 expert dequant, router softmax, top-K selection, per-expert
  gated FFN (gate_proj + up_proj + SiLU + down_proj), weighted sum,
  post-experts RMSNorm. Wired into `decode_token` via GPU/CPU interleave.
- [x] **Metal decode with CPU MoE interleave** — GPU runs dense FFN per
  layer, CPU reads `h_post_attn` (unified memory), runs MoE, adds
  output to `new_h`. Layer scalar correctly applied only to the
  combined FFN+MoE delta (`h_post_attn + scalar * (dense + moe)`),
  not to the full residual.
- [x] **Gemma 4 26B A4B coherent output** — first end-to-end working
  Metal inference (2026-04-24). The four fixes that had to land together:
    1. **Row-padded Q4_K/Q6_K storage** for matrices whose inner dim
       isn't a multiple of 256 (26B A4B's dense `intermediate_size=2112`
       → 8.25 super-blocks per row). Old extraction stored contiguously,
       shader read wrong bytes for every `down_proj` row past 0. See
       `pad_rows_to_256` in `crates/larql-vindex/src/format/weights/write.rs`
       + `inter_padded` dispatch in `metal/decode/mod.rs`.
    2. **Parameter-free router RMSNorm** — HF's `Gemma4TextRouter.norm`
       is `with_scale=False` (no tensor on disk). Added arch trait
       `moe_router_norm_parameter_free()` and the `rms_norm_no_weight`
       branch in `cpu/ops/moe/forward.rs`.
    3. **Outer `post_feedforward_layernorm.weight`** (un-suffixed)
       extracted + applied to `(h1 + h2)` before the residual add —
       distinct from the `_1` dense-branch norm.
    4. **`layer_scalar` scales the whole layer output** (`new_h *=
       layer_scalar`) not the FFN delta — matches HF's final
       `hidden_states *= self.layer_scalar` in `DecoderLayer.forward`.
  Validated end-to-end by residual-diff against HF bf16 (see
  Correctness infrastructure below): L0 `layer_out` cos improved from
  0.7018 → 0.9998; L29 cos from −0.27 → 0.93.
- [ ] **Batched MoE prefill** — current MoE prefill uses token-by-token
  `decode_token` calls (correct, but O(seq_len) serial GPU dispatches
  per layer). Replace with a batched prefill that processes all prompt
  positions in one pass, interleaving GPU dense FFN and CPU MoE at each
  layer. See `crates/larql-compute/src/metal/trait_impl.rs::prefill_q4`
  and `full_pipeline.rs::dispatch_full_pipeline`.
- [ ] **Fix `dispatch_full_pipeline` layer_scalar** — currently scales
  the full residual including `h_post_attn` instead of applying
  `new_h *= layer_scalar` at the end of the layer (HF-accurate). The
  decode path now does this correctly via `apply_whole_layer_scalar`
  in `metal/decode/moe_combine.rs`; prefill path (only matters for
  seq_len>1 with non-MoE `layer_scalar` models) still needs the same.
- [ ] **Chat-template-aware prompting** — 26B A4B is instruct-tuned
  and answers trivia confidently only via the chat template. On raw
  prompts it wanders (HF top-1 on "The capital of France is" is
  `' CAP'`, not `' Paris'`). The architecture regression test now
  asserts against what HF actually produces, but the `run` CLI should
  auto-apply the template for IT models — see P1 "Chat template" below.
- [ ] **MoE-aware forward pass on CPU path** — `predict_q4k` /
  `WeightFfn::forward` has no MoE. The non-Metal CPU path produces
  wrong output on Gemma 4 26B. Wire `cpu_moe_forward` into
  `larql-inference/src/forward/layer.rs`.
- [ ] Wire `RouterIndex` (already exists at
  `crates/larql-vindex/src/index/router.rs`) into the client-side
  forward pass so the router runs locally.

### Phase 2 — Remote expert protocol (Act 2 wire format)

- [ ] `POST /v1/expert/{layer}/{expert_id}` — input residual, output
  residual delta (hidden-size).
- [ ] `POST /v1/expert/batch` — list of `{layer, expert_id, residual}`,
  returns list of deltas. Collapses a layer's K experts into one HTTP
  round trip per server.
- [ ] `--experts 0-31` flag on `larql serve` — load + serve a subset
  of expert IDs so experts can be sharded across machines.
- [ ] `RemoteExpertBackend` in `larql-inference` — MoE-path analog of
  `RemoteWalkBackend`. Handles the sharding map (expert ID range →
  URL), parallel per-layer dispatch, per-expert error handling.

### Phase 3 — LQL / CLI ergonomics

- [ ] `USE "..." WALK ONLY WITH EXPERTS REMOTE { "range": "url", ... };`
  grammar. Extend `crates/larql-lql/src/parser/lifecycle.rs` + executor.
- [ ] `RESHARD EXPERTS { ... };` statement for live redistribution
  (for the "kill one shard, rewire on the fly" proof shot).
- [ ] `larql run --experts '0-31=URL1,32-63=URL2'` CLI flag (MoE
  counterpart to `--ffn`).

### Phase 4 — Data prep

- [ ] `larql slice <vindex> --parts attn,embed,norms,router,index,tokenizer`
  (new subcommand) — carve an attention-only / router-only vindex out
  of a full one without re-extracting from the source model.

### Phase 5 — Deferred until film

- [ ] GPU attention on the client side. `run_attention_block_gpu`
  already exists in `crates/larql-inference/src/attention/gpu.rs` but
  isn't the default path in `forward/layer.rs`. Wire Metal/CUDA into
  the walk-only forward pass so client-side attention runs on GPU
  while FFN/experts go remote.

---

## P1 — Generation UX (chat template, sampling, stopping)

The current `larql run` output loops ("ParisatthecapitalofFranceis...") because
three standard inference features are missing. All are independent and any one
improves the experience.

### Chat template
**Status**: Not started
**Impact**: High — instruction-tuned models (Gemma 3/4 IT, Mistral-Instruct)
loop or produce garbage without their expected prompt format.

`larql run` sends raw text to the model. IT models expect a structured
turn format, e.g. Gemma 4:
```
<start_of_turn>user
The capital of France is<end_of_turn>
<start_of_turn>model
```
Without it, the model sees a bare continuation task and loops greedily.

Fix: read `tokenizer_config.json` from the vindex (already present for
HF-extracted models — lives next to `config.json`). Parse the
`chat_template` Jinja field. Apply it in `larql run` before tokenising.
`minijinja` crate is the standard Rust choice. `larql chat` should always
apply the template; `larql run` can expose `--no-chat-template` for raw use.

### EOS detection and stop strings
**Status**: Partial — `generate.rs` checks for `<eos>`, `</s>`,
`<|endoftext|>` but Gemma 4 uses `<end_of_turn>` which is not in that list.
**Impact**: High — without EOS stopping, greedy decode runs to `--max-tokens`.

Fix: read `eos_token_id` (and `eos_token_ids` list) from `config.json`;
also read `stop_strings` from `generation_config.json` (Gemma 4 lists
`<end_of_turn>` there). Check decoded token string + token ID at every
step in `generate.rs`. `run_cmd.rs` could expose `--stop STRING` for
overrides.

### Token spacing / detokenisation display
**Status**: Not started
**Impact**: Medium — "Paris at the capital..." prints as "Parisatthecapital".

HuggingFace tokenizers use a leading-space convention (`▁Paris`) — the
`tokenizers` crate's `decode` already handles this when
`skip_special_tokens = true`. The bug is likely that `tokenizer.decode`
is called per-token with `false` (keeps `▁` prefix stripped) instead of
accumulating and decoding the full sequence, or that `trim()` is stripping
the leading space. Fix in `generate.rs` decode loop: `decode(&[tid], false)`
and keep the raw string; only trim the very first token.

### Sampling (temperature / top-p / top-k)
**Status**: Not started
**Impact**: Medium for quality, needed for non-deterministic output.

Current path is always greedy (argmax). Add `--temperature F`, `--top-p F`,
`--top-k N` flags to `run_cmd.rs`. Sampling happens after the lm_head
scores are computed in `generate.rs` — no GPU changes required.

### Repetition penalty
**Status**: Not started
**Impact**: Medium — practical fix for the greedy looping problem without
requiring a full chat template. Useful for raw-prompt (`larql run`) and
base models where no chat template exists.

Add `--repetition-penalty F` (default 1.0 = off). Before argmax / sampling,
divide each token's logit by the penalty if that token appears in the
recently generated window. Standard implementation: logit ÷ penalty for
tokens in the last N generated positions. No GPU changes required — purely
a logits post-processing step in `generate.rs`.

### Multi-turn conversation state
**Status**: Not started — `larql chat` resets KV cache per turn today.
**Impact**: High — "chat" implies the model remembers what it said. Without
this, each line in chat mode is an independent cold-start forward pass.

Fix: maintain a running `token_ids` buffer across turns in `run_cmd.rs`.
After each model response, append the response token IDs to the buffer
before the next user turn. Wrap each turn pair in the chat template
(`<start_of_turn>user … model …`) incrementally. Pass the full buffer
to `generate()` so the KV cache grows across turns. Expose `--max-context N`
to bound memory (evict oldest turns when the context window fills).

### Token streaming

### Long context / dynamic KV cache
**Status**: Hard-capped at 4096 tokens today.
**Impact**: High — Gemma 4's headline feature is 1M context. 4096 is a
non-starter for long conversations and the demo's "database" framing.

Two parts:
1. **Configurable max** — expose `--max-context N` (default 8192).
   `KVCache::new_per_layer` already takes `max_seq`; thread `N` through
   `prefill_q4` / `decode_token` call sites in `generate.rs`.
2. **Dynamic growth** — when `current_len` reaches `max_seq`, either
   evict the oldest window (sliding, already implemented as
   `--kv-cache markov-bounded`) or double the buffer. The Metal KV
   cache buffers are pre-allocated; growth requires a realloc + copy on
   the GPU side. A simpler interim: warn and truncate at `max_seq`,
   document as a known limit.
**Status**: Not started
**Impact**: High for UX — without streaming, the CLI is silent until all
`--max-tokens` are done. A 64-token run on Gemma 4 26B takes ~10s with no
output; streaming makes it feel interactive immediately.

Fix: `generate.rs` currently collects tokens into a `Vec` and returns.
Change to accept a `on_token: impl FnMut(&str, f64)` callback (or a
`std::sync::mpsc::Sender`). In `run_cmd.rs`, the callback prints each token
to stdout and flushes. The `larql serve` OpenAI-compatible path (`/v1/chat/completions`
with `stream: true`) would use SSE chunks from the same callback.
Chat mode in `run_cmd.rs` already flushes stdout per turn — streaming
just moves the flush inside the generate loop.

### OpenAI-compatible `/v1/chat/completions`
**Status**: Not started — `larql serve` has custom endpoints but no
OpenAI-compatible chat surface.
**Impact**: High for adoption — makes LARQL a drop-in backend for
Continue.dev, Open WebUI, LiteLLM, and any tool that speaks the
OpenAI API. The "you can do this too" demo moment needs a working URL.

With chat template + streaming landing, this is largely wiring:
- `POST /v1/chat/completions` — accept `{model, messages, stream,
  temperature, max_tokens}`, apply the model's chat template to the
  `messages` array, call `generate()`, return `ChatCompletionResponse`
  (non-stream) or SSE `data: {"choices":[{"delta":...}]}` chunks (stream).
- `GET /v1/models` — return the loaded vindex name so clients can
  enumerate available models.
- Wire into `larql-server/src/routes/` alongside the existing endpoints.

### Auto-extract on `larql run hf://`
**Status**: Not started.
**Impact**: High for adoption — the current flow is `larql extract` →
`larql link` → `larql run`. Three commands before inference starts.
The "you can do this too" moment needs one.

Fix: in `cache::resolve_model`, if the shorthand looks like `hf://owner/name`
and no cached vindex matches, offer to run `larql extract` inline
(with a confirmation prompt or `--yes` flag). Download the safetensors
from HuggingFace, stream-extract to a temp directory, move to the
local cache, then proceed with inference. Re-uses the existing
`larql extract` pipeline — the new code is only in the cache resolver
and a progress display wrapper.

### Gemma 3 4B regression smoke test
**Status**: Not started — no CI check verifies correctness after
compute / inference changes.
**Impact**: Medium — after the MoE and layer_scalar changes, nothing
formally verifies Gemma 3 4B still produces "Paris" at expected
probability. One bad merge could silently break the most-used model.

Fix: add a `tests/integration/` test (or `larql-cli` example) that
loads `gemma3-4b-q4k-streaming` (already in the local cache), runs
`larql run "The capital of France is" -n 1 --metal`, and asserts the
first token is "Paris". Gate on `CI_INTEGRATION=1` so it doesn't run
on every PR but does run before release branches.

---

## P1 — Autoregressive generation quality

### CPU KV cache for autoregressive generation — **SHIPPED**

Two-phase autoregressive decoder in `larql-inference/src/forward/kv_generate.rs`:

- **Prefill** uses `run_attention_with_kv` to capture post-RoPE K and
  post-V-norm V per layer into a `KvCache`.
- **Decode** step in `crates/larql-inference/src/attention/decode.rs`:
  `run_attention_block_decode_step` takes the new token's hidden +
  the layer's existing cache, computes Q/K/V for just that row with
  `apply_rope_partial_at(position=cached_len)`, concatenates the new
  K/V onto the cache, runs `gqa_attention_decode_step` (O(cached_len)
  per head), returns updated cache.

Backend-agnostic via `FfnBackend` — works with `WalkFfn` (local) and
`RemoteWalkBackend` (FFN over HTTP). Measured on Gemma 3 4B f32:

- **Local, no cache (before):** ~1.2 s per decode step, O(N²) growing
- **Local, KV-cached (now):** ~0.6 s/token steady
- **Remote FFN, KV-cached (now):** ~0.5-0.6 s/token steady — same
  protocol as the no-cache version, just many fewer tokens re-shipped

Limitations:
- Skips Gemma 4 E2B per-layer embeddings (PLE) and layer-scalar
  application in the decode loop. Fine for Gemma 3. For full
  Gemma 4 correctness wire `apply_per_layer_embedding` + `apply_layer_scalar`
  into `generate_cached`'s decode layer.
- Q4K CPU path still uses its own no-cache loop (`run_q4k_generate_cpu`).
  Q4K + Metal shader `generate()` remains the fast Q4K path.

### KV cache strategy selector — **SHIPPED (partial)**

`larql run --kv-cache <strategy>` selects how past-token state is kept:

- `standard` *(default)* — full FP32 K/V, unbounded. Shipped.
- `markov-bounded` — sliding window (StreamingLLM-style). Shipped.
  Pass `--context-window N` for the window size. Older tokens drop
  off; memory stays O(window) regardless of generation length.
- `none` — re-run full forward per decode step. O(N²). Shipped as
  correctness fallback.

Not yet wired into the live decode path (all in `crates/kv-cache-benchmark/`):

- `markov-full` — active residual window + cold-tier reconstruction
  via checkpoint layers. Compressed storage via residuals not K/V.
  See `crates/kv-cache-benchmark/src/markov_residual/`. Needs a
  reconstruction primitive that rehydrates K/V for cold-tier
  positions from `token_ids + checkpoint_residual`.
- `turboquant` — per-tensor Q4/Q8 compression of cached K/V. See
  `crates/kv-cache-benchmark/src/turboquant/`. Needs per-step
  quantize/dequantize around the cache append.
- `graph-walk` — experimental, unclear production viability.

### Shader attention + remote FFN

### Metal speedup for non-Q4K decode

**Status:** backend is auto-detected and threaded through
`generate_cached_backend`, but in practice **single-token decode
matmuls stay on CPU** because they fall below the Metal backend's
calibrated FLOP threshold (~500M). Per-layer projections on 4B are
only 5-7M FLOP each — far under the break-even point where GPU
dispatch overhead is worth paying.

**What this means today:**
- `larql run` on f16/f32 vindexes uses CPU BLAS projections regardless
  of `--metal` availability. The KV cache is still the decisive win
  (~6× speedup vs no-cache).
- `larql run --metal` on a **Q4K vindex** routes to
  `larql_inference::layer_graph::generate` (the shader
  `full_pipeline_q4` — all layers fused in one command buffer, KV-
  cached decode on GPU). This is the real GPU path.

**What would actually win on f16/f32:**
1. **Fused f16 full_pipeline shader** — same structure as Q4K's
   `full_pipeline` but with f16 weights. Multi-day shader work.
2. **Batched / speculative decode** — emit N tokens per forward pass
   (draft model, Medusa heads, or speculative sampling). N×M FLOP
   per matmul would clear the threshold. Compatible with remote FFN
   if the batching happens client-side.

See `crates/larql-compute/benches/{linalg,matmul}.rs` and the
many `crates/larql-compute/examples/profile_*.rs` for the measured
GPU-vs-CPU break-even curves — the threshold isn't arbitrary.

### Shader attention + remote FFN (Act 2 endgame)

Q4K + Metal + remote FFN — the ultimate Act 2 configuration. The
shader pipeline (`full_pipeline_q4` / `decode_token`) currently
dispatches attention AND FFN as fused GPU kernels reading from the
Q4K mmap. For remote FFN we'd need to decompose per-layer into:
attention-only GPU kernel → copy residual to host → HTTP round trip
→ copy FFN output back to GPU → next layer's attention. Per-layer
host+network hop kills throughput unless we batch across layers or
use async pipelining.

Worth doing for the Act 2 demo but non-trivial. See
`larql-inference/src/layer_graph/{generate,pipeline_layer,prefill}.rs`
— the fused paths need splitting at the attention/FFN seam.

## P1 — Loose ends in shipped features

### `compute` crate hygiene — six follow-ups from the q4_matvec_v4 review

The 75 %-row-drop bug (closed 2026-04-25, see ship log) was a
symptom: dispatch geometry constants imported separately from the
pipeline kernel name, so the two could silently desync. Walking the
crate to look for the same bug class in other shaders surfaced
several modularity/maintainability issues. Each is its own follow-up.

#### P0a — Stamp pipeline + geometry on a single handle (open)

Today `Q4Pipelines.matvec` is a bare `ComputePipelineState`; geometry
constants (`ROWS_PER_TG`, `THREADS_PER_TG`) are imported separately
from the shader module name at every dispatch site. There were 6
sites, all hand-wired to `crate::metal::shaders::q4_matvec` while the
pipeline was actually built from `q4_matvec_v4` — that mismatch is
exactly how the row-drop bug landed. Other shaders with the same
shape (`q4k_matvec`, `q4kf_qkv_proj`, `q6k_matvec`, `q4k_ffn_gate_up`)
have the same latent risk.

Replace bare pipelines with `KernelHandle { state, rows_per_tg,
threads_per_tg, name }`. Dispatchers read `q4.matvec.rows_per_tg` —
single source of truth, swap kernel = swap struct field. Pinned by a
contract test like `q4_matvec_dispatch_geometry_matches_v4_kernel`
applied to every shader family.

#### P0b — Delete unused `q4_matvec_v2/v3/v5` shaders (open)

Five `q4_matvec_v*` files in `crates/larql-compute/src/metal/shaders/`,
only `_v4` is wired up. v2/v3/v5 are dead weight, all reachable by
name from `library.get_function()` — the row-drop bug literally was
importing the *wrong* one's constants. Delete v2/v3/v5; if any are
still useful for benchmarking move them under `experimental/` behind
a feature flag.

#### P1a — Unify per-quant matvec into one `quant_matvec` trait method (open)

`ComputeBackend` has separate `q4_matvec`, `q4k_matvec`, `q6k_matvec`
methods (and CPU has internal `q8_matvec`, FP4 will need its own).
Adding a quant touches 7-9 places: cpu kernel + metal shader + metal
op + pipeline field + trait method + cpu impl + metal impl +
`QuantFormat` enum + `prefill::encode_quant_matvec_at_offset` +
`metal/stages/quant_matvec.rs`. The match-on-format already exists in
`metal/stages/quant_matvec.rs:36-133`; lift it to the trait. Adding
FP4 should drop to 1 enum variant + 1 match arm + 1 shader + 1 cpu
kernel.

#### P1b — Criterion bench suite covering all quants × cpu/metal (open)

Two criterion benches today (`benches/matmul.rs`, `benches/linalg.rs`)
both CPU only. No Q4_K / Q6_K / Q4_KF / Q8_0 benches, no CPU-vs-Metal
comparison at the same shape, no regression-detector bench (the
75 %-row drop would have shown as a 4× throughput cliff on a Q4_0
lm-head bench three weeks before goldens caught it). 26
`examples/profile_*.rs` files do ad-hoc benchmarking with no
historical baselines.

Consolidate into `benches/quant_matvec.rs` with groups per format
(Q4_0, Q4_K, Q4_KF, Q6_K, Q8_0) × per shape (decode-token N=2560,
prefill-seq=128, lm-head N=262144) × per backend (cpu, metal). HTML
output under `target/criterion/`. Prune the profile examples.

#### P2a — Trait split + Capability enum (open)

`ComputeBackend` is 27 methods, half are `Option<>`-returning
capability probes mixing f32 matmul, per-quant matvec, KV cache, MoE,
decode, prefill, profiling, MoE remote hook, split-profile timing.
Split into smaller traits: `MatMul` (f32/f16), `QuantMatVec` (one
method, dispatch on `QuantFormat`), `DecodeBackend` (token / prefill
/ KV), `ProfileSplit`. Backends opt in via blanket impls or a
capability bitset. Callers branch on `backend.supports(Capability::…)`
instead of `Option::is_some()`.

#### P2b — Decompose `ops/full_pipeline.rs`, drop `decode_profile.rs` (open)

Three big files trending past comprehension:
- `metal/ops/full_pipeline.rs` — 942 LOC
- `metal/decode/mod.rs` — 707 LOC (already shrunk from 1080 in the
  Decode-vs-prefill parity work; same pattern applies)
- `metal/decode_profile.rs` — 567 LOC, looks like `decode/mod.rs`
  plus per-stage timing (DRY violation)

Apply the `encode_qkv` / `encode_ffn` extraction pattern to
`full_pipeline.rs`. Replace `decode_profile.rs` with an opt-in
`Profile` wrapper that decorates `decode/mod.rs` so timing logic
isn't a duplicate decode path.

### `--compact` loader reconstruction — WalkFfn-only today

`larql extract --compact` drops `up_weights.bin` + `down_weights.bin`
from the extract. `WalkFfn` (the production inference path) works fine
— it reads feature-major `{up,down}_features.bin` directly. The dense
ground-truth path (`WeightFfn`, used by `larql dev walk --compare` for
validation) panics with a clear message.

**Why deferred.** The naive fix is to reconstitute
`Array2<f32>` tensors in `ModelWeights.tensors` at load time. For
`down_proj` this requires a transpose (feature-major `[intermediate,
hidden]` → safetensors `[hidden, intermediate]`) which means an owned
copy — **~27 GB of extra heap on 31B**, not viable.

**Proper fix.** Refactor `WeightFfn::forward` (or `ModelWeights`) to
accept feature-major views and pass the transpose flag through to BLAS
gemm. Cross-cutting change: `crates/larql-inference/src/ffn/weight.rs`,
`crates/larql-inference/src/model.rs`, and the `dot_proj` helpers. ~1
focused session.

**Impact.** Unblocks `--compact --compare` for validation workflows.
Does not affect `larql run` or the demo.

### MoE compact mode — refused today

`larql extract --compact` on an MoE architecture refuses with:
> *"ffn_compact not yet supported for MoE architectures — per-expert
> feature-major files don't exist yet"*

**Why deferred.** Two blockers:

1. **Router lives in `up_weights.bin`.** The MoE write path stuffs
   per-expert up weights *and* the router matrix together into
   `up_weights.bin`. Skipping that file loses the router, so the model
   can't dispatch to experts at all. Fix: split the router into its
   own file (`router_weights.bin` already exists as the intended home
   — see `crates/larql-vindex/src/index/router.rs`).
2. **No per-expert feature-major files.** `up_features.bin` /
   `down_features.bin` are single-matrix-per-layer. MoE-compact would
   need per-expert equivalents (~N× the file count or a new layout),
   plus a tool that produces them. No consumer exists yet.

**When to do it.** Pairs naturally with Phase 1 (MoE inference path)
and Phase 2 (per-expert server endpoint). Building those requires a
per-expert-addressable storage layout anyway; compact-MoE falls out of
it.

### `larql dev walk --compact` compatibility

`larql dev walk --compare` against a `--compact` vindex panics (see
above). The panic message points at `WalkFfn` but doesn't explain
`--compare` is the specific operation that's blocked. Improve the
error or disable the `--compare` flag at arg-parse time when the
target vindex is compact.

### Cross-vindex dedup (tokenizer, down_meta)

Tokenizer (~32 MB) and `down_meta.bin` (~30 MB) are identical across
different-precision extracts of the same base model. With ~7 linked
vindexes in the local cache that's ~200 MB of duplicate data. Low
priority — worth doing as a content-addressed store if the cache
grows, otherwise skip.

---

## P2 — Demo production

### Pre-film checklist for the Gemma 4 MoE video

- [ ] Confirm Gemma 4 26B A4B config once the model card is public:
  expert count per layer, top-K, exact active-param figure, GQA ratio.
  Every `~` figure in `docs/demo-script-gemma4-moe.md` needs a real
  number before recording.
- [ ] Measure real footprint + latency on `google/gemma-4-31b-it` for
  Act 1. Replace every `~` in the Act 1 section.
- [ ] Reliability pass on `RemoteWalkBackend` (timeouts, retries,
  mid-layer failure, partial shard outage). A hung HTTP call during
  recording kills the take.
- [ ] `RemoteExpertBackend` (doesn't exist yet — see Phase 2) same
  pass.
- [ ] Decide the repo-public date. `cargo install larql-cli && larql
  serve` should be live the week the video drops so "you can do this
  too" lands with a working command.
- [ ] Pick expert IDs for the Video 3 teaser swap — one that fires on
  medical prompts, one that doesn't — so the "replace expert 42 at
  layer 18" shot lands concretely.

### Memory-footprint `--ffn-only` on the server

`larql serve --ffn-only` today is an operating-mode declaration — it
disables `/v1/infer`, advertises `mode: ffn-service` in `/v1/stats`,
but still loads full `ModelWeights` into RAM. A real FFN-service
doesn't need attention weights resident.

Add `load_model_weights_ffn_only` to `larql-vindex` that skips
attention tensors on the server side. Payoff: serve an MoE without
the attention weights taking a third of RAM.

---

## Done (ship log)

### Metal `q4_matvec_v4` 75 %-row drop on tied-embedding LM-head — closed (2026-04-25)

CPU and Metal disagreed on the next-token argmax for Gemma 3 4B and
Gemma 4 31B because Metal's Q4_0 matvec was only writing 25 % of
output rows at vocab scale. The other 75 % stayed at the buffer's
zero-init value. Llama 2 / Mistral were unaffected (their LM head
goes through the f32 path; Gemma 3/4 are tied-embedding and route
through the synthesised Q4_0 path against the f16 embedding table).

**Symptom.** `test_logits_goldens.rs` recorded *separate* CPU and
Metal goldens on Gemma 3 4B (Metal top-1 = token 50429 logit 2874,
CPU top-1 = token 256240 logit 3632) and Gemma 4 31B. Llama 2 +
Mistral matched bit-for-bit across backends.

**Root cause.** `ops/q4_matvec.rs` and 5 sibling dispatch sites
imported geometry constants from `crate::metal::shaders::q4_matvec`
(`ROWS_PER_TG=32`, `THREADS_PER_TG=1024`) — but the pipeline at
`metal/mod.rs:124` was built from `q4_matvec_v4`, whose row mapping
is hardcoded `row_idx = tg_id * 8 + sg_id`. `num_tgs = N/32` over-
divided; each TG only consumed 8 unique row addresses; result =
exactly `N/4` rows ever written. The "2 of 8 simdgroups firing"
hypothesis in the original write-up was wrong — Metal *did* dispatch
all 32 simdgroups, but v4's row map only consumed sg_id 0..7
uniquely; the remaining sg_ids race-wrote rows already covered by
the previous TG.

**Fix.** One-line import change in 6 files: `use … shaders::q4_matvec`
→ `use … shaders::q4_matvec_v4`. Diagnosed and shipped same day.

**Pinned by.** `crates/larql-compute/tests/test_kernel_lm_head_gemv.rs`
gained four new un-gated regression tests:
- `q4_matvec_metal_writes_every_row_small_n` (N=1024 × K=256)
- `q4_matvec_metal_writes_every_row_misaligned_n` (N=1027,
  not a multiple of ROWS_PER_TG)
- `q4_matvec_dispatch_geometry_matches_v4_kernel` (N=64 — the
  smallest size where the geometry mismatch manifests)
- `q4_matvec_pipeline_max_threads_per_tg` (asserts pipeline cap ≥
  requested TG size; pre-fix this only logged, now it fails loudly)

The two gated vocab-scale tests (`q4_matvec_cpu_vs_metal_at_vocab_scale`,
`q4_matvec_cutoff_sweep`) gained assertions that every output row is
non-zero. `q4_matvec_matches_cpu` in `test_metal_shaders.rs` (rows=10240)
which had been silently failing with `max diff 1831` is now clean.

`test_logits_goldens.rs` per-arch top-5 sets collapsed to one golden
across CPU + Metal, as predicted in the original entry's "After the
fix, they should converge."

**Aftershocks.** The bug was a symptom of geometry constants imported
separately from pipeline kernel name — six follow-ups landed in P1
(`compute` crate hygiene) to kill the bug class entirely:
`KernelHandle` consolidation, dead-shader cleanup, unified
`quant_matvec`, criterion bench suite, trait split + capability enum,
and decomposition of the three remaining oversized files.

### Decode-vs-prefill parity on Gemma 4 31B — closed (2026-04-25)

`test_decode_consistency::decode_consistency_gemma4_31b_dense` was the
single failing test in the parity suite. Metal KV-cached `decode_token`
produced an L0 hidden state with `cos=0.996586, max_abs=1.270`
(2.7 % of the reference layer norm) versus a fresh CPU prefill at the
same effective sequence length, compounding to `cos≈0.76` at L59. Now
matches across all four architectures.

**Diagnosis path.** Built coverage outward from the parity suite until
the gap localised to a single file pair:

1. **kv_cache_append + cache layout/stride hand-off** —
   `test_kernel_kv_cache_append.rs` (14 tests). Pinned the writer
   shader byte-for-byte and the prefill→decode bulk-copy contract
   end-to-end. Cleared as the cause.
2. **rope_at_pos vs rope_at_pos_batched** —
   `test_kernel_rope_at_pos.rs` (6 tests). The two RoPE shaders prefill
   and decode use are bit-identical at the parity-bug geometry
   (head_dim=512, partial 25 %, base=500 000). Cleared.
3. **qk_norm-as-V-norm vs v_norm_batched** — `test_kernel_qk_norm.rs`
   (7 tests). Prefill applies V-norm via the qk_norm shader with
   weight=1, offset=0; decode uses the dedicated v_norm_batched
   shader. Pinned bit-equal at the parity-bug geometry. Cleared.
4. **Per-stage residual capture** —
   `larql_inference::residual_diff::stages::StageCapture` +
   `compare_stages` + `test_decode_stage_bisect.rs`. Extended Metal
   decode with a stage-dump hook (`LARQL_DECODE_DUMP_LAYERS=<dir>` +
   `LARQL_STAGE_DUMP_LAYER=<L>` writes `decode_layer_NN_<stage>.f32`,
   names matching the existing Metal-prefill set). The bisect test
   localised the divergence: every attention-side stage matched at
   `cos=1.0`; the first divergence was at `ffn_out_raw` / `down_out`
   with `cos=0.97 max_abs=5.7 (rel 4.4 %)`.
5. **Kernel test for q4k_ffn_gate_up** —
   `test_kernel_q4k_ffn_gate_up.rs`. Showed catastrophic divergence
   (`cos=-0.08`) at K > 4096 in synthetic, traced to the
   `Q4K_GU_MAX_K = 4096` shared-memory cap.

**Root cause.** Two Metal shaders — `q4k_matvec` and
`q4k_ffn_gate_up` — cached the input vector X in a
`threadgroup float Xsh[4096]` tile. For any `K > 4096` (Gemma 4 31B's
`hidden = 5376`) the tile-load loop wrote past the buffer (Metal UB)
and the dot product later read garbage from those slots. The sibling
`q4k_qkv_proj` had always read X directly from device memory and ran
cleanly at the same K — confirming the fix shape.

**Fix.** Drop the `Xsh[]` tile from both shaders, read X directly
from device memory inside the inner loop. Apple Silicon's L1/L2
cache amortises the repeated reads across the threadgroup's
8 simdgroups. `crates/larql-compute/src/metal/shaders/q4k_matvec.rs`
+ `q4k_ffn_gate_up.rs`, ~10 lines removed each.

**Pinned by.** `test_kernel_q4k_ffn_gate_up::q4k_ffn_gate_up_just_past_max_k_4352`
(one super-block past the old cap) and `..._gemma4_31b_dense`
(production geometry). The previously-`#[ignore]`d cases now pass.

**Decode-side modularisation that fell out of this work.** Pulling
the per-stage dump in cleanly required `decode/mod.rs` to host a few
helper modules: extracted Step 1 (input norm + fused QKV) into
`decode/encode_qkv.rs` and Step 6 (format-aware FFN) into
`decode/encode_ffn.rs`. Behaviour byte-identical; `decode/mod.rs`
went from 1080 → 707 lines.

### Backend parity testing infrastructure + 2 shader fixes (2026-04-24)

Replaced the ad-hoc env-var-driven dump scaffolding (`LARQL_CPU_DUMP_LAYERS`,
`LARQL_METAL_DUMP_LAYERS`, `LARQL_DECODE_DUMP_LAYERS`,
`LARQL_STAGE_DUMP_LAYER`, `LARQL_DUMP_L0`, …) with a typed in-memory
parity API and split the kernel test surface into focused files. Two
real shader bugs surfaced and got fixed in the process.

**New module — `larql_inference::residual_diff`** (3 files):

- `capture.rs`: `ResidualCapture::cpu_prefill / metal_prefill /
  metal_decode` — drives the corresponding production forward path,
  reads its per-layer hidden state into a `Vec<Vec<f32>>`, returns a
  typed struct. Tempfile + env-var plumbing is private to the module.
- `compare.rs`: `compare_captures(a, b, ParityThreshold::tight())`
  → `ParityReport` with first-bad-layer detail, `assert_clean()` for
  test ergonomics. f64-accumulated cos + relative max-abs metrics so
  the same threshold travels across `hidden ∈ {2560, 4096, 5376}`.
- `mod.rs`: 12 unit tests covering shape mismatch, threshold
  semantics, env-var save/restore, dump-file decoding.

**New tests, all driven by the module above or the shared `tests/common/mod.rs`**:

- `larql-inference/tests/test_cpu_metal_parity.rs` (4 tests) —
  refactored. No more env-var setup in the test body. Asserts
  per-layer cos ≥ 0.99995 / rel max_abs ≤ 1 % across all four test
  vindexes.
- `larql-inference/tests/test_decode_consistency.rs` (4 tests) —
  NEW. Asserts `Metal prefill(N) + decode(1) ==
  CPU prefill(N+1).last_position()` per layer. Initially failed for
  Gemma 4 31B; closed 2026-04-25 by the q4k_matvec / q4k_ffn_gate_up
  shared-memory-cap fix (see "Decode-vs-prefill parity on Gemma 4 31B —
  closed" entry above).
- `larql-compute/tests/common/mod.rs` — `get_metal`, `max_diff`,
  `cos_sim` shared helpers across kernel test files.
- `larql-compute/tests/test_kernel_v_norm.rs` (3 tests) — see fixes
  below.
- `larql-compute/tests/test_kernel_kv_attention.rs` (5 tests) —
  pins `kv_attention` against a CPU softmax reference at Llama-2 /
  Gemma 3 / Gemma 4 sliding / Gemma 4 global / long-context T=512.
- `larql-compute/tests/test_kernel_rope.rs` (5 tests) — pins
  `rope_at_pos_batched` at the Gemma 4 global head_dim=512 partial
  RoPE geometry.

**Shader bugs caught + fixed**:

- `metal/shaders/v_norm.rs::v_norm_batched` — the original used
  `[[thread_position_in_grid]]: uint2` with `dispatch_threads`. On M3
  the 2D form silently dispatched only the first TG along Y, so heads
  1+ stayed at the buffer's initial state (zero). Caught by
  `v_norm_batched_all_ones_4x256`. Fix: switched to a single-`uint`
  `[[threadgroup_position_in_grid]]` with one TG per head, mirroring
  the `qk_norm` shader's pattern.
- Same shader, separate latent issue: in production decode the
  shader runs in-place (`x` and `out` aliased), and every thread
  re-read the full head for `sum_sq` while other threads were
  writing. Caught by `v_norm_batched_in_place_matches_reference`.
  Fix: cooperative threadgroup-shared partial-sum reduction with an
  explicit barrier between the read and write phases.

**File-size cleanup**: `test_metal_shaders.rs` shrank 3581 → 3405
lines. Future kernel tests live in dedicated `test_kernel_*.rs`
files using `tests/common/mod.rs` for shared helpers — additions
won't grow the legacy file further.

### Gemma 4 26B A4B end-to-end correctness (2026-04-24)
Closed four independent gaps that together produced garbage output on
the hybrid-MoE 26B A4B model; aligned non-MoE models (Gemma 3 4B,
Gemma 4 31B, Mistral 7B) were unaffected and continue to pass. See
`crates/larql-compute/ROADMAP.md` P0.5 for full per-fix detail.

- **Q4_K/Q6_K row alignment** — 26B A4B's `intermediate_size=2112`
  isn't a multiple of 256, breaking `down_proj` matvec on any
  matrix whose inner dim isn't super-block-aligned. Fix: per-row
  zero-pad during extraction (`pad_rows_to_256`), dispatch with
  `K = inter_padded`. Future vindexes with any non-256 inner dim
  now work automatically.
- **Parameter-free router RMSNorm** — Gemma 4's `Gemma4TextRouter.norm`
  has no learned weight. Added arch flag + `rms_norm_no_weight`.
- **Outer `post_feedforward_layernorm`** extracted and wired — was
  being conflated with the `_1` dense-branch norm.
- **`layer_scalar` applied to whole layer output** not the FFN
  delta — matches HF's `hidden_states *= self.layer_scalar`.

### Correctness infrastructure (2026-04-24)
Tooling to keep the above from regressing, and to localise any
future cross-model forward-pass bug to the right layer / block:

- **Architecture regression suite** —
  `crates/larql-inference/tests/test_arch_golden.rs` runs one
  `#[test]` per `(arch × backend)`. Skip-if-missing for vindex
  cache, so CI stays green but local runs catch breakage
  immediately. Covers Gemma 3, Gemma 4 dense, Gemma 4 hybrid MoE,
  Llama 2 base, Mistral 7B base across GPU + CPU backends.
- **HF-reference residual diff** — `LARQL_DUMP_RESIDUALS=<path>`
  writes every layer's `layer_in` / `h_post_attn` / `layer_out` in
  a binary format symmetric with `/tmp/hf_residuals.py` (hooks
  `Gemma4TextDecoderLayer` in HF transformers). `/tmp/diff_residuals.py`
  prints per-layer cosine + RMS-delta and points at the first
  layer where attention vs FFN diverges. Caught the row-alignment
  bug by bisecting L0 sub-components (attention matched at
  cos=0.9989; down_proj matvec dropped to 0.023).
- **L0 intermediate dumps** (`LARQL_DUMP_L0=<dir>`) — writes
  gate_out, up_out, GEGLU act, down_out, h1, moe_out for the first
  layer. `/tmp/diff_l0_gate_up.py` computes HF's manual MLP from
  the captured pre-norm input and diffs each projection.
- **Vindex surgical patcher** —
  `crates/larql-cli/examples/patch_down_proj.rs` re-quantises
  `layers.N.mlp.down_proj.weight` entries with row-padding from an
  existing vindex. Avoids a ~hour-long 42 GB re-extract when only
  one tensor class needs redoing.

### CLI redesign (primary / dev split)
- New verbs: `run`, `chat`, `pull`, `list`, `show`, `rm`, `link`.
- Research commands moved under `larql dev <subcmd>`; legacy names
  transparently trampolined.
- Dual cache (HuggingFace hub + `~/.cache/larql/local/`) with
  shorthand resolution and source disambiguation.
- `larql serve --ffn-only` flag propagated through CLI → server →
  `/v1/stats`.

### Phase 0 — dense remote FFN baseline
- `POST /v1/walk-ffn` extended with `full_output: true` +
  `seq_len: N`. Server runs the architecture-correct `WalkFfn`,
  returns `[seq_len × hidden]` row-major.
- gRPC mirror (`WalkFfnRequest` / `WalkFfnLayerResult` proto fields).
- `RemoteWalkBackend` in `larql-inference` implements `FfnBackend`,
  slots into `predict_with_ffn` unchanged.
- `larql run --ffn URL` + `larql dev walk --ffn-remote URL` CLI flags.
- `examples/remote_walk_parity.rs` localhost parity probe.

### Vindex size reductions
- `--quant q4k` defaults gate_vectors + embeddings to f16 (previously
  f32 — silent ~32% bloat on every q4k extract).
- `--compact` skips `up_weights.bin` + `down_weights.bin` (saves 3.4
  GB on 4B f16 / ~14 GB proportionally on 31B non-Q4K).
- `--drop-gate-vectors` skips `gate_vectors.bin` on Q4K extracts;
  loader reconstructs from `interleaved_q4k.bin` at load time. 2.3 s
  on 4B / ~12 s on 31B cost, saves 1.7 GB / 13.9 GB respectively.
  Measured via `crates/larql-vindex/examples/bench_gate_dequant.rs`.

### Decoupled-inference memory asymmetry (real, pre-load filtered)
- `LoadWeightsOptions { skip_attn, skip_ffn, skip_lm_head, skip_embed }`
  filters weight manifest entries before mmap+decode — peak RSS
  reflects only what the caller wanted (no allocator-pooling lie).
- Server `--ffn-only`: skips attn + ffn + lm_head + embed at load.
  Walk-ffn endpoint uses `walk_ffn_full_mmap` which reads
  feature-major mmap, not heap tensors.
- Client `--ffn URL`: skips FFN tensors at load. Attention + embed +
  norms + lm_head only on heap.
- Measured on Gemma 3 4B f32 (`gemma3-4b-v2.vindex`):
  - Server RSS: 12.8 GB idle → **12.8 GB through inference** (never grew)
  - Client load: 22.5 s → **7.9 s** (2.8× faster)
  - Forward pass: 3.83 s → **0.83 s** (4.6× faster — no FFN tensor
    touches on the client)
  - Paris @ 80.66% — bit-identical to local unlimited-K walk
- Drop-post-load helpers (`ModelWeights::drop_{attn,ffn,lm_head,embed}_weights`)
  still exist but Rust's system allocator pools freed memory —
  post-load drops reduce heap accounting but not process RSS.
  Superseded by the pre-load filter for the demo path.
- `larql serve` now resolves cache shorthands (`larql serve gemma4-31b-q4k`
  works, not just full paths) via the same `cache::resolve_model`
  logic `larql run` uses.
- `larql run` / `larql dev walk` default `--top-k` to `usize::MAX`
  (unlimited). The old `top-k=10` default silently produced garbage
  on stale/low-K vindexes; removing the cap matches the server's
  `WalkFfn::new_unlimited` behavior.

### Extract tiers + default flip
- New `ExtractLevel::Attention` tier sits between `Browse` and
  `Inference`: includes attention + norms but not FFN. This is the
  first-class way to carve a client-side vindex for the Act 2 demo
  (`larql extract <model> --level attention`). No more ad-hoc slicing.
- Strict `Browse < Attention < Inference < All` ordering + helper
  methods (`writes_attn()` / `writes_ffn()` / `writes_lm_head()`)
  drive what each tier writes. Writers now actually honor the
  boundaries — previously only Browse was meaningfully different from
  non-Browse.
- **Default flip.** `larql extract` now defaults to `--level inference`
  + f16. The common case (`larql extract <model> -o x.vindex`) produces
  an inference-ready vindex out of the box, no flags needed. `--f32`
  opts out of f16 for the rare case someone wants it.

### Gemma 4 config plumbing
- Fixed three missing `final_logit_softcapping` initializers
  (pre-existing compile break on the `architecture-b` branch).
- Dropped an unused `mut` on a closure binding in
  `format/weights/write.rs`.

### Test coverage
- **490 tests across 14 suites**, zero warnings.
- New: cache resolution (19), argv trampoline (8),
  `RemoteWalkBackend` wire format + config + error shape (10), server
  validation + stats mode advertisement (7), local-cache scan
  end-to-end.

---

## Non-goals

- **Not a general model-serving framework.** LARQL's pitch is "the
  model is the database"; inference is a vehicle for the interpretable
  vindex, not the product. We optimize for composability, editability,
  and the demo narrative — not raw throughput against vLLM/TensorRT.
- **Not a training system.** `COMPILE` writes into weights; that's
  patch-level edits, not gradient descent. Stays out of scope.
- **Not HF-compatible on the output side.** We extract *from* HF
  models but the vindex format is our own. A vindex is not meant to be
  loadable by `transformers.AutoModel`.
