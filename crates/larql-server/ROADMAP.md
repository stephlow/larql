# Roadmap — larql-server / larql-router

## Current state (as of 2026-05-01)

### 2026-05-01 — HTTP CPU-path optimisation session

End-to-end Gemma 4 26B-A4B grid jumped from ~17.7 → ~19.7 tok/s on
M3 Max with one local gRPC shard. New per-call wire format,
streaming-overlap default-on, UDS transport, TCP_NODELAY, f16 wire
opt-in. See `Completed` section below for the full per-change list.

### Inherited state (2026-04-26)

- Code quality pass complete: modularity refactor + magic string cleanup + test restructure (see Completed below).
- Follow-up review fixes complete: rate limiting no longer trusts
  `X-Forwarded-For` by default, route/path strings are centralized,
  server loader options are grouped, embed errors use the standard JSON
  error envelope, and server-local clippy allows were reduced.
- Test coverage: **74.2% line / 81.2% function** at the 2026-04-26
  baseline (478 tests). 2026-05-01 (post Q1 cleanup): **131 lib tests +
  37 integration files (~580 tests total), all green**.
- Q1 code-quality cleanup (2026-05-01) shipped 9 of 10 items: 1044-LOC
  `routes/expert.rs` split into 7 focused files; 656-LOC `main.rs` reduced
  to 26 LOC with `bootstrap::serve(cli)` as the orchestration point; new
  `env_flags.rs` (single source of truth for `LARQL_*` knobs) and `wire.rs`
  (shared content-type detection); body-size / JSON-content-type / Cli
  default literals all lifted to typed consts. Q1.10 (stream.rs WebSocket
  state machine) deferred until N0.1 SSE infrastructure lands. See
  Completed → "2026-05-01 (continued) — Q1 code-quality cleanup".
- Server-local clippy was clean at the 2026-04-26 baseline with
  `cargo clippy -p larql-server --tests --no-deps -- -D warnings`,
  re-verified clean post-Q1 on 2026-05-01.
  The dependency-checking form still stops in `larql-vindex`; that is
  tracked outside this server-only pass.
- Examples and synthetic benchmarks checked on 2026-04-26 and re-verified
  2026-05-01: `server_demo`, `embed_demo`, `server_bench --release`,
  `bench_expert_server` (live MoE bench) all pass. `bench_embed_server`
  builds but requires a real vindex path to execute.
- Grid route-table checks are now covered by `cargo test -p larql-router`
  (20 tests, including 7 grid-state tests) plus server announce-envelope tests.
- 2-shard local grid validated end-to-end on Gemma 4 26B-A4B (30 layers,
  inclusive layer ranges 0-14 + 15-29).
- W2 feature-major down retrofittable in-place via
  `larql convert add-feature-major-down --input <vindex>` (1.12 s for
  30 layers, 152 MB output).
- Live W2 surface on `GET /v1/stats.q4k_ffn`:
  `{cache_slots, cache_bytes, feature_major_down}`.
- `--warmup-hnsw` flag eager-builds HNSW across owned layers at boot
  (~325 ms for 15-layer shards on Gemma 26B).
- Grid memory profile (per-shard, single-machine): **9.1 GB RSS**,
  6.7 GB MALLOC_LARGE (gate f32 cache), `down_features_q4k.bin`
  resident at 0 K (capability, not yet exercised on dense path).

## Live perf snapshot (M3 Max, 2-shard grid, 26B-A4B)

### Dense walk-ffn / gate-KNN path

| Operation | Cold | Warm |
|---|---|---|
| `walk-ffn` 1 layer (router) | 12.8 ms | **0.2–0.3 ms** |
| `walk-ffn` 6 layers fanout | — | **1.3 ms** |
| `walk-ffn` 12 layers fanout | 64 ms | 2.6 ms |
| `walk-ffn` 24 layers fanout | 75 ms | 5.0 ms |
| `walk-ffn` 30 layers (full) | 30 ms | **5.9 ms** |
| `walk` (gate KNN, 30L) | — | 8.4 ms |
| 8-way concurrent × 15L fan-out | 112 ms wall | ~1070 layer-evals/sec |

P99 under 8-way contention: 24 ms.

### Remote MoE expert path (Gemma 4 26B-A4B, single in-process shard, layer 15, top-K=8)

`bench_expert_server` against per-layer Q4_K vindex
(`output/gemma4-26b-a4b-q4k.vindex`). Hidden=2816, 128 experts,
moe_intermediate=704, 30 MoE layers.

**bench numbers (2026-05-01, post NEON SDOT + scratch reuse + layer-batch
endpoint + cache cap=256):**

| Operation | Result |
|---|---|
| Vindex load | 5.2 s, +6.0 GB RSS |
| Lazy `get_or_load_weights()` | 1.3 s, +2.8 GB RSS |
| Per-expert bytes (one bench layer, all 128) | 285 MB gate_up + 156 MB down (Q4_K) |
| `forward_moe` warm (router + layer-batch HTTP + combine) | **0.80 ms** mean / 0.79 p50 / 1.09 p99 |
| `cpu_moe_forward` floor (no HTTP, same weights) | **0.37 ms** mean / 0.37 p50 / 0.49 p99 |
| 30-layer sweep (1 decode-step's worth of MoE blocks) | **24.8 ms** (0.83 ms/layer) |
| Steady RSS | **10.5 GB** |

**End-to-end Gemma 4 26B-A4B grid generation (`larql run --moe-shards`,
M3 Max, single local shard, 100-token poem, 3-run avg)**:

| Mode | tok/s |
|---|---|
| HTTP unary (`http://...` shard) | **17.8** |
| gRPC unary (`grpc://...` + `LARQL_MOE_NO_SPLIT=1`) | 17.7 |
| **gRPC + SPLIT overlap (default for gRPC)** | **19.7** |
| UDS HTTP/1.1 (`unix:///path` shard) | 18.2 |
| UDS + f16 wire (`LARQL_MOE_WIRE_F16=1`) | 20.5 (warm); within noise vs UDS f32 |

**Per-call HTTP overhead (loopback, post TCP_NODELAY)**:

| Stage | TCP HTTP | UDS HTTP | gRPC streaming |
|---|---|---|---|
| Server compute (run_experts_cpu_batch) | ~400 µs | ~400 µs | ~400 µs |
| spawn_blocking transition | ~25 µs | ~25 µs | ~25 µs |
| Transport RTT + axum dispatch | ~100 µs | ~50 µs | ~30 µs (multiplexed) |
| Encode + decode | ~5 µs | ~5 µs | ~5 µs (binary protobuf) |
| **Total per-call** | **~660 µs** | **~510 µs** | **~460 µs** |

For comparison, the historical baseline before any of this session's work
was 4.86 ms `forward_moe` warm and 16.6 GB steady RSS on the BF16
monolith (per-expert refactor + Q4_K migration cut that to 1.91 ms / 9.7
GB at 2026-04-26). The 2026-05-01 session took 1.91 ms → 0.80 ms
(another 2.4×) on the same per-call measurement, 56 ms → 24.8 ms
(2.3×) on the 30-layer sweep, and end-to-end ~17.7 → ~19.7 tok/s
(+12%) on the production grid. Cumulative session-on-session win is
**8.6× from the 2.3 tok/s pre-Q4K baseline** (see
`larql-inference/ROADMAP.md → M-CPU-1..6`).

---

## Great new functionality (next big-ticket items)

The numbered F0..F23 items below are mostly **incremental polish**
(metrics, shutdown drain, RBAC, OpenAPI, etc.) — necessary but not
load-bearing for new use cases. The items in this section are
**new capabilities** that would unlock production deployment shapes
the server can't currently serve. Ranked by how much they expand
the addressable surface, not by implementation effort.

### N0. OpenAI API compatibility (Chat Completions, Completions, Responses, Embeddings)

**Status**: Not started. Supersedes the older F10 ("OpenAI-compat
`/v1/chat/completions`") which scoped only the chat endpoint
shallowly. **Highest-leverage item in this section** — every
existing OpenAI client (Python `openai` SDK, JS `openai`, LangChain,
LlamaIndex, Cursor, Continue, Aider, hundreds of agent frameworks,
every dashboard/eval harness in the ecosystem) becomes a larql client
the day this lands. Without it we're a niche-internal tool;
with it we're a drop-in target.

**Scope** — five endpoints, mapped onto our existing inference path:

#### N0.1 `POST /v1/chat/completions` (Chat Completions API)

The current standard. Most clients still talk this even after the
Responses API launched. Non-streaming + streaming via SSE.

```
Request:  {model, messages: [{role, content, tool_calls?, tool_call_id?}],
           temperature?, top_p?, max_tokens?, stream?, tools?, tool_choice?,
           response_format?, seed?, stop?, n?, frequency_penalty?,
           presence_penalty?, logprobs?, top_logprobs?, user?}
Response: {id, object: "chat.completion", created, model,
           choices: [{index, message: {role: "assistant", content,
                       tool_calls?}, finish_reason, logprobs?}],
           usage: {prompt_tokens, completion_tokens, total_tokens}}
SSE chunk: {id, object: "chat.completion.chunk", created, model,
            choices: [{index, delta: {role?, content?, tool_calls?},
                       finish_reason?}]}
SSE terminator: `data: [DONE]\n\n`
```

Translation layer:
- `messages` → render via existing `chat::render_user_prompt` (per
  family chat template) → `encode_prompt` → `generate_streaming`.
- `stream: true` → wrap `generate_streaming`'s `on_token` callback in
  an SSE encoder; emit one chunk per token.
- `tools` → constrained-decoding mask routing the model toward valid
  tool-call JSON. Depends on N0.6 (JSON schema → grammar).
- `response_format: {type: "json_object"}` or
  `response_format: {type: "json_schema", json_schema: {...}}` → same
  constrained-decoding hook.
- `stop` strings → augment the existing `EosConfig` for the duration
  of the call.
- `seed` → pass through to `SamplingConfig` (already supported).

#### N0.2 `POST /v1/completions` (Legacy Completions API)

Older but still widely used (especially by older eval harnesses and
embedding/reranker pipelines that haven't migrated). Simpler shape:

```
Request:  {model, prompt: string | string[], max_tokens?, temperature?,
           top_p?, stream?, logprobs?, echo?, stop?, n?, best_of?,
           seed?, suffix?}
Response: {id, object: "text_completion", created, model,
           choices: [{text, index, finish_reason, logprobs?}],
           usage: {...}}
```

Strictly easier than N0.1 — no chat template, no tool calls, no
multi-message context. Maps directly to `encode_prompt` +
`generate_streaming`. Could ship first as a smoke-test of the
overall translation layer.

#### N0.3 `POST /v1/responses` (Responses API — newer, stateful)

OpenAI's 2025 successor to chat completions. Designed for stateful
multi-turn agents with built-in tool execution + reasoning content.
Pairs naturally with **N1 (stateful chat sessions)** — the
`previous_response_id` field references prior turns whose KV-cache
the server kept resident.

```
Request:  {model, input: string | InputItem[], previous_response_id?,
           instructions?, tools?, tool_choice?, response_format?,
           reasoning?, store?, metadata?, parallel_tool_calls?}

InputItem variants: text input ({type: "message", role, content}),
                    function-call output ({type: "function_call_output",
                    call_id, output}), file references, etc.

Response: {id, object: "response", created_at, status: "completed"|...,
           model, output: [
             {type: "message", role: "assistant", content: [{type: "output_text", text}]},
             {type: "function_call", call_id, name, arguments},
             {type: "reasoning", content},  // for o1 / DeepSeek-R1 style models
           ],
           usage: {input_tokens, output_tokens, reasoning_tokens, total_tokens},
           previous_response_id}
```

Implementation path:
- Without N1: each call is a fresh prefill (server-side response storage
  optional via `store: true` — return `id` for retrieval but don't
  reuse KV-cache).
- With N1: `previous_response_id` → look up the session's KV-cache,
  continue from that state (zero re-prefill on the prior turns).
- Reasoning content (DeepSeek-R1 / Gemma-thinking-style models): emit
  thinking traces as a separate `output[]` entry.

#### N0.4 `POST /v1/embeddings` (Embeddings API)

Existing `/v1/embed` endpoint already does this work; just needs an
OpenAI-shape wrapper.

```
Request:  {model, input: string | string[] | int[] | int[][],
           encoding_format?: "float" | "base64", dimensions?}
Response: {object: "list", data: [{object: "embedding", embedding: [...],
           index}], model, usage: {prompt_tokens, total_tokens}}
```

Two important nuances:
- `input` accepts strings (we tokenise) or pre-tokenised arrays
  (we embed directly via existing `/v1/embed`).
- `encoding_format: "base64"` returns embeddings as
  base64-encoded f32 little-endian bytes — ~33% smaller wire than
  the JSON float array form. Many production clients default to
  base64.

#### N0.5 `GET /v1/models` (already exists, needs OpenAI shape)

Current response shape doesn't match OpenAI's. Reshape:

```
{object: "list", data: [
   {id, object: "model", created, owned_by: "larql", parent?, ...}
]}
```

Trivial — existing route just needs the wrapper.

#### N0.6 Constrained decoding (JSON schema + GBNF grammar)

`response_format: {type: "json_schema"}` and `tools` both require
the decoder to emit only tokens that keep the output grammar-valid.
Today the inference-side decoder has a regex/grammar hook
(`EosConfig` / sampling pipeline already supports "stop strings");
need to extend with a real GBNF parser + JSON Schema → GBNF compiler.

Implementation is well-trodden — port from llama.cpp's `grammar.cpp` /
`grammar-parser.cpp` (well-defined spec; ~1000 LOC). Tracked
separately as F17 in this ROADMAP, but N0 makes it load-bearing.

#### Cross-cutting concerns

- **Streaming framing**: SSE format is `data: {json}\n\n` per chunk,
  terminated by `data: [DONE]\n\n`. axum has `axum::response::sse`
  out of the box.
- **Authentication**: the existing `--api-key` Bearer token mechanism
  works as-is; OpenAI clients send `Authorization: Bearer sk-...`.
- **Model identity**: `model` field in the request maps to a vindex
  ID. For single-model servers, ignore. For multi-model, route via
  the existing model-id mux.
- **Usage tokens**: track `prompt_tokens` (count from
  `encode_prompt`'s output) and `completion_tokens` (count tokens
  generated). Trivial bookkeeping.
- **Error envelope**: OpenAI uses `{error: {message, type, param,
  code}}` — slightly different from our `{error: "..."}`. Add an
  OpenAI-shape error mapper at the route layer.
- **Rate limit headers**: `x-ratelimit-limit-requests`,
  `x-ratelimit-remaining-requests`, etc. — pairs with our existing
  `--rate-limit` machinery.

#### Build order recommendation

1. **N0.5 + N0.4 + N0.2** (Models + Embeddings + Completions) —
   smallest, no streaming, validates the OpenAI shape + auth.
   Makes the server immediately usable for embedding-only and
   text-completion workloads.
2. **N0.1 non-streaming** (Chat Completions, no tools, no
   constrained output yet) — covers ~80% of real chat usage.
3. **N0.1 streaming** (SSE) — every chat UI assumes this.
4. **N0.6** (constrained decoding) — unblocks tools + structured
   output.
5. **N0.1 with tools + JSON mode** — production-grade chat.
6. **N0.3 (Responses API)** — pairs with N1 for stateful continuation.

#### Implementation surface (rough)

- N0.5: ~30 LOC (just a wrapper)
- N0.4: ~150 LOC (translate input format, base64 encoding)
- N0.2: ~250 LOC (legacy completions, simpler)
- N0.1 non-streaming: ~400 LOC
- N0.1 streaming SSE: +200 LOC
- N0.6 GBNF + JSON Schema: ~1200 LOC (port from llama.cpp)
- N0.1 with tools + JSON mode: +300 LOC (depends on N0.6)
- N0.3 Responses API (stateless): ~500 LOC
- N0.3 stateful (with N1): +200 LOC on top

**Total**: ~3200 LOC, shippable in slices. The first 5-day slice
(items 1-3 above) is enough to make larql-server a viable target for
most existing clients.

#### Files

New `routes/openai/` directory — one file per endpoint. Shared
`routes/openai/types.rs` for the request/response schemas (use
`serde` to match the OpenAI shape exactly; let serde-rename do the
heavy lifting for camelCase conversions). Wire into
`routes/mod.rs::single_model_router` alongside the existing routes;
multi-model routing via `model` field in the request body.

#### Why this beats every other N item on leverage

- N1 (sessions) is great but only useful if you have a client to use
  it with. **N0 brings every existing client.**
- N4 (multimodal) is an addressable-market expansion, not a
  client-acquisition unlock.
- N5 (federated knowledge graph) is unique but needs a custom
  client until OpenAI adds federated DESCRIBE to their spec (never).
- N0 is the move that makes everything else discoverable. Ship it
  first.

---

### N1. Stateful chat sessions (KV-cache as a first-class resource)

**Why this is the biggest gap.** Every production LLM API today is
session-aware: client sends the new turn, server remembers prior context
via KV-cache. larql-server's `/v1/infer` is single-shot — every request
re-prefills from scratch. For a 4 K context that's ~100 ms of wasted
compute per turn; for 16 K it's seconds. We're not competitive with
vLLM / TGI / OpenAI for any chat workload.

The pieces exist or are tracked piecemeal — F7 (KV-cache prefix
sharing), F22 (persistent patches as a precedent for session
persistence), the chat session machinery already in
`larql-inference::layer_graph::generate::chat_session` — but no
end-to-end story.

**Proposal**:
- `POST /v1/sessions` → returns `{session_id}` + initial state
- `POST /v1/sessions/{id}/append` → adds user message, generates assistant
  reply, returns SSE stream. KV-cache stays resident.
- `GET /v1/sessions/{id}` → describes current state (msg count, token
  count, model, adapter, last activity).
- `DELETE /v1/sessions/{id}` → frees KV-cache.
- Eviction policy: per-session TTL, total-RSS budget, LRU under
  pressure. Surfaces in `/v1/stats.sessions`.
- Pairs with **N3 (LoRA hot-load)** — sessions can pin a specific adapter.

**Implementation surface**: ~600 LOC. New `routes/sessions.rs`,
new `state::SessionStore`, hook into the existing `generate_streaming`
+ `Detokenizer` machinery. Roughly half the work is the eviction /
budget management — non-trivial but well-scoped.

### N2. Asynchronous batch inference job queue

**Why**: Real-time chat is one model; **bulk inference** (RAG document
processing, embedding pre-compute, reranker scoring, evaluation
harnesses) is another. They have very different SLOs. A batch job
submitter doesn't care about per-token latency; it cares about
throughput, cost, and being able to run while the cluster is otherwise
idle. Today users have to wrap `/v1/infer` in their own retry/queue
glue.

**Proposal**:
- `POST /v1/jobs` → submit `{prompts: [...], model_id, params}` →
  returns `{job_id}`.
- `GET /v1/jobs/{id}` → status + partial results.
- `POST /v1/jobs/{id}/cancel`.
- Optional `webhook_url` in the submit body for completion callback.
- Worker pool: independent rayon thread pool, capped concurrency,
  prioritises real-time `/v1/infer` traffic (job worker yields when a
  real-time request arrives).
- Persistence: jobs survive restarts (write-ahead log to disk).

**Pairs with**: F12 (batched infer in same request), F22 (persistent
state). Together those two are the building blocks; this item is the
asynchronous wrapper.

**Implementation surface**: ~800 LOC. New `routes/jobs.rs`, new
`worker::Pool`, persistence to a `jobs/` directory. The hardest piece
is the priority scheduler — getting it wrong means batch starves
real-time or vice versa.

### N3. LoRA / adapter hot-loading per session

**Why**: Multi-tenant production. Today every tenant either gets the
same base model or has to spin up a separate process. Real production
serving (Anthropic, OpenAI, Together, Replicate) supports per-request
adapter swap. Adapters are 10-100 MB vs the 16 GB base model —
hot-loading hundreds of them is feasible if we have the surface.

**Proposal**:
- `POST /v1/adapters/load` → `{adapter_id, source: "hf://..."|"file://..."|"http://...",
  model_id}` → loads into RAM.
- `GET /v1/adapters` → list loaded adapters with size + last-used.
- `DELETE /v1/adapters/{id}` → evict.
- Inference / sessions take an optional `adapter_id` field — applies
  the LoRA delta to gate/up/down/q/k/v/o matmuls per layer per call.
- Eviction: LRU + total-RSS budget, configurable.

**Pairs with**: N1 (sessions pin adapters). Independent enough to ship
first if N1 is too heavy.

**Implementation surface**: ~500 LOC. The LoRA forward-pass plumbing
already exists at the inference-crate level (per
`larql-inference/ROADMAP.md` § F4 LoRA loading). The server piece is
the lifecycle + RSS management.

### N4. Multimodal API surface (vision tower, mixed image+text infer)

**Why**: Gemma 3/4 ships vision variants; Llama 3.2 too. The vindex
extractor already handles vision tower weights (per
`larql-inference/ROADMAP.md → vision`). We're missing the API
surface — there's no way to send an image to the server today.

**Proposal**:
- `POST /v1/embed/image` → multipart upload → vision tower forward →
  returns `{embedding: [...], hidden_size}`.
- `POST /v1/infer` accepts `images: [base64, ...]` field; server
  routes through the vision tower then concatenates with text tokens
  for the language decoder.
- `POST /v1/sessions/{id}/append` accepts images for multimodal chat.

**Implementation surface**: ~400 LOC server-side once the inference
crate's vision forward path is exposed (currently tracked separately).
Big use-case unlock: docVQA, ChartQA, image classification, image
embedding service.

### N5. Federated knowledge graph over multiple vindexes

**Why**: The DESCRIBE/WALK/SELECT trio makes a vindex a queryable
knowledge graph. Multi-model serving (`--dir`) puts multiple
graphs side-by-side — but each is queried independently. There's no
way to ask "describe France using Gemma's knowledge AND Llama's
knowledge AND my custom vindex". This is a unique capability the
larql architecture enables that nothing else (vLLM, TGI, OpenAI) can
do, and it's invisible.

**Proposal**:
- `GET /v1/federated/describe?entity=X&models=gemma,llama,custom` →
  merges edges across vindexes, sourcing each edge with its origin
  model.
- `POST /v1/federated/select` with cross-model joins ("entities
  Gemma calls capitals AND Llama calls capitals").
- New LQL syntax: `DESCRIBE "France" USING gemma, llama;` already
  hinted in the REPL doc (`USE REMOTE`); the server-side surface is
  the missing half.
- Surfacing model disagreement is a research-grade capability:
  "Gemma says Paris is the capital of France with score 1436;
  Llama says Lyon with score 320. Confidence-weighted merge?"

**Implementation surface**: ~600 LOC. New `routes/federated.rs`,
extends multi-model serving to do cross-model fan-out + merge.

### N6. Live blue-green vindex deployment

**Why**: Production model rollouts. Today swapping a vindex requires
restart (modulo F8 hot-swap, which is admin-only and atomic). True
blue-green wants: load v2 alongside v1, route X% of traffic, observe
metric drift, ramp or rollback.

**Proposal**:
- `POST /v1/admin/deploy` → load `v2.vindex` alongside the active
  `v1.vindex`, returns `{green_id}`.
- `POST /v1/admin/traffic` → set weighted routing
  (`{"v1": 0.9, "v2": 0.1}`).
- `GET /v1/stats.deployment` → per-vindex per-endpoint p50/p99/error
  rate side-by-side. Pairs with F3 metrics.
- `POST /v1/admin/promote/{id}` → atomically swap routing to 100%
  green; old vindex becomes stale-evictable.

**Pairs with**: F8 (admin endpoints), F3 (metrics for traffic
comparison). N6 is the **product** built on top of those primitives.

**Implementation surface**: ~700 LOC. New `routes/admin/deploy.rs`,
extends `AppState` to hold multiple model versions, weighted routing
logic in the request entry points.

---

## P0: Active

### F-FLY. Remote multi-shard deployment on fly.io

**Status**: Not started — next session.

**Goal**: validate the HTTP CPU-path optimisations from the 2026-05-01 session
on a real network (LAN-class RTT ≥ 100 µs), not just M3 Max loopback. Most
of what we shipped is designed to win on real links but is invisible on
loopback (TCP_NODELAY, f16 wire). This is the apples-to-apples test that
tells us whether the in-room engineering translates to a deployable grid.

**Setup target (~2 hosts, then 4-8 if Phase 1 looks good)**:

- 1× client host (Mac dev box or fly.io VM): runs `larql run --moe-shards`
  with attention + dense FFN compute. Holds the 2 GB attention/router/dense
  weight set.
- N× shard hosts (fly.io VMs, ~16 GB RAM each): each runs
  `larql-server --experts START-END --grpc-port 9081 --uds-path ...`
  on a slice of the expert table. 26B-A4B has 128 experts × 30 layers;
  e.g., 4 shards × 32 experts × 30 layers ≈ 4 GB Q4_K + 2 GB working set
  per shard.
- Network: same fly.io region (intra-DC ~0.5 ms RTT) for Phase 1; a second
  region (cross-region ~30-100 ms RTT) for Phase 2 to stress the streaming
  overlap.

**What we expect to learn from this**:

1. Whether the **f16 wire** opt-in actually wins on real links (estimate:
   +3-5% on 1 Gbps, more on slower). On loopback it was within noise; we
   need real RTT to see the wire-bytes saving translate.
2. Whether **gRPC SPLIT default** (now on by default for gRPC) holds its
   ~12% steady-state win when the network leg is bigger than the dense
   FFN GPU leg (instead of comparable). The overlap math says the win
   grows when RTT > dense_FFN_time.
3. End-to-end tok/s ceiling on a real grid — we currently know loopback
   is ~19.7 tok/s; a multi-host grid should be slower per-token but
   throughput-scalable (more shards per host = more concurrent expert work).
4. Whether **predispatch (`batch` dispatch mode)** actually breaks
   generation on every multi-host setup or just on M3 Max loopback. We
   saw garbage output on loopback; might be a different story with real
   network timing.

**Prerequisites already in place** (from this session):

- gRPC streaming default-on for gRPC shards (~12% loopback gain,
  expected to grow on RTT-heavier links)
- TCP_NODELAY on accepted connections (defensive against tail-packet
  stalls on real LAN)
- f16 wire as opt-in (`LARQL_MOE_WIRE_F16=1`)
- Unix domain sockets (`--uds-path`, `unix:///path` URL) for same-host
  shard collocation
- `LARQL_HTTP_TIMING=1` per-call instrumentation (encode / send_total /
  recv_body / decode breakdown)
- `LARQL_MOE_TIMING=1` per-token MoE summary (route / collect / server
  compute / network estimate)
- 9.6× CPU MoE speedup on the shard side (bench: 30-layer sweep
  221 → 22.9 ms; production: 2.3 → ~19.7 tok/s end-to-end on M3 Max
  loopback)

**fly.io specifics worth pinning down before deploy**:

- VM size for shards: 26B-A4B vindex is ~16 GB on disk; needs ~10 GB
  RSS at warmup. `performance-cpu-2x` (~7 GB RAM) won't fit a full
  shard; need `performance-cpu-4x` (~14 GB) at minimum, or shard the
  vindex finer.
- Vindex distribution: cheapest is to ship the full 16 GB to each shard
  and let `--experts START-END` cap working set; alternative is per-shard
  vindex slicing (`larql slice` exists but needs a per-shard variant).
- Persistent volume vs in-memory: with `--warmup-walk-ffn` the boot
  cost is ~6-7 s; if VMs reboot per deploy, that adds up. Consider
  fly.io persistent volumes for the vindex.
- Health check: `/v1/health` is already there.
- Authentication: the existing `--api-key` flag works but a multi-tenant
  fly.io setup probably wants per-shard token rotation (out of scope for
  Phase 1).

### F0. CPU MoE correctness — server path correct, local path TBD

**Status**: Server-side resolved 2026-04-30 (gRPC grid + layer-batch
HTTP path generates correct "Paris" / coherent poem output on
`output/gemma4-26b-a4b-q4k.vindex`). Local in-process `larql run`
without `--moe-shards` not re-validated this session — the kernel work
in `larql-inference/ROADMAP.md → M-CPU-1..6` likely fixed the
underlying issue (NEON SDOT direct-Q4K + scratch reuse + correct
hybrid-combine ordering all share the same code path the local CPU
inference uses), but a smoke-test run is the cheapest way to confirm.

The remaining open item: 2026-04-27 historical analysis below describes
the bug as it existed THEN; most of the suspects have since been
addressed by the per-expert refactor + the M-CPU work. Re-running
`larql run output/gemma4-26b-a4b-q4k.vindex "The capital of France is"`
(no `--moe-shards`) and checking the output is "Paris" would close this
out.

**Historical context (2026-04-27, pre-M-CPU work):**

The per-expert refactor + `experts_packed.bin` removal landed without a
correctness end-to-end check. `larql run` on the 26B-A4B vindex via the CPU
MoE path produces incoherent text ("ever own로 el"), while `larql run --metal`
on the same vindex produces "Paris." The server-side remote-expert endpoint
inherits the same bug because `run_single_expert` and `cpu_moe_forward` share
the same per-expert compute.

**What I tried that did not help:**
- Aligning `cpu_moe_forward`'s router-norm input to `h_norm` (matching Metal's
  `cpu_moe_route(&h_norm, ...)` convention) — different garbage, not "Paris".
- Swapping gate/up row order in the `[2*inter, hidden]` slice — different
  garbage, not "Paris".
- Verified `dequantize_q4_k` is bit-identical to the `larql_models` reference
  via `tests/test_q4k_parity.rs` on synthetic ramp data (3 super-blocks of
  varied content, plus round-trip-within-noise).
- Verified `inter_padded` handling matches Metal's convention (zero-pad
  hidden_state to `inter_padded`, dequant down at `hidden * inter_padded`).

**What's still suspect:**
- Q4_K dequant on the **real per-layer file's bytes** has not been compared
  against Metal's GPU dequant. Synthetic parity ≠ real-data parity.
- The **gate/up convention in HF Gemma 4** could differ from what
  `quantize_moe_entries` assumes about the source BF16 layout.
- BLAS `sgemv` on Apple Accelerate vs Metal's `q4k_matvec` shader could have
  precision drift at 26B scale, though both should be IEEE-754 correct.

**Why the bench numbers were misleading:**
`bench_expert_server` measured `forward_moe` warm at 1.91 ms and the
`cpu_moe_forward` floor at 0.10 ms. Post-fix the floor jumped to 1.81 ms (18×).
The 0.10 ms number was the buggy old code silently returning empty buffers
when the dequant length didn't match the bytes — fast because no work was
happening. This was not flagged because no test compared **output values**,
only latency.

**Diagnosis status (2026-04-27, via `larql parity` + dump-and-diff):**

Layer-by-layer cosine-similarity diff between CPU `predict_q4k` and Metal
`predict_q4k_metal` on the 26B-A4B vindex, using `LARQL_CPU_DUMP_LAYERS` +
`LARQL_DUMP_RESIDUALS`:

| Stage at layer 0 | cos(cpu, metal) |
|---|---|
| h_embed (input to layer 0) | 1.000000 |
| h_post_attn (post-attention) | 1.000000 |
| layer_out (post-FFN+MoE+combine) | **0.626708** ← divergence |

Attention is correct on layer 0; the divergence is in the **FFN + MoE +
combine** between `h_post_attn` and `layer_out`. The CPU MoE block routes
to the same top-K experts as Metal at layer 0 (verified via `MOE_DEBUG=1`:
both pick `[79, 114, 16, 92, 89, 101, 67, 46]` with the same `moe_out_rms`).
Per-expert math is provably correct (parity test). The bug is therefore in
how `run_moe_layer_cpu` composes h1 (dense), h2 (MoE), the outer
post-FFN norm, and `layer_scalar` — and it has drifted from Metal's
`metal/decode/moe_combine.rs::apply_outer_combine`.

`larql parity` v1 shipped (CLI subcommand, `larql-cli/src/commands/diagnostics/parity.rs`)
with `--component moe-expert` + `--component moe-block` and `--backends reference,cpu`.
Run on the 26B-A4B vindex the tool reports:

| Component | reference vs cpu max abs diff | Verdict |
|---|---|---|
| `moe-expert` layer 0 / expert 0 | 4.3 × 10⁻⁶ | within fp32+BLAS noise |
| `moe-block` layer 0 (router → top-K → K experts → sum → post-norm) | 8.4 × 10⁻⁵ | within fp32+BLAS noise |

So the entire MoE expert pathway — Q4_K dequant, gate matmul, up matmul,
activation, down matmul, router, top-K, weighted sum, post-experts norm — is
mathematically correct end-to-end. The bug producing garbage on `larql run`
is **outside** the MoE block. Suspect surface area:

- attention block (Q/K/V proj, RoPE, softmax, O proj) — Metal vs CPU
- hybrid combine: `h1 + h2 → moe_post_outer_norm → + h_post_attn` in
  `larql-inference/src/vindex/q4k_forward.rs::layer_step`
- `apply_layer_scalar` and PLE (`apply_per_layer_embedding`) afterwards
- per-position iteration loop on prefill (`for pos in 0..seq_len`)

**Root cause (further localised 2026-04-27):**

The CPU and Metal paths use **two different forward implementations** for
hybrid-MoE Q4_K vindexes — they have drifted:

- **Metal**: `predict_q4k_metal` builds `FullPipelineLayer` per layer and
  calls `backend.decode_token(&layers, ...)`. Hybrid MoE handled by
  `decode_token_with_moe` → `gpu_moe_dispatch`. This works.
- **CPU**: legacy `q4k_forward.rs::predict_q4k_step` →
  `run_moe_layer_cpu` (hand-rolled) → `cpu_moe_forward` per position +
  hand-rolled hybrid combine (`combined = h1 + h2`,
  `combined_normed = outer_norm(combined)`, `h_out = h_post_attn + combined_normed`).
  Doc comment in that function says it's "verified against HF bf16 via
  residual-cosine diff in the Metal `diag.rs` dumps" — but the file has
  since drifted from Metal and the verification is stale. This produces
  garbage end-to-end on Gemma 4 26B-A4B.

Routing-convention fix (apply router_norm to `h_norm`, not raw `h`,
matching Metal's `cpu_moe_route(&h_norm, ...)`) was applied to
`cpu_moe_forward` and `MoeRouterWeights::route`, with regression tests in
`larql-compute/src/cpu/ops/moe/mod.rs`. Necessary but not sufficient — the
hybrid combine in `run_moe_layer_cpu` is still wrong.

**Next steps for F0 (proper fix):**

The cleanest path is to **delete `run_moe_layer_cpu` and route CPU
predictions through the same `FullPipelineLayer` + `decode_token` pipeline
Metal uses**, swapping `MetalBackend` for `CpuBackend`. That requires
`CpuBackend::decode_token` to support Q4 layers (it currently doesn't —
`predict_q4k_metal` literally `expect()`s "need Metal with Q4 kernels").

Either:
- Implement `CpuBackend::decode_token` for Q4 layers — substantial work
  porting the Metal kernels' algorithm to CPU + BLAS, but unifies the two
  paths and resolves all class-of-bug drifts at once.
- Patch `run_moe_layer_cpu` to match Metal's exact hybrid combine. Faster
  but leaves the dual-path drift surface in place; another knob will go
  out of sync next session.

A `larql parity --component layer` (parity v2) component would catch this
class of bug going forward — diffing the **full hybrid layer output**
between CPU and Metal would have surfaced the combine drift immediately.
That's the right next investment.

**Implication for the remote-MoE story:**
The wire format, `--experts` shard ownership (with the off-by-one fix),
the per-expert byte-table API, and the per-layer Q4_K layout all work
correctly. What does **not** work is the CPU numerical compute on the
server side. Until F0 is closed, "remote MoE on Gemma 4 26B-A4B" is
plumbing-correct but inference-incorrect — clients pointing at a remote
larql-server shard will get garbage output. Workaround: use `--metal` for
all-local generation; remote-MoE is on hold.

---

Functional gaps from the 2026-04-27 server review. Numbering is stable so we
can reference items in commits and reviews.

### F1. Router-side expert-shard fan-out
**Files**: `crates/larql-router/src/main.rs`, `crates/larql-router/src/grid.rs`,
`crates/larql-router-protocol/proto/*.proto`.
The grid router fans out `walk-ffn` by layer ranges only. For MoE, the
remote-expert client (`RemoteMoeBackend` in `larql-inference`) carries the
expert→shard map itself; nothing on the router side. Means clients can't just
point at the router for MoE. Add `POST /v1/expert/{layer}/{id}` and
`POST /v1/expert/batch` to the router, with shard discovery via the existing
gRPC announce stream. Pairs with **F11** (topology endpoint).

### F2. Streaming HTTP infer (SSE)
**Files**: `crates/larql-server/src/routes/infer.rs` (new sibling
`infer_stream.rs`).
`/v1/infer` is single-shot — full output buffered, no incremental tokens. WS
has it (`WS_CMD_INFER`) but most chat UIs talk SSE. Add
`POST /v1/infer/stream` with `text/event-stream`. Same generation loop, yield
each token. Mid-generation cancellation on client disconnect (see **F16**).

### F3. `/metrics` (Prometheus)
**Files**: `crates/larql-server/src/main.rs`, new `crates/larql-server/src/metrics.rs`.
No latency histograms, no per-endpoint counters, no rate-limit drops, no
shard-call durations today. Wire `metrics` + `metrics-exporter-prometheus` (or
hand-rolled). Histograms for: `walk-ffn` per `layer_count`, `forward_moe` per
`top_k`, queue wait, auth failures, rate-limit drops, shard-call latency.

### F4. Graceful shutdown with in-flight drain
**Files**: `crates/larql-server/src/main.rs`.
SIGTERM today probably cuts long-running walks. Standard axum + tokio shutdown
signal: stop accepting, drain N seconds (configurable), hard-kill. Important
for grid rolling restarts.

### F5. Readiness vs liveness split
**Files**: `crates/larql-server/src/routes/health.rs`, `routes/mod.rs`.
`/v1/health` returns `{status, uptime, requests_served}`. Add `GET /v1/ready`
returning 503 until weights are loaded (under `--warmup-walk-ffn` or first
lazy load); include `model_id`, `mode`, `version`, `git_sha`, `format`
(per-layer vs legacy) in the readiness payload. Standard k8s liveness/readiness
split.

---

## P1: Active

### Q1.10 Reduce `routes/stream.rs::handle_stream_infer` (327 LOC) — deferred

The remaining open code-quality item from the 2026-05-01 audit. The other
nine (Q1.1–Q1.9) shipped — see "Completed → 2026-05-01 (continued) — Q1
code-quality cleanup". Q1.10 is deferred until N0.1 (OpenAI Chat
Completions SSE) forces a similar streaming state-machine shape; the
two should share infrastructure. Effort estimate: ~3 hours when picked up.

---

### F6. Replica round-robin + retry on shard failure
**Files**: `crates/larql-router/src/grid.rs`.
Router picks first owning shard; no load-balancing across replicas, no retry
on 5xx. `--shards "0-15=A,0-15=B"` doesn't fan evenly today.

### F7. KV-cache prefix sharing for chat
**Files**: `crates/larql-inference/src/layer_graph/generate/*`,
`crates/larql-server/src/routes/infer.rs`.
Every `/v1/infer` call is fresh prefill. For chat (long shared system prompt +
short user turn) prefix-caching is a 5–10× decode-time win. Needs a
`session_id`-keyed KV cache.

### F8. Vindex hot-swap admin endpoints
**Files**: `crates/larql-server/src/routes/` (new `admin.rs`),
`crates/larql-server/src/state.rs` (mutable model registry).
`POST /v1/admin/vindex/load`, `DELETE /v1/admin/vindex/{id}`,
`POST /v1/admin/vindex/reload`. Admin-key-gated (see **F14**). Otherwise every
model swap is a process restart.

### F9. Binary wire format for `expert/batch`
**Files**: `crates/larql-server/src/routes/expert.rs`,
`crates/larql-inference/src/ffn/moe_remote.rs`.
A K=8 batch on Gemma 4 26B-A4B is ~90 KB JSON per call. The
`application/x-larql-ffn` binary format already exists for `walk-ffn`; mirror
it for `expert/batch`. Expected 3–5× wire reduction.

### F10. OpenAI-compat `/v1/chat/completions` — superseded by N0

This item scoped only the chat completions endpoint shallowly. See
**N0** in the "Great new functionality" section above for the full
plan: chat completions + completions + responses + embeddings +
models, with streaming, tools, structured output, and constrained
decoding. F10 is left here for cross-references; the work happens
under N0.

### F11. Expert topology endpoint
**Files**: new `crates/larql-server/src/routes/topology.rs`.
`GET /v1/expert/topology` returns `{model_id, layers, num_experts, owned: [start,end]}`.
Lets clients build the shard map dynamically instead of having it baked in.
Pairs with **F1** (router fan-out).

### F12. Batched infer
**Files**: `crates/larql-server/src/routes/infer.rs`.
`/v1/infer` takes one prompt today. RAG workloads send N prompts; one batched
call across them amortises router/dispatch overhead. Either accept
`prompts: [...]` or new `/v1/infer/batch`.

### T3. Review follow-up — server hygiene ✅ done 2026-04-26

**Scope**: follow-up from review of `larql-server` focused on magic strings,
modularity, cleanliness, tests, and clippy.

Shipped:
- `X-Forwarded-For` is ignored by default for rate limiting; new
  `--trust-forwarded-for` opt-in is for deployments behind a trusted proxy.
- HTTP protocol constants added for shared health path, API prefix,
  bearer prefix, and binary FFN content type.
- Route path literals in `routes/mod.rs` centralized as named constants so
  single-model and multi-model routing drift is easier to spot.
- `load_single_vindex` now takes a `LoadVindexOptions` struct instead of
  an 11-argument call and repeated `too_many_arguments` clippy allows.
- Embed endpoints now return the standard `{"error": ...}` JSON envelope
  for errors instead of a mix of plain text and JSON.
- Server-local clippy cleanup removed the repeated `too_many_arguments`
  exemptions from the vindex loading path.

Follow-up worth keeping open:
- Consider a route-registration macro/table if route count keeps growing.

### T1. Test coverage — functional tokenizer + uncovered routes ✅ done 2026-04-26

**Outcome**: 49.1% → **58.0% line**, 56.4% → **65.3% function**. 345 → 402 tests.

**Root cause fixed**: added `functional_tokenizer()` (WordLevel, France→0 etc.) to
`tests/common/mod.rs`. The empty BPE tokenizer that previously blocked all
tokenize-dependent routes is now supplemented by a real in-memory tokenizer that
maps test words to embeddings with known KNN hits.

**Files moved:**

| File | Before | After |
|---|---|---|
| `band_utils.rs` | 35% | **100%** |
| `routes/describe.rs` | 48% | **95%** |
| `routes/walk.rs` | 38% | **96%** |
| `ratelimit.rs` | 70% | **98%** |
| `routes/walk_ffn.rs` | 54% | **77%** |
| `routes/patches.rs` | 63% | **91%** |
| `routes/relations.rs` | 83% | **91%** |

**Remaining hard ceiling** (no path forward without real weights or real sockets):

| File | Coverage | Reason |
|---|---|---|
| `grpc.rs` | 0% | Needs full gRPC server+client; defer |
| `routes/stream.rs` | 0% | WebSocket — needs `tokio-tungstenite`; defer |
| `routes/explain.rs` | 11% | Calls `get_or_load_weights()`; rest gated on real model |
| `embed_store.rs` | 25% | Reads real f16 embedding files |
| `main.rs` | 0% | CLI entrypoint; skip |

### T2. Test coverage — remaining reachable paths ✅ done 2026-04-26

**Current**: 74.2% line / 81.2% function. 478 tests.

**Completed this pass:**
- `grpc.rs` 0% → **65%** — 28 direct gRPC handler tests (health, stats, describe, walk, select, relations, walk_ffn, infer, stream_describe)
- Magic strings: `"probe"` → `PROBE_RELATION_SOURCE`; `"ok"` → `HEALTH_STATUS_OK`; infer mode strings in grpc.rs; WebSocket message types in stream.rs (`WS_TYPE_*`, `WS_CMD_*`)
- `embed_store.rs` 25% → **98% line** — tiny f16 mmap fixtures cover open, size validation, lookup, L1 cap, out-of-range, subnormal/inf/nan conversion.
- `announce.rs` 6% → **56% line** — extracted deterministic message builders for announce, heartbeat, dropping, and grid bearer metadata.
- `main.rs` boot/loading/discovery helpers moved into `bootstrap.rs`; `bootstrap.rs` has **92% function** coverage for parse/discovery/serve-alias/options behavior.
- `routes/stream.rs` 0% → **65% line** — WebSocket JSON message builders plus pure describe-message planning cover missing-entity, no-model, and functional edge streaming cases.
- `routes/infer.rs` 32% → **56% line** and `routes/explain.rs` 18% → **46% line** via request/default deserialization tests and response-formatting helpers.
- `routes/embed.rs` 67% → **87% line** — binary embed/logits parsing extracted into helpers; HTTP tests cover binary success, malformed JSON, truncated binary input, hidden-size mismatches, no-model errors, and cacheable single-token JSON/binary responses.
- `routes/walk_ffn.rs` 77% → **80% line** — validation helpers now cover layer selection precedence, missing layers, seq_len handling, overflow, and latency rounding.

**Remaining hard ceiling:**

| File | Current | Gap | What to add |
|---|---|---|---|
| `main.rs` | 0% | 237 lines | Tokio binary entrypoint; boot orchestration is covered through `bootstrap.rs` |
| `bootstrap.rs` | 43% | 134 lines | Real vindex load path still requires filesystem fixtures with full vindex assets |
| `routes/stream.rs` | 65% | 148 lines | Full WebSocket socket loop still needs a client harness such as `tokio-tungstenite` |
| `routes/explain.rs` | 46% | 167 lines | Main path gated on `get_or_load_weights()` and real inference trace |
| `routes/infer.rs` | 56% | 82 lines | Prediction paths need real or injectable inference backend |
| `routes/embed.rs` | 87% | 74 lines | Remaining positive logits path requires loadable weights/lm_head fixture |
| `routes/walk_ffn.rs` | 80% | 125 lines | Remaining full-output path requires loadable weights/FFN fixture |
| `routes/warmup.rs` | 80% | ~15 lines | `warmup_hnsw=true` warn path (HNSW not enabled) |
| `announce.rs` | 56% | ~78 lines | Remaining gap is live gRPC stream lifecycle and retry loop |

### G1. Cold-start profile ✅ done 2026-04-26
**Findings**: walk-ffn cold cost decomposes into two distinct phases:

1. **First walk-ffn ever**: ~1.27 s + ~2.9 GB RSS — lazy
   `get_or_load_weights` builds the f32-decoded gate-vector cache,
   loads `lm_head.bin` + `norms.bin`. One-shot regardless of which
   layer was requested. Confirmed not Metal init: a prior gate-KNN
   walk only adds 2 MB.
2. **First touch of each new layer**: ~17 ms + ~11 MB RSS — kernel
   page-fault for the layer's `interleaved_q4k.bin` slice (gate +
   up + down, ~22 MB on disk). Linear in number of cold layers.

Warm steady state is **0.2–0.3 ms/layer**. The 50× cold:warm ratio
is mostly phase 1; phase 2 is ~50× cheaper.

Conclusion: the win lives in phase 1 — pre-load weights at boot.
Mmap prefetch is a 12 ms one-shot for all 30 layers (negligible).
Both wired in **G2** below.

### G2. `/v1/warmup` endpoint + `--warmup-walk-ffn` flag ✅ done 2026-04-26
**Impact (measured on Gemma 26B)**: first walk-ffn **1247 ms → 12.6 ms (99×)** at the cost of +3.2 GB pre-allocated RSS and ~1.3 s boot delay.

Shipped:
- `POST /v1/warmup` accepting `{layers, skip_weights, warmup_hnsw}`
  (all optional). Returns `{weights_loaded, weights_load_ms,
  layers_prefetched, prefetch_ms, hnsw_built, hnsw_warmup_ms,
  total_ms}`.
- `larql-server --warmup-walk-ffn` boot flag — calls the same code
  path before the listener binds. Goes through
  `warmup_model_async` (`spawn_blocking`) because the boot point
  is already inside the tokio runtime.
- The endpoint runs the work on a blocking pool so the runtime
  stays responsive.

### G3. Dual-host gRPC self-assembling grid ✅ done 2026-04-26
**Live-validated** (single-host two-port simulation, exercises the
same code path as a real LAN-distributed grid):

- Shards launched with `--join http://router:50052 --grid-key <s>
  --public-url http://shard:port` register automatically; router
  logs `Grid: server joined layers=0-14` and updates coverage.
- `total_layers_covered` field on the router is the operator's
  view of grid completeness.
- Killed shard A → router logs `Grid: server left`, coverage drops.
  Layer-5 request returns HTTP 400 `"layer 5 has no owning shard"`
  (clean error, not hang). Layer 22 (live shard B) stays at 0.3 ms.
- Restart killed shard → it auto-rejoins, coverage returns to 30,
  layer 5 routes successfully (cold-page first request: 13.9 ms).
- README "Recommended setup" updated with the `--grid-port` /
  `--join` recipe (separate edit pending).

The gRPC mechanism is production-ready as of this validation.
True cross-host RTT measurement is forward-looking (G3a below).

### G3a. Cross-host RTT measurement *(forward-looking)*
**Status**: open. Requires two physical machines on the same LAN.
The same-host validation establishes correctness; cross-host
measures the additional TCP overhead per fan-out.

## P2: Forward-looking

### G4. mmap residency control endpoint
**Impact**: For long-running shards under memory pressure, expose
`POST /v1/mmap/advise {layers, advice: "willneed"|"dontneed"}` so
operators can trim RSS or pre-warm specific layer ranges without
restarting.

### G5. Per-shard expert routing
**Impact**: For DeepSeek-V3+/Kimi K-class models (1k+ experts), shard
by expert ID within a layer rather than by layer range. Needs an
`ExpertRoute` message type in `larql-router-protocol` and
GridState dispatch updates. Mentioned in larql-vindex P2. Subsumed by
**F1** (router-side expert fan-out) at the router layer; G5 covers the
router-protocol changes specifically.

### G6. Live router-shard topology change
**Impact**: Today shards are static (`--shards` flag at router boot).
For ops convenience, expose `POST /v1/router/shards` (admin-gated)
to add/remove a shard without restarting the router. Pair with
`--grid-port` health checks.

### F13. OpenTelemetry tracing exporter
**Files**: `crates/larql-server/src/main.rs`.
Per-request spans across HTTP→shard fan-out. `tracing_subscriber::fmt` is the
only output today. Wire `tracing-opentelemetry` + OTLP exporter, configurable
via `--otel-endpoint`. Pairs with **F3** (metrics).

### F14. Per-key quotas + audit log
**Files**: `crates/larql-server/src/auth.rs`, `crates/larql-server/src/main.rs`.
Single API key today; no per-key quotas, no rotation, no scoped tokens. Add
`--api-keys keys.toml` (name + role + per-key rate). Structured audit on
patches + admin ops to a configurable sink (file / stdout / OTel).

### F15. RBAC (read-only vs admin keys)
**Files**: `crates/larql-server/src/auth.rs`, all mutating routes.
Today any key can patch the loaded model. Add `role` per key
(read / infer / patch / admin). Mutating endpoints (`patches/apply`,
`insert`, future `admin/*`) require the matching role.

### F16. Mid-generation cancellation on HTTP infer
**Files**: `crates/larql-server/src/routes/infer.rs`.
Client disconnect on `/v1/infer` waits for the full max_tokens. Wire
`tokio::select!` against an axum `OnUpgrade`-style cancellation token (or just
poll the connection on each decode step) to abort early.

### F17. Structured-output / grammar-constrained generation
**Files**: `crates/larql-inference/src/layer_graph/generate/*`,
`crates/larql-server/src/routes/infer.rs`.
`{format: "json", schema: ...}` or `{grammar: "gbnf:..."}` on `/v1/infer`.
Constrains decoding by masking the logits to grammar-valid tokens. Standard
ML-server feature; missing today.

### F18. Log-prob / perplexity endpoint
**Files**: new `crates/larql-server/src/routes/logprobs.rs`.
`POST /v1/logprobs {prompt, top_k}` — return per-token log-probabilities.
Needed for ranking, classification, and eval workflows.

### F19. OpenAPI schema route
**Files**: new derive macro setup using `utoipa` (or hand-rolled).
`GET /openapi.json`. Required for SDK codegen, `kubectl explain`-style
tooling, and external API consumers. Today external consumers read the
README.

### F20. Compression negotiation
**Files**: `crates/larql-server/src/main.rs`.
No `Content-Encoding: gzip|zstd` advertised; relies on a reverse proxy. Wire
`tower-http::compression`. Particularly useful for `walk-ffn` JSON responses
on slow links.

### F21. `/v1/stats` per-layer mmap residency
**Files**: `crates/larql-server/src/routes/stats.rs`.
Existing `q4k_ffn` block exposes cache slots/bytes; extend with per-layer
hot/cold (resident vs paged-out) so operators can see what `--release-mmap-after-request`
actually buys them.

### F22. Persistent patches
**Files**: `crates/larql-server/src/session.rs`,
`crates/larql-server/src/routes/patches.rs`.
Patches are session-scoped today; no on-disk overlay. Add a durable
`POST /v1/patches/save` + auto-apply on boot. Pairs with **F8** (hot-swap)
so a patched model survives restart.

### F23. Python HTTP client SDK
**Files**: new `crates/larql-python/src/http_client.rs` (or new crate).
`larql-python` is walk-only against a local vindex; no HTTP client. Add a
`pip install larql` package speaking the server's HTTP API (sync + async),
mirroring the OpenAI Python SDK shape. Pairs with **F10** (OpenAI compat) so
the SDK is a thin wrapper over the OpenAI client.

---

## Completed

### 2026-05-01 (continued) — Q1 code-quality cleanup (9 of 10 items)

The Q1 audit catalogue from earlier the same day, executed in a follow-on
session. All public APIs preserved; existing test surface unchanged.
Q1.10 (stream.rs WebSocket state machine) deferred until N0.1 (OpenAI
Chat Completions SSE) forces a similar shape.

| Item | Outcome |
|---|---|
| **Q1.1** Split `routes/expert.rs` (1044 LOC, 6 concerns) | New `routes/expert/{mod,single,batch_legacy,layer_batch,cpu,metal,warmup}.rs` directory. mod.rs (90 LOC) re-exports the historical public surface (`run_expert`, `run_experts_cpu_batch`, `run_experts_metal_batch`, `warmup_*`, `handle_*`); each sibling file is ~100-225 LOC with one clear concern. `metal.rs` is `#[cfg(feature = "metal-experts")]`-gated so non-Metal builds compile clean. |
| **Q1.2** Centralise env-var flags into `src/env_flags.rs` | New module with one `pub const` per `LARQL_*` name + cached presence accessors backed by `std::sync::OnceLock` (process-wide, not TLS — env vars don't change at runtime). Replaced 12 raw `std::env::var(...)` call sites in `routes/expert/*` and `grpc_expert.rs`; removed two ad-hoc `thread_local! { static HTTP_TIMING ... }` blocks. README env-var table now references the same names that show up in `env_flags::*`. |
| **Q1.3 + Q1.9** Shared `wire::has_content_type` | New `src/wire.rs` with `has_content_type(headers, expected) -> bool` (uses `contains` so parameterised types like `application/json; charset=utf-8` match). Replaced 4 inline header-detection patterns in `routes/walk_ffn.rs`, `routes/embed.rs` (×2), `routes/expert/batch_legacy.rs`. 4 unit tests cover exact-match, parameterised, mismatch, and missing-header cases. |
| **Q1.4** Body-size limit constants | `REQUEST_BODY_LIMIT_BYTES = 64 MB` and `REQUEST_BODY_LIMIT_LARGE_BYTES = 256 MB` in `src/http.rs`. Replaced 3 bare literals; `EXPERT_BATCH_BODY_LIMIT` in `routes/mod.rs` now references the same const. |
| **Q1.5** `JSON_CONTENT_TYPE` const | Added to `src/http.rs` next to `BINARY_FFN_CONTENT_TYPE`. Replaced 3 bare `"application/json"` literals across walk_ffn / embed / expert. |
| **Q1.6** Typed `DEFAULT_*` consts | `DEFAULT_PORT`, `DEFAULT_HOST`, `DEFAULT_HNSW_EF_SEARCH`, `DEFAULT_MAX_CONCURRENT`, `DEFAULT_DESCRIBE_CACHE_TTL_SECS`, `DEFAULT_LOG_LEVEL`, `DEFAULT_SESSION_TTL_SECS`, etc. Moved into `bootstrap.rs` (alongside the new `Cli` struct from Q1.8); `clap` now uses `default_value_t = ...`. `SessionManager::new` references the same `DEFAULT_SESSION_TTL_SECS` instead of re-encoding `3600`. |
| **Q1.7** `announce.rs` reconnect/heartbeat consts | `RECONNECT_INITIAL_BACKOFF` / `RECONNECT_MAX_BACKOFF` / `HEARTBEAT_INTERVAL` lifted to module consts; the previous `Duration::from_secs(1) / 60 / 10` magic numbers are gone. |
| **Q1.8** Reduce `main.rs::main` (656 LOC → 26 LOC) | Moved `Cli` struct + `pub async fn serve(cli: Cli)` into `bootstrap.rs`. `main.rs` is now: parse Cli, install tracing, call `bootstrap::serve(cli).await`. Boot orchestration (vindex loading, warmups, listener+TLS+UDS, gRPC, grid announce) is callable from anywhere that wants to drive the server without going through `clap::Parser::parse_from`. |
| **Q1.10** stream.rs reduction | **Deferred** — see P1: Active. Bundling with N0.1 SSE infrastructure when that lands. |
| Tests | 126 → **131 lib tests** (4 new for `wire::has_content_type`, 1 for `env_flags::names_are_larql_prefixed_and_unique`); 37 integration tests unchanged; ~580 tests across lib + integration, 0 failures. |
| Clippy | `cargo clippy -p larql-server --tests --no-deps -- -D warnings` clean. |
| `cargo fmt -p larql-server -- --check` | Clean. |

LOC delta (per-file):

| File | Before | After |
|---|---|---|
| `main.rs` | 656 | **26** |
| `bootstrap.rs` | 464 | 1073 (Cli + serve moved in) |
| `routes/expert.rs` | 1044 | (deleted) |
| `routes/expert/mod.rs` | — | 90 |
| `routes/expert/single.rs` | — | 155 |
| `routes/expert/batch_legacy.rs` | — | 105 |
| `routes/expert/layer_batch.rs` | — | 226 |
| `routes/expert/cpu.rs` | — | 195 |
| `routes/expert/metal.rs` | — | 204 |
| `routes/expert/warmup.rs` | — | 140 |
| `env_flags.rs` (new) | — | 122 |
| `wire.rs` (new) | — | 64 |

The bulk of the `bootstrap.rs` size growth is the Cli struct (~200 LOC of
clap doc-comments + `#[arg]` attributes) and the `serve` function body
that used to live in `main`. The orchestration is unchanged; only its
location moved.

### 2026-05-01 — HTTP CPU-path optimisations + UDS transport + layer-batch wire

End-to-end ~17.7 → ~19.7 tok/s on Gemma 4 26B-A4B (M3 Max, single local
gRPC shard, 100-token poem). Per-call HTTP overhead dropped from ~660 µs
to ~460 µs on gRPC streaming, ~510 µs on UDS, ~660 µs on TCP HTTP (now
with TCP_NODELAY). All optimisations preserve bit-exact semantics
(verified by output equivalence on the same prompts).

| Item | Outcome |
|---|---|
| **`POST /v1/experts/layer-batch`** new endpoint | One residual + K (expert_id, weight) pairs → one router-weighted-sum response. Replaces the K-residual-copies legacy `/v1/expert/batch` for the common-case `forward_moe`. Saves ~2.6 MB/token of redundant wire data + K-1 redundant `pre_experts_norm` + Q8_K quants on the server. |
| **`POST /v1/experts/layer-batch-f16`** new endpoint | f16 variant — halves wire bytes (5.5 KB request + response). Opt-in via `LARQL_MOE_WIRE_F16=1` for LAN deployments. f16 conversion CPU cost (~9 µs/call) cancels the wire saving on loopback; expected +3-5% gain on 1 Gbps Ethernet. |
| **Unix Domain Socket transport** (`--uds-path`, `unix://` URL) | Hand-rolled HTTP/1.1 over `UnixStream` (no new dep). Saves ~150 µs/call on loopback (~3% end-to-end). Persistent stream behind a `Mutex`, lazy reconnect on disconnect. Same wire format as TCP HTTP, so f16 + layer-batch semantics carry through unchanged. |
| **TCP_NODELAY on accepted connections** | `axum::serve::ListenerExt::tap_io` hook calls `set_nodelay(true)` per accept. Defensive against tail-packet stalls (40-200 ms on Linux/BSD delayed ACK) on real LAN; within noise on loopback. |
| **gRPC SPLIT default-on for gRPC shards** | Streaming fire/collect overlap now default for `grpc://` shards. Reliably ~12% steady-state win on M3 Max loopback (re-measured 19.5 vs 17.7 tok/s, alternating-cooled). The historical "20 → 4 tok/s catastrophic regression" warning predates the Metal MoE accuracy fix and the predispatch refactor; under thermal pressure both unary + SPLIT regress similarly, but stable-state SPLIT wins. Set `LARQL_MOE_NO_SPLIT=1` to opt out. |
| Per-call timing instrumentation | `LARQL_HTTP_TIMING=1` (server: decode / spawn_overhead / compute / encode µs; client: encode / send_total / recv_body / decode µs). `LARQL_MOE_TIMING=1` (per-token: per-layer route+fire / collect / server compute estimate / network estimate). Used for the diagnostic round that found `__powisf2` libcall in the f16 decode hot path (now bit-manipulated). |
| Test suite restored | 7+ test files had `LoadedModel { ... }` literals missing the `unit_filter` field added recently — all 9 LoadedModel literal sites in tests/ + tests/common/ patched. Test count went from 119 lib-only (broken integration tests) to **494 total across lib + 14 integration test files, all green**. |
| README + docs updated | `README.md` rewrite: new headline mentioning MoE grid as first-class use case, full env-var reference table, refreshed CLI Options with `--uds-path`/`--units`, rewritten "Remote MoE shard topology" recipe with current numbers, new `/v1/experts/layer-batch[-f16]` API section, accurate Crate Structure (28 source files vs the 16 the doc previously listed). `docs/server-spec.md`: §4.5 Remote MoE Expert Endpoints added, §13.4 dropped "planned" status, §10.2 fly.io references `F-FLY`. |
| `bench_expert_server` re-validated | Refreshed numbers in the Live perf snapshot section above. `cpu_moe_forward` floor 0.10 → 0.37 ms (the 0.10 was a buggy measurement on empty buffers — see prior compute ROADMAP). `forward_moe` warm 1.91 → 0.80 ms. 30-layer sweep 56 → 24.8 ms. RSS unchanged at ~10.5 GB. |

Tried-but-reverted (kept in source for future hardware where the trade
may flip):
- `tokio::task::block_in_place` instead of `spawn_blocking` — server-side
  faster (no transition cost) but tokio kept spawning replacement OS
  workers when every request blocked, regressing sweep ~0.3 ms.
- f16 wire as default — within noise on loopback (CPU conversion cancels
  wire saving); kept as opt-in for LAN.

### 2026-05-01 (continued) — larql-server review pass

Same calendar day, separate session. Audit + fixes across the entire
larql-server crate to land a clean baseline alongside the perf work.

| Item | Outcome |
|---|---|
| Test suite restored | 7+ stale `LoadedModel` test fixtures + 1 stale `PatchOp` example fixture missing recently-added struct fields. All 9 LoadedModel literal sites + 1 PatchOp site patched. **Test count went 119 lib-only → 501 across lib + 14 integration files; all green.** |
| `bench_expert_server` extended | New `--uds` and `--wire f32\|f16` flags. Spawns server bound to both TCP and UDS so the bench can A/B per-call cost. Confirmed UDS gives ~10% loopback win (0.82 → 0.74 ms `forward_moe` warm); f16 is a clear LOSS on loopback (1.05 ms — CPU conversion dominates) but expected to win on LAN. |
| README rewrite | Added env-var reference table, `/v1/experts/layer-batch[-f16]` API section, "Remote MoE shard topology" recipe with current numbers, accurate Crate Structure (28 source files vs the 16 the doc previously listed), "What's coming" section pointing to N0..N6 + F-FLY. ~880 → ~1110 LOC. |
| `docs/server-spec.md` updated | §3 CLI flags get `--uds-path` / `--units` / `--warmup-walk-ffn` / env-var section. New §4.5 Remote MoE Expert Endpoints (full layer-batch + f16 + transport coverage). §13.4 dropped "planned" status. §10.2 fly.io references `F-FLY`. |
| ROADMAP additions | New "Great new functionality" section (N0..N6) at the top — N0 is OpenAI API compatibility (chat completions + completions + responses + embeddings + models), highest-leverage item. F-FLY at top of P0: Active. F0 status updated (server path correct, local in-process TBD). Q1 (code-quality review) added at P1 with 10 sub-items targeting modularity + magic literals. |
| `cargo clippy -p larql-server --tests --no-deps -- -D warnings` | Was failing on 6 errors (manual `is_multiple_of`, `let_unit_value`, dead env-var unpacks, `path_used` unused initial assignment). All fixed. Server-only clippy now clean. |
| `cargo fmt -p larql-server -- --check` | Clean. |
| Coverage | 69.24% line / 75.64% function via `cargo llvm-cov`. Slight regression from 74.2/81.2 baseline attributable to new code added without proportional tests; mitigated by adding `topology.rs` tests (3) + `routes/expert.rs` `layer_batch_wire_tests` mod (4). |
| Code-quality findings catalogued | New Q1 section in ROADMAP with 10 concrete items (Q1.1 split `routes/expert.rs` 1049 LOC, Q1.2 centralise env flags into `src/env_flags.rs`, etc.) — all with file:line references and effort estimates. Total ~7-8 hours for the full sweep. |
| README + ROADMAP doublecheck | Fixed `gemma3-4b.vindex` references (file doesn't exist; replaced with `gemma3-4b-v2.vindex` which does), removed stale `ADR-009` reference (no such file), harmonised the two perf reference tables (Examples vs Recommended setups now reference each other), updated stale "2026-04-26" date stamp. |

### 2026-04-26 — Per-expert byte table refactor + `experts_packed.bin` removal

`MoeLayerWeights.experts_{gate_up,down}` migrated from `&[u8]` (monolith +
`expert_idx * stride` arithmetic in the compute path) to `Vec<&[u8]>`
(per-expert slice table). The CPU MoE consumer (`cpu_moe_forward` and
`run_single_expert{,_with_norm}`) now indexes by expert id directly, with
format dispatch (BF16 vs Q4_K) at the cache layer.

| Item | Outcome |
|---|---|
| `larql-compute` | `cpu/ops/moe/{cache,expert,forward,mod}.rs` and `pipeline.rs::MoeLayerWeights`. `cached_dequant(bytes, format, expected_floats)` dispatches BF16/Q4_K. `expert_byte_slice` deleted. Tests updated. 94/94 pass. |
| `larql-vindex` | `cpu/ops/q4_common.rs::dequantize_q4_k` lifted to module scope so the compute crate can dequant Q4_K without a `larql-models` dependency. |
| `larql-inference` | `build_moe_weights` builds per-expert tables from either `weights.get_layer_entry_bytes(...)` (per-layer Q4_K) or BF16 stride slicing (legacy). `QuantFormat` re-exported. |
| `larql-server` | `routes/expert.rs::run_expert` resolves per-expert bytes through whichever path the vindex provides; honours `expert_filter` ownership. `tests/test_expert_endpoint.rs` updated to slice synthetic monoliths into per-expert tables. 4/4 parity tests pass. |
| 26B-A4B vindex | `weight_manifest.json` stripped of `packed_bf16` rows for experts (60 → 421 entries). `experts_packed.bin` deleted (43 GB freed; vindex 58 → 16 GB). |
| Bench parity | `bench_expert_server` re-runs end-to-end against the per-layer-only vindex. `forward_moe` warm latency unchanged at 1.91 ms (was 1.93 ms when monolith was still on disk). 30-layer sweep at 56 ms (cold-page sweep on BF16 monolith was 866 ms). |

`bench_expert_server` and the parity tests both detect the format
automatically (`weights.has_per_layer_ffn()`); legacy BF16 vindexes still work
unchanged. Future MoE vindexes only emit per-layer files — the q4k extractor
at `format/weights/write_q4k/mod.rs` already does this.

### 2026-04-30 — gRPC grid: end-to-end accuracy

The grid produced semantically wrong text on Gemma 4 26B-A4B-it ("The capital
of France is **not specified in the text**…") despite each shard correctly
running its expert FFN. Root cause was on the **client** side
(`larql-inference::layer_graph::grid`) — chat-template handling, detokeniser,
EOS detection, and special-token suppression — not the shard server. The
server work here was confirming the contract: shards return correct expert
outputs given the right top-K input. Documenting for future grid changes.

| Item | Notes |
|------|-------|
| Server shards verified correct | A 2-shard split (experts 0-63 on `:9081`, 64-127 on `:9082`) running against the unit manifest serves expert outputs that, when combined client-side with the proper detokenisation + EOS + special-token suppression + default system prompt, produce "**Paris**" as the answer |
| Shard contract: per-(layer, expert) ownership via `--units` | The `parse_unit_manifest` path is what the client's `--moe-units-manifest` resolves against; ownership is the strict source of truth and `forward_moe_seq` rejects layers/experts not owned by any shard |
| Decode throughput (loopback, M3 Max) | 2.3 tok/s end-to-end on the 26B-A4B with two shards in the same process — expected to climb meaningfully when shards run on separate hosts (less GPU contention with the client) |

### 2026-04-30 — Metal expert dispatch: 3.7× speedup found, blocked on kernel bug

`LARQL_MOE_TIMING=1` showed the grid bottleneck is **server compute = 95%** of token wall time (network = 2%, route+fire = 3%). Per layer: 8.36ms server / 0.18ms net. Each shard runs its 4 picked experts (gate + GELU + down) on CPU-rayon BLAS — that's where the time goes. Sub-arc:

| Item | Notes |
|------|-------|
| Bottleneck localised | CPU experts = 250ms/token (95%) on the loopback 2-shard setup. Network = 5ms (2%). The grid-side overhead is negligible — accelerating the shard's expert math is the only meaningful lever |
| `--features metal-experts` measured: **3.7× speedup** | Server with Metal expert dispatch: 264ms → 117ms per token, 2.3 tok/s → **9.4 tok/s** (preselected path → 11.2 tok/s). Significant — server compute drops from 250ms → 115ms |
| **Accuracy bug blocks shipping** | Metal expert kernel (`MetalBackend::run_experts_preselected_metal` and `_prestaged_metal`, both routes) produces numerically wrong outputs for Gemma 4 26B-A4B-it MoE shape (cos≈0.7 vs CPU, \|metal\|≈70% of \|cpu\|). End-to-end output: "**Paris**" via CPU vs "answer is in the context" via Metal. Same kernels are correct for dense FFN at inter=2560/10240/21504 — bug is specific to MoE inter=704 dispatch |
| Workaround: default to CPU even on metal-experts builds | `run_experts_metal_batch` now early-returns `None` unless `LARQL_USE_METAL_EXPERTS=1` is set. Shipping correctness over speed; the Metal path stays opt-in for kernel-debug runs |
| Diagnostic: `LARQL_METAL_VS_CPU_DEBUG=1` | Server-side per-call A/B compare in `run_experts_metal_batch` — runs both Metal and CPU on the same input, prints max\|Δ\|, \|metal\|, \|cpu\|, cos. Ready to use when someone digs into the kernel |
| See also | `larql-compute/ROADMAP.md` "Open: Metal MoE expert kernel — accuracy bug at inter=704" for the kernel-side investigation plan |

### 2026-04-26 — examples, synthetic benchmark, grid checks

| Item | Outcome |
|---|---|
| `server_demo` | Runs locally with synthetic data; fixed invalid probe-label JSON comma output and updated rate-limit text for `--trust-forwarded-for`. |
| `embed_demo` | Runs locally with synthetic embed/logits/token responses and binary-wire examples. |
| `server_bench --release` | Synthetic benchmark completed: `gate_knn` top-5 0.022 ms/op, 8-layer `walk` 0.203 ms/op, single-layer `walk-ffn` 0.032 ms/op, batched 8-layer `walk-ffn` 0.321 ms/op, describe simulation 0.298 ms/op, 512-token embed prefill 0.114 ms/op. |
| `bench_embed_server` | Example builds under `cargo check -p larql-server --examples`; execution requires a real vindex path. |
| Grid unit coverage | Added `GridState` tests for inclusive ranges, default single-model routing, least-loaded replica selection, deregistration, batched gap reporting, and status gaps. `cargo test -p larql-router` now runs 20 tests. |
| Docs | Updated server README examples/benchmarks/testing, router README validation, and router spec validation commands. |

### 2026-04-26 — coverage round-6 (embed + walk-ffn reachable gaps)

| Item | Outcome |
|---|---|
| `routes/embed.rs` modularity | Extracted binary embed/logits parse helpers and binary embed response encoder |
| `routes/embed.rs` coverage | **66.7% → 86.5% line**, **70.7% → 86.3% function** |
| `routes/walk_ffn.rs` coverage | **76.7% → 79.5% line**, **77.3% → 82.0% function** |
| Tests | 458 → **478** tests |
| Coverage | **71.9% → 74.2% line**, **78.9% → 81.2% function** |

### 2026-04-26 — modularity + coverage round-5

| Item | Outcome |
|---|---|
| Boot/loading modularity | Moved parse/discovery/vindex-load helpers out of `main.rs` into `bootstrap.rs`; binary now keeps CLI orchestration while library code is directly testable |
| `routes/stream.rs` | Extracted pure `stream_describe_messages`; describe stream behavior can be tested without a WebSocket client |
| `routes/infer.rs` | Extracted mode selection and prediction formatting helpers |
| `routes/explain.rs` | Extracted band mapping, probability/gate/attention rounding, prediction formatting, and lens formatting helpers |
| Clippy | Server-local clippy clean with `--no-deps`; full dependency-checking command is blocked by existing `larql-vindex` warnings |
| Coverage | **69.2% → 71.9% line**, **77.1% → 78.9% function** (458 tests) |

### 2026-04-26 — coverage round-4 (T2 reachable gaps)

| Item | Outcome |
|---|---|
| `embed_store.rs` | 25% → **98% line** with tiny f16 mmap fixtures and L1 cache behavior tests |
| `announce.rs` | 6% → **56% line** by extracting/test-covering announce, heartbeat, dropping, and bearer helpers |
| `main.rs` | 0% → **23% line** with binary unit tests for parse/discovery/serve-alias helpers |
| `routes/stream.rs` | 0% → **28% line** with pure WebSocket message shape builders |
| `routes/infer.rs`, `routes/explain.rs` | Default/request deserialization coverage added; full paths remain weight-gated |
| Coverage | 63.9% → **69.2% line**, 73.4% → **77.1% function** (430 → 458 tests) |

### 2026-04-26 — coverage round-3 (T2 partial) + magic strings round-2

| Item | Outcome |
|---|---|
| `test_grpc.rs` — 28 new gRPC handler tests | Direct method calls on `VindexGrpcService` — no network socket; health, stats, describe, walk, select, relations, walk_ffn, infer, stream_describe |
| `grpc.rs` coverage | 0% → **65%** (169 lines uncovered, all gated on real model weights or gRPC streaming) |
| Magic strings — `"probe"` | `PROBE_RELATION_SOURCE` constant in `band_utils.rs`; used in describe.rs, grpc.rs, stream.rs |
| Magic strings — `"ok"` | `HEALTH_STATUS_OK` constant; used in grpc.rs health handler |
| Magic strings — gRPC modes | `INFER_MODE_WALK/DENSE/COMPARE` applied to grpc.rs (was using bare strings) |
| Magic strings — WebSocket types | `WS_TYPE_ERROR/LAYER/DONE/PREDICTION/INFER_DONE` and `WS_CMD_DESCRIBE/INFER` in stream.rs |
| Coverage | 57.2% → **63.3% line**, 65.3% → **73.2% function** (402 → 430 tests) |

### 2026-04-26 — coverage round-2 (T1)

| Item | Outcome |
|---|---|
| `functional_tokenizer()` in common | WordLevel tokenizer (France→0, …) added to test infra; unblocks describe/walk/walk-ffn body paths |
| `test_http_full_routes.rs` | 39 new HTTP integration tests exercising full describe/walk/walk-ffn code paths |
| `test_unit_band_utils.rs` | 13 pure unit tests for `band_utils.rs` constants + helpers |
| Infer + ratelimit branches | `infer_disabled=false` model builder; ratelimit middleware axum tests |
| Coverage | 49.1% → **58.0% line**, 56.4% → **65.3% function** (345 → 402 tests) |

### 2026-04-26 — code quality round-1

| Item | Outcome |
|---|---|
| Modularity — deduplicate `session_id()` | 3 identical private fn definitions → 1 `pub fn extract_session_id` in `session.rs` |
| Modularity — `get_layer_bands()` / `filter_layers_by_band()` | 5 / 3 duplicated blocks → `src/band_utils.rs` |
| Modularity — `model_or_err()` | 25 repeated `ok_or_else(NotFound)` sites → `AppState::model_or_err()` |
| Modularity — `elapsed_ms()` | 20 repeated latency-rounding expressions → `src/state::elapsed_ms()` |
| Magic strings — band names | `"syntax"/"knowledge"/"output"/"all"` → `BAND_*` constants in `band_utils.rs` |
| Magic strings — infer modes | `"walk"/"dense"/"compare"` → `INFER_MODE_*` constants |
| Magic strings — insert modes | `"constellation"/"embedding"` → `INSERT_MODE_*` constants |
| Magic strings — patch names | `"unnamed"/"inline-patch"` → `PATCH_UNNAMED`/`PATCH_INLINE_NAME` constants |
| Magic strings — HTTP headers | `"x-session-id"` → `HEADER_SESSION_ID`; `"etag"/"cache-control"/"if-none-match"` → axum `header::*` |
| Test restructure | `test_api.rs` (2600 L) + `test_http.rs` (1400 L) → 10 focused files (100–350 L each) + `tests/common/mod.rs` |
| Coverage baseline | 39.7% → **49.1% line**, 41.6% → **56.4% function** (345 tests, 0 failures) |

### 2026-04-26 — perf round-1 (G1+G2+G3)

| Item | Outcome |
|---|---|
| G1 cold-start profile | Two-phase: 1.27 s lazy weight load + 17 ms/layer mmap page-in. Warm steady state 0.2–0.3 ms/layer. |
| G2 `/v1/warmup` + `--warmup-walk-ffn` | First walk-ffn 1247 ms → 12.6 ms (99×). Boot trades ~1.3 s + 3.2 GB pre-allocation. HTTP endpoint also exposed for live re-warm. |
| G3 self-assembling gRPC grid | Live-validated `--grid-port` + `--join`: auto-join, coverage tracking, graceful failure (clean HTTP 400 on uncovered layer), auto-recovery on rejoin. |

### 2026-04-26 — W2 retrofit + grid validation

| Item | Outcome |
|---|---|
| `--warmup-hnsw` flag | Eager-builds HNSW across owned layers at boot via `warmup_hnsw_all_layers()`. Reports correct owned-layer count under `--layers`. |
| Boot log: W2 status | `Down features Q4K: loaded (W2 — per-feature decode skips q4k_ffn_layer cache)` when `down_features_q4k.bin` is present. |
| `/v1/stats.q4k_ffn` field | `{cache_slots, cache_bytes, feature_major_down}` — operators can verify W2 active + cache empty in steady state. |
| `larql convert add-feature-major-down` | New CLI subcommand. Retrofits an existing Q4K vindex without re-quantising the rest. 30 layers / 152 MB / 1.12 s on Gemma 26B. Idempotent. |
| Live grid validation | 2-shard layer-range split (0-14 + 15-29) on real 26B vindex, full fan-out via router, 8-way concurrent stress, 0.2 ms warm per-layer, 5.9 ms full-30-layer fan-out. |

### Pre-2026-04-26 — foundations (already in place)

- HTTP API: `/v1/walk`, `/v1/walk-ffn`, `/v1/stats`, `/v1/health`,
  `/v1/infer`, `/v1/insert`, `/v1/expert/{layer}/{id}`, etc.
- `--layers START-END` shard slicing (mmap pages outside range stay
  paged out, RSS proportional to shard size).
- `--max-q4k-cache-layers` LRU bound on the legacy Q4K dequant cache.
- `--ffn-only` / `--embed-only` mode flags.
- gRPC self-assembling grid (`--grid-port` / `--join` / `--grid-key`).
- Bench rig daemon-aware (`larql-vindex` benches refuse if a server
  shares the host; override with `LARQL_BENCH_ALLOW_DAEMONS=1`).
