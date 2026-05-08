# LARQL Roadmap

Top-level plan. Per-crate detail lives in each crate's own `ROADMAP.md`.
This file tracks the demo narrative, the critical path, and cross-crate sequencing.

---

## Crate roadmaps

| Crate | Owns |
|---|---|
| [larql-compute](crates/larql-compute/ROADMAP.md) | Metal GPU kernels, MoE prefill, platform expansion |
| [larql-inference](crates/larql-inference/ROADMAP.md) | Forward pass, generation quality, KV engines |
| [larql-server](crates/larql-server/ROADMAP.md) | HTTP API, gRPC grid, remote expert protocol |
| [larql-router](crates/larql-router/ROADMAP.md) | Grid routing, self-balancing, QUIC transport |
| [larql-cli](crates/larql-cli/ROADMAP.md) | CLI UX, sampling flags, streaming display |
| [larql-lql](crates/larql-lql/ROADMAP.md) | LQL grammar, INSERT/SELECT/USE extensions |
| [larql-core](crates/larql-core/ROADMAP.md) | Graph data model, algorithms, serialization |
| [larql-vindex](crates/larql-vindex/ROADMAP.md) | Vindex format, storage, extraction |
| [larql-models](crates/larql-models/ROADMAP.md) | Architecture definitions, model loading |
| larql-boundary | Confidence-gated BOUNDARY ref codec; cold-context residual storage |

---

## Current state (2026-05-08)

- **~950 tests passing** across the workspace (server 216 lib + 725 integration, router 10+23), 0 build errors.
- **Primary CLI verbs** in place: `run`, `chat`, `pull`, `list`, `show`, `rm`, `link`, `serve`, `bench`.
- **Gemma 3 4B Metal**: **83–84 tok/s** (Ollama steady: 98.5–99.7). **Gap: 1.18×** (was 1.30× before the 2026-05-02 dispatch-geometry fix).
- **Gemma 4 26B A4B Metal**: **19.4 tok/s** (was 5.1 — bug-locked under the same dispatch-geometry mismatch; correct multilingual output now).
- **Grid (CPU MoE on remote shards)**: 18.3 tok/s 1-shard / 17.3 tok/s 2-shard local-loopback. Multi-host LAN/cross-region scaling unblocked.
- **Remote FFN (dense)**: `larql run --ffn URL` + `larql serve --ffn-only` wired end-to-end.
- **gRPC grid**: 2-shard self-assembling grid live-validated on 26B A4B.
- **4 KV-cache engines**: MarkovRS (287×), UnlimitedContext (254×), TurboQuant (4×), Apollo (20,000×) — all at ~95 tok/s on Gemma 3 4B Metal.
- **Wire format negotiation** (2026-05-07): f16 is now the default for all grid traffic (50% bandwidth reduction). i8 symmetric quantised residuals available opt-in (`LARQL_I8_WIRE=1`, 75% reduction). Content-type negotiation via `Accept` header; f32 fallback for non-grid clients.
- **Per-layer latency routing** (2026-05-07): `HeartbeatMsg.layer_stats` carries EMA avg_ms + p99_ms per layer; router routes to the server with lowest per-layer latency (falls back to requests_in_flight when no data yet).
- **WebSocket token streaming** (2026-05-07): `WS /v1/stream` now supports `{"type":"generate","prompt":"...","max_tokens":N}` command with per-token frames and cancel support. SSE streaming on `/v1/chat/completions` was already fully wired.
- **Criterion benchmarks** (2026-05-07): `make bench-wire` (wire codec encode/decode MB/s) and `make bench-routing` (route/heartbeat/rebuild ns/op). `larql-router` now has a library crate (`larql_router::grid`) for test/bench use.
- **Dynamic rebalancing** (2026-05-08): `rebalancer.rs` background task with configurable threshold (--rebalance-interval, --rebalance-threshold). Router detects sustained per-layer latency imbalance and sends `UnassignMsg` to the slow shard; server drains in-flight requests (up to 30s), sends `DroppingMsg`, and re-enters available pool. Real `requests_in_flight` counter wired into heartbeats via `RifGuard` in walk_ffn handler.
- **CI regression gate** (2026-05-08): `scripts/bench-grid-regress.sh` + `scripts/bench_compare.py` + `bench/baselines/`. First run auto-saves baseline; subsequent runs fail if tok/s drops >5% or p99 rises >10%.
- **Shannon arc closed** (2026-05-08): Exps 42–44 prove cross-entropy is a real wire format (Exp 42: 2.0 bits/char vs 6.3 gzip), residual stream is compressible (Exp 43: int8-clip3σ, 98.7% top-1, KL=2.0 nats), gate calibrated at threshold=2.16 (Exp 44: accept=68.9%, early-div=4.8%).
- **`larql-boundary` crate shipped** (2026-05-08): Phases 1–3 of BOUNDARY_REF_PROTOCOL. int8-clip3σ + bf16 codec, per-boundary confidence metadata, calibrated confidence gate. 100% function coverage, CI on Linux/Windows/macOS, 3 examples (encode_decode, gate_decision, accuracy). Phase 4 (server integration) not started.

---

## Demo narrative

### Act 1 — "The model is the database"
Run Gemma 3 4B or 4 26B locally. The vindex is the model; `larql run` queries it.
Show: latency, footprint, `larql walk` tracing a fact through layers.

**Status**: Works end-to-end. Needs chat-template + EOS fix so it doesn't loop.

### Act 2 — "The experts live elsewhere"
Split a MoE model across machines. Client holds attention weights; each shard
holds a subset of expert IDs. The forward pass fans out to shards per token.

**Status**: Server-side grid works. Missing: remote expert endpoints (`/v1/expert/*`),
`RemoteExpertBackend` client, chat-template-aware prompting.

### Act 3 — "Replace an expert"
Swap expert 42 at layer 18 for a custom one. Observe the model's behaviour change.

**Status**: Expert ID selection TBD. Requires Act 2 first.

---

## P0 — Mechanistic surface (lazarus parity)

Driver: replace the chuk-mlx engine in `chuk-mcp-lazarus` with larql. Lazarus
exposes ~77 inference-time MCP tools (capture, ablate, patch, steer, probe,
DLA, KV-surgery). Larql is currently strong on weight-level edits (MEMIT, KNN,
LQL) and weak on inference-time inspection/intervention. The 77 tools collapse
to one missing primitive: a **programmatic forward-hook system**. Once that
lands the rest is mostly Python wrappers.

| # | Item | Crate | Status |
|---|------|-------|--------|
| M1 | `LayerHook` trait + CPU plumbing (read + write) | larql-inference | shipped |
| M2 | `RecordHook`, `ZeroAblateHook`, `SteerHook`, `CompositeHook` | larql-inference | shipped |
| M3 | Activation patching (cross-prompt residual swap) | larql-inference | shipped |
| M4 | Full logit lens — `logit_lens_topk`, `track_token`, `track_race` | larql-inference | shipped |
| M5 | `KvCache::{get_layer, set_layer, clear_layer, clone_layer_from, clone_layer_position_range}` | larql-inference | shipped |
| M6 | Hooks during multi-token generation (`generate_cached_hooked` on CPU; Metal `generate` stays fast by design) | larql-inference | shipped |
| M7 | `W_E` / `W_U` + `embedding_neighbors` + `project_through_unembed` | larql-inference | shipped |
| M8 | pyo3 `PyWalkModel` mech-interp methods (capture / ablate / steer / patch / lens / generate_with_hooks) | larql-python | shipped |

Detail in `larql-inference/ROADMAP.md` § Mechanistic hooks (lazarus parity).

---

## P0 — Best-in-class mechanistic interpretability engine

Driver: make LARQL's executed mechanisms queryable, attributable, patchable,
and reproducible. This is the layer above lazarus parity: not just hooks, but
evidence-grade traces and causal operators over the actual vindex-backed
inference path.

| # | Item | Crate | Status |
|---|------|-------|--------|
| MI0 | Faithful residual DAG: TRACE uses the canonical layer runner and pins additive reconstruction | larql-inference | shipped |
| MI1 | Python `WalkModel.trace()` / `patch_activations()` use `WalkFfn` instead of dense fallback | larql-python + larql-inference | shipped |
| MI2 | Backend-parametric donor capture and activation patching | larql-inference | shipped |
| MI3 | Strict trace artifacts: complete ordered chains, exact file length, `TRACE SAVE` requires `POSITIONS ALL` | larql-inference + larql-lql | shipped |
| MI4 | Golden parity: TRACE final residual/logits match canonical forward; extend to WalkFfn, patched vindex, Q4K, MoE | larql-inference | partial — dense/custom backend pinned |
| MI5 | Rich attribution objects: attention-head writes, FFN feature activations, router/expert decisions, provenance | larql-inference + larql-python | planned |
| MI6 | Causal operators beyond residual replacement: head/feature/router/expert/KV patching | larql-inference + larql-python | planned |
| MI7 | Q4K/MoE trace and patch parity with explicit precision caveats | larql-inference + larql-vindex | planned |
| MI8 | Python experiment ergonomics: batched prompts, donor/recipient alignment, causal metrics, reproducibility metadata | larql-python | planned |

Near-term order: finish MI4 parity coverage, then add attribution records where
the forward path already exposes data, then expand patching operators one
mechanism at a time.

---

## P1 — Research stack promotion: OV/RD → engine primitives

Driver: make LARQL one of the strongest practical mechanistic
interpretability stacks by promoting reusable experiment plumbing into
stable engine APIs, while leaving fast-moving hypotheses in
`larql dev ov-rd` and Python artifact analysis.

| # | Item | Crate | Status |
|---|------|-------|--------|
| R1 | Promote Q4K per-layer tensor insertion/removal from `ov_rd` into `larql-inference::vindex` | larql-inference | shipped |
| R2 | Add Q4K hidden forward with `LayerHook`/intervention support | larql-inference | shipped |
| R3 | Add pre-W_O capture/replacement hook adapters so experiments stop manually driving full layer loops | larql-inference | shipped |
| R4 | Define a compact research trace artifact contract for prompt ids, tokens, layer inputs, pre-W_O rows, oracle codes, logits, and metrics | larql-inference + larql-cli | planned |
| R5 | Keep PQ/address/codebook experiments in `larql dev ov-rd`; move only stable runtime contracts into engines | larql-cli | ongoing |

Rule of thumb: engine code owns reusable capture/intervention/runtime
primitives; `ov_rd` owns experiment orchestration, PQ variants, address
probes, and report schemas until a runtime contract survives repeated
experiments.

---

## P1 — Grid transport, self-balancing & benchmarking

Driver: minimum latency across on-device/LAN/WAN; elastic scaling without
manual shard pre-loading; reproducible, architecture-agnostic performance
evidence. All work is model-family-neutral — no hardcoded layer counts,
hidden sizes, or architecture assumptions.

Spec: ADR-0009 (wire format), ADR-0010 (QUIC), ADR-0011 (self-balancing),
ADR-0012 (benchmarking).

| # | Item | Crates | Status |
|---|------|--------|--------|
| GT1 | f16 wire default for all grid traffic; `LARQL_F16_WIRE_DISABLE` opt-out; Accept header negotiation | larql-server + larql-inference | **shipped 2026-05-07** |
| GT2 | i8 symmetric quantised residuals on wire; `LARQL_I8_WIRE=1` opt-in; per-position scale | larql-server + larql-inference | **shipped 2026-05-07** |
| GT3 | `LayerLatency` in `HeartbeatMsg` (proto + EMA tracker in server + per-layer routing in router) | larql-router-protocol + larql-server + larql-router | **shipped 2026-05-07** |
| GT4 | WebSocket token streaming (`generate` cmd + cancel); SSE for `/v1/chat/completions` confirmed wired | larql-server | **shipped 2026-05-07** |
| GT5 | Mode B gap-fill: `AvailableMsg → AssignMsg → download → ReadyMsg`; new `shard_loader.rs` | larql-router + larql-server | planned |
| GT6 | Dynamic rebalancing: `UnassignMsg` drain protocol + `rebalancer.rs` background task | larql-router + larql-server | **shipped 2026-05-08** |
| GT7 | QUIC transport for grid (`quinn` feature-gated); 0-RTT reconnect; per-stream independence for expert fan-out | larql-router + larql-server | planned |
| GT8 | `larql bench --bench-grid / --wire / --transport / --concurrent / --output json`; arch-agnostic from vindex config | larql-cli | planned |
| GT9 | Criterion micro-benchmarks: `wire_codec.rs` (encode/decode MB/s) + `routing.rs` (route/heartbeat/rebuild ns/op) | larql-inference + larql-router | **shipped 2026-05-07** |
| GT10 | CI regression gate: `scripts/bench-grid-regress.sh` + `bench/baselines/` committed JSONs | scripts/ | **shipped 2026-05-08** |

**Implementation order** (each step is a shippable increment):
~~GT3~~ → ~~GT1~~ → ~~GT2~~ → ~~GT4~~ → ~~GT9~~ → ~~GT5~~ → ~~GT6~~ → ~~GT8~~ → ~~GT10~~ → GT7

---

## P0 — Interpretability truthfulness + commit semantics

Driver: make the current edit model honest before the demo, then earn the
stronger "INSERT commits into weights" story. Today default `INSERT MODE KNN`
is a retrieval overlay persisted in `knn_store.bin`; `COMPILE INTO VINDEX`
bakes compose/MEMIT overlays but carries that KNN sidecar forward. That is a
snapshot/package operation, not a mechanical commit of the journal into FFN
features.

| # | Item | Crate | Status |
|---|------|-------|--------|
| T1 | Tag KNN overrides visibly in `INFER`, `EXPLAIN INFER`, and `TRACE` as post-logits retrieval events, including the model's unoverridden top-1 | larql-lql + larql-inference | planned |
| T2 | Fix decomposed `TRACE` to route through the shared layer sequence, including PLE/layer-scalar deltas or equivalent captured intermediates | larql-inference | shipped |
| T3 | Make Python `WalkModel.trace()` use the vindex `WalkFfn`/patch overlay rather than dense `WeightFfn` | larql-python + larql-inference | shipped |
| T4 | Replace gate-KNN absolute-dot feature ranking in interpretability displays with post-activation magnitude, or filter ghost negative gates after activation | larql-vindex + larql-inference | planned |
| T5 | Fix L1 FFN cache activation capture: cache activations with outputs or bypass cache when activations are requested | larql-inference | planned |
| T6 | Rename residual-capture embedding-neighbor fields (`top_token`) or add separate true logit-lens fields | larql-inference + larql-models | planned |
| T7 | Pin TRACE evidence with final residual/logit parity tests across dense, custom backend, WalkFfn, patched vindex, Q4K, and MoE paths | larql-inference | partial |
| C1 | Add explicit compile modes: default commit/materialize semantics vs `SNAPSHOT` preserving `knn_store.bin` | larql-lql + larql-vindex | design |
| C2 | Implement KNN materialization by lowering retrieval entries into compose/MEMIT/FFN edits, then dropping or marking committed sidecar entries | larql-lql + larql-vindex + larql-inference | planned |
| C3 | Add acceptance tests: session KNN equivalence, trace conversion, and generalization beyond stored prompts | larql-lql + larql-inference | planned |

Acceptance target for materialization:

```text
INFER(session_with_knn, q) == INFER(materialized_vindex, q)
```

for affected canonical prompts, plus a stronger trace/generalization check:
session trace reports pending retrieval; materialized trace shows residual/FFN
evidence; nearby unstored prompts behave through the materialized edit rather
than through a lookup sidecar.

Until C1-C3 ship, video language should distinguish three mechanisms:
KNN journal/retrieval overlay, compose FFN overlay, and compiled/baked weights.

---

## P1 — Model architecture independence hardening

Driver: keep LARQL from becoming "Gemma-shaped with exceptions." The core
`ModelArchitecture` trait is the right boundary, but several production paths
still infer family from strings, pass scalar attention geometry through
per-layer pipelines, or advertise architectures whose extraction/inference
contracts are incomplete.

| # | Item | Crate | Status |
|---|------|-------|--------|
| AI1 | Gate supported architecture families by executable contracts: extraction, vindex weight writing, forward/decode, trace, and prompt rendering | larql-models + larql-vindex + larql-inference | planned |
| AI2 | Implement or explicitly reject MLA architectures in vindex writers and inference; DeepSeek is detected today but `mla_*` tensors are not consumed outside `larql-models` | larql-models + larql-vindex + larql-inference | planned |
| AI3 | Remove scalar attention-geometry fallbacks from backend decode APIs; allocate KV/cache/scratch from `FullPipelineLayer` per-layer shapes everywhere | larql-compute + larql-inference | planned |
| AI4 | Replace vector-only extraction's model-name family guesses with explicit metadata or validated architecture input | larql-vindex | planned |
| AI5 | Roll validated loading/detection through inference, extraction, CLI, and server entry points where missing config should fail fast | larql-models consumers | planned |
| AI6 | Harden vindex extraction/write paths with explicit capability gates, named manifest/tensor tags, and tests proving unsupported attention layouts fail before writing partial indexes | larql-vindex + larql-models | next |

Acceptance target: adding a new transformer architecture should require changes
inside `larql-models::architectures/*` and explicit capability decisions at
storage/forward boundaries, not incidental string matches or hidden Gemma/Llama
defaults in extraction and decode.

---

## Critical path (P0 — what blocks the demo)

Items in order. Each depends on the one above it.

| # | Item | Crate | Status |
|---|------|-------|--------|
| 1 | Chat template + EOS stop | larql-inference + larql-cli | not started |
| 2 | Token streaming | larql-inference + larql-cli | not started |
| 3 | **Per-layer FFN format** (`layers/`, GPU dispatch) Phase 2: pre-alloc buffers | larql-vindex + larql-compute | shipped — `MoeScratch` pre-allocates once per decode call; combined with the 2026-05-02 dispatch-geometry fix, 26B A4B Metal now runs at **19.4 tok/s** (was bug-locked at 5.1) |
| 4 | MoE-aware CPU forward pass (non-Metal fallback) | larql-inference | not started |
| 5 | Wire `RouterIndex` client-side | larql-inference | not started |
| 6 | `POST /v1/expert/{layer}/{expert_id}` | larql-server | not started |
| 7 | `POST /v1/expert/batch` | larql-server | not started |
| 8 | `--experts 0-31` flag on `larql serve` | larql-server | not started |
| 9 | `RemoteExpertBackend` client | larql-inference | not started |
| 10 | Reliability pass (timeouts, retries) | larql-server | not started |

Items 1–2 are needed for Act 1. Item 3's MoE performance gate landed
2026-05-02: 26B A4B Metal now runs at 19.4 tok/s (was 5.1, bug-locked
under the dispatch-geometry mismatch in `moe_dispatch.rs`).
`LARQL_SKIP_MOE=1` ceiling 56.8 tok/s — remaining headroom is real
expert-dispatch work, not allocation. Items 4–10 are needed for Act 2. See
`larql-vindex/ROADMAP.md P0` and `larql-server/ROADMAP.md` (F-COLLECT,
F-LOCAL-MOE, G-SCALE) for the next levers.

---

## P1 — Boundary refs and cold-context storage

Driver: replace unbounded KV retention in long-context and multi-host scenarios
with compact, contract-bearing residual checkpoints. Hot KV window stays bounded;
older context is represented as 2564-byte compressed residual frames.

```
KV for the present. Residual boundaries for memory.
```

Foundation: `crates/larql-boundary/` (Phases 1–3 shipped).  
Protocol spec: `experiments/43_residual_stream_codec/BOUNDARY_REF_PROTOCOL.md`.  
Calibration data: `experiments/44_boundary_gate_calibration/`.

The existing `BoundaryStore` in `larql-inference/src/trace/boundary.rs` stores raw
bf16 residuals. `larql-boundary` adds the 2× compressed path on top of it. Phase 4
connects them to the running server.

| # | Item | Crate | Status |
|---|------|-------|--------|
| BR1 | int8-clip3σ + bf16 codec (Phase 1) | larql-boundary | **shipped** |
| BR2 | Per-boundary metadata + calibrated gate at threshold=2.16 (Phase 2–3) | larql-boundary | **shipped** |
| BR3 | BoundaryFrame wire format + A/B/C/D/E contract taxonomy | larql-boundary | **shipped** |
| BR4 | Phase 4: bounded KV eviction + durability-first capture (Option A) | larql-server + larql-inference | not started |
| BR5 | Phase 4: boundary archive (disk/remote) + restore path | larql-server + larql-inference | not started |
| BR6 | Phase 5: boundary frames over gRPC grid (protobuf schema defined) | larql-router + larql-server | not started |
| BR7 | Track B: per-channel codec (int4 + outlier side-channel, ≤1024 bytes) | larql-boundary | not started |
| BR8 | Gate calibration n≥300 to tighten 95% CI below 1.6%–10.7% | experiments/44 | not started |

**What D-@high actually contracts:** first ~5 continuation tokens safe at 4.8%
early-div (95% CI 1.6%–10.7%, n=62). Total 20-token divergence is ~20% regardless
of threshold — cascade compounds past step 5. Use for boundary-to-fresh-decode; not
for long uninterrupted continuation. See BOUNDARY_REF_PROTOCOL §6.

**Immediate unblocking item:** BR4 (Phase 4 server integration). The eviction
ordering decision (durability-first Option A: capture → gate → fsync → evict KV)
is specified in the protocol; implementation in `larql-server` can start from it
directly.

---

## P1 — Generation UX (parallel to critical path)

Details in `larql-inference/ROADMAP.md` and `larql-cli/ROADMAP.md`.

- Sampling: `--temperature`, `--top-p`, `--top-k`, `--repetition-penalty`
- Multi-turn state: running KV across `larql chat` turns
- Long context: `--max-context N`, dynamic KV buffer growth
- OpenAI-compatible `/v1/chat/completions` (after streaming lands)
- Auto-extract on `larql run hf://owner/name`
- Gemma 3 4B regression smoke test (gate on `CI_INTEGRATION=1`)

---

## P2 — Film checklist

- [ ] Confirm Gemma 4 26B A4B public config (expert count, top-K, active-param figure, GQA ratio). Replace every `~` in `docs/demo-script-gemma4-moe.md`.
- [ ] Measure real footprint + latency on `google/gemma-4-31b-it` for Act 1.
- [ ] Reliability pass on `RemoteWalkBackend` (timeouts, retries, partial shard outage).
- [ ] `RemoteExpertBackend` same reliability pass.
- [ ] Decide repo-public date. `cargo install larql-cli && larql serve` must be live the week the video drops.
- [ ] Pick expert IDs for the Act 3 swap shot — one that fires on medical prompts, one that doesn't.

---

## Loose ends (shipped features with open follow-ups)

| Item | Crate | Detail |
|---|---|---|
| `KernelHandle` spread to 9 remaining tiled shaders | larql-compute | Mechanical, same pattern as q4_matvec_v4 |
| `dispatch_full_pipeline` 30+ params | larql-compute | Bundle into `FullPipelineRefs<'_>` context |
| `QuantFormat` match spread (14 files) | larql-compute | Introduce `FormatRoute` enum |
| `ProfileTimings` producer | larql-compute | Wire commit/wait boundaries into decode_token |
| Benches in CI | larql-compute | GHA workflow written, needs trigger merged |
| `--compact` loader for non-MoE models | larql-vindex | `WeightFfn::forward` panics on compact vindex |
| MoE compact mode | larql-vindex | Blocked on per-expert feature-major files |
| Fix `dispatch_full_pipeline` layer_scalar (dense) | larql-compute | Non-urgent: Gemma 3 4B has scalar=0 |
| Cross-vindex dedup (tokenizer, down_meta) | larql-vindex | Low priority, ~200 MB duplicated at 7 vindexes |
