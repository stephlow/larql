# LARQL Roadmap

Top-level plan. Per-crate detail lives in each crate's own `ROADMAP.md`.
This file tracks the demo narrative, the critical path, and cross-crate sequencing.

---

## Engine purpose (load-bearing — read first)

### The ultimate aim

> **Serve the largest models at blazing speed on consumer hardware, with as little GPU as possible — ideally eventually none.**

Frontier-scale models (100B–1T+ params) are physically incompatible with
consumer hardware under naïve dense matmul: a 671B Q4 model touches
~336 GB per forward pass; consumer DDR5 is ~50 GB/s; that's 6.7 sec/token.
The bandwidth wall cannot be beaten by faster compute. The *only* path
through is **touching fewer weights per token** — sparse retrieval over a
queryable weight database. Vindex was always for this.

Every invention in the codebase serves this aim:

| Invention | Role |
|---|---|
| Vindex (model-as-database) | Sparse access to weights, not dense matmul |
| LQL | Address language for sparse retrieval |
| WalkFfn (gate KNN → down lookup) | The actual sparse-FFN inference path |
| MoE expert grid (gRPC self-assembling) | Distribute models that exceed one machine across consumer machines |
| Layer sharding (`--layers`, `--shards`) | Same, by layer |
| Exp 26 (FP4 native-friendly) | 2× memory shrink without QAT (Gemma 3 4B proven) |
| Exp 27 (hash routing top-2048 mask) | 5× fewer FFN weights touched at KL=0.03 |
| MEMIT / COMPOSE / AOT | Compile programs into smaller weight footprints |
| WASM-in-FFN | Replace heavy kernels with cheap programs where the math allows |
| Boundary refs / residual codec | Compress KV for long context on bandwidth-bound hardware |
| Shannon arc (1 bit/char on Frankenstein) | Theoretical compression ceiling — how far this can go |
| Mech-interp surface (M1–M8) | Discover *which* weights actually do the work; rest stays on disk |
| Cross-arch coverage | The technique stack must generalise |

Combined effect (rough math, conservative): hash routing 5× × FP4 2× × KV
compression 10× = **100× effective bandwidth reduction** on the right
corpus. 670 GB model → 6.7 GB-equivalent traffic → ~134 ms/token on
consumer DDR5. That's blazing.

### Two permanent tracks

The aim demands both competitive performance *now* and progress toward
GPU-free *eventually*. These are co-equal tracks, neither sacrifices to
the other:

1. **GPU track** — maintains competitive baseline against ollama / vLLM /
   llama.cpp on Metal (and eventually CUDA/ROCm if substrate-relevant
   experiments demand them). Permanent. Never demoted in favour of CPU
   work. Without this, every claim measured on the engine fails the
   credibility threshold below.

2. **CPU track** — drives toward "blazing big models on consumer hardware
   without GPU." The ultimate aim. Built **in addition to**, not instead
   of, the GPU track.

**Architecture rule that makes the dual-track tractable**: vindex /
WalkFfn / sparse retrieval is the shared invention. Only kernels differ.
No GPU-only paths in the core design. Every technique developed on one
track must have a path to the other, or be architected
device-agnostically from the start (the verify-loop in MTP2 is a current
example: device-agnostic decode with device-specific kernels under it).

### Why "research substrate" framing is the means, not the end

LARQL **is** a research substrate — but substrate-for-its-own-sake isn't
the goal. The substrate exists because the techniques that make the
ultimate aim possible (sparse retrieval, hash routing, FP4, KV
compression, expert sharding, AOT compilation, boundary refs) have to
be developed *somewhere*. LARQL is that somewhere.

This means:

- Adoption, OpenAI-API ergonomics, multi-tenant batched serving, MCP
  ergonomics, and other "production engine" concerns are out of scope
  **except** where they accelerate experiments or affect measurement
  credibility.
- LARQL is not a production inference engine and will not become one in
  the *commercial* sense. But it must operate at production-engine
  baseline performance on its leading device class — otherwise the
  techniques developed on it can't be credibly compared against
  state-of-the-art.

### Baseline-credibility threshold (acceptance criterion)

> LARQL must be within **10% of llama.cpp / ollama** on the matching
> model + quantisation + context-length configuration **on the device
> class the claim is being made on**, before any *"+N% from technique X"*
> claim is published. CPU technique → CPU baseline. GPU technique → GPU
> baseline.

Current state (2026-05-09):

| Track | Configuration | LARQL | State-of-the-art | Gap | Threshold? |
|---|---|---|---|---|---|
| **GPU (Metal)** | Gemma 3 4B decode | 88 tok/s | ollama ~103 | 17% behind | over (defensible-with-caveat) |
| **GPU (Metal)** | Gemma 3 4B prefill (340 tok) | per-pos matvec | gemm | 14× behind | far over |
| **GPU (Metal)** | Gemma 4 + MTP (when adopted) | 88 tok/s no-MTP | ~225 with MTP | ~2.6× behind | far over |
| **CPU** | Gemma 3 4B decode | not measured | llama.cpp CPU baseline | unknown | not measurable yet |
| **CPU** | Gemma 4 26B-A4B decode | currently grid 18.3 tok/s | unknown | unknown | not measurable yet |

Items the threshold makes load-bearing (not optional) on the **GPU track**:
- **D-ATTN-MTG** — flash attention; without it, attention-mechanism deltas are muddied by missing baseline.
- **D-PREFILL-MM2** — `simdgroup_matrix` matmul; until landed, prefill claims fail the threshold.
- **D-METAL-PLE** — without it, every Gemma 4 E2B experiment runs CPU-fallback and any delta is unattributable.
- **MTP1–MTP6** — Gemma 4 MTP drafters are now part of the state-of-the-art baseline (Ollama supports them).
- **AI1–AI6** — cross-arch deltas need clean arch boundaries.
- **Coverage → 90%** — measurement integrity needs correctness trust.

Items the threshold makes load-bearing on the **CPU track** (see new
"P0 — CPU path to blazing" section below):
- Critical-path #4 — CPU MoE forward pass.
- WalkFfn as primary CPU decode path.
- Hash-routed FFN (exp 27 → product).
- FP4 productisation (exp 26 → product).
- mmap'd vindex with lazy disk-resident edges.
- AMX / AVX-512 / Apple AMX kernels.
- KV compression as default for long context.
- BR4 (boundary refs Phase 4).

Items the threshold makes **explicitly out of scope** (both tracks):
- **CB1, CB2** (continuous batching, PagedAttention) — concurrency-throughput, not single-stream baseline.
- **MCP1** (MCP server) — UX, doesn't change measurement.
- **TM1** (thinking-mode toggle) — UX, doesn't change measurement.
- OpenAI API compatibility beyond what experiments call.

See `docs/positioning.md` for the full framing and competitor diff.

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

## Current state (2026-05-09)

- **~950 tests passing** across the workspace (server 216 lib + 725 integration, router 10+23), 0 build errors.
- **Primary CLI verbs** in place: `run`, `chat`, `pull`, `list`, `show`, `rm`, `link`, `serve`, `bench`.
- **Gemma 3 4B Metal**: **88 tok/s** (Ollama steady: ~103). **Gap: 1.17×** (was 1.18× pre QKV defuse, 1.30× pre 2026-05-02 dispatch-geometry fix). **Acceptance criterion (~85 tok/s, 1.16×) met.**
- **Gemma 4 26B A4B Metal**: **19.4 tok/s** (was 5.1 — bug-locked under the same dispatch-geometry mismatch; correct multilingual output now).
- **Cross-arch coverage validated** (2026-05-09): Gemma 3, Gemma 4 31B dense, Llama 2 7B, Mistral 7B all dispatch correctly through Metal. Gemma 4 E2B falls back to CPU (deliberate — Metal doesn't yet implement Per-Layer Embeddings; diagnosed and tracked as D-METAL-PLE).
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
- **QKV defuse + cleanup pass** (2026-05-09): default flipped from fused `q4k_q6k_qkv_proj_normed` to separate `rms_norm` + non-fused `q4k_q6k_qkv_proj` (+1.6–1.8 tok/s on Gemma 3 4B, +0.4 tok/s on Gemma 4 26B A4B post-thermal-cooldown cross-arch validation, ADR-016). Cross-arch bench captured for 4 model families. Shader inventory survey (47 shaders) + retention rationale doc-blocks added to opt-in shaders. New ADRs: [017 — shader retention under model agnosticity](crates/larql-compute/docs/adr/017-shader-retention-model-agnosticity.md), [018 — architecture → shader routing](crates/larql-compute/docs/adr/018-architecture-shader-routing.md). New docs: [shader-inventory](crates/larql-compute/docs/shader-inventory.md), [architecture-shader-map](crates/larql-compute/docs/architecture-shader-map.md), [llama-cpp-comparison](crates/larql-compute/docs/llama-cpp-comparison.md). One verifiable orphan deleted (`q4k_qkv_proj_v2`).
- **`make bench-cross-arch` shipped** (2026-05-09): runs `larql bench` across the model matrix (Gemma 3 4B, Gemma 4 31B dense, Gemma 4 26B A4B MoE, Llama 2 7B, Mistral 7B). `--save-baseline` / `--compare` modes; `bench/baselines/cross-arch/`. Operationalises ADR-017 model-agnosticity check; multi-arch sweep surfaces thermal artifacts as "every arch regresses simultaneously." Run on a cool machine before saving baselines.
- **D-RMS-FUSE Phase 1 implemented + falsified end-to-end** (2026-05-09): fused post-FFN `residual_add` + next-layer input rms_norm via `residual_norm_store` for the non-Gemma path. Bit-identical parity across Llama 2 7B, Mistral 7B, Gemma 3 4B (Gemma untouched — already triple-fused). End-to-end null vs drift on Llama 2 / Mistral. Kept opt-in `LARQL_FUSED_PRELAYER_NORM=1` per ADR-017 retention. Predicted ~0.2 ms/tok savings collapsed to zero — ADR-015 magnitude-compression at the extreme. Lesson: dispatch-overhead estimates (~7 µs/dispatch) over-predict savings when the kernel being skipped is also short.
- **Gemma 4 E2B 30× anomaly diagnosed** (2026-05-09): root cause = Per-Layer Embeddings (PLE) not implemented in Metal; `gpu.rs:372-374` deliberately routes E2B to CPU. Tracked as **D-METAL-PLE** (1-2 day Metal port of `forward/ple.rs`, 80-150× expected speedup for E2B; unlocks future PLE-using arches like Gemma 4 E4B).
- **larql-compute coverage audit + improvement** (2026-05-09): `cargo llvm-cov` reports **56.03% → 64.81% line coverage** (+8.78 pp; 2,575 newly-covered lines, 22.2% reduction in uncovered LoC). Three rounds: (1) deleted `metal/prefill.rs` (591 LoC of `#[allow(dead_code)]` orphan); (2) targeted tests on small helpers — `tg_width` math (qk_norm 0% → 23%), `scale_vector` dispatch (layer_scalar 12% → 97%), `residual_norm_store` shader parity for D-RMS-FUSE; (3) synthetic end-to-end Metal decode tests (`tests/test_metal_decode_synthetic.rs`, NEW) covering Llama-style + Gemma-3-style + D-RMS-FUSE off-vs-on parity, which lifted `decode/mod.rs` 7% → 61%, `encode_attn` 0% → 46%, `encode_post_ffn` 0% → 83%, `encode_qkv` 0% → 30%, `encode_ffn` 0% → 23%. Coverage policy (`coverage-policy.json`) targets 90% per-file / 93.5% total — current is below but no longer a wide gulf. Largest remaining gaps: `metal/trait_impl/decode.rs` (627 LoC at 21% — MoE / split-profile trait methods), `metal/decode/encode_ffn.rs` (1008 LoC at 23% — Q4_KF / MoE branches), `metal/diag/*.rs` (~3000 LoC at 0% — diagnostic / dev-only).
- **Positioning vs ollama / vLLM / llama.cpp documented** (2026-05-09): [docs/positioning.md](docs/positioning.md). Three-category framing (local single-user / batched serving / research+edit); feature matrix; per-competitor gap analysis; surfaces missing items now tracked under P2 § "Competitive parity" below.
- **Google released Gemma 4 MTP drafters** (2026-05-05, 4 days ago): `google/gemma-4-{E2B,E4B,26B-A4B,31B}-it-assistant` — every Gemma 4 variant LARQL supports. 0.4B BF16 ~4-layer drafter for the 26B-A4B target. Architecture: shared input embeddings + shared KV cache + target last-layer activations concatenated with token embeddings then down-projected to drafter dimension. Measured **2.2× decode speedup on Apple Silicon at speculative batch 4–8** (Google blog), up to 3× generally. Apache 2.0 / CC-BY-4.0. Supported engines: HF Transformers, MLX, vLLM, SGLang, **Ollama**, LiteRT-LM (notably not llama.cpp). Competitive implication: the LARQL gap on Gemma 4 widens from 1.17× to ~2.6× as users adopt MTP on Ollama. Red Hat AI also released an EAGLE-3 speculator for `gemma-4-26B-A4B-it` (0.9B drafter). MTP1 promoted from P2 to **P1** — see new section below.

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
| 4 | MoE-aware CPU forward pass (non-Metal fallback) | larql-inference | not started — **promoted to P0 of the CPU track as C1; see "P0 — CPU path to blazing"** |
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

## P0 — CPU path to blazing (the ultimate-aim track)

Driver: the ultimate aim ("largest models at blazing speed on consumer
hardware, ideally without GPU") demands a permanent CPU track in
parallel with the GPU competitive-baseline track. CPU work is built
**in addition to** Metal work, not instead of it. Every item here is
either device-agnostic by construction (sparse retrieval) or has a
matched GPU twin (so the technique stack stays portable).

The bandwidth math is the gating constraint: 50 GB/s consumer DDR5
means a 671B Q4 model is 6.7 sec/token under naïve dense matmul.
Combined sparse-retrieval techniques (hash routing 5× × FP4 2× × KV
compression 10× = ~100×) make this ~134 ms/token — the actual
"blazing on consumer hardware" target.

| # | Item | Crate | Status | Notes |
|---|------|-------|--------|-------|
| C1 | Critical-path #4 — MoE-aware CPU forward pass (non-Metal fallback) | larql-inference | not started | **Promoted from critical path #4 to P0 of this track**. Currently CPU MoE has no production path; everything routes through Metal or grid. Without C1, CPU track has no decode loop to measure. |
| C2 | WalkFfn as **primary** CPU decode path (not research-only mode) | larql-inference | partial — exists, not productionised | Currently `WeightFfn::forward` is the dense fallback; switch the default for vindex-loaded models to `WalkFfn`. Bench numbers required. Cross-references CPU MoE work in C1. |
| C3 | Hash-routed FFN (exp 27 → product) — top-k mask on gate scores, ~5× bandwidth reduction at KL ≤0.05 | larql-inference + larql-vindex | research only | Exp 27 proved Gemma 3 4B L0 at top-2048/d_ffn (20% mask) gives KL=0.030. Productise as a vindex flag (`hash_route_topk: usize`) + per-layer threshold table; mask threshold per-layer (L0 vs late layers behave differently). Device-agnostic. |
| C4 | FP4 productisation (exp 26 → product) — native FP4 quantisation tier (`Q4_K → FP4`) | larql-vindex + larql-compute | research only | Exp 26 proved gemma3-4b-f16.vindex is 99.83% FP4-friendly per-feature without QAT (down is the tail at 99.65%). Add `Quantisation::FP4` variant; CPU-first kernel; Metal twin in larql-compute. ~2× memory shrink vs Q4_K. |
| C5 | mmap'd vindex with lazy disk-resident edges — only resident pages for active edges per token | larql-vindex + larql-inference | not started | Today vindex loads whole layer tensors into RAM. For models bigger than RAM, mmap the vindex file and let the OS page in only the gate-KNN-resolved edges. Pairs with C2 and C3: when only 20% of edges fire, only those pages are read. |
| C6 | AMX / AVX-512 / Apple AMX kernels for residual compute | larql-compute (CPU side) | partial — Accelerate BLAS, AMX through it | Current CPU path uses ndarray + Accelerate; promote to direct AMX intrinsics on Apple Silicon, AVX-512 on x86. Compute that *does* happen needs to be as good as it gets, since bandwidth is what's left over. |
| C7 | KV compression as **default** for long context (Apollo / MarkovRS / UnlimitedContext / TurboQuant) | larql-inference | shipped (4 engines) but opt-in | Currently `LARQL_KV_ENGINE` selects; promote one of the 4 engines as default for context > N (probably Apollo's 20,000× on the right corpus, MarkovRS at 287× as a more general fallback). Long context on CPU is unaffordable without this. |
| C8 | BR4 (Boundary refs Phase 4 — bounded KV eviction + durability-first capture) | larql-server + larql-inference | not started | See § "P1 — Boundary refs and cold-context storage" below. The CPU track makes BR4 load-bearing because long-context CPU inference can't keep raw KV in RAM. |
| C9 | Distributed-load-balancing for "model spans 4 consumer machines" | larql-router + larql-server | shipped (grid + rebalancer); needs production polish | gRPC self-assembling grid + dynamic rebalancing already shipped (2026-05-08). Productionise for the case where the largest model exceeds *any* available consumer machine individually — layer + expert sharding across 2–8 nodes on local network. |
| C10 | CPU bench harness — `larql bench --cpu` with per-stage breakdown matched against `llama.cpp -ngl 0` | larql-cli + bench/ | not started | Currently `larql bench` measures Metal-only. CPU-track baseline-credibility threshold can't be enforced without this. First test: Gemma 3 4B Q4_K on M3 Max CPU vs `llama.cpp -ngl 0`. Then Llama 2 7B + Mistral 7B for cross-arch CPU. |
| C11 | Architecture rule enforcement — CI check for "no GPU-only paths in core" | scripts/ + crate boundaries | not started | Static check: anything in `larql-inference` core (not `metal/`, not `cpu/`) must compile and pass tests with Metal feature off. Prevents the dual-track from drifting into Metal-locked code. |

**Implementation order**: C10 → C1 → C2 → C7 → C3 → C4 → C5 → C6 → C8 → C9 → C11.

C10 first because the threshold can't be enforced without measurement.
C1+C2+C7 give you a working CPU decode path with bearable long-context.
C3+C4+C5 are the bandwidth-shrinking techniques that make the ultimate
aim possible. C6 squeezes the compute that remains. C8+C9 unblock
genuinely large models. C11 prevents architectural drift.

**Acceptance**:
1. **Short-term** (C10 + C1 + C2): CPU Gemma 3 4B Q4_K decode within 10% of `llama.cpp -ngl 0` on M3 Max CPU.
2. **Medium-term** (+C3 + C4 + C7): CPU Gemma 3 4B FP4 + hash-routed decode at ≥2× the dense Q4_K CPU baseline.
3. **Long-term** (+C5 + C8 + C9): Gemma 4 26B-A4B (or larger) decode on a single 64GB consumer machine at ≥10 tok/s, no GPU.
4. **Ultimate** (full stack + frontier model): 100B-class model on consumer hardware at ≥5 tok/s, no GPU. Stretch goal: 671B-class via C9 multi-machine grid.

---

## P1 — Gemma 4 MTP drafter support (promoted from P2 2026-05-09)

Driver: Google released MTP drafters for every Gemma 4 variant on
**2026-05-05** (see Current state bullet above). Apple Silicon decode
speedup measured at **~2.2× at speculative batch 4–8**. Ollama already
supports MTP out-of-the-box; without this, the LARQL gap on Gemma 4
widens from 1.17× to ~2.6× as users adopt the drafters.

The drafters are the *exact* models LARQL is built around:
`google/gemma-4-{E2B,E4B,26B-A4B,31B}-it-assistant`. Apache 2.0 (code) +
CC-BY-4.0 (weights). The 26B-A4B drafter is 0.4B BF16 (~4 layers).

Architecture (from Google blog + ai.google.dev/gemma/docs/mtp):

1. Drafter shares the **input embedding table** with the target model.
2. Drafter consumes the target's **last-layer activations** at each
   accepted position, concatenates them with the next token embedding,
   and **down-projects to drafter dimension**.
3. Drafter and target share the **KV cache** (single physical cache,
   different read/write patterns).
4. E2B/E4B variants add an "Efficient Embedder" clustering layer that
   restricts drafter computation to selected token clusters.

| # | Item | Crate | Status | Notes |
|---|------|-------|--------|-------|
| MTP1 | `gemma-4-*-it-assistant` HF safetensors loader + `MtpDrafter` arch in larql-models | larql-models + larql-vindex | not started | New arch trait variant `MtpDrafter`; vindex extraction must handle the embedding-sharing reference (drafter doesn't carry its own embed table). Decide vindex layout: separate `*.assistant.vindex` sidecar vs unified `*.with-mtp.vindex` |
| MTP2 | Verify-loop decode (`generate_speculative`) — draft k tokens with drafter, verify k+1 with one target forward, accept longest matching prefix, rollback rejected positions | larql-inference | not started | Needs k as runtime param (default 4–8 per Google's batch-size sweet spot); reuse existing KV management; rollback logic touches `KvCache::clear_layer_position_range` (already shipped under M5) |
| MTP3 | Last-layer-activation feedback path — capture target's final residual at accepted positions, feed into drafter's input projection, down-project to drafter hidden | larql-inference + larql-compute | not started | New Metal kernel: concatenate-and-project (or two separate dispatches if fusion regresses, ADR-015 lesson). Activation buffer reuses M1–M4 capture infrastructure |
| MTP4 | Shared KV cache between target and drafter — single cache, separate write heads | larql-inference | not started | Drafter writes K/V using its own attention parameters but same cache slots; verification reads target's writes. May need `KvCache::view_for_arch` or similar |
| MTP5 | Efficient Embedder clustering layer (E2B/E4B only) | larql-models + larql-compute | not started | Restrict drafter computation to top-N token clusters; smaller-model-only optimisation; defer until MTP1–MTP4 prove out on 26B-A4B |
| MTP6 | `larql bench --mtp` — measure speculative-batch sweep (k=1..16), token-acceptance rate, end-to-end tok/s vs no-MTP baseline | larql-cli + bench/ | not started | Confirms the 2.2× number on M3 Max before promoting to default |
| SD1 | Generic speculative-decoding framework (n-gram draft / EAGLE / external draft model) — share MTP2's verify loop | larql-inference | not started | Broader machinery; promoted from P2 alongside MTP1. Build MTP2 first (concrete spec, immediate users); generalise to SD1 once the verify loop pattern is stable |
| SD2 | EAGLE-3 speculator support — Red Hat AI released `gemma-4-26B-A4B-it` EAGLE-3 (0.9B drafter); same machinery as MTP, different drafter loading | larql-models + larql-inference | not started | Validates SD1 generality on a non-Google drafter for a model we already support |

**Implementation order**: MTP1 → MTP2 → MTP3 → MTP4 → MTP6 (validate
2.2× number) → MTP5 (E2B/E4B optimisation) → SD1 (generalise) → SD2
(EAGLE-3 drop-in).

**Acceptance**: Gemma 4 26B-A4B Metal decode goes from 19.4 tok/s to
≥35 tok/s at speculative batch 4–8 with bit-identical token output vs
no-MTP baseline (Google guarantees identical-quality output; verify with
parity test across the existing cross-arch corpus).

**Why P1 not critical-path**: doesn't block the demo (Acts 1–3) — but
it *does* block any future tok/s comparison with Ollama on Gemma 4. If
the comparison story matters, MTP1–MTP4 should land before any public
benchmark refresh.

---

## P1 — Boundary refs and cold-context storage

Driver: replace unbounded KV retention in long-context and multi-host scenarios
with compact, contract-bearing residual checkpoints. Hot KV window stays bounded;
older context is represented as 2564-byte compressed residual frames.

```
KV for the present. Residual boundaries for memory.
```

Foundation: `crates/larql-boundary/` (Phases 1–3 shipped).
Protocol spec: `~/chris-source/chris-experiments/shannon/43_residual_stream_codec/BOUNDARY_REF_PROTOCOL.md`.
Calibration data: `~/chris-source/chris-experiments/shannon/44_boundary_gate_calibration/`.

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
| BR8 | Gate calibration n≥300 to tighten 95% CI below 1.6%–10.7% | ~/chris-source/chris-experiments/shannon/44_boundary_gate_calibration | not started |

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

## P2 — Competitive parity (positioning analysis 2026-05-09)

Driver: items surfaced by [docs/positioning.md](docs/positioning.md) that the
ollama / vLLM / llama.cpp comparison treats as table stakes but LARQL doesn't
yet ship.

**Re-evaluated 2026-05-09 under the substrate framing** (see "Engine purpose"
above). Each item is now scored by *"does this affect the credibility of
measured technique deltas, or accelerate experiments?"* Items that only serve
"becoming a production engine" are explicitly **dropped or deferred** — LARQL
will never be a production engine, so spending engineering on production-engine
features that don't tighten the experiment loop is scope creep.

| # | Item | Crate | Substrate verdict | Notes |
|---|------|-------|-------------------|-------|
| CB1 | Continuous batching engine — iteration-level scheduler | larql-inference + larql-server | **DROPPED** | Pure concurrency-throughput; doesn't affect single-stream baseline; doesn't accelerate any experiment. Re-open only if a future experiment needs concurrent decode. |
| CB2 | PagedAttention KV allocator | larql-inference | **DROPPED** | Pairs with CB1; useless without it. |
| CB3 | Concurrent stress benchmark | larql-server + bench/ | **DROPPED** | Measures a property the substrate framing doesn't care about. |
| MCP1 | MCP client + server in `larql serve` | larql-server | **DEFERRED** | Re-open only if a research workflow needs LARQL as an MCP-callable tool from inside an agent loop. Otherwise UX. |
| TM1 | Thinking-mode toggle | larql-inference + larql-server | **DEFERRED** | Re-open only if reasoning-trace structure becomes part of an experiment (e.g. probing thinking tokens). |
| RD1 | RMS-norm + scalar-mul pre-fusion shader (ADR-016 follow-up) | larql-compute | **KEEP** (small) | Affects baseline by ~0.1 ms/layer × 34 = ~3.4 ms; below baseline-credibility threshold floor but pure win. |
| (MTP1–MTP6 promoted to P1 — see "P1 — Gemma 4 MTP drafter support" below) | | | KEEP | Both substrate (new mechanism to study) and baseline (Ollama supports it on Gemma 4). |
| (SD1–SD2 promoted to P1) | | | KEEP | Reusable verification machinery; supports any future drafter-based technique. |

**Decision recorded 2026-05-09**: multi-tenant batched serving is out of
scope. LARQL will never be a production engine; the substrate framing's
"engine purpose" section above makes the call explicit. CB1, CB2, CB3 are
dropped. Re-open only if a specific *experiment* needs concurrent decode
(currently none does).

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
