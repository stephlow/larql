# Roadmap — larql-server / larql-router

## Current state (as of 2026-04-26)

- Code quality pass complete: modularity refactor + magic string cleanup + test restructure (see Completed below).
- Follow-up review fixes complete: rate limiting no longer trusts
  `X-Forwarded-For` by default, route/path strings are centralized,
  server loader options are grouped, embed errors use the standard JSON
  error envelope, and server-local clippy allows were reduced.
- Test coverage: **63.3% line / 73.2% function** (430 tests, 0 failures). gRPC handler tests unblocked grpc.rs (0%→65%). Magic strings eliminated across stream.rs, grpc.rs, describe.rs.
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

---

## P0: Active

Nothing critical-path is blocking right now.

---

## P0: Remote expert protocol (Act 2)

These items are the wire-format half of the "experts live elsewhere" demo.
The inference-side counterpart (`RemoteExpertBackend`, `cpu_moe_forward`) is
tracked in `larql-inference/ROADMAP.md`.

### `POST /v1/expert/{layer}/{expert_id}`
**Status**: Not started  
Accept a residual vector (hidden-size f32 or bf16), run that expert's gated FFN
(gate + up + SiLU + down), return the residual delta. Endpoint already declared
in the completed-items list below as a stub; needs a real handler wired to
`ModelWeights`.

### `POST /v1/expert/batch`
**Status**: Not started  
Body: list of `{layer, expert_id, residual}`. Returns a matching list of deltas.
Collapses a layer's K active experts into one HTTP round trip per server, avoiding
K separate requests under MoE top-K dispatch.

### `--experts 0-31` flag on `larql serve`
**Status**: Not started  
**Files**: `src/main.rs` (CLI), `src/state.rs`  
Load and serve only the specified expert ID subset. Allows horizontal sharding
of a large MoE model across machines: `larql serve --experts 0-31` on host A,
`--experts 32-63` on host B. Experts outside the owned range return HTTP 404.

### `load_model_weights_ffn_only` — skip attention tensors on `--ffn-only`
**Status**: Not started  
**Files**: `src/state.rs`  
`larql serve --ffn-only` currently loads `ModelWeights` in full (attention,
norms, embeddings). Add `load_model_weights_ffn_only` that skips attention
tensors to reduce RSS on expert-only shard machines. Expert servers have no
use for Q/K/V projections or the lm_head.

### `RemoteExpertBackend` — note
Implementation lives in `larql-inference` (sharding map, parallel dispatch,
per-expert error handling). This server owns the endpoint definitions and the
`--experts` flag; larql-inference owns the client-side routing.

---

## P1: Active

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
- Move boot/loading/discovery from `main.rs` into a library module if CLI
  startup needs deeper unit coverage.
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

### T2. Test coverage — remaining reachable paths *(in progress)*

**Current**: 63.3% line / 73.2% function. 430 tests.

**Completed this pass:**
- `grpc.rs` 0% → **65%** — 28 direct gRPC handler tests (health, stats, describe, walk, select, relations, walk_ffn, infer, stream_describe)
- Magic strings: `"probe"` → `PROBE_RELATION_SOURCE`; `"ok"` → `HEALTH_STATUS_OK`; infer mode strings in grpc.rs; WebSocket message types in stream.rs (`WS_TYPE_*`, `WS_CMD_*`)

**Still addressable without real weights:**

| File | Current | Gap | What to add |
|---|---|---|---|
| `routes/stream.rs` | 0% | 219 lines | WebSocket inner functions — needs `tokio-tungstenite` or direct `grpc_stream_describe`-style testing |
| `routes/explain.rs` | 11% | 152 lines | Gated on `get_or_load_weights()`; only handler scaffold reachable |
| `routes/infer.rs` | 31% | ~70 lines | `has_model_weights=false` + `infer_disabled=false` → 503 |
| `routes/warmup.rs` | 80% | ~15 lines | `warmup_hnsw=true` warn path (HNSW not enabled) |
| `embed_store.rs` | 25% | ~72 lines | Reads real f16 files; hard to test in-process |
| `announce.rs` | 6% | ~98 lines | gRPC stream to real router — defer |

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
GridState dispatch updates. Mentioned in larql-vindex P2.

### G6. Live router-shard topology change
**Impact**: Today shards are static (`--shards` flag at router boot).
For ops convenience, expose `POST /v1/router/shards` (admin-gated)
to add/remove a shard without restarting the router. Pair with
`--grid-port` health checks.

---

## Completed

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
