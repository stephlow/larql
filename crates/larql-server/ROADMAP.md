# Roadmap — larql-server / larql-router

## Current state (as of 2026-04-26)

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
