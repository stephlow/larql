# Changelog — larql-kv

All notable changes to `larql-kv` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/) conventions
with dated entries (`YYYY-MM-DD`) instead of semantic versions during the
pre-1.0 phase. Forward-looking work lives in [`ROADMAP.md`](ROADMAP.md).

## [2026-05-09] — Initial extraction from larql-inference

Genesis commit. The crate was carved out of
`larql-inference/src/engines/` (~5,540 LOC) where the four KV engines and
the supporting trait/dispatch had grown into a self-contained subsystem
with a real second consumer (`kv-cache-benchmark`) already importing it
through compatibility shims.

### Moved into larql-kv

| Component | Origin | Notes |
|---|---|---|
| `KvEngine` trait, `EngineKind`, `EngineInfo` | `engines/mod.rs` | Now the crate root. |
| `accuracy` module | `engines/accuracy.rs` | `softmax` re-exported from `larql_inference::forward::softmax` instead of being internal. |
| `profiler` module | `engines/profiler.rs` | Verbatim. |
| `engines::apollo` | `engines/kv_engines/apollo/` | Drop the redundant `kv_engines/` middle path. |
| `engines::markov_residual` | `engines/kv_engines/markov_residual/` | |
| `engines::turbo_quant` | `engines/kv_engines/turbo_quant/` | |
| `engines::unlimited_context` | `engines/kv_engines/unlimited_context/` | |

All `crate::{attention,ffn,forward,layer_graph,model,residual,vindex}::*`
paths inside the moved code rewritten to `larql_inference::*`.

### Stayed in larql-inference

- `engines::test_utils` — relocated to `larql_inference::test_utils`. ~20
  internal tests across `attention/`, `forward/`, `ffn/`, `layer_graph/`,
  `trace/`, `vindex/walk_ffn/` use these synthetic-weight fixtures and
  cannot follow into a downstream crate without a circular dep.

### Public-API surface widened in larql-inference

- `DEFAULT_GPU_KV_CACHE_MAX_SEQ` lifted from `pub(crate)` to `pub` in
  `layer_graph::pipeline_layer` so engines can read it from the new home.

### Removed re-exports from `larql_inference::*`

The following used to be at the `larql_inference` crate root or in
`research::*` and now live in `larql-kv`:

- `EngineInfo`, `EngineKind`, `KvEngine`
- `MarkovResidualEngine`, `UnlimitedContextEngine`
- `compare_hidden`, `cosine_similarity`, `js_divergence`, `kl_divergence`,
  `mse`, `softmax`, `HiddenAccuracy`

Downstream consumers should add `larql-kv` to their Cargo.toml and import
from there.

### Consumer updates

- `larql-cli` — `bench_cmd.rs` now imports `EngineKind` and
  `kv_memory_bytes_for_seq` from `larql_kv`. Workspace metal feature gained
  `larql-kv/metal`.
- `kv-cache-benchmark` — compat shims (`apollo/`, `turboquant/`,
  `unlimited_context/`, `real_model/markov_layer.rs`) now re-export from
  `larql_kv` directly. README updated.
- `larql-inference` examples — `apollo_rd_backend.rs` imports from
  `larql_kv::apollo`; `mech_interp_demo.rs` uses
  `larql_inference::test_utils`.

### Rationale

The four engines collectively share a trait and dispatch but diverge on
state management. Keeping them inside `larql-inference` meant every change
to a single engine recompiled the whole inference crate (transformer
forward pass, mech-interp surface, layer graphs). They are also the
target of independent benchmarking — the `kv-cache-benchmark` crate already
treated them as separable. Splitting tightens the API contract between
"transformer forward" (larql-inference) and "KV state strategy" (larql-kv).

The cut was clean: every primitive engines depend on (`ModelWeights`,
`BackendFfn`, `WalkFfn`, `KvCache`, `forward_*`, `rms_norm_heads`, …) was
already public in larql-inference, so this extraction did not require
designing new API.
