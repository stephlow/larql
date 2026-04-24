//! Shared walk-path helpers.

use crate::vindex::walk_config::WalkFfnConfig;

/// True when the user asked for full-K (K ≥ feature count) — the signal
/// that we should route the walk through batched gemm rather than a
/// per-feature loop. Treats `usize::MAX` (set by `::dense` / `--k full`)
/// as full-K; also caches the check when top-K happens to exceed the
/// layer's feature count.
#[inline]
pub(super) fn hits_len_ge_intermediate(config: &WalkFfnConfig, layer: usize, intermediate: usize) -> bool {
    match config.k_for(layer) {
        Some(k) => k >= (intermediate * 8) / 10,
        None => true,
    }
}

/// Dispatch-trace entry: records which walk path fired for a given
/// `(forward_call, layer)`. Enabled via `WalkFfn::with_dispatch_trace()`.
///
/// Each walk path function calls `ctx.trace_path(layer, "name")` on
/// exit. Tests assert the expected sequence; the Q2 debugging flow
/// uses the trace to identify which path consumed a given vindex.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispatchEntry {
    pub layer: usize,
    pub path: &'static str,
}

/// Names pinned by the dispatch-trace tests. Renaming a walk path
/// breaks the trace consumer tests; update this list when that
/// happens, not the individual call sites.
pub const TRACE_NAMES: &[&str] = &[
    "override:sparse",
    "sparse:gemv_full_k",
    "sparse:parallel_q4k_down",
    "sparse:serial",
    "fp4_storage:sparse",
    "interleaved_q4:metal",
    "interleaved_q4:cpu",
    "interleaved",
    "full_mmap",
    "interleaved_q4k:dequant",
    "exact",
    "weights_fallback:sparse",
    "weights_fallback:override",
    "l1_cache_hit",
    "zero_features_dense",
];
