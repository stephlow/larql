//! `FfnRowAccess` — unified FFN row dispatch over native + Q4 + FP4
//! backends, plus the `GateIndex` compatibility composition.
//!
//! These two traits are grouped together because the blanket impls
//! cascade: every type that implements the three storage traits gets
//! `FfnRowAccess` for free, and every type that adds `GateLookup` +
//! `PatchOverrides` on top gets `GateIndex`. The dispatch logic in
//! `ffn_row_*` is the load-bearing default that keeps walk-kernel
//! callers storage-agnostic.

use super::StorageBucket;
use super::{Fp4FfnAccess, GateLookup, NativeFfnAccess, PatchOverrides, QuantizedFfnAccess};
use crate::index::storage::ffn_store::FFN_DOWN;

/// Unified FFN row operations over native, Q4K/Q6K, and FP4/FP8 storage.
pub trait FfnRowAccess: NativeFfnAccess + QuantizedFfnAccess + Fp4FfnAccess {
    // ── Unified FFN row access ─────────────────────────────────────────────
    //
    // One entry point per operation; the walk kernel calls these and
    // doesn't have to care about storage format. Default impls below
    // dispatch through the priority chain:
    //   1. FP4/FP8 (exp 26) — tried first when `has_fp4_storage()` is true
    //   2. Native f32 mmap  — interleaved / up_features / down_features
    //   3. Q4K interleaved  — `q4k_ffn_row_*` with via-cache for down
    //
    // Each step returns early on success. If every backend declines,
    // returns `None` / `false`.
    //
    // Overriding these in a concrete impl is rarely correct — the default
    // logic is the contract. Override the *specific* backend methods
    // (`fp4_ffn_row_dot`, `q4k_ffn_row_dot`, etc.) instead.

    /// Unified fused dequant + dot. `component`: 0=gate, 1=up, 2=down.
    /// Returns the dot product `row(layer, component, feat) · x` from
    /// whichever backend is loaded, or `None` if no backend covers this
    /// coordinate.
    fn ffn_row_dot(&self, layer: usize, component: usize, feat: usize, x: &[f32]) -> Option<f32> {
        // 1. FP4/FP8 backend (if loaded). fp4_ffn_row_dot returns None
        //    when the projection's precision tag is f16/f32 (caller
        //    falls through to native).
        if self.has_fp4_storage() {
            if let Some(dot) = self.fp4_ffn_row_dot(layer, component, feat, x) {
                return Some(dot);
            }
        }
        // 2. Native f32 mmap.
        let x_view = ndarray::ArrayView1::from(x);
        match component {
            0 => {
                if let Some(m) = self.interleaved_gate(layer) {
                    if feat < m.nrows() && m.ncols() == x.len() {
                        return Some(m.row(feat).dot(&x_view));
                    }
                }
            }
            1 => {
                if let Some(m) = self.interleaved_up(layer) {
                    if feat < m.nrows() && m.ncols() == x.len() {
                        return Some(m.row(feat).dot(&x_view));
                    }
                }
                if let Some(m) = self.up_layer_matrix(layer) {
                    if feat < m.nrows() && m.ncols() == x.len() {
                        return Some(m.row(feat).dot(&x_view));
                    }
                }
            }
            2 => {
                if let Some(row) = self.down_feature_vector(layer, feat) {
                    if row.len() == x.len() {
                        return Some(ndarray::ArrayView1::from(row).dot(&x_view));
                    }
                }
                if let Some(m) = self.interleaved_down(layer) {
                    if feat < m.nrows() && m.ncols() == x.len() {
                        return Some(m.row(feat).dot(&x_view));
                    }
                }
                if let Some(m) = self.down_layer_matrix(layer) {
                    if feat < m.nrows() && m.ncols() == x.len() {
                        return Some(m.row(feat).dot(&x_view));
                    }
                }
            }
            _ => {}
        }
        // 3. Q4K fallback.
        if self.has_interleaved_q4k() {
            return self.q4k_ffn_row_dot(layer, component, feat, x);
        }
        None
    }

    /// Unified fused dequant + scaled-add: `out[i] += alpha * row[i]`.
    /// Returns `true` on success, `false` if no backend covers the
    /// coordinate (or shapes don't match).
    fn ffn_row_scaled_add(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        alpha: f32,
        out: &mut [f32],
    ) -> bool {
        if self.has_fp4_storage() && self.fp4_ffn_row_scaled_add(layer, component, feat, alpha, out)
        {
            return true;
        }
        let mut out_view = ndarray::ArrayViewMut1::from(&mut out[..]);
        match component {
            0 => {
                if let Some(m) = self.interleaved_gate(layer) {
                    if feat < m.nrows() && m.ncols() == out_view.len() {
                        out_view.scaled_add(alpha, &m.row(feat));
                        return true;
                    }
                }
            }
            1 => {
                if let Some(m) = self.interleaved_up(layer) {
                    if feat < m.nrows() && m.ncols() == out_view.len() {
                        out_view.scaled_add(alpha, &m.row(feat));
                        return true;
                    }
                }
                if let Some(m) = self.up_layer_matrix(layer) {
                    if feat < m.nrows() && m.ncols() == out_view.len() {
                        out_view.scaled_add(alpha, &m.row(feat));
                        return true;
                    }
                }
            }
            2 => {
                if let Some(row) = self.down_feature_vector(layer, feat) {
                    if row.len() == out_view.len() {
                        out_view.scaled_add(alpha, &ndarray::ArrayView1::from(row));
                        return true;
                    }
                }
                if let Some(m) = self.interleaved_down(layer) {
                    if feat < m.nrows() && m.ncols() == out_view.len() {
                        out_view.scaled_add(alpha, &m.row(feat));
                        return true;
                    }
                }
                if let Some(m) = self.down_layer_matrix(layer) {
                    if feat < m.nrows() && m.ncols() == out_view.len() {
                        out_view.scaled_add(alpha, &m.row(feat));
                        return true;
                    }
                }
            }
            _ => return false,
        }
        if self.has_interleaved_q4k() {
            if component == FFN_DOWN {
                // W2: prefer the feature-major down file when present —
                // a single row decode beats the whole-layer dequant +
                // transpose path. Fall back to the cache for vindexes
                // extracted before the feature-major down emit landed.
                if self.q4k_down_feature_scaled_add(layer, feat, alpha, out) {
                    return true;
                }
                return self.q4k_ffn_row_scaled_add_via_cache(layer, component, feat, alpha, out);
            }
            return self.q4k_ffn_row_scaled_add(layer, component, feat, alpha, out);
        }
        false
    }

    /// Unified decode-into-buffer. `out.len()` must equal the row width.
    fn ffn_row_into(&self, layer: usize, component: usize, feat: usize, out: &mut [f32]) -> bool {
        if self.has_fp4_storage() && self.fp4_ffn_row_into(layer, component, feat, out) {
            return true;
        }
        let copy_row = |row: ndarray::ArrayView1<'_, f32>, out: &mut [f32]| -> bool {
            if row.len() != out.len() {
                return false;
            }
            for (i, &v) in row.iter().enumerate() {
                out[i] = v;
            }
            true
        };
        match component {
            0 => {
                if let Some(m) = self.interleaved_gate(layer) {
                    if feat < m.nrows() {
                        return copy_row(m.row(feat), out);
                    }
                }
            }
            1 => {
                if let Some(m) = self.interleaved_up(layer) {
                    if feat < m.nrows() {
                        return copy_row(m.row(feat), out);
                    }
                }
                if let Some(m) = self.up_layer_matrix(layer) {
                    if feat < m.nrows() {
                        return copy_row(m.row(feat), out);
                    }
                }
            }
            2 => {
                if let Some(row) = self.down_feature_vector(layer, feat) {
                    return copy_row(ndarray::ArrayView1::from(row), out);
                }
                if let Some(m) = self.interleaved_down(layer) {
                    if feat < m.nrows() {
                        return copy_row(m.row(feat), out);
                    }
                }
                if let Some(m) = self.down_layer_matrix(layer) {
                    if feat < m.nrows() {
                        return copy_row(m.row(feat), out);
                    }
                }
            }
            _ => return false,
        }
        if self.has_interleaved_q4k() {
            return self.q4k_ffn_row_into(layer, component, feat, out);
        }
        false
    }

    /// Bucket the index's primary FFN storage falls into. Encapsulates the
    /// `has_*`-flag logic so audits and tooling (e.g. `walk_path_audit`)
    /// don't scatter flag-checks across their bucketing logic.
    ///
    /// Priority mirrors `ffn_row_dot`'s dispatch chain (FP4 first, then
    /// native f32, then Q4K), so the bucket reflects what data the
    /// unified row dispatch will *actually* walk on a mixed-format vindex
    /// — not just which flags happen to be set.
    ///
    /// New storage formats should update this default impl so downstream
    /// consumers automatically pick up the right bucket. Override only
    /// when an implementer wants to pin the bucket explicitly (rare).
    fn primary_storage_bucket(&self) -> StorageBucket {
        if self.has_fp4_storage() {
            StorageBucket::Fp4
        } else if self.has_interleaved() || self.has_full_mmap_ffn() || self.has_down_features() {
            // Native f32 mmap available; ffn_row_* dispatch prefers it
            // over Q4K, so sparse on a mixed (f32 + Q4K) vindex walks
            // f32 features and lands in the Exact bucket.
            StorageBucket::Exact
        } else if self.has_interleaved_q4k() || self.has_interleaved_q4() {
            StorageBucket::Quantized
        } else {
            StorageBucket::Exact
        }
    }
}

impl<T> FfnRowAccess for T where T: NativeFfnAccess + QuantizedFfnAccess + Fp4FfnAccess + ?Sized {}

/// Compatibility trait for consumers that need the whole vindex surface.
///
/// New code should prefer the narrower traits above (`GateLookup`,
/// `PatchOverrides`, `NativeFfnAccess`, `QuantizedFfnAccess`,
/// `Fp4FfnAccess`, or `FfnRowAccess`) when it does not need the full
/// combined API.
pub trait GateIndex: GateLookup + PatchOverrides + FfnRowAccess {}

impl<T> GateIndex for T where T: GateLookup + PatchOverrides + FfnRowAccess + ?Sized {}

#[cfg(test)]
mod tests {
    //! Coverage for the unified `ffn_row_*` dispatch chain. Each test
    //! pins one branch of the priority cascade (FP4 → native f32 → Q4_K)
    //! against a stub that lights up exactly one backend so we can see
    //! which path the dispatch took.

    use super::*;
    use ndarray::Array2;

    /// Configurable stub. Each backend bool toggles which `has_*`
    /// flag returns true and which row methods route through.
    #[derive(Default)]
    struct Stub {
        // Native f32 mmap matrices, keyed by (component → matrix).
        gate: Option<Array2<f32>>,
        up: Option<Array2<f32>>,
        down: Option<Array2<f32>>,
        // Alternate native arms (the dispatch chain tries
        // `interleaved_*` first, then falls through to `*_layer_matrix`).
        up_layer: Option<Array2<f32>>,
        down_layer: Option<Array2<f32>>,
        // Per-feature down vectors (preferred over interleaved_down in
        // the component=2 chain).
        down_feature: Option<Vec<f32>>,
        // FP4 — when set, `fp4_ffn_row_*` returns predetermined sentinels
        // so dispatch routing is observable.
        fp4_dot: Option<f32>,
        fp4_scaled_add_returns: bool,
        fp4_into_returns: bool,
        // Q4_K — when set, `q4k_ffn_row_*` returns predetermined sentinels.
        q4k_dot: Option<f32>,
        q4k_scaled_add_returns: bool,
        q4k_into_returns: bool,
        q4k_down_feature_returns: bool,
    }

    impl NativeFfnAccess for Stub {
        fn has_interleaved(&self) -> bool {
            self.gate.is_some() || self.up.is_some() || self.down.is_some()
        }
        fn has_full_mmap_ffn(&self) -> bool {
            self.up_layer.is_some() || self.down_layer.is_some()
        }
        fn has_down_features(&self) -> bool {
            self.down_feature.is_some()
        }
        fn interleaved_gate(&self, _: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
            self.gate.as_ref().map(|m| m.view())
        }
        fn interleaved_up(&self, _: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
            self.up.as_ref().map(|m| m.view())
        }
        fn interleaved_down(&self, _: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
            self.down.as_ref().map(|m| m.view())
        }
        fn up_layer_matrix(&self, _: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
            self.up_layer.as_ref().map(|m| m.view())
        }
        fn down_layer_matrix(&self, _: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
            self.down_layer.as_ref().map(|m| m.view())
        }
        fn down_feature_vector(&self, _: usize, _: usize) -> Option<&[f32]> {
            self.down_feature.as_deref()
        }
    }

    impl QuantizedFfnAccess for Stub {
        fn has_interleaved_q4k(&self) -> bool {
            self.q4k_dot.is_some() || self.q4k_scaled_add_returns || self.q4k_into_returns
        }
        fn q4k_ffn_row_dot(&self, _: usize, _: usize, _: usize, _: &[f32]) -> Option<f32> {
            self.q4k_dot
        }
        fn q4k_ffn_row_scaled_add(
            &self,
            _: usize,
            _: usize,
            _: usize,
            _: f32,
            _: &mut [f32],
        ) -> bool {
            self.q4k_scaled_add_returns
        }
        fn q4k_ffn_row_scaled_add_via_cache(
            &self,
            _: usize,
            _: usize,
            _: usize,
            _: f32,
            _: &mut [f32],
        ) -> bool {
            self.q4k_scaled_add_returns
        }
        fn q4k_down_feature_scaled_add(&self, _: usize, _: usize, _: f32, _: &mut [f32]) -> bool {
            self.q4k_down_feature_returns
        }
        fn q4k_ffn_row_into(&self, _: usize, _: usize, _: usize, _: &mut [f32]) -> bool {
            self.q4k_into_returns
        }
    }

    impl Fp4FfnAccess for Stub {
        fn has_fp4_storage(&self) -> bool {
            self.fp4_dot.is_some() || self.fp4_scaled_add_returns || self.fp4_into_returns
        }
        fn fp4_ffn_row_dot(&self, _: usize, _: usize, _: usize, _: &[f32]) -> Option<f32> {
            self.fp4_dot
        }
        fn fp4_ffn_row_scaled_add(
            &self,
            _: usize,
            _: usize,
            _: usize,
            _: f32,
            _: &mut [f32],
        ) -> bool {
            self.fp4_scaled_add_returns
        }
        fn fp4_ffn_row_into(&self, _: usize, _: usize, _: usize, _: &mut [f32]) -> bool {
            self.fp4_into_returns
        }
    }

    fn one_row(values: &[f32]) -> Array2<f32> {
        Array2::from_shape_vec((1, values.len()), values.to_vec()).unwrap()
    }

    // ── ffn_row_dot ─────────────────────────────────────────────

    #[test]
    fn dot_routes_through_fp4_first() {
        // FP4 sentinel wins even when native + Q4_K are both available.
        let s = Stub {
            fp4_dot: Some(99.0),
            gate: Some(one_row(&[1.0, 1.0, 1.0])),
            q4k_dot: Some(7.0),
            ..Default::default()
        };
        let x = [1.0, 1.0, 1.0];
        assert_eq!(s.ffn_row_dot(0, 0, 0, &x), Some(99.0));
    }

    #[test]
    fn dot_falls_through_to_native_gate_when_fp4_declines() {
        // FP4 storage present, but `fp4_ffn_row_dot` returns None — the
        // dispatch must fall through to native, NOT to Q4_K.
        let s = Stub {
            fp4_into_returns: true, // has_fp4_storage = true
            fp4_dot: None,
            gate: Some(one_row(&[2.0, 3.0, 4.0])),
            q4k_dot: Some(7.0),
            ..Default::default()
        };
        let x = [1.0, 1.0, 1.0];
        assert_eq!(s.ffn_row_dot(0, 0, 0, &x), Some(9.0));
    }

    #[test]
    fn dot_native_up_then_q4k_fallback() {
        // Component=1: should try interleaved_up first, then q4k.
        let s = Stub {
            up: Some(one_row(&[5.0])),
            q4k_dot: Some(99.0),
            ..Default::default()
        };
        assert_eq!(s.ffn_row_dot(0, 1, 0, &[2.0]), Some(10.0));
    }

    #[test]
    fn dot_native_down_via_interleaved() {
        let s = Stub {
            down: Some(one_row(&[3.0, 3.0])),
            ..Default::default()
        };
        assert_eq!(s.ffn_row_dot(0, 2, 0, &[1.0, 1.0]), Some(6.0));
    }

    #[test]
    fn dot_q4k_used_when_no_native() {
        let s = Stub {
            q4k_dot: Some(42.0),
            ..Default::default()
        };
        assert_eq!(s.ffn_row_dot(0, 0, 0, &[1.0]), Some(42.0));
    }

    #[test]
    fn dot_returns_none_when_nothing_loaded() {
        let s = Stub::default();
        assert!(s.ffn_row_dot(0, 0, 0, &[1.0]).is_none());
    }

    #[test]
    fn dot_invalid_component_returns_none() {
        let s = Stub {
            gate: Some(one_row(&[1.0])),
            ..Default::default()
        };
        // component=99 has no native arm; q4k isn't loaded either.
        assert!(s.ffn_row_dot(0, 99, 0, &[1.0]).is_none());
    }

    #[test]
    fn dot_native_shape_mismatch_falls_through_to_q4k() {
        // Native gate has 3 cols but x has 2 — shape mismatch falls
        // through to the Q4_K fallback.
        let s = Stub {
            gate: Some(one_row(&[1.0, 1.0, 1.0])),
            q4k_dot: Some(123.0),
            ..Default::default()
        };
        assert_eq!(s.ffn_row_dot(0, 0, 0, &[1.0, 1.0]), Some(123.0));
    }

    // ── ffn_row_scaled_add ──────────────────────────────────────

    #[test]
    fn scaled_add_native_gate_writes_alpha_times_row() {
        let s = Stub {
            gate: Some(one_row(&[2.0, 4.0])),
            ..Default::default()
        };
        let mut out = [10.0_f32, 10.0];
        assert!(s.ffn_row_scaled_add(0, 0, 0, 0.5, &mut out));
        assert_eq!(out, [11.0, 12.0]);
    }

    #[test]
    fn scaled_add_down_prefers_feature_major_q4k() {
        // When both Q4K caches and feature-major-down report success,
        // feature-major must win for component=2. We can't directly
        // observe which path ran, but `q4k_down_feature_returns=true`
        // returns first; with `q4k_scaled_add_returns=false` we'd hit
        // the cache fallback (and return false). Setting cache=false
        // pins that feature-major was the chosen path.
        let s = Stub {
            q4k_dot: Some(0.0), // make has_interleaved_q4k = true
            q4k_down_feature_returns: true,
            q4k_scaled_add_returns: false,
            ..Default::default()
        };
        let mut out = [0.0_f32; 4];
        assert!(s.ffn_row_scaled_add(0, 2, 0, 1.0, &mut out));
    }

    #[test]
    fn scaled_add_falls_back_to_cache_when_feature_major_declines() {
        // Feature-major returns false; cache returns true.
        let s = Stub {
            q4k_into_returns: true, // make has_interleaved_q4k = true
            q4k_down_feature_returns: false,
            q4k_scaled_add_returns: true,
            ..Default::default()
        };
        let mut out = [0.0_f32; 4];
        assert!(s.ffn_row_scaled_add(0, 2, 0, 1.0, &mut out));
    }

    #[test]
    fn scaled_add_returns_false_when_no_backend_covers() {
        let s = Stub::default();
        let mut out = [0.0_f32; 4];
        assert!(!s.ffn_row_scaled_add(0, 0, 0, 1.0, &mut out));
        assert_eq!(out, [0.0_f32; 4], "must not modify out on failure");
    }

    #[test]
    fn scaled_add_invalid_component_returns_false_early() {
        // component 99 short-circuits to false before any q4k probe.
        let s = Stub {
            q4k_scaled_add_returns: true,
            ..Default::default()
        };
        let mut out = [0.0_f32; 4];
        assert!(!s.ffn_row_scaled_add(0, 99, 0, 1.0, &mut out));
    }

    // ── ffn_row_into ────────────────────────────────────────────

    #[test]
    fn into_native_up_copies_row_verbatim() {
        let s = Stub {
            up: Some(one_row(&[1.0, 2.0, 3.0])),
            ..Default::default()
        };
        let mut out = [0.0_f32; 3];
        assert!(s.ffn_row_into(0, 1, 0, &mut out));
        assert_eq!(out, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn into_falls_through_to_q4k_when_native_declines() {
        let s = Stub {
            q4k_into_returns: true,
            ..Default::default()
        };
        let mut out = [0.0_f32; 3];
        assert!(s.ffn_row_into(0, 0, 0, &mut out));
    }

    #[test]
    fn into_returns_false_with_no_backend() {
        let s = Stub::default();
        let mut out = [0.0_f32; 3];
        assert!(!s.ffn_row_into(0, 0, 0, &mut out));
    }

    #[test]
    fn into_shape_mismatch_returns_false() {
        // Native gate exists but row width != out width.
        let s = Stub {
            gate: Some(one_row(&[1.0, 2.0])),
            ..Default::default()
        };
        let mut out = [0.0_f32; 3]; // wrong size
        assert!(!s.ffn_row_into(0, 0, 0, &mut out));
    }

    // ── primary_storage_bucket ──────────────────────────────────

    #[test]
    fn bucket_fp4_when_fp4_storage_present() {
        let s = Stub {
            fp4_into_returns: true,
            ..Default::default()
        };
        assert_eq!(s.primary_storage_bucket(), StorageBucket::Fp4);
    }

    #[test]
    fn bucket_exact_when_native_present() {
        let s = Stub {
            gate: Some(one_row(&[1.0])),
            ..Default::default()
        };
        assert_eq!(s.primary_storage_bucket(), StorageBucket::Exact);
    }

    #[test]
    fn bucket_quantized_when_only_q4k() {
        let s = Stub {
            q4k_into_returns: true, // has_interleaved_q4k = true
            ..Default::default()
        };
        assert_eq!(s.primary_storage_bucket(), StorageBucket::Quantized);
    }

    #[test]
    fn bucket_exact_when_nothing_loaded() {
        // The empty-vindex / placeholder case lands in Exact (the
        // "default null" bucket — see ffn_row.rs comment).
        let s = Stub::default();
        assert_eq!(s.primary_storage_bucket(), StorageBucket::Exact);
    }

    #[test]
    fn bucket_priority_fp4_over_exact() {
        // Both FP4 and native are loaded — FP4 wins.
        let s = Stub {
            fp4_into_returns: true,
            gate: Some(one_row(&[1.0])),
            ..Default::default()
        };
        assert_eq!(s.primary_storage_bucket(), StorageBucket::Fp4);
    }

    // ── ffn_row_dot — additional native arms ─────────────────────

    #[test]
    fn dot_component_1_falls_through_to_up_layer_matrix() {
        // No interleaved_up, but up_layer_matrix is present — second
        // native arm in the component=1 chain.
        let s = Stub {
            up_layer: Some(one_row(&[3.0, 3.0])),
            ..Default::default()
        };
        assert_eq!(s.ffn_row_dot(0, 1, 0, &[1.0, 2.0]), Some(9.0));
    }

    #[test]
    fn dot_component_2_uses_down_feature_vector_first() {
        // First native arm for component=2: per-feature down vectors.
        let s = Stub {
            down_feature: Some(vec![2.0, 3.0]),
            // Both lower-priority arms set — must NOT be selected.
            down: Some(one_row(&[99.0, 99.0])),
            down_layer: Some(one_row(&[88.0, 88.0])),
            ..Default::default()
        };
        assert_eq!(s.ffn_row_dot(0, 2, 0, &[1.0, 1.0]), Some(5.0));
    }

    #[test]
    fn dot_component_2_falls_through_to_down_layer_matrix() {
        // No down_feature, no interleaved_down — third arm wins.
        let s = Stub {
            down_layer: Some(one_row(&[4.0, 4.0])),
            ..Default::default()
        };
        assert_eq!(s.ffn_row_dot(0, 2, 0, &[1.0, 0.5]), Some(6.0));
    }

    #[test]
    fn dot_component_2_skips_down_feature_on_shape_mismatch() {
        // down_feature width != x width → that arm declines, fall through
        // to interleaved_down.
        let s = Stub {
            down_feature: Some(vec![1.0, 2.0, 3.0]), // 3-wide
            down: Some(one_row(&[5.0, 5.0])),        // 2-wide
            ..Default::default()
        };
        assert_eq!(s.ffn_row_dot(0, 2, 0, &[1.0, 2.0]), Some(15.0));
    }

    // ── ffn_row_scaled_add — additional native arms ──────────────

    #[test]
    fn scaled_add_component_1_via_up_layer_matrix() {
        let s = Stub {
            up_layer: Some(one_row(&[1.0, 2.0])),
            ..Default::default()
        };
        let mut out = [10.0_f32, 10.0];
        assert!(s.ffn_row_scaled_add(0, 1, 0, 2.0, &mut out));
        assert_eq!(out, [12.0, 14.0]);
    }

    #[test]
    fn scaled_add_component_2_via_down_feature_vector() {
        let s = Stub {
            down_feature: Some(vec![3.0, 4.0]),
            ..Default::default()
        };
        let mut out = [0.0_f32; 2];
        assert!(s.ffn_row_scaled_add(0, 2, 0, 1.0, &mut out));
        assert_eq!(out, [3.0, 4.0]);
    }

    #[test]
    fn scaled_add_component_2_via_down_layer_matrix() {
        let s = Stub {
            down_layer: Some(one_row(&[1.0, 1.0])),
            ..Default::default()
        };
        let mut out = [0.0_f32; 2];
        assert!(s.ffn_row_scaled_add(0, 2, 0, 0.5, &mut out));
        assert_eq!(out, [0.5, 0.5]);
    }

    // ── ffn_row_into — additional native arms ────────────────────

    #[test]
    fn into_component_1_via_up_layer_matrix() {
        let s = Stub {
            up_layer: Some(one_row(&[7.0, 8.0])),
            ..Default::default()
        };
        let mut out = [0.0_f32; 2];
        assert!(s.ffn_row_into(0, 1, 0, &mut out));
        assert_eq!(out, [7.0, 8.0]);
    }

    #[test]
    fn into_component_2_via_down_feature_vector() {
        let s = Stub {
            down_feature: Some(vec![1.5, 2.5, 3.5]),
            ..Default::default()
        };
        let mut out = [0.0_f32; 3];
        assert!(s.ffn_row_into(0, 2, 0, &mut out));
        assert_eq!(out, [1.5, 2.5, 3.5]);
    }

    #[test]
    fn into_component_2_via_interleaved_down() {
        let s = Stub {
            down: Some(one_row(&[9.0, 9.0])),
            ..Default::default()
        };
        let mut out = [0.0_f32; 2];
        assert!(s.ffn_row_into(0, 2, 0, &mut out));
        assert_eq!(out, [9.0, 9.0]);
    }

    #[test]
    fn into_component_2_via_down_layer_matrix() {
        let s = Stub {
            down_layer: Some(one_row(&[2.0, 4.0])),
            ..Default::default()
        };
        let mut out = [0.0_f32; 2];
        assert!(s.ffn_row_into(0, 2, 0, &mut out));
        assert_eq!(out, [2.0, 4.0]);
    }

    #[test]
    fn into_invalid_component_returns_false() {
        let s = Stub {
            gate: Some(one_row(&[1.0])),
            ..Default::default()
        };
        let mut out = [0.0_f32; 1];
        assert!(!s.ffn_row_into(0, 99, 0, &mut out));
    }

    // ── primary_storage_bucket — additional native triggers ──────

    #[test]
    fn bucket_exact_when_full_mmap_ffn_only() {
        let s = Stub {
            up_layer: Some(one_row(&[1.0])),
            ..Default::default()
        };
        assert_eq!(s.primary_storage_bucket(), StorageBucket::Exact);
    }

    #[test]
    fn bucket_exact_when_down_features_only() {
        let s = Stub {
            down_feature: Some(vec![1.0]),
            ..Default::default()
        };
        assert_eq!(s.primary_storage_bucket(), StorageBucket::Exact);
    }

    #[test]
    fn bucket_quantized_when_only_q4_legacy() {
        // QuantizedFfnAccess::has_interleaved_q4 default is false; we
        // can't trigger it via the Stub since we only override q4k.
        // Instead exercise the q4_default-true path through a wrapper
        // type that overrides only `has_interleaved_q4`.
        struct OnlyQ4;
        impl NativeFfnAccess for OnlyQ4 {}
        impl QuantizedFfnAccess for OnlyQ4 {
            fn has_interleaved_q4(&self) -> bool {
                true
            }
        }
        impl Fp4FfnAccess for OnlyQ4 {}
        assert_eq!(OnlyQ4.primary_storage_bucket(), StorageBucket::Quantized);
    }

    // ── Q4_K fallback for non-down components ────────────────────

    #[test]
    fn scaled_add_component_0_falls_through_to_q4k() {
        // No native gate, but Q4_K reports success — the
        // non-component-2 branch of the q4k tail should run.
        let s = Stub {
            q4k_scaled_add_returns: true,
            q4k_dot: Some(0.0), // make has_interleaved_q4k = true
            ..Default::default()
        };
        let mut out = [0.0_f32; 4];
        assert!(s.ffn_row_scaled_add(0, 0, 0, 1.0, &mut out));
    }

    #[test]
    fn scaled_add_component_1_falls_through_to_q4k() {
        let s = Stub {
            q4k_scaled_add_returns: true,
            q4k_dot: Some(0.0),
            ..Default::default()
        };
        let mut out = [0.0_f32; 4];
        assert!(s.ffn_row_scaled_add(0, 1, 0, 1.0, &mut out));
    }

    #[test]
    fn dot_component_1_native_shape_mismatch_falls_through_to_q4k() {
        // interleaved_up exists but cols != x.len() — must fall through.
        let s = Stub {
            up: Some(one_row(&[1.0, 1.0, 1.0])), // 3 cols
            q4k_dot: Some(77.0),
            ..Default::default()
        };
        assert_eq!(s.ffn_row_dot(0, 1, 0, &[1.0, 1.0]), Some(77.0));
    }

    #[test]
    fn dot_component_2_all_native_shape_mismatch_falls_through_to_q4k() {
        // Every component=2 native arm has a shape mismatch — final
        // fall-through hits q4k.
        let s = Stub {
            down_feature: Some(vec![1.0, 2.0, 3.0]), // 3-wide
            down: Some(one_row(&[1.0, 1.0, 1.0])),   // 3-wide
            down_layer: Some(one_row(&[1.0, 1.0, 1.0])),
            q4k_dot: Some(55.0),
            ..Default::default()
        };
        assert_eq!(s.ffn_row_dot(0, 2, 0, &[1.0, 2.0]), Some(55.0));
    }

    #[test]
    fn into_feat_out_of_range_falls_through_to_q4k() {
        // feat >= nrows of interleaved_up → the inner if-let block
        // doesn't return; the function continues to up_layer_matrix
        // and finally to q4k.
        let s = Stub {
            up: Some(one_row(&[1.0])),
            q4k_into_returns: true,
            ..Default::default()
        };
        let mut out = [0.0_f32; 1];
        assert!(s.ffn_row_into(0, 1, 99, &mut out));
    }
}
