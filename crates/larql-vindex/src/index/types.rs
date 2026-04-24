//! Shared types and traits for the vindex index.

use ndarray::{Array1, Array2};
use larql_models::TopKEntry;

/// Metadata for a single FFN feature (from extraction).
#[derive(Clone)]
pub struct FeatureMeta {
    pub top_token: String,
    pub top_token_id: u32,
    pub c_score: f32,
    pub top_k: Vec<TopKEntry>,
}

/// A single step in the walk trace — one feature that fired at one layer.
pub struct WalkHit {
    pub layer: usize,
    pub feature: usize,
    pub gate_score: f32,
    pub meta: FeatureMeta,
}

/// Result of a walk — per-layer feature activations with full metadata.
pub struct WalkTrace {
    pub layers: Vec<(usize, Vec<WalkHit>)>,
}

/// Trait for gate-based feature lookup.
///
/// Both `VectorIndex` (base, readonly) and `PatchedVindex` (with overlay)
/// implement this trait, allowing `WalkFfn` and other consumers to work
/// transparently with patched or unpatched indexes.
pub trait GateIndex: Send + Sync {
    fn gate_knn(&self, layer: usize, residual: &Array1<f32>, top_k: usize) -> Vec<(usize, f32)>;
    fn feature_meta(&self, layer: usize, feature: usize) -> Option<FeatureMeta>;
    fn num_features(&self, layer: usize) -> usize;
    fn down_override(&self, _layer: usize, _feature: usize) -> Option<&[f32]> { None }
    /// Up vector override at (layer, feature). Used by INSERT to write
    /// the slot's up component when installing a constellation fact.
    /// `walk_ffn_sparse` checks this before reading from `up_layer_matrix`,
    /// matching the parallel pattern for `down_override`.
    fn up_override(&self, _layer: usize, _feature: usize) -> Option<&[f32]> { None }
    /// Gate vector override at (layer, feature). Lives in the patch
    /// overlay (`PatchedVindex.overrides_gate`). Used by the sparse
    /// inference fallback to recompute `silu(gate_override · x)` so
    /// the strong installed gate actually drives the activation —
    /// without this, gather-from-dense reads the original weak slot.
    fn gate_override(&self, _layer: usize, _feature: usize) -> Option<&[f32]> { None }
    /// Check if any down vector overrides or gate overrides exist at this layer.
    fn has_overrides_at(&self, _layer: usize) -> bool { false }
    fn down_feature_vector(&self, _layer: usize, _feature: usize) -> Option<&[f32]> { None }
    fn has_down_features(&self) -> bool { false }
    fn down_layer_matrix(&self, _layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> { None }
    fn gate_scores_batch(&self, _layer: usize, _x: &Array2<f32>) -> Option<Array2<f32>> { None }
    /// Backend-aware variant of `gate_scores_batch`. When `backend` is a
    /// Metal `ComputeBackend` and `x` is a single row, implementations
    /// can dispatch `f32_gemv` instead of CPU BLAS — the gate matmul is
    /// the dominant per-layer cost on 31B decode (60 % of token time).
    /// Default implementation ignores the backend and calls the legacy
    /// method.
    fn gate_scores_batch_backend(
        &self,
        layer: usize,
        x: &Array2<f32>,
        _backend: Option<&dyn larql_compute::ComputeBackend>,
    ) -> Option<Array2<f32>> {
        self.gate_scores_batch(layer, x)
    }
    fn up_layer_matrix(&self, _layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> { None }
    fn has_full_mmap_ffn(&self) -> bool { false }
    fn has_interleaved(&self) -> bool { false }
    fn interleaved_gate(&self, _layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> { None }
    fn interleaved_up(&self, _layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> { None }
    fn interleaved_down(&self, _layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> { None }
    fn prefetch_interleaved_layer(&self, _layer: usize) {}
    fn has_interleaved_q4(&self) -> bool { false }
    fn interleaved_q4_gate(&self, _layer: usize) -> Option<ndarray::Array2<f32>> { None }
    fn interleaved_q4_up(&self, _layer: usize) -> Option<ndarray::Array2<f32>> { None }
    fn interleaved_q4_down(&self, _layer: usize) -> Option<ndarray::Array2<f32>> { None }
    fn prefetch_interleaved_q4_layer(&self, _layer: usize) {}
    fn interleaved_q4_mmap_ref(&self) -> Option<&[u8]> { None }
    fn has_interleaved_q4k(&self) -> bool { false }
    fn interleaved_q4k_mmap_ref(&self) -> Option<&[u8]> { None }
    /// Per-layer FFN Q4_K/Q6_K slices — [gate, up, down] with format tags.
    /// `None` when the FFN manifest wasn't emitted (older vindexes).
    fn interleaved_q4k_layer_data(&self, _layer: usize) -> Option<[(&[u8], &str); 3]> { None }

    /// Dequantised Q4K/Q6K FFN matrix for `(layer, component)` where
    /// `component` is 0=gate, 1=up, 2=down. Lazily decoded and cached.
    /// Returns `None` when the vindex has no Q4K interleaved data.
    fn q4k_ffn_layer(&self, _layer: usize, _component: usize)
        -> Option<std::sync::Arc<Vec<f32>>> { None }

    /// Decode one row of a Q4K FFN matrix without caching. Small-memory
    /// alternative to `q4k_ffn_layer`. See `VectorIndex::q4k_ffn_row_into`.
    fn q4k_ffn_row_into(&self, _layer: usize, _component: usize, _feat: usize, _out: &mut [f32]) -> bool {
        false
    }

    /// Fused Q4K/Q6K decode + dot — returns `dot(dequant(row), x)` without
    /// materialising the decoded row. See `VectorIndex::q4k_ffn_row_dot`.
    fn q4k_ffn_row_dot(&self, _layer: usize, _component: usize, _feat: usize, _x: &[f32]) -> Option<f32> {
        None
    }

    /// TEMP diagnostic — route row-dot through full-layer cache.
    fn q4k_ffn_row_dot_via_cache(&self, _layer: usize, _component: usize, _feat: usize, _x: &[f32]) -> Option<f32> {
        None
    }
    fn q4k_ffn_row_scaled_add_via_cache(&self, _layer: usize, _component: usize, _feat: usize, _alpha: f32, _out: &mut [f32]) -> bool {
        false
    }

    /// Fused Q4K/Q6K decode + scaled-add — `out += alpha * dequant(row)`
    /// without materialising the decoded row.
    fn q4k_ffn_row_scaled_add(&self, _layer: usize, _component: usize, _feat: usize, _alpha: f32, _out: &mut [f32]) -> bool {
        false
    }

    // ── FP4 / FP8 FFN storage (exp 26) ─────────────────────────────────────
    //
    // These mirror the `q4k_ffn_row_*` family for the FP4 block format. All
    // default to "no data" so overlays / GateIndex impls that don't carry
    // FP4 storage work unchanged.

    /// Whether this index has FP4/FP8 FFN storage attached.
    fn has_fp4_storage(&self) -> bool { false }

    /// FP4/FP8 fused dequant + dot. `component`: 0=gate, 1=up, 2=down.
    fn fp4_ffn_row_dot(&self, _layer: usize, _component: usize, _feat: usize, _x: &[f32]) -> Option<f32> {
        None
    }

    /// FP4/FP8 fused dequant + scaled-add: `out += alpha * dequant(row)`.
    fn fp4_ffn_row_scaled_add(&self, _layer: usize, _component: usize, _feat: usize, _alpha: f32, _out: &mut [f32]) -> bool {
        false
    }

    /// FP4/FP8 dequantise one row into `out`.
    fn fp4_ffn_row_into(&self, _layer: usize, _component: usize, _feat: usize, _out: &mut [f32]) -> bool {
        false
    }

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
    fn ffn_row_scaled_add(&self, layer: usize, component: usize, feat: usize, alpha: f32, out: &mut [f32]) -> bool {
        if self.has_fp4_storage()
            && self.fp4_ffn_row_scaled_add(layer, component, feat, alpha, out) {
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
            // Q4K down is stored transposed — per-row decode reads
            // hidden-dim rows, not feature vectors. Use the cached
            // whole-layer decode path for down; direct row decode for gate/up.
            if component == 2 {
                return self.q4k_ffn_row_scaled_add_via_cache(layer, component, feat, alpha, out);
            }
            return self.q4k_ffn_row_scaled_add(layer, component, feat, alpha, out);
        }
        false
    }

    /// Unified decode-into-buffer. `out.len()` must equal the row width.
    fn ffn_row_into(&self, layer: usize, component: usize, feat: usize, out: &mut [f32]) -> bool {
        if self.has_fp4_storage()
            && self.fp4_ffn_row_into(layer, component, feat, out) {
            return true;
        }
        let copy_row = |row: ndarray::ArrayView1<'_, f32>, out: &mut [f32]| -> bool {
            if row.len() != out.len() { return false; }
            for (i, &v) in row.iter().enumerate() { out[i] = v; }
            true
        };
        match component {
            0 => {
                if let Some(m) = self.interleaved_gate(layer) {
                    if feat < m.nrows() { return copy_row(m.row(feat), out); }
                }
            }
            1 => {
                if let Some(m) = self.interleaved_up(layer) {
                    if feat < m.nrows() { return copy_row(m.row(feat), out); }
                }
                if let Some(m) = self.up_layer_matrix(layer) {
                    if feat < m.nrows() { return copy_row(m.row(feat), out); }
                }
            }
            2 => {
                if let Some(row) = self.down_feature_vector(layer, feat) {
                    return copy_row(ndarray::ArrayView1::from(row), out);
                }
                if let Some(m) = self.interleaved_down(layer) {
                    if feat < m.nrows() { return copy_row(m.row(feat), out); }
                }
                if let Some(m) = self.down_layer_matrix(layer) {
                    if feat < m.nrows() { return copy_row(m.row(feat), out); }
                }
            }
            _ => return false,
        }
        if self.has_interleaved_q4k() {
            return self.q4k_ffn_row_into(layer, component, feat, out);
        }
        false
    }

    /// Direct Q4K/Q6K matmul — `Y = X @ W.T` against the layer's Q4K bytes.
    /// See `VectorIndex::q4k_matmul_transb`. `x` is `[x_rows, w_cols]`.
    /// `backend` (when provided) routes through Metal/CPU-SIMD kernels.
    fn q4k_matmul_transb(
        &self,
        _layer: usize,
        _component: usize,
        _x: &[f32],
        _x_rows: usize,
        _backend: Option<&dyn larql_compute::ComputeBackend>,
    ) -> Option<Vec<f32>> {
        None
    }

    /// Gate KNN via Q4 matvec — scored by a ComputeBackend.
    /// Returns None if Q4 gate data isn't loaded or backend doesn't support Q4.
    fn gate_knn_q4(
        &self,
        _layer: usize,
        _residual: &Array1<f32>,
        _top_k: usize,
        _backend: &dyn larql_compute::ComputeBackend,
    ) -> Option<Vec<(usize, f32)>> { None }

    /// Per-feature gate scoring: iterate all features, dot product each one.
    /// No matrix multiplication — each feature scored individually.
    /// Returns (feature_index, score) sorted by absolute score descending.
    fn gate_walk(&self, _layer: usize, _residual: &Array1<f32>, _top_k: usize) -> Option<Vec<(usize, f32)>> {
        None // Override in VectorIndex to use mmap
    }

    fn gate_knn_batch(&self, layer: usize, x: &Array2<f32>, top_k: usize) -> Vec<usize> {
        let seq_len = x.shape()[0];
        let mut all = std::collections::BTreeSet::new();
        for s in 0..seq_len {
            let row = x.row(s).to_owned();
            for (feat, _) in self.gate_knn(layer, &row, top_k) {
                all.insert(feat);
            }
        }
        all.into_iter().collect()
    }
}

/// Progress callbacks for index loading.
pub trait IndexLoadCallbacks {
    fn on_file_start(&mut self, _component: &str, _path: &str) {}
    fn on_progress(&mut self, _records: usize) {}
    fn on_file_done(&mut self, _component: &str, _records: usize, _elapsed_ms: f64) {}
}

pub struct SilentLoadCallbacks;
impl IndexLoadCallbacks for SilentLoadCallbacks {}

/// Per-layer gate vector offset info for mmap mode.
#[derive(Clone)]
pub struct GateLayerSlice {
    pub float_offset: usize,
    pub num_features: usize,
}

/// Per-layer Q4 gate data offset info.
#[derive(Clone)]
pub struct GateQ4Slice {
    pub byte_offset: usize,
    pub byte_len: usize,
    pub num_features: usize,
}

/// Mmap'd down_meta.bin — reads individual feature records on demand.
#[derive(Clone)]
pub struct DownMetaMmap {
    pub(crate) mmap: std::sync::Arc<memmap2::Mmap>,
    pub(crate) layer_offsets: Vec<usize>,
    pub(crate) layer_num_features: Vec<usize>,
    pub(crate) top_k_count: usize,
    pub(crate) tokenizer: std::sync::Arc<tokenizers::Tokenizer>,
}

impl DownMetaMmap {
    fn record_size(&self) -> usize {
        8 + self.top_k_count * 8
    }

    pub fn feature_meta(&self, layer: usize, feature: usize) -> Option<FeatureMeta> {
        if layer >= self.layer_offsets.len() { return None; }
        let num_features = self.layer_num_features[layer];
        if num_features == 0 || feature >= num_features { return None; }

        let offset = self.layer_offsets[layer] + feature * self.record_size();
        let rec_size = self.record_size();
        if offset + rec_size > self.mmap.len() { return None; }

        let b = &self.mmap[offset..offset + rec_size];
        let top_token_id = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        let c_score = f32::from_le_bytes([b[4], b[5], b[6], b[7]]);

        if top_token_id == 0 && c_score == 0.0 { return None; }

        let mut top_k = Vec::new();
        for i in 0..self.top_k_count {
            let o = 8 + i * 8;
            let tid = u32::from_le_bytes([b[o], b[o+1], b[o+2], b[o+3]]);
            let logit = f32::from_le_bytes([b[o+4], b[o+5], b[o+6], b[o+7]]);
            if tid > 0 || logit != 0.0 {
                let token = self.tokenizer.decode(&[tid], true)
                    .unwrap_or_else(|_| format!("T{tid}")).trim().to_string();
                top_k.push(TopKEntry { token, token_id: tid, logit });
            }
        }

        let top_token = self.tokenizer.decode(&[top_token_id], true)
            .unwrap_or_else(|_| format!("T{top_token_id}")).trim().to_string();

        Some(FeatureMeta { top_token, top_token_id, c_score, top_k })
    }

    pub fn num_features(&self, layer: usize) -> usize {
        self.layer_num_features.get(layer).copied().unwrap_or(0)
    }

    pub fn total_features(&self) -> usize {
        self.layer_num_features.iter().sum()
    }
}
