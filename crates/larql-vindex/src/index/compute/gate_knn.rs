//! Gate KNN dispatch — brute-force, batched, and HNSW. Storage-side
//! resolution (mmap fast path, decode caches, LRU bookkeeping) lives
//! in `crate::index::storage::gate_store`; this module only orchestrates
//! the dot-product → top-K compute.

use ndarray::{Array1, Array2, ArrayView2};

use crate::index::core::VectorIndex;
use crate::index::storage::gate_store::{gate_gemv_gpu, gate_matmul, gemv};
use crate::index::types::*;

/// Gate KNN methods for VectorIndex.
impl VectorIndex {
    /// Gate KNN: find the top-K features at a layer whose gate vectors have
    /// the highest dot product with the input residual. Uses BLAS matmul.
    ///
    /// In mmap mode, slices directly from the mmap'd file — zero heap allocation.
    /// Returns (feature_index, dot_product) sorted by absolute magnitude descending.
    pub fn gate_knn(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        // HNSW path
        if self
            .gate
            .hnsw_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            if let Some(results) = self.gate_knn_hnsw(layer, residual, top_k) {
                return results;
            }
        }

        // Fast path: f32 mmap zero-copy (no allocation, no clone)
        if let Some(scores) = self.gate_knn_mmap_fast(layer, residual) {
            return Self::top_k_from_scores(&scores, top_k);
        }

        // Fallback: resolve_gate (copies data for heap/f16 paths)
        let gate = match self.resolve_gate(layer) {
            Some(g) => g,
            None => return vec![],
        };
        let view = gate.view(self.hidden_size);
        let scores = gemv(&view, residual);
        Self::top_k_from_scores(&scores, top_k)
    }

    /// Batched gate walk: scores all features via a single BLAS `gemv`, then
    /// extracts the top-K. Despite the name, this is batched matrix-vector —
    /// see [`Self::gate_walk_pure`] for a true per-feature implementation.
    pub fn gate_walk(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        top_k: usize,
    ) -> Option<Vec<(usize, f32)>> {
        let num_features = self.num_features(layer);
        if num_features == 0 {
            return None;
        }

        // Get gate data as contiguous f32 (from mmap or warmed cache)
        let gate_data: &[f32];
        let _owned: Vec<f32>;

        // Try zero-copy f32 mmap first
        let mmap_slice = if self.gate.gate_mmap_dtype == crate::config::dtype::StorageDtype::F32 {
            self.gate.gate_mmap_bytes.as_ref().and_then(|mmap| {
                let slice = self.gate.gate_mmap_slices.get(layer)?;
                if slice.num_features == 0 {
                    return None;
                }
                let byte_offset = slice.float_offset * 4;
                let byte_end = byte_offset + slice.num_features * self.hidden_size * 4;
                if byte_end > mmap.len() {
                    return None;
                }
                Some(unsafe {
                    std::slice::from_raw_parts(
                        mmap[byte_offset..byte_end].as_ptr() as *const f32,
                        slice.num_features * self.hidden_size,
                    )
                })
            })
        } else {
            None
        };

        if let Some(data) = mmap_slice {
            gate_data = data;
        } else {
            // Fallback: resolve gate (may clone)
            let gate = self.resolve_gate(layer)?;
            _owned = gate.data;
            gate_data = &_owned;
        }

        let hidden = self.hidden_size;

        // Single BLAS gemv: gate[N, hidden] × residual[hidden] → scores[N].
        let gate_view = ArrayView2::from_shape((num_features, hidden), gate_data).unwrap();
        let scores = gemv(&gate_view, residual);
        Some(Self::top_k_from_scores(&scores, top_k))
    }

    /// Gate KNN within a specific feature range (for MoE expert-scoped queries).
    /// Only computes dot products for features [feat_start..feat_end].
    /// Returns (global_feature_index, score) pairs.
    pub fn gate_knn_expert(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        feat_start: usize,
        feat_end: usize,
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        // If promoted to heap, use heap path
        if let Some(Some(ref matrix)) = self.gate.gate_vectors.get(layer) {
            let end = feat_end.min(matrix.shape()[0]);
            if feat_start >= end {
                return vec![];
            }
            let slice = matrix.slice(ndarray::s![feat_start..end, ..]);
            let scores = gemv(&slice, residual);
            let mut hits = Self::top_k_from_scores(&scores, top_k);
            for hit in &mut hits {
                hit.0 += feat_start;
            }
            return hits;
        }

        if let Some(ref mmap) = self.gate.gate_mmap_bytes {
            if let Some(slice) = self.gate.gate_mmap_slices.get(layer) {
                if slice.num_features == 0 || feat_start >= slice.num_features {
                    return vec![];
                }
                let end = feat_end.min(slice.num_features);
                let bpf = crate::config::dtype::bytes_per_float(self.gate.gate_mmap_dtype);

                // Compute byte range for just this expert's features
                let layer_byte_start = slice.float_offset * bpf;
                let expert_byte_start = layer_byte_start + feat_start * self.hidden_size * bpf;
                let expert_byte_end = layer_byte_start + end * self.hidden_size * bpf;
                let n_features = end - feat_start;

                if expert_byte_end > mmap.len() {
                    return vec![];
                }

                match self.gate.gate_mmap_dtype {
                    crate::config::dtype::StorageDtype::F32 => {
                        let data = unsafe {
                            let ptr =
                                mmap[expert_byte_start..expert_byte_end].as_ptr() as *const f32;
                            std::slice::from_raw_parts(ptr, n_features * self.hidden_size)
                        };
                        let view =
                            ndarray::ArrayView2::from_shape((n_features, self.hidden_size), data)
                                .unwrap();
                        let scores = gemv(&view, residual);
                        let mut hits = Self::top_k_from_scores(&scores, top_k);
                        // Offset indices to global feature space
                        for hit in &mut hits {
                            hit.0 += feat_start;
                        }
                        return hits;
                    }
                    crate::config::dtype::StorageDtype::F16 => {
                        let raw = &mmap[expert_byte_start..expert_byte_end];
                        let floats = larql_models::quant::half::decode_f16(raw);
                        let view = ndarray::ArrayView2::from_shape(
                            (n_features, self.hidden_size),
                            &floats,
                        )
                        .unwrap();
                        let scores = gemv(&view, residual);
                        let mut hits = Self::top_k_from_scores(&scores, top_k);
                        for hit in &mut hits {
                            hit.0 += feat_start;
                        }
                        return hits;
                    }
                }
            }
        }
        // Fallback: full KNN filtered (slower)
        self.gate_knn(layer, residual, top_k * 10)
            .into_iter()
            .filter(|(f, _)| *f >= feat_start && *f < feat_end)
            .take(top_k)
            .collect()
    }

    /// Pick the K scores with the largest absolute value out of N. Single
    /// scan with a min-heap of capacity K; allocation is O(K), not O(N).
    /// On Gemma 4B (N=10240, K=10, 34-layer walk) this is ~5.4 MB less
    /// allocation per token vs the previous Vec+select_nth approach. Mmap
    /// stays untouched — only the score-extract heap shrinks.
    pub(crate) fn top_k_from_scores(scores: &Array1<f32>, top_k: usize) -> Vec<(usize, f32)> {
        top_k_by_abs(scores.iter().copied(), top_k)
    }

    /// Full walk: gate KNN at each layer, annotated with down token metadata.
    pub fn walk(&self, residual: &Array1<f32>, layers: &[usize], top_k: usize) -> WalkTrace {
        let mut trace_layers = Vec::with_capacity(layers.len());

        for &layer in layers {
            let hits = self.gate_knn(layer, residual, top_k);
            let walk_hits: Vec<WalkHit> = hits
                .into_iter()
                .filter_map(|(feature, gate_score)| {
                    let meta = self.feature_meta(layer, feature)?;
                    Some(WalkHit {
                        layer,
                        feature,
                        gate_score,
                        meta,
                    })
                })
                .collect();
            trace_layers.push((layer, walk_hits));
        }

        WalkTrace {
            layers: trace_layers,
        }
    }

    /// Batched gate KNN: compute scores for ALL sequence positions in one BLAS gemm.
    ///
    /// Input: x is [seq_len, hidden]. Computes gate_vectors @ x^T = [features, seq_len].
    /// Returns the union of per-position top-K feature indices (sorted).
    /// One gemm replaces seq_len separate gemv calls.
    ///
    /// Per-position top-K extraction runs in parallel via rayon when
    /// `seq_len >= PARALLEL_TOPK_THRESHOLD` (16 — below that the rayon
    /// scheduling overhead matches or exceeds the per-position savings;
    /// at seq_len 64 the parallel branch saves ~7 % and at seq_len 256
    /// it saves ~24 % on Gemma-shape gates).
    pub fn gate_knn_batch(&self, layer: usize, x: &Array2<f32>, top_k: usize) -> Vec<usize> {
        let seq_len = x.shape()[0];
        if seq_len == 0 {
            return vec![];
        }

        // Fast path: zero-copy f32 mmap/warmed
        let scores_2d = if let Some(s) = self.gate_scores_2d_fast(layer, x) {
            s
        } else if let Some(gate) = self.resolve_gate(layer) {
            gate_matmul(&gate.view(self.hidden_size), &x.view())
        } else {
            return vec![];
        };

        // scores_2d is [num_features, seq_len].
        // For each position, take top-K features; union the indices.
        let num_features = scores_2d.shape()[0];
        let k = top_k.min(num_features);

        const PARALLEL_TOPK_THRESHOLD: usize = 16;
        let position_hits: Vec<Vec<usize>> = if seq_len >= PARALLEL_TOPK_THRESHOLD {
            use rayon::prelude::*;
            (0..seq_len)
                .into_par_iter()
                .map(|s| {
                    top_k_by_abs(scores_2d.column(s).iter().copied(), k)
                        .into_iter()
                        .map(|(idx, _)| idx)
                        .collect()
                })
                .collect()
        } else {
            (0..seq_len)
                .map(|s| {
                    top_k_by_abs(scores_2d.column(s).iter().copied(), k)
                        .into_iter()
                        .map(|(idx, _)| idx)
                        .collect()
                })
                .collect()
        };

        let mut feature_set = std::collections::BTreeSet::new();
        for hits in position_hits {
            feature_set.extend(hits);
        }
        feature_set.into_iter().collect()
    }

    // Feature store methods (load_down/up_features, down/up_layer_matrix, warmup)
    // are in feature_store.rs

    /// Compute gate scores for all features × all positions in one BLAS gemm.
    /// Returns [seq_len, intermediate] matrix = x @ gate_vectors^T.
    /// These scores are the gate projections — the same as x @ W_gate.T.
    pub fn gate_scores_batch(&self, layer: usize, x: &Array2<f32>) -> Option<Array2<f32>> {
        self.gate_scores_batch_backend(layer, x, None)
    }

    /// Backend-aware gate scores. When `backend` is present and `x` is
    /// a single row (seq_len == 1), route through `f32_gemv` — the
    /// same row-per-simdgroup path that closed lm_head. On Gemma 4 31B
    /// decode (hidden = 5376, ~18 K features, 60 layers) the CPU-BLAS
    /// path clocks ~4.3 ms/layer × 60 = 258 ms/token = 60 % of decode.
    /// Metal f32_gemv was measured at ~1 ms/layer on the lm_head of
    /// similar shape, so the upside is ~200 ms/token.
    pub fn gate_scores_batch_backend(
        &self,
        layer: usize,
        x: &Array2<f32>,
        backend: Option<&dyn larql_compute::ComputeBackend>,
    ) -> Option<Array2<f32>> {
        if x.shape()[0] == 0 {
            return None;
        }

        // Metal gemv fast path (decode / single-row prefill).
        if let Some(be) = backend {
            if x.shape()[0] == 1 {
                if let Some(scores_2d) = self.gate_scores_2d_gpu(layer, x, be) {
                    return Some(scores_2d.t().to_owned());
                }
            }
        }

        // BLAS paths — warmed f32 / mmap f32 / lazy-decoded f16.
        let scores_2d = if let Some(s) = self.gate_scores_2d_fast(layer, x) {
            s
        } else {
            let gate = self.resolve_gate(layer)?;
            gate_matmul(&gate.view(self.hidden_size), &x.view())
        };
        Some(scores_2d.t().to_owned())
    }

    /// Zero-copy GPU gate scores for f32 mmap/warmed, single-row `x`.
    /// Matches `gate_scores_2d_fast` shape contract: returns [N, 1].
    fn gate_scores_2d_gpu(
        &self,
        layer: usize,
        x: &Array2<f32>,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Option<Array2<f32>> {
        // Warmed cache (f32 heap).
        {
            let warmed = self.gate.warmed_gates.read().unwrap();
            if let Some(Some(ref data)) = warmed.get(layer) {
                let nf = self
                    .gate
                    .gate_mmap_slices
                    .get(layer)
                    .map(|s| s.num_features)
                    .unwrap_or(0);
                if nf > 0 {
                    let view =
                        ArrayView2::from_shape((nf, self.hidden_size), data.as_slice()).unwrap();
                    if let Some(scores) = gate_gemv_gpu(&view, &x.view(), backend) {
                        return Some(scores);
                    }
                }
            }
        }
        // f32 mmap (zero-copy, the production path for f32 gate vectors).
        if self.gate.gate_mmap_dtype == crate::config::dtype::StorageDtype::F32 {
            if let Some(ref mmap) = self.gate.gate_mmap_bytes {
                if let Some(slice) = self.gate.gate_mmap_slices.get(layer) {
                    if slice.num_features == 0 {
                        return None;
                    }
                    let byte_offset = slice.float_offset * 4;
                    let byte_end = byte_offset + slice.num_features * self.hidden_size * 4;
                    if byte_end > mmap.len() {
                        return None;
                    }
                    let data = unsafe {
                        let ptr = mmap[byte_offset..byte_end].as_ptr() as *const f32;
                        std::slice::from_raw_parts(ptr, slice.num_features * self.hidden_size)
                    };
                    let view = ArrayView2::from_shape((slice.num_features, self.hidden_size), data)
                        .unwrap();
                    if let Some(scores) = gate_gemv_gpu(&view, &x.view(), backend) {
                        return Some(scores);
                    }
                }
            }
        }
        // f16 mmap: zero-copy pass of raw f16 bytes to Metal's f16_gemv
        // shader, skipping the f16→f32 decode cache entirely. On 31B with
        // an ~18 K × 5376 gate matrix (387 MB f32, 194 MB f16) halving
        // the memory bandwidth is the difference between hitting the
        // CPU-BLAS ceiling and going faster on Metal.
        if self.gate.gate_mmap_dtype == crate::config::dtype::StorageDtype::F16 && x.shape()[0] == 1
        {
            let slice = self.gate.gate_mmap_slices.get(layer)?;
            if slice.num_features == 0 {
                return None;
            }
            let mmap = self.gate.gate_mmap_bytes.as_ref()?;
            let byte_offset = slice.float_offset * 2;
            let byte_end = byte_offset + slice.num_features * self.hidden_size * 2;
            if byte_end <= mmap.len() {
                let raw = &mmap[byte_offset..byte_end];
                let x_row = x.row(0);
                if let Some(x_slice) = x_row.as_slice() {
                    if let Some(scores) =
                        backend.f16_gemv_force(raw, x_slice, slice.num_features, self.hidden_size)
                    {
                        return Array2::from_shape_vec((slice.num_features, 1), scores).ok();
                    }
                }
            }
        }
        None
    }

    /// Zero-copy batch gate scores for f32 mmap/warmed — returns [features, seq].
    fn gate_scores_2d_fast(&self, layer: usize, x: &Array2<f32>) -> Option<Array2<f32>> {
        // Warmed cache
        {
            let warmed = self.gate.warmed_gates.read().unwrap();
            if let Some(Some(ref data)) = warmed.get(layer) {
                let nf = self
                    .gate
                    .gate_mmap_slices
                    .get(layer)
                    .map(|s| s.num_features)
                    .unwrap_or(0);
                if nf > 0 {
                    let view =
                        ArrayView2::from_shape((nf, self.hidden_size), data.as_slice()).unwrap();
                    return Some(gate_matmul(&view, &x.view()));
                }
            }
        }
        // f32 mmap
        if self.gate.gate_mmap_dtype == crate::config::dtype::StorageDtype::F32 {
            if let Some(ref mmap) = self.gate.gate_mmap_bytes {
                if let Some(slice) = self.gate.gate_mmap_slices.get(layer) {
                    if slice.num_features == 0 {
                        return None;
                    }
                    let byte_offset = slice.float_offset * 4;
                    let byte_end = byte_offset + slice.num_features * self.hidden_size * 4;
                    if byte_end > mmap.len() {
                        return None;
                    }
                    let data = unsafe {
                        let ptr = mmap[byte_offset..byte_end].as_ptr() as *const f32;
                        std::slice::from_raw_parts(ptr, slice.num_features * self.hidden_size)
                    };
                    let view = ArrayView2::from_shape((slice.num_features, self.hidden_size), data)
                        .unwrap();
                    return Some(gate_matmul(&view, &x.view()));
                }
            }
        }
        // f16 mmap — lazy decode into cache, then borrow (no per-call clone).
        // Holding the Mutex for the matmul is fine: forward passes are serial
        // per-layer, and this replaces a 462MB clone with a direct view.
        if self.gate.gate_mmap_dtype == crate::config::dtype::StorageDtype::F16 {
            let slice = self.gate.gate_mmap_slices.get(layer)?;
            if slice.num_features == 0 {
                return None;
            }
            let mmap = self.gate.gate_mmap_bytes.as_ref()?;
            let mut cache = self.gate.f16_decode_cache.lock().unwrap();
            if cache.len() <= layer {
                cache.resize(layer + 1, None);
            }
            let miss = cache[layer].is_none();
            if miss {
                let byte_offset = slice.float_offset * 2;
                let byte_end = byte_offset + slice.num_features * self.hidden_size * 2;
                if byte_end > mmap.len() {
                    return None;
                }
                let raw = &mmap[byte_offset..byte_end];
                cache[layer] = Some(larql_models::quant::half::decode_f16(raw));
            }
            self.touch_gate_cache_lru(layer, miss, &mut cache);
            let data = cache[layer].as_ref().unwrap();
            let view =
                ArrayView2::from_shape((slice.num_features, self.hidden_size), data.as_slice())
                    .unwrap();
            return Some(gate_matmul(&view, &x.view()));
        }
        None
    }

    /// Enable HNSW search. Indexes are built lazily on first query per layer.
    ///
    /// `ef_search`: beam width for search (50-200). Higher = better recall, slower.
    pub fn enable_hnsw(&self, ef_search: usize) {
        self.gate
            .hnsw_enabled
            .store(true, std::sync::atomic::Ordering::Relaxed);
        self.gate
            .hnsw_ef_search
            .store(ef_search, std::sync::atomic::Ordering::Relaxed);
    }

    /// Disable HNSW, revert to brute-force matmul.
    pub fn disable_hnsw(&self) {
        self.gate
            .hnsw_enabled
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether HNSW is currently enabled.
    pub fn is_hnsw_enabled(&self) -> bool {
        self.gate
            .hnsw_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get the gate vector matrix for a layer as owned contiguous f32.
    /// Used by HNSW build which needs owned data.
    fn gate_matrix_f32(&self, layer: usize) -> Option<(Vec<f32>, usize)> {
        let gate = self.resolve_gate(layer)?;
        Some((gate.data, gate.num_features))
    }

    /// Build a fresh HNSW for `layer` *without* holding the cache lock.
    /// Returns `None` when the layer has no gate data (caller decides
    /// what to do). Two callers race-safely concurrent on different
    /// layers since this never touches `hnsw_cache`.
    fn build_hnsw_layer(&self, layer: usize) -> Option<super::hnsw::HnswLayer> {
        let (data, num_features) = self.gate_matrix_f32(layer)?;
        let view = ArrayView2::from_shape((num_features, self.hidden_size), &data).unwrap();
        Some(super::hnsw::HnswLayer::build(&view, 8, 32))
    }

    /// Atomically install `hnsw` at `layer` if no other thread already
    /// did. A concurrent racer's index is dropped — the loss is one
    /// duplicated build, not a corrupted cache.
    fn install_hnsw_layer(&self, layer: usize, hnsw: super::hnsw::HnswLayer) {
        let mut cache = self.gate.hnsw_cache.lock().unwrap();
        if cache.len() <= layer {
            cache.resize_with(layer + 1, || None);
        }
        if cache[layer].is_none() {
            cache[layer] = Some(hnsw);
        }
    }

    /// Get or build the HNSW index for a layer (lazy). Holds the cache
    /// lock only briefly at check + install — the ~76 ms build itself
    /// runs lock-free, so concurrent KNN queries on other layers don't
    /// block on this layer's build.
    fn get_or_build_hnsw(&self, layer: usize) -> bool {
        {
            let cache = self.gate.hnsw_cache.lock().unwrap();
            if cache.get(layer).and_then(|s| s.as_ref()).is_some() {
                return true;
            }
        }
        let Some(hnsw) = self.build_hnsw_layer(layer) else {
            return false;
        };
        self.install_hnsw_layer(layer, hnsw);
        true
    }

    /// Eager-build HNSW for every layer, in parallel. One-shot startup
    /// helper for grid servers and interp pipelines that will query all
    /// layers — single call replaces N × ~76 ms lazy builds with one
    /// parallel batch (≈ 76 ms ÷ N_threads on the slowest layer's bound).
    /// Already-built layers are skipped.
    ///
    /// Holds the cache lock only at the snapshot + install boundaries;
    /// the per-layer build runs lock-free across rayon's pool. Memory
    /// note — each parallel build clones its layer's gate data
    /// (`gate_matrix_f32`), so peak transient RSS is ≈
    /// `min(num_layers, num_threads) × layer_gate_bytes`. Shrink with
    /// `rayon::ThreadPoolBuilder::num_threads(...).build_scoped(...)`
    /// if you need to bound it.
    pub fn warmup_hnsw_all_layers(&self) {
        use rayon::prelude::*;
        let num_layers = self.num_layers;
        let to_build: Vec<usize> = {
            let cache = self.gate.hnsw_cache.lock().unwrap();
            (0..num_layers)
                .filter(|&l| cache.get(l).and_then(|s| s.as_ref()).is_none())
                .collect()
        };
        if to_build.is_empty() {
            return;
        }
        let built: Vec<(usize, super::hnsw::HnswLayer)> = to_build
            .par_iter()
            .filter_map(|&l| self.build_hnsw_layer(l).map(|h| (l, h)))
            .collect();
        for (layer, hnsw) in built {
            self.install_hnsw_layer(layer, hnsw);
        }
    }

    /// Gate KNN via HNSW: graph search instead of brute-force matmul.
    ///
    /// Re-rank uses a zero-copy view onto the gate data when the layer
    /// is f32-mmap'd; only the f16-mmap and heap paths fall back to
    /// `gate_matrix_f32` (which clones). Dense 4B with f32 mmap pays
    /// only the search cost; the 100 MB-per-query clone is gone.
    ///
    /// **Ranking semantics.** The brute-force `gate_knn` path returns
    /// the top-K features by |dot| (absolute magnitude — matches the
    /// gate-activation strength regardless of sign). HNSW's internal
    /// rank is by signed dot, which would systematically drop
    /// large-negative features. We oversample HNSW (4× top_k) and then
    /// re-rank by abs at the seam to match the brute path's semantics.
    fn gate_knn_hnsw(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        top_k: usize,
    ) -> Option<Vec<(usize, f32)>> {
        if !self.get_or_build_hnsw(layer) {
            return None;
        }

        let ef = self
            .gate
            .hnsw_ef_search
            .load(std::sync::atomic::Ordering::Relaxed);
        // Oversample so the abs-rank seam below has signed candidates
        // from both tails to choose from.
        let hnsw_k = top_k.saturating_mul(4).max(top_k);
        let cache = self.gate.hnsw_cache.lock().unwrap();
        let hnsw = cache[layer].as_ref()?;

        let mut candidates = if self.gate.gate_mmap_dtype == crate::config::dtype::StorageDtype::F32
            && self.gate.gate_mmap_bytes.is_some()
        {
            // Zero-copy view onto f32-mmap.
            let mmap = self.gate.gate_mmap_bytes.as_ref().unwrap();
            let slice = self.gate.gate_mmap_slices.get(layer)?;
            if slice.num_features == 0 {
                return None;
            }
            let byte_offset = slice.float_offset * 4;
            let byte_end = byte_offset + slice.num_features * self.hidden_size * 4;
            if byte_end > mmap.len() {
                return None;
            }
            let data = unsafe {
                let ptr = mmap[byte_offset..byte_end].as_ptr() as *const f32;
                std::slice::from_raw_parts(ptr, slice.num_features * self.hidden_size)
            };
            let view =
                ArrayView2::from_shape((slice.num_features, self.hidden_size), data).unwrap();
            hnsw.search(&view, residual, hnsw_k, ef)
        } else {
            // Fallback (f16 mmap or heap): owned clone.
            let (data, num_features) = self.gate_matrix_f32(layer)?;
            let view = ArrayView2::from_shape((num_features, self.hidden_size), &data).unwrap();
            hnsw.search(&view, residual, hnsw_k, ef)
        };

        // Re-rank by |dot| to match brute-force semantics.
        candidates.sort_unstable_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(top_k);
        Some(candidates)
    }

    /// Adaptive gate KNN — automatically picks the fastest path per layer.
    ///
    /// Dispatch order:
    /// 1. Pinned Q4 → backend.q4_matvec (pre-loaded, no page faults)
    /// 2. Mmap Q4 → backend.q4_matvec (paged on demand)
    /// 3. f32 mmap/heap → BLAS brute-force (fallback)
    ///
    /// The residency manager tracks which layers are pinned.
    /// More memory budget → more pinned layers → faster walk.
    pub fn gate_knn_adaptive(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        top_k: usize,
        residency: &mut crate::index::storage::residency::ResidencyManager,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Vec<(usize, f32)> {
        residency.record_access(layer);

        // 1. Pinned Q4 (fastest — data already in RAM)
        if let Some(q4_data) = residency.pinned_q4(layer) {
            if backend.has_q4() {
                let x = residual.as_slice().unwrap();
                let (q8_x, q8_scales) = larql_compute::cpu::q4::quantize_to_q8(x);
                let num_features = self.num_features(layer);
                if let Some(scores_vec) =
                    backend.q4_matvec(q4_data, &q8_x, &q8_scales, num_features, self.hidden_size)
                {
                    return Self::top_k_from_scores(&Array1::from_vec(scores_vec), top_k);
                }
            }
        }

        // 2. Mmap Q4 (Q4 file loaded but not pinned — OS pages on demand)
        if let Some(hits) = self.gate_knn_q4(layer, residual, top_k, backend) {
            return hits;
        }

        // 3. f32 brute-force (fallback)
        self.gate_knn(layer, residual, top_k)
    }

    /// Gate KNN via Q4 matvec — scored by a ComputeBackend.
    ///
    /// The vindex provides the raw Q4 data. The backend scores it.
    /// Works with any backend: CPU C kernel, Metal GPU, CUDA, WASM.
    ///
    /// Returns None if Q4 gate data isn't loaded or backend doesn't support Q4.
    pub fn gate_knn_q4(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        top_k: usize,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Option<Vec<(usize, f32)>> {
        if !backend.has_q4() {
            return None;
        }
        let q4_data = self.gate_q4_data(layer)?;
        let slice = self.gate.gate_q4_slices.get(layer)?;
        if slice.num_features == 0 {
            return None;
        }

        let (q8_x, q8_scales) =
            larql_compute::cpu::q4::quantize_to_q8(residual.as_slice().unwrap());
        let scores_vec = backend.q4_matvec(
            q4_data,
            &q8_x,
            &q8_scales,
            slice.num_features,
            self.hidden_size,
        )?;

        let scores = Array1::from_vec(scores_vec);
        Some(Self::top_k_from_scores(&scores, top_k))
    }
}

/// Walk an iterator of f32 scores once, keep the K with largest |value|,
/// return them sorted by |value| descending (matching the prior Vec+select
/// behaviour at the call sites). Does not allocate beyond a `BinaryHeap`
/// of capacity K — for K=10 that's 240 B regardless of input length.
///
/// Panics on NaN inputs to preserve the previous `partial_cmp(...).unwrap()`
/// contract — gate scores from BLAS gemv are NaN-free as long as the
/// inputs are.
fn top_k_by_abs<I>(scores: I, top_k: usize) -> Vec<(usize, f32)>
where
    I: IntoIterator<Item = f32>,
{
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    if top_k == 0 {
        return Vec::new();
    }

    /// Wrapper that orders by `|val|`. Inverted `Ord` so `BinaryHeap`
    /// (max-heap by default) acts as a *min-heap on |val|*: `peek()`
    /// gives the smallest |val| currently in the heap, which is the
    /// candidate to evict when a bigger |val| arrives.
    #[derive(Copy, Clone)]
    struct AbsScore {
        idx: usize,
        val: f32,
    }
    impl PartialEq for AbsScore {
        fn eq(&self, other: &Self) -> bool {
            self.val.abs() == other.val.abs()
        }
    }
    impl Eq for AbsScore {}
    impl PartialOrd for AbsScore {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for AbsScore {
        fn cmp(&self, other: &Self) -> Ordering {
            // Reversed: smaller |val| ranks higher → max-heap pops it first.
            other.val.abs().partial_cmp(&self.val.abs()).unwrap()
        }
    }

    let mut heap: BinaryHeap<AbsScore> = BinaryHeap::with_capacity(top_k);
    for (i, v) in scores.into_iter().enumerate() {
        if heap.len() < top_k {
            heap.push(AbsScore { idx: i, val: v });
        } else if v.abs() > heap.peek().unwrap().val.abs() {
            heap.pop();
            heap.push(AbsScore { idx: i, val: v });
        }
    }

    let mut out: Vec<(usize, f32)> = heap.into_iter().map(|a| (a.idx, a.val)).collect();
    out.sort_unstable_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
    out
}

#[cfg(test)]
mod tests {
    use super::top_k_by_abs;
    use ndarray::Array1;

    #[test]
    fn top_k_by_abs_basic_ordering() {
        let scores: Vec<f32> = vec![0.1, -0.9, 0.5, 0.3];
        let result = top_k_by_abs(scores, 2);
        assert_eq!(result.len(), 2);
        // Top-2 by |val|: index 1 (|-0.9|=0.9) then index 2 (|0.5|=0.5).
        assert_eq!(result[0].0, 1);
        assert!((result[0].1 - (-0.9)).abs() < 1e-6);
        assert_eq!(result[1].0, 2);
    }

    #[test]
    fn top_k_by_abs_negative_values_selected_by_magnitude() {
        let scores: Vec<f32> = vec![1.0, -2.0, 0.5];
        let result = top_k_by_abs(scores, 1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 1); // |-2.0| is largest
    }

    #[test]
    fn top_k_by_abs_k_larger_than_input() {
        let scores: Vec<f32> = vec![1.0, 2.0];
        let result = top_k_by_abs(scores, 10);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn top_k_by_abs_k_zero_returns_empty() {
        let scores: Vec<f32> = vec![1.0, 2.0, 3.0];
        let result = top_k_by_abs(scores, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn top_k_by_abs_empty_input_returns_empty() {
        let result = top_k_by_abs(std::iter::empty::<f32>(), 5);
        assert!(result.is_empty());
    }

    #[test]
    fn top_k_by_abs_result_sorted_descending() {
        let scores: Vec<f32> = vec![0.3, 0.1, 0.9, 0.5, 0.7];
        let result = top_k_by_abs(scores, 3);
        assert_eq!(result.len(), 3);
        for w in result.windows(2) {
            assert!(w[0].1.abs() >= w[1].1.abs(), "not sorted: {:?}", result);
        }
    }

    #[test]
    fn top_k_from_scores_via_array1() {
        use crate::index::VectorIndex;
        let arr = Array1::from_vec(vec![0.1f32, -0.9, 0.5]);
        let result = VectorIndex::top_k_from_scores(&arr, 2);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, 1); // |-0.9| largest
    }
}
