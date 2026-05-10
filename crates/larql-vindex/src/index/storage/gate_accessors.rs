//! `VectorIndex` metadata + gate-vector accessors and one-time setup.
//!
//! Pulls the read-only getters and `warmup` out of `gate.rs` so the
//! KNN-dispatch file stays focused on hot-path search code.
//!
//! - `feature_meta`, `down_meta_at`, `loaded_layers`, `num_features`,
//!   `num_features_at`, `total_gate_vectors`, `total_down_meta`:
//!   metadata readers (heap + mmap aware).
//! - `gate_vector`, `gate_vectors_at`, `gate_vectors_flat`:
//!   raw gate-matrix accessors (heap + mmap, single-row + bulk).
//! - `warmup`: pre-decode f16 mmap to f32 once so per-query KNN avoids
//!   re-decoding on every dispatch.

use ndarray::Array2;

use crate::index::core::VectorIndex;
use crate::index::storage::vindex_storage::VindexStorage;
use crate::index::types::*;

impl VectorIndex {
    /// Look up metadata for a specific feature.
    /// Checks heap first (mutation overrides), then mmap (production read path).
    pub fn feature_meta(&self, layer: usize, feature: usize) -> Option<FeatureMeta> {
        // Heap path first — catches mutation overrides (INSERT/UPDATE)
        if let Some(meta) = self
            .metadata
            .down_meta
            .get(layer)
            .and_then(|v| v.as_ref())
            .and_then(|metas| metas.get(feature))
            .and_then(|m| m.clone())
        {
            return Some(meta);
        }
        // Mmap path (production — zero heap, no mutations)
        if let Some(ref dm) = self.metadata.down_meta_mmap {
            return dm.feature_meta(layer, feature);
        }
        None
    }

    /// Human-readable description of what the walk kernel will actually
    /// do on this vindex. Use to sanity-check a loaded vindex — if the
    /// description says "weights fallback" or "dense (legacy)", the
    /// vindex is not being used for FFN storage and that is probably
    /// not what the caller expected.
    ///
    /// Emitted by [`crate::format::load::load_vindex`] at load time
    /// when `LARQL_VINDEX_DESCRIBE=1` and by the CLI `--describe`
    /// flag. Also useful from tests to assert the expected storage
    /// backend is attached.
    pub fn describe_ffn_backend(&self) -> String {
        // Mirror the walk_ffn routing priority order (see
        // larql-inference::vindex::walk_ffn/mod.rs routing table).
        let mut parts = Vec::new();
        if self.ffn.fp4_storage.is_some() {
            let fp4 = self.ffn.fp4_storage.as_ref().unwrap();
            let g = fp4.manifest.projections.gate.precision;
            let u = fp4.manifest.projections.up.precision;
            let d = fp4.manifest.projections.down.precision;
            parts.push(format!("FP4 sparse (gate={g}, up={u}, down={d})"));
        }
        if self.storage.has_interleaved_q4k() {
            parts.push("Q4K interleaved".into());
        }
        if self.storage.has_interleaved_q4() {
            parts.push("Q4_0 interleaved".into());
        }
        if self.storage.has_interleaved_f32() {
            parts.push("f32 interleaved".into());
        }
        if self.storage.has_up_features() && self.storage.has_down_features() {
            parts.push("full mmap (up+down f32)".into());
        }
        if self.storage.has_gate_vectors() {
            parts.push(format!("gate KNN ({:?} mmap)", self.storage.gate_dtype()));
        }
        if parts.is_empty() {
            "weights fallback (safetensors — vindex not wired)".into()
        } else {
            parts.join(", ")
        }
    }

    /// Number of features indexed at a layer.
    ///
    /// Check order: legacy gate mmap slices → legacy heap gate vectors
    /// → FP4 storage's per-layer feature counts (exp 26). The FP4
    /// fallback fires when an FP4-only vindex has no legacy
    /// `gate_vectors.bin` mapped — without this, the walk kernel
    /// sees `num_features == 0` and falls through to the safetensors
    /// weights path, silently bypassing the vindex entirely.
    pub fn num_features(&self, layer: usize) -> usize {
        if self.storage.has_gate_vectors() {
            let n = self
                .storage
                .gate_layer_slice(layer)
                .map(|s| s.num_features)
                .unwrap_or(0);
            if n > 0 {
                return n;
            }
        }
        if let Some(n) = self
            .gate
            .gate_vectors
            .get(layer)
            .and_then(|v| v.as_ref())
            .map(|m| m.shape()[0])
        {
            if n > 0 {
                return n;
            }
        }
        // FP4 storage fallback — layer_features is populated from
        // `index.json.layers[]` at load time.
        if let Some(ref fp4) = self.ffn.fp4_storage {
            if let Some(&n) = fp4.layer_features.get(layer) {
                return n;
            }
        }
        0
    }

    /// Total gate vectors loaded across all layers.
    pub fn total_gate_vectors(&self) -> usize {
        if self.storage.has_gate_vectors() {
            return self
                .storage
                .gate_layer_slices()
                .iter()
                .map(|s| s.num_features)
                .sum();
        }
        self.gate
            .gate_vectors
            .iter()
            .filter_map(|v| v.as_ref())
            .map(|m| m.shape()[0])
            .sum()
    }

    /// Total down metadata entries loaded across all layers.
    pub fn total_down_meta(&self) -> usize {
        if let Some(ref dm) = self.metadata.down_meta_mmap {
            return dm.total_features();
        }
        self.metadata
            .down_meta
            .iter()
            .filter_map(|v| v.as_ref())
            .map(|metas| metas.iter().filter(|m| m.is_some()).count())
            .sum()
    }

    /// Layers that have gate vectors loaded.
    pub fn loaded_layers(&self) -> Vec<usize> {
        if self.storage.has_gate_vectors() {
            return self
                .storage
                .gate_layer_slices()
                .iter()
                .enumerate()
                .filter(|(_, s)| s.num_features > 0)
                .map(|(i, _)| i)
                .collect();
        }
        self.gate
            .gate_vectors
            .iter()
            .enumerate()
            .filter_map(|(i, v)| v.as_ref().map(|_| i))
            .collect()
    }

    /// Access down metadata for a specific layer.
    pub fn down_meta_at(&self, layer: usize) -> Option<&[Option<FeatureMeta>]> {
        self.metadata
            .down_meta
            .get(layer)
            .and_then(|v| v.as_ref())
            .map(|v| v.as_slice())
    }

    /// Access gate vectors matrix for a specific layer (heap mode only).
    /// Returns None in mmap mode — use gate_knn() directly instead.
    pub fn gate_vectors_at(&self, layer: usize) -> Option<&Array2<f32>> {
        self.gate.gate_vectors.get(layer).and_then(|v| v.as_ref())
    }

    /// Extract a single gate vector for a feature. Works in both heap and mmap mode.
    /// Returns the raw f32 vector (hidden_size elements).
    pub fn gate_vector(&self, layer: usize, feature: usize) -> Option<Vec<f32>> {
        // Heap path
        if let Some(Some(matrix)) = self.gate.gate_vectors.get(layer) {
            if feature < matrix.shape()[0] {
                return Some(matrix.row(feature).to_vec());
            }
            return None;
        }
        // Mmap path
        if let Some(view) = self.storage.gate_layer_view(layer) {
            if feature >= view.slice.num_features {
                return None;
            }
            let bpf = crate::config::dtype::bytes_per_float(view.dtype);
            let byte_offset = (view.slice.float_offset + feature * self.hidden_size) * bpf;
            let byte_count = self.hidden_size * bpf;
            let mmap: &[u8] = view.bytes.as_ref();
            if byte_offset + byte_count > mmap.len() {
                return None;
            }
            let raw = &mmap[byte_offset..byte_offset + byte_count];
            return Some(crate::config::dtype::decode_floats(raw, view.dtype));
        }
        None
    }

    /// Extract all gate vectors at a layer as flat f32 data.
    /// Returns (flat_data, num_features, hidden_size). Works in both heap and mmap mode.
    /// Use for bulk operations (SVD, PCA, numpy export).
    pub fn gate_vectors_flat(&self, layer: usize) -> Option<(Vec<f32>, usize, usize)> {
        // Heap path
        if let Some(Some(matrix)) = self.gate.gate_vectors.get(layer) {
            let (rows, cols) = (matrix.shape()[0], matrix.shape()[1]);
            if let Some(data) = matrix.as_slice() {
                return Some((data.to_vec(), rows, cols));
            }
            // Non-contiguous — copy row by row
            let mut data = Vec::with_capacity(rows * cols);
            for r in 0..rows {
                data.extend(matrix.row(r).iter());
            }
            return Some((data, rows, cols));
        }
        // Mmap path
        if let Some(view) = self.storage.gate_layer_view(layer) {
            if view.slice.num_features == 0 {
                return None;
            }
            let bpf = crate::config::dtype::bytes_per_float(view.dtype);
            let byte_offset = view.slice.float_offset * bpf;
            let byte_count = view.slice.num_features * self.hidden_size * bpf;
            let mmap: &[u8] = view.bytes.as_ref();
            if byte_offset + byte_count > mmap.len() {
                return None;
            }
            let raw = &mmap[byte_offset..byte_offset + byte_count];
            let data = crate::config::dtype::decode_floats(raw, view.dtype);
            return Some((data, view.slice.num_features, self.hidden_size));
        }
        None
    }

    /// Number of features at a layer (works in both heap and mmap mode).
    pub fn num_features_at(&self, layer: usize) -> usize {
        if self.storage.has_gate_vectors() {
            self.storage
                .gate_layer_slice(layer)
                .map(|s| s.num_features)
                .unwrap_or(0)
        } else {
            self.num_features(layer)
        }
    }

    /// Release (ask the kernel to evict) resident pages for every mmap'd
    /// file this index holds. Best-effort: calls `madvise(MADV_DONTNEED)`
    /// on each mapping. On Linux this immediately drops clean pages from
    /// RSS; on Darwin MADV_DONTNEED is advisory and the kernel may delay.
    ///
    /// Use when serving as a long-lived FFN endpoint with a hard RSS
    /// cap — the next request will re-fault whatever pages it needs.
    /// Layer sharding (`--layers`) is the preferred route because it
    /// prevents out-of-shard pages from ever being touched; this method
    /// is for single-shard-holds-everything topologies that still want
    /// to bound RSS between requests.
    pub fn release_mmap_pages(&self) {
        // Linux: MADV_DONTNEED immediately drops clean pages from RSS.
        // Darwin: MADV_DONTNEED is advisory for shared file-backed
        // mmap; the kernel may defer release until memory pressure.
        //
        // Every mmap-backed file in the vindex is registered with
        // `MmapStorage::mmap_handles` (the setters track them as they
        // install bytes). One call covers all of them.
        // Heap-backed entries (synth lm_head) aren't registered, so
        // iterating is safe.
        //
        // Safety: `unchecked_advise` requires no live references into
        // the mmap during the call. The server calls this from the
        // walk-ffn handler AFTER the per-request borrow of `patched`
        // (and any derived byte slices) has dropped.
        self.storage.release_pages();
    }

    /// Pre-decode f16 gate vectors to f32 for lock-free access.
    /// For f32 vindexes this is a no-op — the mmap path is already zero-copy.
    pub fn warmup(&self) {
        if self.storage.gate_dtype() == crate::config::dtype::StorageDtype::F32 {
            return;
        }

        let Some(bytes) = self.storage.gate_bytes_view() else {
            return;
        };
        let mmap: &[u8] = bytes.as_ref();
        let dtype = self.storage.gate_dtype();
        let mut warmed = self.gate.warmed_gates.write().unwrap();
        if warmed.len() < self.num_layers {
            warmed.resize_with(self.num_layers, || None);
        }
        for layer in 0..self.num_layers {
            if warmed[layer].is_some() {
                continue;
            }
            if let Some(slice) = self.storage.gate_layer_slice(layer) {
                if slice.num_features == 0 {
                    continue;
                }
                let bpf = crate::config::dtype::bytes_per_float(dtype);
                let byte_offset = slice.float_offset * bpf;
                let byte_count = slice.num_features * self.hidden_size * bpf;
                let byte_end = byte_offset + byte_count;
                if byte_end > mmap.len() {
                    continue;
                }
                let raw = &mmap[byte_offset..byte_end];
                warmed[layer] = Some(larql_models::quant::half::decode_f16(raw));
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════
// release_mmap_pages smoke test
//
// RSS assertions are intentionally avoided: MADV_DONTNEED is advisory
// on macOS, racy on Linux under memory pressure, and flaky in CI. The
// contract we can meaningfully assert is that the method doesn't
// panic and leaves the index usable for subsequent queries.
// ══════════════════════════════════════════════════════════════

#[cfg(test)]
mod release_mmap_pages_tests {
    use crate::config::dtype::StorageDtype;
    use crate::index::core::VectorIndex;
    use crate::index::types::GateLayerSlice;
    use ndarray::{Array1, Array2};

    #[test]
    fn release_mmap_pages_no_panic_on_heap_only_index() {
        // Heap-only index: no mmaps at all — release_mmap_pages must no-op.
        let hidden = 4;
        let gate0 = Array2::<f32>::zeros((2, hidden));
        let idx = VectorIndex::new(vec![Some(gate0)], vec![None], 1, hidden);
        assert!(!idx.is_mmap(), "heap-only index sanity check");
        // Must not panic — there are literally no mmaps to advise.
        idx.release_mmap_pages();
    }

    #[test]
    fn release_mmap_pages_no_panic_with_f16_gate_mmap() {
        // f16 mmap-backed index — exercises the `gate_mmap_bytes` arm
        // of `release_mmap_pages` on a valid mapping.
        let num_features = 2;
        let hidden = 4;
        let floats = num_features * hidden;
        let bytes = floats * 2;
        let mut anon = memmap2::MmapMut::map_anon(bytes).unwrap();
        let data = vec![1.0f32; floats];
        let encoded = larql_models::quant::half::encode_f16(&data);
        anon[..bytes].copy_from_slice(&encoded);
        let mmap = anon.make_read_only().unwrap();
        let slices = vec![GateLayerSlice {
            float_offset: 0,
            num_features,
        }];
        let idx = VectorIndex::new_mmap(mmap, slices, StorageDtype::F16, None, 1, hidden);
        assert!(idx.is_mmap(), "mmap-backed index sanity check");

        // Baseline query to force at least one page fault + cache decode.
        let q = Array1::from_vec(vec![1.0f32; hidden]);
        let _ = idx.gate_knn(0, &q, 1);

        // Must not panic — the mmap is live and held by Arc.
        idx.release_mmap_pages();

        // And the index must stay usable afterwards — `gate_knn` will
        // re-fault whatever pages the kernel actually evicted.
        let hits = idx.gate_knn(0, &q, 1);
        assert!(
            !hits.is_empty(),
            "gate_knn must still work after page release"
        );
    }
}

#[cfg(test)]
mod accessor_tests {
    //! Coverage for the read-only accessors: heap + mmap branches of
    //! `feature_meta` / `num_features` / `total_*` / `gate_vector` /
    //! `gate_vectors_flat` / `loaded_layers` / `down_meta_at` /
    //! `gate_vectors_at`, plus `describe_ffn_backend` and `warmup`.
    use super::*;
    use crate::config::dtype::StorageDtype;
    use crate::index::core::VectorIndex;
    use crate::index::types::GateLayerSlice;
    use larql_models::TopKEntry;
    use ndarray::Array2;

    fn meta(token: &str) -> FeatureMeta {
        FeatureMeta {
            top_token: token.into(),
            top_token_id: 1,
            c_score: 0.5,
            top_k: vec![TopKEntry {
                token: token.into(),
                token_id: 1,
                logit: 0.5,
            }],
        }
    }

    /// Build an f16-backed mmap from a flat f32 buffer.
    fn f16_mmap_from(floats: &[f32]) -> memmap2::Mmap {
        let bytes = floats.len() * 2;
        let mut anon = memmap2::MmapMut::map_anon(bytes).unwrap();
        let encoded = larql_models::quant::half::encode_f16(floats);
        anon[..bytes].copy_from_slice(&encoded);
        anon.make_read_only().unwrap()
    }

    // ── feature_meta ──

    #[test]
    fn feature_meta_returns_none_when_neither_path_populated() {
        let v = VectorIndex::empty(2, 4);
        assert!(v.feature_meta(0, 0).is_none());
    }

    #[test]
    fn feature_meta_uses_heap_path_when_down_meta_populated() {
        let mut v = VectorIndex::empty(2, 4);
        v.metadata.down_meta[0] = Some(vec![Some(meta("Paris")), None]);
        let m = v.feature_meta(0, 0).expect("heap meta present");
        assert_eq!(m.top_token, "Paris");
        // Sibling slot empty → None.
        assert!(v.feature_meta(0, 1).is_none());
    }

    #[test]
    fn feature_meta_returns_none_for_oob_layer() {
        let v = VectorIndex::empty(2, 4);
        assert!(v.feature_meta(99, 0).is_none());
    }

    // ── num_features ──

    #[test]
    fn num_features_returns_zero_for_empty_index() {
        let v = VectorIndex::empty(2, 4);
        for layer in 0..2 {
            assert_eq!(v.num_features(layer), 0);
        }
    }

    #[test]
    fn num_features_reads_heap_gate_shape() {
        let mut v = VectorIndex::empty(2, 4);
        v.gate.gate_vectors[0] = Some(Array2::<f32>::zeros((7, 4)));
        v.gate.gate_vectors[1] = Some(Array2::<f32>::zeros((3, 4)));
        assert_eq!(v.num_features(0), 7);
        assert_eq!(v.num_features(1), 3);
    }

    #[test]
    fn num_features_reads_mmap_slices() {
        let floats = vec![1.0_f32; 8];
        let mmap = f16_mmap_from(&floats);
        let slices = vec![
            GateLayerSlice {
                float_offset: 0,
                num_features: 4,
            },
            GateLayerSlice {
                float_offset: 16,
                num_features: 0,
            },
        ];
        let v = VectorIndex::new_mmap(mmap, slices, StorageDtype::F16, None, 2, 4);
        assert_eq!(v.num_features(0), 4);
        // Slice with num_features = 0 falls through to fp4 (None) → 0.
        assert_eq!(v.num_features(1), 0);
    }

    #[test]
    fn num_features_oob_layer_returns_zero() {
        let v = VectorIndex::empty(2, 4);
        assert_eq!(v.num_features(99), 0);
    }

    // ── total_gate_vectors / total_down_meta ──

    #[test]
    fn total_gate_vectors_sums_heap_layers() {
        let mut v = VectorIndex::empty(3, 4);
        v.gate.gate_vectors[0] = Some(Array2::<f32>::zeros((5, 4)));
        v.gate.gate_vectors[2] = Some(Array2::<f32>::zeros((7, 4)));
        assert_eq!(v.total_gate_vectors(), 12);
    }

    #[test]
    fn total_gate_vectors_sums_mmap_slices() {
        let floats = vec![1.0_f32; 16];
        let mmap = f16_mmap_from(&floats);
        let slices = vec![
            GateLayerSlice {
                float_offset: 0,
                num_features: 2,
            },
            GateLayerSlice {
                float_offset: 8,
                num_features: 2,
            },
        ];
        let v = VectorIndex::new_mmap(mmap, slices, StorageDtype::F16, None, 2, 4);
        assert_eq!(v.total_gate_vectors(), 4);
    }

    #[test]
    fn total_down_meta_counts_heap_metas() {
        let mut v = VectorIndex::empty(3, 4);
        v.metadata.down_meta[0] = Some(vec![Some(meta("a")), None, Some(meta("b"))]);
        v.metadata.down_meta[2] = Some(vec![Some(meta("c"))]);
        assert_eq!(v.total_down_meta(), 3);
    }

    #[test]
    fn total_down_meta_zero_when_empty() {
        let v = VectorIndex::empty(2, 4);
        assert_eq!(v.total_down_meta(), 0);
    }

    // ── loaded_layers ──

    #[test]
    fn loaded_layers_returns_indices_with_heap_gate() {
        let mut v = VectorIndex::empty(4, 4);
        v.gate.gate_vectors[0] = Some(Array2::<f32>::zeros((2, 4)));
        v.gate.gate_vectors[2] = Some(Array2::<f32>::zeros((2, 4)));
        v.gate.gate_vectors[3] = Some(Array2::<f32>::zeros((2, 4)));
        assert_eq!(v.loaded_layers(), vec![0, 2, 3]);
    }

    #[test]
    fn loaded_layers_filters_zero_feature_mmap_slices() {
        let floats = vec![1.0_f32; 16];
        let mmap = f16_mmap_from(&floats);
        let slices = vec![
            GateLayerSlice {
                float_offset: 0,
                num_features: 2,
            },
            GateLayerSlice {
                float_offset: 0,
                num_features: 0,
            }, // empty layer
            GateLayerSlice {
                float_offset: 8,
                num_features: 2,
            },
        ];
        let v = VectorIndex::new_mmap(mmap, slices, StorageDtype::F16, None, 3, 4);
        assert_eq!(v.loaded_layers(), vec![0, 2]);
    }

    #[test]
    fn loaded_layers_empty_when_nothing_loaded() {
        let v = VectorIndex::empty(3, 4);
        assert!(v.loaded_layers().is_empty());
    }

    // ── down_meta_at / gate_vectors_at ──

    #[test]
    fn down_meta_at_returns_layer_slice() {
        let mut v = VectorIndex::empty(2, 4);
        v.metadata.down_meta[1] = Some(vec![Some(meta("x"))]);
        assert!(v.down_meta_at(0).is_none());
        let slice = v.down_meta_at(1).expect("layer 1 present");
        assert_eq!(slice.len(), 1);
    }

    #[test]
    fn gate_vectors_at_returns_matrix_only_in_heap_mode() {
        let mut v = VectorIndex::empty(2, 4);
        v.gate.gate_vectors[0] = Some(Array2::<f32>::zeros((2, 4)));
        assert_eq!(v.gate_vectors_at(0).unwrap().shape(), &[2, 4]);
        assert!(v.gate_vectors_at(1).is_none());
        assert!(v.gate_vectors_at(99).is_none());
    }

    // ── gate_vector ──

    #[test]
    fn gate_vector_heap_returns_row() {
        let mut v = VectorIndex::empty(1, 4);
        let mut m = Array2::<f32>::zeros((3, 4));
        for j in 0..4 {
            m[[1, j]] = (j + 10) as f32;
        }
        v.gate.gate_vectors[0] = Some(m);
        let row = v.gate_vector(0, 1).unwrap();
        assert_eq!(row, vec![10.0, 11.0, 12.0, 13.0]);
    }

    #[test]
    fn gate_vector_heap_returns_none_for_oob_feature() {
        let mut v = VectorIndex::empty(1, 4);
        v.gate.gate_vectors[0] = Some(Array2::<f32>::zeros((2, 4)));
        assert!(v.gate_vector(0, 99).is_none());
    }

    #[test]
    fn gate_vector_returns_none_when_nothing_loaded() {
        let v = VectorIndex::empty(2, 4);
        assert!(v.gate_vector(0, 0).is_none());
    }

    #[test]
    fn gate_vector_mmap_returns_decoded_floats() {
        let floats = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2 features × 4 hidden
        let mmap = f16_mmap_from(&floats);
        let slices = vec![GateLayerSlice {
            float_offset: 0,
            num_features: 2,
        }];
        let v = VectorIndex::new_mmap(mmap, slices, StorageDtype::F16, None, 1, 4);
        let row = v.gate_vector(0, 1).unwrap();
        // f16 round-trip is lossy for 5..8 but they fit exactly.
        assert!((row[0] - 5.0).abs() < 1e-3);
        assert!((row[3] - 8.0).abs() < 1e-3);
    }

    #[test]
    fn gate_vector_mmap_returns_none_for_oob_feature() {
        let floats = vec![1.0_f32; 8];
        let mmap = f16_mmap_from(&floats);
        let slices = vec![GateLayerSlice {
            float_offset: 0,
            num_features: 2,
        }];
        let v = VectorIndex::new_mmap(mmap, slices, StorageDtype::F16, None, 1, 4);
        assert!(v.gate_vector(0, 99).is_none());
    }

    // ── gate_vectors_flat ──

    #[test]
    fn gate_vectors_flat_heap_returns_data_rows_cols() {
        let mut v = VectorIndex::empty(1, 4);
        let mut m = Array2::<f32>::zeros((2, 4));
        for r in 0..2 {
            for j in 0..4 {
                m[[r, j]] = (r * 10 + j) as f32;
            }
        }
        v.gate.gate_vectors[0] = Some(m);
        let (data, rows, cols) = v.gate_vectors_flat(0).unwrap();
        assert_eq!(rows, 2);
        assert_eq!(cols, 4);
        assert_eq!(data, vec![0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0]);
    }

    #[test]
    fn gate_vectors_flat_returns_none_for_unloaded_layer() {
        let v = VectorIndex::empty(2, 4);
        assert!(v.gate_vectors_flat(0).is_none());
    }

    #[test]
    fn gate_vectors_flat_mmap_returns_decoded_layer() {
        let floats = vec![1.0_f32; 8];
        let mmap = f16_mmap_from(&floats);
        let slices = vec![GateLayerSlice {
            float_offset: 0,
            num_features: 2,
        }];
        let v = VectorIndex::new_mmap(mmap, slices, StorageDtype::F16, None, 1, 4);
        let (data, rows, cols) = v.gate_vectors_flat(0).unwrap();
        assert_eq!(rows, 2);
        assert_eq!(cols, 4);
        assert_eq!(data.len(), 8);
    }

    #[test]
    fn gate_vectors_flat_mmap_returns_none_when_zero_features() {
        let floats = vec![1.0_f32; 4];
        let mmap = f16_mmap_from(&floats);
        let slices = vec![GateLayerSlice {
            float_offset: 0,
            num_features: 0,
        }];
        let v = VectorIndex::new_mmap(mmap, slices, StorageDtype::F16, None, 1, 4);
        assert!(v.gate_vectors_flat(0).is_none());
    }

    // ── num_features_at ──

    #[test]
    fn num_features_at_heap_path_matches_num_features() {
        let mut v = VectorIndex::empty(2, 4);
        v.gate.gate_vectors[0] = Some(Array2::<f32>::zeros((6, 4)));
        assert_eq!(v.num_features_at(0), 6);
        assert_eq!(v.num_features_at(1), 0);
    }

    #[test]
    fn num_features_at_mmap_path_uses_slice_count() {
        let floats = vec![1.0_f32; 16];
        let mmap = f16_mmap_from(&floats);
        let slices = vec![
            GateLayerSlice {
                float_offset: 0,
                num_features: 4,
            },
            GateLayerSlice {
                float_offset: 16,
                num_features: 0,
            },
        ];
        let v = VectorIndex::new_mmap(mmap, slices, StorageDtype::F16, None, 2, 4);
        assert_eq!(v.num_features_at(0), 4);
        assert_eq!(v.num_features_at(1), 0);
        // OOB layer → 0.
        assert_eq!(v.num_features_at(99), 0);
    }

    // ── describe_ffn_backend ──

    #[test]
    fn describe_ffn_backend_reports_weights_fallback_when_empty() {
        let v = VectorIndex::empty(1, 4);
        let s = v.describe_ffn_backend();
        assert!(s.contains("weights fallback"), "got: {s}");
    }

    #[test]
    fn describe_ffn_backend_reports_gate_mmap_dtype() {
        let floats = vec![1.0_f32; 4];
        let mmap = f16_mmap_from(&floats);
        let slices = vec![GateLayerSlice {
            float_offset: 0,
            num_features: 1,
        }];
        let v = VectorIndex::new_mmap(mmap, slices, StorageDtype::F16, None, 1, 4);
        let s = v.describe_ffn_backend();
        assert!(s.contains("gate KNN"), "got: {s}");
        assert!(s.contains("F16"), "got: {s}");
    }

    // ── warmup ──

    #[test]
    fn warmup_is_noop_for_f32_mmap() {
        // f32 path returns immediately — warmed_gates stays empty.
        let bytes = 16; // 4 floats × 4 bytes
        let anon = memmap2::MmapMut::map_anon(bytes).unwrap();
        let mmap = anon.make_read_only().unwrap();
        let slices = vec![GateLayerSlice {
            float_offset: 0,
            num_features: 1,
        }];
        let v = VectorIndex::new_mmap(mmap, slices, StorageDtype::F32, None, 1, 4);
        v.warmup();
        let warmed = v.gate.warmed_gates.read().unwrap();
        assert!(warmed.iter().all(|s| s.is_none()), "f32 path no-ops");
    }

    #[test]
    fn warmup_decodes_f16_into_warmed_gates() {
        let floats = vec![1.0_f32, 2.0, 3.0, 4.0]; // 1 feature × 4 hidden
        let mmap = f16_mmap_from(&floats);
        let slices = vec![GateLayerSlice {
            float_offset: 0,
            num_features: 1,
        }];
        let v = VectorIndex::new_mmap(mmap, slices, StorageDtype::F16, None, 1, 4);
        v.warmup();
        let warmed = v.gate.warmed_gates.read().unwrap();
        let layer0 = warmed[0].as_ref().expect("layer 0 warmed");
        assert_eq!(layer0.len(), 4);
        for (i, want) in [1.0_f32, 2.0, 3.0, 4.0].iter().enumerate() {
            assert!((layer0[i] - want).abs() < 1e-3, "f16 round-trip");
        }
    }

    #[test]
    fn warmup_skips_zero_feature_layers() {
        let floats = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mmap = f16_mmap_from(&floats);
        let slices = vec![
            GateLayerSlice {
                float_offset: 0,
                num_features: 1,
            },
            GateLayerSlice {
                float_offset: 0,
                num_features: 0,
            },
        ];
        let v = VectorIndex::new_mmap(mmap, slices, StorageDtype::F16, None, 2, 4);
        v.warmup();
        let warmed = v.gate.warmed_gates.read().unwrap();
        assert!(warmed[0].is_some());
        assert!(warmed[1].is_none(), "empty layer left None");
    }

    #[test]
    fn warmup_is_idempotent() {
        let floats = vec![1.0_f32; 4];
        let mmap = f16_mmap_from(&floats);
        let slices = vec![GateLayerSlice {
            float_offset: 0,
            num_features: 1,
        }];
        let v = VectorIndex::new_mmap(mmap, slices, StorageDtype::F16, None, 1, 4);
        v.warmup();
        v.warmup(); // second call short-circuits per layer
        let warmed = v.gate.warmed_gates.read().unwrap();
        assert!(warmed[0].is_some());
    }

    #[test]
    fn warmup_no_op_without_mmap() {
        // Heap-only index — no gate mmap → early return regardless
        // of dtype. After step 6 the dtype lives on `MmapStorage`,
        // not the substore; an empty storage stays at the F32 default
        // and the warmup early-returns on the dtype check anyway.
        let v = VectorIndex::empty(1, 4);
        v.warmup();
        let warmed = v.gate.warmed_gates.read().unwrap();
        assert!(warmed.iter().all(|s| s.is_none()));
    }
}
