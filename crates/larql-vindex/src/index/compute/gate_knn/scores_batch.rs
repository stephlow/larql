//! Full-batch score computation feeding the dispatch entry points.
//!
//! `gate_scores_batch` is the public API used by inference for the
//! seq_len-wide gate matmul; `gate_scores_batch_backend` adds the GPU
//! gemv fast path for single-row decode. The two private helpers
//! (`gate_scores_2d_gpu`, `gate_scores_2d_fast`) own the zero-copy
//! mmap/warmed slicing logic and the f16 lazy-decode cache.

use ndarray::{Array2, ArrayView2};

use crate::index::core::VectorIndex;
use crate::index::storage::gate_store::{gate_gemv_gpu, gate_matmul};

impl VectorIndex {
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
    pub(super) fn gate_scores_2d_fast(&self, layer: usize, x: &Array2<f32>) -> Option<Array2<f32>> {
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
}
