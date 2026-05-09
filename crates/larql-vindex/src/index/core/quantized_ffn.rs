//! `impl QuantizedFfnAccess for VectorIndex`.
//!
//! Delegation shim for the Q4-family FFN paths: legacy interleaved Q4,
//! interleaved Q4_K, the dequant cache, and the per-row matmul / dot /
//! scaled-add helpers. Two methods (`interleaved_q4_mmap_ref`,
//! `interleaved_q4k_mmap_ref`) are real implementations because they
//! reach directly into the `FfnStore` mmap handles.

use super::VectorIndex;
use crate::index::types::QuantizedFfnAccess;

impl QuantizedFfnAccess for VectorIndex {
    fn has_interleaved_q4(&self) -> bool {
        self.has_interleaved_q4()
    }

    fn interleaved_q4_gate(&self, layer: usize) -> Option<ndarray::Array2<f32>> {
        self.interleaved_q4_gate(layer)
    }

    fn interleaved_q4_up(&self, layer: usize) -> Option<ndarray::Array2<f32>> {
        self.interleaved_q4_up(layer)
    }

    fn interleaved_q4_down(&self, layer: usize) -> Option<ndarray::Array2<f32>> {
        self.interleaved_q4_down(layer)
    }

    fn prefetch_interleaved_q4_layer(&self, layer: usize) {
        self.prefetch_interleaved_q4_layer(layer)
    }

    fn interleaved_q4_mmap_ref(&self) -> Option<&[u8]> {
        self.ffn
            .interleaved_q4_mmap
            .as_ref()
            .map(|m| m.as_ref() as &[u8])
    }

    fn has_interleaved_q4k(&self) -> bool {
        self.has_interleaved_q4k()
    }

    fn interleaved_q4k_mmap_ref(&self) -> Option<&[u8]> {
        self.ffn
            .interleaved_q4k_mmap
            .as_ref()
            .map(|m| m.as_ref() as &[u8])
    }

    fn prefetch_interleaved_q4k_layer(&self, layer: usize) {
        self.prefetch_interleaved_q4k_layer(layer)
    }

    fn interleaved_q4k_layer_data(&self, layer: usize) -> Option<[(&[u8], &str); 3]> {
        VectorIndex::interleaved_q4k_layer_data(self, layer)
    }

    fn q4k_ffn_layer(&self, layer: usize, component: usize) -> Option<std::sync::Arc<Vec<f32>>> {
        VectorIndex::q4k_ffn_layer(self, layer, component)
    }

    fn q4k_ffn_row_into(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        out: &mut [f32],
    ) -> bool {
        VectorIndex::q4k_ffn_row_into(self, layer, component, feat, out)
    }

    fn q4k_ffn_row_dot(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        x: &[f32],
    ) -> Option<f32> {
        VectorIndex::q4k_ffn_row_dot(self, layer, component, feat, x)
    }

    fn q4k_ffn_row_scaled_add_via_cache(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        alpha: f32,
        out: &mut [f32],
    ) -> bool {
        VectorIndex::q4k_ffn_row_scaled_add_via_cache(self, layer, component, feat, alpha, out)
    }

    fn has_down_features_q4k(&self) -> bool {
        VectorIndex::has_down_features_q4k(self)
    }

    fn q4k_down_feature_scaled_add(
        &self,
        layer: usize,
        feat: usize,
        alpha: f32,
        out: &mut [f32],
    ) -> bool {
        VectorIndex::q4k_down_feature_scaled_add(self, layer, feat, alpha, out)
    }

    fn q4k_ffn_row_scaled_add(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        alpha: f32,
        out: &mut [f32],
    ) -> bool {
        VectorIndex::q4k_ffn_row_scaled_add(self, layer, component, feat, alpha, out)
    }

    fn q4k_matmul_transb(
        &self,
        layer: usize,
        component: usize,
        x: &[f32],
        x_rows: usize,
        backend: Option<&dyn larql_compute::ComputeBackend>,
    ) -> Option<Vec<f32>> {
        VectorIndex::q4k_matmul_transb(self, layer, component, x, x_rows, backend)
    }
}
