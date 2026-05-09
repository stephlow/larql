//! `impl NativeFfnAccess for VectorIndex`.
//!
//! Delegation shim for f32-native FFN access — `down`/`up` row reads,
//! interleaved layer matrices, and the `prefetch_interleaved_layer`
//! madvise hint. Inherent methods live in `index::storage::ffn_store`.

use super::VectorIndex;
use crate::index::types::NativeFfnAccess;

impl NativeFfnAccess for VectorIndex {
    fn down_feature_vector(&self, layer: usize, feature: usize) -> Option<&[f32]> {
        self.down_feature_vector(layer, feature)
    }

    fn has_down_features(&self) -> bool {
        self.ffn.down_features_mmap.is_some()
    }

    fn down_layer_matrix(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.down_layer_matrix(layer)
    }

    fn up_layer_matrix(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.up_layer_matrix(layer)
    }

    fn has_full_mmap_ffn(&self) -> bool {
        self.has_full_mmap_ffn()
    }

    fn has_interleaved(&self) -> bool {
        self.has_interleaved()
    }

    fn interleaved_gate(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.interleaved_gate(layer)
    }

    fn interleaved_up(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.interleaved_up(layer)
    }

    fn interleaved_down(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.interleaved_down(layer)
    }

    fn prefetch_interleaved_layer(&self, layer: usize) {
        self.prefetch_interleaved_layer(layer)
    }
}
