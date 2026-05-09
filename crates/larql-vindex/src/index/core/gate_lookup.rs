//! `impl GateLookup for VectorIndex`.
//!
//! Thin delegation shim over the inherent `gate_*` methods that live
//! on `VectorIndex` itself (defined in `index::compute::gate_knn` and
//! `index::storage::gate_accessors`). Keeping the trait impl separate
//! from the inherent impl makes the capability surface easy to read
//! without scrolling through the storage implementation.

use ndarray::{Array1, Array2};

use super::VectorIndex;
use crate::index::types::{FeatureMeta, GateLookup};

impl GateLookup for VectorIndex {
    fn gate_knn(&self, layer: usize, residual: &Array1<f32>, top_k: usize) -> Vec<(usize, f32)> {
        self.gate_knn(layer, residual, top_k)
    }

    fn feature_meta(&self, layer: usize, feature: usize) -> Option<FeatureMeta> {
        self.feature_meta(layer, feature)
    }

    fn num_features(&self, layer: usize) -> usize {
        self.num_features(layer)
    }

    fn gate_knn_batch(&self, layer: usize, x: &Array2<f32>, top_k: usize) -> Vec<usize> {
        self.gate_knn_batch(layer, x, top_k)
    }

    fn gate_knn_q4(
        &self,
        layer: usize,
        residual: &ndarray::Array1<f32>,
        top_k: usize,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Option<Vec<(usize, f32)>> {
        // Delegate to VectorIndex's existing gate_knn_q4 method
        VectorIndex::gate_knn_q4(self, layer, residual, top_k, backend)
    }

    fn gate_scores_batch(&self, layer: usize, x: &Array2<f32>) -> Option<Array2<f32>> {
        self.gate_scores_batch(layer, x)
    }

    fn gate_scores_batch_backend(
        &self,
        layer: usize,
        x: &Array2<f32>,
        backend: Option<&dyn larql_compute::ComputeBackend>,
    ) -> Option<Array2<f32>> {
        self.gate_scores_batch_backend(layer, x, backend)
    }
}
