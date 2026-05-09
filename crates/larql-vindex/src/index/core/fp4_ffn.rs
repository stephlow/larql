//! `impl Fp4FfnAccess for VectorIndex`.
//!
//! Delegation shim for the FP4 (exp-26) FFN access paths.

use super::VectorIndex;
use crate::index::types::Fp4FfnAccess;

impl Fp4FfnAccess for VectorIndex {
    fn has_fp4_storage(&self) -> bool {
        VectorIndex::has_fp4_storage(self)
    }

    fn fp4_ffn_row_dot(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        x: &[f32],
    ) -> Option<f32> {
        VectorIndex::fp4_ffn_row_dot(self, layer, component, feat, x)
    }

    fn fp4_ffn_row_scaled_add(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        alpha: f32,
        out: &mut [f32],
    ) -> bool {
        VectorIndex::fp4_ffn_row_scaled_add(self, layer, component, feat, alpha, out)
    }

    fn fp4_ffn_row_into(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        out: &mut [f32],
    ) -> bool {
        VectorIndex::fp4_ffn_row_into(self, layer, component, feat, out)
    }
}
