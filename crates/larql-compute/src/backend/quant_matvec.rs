//! `QuantMatVec` — quantised matrix × vector operations.
//!
//! [`Self::quant_matvec`] is the unified entry point — `out[N] = W[N, K] · x[K]`
//! with `W` in any [`crate::QuantFormat`]. Adding a new quant format
//! is one match arm in the default impl plus a kernel module.
//!
//! The legacy per-format helpers (`q4_matvec`, `q4k_matvec`,
//! `q6k_matvec`) stay around for hot-path callers that have already
//! pre-quantised their input — but new callers should reach for
//! `quant_matvec` (see ROADMAP P1a).

use crate::QuantFormat;

/// Quantised matvec primitives.
pub trait QuantMatVec {
    /// Format-dispatched matvec.
    ///
    /// `out[N] = W[N, K] · x[K]`. Q4_K / Q4_KF / Q6_K consume f32 input
    /// directly; Q4_0 / Q8_0 internally re-quantise `x` to Q8 (per-32
    /// f32-scaled int8) before dispatching the kernel.
    ///
    /// Returns `None` if the backend doesn't implement the format.
    fn quant_matvec(
        &self,
        format: QuantFormat,
        weights: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        match format {
            QuantFormat::Q4_K | QuantFormat::Q4_KF => {
                self.q4k_matvec(weights, x, num_rows, hidden)
            }
            QuantFormat::Q6_K => self.q6k_matvec(weights, x, num_rows, hidden),
            QuantFormat::Q4_0 | QuantFormat::Q8_0 => {
                let (q8_x, q8_scales) =
                    crate::cpu::ops::q4_common::quantize_to_q8(x);
                self.q4_matvec(weights, &q8_x, &q8_scales, num_rows, hidden)
            }
        }
    }

    // ── Per-format helpers ──
    //
    // These exist because the hot decode path pre-quantises its input
    // once and reuses it across many gate/up matvecs in a layer; the
    // unified `quant_matvec` re-quantises every call. Migration to a
    // pre-quantised path on `quant_matvec` is its own follow-up.

    /// Q4_0 × Q8 matvec. `Some` if the backend supports Q4_0.
    fn q4_matvec(
        &self,
        _q4_data: &[u8], _q8_x: &[i8], _q8_scales: &[f32],
        _num_rows: usize, _hidden: usize,
    ) -> Option<Vec<f32>> { None }

    /// Q4 vector-matrix: `out[K] = activation[N] @ Q4[N, K]`.
    fn q4_vecmat(
        &self,
        _activation: &[f32], _q4_data: &[u8],
        _intermediate: usize, _hidden: usize,
    ) -> Option<Vec<f32>> { None }

    /// Batched gate+up Q4 matvec for ALL seq positions in one submission.
    #[allow(clippy::type_complexity)]
    fn q4_matvec_pair_batch(
        &self,
        _gate_q4: &[u8], _up_q4: &[u8],
        _x_matrix: &[f32], _seq_len: usize,
        _num_rows: usize, _hidden: usize,
    ) -> Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)> { None }

    /// Q4_K matvec: `scores[N] = Q4_K[N, K] @ f32_x[K]`.
    fn q4k_matvec(
        &self,
        _q4k_data: &[u8], _x: &[f32],
        _num_rows: usize, _hidden: usize,
    ) -> Option<Vec<f32>> { None }

    /// Q6_K matvec: `scores[N] = Q6_K[N, K] @ f32_x[K]`.
    fn q6k_matvec(
        &self,
        _q6k_data: &[u8], _x: &[f32],
        _num_rows: usize, _hidden: usize,
    ) -> Option<Vec<f32>> { None }

    /// Whether this backend implements any Q4 fused operation.
    fn has_q4(&self) -> bool { false }
}
