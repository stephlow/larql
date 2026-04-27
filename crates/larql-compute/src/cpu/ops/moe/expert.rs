//! Per-expert gated-FFN execution (gate_proj, up_proj, activation, down_proj).
//!
//! Used by the in-process MoE forward pass (`cpu_moe_forward`) and by the
//! remote expert server endpoint when one expert's work is delegated to a
//! shard. The BF16 expert weights are dequantized on demand so only the
//! selected experts pay the conversion cost.

use super::cache::cached_dequant;
use super::math::{gelu_tanh, matmul_vec, rms_norm, silu};

/// Run a single expert's gated FFN given a pre-normed input vector.
///
/// `gate_up_bytes` and `down_bytes` carry exactly one expert's weights — the
/// caller picks the right per-expert byte range (per-layer `layers/{L}/{e}`
/// mmap entries or a stride into a legacy monolith). `format` tells the
/// dequantiser how to decode them. Returns the expert's output (not yet
/// weighted by router probability). `h_norm` must already be RMS-normed —
/// use `run_single_expert_with_norm` when you have the raw residual.
#[allow(clippy::too_many_arguments)]
pub fn run_single_expert(
    h_norm: &[f32],
    gate_up_bytes: &[u8],
    down_bytes: &[u8],
    inter: usize,
    format: crate::QuantFormat,
    activation: crate::Activation,
) -> Vec<f32> {
    let hidden = h_norm.len();
    if inter == 0 || hidden == 0 {
        return vec![0.0f32; hidden];
    }

    // Storage layout (matches `format/weights/write_layers.rs::quantize_moe_entries`):
    //   gate_up: [2*inter, hidden]              never padded
    //   down:    [hidden, inter_padded]         Q4_K pads inter→256 multiple
    // BF16 has no padding for either. See `forward::cpu_moe_forward` for the
    // expanded explanation; this single-expert path mirrors it exactly so the
    // remote-expert HTTP endpoint and local in-process MoE share the same
    // numerics.
    let inter_padded = match format {
        crate::QuantFormat::Q4_K => {
            let block = larql_models::quant::ggml::Q4_K_BLOCK_ELEMS;
            inter.div_ceil(block) * block
        }
        _ => inter,
    };

    let gate_up_w = cached_dequant(gate_up_bytes, format, 2 * inter * hidden);
    if gate_up_w.is_empty() {
        return vec![0.0f32; hidden];
    }
    let gate_w = &gate_up_w[..inter * hidden];
    let up_w = &gate_up_w[inter * hidden..2 * inter * hidden];

    let gate_out = matmul_vec(h_norm, gate_w, inter, hidden);
    let up_out = matmul_vec(h_norm, up_w, inter, hidden);

    // Build inner activation at `inter_padded` so the down matmul (which
    // expects `inter_padded` columns under Q4_K) sees zero in the padding.
    let mut hidden_state: Vec<f32> = vec![0.0f32; inter_padded];
    for j in 0..inter {
        let g = gate_out[j];
        let u = up_out[j];
        hidden_state[j] = match activation {
            crate::Activation::GeluTanh => gelu_tanh(g) * u,
            _ => silu(g) * u,
        };
    }

    let down_w = cached_dequant(down_bytes, format, hidden * inter_padded);
    if down_w.is_empty() {
        return vec![0.0f32; hidden];
    }
    matmul_vec(&hidden_state, &down_w, hidden, inter_padded)
}

/// Apply pre-experts norm then run a single expert. Used by the remote
/// expert server endpoint where the raw residual arrives from the client.
#[allow(clippy::too_many_arguments)]
pub fn run_single_expert_with_norm(
    h: &[f32],
    gate_up_bytes: &[u8],
    down_bytes: &[u8],
    inter: usize,
    pre_experts_norm: &[f32],
    norm_offset: f32,
    eps: f32,
    format: crate::QuantFormat,
    activation: crate::Activation,
) -> Vec<f32> {
    let h_norm = rms_norm(h, pre_experts_norm, eps, norm_offset);
    run_single_expert(
        &h_norm,
        gate_up_bytes,
        down_bytes,
        inter,
        format,
        activation,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Activation, QuantFormat};

    // BF16 encoding for common values (little-endian: low byte first).
    fn bf16_bytes(v: f32) -> [u8; 2] {
        let bits = v.to_bits();
        let hi = (bits >> 16) as u16;
        hi.to_le_bytes()
    }

    fn fill_bf16(len: usize, val: f32) -> Vec<u8> {
        let b = bf16_bytes(val);
        let mut v = vec![0u8; len * 2];
        for i in 0..len {
            v[i * 2] = b[0];
            v[i * 2 + 1] = b[1];
        }
        v
    }

    #[test]
    fn zero_inter_returns_zero_vec() {
        let h = vec![1.0f32; 4];
        let out = run_single_expert(&h, &[], &[], 0, QuantFormat::BF16, Activation::Silu);
        assert_eq!(out, vec![0.0f32; 4]);
    }

    #[test]
    fn zero_hidden_returns_empty() {
        let h: Vec<f32> = vec![];
        let out = run_single_expert(&h, &[], &[], 0, QuantFormat::BF16, Activation::Silu);
        assert_eq!(out.len(), 0);
    }

    #[test]
    fn nonzero_weights_produce_nonzero_output() {
        let hidden = 4;
        let inter = 2;
        // One expert's worth of all-1.0 BF16 weights.
        let gate_up = fill_bf16(2 * inter * hidden, 1.0);
        let down = fill_bf16(hidden * inter, 1.0);
        let h = vec![1.0f32; hidden];
        let out = run_single_expert(
            &h,
            &gate_up,
            &down,
            inter,
            QuantFormat::BF16,
            Activation::Silu,
        );
        assert_eq!(out.len(), hidden);
        assert!(
            out.iter().any(|v| v.abs() > 0.01),
            "expected nonzero output, got {out:?}"
        );
    }

    #[test]
    fn with_norm_matches_manual_prenorm() {
        let hidden = 4;
        let inter = 2;
        let gate_up = fill_bf16(2 * inter * hidden, 1.0);
        let down = fill_bf16(hidden * inter, 1.0);
        let h = vec![1.0f32, 2.0, 3.0, 4.0];
        let norm_w = vec![1.0f32; hidden];
        let eps = 1e-6_f32;

        let rms = (h.iter().map(|v| v * v).sum::<f32>() / h.len() as f32 + eps).sqrt();
        let h_normed: Vec<f32> = h
            .iter()
            .zip(norm_w.iter())
            .map(|(&x, &w)| x / rms * w)
            .collect();

        let direct = run_single_expert(
            &h_normed,
            &gate_up,
            &down,
            inter,
            QuantFormat::BF16,
            Activation::Silu,
        );
        let via_norm = run_single_expert_with_norm(
            &h,
            &gate_up,
            &down,
            inter,
            &norm_w,
            0.0,
            eps,
            QuantFormat::BF16,
            Activation::Silu,
        );

        let max_diff: f32 = direct
            .iter()
            .zip(&via_norm)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(
            max_diff < 1e-4,
            "with_norm diverges from manual prenorm: max_diff={max_diff}"
        );
    }

    #[test]
    fn gelu_tanh_differs_from_silu() {
        let hidden = 4;
        let inter = 2;
        let gate_up = fill_bf16(2 * inter * hidden, 1.0);
        let down = fill_bf16(hidden * inter, 1.0);
        let h = vec![0.5f32; hidden];
        let silu_out = run_single_expert(
            &h,
            &gate_up,
            &down,
            inter,
            QuantFormat::BF16,
            Activation::Silu,
        );
        let gelu_out = run_single_expert(
            &h,
            &gate_up,
            &down,
            inter,
            QuantFormat::BF16,
            Activation::GeluTanh,
        );
        let max_diff: f32 = silu_out
            .iter()
            .zip(&gelu_out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(
            max_diff > 0.01,
            "SiLU and GeluTanh should diverge; max_diff={max_diff}"
        );
    }
}
