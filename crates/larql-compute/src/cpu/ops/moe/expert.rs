//! Per-expert gated-FFN execution (gate_proj, up_proj, activation, down_proj).
//!
//! Used by the in-process MoE forward pass (`cpu_moe_forward`) and by the
//! remote expert server endpoint when one expert's work is delegated to a
//! shard. The BF16 expert weights are dequantized on demand so only the
//! selected experts pay the conversion cost.

use super::cache::cached_dequant;
use super::math::{gelu_tanh, matmul_vec, rms_norm, silu};

fn expert_byte_slice(packed: &[u8], expert_idx: usize, out_rows: usize, in_cols: usize) -> &[u8] {
    let bytes_per_expert = out_rows * in_cols * 2;
    let start = expert_idx * bytes_per_expert;
    &packed[start..start + bytes_per_expert]
}

/// Run a single expert's gated FFN given a pre-normed input vector.
///
/// Returns the expert's output (not yet weighted by router probability).
/// `h_norm` must already be RMS-normed — use `run_single_expert_with_norm`
/// when you have the raw residual.
pub fn run_single_expert(
    h_norm: &[f32],
    experts_gate_up: &[u8],
    experts_down: &[u8],
    expert_idx: usize,
    inter: usize,
    activation: crate::Activation,
) -> Vec<f32> {
    let hidden = h_norm.len();
    if inter == 0 || hidden == 0 { return vec![0.0f32; hidden]; }

    let gate_up_bytes = expert_byte_slice(experts_gate_up, expert_idx, 2 * inter, hidden);
    let gate_up_w = cached_dequant(gate_up_bytes);
    let gate_w = &gate_up_w[..inter * hidden];
    let up_w = &gate_up_w[inter * hidden..];

    let gate_out = matmul_vec(h_norm, gate_w, inter, hidden);
    let up_out = matmul_vec(h_norm, up_w, inter, hidden);

    let hidden_state: Vec<f32> = gate_out.iter().zip(up_out.iter())
        .map(|(&g, &u)| match activation {
            crate::Activation::GeluTanh => gelu_tanh(g) * u,
            _ => silu(g) * u,
        })
        .collect();

    let down_bytes = expert_byte_slice(experts_down, expert_idx, hidden, inter);
    let down_w = cached_dequant(down_bytes);
    matmul_vec(&hidden_state, &down_w, hidden, inter)
}

/// Apply pre-experts norm then run a single expert. Used by the remote
/// expert server endpoint where the raw residual arrives from the client.
#[allow(clippy::too_many_arguments)]
pub fn run_single_expert_with_norm(
    h: &[f32],
    experts_gate_up: &[u8],
    experts_down: &[u8],
    expert_idx: usize,
    inter: usize,
    pre_experts_norm: &[f32],
    norm_offset: f32,
    eps: f32,
    activation: crate::Activation,
) -> Vec<f32> {
    let h_norm = rms_norm(h, pre_experts_norm, eps, norm_offset);
    run_single_expert(&h_norm, experts_gate_up, experts_down, expert_idx, inter, activation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Activation;

    // BF16 encoding for common values (little-endian: low byte first).
    fn bf16_bytes(v: f32) -> [u8; 2] {
        let bits = v.to_bits();
        let hi = (bits >> 16) as u16;
        hi.to_le_bytes()
    }

    fn fill_bf16(len: usize, val: f32) -> Vec<u8> {
        let b = bf16_bytes(val);
        let mut v = vec![0u8; len * 2];
        for i in 0..len { v[i * 2] = b[0]; v[i * 2 + 1] = b[1]; }
        v
    }

    #[test]
    fn zero_inter_returns_zero_vec() {
        let h = vec![1.0f32; 4];
        let out = run_single_expert(&h, &[], &[], 0, 0, Activation::Silu);
        assert_eq!(out, vec![0.0f32; 4]);
    }

    #[test]
    fn zero_hidden_returns_empty() {
        let h: Vec<f32> = vec![];
        let out = run_single_expert(&h, &[], &[], 0, 0, Activation::Silu);
        assert_eq!(out.len(), 0);
    }

    #[test]
    fn nonzero_weights_produce_nonzero_output() {
        let hidden = 4;
        let inter = 2;
        // gate_up: [2*inter, hidden], down: [hidden, inter] — all 1.0 BF16
        let gate_up = fill_bf16(2 * inter * hidden, 1.0);
        let down = fill_bf16(hidden * inter, 1.0);
        let h = vec![1.0f32; hidden];
        let out = run_single_expert(&h, &gate_up, &down, 0, inter, Activation::Silu);
        assert_eq!(out.len(), hidden);
        assert!(out.iter().any(|v| v.abs() > 0.01), "expected nonzero output, got {out:?}");
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

        // Manually apply RMS norm: h_norm[i] = h[i] / rms * w[i]
        let rms = (h.iter().map(|v| v * v).sum::<f32>() / h.len() as f32 + eps).sqrt();
        let h_normed: Vec<f32> = h.iter().zip(norm_w.iter()).map(|(&x, &w)| x / rms * w).collect();

        let direct = run_single_expert(&h_normed, &gate_up, &down, 0, inter, Activation::Silu);
        let via_norm = run_single_expert_with_norm(&h, &gate_up, &down, 0, inter, &norm_w, 0.0, eps, Activation::Silu);

        let max_diff: f32 = direct.iter().zip(&via_norm).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(max_diff < 1e-4, "with_norm diverges from manual prenorm: max_diff={max_diff}");
    }

    #[test]
    fn gelu_tanh_differs_from_silu() {
        // Use h = [0.5; 4]: gate_out = 2.0 per row, where silu(2) ≠ gelu_tanh(2)
        let hidden = 4;
        let inter = 2;
        let gate_up = fill_bf16(2 * inter * hidden, 1.0);
        let down = fill_bf16(hidden * inter, 1.0);
        let h = vec![0.5f32; hidden];
        let silu_out = run_single_expert(&h, &gate_up, &down, 0, inter, Activation::Silu);
        let gelu_out = run_single_expert(&h, &gate_up, &down, 0, inter, Activation::GeluTanh);
        let max_diff: f32 = silu_out.iter().zip(&gelu_out).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(max_diff > 0.01, "SiLU and GeluTanh should diverge; max_diff={max_diff}");
    }
}
