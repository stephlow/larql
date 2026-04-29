//! Per-expert gated-FFN execution (gate_proj, up_proj, activation, down_proj).
//!
//! Used by the in-process MoE forward pass (`cpu_moe_forward`) and by the
//! remote expert server endpoint when one expert's work is delegated to a
//! shard. The BF16 expert weights are dequantized on demand so only the
//! selected experts pay the conversion cost.

use super::cache::cached_dequant;
use super::math::{gelu_tanh, matmul_vec, matmul_vec_into, rms_norm, silu};
use crate::cpu::ops::q4_common::q4k_matvec_into;

/// Per-call scratch for `run_single_expert_with_scratch` — preallocate once
/// per gRPC frame and reuse across all K active experts.  Keeps allocation
/// off the hot path: at Gemma 4 26B-A4B sizes the un-pooled version was
/// minting ~360 fresh ~11KB Vecs per token per shard.
///
/// Sized for one expert's worth of intermediate buffers.  Per-call cost on
/// reuse is O(0) — just zeros the activation buffer's padding columns.
pub struct ExpertScratch {
    /// `[inter]` — gate matvec output before activation.
    pub gate_out: Vec<f32>,
    /// `[inter]` — up matvec output.
    pub up_out: Vec<f32>,
    /// `[inter_padded]` — activation buffer fed into down.  Padding columns
    /// (`inter..inter_padded`) are zero-initialised once and re-used
    /// untouched across calls (down's matvec reads them as zero).
    pub act: Vec<f32>,
    /// `[hidden]` — final expert output.
    pub out: Vec<f32>,
}

impl ExpertScratch {
    /// Allocate scratch sized for `(hidden, inter, inter_padded)`.  Call
    /// once per gRPC frame; share `&mut` across the K experts.
    pub fn new(hidden: usize, inter: usize, inter_padded: usize) -> Self {
        Self {
            gate_out: vec![0.0f32; inter],
            up_out: vec![0.0f32; inter],
            act: vec![0.0f32; inter_padded],
            out: vec![0.0f32; hidden],
        }
    }
}

/// Apply pre_experts_norm once per frame and return the normed residual.
/// Hoisting this out of `run_single_expert*` saves K-1 redundant rms_norm
/// passes per layer (the input residual is identical for every expert in
/// the layer's top-K — they all receive the same h_norm by design).
pub fn pre_experts_norm(
    h: &[f32],
    pre_experts_norm: &[f32],
    norm_offset: f32,
    eps: f32,
) -> Vec<f32> {
    if pre_experts_norm.is_empty() {
        return h.to_vec();
    }
    rms_norm(h, pre_experts_norm, eps, norm_offset)
}

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

/// Allocation-free variant of `run_single_expert`: writes into the caller's
/// `ExpertScratch` instead of allocating gate / up / activation / output
/// buffers per call.  Used by the streaming expert server's hot path where
/// allocation churn would dominate at K=8 × 30 layers per token.
///
/// `h_norm` is already pre-normed (see `pre_experts_norm`).  Returns a
/// borrow of `scratch.out` so the caller can `clone_from_slice` into the
/// per-shard accumulator before reusing the scratch for the next expert.
#[allow(clippy::too_many_arguments)]
pub fn run_single_expert_into<'s>(
    scratch: &'s mut ExpertScratch,
    h_norm: &[f32],
    gate_up_bytes: &[u8],
    down_bytes: &[u8],
    inter: usize,
    format: crate::QuantFormat,
    activation: crate::Activation,
) -> &'s [f32] {
    let hidden = h_norm.len();
    if inter == 0 || hidden == 0 {
        for v in scratch.out.iter_mut() {
            *v = 0.0;
        }
        return &scratch.out;
    }

    let inter_padded = match format {
        crate::QuantFormat::Q4_K => {
            let block = larql_models::quant::ggml::Q4_K_BLOCK_ELEMS;
            inter.div_ceil(block) * block
        }
        _ => inter,
    };
    debug_assert_eq!(scratch.gate_out.len(), inter);
    debug_assert_eq!(scratch.up_out.len(), inter);
    debug_assert_eq!(scratch.act.len(), inter_padded);
    debug_assert_eq!(scratch.out.len(), hidden);

    // Per-stage timing: enabled by `LARQL_MOE_EXPERT_TIMING=1`.  Hot path
    // gate; the env-var check is cached in TLS to avoid a syscall per call.
    thread_local! {
        static EXPERT_TIMING: bool =
            std::env::var("LARQL_MOE_EXPERT_TIMING").is_ok();
    }
    let timing = EXPERT_TIMING.with(|t| *t);
    let mut t = std::time::Instant::now();

    // Q4_K direct matvec is available via `LARQL_Q4K_DIRECT=1` but stays
    // OFF by default — on Apple Silicon the scalar inner loop loses to
    // BLAS sgemv on cached f32 weights (BLAS uses AMX, ~5× more compute
    // throughput than scalar Rust).  Will become the right default once
    // we ship a NEON-vectorized version.
    thread_local! {
        static Q4K_DIRECT: bool =
            std::env::var("LARQL_Q4K_DIRECT").is_ok();
    }
    let q4k_direct = Q4K_DIRECT.with(|v| *v);
    let q4k_path = q4k_direct && matches!(format, crate::QuantFormat::Q4_K);

    let gate_w_size = inter * hidden;
    let gate_up_w_f32 = if q4k_path {
        Vec::new()
    } else {
        let v = cached_dequant(gate_up_bytes, format, 2 * inter * hidden);
        if v.is_empty() {
            for v in scratch.out.iter_mut() {
                *v = 0.0;
            }
            return &scratch.out;
        }
        v.to_vec()
    };
    let t_cache_gu = if timing { Some(t.elapsed()) } else { None };
    if timing { t = std::time::Instant::now(); }

    if q4k_path {
        let row_block_bytes = (hidden / 256) * 144;
        let half = inter * row_block_bytes;
        let gate_bytes = &gate_up_bytes[..half];
        let up_bytes = &gate_up_bytes[half..2 * half];
        q4k_matvec_into(&mut scratch.gate_out, h_norm, gate_bytes, inter, hidden);
        let t_gate = if timing { Some(t.elapsed()) } else { None };
        if timing { t = std::time::Instant::now(); }
        q4k_matvec_into(&mut scratch.up_out, h_norm, up_bytes, inter, hidden);
        let t_up = if timing { Some(t.elapsed()) } else { None };
        if timing { t = std::time::Instant::now(); }
        for j in 0..inter {
            let g = scratch.gate_out[j];
            let u = scratch.up_out[j];
            scratch.act[j] = match activation {
                crate::Activation::GeluTanh => gelu_tanh(g) * u,
                _ => silu(g) * u,
            };
        }
        let t_act = if timing { Some(t.elapsed()) } else { None };
        if timing { t = std::time::Instant::now(); }
        q4k_matvec_into(&mut scratch.out, &scratch.act, down_bytes, hidden, inter_padded);
        let t_down = if timing { Some(t.elapsed()) } else { None };
        if timing {
            eprintln!(
                "[run_expert] q4k_direct cache_gu={:.0}us gate={:.0}us up={:.0}us \
                 act={:.0}us cache_dn=0us down={:.0}us",
                t_cache_gu.unwrap().as_secs_f64() * 1e6,
                t_gate.unwrap().as_secs_f64() * 1e6,
                t_up.unwrap().as_secs_f64() * 1e6,
                t_act.unwrap().as_secs_f64() * 1e6,
                t_down.unwrap().as_secs_f64() * 1e6,
            );
        }
        return &scratch.out;
    }

    // Default path: f32 dequant cache + BLAS sgemv (Apple AMX / OpenBLAS).
    let gate_w = &gate_up_w_f32[..gate_w_size];
    let up_w = &gate_up_w_f32[gate_w_size..2 * gate_w_size];
    matmul_vec_into(&mut scratch.gate_out, h_norm, gate_w, inter, hidden);
    let t_gate = if timing { Some(t.elapsed()) } else { None };
    if timing { t = std::time::Instant::now(); }

    matmul_vec_into(&mut scratch.up_out, h_norm, up_w, inter, hidden);
    let t_up = if timing { Some(t.elapsed()) } else { None };
    if timing { t = std::time::Instant::now(); }

    // Build inner activation at `inter_padded`; padding columns
    // (`inter..inter_padded`) stay at their zero-initialised value across
    // reuses since we never write them.
    for j in 0..inter {
        let g = scratch.gate_out[j];
        let u = scratch.up_out[j];
        scratch.act[j] = match activation {
            crate::Activation::GeluTanh => gelu_tanh(g) * u,
            _ => silu(g) * u,
        };
    }
    let t_act = if timing { Some(t.elapsed()) } else { None };
    if timing { t = std::time::Instant::now(); }

    let down_w = cached_dequant(down_bytes, format, hidden * inter_padded);
    if down_w.is_empty() {
        for v in scratch.out.iter_mut() {
            *v = 0.0;
        }
        return &scratch.out;
    }
    let t_cache_dn = if timing { Some(t.elapsed()) } else { None };
    if timing { t = std::time::Instant::now(); }

    matmul_vec_into(&mut scratch.out, &scratch.act, &down_w, hidden, inter_padded);
    let t_down = if timing { Some(t.elapsed()) } else { None };

    if timing {
        eprintln!(
            "[run_expert] cache_gu={:.0}us gate={:.0}us up={:.0}us act={:.0}us \
             cache_dn={:.0}us down={:.0}us",
            t_cache_gu.unwrap().as_secs_f64() * 1e6,
            t_gate.unwrap().as_secs_f64() * 1e6,
            t_up.unwrap().as_secs_f64() * 1e6,
            t_act.unwrap().as_secs_f64() * 1e6,
            t_cache_dn.unwrap().as_secs_f64() * 1e6,
            t_down.unwrap().as_secs_f64() * 1e6,
        );
    }
    &scratch.out
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
