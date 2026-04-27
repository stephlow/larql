//! Full MoE block forward pass: router → top-k → weighted sum of expert outputs.
//!
//! Flow (matching HF Gemma 4, with fallbacks for architectures that omit
//! some weights):
//!
//! 1. `pre_experts_norm(h)` — input for the expert matmuls.
//! 2. `router_norm(h) * router_scale * router_input_scalar` — input for the
//!    router projection. Falls back to the experts' pre-norm when the router
//!    has no dedicated norm weight.
//! 3. Softmax over all experts → top-k → renormalize weights to sum to 1 →
//!    multiply by `per_expert_scale`.
//! 4. For each selected expert: `down_proj(act(gate_proj(h_norm)) * up_proj(h_norm))`
//!    weighted by the router probability, accumulated into `expert_out`.
//! 5. `post_experts_norm(expert_out)` — matches HF's `post_feedforward_layernorm_2`.

use crate::MoeLayerWeights;

use super::cache::cached_dequant;
use super::math::{gelu_tanh, matmul_vec, rms_norm, rms_norm_no_weight, silu, softmax, top_k};

/// Run the MoE expert block for one token.
///
/// `h` — residual stream at this layer (hidden_size f32 values).
/// Returns the expert block contribution to add to the dense FFN output.
/// If `moe` is missing required fields, returns a zero vector of hidden_size.
pub fn cpu_moe_forward(
    h: &[f32],
    moe: &MoeLayerWeights<'_>,
    norm_offset: f32,
    eps: f32,
) -> Vec<f32> {
    let hidden = h.len();
    let num_experts = moe.num_experts;
    let top_k_val = moe.top_k;
    let inter = moe.intermediate_size;

    if num_experts == 0 || top_k_val == 0 || inter == 0 {
        return vec![0.0f32; hidden];
    }
    if moe.router_proj.is_empty() || moe.experts_gate_up.is_empty() || moe.experts_down.is_empty() {
        return vec![0.0f32; hidden];
    }
    // Diagnostic: bypass the expert block entirely. Dense FFN alone flows
    // through the normal path; if this produces legible output, the MoE
    // block is the broken piece. If still garbage, look upstream.
    if std::env::var("SKIP_MOE").is_ok() {
        return vec![0.0f32; hidden];
    }

    // 1. Pre-experts norm — input for the expert matmuls.
    //
    //    The router norm composes ON TOP of this — verified by `larql parity
    //    --component moe-block`: raw-h routing and h_norm routing pick
    //    different top-K experts on the 26B-A4B vindex (e.g. layer 0:
    //    [55,101,126,12,52,114,84,79] vs [101,52,126,55,12,34,68,79], 2 of 8
    //    differ). Metal's `gpu_moe_dispatch` calls `cpu_moe_route(&h_norm,
    //    ...)` and produces correct generation ("Paris."); CPU paths that
    //    route on raw h produce garbage. Aligning to Metal here.
    let h_norm = rms_norm(h, moe.pre_experts_norm, eps, norm_offset);

    // 2. Router input norm. Resolution order:
    //      1. learned router_norm weight (architectures that ship one),
    //      2. parameter-free RMSNorm (HF Gemma 4 — `Gemma4RMSNorm(with_scale=False)`),
    //      3. fallback: just use the pre-experts-norm output directly.
    //    All three apply on top of h_norm so the routing matches Metal.
    let router_in_normed: Vec<f32> = if !moe.router_norm.is_empty() {
        rms_norm(&h_norm, moe.router_norm, eps, norm_offset)
    } else if moe.router_norm_parameter_free {
        rms_norm_no_weight(&h_norm, eps)
    } else {
        h_norm.clone()
    };

    // 3. Router scale (learned per-hidden-dim vector) + optional scalar
    //    (Gemma 4: `scalar_root_size = hidden_size^-0.5`). Applied after the
    //    router norm, before the projection.
    let mut router_in: Vec<f32> = if !moe.router_scale.is_empty() {
        router_in_normed
            .iter()
            .zip(moe.router_scale.iter())
            .map(|(a, b)| a * b)
            .collect()
    } else {
        router_in_normed
    };
    if moe.router_input_scalar != 1.0 && moe.router_input_scalar != 0.0 {
        for v in router_in.iter_mut() {
            *v *= moe.router_input_scalar;
        }
    }

    // 4. Router projection: [hidden] → [num_experts]
    let mut logits = matmul_vec(&router_in, moe.router_proj, num_experts, hidden);

    // 5. Softmax
    softmax(&mut logits);

    // 6. Top-k selection
    let (expert_indices, mut expert_weights) = top_k(&logits, top_k_val);

    // Debug: print routing per layer if MOE_DEBUG=1
    static DEBUG_LAYER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    if std::env::var("MOE_DEBUG").is_ok() {
        let layer_n = DEBUG_LAYER.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 30;
        let h_rms = (h.iter().map(|v| v * v).sum::<f32>() / h.len() as f32).sqrt();
        let hn_rms = (h_norm.iter().map(|v| v * v).sum::<f32>() / h_norm.len() as f32).sqrt();
        let ri_rms =
            (router_in.iter().map(|v| v * v).sum::<f32>() / router_in.len().max(1) as f32).sqrt();
        let logit_max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let logit_min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
        let pnorm_rms = (moe.pre_experts_norm.iter().map(|v| v * v).sum::<f32>()
            / moe.pre_experts_norm.len().max(1) as f32)
            .sqrt();
        let rnorm_rms = (moe.router_norm.iter().map(|v| v * v).sum::<f32>()
            / moe.router_norm.len().max(1) as f32)
            .sqrt();
        let rscale_rms = (moe.router_scale.iter().map(|v| v * v).sum::<f32>()
            / moe.router_scale.len().max(1) as f32)
            .sqrt();
        eprintln!("[L{layer_n:02}] h_rms={h_rms:.2} hn_rms={hn_rms:.2} router_in_rms={ri_rms:.2} | pnorm_rms={pnorm_rms:.2} rnorm_rms={rnorm_rms:.2} rscale_rms={rscale_rms:.2} scalar={:.4} | logits [{logit_min:.3}..{logit_max:.3}] | experts:{expert_indices:?}", moe.router_input_scalar);
    }

    // 7. Renormalize selected weights to sum to 1 (Gemma 4 gemma4_top_k_softmax).
    // After softmax over all 128 experts, the selected top-8 weights sum to
    // ~0.5-0.7, not 1.0.  Renormalising ensures the expert block contributes
    // at the correct scale.  Without this the expert residual is undersized
    // every layer and the model output is garbage.
    let weight_sum: f32 = expert_weights.iter().sum();
    if weight_sum > 0.0 {
        for w in &mut expert_weights {
            *w /= weight_sum;
        }
    }

    // 8. Per-expert output scale (Gemma 4 learned per-expert scale)
    if !moe.router_per_expert_scale.is_empty() {
        for (i, &ei) in expert_indices.iter().enumerate() {
            if ei < moe.router_per_expert_scale.len() {
                expert_weights[i] *= moe.router_per_expert_scale[ei];
            }
        }
    }

    // 9. Run each selected expert's gated FFN (BF16 dequant on demand).
    //    Experts are independent — their only shared input is `h_norm` and
    //    their outputs are summed. Parallelise across the top-K experts with
    //    rayon so BLAS-accelerated gemv on each core overlaps. `moe.activation`
    //    is a plain enum (Copy), and `cached_dequant` hands out shared
    //    Arc<Vec<f32>> values that are Sync, so the closure is Send+Sync.
    //
    //    gate_up layout: [num_experts, 2*inter, hidden]  (gate rows first, then up rows)
    //    down layout:    [num_experts, hidden, inter]
    use rayon::prelude::*;
    let activation = moe.activation;
    let format = moe.expert_data_format;
    // Storage layout per Gemma 4 26B-A4B (and the per-layer Q4_K writer):
    //   gate_up: [2*inter, hidden]              — never padded; quantises
    //                                             cleanly because hidden is
    //                                             already a 256-multiple.
    //   down:    [hidden, inter_padded]         — Q4_K pads `inter` up to
    //                                             the next 256 super-block
    //                                             (704 → 768). BF16 stores
    //                                             un-padded.
    // Mirror Metal's `inter_padded` handling (`metal/moe_dispatch.rs`):
    // dequant down at the padded width, zero-pad the hidden_state so
    // the matmul reads `inter_padded` columns with the padding
    // contributing zero.
    let inter_padded = match format {
        crate::QuantFormat::Q4_K => {
            let block = larql_models::quant::ggml::Q4_K_BLOCK_ELEMS;
            inter.div_ceil(block) * block
        }
        _ => inter,
    };
    let per_expert: Vec<(f32, Vec<f32>)> = expert_indices
        .par_iter()
        .zip(expert_weights.par_iter())
        .filter_map(|(&ei, &weight)| {
            if weight == 0.0 {
                return None;
            }
            // Per-expert byte slices come straight from the mmap-backed
            // tables; cached_dequant LRU-keys on the byte pointer so a
            // re-selected expert skips both allocation and decode.
            let gate_up_bytes = *moe.experts_gate_up.get(ei)?;
            let gate_up_w = cached_dequant(gate_up_bytes, format, 2 * inter * hidden);
            if gate_up_w.is_empty() {
                return None;
            }
            let gate_w = &gate_up_w[..inter * hidden];
            let up_w = &gate_up_w[inter * hidden..2 * inter * hidden];

            let gate_out = matmul_vec(&h_norm, gate_w, inter, hidden);
            let up_out = matmul_vec(&h_norm, up_w, inter, hidden);

            // Gated activation: ACT(gate) * up.  Gemma 4 uses GELU-tanh; Mixtral uses SiLU.
            // Build the inner activation at `inter_padded` so the down matmul
            // (which expects `inter_padded` columns under Q4_K) sees zero in
            // the padding region.
            let mut hidden_state: Vec<f32> = vec![0.0f32; inter_padded];
            for j in 0..inter {
                let g = gate_out[j];
                let u = up_out[j];
                hidden_state[j] = match activation {
                    crate::Activation::GeluTanh => gelu_tanh(g) * u,
                    _ => silu(g) * u,
                };
            }

            let down_bytes = *moe.experts_down.get(ei)?;
            let down_w = cached_dequant(down_bytes, format, hidden * inter_padded);
            if down_w.is_empty() {
                return None;
            }
            let expert_contribution = matmul_vec(&hidden_state, &down_w, hidden, inter_padded);
            Some((weight, expert_contribution))
        })
        .collect();

    let mut expert_out = vec![0.0f32; hidden];
    for (weight, contribution) in &per_expert {
        for (acc, &val) in expert_out.iter_mut().zip(contribution.iter()) {
            *acc += val * *weight;
        }
    }

    // 10. Post-experts norm (HF `post_feedforward_layernorm_2`)
    let result = rms_norm(&expert_out, moe.post_experts_norm, eps, norm_offset);

    if std::env::var("MOE_DEBUG").is_ok() {
        let pre_rms =
            (expert_out.iter().map(|v| v * v).sum::<f32>() / expert_out.len() as f32).sqrt();
        let post_rms = (result.iter().map(|v| v * v).sum::<f32>() / result.len() as f32).sqrt();
        let pnorm2_rms = (moe.post_experts_norm.iter().map(|v| v * v).sum::<f32>()
            / moe.post_experts_norm.len().max(1) as f32)
            .sqrt();
        eprintln!(
            "  pre_norm_rms={pre_rms:.3} post_norm2_rms={pnorm2_rms:.3} moe_out_rms={post_rms:.3}"
        );
    }

    result
}
