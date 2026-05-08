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

use super::cache::try_cached_dequant;
use super::expert::{run_single_expert_q4k_q8k_into, ExpertScratch};
use super::math::{gelu_tanh, matmul_vec, rms_norm, rms_norm_no_weight, silu, softmax, top_k};
use crate::cpu::ops::q4k_q8k_dot::quantize_x_to_q8k;
use crate::options;

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
    // Per-stage timing for bottleneck diagnosis.  Enable with
    // `LARQL_MOE_FWD_TIMING=1`.  Cached in TLS to avoid syscalls
    // per call on the hot path.
    thread_local! {
        static FWD_TIMING: bool = options::env_flag(options::ENV_MOE_FWD_TIMING);
    }
    let timing = FWD_TIMING.with(|t| *t);
    let t_start = std::time::Instant::now();

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
    if options::skip_moe_enabled() {
        return vec![0.0f32; hidden];
    }

    // 1. Pre-experts norm — input for the expert matmuls.
    //
    //    The router norm composes ON TOP of this. Empirically the trained
    //    Gemma 4 26B-A4B weights expect router input = pre_experts_norm(h),
    //    not raw h, even though HF's modeling_gemma4.py reads the raw
    //    residual. Switching to the HF convention degrades generation to
    //    token repetition; this matches Metal's `gpu_moe_dispatch`
    //    convention so all backends agree.
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
    if options::moe_debug_enabled() {
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

    let t_pre_par = t_start.elapsed();

    // Q4_K direct-from-mmap path: quantise h_norm to Q8_K once per layer
    // (shared across all K active experts) and use the SDOT-based integer
    // matvec.  Bypasses the f32 dequant cache entirely — at Gemma 4 26B-A4B
    // sizes the f32 cache is 5.7 GB walked per token and DRAM-bandwidth
    // bound; direct-Q4K is ~1.4 GB.  Set `LARQL_DISABLE_Q4K_DIRECT=1` to
    // fall back to the BLAS-on-cached-f32 path for kernel-debug A/B runs.
    let q4k_direct = matches!(format, crate::QuantFormat::Q4_K)
        && hidden.is_multiple_of(256)
        && !options::env_flag(options::ENV_DISABLE_Q4K_DIRECT);
    let t_q8k_quant_start = std::time::Instant::now();
    let h_norm_q8k = q4k_direct.then(|| quantize_x_to_q8k(&h_norm));
    let t_q8k_quant = t_q8k_quant_start.elapsed();
    let t_par_start = std::time::Instant::now();

    // Per-rayon-thread scratch buffers (gate_out / up_out / act / act_q8k /
    // out).  Allocated lazily on first hit, reused across all subsequent
    // expert calls on the same worker.  Replaces the prior pattern of
    // `vec![0; ...]` allocs per expert call (5 distinct heap allocs per
    // call × K=8 × 30 layers = 1200 allocs/token, with occasional 150 µs
    // spikes from the allocator's slow path that drag par_iter wall up).
    thread_local! {
        static SCRATCH: std::cell::RefCell<Option<ExpertScratch>> =
            const { std::cell::RefCell::new(None) };
    }

    use rayon::prelude::*;
    let expert_out = expert_indices
        .par_iter()
        .zip(expert_weights.par_iter())
        .filter(|(_, &w)| w != 0.0)
        .fold(
            || vec![0.0f32; hidden],
            |mut acc, (&ei, &w)| {
                let Some(&gate_up_bytes) = moe.experts_gate_up.get(ei) else {
                    return acc;
                };
                let Some(&down_bytes) = moe.experts_down.get(ei) else {
                    return acc;
                };

                SCRATCH.with(|cell| {
                    let mut borrow = cell.borrow_mut();
                    let scratch = borrow
                        .get_or_insert_with(|| ExpertScratch::new(hidden, inter, inter_padded));
                    if scratch.gate_out.len() != inter
                        || scratch.act.len() != inter_padded
                        || scratch.out.len() != hidden
                    {
                        *scratch = ExpertScratch::new(hidden, inter, inter_padded);
                    }

                    if let Some(q8k) = h_norm_q8k.as_ref() {
                        // Q4_K direct path — single source of truth in
                        // `expert::run_single_expert_q4k_q8k_into`.  Reuses
                        // the scratch's act_q8k buffer too.
                        let h2 = run_single_expert_q4k_q8k_into(
                            scratch,
                            q8k,
                            gate_up_bytes,
                            down_bytes,
                            inter,
                            activation,
                        );
                        for (a, &v) in acc.iter_mut().zip(h2.iter()) {
                            *a += w * v;
                        }
                        return;
                    }

                    // Fallback: BF16 / F32 / Q4_K-with-disable — original
                    // f32 cache path.  Inlined here to avoid pulling the
                    // per-call rms_norm / format dispatch from the legacy
                    // `run_single_expert_into` that doesn't share scratch.
                    let gate_up_w = try_cached_dequant(gate_up_bytes, format, 2 * inter * hidden)
                        .unwrap_or_else(|err| panic!("{err}"));
                    if gate_up_w.is_empty() {
                        return;
                    }
                    let gate_w = &gate_up_w[..inter * hidden];
                    let up_w = &gate_up_w[inter * hidden..2 * inter * hidden];

                    let gate_out = matmul_vec(&h_norm, gate_w, inter, hidden);
                    let up_out = matmul_vec(&h_norm, up_w, inter, hidden);

                    for j in 0..inter {
                        let g = gate_out[j];
                        let u = up_out[j];
                        scratch.act[j] = match activation {
                            crate::Activation::GeluTanh => gelu_tanh(g) * u,
                            _ => silu(g) * u,
                        };
                    }

                    let down_w = try_cached_dequant(down_bytes, format, hidden * inter_padded)
                        .unwrap_or_else(|err| panic!("{err}"));
                    if down_w.is_empty() {
                        return;
                    }
                    let expert_contribution =
                        matmul_vec(&scratch.act, &down_w, hidden, inter_padded);
                    for (a, &v) in acc.iter_mut().zip(expert_contribution.iter()) {
                        *a += w * v;
                    }
                });
                acc
            },
        )
        .reduce(
            || vec![0.0f32; hidden],
            |mut a, b| {
                for (x, &y) in a.iter_mut().zip(b.iter()) {
                    *x += y;
                }
                a
            },
        );

    let t_par = t_par_start.elapsed();
    let t_sum = std::time::Duration::ZERO;

    // 10. Post-experts norm (HF `post_feedforward_layernorm_2`)
    let t_post_start = std::time::Instant::now();
    let result = rms_norm(&expert_out, moe.post_experts_norm, eps, norm_offset);
    let t_post = t_post_start.elapsed();

    if timing {
        eprintln!(
            "[cpu_moe_forward] K={} pre_par={:.0}us q8k_quant={:.0}us \
             par_iter={:.0}us sum={:.0}us post_norm={:.0}us total={:.0}us",
            expert_indices.len(),
            t_pre_par.as_secs_f64() * 1e6,
            t_q8k_quant.as_secs_f64() * 1e6,
            t_par.as_secs_f64() * 1e6,
            t_sum.as_secs_f64() * 1e6,
            t_post.as_secs_f64() * 1e6,
            t_start.elapsed().as_secs_f64() * 1e6,
        );
    }

    if options::moe_debug_enabled() {
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
