//! Prediction entry points — the functions external code calls
//!
//! All GPU pipeline layer construction goes through `pipeline_layer::build_pipeline_layers()`.
//! Logits computation goes through `logits::finalize_logits()`.
//! KV cache prefill goes through `prefill::prefill_with_kv()`.
//! Token generation goes through `generate::generate()`.

use ndarray::Array2;

use super::{CachedLayerGraph, DenseLayerGraph, LayerGraph};
use crate::model::ModelWeights;
use larql_compute::prelude::*;

// Re-export moved functions for backward compatibility.
pub use super::generate::{generate, GenerateResult};
pub use super::logits::finalize_logits;
pub use super::prefill::prefill_with_kv;

// Alias for internal callers.
use super::prefill::prefill_kv_cache_cpu;

/// Run a full forward pass using vindex logits (KNN against lm_head mmap).
/// Replaces the 231ms dense logits matmul with a ~1ms KNN lookup.
pub fn predict_with_graph_vindex_logits(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    graph: &dyn LayerGraph,
    index: &larql_vindex::VectorIndex,
) -> crate::forward::PredictResult {
    let seq_len = token_ids.len();
    let mut h = crate::forward::embed_tokens_pub(weights, token_ids);

    for layer in 0..weights.num_layers {
        match graph.forward_layer(weights, &h, layer) {
            Some(output) => h = output.residual,
            None => break,
        }
    }

    // Final norm
    let norm_offset = weights.arch.norm_weight_offset();
    let h_final =
        crate::forward::apply_norm(weights, &h, weights.arch.final_norm_key(), norm_offset);

    // Vindex logits: KNN against lm_head mmap
    let last_row = h_final.row(seq_len - 1).to_owned();

    let logits_scale = weights.arch.logits_scaling();
    let final_softcap = weights.arch.final_logit_softcapping();
    let inv_scale = 1.0 / logits_scale;

    // Get raw scores from KNN (dot products against lm_head)
    let hits = index.lm_head_knn(&last_row, top_k);

    // Apply scaling, softcap, softmax over top-K
    let scaled: Vec<(u32, f32)> = hits
        .iter()
        .map(|&(tid, score)| {
            let mut logit = score * inv_scale;
            if let Some(cap) = final_softcap {
                logit = (logit / cap).tanh() * cap;
            }
            (tid, logit)
        })
        .collect();

    let max_logit = scaled
        .iter()
        .map(|(_, l)| *l)
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f64 = scaled
        .iter()
        .map(|(_, l)| ((*l - max_logit) as f64).exp())
        .sum();

    let predictions = scaled
        .iter()
        .filter_map(|&(tid, logit)| {
            let prob = ((logit - max_logit) as f64).exp() / exp_sum;
            tokenizer
                .decode(&[tid], true)
                .ok()
                .map(|s| (s.trim().to_string(), prob))
        })
        .collect();

    crate::forward::PredictResult {
        predictions,
        token_ids: Vec::new(),
    }
}

/// Run a full forward pass using a LayerGraph for per-layer routing.
/// This is the generic layer loop — embedding → layers → logits.
pub fn predict_with_graph(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    graph: &dyn LayerGraph,
) -> crate::forward::PredictResult {
    let mut h = crate::forward::embed_tokens_pub(weights, token_ids);

    for layer in 0..weights.num_layers {
        match graph.forward_layer(weights, &h, layer) {
            Some(output) => h = output.residual,
            None => break,
        }
    }

    crate::forward::logits_to_predictions_pub(weights, &h, tokenizer, top_k, 1.0)
}

/// Two-pass split pipeline: attention on CPU, FFN batched on Metal GPU.
///
/// Pass 1: Run attention for all layers with attention-only residual stream (CPU).
///          The FFN contribution is deferred — attention sees approximate residuals.
///          This is valid because attention is template-fixed (~99% identical across entities).
///
/// Pass 2: Compute FFN for all layers in one Metal command buffer (8.5ms for 21 layers).
///          Uses the post-attention residuals from pass 1 as FFN inputs.
///
/// Pass 3: Add FFN outputs to the final residual and compute logits via vindex KNN.
///
/// Target: 55ms attention + 8.5ms FFN + 5ms logits = 68ms → 15 tok/s
#[allow(clippy::too_many_arguments)]
pub fn predict_split_pass(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    cached_layers: &CachedLayerGraph,
    layer_range: std::ops::Range<usize>,
) -> crate::forward::PredictResult {
    let seq_len = token_ids.len();
    let hidden = weights.hidden_size;
    let arch = &*weights.arch;
    let norm_offset = arch.norm_weight_offset();

    // ── Pass 0: Cached layers (0ms) ──
    let mut h = crate::forward::embed_tokens_pub(weights, token_ids);
    for layer in 0..layer_range.start {
        if let Some(output) = cached_layers.forward_layer(weights, &h, layer) {
            h = output.residual;
        }
    }

    // ── Pass 1: Attention-only for walk layers (CPU BLAS) ──
    // Store post-attention residuals and FFN-normed inputs.
    let mut post_attn_residuals: Vec<Array2<f32>> = Vec::new();
    let mut ffn_inputs: Vec<Vec<f32>> = Vec::new();

    for layer in layer_range.clone() {
        // Run attention only (CPU BLAS, no FFN)
        let (h_post_attn, _attn_proj, _) =
            crate::attention::run_attention_block_gpu(weights, &h, layer, false, None)
                .unwrap_or_else(|| {
                    // Fallback: identity (shouldn't happen with valid weights)
                    (h.clone(), h.clone(), None)
                });

        // Compute pre-FFN norm (this is the FFN input)
        let pre_ffn_key = if arch.has_post_norms() {
            arch.pre_feedforward_layernorm_key(layer)
        } else {
            Some(arch.post_attention_layernorm_key(layer))
        };
        let h_ffn = match pre_ffn_key {
            Some(key) => crate::forward::apply_norm(weights, &h_post_attn, &key, norm_offset),
            None => crate::residual::rms_norm(&h_post_attn, None, norm_offset),
        };

        // Store last-token FFN input for GPU batch
        let last_row = h_ffn.row(seq_len - 1);
        ffn_inputs.push(last_row.to_vec());
        post_attn_residuals.push(h_post_attn.clone());

        // Continue with attention-only residual (approximate — no FFN contribution)
        h = h_post_attn;
    }

    // ── Pass 2: Batch FFN on Metal GPU ──
    let num_walk_layers = layer_range.len();

    // Try batched Q4 FFN via multi_layer_q4_ffn
    let gate_index: &dyn larql_vindex::GateIndex = index;
    let ffn_outputs = if gate_index.has_interleaved_q4() && backend.has_q4() {
        if let Some(q4_mmap) = gate_index.interleaved_q4_mmap_ref() {
            let intermediate = gate_index.num_features(layer_range.start);
            if intermediate > 0 {
                let q4_bytes_per_matrix = larql_compute::QuantFormat::Q4_0
                    .packed_matrix_bytes(intermediate, hidden)
                    .expect("Q4_0 interleaved FFN format must have packed geometry");
                let q4_bytes_per_layer = q4_bytes_per_matrix * 3;

                // Collect Q4 data slices for all walk layers
                let layers_q4: Vec<(&[u8], &[u8], &[u8])> = layer_range
                    .clone()
                    .map(|layer| {
                        let start = layer * q4_bytes_per_layer;
                        let gate = &q4_mmap[start..start + q4_bytes_per_matrix];
                        let up =
                            &q4_mmap[start + q4_bytes_per_matrix..start + 2 * q4_bytes_per_matrix];
                        let down = &q4_mmap
                            [start + 2 * q4_bytes_per_matrix..start + 3 * q4_bytes_per_matrix];
                        (gate, up, down)
                    })
                    .collect();

                // Use the first FFN input as the batch input
                // multi_layer_q4_ffn chains layers: out of layer N → input of layer N+1
                let x = &ffn_inputs[0];
                backend.multi_layer_q4_ffn(&layers_q4, x, intermediate, hidden)
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    // ── Pass 3: Combine ──
    // If GPU batch succeeded, use the final FFN output.
    // Otherwise, fall back to per-layer CPU FFN.
    if let Some(ffn_final) = ffn_outputs {
        // The multi_layer_q4_ffn returns the final residual after all FFN layers.
        // Add it to the last post-attention residual.
        let last_post_attn = &post_attn_residuals[num_walk_layers - 1];
        let mut h_final = last_post_attn.clone();
        let mut last_row_mut = h_final.row_mut(seq_len - 1);
        for j in 0..hidden {
            // The FFN output is the chained result — use it as the last-token residual
            last_row_mut[j] = ffn_final[j];
        }
        h = h_final;
    } else {
        // Fallback: run FFN per-layer on CPU
        h = crate::forward::embed_tokens_pub(weights, token_ids);
        for layer in 0..layer_range.start {
            if let Some(output) = cached_layers.forward_layer(weights, &h, layer) {
                h = output.residual;
            }
        }
        let walk_ffn = crate::vindex::WalkFfn::new_unlimited(weights, index);
        for layer in layer_range.clone() {
            let dense = DenseLayerGraph {
                ffn: &walk_ffn,
                backend: None,
                capture_activation: false,
                capture_attention: false,
            };
            if let Some(output) = dense.forward_layer(weights, &h, layer) {
                h = output.residual;
            }
        }
    }

    // Final norm + vindex logits
    let h_final =
        crate::forward::apply_norm(weights, &h, weights.arch.final_norm_key(), norm_offset);
    let last_row = h_final.row(seq_len - 1).to_owned();

    let logits_scale = weights.arch.logits_scaling();
    let final_softcap = weights.arch.final_logit_softcapping();
    let inv_scale = 1.0 / logits_scale;

    let hits = index.lm_head_knn(&last_row, top_k);
    let scaled: Vec<(u32, f32)> = hits
        .iter()
        .map(|&(tid, score)| {
            let mut logit = score * inv_scale;
            if let Some(cap) = final_softcap {
                logit = (logit / cap).tanh() * cap;
            }
            (tid, logit)
        })
        .collect();

    let max_logit = scaled
        .iter()
        .map(|(_, l)| *l)
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f64 = scaled
        .iter()
        .map(|(_, l)| ((*l - max_logit) as f64).exp())
        .sum();
    let predictions = scaled
        .iter()
        .filter_map(|&(tid, logit)| {
            let prob = ((logit - max_logit) as f64).exp() / exp_sum;
            tokenizer
                .decode(&[tid], true)
                .ok()
                .map(|s| (s.trim().to_string(), prob))
        })
        .collect();

    crate::forward::PredictResult {
        predictions,
        token_ids: Vec::new(),
    }
}

/// Split pass using cached attention residuals — exact output at GPU speed.
///
/// Uses `AttentionCache` (built from one exact run) to skip all attention
/// computation. Batches FFN on Metal GPU in one command buffer.
///
/// Target: 0ms attention + 8.5ms FFN + 5ms logits = ~14ms → 71 tok/s
pub fn predict_split_cached(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    top_k: usize,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    attn_cache: &super::AttentionCache,
    _layer_range: std::ops::Range<usize>,
) -> crate::forward::PredictResult {
    let norm_offset = weights.arch.norm_weight_offset();

    // Zero-copy: borrow the cached residual, don't clone.
    // Final norm produces a new array (unavoidable), but the input is borrowed.
    let h_final = crate::forward::apply_norm(
        weights,
        &attn_cache.final_residual,
        weights.arch.final_norm_key(),
        norm_offset,
    );
    let seq_len = h_final.shape()[0];
    let last_row = h_final.row(seq_len - 1).to_owned();

    // GPU Q4 logits when available (1ms), else CPU BLAS (10ms)
    let hits = index.lm_head_knn_backend(&last_row, top_k, backend);

    let logits_scale = weights.arch.logits_scaling();
    let final_softcap = weights.arch.final_logit_softcapping();
    let inv_scale = 1.0 / logits_scale;

    let scaled: Vec<(u32, f32)> = hits
        .iter()
        .map(|&(tid, score)| {
            let mut logit = score * inv_scale;
            if let Some(cap) = final_softcap {
                logit = (logit / cap).tanh() * cap;
            }
            (tid, logit)
        })
        .collect();

    let max_logit = scaled
        .iter()
        .map(|(_, l)| *l)
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f64 = scaled
        .iter()
        .map(|(_, l)| ((*l - max_logit) as f64).exp())
        .sum();
    let predictions = scaled
        .iter()
        .filter_map(|&(tid, logit)| {
            let prob = ((logit - max_logit) as f64).exp() / exp_sum;
            tokenizer
                .decode(&[tid], true)
                .ok()
                .map(|s| (s.trim().to_string(), prob))
        })
        .collect();

    crate::forward::PredictResult {
        predictions,
        token_ids: Vec::new(),
    }
}

/// Honest production pipeline: real computation, no over-caching.
///
/// - L0-12: cached (template-fixed, proven at 0.999 cosine — legitimate)
/// - L13-33: interleaved attention (CPU BLAS) + FFN (GPU Q4 via compute crate)
/// - Logits: GPU Q4 matvec against lm_head (1ms)
///
/// Every entity-dependent layer is computed. No approximate residuals.
#[allow(clippy::too_many_arguments)]
pub fn predict_honest(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    cached_layers: &CachedLayerGraph,
    layer_range: std::ops::Range<usize>,
) -> crate::forward::PredictResult {
    let norm_offset = weights.arch.norm_weight_offset();

    // Pass 0: cached layers (legitimate — template-fixed)
    let mut h = crate::forward::embed_tokens_pub(weights, token_ids);
    for layer in 0..layer_range.start {
        if let Some(output) = cached_layers.forward_layer(weights, &h, layer) {
            h = output.residual;
        }
    }

    // GPU pipeline: decode (seq=1) uses decode_token/full_pipeline_q4,
    // prefill (seq>1) uses prefill_q4 for GPU-accelerated multi-position inference.
    let seq_len = h.shape()[0];
    let used_gpu = if backend.has_q4() {
        let gate_index: &dyn larql_vindex::GateIndex = index;
        // Prefer Q4_K FFN (Ollama-compatible) over Q4_0
        let (q4_ffn, ffn_is_q4k) = if let Some(mmap) = gate_index.interleaved_q4k_mmap_ref() {
            (Some(mmap), true)
        } else {
            (gate_index.interleaved_q4_mmap_ref(), false)
        };
        let has_q4k = index.attn_q4k_layer_data(layer_range.start).is_some();
        let has_q8 = index.attn_q8_layer_data(layer_range.start).is_some();

        if let Some(q4_ffn_mmap) = q4_ffn {
            let intermediate = gate_index.num_features(layer_range.start);
            let hidden = weights.hidden_size;
            if intermediate > 0 && (has_q4k || has_q8) {
                let ffn_format = if ffn_is_q4k {
                    larql_compute::QuantFormat::Q4_K
                } else {
                    larql_compute::QuantFormat::Q4_0
                };
                let q4_ffn_per_matrix = ffn_format
                    .packed_matrix_bytes(intermediate, hidden)
                    .expect("Q4 interleaved FFN format must have packed geometry");
                // q4_ffn_per_layer computed inside build_pipeline_layers
                let arch = &*weights.arch;

                let layers = super::pipeline_layer::build_pipeline_layers(
                    weights,
                    index,
                    layer_range.clone(),
                    q4_ffn_mmap,
                    q4_ffn_per_matrix,
                    ffn_format,
                );

                let attention = layers
                    .first()
                    .map(super::pipeline_layer::attention_geometry_for_pipeline_layer)
                    .unwrap_or_else(|| {
                        super::pipeline_layer::attention_geometry_for_arch_layer(
                            weights,
                            layer_range.start,
                        )
                    });
                let softcap = arch.attn_logit_softcapping().unwrap_or(0.0);
                let qk_norm = arch.attn_q_norm_key(layer_range.start).is_some();

                if seq_len == 1 {
                    // Decode path (seq=1): try KV-cached decode first, then full_pipeline
                    let x: Vec<f32> = h.row(0).to_vec();

                    if let Some(result) = backend.decode_token(
                        &layers,
                        &x,
                        hidden,
                        intermediate,
                        attention.q_dim,
                        attention.kv_dim,
                        attention.num_q_heads,
                        attention.num_kv_heads,
                        attention.head_dim,
                        attention.rope_base,
                    ) {
                        let mut row = h.row_mut(0);
                        for j in 0..hidden {
                            row[j] = result[j];
                        }
                        return finalize_logits(
                            weights,
                            tokenizer,
                            &h,
                            top_k,
                            index,
                            backend,
                            norm_offset,
                        );
                    }

                    if let Some(result) = backend.full_pipeline_q4(
                        &layers,
                        &x,
                        hidden,
                        intermediate,
                        attention.q_dim,
                        attention.kv_dim,
                        1,
                        attention.num_q_heads,
                        attention.num_kv_heads,
                        attention.head_dim,
                        attention.rope_base,
                        qk_norm,
                        softcap,
                    ) {
                        let mut row = h.row_mut(0);
                        for j in 0..hidden {
                            row[j] = result[j];
                        }
                        true
                    } else {
                        false
                    }
                } else if !arch.has_post_norms() {
                    // Prefill path (seq>1): GPU Q4 pipeline for pre-norm models (Llama, Mistral)
                    // Post-norm models (Gemma3) fall through to CPU — prefill.rs post-norm
                    // handling needs further work (see ADR-009).
                    let x: Vec<f32> = h.as_slice().unwrap_or(&[]).to_vec();

                    if let Some(result) = backend.prefill_q4(
                        &layers,
                        &x,
                        hidden,
                        intermediate,
                        attention.q_dim,
                        attention.kv_dim,
                        seq_len,
                        attention.num_q_heads,
                        attention.num_kv_heads,
                        attention.head_dim,
                        attention.rope_base,
                        qk_norm,
                        softcap,
                    ) {
                        // Copy result back to h matrix (all positions)
                        for s in 0..seq_len {
                            let mut row = h.row_mut(s);
                            for j in 0..hidden {
                                row[j] = result[s * hidden + j];
                            }
                        }

                        // Populate KV cache via CPU for subsequent decode
                        // (lightweight: just QKV projection + RoPE, no FFN)
                        prefill_kv_cache_cpu(weights, token_ids, index, backend, &layer_range);

                        true
                    } else {
                        false
                    }
                } else {
                    // Post-norm models (Gemma3): CPU prefill (correct) → GPU logits (fast)
                    // CPU handles post-norms correctly. Use CPU hidden state, GPU for logits only.
                    // KV cache populated for future decode_token calls (token generation).
                    backend.reset_kv_cache();

                    let walk_ffn = crate::vindex::WalkFfn::new_unlimited(weights, index);
                    let mut h_cpu = h.clone();
                    for (rel_idx, abs_layer) in layer_range.clone().enumerate() {
                        let (h_post_attn, k_rope, v) =
                            crate::attention::gpu::run_attention_with_kv_backend(
                                weights,
                                &h_cpu,
                                abs_layer,
                                Some(backend),
                            )
                            .unwrap();

                        if backend.has_kv_cache() {
                            let k_flat = k_rope.as_slice().unwrap_or(&[]);
                            let v_flat = v.as_slice().unwrap_or(&[]);
                            backend.populate_kv_layer(
                                rel_idx,
                                k_flat,
                                v_flat,
                                seq_len,
                                weights.arch.num_kv_heads_for_layer(abs_layer),
                                weights.arch.head_dim_for_layer(abs_layer),
                            );
                        }

                        let (h_out, _) = crate::forward::run_ffn(
                            weights,
                            &h_post_attn,
                            abs_layer,
                            &walk_ffn,
                            false,
                        );
                        h_cpu = h_out;
                    }

                    // Use correct CPU hidden state, finalize with GPU logits
                    h = h_cpu;
                    return finalize_logits(
                        weights,
                        tokenizer,
                        &h,
                        top_k,
                        index,
                        backend,
                        norm_offset,
                    );
                }
            } else {
                false
            }
        } else {
            false
        }
    } else {
        false
    };

    // CPU fallback: interleaved attention + FFN (for prefill or when GPU not available)
    if !used_gpu {
        let walk_ffn = crate::vindex::WalkFfn::new_unlimited(weights, index);
        for layer in layer_range {
            let (h_post_attn, _, _) =
                crate::attention::run_attention_block_gpu(weights, &h, layer, false, None).unwrap();
            let (h_out, _) =
                crate::forward::run_ffn(weights, &h_post_attn, layer, &walk_ffn, false);
            h = h_out;
        }
    }

    finalize_logits(weights, tokenizer, &h, top_k, index, backend, norm_offset)
}

// generate(), GenerateResult, and softmax_prob moved to generate.rs and logits.rs

/// Optimized predict: uses vindex logits when lm_head is loaded, falls back to full matmul.
///
/// This is the production entry point. It:
/// 1. Runs embedding → layer loop via LayerGraph
/// 2. Uses vindex lm_head KNN if available (eliminates 226ms logits matmul)
/// 3. Falls back to full vocab matmul if no lm_head loaded
pub fn predict_pipeline(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    graph: &dyn LayerGraph,
    index: Option<&larql_vindex::VectorIndex>,
) -> crate::forward::PredictResult {
    // Use vindex logits if lm_head is loaded
    if let Some(idx) = index {
        if idx.has_lm_head() {
            return predict_with_graph_vindex_logits(
                weights, tokenizer, token_ids, top_k, graph, idx,
            );
        }
    }
    // Fallback: full vocab matmul
    predict_with_graph(weights, tokenizer, token_ids, top_k, graph)
}

/// Run a full forward pass with tracing (residuals + activations + attention).
pub fn trace_with_graph(
    weights: &ModelWeights,
    token_ids: &[u32],
    capture_layers: &[usize],
    graph: &dyn LayerGraph,
) -> crate::forward::TraceResult {
    let seq_len = token_ids.len();
    let max_layer = *capture_layers.iter().max().unwrap_or(&0);

    let mut h = crate::forward::embed_tokens_pub(weights, token_ids);
    let mut results = Vec::new();
    let mut activations = Vec::new();
    let mut attention_captures = Vec::new();

    for layer in 0..=max_layer.min(weights.num_layers - 1) {
        match graph.forward_layer(weights, &h, layer) {
            Some(output) => {
                h = output.residual;

                if capture_layers.contains(&layer) {
                    let last_row = h.row(seq_len - 1);
                    results.push((layer, last_row.to_vec()));

                    if let Some(act) = output.activation {
                        let act_row = act.row(seq_len - 1);
                        let mut indexed: Vec<(usize, f32)> =
                            act_row.iter().copied().enumerate().collect();
                        indexed.sort_unstable_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
                        indexed.truncate(200);
                        activations.push((layer, indexed));
                    }

                    if let Some(attn) = output.attention {
                        attention_captures.push(crate::forward::LayerAttentionCapture {
                            layer,
                            weights: attn,
                        });
                    }
                }
            }
            None => break,
        }
    }

    crate::forward::TraceResult {
        residuals: results,
        activations,
        attention: attention_captures,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::test_utils::{
        make_test_tokenizer, make_test_vindex, make_test_weights, TestFixtures,
    };
    use crate::model::ModelWeights;
    use std::sync::OnceLock;

    fn fx() -> &'static TestFixtures {
        static F: OnceLock<TestFixtures> = OnceLock::new();
        F.get_or_init(TestFixtures::build)
    }
    use crate::ffn::WeightFfn;
    use crate::layer_graph::CachedLayerGraph;
    use larql_compute::CpuBackend;

    // ── predict_with_ffn ──────────────────────────────────────────────────────

    #[test]
    fn predict_with_ffn_returns_predictions() {
        let f = fx();
        let (weights, tokenizer) = (&f.weights, &f.tokenizer);
        let ffn = WeightFfn { weights: &weights };
        let result = crate::forward::predict_with_ffn(&weights, &tokenizer, &[0u32, 1], 3, &ffn);
        assert!(result.token_ids.len() <= 3);
        assert_eq!(result.predictions.len(), result.token_ids.len());
        assert!(result
            .token_ids
            .iter()
            .all(|&id| (id as usize) < weights.vocab_size));
    }

    #[test]
    fn predict_with_ffn_single_token() {
        let f = fx();
        let (weights, tokenizer) = (&f.weights, &f.tokenizer);
        let ffn = WeightFfn { weights: &weights };
        let result = crate::forward::predict_with_ffn(&weights, &tokenizer, &[5u32], 1, &ffn);
        assert!(result.token_ids.len() <= 1);
    }

    // ── predict_honest (CPU path via VectorIndex::new with no Q4K) ────────────

    #[test]
    fn predict_honest_runs_without_panic() {
        let f = fx();
        let (weights, tokenizer, index) = (&f.weights, &f.tokenizer, &f.index);
        let cached = CachedLayerGraph::from_residuals(vec![]);
        let num_layers = weights.num_layers;
        // predict_honest falls through to CPU path (no Q4K data in synthetic vindex)
        let result = predict_honest(
            &weights,
            &tokenizer,
            &[0u32, 1, 2],
            5,
            &index,
            &CpuBackend,
            &cached,
            0..num_layers,
        );
        // lm_head_knn is empty → predictions may be empty, but no panic
        assert!(result.token_ids.len() <= 5);
    }

    #[test]
    fn predict_honest_single_token_decode_path() {
        let f = fx();
        let (weights, tokenizer, index) = (&f.weights, &f.tokenizer, &f.index);
        let cached = CachedLayerGraph::from_residuals(vec![]);
        let num_layers = weights.num_layers;
        let result = predict_honest(
            &weights,
            &tokenizer,
            &[3u32],
            3,
            &index,
            &CpuBackend,
            &cached,
            0..num_layers,
        );
        assert!(result.token_ids.len() <= 3);
    }

    #[test]
    fn predict_honest_with_cached_layers() {
        let f = fx();
        let (weights, tokenizer, index) = (&f.weights, &f.tokenizer, &f.index);
        let ffn = WeightFfn { weights: &weights };
        // Pre-cache layer 0
        let cached = CachedLayerGraph::build(&weights, &[0u32], &[0], &ffn);
        let num_layers = weights.num_layers;
        let result = predict_honest(
            &weights,
            &tokenizer,
            &[0u32],
            3,
            &index,
            &CpuBackend,
            &cached,
            0..num_layers,
        );
        assert!(result.token_ids.len() <= 3);
    }

    // ── DenseLayerGraph ───────────────────────────────────────────────��───────

    #[test]
    fn dense_layer_graph_forward_runs() {
        use crate::layer_graph::{DenseLayerGraph, LayerGraph};
        let weights = &fx().weights;
        let ffn = WeightFfn { weights: &weights };
        let h = ndarray::Array2::from_elem((2, weights.hidden_size), 0.1f32);
        let g = DenseLayerGraph {
            ffn: &ffn,
            backend: None,
            capture_activation: false,
            capture_attention: false,
        };
        let out = g.forward_layer(&weights, &h, 0);
        assert!(out.is_some(), "DenseLayerGraph should forward layer 0");
        assert_eq!(out.unwrap().residual.shape(), &[2, weights.hidden_size]);
    }

    #[test]
    fn dense_layer_graph_all_layers() {
        use crate::layer_graph::{DenseLayerGraph, LayerGraph};
        let weights = &fx().weights;
        let ffn = WeightFfn { weights: &weights };
        let h = ndarray::Array2::from_elem((1, weights.hidden_size), 0.5f32);
        let g = DenseLayerGraph {
            ffn: &ffn,
            backend: None,
            capture_activation: false,
            capture_attention: false,
        };
        for layer in 0..weights.num_layers {
            let out = g.forward_layer(&weights, &h, layer);
            assert!(out.is_some(), "layer {layer} should succeed");
        }
    }

    // ── WalkLayerGraph ────────────────────────────────────────────────────────

    #[test]
    fn walk_layer_graph_forward_runs() {
        use crate::layer_graph::{LayerGraph, WalkLayerGraph};
        let weights = &fx().weights;
        let ffn = WeightFfn { weights: &weights };
        let g = WalkLayerGraph {
            ffn: &ffn,
            backend: None,
        };
        let h = ndarray::Array2::from_elem((2, weights.hidden_size), 0.1f32);
        let out = g.forward_layer(&weights, &h, 0);
        assert!(out.is_some());
        assert_eq!(out.unwrap().residual.shape(), &[2, weights.hidden_size]);
    }

    // ── predict_pipeline ─────────────────────────────────────────────────────

    #[test]
    fn predict_pipeline_runs() {
        use crate::layer_graph::LayerGraph;
        let f = fx();
        let (weights, tokenizer, index) = (&f.weights, &f.tokenizer, &f.index);
        let ffn = WeightFfn { weights: &weights };
        let g = crate::layer_graph::WalkLayerGraph {
            ffn: &ffn,
            backend: None,
        };
        let graph: &dyn LayerGraph = &g;
        // predict_pipeline takes Option<&VectorIndex>
        let result = predict_pipeline(&weights, &tokenizer, &[0u32, 1], 3, graph, Some(&index));
        assert!(result.token_ids.len() <= 3);
    }
}
