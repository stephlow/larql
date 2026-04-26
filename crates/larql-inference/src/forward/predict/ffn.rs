//! FFN-backend forward passes (custom backend, router, strategy).

use crate::attention::SharedKV;
use crate::ffn::{FfnBackend, LayerFfnRouter};
use crate::model::ModelWeights;
use super::super::embed::embed_tokens;
use super::super::ple::precompute_per_layer_inputs;
use super::super::layer::{run_layer_with_ffn, run_layer_with_capture, run_attention};
use super::types::{PredictResult, PredictResultWithAttention, LayerMode, LayerAttentionCapture};
use super::dense::logits_to_predictions;

/// Run a full forward pass with a custom FFN backend for all layers.
pub fn predict_with_ffn(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    ffn: &dyn FfnBackend,
) -> PredictResult {
    let num_layers = weights.num_layers;
    let mut h = embed_tokens(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);

    let mut kv_cache: std::collections::HashMap<usize, SharedKV> =
        std::collections::HashMap::new();

    for layer in 0..num_layers {
        let shared_kv = weights.arch.kv_shared_source_layer(layer)
            .and_then(|src| kv_cache.get(&src));

        match run_layer_with_ffn(weights, &h, layer, ffn, false, ple_inputs.get(layer), shared_kv) {
            Some((h_new, _, kv_out)) => {
                h = h_new;
                if let Some(kv) = kv_out {
                    kv_cache.insert(layer, kv);
                }
            }
            None => continue,
        }
    }

    logits_to_predictions(weights, &h, tokenizer, top_k, 1.0)
}

/// Run a full forward pass with a custom FFN backend, capturing attention weights
/// and per-layer residuals for logit lens.
pub fn predict_with_ffn_attention(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    ffn: &dyn FfnBackend,
) -> PredictResultWithAttention {
    let num_layers = weights.num_layers;
    let seq_len = token_ids.len();
    let mut h = embed_tokens(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
    let mut attention = Vec::with_capacity(num_layers);
    let mut residuals = Vec::with_capacity(num_layers);

    for layer in 0..num_layers {
        match run_layer_with_capture(weights, &h, layer, ffn, false, true, ple_inputs.get(layer), None) {
            Some((h_new, _, attn_weights, _)) => {
                h = h_new;
                residuals.push((layer, h.row(seq_len - 1).to_vec()));
                if let Some(w) = attn_weights {
                    attention.push(LayerAttentionCapture { layer, weights: w });
                }
            }
            None => continue,
        }
    }

    let result = logits_to_predictions(weights, &h, tokenizer, top_k, 1.0);
    PredictResultWithAttention {
        predictions: result.predictions,
        attention,
        residuals,
    }
}

/// Run a full forward pass with per-layer FFN backend selection.
pub fn predict_with_router(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    router: &LayerFfnRouter,
) -> PredictResult {
    let num_layers = weights.num_layers;
    let mut h = embed_tokens(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);

    for layer in 0..num_layers {
        let ffn = router.get(layer);
        h = match run_layer_with_ffn(weights, &h, layer, ffn, false, ple_inputs.get(layer), None) {
            Some((h_new, _, _)) => h_new,
            None => continue,
        };
    }

    logits_to_predictions(weights, &h, tokenizer, top_k, 1.0)
}

/// Run a forward pass with per-layer strategy: full compute or scalar gain bypass.
pub fn predict_with_strategy(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    strategy: &[LayerMode],
) -> PredictResult {
    let num_layers = weights.num_layers;
    let mut h = embed_tokens(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);

    for (layer, mode) in strategy.iter().enumerate().take(num_layers) {
        match mode {
            LayerMode::Compute(ffn) => {
                h = match run_layer_with_ffn(weights, &h, layer, *ffn, false, ple_inputs.get(layer), None) {
                    Some((h_new, _, _)) => h_new,
                    None => continue,
                };
            }
            LayerMode::ScalarGain(gain) => {
                h *= *gain;
            }
            LayerMode::AttentionOnly => {
                if let Some(h_post_attn) = run_attention(weights, &h, layer) {
                    h = h_post_attn;
                }
            }
        }
    }

    logits_to_predictions(weights, &h, tokenizer, top_k, 1.0)
}
