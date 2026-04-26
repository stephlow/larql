//! Logits computation — final norm + vindex KNN + softmax.

use ndarray::Array2;

use larql_compute::prelude::*;
use crate::model::ModelWeights;

/// Shared logits computation: final norm + vindex KNN + softmax.
pub fn finalize_logits(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    h: &Array2<f32>,
    top_k: usize,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    norm_offset: f32,
) -> crate::forward::PredictResult {
    let h_final = crate::forward::apply_norm(weights, h, weights.arch.final_norm_key(), norm_offset);
    let seq_len = h_final.shape()[0];
    let last_row = h_final.row(seq_len - 1).to_owned();

    let hits = index.lm_head_knn_backend(&last_row, top_k, backend);

    let logits_scale = weights.arch.logits_scaling();
    let final_softcap = weights.arch.final_logit_softcapping();
    let inv_scale = 1.0 / logits_scale;

    let scaled: Vec<(u32, f32)> = hits.iter().map(|&(tid, score)| {
        let mut logit = score * inv_scale;
        if let Some(cap) = final_softcap {
            logit = (logit / cap).tanh() * cap;
        }
        (tid, logit)
    }).collect();

    let max_logit = scaled.iter().map(|(_, l)| *l).fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f64 = scaled.iter().map(|(_, l)| ((*l - max_logit) as f64).exp()).sum();
    let predictions = scaled.iter()
        .filter_map(|&(tid, logit)| {
            let prob = ((logit - max_logit) as f64).exp() / exp_sum;
            tokenizer.decode(&[tid], true).ok()
                .map(|s| (s.trim().to_string(), prob))
        })
        .collect();

    crate::forward::PredictResult { predictions, token_ids: Vec::new() }
}

/// Softmax probability of a single score within a set of hits.
pub(super) fn softmax_prob(score: f32, hits: &[(u32, f32)], logits_scale: f32, softcap: Option<f32>) -> f64 {
    let inv_scale = 1.0 / logits_scale;
    let scaled: Vec<f32> = hits.iter().map(|&(_, s)| {
        let mut l = s * inv_scale;
        if let Some(cap) = softcap { l = (l / cap).tanh() * cap; }
        l
    }).collect();
    let max_l = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f64 = scaled.iter().map(|l| ((*l - max_l) as f64).exp()).sum();
    let mut target = score * inv_scale;
    if let Some(cap) = softcap { target = (target / cap).tanh() * cap; }
    ((target - max_l) as f64).exp() / exp_sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::test_utils::{make_test_weights, make_test_vindex, make_test_tokenizer};
    use larql_compute::CpuBackend;

    #[test]
    fn finalize_logits_runs_without_panic() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let index = make_test_vindex(&weights);
        let h = ndarray::Array2::from_elem((1, weights.hidden_size), 0.1f32);
        let norm_offset = weights.arch.norm_weight_offset();
        let result = finalize_logits(&weights, &tokenizer, &h, 5, &index, &CpuBackend, norm_offset);
        // lm_head_knn returns empty for synthetic vindex → empty predictions
        assert!(result.token_ids.len() <= 5);
    }

    #[test]
    fn softmax_prob_basic() {
        let hits = vec![(0u32, 3.0f32), (1u32, 2.0f32), (2u32, 1.0f32)];
        let p = softmax_prob(3.0, &hits, 1.0, None);
        assert!(p > 0.0 && p <= 1.0, "probability should be in (0,1]");
        // Highest logit should have highest probability
        let p2 = softmax_prob(2.0, &hits, 1.0, None);
        assert!(p > p2, "logit=3 should have higher prob than logit=2");
    }
}
