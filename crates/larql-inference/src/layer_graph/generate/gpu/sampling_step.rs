//! Per-token sample + detokenize + EOS-check helpers shared by the
//! first-token and decode-loop branches of [`super::generate_streaming`].

use crate::layer_graph::generate::detok::Detokenizer;
use crate::layer_graph::generate::eos::EosConfig;
use crate::layer_graph::generate::sampling::Sampler;
use crate::model::ModelWeights;

/// Outcome of a single sampling step: the picked token id, its surface
/// form, the softmax probability, and whether EOS was hit.
pub(super) struct PickedToken {
    pub id: u32,
    pub text: String,
    pub prob: f64,
    pub is_eos: bool,
}

/// Pick a token from a top-K hits vector, push it onto the detokenizer,
/// fire `on_token`, and check EOS. Returns `None` when the sampler
/// rejects the entire distribution (empty hits / all -inf logits) so the
/// caller can break the loop.
pub(super) fn sample_and_emit<F>(
    sampler: &mut Sampler,
    detok: &mut Detokenizer,
    tokenizer: &tokenizers::Tokenizer,
    weights: &ModelWeights,
    eos: &EosConfig,
    hits: &[(u32, f32)],
    generated_ids: &[u32],
    on_token: &mut F,
) -> Option<PickedToken>
where
    F: FnMut(u32, &str, f64),
{
    let picked_id = sampler.sample_from_topk_with_history(hits, generated_ids)?;
    let text = detok.push(picked_id);
    let score = hits
        .iter()
        .find(|(t, _)| *t == picked_id)
        .map(|(_, s)| *s)
        .unwrap_or(0.0);
    let prob = crate::layer_graph::logits::softmax_prob(
        score,
        hits,
        weights.arch.logits_scaling(),
        weights.arch.final_logit_softcapping(),
    );
    on_token(picked_id, &text, prob);
    let is_eos = eos.is_eos_with_tokenizer(picked_id, &text, tokenizer);
    Some(PickedToken {
        id: picked_id,
        text,
        prob,
        is_eos,
    })
}
