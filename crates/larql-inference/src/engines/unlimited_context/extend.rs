//! Multi-token extend with prior K,V checkpoint.
//!
//! Runs a CPU forward pass over new tokens, seeding each layer's attention with
//! an optional prior K,V cache (the window boundary checkpoint). Equivalent to
//! Python `UnlimitedContextEngine.replay_window` inner loop.

use ndarray::Array2;

use crate::attention::{run_attention_block_decode_step, SharedKV};
use crate::ffn::WeightFfn;
use crate::forward::{embed_tokens_pub, run_ffn};
use crate::model::ModelWeights;

pub struct ExtendOutput {
    /// Hidden state at the last processed token, shape (1, hidden).
    pub last_hidden: Array2<f32>,
    /// Per-layer full K,V cache covering `[prior_tokens, new_tokens]`.
    pub kv_cache: Vec<SharedKV>,
    /// Per-layer last-row K,V ready to save as the next boundary checkpoint.
    pub new_checkpoint: Vec<SharedKV>,
}

/// Run the decoder forward over `token_ids` seeded with an optional prior K,V
/// checkpoint at each layer.
///
/// `abs_start` is the absolute position of the *first new token*.
pub fn rs_extend_from_checkpoint(
    weights: &ModelWeights,
    token_ids: &[u32],
    prior_kv: &[SharedKV],
    abs_start: usize,
) -> Option<ExtendOutput> {
    let num_layers = weights.num_layers;
    let ffn = WeightFfn { weights };

    if token_ids.is_empty() { return None; }
    if prior_kv.len() != num_layers { return None; }

    let mut kv_cache: Vec<SharedKV> = prior_kv.to_vec();
    let mut last_hidden: Option<Array2<f32>> = None;

    for (i, &token_id) in token_ids.iter().enumerate() {
        let abs_position = abs_start + i;
        let mut h = embed_tokens_pub(weights, &[token_id]);

        for (layer, kv_slot) in kv_cache.iter_mut().enumerate() {
            let kv_entry: Option<&SharedKV> = if kv_slot.0.shape()[0] > 0 {
                Some(kv_slot)
            } else {
                None
            };

            let (h_post_attn, new_kv) =
                run_attention_block_decode_step(weights, &h, layer, kv_entry, abs_position)?;

            let (h_out, _capture) = run_ffn(weights, &h_post_attn, layer, &ffn, false);
            h = h_out;
            *kv_slot = new_kv;
        }

        last_hidden = Some(h);
    }

    let new_checkpoint: Vec<SharedKV> = kv_cache
        .iter()
        .map(|(k, v)| {
            let n = k.shape()[0];
            let last_k = k.slice(ndarray::s![n - 1..n, ..]).to_owned();
            let last_v = v.slice(ndarray::s![n - 1..n, ..]).to_owned();
            (last_k, last_v)
        })
        .collect();

    Some(ExtendOutput {
        last_hidden: last_hidden?,
        kv_cache,
        new_checkpoint,
    })
}

/// Build an empty (zero-row) K,V seed for use as `prior_kv` when no prior
/// checkpoint exists (first window, or replay of window 0).
pub fn empty_prior(weights: &ModelWeights) -> Vec<SharedKV> {
    let arch = &*weights.arch;
    (0..weights.num_layers)
        .map(|layer| {
            let kv_dim = arch.num_kv_heads_for_layer(layer) * arch.head_dim_for_layer(layer);
            (
                Array2::<f32>::zeros((0, kv_dim)),
                Array2::<f32>::zeros((0, kv_dim)),
            )
        })
        .collect()
}
