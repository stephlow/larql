//! MarkovResidualEngine — residual-stream KV-cache replacement.
//!
//! The pre-layer residual vector is the complete Markov state of the transformer
//! at that position. K/V are recomputed from stored residuals at decode time
//! (KL = 0.0 vs full-KV baseline on Gemma 3 4B).
//!
//! Lifted from `kv-cache-benchmark::real_model::markov_layer`.

use ndarray::{Array2, s};

use crate::model::ModelWeights;
use crate::forward::{embed_tokens_pub, run_ffn, apply_norm, dot_proj, add_bias};
use crate::attention::{run_attention_with_kv, run_attention_block_decode_step, apply_rope_partial_at};
use crate::residual::{rms_norm_heads, rms_norm_heads_no_weight};
use crate::ffn::WeightFfn;
use super::{EngineInfo, KvEngine};

// ─── RsStore ─────────────────────────────────────────────────────────────────

/// Per-layer pre-attention residuals for all stored positions.
///
/// Cold-tier: evicted residuals saved in `cold_residuals` so attention covers
/// the full history at decode time — same as the Python `extend()` replay.
pub struct RsStore {
    pub stored: Vec<Array2<f32>>,
    pub cold_residuals: Option<Vec<Array2<f32>>>,
    pub cold_abs_start: usize,
    pub next_position: usize,
    pub max_window: Option<usize>,
}

impl RsStore {
    pub fn memory_bytes(&self) -> usize {
        let hot: usize = self.stored.iter().map(|s| s.len() * 4).sum();
        let cold: usize = self.cold_residuals.as_ref()
            .map(|c| c.iter().map(|s| s.len() * 4).sum())
            .unwrap_or(0);
        hot + cold
    }

    pub(crate) fn clip_layer(&mut self, layer: usize, cold: &mut Vec<Array2<f32>>) {
        let window = match self.max_window {
            Some(w) => w,
            None => return,
        };
        let s = &self.stored[layer];
        let rows = s.shape()[0];
        if rows <= window {
            cold.push(Array2::zeros((0, s.shape()[1])));
            return;
        }
        let start = rows - window;
        cold.push(s.slice(s![..start, ..]).to_owned());
        self.stored[layer] = s.slice(s![start.., ..]).to_owned();
    }
}

// ─── Engine ──────────────────────────────────────────────────────────────────

pub struct MarkovResidualEngine {
    window_size: Option<usize>,
    store: Option<RsStore>,
}

impl MarkovResidualEngine {
    pub fn new(window_size: Option<usize>) -> Self {
        Self { window_size, store: None }
    }
}

impl KvEngine for MarkovResidualEngine {
    fn name(&self) -> &str { "markov-rs" }

    fn info(&self) -> EngineInfo {
        let config = match self.window_size {
            Some(w) => format!("window={w}"),
            None => "window=full".into(),
        };
        let mem = self.store.as_ref().map_or(0, |s| s.memory_bytes());
        EngineInfo {
            name: "markov-rs".into(),
            description: format!(
                "residual-stream KV replacement — K/V recomputed from stored residuals (mem={:.1}MB)",
                mem as f64 / 1_048_576.0,
            ),
            backend: "cpu".into(),
            config,
        }
    }

    fn prefill(&mut self, weights: &ModelWeights, token_ids: &[u32]) -> Option<Array2<f32>> {
        let result = rs_prefill(weights, token_ids, self.window_size);
        let hidden = result.hidden.clone();
        self.store = Some(result.store);
        Some(hidden)
    }

    fn decode_step(&mut self, weights: &ModelWeights, token_id: u32) -> Option<Array2<f32>> {
        let rs = self.store.take()?;
        let (hidden, new_rs) = rs_decode_step(weights, token_id, rs)?;
        self.store = Some(new_rs);
        Some(hidden)
    }

    fn memory_bytes(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.memory_bytes())
    }
}

// ─── Core functions ───────────────────────────────────────────────────────────

struct RsPrefillResult {
    hidden: Array2<f32>,
    store: RsStore,
}

fn rs_prefill(
    weights: &ModelWeights,
    token_ids: &[u32],
    max_window: Option<usize>,
) -> RsPrefillResult {
    let num_layers = weights.num_layers;
    let seq_len = token_ids.len();
    let ffn = WeightFfn { weights };

    let mut h = embed_tokens_pub(weights, token_ids);
    let mut stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);

    for layer in 0..num_layers {
        stored.push(h.clone());
        let (h_post_attn, _k, _v) = run_attention_with_kv(weights, &h, layer)
            .expect("attention failed during MarkovRS prefill");
        let (h_out, _) = run_ffn(weights, &h_post_attn, layer, &ffn, false);
        h = h_out;
    }

    let mut rs = RsStore {
        stored,
        cold_residuals: None,
        cold_abs_start: 0,
        next_position: seq_len,
        max_window,
    };

    let mut cold: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        rs.clip_layer(layer, &mut cold);
    }
    let cold_rows = cold.first().map_or(0, |c| c.shape()[0]);
    if cold_rows > 0 {
        rs.cold_residuals = Some(cold);
        rs.cold_abs_start = 0;
    }

    RsPrefillResult { hidden: last_row(&h), store: rs }
}

pub fn rs_decode_step(
    weights: &ModelWeights,
    new_token_id: u32,
    rs: RsStore,
) -> Option<(Array2<f32>, RsStore)> {
    let num_layers = weights.num_layers;
    let ffn = WeightFfn { weights };
    let abs_position = rs.next_position;

    let mut h_new = embed_tokens_pub(weights, &[new_token_id]);
    let mut new_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);

    for layer in 0..num_layers {
        let h_hot = &rs.stored[layer];
        let s_hot = h_hot.shape()[0];

        let (h_full, full_abs_start) = if let Some(cold) = &rs.cold_residuals {
            let h_cold = &cold[layer];
            let s_cold = h_cold.shape()[0];
            if s_cold > 0 {
                let hidden = h_hot.shape()[1];
                let mut combined = Array2::<f32>::zeros((s_cold + s_hot, hidden));
                combined.slice_mut(s![..s_cold, ..]).assign(h_cold);
                combined.slice_mut(s![s_cold.., ..]).assign(h_hot);
                (combined, rs.cold_abs_start)
            } else {
                (h_hot.clone(), abs_position.saturating_sub(s_hot))
            }
        } else {
            (h_hot.clone(), abs_position.saturating_sub(s_hot))
        };

        let (k_recomputed, v_recomputed) =
            recompute_kv(weights, &h_full, layer, full_abs_start)?;

        new_stored.push(h_new.clone());

        let (h_post_attn, _new_kv) = run_attention_block_decode_step(
            weights, &h_new, layer, Some(&(k_recomputed, v_recomputed)), abs_position,
        )?;

        let (h_out, _) = run_ffn(weights, &h_post_attn, layer, &ffn, false);
        h_new = h_out;
    }

    let mut updated_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for (stored, new_row) in rs.stored.iter().zip(new_stored.iter()) {
        let s_old = stored.shape()[0];
        let hidden_dim = stored.shape()[1];
        let mut combined = Array2::<f32>::zeros((s_old + 1, hidden_dim));
        combined.slice_mut(s![..s_old, ..]).assign(stored);
        combined.slice_mut(s![s_old.., ..]).assign(new_row);
        updated_stored.push(combined);
    }

    let cold_residuals = rs.cold_residuals;
    let cold_abs_start = rs.cold_abs_start;
    let max_window = rs.max_window;

    let mut updated_rs = RsStore {
        stored: updated_stored,
        cold_residuals,
        cold_abs_start,
        next_position: abs_position + 1,
        max_window,
    };

    let mut overflow: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        updated_rs.clip_layer(layer, &mut overflow);
    }
    let overflow_rows = overflow.first().map_or(0, |c| c.shape()[0]);
    if overflow_rows > 0 {
        match updated_rs.cold_residuals.as_mut() {
            Some(cold) => {
                for layer in 0..num_layers {
                    let hidden = cold[layer].shape()[1];
                    let c_old = cold[layer].shape()[0];
                    let c_new = overflow[layer].shape()[0];
                    let mut merged = Array2::<f32>::zeros((c_old + c_new, hidden));
                    merged.slice_mut(s![..c_old, ..]).assign(&cold[layer]);
                    merged.slice_mut(s![c_old.., ..]).assign(&overflow[layer]);
                    cold[layer] = merged;
                }
            }
            None => {
                updated_rs.cold_residuals = Some(overflow);
            }
        }
    }

    Some((last_row(&h_new), updated_rs))
}

pub(crate) fn recompute_kv(
    weights: &ModelWeights,
    h_stored: &Array2<f32>,
    layer: usize,
    abs_start: usize,
) -> Option<(Array2<f32>, Array2<f32>)> {
    let arch = &*weights.arch;
    let head_dim = arch.head_dim_for_layer(layer);
    let num_kv = arch.num_kv_heads_for_layer(layer);
    let norm_offset = arch.norm_weight_offset();
    let qk_offset = arch.qk_norm_weight_offset();
    let qk_norm_off = if qk_offset != 0.0 { qk_offset } else { norm_offset };

    let h_norm = apply_norm(weights, h_stored, &arch.input_layernorm_key(layer), norm_offset);

    let w_k = weights.tensors.get(&arch.attn_k_key(layer))?;
    let v_from_k = !weights.tensors.contains_key(&arch.attn_v_key(layer));
    let w_v = if v_from_k { w_k } else { weights.tensors.get(&arch.attn_v_key(layer))? };

    let mut k = dot_proj(&h_norm, w_k);
    let mut v = dot_proj(&h_norm, w_v);

    if let Some(bias) = arch.attn_k_bias_key(layer).and_then(|k| weights.vectors.get(&k)) {
        add_bias(&mut k, bias);
    }
    if let Some(bias) = arch.attn_v_bias_key(layer).and_then(|k| weights.vectors.get(&k)) {
        add_bias(&mut v, bias);
    }

    if arch.has_v_norm() {
        v = rms_norm_heads_no_weight(&v, num_kv, head_dim);
    }
    let k_normed = match arch.attn_k_norm_key(layer).and_then(|k| weights.vectors.get(&k)) {
        Some(norm_w) => rms_norm_heads(&k, norm_w, num_kv, head_dim, qk_norm_off),
        None => k,
    };

    let layer_rope_base = arch.rope_base_for_layer(layer);
    let rotary_frac = arch.rotary_fraction_for_layer(layer);
    let k_rope = apply_rope_partial_at(
        &k_normed, num_kv, head_dim, layer_rope_base, rotary_frac, abs_start,
    );

    Some((k_rope, v))
}

fn last_row(h: &Array2<f32>) -> Array2<f32> {
    let last = h.shape()[0] - 1;
    h.slice(s![last..=last, ..]).to_owned()
}
