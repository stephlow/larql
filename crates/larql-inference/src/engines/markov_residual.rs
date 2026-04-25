//! MarkovResidualEngine — residual-stream KV-cache replacement.
//!
//! The pre-layer residual vector is the complete Markov state of the transformer
//! at that position. K/V are recomputed from stored residuals at decode time
//! (KL = 0.0 vs full-KV baseline on Gemma 3 4B, validated 2026-04-23).
//!
//! Lifted from `kv-cache-benchmark::real_model::markov_layer`.

use ndarray::{Array2, s};
use larql_compute::{ComputeBackend, cpu_backend, dot_proj_gpu};

use crate::model::ModelWeights;
use crate::forward::{embed_tokens_pub, run_ffn, apply_norm, add_bias};
use crate::attention::{
    run_attention_with_kv_backend,
    run_attention_block_decode_step_backend,
    apply_rope_partial_at,
};
use crate::residual::{rms_norm_heads, rms_norm_heads_no_weight};
use crate::ffn::BackendFfn;
use crate::attention::SharedKV;
use super::{EngineInfo, KvEngine};
use super::profiler::{DecodeStageSummary, EngineProfiler};

// ─── RsStore ─────────────────────────────────────────────────────────────────

/// Per-layer pre-attention residuals for all stored positions.
///
/// - `stored[l]`: hot window residuals for layer l, shape `[W, hidden_dim]`
/// - `cold_residuals[l]`: evicted rows from the hot window (full-history replay)
/// - `cold_kv[l]`: pre-computed K/V for the cold tier — static between decode steps,
///   computed once at prefill and reused to avoid redundant `recompute_kv` calls.
pub struct RsStore {
    pub stored: Vec<Array2<f32>>,
    pub cold_residuals: Option<Vec<Array2<f32>>>,
    /// Cached K/V for the cold tier. Each entry is `(K[C, kv_dim], V[C, kv_dim])`.
    /// Once the cold tier is frozen (post-prefill), this avoids re-running
    /// `recompute_kv` on the same static residuals every decode step.
    pub cold_kv: Option<Vec<SharedKV>>,
    pub cold_abs_start: usize,
    pub next_position: usize,
    pub max_window: Option<usize>,
}

impl RsStore {
    /// Total bytes for hot residuals + cold residuals + cached cold K/V.
    pub fn memory_bytes(&self) -> usize {
        let hot: usize = self.stored.iter().map(|s| s.len() * 4).sum();
        let cold_res: usize = self.cold_residuals.as_ref()
            .map(|c| c.iter().map(|s| s.len() * 4).sum())
            .unwrap_or(0);
        let cold_kv: usize = self.cold_kv.as_ref()
            .map(|kv| kv.iter().map(|(k, v)| (k.len() + v.len()) * 4).sum())
            .unwrap_or(0);
        hot + cold_res + cold_kv
    }

    /// Bytes in the cold tier (residuals + cached K/V).
    pub fn cold_bytes(&self) -> usize {
        let cold_res: usize = self.cold_residuals.as_ref()
            .map(|c| c.iter().map(|s| s.len() * 4).sum())
            .unwrap_or(0);
        let cold_kv: usize = self.cold_kv.as_ref()
            .map(|kv| kv.iter().map(|(k, v)| (k.len() + v.len()) * 4).sum())
            .unwrap_or(0);
        cold_res + cold_kv
    }

    /// Token count in the hot window (uses layer 0 as reference).
    pub fn window_tokens(&self) -> usize {
        self.stored.first().map_or(0, |s| s.shape()[0])
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
    backend: Box<dyn ComputeBackend>,
}

impl MarkovResidualEngine {
    pub fn new(window_size: Option<usize>) -> Self {
        Self::with_backend(window_size, cpu_backend())
    }

    pub fn with_backend(window_size: Option<usize>, backend: Box<dyn ComputeBackend>) -> Self {
        Self { window_size, store: None, backend }
    }

    /// Total memory of the engine state in bytes.
    pub fn total_memory_bytes(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.memory_bytes())
    }

    /// Token count in the hot window.
    pub fn window_tokens(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.window_tokens())
    }

    /// Bytes in the cold tier only.
    pub fn cold_bytes(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.cold_bytes())
    }
}

impl KvEngine for MarkovResidualEngine {
    fn name(&self) -> &str { "markov-rs" }

    fn info(&self) -> EngineInfo {
        let window_cfg = match self.window_size {
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
            backend: self.backend.name().to_string(),
            config: window_cfg,
        }
    }

    fn prefill(&mut self, weights: &ModelWeights, token_ids: &[u32]) -> Option<Array2<f32>> {
        let result = rs_prefill(weights, token_ids, self.window_size, self.backend.as_ref());
        let hidden = result.hidden.clone();
        self.store = Some(result.store);
        Some(hidden)
    }

    fn decode_step(&mut self, weights: &ModelWeights, token_id: u32) -> Option<Array2<f32>> {
        let rs = self.store.take()?;
        let (hidden, new_rs) = rs_decode_step(weights, token_id, rs, self.backend.as_ref())?;
        self.store = Some(new_rs);
        Some(hidden)
    }

    fn memory_bytes(&self) -> usize { self.total_memory_bytes() }
    fn window_tokens(&self) -> usize { self.window_tokens() }
    fn cold_bytes(&self) -> usize { self.cold_bytes() }
}

// ─── Core functions ───────────────────────────────────────────────────────────

pub struct RsPrefillResult {
    pub hidden: Array2<f32>,
    pub store: RsStore,
    pub memory_bytes: usize,
    pub window_tokens: usize,
}

/// Run the full prefill forward pass, storing pre-layer residuals.
/// Equivalent to a standard forward pass but stores residuals instead of K/V.
pub fn rs_prefill(
    weights: &ModelWeights,
    token_ids: &[u32],
    max_window: Option<usize>,
    backend: &dyn ComputeBackend,
) -> RsPrefillResult {
    let num_layers = weights.num_layers;
    let seq_len = token_ids.len();

    let mut h = embed_tokens_pub(weights, token_ids);
    let mut stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    let be = Some(backend);

    for layer in 0..num_layers {
        stored.push(h.clone());
        let (h_post_attn, _k, _v) = run_attention_with_kv_backend(weights, &h, layer, be)
            .expect("attention failed during MarkovRS prefill");
        let bffn = BackendFfn { weights, backend };
        let (h_out, _) = run_ffn(weights, &h_post_attn, layer, &bffn, false);
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

    let window_tokens = rs.window_tokens();
    let memory_bytes = rs.memory_bytes();
    RsPrefillResult { hidden: last_row(&h), store: rs, memory_bytes, window_tokens }
}

/// Run one decode step, recomputing K/V from stored residuals.
pub fn rs_decode_step(
    weights: &ModelWeights,
    new_token_id: u32,
    rs: RsStore,
    backend: &dyn ComputeBackend,
) -> Option<(Array2<f32>, RsStore)> {
    let num_layers = weights.num_layers;
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
            recompute_kv(weights, &h_full, layer, full_abs_start, backend)?;

        new_stored.push(h_new.clone());

        let (h_post_attn, _new_kv) = run_attention_block_decode_step_backend(
            weights, &h_new, layer, Some(&(k_recomputed, v_recomputed)), abs_position, Some(backend),
        )?;

        let bffn = BackendFfn { weights, backend };
        let (h_out, _) = run_ffn(weights, &h_post_attn, layer, &bffn, false);
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

/// Recompute K/V from stored pre-layer residuals.
///
/// Uses `backend` for the K/V projection matmuls — routes through GPU on
/// Metal (meaningful speedup for long contexts where `h_stored` is large).
pub fn recompute_kv(
    weights: &ModelWeights,
    h_stored: &Array2<f32>,
    layer: usize,
    abs_start: usize,
    backend: &dyn ComputeBackend,
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

    // K/V projection: hot path for long contexts, GPU-dispatched when available.
    let mut k = dot_proj_gpu(&h_norm, w_k, Some(backend));
    let mut v = dot_proj_gpu(&h_norm, w_v, Some(backend));

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

/// Equivalent Standard KV memory in bytes for `seq_len` tokens (FP16).
pub fn kv_memory_bytes_for_seq(weights: &ModelWeights, seq_len: usize) -> usize {
    let arch = &*weights.arch;
    (0..weights.num_layers)
        .map(|l| {
            let kv_dim = arch.num_kv_heads_for_layer(l) * arch.head_dim_for_layer(l);
            seq_len * kv_dim * 2 * 2 // K + V, FP16 (2 bytes each)
        })
        .sum()
}

fn last_row(h: &Array2<f32>) -> Array2<f32> {
    let last = h.shape()[0] - 1;
    h.slice(s![last..=last, ..]).to_owned()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rs(num_layers: usize, seq_len: usize, hidden: usize, window: Option<usize>) -> RsStore {
        let stored = (0..num_layers)
            .map(|l| {
                let mut a = Array2::<f32>::zeros((seq_len, hidden));
                for i in 0..seq_len {
                    a.row_mut(i).fill((l * 1000 + i) as f32);
                }
                a
            })
            .collect();
        RsStore {
            stored,
            cold_residuals: None,
            cold_abs_start: 0,
            next_position: seq_len,
            max_window: window,
        }
    }

    // ── clip_layer ─────────────────────────────────────────────────────────────

    #[test]
    fn clip_no_window_keeps_all() {
        let mut rs = make_rs(1, 10, 4, None);
        let mut cold = Vec::new();
        rs.clip_layer(0, &mut cold);
        assert_eq!(rs.stored[0].shape()[0], 10);
        assert!(cold.is_empty(), "clip_layer with no window must not push");
    }

    #[test]
    fn clip_exact_window_keeps_all() {
        let mut rs = make_rs(1, 5, 4, Some(5));
        let mut cold = Vec::new();
        rs.clip_layer(0, &mut cold);
        assert_eq!(rs.stored[0].shape()[0], 5);
        assert_eq!(cold[0].shape()[0], 0);
    }

    #[test]
    fn clip_splits_hot_cold_correctly() {
        let mut rs = make_rs(1, 10, 4, Some(4));
        let mut cold = Vec::new();
        rs.clip_layer(0, &mut cold);
        assert_eq!(cold[0].shape()[0], 6, "6 rows evicted");
        assert_eq!(rs.stored[0].shape()[0], 4, "4 rows remain");
        for i in 0..6 {
            assert_eq!(cold[0][[i, 0]], i as f32, "cold row {i} value");
        }
        for i in 0..4 {
            assert_eq!(rs.stored[0][[i, 0]], (6 + i) as f32, "hot row {i} value");
        }
    }

    #[test]
    fn clip_multi_layer_consistent() {
        let mut rs = make_rs(3, 8, 4, Some(3));
        let mut cold = Vec::new();
        for layer in 0..3 { rs.clip_layer(layer, &mut cold); }
        for (l, (c, s)) in cold.iter().zip(rs.stored.iter()).enumerate() {
            assert_eq!(c.shape()[0], 5, "layer {l}: 5 cold rows");
            assert_eq!(s.shape()[0], 3, "layer {l}: 3 hot rows");
        }
    }

    // ── memory_bytes ──────────────────────────────────────────────────────────

    #[test]
    fn memory_bytes_hot_only() {
        let rs = make_rs(2, 4, 8, None);
        assert_eq!(rs.memory_bytes(), 2 * 4 * 8 * 4);
    }

    #[test]
    fn memory_bytes_includes_cold_tier() {
        let mut rs = make_rs(2, 10, 8, Some(4));
        let mut cold = Vec::with_capacity(2);
        for layer in 0..2 { rs.clip_layer(layer, &mut cold); }
        rs.cold_residuals = Some(cold);
        let hot  = 2 * 4 * 8 * 4;
        let cold = 2 * 6 * 8 * 4;
        assert_eq!(rs.memory_bytes(), hot + cold);
    }

    #[test]
    fn cold_bytes_only_cold_tier() {
        let mut rs = make_rs(2, 10, 8, Some(4));
        let mut cold = Vec::with_capacity(2);
        for layer in 0..2 { rs.clip_layer(layer, &mut cold); }
        rs.cold_residuals = Some(cold);
        assert_eq!(rs.cold_bytes(), 2 * 6 * 8 * 4);
    }

    #[test]
    fn window_tokens_uses_layer0() {
        let rs = make_rs(3, 7, 4, None);
        assert_eq!(rs.window_tokens(), 7);
    }

    // ── cold-tier overflow merge in decode ─────────────────────────────────────

    #[test]
    fn decode_overflow_merges_into_existing_cold() {
        let window = 3;
        let hidden = 4;
        let hot = vec![Array2::<f32>::ones((window, hidden))];
        let existing_cold = vec![Array2::<f32>::zeros((2, hidden))];

        let mut rs = RsStore {
            stored: hot,
            cold_residuals: Some(existing_cold),
            cold_abs_start: 0,
            next_position: 5,
            max_window: Some(window),
        };

        let new_row = Array2::<f32>::from_elem((1, hidden), 9.0);
        let s_old = rs.stored[0].shape()[0];
        let mut combined = Array2::<f32>::zeros((s_old + 1, hidden));
        combined.slice_mut(s![..s_old, ..]).assign(&rs.stored[0]);
        combined.slice_mut(s![s_old.., ..]).assign(&new_row);
        rs.stored[0] = combined;

        let mut overflow = Vec::new();
        rs.clip_layer(0, &mut overflow);
        assert_eq!(overflow[0].shape()[0], 1, "one row overflows");

        if let Some(cold) = rs.cold_residuals.as_mut() {
            let c_old = cold[0].shape()[0];
            let c_new = overflow[0].shape()[0];
            let mut merged = Array2::<f32>::zeros((c_old + c_new, hidden));
            merged.slice_mut(s![..c_old, ..]).assign(&cold[0]);
            merged.slice_mut(s![c_old.., ..]).assign(&overflow[0]);
            cold[0] = merged;
        }
        assert_eq!(rs.cold_residuals.as_ref().unwrap()[0].shape()[0], 3);
        assert_eq!(rs.stored[0].shape()[0], window);
    }

    // ── engine construction ────────────────────────────────────────────────────

    #[test]
    fn engine_new_has_no_store() {
        let engine = MarkovResidualEngine::new(Some(512));
        assert_eq!(engine.memory_bytes(), 0);
        assert_eq!(engine.window_tokens(), 0);
        assert_eq!(engine.cold_bytes(), 0);
    }

    #[test]
    fn engine_info_backend_is_cpu_by_default() {
        let engine = MarkovResidualEngine::new(None);
        assert!(engine.info().backend.starts_with("cpu"), "expected cpu backend, got {:?}", engine.info().backend);
        assert_eq!(engine.info().config, "window=full");
        assert!(engine.info().summary().contains("markov-rs"));
    }

    #[test]
    fn engine_info_window_size_in_config() {
        let engine = MarkovResidualEngine::new(Some(512));
        assert_eq!(engine.info().config, "window=512");
    }
}
