//! `DecodeBackend` impl for `MetalBackend`.
//!
//! These methods drive the GPU full-pipeline / KV-cached decode /
//! prefill paths. Most of them delegate to dispatchers under
//! `metal::ops::full_pipeline` or to inherent helpers on
//! `MetalBackend` (e.g. `decode_token`, `decode_token_with_moe_fn`).

use crate::backend::DecodeBackend;
use crate::metal::{ops, MetalBackend};

impl DecodeBackend for MetalBackend {
    fn full_pipeline_q4(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize, inter: usize,
        q_dim: usize, kv_dim: usize,
        seq_len: usize,
        num_q_heads: usize, num_kv_heads: usize, head_dim: usize,
        rope_base: f32, use_qk_norm: bool, softcap: f32,
    ) -> Option<Vec<f32>> {
        let geglu = if layers.first().is_some_and(|l| l.activation == crate::Activation::GeluTanh) {
            &self.geglu_gelu_tanh_pipeline
        } else {
            &self.geglu_pipeline
        };
        Some(ops::full_pipeline::dispatch_full_pipeline(
            &self.queue, &self.bufs, &self.q4,
            geglu,
            &self.geglu_gelu_tanh_pipeline,
            &self.silu_pipeline,
            &self.gelu_tanh_pipeline,
            &self.q8_quant_pipeline,
            Some(&self.fused_attn_pipeline),
            &self.q8_matvec_pipeline.state,
            &self.q8_qkv_proj_pipeline,
            &self.q4k_matvec_pipeline.state, &self.q6k_matvec_pipeline.state,
            &self.rms_norm_pipeline, &self.residual_add_pipeline,
            &self.rms_norm_q8_pipeline, &self.residual_norm_q8_pipeline,
            Some(&self.q4k_qkv_proj_pipeline.state),
            Some(&self.q4kf_qkv_proj_pipeline.state),
            Some(&self.q4kf_proj_pipeline.state),
            None,
            Some(&self.qk_norm_pipeline),
            Some(&self.scale_vector_pipeline),
            None,
            layers, x, hidden, inter, q_dim, kv_dim,
            seq_len, num_q_heads, num_kv_heads, head_dim,
            rope_base, use_qk_norm, softcap,
        ))
    }

    fn multi_layer_q4_ffn(
        &self,
        layers_q4: &[(&[u8], &[u8], &[u8])],
        x: &[f32],
        inter: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        Some(MetalBackend::multi_layer_q4_ffn(self, layers_q4, x, inter, hidden))
    }

    fn prefill_q4(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize, inter: usize,
        q_dim: usize, kv_dim: usize,
        seq_len: usize,
        num_q_heads: usize, num_kv_heads: usize, head_dim: usize,
        rope_base: f32, use_qk_norm: bool, softcap: f32,
    ) -> Option<Vec<f32>> {
        let num_layers = layers.len();
        let shapes: Vec<(usize, usize)> = layers.iter()
            .map(|l| (l.num_kv_heads, l.head_dim))
            .collect();
        let mut cache_guard = self.kv_cache.lock().unwrap();
        if cache_guard.is_none() {
            *cache_guard = Some(ops::kv_cache::KVCache::new_per_layer(&self.bufs, &shapes, 4096));
        }
        let kv = cache_guard.as_mut().unwrap();
        while kv.layers.len() < num_layers {
            let (nkv, hd) = shapes[kv.layers.len()];
            kv.layers.push(ops::kv_cache::LayerKVCache::new(&self.bufs, 4096, nkv, hd));
        }

        // Hybrid MoE models (Gemma 4 26B A4B): each layer requires a
        // CPU MoE pass after the GPU dense FFN, so batched
        // dispatch_full_pipeline (GPU-only) would skip MoE entirely.
        // Instead, run token-by-token decode — each call correctly
        // interleaves GPU dense FFN + CPU MoE + GPU scalars. The
        // caller (generate.rs) only uses the last row of the prefill
        // output, so we return a zero-padded vec with only the final
        // position filled.
        let has_moe = layers.iter().any(|l| l.moe.is_some());
        if has_moe {
            let mut last_h = vec![0.0f32; hidden];
            for pos in 0..seq_len {
                let x_pos = &x[pos * hidden..(pos + 1) * hidden];
                last_h = MetalBackend::decode_token(
                    self, kv, layers, x_pos, hidden, inter, q_dim, kv_dim,
                    num_q_heads, num_kv_heads, head_dim, rope_base,
                );
            }
            let mut result = vec![0.0f32; seq_len * hidden];
            let dst_off = seq_len.saturating_sub(1) * hidden;
            result[dst_off..dst_off + hidden].copy_from_slice(&last_h);
            return Some(result);
        }

        let geglu = if layers.first().is_some_and(|l| l.activation == crate::Activation::GeluTanh) {
            &self.geglu_gelu_tanh_pipeline
        } else {
            &self.geglu_pipeline
        };
        Some(ops::full_pipeline::dispatch_full_pipeline(
            &self.queue, &self.bufs, &self.q4,
            geglu,
            &self.geglu_gelu_tanh_pipeline,
            &self.silu_pipeline,
            &self.gelu_tanh_pipeline,
            &self.q8_quant_pipeline,
            Some(&self.fused_attn_pipeline),
            &self.q8_matvec_pipeline.state,
            &self.q8_qkv_proj_pipeline,
            &self.q4k_matvec_pipeline.state, &self.q6k_matvec_pipeline.state,
            &self.rms_norm_pipeline, &self.residual_add_pipeline,
            &self.rms_norm_q8_pipeline, &self.residual_norm_q8_pipeline,
            Some(&self.q4k_qkv_proj_pipeline.state),
            Some(&self.q4kf_qkv_proj_pipeline.state),
            Some(&self.q4kf_proj_pipeline.state),
            Some(&self.rope_at_pos_pipeline),
            Some(&self.qk_norm_pipeline),
            Some(&self.scale_vector_pipeline),
            Some(kv),
            layers, x, hidden, inter, q_dim, kv_dim,
            seq_len, num_q_heads, num_kv_heads, head_dim,
            rope_base, use_qk_norm, softcap,
        ))
    }

    fn has_kv_cache(&self) -> bool { true }

    fn populate_kv_layer(
        &self, layer: usize,
        k_data: &[f32], v_data: &[f32],
        seq_len: usize, num_kv_heads: usize, head_dim: usize,
    ) {
        let mut cache_guard = self.kv_cache.lock().unwrap();
        if cache_guard.is_none() {
            *cache_guard = Some(self.create_kv_cache(layer + 1, 4096, num_kv_heads, head_dim));
        }
        let kv = cache_guard.as_mut().unwrap();
        while kv.layers.len() <= layer {
            kv.layers.push(ops::kv_cache::LayerKVCache::new(&self.bufs, 4096, num_kv_heads, head_dim));
        }

        let lc = &mut kv.layers[layer];
        let total = seq_len * num_kv_heads * head_dim;
        let k_ptr = lc.k_cache.contents() as *mut f32;
        let v_ptr = lc.v_cache.contents() as *mut f32;
        // SAFETY: k_ptr/v_ptr point to pre-allocated Metal buffers
        // sized for max_seq * kv_dim. k_data/v_data are borrow-checked
        // &[f32] params. Copy size is bounded by min(total, src.len()).
        unsafe {
            std::ptr::copy_nonoverlapping(k_data.as_ptr(), k_ptr, total.min(k_data.len()));
            std::ptr::copy_nonoverlapping(v_data.as_ptr(), v_ptr, total.min(v_data.len()));
        }
        lc.current_len = seq_len;
    }

    fn reset_kv_cache(&self) {
        let mut cache_guard = self.kv_cache.lock().unwrap();
        if let Some(ref mut kv) = *cache_guard {
            // Reset sequence position only — keep the GPU buffers
            // (avoids re-allocating ~1 GB on every new prompt).
            for layer in &mut kv.layers {
                layer.current_len = 0;
            }
        }
    }

    fn preallocate_kv_cache_per_layer(
        &self, shapes: &[(usize, usize)], max_seq: usize,
    ) {
        // Replace any existing cache — callers invoke this once per
        // model load, before the first decode dispatch. If we kept an
        // old cache sized with the wrong per-layer dims the first
        // decode would read off the end of a global-layer buffer.
        let mut cache_guard = self.kv_cache.lock().unwrap();
        *cache_guard = Some(self.create_kv_cache_per_layer(shapes, max_seq));
    }

    fn decode_token(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize, inter: usize,
        q_dim: usize, kv_dim: usize,
        num_q_heads: usize, num_kv_heads: usize, head_dim: usize,
        rope_base: f32,
    ) -> Option<Vec<f32>> {
        let num_layers = layers.len();
        let mut cache_guard = self.kv_cache.lock().unwrap();
        if cache_guard.is_none() {
            *cache_guard = Some(self.create_kv_cache(num_layers, 4096, num_kv_heads, head_dim));
        }
        let kv = cache_guard.as_mut().unwrap();
        // Grow if a later call uses a larger model than the first one
        // sized the cache for.
        while kv.layers.len() < num_layers {
            let l = &layers[kv.layers.len()];
            kv.layers.push(ops::kv_cache::LayerKVCache::new(
                &self.bufs, 4096, l.num_kv_heads, l.head_dim,
            ));
        }
        Some(MetalBackend::decode_token(self, kv, layers, x, hidden, inter, q_dim, kv_dim,
            num_q_heads, num_kv_heads, head_dim, rope_base))
    }

    fn decode_token_with_moe(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize, inter: usize,
        q_dim: usize, kv_dim: usize,
        num_q_heads: usize, num_kv_heads: usize, head_dim: usize,
        rope_base: f32,
        moe_fn: &mut dyn FnMut(usize, &[f32]) -> Vec<f32>,
    ) -> Option<Vec<f32>> {
        let num_layers = layers.len();
        let mut cache_guard = self.kv_cache.lock().unwrap();
        if cache_guard.is_none() {
            *cache_guard = Some(self.create_kv_cache(num_layers, 4096, num_kv_heads, head_dim));
        }
        let kv = cache_guard.as_mut().unwrap();
        while kv.layers.len() < num_layers {
            let l = &layers[kv.layers.len()];
            kv.layers.push(ops::kv_cache::LayerKVCache::new(
                &self.bufs, 4096, l.num_kv_heads, l.head_dim,
            ));
        }
        Some(MetalBackend::decode_token_with_moe_fn(self, kv, layers, x,
            hidden, inter, q_dim, kv_dim,
            num_q_heads, num_kv_heads, head_dim, rope_base, Some(moe_fn)))
    }

    fn decode_token_split_profile(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize, inter: usize,
        q_dim: usize, kv_dim: usize,
        num_q_heads: usize, num_kv_heads: usize, head_dim: usize,
        rope_base: f32,
    ) -> (Option<Vec<f32>>, f64, f64, f64) {
        let num_layers = layers.len();
        let mut cache_guard = self.kv_cache.lock().unwrap();
        if cache_guard.is_none() {
            *cache_guard = Some(self.create_kv_cache(num_layers, 4096, num_kv_heads, head_dim));
        }
        let kv = cache_guard.as_mut().unwrap();
        let (res, ta, tgu, td) = MetalBackend::decode_token_split_profile(
            self, kv, layers, x, hidden, inter, q_dim, kv_dim,
            num_q_heads, num_kv_heads, head_dim, rope_base,
        );
        (Some(res), ta, tgu, td)
    }
}
