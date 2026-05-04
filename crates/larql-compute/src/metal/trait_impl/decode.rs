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
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_base: f32,
        use_qk_norm: bool,
        softcap: f32,
    ) -> Option<Vec<f32>> {
        let geglu = if layers
            .first()
            .is_some_and(|l| l.activation == crate::Activation::GeluTanh)
        {
            &self.geglu_gelu_tanh_pipeline
        } else {
            &self.geglu_pipeline
        };
        Some(ops::full_pipeline::dispatch_full_pipeline(
            &self.queue,
            &self.bufs,
            &self.q4,
            geglu,
            &self.geglu_gelu_tanh_pipeline,
            &self.silu_pipeline,
            &self.gelu_tanh_pipeline,
            &self.q8_quant_pipeline,
            Some(&self.fused_attn_pipeline),
            &self.q8_matvec_pipeline.state,
            &self.q8_qkv_proj_pipeline.state,
            &self.q4k_matvec_pipeline,
            Some(&self.q4k_matmul_pipeline),
            &self.q6k_matvec_pipeline,
            &self.rms_norm_pipeline,
            &self.residual_add_pipeline,
            &self.rms_norm_q8_pipeline,
            &self.residual_norm_q8_pipeline,
            Some(&self.q4k_qkv_proj_pipeline.state),
            Some(&self.q4kf_qkv_proj_pipeline.state),
            Some(&self.q4kf_proj_pipeline.state),
            None,
            Some(&self.qk_norm_pipeline),
            Some(&self.scale_vector_pipeline),
            Some(&self.q4k_geglu_silu_down_pipeline),
            Some(&self.q4k_geglu_gelu_tanh_down_pipeline),
            Some(&self.q6k_geglu_silu_down_pipeline),
            Some(&self.q6k_geglu_gelu_tanh_down_pipeline),
            None,
            layers,
            x,
            hidden,
            inter,
            q_dim,
            kv_dim,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rope_base,
            use_qk_norm,
            softcap,
            None, // moe_fn: no MoE callback for full_pipeline_q4
        ))
    }

    fn multi_layer_q4_ffn(
        &self,
        layers_q4: &[(&[u8], &[u8], &[u8])],
        x: &[f32],
        inter: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        Some(MetalBackend::multi_layer_q4_ffn(
            self, layers_q4, x, inter, hidden,
        ))
    }

    fn prefill_q4(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_base: f32,
        use_qk_norm: bool,
        softcap: f32,
    ) -> Option<Vec<f32>> {
        let mut cache_guard = self.kv_cache.lock().unwrap();
        let kv = self.ensure_kv_cache_for_layers(
            &mut cache_guard,
            layers,
            crate::metal::decode::DEFAULT_KV_CACHE_MAX_SEQ,
        );

        let has_moe = layers.iter().any(|l| l.moe.is_some());
        let geglu = if layers
            .first()
            .is_some_and(|l| l.activation == crate::Activation::GeluTanh)
        {
            &self.geglu_gelu_tanh_pipeline
        } else {
            &self.geglu_pipeline
        };

        // Concrete macro to avoid duplicating the 30-param dispatch call.
        macro_rules! run_dispatch {
            ($moe_fn:expr) => {
                ops::full_pipeline::dispatch_full_pipeline(
                    &self.queue,
                    &self.bufs,
                    &self.q4,
                    geglu,
                    &self.geglu_gelu_tanh_pipeline,
                    &self.silu_pipeline,
                    &self.gelu_tanh_pipeline,
                    &self.q8_quant_pipeline,
                    Some(&self.fused_attn_pipeline),
                    &self.q8_matvec_pipeline.state,
                    &self.q8_qkv_proj_pipeline.state,
                    &self.q4k_matvec_pipeline,
                    Some(&self.q4k_matmul_pipeline),
                    &self.q6k_matvec_pipeline,
                    &self.rms_norm_pipeline,
                    &self.residual_add_pipeline,
                    &self.rms_norm_q8_pipeline,
                    &self.residual_norm_q8_pipeline,
                    Some(&self.q4k_qkv_proj_pipeline.state),
                    Some(&self.q4kf_qkv_proj_pipeline.state),
                    Some(&self.q4kf_proj_pipeline.state),
                    Some(&self.rope_at_pos_pipeline),
                    Some(&self.qk_norm_pipeline),
                    Some(&self.scale_vector_pipeline),
                    Some(&self.q4k_geglu_silu_down_pipeline),
                    Some(&self.q4k_geglu_gelu_tanh_down_pipeline),
                    Some(&self.q6k_geglu_silu_down_pipeline),
                    Some(&self.q6k_geglu_gelu_tanh_down_pipeline),
                    Some(kv),
                    layers,
                    x,
                    hidden,
                    inter,
                    q_dim,
                    kv_dim,
                    seq_len,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    rope_base,
                    use_qk_norm,
                    softcap,
                    $moe_fn,
                )
            };
        }

        if has_moe {
            // Per-layer MoE callback: runs CPU experts for all seq_len positions,
            // accumulates into new_h, then applies outer post-FFN norm + layer_scalar.
            // GPU layer_scalar step is skipped for MoE layers in dispatch_full_pipeline
            // (see `is_moe_layer` guard) so this closure owns the combine step.
            let mut moe_closure = |layer_idx: usize, h_post_attn: &[f32], new_h: &mut [f32]| {
                let layer = &layers[layer_idx];
                let moe_block = match layer.moe.as_ref() {
                    Some(m) => m,
                    None => return,
                };
                let layer_eps = layer.eps;
                let layer_norm_offset = layer.norm_offset;

                // 1. CPU MoE for each position: accumulate into new_h.
                for pos in 0..seq_len {
                    let ha = &h_post_attn[pos * hidden..(pos + 1) * hidden];
                    let moe_out = crate::cpu::ops::moe::cpu_moe_forward(
                        ha,
                        moe_block,
                        layer_norm_offset,
                        layer_eps,
                    );
                    let nh = &mut new_h[pos * hidden..(pos + 1) * hidden];
                    for (i, v) in moe_out.iter().enumerate() {
                        nh[i] += v;
                    }
                }

                // 2. Outer post-FFN norm + layer_scalar per position.
                // Matches moe_combine::apply_outer_combine for batched positions.
                for pos in 0..seq_len {
                    let ha = &h_post_attn[pos * hidden..(pos + 1) * hidden];
                    let nh = &mut new_h[pos * hidden..(pos + 1) * hidden];

                    if layer.moe_combined_output_norm {
                        let outer_w = layer.moe_outer_post_norm.or(layer.post_ffn_norm);
                        if let Some(w) = outer_w {
                            let combined: Vec<f32> =
                                nh.iter().zip(ha).map(|(h, a)| h - a).collect();
                            let rms = (combined.iter().map(|v| v * v).sum::<f32>() / hidden as f32
                                + layer_eps)
                                .sqrt();
                            for (i, (&c, &wt)) in combined.iter().zip(w.iter()).enumerate() {
                                nh[i] = ha[i] + c / rms * (wt + layer_norm_offset);
                            }
                        }
                    }

                    let ls = layer.layer_scalar;
                    if ls != 0.0 && ls != 1.0 {
                        for v in nh.iter_mut() {
                            *v *= ls;
                        }
                    }
                }
            };
            return Some(run_dispatch!(Some(
                &mut moe_closure as &mut dyn FnMut(usize, &[f32], &mut [f32])
            )));
        }

        Some(run_dispatch!(None))
    }

    fn has_kv_cache(&self) -> bool {
        true
    }

    fn populate_kv_layer(
        &self,
        layer: usize,
        k_data: &[f32],
        v_data: &[f32],
        seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) {
        let mut cache_guard = self.kv_cache.lock().unwrap();
        if cache_guard.is_none() {
            *cache_guard = Some(self.create_kv_cache(
                layer + 1,
                crate::metal::decode::DEFAULT_KV_CACHE_MAX_SEQ,
                num_kv_heads,
                head_dim,
            ));
        }
        let kv = cache_guard.as_mut().unwrap();
        while kv.layers.len() <= layer {
            kv.layers.push(ops::kv_cache::LayerKVCache::new(
                &self.bufs,
                crate::metal::decode::DEFAULT_KV_CACHE_MAX_SEQ,
                num_kv_heads,
                head_dim,
            ));
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
            for layer in &mut kv.layers {
                layer.current_len = 0;
            }
        }
    }

    fn kv_cache_len(&self) -> usize {
        self.kv_cache
            .lock()
            .unwrap()
            .as_ref()
            .map(|kv| kv.current_len())
            .unwrap_or(0)
    }

    fn truncate_kv_cache(&self, len: usize) {
        if let Some(ref mut kv) = *self.kv_cache.lock().unwrap() {
            for layer in &mut kv.layers {
                layer.current_len = len;
            }
        }
    }

    fn preallocate_kv_cache_per_layer(&self, shapes: &[(usize, usize)], max_seq: usize) {
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
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_base: f32,
    ) -> Option<Vec<f32>> {
        let mut cache_guard = self.kv_cache.lock().unwrap();
        let kv = self.ensure_kv_cache_for_layers(
            &mut cache_guard,
            layers,
            crate::metal::decode::DEFAULT_KV_CACHE_MAX_SEQ,
        );
        Some(MetalBackend::decode_token(
            self,
            kv,
            layers,
            x,
            hidden,
            inter,
            q_dim,
            kv_dim,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rope_base,
        ))
    }

    fn decode_token_with_moe(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_base: f32,
        moe_fn: &mut dyn FnMut(usize, &[f32]) -> Vec<f32>,
    ) -> Option<Vec<f32>> {
        let mut cache_guard = self.kv_cache.lock().unwrap();
        let kv = self.ensure_kv_cache_for_layers(
            &mut cache_guard,
            layers,
            crate::metal::decode::DEFAULT_KV_CACHE_MAX_SEQ,
        );
        Some(MetalBackend::decode_token_with_moe_fn(
            self,
            kv,
            layers,
            x,
            hidden,
            inter,
            q_dim,
            kv_dim,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rope_base,
            Some(moe_fn),
        ))
    }

    fn decode_token_with_moe_split(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_base: f32,
        moe_fire_fn: &mut dyn FnMut(usize, &[f32]),
        moe_collect_fn: &mut dyn FnMut(usize) -> Vec<f32>,
    ) -> Option<Vec<f32>> {
        let mut cache_guard = self.kv_cache.lock().unwrap();
        let kv = self.ensure_kv_cache_for_layers(
            &mut cache_guard,
            layers,
            crate::metal::decode::DEFAULT_KV_CACHE_MAX_SEQ,
        );
        // Wrap fire so its return value is ignored — the decode-loop closure
        // already discards moe_fn's output when split mode is active.
        let mut fire_wrapper = |layer: usize, h: &[f32]| -> Vec<f32> {
            moe_fire_fn(layer, h);
            Vec::new()
        };
        Some(MetalBackend::decode_token_with_moe_split_fn(
            self,
            kv,
            layers,
            x,
            hidden,
            inter,
            q_dim,
            kv_dim,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rope_base,
            Some(&mut fire_wrapper),
            Some(moe_collect_fn),
        ))
    }

    fn decode_token_split_profile(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_base: f32,
    ) -> (Option<Vec<f32>>, f64, f64, f64) {
        // Per-stage GPU timing comes from `decode_token_with_moe_split_fn`
        // when `LARQL_PROFILE_SPLIT=1` is set: paired commit/wait boundaries
        // around the attention vs FFN blocks land per-stage GPU windows in
        // a thread-local. We read them back here. Without the env flag,
        // we fall back to whole-token wall time in `attn_ms` so callers
        // still see something useful — but they should set the flag to
        // get the actual split.
        use crate::metal::decode::profile;
        let t0 = std::time::Instant::now();
        let result = <Self as DecodeBackend>::decode_token(
            self,
            layers,
            x,
            hidden,
            inter,
            q_dim,
            kv_dim,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rope_base,
        );
        let timings = profile::take_last_split_timings().unwrap_or_else(|| {
            // Fallback: whole-token wall time in `attn_ms`. Caller likely
            // forgot to set `LARQL_PROFILE_SPLIT=1`.
            let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
            profile::ProfileTimings {
                attn_ms: total_ms,
                gate_up_ms: 0.0,
                down_ms: 0.0,
            }
        });
        eprintln!("{}", timings.format_summary(layers.len()));
        (result, timings.attn_ms, timings.gate_up_ms, timings.down_ms)
    }
}
