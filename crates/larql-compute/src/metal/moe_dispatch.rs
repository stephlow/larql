//! GPU expert dispatch for per-layer Q4_K MoE models (§5.12).
//!
//! Called when a MoE layer's expert weights are in `QuantFormat::Q4_K`
//! (per-layer files, not BF16 blob). The router runs on CPU (cheap: 2816×128
//! matmul), expert FFNs run on GPU using existing Q4_K shaders.
//!
//! Flow per MoE layer (after the standard GPU commit for `h_post_attn`):
//!
//! 1. CPU: router projection + softmax + top-K + renormalize.
//! 2. CPU→GPU: write K gate+up Q4_K byte slices directly into Metal staging
//!    buffers (shared memory write, single copy per expert).
//! 3. GPU: `q4k_ffn_gate_up` dispatch — all K experts' gate+up in one call.
//! 4. GPU: GELU-tanh activation.
//! 5. CPU→GPU: write K down Q4_K slices into staging buffers.
//! 6. GPU: K × `q4k_matvec` for expert down projections.
//! 7. Commit + wait (one GPU sync for expert compute).
//! 8. CPU: read back K × hidden expert outputs, weighted sum → `moe_out`.

use metal::*;
use std::ffi::c_void;

use super::buffers::read_buffer_f32;
use super::MetalBackend;
use crate::cpu::ops::moe::cpu_moe_route;
use crate::MoeLayerWeights;

impl MetalBackend {
    /// High-level decode step using GPU expert dispatch for Q4_K per-layer format.
    pub fn decode_token_q4k_moe(
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
        norm_eps: f32,
        get_expert: impl Fn(usize, usize) -> Option<(Vec<u8>, Vec<u8>)>,
    ) -> Option<Vec<f32>> {
        let mut kv_guard = self.kv_cache.lock().unwrap();
        if kv_guard.is_none() {
            let shapes: Vec<(usize, usize)> = layers
                .iter()
                .map(|l| (l.num_kv_heads, l.head_dim))
                .collect();
            *kv_guard = Some(super::ops::kv_cache::KVCache::new_per_layer(
                &self.bufs, &shapes, 4096,
            ));
        }
        let kv = kv_guard.as_mut().unwrap();
        while kv.layers.len() < layers.len() {
            let l = kv.layers.len();
            let (nkv, hd) = (layers[l].num_kv_heads, layers[l].head_dim);
            kv.layers.push(super::ops::kv_cache::LayerKVCache::new(
                &self.bufs, 4096, nkv, hd,
            ));
        }

        let mut moe_fn = {
            let get_expert_ref = &get_expert;
            move |layer_idx: usize, h_post_attn: &[f32]| -> Vec<f32> {
                let moe = match layers[layer_idx].moe.as_ref() {
                    Some(m) => m,
                    None => return vec![0.0f32; hidden],
                };
                self.gpu_moe_dispatch(h_post_attn, moe, norm_eps, &|expert_idx| {
                    get_expert_ref(layer_idx, expert_idx)
                })
            }
        };

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
            Some(&mut moe_fn),
        ))
    }

    /// GPU expert dispatch for Q4_K per-layer expert weights.
    ///
    /// Writes expert bytes DIRECTLY into Metal staging buffers (shared memory)
    /// to avoid a triple-copy. Each expert's gate+up / down bytes are copied
    /// from the mmap-backed Vec<u8> into the Metal buffer's `contents()` pointer
    /// in one memcpy — no intermediate staging Vec.
    pub fn gpu_moe_dispatch(
        &self,
        h_post_attn: &[f32],
        moe: &MoeLayerWeights<'_>,
        eps: f32,
        get_expert_bytes: &dyn Fn(usize) -> Option<(Vec<u8>, Vec<u8>)>,
    ) -> Vec<f32> {
        let hidden = h_post_attn.len();
        let inter = moe.intermediate_size;
        let inter_padded = inter.div_ceil(256) * 256;
        let top_k = moe.top_k;

        // ── 1. CPU router ──────────────────────────────────────────────────
        let h_norm = if !moe.pre_experts_norm.is_empty() {
            let rms = (h_post_attn.iter().map(|v| v * v).sum::<f32>() / hidden as f32 + eps).sqrt();
            h_post_attn
                .iter()
                .zip(moe.pre_experts_norm)
                .map(|(x, w)| x / rms * (w + 0.0))
                .collect::<Vec<f32>>()
        } else {
            h_post_attn.to_vec()
        };
        let (expert_indices, expert_weights) = cpu_moe_route(&h_norm, moe, eps);

        // ── 2. Pre-allocate Metal staging buffers, write expert bytes directly ──
        // Q4_K: bytes per row = (hidden / 256) * 144.
        let row_bytes = (hidden / 256) * 144;
        let gate_half_bytes = inter * row_bytes; // bytes for gate rows of one expert
        let up_half_bytes = inter * row_bytes; // bytes for up rows of one expert
        let down_row_bytes = (inter_padded / 256) * 144; // Q4_K down: cols = inter_padded
        let down_expert_bytes = hidden * down_row_bytes; // one expert's down matrix

        // Allocate shared-memory Metal buffers — write CPU→GPU via contents() ptr.
        let gate_buf = self.bufs.output((top_k * gate_half_bytes) as u64);
        let up_buf = self.bufs.output((top_k * up_half_bytes) as u64);
        let gate_ptr = gate_buf.contents() as *mut u8;
        let up_ptr = up_buf.contents() as *mut u8;

        let mut down_bufs: Vec<Buffer> = Vec::with_capacity(top_k);
        let mut valid_weights: Vec<f32> = Vec::with_capacity(top_k);
        let mut valid_count = 0usize;

        for (k, &ei) in expert_indices.iter().enumerate() {
            let Some((gu_bytes, dn_bytes)) = get_expert_bytes(ei) else {
                continue;
            };
            let half = gate_half_bytes;
            if gu_bytes.len() < 2 * half {
                continue;
            }

            // Write gate and up directly into pre-allocated Metal buffer.
            // SAFETY: gate_ptr/up_ptr point to Metal shared memory (MTLResourceOptions::StorageModeShared).
            // Offsets are bounded by `top_k * gate_half_bytes` allocated above.
            // gate: bytes 0..half of gu_bytes
            // up:   bytes half..2*half of gu_bytes
            unsafe {
                std::ptr::copy_nonoverlapping(
                    gu_bytes.as_ptr(),
                    gate_ptr.add(valid_count * gate_half_bytes),
                    gate_half_bytes,
                );
                std::ptr::copy_nonoverlapping(
                    gu_bytes.as_ptr().add(half),
                    up_ptr.add(valid_count * up_half_bytes),
                    up_half_bytes,
                );
            }

            // Down: allocate a Metal buffer and write directly.
            let dn_buf = self.bufs.output(down_expert_bytes as u64);
            let dn_ptr = dn_buf.contents() as *mut u8;
            let copy_len = dn_bytes.len().min(down_expert_bytes);
            unsafe {
                std::ptr::copy_nonoverlapping(dn_bytes.as_ptr(), dn_ptr, copy_len);
            }
            down_bufs.push(dn_buf);
            valid_weights.push(expert_weights[k]);
            valid_count += 1;
        }

        if valid_count == 0 {
            return vec![0.0f32; hidden];
        }

        // ── 3. GPU: q4k_ffn_gate_up for all valid_count experts ──────────
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        let x_buf = self.bufs.transient_from_f32(&h_norm);
        let n_rows = (valid_count * inter) as u32;
        let k_cols = hidden as u32;
        let tgs = (valid_count as u64 * inter as u64)
            .div_ceil(crate::metal::shaders::q4k_ffn_gate_up::ROWS_PER_TG);

        let g_out = self.bufs.output((valid_count * inter * 4) as u64);
        let u_out = self.bufs.output((valid_count * inter * 4) as u64);

        enc.set_compute_pipeline_state(&self.q4k_ffn_gate_up_pipeline.state);
        enc.set_buffer(0, Some(&gate_buf), 0);
        enc.set_buffer(1, Some(&up_buf), 0);
        enc.set_buffer(2, Some(&x_buf), 0);
        enc.set_buffer(3, Some(&g_out), 0);
        enc.set_buffer(4, Some(&u_out), 0);
        enc.set_bytes(5, 4, &n_rows as *const u32 as *const c_void);
        enc.set_bytes(6, 4, &k_cols as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(tgs * 2, 1, 1),
            MTLSize::new(crate::metal::shaders::q4k_ffn_gate_up::THREADS_PER_TG, 1, 1),
        );

        // ── 4. GPU: GELU-tanh activation ─────────────────────────────────
        let act_len = (valid_count * inter) as u32;
        let act_stride = inter_padded;
        let act_buf = self.bufs.output((valid_count * act_stride * 4) as u64);
        enc.set_compute_pipeline_state(&self.geglu_gelu_tanh_pipeline);
        enc.set_buffer(0, Some(&g_out), 0);
        enc.set_buffer(1, Some(&u_out), 0);
        enc.set_buffer(2, Some(&act_buf), 0);
        enc.set_bytes(3, 4, &act_len as *const u32 as *const c_void);
        enc.dispatch_threads(
            MTLSize::new(valid_count as u64 * inter as u64, 1, 1),
            MTLSize::new(256.min(valid_count as u64 * inter as u64), 1, 1),
        );

        // ── 5–6. GPU: down projection per expert ─────────────────────────
        // Each expert e uses act[e*inter_padded..e*inter_padded+inter] as input
        // and produces expert_outs[e*hidden..(e+1)*hidden]. The activation bytes
        // beyond `inter` stay zero from `bufs.output`, so padded Q4_K down rows
        // contribute nothing.
        let n_out = hidden as u32;
        let k_in = inter_padded as u32;
        let down_tgs = (hidden as u64).div_ceil(crate::metal::shaders::q4k_matvec::ROWS_PER_TG);
        let expert_outs = self.bufs.output((valid_count * hidden * 4) as u64);

        for e in 0..valid_count {
            let act_offset = (e * act_stride * 4) as u64;
            let out_offset = (e * hidden * 4) as u64;
            enc.set_compute_pipeline_state(&self.q4k_matvec_pipeline.state);
            enc.set_buffer(0, Some(&down_bufs[e]), 0);
            enc.set_buffer(1, Some(&act_buf), act_offset);
            enc.set_buffer(2, Some(&expert_outs), out_offset);
            enc.set_bytes(3, 4, &n_out as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &k_in as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(down_tgs, 1, 1),
                MTLSize::new(crate::metal::shaders::q4k_matvec::THREADS_PER_TG, 1, 1),
            );
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        // ── 7. CPU: weighted sum ─────────────────────────────────────────
        let all_expert_outputs = read_buffer_f32(&expert_outs, valid_count * hidden);
        let mut moe_out = vec![0.0f32; hidden];
        for e in 0..valid_count {
            let w = valid_weights[e];
            let out_slice = &all_expert_outputs[e * hidden..(e + 1) * hidden];
            for (acc, &v) in moe_out.iter_mut().zip(out_slice) {
                *acc += v * w;
            }
        }

        // Post-experts norm (Gemma 4 `post_feedforward_layernorm_2`).
        if !moe.post_experts_norm.is_empty() {
            let rms = (moe_out.iter().map(|v| v * v).sum::<f32>() / hidden as f32 + eps).sqrt();
            for (v, &w) in moe_out.iter_mut().zip(moe.post_experts_norm) {
                *v = *v / rms * (w + 0.0);
            }
        }
        moe_out
    }
}
