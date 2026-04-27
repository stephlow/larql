//! GPU expert dispatch for per-layer Q4_K MoE models (§5.12).
//!
//! Called when a MoE layer's expert weights are in `QuantFormat::Q4_K`
//! (per-layer files, not BF16 blob). The router runs on CPU (cheap: 2816×128
//! matmul), expert FFNs run on GPU using existing Q4_K shaders.
//!
//! Flow per MoE layer (after the standard GPU commit for `h_post_attn`):
//!
//! 1. CPU: pre-experts norm + router projection + softmax + top-K + renorm.
//! 2. CPU→GPU: write the K selected experts' gate / up / down byte slices
//!    DIRECTLY into pre-allocated Metal staging buffers (one memcpy each).
//! 3. GPU: `q4k_ffn_gate_up` over all K experts in one dispatch.
//! 4. GPU: K × `geglu_gelu_tanh` — one per expert at strided act_buf offset
//!    `e × inter_padded` so down's `K = inter_padded` reads see zero padding.
//! 5. GPU: K × `q4k_matvec` for expert down projections.
//! 6. Commit + wait (one GPU sync per MoE layer).
//! 7. CPU: read back K × hidden expert outputs, weighted sum → `moe_out`.
//!
//! Phase 2 (2026-04-26): all scratch is pre-allocated once per decode call
//! via `MoeScratch::new(...)` and reused across every MoE layer. Previously
//! each layer called `bufs.output(...)` ~10 times (~120ms allocation overhead
//! per token at 30 MoE layers on M3 Max). Buffer sizes are constant per model
//! — `(top_k, hidden, inter_padded)` — so the buffers can stay live for the
//! whole decode and serve every layer's expert routing.

use metal::*;
use std::ffi::c_void;

use super::buffers::{read_buffer_f32, BufferCache};
use super::MetalBackend;
use crate::cpu::ops::moe::cpu_moe_route;
use crate::MoeLayerWeights;

/// Pre-allocated scratch for the whole MoE decode loop.
///
/// All sizes are determined by `(top_k, hidden, intermediate_size)` of the
/// first MoE layer, which is constant across MoE layers in the architectures
/// we currently target (Gemma 4 26B A4B). Sizing assumes Q4_K weights with
/// 256-element super-blocks, 144 bytes per row-block.
///
/// `act_buf` is sized to `top_k × inter_padded` and zero-initialised so the
/// `inter_padded - inter` padding columns of every expert's strided slice
/// contribute nothing through the down projection — required when
/// `moe.intermediate_size` is not a multiple of 256 (e.g. Gemma 4 26B's 2112
/// → inter_padded 2304).
pub(super) struct MoeScratch {
    pub(super) top_k: usize,
    pub(super) inter: usize,
    pub(super) inter_padded: usize,
    pub(super) hidden: usize,
    pub(super) row_bytes: usize,
    pub(super) down_row_bytes: usize,

    pub(super) gate_buf: Buffer,
    pub(super) up_buf: Buffer,
    pub(super) down_bufs: Vec<Buffer>,

    pub(super) x_buf: Buffer,
    pub(super) g_out: Buffer,
    pub(super) u_out: Buffer,
    pub(super) act_buf: Buffer,
    pub(super) expert_outs: Buffer,
}

impl MoeScratch {
    pub(super) fn new(bufs: &BufferCache, top_k: usize, hidden: usize, inter: usize) -> Self {
        let block = larql_models::quant::ggml::Q4_K_BLOCK_ELEMS;
        let bytes_per_block = larql_models::quant::ggml::Q4_K_BLOCK_BYTES;
        let inter_padded = inter.div_ceil(block) * block;
        // Q4_K row stride: one super-block per Q4_K_BLOCK_ELEMS elements,
        // Q4_K_BLOCK_BYTES bytes per super-block.
        let row_bytes = (hidden / block) * bytes_per_block;
        let down_row_bytes = (inter_padded / block) * bytes_per_block;

        let gate_buf = bufs.output((top_k * inter * row_bytes) as u64);
        let up_buf = bufs.output((top_k * inter * row_bytes) as u64);
        let down_bufs: Vec<Buffer> = (0..top_k)
            .map(|_| bufs.output((hidden * down_row_bytes) as u64))
            .collect();

        let x_buf = bufs.output((hidden * 4) as u64);
        let g_out = bufs.output((top_k * inter * 4) as u64);
        let u_out = bufs.output((top_k * inter * 4) as u64);
        let act_buf = bufs.output((top_k * inter_padded * 4) as u64);
        let expert_outs = bufs.output((top_k * hidden * 4) as u64);

        // Zero the padding tails once. GEGLU writes only the first `inter`
        // floats of each expert's `inter_padded`-strided slice, so the
        // remaining `inter_padded - inter` floats stay zero forever.
        unsafe {
            let ptr = act_buf.contents() as *mut f32;
            std::ptr::write_bytes(ptr, 0, top_k * inter_padded);
        }

        Self {
            top_k,
            inter,
            inter_padded,
            hidden,
            row_bytes,
            down_row_bytes,
            gate_buf,
            up_buf,
            down_bufs,
            x_buf,
            g_out,
            u_out,
            act_buf,
            expert_outs,
        }
    }
}

impl MetalBackend {
    /// High-level decode step using GPU expert dispatch for Q4_K per-layer format.
    ///
    /// `get_expert(layer_idx, expert_idx)` returns the (gate+up, down) byte
    /// slices for the requested expert, borrowed from the model weights (mmap).
    /// The borrow only needs to outlive the closure call — `gpu_moe_dispatch`
    /// memcpys both slices into pre-allocated Metal buffers before returning.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_token_q4k_moe<'w, F>(
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
        get_expert: F,
    ) -> Option<Vec<f32>>
    where
        F: Fn(usize, usize) -> Option<(&'w [u8], &'w [u8])>,
    {
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

        // Allocate scratch once for the whole decode call. Sized from the first
        // MoE layer; we assume top_k / intermediate_size are constant across
        // MoE layers (true for Gemma 4 26B A4B and similar). When future
        // architectures violate that we'll need either per-layer scratch or
        // the worst-case max — but no current model exercises that path.
        let scratch = layers
            .iter()
            .find_map(|l| l.moe.as_ref())
            .map(|m| MoeScratch::new(&self.bufs, m.top_k, hidden, m.intermediate_size));
        let scratch_ref = scratch.as_ref();

        let mut moe_fn = {
            let get_expert_ref = &get_expert;
            move |layer_idx: usize, h_post_attn: &[f32]| -> Vec<f32> {
                let moe = match layers[layer_idx].moe.as_ref() {
                    Some(m) => m,
                    None => return vec![0.0f32; hidden],
                };
                let scratch = scratch_ref
                    .expect("MoE layer present but no scratch allocated — model has MoE");
                self.gpu_moe_dispatch_with_scratch(
                    h_post_attn,
                    moe,
                    norm_eps,
                    scratch,
                    |expert_idx| get_expert_ref(layer_idx, expert_idx),
                )
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

    /// GPU expert dispatch with pre-allocated scratch.
    ///
    /// Per call this does:
    ///   - 1 CPU pre-experts norm + router pass (~hidden² FLOPs, cheap).
    ///   - top_k × 2 host→shared-memory memcpys (one per gate+up + one per
    ///     down byte slice); no Metal allocations in the hot path.
    ///   - 1 fused gate+up dispatch + top_k activation dispatches +
    ///     top_k down dispatches → committed and waited on once.
    ///   - 1 readback of `top_k × hidden` f32 expert outputs + CPU weighted sum
    ///     and post-experts norm.
    pub(super) fn gpu_moe_dispatch_with_scratch<'w, F>(
        &self,
        h_post_attn: &[f32],
        moe: &MoeLayerWeights<'_>,
        eps: f32,
        scratch: &MoeScratch,
        get_expert_bytes: F,
    ) -> Vec<f32>
    where
        F: Fn(usize) -> Option<(&'w [u8], &'w [u8])>,
    {
        let hidden = h_post_attn.len();
        let inter = moe.intermediate_size;
        let inter_padded = scratch.inter_padded;
        let top_k = moe.top_k;
        debug_assert_eq!(top_k, scratch.top_k, "MoE top_k drift across layers");
        debug_assert_eq!(
            inter, scratch.inter,
            "MoE intermediate_size drift across layers"
        );
        debug_assert_eq!(
            hidden, scratch.hidden,
            "MoE hidden_size drift across layers"
        );

        // ── 1. CPU pre-experts norm + router ─────────────────────────────
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

        // ── 2. Stage expert weight bytes into pre-allocated Metal buffers ─
        let row_bytes = scratch.row_bytes;
        let gate_half_bytes = inter * row_bytes;
        let up_half_bytes = inter * row_bytes;
        let down_expert_bytes = hidden * scratch.down_row_bytes;

        let gate_ptr = scratch.gate_buf.contents() as *mut u8;
        let up_ptr = scratch.up_buf.contents() as *mut u8;

        let mut valid_weights: Vec<f32> = Vec::with_capacity(top_k);
        let mut valid_count = 0usize;

        for (k, &ei) in expert_indices.iter().enumerate() {
            let Some((gu_bytes, dn_bytes)) = get_expert_bytes(ei) else {
                continue;
            };
            if gu_bytes.len() < 2 * gate_half_bytes {
                continue;
            }

            // Q4_K layout: gate || up, each `inter * row_bytes` bytes.
            // SAFETY: gate_ptr / up_ptr are StorageModeShared Metal buffer
            // contents; offsets are bounded by `top_k * gate_half_bytes`
            // allocated up front (see `MoeScratch::new`). Writes complete
            // before the encoder dispatches the matvec that reads them.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    gu_bytes.as_ptr(),
                    gate_ptr.add(valid_count * gate_half_bytes),
                    gate_half_bytes,
                );
                std::ptr::copy_nonoverlapping(
                    gu_bytes.as_ptr().add(gate_half_bytes),
                    up_ptr.add(valid_count * up_half_bytes),
                    up_half_bytes,
                );
            }

            let dn_dst = scratch.down_bufs[valid_count].contents() as *mut u8;
            let copy_len = dn_bytes.len().min(down_expert_bytes);
            unsafe {
                std::ptr::copy_nonoverlapping(dn_bytes.as_ptr(), dn_dst, copy_len);
            }

            valid_weights.push(expert_weights[k]);
            valid_count += 1;
        }

        if valid_count == 0 {
            return vec![0.0f32; hidden];
        }

        // ── 3. Stage router-normed input into pre-allocated x_buf ─────────
        unsafe {
            let x_ptr = scratch.x_buf.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(h_norm.as_ptr(), x_ptr, hidden);
        }

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        // ── 4. q4k_ffn_gate_up over all valid_count experts at once ──────
        let n_rows = (valid_count * inter) as u32;
        let k_cols = hidden as u32;
        let tgs = (valid_count as u64 * inter as u64)
            .div_ceil(crate::metal::shaders::q4k_ffn_gate_up::ROWS_PER_TG);

        enc.set_compute_pipeline_state(&self.q4k_ffn_gate_up_pipeline.state);
        enc.set_buffer(0, Some(&scratch.gate_buf), 0);
        enc.set_buffer(1, Some(&scratch.up_buf), 0);
        enc.set_buffer(2, Some(&scratch.x_buf), 0);
        enc.set_buffer(3, Some(&scratch.g_out), 0);
        enc.set_buffer(4, Some(&scratch.u_out), 0);
        enc.set_bytes(5, 4, &n_rows as *const u32 as *const c_void);
        enc.set_bytes(6, 4, &k_cols as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(tgs * 2, 1, 1),
            MTLSize::new(crate::metal::shaders::q4k_ffn_gate_up::THREADS_PER_TG, 1, 1),
        );

        // ── 5. GELU-tanh activation per expert (strided to inter_padded) ──
        // Gate/up output is packed at stride `inter`; activation must land at
        // stride `inter_padded` because down reads `K = inter_padded`. One
        // small dispatch per expert with the right offsets gets us strided
        // output without a new shader. valid_count × ~5µs ≪ allocation cost.
        let inter_u32 = inter as u32;
        for e in 0..valid_count {
            let g_offset = (e * inter * 4) as u64;
            let u_offset = (e * inter * 4) as u64;
            let a_offset = (e * inter_padded * 4) as u64;
            enc.set_compute_pipeline_state(&self.geglu_gelu_tanh_pipeline);
            enc.set_buffer(0, Some(&scratch.g_out), g_offset);
            enc.set_buffer(1, Some(&scratch.u_out), u_offset);
            enc.set_buffer(2, Some(&scratch.act_buf), a_offset);
            enc.set_bytes(3, 4, &inter_u32 as *const u32 as *const c_void);
            enc.dispatch_threads(
                MTLSize::new(inter as u64, 1, 1),
                MTLSize::new(256.min(inter as u64), 1, 1),
            );
        }

        // ── 6. Down projection per expert ────────────────────────────────
        let n_out = hidden as u32;
        let k_in = inter_padded as u32;
        let down_tgs = (hidden as u64).div_ceil(crate::metal::shaders::q4k_matvec::ROWS_PER_TG);

        for e in 0..valid_count {
            let act_offset = (e * inter_padded * 4) as u64;
            let out_offset = (e * hidden * 4) as u64;
            enc.set_compute_pipeline_state(&self.q4k_matvec_pipeline.state);
            enc.set_buffer(0, Some(&scratch.down_bufs[e]), 0);
            enc.set_buffer(1, Some(&scratch.act_buf), act_offset);
            enc.set_buffer(2, Some(&scratch.expert_outs), out_offset);
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

        // ── 7. CPU weighted sum + post-experts norm ──────────────────────
        let all_expert_outputs = read_buffer_f32(&scratch.expert_outs, valid_count * hidden);
        let mut moe_out = vec![0.0f32; hidden];
        for e in 0..valid_count {
            let w = valid_weights[e];
            let out_slice = &all_expert_outputs[e * hidden..(e + 1) * hidden];
            for (acc, &v) in moe_out.iter_mut().zip(out_slice) {
                *acc += v * w;
            }
        }

        if !moe.post_experts_norm.is_empty() {
            let rms = (moe_out.iter().map(|v| v * v).sum::<f32>() / hidden as f32 + eps).sqrt();
            for (v, &w) in moe_out.iter_mut().zip(moe.post_experts_norm) {
                *v = *v / rms * (w + 0.0);
            }
        }
        moe_out
    }
}
