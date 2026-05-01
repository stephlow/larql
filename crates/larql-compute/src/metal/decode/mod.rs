use super::*;

mod diag;
mod encode_ffn;
mod encode_qkv;
pub mod gpu_timing;
mod moe_combine;
pub mod profile;

pub use profile::ProfileTimings;

impl MetalBackend {
    /// Create a KV cache for decode mode with uniform per-layer dims.
    pub fn create_kv_cache(
        &self,
        num_layers: usize,
        max_seq: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> ops::kv_cache::KVCache {
        ops::kv_cache::KVCache::new(&self.bufs, num_layers, max_seq, num_kv_heads, head_dim)
    }

    /// Create a KV cache with per-layer shapes for models with asymmetric
    /// attention geometry (Gemma 4 31B sliding=16×256 / global=4×512).
    /// `shapes[i] = (num_kv_heads_i, head_dim_i)` for layer i.
    pub fn create_kv_cache_per_layer(
        &self,
        shapes: &[(usize, usize)],
        max_seq: usize,
    ) -> ops::kv_cache::KVCache {
        ops::kv_cache::KVCache::new_per_layer(&self.bufs, shapes, max_seq)
    }

    /// Decode one token through all layers with KV cache.
    ///
    /// **Single command buffer**, one encoder per layer, no explicit barriers
    /// (Apple Silicon serialises compute dispatches within an encoder).
    ///
    /// Per-layer pipeline (~10 dispatches):
    ///   1. Input norm
    ///   2. Fused QKV projection (Q4_K or Q4_KF)
    ///   3. Batched RoPE (all Q heads), batched RoPE (all K heads)
    ///   4. Batched V-norm (optional, Gemma 4)
    ///   5. KV cache append + KV attend
    ///   6. O projection
    ///   7. Residual + norm (f32 for Q4_K/Q4_KF, +Q8 for Q4_0)
    ///   8. FFN: fused gate+up (Q4_K) or separate gate/up (Q4_KF) + GEGLU + down
    ///   9. Post-FFN residual + optional layer scalar
    ///
    /// Format-aware FFN routing:
    ///   - Q4_KF: llama.cpp-exact kernel (q4kf_proj) with register-cached input
    ///   - Q4_K:  fused gate+up kernel + q4k_matvec (uint4, 8 rows/TG, nr0=2)
    ///   - Q4_0:  legacy Q8-input path
    ///
    /// Decode one token with an optional MoE override function.
    ///
    /// When `moe_fn` is `Some`, it is called instead of `cpu_moe_forward` for
    /// every MoE layer.  Signature: `moe_fn(layer_idx, h_post_attn) -> Vec<f32>`.
    /// The returned vec must have length == `hidden`.  Pass `None` for the
    /// normal local-expert path.
    ///
    /// When `moe_collect_fn` is also `Some` the per-layer pipeline switches to
    /// the split-encoder layout: attention is committed and waited, `moe_fn`
    /// is invoked as a non-blocking *fire* (return value discarded), dense
    /// FFN + post-FFN residual are encoded on a fresh command buffer and
    /// committed without waiting, then `moe_collect_fn(layer)` is called to
    /// retrieve the expert output — letting the remote round trip overlap
    /// with the dense-FFN GPU work.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn decode_token_with_moe_fn(
        &self,
        kv_cache: &mut ops::kv_cache::KVCache,
        layers: &[crate::FullPipelineLayer],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        _num_q_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _rope_base: f32,
        mut moe_fn: Option<&mut dyn FnMut(usize, &[f32]) -> Vec<f32>>,
    ) -> Vec<f32> {
        // Backwards-compat wrapper: forward to the split-aware impl with no
        // collect callback.
        self.decode_token_with_moe_split_fn(
            kv_cache,
            layers,
            x,
            hidden,
            inter,
            q_dim,
            kv_dim,
            _num_q_heads,
            _num_kv_heads,
            _head_dim,
            _rope_base,
            moe_fn
                .as_deref_mut()
                .map(|f| f as &mut dyn FnMut(usize, &[f32]) -> Vec<f32>),
            None,
        )
    }

    /// Split fire / collect variant of `decode_token_with_moe_fn`.  See the
    /// trait method `DecodeBackend::decode_token_with_moe_split` for the
    /// motivating use case (within-layer GPU/MoE overlap).
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn decode_token_with_moe_split_fn(
        &self,
        kv_cache: &mut ops::kv_cache::KVCache,
        layers: &[crate::FullPipelineLayer],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        _num_q_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _rope_base: f32,
        mut moe_fn: Option<&mut dyn FnMut(usize, &[f32]) -> Vec<f32>>,
        mut moe_collect_fn: Option<&mut dyn FnMut(usize) -> Vec<f32>>,
    ) -> Vec<f32> {
        let _gpu_time_token_start = std::time::Instant::now();
        let mut gpu_time = gpu_timing::TokenGpuTime::default();

        let num_layers = layers.len();
        let hidden_val = hidden as u32;
        // Inner dim of down_proj is the intermediate size. Q4_K/Q6_K
        // super-blocks hold 256 values, so when `inter % 256 != 0` each stored
        // row must be padded up to `inter_padded` for the matvec to read the
        // right bytes (see `pad_rows_to_256` in the extractor). The
        // activation buffer fed into down_proj gets allocated at this size
        // and zero-initialised so the padding columns contribute nothing.
        // (The per-stage-as-u32 forms now live inside `encode_ffn`.)
        let block = larql_models::quant::ggml::Q4_K_BLOCK_ELEMS;
        let inter_padded = inter.div_ceil(block) * block;

        // Residual dump (env-gated) for HF-reference diffs. Active only when
        // `LARQL_DUMP_RESIDUALS=<path>` is set.
        let mut residual_dump = diag::ResidualDump::from_env();

        // Input RMS debug (first 3 calls, env-gated).
        static CALL_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let call_n = CALL_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        diag::log_decode_entry(call_n, x, hidden, inter, layers);

        // Scratch buffers are reused across all layers within the encoder.
        // When attention geometry varies layer to layer (Gemma 4 sliding=8192
        // vs global=16384 q_dim) we must size each scratch to the MAX across
        // layers; the outer scalar `q_dim` / `kv_dim` only reflect the first
        // layer's shape. Taking the per-layer max means a global layer's
        // 16384-wide Q output won't overflow a buffer sized for 8192.
        let max_q_dim = layers
            .iter()
            .map(|l| l.num_q_heads * l.head_dim)
            .max()
            .unwrap_or(q_dim);
        let max_kv_dim = layers
            .iter()
            .map(|l| l.num_kv_heads * l.head_dim)
            .max()
            .unwrap_or(kv_dim);

        // Pre-cache weight buffers
        let wq_bufs: Vec<_> = layers
            .iter()
            .map(|l| self.bufs.get_bytes(l.wq.data))
            .collect();
        let wk_bufs: Vec<_> = layers
            .iter()
            .map(|l| self.bufs.get_bytes(l.wk.data))
            .collect();
        let wv_bufs: Vec<_> = layers
            .iter()
            .map(|l| self.bufs.get_bytes(l.wv.data))
            .collect();
        let wo_bufs: Vec<_> = layers
            .iter()
            .map(|l| self.bufs.get_bytes(l.wo.data))
            .collect();
        // Stable across decode calls → cache by slice identity. Skips ~136
        // per-token Metal-buffer allocations for scales/norms on 34-layer
        // Gemma 3. `get_f32` hits the cache from the second decode onward.
        let wq_scale_bufs: Vec<_> = layers
            .iter()
            .map(|l| self.bufs.get_f32(l.wq.scales.unwrap_or(&[])))
            .collect();
        let wk_scale_bufs: Vec<_> = layers
            .iter()
            .map(|l| self.bufs.get_f32(l.wk.scales.unwrap_or(&[])))
            .collect();
        let wv_scale_bufs: Vec<_> = layers
            .iter()
            .map(|l| self.bufs.get_f32(l.wv.scales.unwrap_or(&[])))
            .collect();
        let wo_scale_bufs: Vec<_> = layers
            .iter()
            .map(|l| self.bufs.get_f32(l.wo.scales.unwrap_or(&[])))
            .collect();
        let gate_bufs: Vec<_> = layers
            .iter()
            .map(|l| self.bufs.get_bytes(l.gate.data))
            .collect();
        let up_bufs: Vec<_> = layers
            .iter()
            .map(|l| self.bufs.get_bytes(l.up.data))
            .collect();
        let down_bufs: Vec<_> = layers
            .iter()
            .map(|l| self.bufs.get_bytes(l.down.data))
            .collect();
        let input_norm_bufs: Vec<_> = layers
            .iter()
            .map(|l| self.bufs.get_f32(l.input_norm))
            .collect();
        let post_attn_norm_bufs: Vec<_> = layers
            .iter()
            .map(|l| self.bufs.get_f32(l.post_attn_norm))
            .collect();

        // Two h buffers for ping-pong: even layers write to h_a, odd to h_b.
        let h_init = self.bufs.transient_from_f32(x);
        let h_a = self.bufs.output((hidden * 4) as u64);
        let h_b = self.bufs.output((hidden * 4) as u64);
        let mut h_buf = &h_init;

        // Pre-allocate scratch buffers reused across layers.
        // GPU processes layers sequentially within one cmd buffer, so
        // these buffers are never read and written simultaneously.
        let q_out = self.bufs.output((max_q_dim * 4) as u64);
        let k_out = self.bufs.output((max_kv_dim * 4) as u64);
        let v_out = self.bufs.output((max_kv_dim * 4) as u64);
        let norm_f32_buf = self.bufs.output((hidden * 4) as u64);
        let attn_out_buf = self.bufs.output((max_q_dim * 4) as u64);
        let o_out_buf = self.bufs.output((hidden * 4) as u64);
        let h_post_attn = self.bufs.output((hidden * 4) as u64);
        let ffn_norm_out = self.bufs.output((hidden * 4) as u64);
        let ffn_q8 = self.bufs.output(hidden as u64);
        let ffn_q8s = self.bufs.output((hidden / 32 * 4) as u64);
        let up_out = self.bufs.output((inter * 4) as u64);
        // Sized to `inter_padded` and zero-initialised so down_proj's matvec
        // reads zero for any trailing padding columns. Only the first
        // `inter` floats are written by GEGLU; the rest stay zero across all
        // layers because nothing writes past `inter`.
        let act_buf = self.bufs.output((inter_padded * 4) as u64);
        {
            let ptr = act_buf.contents() as *mut f32;
            unsafe {
                std::ptr::write_bytes(ptr, 0, inter_padded);
            }
        }
        let down_out = self.bufs.output((hidden * 4) as u64);
        let gate_out_scratch = self.bufs.output((inter * 4) as u64);
        // new_h is ping-ponged via h_a/h_b above
        let normed_scratch = self.bufs.output((hidden * 4) as u64);
        let o_q8_scratch = self.bufs.output(max_q_dim as u64);
        let o_q8s_scratch = self.bufs.output((max_q_dim / 32 * 4) as u64);
        let scaled_scratch = self.bufs.output((hidden * 4) as u64);

        // Owned cmd+enc so they can be re-created mid-loop for MoE CPU interleave.
        let has_moe = layers.iter().any(|l| l.moe.is_some());
        // Split mode: when a fire+collect callback pair is present, defer
        // FFN encoding for MoE layers until *after* the remote MoE call has
        // been fired, so dense FFN runs on the GPU in parallel with the
        // network round trip.  Falls back to single-encoder per layer when
        // `moe_collect_fn` is `None` (existing local-MoE / unary HTTP path).
        let split_mode = moe_fn.is_some() && moe_collect_fn.is_some();
        let mut cmd = self.queue.new_command_buffer().to_owned();
        let mut enc = cmd.new_compute_command_encoder().to_owned();
        let mut encoder_ended = false;

        // Diagnostic: run only up to (and including) the specified layer,
        // then dump intermediates and exit. Pinpoints which sub-stage in
        // which layer first produces NaN on real-vindex decode.
        let diag_stop_layer: Option<usize> = std::env::var("LARQL_DECODE_DIAG_LAYER")
            .ok()
            .and_then(|v| v.parse::<usize>().ok());

        for l in 0..num_layers {
            let layer = &layers[l];

            // Snapshot the layer input for HF-reference diff. Must be taken
            // before any compute since `h_buf` = layer-N input at this point
            // (it's the previous layer's `new_h`, or the embedding for L0).
            // GPU buffers are committed + waited at the end of each MoE
            // iteration so the read returns consistent data.
            let layer_in_snapshot: Option<Vec<f32>> = if residual_dump.is_enabled() {
                Some(super::buffers::read_buffer_f32(h_buf, hidden))
            } else {
                None
            };
            let dump_l0_dir = if l == 0 {
                std::env::var("LARQL_DUMP_L0").ok()
            } else {
                None
            };

            let norm_offset = layer.norm_offset;
            let eps = layer.eps;
            let scale = layer.attn_scale;
            let layer_head_dim = layer.head_dim;
            let layer_num_q_heads = layer.num_q_heads;
            let layer_num_kv_heads = layer.num_kv_heads;
            let layer_rope_base = layer.rope_base;
            let layer_rotary_dim = if layer.rotary_dim > 0 {
                layer.rotary_dim
            } else {
                layer_head_dim
            };
            let uses_q4k = layer.wq.format.is_q4k_family();
            let layer_q_dim = layer_num_q_heads * layer_head_dim;
            let layer_kv_dim = layer_num_kv_heads * layer_head_dim;
            let window_size = layer.sliding_window as u32;

            // ── Step 1: Input norm + Q/K/V projection ──
            // Format-aware: Q4_K family routes through fused QKV
            // shaders (uniform / mixed Q4K+Q6K-V / per-projection
            // fallback); Q4_0 routes through fused norm+Q8 then
            // Q8 QKV. Implementation lives in `encode_qkv.rs`.
            self.encode_input_norm_and_qkv(
                &enc,
                layer,
                encode_qkv::QkvBufs {
                    h_in: h_buf,
                    input_norm: &input_norm_bufs[l],
                    input_norm_bias: layer.input_norm_bias,
                    wq: &wq_bufs[l],
                    wk: &wk_bufs[l],
                    wv: &wv_bufs[l],
                    wq_scales: &wq_scale_bufs[l],
                    wk_scales: &wk_scale_bufs[l],
                    wv_scales: &wv_scale_bufs[l],
                    norm_out: &norm_f32_buf,
                    q_out: &q_out,
                    k_out: &k_out,
                    v_out: &v_out,
                    ffn_q8: &ffn_q8,
                    ffn_q8s: &ffn_q8s,
                },
                encode_qkv::QkvDims {
                    hidden,
                    layer_q_dim,
                    layer_kv_dim,
                    eps,
                    norm_offset,
                },
                uses_q4k,
            );

            // ── Step 1.5: QK-norm on Q and K (Gemma 3 / Gemma 4) ──
            //
            // Per-head RMS-norm with learned weight, applied to the raw
            // projection output before RoPE. Without this the Q/K vectors
            // on Gemma 3/4 are unscaled — attention dot products overflow
            // and softmax collapses to NaN by layer 0.
            //
            // Formula (matches CPU `rms_norm_heads_eps`):
            //   out[h, d] = (x[h, d] / sqrt(mean(x_head²) + eps))
            //             * (qk_norm_offset + weight[d])
            //
            // The qk_norm_offset is 0.0 on Gemma 4 and 1.0 on Gemma 2/3.
            // Passed as `offset` to the shader so `offset + weight[d]` does
            // the right thing for both families.
            // ── Steps 1.5 + 2: QK-norm + RoPE ──
            //
            // When `LARQL_FUSED_QK_NORM_ROPE=1` AND the layer has
            // QK-norm weights (Gemma 3/4), use the single fused
            // `qk_norm_rope_fused` kernel — saves 1 dispatch per
            // layer × 34 = ~34/tok. Falls back to the consecutive
            // `qk_norm_qk` + `rope_at_pos_batched_qk` chain for
            // archs without QK-norm or when the env flag is unset.
            // qk_norm+RoPE fusion — proven win 2026-05-01 (~0.10 ms/tok GPU).
            // Default-on; opt out via `LARQL_FUSED_QK_NORM_ROPE=0`.
            let use_fused_qkn_rope = !matches!(
                std::env::var("LARQL_FUSED_QK_NORM_ROPE").as_deref(),
                Ok("0") | Ok("false") | Ok("off") | Ok("no")
            );
            let pos = kv_cache.layers[l].current_len as u32;
            if use_fused_qkn_rope && layer.q_norm_weight.is_some() && layer.k_norm_weight.is_some()
            {
                let q_w = layer.q_norm_weight.unwrap();
                let k_w = layer.k_norm_weight.unwrap();
                let hd_val = layer_head_dim as u32;
                let nq_val = layer_num_q_heads as u32;
                let qk_off = layer.qk_norm_offset;
                let eps = layer.eps;
                let rdim = layer_rotary_dim as u32;
                let mut tg_w: usize = 1;
                while tg_w < layer_head_dim && tg_w < 512 {
                    tg_w <<= 1;
                }
                let q_w_buf = self.bufs.get_f32(q_w);
                let k_w_buf = self.bufs.get_f32(k_w);
                let total_heads = (layer_num_q_heads + layer_num_kv_heads) as u64;
                enc.set_compute_pipeline_state(&self.qk_norm_rope_fused_pipeline);
                enc.set_buffer(0, Some(&q_out), 0);
                enc.set_buffer(1, Some(&k_out), 0);
                enc.set_buffer(2, Some(&q_w_buf), 0);
                enc.set_buffer(3, Some(&k_w_buf), 0);
                enc.set_bytes(4, 4, &hd_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(5, 4, &nq_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(6, 4, &eps as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(7, 4, &qk_off as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(
                    8,
                    4,
                    &layer_rope_base as *const f32 as *const std::ffi::c_void,
                );
                enc.set_bytes(9, 4, &pos as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(10, 4, &rdim as *const u32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(total_heads, 1, 1),
                    MTLSize::new(tg_w as u64, 1, 1),
                );
            } else {
                if let (Some(q_w), Some(k_w)) = (layer.q_norm_weight, layer.k_norm_weight) {
                    let hd_val = layer_head_dim as u32;
                    let nq_val = layer_num_q_heads as u32;
                    let qk_off = layer.qk_norm_offset;
                    let eps = layer.eps;
                    let mut tg_w: usize = 1;
                    while tg_w < layer_head_dim && tg_w < 512 {
                        tg_w <<= 1;
                    }

                    // Fused Q+K norm: one dispatch covers all q_heads+kv_heads.
                    // Saves 1 dispatch per layer × 34 = 34 dispatches/token.
                    let q_w_buf = self.bufs.get_f32(q_w);
                    let k_w_buf = self.bufs.get_f32(k_w);
                    let total_heads = (layer_num_q_heads + layer_num_kv_heads) as u64;
                    enc.set_compute_pipeline_state(&self.qk_norm_qk_pipeline);
                    enc.set_buffer(0, Some(&q_out), 0);
                    enc.set_buffer(1, Some(&k_out), 0);
                    enc.set_buffer(2, Some(&q_w_buf), 0);
                    enc.set_buffer(3, Some(&k_w_buf), 0);
                    enc.set_bytes(4, 4, &hd_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(5, 4, &nq_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(6, 4, &eps as *const f32 as *const std::ffi::c_void);
                    enc.set_bytes(7, 4, &qk_off as *const f32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(
                        MTLSize::new(total_heads, 1, 1),
                        MTLSize::new(tg_w as u64, 1, 1),
                    );
                }

                // ── Step 2: RoPE on Q and K heads (batched — one dispatch each) ──
                let hd = layer_head_dim as u32;
                let rdim = layer_rotary_dim as u32;
                let rope_pairs = (layer_rotary_dim / 2) as u64;
                let num_q = layer_num_q_heads as u32;
                let total_qk_heads = (layer_num_q_heads + layer_num_kv_heads) as u64;
                enc.set_compute_pipeline_state(&self.rope_at_pos_batched_qk_pipeline);
                enc.set_buffer(0, Some(&q_out), 0);
                enc.set_buffer(1, Some(&k_out), 0);
                enc.set_bytes(2, 4, &hd as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(
                    3,
                    4,
                    &layer_rope_base as *const f32 as *const std::ffi::c_void,
                );
                enc.set_bytes(4, 4, &pos as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(5, 4, &rdim as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(6, 4, &num_q as *const u32 as *const std::ffi::c_void);
                enc.dispatch_threads(
                    MTLSize::new(rope_pairs, total_qk_heads, 1),
                    MTLSize::new(rope_pairs.min(256), 1, 1),
                );
            }

            // ── Step 3: V-norm batched (optional, Gemma 4) ──
            // Cooperative reduction: one threadgroup per KV head; threads
            // within a TG share the sum-of-squares via threadgroup memory
            // and a barrier (see `shaders/v_norm.rs`). Round tg width up
            // to a power of two ≤ 512 for the tree reduction.
            if layer.has_v_norm {
                let hd_val = layer_head_dim as u32;
                let num_kv = layer_num_kv_heads as u32;
                let mut tg_w: u64 = 1;
                while tg_w < layer_head_dim as u64 && tg_w < 512 {
                    tg_w <<= 1;
                }
                enc.set_compute_pipeline_state(&self.v_norm_batched_pipeline);
                enc.set_buffer(0, Some(&v_out), 0);
                enc.set_buffer(1, Some(&v_out), 0);
                enc.set_bytes(2, 4, &hd_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(3, 4, &eps as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(4, 4, &num_kv as *const u32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(layer_num_kv_heads as u64, 1, 1),
                    MTLSize::new(tg_w, 1, 1),
                );
            }

            // No explicit barriers — Apple Silicon executes compute dispatches
            // within a single encoder in submission order. Verified by tests.

            let attn_out = &attn_out_buf;
            // Fused KV-append + KV-attention. Each Q-head TG writes its
            // kv_head's new K/V row at position pos = current_len, then
            // proceeds with attention on T = current_len + 1. Eliminates
            // the separate kv_cache_append dispatch
            // (~1 dispatch/layer × 34 ≈ 0.24 ms/tok). Default-on; opt
            // out via `LARQL_FUSED_KV_APPEND_ATTEND=0` for diagnostics.
            let use_fused_kv_aa = !matches!(
                std::env::var("LARQL_FUSED_KV_APPEND_ATTEND").as_deref(),
                Ok("0") | Ok("false") | Ok("off") | Ok("no")
            );
            if use_fused_kv_aa {
                let cache = &kv_cache.layers[l];
                let t_val = (cache.current_len + 1) as u32;
                let hd = cache.head_dim as u32;
                let num_q_val = layer_num_q_heads as u32;
                let num_kv = cache.num_kv_heads as u32;
                enc.set_compute_pipeline_state(&self.kv_append_attend_fused_pipeline);
                enc.set_buffer(0, Some(&q_out), 0);
                enc.set_buffer(1, Some(&cache.k_cache), 0);
                enc.set_buffer(2, Some(&cache.v_cache), 0);
                enc.set_buffer(3, Some(attn_out), 0);
                enc.set_bytes(4, 4, &t_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(5, 4, &hd as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(6, 4, &num_q_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(7, 4, &num_kv as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(8, 4, &scale as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(9, 4, &window_size as *const u32 as *const std::ffi::c_void);
                enc.set_buffer(10, Some(&k_out), 0);
                enc.set_buffer(11, Some(&v_out), 0);
                enc.dispatch_thread_groups(
                    MTLSize::new(layer_num_q_heads as u64, 1, 1),
                    MTLSize::new(256.min(layer_head_dim as u64), 1, 1),
                );
            } else {
                ops::kv_cache::encode_kv_append(
                    &enc,
                    &kv_cache.layers[l],
                    &self.kv_append_pipeline,
                    &k_out,
                    &v_out,
                );
                ops::kv_cache::encode_kv_attend(
                    &enc,
                    &kv_cache.layers[l],
                    &self.kv_attend_pipeline,
                    &q_out,
                    attn_out,
                    layer_num_q_heads,
                    scale,
                    window_size,
                );
            }
            kv_cache.layers[l].current_len += 1;

            // Scratch buffers pre-allocated above — reused each layer.
            let new_h = if l % 2 == 0 { &h_a } else { &h_b };
            if uses_q4k {
                // Q4_K / Q4_KF / Q6_K O-projection via the stage helper.
                use crate::metal::stages::quant_matvec::Pipelines;
                let pipes = Pipelines {
                    q4kf_proj: Some(&self.q4kf_proj_pipeline.state),
                    q4k_matvec_fallback: &self.q4k_proj_pipeline,
                    q6k_matvec: &self.q6k_matvec_pipeline,
                    q4_matvec: &self.q4.matvec,
                    q4k_matmul: None,
                };
                crate::metal::stages::o_proj::encode(
                    &enc,
                    &pipes,
                    &self.q8_quant_pipeline,
                    layer.wo.format,
                    &wo_bufs[l],
                    attn_out,
                    0,
                    &o_q8_scratch,
                    0,
                    &o_q8s_scratch,
                    0,
                    &o_out_buf,
                    0,
                    layer_q_dim,
                    hidden,
                );
            } else {
                // Q8 legacy path: decode-specific `q8_matvec` shader (not in
                // stages::quant_matvec which uses `q4_matvec` for Q4_0/Q8_0
                // with a different buffer layout). Inline.
                let o_q8 = &o_q8_scratch;
                let o_q8s = &o_q8s_scratch;
                let dim_val = layer_q_dim as u32;
                let blocks = (layer_q_dim / 32) as u32;
                enc.set_compute_pipeline_state(&self.q8_quant_pipeline);
                enc.set_buffer(0, Some(attn_out), 0);
                enc.set_buffer(1, Some(o_q8), 0);
                enc.set_buffer(2, Some(o_q8s), 0);
                enc.set_bytes(3, 4, &dim_val as *const u32 as *const std::ffi::c_void);
                enc.dispatch_threads(
                    MTLSize::new(blocks as u64, 1, 1),
                    MTLSize::new(256.min(blocks as u64), 1, 1),
                );

                let o_rows = hidden as u32;
                let o_k = layer_q_dim as u32;
                enc.set_compute_pipeline_state(&self.q8_matvec_pipeline.state);
                enc.set_buffer(0, Some(&wo_bufs[l]), 0);
                enc.set_buffer(1, Some(o_q8), 0);
                enc.set_buffer(2, Some(&wo_scale_bufs[l]), 0);
                enc.set_buffer(3, Some(o_q8s), 0);
                enc.set_buffer(4, Some(&o_out_buf), 0);
                enc.set_bytes(5, 4, &o_rows as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(6, 4, &o_k as *const u32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new((hidden as u64).div_ceil(8), 1, 1),
                    MTLSize::new(256, 1, 1),
                );
            }

            // ── Step 5: Residual + norm (format-aware: Q4_K skips Q8 quantize) ──
            let ffn_uses_q4k = layer.gate.format.is_q4k_family();
            // ffn_norm_out pre-allocated above

            let has_post_norms = layer.has_post_norms;
            if has_post_norms {
                let normed_o = &normed_scratch;
                let pre_ffn_buf = if let Some(pfn) = layer.pre_ffn_norm {
                    self.bufs.get_f32(pfn)
                } else {
                    post_attn_norm_bufs[l].clone()
                };
                // Triple-fused post_attn_norm+residual+ffn_norm+store kernel —
                // proven win 2026-05-01. Default-on; opt out via
                // `LARQL_FUSED_POST_ATTN_NORM=0` if needed for diagnostics.
                let use_fused_post_attn = !matches!(
                    std::env::var("LARQL_FUSED_POST_ATTN_NORM").as_deref(),
                    Ok("0") | Ok("false") | Ok("off") | Ok("no")
                );
                if use_fused_post_attn && ffn_uses_q4k {
                    // Triple-fused: post_attn_norm + residual_norm + h_post_attn
                    // store in ONE dispatch. Replaces (rms_norm +
                    // residual_norm_store) two-dispatch pair on the
                    // has_post_norms path → saves 1 dispatch/layer × 34
                    // = ~34/tok ≈ 0.24 ms/tok. ROADMAP G-3 third fusion.
                    enc.set_compute_pipeline_state(&self.post_attn_residual_norm_store_pipeline);
                    enc.set_buffer(0, Some(h_buf), 0);
                    enc.set_buffer(1, Some(&o_out_buf), 0);
                    enc.set_buffer(2, Some(&post_attn_norm_bufs[l]), 0);
                    enc.set_buffer(3, Some(&pre_ffn_buf), 0);
                    enc.set_buffer(4, Some(&ffn_norm_out), 0);
                    enc.set_buffer(5, Some(&h_post_attn), 0);
                    enc.set_bytes(6, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(7, 4, &eps as *const f32 as *const std::ffi::c_void);
                    enc.set_bytes(8, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(
                        MTLSize::new(1, 1, 1),
                        MTLSize::new(256.min(hidden as u64), 1, 1),
                    );
                    // Skip the unfused chain below — both ffn_norm_out
                    // and h_post_attn are already written.
                } else {
                    // Unfused chain: rms_norm + residual_norm_store
                    // (or _q8). Still 1 dispatch better than the
                    // pre-2026-05-01 3-dispatch chain.
                    {
                        use crate::metal::ops::full_pipeline::encode_rms_norm;
                        encode_rms_norm(
                            &enc,
                            &self.rms_norm_pipeline,
                            &o_out_buf,
                            &post_attn_norm_bufs[l],
                            normed_o,
                            hidden,
                            eps,
                            norm_offset,
                        );
                    }
                    if ffn_uses_q4k {
                        // Q4_K path: residual+norm in ONE dispatch via
                        // `residual_norm_store` — writes both `ffn_norm_out`
                        // (RMS-normed sum, scaled by `pre_ffn_buf`) AND
                        // `h_post_attn` (raw sum). Replaces the
                        // residual_norm + residual_add two-dispatch pair
                        // that was here pre-2026-05-01. Saves 1 dispatch
                        // per layer × 34 = ~34/tok ≈ 0.24 ms/tok end-to-end
                        // (same fusion mechanic as `qk_norm_rope_fused`,
                        // ref. ROADMAP G-3 entry).
                        //
                        // The math is identical to the unfused pair:
                        //   ffn_norm_out = RMS_norm(h + normed_o) * (pre_ffn_buf + offset)
                        //   h_post_attn  = h + normed_o
                        // (`residual_norm_store` shader in
                        // `shaders/fused_ops.rs` was already written for the
                        // !has_post_norms branch; just routing the
                        // has_post_norms branch through it now.)
                        enc.set_compute_pipeline_state(&self.residual_norm_store_pipeline);
                        enc.set_buffer(0, Some(h_buf), 0);
                        enc.set_buffer(1, Some(normed_o), 0);
                        enc.set_buffer(2, Some(&pre_ffn_buf), 0);
                        enc.set_buffer(3, Some(&ffn_norm_out), 0);
                        enc.set_buffer(4, Some(&h_post_attn), 0);
                        enc.set_bytes(5, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(6, 4, &eps as *const f32 as *const std::ffi::c_void);
                        enc.set_bytes(7, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                        enc.dispatch_thread_groups(
                            MTLSize::new(1, 1, 1),
                            MTLSize::new(256.min(hidden as u64), 1, 1),
                        );
                    } else {
                        enc.set_compute_pipeline_state(&self.residual_norm_q8_pipeline);
                        enc.set_buffer(0, Some(h_buf), 0);
                        enc.set_buffer(1, Some(normed_o), 0);
                        enc.set_buffer(2, Some(&pre_ffn_buf), 0);
                        enc.set_buffer(3, Some(&ffn_q8), 0);
                        enc.set_buffer(4, Some(&ffn_q8s), 0);
                        enc.set_buffer(5, Some(&h_post_attn), 0);
                        enc.set_bytes(6, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(7, 4, &eps as *const f32 as *const std::ffi::c_void);
                        enc.set_bytes(8, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                        enc.dispatch_thread_groups(
                            MTLSize::new(1, 1, 1),
                            MTLSize::new(256.min(hidden as u64), 1, 1),
                        );
                    }
                } // close `else { unfused chain }`
            } else if ffn_uses_q4k {
                // Fused: residual_norm_store writes BOTH ffn_norm_out (normed,
                // for FFN input) AND h_post_attn (raw sum, for post-FFN add).
                // Replaces residual_norm + residual_add (saves 34 dispatches/token).
                enc.set_compute_pipeline_state(&self.residual_norm_store_pipeline);
                enc.set_buffer(0, Some(h_buf), 0);
                enc.set_buffer(1, Some(&o_out_buf), 0);
                enc.set_buffer(2, Some(&post_attn_norm_bufs[l]), 0);
                enc.set_buffer(3, Some(&ffn_norm_out), 0);
                enc.set_buffer(4, Some(&h_post_attn), 0);
                enc.set_bytes(5, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(6, 4, &eps as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(7, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(1, 1, 1),
                    MTLSize::new(256.min(hidden as u64), 1, 1),
                );
            } else {
                enc.set_compute_pipeline_state(&self.residual_norm_q8_pipeline);
                enc.set_buffer(0, Some(h_buf), 0);
                enc.set_buffer(1, Some(&o_out_buf), 0);
                enc.set_buffer(2, Some(&post_attn_norm_bufs[l]), 0);
                enc.set_buffer(3, Some(&ffn_q8), 0);
                enc.set_buffer(4, Some(&ffn_q8s), 0);
                enc.set_buffer(5, Some(&h_post_attn), 0);
                enc.set_bytes(6, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(7, 4, &eps as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(8, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(1, 1, 1),
                    MTLSize::new(256.min(hidden as u64), 1, 1),
                );
            }

            // ── Steps 6-7: FFN + post-FFN residual ──
            //
            // Skip when in split mode AND this layer has MoE — they will be
            // re-encoded on a fresh command buffer inside the MoE block so
            // they can run in parallel with the remote MoE round trip.  For
            // non-MoE layers (or non-split mode) we encode them inline as
            // before.
            let defer_ffn_for_split = split_mode && layer.moe.is_some();

            // Stage-timing boundary: when LARQL_DECODE_STAGE_TIMING=1 (and we
            // are NOT already splitting for MoE), close the encoder here so
            // attention CB time can be recorded separately from FFN CB time.
            // Adds ~1 commit/wait per layer (~0.5ms × 30 = ~15ms inflation
            // on Gemma 4) — measurement-only mode, off by default.
            let stage_timing_split =
                !defer_ffn_for_split && std::env::var("LARQL_DECODE_STAGE_TIMING").is_ok();
            if stage_timing_split {
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                gpu_time.record_stage(&cmd, gpu_timing::DecodeStage::Attention);
                cmd = self.queue.new_command_buffer().to_owned();
                enc = cmd.new_compute_command_encoder().to_owned();
                encoder_ended = false;
            }

            if !defer_ffn_for_split {
                // Step 6: FFN (format-aware Q4_KF / Q4_K / Q4_0)
                self.encode_ffn_step(
                    &enc,
                    layer,
                    encode_ffn::FfnBufs {
                        gate_w: &gate_bufs[l],
                        up_w: &up_bufs[l],
                        down_w: &down_bufs[l],
                        ffn_norm_out: &ffn_norm_out,
                        ffn_q8: &ffn_q8,
                        ffn_q8s: &ffn_q8s,
                        gate_out_scratch: &gate_out_scratch,
                        up_out: &up_out,
                        act_buf: &act_buf,
                        down_out: &down_out,
                    },
                    encode_ffn::FfnDims {
                        hidden,
                        inter,
                        inter_padded,
                    },
                    ffn_uses_q4k,
                );

                // Step 7: Post-FFN residual
                if has_post_norms {
                    if let Some(post_ffn) = layer.post_ffn_norm {
                        let post_ffn_buf = self.bufs.get_f32(post_ffn);
                        // Post-FFN norm+residual_add fusion — proven win
                        // 2026-05-01. Default-on; opt out via
                        // `LARQL_FUSED_POST_FFN_NORM=0` for diagnostics.
                        let use_fused_post_ffn = !matches!(
                            std::env::var("LARQL_FUSED_POST_FFN_NORM").as_deref(),
                            Ok("0") | Ok("false") | Ok("off") | Ok("no")
                        );
                        if use_fused_post_ffn {
                            // Fused: rms_norm(down_out) + residual_add(h_post_attn,
                            // normed_ffn) → new_h. Single dispatch, single TG;
                            // saves 1 dispatch/layer × 34 ≈ 0.24 ms/tok end-to-end.
                            // Math identical to the unfused chain — see
                            // `shaders/post_ffn_norm_residual_add.rs`.
                            enc.set_compute_pipeline_state(
                                &self.post_ffn_norm_residual_add_pipeline,
                            );
                            enc.set_buffer(0, Some(&down_out), 0);
                            enc.set_buffer(1, Some(&h_post_attn), 0);
                            enc.set_buffer(2, Some(&post_ffn_buf), 0);
                            enc.set_buffer(3, Some(new_h), 0);
                            enc.set_bytes(
                                4,
                                4,
                                &hidden_val as *const u32 as *const std::ffi::c_void,
                            );
                            enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
                            enc.set_bytes(
                                6,
                                4,
                                &norm_offset as *const f32 as *const std::ffi::c_void,
                            );
                            enc.dispatch_thread_groups(
                                MTLSize::new(1, 1, 1),
                                MTLSize::new(256.min(hidden as u64), 1, 1),
                            );
                        } else {
                            let normed_ffn = &normed_scratch;
                            use crate::metal::ops::full_pipeline::encode_rms_norm;
                            encode_rms_norm(
                                &enc,
                                &self.rms_norm_pipeline,
                                &down_out,
                                &post_ffn_buf,
                                normed_ffn,
                                hidden,
                                eps,
                                norm_offset,
                            );
                            use crate::metal::ops::full_pipeline::encode_residual_add;
                            encode_residual_add(
                                &enc,
                                &self.residual_add_pipeline,
                                &h_post_attn,
                                normed_ffn,
                                new_h,
                                hidden,
                            );
                        }
                    } else {
                        use crate::metal::ops::full_pipeline::encode_residual_add;
                        encode_residual_add(
                            &enc,
                            &self.residual_add_pipeline,
                            &h_post_attn,
                            &down_out,
                            new_h,
                            hidden,
                        );
                    }
                } else {
                    use crate::metal::ops::full_pipeline::encode_residual_add;
                    encode_residual_add(
                        &enc,
                        &self.residual_add_pipeline,
                        &h_post_attn,
                        &down_out,
                        new_h,
                        hidden,
                    );
                }
            }

            h_buf = new_h;
            let _ = &scaled_scratch; // keep binding alive; no longer needed

            // CPU MoE interleave for hybrid MoE models (e.g. Gemma 4 26B A4B).
            // After the GPU dense-FFN pass, flush the encoder, run the expert block
            // on CPU (direct shared-memory access), then restart for the next layer.
            // layer_scalar is applied AFTER MoE so it scales the combined output
            // (dense + MoE). Applying it before would leave the MoE contribution unscaled.
            if has_moe {
                if let Some(ref moe) = layer.moe {
                    enc.end_encoding();
                    cmd.commit();
                    cmd.wait_until_completed();
                    // In split mode the cb we just waited contains ONLY attention
                    // (steps 1-5).  In non-split mode it normally contains
                    // attention + dense FFN; but when stage_timing_split was
                    // active, attention was already committed at its own
                    // boundary so this cb contains only FFN + post-residual.
                    let attn_cb_stage = if defer_ffn_for_split {
                        gpu_timing::DecodeStage::Attention
                    } else if stage_timing_split {
                        gpu_timing::DecodeStage::DenseFfn
                    } else {
                        gpu_timing::DecodeStage::Other
                    };
                    gpu_time.record_stage(&cmd, attn_cb_stage);
                    encoder_ended = true;

                    // MoE and dense FFN run on the SAME input (h_post_attn, the
                    // post-attention residual). Dense FFN output is already in new_h.
                    // Read MoE input from h_post_attn, accumulate MoE output into new_h.
                    let attn_ptr = h_post_attn.contents() as *const f32;
                    let attn_slice = unsafe { std::slice::from_raw_parts(attn_ptr, hidden) };
                    let moe_out = if defer_ffn_for_split {
                        // Split path: fire MoE NOW (h_post_attn is ready), then
                        // encode dense FFN + post-FFN residual on a fresh cb so
                        // GPU runs them while the network round trip is in
                        // flight, then collect.
                        let fire = moe_fn.as_deref_mut().expect("split_mode implies moe_fn");
                        fire(l, attn_slice);

                        // Fresh command buffer for the dense FFN pass.
                        cmd = self.queue.new_command_buffer().to_owned();
                        let ffn_enc = cmd.new_compute_command_encoder();

                        // Step 6: FFN encode on the new cb.
                        self.encode_ffn_step(
                            ffn_enc,
                            layer,
                            encode_ffn::FfnBufs {
                                gate_w: &gate_bufs[l],
                                up_w: &up_bufs[l],
                                down_w: &down_bufs[l],
                                ffn_norm_out: &ffn_norm_out,
                                ffn_q8: &ffn_q8,
                                ffn_q8s: &ffn_q8s,
                                gate_out_scratch: &gate_out_scratch,
                                up_out: &up_out,
                                act_buf: &act_buf,
                                down_out: &down_out,
                            },
                            encode_ffn::FfnDims {
                                hidden,
                                inter,
                                inter_padded,
                            },
                            ffn_uses_q4k,
                        );

                        // Step 7: Post-FFN residual on the new cb.
                        if has_post_norms {
                            if let Some(post_ffn) = layer.post_ffn_norm {
                                let post_ffn_buf = self.bufs.get_f32(post_ffn);
                                let normed_ffn = &normed_scratch;
                                use crate::metal::ops::full_pipeline::encode_rms_norm;
                                encode_rms_norm(
                                    ffn_enc,
                                    &self.rms_norm_pipeline,
                                    &down_out,
                                    &post_ffn_buf,
                                    normed_ffn,
                                    hidden,
                                    eps,
                                    norm_offset,
                                );
                                use crate::metal::ops::full_pipeline::encode_residual_add;
                                encode_residual_add(
                                    ffn_enc,
                                    &self.residual_add_pipeline,
                                    &h_post_attn,
                                    normed_ffn,
                                    new_h,
                                    hidden,
                                );
                            } else {
                                use crate::metal::ops::full_pipeline::encode_residual_add;
                                encode_residual_add(
                                    ffn_enc,
                                    &self.residual_add_pipeline,
                                    &h_post_attn,
                                    &down_out,
                                    new_h,
                                    hidden,
                                );
                            }
                        } else {
                            use crate::metal::ops::full_pipeline::encode_residual_add;
                            encode_residual_add(
                                ffn_enc,
                                &self.residual_add_pipeline,
                                &h_post_attn,
                                &down_out,
                                new_h,
                                hidden,
                            );
                        }
                        ffn_enc.end_encoding();
                        cmd.commit(); // GPU runs FFN async — no wait yet

                        // Collect MoE — blocks on the network round trip.
                        // GPU is doing dense FFN in parallel during this wait.
                        let collect = moe_collect_fn
                            .as_deref_mut()
                            .expect("split_mode implies moe_collect_fn");
                        let result = collect(l);
                        // Wait for the FFN cb (likely already done if MoE was
                        // the longer leg).
                        cmd.wait_until_completed();
                        gpu_time.record_stage(&cmd, gpu_timing::DecodeStage::DenseFfn);
                        result
                    } else if let Some(ref mut f) = moe_fn {
                        f(l, attn_slice)
                    } else {
                        crate::cpu::ops::moe::cpu_moe_forward(
                            attn_slice,
                            moe,
                            layer.norm_offset,
                            layer.eps,
                        )
                    };
                    // Accumulate the MoE contribution into the dense output
                    // buffer: new_h = h_post_attn + _1(dense) + moe_out.
                    let h_ptr = new_h.contents() as *mut f32;
                    unsafe {
                        for (i, v) in moe_out.iter().enumerate() {
                            *h_ptr.add(i) += v;
                        }
                    }

                    // L0-only Gemma-4-MoE intermediate dump for HF-Python
                    // diffs. Helper lives in `diag.rs`. Activated by
                    // `LARQL_DUMP_L0=<dir>`.
                    if l == 0 {
                        if let Some(ref dir) = dump_l0_dir {
                            diag::dump_l0_moe_intermediates(
                                dir,
                                &h_post_attn,
                                &ffn_norm_out,
                                &gate_out_scratch,
                                &up_out,
                                &act_buf,
                                &down_out,
                                new_h,
                                &moe_out,
                                hidden,
                                inter,
                            );
                        }
                    }

                    // Apply the architecture-driven outer combine (outer RMS
                    // norm for Gemma 4 hybrid MoE, or layer_scalar-only for
                    // legacy MoE). See `moe_combine.rs` for the full HF map.
                    moe_combine::apply_outer_combine(layer, new_h, &h_post_attn, hidden);

                    // Optional residual capture for HF-reference diffs.
                    // `layer_in_snapshot` was captured at the top of this
                    // iteration; the command buffer has been waited so
                    // both `h_post_attn` and `new_h` are consistent.
                    if let Some(li) = layer_in_snapshot.as_ref() {
                        let ha = super::buffers::read_buffer_f32(&h_post_attn, hidden);
                        let lo = super::buffers::read_buffer_f32(new_h, hidden);
                        residual_dump.record_layer(l, li, &ha, &lo);
                    }

                    if l + 1 < num_layers {
                        cmd = self.queue.new_command_buffer().to_owned();
                        enc = cmd.new_compute_command_encoder().to_owned();
                        encoder_ended = false;
                    }
                }
            } else {
                // ── Step 8: Optional layer scalar (non-MoE layers) ──
                // GPU in-place scale on new_h before it becomes the next layer's input.
                if layer.layer_scalar != 0.0 {
                    crate::metal::stages::layer_scalar::encode(
                        &enc,
                        &self.scale_vector_pipeline,
                        new_h,
                        1,
                        hidden,
                        layer.layer_scalar,
                    );
                }
            }

            // Optional per-layer end-of-layer dump for decode-path
            // diagnostics. Flushes the encoder so `new_h` is readable,
            // writes `decode_layer_{LL}.f32`, then restarts the encoder
            // for the next layer. Paired with Metal prefill's
            // `metal_layer_{LL}_h_out.f32` hook so the two paths can be
            // diffed at the same layer boundaries. Gated on an env var to
            // keep normal decode free of flush overhead.
            //
            // When `LARQL_STAGE_DUMP_LAYER` names the current layer, also
            // dump every per-sub-stage scratch buffer
            // (`decode_layer_{LL}_{stage}.f32`). Names match the Metal
            // prefill side (`metal_layer_NN_{stage}.f32`) so the two
            // dump dirs can be diffed file-by-file. The end-of-layer
            // commit above is what makes these reads consistent — the
            // scratch buffers persist across layers, so without the
            // per-layer flush we'd be reading the *last* layer's value.
            if let Ok(dir) = std::env::var("LARQL_DECODE_DUMP_LAYERS") {
                if !encoder_ended {
                    enc.end_encoding();
                    cmd.commit();
                    cmd.wait_until_completed();
                    encoder_ended = true;
                }
                let hidden_bytes = super::buffers::read_buffer_f32(new_h, hidden);
                let as_bytes: Vec<u8> = hidden_bytes.iter().flat_map(|v| v.to_le_bytes()).collect();
                let path = format!("{dir}/decode_layer_{l:02}.f32");
                if let Err(e) = std::fs::write(&path, &as_bytes) {
                    eprintln!("[decode-dump] failed to write {path}: {e}");
                }

                // Per-stage dump for the layer named by
                // `LARQL_STAGE_DUMP_LAYER` (default 0). Helper lives in
                // `diag.rs`; the bundle of references is the same one
                // the early-exit diag mode uses.
                let stage_layer = std::env::var("LARQL_STAGE_DUMP_LAYER")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);
                if l == stage_layer {
                    let bufs = diag::LayerDiagBufs {
                        norm_f32_buf: &norm_f32_buf,
                        q_out: &q_out,
                        k_out: &k_out,
                        v_out: &v_out,
                        attn_out_buf: &attn_out_buf,
                        o_out_buf: &o_out_buf,
                        h_post_attn: &h_post_attn,
                        ffn_norm_out: &ffn_norm_out,
                        gate_out_scratch: &gate_out_scratch,
                        up_out: &up_out,
                        act_buf: &act_buf,
                        down_out: &down_out,
                        new_h,
                        hidden,
                        inter,
                        layer_q_dim,
                        layer_kv_dim: layer_num_kv_heads * layer_head_dim,
                    };
                    diag::dump_decode_stage_files(&dir, l, &bufs);
                }

                if l + 1 < num_layers {
                    cmd = self.queue.new_command_buffer().to_owned();
                    enc = cmd.new_compute_command_encoder().to_owned();
                    encoder_ended = false;
                }
            }

            // Diagnostic early-exit after layer `l`. Commits what we have,
            // reads the per-sub-stage buffers, and reports NaN counts.
            if diag_stop_layer == Some(l) {
                if !encoder_ended {
                    enc.end_encoding();
                    cmd.commit();
                    cmd.wait_until_completed();
                }
                let bufs = diag::LayerDiagBufs {
                    norm_f32_buf: &norm_f32_buf,
                    q_out: &q_out,
                    k_out: &k_out,
                    v_out: &v_out,
                    attn_out_buf: &attn_out_buf,
                    o_out_buf: &o_out_buf,
                    h_post_attn: &h_post_attn,
                    ffn_norm_out: &ffn_norm_out,
                    gate_out_scratch: &gate_out_scratch,
                    up_out: &up_out,
                    act_buf: &act_buf,
                    down_out: &down_out,
                    new_h,
                    hidden,
                    inter,
                    layer_q_dim,
                    layer_kv_dim: layer_num_kv_heads * layer_head_dim,
                };
                diag::dump_layer_buffers(l, &bufs);
                return super::buffers::read_buffer_f32(new_h, hidden);
            }
        }

        if !encoder_ended {
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            gpu_time.record(&cmd);
        }

        let result = super::buffers::read_buffer_f32(h_buf, hidden);

        // Print GPU vs CPU split when LARQL_GPU_TIMING=1. Wall covers the
        // entire decode_token_with_moe_fn call including buffer reads;
        // gpu is the sum of MTLCommandBuffer.gpuStartTime/gpuEndTime
        // windows. Delta is CPU encoding + readback overhead.
        let wall_ms = _gpu_time_token_start.elapsed().as_secs_f64() * 1000.0;
        gpu_time.print_if_enabled(wall_ms);

        result
    }

    /// Local-expert path — delegates to `decode_token_with_moe_fn` with no hook.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_token(
        &self,
        kv_cache: &mut ops::kv_cache::KVCache,
        layers: &[crate::FullPipelineLayer],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_base: f32,
    ) -> Vec<f32> {
        self.decode_token_with_moe_fn(
            kv_cache,
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
            None,
        )
    }
}
