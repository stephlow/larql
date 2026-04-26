//! Full pipeline: ALL Q4 (attention + FFN) in ONE Metal command buffer.
//!
//! Correct inference path with norms and residual connections:
//!   Per layer:
//!     1. rms_norm(h, input_norm) → h_norm
//!     2. Q4 Q/K/V projections from h_norm
//!     3. Fused attention (RoPE + GQA + softcap)
//!     4. Q4 O projection
//!     5. Post-attn norm (if post_norms) + residual_add(h, o_out) → h
//!     6. rms_norm(h, post_attn_norm) → h_ffn
//!     7. Q4 gate/up → GEGLU → Q4 down
//!     8. Post-FFN norm (if post_norms) + residual_add(h, ffn_out) → h
//!     9. Q8 quantize h → next layer

use std::ffi::c_void;
use metal::*;

use crate::metal::buffers::BufferCache;
use crate::metal::ops::q4_common::Q4Pipelines;

/// Weights for one transformer layer — ALL Q4 + norm weights.
/// Matches `crate::FullPipelineLayer` but with borrowed Metal-friendly data.
pub struct LayerWeights<'a> {
    pub wq_q4: &'a [u8],
    pub wk_q4: &'a [u8],
    pub wv_q4: &'a [u8],
    pub wo_q4: &'a [u8],
    pub gate_q4: &'a [u8],
    pub up_q4: &'a [u8],
    pub down_t_q4: &'a [u8],
}

#[allow(clippy::too_many_arguments)]
pub fn encode_rms_norm(
    enc: &ComputeCommandEncoderRef,
    rms_pipeline: &ComputePipelineState,
    buf_x: &Buffer,
    buf_weight: &Buffer,
    buf_out: &Buffer,
    len: usize,
    eps: f32,
    offset: f32,
) {
    let len_val = len as u32;
    enc.set_compute_pipeline_state(rms_pipeline);
    enc.set_buffer(0, Some(buf_x), 0);
    enc.set_buffer(1, Some(buf_weight), 0);
    enc.set_buffer(2, Some(buf_out), 0);
    enc.set_bytes(3, 4, &len_val as *const u32 as *const c_void);
    enc.set_bytes(4, 4, &eps as *const f32 as *const c_void);
    enc.set_bytes(5, 4, &offset as *const f32 as *const c_void);
    // Single threadgroup — cooperative SIMD reduction requires all threads in one TG.
    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256.min(len as u64), 1, 1));
}

pub fn encode_residual_add(
    enc: &ComputeCommandEncoderRef,
    add_pipeline: &ComputePipelineState,
    buf_a: &Buffer,
    buf_b: &Buffer,
    buf_out: &Buffer,
    len: usize,
) {
    let len_val = len as u32;
    enc.set_compute_pipeline_state(add_pipeline);
    enc.set_buffer(0, Some(buf_a), 0);
    enc.set_buffer(1, Some(buf_b), 0);
    enc.set_buffer(2, Some(buf_out), 0);
    enc.set_bytes(3, 4, &len_val as *const u32 as *const c_void);
    enc.dispatch_threads(MTLSize::new(len as u64, 1, 1), MTLSize::new(256.min(len as u64), 1, 1));
}

/// Q4_0 matvec with explicit input/output offsets (bytes).
/// Same as `encode_q4_matvec` but lets the caller point at a specific row of
/// a multi-position staging buffer — used in prefill (`seq_len > 1`) where
/// Run all layers in ONE Metal command buffer with correct norms and residuals.
///
/// Multi-position aware: processes `seq_len >= 1` tokens through every stage.
/// For `seq_len == 1` this is the decode path; for `seq_len > 1` it is the
/// prefill path and populates the KV cache for subsequent decode.
///
/// Architecture coverage:
/// - Pre-norm (Llama / Mistral / Qwen): `has_post_norms = false`, `use_qk_norm = false`
/// - Post-norm + QK-norm (Gemma 3 / Gemma 4): `has_post_norms = true`, `use_qk_norm = true`
/// - Gated FFN (default) + Standard FFN (StarCoder2)
/// - SiLU + GELU-tanh activations
/// - Q4_K / Q6_K / Q4_KF / Q8_0 attention weights (Q4_K/Q6_K/Q4_KF take f32 input;
///   Q8_0 takes Q8 input via fused norm+Q8 shader)
///
/// QK-norm ordering: when `use_qk_norm` is true and `qk_norm_pipeline` is
/// supplied, QK-norm is applied **before** RoPE (matching `decode_token` and
/// the Gemma 3/4 reference implementations). `fused_attention` is then called
/// with `use_qk_norm = 0` to avoid a second normalisation.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_full_pipeline(
    queue: &CommandQueue,
    bufs: &BufferCache,
    q4: &Q4Pipelines,
    geglu_pipeline: &ComputePipelineState,
    geglu_gelu_tanh_pipeline: &ComputePipelineState,
    silu_pipeline: &ComputePipelineState,
    gelu_tanh_pipeline: &ComputePipelineState,
    q8_quant_pipeline: &ComputePipelineState,
    fused_attn_pipeline: Option<&ComputePipelineState>,
    _q8_matvec_pipeline: &ComputePipelineState,
    q8_qkv_proj_pipeline: &ComputePipelineState,
    q4k_matvec_pipeline: &crate::metal::kernel::KernelHandle,
    q6k_matvec_pipeline: &crate::metal::kernel::KernelHandle,
    rms_norm_pipeline: &ComputePipelineState,
    residual_add_pipeline: &ComputePipelineState,
    rms_norm_q8_pipeline: &ComputePipelineState,
    _residual_norm_q8_pipeline: &ComputePipelineState,
    q4k_qkv_proj_pipeline: Option<&ComputePipelineState>,
    q4kf_qkv_proj_pipeline: Option<&ComputePipelineState>,
    q4kf_proj_pipeline: Option<&ComputePipelineState>,
    rope_at_pos_pipeline: Option<&ComputePipelineState>,
    qk_norm_pipeline: Option<&ComputePipelineState>,
    scale_vector_pipeline: Option<&ComputePipelineState>,
    // Fused activation+down kernels (KernelHandles). Engaged when
    // down.format ∈ {Q4_K, Q6_K} — saves one dispatch + an
    // inter-sized activation buffer write/read per position. None
    // for backends that don't have these compiled.
    fused_q4k_geglu_silu_down: Option<&crate::metal::kernel::KernelHandle>,
    fused_q4k_geglu_gelu_tanh_down: Option<&crate::metal::kernel::KernelHandle>,
    fused_q6k_geglu_silu_down: Option<&crate::metal::kernel::KernelHandle>,
    fused_q6k_geglu_gelu_tanh_down: Option<&crate::metal::kernel::KernelHandle>,
    mut kv_cache: Option<&mut crate::metal::ops::kv_cache::KVCache>,
    layers: &[crate::FullPipelineLayer],
    x: &[f32],
    hidden: usize,
    inter: usize,
    q_dim: usize,
    _kv_dim: usize,
    seq_len: usize,
    _num_q_heads: usize,
    _num_kv_heads: usize,
    _head_dim: usize,
    _rope_base: f32, // global fallback; per-layer layers[l].rope_base used in loop
    use_qk_norm: bool,
    softcap: f32,
    // Optional per-layer MoE callback for hybrid MoE models (e.g. Gemma 4 26B A4B).
    // When provided, the function commits the GPU command buffer after each MoE layer,
    // calls this closure with `(layer_idx, h_post_attn, new_h)` (both slices are
    // `[seq_len × hidden]`), and restarts the command buffer for the next layer.
    // The closure is responsible for running CPU MoE and accumulating the result
    // into `new_h`, as well as applying any outer post-FFN norm and layer_scalar.
    // The GPU layer_scalar step (step 11) is skipped for layers where the callback
    // fires so the closure can apply it correctly after combining dense + MoE.
    // Pass `None` for models without MoE — behaviour is identical to the prior API.
    mut moe_fn: Option<&mut dyn FnMut(usize, &[f32], &mut [f32])>,
) -> Vec<f32> {
    let num_layers = layers.len();

    // All per-layer scratch + cached weight buffers in one struct.
    // See `LayerBuffers::allocate` for the sizing rationale (Gemma 4
    // mixed sliding/global geometry, Q8 staging shared between the
    // attention-input and O-projection paths, etc.).
    let lb = super::buffers::LayerBuffers::allocate(
        bufs, layers, x, hidden, inter, seq_len, q_dim,
    );
    // Local aliases to keep the orchestration body readable. Using
    // shared references means the body's existing `wq_bufs[l]` etc.
    // resolve through `Vec<Buffer>` indexing unchanged.
    // Q/K/V weight & scale buffers are consumed inside the
    // input-norm + QKV stage helper (`stages::encode_input_norm_and_qkv`)
    // — the helper reads them off `lb` directly. The rest of the body
    // only needs `wo` (for o_proj).
    let wo_bufs        = &lb.wo;
    let gate_bufs      = &lb.gate;
    let up_bufs        = &lb.up;
    let down_bufs      = &lb.down;
    let post_attn_norm_bufs = &lb.post_attn_norm;
    let pre_ffn_norm_bufs  = &lb.pre_ffn_norm;
    let post_ffn_norm_bufs = &lb.post_ffn_norm;
    let h_bufs         = &lb.h;
    let q_outs         = &lb.q_out;
    let k_outs         = &lb.k_out;
    let v_outs         = &lb.v_out;
    let attn_outs      = &lb.attn_out;
    let o_outs         = &lb.o_out;
    let h_post_attns   = &lb.h_post_attn;
    let ffn_norm_outs  = &lb.ffn_norm_out;
    let gate_outs      = &lb.gate_out;
    let up_outs        = &lb.up_out;
    let act_bufs_vec   = &lb.act_buf;
    let down_outs      = &lb.down_out;
    let q8_bufs        = &lb.q8;
    let q8s_bufs       = &lb.q8s;
    let ffn_q8_bufs    = &lb.ffn_q8;
    let ffn_q8s_bufs   = &lb.ffn_q8s;
    let q8_row_max     = lb.q8_row_max;
    let q8s_row_bytes  = lb.q8s_row_bytes;

    // Per-layer GPU commit mode: used for hybrid MoE models where the CPU
    // expert block runs after each layer's dense FFN. When active, we commit
    // after every layer that has MoE (not once at the end), restart the
    // command buffer, and call the caller-supplied closure.
    let needs_per_layer_commit = moe_fn.is_some() && layers.iter().any(|l| l.moe.is_some());

    let mut cmd = queue.new_command_buffer().to_owned();
    let dump_path = std::env::var("LARQL_METAL_DUMP_LAYERS").ok();
    super::dump::dump_h_embed(dump_path.as_deref(), &lb, seq_len, hidden);

    for l in 0..num_layers {
        let eps = layers[l].eps;
        let layer_rope_base = layers[l].rope_base;
        let layer_head_dim = layers[l].head_dim;
        let layer_num_q_heads = layers[l].num_q_heads;
        let layer_num_kv_heads = layers[l].num_kv_heads;
        let layer_q_dim = layer_num_q_heads * layer_head_dim;
        let layer_kv_dim = layer_num_kv_heads * layer_head_dim;
        let layer_attn_scale = layers[l].attn_scale;
        let norm_offset = layers[l].norm_offset;
        let has_post_norms = layers[l].has_post_norms;

        // ── 1+3. Input norm + Q/K/V projections (format-aware) ──
        //
        // Per-position offsets (bytes). `layer_q_dim` / `layer_kv_dim`
        // are the **this layer's** actual dimensions — Gemma 4
        // alternates sliding (head_dim=256) and global (head_dim=512)
        // layers so these differ per layer. Offsets into the per-layer
        // allocated buffers use the per-layer dims; `q_dim` / `kv_dim`
        // are only used as fallback stride for the Q8 staging bucket.
        let h_off = |p: usize| (p * hidden * 4) as u64;
        let q_off = |p: usize| (p * layer_q_dim * 4) as u64;
        let q8_off = |p: usize| (p * q8_row_max) as u64;
        let q8s_off = |p: usize| (p * q8s_row_bytes) as u64;
        let qm_pipes = crate::metal::stages::quant_matvec::Pipelines {
            q4kf_proj: q4kf_proj_pipeline,
            q4k_matvec_fallback: q4k_matvec_pipeline,
            q6k_matvec: q6k_matvec_pipeline,
            q4_matvec: &q4.matvec,
        };
        super::stages::encode_input_norm_and_qkv(
            cmd.as_ref(),
            &layers[l], l, seq_len, hidden,
            &super::stages::LayerCtx {
                eps, norm_offset,
                layer_q_dim, layer_kv_dim,
                q8_row_max, q8s_row_bytes,
            },
            &super::stages::InputNormQkvPipes {
                rms_norm: rms_norm_pipeline,
                rms_norm_q8: rms_norm_q8_pipeline,
                q8_qkv_proj: q8_qkv_proj_pipeline,
                q4kf_qkv_proj: q4kf_qkv_proj_pipeline,
                q4k_qkv_proj: q4k_qkv_proj_pipeline,
                qm_pipes,
            },
            &lb,
        );
        // qm_pipes is recomputed below for the FFN/down stages because
        // it borrows from local references that were moved into the
        // helper above.
        let qm_pipes = crate::metal::stages::quant_matvec::Pipelines {
            q4kf_proj: q4kf_proj_pipeline,
            q4k_matvec_fallback: q4k_matvec_pipeline,
            q6k_matvec: q6k_matvec_pipeline,
            q4_matvec: &q4.matvec,
        };

        // ── 3 (pre). Optional parameter-free V-norm (Gemma 4). ──
        if layers[l].has_v_norm {
            if let Some(qk_norm_pipe) = qk_norm_pipeline {
                let ones: Vec<f32> = vec![1.0; layer_head_dim];
                let ones_buf = bufs.transient_from_f32(&ones);
                let enc = cmd.new_compute_command_encoder();
                crate::metal::stages::qk_norm::encode_v_norm(
                    enc, qk_norm_pipe,
                    &v_outs[l], &ones_buf,
                    seq_len, layer_num_kv_heads, layer_head_dim, eps,
                );
                enc.end_encoding();
            }
        }

        // Stage dump: Q just after QKV projection, before QK-norm.
        cmd = super::dump::dump_layer0_q_after_stage(
            dump_path.as_deref(), queue, cmd, &lb, "raw",
            seq_len, layer_q_dim, l,
        );

        // ── 3a. QK-norm on Q and K (pre-RoPE). Gemma 3 / Gemma 4. ──
        let applied_prerope_qk_norm = if use_qk_norm {
            if let (Some(qk_norm_pipe), Some(q_w_slice), Some(k_w_slice)) =
                (qk_norm_pipeline, layers[l].q_norm_weight, layers[l].k_norm_weight)
            {
                let q_w_buf = bufs.get_f32(q_w_slice);
                let k_w_buf = bufs.get_f32(k_w_slice);
                let enc = cmd.new_compute_command_encoder();
                crate::metal::stages::qk_norm::encode_qk_norm(
                    enc, qk_norm_pipe,
                    &q_outs[l], &q_w_buf,
                    &k_outs[l], &k_w_buf,
                    seq_len, layer_num_q_heads, layer_num_kv_heads, layer_head_dim,
                    eps, layers[l].qk_norm_offset,
                );
                enc.end_encoding();
                true
            } else {
                // use_qk_norm requested but pipeline or weights missing —
                // fall back to fused_attention's internal QK-norm (legacy path).
                false
            }
        } else {
            false
        };

        // Stage dump: Q after QK-norm, before RoPE.
        cmd = super::dump::dump_layer0_q_after_stage(
            dump_path.as_deref(), queue, cmd, &lb, "after_qk_norm",
            seq_len, layer_q_dim, l,
        );

        // ── 3b. Apply RoPE separately when populating KV cache ──
        let use_separate_rope = kv_cache.is_some() && rope_at_pos_pipeline.is_some();
        if use_separate_rope {
            let enc = cmd.new_compute_command_encoder();
            crate::metal::stages::rope::encode(
                enc, rope_at_pos_pipeline.unwrap(),
                &q_outs[l], &k_outs[l],
                seq_len, layer_num_q_heads, layer_num_kv_heads, layer_head_dim,
                layers[l].rotary_dim, layer_rope_base,
            );
            enc.end_encoding();
        }

        // ── 4. Fused attention (RoPE + GQA + softcap, multi-position). ──
        if let Some(fused_pipeline) = fused_attn_pipeline {
            let enc = cmd.new_compute_command_encoder();
            crate::metal::stages::attention::encode(
                enc, fused_pipeline,
                &q_outs[l], &k_outs[l], &v_outs[l], &attn_outs[l],
                seq_len, layer_num_q_heads, layer_num_kv_heads, layer_head_dim,
                layer_attn_scale, layer_rope_base,
                crate::metal::stages::attention::Flags {
                    // Caller pre-applied QK-norm: tell shader to skip its internal
                    // normalisation so we don't double-normalise.
                    use_qk_norm: use_qk_norm && !applied_prerope_qk_norm,
                    skip_rope: use_separate_rope,
                    softcap,
                    rotary_dim: layers[l].rotary_dim as u32,
                },
            );
            enc.end_encoding();
        }

        // ── 5. O projection. Per position. ──
        for pos in 0..seq_len {
            let enc = cmd.new_compute_command_encoder();
            crate::metal::stages::o_proj::encode(
                enc, &qm_pipes, q8_quant_pipeline,
                layers[l].wo.format,
                &wo_bufs[l],
                &attn_outs[l], q_off(pos),
                &q8_bufs[l], q8_off(pos),
                &q8s_bufs[l], q8s_off(pos),
                &o_outs[l], h_off(pos),
                layer_q_dim, hidden,
            );
            enc.end_encoding();
        }

        // ── 6. Post-attention residual + pre-FFN norm (+ optional Q8 quant). ──
        //
        // Two output representations are needed here:
        //   (a) ffn_norm_outs[l]  — f32 per position; consumed by Q4_K / Q4_KF /
        //                            Q6_K FFN which expect f32 input.
        //   (b) ffn_q8_bufs[l] + ffn_q8s_bufs[l] — Q8 + scales per position;
        //       consumed only by Q4_0 / Q8_0 FFN.
        // `h_post_attns[l]` holds the post-residual f32 hidden state for the
        // final residual add at the end of this layer (step 10).
        let ffn_format = layers[l].gate.format;
        let ffn_needs_q8 = matches!(ffn_format,
            crate::QuantFormat::Q4_0 | crate::QuantFormat::Q8_0);
        let pre_ffn_weight_buf: &metal::Buffer = if has_post_norms {
            pre_ffn_norm_bufs[l].as_ref().unwrap_or(&post_attn_norm_bufs[l])
        } else {
            &post_attn_norm_bufs[l]
        };
        {
            let mut scratch = |bytes: u64| bufs.output(bytes);
            let enc = cmd.new_compute_command_encoder();
            crate::metal::stages::residual::encode_post_attn(
                enc, rms_norm_pipeline, residual_add_pipeline, q8_quant_pipeline,
                &mut scratch,
                &h_bufs[l], &o_outs[l], &h_post_attns[l], &ffn_norm_outs[l],
                &post_attn_norm_bufs[l], pre_ffn_weight_buf,
                &ffn_q8_bufs[l], &ffn_q8s_bufs[l],
                seq_len, hidden, eps, norm_offset,
                has_post_norms, ffn_needs_q8,
                (hidden * 4) as u64,
                hidden as u64,
                (hidden.div_ceil(32) * 4) as u64,
            );
            enc.end_encoding();
        }

        // ── 7-9. FFN: gate+up → activation → down. Format-aware per position. ──
        {
            use crate::metal::stages::ffn;
            let act = match layers[l].activation {
                crate::Activation::GeluTanh => ffn::Activation::GeluTanh,
                _ => ffn::Activation::SiLU,
            };
            let h_stride = (hidden * 4) as u64;
            let inter_stride = (inter * 4) as u64;
            let q8_stride = hidden as u64;
            let q8s_stride = (hidden.div_ceil(32) * 4) as u64;

            let enc = cmd.new_compute_command_encoder();
            if layers[l].ffn_type == crate::FfnType::Standard {
                ffn::encode_standard(
                    enc, &qm_pipes, silu_pipeline, gelu_tanh_pipeline,
                    layers[l].up.format, layers[l].down.format, act,
                    &up_bufs[l], &down_bufs[l],
                    &ffn_norm_outs[l], &ffn_q8_bufs[l], &ffn_q8s_bufs[l],
                    &up_outs[l], &act_bufs_vec[l], &down_outs[l],
                    seq_len, inter, hidden,
                    h_stride, inter_stride, q8_stride, q8s_stride,
                );
            } else {
                ffn::encode_gated(
                    enc, &qm_pipes, geglu_pipeline, geglu_gelu_tanh_pipeline,
                    ffn::FusedGegluDown {
                        q4k_silu: fused_q4k_geglu_silu_down,
                        q4k_gelu_tanh: fused_q4k_geglu_gelu_tanh_down,
                        q6k_silu: fused_q6k_geglu_silu_down,
                        q6k_gelu_tanh: fused_q6k_geglu_gelu_tanh_down,
                    },
                    layers[l].gate.format, layers[l].up.format, layers[l].down.format, act,
                    &gate_bufs[l], &up_bufs[l], &down_bufs[l],
                    &ffn_norm_outs[l], &ffn_q8_bufs[l], &ffn_q8s_bufs[l],
                    &gate_outs[l], &up_outs[l], &act_bufs_vec[l], &down_outs[l],
                    seq_len, inter, hidden,
                    h_stride, inter_stride, q8_stride, q8s_stride,
                );
            }
            enc.end_encoding();
        }

        // ── 10. Post-FFN: optional norm, then residual add → h for next layer. ──
        {
            let mut scratch = |bytes: u64| bufs.output(bytes);
            let enc = cmd.new_compute_command_encoder();
            crate::metal::stages::residual::encode_post_ffn(
                enc, rms_norm_pipeline, residual_add_pipeline,
                &mut scratch,
                &down_outs[l], &h_post_attns[l], &h_bufs[l + 1],
                post_ffn_norm_bufs[l].as_ref(),
                seq_len, hidden, eps, norm_offset,
                has_post_norms,
                (hidden * 4) as u64,
            );
            enc.end_encoding();
        }

        // ── 11. Per-layer residual scalar (Gemma 4). ──
        // Skipped for MoE layers in per-layer-commit mode: the moe_fn
        // closure applies layer_scalar after combining dense + MoE output,
        // which is the correct application point (HF: `hidden *= layer_scalar`
        // after the full FFN block including experts).
        let is_moe_layer = needs_per_layer_commit && layers[l].moe.is_some();
        if !is_moe_layer {
            if let Some(scale_pipe) = scale_vector_pipeline {
                let enc = cmd.new_compute_command_encoder();
                crate::metal::stages::layer_scalar::encode(
                    enc, scale_pipe, &h_bufs[l + 1], seq_len, hidden, layers[l].layer_scalar,
                );
                enc.end_encoding();
            }
        }

        // End-of-layer dump (LARQL_METAL_DUMP_LAYERS=<dir>) — bisects
        // CPU/Metal drift layer-by-layer.
        cmd = super::dump::dump_layer_snapshots(
            dump_path.as_deref(), queue, cmd, &lb,
            layers, l, seq_len, hidden, inter,
        );

        // ── Per-layer MoE interleave. ──
        // After the dense FFN is committed, run the CPU expert block for
        // each prompt position and accumulate into `h_bufs[l+1]`. Then
        // restart the command buffer for the next layer.
        if needs_per_layer_commit {
            cmd.commit();
            cmd.wait_until_completed();

            // KV cache: copy this layer's K/V before the caller reads
            // `h_post_attn` or touches `new_h`.
            if let Some(kv) = kv_cache.as_mut() {
                super::kv_copy::populate_kv_one_layer(
                    kv, bufs, &lb, &layers[l], l, seq_len,
                );
            }

            if is_moe_layer {
                if let Some(ref mut f) = moe_fn {
                    let ha_ptr = lb.h_post_attn[l].contents() as *const f32;
                    let h_ptr = lb.h[l + 1].contents() as *mut f32;
                    // SAFETY: GPU finished (wait_until_completed). Both buffers
                    // are pre-allocated for `seq_len * hidden` f32s.
                    let ha = unsafe { std::slice::from_raw_parts(ha_ptr, seq_len * hidden) };
                    let h = unsafe { std::slice::from_raw_parts_mut(h_ptr, seq_len * hidden) };
                    f(l, ha, h);
                }
            }

            if l < num_layers - 1 {
                cmd = queue.new_command_buffer().to_owned();
            }
        }
    }

    if !needs_per_layer_commit {
        cmd.commit();
        cmd.wait_until_completed();

        // Post-commit: populate persistent KV cache from GPU-computed
        // RoPE'd K/V (buffers are readable now that the command buffer is
        // finished).
        super::kv_copy::populate_kv_after_commit(
            kv_cache, bufs, &lb, layers, seq_len,
        );
    }

    // Read final hidden state — `seq_len * hidden` floats, caller reshapes
    // to [seq_len, hidden] (see `layer_graph::generate`).
    crate::metal::buffers::read_buffer_f32(&h_bufs[num_layers], seq_len * hidden)
}
