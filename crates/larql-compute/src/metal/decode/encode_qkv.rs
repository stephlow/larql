//! Step 1 of the decode pipeline: input norm + fused Q/K/V projection.
//!
//! Two top-level paths gated on `uses_q4k`:
//!   - **Q4_K family** (Q4_K, Q6_K, Q4_KF) — RMS or LayerNorm into f32,
//!     then a fused QKV shader keyed on the (wq.fmt, wk.fmt, wv.fmt)
//!     triplet:
//!       * uniform Q4_K / Q4_KF → `q4k_qkv_proj` / `q4kf_qkv_proj`
//!       * Q4_K Q/K + Q6_K V (Gemma 3 / 4 Ollama convention) →
//!         `q4k_q6k_qkv_proj`
//!       * anything else → per-projection fallback through `quant_matvec`
//!   - **Q4_0** (legacy Q8 input) — fused norm+Q8 quantize, then
//!     `q8_qkv_proj`.
//!
//! Used to live inline in `decode_token_with_moe_fn`. Pulled out here
//! so the hot decode function stays scannable.

use metal::{ComputeCommandEncoderRef, MTLSize};

use crate::metal::MetalBackend;
use crate::FullPipelineLayer;

/// Buffer references the QKV step reads or writes.
pub(super) struct QkvBufs<'a> {
    // Input
    pub h_in: &'a metal::Buffer,
    // Per-layer weights + scales
    pub input_norm: &'a metal::Buffer,
    pub input_norm_bias: Option<&'a [f32]>,
    pub wq: &'a metal::Buffer,
    pub wk: &'a metal::Buffer,
    pub wv: &'a metal::Buffer,
    pub wq_scales: &'a metal::Buffer, // Q4_0 path only; ignored otherwise
    pub wk_scales: &'a metal::Buffer,
    pub wv_scales: &'a metal::Buffer,
    // Outputs
    pub norm_out: &'a metal::Buffer,
    pub q_out: &'a metal::Buffer,
    pub k_out: &'a metal::Buffer,
    pub v_out: &'a metal::Buffer,
    // Scratch (Q4_0 path only)
    pub ffn_q8: &'a metal::Buffer,
    pub ffn_q8s: &'a metal::Buffer,
}

#[derive(Copy, Clone)]
pub(super) struct QkvDims {
    pub hidden: usize,
    pub layer_q_dim: usize,
    pub layer_kv_dim: usize,
    pub eps: f32,
    pub norm_offset: f32,
}

impl MetalBackend {
    /// Encode input norm + fused QKV projection. `uses_q4k` selects the
    /// top-level path; the layer's per-projection formats select the
    /// inner shader. Behaviour mirrors the inline form previously in
    /// `decode/mod.rs` byte-for-byte.
    pub(super) fn encode_input_norm_and_qkv(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: &FullPipelineLayer,
        bufs: QkvBufs<'_>,
        dims: QkvDims,
        uses_q4k: bool,
    ) {
        if uses_q4k {
            self.encode_q4k_input_norm(enc, layer, &bufs, dims);
            self.encode_q4k_qkv(enc, layer, &bufs, dims);
        } else {
            self.encode_q4_0_norm_and_qkv(enc, layer, &bufs, dims);
        }
    }

    // ── Q4_K family: norm → f32, then fused QKV shader ───────────────────────

    fn encode_q4k_input_norm(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: &FullPipelineLayer,
        bufs: &QkvBufs<'_>,
        dims: QkvDims,
    ) {
        use crate::metal::ops::full_pipeline::encode_rms_norm;
        let QkvDims { hidden, eps, norm_offset, .. } = dims;

        if layer.norm_type == crate::NormType::LayerNorm {
            let len_val = hidden as u32;
            if let Some(bias) = bufs.input_norm_bias {
                let bias_buf = self.bufs.get_f32(bias);
                enc.set_compute_pipeline_state(&self.layer_norm_pipeline);
                enc.set_buffer(0, Some(bufs.h_in), 0);
                enc.set_buffer(1, Some(bufs.input_norm), 0);
                enc.set_buffer(2, Some(&bias_buf), 0);
                enc.set_buffer(3, Some(bufs.norm_out), 0);
                enc.set_bytes(4, 4, &len_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(6, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
            } else {
                enc.set_compute_pipeline_state(&self.layer_norm_no_bias_pipeline);
                enc.set_buffer(0, Some(bufs.h_in), 0);
                enc.set_buffer(1, Some(bufs.input_norm), 0);
                enc.set_buffer(2, Some(bufs.norm_out), 0);
                enc.set_bytes(3, 4, &len_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(4, 4, &eps as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(5, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
            }
            enc.dispatch_threads(
                MTLSize::new(hidden as u64, 1, 1),
                MTLSize::new(256.min(hidden as u64), 1, 1),
            );
        } else {
            encode_rms_norm(
                enc, &self.rms_norm_pipeline,
                bufs.h_in, bufs.input_norm, bufs.norm_out,
                hidden, eps, norm_offset,
            );
        }
    }

    fn encode_q4k_qkv(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: &FullPipelineLayer,
        bufs: &QkvBufs<'_>,
        dims: QkvDims,
    ) {
        let QkvDims { hidden, layer_q_dim, layer_kv_dim, .. } = dims;

        // Three paths, in priority order: uniform Q4_K/Q4_KF → fused
        // single shader; mixed Q4_K Q/K + Q6_K V → dedicated shader;
        // anything else → per-projection fallback.
        let uniform_q4k = layer.wq.format == layer.wk.format
            && layer.wk.format == layer.wv.format
            && layer.wq.format != crate::QuantFormat::Q6_K;
        let mixed_q4k_q6k_v = layer.wq.format == crate::QuantFormat::Q4_K
            && layer.wk.format == crate::QuantFormat::Q4_K
            && layer.wv.format == crate::QuantFormat::Q6_K;

        if uniform_q4k {
            let fused_pipe = if layer.wq.format == crate::QuantFormat::Q4_KF {
                &self.q4kf_qkv_proj_pipeline
            } else {
                &self.q4k_qkv_proj_pipeline
            };
            crate::metal::stages::qkv_proj::encode_fused_f32(
                enc, &fused_pipe.state,
                bufs.wq, bufs.wk, bufs.wv,
                bufs.norm_out, 0,
                bufs.q_out, 0, bufs.k_out, 0, bufs.v_out, 0,
                layer_q_dim, layer_kv_dim, hidden,
            );
        } else if mixed_q4k_q6k_v {
            use crate::metal::shaders::q4k_q6k_qkv_proj as sh;
            let total_rows = (layer_q_dim + layer_kv_dim + layer_kv_dim) as u64;
            let num_tgs = total_rows.div_ceil(sh::ROWS_PER_TG);
            let q_rows_u = layer_q_dim as u32;
            let k_rows_u = layer_kv_dim as u32;
            let v_rows_u = layer_kv_dim as u32;
            let k_u = hidden as u32;
            enc.set_compute_pipeline_state(&self.q4k_q6k_qkv_proj_pipeline.state);
            enc.set_buffer(0, Some(bufs.wq), 0);
            enc.set_buffer(1, Some(bufs.wk), 0);
            enc.set_buffer(2, Some(bufs.wv), 0);
            enc.set_buffer(3, Some(bufs.norm_out), 0);
            enc.set_buffer(4, Some(bufs.q_out), 0);
            enc.set_buffer(5, Some(bufs.k_out), 0);
            enc.set_buffer(6, Some(bufs.v_out), 0);
            enc.set_bytes(7, 4, &q_rows_u as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(8, 4, &k_rows_u as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(9, 4, &v_rows_u as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(10, 4, &k_u as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(num_tgs, 1, 1),
                MTLSize::new(sh::THREADS_PER_TG, 1, 1),
            );
        } else {
            // Mixed-but-unsupported (e.g. Q4_KF + Q6_K, or Q4_0 legacy):
            // per-projection dispatch through the format-aware helper.
            use crate::metal::stages::qkv_proj::{self, Proj};
            use crate::metal::stages::quant_matvec::Pipelines;
            let pipes = Pipelines {
                q4kf_proj: Some(&self.q4kf_proj_pipeline.state),
                q4k_matvec_fallback: &self.q4k_matvec_pipeline.state,
                q6k_matvec: &self.q6k_matvec_pipeline.state,
                q4_matvec: &self.q4.matvec,
            };
            qkv_proj::encode_per_proj(
                enc, &pipes,
                bufs.norm_out, 0,
                // Q8 bufs unused for f32-input formats — pass norm as a
                // harmless placeholder.
                bufs.norm_out, 0, bufs.norm_out, 0,
                [
                    Proj { format: layer.wq.format, w_buf: bufs.wq, out_buf: bufs.q_out, out_off: 0, rows: layer_q_dim },
                    Proj { format: layer.wk.format, w_buf: bufs.wk, out_buf: bufs.k_out, out_off: 0, rows: layer_kv_dim },
                    Proj { format: layer.wv.format, w_buf: bufs.wv, out_buf: bufs.v_out, out_off: 0, rows: layer_kv_dim },
                ],
                hidden,
            );
        }
    }

    // ── Q4_0 legacy: norm+Q8 → Q8 QKV ────────────────────────────────────────

    fn encode_q4_0_norm_and_qkv(
        &self,
        enc: &ComputeCommandEncoderRef,
        _layer: &FullPipelineLayer,
        bufs: &QkvBufs<'_>,
        dims: QkvDims,
    ) {
        let QkvDims { hidden, layer_q_dim, layer_kv_dim, eps, norm_offset } = dims;
        let hidden_val = hidden as u32;

        // Fused norm + Q8 quantize (in-place into the FFN scratch
        // buffers — they're re-quantised before the FFN dispatch).
        enc.set_compute_pipeline_state(&self.rms_norm_q8_pipeline);
        enc.set_buffer(0, Some(bufs.h_in), 0);
        enc.set_buffer(1, Some(bufs.input_norm), 0);
        enc.set_buffer(2, Some(bufs.ffn_q8), 0);
        enc.set_buffer(3, Some(bufs.ffn_q8s), 0);
        enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
        enc.set_bytes(6, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(256.min(hidden as u64), 1, 1),
        );

        let total_rows = (layer_q_dim + layer_kv_dim + layer_kv_dim) as u32;
        let q_rows = layer_q_dim as u32;
        let k_rows = layer_kv_dim as u32;
        let v_rows = layer_kv_dim as u32;
        let k_val = hidden as u32;
        enc.set_compute_pipeline_state(&self.q8_qkv_proj_pipeline);
        enc.set_buffer(0, Some(bufs.wq), 0);
        enc.set_buffer(1, Some(bufs.wk), 0);
        enc.set_buffer(2, Some(bufs.wv), 0);
        enc.set_buffer(3, Some(bufs.ffn_q8), 0);
        enc.set_buffer(4, Some(bufs.wq_scales), 0);
        enc.set_buffer(5, Some(bufs.wk_scales), 0);
        enc.set_buffer(6, Some(bufs.wv_scales), 0);
        enc.set_buffer(7, Some(bufs.ffn_q8s), 0);
        enc.set_buffer(8, Some(bufs.q_out), 0);
        enc.set_buffer(9, Some(bufs.k_out), 0);
        enc.set_buffer(10, Some(bufs.v_out), 0);
        enc.set_bytes(11, 4, &q_rows as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(12, 4, &k_rows as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(13, 4, &v_rows as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(14, 4, &k_val as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            MTLSize::new((total_rows as u64).div_ceil(8), 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }
}
