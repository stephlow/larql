//! Step 6 of the decode pipeline: format-aware FFN dispatch.
//!
//! Three production paths on the same `(gate, up, down)` triplet:
//!   - **Q4_KF** — llama.cpp-exact kernel; fused gate+up; `act_buf` then
//!     down via `quant_matvec` (mixed-quant aware).
//!   - **Q4_K** — our kernel; fused gate+up; down via `quant_matvec`
//!     (Gemma 3 4B ships Q6_K down even when gate/up are Q4_K).
//!   - **Q4_0** (legacy) — Q8-input matvec for gate/up; `q4.f32_matvec`
//!     for down.
//!
//! Used to live inline in `decode_token_with_moe_fn`; pulled out here
//! so `decode/mod.rs` stays readable. Behaviour is byte-identical to
//! the original block.
//!
//! All buffer + pipeline references are held in `FfnBufs` and
//! `FfnDims` so the encoder method has a manageable signature.

use metal::{ComputeCommandEncoderRef, MTLSize};

use crate::metal::MetalBackend;
use crate::FullPipelineLayer;

/// Buffer references the FFN block reads or writes. The encoder is
/// passed separately so the method can also borrow `&self`.
pub(super) struct FfnBufs<'a> {
    // Weights for this layer
    pub gate_w: &'a metal::Buffer,
    pub up_w: &'a metal::Buffer,
    pub down_w: &'a metal::Buffer,
    // Inputs
    pub ffn_norm_out: &'a metal::Buffer, // f32 input (Q4_K / Q4_KF paths)
    pub ffn_q8: &'a metal::Buffer,       // Q8 input bytes (Q4_0 path)
    pub ffn_q8s: &'a metal::Buffer,      // Q8 input scales (Q4_0 path)
    // Scratch (gate output reused even on non-gated paths)
    pub gate_out_scratch: &'a metal::Buffer,
    pub up_out: &'a metal::Buffer,
    pub act_buf: &'a metal::Buffer,
    // Output
    pub down_out: &'a metal::Buffer,
}

#[derive(Copy, Clone)]
pub(super) struct FfnDims {
    pub hidden: usize,
    pub inter: usize,
    /// `inter` rounded up to the next multiple of 256 — used by the Q4K
    /// down dispatch when storage is per-row-padded super-blocks.
    pub inter_padded: usize,
}

impl MetalBackend {
    /// Encode the full FFN block (gate / up / activation / down) into
    /// the encoder. `ffn_uses_q4k` selects the path; the function
    /// returns the same `down_out` buffer the caller passed in via
    /// `bufs`. No commit/flush — the caller owns encoder lifecycle.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_ffn_step(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: &FullPipelineLayer,
        bufs: FfnBufs<'_>,
        dims: FfnDims,
        ffn_uses_q4k: bool,
    ) {
        let FfnDims { hidden, inter, inter_padded } = dims;
        let inter_val = inter as u32;
        let inter_padded_val = inter_padded as u32;
        let hidden_val = hidden as u32;

        let ffn_is_q4kf = layer.gate.format == crate::QuantFormat::Q4_KF;

        if ffn_is_q4kf {
            self.encode_q4kf_ffn(enc, layer, &bufs, hidden, inter, hidden_val, inter_val);
        } else if ffn_uses_q4k {
            self.encode_q4k_ffn(enc, layer, &bufs, hidden, inter, inter_padded,
                hidden_val, inter_val, inter_padded_val);
        } else {
            self.encode_q4_0_ffn(enc, layer, &bufs, hidden, inter, hidden_val, inter_val);
        }
    }

    // ── Q4_KF (GGUF) ─────────────────────────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    fn encode_q4kf_ffn(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: &FullPipelineLayer,
        bufs: &FfnBufs<'_>,
        hidden: usize,
        inter: usize,
        hidden_val: u32,
        inter_val: u32,
    ) {
        use crate::metal::shaders::q4kf_qkv_proj as q4kf;
        use crate::metal::shaders::q4kf_ffn_gate_up as q4kf_gu;
        let n_tgs_down = (hidden as u64).div_ceil(q4kf::ROWS_PER_TG);

        if layer.is_gated() {
            // Fused gate+up
            let n_tgs_per_mat = (inter as u64).div_ceil(q4kf_gu::ROWS_PER_TG);
            enc.set_compute_pipeline_state(&self.q4kf_ffn_gate_up_pipeline.state);
            enc.set_buffer(0, Some(bufs.gate_w), 0);
            enc.set_buffer(1, Some(bufs.up_w), 0);
            enc.set_buffer(2, Some(bufs.ffn_norm_out), 0);
            enc.set_buffer(3, Some(bufs.gate_out_scratch), 0);
            enc.set_buffer(4, Some(bufs.up_out), 0);
            enc.set_bytes(5, 4, &inter_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(6, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(n_tgs_per_mat * 2, 1, 1),
                MTLSize::new(q4kf_gu::THREADS_PER_TG, 1, 1),
            );

            // GEGLU
            self.encode_geglu(enc, layer, bufs, inter_val, inter as u64);

            // Down — format-aware (mixed Q4_KF + Q6_K is a real config)
            self.encode_qmv_down(enc, layer, bufs, hidden, inter);
            let _ = n_tgs_down;
        } else {
            // Standard FFN: up + activation + down
            let n_tgs_up = (inter as u64).div_ceil(q4kf::ROWS_PER_TG);
            enc.set_compute_pipeline_state(&self.q4kf_proj_pipeline.state);
            enc.set_buffer(0, Some(bufs.up_w), 0);
            enc.set_buffer(1, Some(bufs.ffn_norm_out), 0);
            enc.set_buffer(2, Some(bufs.up_out), 0);
            enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(MTLSize::new(n_tgs_up, 1, 1), MTLSize::new(q4kf::THREADS_PER_TG, 1, 1));

            self.encode_activation(enc, layer, bufs.up_out, bufs.act_buf, inter_val, inter as u64);

            enc.set_compute_pipeline_state(&self.q4kf_proj_pipeline.state);
            enc.set_buffer(0, Some(bufs.down_w), 0);
            enc.set_buffer(1, Some(bufs.act_buf), 0);
            enc.set_buffer(2, Some(bufs.down_out), 0);
            enc.set_bytes(3, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(MTLSize::new(n_tgs_down, 1, 1), MTLSize::new(q4kf::THREADS_PER_TG, 1, 1));
        }
    }

    // ── Q4_K ─────────────────────────────────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    fn encode_q4k_ffn(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: &FullPipelineLayer,
        bufs: &FfnBufs<'_>,
        hidden: usize,
        inter: usize,
        inter_padded: usize,
        hidden_val: u32,
        inter_val: u32,
        inter_padded_val: u32,
    ) {
        use crate::metal::shaders::q4k_matvec as q4k;
        use crate::metal::shaders::q4k_ffn_gate_up as q4k_gu;
        let n_tgs_down = (hidden as u64).div_ceil(q4k::ROWS_PER_TG);

        if layer.is_gated() {
            let n_tgs_per_mat = (inter as u64).div_ceil(q4k_gu::ROWS_PER_TG);
            enc.set_compute_pipeline_state(&self.q4k_ffn_gate_up_pipeline.state);
            enc.set_buffer(0, Some(bufs.gate_w), 0);
            enc.set_buffer(1, Some(bufs.up_w), 0);
            enc.set_buffer(2, Some(bufs.ffn_norm_out), 0);
            enc.set_buffer(3, Some(bufs.gate_out_scratch), 0);
            enc.set_buffer(4, Some(bufs.up_out), 0);
            enc.set_bytes(5, 4, &inter_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(6, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(n_tgs_per_mat * 2, 1, 1),
                MTLSize::new(q4k_gu::THREADS_PER_TG, 1, 1),
            );

            self.encode_geglu(enc, layer, bufs, inter_val, inter as u64);

            // Down projection — format-aware. Gemma 3 4B ships Q6_K
            // down even when gate/up are Q4_K. `inter_padded` matches
            // the stored super-block layout.
            use crate::metal::stages::quant_matvec::{self as qmv, Pipelines};
            let pipes = Pipelines {
                q4kf_proj: Some(&self.q4kf_proj_pipeline.state),
                q4k_matvec_fallback: &self.q4k_matvec_pipeline.state,
                q6k_matvec: &self.q6k_matvec_pipeline.state,
                q4_matvec: &self.q4.matvec,
            };
            qmv::encode(
                enc, layer.down.format, bufs.down_w,
                bufs.act_buf, 0,
                bufs.act_buf, 0, bufs.act_buf, 0, // Q8 unused for f32 input
                bufs.down_out, 0,
                &pipes,
                hidden, inter_padded,
            );
            let _ = n_tgs_down;
        } else {
            let n_tgs_up = (inter as u64).div_ceil(q4k::ROWS_PER_TG);
            enc.set_compute_pipeline_state(&self.q4k_matvec_pipeline.state);
            enc.set_buffer(0, Some(bufs.up_w), 0);
            enc.set_buffer(1, Some(bufs.ffn_norm_out), 0);
            enc.set_buffer(2, Some(bufs.up_out), 0);
            enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(MTLSize::new(n_tgs_up, 1, 1), MTLSize::new(q4k::THREADS_PER_TG, 1, 1));

            self.encode_activation(enc, layer, bufs.up_out, bufs.act_buf, inter_val, inter as u64);

            enc.set_compute_pipeline_state(&self.q4k_matvec_pipeline.state);
            enc.set_buffer(0, Some(bufs.down_w), 0);
            enc.set_buffer(1, Some(bufs.act_buf), 0);
            enc.set_buffer(2, Some(bufs.down_out), 0);
            enc.set_bytes(3, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &inter_padded_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(MTLSize::new(n_tgs_down, 1, 1), MTLSize::new(q4k::THREADS_PER_TG, 1, 1));
        }
    }

    // ── Q4_0 (legacy Q8 input path) ──────────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    fn encode_q4_0_ffn(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: &FullPipelineLayer,
        bufs: &FfnBufs<'_>,
        hidden: usize,
        inter: usize,
        hidden_val: u32,
        inter_val: u32,
    ) {
        // Geometry travels with the q4 matvec KernelHandle — single source
        // of truth, can't drift from the kernel's row map.
        let kernel = &self.q4.matvec;
        let n_tgs_ffn = (inter as u64).div_ceil(kernel.rows_per_tg);
        let tg_size = MTLSize::new(kernel.threads_per_tg, 1, 1);

        if layer.is_gated() {
            // Gate
            enc.set_compute_pipeline_state(&kernel.state);
            enc.set_buffer(0, Some(bufs.gate_w), 0);
            enc.set_buffer(1, Some(bufs.ffn_q8), 0);
            enc.set_buffer(2, Some(bufs.ffn_q8s), 0);
            enc.set_buffer(3, Some(bufs.gate_out_scratch), 0);
            enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(MTLSize::new(n_tgs_ffn, 1, 1), tg_size);
            // Up (reuse pipeline + bindings, swap matrix and out)
            enc.set_buffer(0, Some(bufs.up_w), 0);
            enc.set_buffer(3, Some(bufs.up_out), 0);
            enc.dispatch_thread_groups(MTLSize::new(n_tgs_ffn, 1, 1), tg_size);

            self.encode_geglu(enc, layer, bufs, inter_val, inter as u64);
        } else {
            enc.set_compute_pipeline_state(&kernel.state);
            enc.set_buffer(0, Some(bufs.up_w), 0);
            enc.set_buffer(1, Some(bufs.ffn_q8), 0);
            enc.set_buffer(2, Some(bufs.ffn_q8s), 0);
            enc.set_buffer(3, Some(bufs.up_out), 0);
            enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(MTLSize::new(n_tgs_ffn, 1, 1), tg_size);

            self.encode_activation(enc, layer, bufs.up_out, bufs.act_buf, inter_val, inter as u64);
        }

        // Down via Q4_0 f32-input matvec (fixed pipeline, no
        // format-aware routing — Q4_0 vindexes are uniform-format).
        enc.set_compute_pipeline_state(&self.q4.f32_matvec);
        enc.set_buffer(0, Some(bufs.down_w), 0);
        enc.set_buffer(1, Some(bufs.act_buf), 0);
        enc.set_buffer(2, Some(bufs.down_out), 0);
        enc.set_bytes(3, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
        enc.dispatch_threads(MTLSize::new(hidden as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    // ── Shared sub-steps ─────────────────────────────────────────────────────

    fn encode_geglu(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: &FullPipelineLayer,
        bufs: &FfnBufs<'_>,
        inter_val: u32,
        inter_threads: u64,
    ) {
        let geglu = match layer.activation {
            crate::Activation::GeluTanh => &self.geglu_gelu_tanh_pipeline,
            _ => &self.geglu_pipeline,
        };
        enc.set_compute_pipeline_state(geglu);
        enc.set_buffer(0, Some(bufs.gate_out_scratch), 0);
        enc.set_buffer(1, Some(bufs.up_out), 0);
        enc.set_buffer(2, Some(bufs.act_buf), 0);
        enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
        enc.dispatch_threads(MTLSize::new(inter_threads, 1, 1), MTLSize::new(256, 1, 1));
    }

    fn encode_activation(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: &FullPipelineLayer,
        in_buf: &metal::Buffer,
        out_buf: &metal::Buffer,
        inter_val: u32,
        inter_threads: u64,
    ) {
        let pipe = match layer.activation {
            crate::Activation::GeluTanh => &self.gelu_tanh_pipeline,
            _ => &self.silu_pipeline,
        };
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(in_buf), 0);
        enc.set_buffer(1, Some(out_buf), 0);
        enc.set_bytes(2, 4, &inter_val as *const u32 as *const std::ffi::c_void);
        enc.dispatch_threads(MTLSize::new(inter_threads, 1, 1), MTLSize::new(256, 1, 1));
    }

    fn encode_qmv_down(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: &FullPipelineLayer,
        bufs: &FfnBufs<'_>,
        hidden: usize,
        inter: usize,
    ) {
        use crate::metal::stages::quant_matvec::{self as qmv, Pipelines};
        let pipes = Pipelines {
            q4kf_proj: Some(&self.q4kf_proj_pipeline.state),
            q4k_matvec_fallback: &self.q4k_matvec_pipeline.state,
            q6k_matvec: &self.q6k_matvec_pipeline.state,
            q4_matvec: &self.q4.matvec,
        };
        qmv::encode(
            enc, layer.down.format, bufs.down_w,
            bufs.act_buf, 0,
            bufs.act_buf, 0, bufs.act_buf, 0,
            bufs.down_out, 0,
            &pipes,
            hidden, inter,
        );
    }
}
