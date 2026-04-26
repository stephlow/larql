//! Metal GPU compute backend — Apple Silicon.
//!
//! All operations go through the [`ComputeBackend`] trait. Metal-specific
//! optimisations: simdgroup reductions, uint4 vectorised loads, native half reads,
//! zero-copy mmap buffers, single command buffer pipeline.
//!
//! ## Modules
//!
//! - `shaders/`:  Metal Shading Language — one file per kernel (~30 shader files)
//! - `ops/`:      GPU dispatch — one file per operation (6 dispatchers)
//! - `buffers`:   GPU buffer cache (zero-copy mmap, transient allocation)
//! - `f32_ops`:   f32 tiled matmul dispatch with GPU/CPU routing
//! - `calibrate`: CPU vs GPU auto-calibration
//!
//! ## Performance (M3 Max, Gemma 3 4B, 34 layers)
//!
//! - Full decode: ~0.38ms/layer, ~77 tok/s (Q4_KF path)
//! - vs Ollama: ~1.0–1.25× (at parity)
//! - Q4_K matvec: uint4 loads, 8 rows/TG, multi-row (nr0=2)
//! - KV attention: simd_max/simd_sum reductions, float4 Q·K dot products

pub mod buffers;
pub mod calibrate;
mod decode;
mod decode_hybrid;
/// Diagnostic and profiling tools — kernel bandwidth, decode-stage timing,
/// layer-level residual dumps. See `diag/mod.rs` for the full index.
pub mod diag;
mod direct_ops;
pub mod f32_ops;
pub mod kernel; // KernelHandle: pipeline + dispatch geometry, bundled
mod moe_dispatch;
pub mod ops; // modular: ops/mod.rs → one file per operation
mod pipeline;
mod prefill;
pub mod shaders; // modular: shaders/mod.rs → one file per shader
pub mod stages; // modular: stages/mod.rs → one file per pipeline stage
mod trait_impl;

use metal::*;
use std::sync::atomic::{AtomicUsize, Ordering};

use buffers::BufferCache;
use f32_ops::F32Ops;
use kernel::KernelHandle;
use ops::q4_common::Q4Pipelines;

/// Metal GPU compute backend.
///
/// ## Pipeline field convention
///
/// Fields fall into two camps:
///
/// - **`KernelHandle`** — simdgroup-tiled kernels with hard-coded row
///   maps (`row_idx = tg_id * ROWS_PER_TG + sg_id`). Geometry travels
///   with the pipeline; dispatchers read `kernel.rows_per_tg` /
///   `kernel.threads_per_tg` rather than importing constants from a
///   shader module. This is the bug class the q4_matvec_v4 75 %-row
///   drop introduced (see ROADMAP ship log).
///
/// - **`ComputePipelineState`** — flat `dispatch_threads` kernels
///   (one thread per output element / row) or attention-shape
///   kernels (per-head dispatch). No row-map drift risk because the
///   dispatcher already specifies the geometry per call.
///
/// Twelve simdgroup-tiled fields use `KernelHandle`. The rest stay
/// bare. Decision per remaining field:
/// - `geglu_*`, `silu`, `gelu_tanh`, `residual_add`, `scale_vector` →
///   element-wise, flat dispatch.
/// - `rms_norm*`, `layer_norm*`, `v_norm*`, `qk_norm`, `residual_norm*`
///   → per-row reduction, flat dispatch (one threadgroup per row).
/// - `causal_attn`, `fused_attn`, `kv_attend`, `kv_append` → attention
///   geometry (per-head/per-position), not row-tiled.
/// - `rope_*`, `q8_quant` → flat dispatch_threads.
pub struct MetalBackend {
    queue: CommandQueue,
    bufs: BufferCache,
    f32_ops: F32Ops,
    pub q4: Q4Pipelines,
    causal_attn_pipeline: ComputePipelineState,
    pub fused_attn_pipeline: ComputePipelineState,
    pub geglu_pipeline: ComputePipelineState,
    pub geglu_gelu_tanh_pipeline: ComputePipelineState,
    q8_quant_pipeline: ComputePipelineState,
    pub kv_attend_pipeline: ComputePipelineState,
    pub kv_append_pipeline: ComputePipelineState,
    pub q8_matvec_pipeline: KernelHandle,
    pub rms_norm_pipeline: ComputePipelineState,
    pub residual_add_pipeline: ComputePipelineState,
    pub q8_qkv_proj_pipeline: KernelHandle,
    pub q4k_matvec_pipeline: KernelHandle,
    pub q4k_ffn_gate_up_pipeline: KernelHandle,
    pub q4kf_ffn_gate_up_pipeline: KernelHandle,
    pub q4k_geglu_silu_down_pipeline: KernelHandle,
    pub q4k_geglu_gelu_tanh_down_pipeline: KernelHandle,
    /// Fused GEGLU activation + Q6_K down projection — production
    /// FFN path on Gemma 3/4 / Llama 2 / Mistral (Ollama convention
    /// is Q4_K gate/up + Q6_K down). Mirrors the Q4_K twins above.
    pub q6k_geglu_silu_down_pipeline: KernelHandle,
    pub q6k_geglu_gelu_tanh_down_pipeline: KernelHandle,
    pub q6k_matvec_pipeline: KernelHandle,
    pub rope_at_pos_pipeline: ComputePipelineState,
    pub rope_at_pos_batched_pipeline: ComputePipelineState,
    pub q4k_qkv_proj_pipeline: KernelHandle,
    /// Fused mixed-quant QKV: Q4_K Q/K rows + Q6_K V rows in one dispatch.
    /// Gemma 3 4B / Gemma 4 ship `V` as Q6_K; without this shader decode
    /// falls through to three per-projection dispatches per layer.
    pub q4k_q6k_qkv_proj_pipeline: KernelHandle,
    pub q4k_q6k_qkv_proj_normed_pipeline: KernelHandle,
    pub q4k_proj_pipeline: KernelHandle,
    pub q4kf_qkv_proj_pipeline: KernelHandle,
    pub q4kf_proj_pipeline: KernelHandle,
    // Standalone activations (non-gated FFN)
    pub silu_pipeline: ComputePipelineState,
    pub gelu_tanh_pipeline: ComputePipelineState,
    // LayerNorm (StarCoder2)
    pub layer_norm_pipeline: ComputePipelineState,
    pub layer_norm_no_bias_pipeline: ComputePipelineState,
    // V-norm (Gemma 4)
    pub v_norm_pipeline: ComputePipelineState,
    pub v_norm_batched_pipeline: ComputePipelineState,
    pub qk_norm_pipeline: ComputePipelineState,
    pub qk_norm_qk_pipeline: ComputePipelineState,
    pub rope_at_pos_batched_qk_pipeline: ComputePipelineState,
    // Scale vector (per-layer scalar, Gemma 4)
    pub scale_vector_pipeline: ComputePipelineState,
    /// KV cache for decode mode — initialized on first decode_token call.
    kv_cache: std::sync::Mutex<Option<ops::kv_cache::KVCache>>,
    pub rms_norm_q8_pipeline: ComputePipelineState,
    pub residual_norm_pipeline: ComputePipelineState,
    pub residual_norm_q8_pipeline: ComputePipelineState,
    pub residual_norm_store_pipeline: ComputePipelineState,
    /// Dedicated row-per-simdgroup f32 gemv for the LM head. Used in
    /// autoregressive decode where `matmul_transb(query, lm_head)` shows
    /// up as the dominant per-token cost.
    pub f32_gemv_pipeline: KernelHandle,
    pub f32_argmax_partial_pipeline: ComputePipelineState,
    /// Per-TG top-K reduction over a scores buffer. Produces `K_TOPK = 8`
    /// (val, idx) pairs per TG; CPU final reduction merges into the caller's
    /// requested top-k. Used by the lm_head top_k=5 path on Gemma 3/4.
    pub f32_topk_partial_pipeline: ComputePipelineState,
    /// Same layout as [`Self::f32_gemv_pipeline`], but with a `half`
    /// weight matrix. Halves bandwidth for tied-embedding models whose
    /// lm_head would otherwise live as a 5.6 GB f32 clone on 31B.
    pub f16_gemv_pipeline: KernelHandle,
    flop_threshold: AtomicUsize,
}

impl MetalBackend {
    /// Create a Metal backend. Returns None if no Metal device is available.
    pub fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let queue = device.new_command_queue();

        let opts = CompileOptions::new();
        let all_src = shaders::all_shaders();
        let library = device
            .new_library_with_source(&all_src, &opts)
            .map_err(|e| eprintln!("[metal] shader compile error: {e}"))
            .ok()?;

        use kernel::get_shader_pipeline;

        let f32_ops = F32Ops {
            sgemm_pipeline: get_shader_pipeline::<shaders::sgemm::Kernel>(&device, &library)?,
            transb_pipeline: get_shader_pipeline::<shaders::sgemm_transb::Kernel>(
                &device, &library,
            )?,
        };

        let causal_attn_pipeline =
            get_shader_pipeline::<shaders::causal_attention::Kernel>(&device, &library)?;

        // Q4 family pipelines.
        //
        // `matvec` is simdgroup-tiled. Its kernel name + row map +
        // threads-per-TG live in `shaders/q4_matvec_v4.rs` via the
        // `TiledKernel` impl on the `Kernel` marker; binding it here
        // is one type-parameter line. To swap to a future v6, change
        // `q4_matvec_v4::Kernel` → `q4_matvec_v6::Kernel` here and
        // nothing else. See `metal::kernel` and the q4_matvec_v4
        // 75 %-row-drop ship-log entry.
        //
        // `vecmat` and `f32_matvec` use flat `dispatch_threads` — no
        // per-TG geometry, bare pipeline state is enough.
        let q4 = Q4Pipelines {
            matvec: KernelHandle::from_kernel::<shaders::q4_matvec_v4::Kernel>(&device, &library)?,
            vecmat: get_shader_pipeline::<shaders::q4_vecmat::Kernel>(&device, &library)?,
            f32_matvec: get_shader_pipeline::<shaders::q4_f32_matvec::Kernel>(&device, &library)?,
        };

        let bufs = BufferCache::new(&device);

        let geglu_pipeline = get_shader_pipeline::<shaders::geglu::SiluKernel>(&device, &library)?;
        let geglu_gelu_tanh_pipeline =
            get_shader_pipeline::<shaders::geglu::GeluTanhKernel>(&device, &library)?;
        let q8_quant_pipeline =
            get_shader_pipeline::<shaders::quantize_q8::Kernel>(&device, &library)?;

        // Q8 matvec for attention projections (KernelHandle — geometry travels with kernel).
        let q8_matvec_pipeline =
            KernelHandle::from_kernel::<shaders::q8_matvec::Kernel>(&device, &library)?;

        // Norm and residual ops
        let rms_norm_pipeline =
            get_shader_pipeline::<shaders::residual_inject::RmsNormKernel>(&device, &library)?;
        let residual_add_pipeline =
            get_shader_pipeline::<shaders::residual_inject::ResidualAddKernel>(&device, &library)?;

        // Q4_K + Q6_K matvec (KernelHandle).
        let q4k_matvec_pipeline =
            KernelHandle::from_kernel::<shaders::q4k_matvec::Kernel>(&device, &library)?;
        let q6k_matvec_pipeline =
            KernelHandle::from_kernel::<shaders::q6k_matvec::Kernel>(&device, &library)?;

        // Fused Q4_K / Q4_KF FFN gate+up (KernelHandle).
        let q4k_ffn_gate_up_pipeline =
            KernelHandle::from_kernel::<shaders::q4k_ffn_gate_up::Kernel>(&device, &library)?;
        let q4kf_ffn_gate_up_pipeline =
            KernelHandle::from_kernel::<shaders::q4kf_ffn_gate_up::Kernel>(&device, &library)?;
        // Fused activation+down (KernelHandle).
        let q4k_geglu_silu_down_pipeline =
            KernelHandle::from_kernel::<shaders::q4k_geglu_down::SiluKernel>(&device, &library)?;
        let q4k_geglu_gelu_tanh_down_pipeline = KernelHandle::from_kernel::<
            shaders::q4k_geglu_down::GeluTanhKernel,
        >(&device, &library)?;
        let q6k_geglu_silu_down_pipeline =
            KernelHandle::from_kernel::<shaders::q6k_geglu_down::SiluKernel>(&device, &library)?;
        let q6k_geglu_gelu_tanh_down_pipeline = KernelHandle::from_kernel::<
            shaders::q6k_geglu_down::GeluTanhKernel,
        >(&device, &library)?;

        // Fused Q8 QKV projection (KernelHandle).
        let q8_qkv_proj_pipeline =
            KernelHandle::from_kernel::<shaders::q8_attn_proj::QkvKernel>(&device, &library)?;

        // Fused ops (norm+quantize, residual+norm, residual+norm+quantize)
        let rms_norm_q8_pipeline =
            get_shader_pipeline::<shaders::fused_ops::RmsNormQ8Kernel>(&device, &library)?;
        let residual_norm_pipeline =
            get_shader_pipeline::<shaders::fused_ops::ResidualNormKernel>(&device, &library)?;
        let residual_norm_q8_pipeline =
            get_shader_pipeline::<shaders::fused_ops::ResidualNormQ8Kernel>(&device, &library)?;
        let residual_norm_store_pipeline =
            get_shader_pipeline::<shaders::fused_ops::ResidualNormStoreKernel>(&device, &library)?;

        // Dedicated f32 / f16 gemv for the LM head (KernelHandle).
        let f32_gemv_pipeline =
            KernelHandle::from_kernel::<shaders::f32_gemv::Kernel>(&device, &library)?;
        let f32_argmax_partial_pipeline =
            get_shader_pipeline::<shaders::f32_gemv::ArgmaxKernel>(&device, &library)?;
        let f32_topk_partial_pipeline =
            get_shader_pipeline::<shaders::f32_gemv::TopKKernel>(&device, &library)?;
        let f16_gemv_pipeline =
            KernelHandle::from_kernel::<shaders::f16_gemv::Kernel>(&device, &library)?;

        // RoPE at position (for KV-cached decode)
        let rope_at_pos_pipeline =
            get_shader_pipeline::<shaders::rope::RopeAtPosKernel>(&device, &library)?;
        let rope_at_pos_batched_pipeline =
            get_shader_pipeline::<shaders::rope::RopeAtPosBatchedKernel>(&device, &library)?;

        // Fused Q4_K QKV projection (KernelHandle).
        let q4k_qkv_proj_pipeline =
            KernelHandle::from_kernel::<shaders::q4k_qkv_proj::QkvKernel>(&device, &library)?;
        let q4k_q6k_qkv_proj_pipeline =
            KernelHandle::from_kernel::<shaders::q4k_q6k_qkv_proj::Kernel>(&device, &library)?;
        let q4k_q6k_qkv_proj_normed_pipeline = KernelHandle::from_kernel::<
            shaders::q4k_q6k_qkv_proj::NormedKernel,
        >(&device, &library)?;
        let q4k_proj_pipeline =
            KernelHandle::from_kernel::<shaders::q4k_qkv_proj::ProjKernel>(&device, &library)?;

        // Q4_KF: pre-baked scales (faster inference) — KernelHandle.
        let q4kf_qkv_proj_pipeline =
            KernelHandle::from_kernel::<shaders::q4kf_qkv_proj::QkvKernel>(&device, &library)?;
        let q4kf_proj_pipeline =
            KernelHandle::from_kernel::<shaders::q4kf_qkv_proj::ProjKernel>(&device, &library)?;

        // Fused attention (RoPE + GQA + softcap)
        let fused_attn_pipeline =
            get_shader_pipeline::<shaders::fused_attention::Kernel>(&device, &library)?;

        // Standalone activations (non-gated FFN)
        let silu_pipeline =
            get_shader_pipeline::<shaders::activation::SiluKernel>(&device, &library)?;
        let gelu_tanh_pipeline =
            get_shader_pipeline::<shaders::activation::GeluTanhKernel>(&device, &library)?;

        // LayerNorm (StarCoder2, GPT-2)
        let layer_norm_pipeline =
            get_shader_pipeline::<shaders::layer_norm::Kernel>(&device, &library)?;
        let layer_norm_no_bias_pipeline =
            get_shader_pipeline::<shaders::layer_norm::NoBiasKernel>(&device, &library)?;

        // V-norm (parameter-free RMSNorm, Gemma 4)
        let v_norm_pipeline = get_shader_pipeline::<shaders::v_norm::Kernel>(&device, &library)?;
        let v_norm_batched_pipeline =
            get_shader_pipeline::<shaders::v_norm::BatchedKernel>(&device, &library)?;

        // QK-norm (learned-weight per-head RMSNorm, Gemma 3/4)
        let qk_norm_pipeline = get_shader_pipeline::<shaders::qk_norm::Kernel>(&device, &library)?;
        let qk_norm_qk_pipeline =
            get_shader_pipeline::<shaders::qk_norm::QkKernel>(&device, &library)?;
        let rope_at_pos_batched_qk_pipeline =
            get_shader_pipeline::<shaders::rope::RopeAtPosBatchedQkKernel>(&device, &library)?;

        // Scale vector (per-layer scalar multiplier, Gemma 4)
        let scale_vector_pipeline =
            get_shader_pipeline::<shaders::residual_inject::ScaleVectorKernel>(&device, &library)?;

        // KV cache attention
        let kv_attend_pipeline =
            get_shader_pipeline::<shaders::kv_attention::AttendKernel>(&device, &library)?;
        let kv_append_pipeline =
            get_shader_pipeline::<shaders::kv_attention::AppendKernel>(&device, &library)?;

        Some(Self {
            queue,
            bufs,
            f32_ops,
            q4,
            causal_attn_pipeline,
            fused_attn_pipeline,
            geglu_pipeline,
            geglu_gelu_tanh_pipeline,
            q8_quant_pipeline,
            kv_attend_pipeline,
            kv_append_pipeline,
            q8_matvec_pipeline,
            rms_norm_pipeline,
            residual_add_pipeline,
            q8_qkv_proj_pipeline,
            q4k_matvec_pipeline,
            q4k_ffn_gate_up_pipeline,
            q4kf_ffn_gate_up_pipeline,
            q4k_geglu_silu_down_pipeline,
            q4k_geglu_gelu_tanh_down_pipeline,
            q6k_geglu_silu_down_pipeline,
            q6k_geglu_gelu_tanh_down_pipeline,
            q6k_matvec_pipeline,
            rope_at_pos_pipeline,
            rope_at_pos_batched_pipeline,
            q4k_qkv_proj_pipeline,
            q4k_q6k_qkv_proj_pipeline,
            q4k_q6k_qkv_proj_normed_pipeline,
            q4k_proj_pipeline,
            q4kf_qkv_proj_pipeline,
            q4kf_proj_pipeline,
            silu_pipeline,
            gelu_tanh_pipeline,
            layer_norm_pipeline,
            layer_norm_no_bias_pipeline,
            v_norm_pipeline,
            v_norm_batched_pipeline,
            qk_norm_pipeline,
            qk_norm_qk_pipeline,
            rope_at_pos_batched_qk_pipeline,
            scale_vector_pipeline,
            kv_cache: std::sync::Mutex::new(None),
            rms_norm_q8_pipeline,
            residual_norm_pipeline,
            residual_norm_q8_pipeline,
            residual_norm_store_pipeline,
            f32_gemv_pipeline,
            f32_argmax_partial_pipeline,
            f32_topk_partial_pipeline,
            f16_gemv_pipeline,
            flop_threshold: AtomicUsize::new(calibrate::DEFAULT_FLOP_THRESHOLD),
        })
    }

    /// Auto-calibrate CPU vs GPU threshold.
    pub fn calibrate(&self) {
        let threshold = calibrate::calibrate(&self.f32_ops, &self.queue, &self.bufs);
        self.flop_threshold.store(threshold, Ordering::Relaxed);
    }

    pub fn flop_threshold(&self) -> usize {
        self.flop_threshold.load(Ordering::Relaxed)
    }
    pub fn set_flop_threshold(&self, t: usize) {
        self.flop_threshold
            .store(t.max(calibrate::MIN_FLOP_FLOOR), Ordering::Relaxed);
    }
    pub fn cache_size(&self) -> usize {
        self.bufs.len()
    }
    pub fn bufs(&self) -> &BufferCache {
        &self.bufs
    }
    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }

    /// Access the KV cache for hybrid decode (GPU attention + CPU FFN).
    /// Creates the cache on first access.
    pub fn kv_cache_mut(
        &self,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> std::sync::MutexGuard<'_, Option<ops::kv_cache::KVCache>> {
        let mut guard = self.kv_cache.lock().unwrap();
        if guard.is_none() {
            *guard = Some(self.create_kv_cache(num_layers, 4096, num_kv_heads, head_dim));
        }
        guard
    }
}
