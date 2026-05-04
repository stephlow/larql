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
pub use decode::profile::take_last_split_timings;
pub use moe_dispatch::MoeScratch;
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
    pub kv_attend_long_pipeline: ComputePipelineState,
    pub kv_append_pipeline: ComputePipelineState,
    /// Fused KV-append + KV-attention. Each Q-head TG cooperatively
    /// writes its kv_head's new K/V row to cache at position `pos`,
    /// then proceeds with attention over T = pos + 1. Eliminates the
    /// `kv_cache_append` dispatch (~1 dispatch/layer × 34 ≈ 0.24 ms/tok).
    /// Default-on; opt out via `LARQL_FUSED_KV_APPEND_ATTEND=0`. See
    /// `shaders/kv_append_attend_fused.rs`.
    pub kv_append_attend_fused_pipeline: ComputePipelineState,
    /// Fused **QK-norm + RoPE + KV-cache append + attention** —
    /// collapses three dispatches (qk_norm_rope_fused +
    /// kv_append_attend_fused, plus the implicit kv_append phase) into
    /// one. Each Q-head TG normalises+ropes its Q (kept in TG memory),
    /// normalises+ropes+writes its kv_head's K row to cache, streams V
    /// to cache, then attends. Saves 1 dispatch/layer × 34 ≈ 0.2 ms/tok.
    /// Default-on; opt out via `LARQL_FUSED_ATTN=0`. See
    /// `shaders/attn_fused.rs`.
    pub attn_fused_pipeline: ComputePipelineState,
    pub q8_matvec_pipeline: KernelHandle,
    pub rms_norm_pipeline: ComputePipelineState,
    pub residual_add_pipeline: ComputePipelineState,
    pub q8_qkv_proj_pipeline: KernelHandle,
    /// Production-active Q4_K matvec pipeline. Holds 8sg by default
    /// (2026-04-28; profiler showed 55% of LPDDR5X peak with 4sg).
    /// All dispatch sites use this transparently. Tests reach the
    /// explicit variants via `q4k_matvec_4sg_pipeline` /
    /// `q4k_matvec_8sg_pipeline`.
    pub q4k_matvec_pipeline: KernelHandle,
    /// Always-4sg Q4_K matvec (production until 2026-04-28). Kept as
    /// the explicit fallback / opt-out via `LARQL_Q4K_MATVEC_8SG=0`.
    pub q4k_matvec_4sg_pipeline: KernelHandle,
    /// Always-8sg Q4_K matvec (256 threads/TG, 8 rows/TG). Bit-identical
    /// output to 4sg. Default-on for `q4k_matvec_pipeline`.
    pub q4k_matvec_8sg_pipeline: KernelHandle,
    /// Stride-32 Q4_K matvec — same Q4_K input format as `q4k_matvec`
    /// but each lane accumulates the contribution of every element
    /// `i where i % 32 == lane`, mirroring `f16_gemv`'s reduction tree.
    /// Use this for the LM head when the regular `q4k_matvec`'s
    /// block-aware lane split (`ix = lane & 1u`) drifts enough vs CPU
    /// to flip top-1 on close-call tokens. See
    /// `shaders/q4k_matvec_stride32.rs` for the rationale.
    pub q4k_matvec_stride32_pipeline: KernelHandle,
    /// Q4_K matmul (gemm) — `[N, K] × [M, K] → [M, N]`. Used by prefill
    /// and seq>1 dispatch when amortising dequant across positions is
    /// worth the per-thread accumulator footprint. Decode (M=1) still
    /// routes through `q4k_matvec_pipeline` for minimal register pressure.
    pub q4k_matmul_pipeline: KernelHandle,
    pub q4k_ffn_gate_up_pipeline: KernelHandle,
    /// Experimental Q4_K gate+up with f16 inner accumulators — opt-in
    /// via `LARQL_F16_ACC=1` while precision is being validated.
    /// Hypothesis: 2× f16 FMA throughput on Apple GPUs frees ALU cycles
    /// even on bandwidth-bound kernels. See
    /// `shaders/q4k_ffn_gate_up_f16acc.rs`.
    pub q4k_ffn_gate_up_f16acc_pipeline: KernelHandle,
    /// Experimental Q4_K gate+up with 8 simdgroups per TG (256 threads,
    /// 8 rows/TG) instead of the production 4 simdgroups (128 threads,
    /// 4 rows/TG). Same per-thread register footprint (nr0=1) so no
    /// register pressure regression; doubled threads per TG should
    /// improve within-TG latency hiding. Off by default; opt-in via
    /// `LARQL_GATE_UP_8SG=1` while perf is being measured. See
    /// `shaders/q4k_ffn_gate_up_8sg.rs`.
    pub q4k_ffn_gate_up_8sg_pipeline: KernelHandle,
    /// Cooperative-scale-load Q4_K gate+up — same Q4_K input as
    /// `q4k_ffn_gate_up_pipeline`, but the per-super-block dequant
    /// header (`d`/`dmin`/8 sub-block scales/mins) is decoded once
    /// per simdgroup per super-block and broadcast via
    /// `simd_broadcast`/`simd_shuffle`, eliminating 32× redundant
    /// ALU on the production critical path. Aimed at the
    /// 187 GB/s = 47%-of-peak ALU bottleneck flagged in
    /// `metal/diag/kernel_profile.rs`. Opt-in via
    /// `LARQL_GATE_UP_COOP=1` while perf is being measured. See
    /// `shaders/q4k_ffn_gate_up_coop.rs`.
    pub q4k_ffn_gate_up_coop_pipeline: KernelHandle,
    /// NR0=2 multi-row + shared-X-vector Q4_K gate+up — same Q4_K
    /// input as `q4k_ffn_gate_up_pipeline`, but each simdgroup handles
    /// 2 output rows in parallel with `xl[16]` loaded once and reused
    /// across both. Mirrors llama.cpp's `N_R0_Q4_K = 2` shape. Aimed
    /// at the X-cache-traffic bottleneck diagnosed by step-by-step
    /// vs-ollama comparison (2026-05-01). Opt-in via
    /// `LARQL_GATE_UP_NR2=1`. See `shaders/q4k_ffn_gate_up_nr2.rs`.
    pub q4k_ffn_gate_up_nr2_pipeline: KernelHandle,
    pub q4kf_ffn_gate_up_pipeline: KernelHandle,
    pub q4k_geglu_silu_down_pipeline: KernelHandle,
    pub q4k_geglu_gelu_tanh_down_pipeline: KernelHandle,
    /// Fused GEGLU activation + Q6_K down projection — production
    /// FFN path on Gemma 3/4 / Llama 2 / Mistral (Ollama convention
    /// is Q4_K gate/up + Q6_K down). Mirrors the Q4_K twins above.
    pub q6k_geglu_silu_down_pipeline: KernelHandle,
    pub q6k_geglu_gelu_tanh_down_pipeline: KernelHandle,
    /// Cached-activation Q6_K GELU-tanh + down — TG memory holds
    /// `tg_act[256]` (one fully-activated element per super-block
    /// position) so the inner FMA loop reads pre-computed activations
    /// instead of recomputing `tanh()` per row. Eliminates the 4×
    /// `tanh()` redundancy that made the original
    /// `q6k_geglu_gelu_tanh_down` regress on Gemma 3 4B (per the
    /// 2026-04-26 finding documented in `encode_ffn.rs`). Saves
    /// 1 dispatch per layer × 34 = ~34/tok plus the redundant
    /// activation compute. Opt-in via `LARQL_FUSED_Q6K_DOWN=1`. See
    /// `shaders/q6k_geglu_gelu_tanh_down_cached.rs`.
    pub q6k_geglu_gelu_tanh_down_cached_pipeline: KernelHandle,
    /// Production-active Q6_K matvec pipeline. Holds 8sg by default,
    /// 4sg when `LARQL_Q6K_8SG=0` is set at startup. All dispatch
    /// sites use this transparently; tests reach the explicit
    /// variants via `q6k_matvec_4sg_pipeline` / `q6k_matvec_8sg_pipeline`.
    pub q6k_matvec_pipeline: KernelHandle,
    /// Always-4sg Q6_K matvec (production until 2026-04-28). Kept as
    /// the explicit fallback / opt-out via `LARQL_Q6K_8SG=0`.
    pub q6k_matvec_4sg_pipeline: KernelHandle,
    /// Always-8sg Q6_K matvec (256 threads/TG, 8 rows/TG). Bit-identical
    /// output to 4sg (same math, only TG dispatch geometry changed).
    /// Default-on for `q6k_matvec_pipeline` as of 2026-04-28. See
    /// `shaders/q6k_matvec_8sg.rs`.
    pub q6k_matvec_8sg_pipeline: KernelHandle,
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
    /// Fused QK-norm + RoPE — replaces the consecutive
    /// `qk_norm_qk` + `rope_at_pos_batched_qk` dispatches with one
    /// kernel: each TG handles one head, RMS-norms it, applies
    /// per-d weight scale, then in-place RoPE. Saves 1 dispatch per
    /// layer × 34 = ~34/tok. Opt-in via `LARQL_FUSED_QK_NORM_ROPE=1`.
    /// See `shaders/qk_norm_rope_fused.rs`.
    pub qk_norm_rope_fused_pipeline: ComputePipelineState,
    /// Triple-fusion: post_attn_norm + residual + ffn_norm + h_post_attn
    /// store. Replaces the 3-dispatch chain (rms_norm + residual_norm +
    /// residual_add) for the `has_post_norms` decode path with a
    /// single kernel doing two sequential RMS reductions and one
    /// fused residual+norm+store. Saves ~34 dispatches/tok.
    /// Opt-in via `LARQL_FUSED_POST_ATTN_NORM=1`.
    /// See `shaders/post_attn_residual_norm_store.rs`.
    pub post_attn_residual_norm_store_pipeline: ComputePipelineState,
    /// Fused post-FFN norm + residual_add. Replaces the consecutive
    /// `rms_norm(down_out)` + `residual_add(h_post_attn, normed_ffn)`
    /// dispatches at the end of each layer in the
    /// `has_post_norms + post_ffn_norm` decode path. Saves
    /// 1 dispatch / layer × 34 ≈ 0.24 ms/tok.
    /// Opt-in via `LARQL_FUSED_POST_FFN_NORM=1`.
    /// See `shaders/post_ffn_norm_residual_add.rs`.
    pub post_ffn_norm_residual_add_pipeline: ComputePipelineState,
    pub rope_at_pos_batched_qk_pipeline: ComputePipelineState,
    // Scale vector (per-layer scalar, Gemma 4)
    pub scale_vector_pipeline: ComputePipelineState,
    /// KV cache for decode mode — initialized on first decode_token call.
    kv_cache: std::sync::Mutex<Option<ops::kv_cache::KVCache>>,
    /// Pre-allocated MoE scratch for `decode_token_q4k_moe` — keyed
    /// by `(top_k, hidden, intermediate_size)`. Reused across decode
    /// calls so the ~15 buffer allocations (~120ms on Gemma 4 26B-A4B,
    /// M3 Max) only happen at first use, not per token. Mirrors the
    /// shape cache `larql-server` keeps in `state.rs::moe_scratches`,
    /// pulled inside the backend so the local decode path benefits
    /// without each caller threading a cache through.
    moe_scratch: std::sync::Mutex<Option<moe_dispatch::MoeScratch>>,
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
        // Q4_K matvec: production default is 8sg (256 threads/TG, 8
        // rows/TG) as of 2026-04-28 — production-batched profiler
        // showed q4k_matvec at 220 GB/s = 55% of LPDDR5X peak, the
        // most-under-utilised matvec by far. 8sg gives access to the
        // remaining bandwidth slack the same way it did for gate+up.
        // Set `LARQL_Q4K_MATVEC_8SG=0` at startup to opt out.
        let q4k_matvec_4sg_pipeline =
            KernelHandle::from_kernel::<shaders::q4k_matvec::Kernel>(&device, &library)?;
        let q4k_matvec_8sg_pipeline =
            KernelHandle::from_kernel::<shaders::q4k_matvec_8sg::Kernel>(&device, &library)?;
        let q4k_matvec_stride32_pipeline =
            KernelHandle::from_kernel::<shaders::q4k_matvec_stride32::Kernel>(&device, &library)?;
        let q4k_matvec_use_4sg = matches!(
            std::env::var("LARQL_Q4K_MATVEC_8SG").as_deref(),
            Ok("0") | Ok("false") | Ok("off") | Ok("no")
        );
        let q4k_matvec_pipeline = if q4k_matvec_use_4sg {
            q4k_matvec_4sg_pipeline.clone()
        } else {
            q4k_matvec_8sg_pipeline.clone()
        };
        let q4k_matmul_pipeline =
            KernelHandle::from_kernel::<shaders::q4k_matmul::Kernel>(&device, &library)?;
        // Q6_K matvec: production default is the 4-simdgroup variant.
        // Tried 8sg (256 threads/TG, 8 rows/TG, kernel-isolated 1.96×
        // speedup) on 2026-04-28 — end-to-end was at parity, slightly
        // worse on quiet GPU (77.6 → 77.1 tok/s, 0.08 ms/tok). q6k was
        // already at 84% of LPDDR5X peak (vs gate+up's 68%), so the
        // ALU/scheduling slack the 8sg variant exposes is too small
        // to recover end-to-end. Both pipelines are kept — tests use
        // them explicitly, opt-IN via `LARQL_Q6K_8SG=1` for callers
        // who want to retry on different hardware.
        let q6k_matvec_4sg_pipeline =
            KernelHandle::from_kernel::<shaders::q6k_matvec::Kernel>(&device, &library)?;
        let q6k_matvec_8sg_pipeline =
            KernelHandle::from_kernel::<shaders::q6k_matvec_8sg::Kernel>(&device, &library)?;
        let q6k_use_8sg = matches!(
            std::env::var("LARQL_Q6K_8SG").as_deref(),
            Ok("1") | Ok("true") | Ok("on") | Ok("yes")
        );
        let q6k_matvec_pipeline = if q6k_use_8sg {
            q6k_matvec_8sg_pipeline.clone()
        } else {
            q6k_matvec_4sg_pipeline.clone()
        };

        // Fused Q4_K / Q4_KF FFN gate+up (KernelHandle).
        let q4k_ffn_gate_up_pipeline =
            KernelHandle::from_kernel::<shaders::q4k_ffn_gate_up::Kernel>(&device, &library)?;
        let q4k_ffn_gate_up_f16acc_pipeline = KernelHandle::from_kernel::<
            shaders::q4k_ffn_gate_up_f16acc::Kernel,
        >(&device, &library)?;
        let q4k_ffn_gate_up_8sg_pipeline =
            KernelHandle::from_kernel::<shaders::q4k_ffn_gate_up_8sg::Kernel>(&device, &library)?;
        let q4k_ffn_gate_up_coop_pipeline =
            KernelHandle::from_kernel::<shaders::q4k_ffn_gate_up_coop::Kernel>(&device, &library)?;
        let q4k_ffn_gate_up_nr2_pipeline =
            KernelHandle::from_kernel::<shaders::q4k_ffn_gate_up_nr2::Kernel>(&device, &library)?;
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
        let q6k_geglu_gelu_tanh_down_cached_pipeline = KernelHandle::from_kernel::<
            shaders::q6k_geglu_gelu_tanh_down_cached::Kernel,
        >(&device, &library)?;
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
        let qk_norm_rope_fused_pipeline =
            get_shader_pipeline::<shaders::qk_norm_rope_fused::Kernel>(&device, &library)?;
        let post_attn_residual_norm_store_pipeline = get_shader_pipeline::<
            shaders::post_attn_residual_norm_store::Kernel,
        >(&device, &library)?;
        let post_ffn_norm_residual_add_pipeline =
            get_shader_pipeline::<shaders::post_ffn_norm_residual_add::Kernel>(&device, &library)?;
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
        let kv_attend_long_pipeline =
            get_shader_pipeline::<shaders::kv_attention::AttendLongKernel>(&device, &library)?;
        let kv_append_pipeline =
            get_shader_pipeline::<shaders::kv_attention::AppendKernel>(&device, &library)?;
        let kv_append_attend_fused_pipeline =
            get_shader_pipeline::<shaders::kv_append_attend_fused::Kernel>(&device, &library)?;
        let attn_fused_pipeline =
            get_shader_pipeline::<shaders::attn_fused::Kernel>(&device, &library)?;

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
            kv_attend_long_pipeline,
            kv_append_pipeline,
            kv_append_attend_fused_pipeline,
            attn_fused_pipeline,
            q8_matvec_pipeline,
            rms_norm_pipeline,
            residual_add_pipeline,
            q8_qkv_proj_pipeline,
            q4k_matvec_pipeline,
            q4k_matvec_4sg_pipeline,
            q4k_matvec_8sg_pipeline,
            q4k_matvec_stride32_pipeline,
            q4k_matmul_pipeline,
            q4k_ffn_gate_up_pipeline,
            q4k_ffn_gate_up_f16acc_pipeline,
            q4k_ffn_gate_up_8sg_pipeline,
            q4k_ffn_gate_up_coop_pipeline,
            q4k_ffn_gate_up_nr2_pipeline,
            q4kf_ffn_gate_up_pipeline,
            q4k_geglu_silu_down_pipeline,
            q4k_geglu_gelu_tanh_down_pipeline,
            q6k_geglu_silu_down_pipeline,
            q6k_geglu_gelu_tanh_down_pipeline,
            q6k_geglu_gelu_tanh_down_cached_pipeline,
            q6k_matvec_pipeline,
            q6k_matvec_4sg_pipeline,
            q6k_matvec_8sg_pipeline,
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
            qk_norm_rope_fused_pipeline,
            post_attn_residual_norm_store_pipeline,
            post_ffn_norm_residual_add_pipeline,
            rope_at_pos_batched_qk_pipeline,
            scale_vector_pipeline,
            kv_cache: std::sync::Mutex::new(None),
            moe_scratch: std::sync::Mutex::new(None),
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
        let shapes = vec![(num_kv_heads, head_dim); num_layers];
        self.ensure_kv_cache_for_shapes(&mut guard, &shapes, decode::DEFAULT_KV_CACHE_MAX_SEQ);
        guard
    }

    /// Access the KV cache using per-layer pipeline geometry.
    ///
    /// This is the preferred path for heterogeneous attention layouts; it
    /// avoids the legacy uniform `(num_kv_heads, head_dim)` fallback.
    pub fn kv_cache_mut_for_layers(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
    ) -> std::sync::MutexGuard<'_, Option<ops::kv_cache::KVCache>> {
        let mut guard = self.kv_cache.lock().unwrap();
        self.ensure_kv_cache_for_layers(&mut guard, layers, decode::DEFAULT_KV_CACHE_MAX_SEQ);
        guard
    }

    /// Access the KV cache using explicit per-layer geometry.
    ///
    /// Use this when call sites pass absolute layer indices and only hold a
    /// slice of pipeline layers locally.
    pub fn kv_cache_mut_for_shapes(
        &self,
        shapes: &[(usize, usize)],
    ) -> std::sync::MutexGuard<'_, Option<ops::kv_cache::KVCache>> {
        let mut guard = self.kv_cache.lock().unwrap();
        self.ensure_kv_cache_for_shapes(&mut guard, shapes, decode::DEFAULT_KV_CACHE_MAX_SEQ);
        guard
    }
}
