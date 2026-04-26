//! Pipeline layer types — per-layer architecture parameters for the compute pipeline.
//!
//! These types carry all model-specific behavior per-layer:
//! norm type, activation, attention geometry, RoPE, FFN type, etc.
//! The compute backends read these fields per-layer — no hardcoded
//! model assumptions in the execution path.

/// Quantization format for a weight tensor.
/// Names match GGUF conventions (Q4_K, Q6_K, etc.).
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_camel_case_types)]
pub enum QuantFormat {
    Q4_0,   // 18 bytes per 32 values (one f16 scale)
    Q4_K,   // 144 bytes per 256 values (GGUF-canonical, Ollama-compatible)
    Q4_KF,  // 160 bytes per 256 values (pre-baked half scales — fast decode)
    Q6_K,   // 210 bytes per 256 values (6-bit with sub-block scales)
    Q8_0,   // int8 values + separate f32 scales
    BF16,   // raw bfloat16 (2 bytes per value, no quantization scales)
    F16,    // raw float16  (2 bytes per value)
    F32,    // raw float32  (4 bytes per value)
}

/// A quantized weight matrix — raw bytes with format tag.
#[derive(Clone, Copy)]
pub struct QuantWeight<'a> {
    pub data: &'a [u8],
    pub scales: Option<&'a [f32]>,  // only for Q8_0 (separate scale array)
    pub format: QuantFormat,
}

/// Norm type for layer normalization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NormType {
    /// RMSNorm — Llama, Gemma, Qwen, most modern models.
    RmsNorm,
    /// Standard LayerNorm (mean-subtraction + variance normalization) — StarCoder2, GPT-2.
    LayerNorm,
}

/// FFN type: gated (gate+up→GEGLU→down) vs standard (up→activation→down).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FfnType {
    /// Gated: SiLU(x @ gate.T) * (x @ up.T) @ down.T — Llama, Gemma, Mistral.
    Gated,
    /// Standard: activation(x @ up.T) @ down.T — StarCoder2, GPT-2.
    Standard,
}

/// Activation function for FFN.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Activation {
    /// SiLU / Swish — Llama, Mistral, Qwen.
    Silu,
    /// GELU with tanh approximation — Gemma, StarCoder2.
    GeluTanh,
}

/// Hybrid MoE (Mixture-of-Experts) weights for one layer.
///
/// Gemma 4 26B A4B runs a dense MLP and an expert block in parallel per layer,
/// summing their outputs. This struct carries the expert-block tensors.
pub struct MoeLayerWeights<'a> {
    /// Expert gate+up weight bytes. Format declared by `expert_data_format`.
    ///
    /// Legacy BF16 layout: [num_experts, 2 * inter, hidden] contiguous.
    /// Per-layer Q4_K layout: NOT used here — per-layer format exposes
    /// individual expert slices via `ModelWeights::get_layer_entry_bytes`.
    /// When `expert_data_format == QuantFormat::Q4_K`, dispatch via
    /// `get_layer_entry_bytes` rather than these fields.
    pub experts_gate_up: &'a [u8],
    /// Expert down weight bytes. See `experts_gate_up` note.
    pub experts_down: &'a [u8],
    /// Format of the expert weight bytes. `Q4_K` = per-layer Q4_K files
    /// (GPU-dispatchable); anything else = legacy BF16 (CPU dequant path).
    #[allow(dead_code)]
    pub expert_data_format: QuantFormat,
    /// Router linear projection weight [num_experts, hidden_size].
    pub router_proj: &'a [f32],
    /// Router learned input-scale [hidden_size].
    pub router_scale: &'a [f32],
    /// Router per-expert output-scale [num_experts].
    pub router_per_expert_scale: &'a [f32],
    /// Router's own RMS-norm weight applied to the router input before projection.
    /// Empty slice → fall back to parameter-free RMSNorm (if the flag below
    /// is set) or to `pre_experts_norm`.
    pub router_norm: &'a [f32],
    /// Parameter-free router RMSNorm: apply `x / sqrt(mean(x²) + eps)` on
    /// the router input when `router_norm` is empty. HF Gemma 4 sets this
    /// true (`Gemma4RMSNorm(with_scale=False)` — no learned weight on disk).
    pub router_norm_parameter_free: bool,
    /// Scalar multiplier on the router input after the norm and `router_scale`.
    /// HF Gemma 4: `hidden_size^-0.5`. Use `1.0` to disable.
    pub router_input_scalar: f32,
    /// Pre-norm applied to the expert matmuls' input (not the router's). [hidden_size].
    pub pre_experts_norm: &'a [f32],
    /// Post-norm for dense FFN output (replaces plain post_ffn_norm). [hidden_size].
    pub post_ffn1_norm: &'a [f32],
    /// Post-norm for expert block output. [hidden_size].
    pub post_experts_norm: &'a [f32],
    /// Total number of routed experts.
    pub num_experts: usize,
    /// Experts activated per token (top-K).
    pub top_k: usize,
    /// Per-expert intermediate (hidden) dimension.
    pub intermediate_size: usize,
    /// Activation function for expert MLPs. Gemma 4 uses GeluTanh; Mixtral/others use Silu.
    pub activation: Activation,
}

/// Per-layer quantized weights for the full pipeline.
///
/// Carries all architecture-specific behavior per-layer — no model
/// type strings or hardcoded constants in the compute path.
/// Supports Q4_K/Q6_K (Ollama strategy) or Q8_0 (higher precision fallback).
pub struct FullPipelineLayer<'a> {
    // ── Attention weights ──
    pub wq: QuantWeight<'a>,
    pub wk: QuantWeight<'a>,
    pub wv: QuantWeight<'a>,
    pub wo: QuantWeight<'a>,

    // ── FFN weights ──
    /// Gate projection (only used when ffn_type == Gated).
    pub gate: QuantWeight<'a>,
    pub up: QuantWeight<'a>,
    pub down: QuantWeight<'a>,

    // ── Norm weights (f32 vectors, hidden_size elements) ──
    pub input_norm: &'a [f32],
    pub post_attn_norm: &'a [f32],
    pub pre_ffn_norm: Option<&'a [f32]>,
    pub post_ffn_norm: Option<&'a [f32]>,
    /// Norm bias (only for LayerNorm). None for RMSNorm.
    pub input_norm_bias: Option<&'a [f32]>,
    pub post_attn_norm_bias: Option<&'a [f32]>,

    // ── Per-layer architecture parameters ──
    /// Norm weight offset: 0.0 (Llama, Gemma 4), 1.0 (Gemma 2/3).
    pub norm_offset: f32,
    /// QK norm weight offset: 0.0 (Llama, Gemma 4), 1.0 (Gemma 2/3).
    pub qk_norm_offset: f32,
    /// RMSNorm epsilon. Default: 1e-6.
    pub eps: f32,
    /// Whether this model uses post-norms (4 norms per layer: Gemma 2/3/4).
    pub has_post_norms: bool,
    /// Norm type: RMSNorm (default) or LayerNorm (StarCoder2).
    pub norm_type: NormType,
    /// FFN type: Gated (default) or Standard (StarCoder2).
    pub ffn_type: FfnType,
    /// Activation function for the FFN.
    pub activation: Activation,
    /// Attention scale for this layer. Default: 1/sqrt(head_dim).
    /// Gemma 4 (with QK-norm): 1.0.
    pub attn_scale: f32,
    /// Head dimension for this layer. Gemma 4: 256 (sliding) or 512 (global).
    pub head_dim: usize,
    /// Number of Q heads for this layer.
    pub num_q_heads: usize,
    /// Number of KV heads for this layer.
    pub num_kv_heads: usize,
    /// RoPE base frequency for this layer. Gemma 3/4: 10k (sliding) or 1M (global).
    pub rope_base: f32,
    /// Dimensions to apply RoPE to. 0 = full head_dim. Gemma 4 global: head_dim * 0.25.
    pub rotary_dim: usize,
    /// Sliding window size. 0 = full attention (no window).
    pub sliding_window: usize,
    /// Whether to apply parameter-free V-norm (Gemma 4).
    pub has_v_norm: bool,
    /// Per-layer scalar multiplier. 0.0 = disabled (no scaling). Gemma 4: learned scalar.
    pub layer_scalar: f32,
    /// QK-norm weight for Q heads (Gemma 3 / Gemma 4). Length = head_dim.
    /// Applied per-head as RMS-norm before RoPE. `None` means skip QK-norm.
    pub q_norm_weight: Option<&'a [f32]>,
    /// QK-norm weight for K heads. Same shape as `q_norm_weight`.
    pub k_norm_weight: Option<&'a [f32]>,
    /// FFN bias on up projection (StarCoder2). None = no bias.
    pub ffn_up_bias: Option<&'a [f32]>,
    /// FFN bias on down projection (StarCoder2). None = no bias.
    pub ffn_down_bias: Option<&'a [f32]>,

    /// Hybrid MoE block (Gemma 4 26B A4B: dense MLP + expert block, outputs summed).
    /// None for all dense models.
    pub moe: Option<MoeLayerWeights<'a>>,

    /// When true, a final RMS norm is applied to the combined (dense + expert)
    /// output before the residual add. Gemma 4 26B A4B: true. Other models:
    /// false (use `layer_scalar` instead).
    pub moe_combined_output_norm: bool,

    /// Outer post-FFN norm weight applied to `(h1 + h2)` before the residual
    /// add. When present and `moe_combined_output_norm` is true, this weight
    /// is used instead of `post_ffn_norm` for the combined norm.
    /// HF Gemma 4: `layers.N.post_feedforward_layernorm.weight` (un-suffixed,
    /// distinct from the `_1` dense-branch norm stored in `post_ffn_norm`).
    /// `None` → fall back to `post_ffn_norm` (legacy behavior).
    pub moe_outer_post_norm: Option<&'a [f32]>,
}

impl<'a> FullPipelineLayer<'a> {
    /// Whether this layer uses gated FFN (gate + up → GEGLU → down).
    pub fn is_gated(&self) -> bool {
        self.ffn_type == FfnType::Gated
    }

    /// Whether this layer has a hybrid MoE block alongside the dense FFN.
    /// When true, the forward pass runs both branches and sums their outputs.
    pub fn is_hybrid_moe(&self) -> bool {
        self.moe.is_some()
    }
}

// ── Backward compatibility: convert old-style bool to new enums ──

impl From<bool> for Activation {
    /// `true` = GeluTanh (Gemma), `false` = Silu (Llama).
    fn from(use_gelu_tanh: bool) -> Self {
        if use_gelu_tanh { Activation::GeluTanh } else { Activation::Silu }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_qw(data: &[u8]) -> QuantWeight<'_> {
        QuantWeight { data, scales: None, format: QuantFormat::Q4_0 }
    }

    fn minimal_layer<'a>(
        data: &'a [u8],
        norms: &'a [f32],
        ffn_type: FfnType,
        moe: Option<MoeLayerWeights<'a>>,
    ) -> FullPipelineLayer<'a> {
        let qw = minimal_qw(data);
        FullPipelineLayer {
            wq: qw, wk: qw, wv: qw, wo: qw,
            gate: qw, up: qw, down: qw,
            input_norm: norms, post_attn_norm: norms,
            pre_ffn_norm: None, post_ffn_norm: None,
            input_norm_bias: None, post_attn_norm_bias: None,
            norm_offset: 0.0, qk_norm_offset: 0.0, eps: 1e-6,
            has_post_norms: false, norm_type: NormType::RmsNorm,
            ffn_type, activation: Activation::Silu,
            attn_scale: 0.5, head_dim: 4, num_q_heads: 1, num_kv_heads: 1,
            rope_base: 10000.0, rotary_dim: 0, sliding_window: 0,
            has_v_norm: false, layer_scalar: 0.0,
            q_norm_weight: None, k_norm_weight: None,
            ffn_up_bias: None, ffn_down_bias: None,
            moe, moe_combined_output_norm: false, moe_outer_post_norm: None,
        }
    }

    #[test]
    fn activation_from_bool() {
        assert_eq!(Activation::from(true), Activation::GeluTanh);
        assert_eq!(Activation::from(false), Activation::Silu);
    }

    #[test]
    fn is_gated_matches_ffn_type() {
        let norms = [1.0f32; 4];
        let gated = minimal_layer(&[], &norms, FfnType::Gated, None);
        let standard = minimal_layer(&[], &norms, FfnType::Standard, None);
        assert!(gated.is_gated());
        assert!(!standard.is_gated());
    }

    #[test]
    fn is_hybrid_moe_reflects_option() {
        let norms = [1.0f32; 4];
        let no_moe = minimal_layer(&[], &norms, FfnType::Gated, None);
        assert!(!no_moe.is_hybrid_moe());

        let moe = MoeLayerWeights {
            experts_gate_up: &[], experts_down: &[],
            router_proj: &[], router_scale: &[], router_per_expert_scale: &[],
            router_norm: &[], router_norm_parameter_free: false,
            router_input_scalar: 1.0, pre_experts_norm: &[],
            post_ffn1_norm: &[], post_experts_norm: &[],
            num_experts: 2, top_k: 1, intermediate_size: 4,
            activation: Activation::Silu,
            expert_data_format: QuantFormat::BF16,
        };
        let with_moe = minimal_layer(&[], &norms, FfnType::Gated, Some(moe));
        assert!(with_moe.is_hybrid_moe());
    }

    #[test]
    fn quant_format_equality() {
        assert_eq!(QuantFormat::Q4_K, QuantFormat::Q4_K);
        assert_ne!(QuantFormat::Q4_K, QuantFormat::Q6_K);
        assert_ne!(QuantFormat::Q4_0, QuantFormat::Q4_KF);
    }
}
