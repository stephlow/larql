//! Full transformer forward pass.
//!
//! Runs tokens through embedding → layers → final norm → logits.
//! Uses the ModelArchitecture trait for model-specific behavior
//! and FfnBackend trait for swappable FFN computation.
//!
//! Submodules:
//! - `ops`: Small math utilities (dot_proj, add_bias, apply_norm)
//! - `embed`: Token embedding with architecture-specific scaling
//! - `ple`: Per-Layer Embeddings (gated per-layer token embeddings)
//! - `layer`: Single-layer dispatch (attention + FFN + PLE + scalar)
//! - `predict`: Logits computation and all predict_* entry points
//!   - `predict/types`: Result structs and LayerMode enum
//!   - `predict/raw`: RawForward and raw logit forward passes
//!   - `predict/dense`: Dense weight forward passes and logit projection
//!   - `predict/ffn`: Custom FFN backend, router, and strategy forward passes
//! - `trace`: Residual/activation capture and calibration

pub mod embed;
pub mod infer_patched;
pub mod kv_generate;
pub mod layer;
pub mod memit;
pub mod ops;
pub mod ple;
pub mod predict;
pub mod target_delta;
pub mod trace;

// ── Re-export ops so all `super::apply_norm` / `crate::forward::*` paths work ──
pub use ops::{add_bias, apply_norm, dot_proj, softmax};

// ── Re-export types from predict::types so `trace.rs` and other siblings
//    can still `use super::{TraceResult, LayerAttentionCapture, ...}` ──
pub use predict::types::{
    LayerAttentionCapture, LayerMode, PredictResult, PredictResultWithAttention,
    PredictResultWithResiduals, TraceResult,
};

// ── Re-exports: preserve all `crate::forward::*` paths ──

pub use embed::embed_tokens_pub;
pub use infer_patched::{
    apply_knn_override, infer_patched, infer_patched_q4k, walk_trace_from_residuals,
    InferPatchedResult, KnnOverride, KNN_COSINE_THRESHOLD,
};
pub use kv_generate::{
    generate_cached, generate_cached_backend, generate_cached_constrained,
    generate_cached_with_window,
};
pub use layer::{run_attention_public, run_ffn, run_layer_with_ffn};
pub use memit::{run_memit, run_memit_with_target_opt, MemitFact, MemitFactResult, MemitResult};
pub use predict::{
    forward_from_layer, forward_raw_logits, forward_raw_logits_with_prefix, hidden_to_raw_logits,
    logit_lens_top1, logits_to_predictions_pub, predict, predict_from_hidden,
    predict_from_hidden_with_ffn, predict_with_ffn, predict_with_ffn_attention,
    predict_with_ffn_trace, predict_with_router, predict_with_strategy, predict_with_temperature,
    RawForward,
};
pub use target_delta::{TargetDelta, TargetDeltaOpts};
pub use trace::{
    calibrate_scalar_gains, capture_decoy_residuals, capture_ffn_activation_matrix,
    capture_residuals, capture_spec_residuals, estimate_ffn_covariance, forward_to_layer,
    trace_forward, trace_forward_full, trace_forward_with_ffn, SpecCapture,
};
