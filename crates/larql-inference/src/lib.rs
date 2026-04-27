extern crate blas_src;

pub mod attention;
pub mod capture;
pub mod chat;
pub mod engines;
pub mod error;
pub mod experts;
pub mod ffn;
pub mod forward;
pub mod layer_graph;
pub mod model;
pub mod prompt;
pub mod residual;
pub mod residual_diff;
pub mod tokenizer;
pub mod trace;
pub mod trie;
pub mod vindex;
pub mod walker;

// Re-export dependencies for downstream crates.
pub use larql_models;
pub use larql_vindex;
pub use ndarray;
pub use safetensors;
pub use tokenizers;

// Backend re-exports (from larql-compute).
pub use larql_compute::cpu::ops::moe::{
    cpu_moe_forward, run_single_expert, run_single_expert_with_norm,
};
pub use larql_compute::Activation as ComputeActivation;
pub use larql_compute::CpuBackend;
pub use larql_compute::MoeLayerWeights;
pub use larql_compute::QuantFormat;
pub use larql_compute::{
    cpu_backend, default_backend, dot_proj_gpu, matmul_gpu, ComputeBackend, MatMulOp,
};

/// Map a model's activation function to the compute-layer `Activation` enum.
pub fn activation_from_arch(
    arch: &dyn larql_models::ModelArchitecture,
) -> larql_compute::Activation {
    match arch.activation() {
        larql_models::Activation::GeluTanh => larql_compute::Activation::GeluTanh,
        _ => larql_compute::Activation::Silu,
    }
}
#[cfg(feature = "metal")]
pub use larql_compute::MetalBackend;

// Re-export essentials at crate root.
pub use attention::AttentionWeights;
pub use capture::{
    CaptureCallbacks, CaptureConfig, InferenceModel, TopKEntry, VectorFileHeader, VectorRecord,
};
pub use chat::{wrap_chat_prompt, wrap_prompt_raw, wrap_with_vindex_template, ChatWrap};
pub use error::InferenceError;
pub use ffn::graph_backend::{GateIndex, IndexBuildCallbacks, SilentIndexCallbacks};
pub use ffn::{
    BackendFfn, FfnBackend, LayerFfnRouter, MoeRouterWeights, RemoteFfnConfig, RemoteFfnError,
    RemoteLatencyStats, RemoteMoeBackend, RemoteMoeError, RemoteWalkBackend, ShardConfig,
    SparseFfn, WeightFfn,
};
pub use forward::{
    apply_knn_override, calibrate_scalar_gains, capture_decoy_residuals,
    capture_ffn_activation_matrix, capture_residuals, capture_spec_residuals,
    estimate_ffn_covariance, forward_from_layer, forward_raw_logits, forward_to_layer,
    generate_cached_constrained, hidden_to_raw_logits, infer_patched, infer_patched_q4k,
    logit_lens_top1, predict, predict_from_hidden, predict_from_hidden_with_ffn, predict_with_ffn,
    predict_with_ffn_attention, predict_with_ffn_trace, predict_with_router, predict_with_strategy,
    run_memit, run_memit_with_target_opt, trace_forward, trace_forward_full,
    trace_forward_with_ffn, walk_trace_from_residuals, InferPatchedResult, KnnOverride,
    LayerAttentionCapture, LayerMode, MemitFact, MemitFactResult, MemitResult, PredictResult,
    PredictResultWithAttention, PredictResultWithResiduals, RawForward, SpecCapture, TargetDelta,
    TargetDeltaOpts, TraceResult, KNN_COSINE_THRESHOLD,
};
pub use layer_graph::{
    build_adaptive_graph,
    detect_template,
    generate,
    generate_streaming,
    generate_with_sampling,
    // Expert grid generation
    grid::{generate_with_remote_moe, GridGenerateResult},
    hybrid::predict_hybrid,
    predict_honest,
    predict_pipeline,
    predict_split_cached,
    predict_split_pass,
    predict_with_graph,
    predict_with_graph_vindex_logits,
    trace_with_graph,
    AttentionCache,
    CachedLayerGraph,
    // Multi-turn chat session
    ChatMLRenderer,
    ChatSession,
    DenseLayerGraph,
    // Generation building blocks (EOS, detok, sampling)
    Detokenizer,
    EosConfig,
    GemmaRenderer,
    GenerateResult,
    GuidedWalkLayerGraph,
    // Production
    LayerGraph,
    LayerOutput,
    Llama3Renderer,
    PerLayerGraph,
    PipelinedLayerGraph,
    Sampler,
    SamplingConfig,
    TurnRenderer,
    // Analysis/validation
    TemplatePattern,
    TemplateUniverse,
    WalkLayerGraph,
};
pub use model::{load_model_dir, resolve_model_path, ModelWeights};
pub use tokenizer::{decode_token, decode_token_raw, encode_prompt, load_tokenizer};
pub use trace::{
    trace as trace_decomposed, trace_residuals, AnswerWaypoint, BoundaryStore, BoundaryWriter,
    ContextStore, ContextTier, ContextWriter, LayerSummary, ResidualTrace, TraceNode,
    TracePositions, TraceStore, TraceWriter,
};
pub use vindex::{open_inference_vindex, predict_q4k, FfnL1Cache, WalkFfn, WalkFfnConfig};

// Engine re-exports.
pub use engines::accuracy::{
    compare_hidden, cosine_similarity, js_divergence, kl_divergence, mse, softmax, HiddenAccuracy,
};
pub use engines::markov_residual::MarkovResidualEngine;
pub use engines::unlimited_context::UnlimitedContextEngine;
pub use engines::{EngineInfo, EngineKind, KvEngine};

// Walker re-exports.
pub use walker::attention_walker::{AttentionLayerResult, AttentionWalker};
pub use walker::vector_extractor::{
    ExtractCallbacks, ExtractConfig, ExtractSummary, VectorExtractor,
};
pub use walker::weight_walker::{
    walk_model, LayerResult, LayerStats, WalkCallbacks, WalkConfig, WeightWalker,
};
