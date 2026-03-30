extern crate blas_src;

pub mod attention;
pub mod capture;
pub mod error;
pub mod ffn;
pub mod forward;
pub mod graph_ffn;
pub mod model;
pub mod route_ffn;
pub mod residual;
pub mod tokenizer;
pub mod vector_index;
pub mod walker;

// Re-export dependencies for downstream crates.
pub use larql_models;
pub use ndarray;
pub use safetensors;
pub use tokenizers;

// Re-export essentials at crate root.
pub use capture::{
    CaptureCallbacks, CaptureConfig, InferenceModel, TopKEntry, VectorFileHeader, VectorRecord,
};
pub use error::InferenceError;
pub use ffn::{FfnBackend, LayerFfnRouter, SparseFfn, WeightFfn};
pub use attention::AttentionWeights;
pub use forward::{
    capture_residuals, predict, predict_with_ffn, predict_with_router, trace_forward,
    trace_forward_full, trace_forward_with_ffn, LayerAttentionCapture, PredictResult, TraceResult,
};
pub use graph_ffn::{GateIndex, GraphFfn, IndexBuildCallbacks, SilentIndexCallbacks};
pub use route_ffn::{RouteFfn, RouteGuidedFfn, RouteTable};
pub use vector_index::{
    load_feature_labels, load_model_weights_from_vindex, load_vindex_config,
    load_vindex_embeddings, load_vindex_tokenizer, write_model_weights, VectorIndex, VindexConfig,
    WalkFfn, WalkTrace,
};
pub use model::{load_model_dir, resolve_model_path, ModelWeights};
pub use tokenizer::{decode_token, load_tokenizer};

// Walker re-exports.
pub use walker::attention_walker::{AttentionLayerResult, AttentionWalker};
pub use walker::vector_extractor::{
    ExtractCallbacks, ExtractConfig, ExtractSummary, VectorExtractor,
};
pub use walker::weight_walker::{
    walk_model, LayerResult, LayerStats, WalkCallbacks, WalkConfig, WeightWalker,
};
