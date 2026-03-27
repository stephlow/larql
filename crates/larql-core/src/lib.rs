#[cfg(feature = "walker")]
extern crate blas_src;

pub mod algo;
pub mod core;
pub mod engine;
pub mod io;
pub mod walker;

// Re-export the essential types at crate root.
pub use core::edge::Edge;
pub use core::enums::{MergeStrategy, SourceType};
pub use core::graph::Graph;
pub use core::schema::Schema;

pub use engine::bfs::{extract_bfs, BfsCallbacks, BfsConfig, BfsResult};
pub use engine::chain::{chain_tokens, ChainResult};
pub use engine::provider::{ModelProvider, PredictionResult, TokenPrediction};
pub use engine::templates::TemplateRegistry;

pub use io::checkpoint::CheckpointLog;
pub use io::format::Format;
pub use io::json::{load_json, save_json};
pub use io::{load, save, load_with_format, save_with_format, to_bytes, from_bytes};

pub use algo::merge::merge_graphs;
pub use algo::shortest_path::shortest_path;

#[cfg(feature = "walker")]
pub use walker::weight_walker::{
    walk_model, resolve_model_path, WeightWalker, WalkConfig, WalkCallbacks, LayerResult, LayerStats,
};
#[cfg(feature = "walker")]
pub use walker::attention_walker::{AttentionWalker, AttentionLayerResult};
#[cfg(feature = "walker")]
pub use walker::vector_extractor::{VectorExtractor, ExtractConfig, ExtractCallbacks, ExtractSummary};
