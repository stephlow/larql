//! Vindex — the queryable model format.
//!
//! Decompile, browse, edit, and recompile neural networks.
//! This crate owns the complete vindex lifecycle:
//! extract, load, query, mutate, patch, save, compile.

extern crate blas_src;

// ── Module structure ──
pub mod clustering;
pub mod config;
pub mod describe;
pub mod error;
pub mod extract;
pub mod format;
pub mod index;
pub mod patch;

// ── Re-export dependencies ──
pub use ndarray;
pub use tokenizers;

// ── Re-export essentials at crate root ──

// Config
pub use config::dtype::StorageDtype;
pub use config::types::{
    DownMetaRecord, DownMetaTopK, ExtractLevel, LayerBands, MoeConfig,
    VindexConfig, VindexLayerInfo, VindexModelConfig, VindexSource,
};

// Error
pub use error::VindexError;

// Index
pub use index::core::{
    FeatureMeta, IndexLoadCallbacks, SilentLoadCallbacks, VectorIndex, WalkHit, WalkTrace,
};

// Describe
pub use describe::{DescribeEdge, LabelSource};

// Extract
pub use extract::{
    build_vindex, build_vindex_resume, build_vindex_from_vectors,
    IndexBuildCallbacks, SilentBuildCallbacks,
};

// Format
pub use format::checksums;
pub use format::down_meta;
pub use format::load::{
    load_feature_labels, load_vindex_config, load_vindex_embeddings, load_vindex_tokenizer,
};
pub use format::loader::{load_model_dir, resolve_model_path};
pub use format::weights::{write_model_weights, load_model_weights};

// Patch
pub use patch::core::{PatchOp, PatchedVindex, VindexPatch};
