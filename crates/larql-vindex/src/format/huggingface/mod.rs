//! HuggingFace Hub integration — download, publish, and discovery
//! for vindex-shaped dataset repos.
//!
//! ```text
//! # Download a vindex
//! larql> USE "hf://chrishayuk/gemma-3-4b-it-vindex";
//!
//! # Upload a vindex
//! larql publish gemma3-4b.vindex --repo chrishayuk/gemma-3-4b-it-vindex
//! ```
//!
//! Module split (post 2026-04-25 audit):
//! - [`download`]  — `hf://` resolution, snapshot caching, conditional fetch
//! - [`publish`]   — repo creation, file uploads, LFS protocol, callbacks
//! - [`discovery`] — collections, repo existence, item fetch
//!
//! Shared constants live here. Each submodule re-imports them via
//! `use super::{VINDEX_CORE_FILES, VINDEX_WEIGHT_FILES}`.

use crate::format::filenames::*;

/// The files that make up a vindex, in priority order for lazy
/// loading. Used by `download` to decide which pieces a partial
/// fetch should include first, and by `publish` to walk the upload
/// list deterministically.
pub(crate) const VINDEX_CORE_FILES: &[&str] = &[
    INDEX_JSON,
    TOKENIZER_JSON,
    GATE_VECTORS_BIN,
    EMBEDDINGS_BIN,
    DOWN_META_BIN,
    "down_meta.jsonl",
    RELATION_CLUSTERS_JSON,
    FEATURE_LABELS_JSON,
];

pub(crate) const VINDEX_WEIGHT_FILES: &[&str] = &[
    ATTN_WEIGHTS_BIN,
    NORMS_BIN,
    UP_WEIGHTS_BIN,
    DOWN_WEIGHTS_BIN,
    LM_HEAD_BIN,
    WEIGHT_MANIFEST_JSON,
];

pub mod discovery;
pub mod download;
pub mod publish;

// Re-export the previous flat-module surface so callers don't have to
// pick a submodule.
pub use discovery::{
    add_collection_item, dataset_repo_exists, ensure_collection, fetch_collection_items,
    repo_exists, CollectionItem,
};
pub use download::{
    download_hf_weights, resolve_hf_vindex, resolve_hf_vindex_with_progress, DownloadProgress,
};
pub use publish::{
    publish_vindex, publish_vindex_with_opts, PublishCallbacks, PublishOptions,
    SilentPublishCallbacks,
};

/// Check if a path is an `hf://` reference. Lives here (not under
/// `download`) because callers in `publish` and `discovery` test it
/// too.
pub fn is_hf_path(path: &str) -> bool {
    path.starts_with("hf://")
}
