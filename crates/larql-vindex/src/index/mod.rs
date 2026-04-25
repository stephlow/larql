//! VectorIndex — the in-memory KNN engine, mutation interface, MoE
//! router, and HNSW index.
//!
//! Top-level structure (post 2026-04-25 reorg):
//! - `types`      — FeatureMeta, GateIndex trait, WalkHit, callbacks
//! - `core`       — VectorIndex struct + constructors + loading
//! - `compute/`   — KNN dispatch, HNSW, MoE routing (read-only over storage)
//! - `storage/`   — mmap loaders, residency, decode caches
//! - `mutate/`    — INSERT / DELETE, NDJSON heap loaders, persistence
//! - `gate`, `walk`, `accessors`, `attn`, `lm_head`, `fp4_storage` —
//!   pending split into compute/ and storage/ in a follow-up pass

pub mod types;
pub mod core;
mod gate;
mod gate_trait;
mod walk;
#[cfg(test)]
mod ffn_dispatch_tests;
pub mod compute;
pub mod storage;
pub mod mutate;

pub use core::*;
pub use compute::router::RouterIndex;
pub use storage::residency::{ResidencyManager, LayerState};

// Backwards-compatible aliases at the old paths. In-tree code is
// migrated incrementally; external callers can reach the modules by
// either name. Drop these once `crate::index::{hnsw,attn,lm_head,…}`
// users are all updated.
pub use compute::hnsw;
pub use compute::router;
pub use storage::residency;
pub use storage::attn;
pub use storage::lm_head;
pub use storage::accessors;
pub use storage::fp4_storage;
