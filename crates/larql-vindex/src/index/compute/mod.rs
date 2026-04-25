//! Compute layer — KNN dispatch, HNSW search, MoE routing.
//! Reads from `crate::index::storage` and `crate::index::core`;
//! never touches mmap bytes directly (always via store accessors).

pub mod hnsw;
pub mod router;

pub use router::RouterIndex;
