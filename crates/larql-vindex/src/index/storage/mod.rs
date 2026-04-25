//! Storage layer — mmap loaders, slicing, decode caches, residency
//! management. These modules touch raw bytes and own the read-side
//! invariants (alignment, layer ranges, page-cache hints).
//!
//! Pure dispatch and KNN compute live in `crate::index::compute`;
//! mutation paths live in `crate::index::mutate`.

pub mod accessors;
pub mod attn;
pub mod fp4_storage;
pub mod lm_head;
pub mod residency;

pub use residency::{LayerState, ResidencyManager};
