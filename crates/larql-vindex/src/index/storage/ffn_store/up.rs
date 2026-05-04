//! Feature-major up projections (`up_features.bin`, f32 mmap).
//!
//! Mirror of `down.rs` for the up matrix. `has_full_mmap_ffn` lives
//! here because it's the one cross-cutting predicate (up + down both
//! loaded) — kept on the up side since the up loader is the second
//! to fire by convention.

use std::sync::Arc;

use crate::error::VindexError;
use crate::format::filenames::UP_FEATURES_BIN;
use crate::index::core::VectorIndex;
use crate::mmap_util::mmap_demand_paged;

impl VectorIndex {
    /// Load feature-major up vectors from up_features.bin.
    pub fn load_up_features(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join(UP_FEATURES_BIN);
        if !path.exists() {
            return Err(VindexError::Parse(
                "up_features.bin not found. Run: cargo run --release -p larql-vindex --example build_up_features -- <vindex>".into()
            ));
        }
        let file = std::fs::File::open(&path)?;
        // Demand-paged: only activated feature vectors are read per token.
        let mmap = unsafe { mmap_demand_paged(&file)? };
        self.ffn.up_features_mmap = Some(Arc::new(mmap));
        Ok(())
    }

    /// Get the full up matrix for a layer: [intermediate, hidden] zero-copy view.
    pub fn up_layer_matrix(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        let mmap = self.ffn.up_features_mmap.as_ref()?;
        let intermediate = self.num_features(layer);
        if intermediate == 0 {
            return None;
        }
        let floats_per_layer = intermediate * self.hidden_size;
        let bytes_per_layer = floats_per_layer * 4;
        let start = self.ffn_layer_byte_offset(layer, 1);
        let end = start + bytes_per_layer;
        if end > mmap.len() {
            return None;
        }
        let data = unsafe {
            let ptr = mmap[start..end].as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, floats_per_layer)
        };
        ndarray::ArrayView2::from_shape((intermediate, self.hidden_size), data).ok()
    }

    /// Whether both up and down feature-major mmaps are loaded.
    pub fn has_full_mmap_ffn(&self) -> bool {
        self.ffn.down_features_mmap.is_some() && self.ffn.up_features_mmap.is_some()
    }
}
