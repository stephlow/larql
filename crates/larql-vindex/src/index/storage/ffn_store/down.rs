//! Feature-major down projections (`down_features.bin`, f32 mmap).
//!
//! Zero-copy slicing — the per-feature down vector is a `&[f32]` view
//! straight into the mmap, no decode, no clone. Per-layer offsets go
//! through `ffn_layer_byte_offset` so variable per-layer feature counts
//! (MoE shards) address correctly.

use std::sync::Arc;

use crate::error::VindexError;
use crate::format::filenames::DOWN_FEATURES_BIN;
use crate::index::core::VectorIndex;
use crate::mmap_util::mmap_demand_paged;

impl VectorIndex {
    /// Load feature-major down vectors from down_features.bin.
    pub fn load_down_features(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join(DOWN_FEATURES_BIN);
        if !path.exists() {
            return Err(VindexError::Parse(
                "down_features.bin not found. Run: cargo run --release -p larql-vindex --example build_down_features -- <vindex>".into()
            ));
        }
        let file = std::fs::File::open(&path)?;
        // Demand-paged: only the activated feature vectors are read per token.
        let mmap = unsafe { mmap_demand_paged(&file)? };
        self.ffn.down_features_mmap = Some(Arc::new(mmap));
        Ok(())
    }

    /// Whether feature-major down vectors are loaded.
    pub fn has_down_features(&self) -> bool {
        self.ffn.down_features_mmap.is_some()
    }

    /// Get a feature's contiguous down vector from the mmap'd feature-major file.
    /// Returns `[hidden_size]` f32 slice — zero-copy from mmap.
    pub fn down_feature_vector(&self, layer: usize, feature: usize) -> Option<&[f32]> {
        let mmap = self.ffn.down_features_mmap.as_ref()?;
        let intermediate = self.num_features(layer);
        if intermediate == 0 || feature >= intermediate {
            return None;
        }

        let layer_offset = self.ffn_layer_byte_offset(layer, 1);
        let feature_offset = feature * self.hidden_size * 4;
        let start = layer_offset + feature_offset;
        let end = start + self.hidden_size * 4;

        if end > mmap.len() {
            return None;
        }

        let data = unsafe {
            let ptr = mmap[start..end].as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, self.hidden_size)
        };
        Some(data)
    }

    /// Get the full down matrix for a layer: [intermediate, hidden] zero-copy view.
    pub fn down_layer_matrix(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        let mmap = self.ffn.down_features_mmap.as_ref()?;
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
}
