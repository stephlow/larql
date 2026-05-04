//! Interleaved FFN data — `[gate|up|down]` packed per layer in one
//! contiguous f32 file (`interleaved.bin`).
//!
//! Eliminates TLB thrash from three separate mmap files. Per-layer
//! prefetch lets a forward pass tell the kernel which layer's bytes
//! are about to be read.

use std::sync::Arc;

use crate::error::VindexError;
use crate::format::filenames::INTERLEAVED_BIN;
use crate::index::core::VectorIndex;
use crate::mmap_util::mmap_demand_paged;

impl VectorIndex {
    /// Load interleaved FFN data: [gate|up|down] per layer in one contiguous file.
    /// Eliminates TLB thrash from 3 separate mmap files.
    pub fn load_interleaved(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join(INTERLEAVED_BIN);
        if !path.exists() {
            return Err(VindexError::Parse(
                "interleaved.bin not found. Run: cargo run --release -p larql-vindex --example build_interleaved -- <vindex>".into()
            ));
        }
        let file = std::fs::File::open(&path)?;
        // Demand-paged: per-layer prefetch issued at query time via prefetch_interleaved_layer.
        let mmap = unsafe { mmap_demand_paged(&file)? };
        self.ffn.interleaved_mmap = Some(Arc::new(mmap));
        Ok(())
    }

    /// Whether interleaved FFN data is loaded.
    pub fn has_interleaved(&self) -> bool {
        self.ffn.interleaved_mmap.is_some()
    }

    /// Get gate matrix for a layer from the interleaved file: [intermediate, hidden].
    pub fn interleaved_gate(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        let mmap = self.ffn.interleaved_mmap.as_ref()?;
        let intermediate = self.num_features(layer);
        if intermediate == 0 {
            return None;
        }
        let matrix_floats = intermediate * self.hidden_size;
        let matrix_bytes = matrix_floats * 4;
        let start = self.ffn_layer_byte_offset(layer, 3); // gate is first
        let end = start + matrix_bytes;
        if end > mmap.len() {
            return None;
        }
        let data = unsafe {
            let ptr = mmap[start..end].as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, matrix_floats)
        };
        ndarray::ArrayView2::from_shape((intermediate, self.hidden_size), data).ok()
    }

    /// Get up matrix for a layer from the interleaved file: [intermediate, hidden].
    pub fn interleaved_up(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        let mmap = self.ffn.interleaved_mmap.as_ref()?;
        let intermediate = self.num_features(layer);
        if intermediate == 0 {
            return None;
        }
        let matrix_floats = intermediate * self.hidden_size;
        let matrix_bytes = matrix_floats * 4;
        let start = self.ffn_layer_byte_offset(layer, 3) + matrix_bytes; // up is second
        let end = start + matrix_bytes;
        if end > mmap.len() {
            return None;
        }
        let data = unsafe {
            let ptr = mmap[start..end].as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, matrix_floats)
        };
        ndarray::ArrayView2::from_shape((intermediate, self.hidden_size), data).ok()
    }

    /// Get down matrix for a layer from the interleaved file: [intermediate, hidden].
    pub fn interleaved_down(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        let mmap = self.ffn.interleaved_mmap.as_ref()?;
        let intermediate = self.num_features(layer);
        if intermediate == 0 {
            return None;
        }
        let matrix_floats = intermediate * self.hidden_size;
        let matrix_bytes = matrix_floats * 4;
        let start = self.ffn_layer_byte_offset(layer, 3) + matrix_bytes * 2; // down is third
        let end = start + matrix_bytes;
        if end > mmap.len() {
            return None;
        }
        let data = unsafe {
            let ptr = mmap[start..end].as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, matrix_floats)
        };
        ndarray::ArrayView2::from_shape((intermediate, self.hidden_size), data).ok()
    }

    /// Prefetch next layer's interleaved data into page cache.
    pub fn prefetch_interleaved_layer(&self, layer: usize) {
        #[cfg(unix)]
        if let Some(ref mmap) = self.ffn.interleaved_mmap {
            let intermediate = self.num_features(layer);
            if intermediate == 0 {
                return;
            }
            let matrix_bytes = intermediate * self.hidden_size * 4;
            let layer_bytes = matrix_bytes * 3;
            let start = self.ffn_layer_byte_offset(layer, 3);
            let end = (start + layer_bytes).min(mmap.len());
            if start >= mmap.len() {
                return;
            }
            unsafe {
                let ptr = mmap[start..].as_ptr() as *mut libc::c_void;
                libc::madvise(ptr, end - start, libc::MADV_WILLNEED);
            }
        }
    }
}
