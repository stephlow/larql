//! Q4_0 interleaved FFN data (`interleaved_q4.bin`).
//!
//! Loaders + per-component dequant. Q4_K/Q6_K (the Ollama-compatible
//! variant) lives in the sibling `interleaved_q4k.rs`; this file is
//! the predecessor format used before the K-quant rollout.

use std::sync::Arc;

use crate::error::VindexError;
use crate::format::filenames::INTERLEAVED_Q4_BIN;
use crate::index::core::VectorIndex;
use crate::mmap_util::mmap_demand_paged;

impl VectorIndex {
    /// Load Q4_0 interleaved FFN data.
    pub fn load_interleaved_q4(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join(INTERLEAVED_Q4_BIN);
        if !path.exists() {
            return Err(VindexError::Parse("interleaved_q4.bin not found".into()));
        }
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { mmap_demand_paged(&file)? };
        self.ffn.interleaved_q4_mmap = Some(Arc::new(mmap));
        Ok(())
    }

    pub fn has_interleaved_q4(&self) -> bool {
        self.ffn.interleaved_q4_mmap.is_some()
    }

    /// Dequantize one matrix from Q4 interleaved file → f32 Array2.
    /// component: 0=gate, 1=up, 2=down
    fn dequant_q4_matrix(&self, layer: usize, component: usize) -> Option<ndarray::Array2<f32>> {
        let mmap = self.ffn.interleaved_q4_mmap.as_ref()?;
        let intermediate = self.num_features(layer);
        if intermediate == 0 {
            return None;
        }

        let floats_per_matrix = intermediate * self.hidden_size;
        let q4_bytes_per_matrix = floats_per_matrix / larql_models::quant::ggml::Q4_0_BLOCK_ELEMS
            * larql_models::quant::ggml::Q4_0_BLOCK_BYTES;
        let q4_bytes_per_layer = q4_bytes_per_matrix * 3;

        let start = layer * q4_bytes_per_layer + component * q4_bytes_per_matrix;
        let end = start + q4_bytes_per_matrix;
        if end > mmap.len() {
            return None;
        }

        let q4_data = &mmap[start..end];
        let floats = larql_models::quant::ggml::dequantize_q4_0(q4_data, floats_per_matrix).ok()?;
        ndarray::Array2::from_shape_vec((intermediate, self.hidden_size), floats).ok()
    }

    /// Get gate matrix from Q4 interleaved file, dequantized to f32.
    pub fn interleaved_q4_gate(&self, layer: usize) -> Option<ndarray::Array2<f32>> {
        self.dequant_q4_matrix(layer, 0)
    }

    /// Get up matrix from Q4 interleaved file, dequantized to f32.
    pub fn interleaved_q4_up(&self, layer: usize) -> Option<ndarray::Array2<f32>> {
        self.dequant_q4_matrix(layer, 1)
    }

    /// Get down matrix from Q4 interleaved file, dequantized to f32.
    pub fn interleaved_q4_down(&self, layer: usize) -> Option<ndarray::Array2<f32>> {
        self.dequant_q4_matrix(layer, 2)
    }

    /// Prefetch next layer's Q4 data.
    pub fn prefetch_interleaved_q4_layer(&self, layer: usize) {
        #[cfg(unix)]
        if let Some(ref mmap) = self.ffn.interleaved_q4_mmap {
            let intermediate = self.num_features(layer);
            if intermediate == 0 {
                return;
            }
            let q4_bytes_per_matrix = intermediate * self.hidden_size
                / larql_models::quant::ggml::Q4_0_BLOCK_ELEMS
                * larql_models::quant::ggml::Q4_0_BLOCK_BYTES;
            let q4_bytes_per_layer = q4_bytes_per_matrix * 3;
            let start = layer * q4_bytes_per_layer;
            let end = (start + q4_bytes_per_layer).min(mmap.len());
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
