//! `ProjectionStore` — owns lm_head and attention weight mmaps.
//!
//! Carved out of `VectorIndex` in the 2026-04-25 reorg. Method
//! implementations stay in `storage/lm_head.rs` and `storage/attn.rs`
//! (they need the full index for shape info).

use std::sync::Arc;

pub struct ProjectionStore {
    /// Mmap'd lm_head (output projection): `[vocab_size, hidden_size]`, f32.
    pub lm_head_mmap: Option<Arc<memmap2::Mmap>>,
    /// Mmap'd lm_head as f16 — typically the tied-embedding case.
    pub lm_head_f16_mmap: Option<Arc<memmap2::Mmap>>,
    /// Q4_0 lm_head mmap.
    pub lm_head_q4_mmap: Option<Arc<memmap2::Mmap>>,
    /// Q4_0 lm_head synthesised in RAM from f16 embeddings at load time.
    pub lm_head_q4_synth: Option<Arc<Vec<u8>>>,
    /// Q4_K / Q6_K attention weights (Ollama-compatible).
    pub attn_q4k_mmap: Option<Arc<memmap2::Mmap>>,
    /// Per-matrix (offset, length, format) for `attn_q4k_mmap`.
    pub attn_q4k_manifest: Option<Vec<(usize, usize, String)>>,
    /// Q4_0 attention weights (full-pipeline GPU path).
    pub attn_q4_mmap: Option<Arc<memmap2::Mmap>>,
    /// Per-matrix (offset, length) for `attn_q4_mmap`.
    pub attn_q4_manifest: Option<Vec<(usize, usize)>>,
    /// Q8_0 attention weights (higher-precision option).
    pub attn_q8_mmap: Option<Arc<memmap2::Mmap>>,
    /// Per-matrix (offset, vals_len, scales_len) for `attn_q8_mmap`.
    pub attn_q8_manifest: Option<Vec<(usize, usize, usize)>>,
}

impl ProjectionStore {
    pub fn empty() -> Self {
        Self {
            lm_head_mmap: None,
            lm_head_f16_mmap: None,
            lm_head_q4_mmap: None,
            lm_head_q4_synth: None,
            attn_q4k_mmap: None,
            attn_q4k_manifest: None,
            attn_q4_mmap: None,
            attn_q4_manifest: None,
            attn_q8_mmap: None,
            attn_q8_manifest: None,
        }
    }
}

impl Clone for ProjectionStore {
    fn clone(&self) -> Self {
        Self {
            lm_head_mmap: self.lm_head_mmap.clone(),
            lm_head_f16_mmap: self.lm_head_f16_mmap.clone(),
            lm_head_q4_mmap: self.lm_head_q4_mmap.clone(),
            lm_head_q4_synth: self.lm_head_q4_synth.clone(),
            attn_q4k_mmap: self.attn_q4k_mmap.clone(),
            attn_q4k_manifest: self.attn_q4k_manifest.clone(),
            attn_q4_mmap: self.attn_q4_mmap.clone(),
            attn_q4_manifest: self.attn_q4_manifest.clone(),
            attn_q8_mmap: self.attn_q8_mmap.clone(),
            attn_q8_manifest: self.attn_q8_manifest.clone(),
        }
    }
}
