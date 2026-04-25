//! Per-window token-ID archive (COLD tier).
//!
//! Append-only; never evicted. Provides the raw token stream for replay.
//! Four bytes per token (u32), regardless of model size.

use std::collections::HashMap;

#[derive(Default)]
pub struct TokenArchive {
    tokens: HashMap<usize, Vec<u32>>,
    abs_offsets: HashMap<usize, usize>,
}

impl TokenArchive {
    pub fn new() -> Self { Self::default() }

    pub fn archive(&mut self, window_id: usize, token_ids: Vec<u32>, abs_offset: usize) {
        self.tokens.insert(window_id, token_ids);
        self.abs_offsets.insert(window_id, abs_offset);
    }

    /// Return `(token_ids, abs_offset)` for a window.
    pub fn retrieve(&self, window_id: usize) -> Option<(&[u32], usize)> {
        let toks = self.tokens.get(&window_id)?;
        let off = *self.abs_offsets.get(&window_id)?;
        Some((toks.as_slice(), off))
    }

    pub fn len(&self) -> usize { self.tokens.len() }
    pub fn is_empty(&self) -> bool { self.tokens.is_empty() }
    pub fn total_tokens(&self) -> usize { self.tokens.values().map(|t| t.len()).sum() }
    pub fn total_bytes(&self) -> usize { self.tokens.values().map(|t| t.len() * 4).sum() }
}
