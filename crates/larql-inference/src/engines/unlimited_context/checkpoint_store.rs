//! Per-window boundary K,V checkpoint store (WARM tier).
//!
//! Each checkpoint is the K,V at the last position of a closed window — one
//! (K, V) pair per layer. Bytes per checkpoint on Gemma 3 4B ≈ 278 KB (f32).

use std::collections::HashMap;
use crate::attention::SharedKV;

#[derive(Default)]
pub struct CheckpointStore {
    kv: HashMap<usize, Vec<SharedKV>>,
    abs_pos: HashMap<usize, usize>,
}

impl CheckpointStore {
    pub fn new() -> Self { Self::default() }

    /// Save the last-position K,V for a closed window.
    /// `kv_last[layer]` must have shape (1, kv_dim) for both K and V.
    pub fn save(&mut self, window_id: usize, kv_last: Vec<SharedKV>, abs_pos: usize) {
        debug_assert!(
            kv_last.iter().all(|(k, v)| k.shape()[0] == 1 && v.shape()[0] == 1),
            "checkpoint must be single-row K/V per layer"
        );
        self.kv.insert(window_id, kv_last);
        self.abs_pos.insert(window_id, abs_pos);
    }

    pub fn load(&self, window_id: usize) -> Option<(Vec<SharedKV>, usize)> {
        let kv = self.kv.get(&window_id)?.clone();
        let pos = *self.abs_pos.get(&window_id)?;
        Some((kv, pos))
    }

    pub fn contains(&self, window_id: usize) -> bool { self.kv.contains_key(&window_id) }
    pub fn len(&self) -> usize { self.kv.len() }
    pub fn is_empty(&self) -> bool { self.kv.is_empty() }

    pub fn evict(&mut self, window_ids: &[usize]) {
        for id in window_ids {
            self.kv.remove(id);
            self.abs_pos.remove(id);
        }
    }

    pub fn total_bytes(&self) -> usize {
        self.kv
            .values()
            .flat_map(|layers| layers.iter())
            .map(|(k, v)| (k.len() + v.len()) * 4)
            .sum()
    }
}
