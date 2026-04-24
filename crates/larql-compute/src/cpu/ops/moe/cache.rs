//! Bounded LRU cache for dequantised MoE expert weights.
//!
//! Gemma 4 26B A4B has 128 experts × 60 layers × ~312 MB (gate_up + down per
//! expert). The router picks 8-per-token, so the naive path decodes ~150 GB
//! of BF16 → f32 per generated token. In practice many tokens share experts,
//! so a bounded LRU keyed by the mmap pointer lets repeat hits skip the
//! dequant + allocation entirely.
//!
//! Key = mmap pointer (the `&[u8]` byte slice for one expert's packed tensor).
//! The mmap is stable for the life of the process, so the pointer uniquely
//! identifies `(layer, expert, kind)` without threading those ids down.
//!
//! Value = `Arc<Vec<f32>>`. Cloning on hit is O(1) — real allocation + BF16→f32
//! conversion runs exactly once per cached entry.
//!
//! Sizing: `LARQL_MOE_CACHE_ENTRIES` env var caps the entry count (default 64).
//! With 312 MB/entry on 26B A4B the default is ~20 GB — small enough to fit
//! alongside the mmap'd vindex on 64+ GB Macs. Set to 0 to disable.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, OnceLock};

/// LRU cache entry: dequantised expert weights.
pub(super) type ExpertF32 = Arc<Vec<f32>>;

/// Cache key — the byte slice's start pointer is stable across the lifetime
/// of the mmap, so different experts in the same packed tensor get distinct
/// keys via their offset. `usize` wrapping the pointer lets the map be Send.
type Key = usize;

struct Inner {
    map: std::collections::HashMap<Key, ExpertF32>,
    order: VecDeque<Key>,
    cap: usize,
}

impl Inner {
    fn new(cap: usize) -> Self {
        Self {
            map: std::collections::HashMap::with_capacity(cap.saturating_add(1)),
            order: VecDeque::with_capacity(cap.saturating_add(1)),
            cap,
        }
    }

    fn get(&mut self, key: Key) -> Option<ExpertF32> {
        let v = self.map.get(&key)?.clone();
        // LRU touch: move to back without reordering the map. Linear in the
        // VecDeque; for cap=64 this is a handful of pointer moves per lookup
        // and stays well below the BLAS cost we're amortising.
        if let Some(pos) = self.order.iter().position(|k| *k == key) {
            self.order.remove(pos);
            self.order.push_back(key);
        }
        Some(v)
    }

    fn insert(&mut self, key: Key, val: ExpertF32) {
        if self.cap == 0 { return; }
        if self.map.contains_key(&key) {
            // Already present (a concurrent inserter raced us); don't duplicate.
            return;
        }
        while self.map.len() >= self.cap {
            if let Some(victim) = self.order.pop_front() {
                self.map.remove(&victim);
            } else {
                break;
            }
        }
        self.order.push_back(key);
        self.map.insert(key, val);
    }
}

fn cell() -> &'static Mutex<Inner> {
    static CELL: OnceLock<Mutex<Inner>> = OnceLock::new();
    CELL.get_or_init(|| {
        let cap = std::env::var("LARQL_MOE_CACHE_ENTRIES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(64);
        Mutex::new(Inner::new(cap))
    })
}

/// Return a cached Arc<Vec<f32>> for `bytes` (the BF16 packed expert slice),
/// dequantising + inserting on miss. On hit, no allocation happens.
pub(super) fn cached_dequant(bytes: &[u8]) -> ExpertF32 {
    let key = bytes.as_ptr() as usize;
    // Fast path: read-only hit under the mutex.
    if let Ok(mut inner) = cell().lock() {
        if let Some(hit) = inner.get(key) {
            return hit;
        }
    }
    // Miss: dequantise OUTSIDE the lock, then insert.
    let decoded = super::math::bf16_to_f32(bytes);
    let arc = Arc::new(decoded);
    if let Ok(mut inner) = cell().lock() {
        inner.insert(key, arc.clone());
    }
    arc
}
