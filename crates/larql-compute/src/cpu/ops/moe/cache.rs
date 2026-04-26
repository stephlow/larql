//! Bounded LRU cache for dequantised MoE expert weights.
//!
//! Gemma 4 26B A4B has 128 experts × 30 MoE layers. Per-layer Q4_K storage:
//! ~24 MB f32 per expert (gate_up + down combined). The router picks
//! top-K=8 per layer, so a naive decode path runs ~5.7 GB of Q4_K → f32
//! per token. In practice prompts route consistently to the same experts;
//! a bounded LRU keyed by the mmap pointer lets repeat hits skip both
//! allocation and decode.
//!
//! Key = mmap pointer (the `&[u8]` byte slice for one expert's packed
//! tensor). The mmap is stable for the life of the process, so the pointer
//! uniquely identifies `(layer, expert, kind)` even after the per-expert
//! byte-table refactor — `experts_gate_up[ei]` is still backed by the same
//! mmap range across calls.
//!
//! Value = `Arc<Vec<f32>>`. Cloning on hit is O(1) — real allocation +
//! dequant runs exactly once per cached entry.
//!
//! Sizing: `LARQL_MOE_CACHE_ENTRIES` env var caps the entry count
//! (default 64). 64 × ~24 MB ≈ 1.5 GB at Gemma 4 26B-A4B Q4_K dimensions.
//! For workloads with high expert diversity (many distinct prompts, large
//! `top_k`, or models with more experts) raise this to 128 or 256 to cover
//! the working set. Set to 0 to disable caching entirely.
//!
//! Format dispatch (BF16 / Q4_K / F32) is on the dequant path, not the
//! cache key — same bytes always dequant to the same f32 vector regardless
//! of the format tag, so a single key works for all formats.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, OnceLock};

/// LRU cache entry: dequantised expert weights.
pub(super) type ExpertF32 = Arc<Vec<f32>>;

/// Cache key — in production the byte slice's start pointer is stable across
/// the lifetime of the mmap, so different experts in the same packed tensor get
/// distinct keys via their offset. Tests use short heap Vecs whose addresses can
/// be recycled between cases, so include a content fingerprint under `cfg(test)`.
#[cfg(not(test))]
type Key = usize;

#[cfg(test)]
type Key = (usize, usize, u64);

#[cfg(not(test))]
fn cache_key(bytes: &[u8]) -> Key {
    bytes.as_ptr() as usize
}

#[cfg(test)]
fn cache_key(bytes: &[u8]) -> Key {
    use std::hash::{Hash, Hasher};

    let mut h = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut h);
    (bytes.as_ptr() as usize, bytes.len(), h.finish())
}

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
        if self.cap == 0 {
            return;
        }
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

/// Return a cached Arc<Vec<f32>> for `bytes`, dequantising under `format` on
/// miss. `expected_floats` is required for block formats (Q4_K) where the
/// output length is not derivable from the input length without padding info;
/// it's ignored for raw BF16. On hit, no allocation happens.
pub(super) fn cached_dequant(
    bytes: &[u8],
    format: crate::QuantFormat,
    expected_floats: usize,
) -> ExpertF32 {
    let key = cache_key(bytes);
    // Fast path: read-only hit under the mutex. Cache key is just the byte
    // slice identity — same bytes always dequant to the same output.
    if let Ok(mut inner) = cell().lock() {
        if let Some(hit) = inner.get(key) {
            return hit;
        }
    }
    // Miss: dequantise OUTSIDE the lock, then insert.
    let decoded = match format {
        crate::QuantFormat::BF16 => super::math::bf16_to_f32(bytes),
        crate::QuantFormat::Q4_K => {
            crate::cpu::ops::q4_common::dequantize_q4_k(bytes, expected_floats)
        }
        crate::QuantFormat::F32 => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        _ => {
            // Other formats not yet wired into the CPU MoE expert path.
            // Empty fallback → caller treats as a skipped expert.
            Vec::new()
        }
    };
    let arc = Arc::new(decoded);
    if let Ok(mut inner) = cell().lock() {
        inner.insert(key, arc.clone());
    }
    arc
}

#[cfg(test)]
mod cache_format_tests {
    use super::*;
    use crate::QuantFormat;

    /// BF16 path: 2 bytes per float, no padding. Round-trip a fixed value.
    #[test]
    fn bf16_dispatch_round_trip() {
        // 4 BF16 values of 1.0 (0x3F80 little-endian = [0x80, 0x3F]).
        let bytes = vec![0x80u8, 0x3F, 0x80, 0x3F, 0x80, 0x3F, 0x80, 0x3F];
        let out = cached_dequant(&bytes, QuantFormat::BF16, 4);
        assert_eq!(out.len(), 4);
        for v in out.iter() {
            assert!((v - 1.0).abs() < 1e-3, "BF16 1.0 round-trip got {v}");
        }
    }

    /// Q4_K path: 144 bytes per 256 floats. Quantise→dequantise round-trip
    /// must come back within Q4 quantisation noise.
    #[test]
    fn q4k_dispatch_round_trip() {
        // 256-element ramp [-1, 1] — same fixture used by q4_common tests.
        let data: Vec<f32> = (0..256).map(|i| (i as f32 / 255.0) * 2.0 - 1.0).collect();
        let bytes = crate::cpu::ops::q4_common::quantize_q4_k(&data);
        assert_eq!(bytes.len(), 144);

        let out = cached_dequant(&bytes, QuantFormat::Q4_K, 256);
        assert_eq!(out.len(), 256);
        let max_err: f32 = data
            .iter()
            .zip(&*out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        // Q4 nibble step ≈ 0.13 over 2.0 range; allow 2× for sub-block bias.
        assert!(max_err < 0.12, "Q4_K round-trip max error {max_err}");
    }

    /// F32 path: passthrough.
    #[test]
    fn f32_dispatch_passthrough() {
        let data: Vec<f32> = vec![1.0, -2.5, 3.14, 0.0];
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out = cached_dequant(&bytes, QuantFormat::F32, data.len());
        assert_eq!(out.len(), data.len());
        for (a, b) in data.iter().zip(&*out) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    /// Unsupported formats fall back to empty (caller treats as skipped expert).
    #[test]
    fn unsupported_format_returns_empty() {
        let bytes = vec![0u8; 18];
        let out = cached_dequant(&bytes, QuantFormat::Q4_0, 32);
        assert!(
            out.is_empty(),
            "Q4_0 not implemented for MoE → empty fallback"
        );
    }

    /// Out-of-bounds Q4_K input returns empty (no panic).
    #[test]
    fn q4k_truncated_input_returns_empty() {
        let bytes = vec![0u8; 100]; // 100 < 144 = one super-block
        let out = cached_dequant(&bytes, QuantFormat::Q4_K, 256);
        assert!(out.is_empty(), "truncated Q4_K → empty (caller skips)");
    }

    /// Q4_K with non-multiple-of-256 expected_floats returns empty.
    #[test]
    fn q4k_misaligned_length_returns_empty() {
        let bytes = vec![0u8; 144];
        let out = cached_dequant(&bytes, QuantFormat::Q4_K, 200);
        assert!(out.is_empty(), "expected_floats not a 256 multiple → empty");
    }
}
