//! Q4_K/Q6_K dequant cache — `q4k_ffn_layer` lazily decodes a whole
//! layer to f32 (transposing down from `[hidden, intermediate]` to
//! feature-major), shares the result via `Arc`, and bounds memory
//! via an LRU controlled by `set_q4k_ffn_cache_max_layers`.
//!
//! **The cache is the legacy path.** Production Metal decode bypasses
//! it entirely (`q4k_matmul_transb` streams Q4_K bytes through the
//! GPU). The W2 feature-major down emit (see
//! `format/weights/write_q4k/feature_major_down.rs` + the
//! `q4k_down_feature_scaled_add` dispatch) replaces the cache for
//! per-feature down decode when `down_features_q4k.bin` is present.
//! The cache stays as the fallback for vindexes extracted before
//! W2 landed.
//!
//! Carved out of `ffn_store.rs` in the 2026-04-25 modularity pass.

use crate::index::core::VectorIndex;

impl VectorIndex {
    /// Diagnostic: count of populated `q4k_ffn_cache` slots and the
    /// total f32 bytes they hold. Used by perf probes that need to know
    /// whether a decode actually exercised the dequant cache (the hot
    /// path on Metal does NOT — it streams Q4_K bytes through
    /// `q4k_matmul_transb`). Returns `(populated_slots, bytes)`.
    pub fn q4k_ffn_cache_stats(&self) -> (usize, usize) {
        let cache = self.ffn.q4k_ffn_cache.lock().unwrap();
        let mut slots = 0usize;
        let mut bytes = 0usize;
        for slot in cache.iter() {
            for arc in slot.iter().flatten() {
                slots += 1;
                bytes += arc.len() * std::mem::size_of::<f32>();
            }
        }
        (slots, bytes)
    }

    /// Cap the number of layers held in `q4k_ffn_cache`. Mirror of
    /// `set_gate_cache_max_layers` for the FFN dequant cache. `0`
    /// (default) means unbounded. Setting a smaller cap shrinks the
    /// cache eagerly via the LRU.
    ///
    /// Recommended: `8` for a CPU-only Gemma 3 4B server (≈ 840 MB
    /// down-leg ceiling). Metal-backed runs do not need this — the
    /// full-K fast path bypasses the cache entirely. With W2
    /// feature-major down enabled at extract time, the cache is
    /// only used for non-Q4K interleaved fallback paths and can
    /// be capped at 1.
    pub fn set_q4k_ffn_cache_max_layers(&self, max_layers: usize) {
        self.ffn
            .q4k_ffn_cache_max_layers
            .store(max_layers, std::sync::atomic::Ordering::Relaxed);
        if max_layers > 0 {
            let mut cache = self.ffn.q4k_ffn_cache.lock().unwrap();
            let mut lru = self.ffn.q4k_ffn_cache_lru.lock().unwrap();
            while lru.len() > max_layers {
                if let Some(evict) = lru.pop_back() {
                    if evict < cache.len() {
                        cache[evict] = [None, None, None];
                    }
                }
            }
        }
    }

    /// Record an access to a Q4_K-cached layer and evict if the LRU
    /// has grown beyond `q4k_ffn_cache_max_layers`. Must be called
    /// with `cache` already locked by the caller; `just_inserted` is
    /// true when this call just dequantised a fresh layer.
    fn touch_q4k_ffn_cache_lru(
        &self,
        layer: usize,
        just_inserted: bool,
        cache: &mut [[Option<std::sync::Arc<Vec<f32>>>; 3]],
    ) {
        let max = self
            .ffn
            .q4k_ffn_cache_max_layers
            .load(std::sync::atomic::Ordering::Relaxed);
        if max == 0 {
            return;
        }
        let mut lru = self.ffn.q4k_ffn_cache_lru.lock().unwrap();
        if let Some(pos) = lru.iter().position(|&l| l == layer) {
            lru.remove(pos);
        }
        lru.push_front(layer);
        if just_inserted {
            while lru.len() > max {
                if let Some(evict) = lru.pop_back() {
                    if evict < cache.len() && evict != layer {
                        cache[evict] = [None, None, None];
                    }
                }
            }
        }
    }

    /// Dequantise one Q4K/Q6K FFN matrix on demand, caching the result.
    /// `component`: 0=gate, 1=up, 2=down. Returns `None` when no Q4K
    /// interleaved mmap is loaded. First access per (layer, component)
    /// pays a ~200ms–1s dequant cost (varies with intermediate size);
    /// later accesses are a single `Arc` clone.
    ///
    /// **Memory cost.** Caching a 31B layer's up+down is ~1.85GB of f32
    /// heap. For fine-grained inference prefer [`Self::q4k_ffn_row_into`],
    /// which decodes a single feature into a caller-provided buffer
    /// without populating the cache.
    pub fn q4k_ffn_layer(
        &self,
        layer: usize,
        component: usize,
    ) -> Option<std::sync::Arc<Vec<f32>>> {
        if component > 2 {
            return None;
        }
        {
            let mut cache = self.ffn.q4k_ffn_cache.lock().unwrap();
            if let Some(slot) = cache.get(layer) {
                if let Some(ref arc) = slot[component] {
                    let arc = arc.clone();
                    // Hit — bump LRU but don't evict (just_inserted=false).
                    self.touch_q4k_ffn_cache_lru(layer, false, &mut cache);
                    return Some(arc);
                }
            }
        }
        let slices = self.interleaved_q4k_layer_data(layer)?;
        let (bytes, format) = slices[component];
        let intermediate = self.num_features(layer);
        if intermediate == 0 {
            return None;
        }
        let hidden = self.hidden_size;
        let n = intermediate * hidden;
        let padded = n.div_ceil(256) * 256;
        let info = crate::quant::registry::lookup(format)?;
        let decoded = (info.dequantize)(bytes, padded).ok()?;
        // Gate (0) and up (1) are stored row-major [intermediate, hidden] — row
        // `feat` already contains that feature's weight vector.
        //
        // Down (2) is stored row-major [hidden, intermediate] (the native PyTorch
        // nn.Linear(intermediate, hidden) orientation). To give callers a
        // feature-major view matching gate/up, we transpose here: after the flip
        // arc[feat*hidden..(feat+1)*hidden] is feature `feat`'s down vector.
        let final_data: Vec<f32> = if component == 2 {
            let mut t = vec![0.0f32; n];
            for h in 0..hidden {
                let src_row = &decoded[h * intermediate..(h + 1) * intermediate];
                for (i, &v) in src_row.iter().enumerate() {
                    t[i * hidden + h] = v;
                }
            }
            t
        } else {
            decoded.into_iter().take(n).collect()
        };
        let arc = std::sync::Arc::new(final_data);
        {
            let mut cache = self.ffn.q4k_ffn_cache.lock().unwrap();
            if let Some(slot) = cache.get_mut(layer) {
                slot[component] = Some(arc.clone());
            }
            // Fresh insert — bump LRU and evict if over the cap.
            self.touch_q4k_ffn_cache_lru(layer, true, &mut cache);
        }
        Some(arc)
    }

    /// Cache-based scaled-add — decodes the whole layer (`q4k_ffn_layer`)
    /// on first access, then serves `out += alpha * row` from the cached
    /// feature-major matrix. Required for down: it is stored transposed
    /// on disk (`[hidden, intermediate]`), so a per-row decode reads
    /// hidden-dim rows rather than feature vectors.
    ///
    /// Superseded by `q4k_down_feature_scaled_add` when
    /// `down_features_q4k.bin` is present (W2). Stays here as the
    /// fallback for legacy vindexes.
    #[inline]
    pub fn q4k_ffn_row_scaled_add_via_cache(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        alpha: f32,
        out: &mut [f32],
    ) -> bool {
        let Some(arc) = self.q4k_ffn_layer(layer, component) else {
            return false;
        };
        let hidden = self.hidden_size;
        let row_start = feat * hidden;
        let row_end = row_start + hidden;
        if row_end > arc.len() || out.len() != hidden {
            return false;
        }
        for i in 0..hidden {
            out[i] += alpha * arc[row_start + i];
        }
        true
    }
}
