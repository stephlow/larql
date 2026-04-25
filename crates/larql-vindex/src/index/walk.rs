//! Walk FFN data — mmap'd feature-major down and up projection vectors.
//!
//! Manages down_features.bin and up_features.bin — [intermediate, hidden] per layer,
//! f32 files where each feature's vector is contiguous for zero-copy BLAS access.

use std::sync::Arc;

use crate::error::VindexError;

use super::core::VectorIndex;

use crate::format::filenames::{
    DOWN_FEATURES_BIN, GATE_VECTORS_Q4_BIN, INTERLEAVED_BIN,
    INTERLEAVED_Q4_BIN, INTERLEAVED_Q4K_BIN, INTERLEAVED_Q4K_MANIFEST_JSON,
    UP_FEATURES_BIN,
};
use crate::mmap_util::{mmap_demand_paged, mmap_optimized};

/// Feature store methods for VectorIndex.
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
        self.down_features_mmap = Some(Arc::new(mmap));
        Ok(())
    }

    /// Whether feature-major down vectors are loaded.
    pub fn has_down_features(&self) -> bool {
        self.down_features_mmap.is_some()
    }

    /// Get a feature's contiguous down vector from the mmap'd feature-major file.
    /// Returns `[hidden_size]` f32 slice — zero-copy from mmap.
    pub fn down_feature_vector(&self, layer: usize, feature: usize) -> Option<&[f32]> {
        let mmap = self.down_features_mmap.as_ref()?;
        let intermediate = self.num_features(layer);
        if intermediate == 0 || feature >= intermediate { return None; }

        let layer_floats = intermediate * self.hidden_size;
        let layer_offset = layer * layer_floats * 4;
        let feature_offset = feature * self.hidden_size * 4;
        let start = layer_offset + feature_offset;
        let end = start + self.hidden_size * 4;

        if end > mmap.len() { return None; }

        let data = unsafe {
            let ptr = mmap[start..end].as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, self.hidden_size)
        };
        Some(data)
    }

    /// Get the full down matrix for a layer: [intermediate, hidden] zero-copy view.
    pub fn down_layer_matrix(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        let mmap = self.down_features_mmap.as_ref()?;
        let intermediate = self.num_features(layer);
        if intermediate == 0 { return None; }

        let floats_per_layer = intermediate * self.hidden_size;
        let bytes_per_layer = floats_per_layer * 4;
        let start = layer * bytes_per_layer;
        let end = start + bytes_per_layer;
        if end > mmap.len() { return None; }

        let data = unsafe {
            let ptr = mmap[start..end].as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, floats_per_layer)
        };
        ndarray::ArrayView2::from_shape((intermediate, self.hidden_size), data).ok()
    }

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
        self.up_features_mmap = Some(Arc::new(mmap));
        Ok(())
    }

    /// Get the full up matrix for a layer: [intermediate, hidden] zero-copy view.
    pub fn up_layer_matrix(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        let mmap = self.up_features_mmap.as_ref()?;
        let intermediate = self.num_features(layer);
        if intermediate == 0 { return None; }
        let floats_per_layer = intermediate * self.hidden_size;
        let bytes_per_layer = floats_per_layer * 4;
        let start = layer * bytes_per_layer;
        let end = start + bytes_per_layer;
        if end > mmap.len() { return None; }
        let data = unsafe {
            let ptr = mmap[start..end].as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, floats_per_layer)
        };
        ndarray::ArrayView2::from_shape((intermediate, self.hidden_size), data).ok()
    }

    /// Whether both up and down feature-major mmaps are loaded.
    pub fn has_full_mmap_ffn(&self) -> bool {
        self.down_features_mmap.is_some() && self.up_features_mmap.is_some()
    }

    // ── Interleaved FFN data: gate+up+down packed per layer ──

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
        self.interleaved_mmap = Some(Arc::new(mmap));
        Ok(())
    }

    /// Whether interleaved FFN data is loaded.
    pub fn has_interleaved(&self) -> bool {
        self.interleaved_mmap.is_some()
    }

    /// Get gate matrix for a layer from the interleaved file: [intermediate, hidden].
    pub fn interleaved_gate(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        let mmap = self.interleaved_mmap.as_ref()?;
        let intermediate = self.num_features(layer);
        if intermediate == 0 { return None; }
        let matrix_floats = intermediate * self.hidden_size;
        let matrix_bytes = matrix_floats * 4;
        let layer_bytes = matrix_bytes * 3; // gate + up + down
        let start = layer * layer_bytes; // gate is first
        let end = start + matrix_bytes;
        if end > mmap.len() { return None; }
        let data = unsafe {
            let ptr = mmap[start..end].as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, matrix_floats)
        };
        ndarray::ArrayView2::from_shape((intermediate, self.hidden_size), data).ok()
    }

    /// Get up matrix for a layer from the interleaved file: [intermediate, hidden].
    pub fn interleaved_up(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        let mmap = self.interleaved_mmap.as_ref()?;
        let intermediate = self.num_features(layer);
        if intermediate == 0 { return None; }
        let matrix_floats = intermediate * self.hidden_size;
        let matrix_bytes = matrix_floats * 4;
        let layer_bytes = matrix_bytes * 3;
        let start = layer * layer_bytes + matrix_bytes; // up is second
        let end = start + matrix_bytes;
        if end > mmap.len() { return None; }
        let data = unsafe {
            let ptr = mmap[start..end].as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, matrix_floats)
        };
        ndarray::ArrayView2::from_shape((intermediate, self.hidden_size), data).ok()
    }

    /// Get down matrix for a layer from the interleaved file: [intermediate, hidden].
    pub fn interleaved_down(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        let mmap = self.interleaved_mmap.as_ref()?;
        let intermediate = self.num_features(layer);
        if intermediate == 0 { return None; }
        let matrix_floats = intermediate * self.hidden_size;
        let matrix_bytes = matrix_floats * 4;
        let layer_bytes = matrix_bytes * 3;
        let start = layer * layer_bytes + matrix_bytes * 2; // down is third
        let end = start + matrix_bytes;
        if end > mmap.len() { return None; }
        let data = unsafe {
            let ptr = mmap[start..end].as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, matrix_floats)
        };
        ndarray::ArrayView2::from_shape((intermediate, self.hidden_size), data).ok()
    }

    /// Prefetch next layer's interleaved data into page cache.
    pub fn prefetch_interleaved_layer(&self, layer: usize) {
        #[cfg(unix)]
        if let Some(ref mmap) = self.interleaved_mmap {
            let intermediate = self.num_features(layer);
            if intermediate == 0 { return; }
            let matrix_bytes = intermediate * self.hidden_size * 4;
            let layer_bytes = matrix_bytes * 3;
            let start = layer * layer_bytes;
            let end = (start + layer_bytes).min(mmap.len());
            if start >= mmap.len() { return; }
            unsafe {
                let ptr = mmap[start..].as_ptr() as *mut libc::c_void;
                libc::madvise(ptr, end - start, libc::MADV_WILLNEED);
            }
        }
    }

    // ── Q4 interleaved: quantized gate+up+down per layer ──

    /// Load Q4_0 interleaved FFN data.
    pub fn load_interleaved_q4(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join(INTERLEAVED_Q4_BIN);
        if !path.exists() {
            return Err(VindexError::Parse("interleaved_q4.bin not found".into()));
        }
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { mmap_demand_paged(&file)? };
        self.interleaved_q4_mmap = Some(Arc::new(mmap));
        Ok(())
    }

    pub fn has_interleaved_q4(&self) -> bool {
        self.interleaved_q4_mmap.is_some()
    }

    /// Load Q4_K/Q6_K interleaved FFN data (Ollama-compatible, matches attn format).
    ///
    /// Also reads the optional `interleaved_q4k_manifest.json` sidecar emitted
    /// by the streaming Q4 writer. When the manifest is present callers get
    /// per-matrix layout (offsets, lengths, formats) via
    /// [`VectorIndex::interleaved_q4k_layer_data`]. When it's absent — older
    /// vindexes from `build_q4k_weights.rs` — callers fall back to the legacy
    /// uniform-stride path.
    pub fn load_interleaved_q4k(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join(INTERLEAVED_Q4K_BIN);
        if !path.exists() {
            return Err(VindexError::Parse("interleaved_q4k.bin not found".into()));
        }
        let file = std::fs::File::open(&path)?;
        // Demand-paged: the q4k forward walk reads only the activated features'
        // byte ranges per layer, not the entire 13 GB file.
        let mmap = unsafe { mmap_demand_paged(&file)? };
        self.interleaved_q4k_mmap = Some(Arc::new(mmap));

        let manifest_path = dir.join(INTERLEAVED_Q4K_MANIFEST_JSON);
        if manifest_path.exists() {
            let json: Vec<serde_json::Value> = serde_json::from_str(
                &std::fs::read_to_string(&manifest_path)
                    .map_err(|e| VindexError::Parse(e.to_string()))?,
            )
            .map_err(|e| VindexError::Parse(e.to_string()))?;

            let entries: Vec<(usize, usize, String)> = json
                .iter()
                .map(|e| {
                    let offset = e["offset"].as_u64().unwrap_or(0) as usize;
                    let length = e["length"].as_u64().unwrap_or(0) as usize;
                    let format = e["format"].as_str().unwrap_or("Q4_K").to_string();
                    (offset, length, format)
                })
                .collect();
            self.interleaved_q4k_manifest = Some(entries);
        }
        Ok(())
    }

    pub fn has_interleaved_q4k(&self) -> bool {
        self.interleaved_q4k_mmap.is_some()
    }

    /// Per-layer Q4_K/Q6_K FFN slices — [gate, up, down] with formats.
    ///
    /// Returns `None` when the FFN manifest wasn't present at load time
    /// (caller should fall back to uniform-stride). Returns `Some` iff the
    /// manifest has 3 entries for `layer`; downstream kernels dispatch on
    /// the format string (`"Q4_K"` or `"Q6_K"`).
    pub fn interleaved_q4k_layer_data(&self, layer: usize) -> Option<[(&[u8], &str); 3]> {
        let mmap = self.interleaved_q4k_mmap.as_ref()?;
        let manifest = self.interleaved_q4k_manifest.as_ref()?;
        let base = layer * 3;
        if base + 2 >= manifest.len() {
            return None;
        }
        let mut out: [(&[u8], &str); 3] = [(&[], ""); 3];
        for i in 0..3 {
            let (offset, length, ref format) = manifest[base + i];
            out[i] = (&mmap[offset..offset + length], format.as_str());
        }
        Some(out)
    }

    /// Dequantize one matrix from Q4 interleaved file → f32 Array2.
    /// component: 0=gate, 1=up, 2=down
    fn dequant_q4_matrix(&self, layer: usize, component: usize) -> Option<ndarray::Array2<f32>> {
        let mmap = self.interleaved_q4_mmap.as_ref()?;
        let intermediate = self.num_features(layer);
        if intermediate == 0 { return None; }

        let floats_per_matrix = intermediate * self.hidden_size;
        let q4_bytes_per_matrix = floats_per_matrix / 32 * 18; // Q4_0: 18 bytes per 32 elements
        let q4_bytes_per_layer = q4_bytes_per_matrix * 3;

        let start = layer * q4_bytes_per_layer + component * q4_bytes_per_matrix;
        let end = start + q4_bytes_per_matrix;
        if end > mmap.len() { return None; }

        let q4_data = &mmap[start..end];
        let floats = larql_models::quant::ggml::dequantize_q4_0(q4_data, floats_per_matrix).ok()?;
        ndarray::Array2::from_shape_vec((intermediate, self.hidden_size), floats).ok()
    }

    /// Diagnostic: count of populated `q4k_ffn_cache` slots and the
    /// total f32 bytes they hold. Used by perf probes that need to know
    /// whether a decode actually exercised the dequant cache (the hot
    /// path on Metal does NOT — it streams Q4_K bytes through
    /// `q4k_matmul_transb`). Returns `(populated_slots, bytes)`.
    pub fn q4k_ffn_cache_stats(&self) -> (usize, usize) {
        let cache = self.q4k_ffn_cache.lock().unwrap();
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
    /// full-K fast path bypasses the cache entirely.
    pub fn set_q4k_ffn_cache_max_layers(&self, max_layers: usize) {
        self.q4k_ffn_cache_max_layers
            .store(max_layers, std::sync::atomic::Ordering::Relaxed);
        if max_layers > 0 {
            let mut cache = self.q4k_ffn_cache.lock().unwrap();
            let mut lru = self.q4k_ffn_cache_lru.lock().unwrap();
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
            .q4k_ffn_cache_max_layers
            .load(std::sync::atomic::Ordering::Relaxed);
        if max == 0 {
            return;
        }
        let mut lru = self.q4k_ffn_cache_lru.lock().unwrap();
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
    pub fn q4k_ffn_layer(&self, layer: usize, component: usize)
        -> Option<std::sync::Arc<Vec<f32>>>
    {
        if component > 2 { return None; }
        {
            let mut cache = self.q4k_ffn_cache.lock().unwrap();
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
        if intermediate == 0 { return None; }
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
            let mut cache = self.q4k_ffn_cache.lock().unwrap();
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
    #[inline]
    pub fn q4k_ffn_row_scaled_add_via_cache(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        alpha: f32,
        out: &mut [f32],
    ) -> bool {
        let Some(arc) = self.q4k_ffn_layer(layer, component) else { return false; };
        let hidden = self.hidden_size;
        let row_start = feat * hidden;
        let row_end = row_start + hidden;
        if row_end > arc.len() || out.len() != hidden { return false; }
        for i in 0..hidden {
            out[i] += alpha * arc[row_start + i];
        }
        true
    }

    /// Cache-based dot — same role as `q4k_ffn_row_scaled_add_via_cache`
    /// but for the up leg. Currently unused (up is row-major on disk so
    /// per-row decode is enough); kept for diagnostics and test parity.
    /// If this works and the per-row version doesn't, the bug is in the
    /// row-offset calculation or per-row byte slicing.
    #[inline]
    pub fn q4k_ffn_row_dot_via_cache(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        x: &[f32],
    ) -> Option<f32> {
        let arc = self.q4k_ffn_layer(layer, component)?;
        let hidden = self.hidden_size;
        let row_start = feat * hidden;
        let row_end = row_start + hidden;
        if row_end > arc.len() { return None; }
        let mut acc = 0.0f32;
        for (i, &xv) in x.iter().enumerate() {
            acc += arc[row_start + i] * xv;
        }
        Some(acc)
    }

    /// Direct Q4K/Q6K matmul — Y = X @ W.T, where W is the FFN matrix
    /// stored as Q4K/Q6K bytes in the vindex. Decodes and FMAs fused,
    /// parallelised across W rows. Zero extra RAM (no f32 cache).
    ///
    /// `x` is `[x_rows, w_cols]` row-major. `component` selects the layer's
    /// gate (0) / up (1) / down (2) Q4K slice. On return the output is
    /// `[x_rows, w_rows]` row-major where `w_rows` equals the slice's
    /// shape-0 (intermediate for gate/up, hidden for down).
    ///
    /// Dispatches to the backend's `q4k_matvec` / `q6k_matvec` when a
    /// compute backend is provided (Metal on Apple Silicon, CPU-SIMD
    /// otherwise) — one submission per X row. Falls back to the rayon
    /// + CPU-NEON scalar path when no backend is attached.
    pub fn q4k_matmul_transb(
        &self,
        layer: usize,
        component: usize,
        x: &[f32],
        x_rows: usize,
        backend: Option<&dyn larql_compute::ComputeBackend>,
    ) -> Option<Vec<f32>> {
        use rayon::prelude::*;
        if component > 2 { return None; }
        let slices = self.interleaved_q4k_layer_data(layer)?;
        let (bytes, format) = slices[component];

        let intermediate = self.num_features(layer);
        let hidden = self.hidden_size;
        let (w_rows, w_cols) = match component {
            0 | 1 => (intermediate, hidden),
            2     => (hidden, intermediate),
            _     => return None,
        };
        if x.len() != x_rows * w_cols { return None; }
        if w_cols % 256 != 0 { return None; }

        // Backend per-row dispatch is *slower* than CPU-NEON here because
        // each q4k_matvec call pays a Metal submission (~15 ms). With x_rows
        // × layers × 3 components we'd spend all our time in dispatch.
        // A batched Metal shader (one submission per layer) would fix this,
        // but we don't have it wired yet — keep the hook for future use.
        let _ = backend;

        // Format dispatch via the registry — one lookup, no inline 144/210
        // magic, no silent `_ => 0.0` arm scattered in the hot loop.
        let info = crate::quant::registry::lookup(format)?;
        let row_dot = info.row_dot?;
        let bytes_per_w_row = info.bytes_per_row(w_cols)?;

        // CPU fallback: rayon over W rows, NEON per-row dot.
        let mut y_t = vec![0.0f32; w_rows * x_rows];
        y_t.par_chunks_mut(x_rows).enumerate().for_each(|(j, slot)| {
            let w_row_start = j * bytes_per_w_row;
            let w_row = &bytes[w_row_start..w_row_start + bytes_per_w_row];
            for i in 0..x_rows {
                let x_row = &x[i * w_cols..(i + 1) * w_cols];
                slot[i] = row_dot(w_row, x_row).unwrap_or(0.0);
            }
        });
        let mut y = vec![0.0f32; x_rows * w_rows];
        for j in 0..w_rows {
            let src_base = j * x_rows;
            for i in 0..x_rows {
                y[i * w_rows + j] = y_t[src_base + i];
            }
        }
        Some(y)
    }

    /// Fused Q4K/Q6K decode + dot with `x` for one feature. Returns `None`
    /// if the row isn't available. This is ~2× faster than the
    /// `q4k_ffn_row_into` → BLAS sdot sequence because it skips the Vec
    /// allocation, the intermediate copy, and keeps the decoded data in
    /// registers.
    #[inline]
    pub fn q4k_ffn_row_dot(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        x: &[f32],
    ) -> Option<f32> {
        if component > 2 || x.len() != self.hidden_size { return None; }
        let slices = self.interleaved_q4k_layer_data(layer)?;
        let (bytes, format) = slices[component];
        let hidden = self.hidden_size;
        if feat >= self.num_features(layer) { return None; }
        let info = crate::quant::registry::lookup(format)?;
        let row_dot = info.row_dot?;
        let bytes_per_row = info.bytes_per_row(hidden)?;
        let start = feat * bytes_per_row;
        let end = start + bytes_per_row;
        if end > bytes.len() { return None; }
        row_dot(&bytes[start..end], x).ok()
    }

    /// Fused Q4K/Q6K decode + scaled-add into `out` for one feature.
    /// Counterpart to `q4k_ffn_row_dot` for the down leg.
    #[inline]
    pub fn q4k_ffn_row_scaled_add(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        alpha: f32,
        out: &mut [f32],
    ) -> bool {
        if component > 2 || out.len() != self.hidden_size { return false; }
        let Some(slices) = self.interleaved_q4k_layer_data(layer) else { return false; };
        let (bytes, format) = slices[component];
        let hidden = self.hidden_size;
        if feat >= self.num_features(layer) { return false; }
        let Some(info) = crate::quant::registry::lookup(format) else { return false; };
        let Some(scaled_add) = info.row_scaled_add else { return false; };
        let Some(bytes_per_row) = info.bytes_per_row(hidden) else { return false; };
        let start = feat * bytes_per_row;
        let end = start + bytes_per_row;
        if end > bytes.len() { return false; }
        scaled_add(&bytes[start..end], alpha, out).is_ok()
    }

    /// Decode one row of a Q4K/Q6K FFN matrix directly into `out` without
    /// caching. `component`: 0=gate, 1=up, 2=down; `feat` is the feature
    /// (row) index; `out` must have length `hidden_size`. Returns `false`
    /// when the vindex has no Q4K data or shape is invalid.
    ///
    /// Row-level decode is the small-memory path for very large models
    /// (~30B+) where caching entire dequantised layers blows the RAM
    /// budget. Cost is ~50–70μs per row for hidden≈5376; at K=100 on a
    /// 60-layer model that's ~60 × 100 × 2 decodes × 60μs ≈ 720ms per
    /// forward pass.
    pub fn q4k_ffn_row_into(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        out: &mut [f32],
    ) -> bool {
        if component > 2 || out.len() != self.hidden_size { return false; }
        let Some(slices) = self.interleaved_q4k_layer_data(layer) else { return false; };
        let (bytes, format) = slices[component];
        let hidden = self.hidden_size;
        if feat >= self.num_features(layer) { return false; }

        let Some(info) = crate::quant::registry::lookup(format) else { return false; };
        let Some(bytes_per_row) = info.bytes_per_row(hidden) else { return false; };
        let start = feat * bytes_per_row;
        let end = start + bytes_per_row;
        if end > bytes.len() { return false; }
        match (info.dequantize)(&bytes[start..end], hidden) {
            Ok(v) => { out.copy_from_slice(&v[..hidden]); true }
            Err(_) => false,
        }
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
        if let Some(ref mmap) = self.interleaved_q4_mmap {
            let intermediate = self.num_features(layer);
            if intermediate == 0 { return; }
            let q4_bytes_per_matrix = intermediate * self.hidden_size / 32 * 18;
            let q4_bytes_per_layer = q4_bytes_per_matrix * 3;
            let start = layer * q4_bytes_per_layer;
            let end = (start + q4_bytes_per_layer).min(mmap.len());
            if start >= mmap.len() { return; }
            unsafe {
                let ptr = mmap[start..].as_ptr() as *mut libc::c_void;
                libc::madvise(ptr, end - start, libc::MADV_WILLNEED);
            }
        }
    }

    /// Prefetch next layer's Q4_K/Q6_K FFN data into the page cache via
    /// MADV_WILLNEED. Counterpart of [`Self::prefetch_interleaved_q4_layer`].
    /// Issues one madvise spanning the layer's gate+up+down matrices.
    ///
    /// When the FFN manifest is loaded (the streaming-writer path), the
    /// span is computed from the layer's three manifest entries — handles
    /// mixed Q4_K/Q6_K layouts where down may be Q6_K (210 B/256) while
    /// gate/up are Q4_K (144 B/256). Without a manifest, falls back to
    /// the legacy uniform Q4_K stride (144 B/256 across all three
    /// matrices) — matches the build_q4k_weights writer.
    pub fn prefetch_interleaved_q4k_layer(&self, layer: usize) {
        #[cfg(unix)]
        if let Some(ref mmap) = self.interleaved_q4k_mmap {
            let intermediate = self.num_features(layer);
            if intermediate == 0 { return; }
            let (start, len) = if let Some(ref manifest) = self.interleaved_q4k_manifest {
                let base = layer * 3;
                if base + 2 >= manifest.len() { return; }
                let s = manifest[base].0;
                let (last_off, last_len, _) = &manifest[base + 2];
                let e = (last_off + last_len).min(mmap.len());
                if s >= mmap.len() || e <= s { return; }
                (s, e - s)
            } else {
                // Uniform-stride fallback: matches build_q4k_weights's
                // Q4_K-only writer. Q4_K is 144 bytes per 256 elements.
                let blocks_per_matrix = intermediate * self.hidden_size / 256;
                let bytes_per_matrix = blocks_per_matrix * 144;
                let bytes_per_layer = bytes_per_matrix * 3;
                let s = layer * bytes_per_layer;
                let e = (s + bytes_per_layer).min(mmap.len());
                if s >= mmap.len() || e <= s { return; }
                (s, e - s)
            };
            unsafe {
                let ptr = mmap[start..].as_ptr() as *mut libc::c_void;
                libc::madvise(ptr, len, libc::MADV_WILLNEED);
            }
        }
    }

    // warmup() is in gate.rs (it's a gate cache operation)

    // ── Q4 gate vectors for fast KNN via larql-compute ──

    /// Load Q4_0 gate vectors from gate_vectors_q4.bin.
    ///
    /// File layout: layers packed contiguously, each layer is
    /// [num_features × hidden] in Q4_0 format (18 bytes per 32 elements).
    /// The per-layer feature count comes from gate_mmap_slices (must load
    /// f32/f16 gates first for the slice metadata, or pass feature counts).
    pub fn load_gate_vectors_q4(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join(GATE_VECTORS_Q4_BIN);
        if !path.exists() {
            return Err(VindexError::Parse("gate_vectors_q4.bin not found".into()));
        }
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { mmap_optimized(&file)? };

        // Compute per-layer byte offsets from feature counts
        let mut slices = Vec::with_capacity(self.num_layers);
        let mut offset = 0usize;
        for layer in 0..self.num_layers {
            let num_features = self.num_features(layer);
            let floats = num_features * self.hidden_size;
            let q4_bytes = floats / 32 * 18; // Q4_0: 18 bytes per 32 elements
            slices.push(super::types::GateQ4Slice {
                byte_offset: offset,
                byte_len: q4_bytes,
                num_features,
            });
            offset += q4_bytes;
        }

        self.gate_q4_mmap = Some(Arc::new(mmap));
        self.gate_q4_slices = slices;
        Ok(())
    }

    /// Whether Q4 gate vectors are loaded.
    pub fn has_gate_q4(&self) -> bool {
        self.gate_q4_mmap.is_some()
    }

    /// Get Q4 data slice for a layer's gate vectors. Returns the raw Q4_0 bytes.
    pub fn gate_q4_data(&self, layer: usize) -> Option<&[u8]> {
        let mmap = self.gate_q4_mmap.as_ref()?;
        let slice = self.gate_q4_slices.get(layer)?;
        if slice.byte_len == 0 { return None; }
        let end = slice.byte_offset + slice.byte_len;
        if end > mmap.len() { return None; }
        Some(&mmap[slice.byte_offset..end])
    }

    // ── FP4 / FP8 FFN storage (exp 26) ────────────────────────────────────

    /// Load FP4 / FP8 FFN projection mmaps from `dir` using the `fp4`
    /// manifest in `config`. Non-fatal: if `config.fp4` is None, no
    /// storage is attached and the method returns Ok. Errors on
    /// malformed manifests (e.g. file sizes that don't match the
    /// per-layer feature counts).
    pub fn load_fp4_storage(
        &mut self,
        dir: &std::path::Path,
        config: &crate::config::types::VindexConfig,
    ) -> Result<(), VindexError> {
        let Some(ref manifest) = config.fp4 else { return Ok(()); };
        let layer_features: Vec<usize> = config.layers.iter().map(|l| l.num_features).collect();
        let storage = super::fp4_storage::Fp4Storage::load(
            dir,
            manifest.clone(),
            layer_features,
            config.hidden_size,
        )?;
        self.fp4_storage = Some(std::sync::Arc::new(storage));
        Ok(())
    }

    /// Whether FP4/FP8 FFN storage is attached.
    pub fn has_fp4_storage(&self) -> bool {
        self.fp4_storage.is_some()
    }

    /// Fused dequant + dot for one FFN feature when FP4/FP8 storage is
    /// attached. `component` is 0=gate, 1=up, 2=down. Returns `None`
    /// if no FP4 storage is attached, if the projection is stored in
    /// f16/f32 (caller falls back to the legacy path), or if the
    /// coordinates are out of range.
    #[inline]
    pub fn fp4_ffn_row_dot(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        x: &[f32],
    ) -> Option<f32> {
        let fp4 = self.fp4_storage.as_ref()?;
        fp4.row_dot(layer, component, feat, x)
    }

    /// Fused dequant + scaled-add for the FP4/FP8 path.
    #[inline]
    pub fn fp4_ffn_row_scaled_add(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        alpha: f32,
        out: &mut [f32],
    ) -> bool {
        let Some(fp4) = self.fp4_storage.as_ref() else { return false; };
        fp4.row_scaled_add(layer, component, feat, alpha, out)
    }

    /// Dequantise one FFN feature into the caller's buffer (FP4/FP8 path).
    /// Counterpart of `q4k_ffn_row_into`.
    #[inline]
    pub fn fp4_ffn_row_into(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        out: &mut [f32],
    ) -> bool {
        let Some(fp4) = self.fp4_storage.as_ref() else { return false; };
        fp4.dequant_row_into(layer, component, feat, out)
    }
}
