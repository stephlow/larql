//! FFN storage — mmap loaders, accessors, prefetchers, and the
//! Q4_K/Q6_K dequant cache. Compute-side codec dispatch (matmul +
//! row-level fused decode) lives in
//! `crate::index::compute::q4k_dispatch`.
//!
//! Files managed:
//! - `down_features.bin` / `up_features.bin` — feature-major f32
//!   projections; zero-copy BLAS slicing.
//! - `interleaved.bin` (f32) and `interleaved_q4{,k}.bin` — packed
//!   gate/up/down per layer.
//! - Q4_0 gate-vector mmap, FP4/FP8 storage handle.
//!
//! The cache (`q4k_ffn_cache`) is bounded by
//! `set_q4k_ffn_cache_max_layers`; only the CPU per-position fallback
//! populates it (Metal full-K decode streams Q4_K bytes through
//! `compute::q4k_dispatch::q4k_matmul_transb`).

use std::sync::{Arc, Mutex};

use crate::error::VindexError;

use crate::index::core::VectorIndex;

use crate::format::filenames::{
    DOWN_FEATURES_BIN, GATE_VECTORS_Q4_BIN, INTERLEAVED_BIN,
    INTERLEAVED_Q4_BIN, INTERLEAVED_Q4K_BIN, INTERLEAVED_Q4K_MANIFEST_JSON,
    UP_FEATURES_BIN,
};
use crate::mmap_util::{mmap_demand_paged, mmap_optimized};

// ── FfnStore composed-substore ─────────────────────────────────────────

/// Per-layer Q4_K/Q6_K FFN dequant cache: outer index = layer, inner array =
/// `[gate, up, down]`. `Arc` shares the decoded matrix across `VectorIndex`
/// clones; `Mutex` guards LRU eviction.
pub type Q4kFfnCache = Mutex<Vec<[Option<Arc<Vec<f32>>>; 3]>>;

pub struct FfnStore {
    /// Feature-major down projections (f32 mmap).
    pub down_features_mmap: Option<Arc<memmap2::Mmap>>,
    /// Feature-major up projections (f32 mmap).
    pub up_features_mmap: Option<Arc<memmap2::Mmap>>,
    /// Interleaved [gate|up|down] FFN data (f32, packed per layer).
    pub interleaved_mmap: Option<Arc<memmap2::Mmap>>,
    /// Q4_0 quantized interleaved FFN.
    pub interleaved_q4_mmap: Option<Arc<memmap2::Mmap>>,
    /// Q4_K / Q6_K quantized interleaved FFN (Ollama-compatible).
    pub interleaved_q4k_mmap: Option<Arc<memmap2::Mmap>>,
    /// Per-matrix (offset, length, format) entries — 3 per layer in
    /// `[gate, up, down]` order.
    pub interleaved_q4k_manifest: Option<Vec<(usize, usize, String)>>,
    /// Per-layer lazy dequant cache for Q4_K/Q6_K FFN tensors.
    /// `q4k_ffn_cache[layer][c]` is the dequantised
    /// `[intermediate × hidden]` matrix for component `c`
    /// (0=gate, 1=up, 2=down). LRU-bounded by
    /// `q4k_ffn_cache_max_layers`.
    pub q4k_ffn_cache: Q4kFfnCache,
    /// LRU of layers held in `q4k_ffn_cache`. Front = newest.
    pub q4k_ffn_cache_lru: Mutex<std::collections::VecDeque<usize>>,
    /// Cap on `q4k_ffn_cache`. 0 = unlimited (default).
    pub q4k_ffn_cache_max_layers: std::sync::atomic::AtomicUsize,
    /// FP4 / FP8 FFN storage (exp 26).
    pub fp4_storage: Option<Arc<crate::index::fp4_storage::Fp4Storage>>,
}

impl FfnStore {
    pub fn empty(num_layers: usize) -> Self {
        Self {
            down_features_mmap: None,
            up_features_mmap: None,
            interleaved_mmap: None,
            interleaved_q4_mmap: None,
            interleaved_q4k_mmap: None,
            interleaved_q4k_manifest: None,
            q4k_ffn_cache: Mutex::new(
                (0..num_layers).map(|_| [None, None, None]).collect(),
            ),
            q4k_ffn_cache_lru: Mutex::new(std::collections::VecDeque::new()),
            q4k_ffn_cache_max_layers: std::sync::atomic::AtomicUsize::new(0),
            fp4_storage: None,
        }
    }
}

impl Clone for FfnStore {
    fn clone(&self) -> Self {
        use std::sync::atomic::Ordering;
        let nl = self
            .q4k_ffn_cache
            .lock()
            .map(|c| c.len())
            .unwrap_or(0);
        Self {
            down_features_mmap: self.down_features_mmap.clone(),
            up_features_mmap: self.up_features_mmap.clone(),
            interleaved_mmap: self.interleaved_mmap.clone(),
            interleaved_q4_mmap: self.interleaved_q4_mmap.clone(),
            interleaved_q4k_mmap: self.interleaved_q4k_mmap.clone(),
            interleaved_q4k_manifest: self.interleaved_q4k_manifest.clone(),
            q4k_ffn_cache: Mutex::new(
                (0..nl).map(|_| [None, None, None]).collect(),
            ),
            q4k_ffn_cache_lru: Mutex::new(std::collections::VecDeque::new()),
            q4k_ffn_cache_max_layers: std::sync::atomic::AtomicUsize::new(
                self.q4k_ffn_cache_max_layers.load(Ordering::Relaxed),
            ),
            fp4_storage: self.fp4_storage.clone(),
        }
    }
}

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
        let mmap = self.ffn.down_features_mmap.as_ref()?;
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
        self.ffn.up_features_mmap = Some(Arc::new(mmap));
        Ok(())
    }

    /// Get the full up matrix for a layer: [intermediate, hidden] zero-copy view.
    pub fn up_layer_matrix(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        let mmap = self.ffn.up_features_mmap.as_ref()?;
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
        self.ffn.down_features_mmap.is_some() && self.ffn.up_features_mmap.is_some()
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
        let mmap = self.ffn.interleaved_mmap.as_ref()?;
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
        let mmap = self.ffn.interleaved_mmap.as_ref()?;
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
        if let Some(ref mmap) = self.ffn.interleaved_mmap {
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
        self.ffn.interleaved_q4_mmap = Some(Arc::new(mmap));
        Ok(())
    }

    pub fn has_interleaved_q4(&self) -> bool {
        self.ffn.interleaved_q4_mmap.is_some()
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
        self.ffn.interleaved_q4k_mmap = Some(Arc::new(mmap));

        let manifest_path = dir.join(INTERLEAVED_Q4K_MANIFEST_JSON);
        if manifest_path.exists() {
            let json: Vec<serde_json::Value> = serde_json::from_str(
                &std::fs::read_to_string(&manifest_path)
                    .map_err(|e| VindexError::Parse(e.to_string()))?,
            )
            .map_err(|e| VindexError::Parse(e.to_string()))?;

            // Format is required. The previous `unwrap_or("Q4_K")`
            // default silently masked malformed manifests — see
            // ROADMAP P0 "Replace unwrap_or(Q4_K) silent fallbacks".
            let entries: Vec<(usize, usize, String)> = json
                .iter()
                .map(|e| {
                    let offset = e["offset"].as_u64().unwrap_or(0) as usize;
                    let length = e["length"].as_u64().unwrap_or(0) as usize;
                    let tag = e["format"].as_str().ok_or_else(|| VindexError::Parse(
                        "interleaved_q4k_manifest entry missing `format` field".into(),
                    ))?;
                    if crate::quant::registry::lookup(tag).is_none() {
                        return Err(VindexError::Parse(format!(
                            "interleaved_q4k_manifest: unknown format tag {tag:?} \
                             — quant::registry has no entry"
                        )));
                    }
                    Ok((offset, length, tag.to_string()))
                })
                .collect::<Result<Vec<_>, VindexError>>()?;
            self.ffn.interleaved_q4k_manifest = Some(entries);
        }
        Ok(())
    }

    pub fn has_interleaved_q4k(&self) -> bool {
        self.ffn.interleaved_q4k_mmap.is_some()
    }

    /// Per-layer Q4_K/Q6_K FFN slices — [gate, up, down] with formats.
    ///
    /// Returns `None` when the FFN manifest wasn't present at load time
    /// (caller should fall back to uniform-stride). Returns `Some` iff the
    /// manifest has 3 entries for `layer`; downstream kernels dispatch on
    /// the format string (`"Q4_K"` or `"Q6_K"`).
    pub fn interleaved_q4k_layer_data(&self, layer: usize) -> Option<[(&[u8], &str); 3]> {
        let mmap = self.ffn.interleaved_q4k_mmap.as_ref()?;
        let manifest = self.ffn.interleaved_q4k_manifest.as_ref()?;
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
        let mmap = self.ffn.interleaved_q4_mmap.as_ref()?;
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
    /// full-K fast path bypasses the cache entirely.
    pub fn set_q4k_ffn_cache_max_layers(&self, max_layers: usize) {
        self.ffn.q4k_ffn_cache_max_layers
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
        let max = self.ffn.q4k_ffn_cache_max_layers
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
    pub fn q4k_ffn_layer(&self, layer: usize, component: usize)
        -> Option<std::sync::Arc<Vec<f32>>>
    {
        if component > 2 { return None; }
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
        if let Some(ref mmap) = self.ffn.interleaved_q4k_mmap {
            let intermediate = self.num_features(layer);
            if intermediate == 0 { return; }
            let (start, len) = if let Some(ref manifest) = self.ffn.interleaved_q4k_manifest {
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
            slices.push(crate::index::types::GateQ4Slice {
                byte_offset: offset,
                byte_len: q4_bytes,
                num_features,
            });
            offset += q4_bytes;
        }

        self.gate.gate_q4_mmap = Some(Arc::new(mmap));
        self.gate.gate_q4_slices = slices;
        Ok(())
    }

    /// Whether Q4 gate vectors are loaded.
    pub fn has_gate_q4(&self) -> bool {
        self.gate.gate_q4_mmap.is_some()
    }

    /// Get Q4 data slice for a layer's gate vectors. Returns the raw Q4_0 bytes.
    pub fn gate_q4_data(&self, layer: usize) -> Option<&[u8]> {
        let mmap = self.gate.gate_q4_mmap.as_ref()?;
        let slice = self.gate.gate_q4_slices.get(layer)?;
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
        let storage = super::fp4_store::Fp4Storage::load(
            dir,
            manifest.clone(),
            layer_features,
            config.hidden_size,
        )?;
        self.ffn.fp4_storage = Some(std::sync::Arc::new(storage));
        Ok(())
    }

    /// Whether FP4/FP8 FFN storage is attached.
    pub fn has_fp4_storage(&self) -> bool {
        self.ffn.fp4_storage.is_some()
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
        let fp4 = self.ffn.fp4_storage.as_ref()?;
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
        let Some(fp4) = self.ffn.fp4_storage.as_ref() else { return false; };
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
        let Some(fp4) = self.ffn.fp4_storage.as_ref() else { return false; };
        fp4.dequant_row_into(layer, component, feat, out)
    }
}
