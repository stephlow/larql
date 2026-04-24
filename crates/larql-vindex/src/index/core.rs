//! VectorIndex struct and core operations.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use ndarray::Array2;

// Re-export all shared types from types.rs.
pub use super::types::*;

/// The full model as a local vector index.
///
/// Gate vectors for KNN matching + down token metadata for output lookup.
/// Supports two storage modes:
/// - **Heap**: gate vectors copied into per-layer Array2 (in-memory builds, mutations)
/// - **Mmap**: gate vectors sliced directly from mmap'd file (zero-copy, zero heap)
pub struct VectorIndex {
    /// Per-layer gate vectors (heap mode): gate_vectors[layer] is (num_features, hidden_size).
    pub(crate) gate_vectors: Vec<Option<Array2<f32>>>,

    /// Mmap'd gate vector bytes (zero-copy mode). When set, gate_knn slices
    /// directly from this instead of using gate_vectors heap arrays.
    /// For f32: bytes are reinterpreted as &[f32] directly (zero-copy).
    /// For f16: bytes are decoded per-layer on demand.
    /// Arc for Clone support — the mmap is shared, not copied.
    pub(crate) gate_mmap_bytes: Option<Arc<memmap2::Mmap>>,

    /// Storage dtype for mmap'd data (needed for f16 decoding).
    pub(crate) gate_mmap_dtype: crate::config::dtype::StorageDtype,

    /// Per-layer slice info for mmap mode.
    pub(crate) gate_mmap_slices: Vec<GateLayerSlice>,

    /// Per-layer, per-feature output token metadata from down projections.
    /// down_meta[layer][feature] = FeatureMeta with top tokens.
    /// Heap mode: populated during builds or when loaded from JSONL.
    pub(crate) down_meta: Vec<Option<Vec<Option<FeatureMeta>>>>,

    /// Mmap'd down_meta.bin bytes (zero-copy mode).
    /// When set, feature_meta() reads records on demand from the mmap.
    pub(crate) down_meta_mmap: Option<Arc<DownMetaMmap>>,

    /// Number of layers in the model.
    pub num_layers: usize,

    /// Hidden dimension.
    pub hidden_size: usize,

    /// Down vector overrides: custom output vectors for specific features.
    /// When set, sparse_ffn_forward uses this instead of the model's down weight row.
    /// Key: (layer, feature), Value: hidden_size f32 vector.
    pub(crate) down_overrides: HashMap<(usize, usize), Vec<f32>>,

    /// Up vector overrides: custom up vectors for specific features.
    /// Parallel to down_overrides — when set, walk_ffn_sparse uses this
    /// instead of the model's up_features row at that slot. INSERT
    /// writes to this so the slot's activation = silu(gate·x) * (up·x)
    /// reflects the constellation, not the original weak free-slot up.
    /// Key: (layer, feature), Value: hidden_size f32 vector.
    pub(crate) up_overrides: HashMap<(usize, usize), Vec<f32>>,

    /// Lazy decode cache for f16 gate vectors. Each layer decoded once on first
    /// KNN call, then reused. Eliminates repeated f16→f32 conversion.
    pub(crate) f16_decode_cache: Mutex<Vec<Option<Vec<f32>>>>,
    /// LRU queue for `f16_decode_cache`. Back is oldest, front is newest.
    /// Used with `gate_cache_max_layers` to cap decoded-gate heap growth
    /// (a 31B f16 gate table decodes to ~26 GB if all 60 layers are kept).
    pub(crate) gate_cache_lru: Mutex<std::collections::VecDeque<usize>>,
    /// Cap on live entries in `f16_decode_cache`. 0 = unlimited (default —
    /// historical behaviour, max speed). Set via `set_gate_cache_max_layers`
    /// to bound RSS growth. When an insert would exceed the cap, the
    /// least-recently-used layer is dropped.
    pub(crate) gate_cache_max_layers: std::sync::atomic::AtomicUsize,
    pub(crate) warmed_gates: std::sync::RwLock<Vec<Option<Vec<f32>>>>,
    pub(crate) down_features_mmap: Option<Arc<memmap2::Mmap>>,
    pub(crate) up_features_mmap: Option<Arc<memmap2::Mmap>>,
    pub(crate) hnsw_cache: Mutex<Vec<Option<super::hnsw::HnswLayer>>>,
    pub(crate) hnsw_enabled: std::sync::atomic::AtomicBool,
    pub(crate) hnsw_ef_search: std::sync::atomic::AtomicUsize,
    /// Mmap'd lm_head (output projection): [vocab_size, hidden_size], f32.
    pub(crate) lm_head_mmap: Option<Arc<memmap2::Mmap>>,
    /// Mmap'd lm_head as f16 — typically the tied-embedding case where the
    /// vindex's `embeddings.bin` is the output projection. Carried by
    /// `VectorIndex` so `lm_head_knn_backend` can dispatch to Metal's
    /// `f16_gemv` without materialising a 5.6 GB f32 clone on 31B.
    pub(crate) lm_head_f16_mmap: Option<Arc<memmap2::Mmap>>,
    pub vocab_size: usize,
    /// Interleaved FFN data: [gate|up|down] per layer in one contiguous file.
    pub(crate) interleaved_mmap: Option<Arc<memmap2::Mmap>>,
    /// Q4_0 quantized interleaved FFN data (7x smaller, dequant on read).
    pub(crate) interleaved_q4_mmap: Option<Arc<memmap2::Mmap>>,
    /// Q4_K/Q6_K quantized interleaved FFN data (Ollama-compatible, matches attn format).
    pub(crate) interleaved_q4k_mmap: Option<Arc<memmap2::Mmap>>,
    /// Per-matrix (offset, length, format) entries for `interleaved_q4k.bin`,
    /// 3 per layer in [gate, up, down] order. Required because the Ollama
    /// strategy mixes Q4_K (gate/up) with Q6_K (down), so layer stride is
    /// not uniform and callers cannot compute offsets from shape alone.
    pub(crate) interleaved_q4k_manifest: Option<Vec<(usize, usize, String)>>,
    /// Per-layer lazy decode cache for Q4K/Q6K FFN tensors.
    /// `q4k_ffn_cache[layer][c]` is the dequantised `[intermediate × hidden]`
    /// matrix for component `c` (0=gate, 1=up, 2=down). Populated on first
    /// access via `q4k_ffn_layer`. Backs `walk_ffn_sparse`'s f32 view when
    /// no native f32 mmap exists (Q4K-only vindexes).
    #[allow(clippy::type_complexity)]
    pub(crate) q4k_ffn_cache: Mutex<Vec<[Option<Arc<Vec<f32>>>; 3]>>,

    /// Layer range owned by this index instance (start inclusive, end exclusive).
    /// `None` means all layers are owned (default, no sharding).
    /// Set via `load_vindex_with_range` to restrict which layers are served,
    /// preventing accidental page faults into out-of-shard mmap regions.
    pub(crate) layer_range: Option<(usize, usize)>,

    /// Q4_0 gate vectors mmap — for fast Q4 KNN via larql-compute.
    pub(crate) gate_q4_mmap: Option<Arc<memmap2::Mmap>>,
    /// Per-layer byte offset + byte length in gate_q4_mmap.
    pub(crate) gate_q4_slices: Vec<GateQ4Slice>,
    /// Q4_0 lm_head mmap — for GPU Q4 logits (replaces CPU f32 lm_head KNN).
    pub(crate) lm_head_q4_mmap: Option<Arc<memmap2::Mmap>>,
    /// Q4_0 lm_head synthesized in RAM from f16 embeddings at load time.
    pub(crate) lm_head_q4_synth: Option<Arc<Vec<u8>>>,
    /// Q4_K/Q6_K attention weights (Ollama-compatible).
    pub(crate) attn_q4k_mmap: Option<Arc<memmap2::Mmap>>,
    pub(crate) attn_q4k_manifest: Option<Vec<(usize, usize, String)>>,
    /// Q4_0 attention weights mmap — for GPU full pipeline.
    pub(crate) attn_q4_mmap: Option<Arc<memmap2::Mmap>>,
    /// Per-matrix (offset, length) in attn_q4_mmap — from manifest.
    pub(crate) attn_q4_manifest: Option<Vec<(usize, usize)>>,
    /// Q8_0 attention weights mmap — higher precision for attention projections.
    pub(crate) attn_q8_mmap: Option<Arc<memmap2::Mmap>>,
    /// Per-matrix (offset, vals_len, scales_len) in attn_q8_mmap.
    pub(crate) attn_q8_manifest: Option<Vec<(usize, usize, usize)>>,

    /// FP4/FP8 FFN storage (exp 26). Set by `load_fp4_storage` when
    /// `index.json` carries an `fp4` manifest. When present, the walk
    /// kernel should dispatch through the FP4 accessors in preference
    /// to the legacy f16/f32 path.
    pub(crate) fp4_storage: Option<Arc<super::fp4_storage::Fp4Storage>>,
}

impl Clone for VectorIndex {
    /// Clones share mmap/Arc/Vec state with the source, but rebuild the
    /// per-clone caches (`f16_decode_cache`, `gate_cache_lru`, `warmed_gates`,
    /// `hnsw_cache`, `q4k_ffn_cache`) because Mutex/RwLock aren't cloneable
    /// and their contents are per-instance working memory anyway. Atomics
    /// are rebuilt holding the source's current value.
    ///
    /// Fresh-state fields (the caches) are filled by `Self::empty(..)`;
    /// this impl only lists fields whose values are copied from `self`.
    /// Adding a new Arc-like / Vec / Copy-scalar field means appending
    /// one line here. Adding a new Mutex/RwLock field means updating
    /// only `Self::empty`.
    fn clone(&self) -> Self {
        use std::sync::atomic::Ordering;
        Self {
            gate_vectors: self.gate_vectors.clone(),
            gate_mmap_bytes: self.gate_mmap_bytes.clone(),
            gate_mmap_dtype: self.gate_mmap_dtype,
            gate_mmap_slices: self.gate_mmap_slices.clone(),
            down_meta: self.down_meta.clone(),
            down_meta_mmap: self.down_meta_mmap.clone(),
            down_overrides: self.down_overrides.clone(),
            up_overrides: self.up_overrides.clone(),
            gate_cache_max_layers: std::sync::atomic::AtomicUsize::new(
                self.gate_cache_max_layers.load(Ordering::Relaxed),
            ),
            down_features_mmap: self.down_features_mmap.clone(),
            up_features_mmap: self.up_features_mmap.clone(),
            hnsw_enabled: std::sync::atomic::AtomicBool::new(
                self.hnsw_enabled.load(Ordering::Relaxed),
            ),
            hnsw_ef_search: std::sync::atomic::AtomicUsize::new(
                self.hnsw_ef_search.load(Ordering::Relaxed),
            ),
            lm_head_mmap: self.lm_head_mmap.clone(),
            lm_head_f16_mmap: self.lm_head_f16_mmap.clone(),
            vocab_size: self.vocab_size,
            interleaved_mmap: self.interleaved_mmap.clone(),
            interleaved_q4_mmap: self.interleaved_q4_mmap.clone(),
            interleaved_q4k_mmap: self.interleaved_q4k_mmap.clone(),
            interleaved_q4k_manifest: self.interleaved_q4k_manifest.clone(),
            gate_q4_mmap: self.gate_q4_mmap.clone(),
            gate_q4_slices: self.gate_q4_slices.clone(),
            lm_head_q4_mmap: self.lm_head_q4_mmap.clone(),
            lm_head_q4_synth: self.lm_head_q4_synth.clone(),
            attn_q4k_mmap: self.attn_q4k_mmap.clone(),
            attn_q4k_manifest: self.attn_q4k_manifest.clone(),
            attn_q4_mmap: self.attn_q4_mmap.clone(),
            attn_q4_manifest: self.attn_q4_manifest.clone(),
            attn_q8_mmap: self.attn_q8_mmap.clone(),
            attn_q8_manifest: self.attn_q8_manifest.clone(),
            layer_range: self.layer_range,
            fp4_storage: self.fp4_storage.clone(),
            // Everything else — including the Mutex/RwLock caches and
            // the fields also covered explicitly above — uses empty's
            // ground state. Explicit fields listed before this line
            // override empty's defaults (Rust struct FRU semantics).
            ..Self::empty(self.num_layers, self.hidden_size)
        }
    }
}

impl VectorIndex {
    /// Private constructor for the "nothing loaded" state. Every field
    /// is set to its default inert value — Options are `None`, Vecs are
    /// empty or `vec![None; num_layers]` where per-layer slots are
    /// required, caches are freshly allocated Mutex/RwLock/Atomic. The
    /// other `new_*` constructors and `Clone` use `..Self::empty(..)`
    /// to express only the fields they actually set.
    ///
    /// **Single source of truth for new field defaults.** Adding a
    /// field to `VectorIndex` now requires updating the struct
    /// definition and this function. Constructors don't need to change.
    pub(crate) fn empty(num_layers: usize, hidden_size: usize) -> Self {
        Self {
            gate_vectors: vec![None; num_layers],
            gate_mmap_bytes: None,
            gate_mmap_dtype: crate::config::dtype::StorageDtype::F32,
            gate_mmap_slices: Vec::new(),
            down_meta: vec![None; num_layers],
            down_meta_mmap: None,
            num_layers,
            hidden_size,
            down_overrides: HashMap::new(),
            up_overrides: HashMap::new(),
            f16_decode_cache: Mutex::new(vec![None; num_layers]),
            gate_cache_lru: Mutex::new(std::collections::VecDeque::new()),
            gate_cache_max_layers: std::sync::atomic::AtomicUsize::new(0),
            warmed_gates: std::sync::RwLock::new(vec![None; num_layers]),
            down_features_mmap: None,
            up_features_mmap: None,
            hnsw_cache: Mutex::new((0..num_layers).map(|_| None).collect()),
            hnsw_enabled: std::sync::atomic::AtomicBool::new(false),
            hnsw_ef_search: std::sync::atomic::AtomicUsize::new(200),
            lm_head_mmap: None,
            lm_head_f16_mmap: None,
            vocab_size: 0,
            interleaved_mmap: None,
            interleaved_q4_mmap: None,
            interleaved_q4k_mmap: None,
            interleaved_q4k_manifest: None,
            q4k_ffn_cache: Mutex::new((0..num_layers).map(|_| [None, None, None]).collect()),
            layer_range: None,
            gate_q4_mmap: None,
            gate_q4_slices: Vec::new(),
            lm_head_q4_mmap: None,
            lm_head_q4_synth: None,
            attn_q4k_mmap: None,
            attn_q4k_manifest: None,
            attn_q4_mmap: None,
            attn_q4_manifest: None,
            attn_q8_mmap: None,
            attn_q8_manifest: None,
            fp4_storage: None,
        }
    }

    /// Create a new VectorIndex from heap-allocated components (in-memory builds).
    pub fn new(
        gate_vectors: Vec<Option<Array2<f32>>>,
        down_meta: Vec<Option<Vec<Option<FeatureMeta>>>>,
        num_layers: usize,
        hidden_size: usize,
    ) -> Self {
        Self {
            gate_vectors,
            down_meta,
            ..Self::empty(num_layers, hidden_size)
        }
    }

    /// Create a VectorIndex with zero-copy mmap'd gate vectors and down_meta.
    /// No heap allocation — everything read on demand from mmap'd files.
    pub fn new_mmap(
        gate_mmap: memmap2::Mmap,
        gate_slices: Vec<GateLayerSlice>,
        dtype: crate::config::dtype::StorageDtype,
        down_meta_mmap: Option<DownMetaMmap>,
        num_layers: usize,
        hidden_size: usize,
    ) -> Self {
        Self {
            gate_mmap_bytes: Some(Arc::new(gate_mmap)),
            gate_mmap_dtype: dtype,
            gate_mmap_slices: gate_slices,
            down_meta_mmap: down_meta_mmap.map(Arc::new),
            ..Self::empty(num_layers, hidden_size)
        }
    }

    /// Returns true if this index uses mmap'd gate vectors (zero heap copy).
    pub fn is_mmap(&self) -> bool {
        self.gate_mmap_bytes.is_some()
    }

    /// Estimated heap bytes used by gate vectors (0 if mmap'd).
    pub fn gate_heap_bytes(&self) -> usize {
        if self.is_mmap() {
            return 0;
        }
        self.gate_vectors.iter()
            .filter_map(|v| v.as_ref())
            .map(|m| m.len() * std::mem::size_of::<f32>())
            .sum()
    }

    /// Returns true if `layer` is owned by this shard (always true when no
    /// range is set). Use this to guard accessor calls and reject requests
    /// for layers outside the server's owned range before touching mmap pages.
    pub fn is_layer_owned(&self, layer: usize) -> bool {
        match self.layer_range {
            None => true,
            Some((start, end)) => layer >= start && layer < end,
        }
    }

    /// Returns the owned layer range `(start_inclusive, end_exclusive)`, or
    /// `None` if all layers are served.
    pub fn owned_layer_range(&self) -> Option<(usize, usize)> {
        self.layer_range
    }

    /// Set the owned layer range (used by `load_vindex_with_range`).
    pub(crate) fn set_layer_range(&mut self, range: (usize, usize)) {
        self.layer_range = Some(range);
    }
}

#[cfg(test)]
mod refactor_tests {
    //! Coverage for the `empty()` / `new()` / `new_mmap()` / `Clone`
    //! refactor. These tests pin the invariants the refactor promised:
    //! constructors use a single source of truth (`empty`), Clone
    //! preserves Arc refcount (doesn't deep-copy mmap bytes), Clone
    //! resets Mutex/RwLock caches (fresh allocations), atomics carry
    //! their current value across the clone boundary.
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn empty_defaults_for_new_fields() {
        let v = VectorIndex::empty(3, 64);
        assert_eq!(v.num_layers, 3);
        assert_eq!(v.hidden_size, 64);
        assert_eq!(v.gate_vectors.len(), 3);
        assert!(v.gate_vectors.iter().all(|slot| slot.is_none()));
        assert!(v.gate_mmap_bytes.is_none());
        assert!(v.gate_mmap_slices.is_empty());
        assert!(v.down_meta_mmap.is_none());
        assert!(v.down_features_mmap.is_none());
        assert!(v.up_features_mmap.is_none());
        assert!(v.interleaved_mmap.is_none());
        assert!(v.interleaved_q4_mmap.is_none());
        assert!(v.interleaved_q4k_mmap.is_none());
        assert!(v.interleaved_q4k_manifest.is_none());
        assert!(v.gate_q4_mmap.is_none());
        assert!(v.gate_q4_slices.is_empty());
        assert!(v.lm_head_mmap.is_none());
        assert!(v.lm_head_f16_mmap.is_none());
        assert!(v.lm_head_q4_mmap.is_none());
        assert!(v.lm_head_q4_synth.is_none());
        assert!(v.attn_q4k_mmap.is_none());
        assert!(v.attn_q4k_manifest.is_none());
        assert!(v.attn_q4_mmap.is_none());
        assert!(v.attn_q4_manifest.is_none());
        assert!(v.attn_q8_mmap.is_none());
        assert!(v.attn_q8_manifest.is_none());
        assert!(v.fp4_storage.is_none());
        assert_eq!(v.vocab_size, 0);
        assert_eq!(v.layer_range, None);
        assert!(matches!(v.gate_mmap_dtype, crate::StorageDtype::F32));
        // Atomics at their ground state.
        assert!(!v.hnsw_enabled.load(Ordering::Relaxed));
        assert_eq!(v.hnsw_ef_search.load(Ordering::Relaxed), 200);
        assert_eq!(v.gate_cache_max_layers.load(Ordering::Relaxed), 0);
        // Caches sized to num_layers.
        let f16_cache = v.f16_decode_cache.lock().unwrap();
        assert_eq!(f16_cache.len(), 3);
        drop(f16_cache);
        let warm = v.warmed_gates.read().unwrap();
        assert_eq!(warm.len(), 3);
        drop(warm);
        let hnsw = v.hnsw_cache.lock().unwrap();
        assert_eq!(hnsw.len(), 3);
        drop(hnsw);
        let q4k = v.q4k_ffn_cache.lock().unwrap();
        assert_eq!(q4k.len(), 3);
        drop(q4k);
    }

    #[test]
    fn new_preserves_gate_and_down_meta_overrides_empty() {
        let gate = vec![Some(Array2::<f32>::zeros((2, 4))), None];
        let down = vec![None, Some(vec![None; 5])];
        let v = VectorIndex::new(gate.clone(), down.clone(), 2, 4);
        assert_eq!(v.num_layers, 2);
        assert_eq!(v.hidden_size, 4);
        assert!(v.gate_vectors[0].is_some());
        assert_eq!(v.gate_vectors[0].as_ref().unwrap().shape(), &[2, 4]);
        assert!(v.down_meta[1].is_some());
        assert_eq!(v.down_meta[1].as_ref().unwrap().len(), 5);
        // Everything else falls through to empty().
        assert!(v.gate_mmap_bytes.is_none());
        assert!(v.fp4_storage.is_none());
    }

    #[test]
    fn new_mmap_sets_mmap_fields_and_defaults_rest() {
        let bytes = vec![0u8; 1024];
        // Create a zero-backed mmap via a tempfile so we have a real Mmap.
        let tmp = std::env::temp_dir().join(format!("core_mmap_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        let path = tmp.join("fake_gate.bin");
        std::fs::write(&path, &bytes).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };

        let v = VectorIndex::new_mmap(
            mmap,
            Vec::new(),
            crate::StorageDtype::F16,
            None,
            4,
            16,
        );
        assert_eq!(v.num_layers, 4);
        assert_eq!(v.hidden_size, 16);
        assert!(v.gate_mmap_bytes.is_some());
        assert!(matches!(v.gate_mmap_dtype, crate::StorageDtype::F16));
        // Fields not set by new_mmap() come from empty().
        assert!(v.down_features_mmap.is_none());
        assert!(v.fp4_storage.is_none());
        assert_eq!(v.vocab_size, 0);
        let f16_cache = v.f16_decode_cache.lock().unwrap();
        assert_eq!(f16_cache.len(), 4);
        drop(f16_cache);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn clone_shares_arc_mmap_handles() {
        let tmp = std::env::temp_dir().join(format!("core_clone_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        let path = tmp.join("fake_gate.bin");
        std::fs::write(&path, vec![0u8; 64]).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
        let original = VectorIndex::new_mmap(
            mmap, Vec::new(), crate::StorageDtype::F32, None, 2, 8,
        );

        let src_arc = original.gate_mmap_bytes.as_ref().unwrap();
        let src_strong_before = Arc::strong_count(src_arc);

        let cloned = original.clone();
        let src_strong_after = Arc::strong_count(src_arc);

        // Clone should have bumped the refcount (Arc shared, not deep-copied).
        assert_eq!(
            src_strong_after,
            src_strong_before + 1,
            "Arc strong count should increase by 1 on clone"
        );
        // Both should point at the same allocation.
        let cloned_arc = cloned.gate_mmap_bytes.as_ref().unwrap();
        assert!(Arc::ptr_eq(src_arc, cloned_arc), "both must share the mmap");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn clone_preserves_atomic_values() {
        let v = VectorIndex::empty(2, 8);
        v.hnsw_enabled.store(true, Ordering::Relaxed);
        v.hnsw_ef_search.store(42, Ordering::Relaxed);
        v.gate_cache_max_layers.store(7, Ordering::Relaxed);

        let cloned = v.clone();
        assert!(cloned.hnsw_enabled.load(Ordering::Relaxed));
        assert_eq!(cloned.hnsw_ef_search.load(Ordering::Relaxed), 42);
        assert_eq!(cloned.gate_cache_max_layers.load(Ordering::Relaxed), 7);

        // Mutating the clone's atomics must not affect the original.
        cloned.hnsw_enabled.store(false, Ordering::Relaxed);
        assert!(v.hnsw_enabled.load(Ordering::Relaxed));
    }

    #[test]
    fn clone_resets_mutex_caches_to_fresh() {
        let v = VectorIndex::empty(3, 16);

        // Populate a cache entry.
        {
            let mut cache = v.f16_decode_cache.lock().unwrap();
            cache[1] = Some(vec![1.0, 2.0, 3.0]);
        }
        {
            let mut warm = v.warmed_gates.write().unwrap();
            warm[0] = Some(vec![7.0]);
        }

        let cloned = v.clone();

        // Source retains state.
        let src_cache = v.f16_decode_cache.lock().unwrap();
        assert!(src_cache[1].is_some(), "source cache unchanged");
        drop(src_cache);

        // Clone starts fresh.
        let cloned_cache = cloned.f16_decode_cache.lock().unwrap();
        assert_eq!(cloned_cache.len(), 3);
        assert!(cloned_cache.iter().all(|slot| slot.is_none()),
                "clone's cache must be empty");
        drop(cloned_cache);

        let cloned_warm = cloned.warmed_gates.read().unwrap();
        assert!(cloned_warm.iter().all(|slot| slot.is_none()));
        drop(cloned_warm);
    }

    #[test]
    fn clone_preserves_vec_and_hashmap_fields() {
        let mut v = VectorIndex::empty(2, 4);
        v.down_overrides.insert((0, 3), vec![1.0, 2.0, 3.0, 4.0]);
        v.up_overrides.insert((1, 1), vec![5.0; 4]);

        let cloned = v.clone();
        assert_eq!(cloned.down_overrides.get(&(0, 3)), Some(&vec![1.0, 2.0, 3.0, 4.0]));
        assert_eq!(cloned.up_overrides.get(&(1, 1)), Some(&vec![5.0; 4]));

        // Distinct allocations — mutating the clone doesn't affect the source.
        let mut cloned = cloned;
        cloned.down_overrides.insert((1, 0), vec![9.0; 4]);
        assert!(!v.down_overrides.contains_key(&(1, 0)), "source HashMap was aliased");
    }

    #[test]
    fn clone_preserves_layer_range() {
        let mut v = VectorIndex::empty(4, 8);
        v.set_layer_range((1, 3));
        let cloned = v.clone();
        assert_eq!(cloned.layer_range, Some((1, 3)));
        assert_eq!(cloned.owned_layer_range(), Some((1, 3)));
    }

    #[test]
    fn clone_carries_fp4_storage_handle() {
        use super::super::fp4_storage::Fp4Storage;
        use crate::config::types::Fp4Config;

        let manifest = Fp4Config::option_b_default();
        let storage = Fp4Storage {
            manifest,
            gate_mmap: None,
            up_mmap: None,
            down_mmap: None,
            layer_features: vec![4, 4],
            hidden: 256,
        };
        let mut v = VectorIndex::empty(2, 256);
        v.fp4_storage = Some(Arc::new(storage));

        let src_arc = v.fp4_storage.as_ref().unwrap().clone();
        let strong_before = Arc::strong_count(&src_arc);
        let cloned = v.clone();
        let strong_after = Arc::strong_count(&src_arc);

        assert!(cloned.fp4_storage.is_some());
        assert_eq!(strong_after, strong_before + 1, "Arc count must bump");
        assert!(Arc::ptr_eq(&src_arc, cloned.fp4_storage.as_ref().unwrap()));
    }

    #[test]
    fn clone_independent_hnsw_cache_allocation() {
        let v = VectorIndex::empty(3, 16);
        let cloned = v.clone();

        // Mutating clone's HNSW slot must not affect the source.
        {
            let mut c = cloned.hnsw_cache.lock().unwrap();
            c[0] = None; // already None, but force a touch
            assert_eq!(c.len(), 3);
        }
        // Source's HNSW cache must still be intact.
        let src = v.hnsw_cache.lock().unwrap();
        assert_eq!(src.len(), 3);
    }
}
