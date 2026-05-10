//! `MmapStorage` — production `VindexStorage` impl backed by the
//! existing `Arc<Mmap>` substore fields.
//!
//! Step 2 of the migration plan: a parity wrapper that returns the
//! same byte ranges the substore accessors do today, in `Bytes`
//! shape. No behavior change. Substores still own their
//! `Arc<Mmap>` fields; `MmapStorage` holds clones (cheap — one Arc
//! refcount bump per file at construction) and a reused `Bytes`
//! handle per whole-file mmap so per-layer slices are O(1)
//! refcounted.
//!
//! In step 4 the substore byte-yielding accessors get rewritten to
//! forward through `MmapStorage`. In step 5 the substore mmap fields
//! drop entirely. Until then this is purely additive.

use std::sync::Arc;

use bytes::Bytes;

use crate::config::dtype::StorageDtype;
use crate::index::storage::attn::ATTN_TENSORS_PER_LAYER;
use crate::index::storage::ffn_store::{DownFeaturesQ4kEntry, FfnStore, FFN_COMPONENTS_PER_LAYER};
use crate::index::storage::gate_store::GateStore;
use crate::index::storage::projection_store::ProjectionStore;
use crate::index::types::{GateLayerSlice, GateQ4Slice};

use super::sealed::Sealed;
use super::{BytesView, GateLayerView, VindexStorage};

/// Parity wrapper over today's substore mmaps. Implements
/// `VindexStorage` by cloning each substore's `Arc<Mmap>` (or
/// `Arc<Vec<u8>>` for the synth lm_head) and converting once into a
/// reusable `Bytes` whole-file handle. Per-layer accessors slice the
/// whole-file `Bytes` via `Bytes::slice` — O(1) refcounted, no copy.
#[derive(Clone)]
pub struct MmapStorage {
    // ── FFN ──────────────────────────────────────────────────────────
    interleaved_q4k: Option<Bytes>,
    interleaved_q4k_manifest: Option<Vec<(usize, usize, String)>>,
    interleaved_q4: Option<Bytes>,
    down_features_q4k: Option<Bytes>,
    down_features_q4k_manifest: Option<Vec<DownFeaturesQ4kEntry>>,

    // ── Attention ────────────────────────────────────────────────────
    attn_q4k: Option<Bytes>,
    attn_q4k_manifest: Option<Vec<(usize, usize, String)>>,
    attn_q4: Option<Bytes>,
    attn_q4_manifest: Option<Vec<(usize, usize)>>,
    attn_q8: Option<Bytes>,
    attn_q8_manifest: Option<Vec<(usize, usize, usize)>>,

    // ── lm_head ──────────────────────────────────────────────────────
    lm_head_f32: Option<Bytes>,
    lm_head_f16: Option<Bytes>,
    lm_head_q4: Option<Bytes>,

    // ── Gate ─────────────────────────────────────────────────────────
    gate_bytes: Option<Bytes>,
    gate_dtype: StorageDtype,
    gate_slices: Vec<GateLayerSlice>,
    gate_q4_bytes: Option<Bytes>,
    gate_q4_slices: Vec<GateQ4Slice>,
    /// Hidden dim — gate `GateLayerView::slice.float_offset` is in
    /// floats, but a Redis-backed impl will need `hidden_size` to
    /// slice from a flat buffer. Carry it here so the trait surface
    /// doesn't have to.
    #[allow(dead_code)]
    hidden_size: usize,
}

impl Sealed for MmapStorage {}

impl MmapStorage {
    /// Build a parity wrapper from today's substores. Cheap — every
    /// `Arc<Mmap>` clone is a refcount bump; the only allocation is
    /// `Bytes::from_owner` once per whole-file mmap.
    ///
    /// `lm_head_q4_synth` (in-RAM Q4 synthesised from f16 embeddings)
    /// is folded into `lm_head_q4`: callers don't see the difference.
    pub fn from_substores(
        ffn: &FfnStore,
        gate: &GateStore,
        projections: &ProjectionStore,
        hidden_size: usize,
    ) -> Self {
        // lm_head_q4 unifies the mmap and the in-RAM synth fallback.
        // The synth path is `Arc<Vec<u8>>`; the mmap path is
        // `Arc<Mmap>`. Both convert to `Bytes::from_owner` cleanly.
        let lm_head_q4 = projections
            .lm_head_q4_mmap
            .as_ref()
            .map(arc_mmap_to_bytes)
            .or_else(|| {
                projections
                    .lm_head_q4_synth
                    .as_ref()
                    .map(|v| Bytes::from_owner(ArcAsBytes(v.clone())))
            });

        Self {
            // FFN
            interleaved_q4k: ffn.interleaved_q4k_mmap.as_ref().map(arc_mmap_to_bytes),
            interleaved_q4k_manifest: ffn.interleaved_q4k_manifest.clone(),
            interleaved_q4: ffn.interleaved_q4_mmap.as_ref().map(arc_mmap_to_bytes),
            down_features_q4k: ffn.down_features_q4k_mmap.as_ref().map(arc_mmap_to_bytes),
            down_features_q4k_manifest: ffn.down_features_q4k_manifest.clone(),

            // Attention
            attn_q4k: projections.attn_q4k_mmap.as_ref().map(arc_mmap_to_bytes),
            attn_q4k_manifest: projections.attn_q4k_manifest.clone(),
            attn_q4: projections.attn_q4_mmap.as_ref().map(arc_mmap_to_bytes),
            attn_q4_manifest: projections.attn_q4_manifest.clone(),
            attn_q8: projections.attn_q8_mmap.as_ref().map(arc_mmap_to_bytes),
            attn_q8_manifest: projections.attn_q8_manifest.clone(),

            // lm_head
            lm_head_f32: projections.lm_head_mmap.as_ref().map(arc_mmap_to_bytes),
            lm_head_f16: projections.lm_head_f16_mmap.as_ref().map(arc_mmap_to_bytes),
            lm_head_q4,

            // Gate
            gate_bytes: gate.gate_mmap_bytes.as_ref().map(arc_mmap_to_bytes),
            gate_dtype: gate.gate_mmap_dtype,
            gate_slices: gate.gate_mmap_slices.clone(),
            gate_q4_bytes: gate.gate_q4_mmap.as_ref().map(arc_mmap_to_bytes),
            gate_q4_slices: gate.gate_q4_slices.clone(),
            hidden_size,
        }
    }

    /// Inert empty wrapper — every `Option` is `None`. Used by
    /// `VectorIndex::empty()` and tests. Constructed without any of
    /// the substore types so callers don't have to fabricate empty
    /// substores just to get a storage handle.
    pub fn empty(hidden_size: usize) -> Self {
        Self {
            interleaved_q4k: None,
            interleaved_q4k_manifest: None,
            interleaved_q4: None,
            down_features_q4k: None,
            down_features_q4k_manifest: None,
            attn_q4k: None,
            attn_q4k_manifest: None,
            attn_q4: None,
            attn_q4_manifest: None,
            attn_q8: None,
            attn_q8_manifest: None,
            lm_head_f32: None,
            lm_head_f16: None,
            lm_head_q4: None,
            gate_bytes: None,
            gate_dtype: StorageDtype::F32,
            gate_slices: Vec::new(),
            gate_q4_bytes: None,
            gate_q4_slices: Vec::new(),
            hidden_size,
        }
    }
}

/// `Arc<Mmap>` → `Bytes` via `Bytes::from_owner`. Zero-copy: the
/// `Bytes` keeps the `Arc<Mmap>` alive for the lifetime of any
/// outstanding slices.
fn arc_mmap_to_bytes(arc: &Arc<memmap2::Mmap>) -> Bytes {
    Bytes::from_owner(ArcAsBytes(arc.clone()))
}

/// Owner wrapper so `bytes::Bytes::from_owner` (which requires
/// `AsRef<[u8]>` on the owner) accepts an `Arc<T>` where `T` already
/// implements `AsRef<[u8]>`. Both `memmap2::Mmap` and `Vec<u8>` (the
/// in-RAM synth lm_head) qualify; without this wrapper Rust looks for
/// `AsRef<[u8]> for Arc<T>` directly and only finds `AsRef<T>`.
struct ArcAsBytes<T: AsRef<[u8]> + Send + Sync + 'static>(Arc<T>);

impl<T: AsRef<[u8]> + Send + Sync + 'static> AsRef<[u8]> for ArcAsBytes<T> {
    fn as_ref(&self) -> &[u8] {
        (*self.0).as_ref()
    }
}

/// Bounds-check (`offset + length <= bytes.len()`) and build a
/// borrowed `BytesView`. Matches the defensive behavior of every
/// substore accessor that consults a stale-or-corrupt manifest.
fn checked_view<'a>(bytes: &'a Bytes, offset: usize, length: usize) -> Option<BytesView<'a>> {
    let end = offset.checked_add(length)?;
    if end > bytes.len() {
        return None;
    }
    Some(BytesView::new(bytes, offset, length))
}

impl VindexStorage for MmapStorage {
    // ── FFN ───────────────────────────────────────────────────────

    fn interleaved_q4k_layer_data(
        &self,
        layer: usize,
    ) -> Option<[(BytesView<'_>, &str); FFN_COMPONENTS_PER_LAYER]> {
        let bytes = self.interleaved_q4k.as_ref()?;
        let manifest = self.interleaved_q4k_manifest.as_ref()?;
        let base = layer * FFN_COMPONENTS_PER_LAYER;
        if base + FFN_COMPONENTS_PER_LAYER > manifest.len() {
            return None;
        }
        // Validate every entry's range before forming the array.
        for i in 0..FFN_COMPONENTS_PER_LAYER {
            let (offset, length, _) = &manifest[base + i];
            checked_view(bytes, *offset, *length)?;
        }
        let out: [(BytesView<'_>, &str); FFN_COMPONENTS_PER_LAYER] = std::array::from_fn(|i| {
            let (offset, length, format) = &manifest[base + i];
            (BytesView::new(bytes, *offset, *length), format.as_str())
        });
        Some(out)
    }

    fn interleaved_q4k_whole_buffer(&self) -> Option<Bytes> {
        self.interleaved_q4k.clone()
    }

    fn interleaved_q4_whole_buffer(&self) -> Option<Bytes> {
        self.interleaved_q4.clone()
    }

    fn down_features_q4k_layer_data(&self, layer: usize) -> Option<(BytesView<'_>, &str, usize)> {
        let bytes = self.down_features_q4k.as_ref()?;
        let manifest = self.down_features_q4k_manifest.as_ref()?;
        let entry = manifest.get(layer)?;
        let view = checked_view(bytes, entry.offset, entry.length)?;
        Some((view, entry.format.as_str(), entry.padded_width))
    }

    fn gate_q4_layer_data(&self, layer: usize) -> Option<BytesView<'_>> {
        let bytes = self.gate_q4_bytes.as_ref()?;
        let entry = self.gate_q4_slices.get(layer)?;
        if entry.byte_len == 0 {
            return None;
        }
        checked_view(bytes, entry.byte_offset, entry.byte_len)
    }

    // ── Attention ─────────────────────────────────────────────────

    fn attn_q4k_layer_data(
        &self,
        layer: usize,
    ) -> Option<[(BytesView<'_>, &str); ATTN_TENSORS_PER_LAYER]> {
        let bytes = self.attn_q4k.as_ref()?;
        let manifest = self.attn_q4k_manifest.as_ref()?;
        let base = layer * ATTN_TENSORS_PER_LAYER;
        if base + ATTN_TENSORS_PER_LAYER > manifest.len() {
            return None;
        }
        for i in 0..ATTN_TENSORS_PER_LAYER {
            let (offset, length, _) = &manifest[base + i];
            checked_view(bytes, *offset, *length)?;
        }
        let out: [(BytesView<'_>, &str); ATTN_TENSORS_PER_LAYER] = std::array::from_fn(|i| {
            let (offset, length, format) = &manifest[base + i];
            (BytesView::new(bytes, *offset, *length), format.as_str())
        });
        Some(out)
    }

    fn attn_q4_whole_buffer(&self) -> Option<Bytes> {
        self.attn_q4.clone()
    }

    fn attn_q4_layer_slices(
        &self,
        layer: usize,
    ) -> Option<[BytesView<'_>; ATTN_TENSORS_PER_LAYER]> {
        let bytes = self.attn_q4.as_ref()?;
        let manifest = self.attn_q4_manifest.as_ref()?;
        let base = layer * ATTN_TENSORS_PER_LAYER;
        if base + ATTN_TENSORS_PER_LAYER > manifest.len() {
            return None;
        }
        for i in 0..ATTN_TENSORS_PER_LAYER {
            let (offset, length) = &manifest[base + i];
            checked_view(bytes, *offset, *length)?;
        }
        let out: [BytesView<'_>; ATTN_TENSORS_PER_LAYER] = std::array::from_fn(|i| {
            let (offset, length) = &manifest[base + i];
            BytesView::new(bytes, *offset, *length)
        });
        Some(out)
    }

    fn attn_q8_layer_data(
        &self,
        layer: usize,
    ) -> Option<[(BytesView<'_>, BytesView<'_>); ATTN_TENSORS_PER_LAYER]> {
        let bytes = self.attn_q8.as_ref()?;
        let manifest = self.attn_q8_manifest.as_ref()?;
        let base = layer * ATTN_TENSORS_PER_LAYER;
        if base + ATTN_TENSORS_PER_LAYER > manifest.len() {
            return None;
        }
        for i in 0..ATTN_TENSORS_PER_LAYER {
            let (offset, vals_len, scales_len) = manifest[base + i];
            let vals_end = offset.checked_add(vals_len)?;
            let scales_end = vals_end.checked_add(scales_len)?;
            if scales_end > bytes.len() {
                return None;
            }
        }
        let out: [(BytesView<'_>, BytesView<'_>); ATTN_TENSORS_PER_LAYER] =
            std::array::from_fn(|i| {
                let (offset, vals_len, scales_len) = manifest[base + i];
                let vals = BytesView::new(bytes, offset, vals_len);
                let scales = BytesView::new(bytes, offset + vals_len, scales_len);
                (vals, scales)
            });
        Some(out)
    }

    // ── lm_head ───────────────────────────────────────────────────

    fn lm_head_q4_bytes(&self) -> Option<Bytes> {
        self.lm_head_q4.clone()
    }

    fn lm_head_f16_bytes(&self) -> Option<Bytes> {
        self.lm_head_f16.clone()
    }

    fn lm_head_f32_bytes(&self) -> Option<Bytes> {
        self.lm_head_f32.clone()
    }

    // ── Gate ──────────────────────────────────────────────────────

    fn gate_layer_view(&self, layer: usize) -> Option<GateLayerView<'_>> {
        let bytes = self.gate_bytes.as_ref()?;
        let slice = *self.gate_slices.get(layer)?;
        if slice.num_features == 0 {
            return None;
        }
        Some(GateLayerView {
            bytes,
            dtype: self.gate_dtype,
            slice,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::types::GateLayerSlice;

    /// Empty wrapper has every accessor returning `None`.
    #[test]
    fn empty_storage_returns_none_everywhere() {
        let s = MmapStorage::empty(2560);
        assert!(s.interleaved_q4k_layer_data(0).is_none());
        assert!(s.interleaved_q4k_whole_buffer().is_none());
        assert!(s.interleaved_q4_whole_buffer().is_none());
        assert!(s.down_features_q4k_layer_data(0).is_none());
        assert!(s.gate_q4_layer_data(0).is_none());
        assert!(s.attn_q4k_layer_data(0).is_none());
        assert!(s.attn_q4_whole_buffer().is_none());
        assert!(s.attn_q4_layer_slices(0).is_none());
        assert!(s.attn_q8_layer_data(0).is_none());
        assert!(s.lm_head_q4_bytes().is_none());
        assert!(s.lm_head_f16_bytes().is_none());
        assert!(s.lm_head_f32_bytes().is_none());
        assert!(s.gate_layer_view(0).is_none());
    }

    /// A `Bytes`-backed `MmapStorage` with a fabricated FFN Q4_K
    /// manifest must hand back the same byte ranges the manifest
    /// describes.
    #[test]
    fn ffn_q4k_layer_data_matches_manifest() {
        let mut s = MmapStorage::empty(8);
        // 3 layers × 3 components × 16 bytes = 144 bytes.
        let payload: Vec<u8> = (0u8..144).collect();
        s.interleaved_q4k = Some(Bytes::from(payload.clone()));
        s.interleaved_q4k_manifest = Some(
            (0..3 * FFN_COMPONENTS_PER_LAYER)
                .map(|i| (i * 16, 16, "Q4_K".to_string()))
                .collect(),
        );

        for layer in 0..3 {
            let arr = s.interleaved_q4k_layer_data(layer).expect("layer present");
            for (c, (view, fmt)) in arr.iter().enumerate() {
                let global = layer * FFN_COMPONENTS_PER_LAYER + c;
                let expected: &[u8] = &payload[global * 16..(global + 1) * 16];
                assert_eq!(view.as_slice(), expected, "layer {layer} comp {c}");
                assert_eq!(*fmt, "Q4_K");
            }
        }
    }

    /// A stale FFN Q4_K manifest entry that runs past the buffer
    /// must produce `None`, not a slice-bounds panic.
    #[test]
    fn ffn_q4k_layer_data_rejects_out_of_bounds_manifest() {
        let mut s = MmapStorage::empty(8);
        let payload: Vec<u8> = vec![0u8; 32];
        s.interleaved_q4k = Some(Bytes::from(payload));
        // gate fits, up fits, down points past the end.
        s.interleaved_q4k_manifest = Some(vec![
            (0, 8, "Q4_K".to_string()),
            (8, 8, "Q4_K".to_string()),
            (16, 32, "Q4_K".to_string()), // 16 + 32 = 48 > 32
        ]);
        assert!(s.interleaved_q4k_layer_data(0).is_none());
    }

    /// Attention Q8 layer data carries vals + scales spans; both must
    /// fit before any tuple is formed.
    #[test]
    fn attn_q8_layer_data_validates_combined_span() {
        let mut s = MmapStorage::empty(8);
        s.attn_q8 = Some(Bytes::from(vec![0u8; 1024]));
        // Q, K, V fit; O's scales run past 1024.
        s.attn_q8_manifest = Some(vec![
            (0, 64, 16),
            (100, 64, 16),
            (200, 64, 16),
            (1000, 64, 16), // 1000 + 64 + 16 = 1080 > 1024
        ]);
        assert!(s.attn_q8_layer_data(0).is_none());
    }

    /// `GateLayerView<'_>` borrows the dtype + slice + bytes
    /// together. The view is `Copy`, so multiple holders share the
    /// same borrow without refcount touches.
    #[test]
    fn gate_layer_view_round_trip() {
        let mut s = MmapStorage::empty(4);
        s.gate_bytes = Some(Bytes::from(vec![1u8, 2, 3, 4, 5, 6, 7, 8]));
        s.gate_dtype = StorageDtype::F16;
        s.gate_slices = vec![
            GateLayerSlice {
                float_offset: 0,
                num_features: 1,
            },
            GateLayerSlice {
                float_offset: 4,
                num_features: 1,
            },
        ];
        let v0 = s.gate_layer_view(0).expect("layer 0 present");
        assert_eq!(v0.dtype, StorageDtype::F16);
        assert_eq!(v0.slice.num_features, 1);
        let v0_copy = v0; // `Copy`, no clone needed.
        assert_eq!(v0.bytes.as_ptr(), v0_copy.bytes.as_ptr());
    }

    /// `gate_layer_view` returns `None` when the layer's
    /// `num_features` is zero — matches the substore convention for
    /// unowned layers in a sharded `--layers` slice.
    #[test]
    fn gate_layer_view_none_when_layer_unowned() {
        let mut s = MmapStorage::empty(4);
        s.gate_bytes = Some(Bytes::from(vec![0u8; 8]));
        s.gate_slices = vec![GateLayerSlice {
            float_offset: 0,
            num_features: 0,
        }];
        assert!(s.gate_layer_view(0).is_none());
    }

    /// `MmapStorage` clones cheaply — every field is `Bytes` /
    /// `Vec<...>` / `Copy`, so clone is a refcount bump per
    /// whole-file `Bytes`.
    #[test]
    fn mmap_storage_clones_via_refcount() {
        let mut s = MmapStorage::empty(4);
        s.lm_head_f16 = Some(Bytes::from(vec![1u8, 2, 3, 4]));
        let cloned = s.clone();
        assert_eq!(
            s.lm_head_f16.as_ref().unwrap().as_ptr(),
            cloned.lm_head_f16.as_ref().unwrap().as_ptr(),
        );
    }
}
