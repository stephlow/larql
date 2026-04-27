//! Attention weight loaders + per-layer accessors.
//!
//! Loads the per-layer Q / K / V / O projection weights in Q8, Q4_K, or
//! Q4_0 format from `attn_weights_*.bin` files plus their JSON
//! manifests. Mirrors the FFN walk plumbing in `super::walk`; lives in
//! its own file so attention storage isn't tangled with FFN storage.

use std::sync::Arc;

use crate::error::VindexError;
use crate::format::filenames::*;
use crate::mmap_util::mmap_optimized;

use crate::index::core::VectorIndex;

impl VectorIndex {
    /// Load Q8 attention weights + manifest for GPU full pipeline.
    pub fn load_attn_q8(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join(ATTN_WEIGHTS_Q8_BIN);
        if !path.exists() {
            return Err(VindexError::Parse("attn_weights_q8.bin not found".into()));
        }
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { mmap_optimized(&file)? };
        self.projections.attn_q8_mmap = Some(Arc::new(mmap));

        let manifest_path = dir.join(ATTN_WEIGHTS_Q8_MANIFEST_JSON);
        if manifest_path.exists() {
            let json: Vec<serde_json::Value> = serde_json::from_str(
                &std::fs::read_to_string(&manifest_path)
                    .map_err(|e| VindexError::Parse(e.to_string()))?,
            )
            .map_err(|e| VindexError::Parse(e.to_string()))?;

            let entries: Vec<(usize, usize, usize)> = json
                .iter()
                .map(|e| {
                    let offset = e["q8_offset"].as_u64().unwrap_or(0) as usize;
                    let vals_len = e["q8_vals_len"].as_u64().unwrap_or(0) as usize;
                    let scales_len = e["q8_scales_len"].as_u64().unwrap_or(0) as usize;
                    (offset, vals_len, scales_len)
                })
                .collect();
            self.projections.attn_q8_manifest = Some(entries);
        }
        Ok(())
    }

    /// Get per-layer Q8 attention slices: (q_vals, q_scales, k_vals, k_scales, v_vals, v_scales, o_vals, o_scales)
    pub fn attn_q8_layer_data(&self, layer: usize) -> Option<[(&[u8], &[f32]); 4]> {
        let mmap = self.projections.attn_q8_mmap.as_ref()?;
        let manifest = self.projections.attn_q8_manifest.as_ref()?;

        let base = layer * 4;
        if base + 3 >= manifest.len() {
            return None;
        }

        let mut result = [(&[] as &[u8], &[] as &[f32]); 4];
        for i in 0..4 {
            let (offset, vals_len, scales_len) = manifest[base + i];
            let vals = &mmap[offset..offset + vals_len];
            let scales_start = offset + vals_len;
            let scales_data = &mmap[scales_start..scales_start + scales_len];
            let scales = unsafe {
                std::slice::from_raw_parts(scales_data.as_ptr() as *const f32, scales_len / 4)
            };
            result[i] = (vals, scales);
        }
        Some(result)
    }

    /// Load Q4_K/Q6_K attention weights for Ollama-compatible GPU pipeline.
    pub fn load_attn_q4k(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join(ATTN_WEIGHTS_Q4K_BIN);
        if !path.exists() {
            return Err(VindexError::Parse("attn_weights_q4k.bin not found".into()));
        }
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { mmap_optimized(&file)? };

        let manifest_path = dir.join(ATTN_WEIGHTS_Q4K_MANIFEST_JSON);
        if manifest_path.exists() {
            let json: Vec<serde_json::Value> = serde_json::from_str(
                &std::fs::read_to_string(&manifest_path)
                    .map_err(|e| VindexError::Parse(e.to_string()))?,
            )
            .map_err(|e| VindexError::Parse(e.to_string()))?;

            // Each entry: {key, shape, format, offset, length}.
            //
            // Format is required. We used to default to `"Q4_K"` here
            // when the field was missing, which silently masked
            // malformed manifests — see ROADMAP P0 "Replace
            // unwrap_or(Q4_K) silent fallbacks".
            let entries: Vec<(usize, usize, String)> = json
                .iter()
                .map(|e| {
                    let offset = e["offset"].as_u64().unwrap_or(0) as usize;
                    let length = e["length"].as_u64().unwrap_or(0) as usize;
                    let tag = e["format"].as_str().ok_or_else(|| {
                        VindexError::Parse(
                            "attn_weights_q4k_manifest entry missing `format` field".into(),
                        )
                    })?;
                    let qfmt = crate::quant::registry::lookup(tag).ok_or_else(|| {
                        VindexError::Parse(format!(
                            "attn_weights_q4k_manifest: unknown format tag {tag:?} \
                             — quant::registry has no entry"
                        ))
                    })?;

                    // Stride sanity check — catches stale vindexes built
                    // with the legacy 148-byte block_q4_K layout against
                    // the current 144-byte GGUF kernels (the read drifts
                    // 4 bytes per superblock, producing all-NaN output).
                    let key = e["key"].as_str().unwrap_or("<no-key>");
                    let shape: Vec<usize> = e["shape"]
                        .as_array()
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_u64().map(|n| n as usize))
                                .collect()
                        })
                        .unwrap_or_default();
                    if let Some(expected) = qfmt.expected_bytes(&shape) {
                        if expected != length {
                            return Err(VindexError::Parse(format!(
                                "attn_weights_q4k_manifest: tensor {key:?} ({tag}, shape {shape:?}) \
                                 has length {length} but format expects {expected} \
                                 ({} bytes/block × {}). \
                                 Likely cause: vindex built with legacy 148-byte block_q4_K layout — \
                                 rebuild the vindex with current code (`larql q4k <model>` or equivalent).",
                                qfmt.bytes_per_block,
                                length / qfmt.bytes_per_block.max(1),
                            )));
                        }
                    }
                    Ok((offset, length, tag.to_string()))
                })
                .collect::<Result<Vec<_>, VindexError>>()?;
            self.projections.attn_q4k_manifest = Some(entries);
        }
        self.projections.attn_q4k_mmap = Some(Arc::new(mmap));
        Ok(())
    }

    /// Get per-layer Q4_K/Q6_K attention slices: (data, format) for Q, K, V, O.
    pub fn attn_q4k_layer_data(&self, layer: usize) -> Option<[(&[u8], &str); 4]> {
        let mmap = self.projections.attn_q4k_mmap.as_ref()?;
        let manifest = self.projections.attn_q4k_manifest.as_ref()?;
        let base = layer * 4;
        if base + 3 >= manifest.len() {
            return None;
        }

        let mut result: [(&[u8], &str); 4] = [(&[], ""); 4];
        for i in 0..4 {
            let (offset, length, ref format) = manifest[base + i];
            result[i] = (&mmap[offset..offset + length], format.as_str());
        }
        Some(result)
    }

    /// Load Q4 attention weights + manifest for GPU full pipeline.
    pub fn load_attn_q4(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join(ATTN_WEIGHTS_Q4_BIN);
        if !path.exists() {
            return Err(VindexError::Parse("attn_weights_q4.bin not found".into()));
        }
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { mmap_optimized(&file)? };
        self.projections.attn_q4_mmap = Some(Arc::new(mmap));

        // Load manifest with per-matrix offsets
        let manifest_path = dir.join(ATTN_WEIGHTS_Q4_MANIFEST_JSON);
        if manifest_path.exists() {
            let json: Vec<serde_json::Value> = serde_json::from_str(
                &std::fs::read_to_string(&manifest_path)
                    .map_err(|e| VindexError::Parse(e.to_string()))?,
            )
            .map_err(|e| VindexError::Parse(e.to_string()))?;

            let entries: Vec<(usize, usize)> = json
                .iter()
                .map(|e| {
                    let offset = e["q4_offset"].as_u64().unwrap_or(0) as usize;
                    let length = e["q4_length"].as_u64().unwrap_or(0) as usize;
                    (offset, length)
                })
                .collect();
            self.projections.attn_q4_manifest = Some(entries);
        }
        Ok(())
    }

    /// Get raw Q4 attention weight bytes (all layers packed).
    pub fn attn_q4_data(&self) -> Option<&[u8]> {
        self.projections
            .attn_q4_mmap
            .as_ref()
            .map(|m| m.as_ref() as &[u8])
    }

    /// Get per-layer Q4 attention weight slices (Q, K, V, O) using the manifest.
    /// Returns None if manifest or Q4 attn data is not loaded.
    #[allow(clippy::type_complexity)]
    pub fn attn_q4_layer_slices(&self, layer: usize) -> Option<(&[u8], &[u8], &[u8], &[u8])> {
        let mmap = self.projections.attn_q4_mmap.as_ref()?;
        let manifest = self.projections.attn_q4_manifest.as_ref()?;

        // Each layer has 4 tensors: Q, K, V, O
        let base = layer * 4;
        if base + 3 >= manifest.len() {
            return None;
        }

        let q = &manifest[base];
        let k = &manifest[base + 1];
        let v = &manifest[base + 2];
        let o = &manifest[base + 3];

        let q_data = &mmap[q.0..q.0 + q.1];
        let k_data = &mmap[k.0..k.0 + k.1];
        let v_data = &mmap[v.0..v.0 + v.1];
        let o_data = &mmap[o.0..o.0 + o.1];

        Some((q_data, k_data, v_data, o_data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal vindex directory with the given attn_weights_q4k.bin
    /// payload + manifest. Returns a `tempfile::TempDir` whose path can be
    /// passed straight to `load_attn_q4k`.
    fn make_vindex_with_attn_q4k(payload: &[u8], manifest: serde_json::Value) -> tempfile::TempDir {
        let tmp = tempfile::tempdir().expect("tempdir");
        std::fs::write(tmp.path().join(ATTN_WEIGHTS_Q4K_BIN), payload).unwrap();
        std::fs::write(
            tmp.path().join(ATTN_WEIGHTS_Q4K_MANIFEST_JSON),
            serde_json::to_string(&manifest).unwrap(),
        )
        .unwrap();
        tmp
    }

    fn empty_vindex() -> VectorIndex {
        // Layer count and hidden size don't matter for the load_attn_q4k
        // path — both are read from the manifest, not from the index.
        VectorIndex::empty(1, 2560)
    }

    /// Q4_K shape `[2048, 2560]` at the canonical Q4_K_BLOCK_BYTES stride
    /// must load cleanly.
    #[test]
    fn load_attn_q4k_accepts_correct_144_byte_stride() {
        use larql_models::quant::ggml::{Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS};
        let len = 2048 * (2560 / Q4_K_BLOCK_ELEMS) * Q4_K_BLOCK_BYTES; // 2_949_120
        let payload = vec![0u8; len];
        let manifest = serde_json::json!([
            {
                "key": "layers.0.self_attn.q_proj.weight",
                "shape": [2048, 2560],
                "format": "Q4_K",
                "offset": 0,
                "length": len,
            }
        ]);
        let tmp = make_vindex_with_attn_q4k(&payload, manifest);
        let mut idx = empty_vindex();
        idx.load_attn_q4k(tmp.path()).expect("clean stride must load");
    }

    /// Regression: an attn_weights_q4k.bin written with the legacy
    /// 148-byte block_q4_K layout must be rejected at load time. The
    /// kernel reads 144-byte GGUF strides; without this check, every
    /// row's read window drifts by 4 bytes per superblock and the GPU
    /// prefill silently produces all-NaN.
    #[test]
    fn load_attn_q4k_rejects_legacy_148_byte_stride() {
        let bad_len = 2048 * (2560 / 256) * 148; // 3_031_040 — what 8-Apr vindexes have
        let payload = vec![0u8; bad_len];
        let manifest = serde_json::json!([
            {
                "key": "layers.0.self_attn.q_proj.weight",
                "shape": [2048, 2560],
                "format": "Q4_K",
                "offset": 0,
                "length": bad_len,
            }
        ]);
        let tmp = make_vindex_with_attn_q4k(&payload, manifest);
        let mut idx = empty_vindex();
        let err = idx
            .load_attn_q4k(tmp.path())
            .expect_err("legacy 148-byte stride must be rejected");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("rebuild the vindex"),
            "error must guide the user to rebuild — got: {msg}"
        );
        assert!(
            msg.contains("3031040") || msg.contains("2949120"),
            "error must include both lengths so the user can see the drift — got: {msg}"
        );
    }

    /// A length that's neither 144 × n nor 148 × n still gets rejected
    /// (anything that's not the canonical stride is an error).
    #[test]
    fn load_attn_q4k_rejects_arbitrary_wrong_length() {
        let weird_len = 2_949_120 + 17; // off-by-17 — definitely not aligned
        let payload = vec![0u8; weird_len];
        let manifest = serde_json::json!([
            {
                "key": "layers.0.self_attn.q_proj.weight",
                "shape": [2048, 2560],
                "format": "Q4_K",
                "offset": 0,
                "length": weird_len,
            }
        ]);
        let tmp = make_vindex_with_attn_q4k(&payload, manifest);
        let mut idx = empty_vindex();
        idx.load_attn_q4k(tmp.path())
            .expect_err("non-canonical stride must be rejected");
    }

    /// Q6_K stride (210 bytes per 256-element block) must also validate
    /// — V projections in Gemma 3 4B are Q6_K and would suffer the same
    /// silent-drift class of bug.
    #[test]
    fn load_attn_q4k_validates_q6k_v_projection() {
        use larql_models::quant::ggml::{
            K_QUANT_BLOCK_ELEMS, Q4_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES,
        };
        let q4k_len = 1024 * (2560 / K_QUANT_BLOCK_ELEMS) * Q4_K_BLOCK_BYTES; // K proj: 1024 × 1440
        let q6k_len = 1024 * (2560 / K_QUANT_BLOCK_ELEMS) * Q6_K_BLOCK_BYTES; // V proj: 1024 × 2100
        let total = q4k_len + q6k_len;
        let payload = vec![0u8; total];
        let manifest = serde_json::json!([
            {
                "key": "layers.0.self_attn.k_proj.weight",
                "shape": [1024, 2560],
                "format": "Q4_K",
                "offset": 0,
                "length": q4k_len,
            },
            {
                "key": "layers.0.self_attn.v_proj.weight",
                "shape": [1024, 2560],
                "format": "Q6_K",
                "offset": q4k_len,
                "length": q6k_len,
            }
        ]);
        let tmp = make_vindex_with_attn_q4k(&payload, manifest);
        let mut idx = empty_vindex();
        idx.load_attn_q4k(tmp.path())
            .expect("matched Q4_K + Q6_K strides must load");
    }

    /// A Q6_K manifest entry recorded with a Q4_K-sized length (210 vs
    /// 144 confusion at write time) must be rejected.
    #[test]
    fn load_attn_q4k_rejects_q6k_with_q4k_stride() {
        use larql_models::quant::ggml::{K_QUANT_BLOCK_ELEMS, Q4_K_BLOCK_BYTES};
        let wrong_len = 1024 * (2560 / K_QUANT_BLOCK_ELEMS) * Q4_K_BLOCK_BYTES; // Q4_K stride for Q6_K tensor
        let payload = vec![0u8; wrong_len];
        let manifest = serde_json::json!([
            {
                "key": "layers.0.self_attn.v_proj.weight",
                "shape": [1024, 2560],
                "format": "Q6_K",
                "offset": 0,
                "length": wrong_len,
            }
        ]);
        let tmp = make_vindex_with_attn_q4k(&payload, manifest);
        let mut idx = empty_vindex();
        idx.load_attn_q4k(tmp.path())
            .expect_err("Q6_K tensor with Q4_K length must be rejected");
    }
}
