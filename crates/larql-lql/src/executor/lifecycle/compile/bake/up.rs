//! Bake up overlay entries into `up_weights.bin` (or whichever file
//! the manifest points the up_proj tensors at).
//!
//! Dense FFN at inference reads via `load_model_weights`, which
//! consults `weight_manifest.json` to find each tensor's `(file,
//! offset, length, shape)` entry. Layout per layer:
//!
//! - the file the manifest points to (normally `up_weights.bin`)
//! - per-layer tensor at `entry.offset` with `entry.length` bytes
//! - dtype inferred from `byte_count / expected_floats` (4 = f32,
//!   2 = f16)
//! - shape `[num_features, hidden_size]` row-major; feature `f`'s
//!   row starts at `entry.offset + f × hidden × bpf`
//!
//! `up_features.bin` (a separate feature-major f32 file used only by
//! `walk_ffn_sparse`) is NOT touched here. Writing to the wrong file
//! was the root cause of `refine_demo22`'s regression from 8/10 to
//! 0/10 compiled retrieval.

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};

use crate::error::LqlError;
use larql_vindex::format::filenames::WEIGHT_MANIFEST_JSON;

use super::{copy_for_patch, BYTES_PER_F16, BYTES_PER_F32};

/// Manifest key fragment that identifies an up_proj tensor entry.
/// Used to filter `weight_manifest.json` while searching for the
/// per-layer up tensors.
const UP_PROJ_KEY_FRAGMENT: &str = "up_proj";

/// Manifest key fragment that prefixes the layer index in tensor
/// keys (`layers.{N}.…`).
const LAYERS_KEY_FRAGMENT: &str = "layers.";

pub(in crate::executor::lifecycle::compile) fn patch_up_weights(
    source_dir: &std::path::Path,
    dest_dir: &std::path::Path,
    config: &larql_vindex::VindexConfig,
    up_overrides: &HashMap<(usize, usize), Vec<f32>>,
) -> Result<(), LqlError> {
    if up_overrides.is_empty() {
        return Ok(());
    }

    // Read the weight manifest from the SOURCE vindex — the dest
    // copy was hard-linked from source and we haven't modified the
    // manifest.
    let manifest_path = source_dir.join(WEIGHT_MANIFEST_JSON);
    if !manifest_path.exists() {
        // Manifestless vindex — we can't safely locate the up
        // tensors. Skip silently; the compiled vindex still has
        // baked down_weights.bin and overlay gates in
        // gate_vectors.bin so the install is at least partially live.
        return Ok(());
    }
    let layer_up_lookup = parse_manifest_for_up(&manifest_path)?;

    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let expected_floats = intermediate * hidden;

    // File handles cached per file so we don't re-open per write.
    let mut file_cache: HashMap<String, std::fs::File> = HashMap::new();

    for ((layer, feature), up_vec) in up_overrides {
        if up_vec.len() != hidden {
            return Err(LqlError::Execution(format!(
                "up override at L{layer} F{feature} has wrong shape: {} (expected {hidden})",
                up_vec.len()
            )));
        }
        if *feature >= intermediate {
            return Err(LqlError::Execution(format!(
                "up override at L{layer} F{feature} out of range (intermediate = {intermediate})"
            )));
        }

        let Some((file_name, offset, length)) = layer_up_lookup.get(layer) else {
            // No manifest entry for this layer's up projection —
            // skip silently, the layer's up is not materialised.
            continue;
        };

        let bpf = if *length as usize == expected_floats * BYTES_PER_F32 {
            BYTES_PER_F32
        } else if *length as usize == expected_floats * BYTES_PER_F16 {
            BYTES_PER_F16
        } else {
            return Err(LqlError::Execution(format!(
                "up weight for L{layer} has length {length} ≠ \
                 expected {} (f32) or {} (f16)",
                expected_floats * BYTES_PER_F32,
                expected_floats * BYTES_PER_F16,
            )));
        };

        // Lazily open + copy the file if we haven't touched it yet.
        if !file_cache.contains_key(file_name) {
            let src = source_dir.join(file_name);
            let dst = dest_dir.join(file_name);
            if !src.exists() {
                return Err(LqlError::Execution(format!(
                    "weight file {file_name} referenced by manifest but missing from source"
                )));
            }
            copy_for_patch(&src, &dst)?;
            let f = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&dst)
                .map_err(|e| LqlError::exec(format!("open {file_name}"), e))?;
            file_cache.insert(file_name.clone(), f);
        }
        let file = file_cache.get_mut(file_name).unwrap();

        let row_bytes = hidden * bpf;
        let mut row_buf = vec![0u8; row_bytes];
        encode_row(up_vec, bpf, &mut row_buf);

        let feature_offset = offset + (*feature * row_bytes) as u64;
        file.seek(SeekFrom::Start(feature_offset))
            .map_err(|e| LqlError::exec(format!("seek {file_name}"), e))?;
        file.write_all(&row_buf)
            .map_err(|e| LqlError::exec(format!("write {file_name} row"), e))?;
    }
    Ok(())
}

/// Walk the manifest JSON and collect `layer → (file, offset, length)`
/// for every up_proj tensor. Entries without all four fields, or with
/// a key that doesn't match the `layers.{N}.…up_proj` pattern, are
/// silently skipped (works for Llama/Gemma/Mistral; MoE experts and
/// non-standard naming conventions skip and the layer's overlay
/// becomes a silent no-op).
fn parse_manifest_for_up(
    manifest_path: &std::path::Path,
) -> Result<HashMap<usize, (String, u64, u64)>, LqlError> {
    let text = std::fs::read_to_string(manifest_path)
        .map_err(|e| LqlError::exec("read weight_manifest.json", e))?;
    let entries: Vec<serde_json::Value> = serde_json::from_str(&text)
        .map_err(|e| LqlError::exec("parse weight_manifest.json", e))?;

    let mut out: HashMap<usize, (String, u64, u64)> = HashMap::new();
    for entry in &entries {
        let Some(key) = entry.get("key").and_then(|v| v.as_str()) else {
            continue;
        };
        if !key.contains(UP_PROJ_KEY_FRAGMENT) {
            continue;
        }
        let Some(file) = entry.get("file").and_then(|v| v.as_str()) else {
            continue;
        };
        let Some(offset) = entry.get("offset").and_then(|v| v.as_u64()) else {
            continue;
        };
        let Some(length) = entry.get("length").and_then(|v| v.as_u64()) else {
            continue;
        };
        let Some(layer) = parse_layer_from_key(key) else {
            continue;
        };
        out.insert(layer, (file.to_string(), offset, length));
    }
    Ok(out)
}

/// Extract the layer index from a manifest key like
/// `model.layers.{N}.mlp.up_proj.weight` → `N`. Returns `None` if
/// the key doesn't follow the `layers.{int}.…` shape.
fn parse_layer_from_key(key: &str) -> Option<usize> {
    let rest = key.split(LAYERS_KEY_FRAGMENT).nth(1)?;
    let layer_str = rest.split('.').next()?;
    layer_str.parse::<usize>().ok()
}

fn encode_row(up_vec: &[f32], bpf: usize, row_buf: &mut [u8]) {
    if bpf == BYTES_PER_F32 {
        for (i, v) in up_vec.iter().enumerate() {
            row_buf[i * BYTES_PER_F32..(i + 1) * BYTES_PER_F32].copy_from_slice(&v.to_le_bytes());
        }
    } else {
        for (i, v) in up_vec.iter().enumerate() {
            let half_bits = larql_models::quant::half::f32_to_f16(*v);
            row_buf[i * BYTES_PER_F16..(i + 1) * BYTES_PER_F16]
                .copy_from_slice(&half_bits.to_le_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_layer_from_key_extracts_int() {
        assert_eq!(parse_layer_from_key("model.layers.7.mlp.up_proj"), Some(7));
        assert_eq!(parse_layer_from_key("model.layers.0.mlp.up_proj"), Some(0));
        assert_eq!(
            parse_layer_from_key("model.layers.31.mlp.up_proj.weight"),
            Some(31)
        );
    }

    #[test]
    fn parse_layer_from_key_rejects_non_layer_keys() {
        assert_eq!(parse_layer_from_key("model.embed_tokens"), None);
        assert_eq!(parse_layer_from_key("layers.x.mlp"), None);
        assert_eq!(parse_layer_from_key("nothing here"), None);
    }

    #[test]
    fn encode_row_f32_writes_each_cell() {
        let mut buf = vec![0u8; 3 * BYTES_PER_F32];
        encode_row(&[1.5, -2.0, 0.25], BYTES_PER_F32, &mut buf);
        for (i, want) in [1.5f32, -2.0, 0.25].iter().enumerate() {
            let cell = i * BYTES_PER_F32;
            let got = f32::from_le_bytes(buf[cell..cell + BYTES_PER_F32].try_into().unwrap());
            assert_eq!(got, *want);
        }
    }

    #[test]
    fn encode_row_f16_writes_each_cell() {
        let mut buf = vec![0u8; 3 * BYTES_PER_F16];
        encode_row(&[1.0, 2.0, 3.0], BYTES_PER_F16, &mut buf);
        for (i, want) in [1.0f32, 2.0, 3.0].iter().enumerate() {
            let cell = i * BYTES_PER_F16;
            let bits = u16::from_le_bytes(buf[cell..cell + BYTES_PER_F16].try_into().unwrap());
            let got = larql_models::quant::half::f16_to_f32(bits);
            assert!((got - want).abs() < 0.01);
        }
    }

    #[test]
    fn key_fragments_match_actual_manifest_shape() {
        // Pinned: parse_manifest_for_up depends on these exact shapes.
        assert_eq!(UP_PROJ_KEY_FRAGMENT, "up_proj");
        assert_eq!(LAYERS_KEY_FRAGMENT, "layers.");
    }
}
