//! Bake gate overlay entries into `gate_vectors.bin`. File layout
//! follows the per-layer `VindexLayerInfo` records in `config.layers`:
//!
//! - dtype from `config.dtype` (may be f16 or f32)
//! - each layer has an explicit byte `offset` and `length` — layers
//!   are NOT necessarily contiguous or in `layer` order within the
//!   array. Writing at a naive `layer_index × layer_bytes` offset
//!   lands in the wrong slice and corrupts whichever layer actually
//!   lives at that byte position, which wrecks inference across the
//!   whole file (validated by `refine_demo22`: the naive offsets
//!   collapsed compiled-session retrieval from 8/10 to 0/10).
//!
//! Within a layer, feature `f`'s gate is the row at
//! `info.offset + f × hidden × bpf` — contiguous per-feature.

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};

use crate::error::LqlError;
use larql_vindex::format::filenames::GATE_VECTORS_BIN;

use super::{copy_for_patch, BYTES_PER_F16, BYTES_PER_F32};

pub(in crate::executor::lifecycle::compile) fn patch_gate_vectors(
    source_dir: &std::path::Path,
    dest_dir: &std::path::Path,
    config: &larql_vindex::VindexConfig,
    gate_overrides: &HashMap<(usize, usize), Vec<f32>>,
) -> Result<(), LqlError> {
    if gate_overrides.is_empty() {
        return Ok(());
    }
    let src = source_dir.join(GATE_VECTORS_BIN);
    let dst = dest_dir.join(GATE_VECTORS_BIN);
    if !src.exists() {
        return Err(LqlError::Execution(
            "source vindex has no gate_vectors.bin — cannot bake gate overrides".into(),
        ));
    }

    // `dst` was hard-linked from the source earlier in the compile
    // bake's unchanging-files loop, so we need a real copy we own
    // before seek-writing into it.
    copy_for_patch(&src, &dst)?;

    let hidden = config.hidden_size;
    let bpf = larql_vindex::config::dtype::bytes_per_float(config.dtype);

    // Map layer → (byte offset, num_features). Layers that don't
    // appear in `config.layers` have no gate data in the file (e.g.
    // embedding-only layers); an override targeting them is a bug —
    // we error out clearly.
    let mut layer_info: HashMap<usize, (u64, usize)> = HashMap::new();
    for info in &config.layers {
        layer_info.insert(info.layer, (info.offset, info.num_features));
    }

    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&dst)
        .map_err(|e| LqlError::exec("open gate_vectors.bin", e))?;

    let row_bytes = hidden * bpf;
    let mut row_buf = vec![0u8; row_bytes];

    for ((layer, feature), gate_vec) in gate_overrides {
        if gate_vec.len() != hidden {
            return Err(LqlError::Execution(format!(
                "gate override at L{layer} F{feature} has wrong shape: {} (expected {hidden})",
                gate_vec.len()
            )));
        }
        let Some(&(layer_offset, nf)) = layer_info.get(layer) else {
            return Err(LqlError::Execution(format!(
                "gate override at L{layer} F{feature}: layer {layer} not in config.layers \
                 (source vindex has no gate data for this layer)"
            )));
        };
        if *feature >= nf {
            return Err(LqlError::Execution(format!(
                "gate override at L{layer} F{feature} out of range (layer has {nf} features)"
            )));
        }

        encode_row(gate_vec, bpf, &mut row_buf)?;

        let feature_offset = layer_offset + (*feature * row_bytes) as u64;
        file.seek(SeekFrom::Start(feature_offset))
            .map_err(|e| LqlError::exec("seek gate_vectors", e))?;
        file.write_all(&row_buf)
            .map_err(|e| LqlError::exec("write gate_vectors row", e))?;
    }
    Ok(())
}

/// Encode a row of f32 values into the file's native dtype (f32 or f16).
fn encode_row(gate_vec: &[f32], bpf: usize, row_buf: &mut [u8]) -> Result<(), LqlError> {
    if bpf == BYTES_PER_F32 {
        for (i, v) in gate_vec.iter().enumerate() {
            row_buf[i * BYTES_PER_F32..(i + 1) * BYTES_PER_F32].copy_from_slice(&v.to_le_bytes());
        }
    } else if bpf == BYTES_PER_F16 {
        for (i, v) in gate_vec.iter().enumerate() {
            let half_bits = larql_models::quant::half::f32_to_f16(*v);
            row_buf[i * BYTES_PER_F16..(i + 1) * BYTES_PER_F16]
                .copy_from_slice(&half_bits.to_le_bytes());
        }
    } else {
        return Err(LqlError::Execution(format!(
            "unsupported gate_vectors.bin dtype: bpf={bpf}",
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_row_f32_round_trips() {
        let mut buf = vec![0u8; 4 * BYTES_PER_F32];
        let row = vec![1.0f32, -2.5, 3.25, 0.0];
        encode_row(&row, BYTES_PER_F32, &mut buf).unwrap();
        for (i, want) in row.iter().enumerate() {
            let cell = i * BYTES_PER_F32;
            let got = f32::from_le_bytes(buf[cell..cell + BYTES_PER_F32].try_into().unwrap());
            assert_eq!(got, *want);
        }
    }

    #[test]
    fn encode_row_f16_round_trips_within_tolerance() {
        let mut buf = vec![0u8; 4 * BYTES_PER_F16];
        let row = vec![1.0f32, -2.5, 3.25, 0.0];
        encode_row(&row, BYTES_PER_F16, &mut buf).unwrap();
        for (i, want) in row.iter().enumerate() {
            let cell = i * BYTES_PER_F16;
            let bits = u16::from_le_bytes(buf[cell..cell + BYTES_PER_F16].try_into().unwrap());
            let got = larql_models::quant::half::f16_to_f32(bits);
            assert!((got - want).abs() < 0.01);
        }
    }

    #[test]
    fn encode_row_rejects_unsupported_bpf() {
        let mut buf = vec![0u8; 8];
        let err = encode_row(&[1.0, 2.0], 3, &mut buf).unwrap_err();
        assert!(err.to_string().contains("unsupported"));
    }
}
