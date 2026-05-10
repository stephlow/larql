//! Apply MEMIT ΔW deltas additively into `down_weights.bin`. Runs
//! AFTER `patch_down_weights` — column-replace covers legacy arch-A
//! inserts, MEMIT covers compose-mode inserts; both contribute to
//! the final compiled slab.

use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};

use crate::error::LqlError;
use larql_vindex::format::filenames::DOWN_WEIGHTS_BIN;

use super::{detect_down_dtype_bytes, BYTES_PER_F16, BYTES_PER_F32};

pub(in crate::executor::lifecycle::compile) fn apply_memit_deltas_to_down_weights(
    dest_dir: &std::path::Path,
    config: &larql_vindex::VindexConfig,
    results: &[larql_inference::MemitResult],
) -> Result<(), LqlError> {
    let dst = dest_dir.join(DOWN_WEIGHTS_BIN);
    if !dst.exists() {
        return Err(LqlError::Execution(
            "apply_memit_deltas: down_weights.bin not found in output dir".into(),
        ));
    }

    let total = std::fs::metadata(&dst)
        .map_err(|e| LqlError::exec("stat down_weights.bin", e))?
        .len() as usize;

    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let num_layers = config.num_layers;
    let elements_per_layer = hidden * intermediate;
    let total_elements = num_layers * elements_per_layer;

    let dtype_bytes = detect_down_dtype_bytes(total, total_elements)?;
    let layer_bytes = elements_per_layer * dtype_bytes;

    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&dst)
        .map_err(|e| LqlError::exec("open down_weights.bin for MEMIT apply", e))?;

    let mut buf = vec![0u8; layer_bytes];

    for result in results {
        let layer = result.layer;
        if layer >= num_layers {
            return Err(LqlError::Execution(format!(
                "MEMIT result references layer {layer} but vindex has {num_layers} layers"
            )));
        }

        let shape = result.delta_w.shape();
        if shape[0] != hidden || shape[1] != intermediate {
            return Err(LqlError::Execution(format!(
                "MEMIT ΔW shape {:?} mismatches vindex shape [{hidden}, {intermediate}] at L{layer}",
                shape
            )));
        }

        let layer_offset = (layer * layer_bytes) as u64;
        file.seek(SeekFrom::Start(layer_offset))
            .map_err(|e| LqlError::exec("seek down_weights slab", e))?;
        file.read_exact(&mut buf)
            .map_err(|e| LqlError::exec("read down_weights slab", e))?;

        // Row-major layout: cell = (row * intermediate + feat) * dtype_bytes.
        // Skip cells whose delta is exactly 0 — most ΔW are sparse.
        for row in 0..hidden {
            for feat in 0..intermediate {
                let cell = (row * intermediate + feat) * dtype_bytes;
                let delta = result.delta_w[[row, feat]];
                if delta == 0.0 {
                    continue;
                }
                if dtype_bytes == BYTES_PER_F32 {
                    let cur = f32::from_le_bytes([
                        buf[cell],
                        buf[cell + 1],
                        buf[cell + 2],
                        buf[cell + 3],
                    ]);
                    let next = cur + delta;
                    buf[cell..cell + BYTES_PER_F32].copy_from_slice(&next.to_le_bytes());
                } else {
                    let cur_half = u16::from_le_bytes([buf[cell], buf[cell + 1]]);
                    let cur = larql_models::quant::half::f16_to_f32(cur_half);
                    let next = cur + delta;
                    let next_half = larql_models::quant::half::f32_to_f16(next);
                    buf[cell..cell + BYTES_PER_F16].copy_from_slice(&next_half.to_le_bytes());
                }
            }
        }

        file.seek(SeekFrom::Start(layer_offset))
            .map_err(|e| LqlError::exec("seek down_weights slab (write)", e))?;
        file.write_all(&buf)
            .map_err(|e| LqlError::exec("write down_weights slab", e))?;
    }

    Ok(())
}
