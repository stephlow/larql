//! Per-layer FFN weight writer — `layers/layer_{L:02}.weights` format (§5.12).
//!
//! Unified for dense (num_entries=1) and MoE (num_entries=num_experts) models.
//! The file header declares the quantization format; all entries in the file
//! use it uniformly. Structure is orthogonal to quantization: adding a new
//! quant (Q8, FP4, …) is a new `QuantFormat` variant; the file layout is unchanged.
//!
//! Binary layout:
//!   [header]       6 × u32: magic "LYRW", format_version=1, quant_format,
//!                            num_entries, intermediate, hidden
//!   [offset table] num_entries × 4 × u64: gate_up_off, gate_up_bytes,
//!                                          down_off, down_bytes
//!   [entry 0 gate+up] quant_format blocks, shape [2*inter, hidden]
//!   [entry 0 down]    quant_format blocks, shape [hidden, inter_padded]
//!   [entry 1 gate+up] ...

use std::io::{BufWriter, Write};
use std::path::Path;

use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k};
use larql_models::ModelArchitecture;

use crate::VindexError;

/// Format tag written into the file header. Extend as new formats land.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerWeightFormat {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    Q4_0 = 3,
    Q4_K = 4,
    Q6_K = 5,
    Q8_0 = 6,
    FP4 = 7,
}

impl LayerWeightFormat {
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

const MAGIC: u32 = u32::from_le_bytes(*b"LYRW");
const FORMAT_VERSION: u32 = 1;

/// One quantized entry: gate+up bytes and down bytes, both in the same format.
pub struct LayerEntry {
    pub gate_up: Vec<u8>, // Q4_K [2*inter, hidden]
    pub down: Vec<u8>,    // Q6_K [hidden, inter_padded]  (same format as gate_up)
}

/// Write `layers/layer_{L:02}.weights` for one layer.
///
/// `entries`: one element for dense, `num_experts` elements for MoE.
/// All entries use `format` uniformly.
pub fn write_layer_weights(
    dir: &Path,
    layer: usize,
    format: LayerWeightFormat,
    entries: &[LayerEntry],
    inter: usize,
    hidden: usize,
) -> Result<(), VindexError> {
    let layers_dir = dir.join("layers");
    std::fs::create_dir_all(&layers_dir)?;

    let filename = format!("layers/layer_{layer:02}.weights");
    let path = dir.join(&filename);
    let mut f = BufWriter::new(std::fs::File::create(&path)?);

    let num_entries = entries.len() as u32;

    // ── Header (6 × u32) ──
    f.write_all(&MAGIC.to_le_bytes())?;
    f.write_all(&FORMAT_VERSION.to_le_bytes())?;
    f.write_all(&format.as_u32().to_le_bytes())?;
    f.write_all(&num_entries.to_le_bytes())?;
    f.write_all(&(inter as u32).to_le_bytes())?;
    f.write_all(&(hidden as u32).to_le_bytes())?;

    // ── Offset table (num_entries × 4 × u64) ──
    // Compute offsets: header=24 bytes, table=num_entries*32 bytes, then data.
    let header_bytes: u64 = 24;
    let table_bytes: u64 = num_entries as u64 * 32;
    let mut cursor: u64 = header_bytes + table_bytes;

    let mut offsets: Vec<(u64, u64, u64, u64)> = Vec::with_capacity(entries.len());
    for entry in entries {
        let gate_up_off = cursor;
        let gate_up_bytes = entry.gate_up.len() as u64;
        cursor += gate_up_bytes;
        let down_off = cursor;
        let down_bytes = entry.down.len() as u64;
        cursor += down_bytes;
        offsets.push((gate_up_off, gate_up_bytes, down_off, down_bytes));
    }

    for (gate_up_off, gate_up_bytes, down_off, down_bytes) in &offsets {
        f.write_all(&gate_up_off.to_le_bytes())?;
        f.write_all(&gate_up_bytes.to_le_bytes())?;
        f.write_all(&down_off.to_le_bytes())?;
        f.write_all(&down_bytes.to_le_bytes())?;
    }

    // ── Data ──
    for entry in entries {
        f.write_all(&entry.gate_up)?;
        f.write_all(&entry.down)?;
    }
    f.flush()?;
    Ok(())
}

/// BF16 byte slice (2 bytes per element) → f32 Vec.
pub fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| {
            let bits = u32::from(u16::from_le_bytes([b[0], b[1]])) << 16;
            f32::from_bits(bits)
        })
        .collect()
}

/// Quantize an f32 slice to the specified format.
/// Returns the quantized byte Vec.
///
/// The `block_width` is the number of columns (used for padding to the
/// nearest block boundary when required by the format).
pub fn quantize_f32(data: &[f32], format: LayerWeightFormat) -> Vec<u8> {
    match format {
        LayerWeightFormat::Q4_K => quantize_q4_k(data),
        LayerWeightFormat::Q6_K => quantize_q6_k(data),
        LayerWeightFormat::F32 => bytemuck_f32_to_bytes(data),
        LayerWeightFormat::F16 | LayerWeightFormat::BF16 => {
            // Store as f32 — f16/bf16 conversion not yet implemented here.
            // Caller should use F32 format for now.
            bytemuck_f32_to_bytes(data)
        }
        _ => quantize_q4_k(data), // fallback: Q4_K for unimplemented formats
    }
}

fn bytemuck_f32_to_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Pad an [out_rows, in_cols] row-major f32 matrix so `in_cols` is a
/// multiple of 256 (required for Q4_K super-block alignment).
/// Returns the original slice unchanged if already aligned.
pub fn pad_cols_to_256(data: &[f32], out_rows: usize, in_cols: usize) -> (Vec<f32>, usize) {
    let padded = in_cols.div_ceil(256) * 256;
    if padded == in_cols {
        return (data.to_vec(), in_cols);
    }
    let mut v = vec![0.0f32; out_rows * padded];
    for row in 0..out_rows {
        v[row * padded..row * padded + in_cols]
            .copy_from_slice(&data[row * in_cols..(row + 1) * in_cols]);
    }
    (v, padded)
}

/// Build quantized entries for a dense FFN layer from f32 gate/up/down tensors.
///
/// `gate_f32`: [inter, hidden], `up_f32`: [inter, hidden], `down_f32`: [hidden, inter].
/// All entries in the output use `format` uniformly.
pub fn quantize_dense_entry(
    gate_f32: &[f32],
    up_f32: &[f32],
    down_f32: &[f32],
    inter: usize,
    hidden: usize,
    format: LayerWeightFormat,
) -> LayerEntry {
    // gate+up interleaved: [gate rows, up rows] = [2*inter, hidden]
    let mut gate_up_f32 = Vec::with_capacity(2 * inter * hidden);
    gate_up_f32.extend_from_slice(gate_f32);
    gate_up_f32.extend_from_slice(up_f32);
    let gate_up = quantize_f32(&gate_up_f32, format);

    // down: [hidden, inter] padded to 256-element column boundary
    let (down_padded, _) = pad_cols_to_256(down_f32, hidden, inter);
    let down = quantize_f32(&down_padded, format);

    LayerEntry { gate_up, down }
}

/// Build quantized entries for one MoE layer from BF16-packed expert tensors.
///
/// `gate_up_bf16`: [num_experts, 2*moe_inter, hidden] BF16.
/// `down_bf16`:    [num_experts, hidden, moe_inter] BF16.
/// All entries use `format` uniformly — no mixing of formats within a file.
pub fn quantize_moe_entries(
    gate_up_bf16: &[u8],
    down_bf16: &[u8],
    num_experts: usize,
    moe_inter: usize,
    hidden: usize,
    format: LayerWeightFormat,
) -> Vec<LayerEntry> {
    let gate_up_stride = 2 * moe_inter * hidden * 2; // bytes per expert (BF16)
    let down_stride = hidden * moe_inter * 2; // bytes per expert (BF16)

    (0..num_experts)
        .map(|e| {
            let gu_bytes = &gate_up_bf16[e * gate_up_stride..(e + 1) * gate_up_stride];
            let gate_up_f32 = bf16_bytes_to_f32(gu_bytes);
            let gate_up = quantize_f32(&gate_up_f32, format);

            let dn_bytes = &down_bf16[e * down_stride..(e + 1) * down_stride];
            let down_f32_src = bf16_bytes_to_f32(dn_bytes);
            // Pad inter → 256-element boundary (required for block formats like Q4_K)
            let (down_padded, _) = pad_cols_to_256(&down_f32_src, hidden, moe_inter);
            let down = quantize_f32(&down_padded, format);

            LayerEntry { gate_up, down }
        })
        .collect()
}

/// Parse a `layers/layer_{L}.weights` file header and offset table.
///
/// Returns `(format, num_entries, inter, hidden, offsets)` where
/// `offsets[e] = (gate_up_offset, gate_up_bytes, down_offset, down_bytes)`.
pub fn parse_layer_weights_header(
    data: &[u8],
) -> Option<(
    LayerWeightFormat,
    usize,
    usize,
    usize,
    Vec<(usize, usize, usize, usize)>,
)> {
    if data.len() < 24 {
        return None;
    }
    let magic = u32::from_le_bytes(data[0..4].try_into().ok()?);
    if magic != MAGIC {
        return None;
    }
    // format_version at [4..8] — currently ignored, forward-compatible
    let quant_raw = u32::from_le_bytes(data[8..12].try_into().ok()?);
    let format = match quant_raw {
        0 => LayerWeightFormat::F32,
        1 => LayerWeightFormat::F16,
        2 => LayerWeightFormat::BF16,
        3 => LayerWeightFormat::Q4_0,
        4 => LayerWeightFormat::Q4_K,
        5 => LayerWeightFormat::Q6_K,
        6 => LayerWeightFormat::Q8_0,
        7 => LayerWeightFormat::FP4,
        _ => return None,
    };
    let num_entries = u32::from_le_bytes(data[12..16].try_into().ok()?) as usize;
    let inter = u32::from_le_bytes(data[16..20].try_into().ok()?) as usize;
    let hidden = u32::from_le_bytes(data[20..24].try_into().ok()?) as usize;

    let table_start = 24usize;
    let table_end = table_start + num_entries * 32;
    if data.len() < table_end {
        return None;
    }

    let mut offsets = Vec::with_capacity(num_entries);
    for e in 0..num_entries {
        let base = table_start + e * 32;
        let gate_up_off = u64::from_le_bytes(data[base..base + 8].try_into().ok()?) as usize;
        let gate_up_bytes = u64::from_le_bytes(data[base + 8..base + 16].try_into().ok()?) as usize;
        let down_off = u64::from_le_bytes(data[base + 16..base + 24].try_into().ok()?) as usize;
        let down_bytes = u64::from_le_bytes(data[base + 24..base + 32].try_into().ok()?) as usize;
        offsets.push((gate_up_off, gate_up_bytes, down_off, down_bytes));
    }
    Some((format, num_entries, inter, hidden, offsets))
}
