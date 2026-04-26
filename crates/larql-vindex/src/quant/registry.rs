//! GGML quant-format registry — single dispatch table for the formats
//! the vindex reads.
//!
//! Today five places (`walk.rs:dequant`, `walk.rs:row_dot`,
//! `walk.rs:row_scaled_add`, `walk.rs:byte-stride math`,
//! `walk.rs:single-row decode`) match on a `&str` format tag and
//! dispatch by name. That's 25+ string literals and several
//! silent-fallback `_ => None` arms — adding the next format means
//! editing eight files and hoping you didn't miss one of the
//! match arms.
//!
//! The registry collapses that to **one place**. Adding Q5_K is:
//!
//! 1. Implement `quantize_q5_k` / `dequantize_q5_k` / `q5k_row_dot` /
//!    `q5k_row_scaled_add` in `larql-models::quant::ggml`.
//! 2. Add one `QuantFormatInfo` entry to `QUANT_FORMATS` below.
//! 3. (Optionally) extend `crate::config::types::QuantFormat`.
//!
//! Calling code at the seam looks like:
//!
//! ```ignore
//! let info = registry::lookup(format_tag)
//!     .ok_or_else(|| Error::UnknownFormat(format_tag.into()))?;
//! let bytes_per_row = info.bytes_per_row(hidden);
//! info.row_dot(row_bytes, x)
//! ```
//!
//! No more silent `_ => None` arms — `lookup` returns `None` exactly
//! once at the seam, and the caller is forced to handle it.

use larql_models::quant::ggml;

/// Function-pointer signatures that mirror `larql_models::quant::ggml`.
type DequantizeFn = fn(&[u8], usize) -> Result<Vec<f32>, larql_models::ModelError>;
type RowDotFn = fn(&[u8], &[f32]) -> Result<f32, larql_models::ModelError>;
type RowScaledAddFn = fn(&[u8], f32, &mut [f32]) -> Result<(), larql_models::ModelError>;

/// One entry in the format registry. `tag` is the on-disk string
/// (matches what's in `interleaved_q4k_manifest.json`).
pub struct QuantFormatInfo {
    /// Serialized identifier — appears in manifests and the
    /// `QuantBlockFormat` serde enum.
    pub tag: &'static str,

    /// Elements per super-block. The full GGML K-quant family uses
    /// 256; legacy Q4_0 / Q8_0 use 32. Don't hard-code "256" inline.
    pub block_elements: usize,

    /// Bytes per super-block.
    /// - Q4_0: 18 bytes / 32 elements (legacy 4-bit)
    /// - Q4_K: 144 bytes / 256 elements
    /// - Q6_K: 210 bytes / 256 elements
    /// - Q8_0: 34 bytes / 32 elements
    pub bytes_per_block: usize,

    /// Decode `data` (assumed `n_elements`-shaped) into a fresh `Vec<f32>`.
    pub dequantize: DequantizeFn,

    /// Fused dot — `row_bytes` is one row, `x` matches its decoded
    /// element count. `None` for formats without a dedicated kernel.
    pub row_dot: Option<RowDotFn>,

    /// Fused scaled-add — `out += alpha * decode(row_bytes)`. `None`
    /// for formats without a dedicated kernel.
    pub row_scaled_add: Option<RowScaledAddFn>,
}

impl QuantFormatInfo {
    /// Bytes occupied by one row of `n_cols` elements. Returns `None`
    /// if the row isn't a whole number of blocks.
    #[inline]
    pub fn bytes_per_row(&self, n_cols: usize) -> Option<usize> {
        if !n_cols.is_multiple_of(self.block_elements) {
            return None;
        }
        Some((n_cols / self.block_elements) * self.bytes_per_block)
    }

    /// Convenience: dequantise one block and return the f32 vector.
    /// Routes to the registered `dequantize` fn pointer.
    pub fn dequantize_block(&self, bytes: &[u8]) -> Result<Vec<f32>, larql_models::ModelError> {
        (self.dequantize)(bytes, self.block_elements)
    }
}

/// All quant formats the vindex understands as of 2026-04-25. Adding a
/// format = one entry here + the ggml functions it points at. The
/// caller-visible `tag` is the only string literal that should appear
/// in match arms anywhere else; everything else flows through this
/// table.
pub static QUANT_FORMATS: &[QuantFormatInfo] = &[
    QuantFormatInfo {
        tag: "Q4_K",
        block_elements: 256,
        bytes_per_block: 144,
        dequantize: ggml::dequantize_q4_k,
        row_dot: Some(ggml::q4k_row_dot),
        row_scaled_add: Some(ggml::q4k_row_scaled_add),
    },
    QuantFormatInfo {
        tag: "Q6_K",
        block_elements: 256,
        bytes_per_block: 210,
        dequantize: ggml::dequantize_q6_k,
        row_dot: Some(ggml::q6k_row_dot),
        row_scaled_add: Some(ggml::q6k_row_scaled_add),
    },
];

/// Look up a format by its on-disk tag (e.g. `"Q4_K"`). Returns
/// `None` for unknown / typo'd tags — caller must handle this once
/// at the seam instead of having silent fallbacks scattered through
/// match arms.
pub fn lookup(tag: &str) -> Option<&'static QuantFormatInfo> {
    QUANT_FORMATS.iter().find(|f| f.tag == tag)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_tags_unique() {
        let tags: std::collections::HashSet<_> = QUANT_FORMATS.iter().map(|f| f.tag).collect();
        assert_eq!(
            tags.len(),
            QUANT_FORMATS.len(),
            "duplicate format tag in QUANT_FORMATS"
        );
    }

    #[test]
    fn lookup_known_formats() {
        let q4k = lookup("Q4_K").expect("Q4_K should be registered");
        assert_eq!(q4k.block_elements, 256);
        assert_eq!(q4k.bytes_per_block, 144);
        assert!(q4k.row_dot.is_some());
        assert!(q4k.row_scaled_add.is_some());

        let q6k = lookup("Q6_K").expect("Q6_K should be registered");
        assert_eq!(q6k.bytes_per_block, 210);
    }

    #[test]
    fn lookup_unknown_returns_none() {
        // The whole point of the registry: typo'd tags fail loudly at
        // the seam instead of triggering a silent `_ => None` arm.
        assert!(lookup("Q5_K").is_none());
        assert!(lookup("q4_k").is_none()); // case-sensitive — manifest uses "Q4_K"
        assert!(lookup("").is_none());
    }

    #[test]
    fn bytes_per_row_block_aligned() {
        let q4k = lookup("Q4_K").unwrap();
        // hidden = 2560 = 10 × 256 → 10 × 144 = 1440 bytes
        assert_eq!(q4k.bytes_per_row(2560), Some(1440));
        // hidden = 2048 = 8 × 256 → 8 × 144 = 1152 bytes
        assert_eq!(q4k.bytes_per_row(2048), Some(1152));
        // hidden = 100 not a multiple of 256 → None
        assert_eq!(q4k.bytes_per_row(100), None);
    }
}
