//! Data type conversion utilities for vindex storage.
//!
//! Supports f32 (default) and f16 (half precision) storage.
//! f16 halves file sizes with negligible impact on KNN accuracy.

use serde::{Deserialize, Serialize};

/// Storage precision for vindex binary files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StorageDtype {
    F32,
    F16,
}

impl Default for StorageDtype {
    fn default() -> Self {
        Self::F32
    }
}

impl std::fmt::Display for StorageDtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F32 => write!(f, "f32"),
            Self::F16 => write!(f, "f16"),
        }
    }
}

/// Convert f32 slice to f16 bytes (2 bytes per float).
pub fn f32_to_f16_bytes(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 2);
    for &v in data {
        let bits = f32_to_half(v);
        out.extend_from_slice(&bits.to_le_bytes());
    }
    out
}

/// Convert f16 bytes back to f32 vec.
pub fn f16_bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|b| half_to_f32(u16::from_le_bytes([b[0], b[1]])))
        .collect()
}

/// Write f32 data as either f32 or f16 bytes.
pub fn encode_floats(data: &[f32], dtype: StorageDtype) -> Vec<u8> {
    match dtype {
        StorageDtype::F32 => {
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
            };
            bytes.to_vec()
        }
        StorageDtype::F16 => f32_to_f16_bytes(data),
    }
}

/// Read bytes back to f32, handling dtype.
pub fn decode_floats(data: &[u8], dtype: StorageDtype) -> Vec<f32> {
    match dtype {
        StorageDtype::F32 => {
            let floats: &[f32] = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4)
            };
            floats.to_vec()
        }
        StorageDtype::F16 => f16_bytes_to_f32(data),
    }
}

/// Bytes per float for a given dtype.
pub fn bytes_per_float(dtype: StorageDtype) -> usize {
    match dtype {
        StorageDtype::F32 => 4,
        StorageDtype::F16 => 2,
    }
}

fn f32_to_half(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;

    if exp == 255 {
        // Inf/NaN
        return sign | 0x7C00 | if mant != 0 { 0x0200 } else { 0 };
    }

    let exp16 = exp - 127 + 15;

    if exp16 >= 31 {
        // Overflow → Inf
        return sign | 0x7C00;
    }
    if exp16 <= 0 {
        // Underflow → zero (or denorm, but we skip denorms for speed)
        return sign;
    }

    sign | ((exp16 as u16) << 10) | ((mant >> 13) as u16)
}

fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 { return f32::from_bits(sign); }
        let mut e = 1u32;
        let mut m = mant;
        while (m & 0x400) == 0 { m <<= 1; e += 1; }
        return f32::from_bits(sign | ((127 - 15 + 1 - e) << 23) | ((m & 0x3FF) << 13));
    }
    if exp == 31 {
        return f32::from_bits(sign | (0xFF << 23) | (mant << 13));
    }
    f32::from_bits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_f16_round_trip() {
        let values = vec![0.0f32, 1.0, -1.0, 0.5, 100.0, -0.001, 3.14159];
        let encoded = f32_to_f16_bytes(&values);
        assert_eq!(encoded.len(), values.len() * 2);
        let decoded = f16_bytes_to_f32(&encoded);
        assert_eq!(decoded.len(), values.len());
        for (orig, dec) in values.iter().zip(decoded.iter()) {
            let err = (orig - dec).abs();
            assert!(err < 0.01 * orig.abs().max(0.001),
                "f16 round-trip: {} → {} (err={})", orig, dec, err);
        }
    }

    #[test]
    fn encode_decode_f32() {
        let data = vec![1.0f32, 2.0, 3.0];
        let encoded = encode_floats(&data, StorageDtype::F32);
        assert_eq!(encoded.len(), 12); // 3 × 4
        let decoded = decode_floats(&encoded, StorageDtype::F32);
        assert_eq!(decoded, data);
    }

    #[test]
    fn encode_decode_f16() {
        let data = vec![1.0f32, 2.0, 3.0];
        let encoded = encode_floats(&data, StorageDtype::F16);
        assert_eq!(encoded.len(), 6); // 3 × 2
        let decoded = decode_floats(&encoded, StorageDtype::F16);
        for (orig, dec) in data.iter().zip(decoded.iter()) {
            assert!((orig - dec).abs() < 0.01);
        }
    }
}
