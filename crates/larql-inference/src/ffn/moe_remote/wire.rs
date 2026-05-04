use serde::{Deserialize, Serialize};

// ── Binary wire format ────────────────────────────────────────────────────────
//
// Content-Type: application/x-larql-expert
//
// Request:  [N u32][hidden u32] + N × [layer u32][expert_id u32][f32 × hidden]
// Response: [N u32][hidden u32][latency_ms f32] + N × [layer u32][expert_id u32][f32 × hidden]
//
// All integers and floats are little-endian.  This is ~6× smaller than JSON
// for typical 2816-float payloads and avoids serde_json float formatting.

pub const EXPERT_BINARY_CONTENT_TYPE: &str = "application/x-larql-expert";

/// Content type for the `/v1/experts/layer-batch` endpoint — the layer-batched
/// MoE wire format that ships one residual + K (expert_id, weight) pairs and
/// receives back ONE weighted-sum vector.  Eliminates the K-1 redundant
/// residual copies on the wire (~78 KB per call at Gemma 4 26B-A4B sizes)
/// and the K-1 redundant `pre_experts_norm` + Q8_K quantisations on the
/// server (~10-20 µs per layer of CPU work).
pub const LAYER_BATCH_CONTENT_TYPE: &str = "application/x-larql-experts-layer";

/// f16 variant of the layer-batch wire format.  Halves the per-call wire
/// bytes (residual + weighted-sum response): 11 KB → 5.5 KB at hidden=2816.
/// Quantisation is `f32 → IEEE-754 half`, ~3 decimal digits of precision —
/// well within MoE activation noise (Q8_K already adds ~0.4% per-element
/// quant error on the activation in the SDOT path; f16 wire adds another
/// ~0.05% which is negligible).  Mathematically identical when both sides
/// dequantise to f32 before compute.
pub const LAYER_BATCH_F16_CONTENT_TYPE: &str = "application/x-larql-experts-layer-f16";

// ── Layer-batch wire format ───────────────────────────────────────────────────
//
// Content-Type: application/x-larql-experts-layer
//
// Request:  [layer u32][hidden u32][K u32]
//           + hidden × f32  (residual, sent ONCE)
//           + K × [expert_id u32, weight f32]
//
// Response: [hidden u32][latency_ms f32]
//           + hidden × f32  (router-weighted sum across the K experts)
//
// Server-side fast path: the response is the result of
// `run_experts_cpu_batch(layer, residual, expert_ids, expert_weights)` — the
// server applies pre_experts_norm once, quantises h_norm to Q8_K once, and
// fans out the K expert kernels with the shared activation.

/// Encode a layer-batch request.
pub fn encode_layer_batch_request(
    layer: usize,
    residual: &[f32],
    expert_ids: &[u32],
    expert_weights: &[f32],
) -> Vec<u8> {
    let hidden = residual.len();
    let k = expert_ids.len();
    debug_assert_eq!(k, expert_weights.len());
    let mut buf = Vec::with_capacity(12 + hidden * 4 + k * 8);
    buf.extend_from_slice(&(layer as u32).to_le_bytes());
    buf.extend_from_slice(&(hidden as u32).to_le_bytes());
    buf.extend_from_slice(&(k as u32).to_le_bytes());
    for &v in residual {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    for (i, &eid) in expert_ids.iter().enumerate() {
        buf.extend_from_slice(&eid.to_le_bytes());
        buf.extend_from_slice(&expert_weights[i].to_le_bytes());
    }
    buf
}

/// Decode a layer-batch request from raw bytes.  Returns
/// `(layer, residual, expert_ids, expert_weights)` or `None` on truncation.
pub fn decode_layer_batch_request(bytes: &[u8]) -> Option<(usize, Vec<f32>, Vec<u32>, Vec<f32>)> {
    if bytes.len() < 12 {
        return None;
    }
    let layer = u32::from_le_bytes(bytes[0..4].try_into().ok()?) as usize;
    let hidden = u32::from_le_bytes(bytes[4..8].try_into().ok()?) as usize;
    let k = u32::from_le_bytes(bytes[8..12].try_into().ok()?) as usize;
    let want = 12 + hidden * 4 + k * 8;
    if bytes.len() < want {
        return None;
    }
    let mut pos = 12usize;
    let residual: Vec<f32> = bytes[pos..pos + hidden * 4]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    pos += hidden * 4;
    let mut expert_ids = Vec::with_capacity(k);
    let mut expert_weights = Vec::with_capacity(k);
    for _ in 0..k {
        let eid = u32::from_le_bytes(bytes[pos..pos + 4].try_into().ok()?);
        let w = f32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().ok()?);
        expert_ids.push(eid);
        expert_weights.push(w);
        pos += 8;
    }
    Some((layer, residual, expert_ids, expert_weights))
}

/// Encode a layer-batch response (one weighted-sum vector).
pub fn encode_layer_batch_response(weighted_sum: &[f32], latency_ms: f32) -> Vec<u8> {
    let hidden = weighted_sum.len();
    let mut buf = Vec::with_capacity(8 + hidden * 4);
    buf.extend_from_slice(&(hidden as u32).to_le_bytes());
    buf.extend_from_slice(&latency_ms.to_le_bytes());
    for &v in weighted_sum {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

/// Decode a layer-batch response.  Returns the weighted-sum vector or `None`
/// on truncation.  Discards the latency_ms field (informational only).
pub fn decode_layer_batch_response(bytes: &[u8]) -> Option<Vec<f32>> {
    if bytes.len() < 8 {
        return None;
    }
    let hidden = u32::from_le_bytes(bytes[0..4].try_into().ok()?) as usize;
    if bytes.len() < 8 + hidden * 4 {
        return None;
    }
    Some(
        bytes[8..8 + hidden * 4]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
    )
}

// ── f16 wire helpers ──────────────────────────────────────────────────────────
// IEEE-754 binary16 conversion.  Round-to-nearest-even for finite values;
// saturates on overflow; preserves NaN.  Same behaviour as the `half` crate
// but kept inline here so the wire layer doesn't take a new dep.

#[inline(always)]
pub(super) fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;
    if exp == 0xFF {
        // Inf or NaN.
        if mant == 0 {
            return sign | 0x7C00;
        }
        return sign | 0x7C00 | ((mant >> 13) as u16) | 0x0001; // canonical NaN
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 0x1F {
        // Overflow → ±Inf.
        return sign | 0x7C00;
    }
    if new_exp <= 0 {
        // Subnormal or zero.
        if new_exp < -10 {
            return sign;
        }
        let mant_full = mant | 0x80_0000; // implicit leading 1
        let shift = (14 - new_exp) as u32;
        let new_mant = (mant_full >> shift) as u16;
        // Round-to-nearest-even on the dropped bit.
        let round_bit = (mant_full >> (shift - 1)) & 1;
        let sticky = mant_full & ((1u32 << (shift - 1)) - 1);
        let mut out = new_mant;
        if round_bit != 0 && (sticky != 0 || (new_mant & 1) != 0) {
            out += 1;
        }
        return sign | out;
    }
    // Normal.
    let new_mant = (mant >> 13) as u16;
    let round_bit = (mant >> 12) & 1;
    let sticky = mant & 0xFFF;
    let mut combined = ((new_exp as u16) << 10) | new_mant;
    if round_bit != 0 && (sticky != 0 || (new_mant & 1) != 0) {
        combined += 1; // may carry into exponent — that's fine, IEEE-correct
    }
    sign | combined
}

#[inline(always)]
pub(super) fn f16_bits_to_f32(bits: u16) -> f32 {
    // Mirrors `larql_compute::cpu::ops::q4_common::f16_to_f32` (kept inline
    // so the wire layer stays dependency-free).  Bit-exact for all 65536
    // f16 inputs vs the powi reference.
    let bits = bits as u32;
    let sign = (bits & 0x8000) << 16;
    let exp = (bits >> 10) & 0x1F;
    let mant = bits & 0x3FF;
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        let lz = (mant as u16).leading_zeros() - 6;
        let new_mant = (mant << (lz + 14)) & 0x7F_FFFF;
        let new_exp = (127u32 - 14 - lz) << 23;
        return f32::from_bits(sign | new_exp | new_mant);
    }
    if exp == 31 {
        return f32::from_bits(sign | 0x7F80_0000 | (mant << 13));
    }
    let new_exp = (exp + (127 - 15)) << 23;
    f32::from_bits(sign | new_exp | (mant << 13))
}

/// Encode a layer-batch request with f16 residual.  Same shape as the f32
/// version but residual bytes are 2 per element (vs 4).  Header layout
/// `[layer u32][hidden u32][K u32]` is unchanged so the server can size
/// the read slice correctly.
pub fn encode_layer_batch_request_f16(
    layer: usize,
    residual: &[f32],
    expert_ids: &[u32],
    expert_weights: &[f32],
) -> Vec<u8> {
    let hidden = residual.len();
    let k = expert_ids.len();
    debug_assert_eq!(k, expert_weights.len());
    let mut buf = Vec::with_capacity(12 + hidden * 2 + k * 8);
    buf.extend_from_slice(&(layer as u32).to_le_bytes());
    buf.extend_from_slice(&(hidden as u32).to_le_bytes());
    buf.extend_from_slice(&(k as u32).to_le_bytes());
    for &v in residual {
        buf.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
    }
    for (i, &eid) in expert_ids.iter().enumerate() {
        buf.extend_from_slice(&eid.to_le_bytes());
        // Weights stay f32 — only K of them, and they're routing
        // probabilities (small dynamic range, but full f32 precision keeps
        // the renormalised sum exactly 1.0).
        buf.extend_from_slice(&expert_weights[i].to_le_bytes());
    }
    buf
}

/// Decode an f16 layer-batch request.  Reconstructs `residual` to f32 on
/// the server before passing into `run_experts_cpu_batch`.
pub fn decode_layer_batch_request_f16(
    bytes: &[u8],
) -> Option<(usize, Vec<f32>, Vec<u32>, Vec<f32>)> {
    if bytes.len() < 12 {
        return None;
    }
    let layer = u32::from_le_bytes(bytes[0..4].try_into().ok()?) as usize;
    let hidden = u32::from_le_bytes(bytes[4..8].try_into().ok()?) as usize;
    let k = u32::from_le_bytes(bytes[8..12].try_into().ok()?) as usize;
    let want = 12 + hidden * 2 + k * 8;
    if bytes.len() < want {
        return None;
    }
    let mut pos = 12usize;
    let residual: Vec<f32> = bytes[pos..pos + hidden * 2]
        .chunks_exact(2)
        .map(|b| f16_bits_to_f32(u16::from_le_bytes([b[0], b[1]])))
        .collect();
    pos += hidden * 2;
    let mut expert_ids = Vec::with_capacity(k);
    let mut expert_weights = Vec::with_capacity(k);
    for _ in 0..k {
        let eid = u32::from_le_bytes(bytes[pos..pos + 4].try_into().ok()?);
        let w = f32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().ok()?);
        expert_ids.push(eid);
        expert_weights.push(w);
        pos += 8;
    }
    Some((layer, residual, expert_ids, expert_weights))
}

/// Encode the f16 layer-batch response (weighted-sum vector packed as f16).
pub fn encode_layer_batch_response_f16(weighted_sum: &[f32], latency_ms: f32) -> Vec<u8> {
    let hidden = weighted_sum.len();
    let mut buf = Vec::with_capacity(8 + hidden * 2);
    buf.extend_from_slice(&(hidden as u32).to_le_bytes());
    buf.extend_from_slice(&latency_ms.to_le_bytes());
    for &v in weighted_sum {
        buf.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
    }
    buf
}

/// Decode the f16 layer-batch response back to f32 for client-side
/// accumulation.
pub fn decode_layer_batch_response_f16(bytes: &[u8]) -> Option<Vec<f32>> {
    if bytes.len() < 8 {
        return None;
    }
    let hidden = u32::from_le_bytes(bytes[0..4].try_into().ok()?) as usize;
    if bytes.len() < 8 + hidden * 2 {
        return None;
    }
    Some(
        bytes[8..8 + hidden * 2]
            .chunks_exact(2)
            .map(|b| f16_bits_to_f32(u16::from_le_bytes([b[0], b[1]])))
            .collect(),
    )
}

/// Encode a batch of expert requests as binary.
pub fn encode_expert_request(items: &[ExpertCallItem]) -> Vec<u8> {
    let n = items.len();
    let hidden = items.first().map(|r| r.residual.len()).unwrap_or(0);
    let mut buf = Vec::with_capacity(8 + n * (8 + hidden * 4));
    buf.extend_from_slice(&(n as u32).to_le_bytes());
    buf.extend_from_slice(&(hidden as u32).to_le_bytes());
    for item in items {
        buf.extend_from_slice(&(item.layer as u32).to_le_bytes());
        buf.extend_from_slice(&(item.expert_id as u32).to_le_bytes());
        for &v in &item.residual {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    buf
}

/// Decode a binary expert response. Returns None on truncation.
pub fn decode_expert_response(bytes: &[u8]) -> Option<Vec<ExpertResultItem>> {
    if bytes.len() < 12 {
        return None;
    }
    let n = u32::from_le_bytes(bytes[0..4].try_into().ok()?) as usize;
    let hidden = u32::from_le_bytes(bytes[4..8].try_into().ok()?) as usize;
    // bytes[8..12] = latency_ms f32 (informational, skip)
    let mut pos = 12usize;
    let item_bytes = 8 + hidden * 4;
    if bytes.len() < 12 + n * item_bytes {
        return None;
    }
    let mut results = Vec::with_capacity(n);
    for _ in 0..n {
        let layer = u32::from_le_bytes(bytes[pos..pos + 4].try_into().ok()?) as usize;
        let expert_id = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().ok()?) as usize;
        pos += 8;
        let output: Vec<f32> = bytes[pos..pos + hidden * 4]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        pos += hidden * 4;
        results.push(ExpertResultItem {
            layer,
            expert_id,
            output,
        });
    }
    Some(results)
}

/// Decode a binary expert request from the server side.
pub fn decode_expert_request(bytes: &[u8]) -> Option<Vec<ExpertCallItem>> {
    if bytes.len() < 8 {
        return None;
    }
    let n = u32::from_le_bytes(bytes[0..4].try_into().ok()?) as usize;
    let hidden = u32::from_le_bytes(bytes[4..8].try_into().ok()?) as usize;
    let mut pos = 8usize;
    let item_bytes = 8 + hidden * 4;
    if bytes.len() < 8 + n * item_bytes {
        return None;
    }
    let mut items = Vec::with_capacity(n);
    for _ in 0..n {
        let layer = u32::from_le_bytes(bytes[pos..pos + 4].try_into().ok()?) as usize;
        let expert_id = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().ok()?) as usize;
        pos += 8;
        let residual: Vec<f32> = bytes[pos..pos + hidden * 4]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        pos += hidden * 4;
        items.push(ExpertCallItem {
            layer,
            expert_id,
            residual,
        });
    }
    Some(items)
}

/// Encode a batch of expert results as binary (server-side response).
pub fn encode_expert_response(items: &[ExpertResultItem], latency_ms: f32) -> Vec<u8> {
    let n = items.len();
    let hidden = items.first().map(|r| r.output.len()).unwrap_or(0);
    let mut buf = Vec::with_capacity(12 + n * (8 + hidden * 4));
    buf.extend_from_slice(&(n as u32).to_le_bytes());
    buf.extend_from_slice(&(hidden as u32).to_le_bytes());
    buf.extend_from_slice(&latency_ms.to_le_bytes());
    for item in items {
        buf.extend_from_slice(&(item.layer as u32).to_le_bytes());
        buf.extend_from_slice(&(item.expert_id as u32).to_le_bytes());
        for &v in &item.output {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    buf
}

// ── Wire types ────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct BatchRequest<'a> {
    requests: &'a [ExpertCallItem],
}

#[derive(Serialize, Clone)]
pub struct ExpertCallItem {
    pub layer: usize,
    pub expert_id: usize,
    pub residual: Vec<f32>,
}

#[derive(Deserialize)]
struct BatchResponse {
    results: Vec<ExpertResultItem>,
}

#[derive(Deserialize)]
pub struct ExpertResultItem {
    pub layer: usize,
    pub expert_id: usize,
    pub output: Vec<f32>,
}
