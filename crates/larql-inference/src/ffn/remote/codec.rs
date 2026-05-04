//! Binary wire codec for the LARQL FFN remote protocol.
//!
//! See the `super` module doc for the full binary frame layout.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub(super) const BINARY_CT: &str = "application/x-larql-ffn";
pub(super) const BATCH_MARKER: u32 = 0xFFFF_FFFF;

// ── Wire types (JSON fallback) ────────────────────────────────────────────────

#[derive(Serialize)]
#[allow(dead_code)]
pub(super) struct WalkFfnHttpRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layers: Option<Vec<usize>>,
    pub residual: Vec<f32>,
    pub seq_len: usize,
    pub full_output: bool,
}

#[derive(Deserialize)]
pub(super) struct WalkFfnSingleResponse {
    #[allow(dead_code)]
    pub layer: usize,
    pub output: Vec<f32>,
    #[allow(dead_code)]
    pub seq_len: usize,
}

// ── Latency profiling result ──────────────────────────────────────────────────

/// Breakdown returned by [`super::http::RemoteWalkBackend::probe_latency`].
#[derive(Debug, Clone)]
pub struct RemoteLatencyStats {
    /// Wall-clock round-trip (client-measured), averaged over `samples` calls.
    pub total_ms: f64,
    /// FFN compute time reported by the server in the binary response header.
    pub server_ms: f64,
    /// `total_ms - server_ms`: HTTP framing + TCP + serialization overhead.
    pub overhead_ms: f64,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub samples: usize,
}

impl std::fmt::Display for RemoteLatencyStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "layers={} hidden={} samples={}\n  total    {:7.2} ms\n  server   {:7.2} ms  (FFN compute)\n  overhead {:7.2} ms  (HTTP + TCP + framing)",
            self.num_layers, self.hidden_size, self.samples,
            self.total_ms, self.server_ms, self.overhead_ms,
        )
    }
}

// ── Binary codec ──────────────────────────────────────────────────────────────

/// Encode a request as binary.
/// `layer` and `layers` are mutually exclusive; pass `None` for the unused one.
pub(crate) fn encode_binary_request(
    layer: Option<usize>,
    layers: Option<&[usize]>,
    residual: &[f32],
    seq_len: usize,
    full_output: bool,
    top_k: usize,
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(16 + residual.len() * 4);

    if let Some(ls) = layers {
        buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
        buf.extend_from_slice(&(ls.len() as u32).to_le_bytes());
        for &l in ls {
            buf.extend_from_slice(&(l as u32).to_le_bytes());
        }
    } else {
        let l = layer.unwrap_or(0) as u32;
        buf.extend_from_slice(&l.to_le_bytes());
    }

    buf.extend_from_slice(&(seq_len as u32).to_le_bytes());
    buf.extend_from_slice(&(full_output as u32).to_le_bytes());
    buf.extend_from_slice(&(top_k as u32).to_le_bytes());
    for &v in residual {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

/// Decode a binary single-layer full_output response.
/// Returns `(layer, output_floats)`.
pub(crate) fn decode_binary_single(body: &[u8]) -> Result<(usize, Vec<f32>), String> {
    if body.len() < 12 {
        return Err(format!("binary response too short: {} bytes", body.len()));
    }
    let marker = u32::from_le_bytes(body[0..4].try_into().unwrap());
    if marker == BATCH_MARKER {
        return Err("expected single-layer response but got batch marker".into());
    }
    let layer = marker as usize;
    // bytes 4-7: seq_len (ignored here — caller validates against expected shape)
    // bytes 8-11: latency f32
    let floats: Vec<f32> = body[12..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    Ok((layer, floats))
}

/// Decode a binary batch full_output response.
/// Returns a map from layer → output floats.
pub(crate) fn decode_binary_batch(body: &[u8]) -> Result<HashMap<usize, Vec<f32>>, String> {
    if body.len() < 12 {
        return Err(format!(
            "binary batch response too short: {} bytes",
            body.len()
        ));
    }
    let marker = u32::from_le_bytes(body[0..4].try_into().unwrap());

    // Single-layer response — accept it as a batch of 1.
    if marker != BATCH_MARKER {
        let (layer, floats) = decode_binary_single(body)?;
        let mut m = HashMap::new();
        m.insert(layer, floats);
        return Ok(m);
    }

    let num_results = u32::from_le_bytes(body[4..8].try_into().unwrap()) as usize;
    // bytes 8-11: latency f32 (skip)
    let mut offset = 12usize;
    let mut out = HashMap::with_capacity(num_results);

    for _ in 0..num_results {
        if body.len() < offset + 12 {
            return Err("binary batch: truncated result header".into());
        }
        let layer = u32::from_le_bytes(body[offset..offset + 4].try_into().unwrap()) as usize;
        // offset+4: seq_len (skip)
        let num_floats =
            u32::from_le_bytes(body[offset + 8..offset + 12].try_into().unwrap()) as usize;
        offset += 12;
        let bytes_needed = num_floats * 4;
        if body.len() < offset + bytes_needed {
            return Err(format!(
                "binary batch: truncated output for layer {layer}: need {bytes_needed}, have {}",
                body.len() - offset
            ));
        }
        let floats: Vec<f32> = body[offset..offset + bytes_needed]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        offset += bytes_needed;
        out.insert(layer, floats);
    }
    Ok(out)
}

/// Extract the `latency_ms` f32 embedded at bytes 8-11 of a binary response.
/// Returns 0.0 if the body is too short or the value is non-finite.
pub(super) fn extract_response_latency_ms(body: &[u8]) -> f64 {
    if body.len() < 12 {
        return 0.0;
    }
    // Both single-layer and batch responses have latency_ms at offset 8.
    let v = f32::from_le_bytes(body[8..12].try_into().unwrap());
    if v.is_finite() {
        v as f64
    } else {
        0.0
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── JSON serialisation ────────────────────────────────────────────────────

    #[test]
    fn request_serializes_with_seq_len_and_full_output() {
        let req = WalkFfnHttpRequest {
            layer: Some(3),
            layers: None,
            residual: vec![0.1, -0.2, 0.3, 0.4],
            seq_len: 2,
            full_output: true,
        };
        let v: serde_json::Value = serde_json::to_value(&req).unwrap();
        assert_eq!(v["layer"], 3);
        assert_eq!(v["seq_len"], 2);
        assert_eq!(v["full_output"], true);
        assert!(
            v.get("layers").is_none() || v["layers"].is_null(),
            "layers should not appear when None, got: {v}"
        );
        assert_eq!(v["residual"].as_array().unwrap().len(), 4);
    }

    #[test]
    fn response_deserializes_hidden_vector() {
        let json = serde_json::json!({
            "layer": 5,
            "output": [0.1, 0.2, 0.3, 0.4, 0.5],
            "seq_len": 1,
            "latency_ms": 2.5,
        });
        let parsed: WalkFfnSingleResponse = serde_json::from_value(json).unwrap();
        assert_eq!(parsed.layer, 5);
        assert_eq!(parsed.output.len(), 5);
        assert_eq!(parsed.seq_len, 1);
    }

    #[test]
    fn response_deserializes_multi_token_output() {
        let flat: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let json = serde_json::json!({
            "layer": 0,
            "output": flat,
            "seq_len": 3,
        });
        let parsed: WalkFfnSingleResponse = serde_json::from_value(json).unwrap();
        assert_eq!(parsed.output.len(), 12);
        assert_eq!(parsed.seq_len, 3);
    }

    // ── encode_binary_request ─────────────────────────────────────────────────

    #[test]
    fn encode_single_layer_header() {
        let residual = vec![1.0f32, 2.0, 3.0, 4.0];
        let body = encode_binary_request(Some(7), None, &residual, 1, true, 256);
        // First u32 = layer index
        let layer = u32::from_le_bytes(body[0..4].try_into().unwrap());
        assert_eq!(layer, 7);
        let seq_len = u32::from_le_bytes(body[4..8].try_into().unwrap());
        assert_eq!(seq_len, 1);
        let flags = u32::from_le_bytes(body[8..12].try_into().unwrap());
        assert_eq!(flags & 1, 1); // full_output
        let top_k = u32::from_le_bytes(body[12..16].try_into().unwrap());
        assert_eq!(top_k, 256);
        assert_eq!(body.len(), 16 + 4 * 4);
    }

    #[test]
    fn encode_batch_header() {
        let residual = vec![0.5f32; 4];
        let body = encode_binary_request(None, Some(&[5, 20, 30]), &residual, 1, true, 512);
        let marker = u32::from_le_bytes(body[0..4].try_into().unwrap());
        assert_eq!(marker, BATCH_MARKER);
        let num_layers = u32::from_le_bytes(body[4..8].try_into().unwrap());
        assert_eq!(num_layers, 3);
        let l0 = u32::from_le_bytes(body[8..12].try_into().unwrap());
        let l1 = u32::from_le_bytes(body[12..16].try_into().unwrap());
        let l2 = u32::from_le_bytes(body[16..20].try_into().unwrap());
        assert_eq!((l0, l1, l2), (5, 20, 30));
    }

    #[test]
    fn encode_residual_values_preserved() {
        let residual = vec![-1.5f32, 0.0, 3.25];
        let body = encode_binary_request(Some(0), None, &residual, 1, true, 8092);
        let offset = 16; // 4 header u32s × 4 bytes
        let v0 = f32::from_le_bytes(body[offset..offset + 4].try_into().unwrap());
        let v1 = f32::from_le_bytes(body[offset + 4..offset + 8].try_into().unwrap());
        let v2 = f32::from_le_bytes(body[offset + 8..offset + 12].try_into().unwrap());
        assert_eq!(v0.to_bits(), (-1.5f32).to_bits());
        assert_eq!(v1.to_bits(), 0.0f32.to_bits());
        assert!((v2 - 3.25f32).abs() < 1e-5);
    }

    // ── decode_binary_single ──────────────────────────────────────────────────

    fn make_single_response(layer: u32, seq_len: u32, latency: f32, output: &[f32]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&layer.to_le_bytes());
        buf.extend_from_slice(&seq_len.to_le_bytes());
        buf.extend_from_slice(&latency.to_le_bytes());
        for &v in output {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    fn make_batch_response(latency: f32, entries: &[(u32, &[f32])]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
        buf.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        buf.extend_from_slice(&latency.to_le_bytes());
        for &(layer, floats) in entries {
            buf.extend_from_slice(&layer.to_le_bytes());
            buf.extend_from_slice(&1u32.to_le_bytes()); // seq_len
            buf.extend_from_slice(&(floats.len() as u32).to_le_bytes());
            for &v in floats {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        buf
    }

    #[test]
    fn decode_single_response_correct() {
        let output = vec![1.0f32, -2.0, 3.5];
        let body = make_single_response(5, 1, 7.3, &output);
        let (layer, floats) = decode_binary_single(&body).unwrap();
        assert_eq!(layer, 5);
        assert_eq!(floats.len(), 3);
        assert!((floats[0] - 1.0).abs() < 1e-6);
        assert!((floats[1] - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn decode_single_response_rejects_batch_marker() {
        let body = make_batch_response(1.0, &[(5, &[1.0, 2.0])]);
        let result = decode_binary_single(&body);
        assert!(result.is_err());
    }

    #[test]
    fn decode_single_response_too_short() {
        let result = decode_binary_single(&[0u8; 8]);
        assert!(result.is_err());
    }

    // ── decode_binary_batch ───────────────────────────────────────────────────

    #[test]
    fn decode_batch_response_correct() {
        let body = make_batch_response(15.0, &[(5, &[1.0, 2.0]), (20, &[3.0, 4.0])]);
        let map = decode_binary_batch(&body).unwrap();
        assert_eq!(map.len(), 2);
        let v5 = map.get(&5).unwrap();
        assert_eq!(v5.len(), 2);
        assert!((v5[0] - 1.0).abs() < 1e-6);
        let v20 = map.get(&20).unwrap();
        assert!((v20[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn decode_batch_accepts_single_response() {
        // A server returning single-layer response to a same-shard batch.
        let output = vec![7.0f32, 8.0];
        let body = make_single_response(10, 1, 5.0, &output);
        let map = decode_binary_batch(&body).unwrap();
        assert_eq!(map.len(), 1);
        assert!(map.contains_key(&10));
    }

    #[test]
    fn decode_batch_truncated_returns_error() {
        let mut body = make_batch_response(1.0, &[(5, &[1.0, 2.0])]);
        body.truncate(body.len() - 4); // cut off last float
        let result = decode_binary_batch(&body);
        assert!(result.is_err());
    }

    #[test]
    fn binary_request_response_roundtrip() {
        // Encode a single-layer request, then simulate what the server echoes.
        let residual = vec![0.1f32, 0.2, 0.3, 0.4];
        let req = encode_binary_request(Some(5), None, &residual, 1, true, 8092);
        // Simulate server extracting the layer.
        let layer = u32::from_le_bytes(req[0..4].try_into().unwrap());
        assert_eq!(layer, 5);

        // Simulate server response.
        let output = vec![0.9f32, 0.8, 0.7, 0.6];
        let resp = make_single_response(layer, 1, 8.5, &output);
        let (resp_layer, floats) = decode_binary_single(&resp).unwrap();
        assert_eq!(resp_layer as u32, layer);
        assert_eq!(floats, output);
    }
}
