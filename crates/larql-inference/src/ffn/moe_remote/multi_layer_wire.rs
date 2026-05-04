//! Binary wire format for `POST /v1/experts/multi-layer-batch`.
//!
//! Collapses 30 per-layer HTTP requests into one per shard, eliminating the
//! per-request HTTPS overhead (~20 ms × 30 = 600 ms in the predispatch path).
//! The server processes tasks sequentially so rayon runs at full utilisation
//! (no oversubscription); the client parallelises across shards only.
//!
//! Request layout (little-endian):
//!   u32  num_tasks
//!   for each task:
//!     u32  layer
//!     u32  hidden            (residual length = h_post_attn size)
//!     u32  num_experts
//!     f32[hidden]  residual
//!     u32[n]       expert_ids
//!     f32[n]       weights
//!
//! Response layout:
//!   u32  num_results
//!   for each result:
//!     u32  layer
//!     u32  hidden
//!     f32[hidden]  h2         (raw weighted sum; caller applies post-experts norm)

pub const MULTI_LAYER_BATCH_CONTENT_TYPE: &str = "application/x-larql-experts-multi-layer";

/// Q8K-prenormed variant: client sends `h_norm` pre-quantised to Q8_K
/// (already computed during routing — zero extra client compute).  Server
/// skips `pre_experts_norm` + `quantize_h_norm_for_q4k` and calls the
/// matvec directly.  4× smaller upload than the f32 residual path.
///
/// Request layout — same header as f32, but residual field replaced:
///   u32  num_tasks
///   for each task:
///     u32  layer
///     u32  hidden              (= n_blocks × 256)
///     u32  num_experts
///     i8[hidden]  q8k_qs       (quantised activation)
///     f32[n_blocks]  q8k_d     (per-super-block scales)
///     i16[n_blocks × 8]  q8k_sums  (precomputed sub-block sums)
///     u32[num_experts]  expert_ids
///     f32[num_experts]  weights
pub const MULTI_LAYER_BATCH_Q8K_CONTENT_TYPE: &str = "application/x-larql-experts-multi-layer-q8k";

pub struct MultiLayerTask {
    pub layer: usize,
    pub residual: Vec<f32>,
    pub expert_ids: Vec<u32>,
    pub weights: Vec<f32>,
}

/// Q8K-prenormed task: carries already-quantised h_norm so the server skips
/// normalisation and directly calls `q4k_q8k_matvec_into`.
pub struct MultiLayerTaskQ8K {
    pub layer: usize,
    pub hidden: usize,
    /// Flat i8 activation: `qs[block * 256 .. (block+1) * 256]` per block.
    pub qs: Vec<i8>,
    /// Per-super-block f32 scale: `d[block]`.
    pub d: Vec<f32>,
    /// Per-sub-block i16 sums: `sums[block * 8 + sb]`.
    pub sums: Vec<i16>,
    pub expert_ids: Vec<u32>,
    pub weights: Vec<f32>,
}

pub struct MultiLayerResult {
    pub layer: usize,
    pub h2: Vec<f32>,
}

pub fn encode_multi_layer_request(tasks: &[MultiLayerTask]) -> Vec<u8> {
    let cap = 4 + tasks
        .iter()
        .map(|t| 12 + t.residual.len() * 4 + t.expert_ids.len() * 8)
        .sum::<usize>();
    let mut buf = Vec::with_capacity(cap);
    push_u32(&mut buf, tasks.len() as u32);
    for t in tasks {
        push_u32(&mut buf, t.layer as u32);
        push_u32(&mut buf, t.residual.len() as u32);
        push_u32(&mut buf, t.expert_ids.len() as u32);
        for &v in &t.residual {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for &e in &t.expert_ids {
            push_u32(&mut buf, e);
        }
        for &w in &t.weights {
            buf.extend_from_slice(&w.to_le_bytes());
        }
    }
    buf
}

pub fn decode_multi_layer_request(bytes: &[u8]) -> Option<Vec<MultiLayerTask>> {
    let mut pos = 0;
    let n = read_u32(bytes, &mut pos)? as usize;
    let mut tasks = Vec::with_capacity(n);
    for _ in 0..n {
        let layer = read_u32(bytes, &mut pos)? as usize;
        let hidden = read_u32(bytes, &mut pos)? as usize;
        let ne = read_u32(bytes, &mut pos)? as usize;
        let residual = read_f32_slice(bytes, &mut pos, hidden)?;
        let mut expert_ids = Vec::with_capacity(ne);
        for _ in 0..ne {
            expert_ids.push(read_u32(bytes, &mut pos)?);
        }
        let mut weights = Vec::with_capacity(ne);
        for _ in 0..ne {
            weights.push(read_f32(bytes, &mut pos)?);
        }
        tasks.push(MultiLayerTask {
            layer,
            residual,
            expert_ids,
            weights,
        });
    }
    Some(tasks)
}

pub fn encode_multi_layer_response(results: &[MultiLayerResult]) -> Vec<u8> {
    let cap = 4 + results.iter().map(|r| 8 + r.h2.len() * 4).sum::<usize>();
    let mut buf = Vec::with_capacity(cap);
    push_u32(&mut buf, results.len() as u32);
    for r in results {
        push_u32(&mut buf, r.layer as u32);
        push_u32(&mut buf, r.h2.len() as u32);
        for &v in &r.h2 {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    buf
}

pub fn decode_multi_layer_response(bytes: &[u8]) -> Option<Vec<MultiLayerResult>> {
    let mut pos = 0;
    let n = read_u32(bytes, &mut pos)? as usize;
    let mut results = Vec::with_capacity(n);
    for _ in 0..n {
        let layer = read_u32(bytes, &mut pos)? as usize;
        let hidden = read_u32(bytes, &mut pos)? as usize;
        let h2 = read_f32_slice(bytes, &mut pos, hidden)?;
        results.push(MultiLayerResult { layer, h2 });
    }
    Some(results)
}

// ── Q8K-prenormed wire ────────────────────────────────────────────────────────

const ELEMS_PER_Q8K_BLOCK: usize = 256;
const SUMS_PER_Q8K_BLOCK: usize = 8;

pub fn encode_multi_layer_request_q8k(tasks: &[MultiLayerTaskQ8K]) -> Vec<u8> {
    let cap = 4 + tasks
        .iter()
        .map(|t| {
            let nb = t.hidden / ELEMS_PER_Q8K_BLOCK;
            12 // layer + hidden + num_experts
            + t.hidden  // qs (i8)
            + nb * 4    // d (f32)
            + nb * SUMS_PER_Q8K_BLOCK * 2  // sums (i16)
            + t.expert_ids.len() * 8 // expert_ids + weights
        })
        .sum::<usize>();
    let mut buf = Vec::with_capacity(cap);
    push_u32(&mut buf, tasks.len() as u32);
    for t in tasks {
        let nb = t.hidden / ELEMS_PER_Q8K_BLOCK;
        push_u32(&mut buf, t.layer as u32);
        push_u32(&mut buf, t.hidden as u32);
        push_u32(&mut buf, t.expert_ids.len() as u32);
        // Q8K activation
        for &q in &t.qs {
            buf.push(q as u8);
        }
        for &v in &t.d {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for &s in &t.sums {
            buf.extend_from_slice(&s.to_le_bytes());
        }
        debug_assert_eq!(t.qs.len(), t.hidden, "qs length mismatch");
        debug_assert_eq!(t.d.len(), nb, "d length mismatch");
        debug_assert_eq!(
            t.sums.len(),
            nb * SUMS_PER_Q8K_BLOCK,
            "sums length mismatch"
        );
        // Expert routing
        for &e in &t.expert_ids {
            push_u32(&mut buf, e);
        }
        for &w in &t.weights {
            buf.extend_from_slice(&w.to_le_bytes());
        }
    }
    buf
}

pub fn decode_multi_layer_request_q8k(bytes: &[u8]) -> Option<Vec<MultiLayerTaskQ8K>> {
    let mut pos = 0;
    let n = read_u32(bytes, &mut pos)? as usize;
    let mut tasks = Vec::with_capacity(n);
    for _ in 0..n {
        let layer = read_u32(bytes, &mut pos)? as usize;
        let hidden = read_u32(bytes, &mut pos)? as usize;
        let ne = read_u32(bytes, &mut pos)? as usize;
        let nb = hidden / ELEMS_PER_Q8K_BLOCK;
        // Q8K activation
        let qs = read_i8_slice(bytes, &mut pos, hidden)?;
        let d = read_f32_slice(bytes, &mut pos, nb)?;
        let sums = read_i16_slice(bytes, &mut pos, nb * SUMS_PER_Q8K_BLOCK)?;
        // Expert routing
        let mut expert_ids = Vec::with_capacity(ne);
        for _ in 0..ne {
            expert_ids.push(read_u32(bytes, &mut pos)?);
        }
        let mut weights = Vec::with_capacity(ne);
        for _ in 0..ne {
            weights.push(read_f32(bytes, &mut pos)?);
        }
        tasks.push(MultiLayerTaskQ8K {
            layer,
            hidden,
            qs,
            d,
            sums,
            expert_ids,
            weights,
        });
    }
    Some(tasks)
}

fn read_i8_slice(bytes: &[u8], pos: &mut usize, n: usize) -> Option<Vec<i8>> {
    let end = pos.checked_add(n)?;
    if end > bytes.len() {
        return None;
    }
    let v: Vec<i8> = bytes[*pos..end].iter().map(|&b| b as i8).collect();
    *pos = end;
    Some(v)
}

fn read_i16_slice(bytes: &[u8], pos: &mut usize, n: usize) -> Option<Vec<i16>> {
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        let end = pos.checked_add(2)?;
        if end > bytes.len() {
            return None;
        }
        let val = i16::from_le_bytes(bytes[*pos..end].try_into().unwrap());
        *pos = end;
        v.push(val);
    }
    Some(v)
}

fn push_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn read_u32(bytes: &[u8], pos: &mut usize) -> Option<u32> {
    let end = pos.checked_add(4)?;
    if end > bytes.len() {
        return None;
    }
    let v = u32::from_le_bytes(bytes[*pos..end].try_into().unwrap());
    *pos = end;
    Some(v)
}

fn read_f32(bytes: &[u8], pos: &mut usize) -> Option<f32> {
    let end = pos.checked_add(4)?;
    if end > bytes.len() {
        return None;
    }
    let v = f32::from_le_bytes(bytes[*pos..end].try_into().unwrap());
    *pos = end;
    Some(v)
}

fn read_f32_slice(bytes: &[u8], pos: &mut usize, n: usize) -> Option<Vec<f32>> {
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(read_f32(bytes, pos)?);
    }
    Some(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_round_trip() {
        let tasks = vec![
            MultiLayerTask {
                layer: 0,
                residual: vec![1.0, 2.0, 3.0],
                expert_ids: vec![5, 17],
                weights: vec![0.6, 0.4],
            },
            MultiLayerTask {
                layer: 7,
                residual: vec![0.5, -1.0, 2.5],
                expert_ids: vec![42],
                weights: vec![1.0],
            },
        ];
        let encoded = encode_multi_layer_request(&tasks);
        let decoded = decode_multi_layer_request(&encoded).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].layer, 0);
        assert_eq!(decoded[0].residual, vec![1.0, 2.0, 3.0]);
        assert_eq!(decoded[0].expert_ids, vec![5, 17]);
        assert_eq!(decoded[0].weights, vec![0.6, 0.4]);
        assert_eq!(decoded[1].layer, 7);
        assert_eq!(decoded[1].expert_ids, vec![42]);
    }

    #[test]
    fn response_round_trip() {
        let results = vec![
            MultiLayerResult {
                layer: 3,
                h2: vec![0.1, 0.2, 0.3],
            },
            MultiLayerResult {
                layer: 15,
                h2: vec![-1.0, 0.0, 1.0],
            },
        ];
        let encoded = encode_multi_layer_response(&results);
        let decoded = decode_multi_layer_response(&encoded).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].layer, 3);
        assert_eq!(decoded[0].h2, vec![0.1, 0.2, 0.3]);
        assert_eq!(decoded[1].layer, 15);
    }

    #[test]
    fn handles_truncation() {
        assert!(decode_multi_layer_request(&[]).is_none());
        assert!(decode_multi_layer_request(&[0, 0, 0, 1]).is_none()); // claims 1 task but no body
        assert!(decode_multi_layer_response(&[]).is_none());
    }
}
