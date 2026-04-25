//! TurboQuantEngine — WHT + Lloyd-Max K/V cache compression.
//!
//! Algorithm (ICLR 2026 style):
//!   1. Normalize vector → unit norm (store scalar)
//!   2. Walsh-Hadamard rotation (spreads coordinates to Beta distribution)
//!   3. Lloyd-Max scalar quantization (3 or 4 bits per coordinate)
//!   4. Bit-pack indices
//!   5. Decode: unpack → centroids → inverse WHT → rescale
//!
//! The `TurboQuantEngine` wraps this codec around the CPU K/V cache:
//! prefill captures K/V per layer and compresses them; each decode step
//! decompresses the full prior K/V for attention, appends the new token's
//! K/V, then re-compresses and stores the updated cache.

pub mod codebooks;
pub mod lloyd_max;
pub mod packing;
pub mod rotation;

use ndarray::{s, Array2};
use larql_compute::{ComputeBackend, cpu_backend};
use larql_vindex::VectorIndex;

use crate::model::ModelWeights;
use crate::attention::{run_attention_with_kv_backend, run_attention_block_decode_step_backend};
use crate::ffn::BackendFfn;
use crate::vindex::{WalkFfn, WalkFfnConfig};
use crate::forward::{embed_tokens_pub, run_ffn};
use crate::attention::SharedKV;
use crate::engines::{EngineInfo, KvEngine};
use crate::engines::markov_residual::ensure_attn_tensors_dequantised;

// ─── TurboQuant codec ────────────────────────────────────────────────────────

/// WHT + Lloyd-Max codec. Stateless — all operations are deterministic
/// functions of the input vector and the pre-computed codebook.
#[derive(Clone)]
pub struct TurboQuant {
    pub bits: u8, // 3 or 4
}

impl TurboQuant {
    pub fn new(bits: u8) -> Self {
        assert!(bits == 3 || bits == 4, "TurboQuant: bits must be 3 or 4");
        Self { bits }
    }

    /// Encode a single vector: normalize → WHT → quantize → pack.
    pub fn encode_vector(&self, x: &[f32]) -> Vec<u8> {
        let d = x.len();
        let norm = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let x_hat: Vec<f32> = if norm > 1e-12 {
            x.iter().map(|v| v / norm).collect()
        } else {
            vec![0.0; d]
        };
        let y = rotation::wht(&x_hat);
        let codebook = codebooks::get_codebook(d, self.bits);
        let indices: Vec<u8> = y.iter()
            .map(|&val| lloyd_max::quantize_scalar(val, codebook))
            .collect();
        let mut buf = Vec::new();
        buf.extend_from_slice(&norm.to_le_bytes());
        packing::pack_indices(&indices, self.bits, &mut buf);
        buf
    }

    /// Decode a single vector: unpack → centroids → inverse WHT → rescale.
    pub fn decode_vector(&self, encoded: &[u8], dim: usize) -> Vec<f32> {
        let norm = f32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]);
        let indices = packing::unpack_indices(&encoded[4..], dim, self.bits);
        let codebook = codebooks::get_codebook(dim, self.bits);
        let y: Vec<f32> = indices.iter().map(|&i| codebook.centroids[i as usize]).collect();
        let x_hat = rotation::wht(&y);
        x_hat.iter().map(|&v| v * norm).collect()
    }

    pub fn bytes_per_vector(&self, dim: usize) -> usize {
        4 + packing::packed_size(dim, self.bits)
    }
}

// ─── Compressed K/V layer ────────────────────────────────────────────────────

struct CompressedLayer {
    compressed_k: Vec<u8>,
    compressed_v: Vec<u8>,
    num_vecs: usize,
    kv_dim: usize,
    /// Largest power-of-two head dimension detected from kv_dim.
    head_dim: usize,
}

impl CompressedLayer {
    fn compress(kv: &SharedKV, tq: &TurboQuant) -> Self {
        let (k, v) = kv;
        let num_vecs = k.shape()[0];
        let kv_dim   = k.shape()[1];
        let head_dim = detect_head_dim(kv_dim);
        Self {
            compressed_k: compress_matrix(k, tq, head_dim),
            compressed_v: compress_matrix(v, tq, head_dim),
            num_vecs,
            kv_dim,
            head_dim,
        }
    }

    fn decompress(&self, tq: &TurboQuant) -> SharedKV {
        let k = decompress_matrix(&self.compressed_k, self.num_vecs, self.kv_dim, self.head_dim, tq);
        let v = decompress_matrix(&self.compressed_v, self.num_vecs, self.kv_dim, self.head_dim, tq);
        (k, v)
    }

    fn memory_bytes(&self) -> usize {
        self.compressed_k.len() + self.compressed_v.len()
    }
}

fn detect_head_dim(kv_dim: usize) -> usize {
    for &hd in &[256usize, 128, 64, 32] {
        if kv_dim % hd == 0 { return hd; }
    }
    kv_dim // fallback: treat whole row as one head
}

fn compress_matrix(m: &Array2<f32>, tq: &TurboQuant, head_dim: usize) -> Vec<u8> {
    let mut buf = Vec::new();
    for row in m.rows() {
        let row_slice = row.as_slice().expect("non-contiguous row");
        for chunk in row_slice.chunks(head_dim) {
            buf.extend_from_slice(&tq.encode_vector(chunk));
        }
    }
    buf
}

fn decompress_matrix(
    bytes: &[u8],
    num_vecs: usize,
    kv_dim: usize,
    head_dim: usize,
    tq: &TurboQuant,
) -> Array2<f32> {
    let heads_per_vec = kv_dim / head_dim;
    let bytes_per_head = tq.bytes_per_vector(head_dim);
    let mut data = Vec::with_capacity(num_vecs * kv_dim);
    for i in 0..num_vecs {
        for h in 0..heads_per_vec {
            let offset = (i * heads_per_vec + h) * bytes_per_head;
            let decoded = tq.decode_vector(&bytes[offset..offset + bytes_per_head], head_dim);
            data.extend_from_slice(&decoded);
        }
    }
    Array2::from_shape_vec((num_vecs, kv_dim), data).expect("shape mismatch")
}

// ─── Engine ──────────────────────────────────────────────────────────────────

pub struct TurboQuantEngine {
    tq: TurboQuant,
    backend: Box<dyn ComputeBackend>,
    layers: Vec<CompressedLayer>,
    abs_position: usize,
}

impl TurboQuantEngine {
    pub fn new(bits: u8) -> Self {
        Self::with_backend(bits, cpu_backend())
    }

    pub fn with_backend(bits: u8, backend: Box<dyn ComputeBackend>) -> Self {
        Self { tq: TurboQuant::new(bits), backend, layers: Vec::new(), abs_position: 0 }
    }
}

impl KvEngine for TurboQuantEngine {
    fn name(&self) -> &str { "turbo-quant" }

    fn info(&self) -> EngineInfo {
        let mem: usize = self.layers.iter().map(|l| l.memory_bytes()).sum();
        EngineInfo {
            name: "turbo-quant".into(),
            description: format!(
                "{}-bit WHT+Lloyd-Max K/V compression (mem={:.1}MB)",
                self.tq.bits,
                mem as f64 / 1_048_576.0,
            ),
            backend: self.backend.name().to_string(),
            config: format!("bits={}", self.tq.bits),
        }
    }

    fn prefill(&mut self, weights: &ModelWeights, token_ids: &[u32]) -> Option<Array2<f32>> {
        let num_layers = weights.num_layers;
        let be = Some(self.backend.as_ref());
        let mut h = embed_tokens_pub(weights, token_ids);
        self.layers.clear();

        for layer in 0..num_layers {
            let (h_post_attn, k, v) =
                run_attention_with_kv_backend(weights, &h, layer, be)?;
            self.layers.push(CompressedLayer::compress(&(k, v), &self.tq));

            let bffn = BackendFfn { weights, backend: self.backend.as_ref() };
            let (h_out, _) = run_ffn(weights, &h_post_attn, layer, &bffn, false);
            h = h_out;
        }

        self.abs_position = token_ids.len();
        Some(last_row(&h))
    }

    fn decode_step(&mut self, weights: &ModelWeights, token_id: u32) -> Option<Array2<f32>> {
        let num_layers = weights.num_layers;
        let abs_position = self.abs_position;
        let mut h = embed_tokens_pub(weights, &[token_id]);

        for layer in 0..num_layers {
            // Decompress full prior K/V for attention.
            let prior_kv = self.layers[layer].decompress(&self.tq);

            // Decode step returns updated K/V (prior + new token).
            let (h_post_attn, updated_kv) = run_attention_block_decode_step_backend(
                weights, &h, layer, Some(&prior_kv), abs_position,
                Some(self.backend.as_ref()),
            )?;

            // Re-compress the updated cache.
            let arch = &*weights.arch;
            let kv_dim = arch.num_kv_heads_for_layer(layer) * arch.head_dim_for_layer(layer);
            self.layers[layer] = CompressedLayer {
                compressed_k: compress_matrix(&updated_kv.0, &self.tq, detect_head_dim(kv_dim)),
                compressed_v: compress_matrix(&updated_kv.1, &self.tq, detect_head_dim(kv_dim)),
                num_vecs: updated_kv.0.shape()[0],
                kv_dim,
                head_dim: detect_head_dim(kv_dim),
            };

            let bffn = BackendFfn { weights, backend: self.backend.as_ref() };
            let (h_out, _) = run_ffn(weights, &h_post_attn, layer, &bffn, false);
            h = h_out;
        }

        self.abs_position += 1;
        Some(last_row(&h))
    }

    fn memory_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.memory_bytes()).sum()
    }

    /// Q4K path: dequantise attention tensors once (idempotent), use WalkFfn
    /// for FFN. Same approach as MarkovRS CPU Q4K — compresses the resulting
    /// K/V rather than storing raw residuals.
    fn prefill_q4k(
        &mut self,
        weights: &mut ModelWeights,
        index: &VectorIndex,
        token_ids: &[u32],
        backend: &dyn ComputeBackend,
    ) -> Option<Array2<f32>> {
        ensure_attn_tensors_dequantised(weights, index);
        let num_layers = weights.num_layers;
        let be = Some(backend);
        let mut h = embed_tokens_pub(weights, token_ids);
        self.layers.clear();

        for layer in 0..num_layers {
            let (h_post_attn, k, v) = run_attention_with_kv_backend(weights, &h, layer, be)?;
            self.layers.push(CompressedLayer::compress(&(k, v), &self.tq));

            let walk_ffn = WalkFfn::from_config(weights, index, WalkFfnConfig::dense(num_layers))
                .with_backend(backend);
            let (h_out, _) = run_ffn(weights, &h_post_attn, layer, &walk_ffn, false);
            h = h_out;
        }

        self.abs_position = token_ids.len();
        Some(last_row(&h))
    }

    fn decode_step_q4k(
        &mut self,
        weights: &mut ModelWeights,
        index: &VectorIndex,
        token_id: u32,
        backend: &dyn ComputeBackend,
    ) -> Option<Array2<f32>> {
        ensure_attn_tensors_dequantised(weights, index);
        let num_layers = weights.num_layers;
        let abs_position = self.abs_position;
        let mut h = embed_tokens_pub(weights, &[token_id]);

        for layer in 0..num_layers {
            let prior_kv = self.layers[layer].decompress(&self.tq);
            let (h_post_attn, updated_kv) = run_attention_block_decode_step_backend(
                weights, &h, layer, Some(&prior_kv), abs_position, Some(backend),
            )?;
            let arch = &*weights.arch;
            let kv_dim = arch.num_kv_heads_for_layer(layer) * arch.head_dim_for_layer(layer);
            self.layers[layer] = CompressedLayer {
                compressed_k: compress_matrix(&updated_kv.0, &self.tq, detect_head_dim(kv_dim)),
                compressed_v: compress_matrix(&updated_kv.1, &self.tq, detect_head_dim(kv_dim)),
                num_vecs: updated_kv.0.shape()[0],
                kv_dim,
                head_dim: detect_head_dim(kv_dim),
            };
            let walk_ffn = WalkFfn::from_config(weights, index, WalkFfnConfig::dense(num_layers))
                .with_backend(backend);
            let (h_out, _) = run_ffn(weights, &h_post_attn, layer, &walk_ffn, false);
            h = h_out;
        }

        self.abs_position += 1;
        Some(last_row(&h))
    }
}

fn last_row(h: &Array2<f32>) -> Array2<f32> {
    let last = h.shape()[0] - 1;
    h.slice(s![last..=last, ..]).to_owned()
}
