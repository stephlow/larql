//! Walk attention OV circuits — extract routing edges from attention heads.
//!
//! Each attention head is a routing rule. The OV circuit tells you:
//! when this head fires on input X, it copies output Y.
//! That's an edge: X → Y via this head.
//!
//! Confidence scoring:
//! - c_in: input amplification norm (how much this head amplifies the input)
//! - c_out: output logit magnitude (how strongly it projects to the output token)
//! - c: normalized (c_in × c_out) / max per layer
//!
//! Zero forward passes. Pure matrix multiplication.

use crate::core::edge::Edge;
use crate::core::enums::SourceType;
use crate::core::graph::Graph;

use super::safetensors_loader::{ModelWeights, WalkerError};
use super::weight_walker::{resolve_model_path, LayerResult, LayerStats, WalkCallbacks, WalkConfig};

/// Result of walking attention heads at a single layer.
#[derive(Debug, Clone)]
pub struct AttentionLayerResult {
    pub layer: usize,
    pub heads_walked: usize,
    pub edges_found: usize,
    pub elapsed_ms: f64,
    pub stats: LayerStats,
}

/// A raw attention edge before per-layer normalization.
struct RawEdge {
    subject: String,
    relation: String,
    object: String,
    c_in: f32,
    c_out: f32,
    layer: usize,
    head: usize,
}

/// A loaded model ready for attention walking.
pub struct AttentionWalker {
    weights: ModelWeights,
    tokenizer: tokenizers::Tokenizer,
    head_dim: usize,
}

impl AttentionWalker {
    pub fn load(model: &str) -> Result<Self, WalkerError> {
        let model_path = resolve_model_path(model)?;
        let weights = super::safetensors_loader::load_model_dir(&model_path)?;

        let tokenizer_path = model_path.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(WalkerError::MissingTensor(
                "tokenizer.json not found".into(),
            ));
        }
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| WalkerError::Parse(e.to_string()))?;

        let v_key = "layers.0.self_attn.v_proj.weight";
        let head_dim = if let Some(v) = weights.tensors.get(v_key) {
            let v_rows = v.shape()[0];
            if v_rows % 256 == 0 {
                256
            } else if v_rows % 128 == 0 {
                128
            } else if v_rows % 64 == 0 {
                64
            } else {
                128
            }
        } else {
            256
        };

        Ok(Self {
            weights,
            tokenizer,
            head_dim,
        })
    }

    pub fn num_layers(&self) -> usize {
        self.weights.num_layers
    }

    /// Walk all attention heads at a single layer.
    pub fn walk_layer(
        &self,
        layer: usize,
        config: &WalkConfig,
        graph: &mut Graph,
        callbacks: &mut dyn WalkCallbacks,
    ) -> Result<AttentionLayerResult, WalkerError> {
        let start = std::time::Instant::now();

        let prefix = format!("layers.{layer}.self_attn.");
        let w_v = self
            .weights
            .tensors
            .get(&format!("{prefix}v_proj.weight"))
            .ok_or_else(|| WalkerError::MissingTensor(format!("{prefix}v_proj.weight")))?;
        let w_o = self
            .weights
            .tensors
            .get(&format!("{prefix}o_proj.weight"))
            .ok_or_else(|| WalkerError::MissingTensor(format!("{prefix}o_proj.weight")))?;

        let num_kv_heads = w_v.shape()[0] / self.head_dim;
        callbacks.on_layer_start(layer, num_kv_heads);

        let k_inputs = (config.top_k * 10).min(self.weights.vocab_size);

        // Phase 1: collect raw edges with c_in / c_out
        let mut raw_edges: Vec<RawEdge> = Vec::new();

        for h in 0..num_kv_heads {
            callbacks.on_progress(layer, h, num_kv_heads);

            let hd = self.head_dim;
            let v_h = w_v.slice(ndarray::s![h * hd..(h + 1) * hd, ..]);
            let o_h = w_o.slice(ndarray::s![.., h * hd..(h + 1) * hd]);

            // OV circuit: O_h @ V_h -> (hidden, hidden)
            let ov = o_h.dot(&v_h);

            // transformed = embed @ OV.T -> (vocab, hidden)
            let transformed = self.weights.embed.dot(&ov.t());

            // Norms: how much this head amplifies each input (c_in candidate)
            let norms: Vec<f32> = (0..self.weights.vocab_size)
                .map(|i| {
                    let row = transformed.row(i);
                    row.iter().map(|x| x * x).sum::<f32>().sqrt()
                })
                .collect();

            // Top-k inputs by amplification
            let top_inputs = partial_top_k(&norms, k_inputs);

            let relation = format!("L{layer}-H{h}");
            for &(inp_idx, c_in) in &top_inputs {
                let inp_vec = transformed.row(inp_idx);

                // out_logits = embed @ inp_vec -> (vocab,)
                let out_logits: Vec<f32> = (0..self.weights.vocab_size)
                    .map(|j| {
                        self.weights
                            .embed
                            .row(j)
                            .iter()
                            .zip(inp_vec.iter())
                            .map(|(a, b)| a * b)
                            .sum()
                    })
                    .collect();

                let top_out = partial_top_k(&out_logits, config.top_k);

                let inp_token = match decode_token(&self.tokenizer, inp_idx as u32) {
                    Some(t) if !t.is_empty() => t,
                    _ => continue,
                };

                for &(out_idx, c_out) in &top_out {
                    if c_out < config.min_score {
                        continue;
                    }
                    let out_token = match decode_token(&self.tokenizer, out_idx as u32) {
                        Some(t) if !t.is_empty() => t,
                        _ => continue,
                    };

                    raw_edges.push(RawEdge {
                        subject: inp_token.clone(),
                        relation: relation.clone(),
                        object: out_token,
                        c_in,
                        c_out,
                        layer,
                        head: h,
                    });
                }
            }
        }

        // Phase 2: per-layer normalization
        let max_product = raw_edges
            .iter()
            .map(|e| e.c_in * e.c_out)
            .fold(f32::MIN, f32::max)
            .max(f32::EPSILON);

        let mut sum_conf = 0.0f64;
        let mut sum_cin = 0.0f64;
        let mut sum_cout = 0.0f64;
        let mut max_conf = 0.0f64;
        let mut min_conf = 1.0f64;

        // Phase 3: add normalized edges to graph
        for raw in &raw_edges {
            let product = raw.c_in * raw.c_out;
            let confidence = (product / max_product) as f64;

            sum_conf += confidence;
            sum_cin += raw.c_in as f64;
            sum_cout += raw.c_out as f64;
            if confidence > max_conf {
                max_conf = confidence;
            }
            if confidence < min_conf {
                min_conf = confidence;
            }

            let edge = Edge::new(&raw.subject, &raw.relation, &raw.object)
                .with_confidence(confidence)
                .with_source(SourceType::Parametric)
                .with_metadata("layer", serde_json::Value::from(raw.layer as u64))
                .with_metadata("head", serde_json::Value::from(raw.head as u64))
                .with_metadata("circuit", serde_json::Value::String("OV".to_string()))
                .with_metadata("c_in", serde_json::Value::from(round4(raw.c_in as f64)))
                .with_metadata("c_out", serde_json::Value::from(round4(raw.c_out as f64)));
            graph.add_edge(edge);
        }

        let n = raw_edges.len();
        let stats = if n > 0 {
            LayerStats {
                mean_confidence: sum_conf / n as f64,
                max_confidence: max_conf,
                min_confidence: min_conf,
                mean_c_in: sum_cin / n as f64,
                mean_c_out: sum_cout / n as f64,
            }
        } else {
            LayerStats::default()
        };

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        callbacks.on_layer_done(&LayerResult {
            layer,
            features_scanned: num_kv_heads,
            edges_found: n,
            elapsed_ms: elapsed,
            stats: stats.clone(),
        });
        callbacks.on_checkpoint(graph);

        Ok(AttentionLayerResult {
            layer,
            heads_walked: num_kv_heads,
            edges_found: n,
            elapsed_ms: elapsed,
            stats,
        })
    }
}

fn round4(v: f64) -> f64 {
    (v * 10000.0).round() / 10000.0
}

fn partial_top_k(data: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = data.iter().copied().enumerate().collect();
    let k = k.min(indexed.len());
    if k == 0 {
        return vec![];
    }
    indexed.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.truncate(k);
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed
}

fn decode_token(tokenizer: &tokenizers::Tokenizer, id: u32) -> Option<String> {
    tokenizer
        .decode(&[id], true)
        .ok()
        .map(|s| s.trim().to_string())
}
