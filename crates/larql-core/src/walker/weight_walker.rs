//! Walk FFN weight matrices — extract edges directly from model parameters.
//!
//! Every FFN feature is a potential edge. For each feature in a layer:
//! - What tokens activate it (gate projection) → c_in
//! - What tokens it produces (down projection) → c_out
//! - Confidence = c_in × c_out, normalized per-layer to [0, 1]
//!
//! Zero forward passes. Pure matrix multiplication.

use std::path::{Path, PathBuf};

use crate::core::edge::Edge;
use crate::core::enums::SourceType;
use crate::core::graph::Graph;

use super::safetensors_loader::{load_model_dir, ModelWeights, WalkerError};

/// Result of walking a single layer.
#[derive(Debug, Clone)]
pub struct LayerResult {
    pub layer: usize,
    pub features_scanned: usize,
    pub edges_found: usize,
    pub elapsed_ms: f64,
    pub stats: LayerStats,
}

/// Per-layer statistics for validation.
#[derive(Debug, Clone, Default)]
pub struct LayerStats {
    pub mean_confidence: f64,
    pub max_confidence: f64,
    pub min_confidence: f64,
    pub mean_c_in: f64,
    pub mean_c_out: f64,
}

/// Configuration for the weight walker.
pub struct WalkConfig {
    pub top_k: usize,
    pub min_score: f32,
}

impl Default for WalkConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            min_score: 0.02,
        }
    }
}

/// Callbacks for walk progress.
pub trait WalkCallbacks {
    fn on_layer_start(&mut self, _layer: usize, _num_features: usize) {}
    fn on_layer_done(&mut self, _result: &LayerResult) {}
    fn on_checkpoint(&mut self, _graph: &Graph) {}
    fn on_progress(&mut self, _layer: usize, _features_done: usize, _total: usize) {}
}

pub struct SilentWalkCallbacks;
impl WalkCallbacks for SilentWalkCallbacks {}

/// Resolve a model string to a local directory path.
pub fn resolve_model_path(model: &str) -> Result<PathBuf, WalkerError> {
    let as_path = Path::new(model);
    if as_path.is_dir() {
        return Ok(as_path.to_path_buf());
    }

    if model.contains('/') {
        let cache_name = format!("models--{}", model.replace('/', "--"));
        let hf_hub = hf_cache_dir().join(&cache_name).join("snapshots");
        if hf_hub.is_dir() {
            if let Some(snap) = std::fs::read_dir(&hf_hub)
                .ok()
                .and_then(|rd| {
                    rd.filter_map(|e| e.ok())
                        .map(|e| e.path())
                        .find(|p| p.is_dir())
                })
            {
                return Ok(snap);
            }
        }
    }

    Err(WalkerError::NotADirectory(PathBuf::from(model)))
}

fn hf_cache_dir() -> PathBuf {
    if let Ok(hf_home) = std::env::var("HF_HOME") {
        return PathBuf::from(hf_home).join("hub");
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".cache/huggingface/hub")
}

/// A loaded model ready for weight walking.
pub struct WeightWalker {
    weights: ModelWeights,
    tokenizer: tokenizers::Tokenizer,
}

/// A raw edge before per-layer normalization.
struct RawEdge {
    subject: String,
    relation: String,
    object: String,
    c_in: f32,
    c_out: f32,
    layer: usize,
    feature: usize,
}

impl WeightWalker {
    pub fn load(model: &str) -> Result<Self, WalkerError> {
        let model_path = resolve_model_path(model)?;
        let weights = load_model_dir(&model_path)?;

        let tokenizer_path = model_path.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(WalkerError::MissingTensor(
                "tokenizer.json not found".into(),
            ));
        }
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| WalkerError::Parse(e.to_string()))?;

        Ok(Self { weights, tokenizer })
    }

    pub fn num_layers(&self) -> usize {
        self.weights.num_layers
    }

    /// Walk a single layer's FFN weights into the graph.
    ///
    /// Confidence scoring:
    /// - `c_in`: raw gate projection score (input selectivity)
    /// - `c_out`: raw down projection score (output strength)
    /// - `c`: normalized `c_in × c_out`, scaled to [0,1] per layer
    pub fn walk_layer(
        &self,
        layer: usize,
        config: &WalkConfig,
        graph: &mut Graph,
        callbacks: &mut dyn WalkCallbacks,
    ) -> Result<LayerResult, WalkerError> {
        let start = std::time::Instant::now();

        let prefix = format!("layers.{layer}.mlp.");
        let w_gate = self
            .weights
            .tensors
            .get(&format!("{prefix}gate_proj.weight"))
            .ok_or_else(|| WalkerError::MissingTensor(format!("{prefix}gate_proj.weight")))?;
        let w_down = self
            .weights
            .tensors
            .get(&format!("{prefix}down_proj.weight"))
            .ok_or_else(|| WalkerError::MissingTensor(format!("{prefix}down_proj.weight")))?;

        let n_features = w_down.shape()[1];
        callbacks.on_layer_start(layer, n_features);

        // BLAS-accelerated matmuls
        let all_output = self.weights.embed.dot(w_down);
        let all_input = self.weights.embed.dot(&w_gate.t());

        let k = config.top_k.min(self.weights.vocab_size);
        let progress_interval = (n_features / 20).max(1);

        // Phase 1: collect raw edges with c_in / c_out
        let mut raw_edges: Vec<RawEdge> = Vec::new();

        for feat_idx in 0..n_features {
            if feat_idx % progress_interval == 0 {
                callbacks.on_progress(layer, feat_idx, n_features);
            }

            let top_in = partial_top_k_column(&all_input, feat_idx, k);
            let top_out = partial_top_k_column(&all_output, feat_idx, k);

            let mut subjects: Vec<(String, f32)> = Vec::new();
            for (idx, score) in &top_in {
                if *score >= config.min_score {
                    if let Some(tok) = decode_token(&self.tokenizer, *idx as u32) {
                        if !tok.is_empty() {
                            subjects.push((tok, *score));
                        }
                    }
                }
            }

            let mut objects: Vec<(String, f32)> = Vec::new();
            for (idx, score) in &top_out {
                if *score >= config.min_score {
                    if let Some(tok) = decode_token(&self.tokenizer, *idx as u32) {
                        if !tok.is_empty() {
                            objects.push((tok, *score));
                        }
                    }
                }
            }

            if subjects.is_empty() || objects.is_empty() {
                continue;
            }

            let relation = format!("L{layer}-F{feat_idx}");
            for (subj, c_in) in &subjects {
                for (obj, c_out) in &objects {
                    raw_edges.push(RawEdge {
                        subject: subj.clone(),
                        relation: relation.clone(),
                        object: obj.clone(),
                        c_in: *c_in,
                        c_out: *c_out,
                        layer,
                        feature: feat_idx,
                    });
                }
            }
        }

        // Phase 2: per-layer normalization
        // confidence = (c_in × c_out) / max(c_in × c_out) across this layer
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
                .with_metadata("feature", serde_json::Value::from(raw.feature as u64))
                .with_metadata(
                    "c_in",
                    serde_json::Value::from(round4(raw.c_in as f64)),
                )
                .with_metadata(
                    "c_out",
                    serde_json::Value::from(round4(raw.c_out as f64)),
                );
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
        let result = LayerResult {
            layer,
            features_scanned: n_features,
            edges_found: n,
            elapsed_ms: elapsed,
            stats,
        };
        callbacks.on_layer_done(&result);
        callbacks.on_checkpoint(graph);
        Ok(result)
    }
}

fn round4(v: f64) -> f64 {
    (v * 10000.0).round() / 10000.0
}

/// Extract top-k (index, value) pairs from a column using partial sort.
fn partial_top_k_column(
    matrix: &ndarray::Array2<f32>,
    col: usize,
    k: usize,
) -> Vec<(usize, f32)> {
    let nrows = matrix.shape()[0];
    let mut indexed: Vec<(usize, f32)> = Vec::with_capacity(nrows);
    for i in 0..nrows {
        indexed.push((i, matrix[[i, col]]));
    }

    if k >= indexed.len() {
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        return indexed;
    }

    indexed.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.truncate(k);
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed
}

/// Convenience: load model and walk all (or selected) layers.
pub fn walk_model(
    model: &str,
    layers: Option<&[usize]>,
    config: &WalkConfig,
    graph: &mut Graph,
    callbacks: &mut dyn WalkCallbacks,
) -> Result<Vec<LayerResult>, WalkerError> {
    let walker = WeightWalker::load(model)?;

    let layer_indices: Vec<usize> = match layers {
        Some(ls) => ls.to_vec(),
        None => (0..walker.num_layers()).collect(),
    };

    let mut results = Vec::new();
    for &layer in &layer_indices {
        let result = walker.walk_layer(layer, config, graph, callbacks)?;
        results.push(result);
    }

    Ok(results)
}

fn decode_token(tokenizer: &tokenizers::Tokenizer, id: u32) -> Option<String> {
    tokenizer
        .decode(&[id], true)
        .ok()
        .map(|s| s.trim().to_string())
}
