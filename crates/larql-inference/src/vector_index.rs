//! Local vector index — the full model as an in-memory KNN engine.
//!
//! Loads gate vectors and down-projection token metadata from extracted NDJSON
//! files (produced by `vector-extract`). Provides:
//!
//! 1. Gate KNN via BLAS matmul: residual × gate_vectors^T → top-K features
//! 2. Down token lookup: instant array access to precomputed output tokens
//!
//! This is the local equivalent of the SurrealDB walk — same vectors, same KNN,
//! same answer. No HTTP, no JSON serialisation, no round-trip. Array access.
//!
//! Memory: 34 layers × 10240 features × 2560 dim × 4 bytes = ~3.4GB for gate vectors.
//! Down metadata is lightweight (top_k token strings per feature).

use std::collections::HashMap;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use crate::error::InferenceError;
use crate::ffn::{sigmoid, FfnBackend};
use crate::model::ModelWeights;

use larql_models::TopKEntry;

/// Metadata for a single FFN feature (from extraction).
#[derive(Clone)]
pub struct FeatureMeta {
    pub top_token: String,
    pub top_token_id: u32,
    pub c_score: f32,
    pub top_k: Vec<TopKEntry>,
}

/// A single step in the walk trace — one feature that fired at one layer.
pub struct WalkHit {
    pub layer: usize,
    pub feature: usize,
    pub gate_score: f32,
    pub meta: FeatureMeta,
}

/// Result of a walk — per-layer feature activations with full metadata.
pub struct WalkTrace {
    /// Per-layer hits, sorted by gate score descending.
    pub layers: Vec<(usize, Vec<WalkHit>)>,
}

/// Progress callbacks for index loading.
pub trait IndexLoadCallbacks {
    fn on_file_start(&mut self, _component: &str, _path: &str) {}
    fn on_progress(&mut self, _records: usize) {}
    fn on_file_done(&mut self, _component: &str, _records: usize, _elapsed_ms: f64) {}
}

pub struct SilentLoadCallbacks;
impl IndexLoadCallbacks for SilentLoadCallbacks {}

/// The full model as a local vector index.
///
/// Gate vectors for KNN matching + down token metadata for output lookup.
/// Loaded from the NDJSON files produced by `vector-extract`.
pub struct VectorIndex {
    /// Per-layer gate vectors: gate_vectors[layer] is (num_features, hidden_size).
    /// Used for KNN via BLAS matmul.
    gate_vectors: Vec<Option<Array2<f32>>>,

    /// Per-layer, per-feature output token metadata from down projections.
    /// down_meta[layer][feature] = FeatureMeta with top tokens.
    down_meta: Vec<Option<Vec<Option<FeatureMeta>>>>,

    /// Number of layers in the model.
    pub num_layers: usize,

    /// Hidden dimension.
    pub hidden_size: usize,
}

impl VectorIndex {
    /// Load gate vectors from an NDJSON file (ffn_gate.vectors.jsonl).
    ///
    /// Each line is a VectorRecord with layer, feature, vector, top_token, etc.
    /// Vectors are packed into per-layer Array2 matrices for BLAS matmul.
    pub fn load_gates(
        path: &Path,
        callbacks: &mut dyn IndexLoadCallbacks,
    ) -> Result<Self, InferenceError> {
        callbacks.on_file_start("ffn_gate", &path.display().to_string());
        let start = std::time::Instant::now();

        let file = std::fs::File::open(path)?;
        let reader = BufReader::with_capacity(1 << 20, file);

        // First pass: collect all records to determine dimensions
        let mut records: Vec<(usize, usize, Vec<f32>, FeatureMeta)> = Vec::new();
        let mut hidden_size = 0;
        let mut max_layer = 0;
        let mut count = 0;

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let obj: serde_json::Value =
                serde_json::from_str(line).map_err(|e| InferenceError::Parse(e.to_string()))?;

            if obj.get("_header").is_some() {
                if let Some(dim) = obj.get("dimension").and_then(|v| v.as_u64()) {
                    hidden_size = dim as usize;
                }
                continue;
            }

            let layer = obj["layer"].as_u64().unwrap() as usize;
            let feature = obj["feature"].as_u64().unwrap() as usize;

            let vector: Vec<f32> = obj["vector"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();

            if hidden_size == 0 {
                hidden_size = vector.len();
            }

            let top_token = obj["top_token"].as_str().unwrap_or("").to_string();
            let top_token_id = obj["top_token_id"].as_u64().unwrap_or(0) as u32;
            let c_score = obj["c_score"].as_f64().unwrap_or(0.0) as f32;

            let top_k: Vec<TopKEntry> = match obj.get("top_k").and_then(|v| v.as_array()) {
                Some(arr) => arr
                    .iter()
                    .filter_map(|entry| {
                        Some(TopKEntry {
                            token: entry.get("token")?.as_str()?.to_string(),
                            token_id: entry.get("token_id")?.as_u64()? as u32,
                            logit: entry.get("logit")?.as_f64()? as f32,
                        })
                    })
                    .collect(),
                None => vec![],
            };

            let meta = FeatureMeta {
                top_token,
                top_token_id,
                c_score,
                top_k,
            };

            if layer > max_layer {
                max_layer = layer;
            }

            records.push((layer, feature, vector, meta));

            count += 1;
            if count % 10000 == 0 {
                callbacks.on_progress(count);
            }
        }

        let num_layers = max_layer + 1;

        // Group by layer, find max feature per layer
        let mut layer_sizes: HashMap<usize, usize> = HashMap::new();
        for &(layer, feature, _, _) in &records {
            let entry = layer_sizes.entry(layer).or_insert(0);
            if feature + 1 > *entry {
                *entry = feature + 1;
            }
        }

        // Build per-layer matrices
        let mut gate_vectors: Vec<Option<Array2<f32>>> = vec![None; num_layers];
        let mut gate_meta: Vec<Option<Vec<Option<FeatureMeta>>>> = vec![None; num_layers];

        // Pre-allocate
        for (&layer, &num_features) in &layer_sizes {
            gate_vectors[layer] = Some(Array2::zeros((num_features, hidden_size)));
            gate_meta[layer] = Some(vec![None; num_features]);
        }

        // Fill
        for (layer, feature, vector, meta) in records {
            if let Some(ref mut matrix) = gate_vectors[layer] {
                for (j, &val) in vector.iter().enumerate() {
                    matrix[[feature, j]] = val;
                }
            }
            if let Some(ref mut metas) = gate_meta[layer] {
                metas[feature] = Some(meta);
            }
        }

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        callbacks.on_file_done("ffn_gate", count, elapsed_ms);

        Ok(VectorIndex {
            gate_vectors,
            down_meta: gate_meta, // gate meta initially; down loaded separately
            num_layers,
            hidden_size,
        })
    }

    /// Load down-projection token metadata from an NDJSON file (ffn_down.vectors.jsonl).
    ///
    /// Only loads the metadata (top_token, top_k, c_score), NOT the full vectors.
    /// This replaces any gate-file metadata with the down-projection metadata,
    /// which tells you what each feature *outputs* rather than what it *responds to*.
    pub fn load_down_meta(
        &mut self,
        path: &Path,
        callbacks: &mut dyn IndexLoadCallbacks,
    ) -> Result<usize, InferenceError> {
        callbacks.on_file_start("ffn_down", &path.display().to_string());
        let start = std::time::Instant::now();

        let file = std::fs::File::open(path)?;
        let reader = BufReader::with_capacity(1 << 20, file);
        let mut count = 0;

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let obj: serde_json::Value =
                serde_json::from_str(line).map_err(|e| InferenceError::Parse(e.to_string()))?;

            if obj.get("_header").is_some() {
                continue;
            }

            let layer = obj["layer"].as_u64().unwrap() as usize;
            let feature = obj["feature"].as_u64().unwrap() as usize;

            let top_token = obj["top_token"].as_str().unwrap_or("").to_string();
            let top_token_id = obj["top_token_id"].as_u64().unwrap_or(0) as u32;
            let c_score = obj["c_score"].as_f64().unwrap_or(0.0) as f32;

            let top_k: Vec<TopKEntry> = match obj.get("top_k").and_then(|v| v.as_array()) {
                Some(arr) => arr
                    .iter()
                    .filter_map(|entry| {
                        Some(TopKEntry {
                            token: entry.get("token")?.as_str()?.to_string(),
                            token_id: entry.get("token_id")?.as_u64()? as u32,
                            logit: entry.get("logit")?.as_f64()? as f32,
                        })
                    })
                    .collect(),
                None => vec![],
            };

            let meta = FeatureMeta {
                top_token,
                top_token_id,
                c_score,
                top_k,
            };

            if layer < self.num_layers {
                // Ensure layer slot exists
                while self.down_meta.len() <= layer {
                    self.down_meta.push(None);
                }
                if self.down_meta[layer].is_none() {
                    self.down_meta[layer] = Some(Vec::new());
                }
                if let Some(ref mut metas) = self.down_meta[layer] {
                    while metas.len() <= feature {
                        metas.push(None);
                    }
                    metas[feature] = Some(meta);
                }
            }

            count += 1;
            if count % 10000 == 0 {
                callbacks.on_progress(count);
            }
        }

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        callbacks.on_file_done("ffn_down", count, elapsed_ms);

        Ok(count)
    }

    /// Gate KNN: find the top-K features at a layer whose gate vectors have
    /// the highest dot product with the input residual. Uses BLAS matmul.
    ///
    /// Returns (feature_index, dot_product) sorted by absolute magnitude descending.
    pub fn gate_knn(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        let gate_matrix = match self.gate_vectors.get(layer).and_then(|v| v.as_ref()) {
            Some(m) => m,
            None => return vec![],
        };

        // gate_proj = gate_matrix @ residual → (num_features,)
        let scores = gate_matrix.dot(residual);

        // Top-K by dot product magnitude
        let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        let k = top_k.min(indexed.len());
        if k > 0 && k < indexed.len() {
            indexed.select_nth_unstable_by(k, |a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
            indexed.truncate(k);
        }
        indexed.sort_unstable_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        indexed
    }

    /// Full walk: gate KNN at each layer, annotated with down token metadata.
    pub fn walk(
        &self,
        residual: &Array1<f32>,
        layers: &[usize],
        top_k: usize,
    ) -> WalkTrace {
        let mut trace_layers = Vec::with_capacity(layers.len());

        for &layer in layers {
            let hits = self.gate_knn(layer, residual, top_k);
            let walk_hits: Vec<WalkHit> = hits
                .into_iter()
                .filter_map(|(feature, gate_score)| {
                    let meta = self
                        .down_meta
                        .get(layer)
                        .and_then(|v| v.as_ref())
                        .and_then(|metas| metas.get(feature))
                        .and_then(|m| m.as_ref())
                        .cloned()?;
                    Some(WalkHit {
                        layer,
                        feature,
                        gate_score,
                        meta,
                    })
                })
                .collect();
            trace_layers.push((layer, walk_hits));
        }

        WalkTrace {
            layers: trace_layers,
        }
    }

    /// Look up metadata for a specific feature.
    pub fn feature_meta(&self, layer: usize, feature: usize) -> Option<&FeatureMeta> {
        self.down_meta
            .get(layer)
            .and_then(|v| v.as_ref())
            .and_then(|metas| metas.get(feature))
            .and_then(|m| m.as_ref())
    }

    /// Number of features indexed at a layer.
    pub fn num_features(&self, layer: usize) -> usize {
        self.gate_vectors
            .get(layer)
            .and_then(|v| v.as_ref())
            .map(|m| m.shape()[0])
            .unwrap_or(0)
    }

    /// Total gate vectors loaded across all layers.
    pub fn total_gate_vectors(&self) -> usize {
        self.gate_vectors
            .iter()
            .filter_map(|v| v.as_ref())
            .map(|m| m.shape()[0])
            .sum()
    }

    /// Total down metadata entries loaded across all layers.
    pub fn total_down_meta(&self) -> usize {
        self.down_meta
            .iter()
            .filter_map(|v| v.as_ref())
            .map(|metas| metas.iter().filter(|m| m.is_some()).count())
            .sum()
    }

    /// Layers that have gate vectors loaded.
    pub fn loaded_layers(&self) -> Vec<usize> {
        self.gate_vectors
            .iter()
            .enumerate()
            .filter_map(|(i, v)| v.as_ref().map(|_| i))
            .collect()
    }
}

// ── .vindex format ──
//
// A .vindex directory is a self-contained model index:
//
//   model.vindex/
//     index.json          — model config, dimensions, layer map
//     gate_vectors.bin    — raw f32: all gate vectors, layer by layer
//     embeddings.bin      — raw f32: embedding matrix (vocab × hidden)
//     down_meta.jsonl     — per-feature: layer, feature, top_token, top_k
//     tokenizer.json      — HuggingFace tokenizer (copied from model)
//
// No safetensors. No framework. One directory, one binary.

/// Metadata stored in index.json inside a .vindex directory.
#[derive(Serialize, Deserialize)]
pub struct VindexConfig {
    /// Format version.
    pub version: u32,
    /// Original model name (e.g., "google/gemma-3-4b-it").
    pub model: String,
    /// Model family (e.g., "gemma3", "llama").
    pub family: String,
    /// Number of layers.
    pub num_layers: usize,
    /// Hidden dimension.
    pub hidden_size: usize,
    /// Intermediate (FFN) size.
    pub intermediate_size: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Embedding scale factor.
    pub embed_scale: f32,
    /// Per-layer info for gate_vectors.bin layout.
    pub layers: Vec<VindexLayerInfo>,
    /// Top-K tokens stored per feature in down metadata.
    pub down_top_k: usize,
    /// Whether model_weights.bin is present (full inference capable).
    #[serde(default)]
    pub has_model_weights: bool,
    /// Model config for architecture reconstruction.
    #[serde(default)]
    pub model_config: Option<VindexModelConfig>,
}

/// Model configuration stored in the vindex for architecture reconstruction.
#[derive(Serialize, Deserialize, Clone)]
pub struct VindexModelConfig {
    pub model_type: String,
    pub head_dim: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub rope_base: f64,
    #[serde(default)]
    pub sliding_window: Option<usize>,
}

#[derive(Serialize, Deserialize)]
pub struct VindexLayerInfo {
    pub layer: usize,
    pub num_features: usize,
    /// Byte offset into gate_vectors.bin.
    pub offset: u64,
    /// Byte length of this layer's gate data.
    pub length: u64,
}

/// Down metadata entry in the NDJSON file (compact, no vectors).
#[derive(Serialize, Deserialize)]
struct DownMetaRecord {
    #[serde(rename = "l")]
    layer: usize,
    #[serde(rename = "f")]
    feature: usize,
    #[serde(rename = "t")]
    top_token: String,
    #[serde(rename = "i")]
    top_token_id: u32,
    #[serde(rename = "c")]
    c_score: f32,
    #[serde(rename = "k")]
    top_k: Vec<DownMetaTopK>,
}

#[derive(Serialize, Deserialize)]
struct DownMetaTopK {
    #[serde(rename = "t")]
    token: String,
    #[serde(rename = "i")]
    token_id: u32,
    #[serde(rename = "s")]
    logit: f32,
}

/// Callbacks for index build progress.
pub trait IndexBuildCallbacks {
    fn on_stage(&mut self, _stage: &str) {}
    fn on_layer_start(&mut self, _component: &str, _layer: usize, _total: usize) {}
    fn on_feature_progress(&mut self, _component: &str, _layer: usize, _done: usize, _total: usize) {}
    fn on_layer_done(&mut self, _component: &str, _layer: usize, _elapsed_ms: f64) {}
    fn on_stage_done(&mut self, _stage: &str, _elapsed_ms: f64) {}
}

pub struct SilentBuildCallbacks;
impl IndexBuildCallbacks for SilentBuildCallbacks {}

impl VectorIndex {
    /// Build a .vindex from model weights and write it to disk.
    ///
    /// Reads gate vectors and down projections directly from safetensors,
    /// projects down vectors to vocabulary for top-k token metadata,
    /// writes everything to a self-contained directory.
    pub fn build_vindex(
        weights: &ModelWeights,
        tokenizer: &tokenizers::Tokenizer,
        model_name: &str,
        output_dir: &Path,
        down_top_k: usize,
        callbacks: &mut dyn IndexBuildCallbacks,
    ) -> Result<(), InferenceError> {
        std::fs::create_dir_all(output_dir)?;

        let num_layers = weights.num_layers;
        let hidden_size = weights.hidden_size;
        let intermediate_size = weights.intermediate_size;
        let vocab_size = weights.vocab_size;
        let embed_scale = weights.arch.embed_scale();

        // ── 1. Write gate vectors (binary f32) ──
        callbacks.on_stage("gate_vectors");
        let gate_path = output_dir.join("gate_vectors.bin");
        let mut gate_file = BufWriter::new(std::fs::File::create(&gate_path)?);
        let mut layer_infos: Vec<VindexLayerInfo> = Vec::new();
        let mut offset: u64 = 0;

        for layer in 0..num_layers {
            callbacks.on_layer_start("gate", layer, num_layers);
            let start = std::time::Instant::now();

            let gate_key = weights.arch.ffn_gate_key(layer);
            let w_gate = match weights.tensors.get(&gate_key) {
                Some(w) => w,
                None => continue,
            };

            let num_features = w_gate.shape()[0];
            let data = w_gate.as_slice().unwrap();
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
            };
            gate_file.write_all(bytes)?;

            let length = bytes.len() as u64;
            layer_infos.push(VindexLayerInfo {
                layer,
                num_features,
                offset,
                length,
            });
            offset += length;

            callbacks.on_layer_done("gate", layer, start.elapsed().as_secs_f64() * 1000.0);
        }
        gate_file.flush()?;
        callbacks.on_stage_done("gate_vectors", 0.0);

        // ── 2. Write embeddings (binary f32) ──
        callbacks.on_stage("embeddings");
        let embed_path = output_dir.join("embeddings.bin");
        let embed_data = weights.embed.as_slice().unwrap();
        let embed_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(embed_data.as_ptr() as *const u8, embed_data.len() * 4)
        };
        std::fs::write(&embed_path, embed_bytes)?;
        callbacks.on_stage_done("embeddings", 0.0);

        // ── 3. Write down metadata (NDJSON, no vectors) ──
        callbacks.on_stage("down_meta");
        let down_path = output_dir.join("down_meta.jsonl");
        let mut down_file = BufWriter::new(std::fs::File::create(&down_path)?);

        for layer in 0..num_layers {
            callbacks.on_layer_start("down", layer, num_layers);
            let start = std::time::Instant::now();

            let down_key = weights.arch.ffn_down_key(layer);
            let w_down = match weights.tensors.get(&down_key) {
                Some(w) => w,
                None => continue,
            };

            // w_down is (hidden_size, intermediate_size) — columns are features
            let num_features = w_down.shape()[1];

            for feat in 0..num_features {
                if feat % 500 == 0 {
                    callbacks.on_feature_progress("down", layer, feat, num_features);
                }

                // Extract down column for this feature
                let down_col: Vec<f32> = (0..hidden_size)
                    .map(|h| w_down[[h, feat]])
                    .collect();

                // Project to vocabulary: logits = embed @ down_col
                let top_k_entries = project_to_top_k(
                    &weights.embed,
                    &down_col,
                    down_top_k,
                    tokenizer,
                );

                let (top_token, top_token_id, c_score) = if let Some(first) = top_k_entries.first() {
                    (first.token.clone(), first.token_id, first.logit)
                } else {
                    (String::new(), 0, 0.0)
                };

                let record = DownMetaRecord {
                    layer,
                    feature: feat,
                    top_token,
                    top_token_id,
                    c_score,
                    top_k: top_k_entries
                        .iter()
                        .map(|e| DownMetaTopK {
                            token: e.token.clone(),
                            token_id: e.token_id,
                            logit: e.logit,
                        })
                        .collect(),
                };

                serde_json::to_writer(&mut down_file, &record)
                    .map_err(|e| InferenceError::Parse(e.to_string()))?;
                down_file.write_all(b"\n")?;
            }

            callbacks.on_layer_done("down", layer, start.elapsed().as_secs_f64() * 1000.0);
        }
        down_file.flush()?;
        callbacks.on_stage_done("down_meta", 0.0);

        // ── 4. Copy tokenizer ──
        callbacks.on_stage("tokenizer");
        let tokenizer_json = tokenizer
            .to_string(true)
            .map_err(|e| InferenceError::Parse(format!("tokenizer serialize: {e}")))?;
        std::fs::write(output_dir.join("tokenizer.json"), tokenizer_json)?;
        callbacks.on_stage_done("tokenizer", 0.0);

        // ── 5. Write index.json ──
        let config = VindexConfig {
            version: 1,
            model: model_name.to_string(),
            family: weights.arch.family().to_string(),
            num_layers,
            hidden_size,
            intermediate_size,
            vocab_size,
            embed_scale,
            layers: layer_infos,
            down_top_k,
            has_model_weights: false,
            model_config: Some(VindexModelConfig {
                model_type: weights.arch.config().model_type.clone(),
                head_dim: weights.head_dim,
                num_q_heads: weights.num_q_heads,
                num_kv_heads: weights.num_kv_heads,
                rope_base: weights.rope_base,
                sliding_window: weights.arch.config().sliding_window,
            }),
        };

        let config_json = serde_json::to_string_pretty(&config)
            .map_err(|e| InferenceError::Parse(e.to_string()))?;
        std::fs::write(output_dir.join("index.json"), config_json)?;

        Ok(())
    }

    /// Load a VectorIndex from a .vindex directory.
    ///
    /// Reads gate_vectors.bin (mmap'd), down_meta.jsonl, and index.json.
    /// The embeddings and tokenizer are loaded separately via `load_vindex_embeddings`.
    pub fn load_vindex(
        dir: &Path,
        callbacks: &mut dyn IndexLoadCallbacks,
    ) -> Result<Self, InferenceError> {
        // Read config
        let config_path = dir.join("index.json");
        let config_text = std::fs::read_to_string(&config_path)?;
        let config: VindexConfig = serde_json::from_str(&config_text)
            .map_err(|e| InferenceError::Parse(e.to_string()))?;

        let num_layers = config.num_layers;
        let hidden_size = config.hidden_size;

        // Load gate vectors from binary
        callbacks.on_file_start("gate_vectors", &dir.join("gate_vectors.bin").display().to_string());
        let start = std::time::Instant::now();

        let gate_path = dir.join("gate_vectors.bin");
        let gate_bytes = std::fs::read(&gate_path)?;
        let gate_floats: &[f32] = unsafe {
            std::slice::from_raw_parts(
                gate_bytes.as_ptr() as *const f32,
                gate_bytes.len() / 4,
            )
        };

        let mut gate_vectors: Vec<Option<Array2<f32>>> = vec![None; num_layers];
        let mut total_gate = 0;

        for info in &config.layers {
            let float_offset = info.offset as usize / 4;
            let float_count = info.num_features * hidden_size;
            let layer_data = &gate_floats[float_offset..float_offset + float_count];
            let matrix = Array2::from_shape_vec(
                (info.num_features, hidden_size),
                layer_data.to_vec(),
            )
            .map_err(|e| InferenceError::Parse(e.to_string()))?;
            gate_vectors[info.layer] = Some(matrix);
            total_gate += info.num_features;
        }

        callbacks.on_file_done(
            "gate_vectors",
            total_gate,
            start.elapsed().as_secs_f64() * 1000.0,
        );

        // Load down metadata
        callbacks.on_file_start("down_meta", &dir.join("down_meta.jsonl").display().to_string());
        let start = std::time::Instant::now();

        let down_path = dir.join("down_meta.jsonl");
        let down_file = std::fs::File::open(&down_path)?;
        let reader = BufReader::with_capacity(1 << 20, down_file);

        let mut down_meta: Vec<Option<Vec<Option<FeatureMeta>>>> = vec![None; num_layers];
        let mut down_count = 0;

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let record: DownMetaRecord = serde_json::from_str(line)
                .map_err(|e| InferenceError::Parse(e.to_string()))?;

            let layer = record.layer;
            let feature = record.feature;

            let meta = FeatureMeta {
                top_token: record.top_token,
                top_token_id: record.top_token_id,
                c_score: record.c_score,
                top_k: record
                    .top_k
                    .into_iter()
                    .map(|e| TopKEntry {
                        token: e.token,
                        token_id: e.token_id,
                        logit: e.logit,
                    })
                    .collect(),
            };

            if layer < num_layers {
                if down_meta[layer].is_none() {
                    down_meta[layer] = Some(Vec::new());
                }
                if let Some(ref mut metas) = down_meta[layer] {
                    while metas.len() <= feature {
                        metas.push(None);
                    }
                    metas[feature] = Some(meta);
                }
            }

            down_count += 1;
            if down_count % 10000 == 0 {
                callbacks.on_progress(down_count);
            }
        }

        callbacks.on_file_done(
            "down_meta",
            down_count,
            start.elapsed().as_secs_f64() * 1000.0,
        );

        Ok(VectorIndex {
            gate_vectors,
            down_meta,
            num_layers,
            hidden_size,
        })
    }
}

impl VectorIndex {
    /// Build a .vindex from already-extracted NDJSON vector files.
    ///
    /// Reads ffn_gate.vectors.jsonl, ffn_down.vectors.jsonl, and
    /// embeddings.vectors.jsonl, packs them into the binary .vindex format.
    /// Much faster than build_vindex since no vocab projection needed.
    pub fn build_vindex_from_vectors(
        vectors_dir: &Path,
        output_dir: &Path,
        callbacks: &mut dyn IndexBuildCallbacks,
    ) -> Result<(), InferenceError> {
        std::fs::create_dir_all(output_dir)?;

        let gate_path = vectors_dir.join("ffn_gate.vectors.jsonl");
        let down_path = vectors_dir.join("ffn_down.vectors.jsonl");
        let embed_path = vectors_dir.join("embeddings.vectors.jsonl");

        if !gate_path.exists() {
            return Err(InferenceError::Parse(
                format!("ffn_gate.vectors.jsonl not found in {}", vectors_dir.display()),
            ));
        }

        // ── 1. Read gate header for config ──
        let gate_file = std::fs::File::open(&gate_path)?;
        let reader = BufReader::with_capacity(1 << 20, gate_file);
        let first_line = reader.lines().next()
            .ok_or_else(|| InferenceError::Parse("empty gate file".into()))??;
        let header: serde_json::Value = serde_json::from_str(&first_line)
            .map_err(|e| InferenceError::Parse(e.to_string()))?;

        let model_name = header.get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let hidden_size = header.get("dimension")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        // ── 2. Stream gate vectors → binary + collect layer info ──
        callbacks.on_stage("gate_vectors");
        let start = std::time::Instant::now();

        let gate_file = std::fs::File::open(&gate_path)?;
        let reader = BufReader::with_capacity(1 << 20, gate_file);

        // First pass: collect all records to determine layout
        let mut gate_records: Vec<(usize, usize, Vec<f32>)> = Vec::new();
        let mut max_layer: usize = 0;
        let mut count: usize = 0;

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() { continue; }

            let obj: serde_json::Value = serde_json::from_str(line)
                .map_err(|e| InferenceError::Parse(e.to_string()))?;
            if obj.get("_header").is_some() { continue; }

            let layer = obj["layer"].as_u64().unwrap() as usize;
            let feature = obj["feature"].as_u64().unwrap() as usize;
            let vector: Vec<f32> = obj["vector"].as_array().unwrap()
                .iter().map(|v| v.as_f64().unwrap() as f32).collect();

            if layer > max_layer { max_layer = layer; }
            gate_records.push((layer, feature, vector));

            count += 1;
            if count.is_multiple_of(10000) {
                callbacks.on_feature_progress("gate", 0, count, 0);
            }
        }

        let num_layers = max_layer + 1;

        // Find features per layer
        let mut layer_feature_counts: HashMap<usize, usize> = HashMap::new();
        for &(layer, feature, _) in &gate_records {
            let e = layer_feature_counts.entry(layer).or_insert(0);
            if feature + 1 > *e { *e = feature + 1; }
        }

        // Sort records by (layer, feature) for contiguous binary write
        gate_records.sort_unstable_by_key(|r| (r.0, r.1));

        // Write binary
        let bin_path = output_dir.join("gate_vectors.bin");
        let mut bin_file = BufWriter::new(std::fs::File::create(&bin_path)?);
        let mut layer_infos: Vec<VindexLayerInfo> = Vec::new();
        let mut offset: u64 = 0;

        let mut sorted_layers: Vec<usize> = layer_feature_counts.keys().copied().collect();
        sorted_layers.sort();

        for &layer in &sorted_layers {
            let num_features = layer_feature_counts[&layer];
            // Write zeros for all features, then overwrite with actual data
            let mut layer_data = vec![0.0f32; num_features * hidden_size];

            for &(l, feat, ref vec) in &gate_records {
                if l == layer {
                    let dst = feat * hidden_size;
                    layer_data[dst..dst + hidden_size].copy_from_slice(vec);
                }
            }

            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    layer_data.as_ptr() as *const u8,
                    layer_data.len() * 4,
                )
            };
            bin_file.write_all(bytes)?;

            let length = bytes.len() as u64;
            layer_infos.push(VindexLayerInfo { layer, num_features, offset, length });
            offset += length;
        }
        bin_file.flush()?;

        callbacks.on_stage_done("gate_vectors", start.elapsed().as_secs_f64() * 1000.0);

        // ── 3. Stream embeddings → binary ──
        callbacks.on_stage("embeddings");
        let start = std::time::Instant::now();

        let embed_bin_path = output_dir.join("embeddings.bin");
        let mut embed_out = BufWriter::new(std::fs::File::create(&embed_bin_path)?);

        let embed_file = std::fs::File::open(&embed_path)?;
        let reader = BufReader::with_capacity(1 << 20, embed_file);

        let mut vocab_size: usize = 0;
        let mut embed_count: usize = 0;

        // Collect all embeddings (they may not be in order)
        let mut embed_records: Vec<(usize, Vec<f32>)> = Vec::new();

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() { continue; }

            let obj: serde_json::Value = serde_json::from_str(line)
                .map_err(|e| InferenceError::Parse(e.to_string()))?;
            if obj.get("_header").is_some() { continue; }

            let feature = obj["feature"].as_u64().unwrap() as usize;
            let vector: Vec<f32> = obj["vector"].as_array().unwrap()
                .iter().map(|v| v.as_f64().unwrap() as f32).collect();

            if feature + 1 > vocab_size { vocab_size = feature + 1; }
            embed_records.push((feature, vector));

            embed_count += 1;
            if embed_count.is_multiple_of(10000) {
                callbacks.on_feature_progress("embeddings", 0, embed_count, 0);
            }
        }

        // Sort by feature ID and write contiguously
        embed_records.sort_unstable_by_key(|r| r.0);
        let mut embed_data = vec![0.0f32; vocab_size * hidden_size];
        for (feat, vec) in &embed_records {
            let dst = feat * hidden_size;
            embed_data[dst..dst + hidden_size].copy_from_slice(vec);
        }

        let embed_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                embed_data.as_ptr() as *const u8,
                embed_data.len() * 4,
            )
        };
        embed_out.write_all(embed_bytes)?;
        embed_out.flush()?;

        callbacks.on_stage_done("embeddings", start.elapsed().as_secs_f64() * 1000.0);

        // ── 4. Stream down metadata (copy top_k, skip vectors) ──
        callbacks.on_stage("down_meta");
        let start = std::time::Instant::now();

        let down_meta_path = output_dir.join("down_meta.jsonl");
        let mut down_out = BufWriter::new(std::fs::File::create(&down_meta_path)?);

        let down_file = std::fs::File::open(&down_path)?;
        let reader = BufReader::with_capacity(1 << 20, down_file);
        let mut down_count: usize = 0;
        let mut down_top_k_size: usize = 0;

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() { continue; }

            let obj: serde_json::Value = serde_json::from_str(line)
                .map_err(|e| InferenceError::Parse(e.to_string()))?;
            if obj.get("_header").is_some() { continue; }

            let layer = obj["layer"].as_u64().unwrap() as usize;
            let feature = obj["feature"].as_u64().unwrap() as usize;
            let top_token = obj["top_token"].as_str().unwrap_or("").to_string();
            let top_token_id = obj["top_token_id"].as_u64().unwrap_or(0) as u32;
            let c_score = obj["c_score"].as_f64().unwrap_or(0.0) as f32;

            let top_k: Vec<DownMetaTopK> = match obj.get("top_k").and_then(|v| v.as_array()) {
                Some(arr) => {
                    if down_top_k_size == 0 { down_top_k_size = arr.len(); }
                    arr.iter().filter_map(|entry| {
                        Some(DownMetaTopK {
                            token: entry.get("token")?.as_str()?.to_string(),
                            token_id: entry.get("token_id")?.as_u64()? as u32,
                            logit: entry.get("logit")?.as_f64()? as f32,
                        })
                    }).collect()
                }
                None => vec![],
            };

            let record = DownMetaRecord {
                layer, feature, top_token, top_token_id, c_score, top_k,
            };

            serde_json::to_writer(&mut down_out, &record)
                .map_err(|e| InferenceError::Parse(e.to_string()))?;
            down_out.write_all(b"\n")?;

            down_count += 1;
            if down_count.is_multiple_of(10000) {
                callbacks.on_feature_progress("down", 0, down_count, 0);
            }
        }
        down_out.flush()?;

        callbacks.on_stage_done("down_meta", start.elapsed().as_secs_f64() * 1000.0);

        // ── 5. Copy tokenizer if available ──
        // Look for tokenizer.json near the vectors dir or in common locations
        let tokenizer_src = find_tokenizer(vectors_dir);
        if let Some(ref src) = tokenizer_src {
            callbacks.on_stage("tokenizer");
            std::fs::copy(src, output_dir.join("tokenizer.json"))?;
            callbacks.on_stage_done("tokenizer", 0.0);
        }

        // ── 6. Determine embed_scale from model family ──
        // Gemma models use sqrt(hidden_size), others use 1.0
        let intermediate_size = layer_feature_counts.values().max().copied().unwrap_or(0);
        let embed_scale = if model_name.contains("gemma") {
            (hidden_size as f32).sqrt()
        } else {
            1.0
        };
        let family = if model_name.contains("gemma") {
            "gemma3"
        } else if model_name.contains("llama") || model_name.contains("Llama") {
            "llama"
        } else {
            "unknown"
        };

        // ── 7. Write index.json ──
        let config = VindexConfig {
            version: 1,
            model: model_name,
            family: family.to_string(),
            num_layers,
            hidden_size,
            intermediate_size,
            vocab_size,
            embed_scale,
            layers: layer_infos,
            down_top_k: down_top_k_size,
            has_model_weights: false,
            model_config: None,
        };

        let config_json = serde_json::to_string_pretty(&config)
            .map_err(|e| InferenceError::Parse(e.to_string()))?;
        std::fs::write(output_dir.join("index.json"), config_json)?;

        Ok(())
    }
}

/// Try to find tokenizer.json near the vectors directory.
fn find_tokenizer(vectors_dir: &Path) -> Option<std::path::PathBuf> {
    // Check parent directory
    if let Some(parent) = vectors_dir.parent() {
        let p = parent.join("tokenizer.json");
        if p.exists() { return Some(p); }
    }
    // Check vectors dir itself
    let p = vectors_dir.join("tokenizer.json");
    if p.exists() { return Some(p); }
    // Check sibling
    if let Some(parent) = vectors_dir.parent() {
        let p = parent.join("vectors").join("tokenizer.json");
        if p.exists() { return Some(p); }
    }
    None
}

/// Load embeddings from a .vindex directory.
pub fn load_vindex_embeddings(dir: &Path) -> Result<(Array2<f32>, f32), InferenceError> {
    let config_text = std::fs::read_to_string(dir.join("index.json"))?;
    let config: VindexConfig = serde_json::from_str(&config_text)
        .map_err(|e| InferenceError::Parse(e.to_string()))?;

    let embed_bytes = std::fs::read(dir.join("embeddings.bin"))?;
    let embed_floats: Vec<f32> = unsafe {
        std::slice::from_raw_parts(
            embed_bytes.as_ptr() as *const f32,
            embed_bytes.len() / 4,
        )
    }
    .to_vec();

    let embed = Array2::from_shape_vec((config.vocab_size, config.hidden_size), embed_floats)
        .map_err(|e| InferenceError::Parse(e.to_string()))?;

    Ok((embed, config.embed_scale))
}

/// Load tokenizer from a .vindex directory.
pub fn load_vindex_tokenizer(dir: &Path) -> Result<tokenizers::Tokenizer, InferenceError> {
    let path = dir.join("tokenizer.json");
    tokenizers::Tokenizer::from_file(&path).map_err(|e| InferenceError::Parse(e.to_string()))
}

/// Load the vindex config.
pub fn load_vindex_config(dir: &Path) -> Result<VindexConfig, InferenceError> {
    let text = std::fs::read_to_string(dir.join("index.json"))?;
    serde_json::from_str(&text).map_err(|e| InferenceError::Parse(e.to_string()))
}

/// Load feature labels from down_meta.jsonl — fast hash lookup, no vocab projection.
///
/// Returns a map: (layer, feature) → top_token string.
/// Also works with the gate vectors NDJSON from vector-extract (has same fields).
pub fn load_feature_labels(path: &Path) -> Result<HashMap<(usize, usize), String>, InferenceError> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::with_capacity(1 << 20, file);
    let mut labels: HashMap<(usize, usize), String> = HashMap::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let obj: serde_json::Value =
            serde_json::from_str(line).map_err(|e| InferenceError::Parse(e.to_string()))?;

        if obj.get("_header").is_some() {
            continue;
        }

        // Support both compact (l/f/t) and full (layer/feature/top_token) formats
        let layer = obj
            .get("l")
            .or_else(|| obj.get("layer"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let feature = obj
            .get("f")
            .or_else(|| obj.get("feature"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let token = obj
            .get("t")
            .or_else(|| obj.get("top_token"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        labels.insert((layer, feature), token);
    }

    Ok(labels)
}

/// Write all model weights (attention + FFN + norms) to a vindex directory.
///
/// Creates `model_weights.bin` containing all 2D tensors and 1D vectors
/// serialized contiguously, with a `weight_manifest.json` mapping keys to offsets.
/// Updates `index.json` to mark the vindex as inference-capable.
pub fn write_model_weights(
    weights: &ModelWeights,
    dir: &Path,
    callbacks: &mut dyn IndexBuildCallbacks,
) -> Result<(), InferenceError> {
    callbacks.on_stage("model_weights");
    let start = std::time::Instant::now();

    let bin_path = dir.join("model_weights.bin");
    let mut bin_file = BufWriter::new(std::fs::File::create(&bin_path)?);

    #[derive(Serialize)]
    struct WeightEntry {
        key: String,
        kind: String, // "tensor" (2D) or "vector" (1D)
        shape: Vec<usize>,
        offset: u64,
        length: u64,
    }

    let mut entries: Vec<WeightEntry> = Vec::new();
    let mut offset: u64 = 0;

    // Write 2D tensors (attention Q/K/V/O, FFN up/down — gate already in gate_vectors.bin)
    let arch = &*weights.arch;
    let num_layers = weights.num_layers;

    for layer in 0..num_layers {
        callbacks.on_layer_start("weights", layer, num_layers);

        // Attention weights
        for (suffix, key_fn) in &[
            ("q", arch.attn_q_key(layer)),
            ("k", arch.attn_k_key(layer)),
            ("v", arch.attn_v_key(layer)),
            ("o", arch.attn_o_key(layer)),
        ] {
            if let Some(tensor) = weights.tensors.get(key_fn) {
                let data = tensor.as_slice().unwrap();
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
                };
                bin_file.write_all(bytes)?;
                entries.push(WeightEntry {
                    key: key_fn.clone(),
                    kind: "tensor".into(),
                    shape: vec![tensor.shape()[0], tensor.shape()[1]],
                    offset,
                    length: bytes.len() as u64,
                });
                offset += bytes.len() as u64;
            }
            let _ = suffix;
        }

        // FFN up and down (gate is in gate_vectors.bin, but we need it here too for WalkFfn)
        for key in &[
            arch.ffn_gate_key(layer),
            arch.ffn_up_key(layer),
            arch.ffn_down_key(layer),
        ] {
            if let Some(tensor) = weights.tensors.get(key) {
                let data = tensor.as_slice().unwrap();
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
                };
                bin_file.write_all(bytes)?;
                entries.push(WeightEntry {
                    key: key.clone(),
                    kind: "tensor".into(),
                    shape: vec![tensor.shape()[0], tensor.shape()[1]],
                    offset,
                    length: bytes.len() as u64,
                });
                offset += bytes.len() as u64;
            }
        }

        callbacks.on_layer_done("weights", layer, 0.0);
    }

    // Write 1D vectors (all norms)
    for (key, vec) in &weights.vectors {
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(vec.as_ptr() as *const u8, vec.len() * 4)
        };
        bin_file.write_all(bytes)?;
        entries.push(WeightEntry {
            key: key.clone(),
            kind: "vector".into(),
            shape: vec![vec.len()],
            offset,
            length: bytes.len() as u64,
        });
        offset += bytes.len() as u64;
    }

    bin_file.flush()?;

    // Write manifest
    let manifest_json = serde_json::to_string_pretty(&entries)
        .map_err(|e| InferenceError::Parse(e.to_string()))?;
    std::fs::write(dir.join("weight_manifest.json"), manifest_json)?;

    // Update index.json
    let config_path = dir.join("index.json");
    let config_text = std::fs::read_to_string(&config_path)?;
    let mut config: VindexConfig = serde_json::from_str(&config_text)
        .map_err(|e| InferenceError::Parse(e.to_string()))?;

    config.has_model_weights = true;
    config.model_config = Some(VindexModelConfig {
        model_type: weights.arch.config().model_type.clone(),
        head_dim: weights.head_dim,
        num_q_heads: weights.num_q_heads,
        num_kv_heads: weights.num_kv_heads,
        rope_base: weights.rope_base,
        sliding_window: weights.arch.config().sliding_window,
    });

    let config_json = serde_json::to_string_pretty(&config)
        .map_err(|e| InferenceError::Parse(e.to_string()))?;
    std::fs::write(&config_path, config_json)?;

    callbacks.on_stage_done("model_weights", start.elapsed().as_secs_f64() * 1000.0);
    Ok(())
}

/// Load a full ModelWeights from a vindex directory.
///
/// Reads model_weights.bin + embeddings.bin + weight_manifest.json,
/// reconstructs the architecture from index.json config.
/// Returns a ModelWeights that can be used with the existing forward pass.
pub fn load_model_weights_from_vindex(
    dir: &Path,
    callbacks: &mut dyn IndexLoadCallbacks,
) -> Result<ModelWeights, InferenceError> {
    let config = load_vindex_config(dir)?;

    if !config.has_model_weights {
        return Err(InferenceError::Parse(
            "vindex does not contain model weights. Rebuild with: larql extract-index <model> -o <vindex> --include-weights".into(),
        ));
    }

    let model_cfg = config.model_config.as_ref().ok_or_else(|| {
        InferenceError::Parse("vindex missing model_config in index.json".into())
    })?;

    // Reconstruct architecture
    let _arch_config = larql_models::ModelConfig {
        model_type: model_cfg.model_type.clone(),
        num_layers: config.num_layers,
        hidden_size: config.hidden_size,
        intermediate_size: config.intermediate_size,
        head_dim: model_cfg.head_dim,
        num_q_heads: model_cfg.num_q_heads,
        num_kv_heads: model_cfg.num_kv_heads,
        vocab_size: Some(config.vocab_size),
        rope_base: model_cfg.rope_base,
        sliding_window: model_cfg.sliding_window,
    };

    let arch_json = serde_json::json!({
        "model_type": model_cfg.model_type,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_layers,
        "intermediate_size": config.intermediate_size,
        "head_dim": model_cfg.head_dim,
        "num_attention_heads": model_cfg.num_q_heads,
        "num_key_value_heads": model_cfg.num_kv_heads,
        "rope_theta": model_cfg.rope_base,
        "sliding_window": model_cfg.sliding_window,
        "vocab_size": config.vocab_size,
    });
    let arch = larql_models::detect_from_json(&arch_json);

    // Load embeddings
    callbacks.on_file_start("embeddings", &dir.join("embeddings.bin").display().to_string());
    let embed_bytes = std::fs::read(dir.join("embeddings.bin"))?;
    let embed_floats: Vec<f32> = unsafe {
        std::slice::from_raw_parts(
            embed_bytes.as_ptr() as *const f32,
            embed_bytes.len() / 4,
        )
    }
    .to_vec();
    let embed = Array2::from_shape_vec((config.vocab_size, config.hidden_size), embed_floats)
        .map_err(|e| InferenceError::Parse(e.to_string()))?;
    callbacks.on_file_done("embeddings", config.vocab_size, 0.0);

    // Load weight manifest
    callbacks.on_file_start("model_weights", &dir.join("model_weights.bin").display().to_string());
    let manifest_text = std::fs::read_to_string(dir.join("weight_manifest.json"))?;

    #[derive(Deserialize)]
    struct WeightEntry {
        key: String,
        kind: String,
        shape: Vec<usize>,
        offset: u64,
        length: u64,
    }

    let entries: Vec<WeightEntry> = serde_json::from_str(&manifest_text)
        .map_err(|e| InferenceError::Parse(e.to_string()))?;

    // Read binary weight data
    let bin_data = std::fs::read(dir.join("model_weights.bin"))?;
    let all_floats: &[f32] = unsafe {
        std::slice::from_raw_parts(
            bin_data.as_ptr() as *const f32,
            bin_data.len() / 4,
        )
    };

    let mut tensors: HashMap<String, Array2<f32>> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();

    for entry in &entries {
        let float_offset = entry.offset as usize / 4;
        let float_count = entry.length as usize / 4;
        let data = &all_floats[float_offset..float_offset + float_count];

        match entry.kind.as_str() {
            "tensor" => {
                let arr = Array2::from_shape_vec(
                    (entry.shape[0], entry.shape[1]),
                    data.to_vec(),
                )
                .map_err(|e| InferenceError::Parse(e.to_string()))?;
                tensors.insert(entry.key.clone(), arr);
            }
            "vector" => {
                vectors.insert(entry.key.clone(), data.to_vec());
            }
            _ => {}
        }
    }

    callbacks.on_file_done("model_weights", entries.len(), 0.0);

    let cfg = arch.config();
    Ok(ModelWeights {
        tensors,
        vectors,
        embed,
        num_layers: cfg.num_layers,
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        vocab_size: config.vocab_size,
        head_dim: cfg.head_dim,
        num_q_heads: cfg.num_q_heads,
        num_kv_heads: cfg.num_kv_heads,
        rope_base: cfg.rope_base,
        arch,
    })
}

/// Project a vector onto the embedding matrix and return top-k tokens.
fn project_to_top_k(
    embed: &Array2<f32>,
    vector: &[f32],
    k: usize,
    tokenizer: &tokenizers::Tokenizer,
) -> Vec<TopKEntry> {
    let vocab_size = embed.shape()[0];
    let mut scores: Vec<(usize, f32)> = Vec::with_capacity(vocab_size);

    for i in 0..vocab_size {
        let row = embed.row(i);
        let dot: f32 = row.iter().zip(vector.iter()).map(|(a, b)| a * b).sum();
        scores.push((i, dot));
    }

    let k = k.min(scores.len());
    if k > 0 && k < scores.len() {
        scores.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
    }
    scores.truncate(k);
    scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    scores
        .into_iter()
        .filter_map(|(idx, logit)| {
            tokenizer
                .decode(&[idx as u32], true)
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .map(|token| TopKEntry {
                    token,
                    token_id: idx as u32,
                    logit,
                })
        })
        .collect()
}

// ── FfnBackend: WalkFfn ──

/// FFN backend that uses the VectorIndex for gate selection.
///
/// Gate KNN finds which features fire. Then uses the model's actual up/down
/// weights for the sparse computation — same as SparseFfn but with KNN-based
/// feature selection instead of full gate matmul.
///
/// The gate matmul IS the KNN. residual × gate_vectors^T is both the gate
/// computation and the similarity search. Same operation, different framing.
pub struct WalkFfn<'a> {
    pub weights: &'a ModelWeights,
    pub index: &'a VectorIndex,
    pub top_k: usize,
    /// If set, captures walk traces per layer during forward pass.
    trace: std::cell::RefCell<Vec<(usize, Vec<WalkHit>)>>,
}

impl<'a> WalkFfn<'a> {
    pub fn new(weights: &'a ModelWeights, index: &'a VectorIndex, top_k: usize) -> Self {
        Self {
            weights,
            index,
            top_k,
            trace: std::cell::RefCell::new(Vec::new()),
        }
    }

    /// Take the accumulated walk trace (clears internal state).
    pub fn take_trace(&self) -> WalkTrace {
        let layers = self.trace.borrow_mut().drain(..).collect();
        WalkTrace { layers }
    }
}

impl<'a> FfnBackend for WalkFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        let (out, _) = self.forward_with_activation(layer, x);
        out
    }

    fn forward_with_activation(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>) {
        let arch = &*self.weights.arch;
        let w_gate = self.weights.tensors.get(&arch.ffn_gate_key(layer)).unwrap();
        let w_up = self.weights.tensors.get(&arch.ffn_up_key(layer)).unwrap();
        let w_down = self.weights.tensors.get(&arch.ffn_down_key(layer)).unwrap();
        let hidden = x.shape()[1];
        let intermediate = w_gate.shape()[0];
        let seq_len = x.shape()[0];

        let mut full_activation = Array2::<f32>::zeros((seq_len, intermediate));
        let mut out = Array2::<f32>::zeros((seq_len, hidden));

        // For the last sequence position, capture the walk trace
        let last_pos = seq_len - 1;

        let has_index = self.index.num_features(layer) > 0;

        for s in 0..seq_len {
            let x_row = x.row(s);

            // Feature selection: use index if available, fall back to SparseFfn
            let hits: Vec<(usize, f32)> = if has_index {
                let x_vec = x_row.to_owned();
                self.index.gate_knn(layer, &x_vec, self.top_k)
            } else {
                // No index for this layer — compute gate matmul directly (SparseFfn path)
                let gate_proj = w_gate.dot(&x_row);
                let mut indexed: Vec<(usize, f32)> = gate_proj
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(i, v)| (i, v * sigmoid(v)))
                    .collect();
                let k = self.top_k.min(indexed.len());
                if k > 0 && k < indexed.len() {
                    indexed.select_nth_unstable_by(k, |a, b| {
                        b.1.abs().partial_cmp(&a.1.abs()).unwrap()
                    });
                    indexed.truncate(k);
                }
                indexed.sort_unstable_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
                indexed
            };

            let k = hits.len();
            if k == 0 {
                continue;
            }

            // Capture walk trace for last position (only for indexed layers)
            if s == last_pos && has_index {
                let walk_hits: Vec<WalkHit> = hits
                    .iter()
                    .filter_map(|&(feature, gate_score)| {
                        let meta = self.index.feature_meta(layer, feature)?.clone();
                        Some(WalkHit {
                            layer,
                            feature,
                            gate_score,
                            meta,
                        })
                    })
                    .collect();
                self.trace.borrow_mut().push((layer, walk_hits));
            }

            // Compute actual gate activations for selected features
            let up_raw = w_up.as_slice().unwrap();
            let mut up_buf = vec![0.0f32; k * hidden];
            for (i, &(feat, _)) in hits.iter().enumerate() {
                let src = feat * hidden;
                up_buf[i * hidden..(i + 1) * hidden]
                    .copy_from_slice(&up_raw[src..src + hidden]);
            }

            let up_sub =
                ndarray::ArrayView2::from_shape((k, hidden), &up_buf[..k * hidden]).unwrap();
            let up_proj = up_sub.dot(&x_row);

            // Compute actual gate values for the selected features
            for (i, &(feat, _)) in hits.iter().enumerate() {
                let gate_row = w_gate.row(feat);
                let gate_val: f32 = gate_row.iter().zip(x_row.iter()).map(|(a, b)| a * b).sum();
                let silu_gate = gate_val * sigmoid(gate_val);
                let act_val = silu_gate * up_proj[i];
                full_activation[[s, feat]] = act_val;
            }

            // Down projection via dense BLAS gemv on sparse activation
            let act_row = full_activation.row(s);
            let out_vec = w_down.dot(&act_row);
            let mut out_row = out.row_mut(s);
            ndarray::Zip::from(&mut out_row)
                .and(&out_vec)
                .for_each(|o, &v| *o = v);
        }

        (out, full_activation)
    }

    fn name(&self) -> &str {
        "walk"
    }
}
