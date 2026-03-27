//! Extract full vectors from model weight matrices to intermediate NDJSON files.
//!
//! Same safetensors loading and BLAS matmuls as weight_walker, but captures the
//! full float vector alongside top-k token metadata. Output is one `.vectors.jsonl`
//! file per component type (ffn_down, ffn_gate, etc.).
//!
//! Zero forward passes. Pure matrix multiplication.

use std::collections::HashSet;
use std::io::{BufRead, BufWriter, Write};
use std::path::{Path, PathBuf};

use super::safetensors_loader::{load_model_dir, ModelWeights, WalkerError};
use super::weight_walker::resolve_model_path;

// Component name constants — strings, not enums.
pub const COMPONENT_FFN_DOWN: &str = "ffn_down";
pub const COMPONENT_FFN_GATE: &str = "ffn_gate";
pub const COMPONENT_FFN_UP: &str = "ffn_up";
pub const COMPONENT_ATTN_OV: &str = "attn_ov";
pub const COMPONENT_ATTN_QK: &str = "attn_qk";
pub const COMPONENT_EMBEDDINGS: &str = "embeddings";

pub const ALL_COMPONENTS: &[&str] = &[
    COMPONENT_FFN_DOWN,
    COMPONENT_FFN_GATE,
    COMPONENT_FFN_UP,
    COMPONENT_ATTN_OV,
    COMPONENT_ATTN_QK,
    COMPONENT_EMBEDDINGS,
];

/// A single extracted vector with metadata.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct VectorRecord {
    pub id: String,
    pub layer: usize,
    pub index: usize,
    pub vector: Vec<f32>,
    pub dim: usize,
    pub top_token: String,
    pub top_token_id: u32,
    pub c_score: f32,
    pub top_k: Vec<TopKEntry>,
}

/// A top-k token entry with score.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct TopKEntry {
    pub token: String,
    pub token_id: u32,
    pub score: f32,
}

/// Header line written as first line of each NDJSON file.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct VectorFileHeader {
    pub _header: bool,
    pub component: String,
    pub model: String,
    pub dimension: usize,
    pub extraction_date: String,
}

/// Configuration for vector extraction.
pub struct ExtractConfig {
    pub components: Vec<String>,
    pub layers: Option<Vec<usize>>,
    pub top_k: usize,
}

impl Default for ExtractConfig {
    fn default() -> Self {
        Self {
            components: ALL_COMPONENTS.iter().map(|s| s.to_string()).collect(),
            layers: None,
            top_k: 10,
        }
    }
}

/// Callbacks for extraction progress.
pub trait ExtractCallbacks {
    fn on_component_start(&mut self, _component: &str, _total_layers: usize) {}
    fn on_layer_start(&mut self, _component: &str, _layer: usize, _num_vectors: usize) {}
    fn on_progress(&mut self, _component: &str, _layer: usize, _done: usize, _total: usize) {}
    fn on_layer_done(
        &mut self,
        _component: &str,
        _layer: usize,
        _vectors_written: usize,
        _elapsed_ms: f64,
    ) {
    }
    fn on_component_done(&mut self, _component: &str, _total_written: usize) {}
}

pub struct SilentExtractCallbacks;
impl ExtractCallbacks for SilentExtractCallbacks {}

/// Summary of a full extraction run.
pub struct ExtractSummary {
    pub components: Vec<ComponentSummary>,
    pub total_vectors: usize,
    pub elapsed_secs: f64,
}

/// Summary for a single component.
pub struct ComponentSummary {
    pub component: String,
    pub vectors_written: usize,
    pub output_path: PathBuf,
    pub elapsed_secs: f64,
}

/// Streaming NDJSON writer for vector records.
pub struct VectorWriter {
    writer: BufWriter<std::fs::File>,
    count: usize,
}

impl VectorWriter {
    /// Create a new writer, truncating any existing file.
    pub fn create(path: &Path) -> Result<Self, WalkerError> {
        let file = std::fs::File::create(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
            count: 0,
        })
    }

    /// Open an existing file for appending and count existing records.
    pub fn append(path: &Path) -> Result<(Self, usize), WalkerError> {
        // Count existing lines (excluding header)
        let existing = if path.exists() {
            let file = std::fs::File::open(path)?;
            let reader = std::io::BufReader::new(file);
            let total_lines = reader.lines().count();
            if total_lines > 0 {
                total_lines - 1 // subtract header
            } else {
                0
            }
        } else {
            0
        };

        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok((
            Self {
                writer: BufWriter::new(file),
                count: existing,
            },
            existing,
        ))
    }

    /// Write the metadata header as the first line.
    pub fn write_header(&mut self, header: &VectorFileHeader) -> Result<(), WalkerError> {
        serde_json::to_writer(&mut self.writer, header)
            .map_err(|e| WalkerError::Parse(e.to_string()))?;
        self.writer.write_all(b"\n")?;
        Ok(())
    }

    /// Write a single vector record as one NDJSON line.
    pub fn write_record(&mut self, record: &VectorRecord) -> Result<(), WalkerError> {
        serde_json::to_writer(&mut self.writer, record)
            .map_err(|e| WalkerError::Parse(e.to_string()))?;
        self.writer.write_all(b"\n")?;
        self.count += 1;
        Ok(())
    }

    pub fn flush(&mut self) -> Result<(), WalkerError> {
        self.writer.flush()?;
        Ok(())
    }

    pub fn count(&self) -> usize {
        self.count
    }
}

/// Scan an existing NDJSON file for completed layer numbers.
pub fn scan_completed_layers(path: &Path) -> Result<HashSet<usize>, WalkerError> {
    let mut layers = HashSet::new();
    if !path.exists() {
        return Ok(layers);
    }

    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);

    for line in reader.lines() {
        let line = line?;
        // Quick parse: extract "layer" field without full deserialization
        if let Some(pos) = line.find("\"layer\":") {
            let rest = &line[pos + 8..];
            let num_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(layer) = num_str.parse::<usize>() {
                layers.insert(layer);
            }
        }
    }

    Ok(layers)
}

/// A loaded model ready for vector extraction.
pub struct VectorExtractor {
    weights: ModelWeights,
    tokenizer: tokenizers::Tokenizer,
    model_name: String,
}

impl VectorExtractor {
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

        Ok(Self {
            weights,
            tokenizer,
            model_name: model.to_string(),
        })
    }

    pub fn num_layers(&self) -> usize {
        self.weights.num_layers
    }

    pub fn hidden_size(&self) -> usize {
        self.weights.hidden_size
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Extract FFN down vectors for a single layer.
    ///
    /// W_down columns projected through the embedding give the output direction
    /// of each feature — what it WRITES to the residual stream.
    pub fn extract_ffn_down(
        &self,
        layer: usize,
        config: &ExtractConfig,
        writer: &mut VectorWriter,
        callbacks: &mut dyn ExtractCallbacks,
    ) -> Result<usize, WalkerError> {
        let prefix = format!("layers.{layer}.mlp.");
        let w_down = self
            .weights
            .tensors
            .get(&format!("{prefix}down_proj.weight"))
            .ok_or_else(|| WalkerError::MissingTensor(format!("{prefix}down_proj.weight")))?;

        let n_features = w_down.shape()[1];
        callbacks.on_layer_start(COMPONENT_FFN_DOWN, layer, n_features);

        // BLAS matmul: (vocab, hidden) @ (hidden, intermediate) → (vocab, intermediate)
        let all_output = self.weights.embed.dot(w_down);

        let progress_interval = (n_features / 20).max(1);
        let mut count = 0;

        for feat_idx in 0..n_features {
            if feat_idx % progress_interval == 0 {
                callbacks.on_progress(COMPONENT_FFN_DOWN, layer, feat_idx, n_features);
            }

            // Full vector: the vocab-projected column for this feature
            let vector: Vec<f32> = all_output.column(feat_idx).to_vec();

            // Top-k tokens for metadata
            let top_k_pairs = partial_top_k_column(&all_output, feat_idx, config.top_k);
            let top_k: Vec<TopKEntry> = top_k_pairs
                .iter()
                .filter_map(|&(idx, score)| {
                    decode_token(&self.tokenizer, idx as u32).map(|token| TopKEntry {
                        token,
                        token_id: idx as u32,
                        score,
                    })
                })
                .collect();

            let (top_token, top_token_id, c_score) = if let Some(first) = top_k.first() {
                (first.token.clone(), first.token_id, first.score)
            } else {
                (String::new(), 0, 0.0)
            };

            let record = VectorRecord {
                id: format!("L{layer}_F{feat_idx}"),
                layer,
                index: feat_idx,
                dim: vector.len(),
                vector,
                top_token,
                top_token_id,
                c_score,
                top_k,
            };

            writer.write_record(&record)?;
            count += 1;
        }

        writer.flush()?;
        Ok(count)
    }

    /// Extract FFN gate vectors for a single layer.
    ///
    /// W_gate rows projected through the embedding give the input direction
    /// of each feature — what TRIGGERS it.
    pub fn extract_ffn_gate(
        &self,
        layer: usize,
        config: &ExtractConfig,
        writer: &mut VectorWriter,
        callbacks: &mut dyn ExtractCallbacks,
    ) -> Result<usize, WalkerError> {
        let prefix = format!("layers.{layer}.mlp.");
        let w_gate = self
            .weights
            .tensors
            .get(&format!("{prefix}gate_proj.weight"))
            .ok_or_else(|| WalkerError::MissingTensor(format!("{prefix}gate_proj.weight")))?;

        let n_features = w_gate.shape()[0];
        callbacks.on_layer_start(COMPONENT_FFN_GATE, layer, n_features);

        // BLAS matmul: (vocab, hidden) @ (intermediate, hidden).T → (vocab, intermediate)
        let all_input = self.weights.embed.dot(&w_gate.t());

        let progress_interval = (n_features / 20).max(1);
        let mut count = 0;

        for feat_idx in 0..n_features {
            if feat_idx % progress_interval == 0 {
                callbacks.on_progress(COMPONENT_FFN_GATE, layer, feat_idx, n_features);
            }

            // Full vector: the vocab-projected column for this feature
            let vector: Vec<f32> = all_input.column(feat_idx).to_vec();

            // Top-k tokens for metadata
            let top_k_pairs = partial_top_k_column(&all_input, feat_idx, config.top_k);
            let top_k: Vec<TopKEntry> = top_k_pairs
                .iter()
                .filter_map(|&(idx, score)| {
                    decode_token(&self.tokenizer, idx as u32).map(|token| TopKEntry {
                        token,
                        token_id: idx as u32,
                        score,
                    })
                })
                .collect();

            let (top_token, top_token_id, c_score) = if let Some(first) = top_k.first() {
                (first.token.clone(), first.token_id, first.score)
            } else {
                (String::new(), 0, 0.0)
            };

            let record = VectorRecord {
                id: format!("L{layer}_F{feat_idx}"),
                layer,
                index: feat_idx,
                dim: vector.len(),
                vector,
                top_token,
                top_token_id,
                c_score,
                top_k,
            };

            writer.write_record(&record)?;
            count += 1;
        }

        writer.flush()?;
        Ok(count)
    }

    /// Orchestrate extraction of all requested components across requested layers.
    pub fn extract_all(
        &self,
        config: &ExtractConfig,
        output_dir: &Path,
        resume: bool,
        callbacks: &mut dyn ExtractCallbacks,
    ) -> Result<ExtractSummary, WalkerError> {
        std::fs::create_dir_all(output_dir)?;
        let overall_start = std::time::Instant::now();
        let mut summaries = Vec::new();

        let layers: Vec<usize> = match &config.layers {
            Some(ls) => ls.clone(),
            None => (0..self.weights.num_layers).collect(),
        };

        for component in &config.components {
            let file_path = output_dir.join(format!("{component}.vectors.jsonl"));
            let comp_start = std::time::Instant::now();

            // Determine completed layers for resume
            let completed = if resume {
                scan_completed_layers(&file_path)?
            } else {
                HashSet::new()
            };

            let pending: Vec<usize> = layers
                .iter()
                .filter(|l| !completed.contains(l))
                .copied()
                .collect();

            if pending.is_empty() {
                eprintln!("  {component}: all layers already completed, skipping");
                continue;
            }

            callbacks.on_component_start(component, pending.len());

            // Open writer (append for resume, create for fresh)
            let mut writer = if resume && file_path.exists() {
                let (w, existing) = VectorWriter::append(&file_path)?;
                eprintln!(
                    "  {component}: resuming ({} existing records, {} layers remaining)",
                    existing,
                    pending.len()
                );
                w
            } else {
                let mut w = VectorWriter::create(&file_path)?;
                let header = VectorFileHeader {
                    _header: true,
                    component: component.clone(),
                    model: self.model_name.clone(),
                    dimension: self.weights.hidden_size,
                    extraction_date: current_date(),
                };
                w.write_header(&header)?;
                w
            };

            let mut total_written = 0;

            for &layer in &pending {
                let layer_start = std::time::Instant::now();

                let count = match component.as_str() {
                    COMPONENT_FFN_DOWN => {
                        self.extract_ffn_down(layer, config, &mut writer, callbacks)?
                    }
                    COMPONENT_FFN_GATE => {
                        self.extract_ffn_gate(layer, config, &mut writer, callbacks)?
                    }
                    other => {
                        eprintln!("  {other}: not yet implemented, skipping");
                        break;
                    }
                };

                let elapsed_ms = layer_start.elapsed().as_secs_f64() * 1000.0;
                callbacks.on_layer_done(component, layer, count, elapsed_ms);
                total_written += count;
            }

            callbacks.on_component_done(component, total_written);

            summaries.push(ComponentSummary {
                component: component.clone(),
                vectors_written: total_written,
                output_path: file_path,
                elapsed_secs: comp_start.elapsed().as_secs_f64(),
            });
        }

        let total_vectors = summaries.iter().map(|s| s.vectors_written).sum();
        Ok(ExtractSummary {
            components: summaries,
            total_vectors,
            elapsed_secs: overall_start.elapsed().as_secs_f64(),
        })
    }
}

fn current_date() -> String {
    // Simple date without chrono dependency
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let days = now / 86400;
    // Approximate — good enough for a metadata field
    let year = 1970 + (days / 365);
    let remaining = days % 365;
    let month = remaining / 30 + 1;
    let day = remaining % 30 + 1;
    format!("{year}-{month:02}-{day:02}")
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

fn decode_token(tokenizer: &tokenizers::Tokenizer, id: u32) -> Option<String> {
    tokenizer
        .decode(&[id], true)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}
