//! Model weights serialization to/from .vindex directories.
//!
//! Split format (v2): separate files per component, no duplication.
//!   attn_weights.bin  — Q, K, V, O per layer
//!   ffn_weights.bin   — up + down per layer (gate is in gate_vectors.bin)
//!   norms.bin         — all LayerNorm vectors
//!   lm_head.bin       — output projection
//!
//! Legacy format (v1): single model_weights.bin with weight_manifest.json.

use std::collections::HashMap;
use std::io::{BufWriter, Write};
use std::path::Path;

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::error::VindexError;
use larql_models::ModelWeights;

use crate::extract::callbacks::IndexBuildCallbacks;
use crate::config::{VindexConfig, VindexModelConfig};
use crate::index::core::IndexLoadCallbacks; use crate::format::load::load_vindex_config;

#[derive(Serialize, Deserialize)]
struct WeightEntry {
    key: String,
    kind: String,
    shape: Vec<usize>,
    offset: u64,
    length: u64,
    #[serde(default)]
    file: String,
}

/// Write model weights to split component files (v2 format).
///
/// Creates separate files for attention, FFN (up+down only), norms, and lm_head.
/// Gate weights are NOT written — they're already in gate_vectors.bin.
pub fn write_model_weights(
    weights: &ModelWeights,
    dir: &Path,
    callbacks: &mut dyn IndexBuildCallbacks,
) -> Result<(), VindexError> {
    callbacks.on_stage("model_weights");
    let start = std::time::Instant::now();

    // Read dtype from config if available, default to F32
    let dtype = crate::format::load::load_vindex_config(dir)
        .map(|c| c.dtype)
        .unwrap_or(crate::config::dtype::StorageDtype::F32);

    let arch = &*weights.arch;
    let num_layers = weights.num_layers;
    let mut entries: Vec<WeightEntry> = Vec::new();

    // ── Attention weights ──
    let attn_path = dir.join("attn_weights.bin");
    let mut attn_file = BufWriter::new(std::fs::File::create(&attn_path)?);
    let mut attn_offset: u64 = 0;

    for layer in 0..num_layers {
        callbacks.on_layer_start("attn_weights", layer, num_layers);
        for key in &[
            arch.attn_q_key(layer),
            arch.attn_k_key(layer),
            arch.attn_v_key(layer),
            arch.attn_o_key(layer),
        ] {
            if let Some(tensor) = weights.tensors.get(key) {
                let len = write_tensor(&mut attn_file, tensor, dtype)?;
                entries.push(WeightEntry {
                    key: key.clone(),
                    kind: "tensor".into(),
                    shape: vec![tensor.shape()[0], tensor.shape()[1]],
                    offset: attn_offset,
                    length: len,
                    file: "attn_weights.bin".into(),
                });
                attn_offset += len;
            }
        }
        callbacks.on_layer_done("attn_weights", layer, 0.0);
    }
    attn_file.flush()?;

    // ── W_up weights (gate is in gate_vectors.bin, not duplicated) ──
    let up_path = dir.join("up_weights.bin");
    let mut up_file = BufWriter::new(std::fs::File::create(&up_path)?);
    let mut up_offset: u64 = 0;

    // ── W_down weights (full vectors for COMPILE) ──
    let down_path = dir.join("down_weights.bin");
    let mut down_file = BufWriter::new(std::fs::File::create(&down_path)?);
    let mut down_offset: u64 = 0;

    for layer in 0..num_layers {
        callbacks.on_layer_start("up/down_weights", layer, num_layers);

        if arch.is_moe() {
            for expert in 0..arch.num_experts() {
                if let Some(key) = arch.expert_ffn_up_key(layer, expert) {
                    if let Some(tensor) = weights.tensors.get(&key) {
                        let len = write_tensor(&mut up_file, tensor, dtype)?;
                        entries.push(WeightEntry {
                            key, kind: "tensor".into(),
                            shape: vec![tensor.shape()[0], tensor.shape()[1]],
                            offset: up_offset, length: len,
                            file: "up_weights.bin".into(),
                        });
                        up_offset += len;
                    }
                }
                if let Some(key) = arch.expert_ffn_down_key(layer, expert) {
                    if let Some(tensor) = weights.tensors.get(&key) {
                        let len = write_tensor(&mut down_file, tensor, dtype)?;
                        entries.push(WeightEntry {
                            key, kind: "tensor".into(),
                            shape: vec![tensor.shape()[0], tensor.shape()[1]],
                            offset: down_offset, length: len,
                            file: "down_weights.bin".into(),
                        });
                        down_offset += len;
                    }
                }
            }
            // MoE router weights (in up_weights alongside up projections)
            if let Some(key) = arch.moe_router_key(layer) {
                if let Some(tensor) = weights.tensors.get(&key) {
                    let len = write_tensor(&mut up_file, tensor, dtype)?;
                    entries.push(WeightEntry {
                        key, kind: "tensor".into(),
                        shape: vec![tensor.shape()[0], tensor.shape()[1]],
                        offset: up_offset, length: len,
                        file: "up_weights.bin".into(),
                    });
                    up_offset += len;
                }
            }
        } else {
            // Dense: separate up and down files
            let up_key = arch.ffn_up_key(layer);
            if let Some(tensor) = weights.tensors.get(&up_key) {
                let len = write_tensor(&mut up_file, tensor, dtype)?;
                entries.push(WeightEntry {
                    key: up_key, kind: "tensor".into(),
                    shape: vec![tensor.shape()[0], tensor.shape()[1]],
                    offset: up_offset, length: len,
                    file: "up_weights.bin".into(),
                });
                up_offset += len;
            }

            let down_key = arch.ffn_down_key(layer);
            if let Some(tensor) = weights.tensors.get(&down_key) {
                let len = write_tensor(&mut down_file, tensor, dtype)?;
                entries.push(WeightEntry {
                    key: down_key, kind: "tensor".into(),
                    shape: vec![tensor.shape()[0], tensor.shape()[1]],
                    offset: down_offset, length: len,
                    file: "down_weights.bin".into(),
                });
                down_offset += len;
            }
        }

        callbacks.on_layer_done("up/down_weights", layer, 0.0);
    }
    up_file.flush()?;
    down_file.flush()?;

    // ── Norms ──
    let norms_path = dir.join("norms.bin");
    let mut norms_file = BufWriter::new(std::fs::File::create(&norms_path)?);
    let mut norms_offset: u64 = 0;

    for (key, vec) in &weights.vectors {
        let bytes = crate::config::dtype::encode_floats(vec, dtype);
        norms_file.write_all(&bytes)?;
        entries.push(WeightEntry {
            key: key.clone(),
            kind: "vector".into(),
            shape: vec![vec.len()],
            offset: norms_offset,
            length: bytes.len() as u64,
            file: "norms.bin".into(),
        });
        norms_offset += bytes.len() as u64;
    }
    norms_file.flush()?;

    // ── LM Head ──
    let lm_head_path = dir.join("lm_head.bin");
    let lm_data = weights.lm_head.as_slice().unwrap();
    let lm_bytes = crate::config::dtype::encode_floats(lm_data, dtype);
    std::fs::write(&lm_head_path, &lm_bytes)?;
    entries.push(WeightEntry {
        key: "lm_head.weight".into(),
        kind: "tensor".into(),
        shape: vec![weights.lm_head.shape()[0], weights.lm_head.shape()[1]],
        offset: 0,
        length: lm_bytes.len() as u64,
        file: "lm_head.bin".into(),
    });

    // ── Manifest ──
    let manifest_json = serde_json::to_string_pretty(&entries)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    std::fs::write(dir.join("weight_manifest.json"), manifest_json)?;

    // ── Update index.json ──
    let config_path = dir.join("index.json");
    let config_text = std::fs::read_to_string(&config_path)?;
    let mut config: VindexConfig = serde_json::from_str(&config_text)
        .map_err(|e| VindexError::Parse(e.to_string()))?;

    config.has_model_weights = true;
    config.model_config = Some(VindexModelConfig {
        model_type: weights.arch.config().model_type.clone(),
        head_dim: weights.head_dim,
        num_q_heads: weights.num_q_heads,
        num_kv_heads: weights.num_kv_heads,
        rope_base: weights.rope_base,
        sliding_window: weights.arch.config().sliding_window,
        moe: if weights.arch.is_moe() {
            Some(crate::MoeConfig {
                num_experts: weights.arch.num_experts(),
                top_k: weights.arch.num_experts_per_token(),
                shared_expert: weights.arch.num_shared_experts() > 0,
                router_type: "top_k_softmax".into(),
            })
        } else {
            None
        },
    });

    let config_json = serde_json::to_string_pretty(&config)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    std::fs::write(&config_path, config_json)?;

    callbacks.on_stage_done("model_weights", start.elapsed().as_secs_f64() * 1000.0);
    Ok(())
}

fn write_tensor(w: &mut BufWriter<std::fs::File>, tensor: &Array2<f32>, dtype: crate::config::dtype::StorageDtype) -> Result<u64, VindexError> {
    let data = tensor.as_slice().unwrap();
    let bytes = crate::config::dtype::encode_floats(data, dtype);
    w.write_all(&bytes)?;
    Ok(bytes.len() as u64)
}

/// Load a full ModelWeights from a vindex directory.
///
/// Tries split files (v2) first, falls back to model_weights.bin (v1).
pub fn load_model_weights(
    dir: &Path,
    callbacks: &mut dyn IndexLoadCallbacks,
) -> Result<ModelWeights, VindexError> {
    let config = load_vindex_config(dir)?;

    if !config.has_model_weights {
        return Err(VindexError::Parse(
            "vindex does not contain model weights. Rebuild with: larql extract-index <model> -o <vindex> --include-weights".into(),
        ));
    }

    let model_cfg = config.model_config.as_ref().ok_or_else(|| {
        VindexError::Parse("vindex missing model_config in index.json".into())
    })?;

    // Reconstruct architecture from config
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
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    callbacks.on_file_done("embeddings", config.vocab_size, 0.0);

    // Load weight manifest
    let manifest_path = dir.join("weight_manifest.json");
    if !manifest_path.exists() {
        return Err(VindexError::Parse(
            "weight_manifest.json not found".into(),
        ));
    }

    callbacks.on_file_start("model_weights", "weight_manifest.json");
    let manifest_text = std::fs::read_to_string(&manifest_path)?;
    let entries: Vec<WeightEntry> = serde_json::from_str(&manifest_text)
        .map_err(|e| VindexError::Parse(e.to_string()))?;

    // Cache loaded file data to avoid re-reading
    let mut file_cache: HashMap<String, Vec<u8>> = HashMap::new();

    let mut tensors: HashMap<String, Array2<f32>> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut lm_head_loaded: Option<Array2<f32>> = None;

    for entry in &entries {
        // Determine which file to read from
        let filename = if entry.file.is_empty() {
            "model_weights.bin".to_string() // legacy v1 format
        } else {
            entry.file.clone()
        };

        let data = file_cache.entry(filename.clone()).or_insert_with(|| {
            std::fs::read(dir.join(&filename)).unwrap_or_default()
        });

        if data.is_empty() {
            continue;
        }

        let byte_offset = entry.offset as usize;
        let byte_count = entry.length as usize;
        if byte_offset + byte_count > data.len() {
            continue;
        }
        let raw_bytes = &data[byte_offset..byte_offset + byte_count];
        let floats = crate::config::dtype::decode_floats(raw_bytes, config.dtype);
        let slice = &floats[..];

        match entry.kind.as_str() {
            "tensor" => {
                let arr = Array2::from_shape_vec(
                    (entry.shape[0], entry.shape[1]),
                    slice.to_vec(),
                )
                .map_err(|e| VindexError::Parse(e.to_string()))?;

                if entry.key == "lm_head.weight" {
                    lm_head_loaded = Some(arr);
                } else {
                    tensors.insert(entry.key.clone(), arr);
                }
            }
            "vector" => {
                vectors.insert(entry.key.clone(), slice.to_vec());
            }
            _ => {}
        }
    }

    // Gate vectors: read from gate_vectors.bin and inject into tensors
    // (the forward pass needs them as tensors, but they're stored in the query index)
    let gate_bytes = std::fs::read(dir.join("gate_vectors.bin"))?;
    let gate_floats = crate::config::dtype::decode_floats(&gate_bytes, config.dtype);
    let bpf = crate::config::dtype::bytes_per_float(config.dtype);
    for info in &config.layers {
        let float_offset = info.offset as usize / bpf;
        let float_count = info.num_features * config.hidden_size;
        if float_offset + float_count <= gate_floats.len() {
            let gate_data = &gate_floats[float_offset..float_offset + float_count];
            let gate_matrix = Array2::from_shape_vec(
                (info.num_features, config.hidden_size),
                gate_data.to_vec(),
            )
            .map_err(|e| VindexError::Parse(e.to_string()))?;
            let gate_key = arch.ffn_gate_key(info.layer);
            tensors.insert(gate_key, gate_matrix);
        }
    }

    callbacks.on_file_done("model_weights", entries.len(), 0.0);

    let cfg = arch.config();
    let lm_head = lm_head_loaded.unwrap_or_else(|| embed.clone());

    Ok(ModelWeights {
        tensors,
        vectors,
        embed,
        lm_head,
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

/// Find the tokenizer path near a model or vindex directory.
pub fn find_tokenizer_path(dir: &Path) -> Option<std::path::PathBuf> {
    let p = dir.join("tokenizer.json");
    if p.exists() {
        return Some(p);
    }
    if let Some(parent) = dir.parent() {
        let p = parent.join("tokenizer.json");
        if p.exists() {
            return Some(p);
        }
    }
    None
}
