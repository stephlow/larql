//! Model loading — safetensors, MLX, GGUF → ModelWeights.
//!
//! This is the entry point for extracting models into vindexes.
//! Handles dtype conversion (f16, bf16 → f32), HuggingFace cache resolution,
//! and architecture detection.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ndarray::Array2;

use larql_models::ModelWeights;
use crate::error::VindexError;

/// Load all safetensors files from a model directory.
/// Detects architecture from config.json and loads all weight tensors.
pub fn load_model_dir(path: impl AsRef<Path>) -> Result<ModelWeights, VindexError> {
    let path = path.as_ref();
    if !path.is_dir() {
        return Err(VindexError::NotADirectory(path.to_path_buf()));
    }

    let arch = larql_models::detect_architecture(path)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    let prefixes = arch.key_prefixes_to_strip();

    let mut st_files: Vec<PathBuf> = std::fs::read_dir(path)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    st_files.sort();

    if st_files.is_empty() {
        return Err(VindexError::NoSafetensors(path.to_path_buf()));
    }

    let mut tensors: HashMap<String, Array2<f32>> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();

    for st_path in &st_files {
        let file = std::fs::File::open(st_path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let st = safetensors::SafeTensors::deserialize(&mmap)
            .map_err(|e| VindexError::Parse(e.to_string()))?;

        for (name, view) in st.tensors() {
            let key = normalize_key(&name, prefixes);
            let shape = view.shape();
            let data = tensor_to_f32(&view)?;

            match shape.len() {
                2 => {
                    let arr = Array2::from_shape_vec((shape[0], shape[1]), data)
                        .map_err(|e| VindexError::Parse(e.to_string()))?;
                    tensors.insert(key, arr);
                }
                1 => {
                    vectors.insert(key, data);
                }
                _ => {}
            }
        }
    }

    let embed_key = arch.embed_key();
    let embed = tensors
        .get(embed_key)
        .ok_or_else(|| VindexError::MissingTensor(embed_key.into()))?
        .clone();

    let lm_head = tensors
        .get("lm_head.weight")
        .cloned()
        .unwrap_or_else(|| embed.clone());

    let vocab_size = lm_head.shape()[0];
    let cfg = arch.config();

    Ok(ModelWeights {
        tensors,
        vectors,
        embed,
        lm_head,
        num_layers: cfg.num_layers,
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        vocab_size,
        head_dim: cfg.head_dim,
        num_q_heads: cfg.num_q_heads,
        num_kv_heads: cfg.num_kv_heads,
        rope_base: cfg.rope_base,
        arch,
    })
}

/// Resolve a HuggingFace model ID or path to a local directory.
pub fn resolve_model_path(model: &str) -> Result<PathBuf, VindexError> {
    let path = PathBuf::from(model);
    if path.is_dir() {
        return Ok(path);
    }

    // Try HuggingFace cache
    let cache_name = format!("models--{}", model.replace('/', "--"));
    let home = std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    let hf_cache = home.join(format!(".cache/huggingface/hub/{cache_name}/snapshots"));

    if hf_cache.is_dir() {
        if let Some(snapshot) = std::fs::read_dir(&hf_cache)
            .ok()
            .and_then(|mut d| d.next())
            .and_then(|e| e.ok())
        {
            let snapshot_path = snapshot.path();
            if snapshot_path.is_dir() {
                return Ok(snapshot_path);
            }
        }
    }

    Err(VindexError::NotADirectory(path))
}

fn normalize_key(key: &str, prefixes: &[&str]) -> String {
    for prefix in prefixes {
        if let Some(stripped) = key.strip_prefix(prefix) {
            return stripped.to_string();
        }
    }
    key.to_string()
}

fn tensor_to_f32(view: &safetensors::tensor::TensorView<'_>) -> Result<Vec<f32>, VindexError> {
    match view.dtype() {
        safetensors::Dtype::F32 => {
            let bytes = view.data();
            Ok(bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        }
        safetensors::Dtype::F16 => {
            let bytes = view.data();
            Ok(bytes
                .chunks_exact(2)
                .map(|b| half_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect())
        }
        safetensors::Dtype::BF16 => {
            let bytes = view.data();
            Ok(bytes
                .chunks_exact(2)
                .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect())
        }
        other => Err(VindexError::UnsupportedDtype(format!("{other:?}"))),
    }
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

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}
