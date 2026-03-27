//! Load weight tensors from safetensors files in a model directory.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ndarray::Array2;

/// A loaded model's weight tensors and config.
pub struct ModelWeights {
    pub tensors: HashMap<String, Array2<f32>>,
    pub embed: Array2<f32>,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
}

/// Load all safetensors files from a model directory.
/// Strips `language_model.model.` prefix from keys (Gemma 3 convention).
pub fn load_model_dir(path: impl AsRef<Path>) -> Result<ModelWeights, WalkerError> {
    let path = path.as_ref();
    if !path.is_dir() {
        return Err(WalkerError::NotADirectory(path.to_path_buf()));
    }

    // Find safetensors files
    let mut st_files: Vec<PathBuf> = std::fs::read_dir(path)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .is_some_and(|ext| ext == "safetensors")
        })
        .collect();
    st_files.sort();

    if st_files.is_empty() {
        return Err(WalkerError::NoSafetensors(path.to_path_buf()));
    }

    // Load all tensors
    let mut tensors: HashMap<String, Array2<f32>> = HashMap::new();

    for st_path in &st_files {
        let file = std::fs::File::open(st_path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let st = safetensors::SafeTensors::deserialize(&mmap)
            .map_err(|e| WalkerError::Parse(e.to_string()))?;

        for (name, view) in st.tensors() {
            let shape = view.shape();
            if shape.len() != 2 {
                continue; // skip non-2D tensors (norms, etc.)
            }

            let key = normalize_key(&name);
            let data = tensor_to_f32(&view)?;
            let arr = Array2::from_shape_vec((shape[0], shape[1]), data)
                .map_err(|e| WalkerError::Parse(e.to_string()))?;
            tensors.insert(key, arr);
        }
    }

    // Load config
    let config_path = path.join("config.json");
    let config = if config_path.exists() {
        let text = std::fs::read_to_string(&config_path)?;
        serde_json::from_str::<serde_json::Value>(&text)
            .map_err(|e| WalkerError::Parse(e.to_string()))?
    } else {
        serde_json::json!({})
    };
    let text_config = config.get("text_config").unwrap_or(&config);

    let num_layers = text_config["num_hidden_layers"].as_u64().unwrap_or(32) as usize;
    let hidden_size = text_config["hidden_size"].as_u64().unwrap_or(2048) as usize;
    let intermediate_size = text_config["intermediate_size"].as_u64().unwrap_or(8192) as usize;

    // Find embedding matrix
    let embed = tensors
        .get("embed_tokens.weight")
        .ok_or_else(|| WalkerError::MissingTensor("embed_tokens.weight".into()))?
        .clone();

    let vocab_size = embed.shape()[0];

    Ok(ModelWeights {
        tensors,
        embed,
        num_layers,
        hidden_size,
        intermediate_size,
        vocab_size,
    })
}

/// Strip common prefixes from tensor keys.
fn normalize_key(key: &str) -> String {
    let key = key
        .strip_prefix("language_model.model.")
        .or_else(|| key.strip_prefix("model."))
        .unwrap_or(key);
    key.to_string()
}

/// Convert a safetensors tensor view to Vec<f32>.
fn tensor_to_f32(view: &safetensors::tensor::TensorView<'_>) -> Result<Vec<f32>, WalkerError> {
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
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    half_to_f32(bits)
                })
                .collect())
        }
        safetensors::Dtype::BF16 => {
            let bytes = view.data();
            Ok(bytes
                .chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    bf16_to_f32(bits)
                })
                .collect())
        }
        other => Err(WalkerError::UnsupportedDtype(format!("{other:?}"))),
    }
}

fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        // Denormalized
        let mut e = 1u32;
        let mut m = mant;
        while (m & 0x400) == 0 {
            m <<= 1;
            e += 1;
        }
        let exp32 = (127 - 15 + 1 - e) << 23;
        let mant32 = (m & 0x3FF) << 13;
        return f32::from_bits(sign | exp32 | mant32);
    }
    if exp == 31 {
        let exp32 = 0xFF << 23;
        let mant32 = mant << 13;
        return f32::from_bits(sign | exp32 | mant32);
    }

    let exp32 = (exp + 127 - 15) << 23;
    let mant32 = mant << 13;
    f32::from_bits(sign | exp32 | mant32)
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[derive(Debug, thiserror::Error)]
pub enum WalkerError {
    #[error("not a directory: {0}")]
    NotADirectory(PathBuf),
    #[error("no safetensors files in {0}")]
    NoSafetensors(PathBuf),
    #[error("missing tensor: {0}")]
    MissingTensor(String),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("unsupported dtype: {0}")]
    UnsupportedDtype(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
