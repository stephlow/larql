//! Binary down_meta format — compact storage for per-feature output metadata.
//!
//! Replaces down_meta.jsonl (~160 MB) with a binary format (~30 MB for top_k=10).
//! Token strings are resolved at read time via the tokenizer.
//!
//! File: down_meta.bin
//! Format:
//!   Header (16 bytes): magic, version, num_layers, top_k
//!   Per layer: num_features (u32), then fixed-size records
//!   Per feature: top_token_id (u32), c_score (f32), top_k × (token_id u32, logit f32)

use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::error::VindexError;
use crate::index::FeatureMeta;

const MAGIC: u32 = 0x444D4554; // "DMET"
const FORMAT_VERSION: u32 = 1;

/// Write down_meta in binary format.
pub fn write_binary(
    dir: &Path,
    down_meta: &[Option<Vec<Option<FeatureMeta>>>],
    top_k_count: usize,
) -> Result<usize, VindexError> {
    let path = dir.join("down_meta.bin");
    let file = std::fs::File::create(&path)?;
    let mut w = BufWriter::new(file);
    let mut total = 0usize;

    let num_layers = down_meta.len() as u32;

    // Header
    w.write_all(&MAGIC.to_le_bytes())?;
    w.write_all(&FORMAT_VERSION.to_le_bytes())?;
    w.write_all(&num_layers.to_le_bytes())?;
    w.write_all(&(top_k_count as u32).to_le_bytes())?;

    // Per layer
    for layer_meta in down_meta.iter() {
        match layer_meta {
            Some(features) => {
                let num_features = features.len() as u32;
                w.write_all(&num_features.to_le_bytes())?;

                for meta_opt in features {
                    match meta_opt {
                        Some(meta) => {
                            w.write_all(&meta.top_token_id.to_le_bytes())?;
                            w.write_all(&meta.c_score.to_le_bytes())?;

                            // Write exactly top_k_count entries (pad with zeros)
                            for i in 0..top_k_count {
                                if i < meta.top_k.len() {
                                    w.write_all(&meta.top_k[i].token_id.to_le_bytes())?;
                                    w.write_all(&meta.top_k[i].logit.to_le_bytes())?;
                                } else {
                                    w.write_all(&0u32.to_le_bytes())?;
                                    w.write_all(&0f32.to_le_bytes())?;
                                }
                            }
                            total += 1;
                        }
                        None => {
                            // Empty feature: token_id=0, c_score=0, all top_k zeros
                            w.write_all(&0u32.to_le_bytes())?;
                            w.write_all(&0f32.to_le_bytes())?;
                            for _ in 0..top_k_count {
                                w.write_all(&0u32.to_le_bytes())?;
                                w.write_all(&0f32.to_le_bytes())?;
                            }
                        }
                    }
                }
            }
            None => {
                w.write_all(&0u32.to_le_bytes())?; // 0 features
            }
        }
    }

    w.flush()?;
    Ok(total)
}

/// Read down_meta from binary format.
/// Token strings are resolved via the tokenizer.
pub fn read_binary(
    dir: &Path,
    tokenizer: &tokenizers::Tokenizer,
) -> Result<(Vec<Option<Vec<Option<FeatureMeta>>>>, usize), VindexError> {
    let path = dir.join("down_meta.bin");
    let file = std::fs::File::open(&path)?;
    let mut r = BufReader::new(file);

    // Header
    let magic = read_u32(&mut r)?;
    if magic != MAGIC {
        return Err(VindexError::Parse(format!(
            "invalid down_meta.bin magic: expected 0x{MAGIC:08X}, got 0x{magic:08X}"
        )));
    }
    let version = read_u32(&mut r)?;
    if version != FORMAT_VERSION {
        return Err(VindexError::Parse(format!(
            "unsupported down_meta.bin version: {version}"
        )));
    }
    let num_layers = read_u32(&mut r)? as usize;
    let top_k_count = read_u32(&mut r)? as usize;

    let mut down_meta: Vec<Option<Vec<Option<FeatureMeta>>>> = Vec::with_capacity(num_layers);
    let mut total = 0usize;

    for _ in 0..num_layers {
        let num_features = read_u32(&mut r)? as usize;
        if num_features == 0 {
            down_meta.push(None);
            continue;
        }

        let mut features: Vec<Option<FeatureMeta>> = Vec::with_capacity(num_features);
        for _ in 0..num_features {
            let top_token_id = read_u32(&mut r)?;
            let c_score = read_f32(&mut r)?;

            let mut top_k = Vec::with_capacity(top_k_count);
            for _ in 0..top_k_count {
                let token_id = read_u32(&mut r)?;
                let logit = read_f32(&mut r)?;
                if token_id > 0 || logit != 0.0 {
                    let token = tokenizer
                        .decode(&[token_id], true)
                        .unwrap_or_else(|_| format!("T{token_id}"))
                        .trim()
                        .to_string();
                    top_k.push(larql_models::TopKEntry {
                        token,
                        token_id,
                        logit,
                    });
                }
            }

            if top_token_id == 0 && c_score == 0.0 && top_k.is_empty() {
                features.push(None);
            } else {
                let top_token = tokenizer
                    .decode(&[top_token_id], true)
                    .unwrap_or_else(|_| format!("T{top_token_id}"))
                    .trim()
                    .to_string();
                features.push(Some(FeatureMeta {
                    top_token,
                    top_token_id,
                    c_score,
                    top_k,
                }));
                total += 1;
            }
        }

        down_meta.push(Some(features));
    }

    Ok((down_meta, total))
}

/// Check if a binary down_meta.bin exists in the directory.
pub fn has_binary(dir: &Path) -> bool {
    dir.join("down_meta.bin").exists()
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32, VindexError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_f32<R: Read>(r: &mut R) -> Result<f32, VindexError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}
