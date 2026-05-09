//! Q4_K weight loader — reconstructs the minimum `ModelWeights` needed
//! to drive a Q4_K vindex forward pass: embeddings, norms, optional
//! `lm_head`, and packed-byte-range references for `attn_weights_q4k.bin`,
//! `interleaved_q4k.bin`, and the per-layer `layers/layer_{L:02}.weights`
//! files. The forward pass dequantises on demand.

use std::collections::HashMap;
use std::path::Path;

use ndarray::Array2;

use larql_models::ModelWeights;

use crate::error::VindexError;
use crate::format::filenames::*;
use crate::format::load::load_vindex_config;
use crate::index::core::IndexLoadCallbacks;

use super::super::write_f32::{kind, WeightEntry};
use super::expert_in_shard;

/// Expert-shard variant of [`super::load_model_weights_q4k`].
///
/// Identical to the full loader except that when `expert_filter` is `Some((start,
/// end_excl))`, per-layer expert entries outside `[start, end_excl)` are not
/// inserted into `packed_byte_ranges`. Only the owned experts' byte-range
/// records are kept; the mmap of each layer file still covers the whole file
/// (the OS pages for unowned experts simply stay unpopulated).
///
/// A mini-process launched with `--experts 0-15` sets
/// `expert_filter = Some((0, 16))` and loads only experts 0–15, reducing
/// steady-state RSS from ~15 GB (all 128 experts) to ~120 MB (16 experts × 30
/// layers × 4 MB each).
pub fn load_model_weights_q4k_shard(
    dir: &Path,
    callbacks: &mut dyn IndexLoadCallbacks,
    expert_filter: Option<(usize, usize)>,
) -> Result<ModelWeights, VindexError> {
    let config = load_vindex_config(dir)?;

    if !config.has_model_weights {
        return Err(VindexError::Parse(
            "vindex does not contain model weights. Rebuild with --level all --quant q4k".into(),
        ));
    }
    if config.quant != crate::QuantFormat::Q4K {
        return Err(VindexError::Parse(format!(
            "load_model_weights_q4k expects a Q4_K vindex, got quant={}",
            config.quant,
        )));
    }

    let model_cfg = config
        .model_config
        .as_ref()
        .ok_or_else(|| VindexError::Parse("vindex missing model_config in index.json".into()))?;

    // Reconstruct architecture (same as load_model_weights — Gemma 4 per-layer
    // geometry propagates through model_cfg).
    let mut arch_obj = serde_json::json!({
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
    let obj = arch_obj.as_object_mut().unwrap();
    if let Some(v) = model_cfg.global_head_dim {
        obj.insert("global_head_dim".into(), v.into());
    }
    if let Some(v) = model_cfg.num_global_kv_heads {
        obj.insert("num_global_key_value_heads".into(), v.into());
    }
    if let Some(v) = model_cfg.partial_rotary_factor {
        obj.insert("partial_rotary_factor".into(), v.into());
    }
    if let Some(v) = model_cfg.sliding_window_pattern {
        obj.insert("sliding_window_pattern".into(), v.into());
    }
    if let Some(ref v) = model_cfg.layer_types {
        obj.insert(
            "layer_types".into(),
            serde_json::to_value(v).unwrap_or_default(),
        );
    }
    if model_cfg.attention_k_eq_v {
        obj.insert("attention_k_eq_v".into(), true.into());
    }
    if let Some(v) = model_cfg.num_kv_shared_layers {
        obj.insert("num_kv_shared_layers".into(), v.into());
    }
    if let Some(v) = model_cfg.per_layer_embed_dim {
        obj.insert("hidden_size_per_layer_input".into(), v.into());
    }
    if let Some(v) = model_cfg.rope_local_base {
        obj.insert("rope_local_base_freq".into(), v.into());
    }
    if let Some(v) = model_cfg.query_pre_attn_scalar {
        obj.insert("query_pre_attn_scalar".into(), v.into());
    }
    if let Some(v) = model_cfg.final_logit_softcapping {
        obj.insert("final_logit_softcapping".into(), v.into());
    }
    if let Some(ref moe) = model_cfg.moe {
        obj.insert("num_experts".into(), moe.num_experts.into());
        obj.insert("top_k_experts".into(), moe.top_k.into());
        if let Some(v) = moe.moe_intermediate_size {
            obj.insert("moe_intermediate_size".into(), v.into());
        }
        if moe.hybrid {
            obj.insert("enable_moe_block".into(), true.into());
        }
    }
    let arch = larql_models::detect_from_json(&arch_obj);

    // Embeddings — required for token lookup at layer 0.
    callbacks.on_file_start(
        "embeddings",
        &dir.join(EMBEDDINGS_BIN).display().to_string(),
    );
    let embed_file = std::fs::File::open(dir.join(EMBEDDINGS_BIN))?;
    let embed_mmap = unsafe { memmap2::Mmap::map(&embed_file)? };
    let expected_f32 = config.vocab_size * config.hidden_size * 4;
    let embed_dtype = if embed_mmap.len() == expected_f32 {
        crate::config::dtype::StorageDtype::F32
    } else {
        crate::config::dtype::StorageDtype::F16
    };
    let embed_floats = crate::config::dtype::decode_floats(&embed_mmap, embed_dtype);
    let embed = Array2::from_shape_vec((config.vocab_size, config.hidden_size), embed_floats)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    callbacks.on_file_done("embeddings", config.vocab_size, 0.0);

    // norms.bin (f32) — loaded via weight_manifest.json, filtered to vector entries.
    let manifest_path = dir.join(WEIGHT_MANIFEST_JSON);
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut tensors: HashMap<String, larql_models::WeightArray> = HashMap::new();
    let mut packed_mmaps: HashMap<String, memmap2::Mmap> = HashMap::new();
    let mut packed_byte_ranges: HashMap<String, (String, usize, usize)> = HashMap::new();
    let mut lm_head_loaded: Option<larql_models::WeightArray> = None;

    if manifest_path.exists() {
        let manifest_text = std::fs::read_to_string(&manifest_path)?;
        let entries: Vec<WeightEntry> =
            serde_json::from_str(&manifest_text).map_err(|e| VindexError::Parse(e.to_string()))?;

        let mut mmap_cache: HashMap<String, memmap2::Mmap> = HashMap::new();
        for entry in &entries {
            if entry.file.is_empty() {
                continue;
            }
            if entry.kind != kind::VECTOR
                && entry.kind != kind::TENSOR_Q4K
                && entry.kind != kind::TENSOR_F16
                && entry.kind != kind::PACKED_BF16
            {
                continue;
            }

            if !mmap_cache.contains_key(&entry.file) {
                let fpath = dir.join(&entry.file);
                if let Ok(f) = std::fs::File::open(&fpath) {
                    if let Ok(m) = unsafe { memmap2::Mmap::map(&f) } {
                        mmap_cache.insert(entry.file.clone(), m);
                    }
                }
            }
            let data = match mmap_cache.get(&entry.file) {
                Some(m) => m.as_ref(),
                None => continue,
            };
            let byte_offset = entry.offset as usize;
            let byte_count = entry.length as usize;
            if byte_offset + byte_count > data.len() {
                continue;
            }
            let raw_bytes = &data[byte_offset..byte_offset + byte_count];

            if entry.kind == kind::PACKED_BF16 {
                // Record the byte range into the mmap — do NOT clone (could be 43 GB).
                // The mmap stays alive in packed_mmaps; get_packed_bytes() returns the slice.
                packed_byte_ranges.insert(
                    entry.key.clone(),
                    (entry.file.clone(), byte_offset, byte_count),
                );
            } else if entry.kind == kind::VECTOR {
                let expected_floats: usize = entry.shape.iter().product();
                let actual_dtype = if byte_count == expected_floats * 4 {
                    crate::config::dtype::StorageDtype::F32
                } else if byte_count == expected_floats * 2 {
                    crate::config::dtype::StorageDtype::F16
                } else {
                    config.dtype
                };
                let floats = crate::config::dtype::decode_floats(raw_bytes, actual_dtype);
                vectors.insert(entry.key.clone(), floats);
            } else {
                // tensor_q4k / tensor_f16: 2D tensor (PLE weights for Gemma 4
                // E2B). Decode to f32 and insert into weights.tensors so
                // `ple.rs` can look it up like any other dense matrix.
                if entry.shape.len() != 2 {
                    continue;
                }
                let rows = entry.shape[0];
                let cols = entry.shape[1];
                let n = rows * cols;
                let floats: Option<Vec<f32>> = if entry.kind == kind::TENSOR_Q4K {
                    let padded = n.div_ceil(larql_models::quant::ggml::Q4_K_BLOCK_ELEMS)
                        * larql_models::quant::ggml::Q4_K_BLOCK_ELEMS;
                    larql_models::quant::ggml::dequantize_q4_k(raw_bytes, padded).ok()
                } else {
                    // tensor_f16 — raw bytes are IEEE half-precision.
                    Some(crate::config::dtype::decode_floats(
                        raw_bytes,
                        crate::config::dtype::StorageDtype::F16,
                    ))
                };
                if let Some(floats) = floats {
                    if floats.len() >= n {
                        if let Ok(arr) = Array2::from_shape_vec((rows, cols), floats[..n].to_vec())
                        {
                            tensors.insert(entry.key.clone(), arr.into_shared());
                        }
                    }
                }
            }
        }
        // Move packed file mmaps into the outer map so they outlive this block.
        for (filename, mmap) in mmap_cache {
            if packed_byte_ranges.values().any(|(f, _, _)| f == &filename) {
                packed_mmaps.insert(filename, mmap);
            }
        }
    }

    // ── Per-layer FFN weights: layers/layer_{L:02}.weights (§5.12) ──────────
    // Loaded when index.json carries `ffn_layout: "per_layer"`. For each
    // layer file: mmap it, parse the header + offset table, record per-entry
    // byte ranges keyed as `"layers/{layer}/{entry}/gate_up"` and `"layers/{layer}/{entry}/down"`.
    if config.ffn_layout == Some(crate::config::FfnLayout::PerLayer) {
        use super::super::write_layers::parse_layer_weights_header;
        use crate::format::filenames::layer_weights_filename;
        for l in 0..config.num_layers {
            let filename = layer_weights_filename(l);
            let fpath = dir.join(&filename);
            if !fpath.exists() {
                continue;
            }
            if let Ok(f) = std::fs::File::open(&fpath) {
                if let Ok(mmap) = unsafe { memmap2::Mmap::map(&f) } {
                    if let Some((_fmt, _num_entries, _inter, _hidden, offsets)) =
                        parse_layer_weights_header(&mmap)
                    {
                        // Use the shared key builder from larql-models so the
                        // loader and `ModelWeights::get_layer_entry_bytes` stay
                        // in lockstep. Drift here causes silent None returns.
                        for (e, (gu_off, gu_bytes, dn_off, dn_bytes)) in offsets.iter().enumerate()
                        {
                            if !expert_in_shard(e, expert_filter) {
                                continue;
                            }
                            packed_byte_ranges.insert(
                                larql_models::weights::per_layer_ffn_key(
                                    l,
                                    e,
                                    larql_models::weights::PER_LAYER_FFN_GATE_UP,
                                ),
                                (filename.clone(), *gu_off, *gu_bytes),
                            );
                            packed_byte_ranges.insert(
                                larql_models::weights::per_layer_ffn_key(
                                    l,
                                    e,
                                    larql_models::weights::PER_LAYER_FFN_DOWN,
                                ),
                                (filename.clone(), *dn_off, *dn_bytes),
                            );
                        }
                        packed_mmaps.insert(filename, mmap);
                    }
                }
            }
        }
    }

    // lm_head_q4.bin (Q4_K of the output projection) — dequant to f32. If
    // absent (tied embeddings), fall back to embed.clone() below.
    let lm_q4_path = dir.join(LM_HEAD_Q4_BIN);
    if lm_q4_path.exists() {
        let bytes = std::fs::read(&lm_q4_path)?;
        let num_floats = config.vocab_size * config.hidden_size;
        let padded = num_floats.div_ceil(larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS)
            * larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS;
        if let Ok(floats) = larql_models::quant::ggml::dequantize_q4_k(&bytes, padded) {
            if floats.len() >= num_floats {
                if let Ok(arr) = Array2::from_shape_vec(
                    (config.vocab_size, config.hidden_size),
                    floats[..num_floats].to_vec(),
                ) {
                    lm_head_loaded = Some(arr.into_shared());
                }
            }
        }
    }

    let cfg = arch.config();
    let embed = embed.into_shared();
    let lm_head = lm_head_loaded.unwrap_or_else(|| embed.clone());

    Ok(ModelWeights {
        tensors,
        vectors,
        raw_bytes: std::collections::HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_mmaps,
        packed_byte_ranges,
        embed,
        lm_head,
        position_embed: None,
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

