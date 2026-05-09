//! Resume an interrupted vindex build — rebuilds clustering, tokenizer,
//! and `index.json` from the partial output of an earlier run.

use std::path::Path;

use larql_models::ModelWeights;

use crate::config::{VindexConfig, VindexModelConfig};
use crate::error::VindexError;
use crate::extract::build_helpers::{
    build_whole_word_vocab, chrono_now, compute_gate_top_tokens, compute_offset_direction,
    run_clustering_pipeline, ClusterData,
};
use crate::extract::callbacks::IndexBuildCallbacks;
use crate::extract::constants::{DEFAULT_DOWN_TOP_K, FIRST_CONTENT_TOKEN_ID};
use crate::extract::stage_labels::*;
use crate::format::filenames::*;

use super::knowledge_layer_range;

/// Resume an interrupted vindex build.
/// Assumes gate_vectors.bin, embeddings.bin, and down_meta.jsonl exist.
/// Runs: relation clustering + tokenizer + index.json.
pub fn build_vindex_resume(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    model_name: &str,
    output_dir: &Path,
    callbacks: &mut dyn IndexBuildCallbacks,
) -> Result<(), VindexError> {
    let num_layers = weights.num_layers;
    let hidden_size = weights.hidden_size;
    let intermediate_size = weights.intermediate_size;
    let vocab_size = weights.vocab_size;
    let embed_scale = weights.arch.embed_scale();

    // Reconstruct layer_infos from gate_vectors.bin
    let gate_path = output_dir.join(GATE_VECTORS_BIN);
    let gate_size = std::fs::metadata(&gate_path)?.len();
    let bytes_per_layer = (intermediate_size * hidden_size * 4) as u64;
    let mut layer_infos = Vec::new();
    for layer in 0..num_layers {
        layer_infos.push(crate::config::VindexLayerInfo {
            layer,
            num_features: intermediate_size,
            offset: layer as u64 * bytes_per_layer,
            length: bytes_per_layer,
            num_experts: None,
            num_features_per_expert: None,
        });
    }
    eprintln!(
        "  Reconstructed {} layer infos from gate_vectors.bin ({:.1} GB)",
        layer_infos.len(),
        gate_size as f64 / 1e9
    );

    // Read down_meta.jsonl to collect cluster directions for the model's
    // knowledge band.
    let family = weights.arch.family().to_string();
    let knowledge_layers = knowledge_layer_range(&family, num_layers);
    let mut cluster_directions: Vec<f32> = Vec::new();
    let mut cluster_features: Vec<(usize, usize)> = Vec::new();
    let mut cluster_top_tokens: Vec<String> = Vec::new();
    let mut cluster_input_tokens: Vec<String> = Vec::new();
    let mut cluster_output_tokens: Vec<String> = Vec::new();

    eprintln!("  Building whole-word vocabulary...");
    let (ww_ids, ww_embed) =
        build_whole_word_vocab(tokenizer, &weights.embed, vocab_size, hidden_size);

    let mut gate_top_tokens_per_layer: std::collections::HashMap<usize, Vec<String>> =
        std::collections::HashMap::new();
    if let Some((cluster_layer_min, cluster_layer_max)) = knowledge_layers {
        eprintln!(
            "  Computing gate input tokens for L{}-{}...",
            cluster_layer_min,
            cluster_layer_max.saturating_sub(1)
        );
        for layer in cluster_layer_min..cluster_layer_max {
            let layer_start = std::time::Instant::now();
            let tokens = compute_gate_top_tokens(
                weights,
                tokenizer,
                layer,
                intermediate_size,
                &ww_ids,
                &ww_embed,
            );
            gate_top_tokens_per_layer.insert(layer, tokens);
            eprintln!(
                "    gate L{:2}: {:.1}s",
                layer,
                layer_start.elapsed().as_secs_f64()
            );
        }
    } else {
        eprintln!("  Skipping relation clustering: no knowledge band for this model");
    }
    eprintln!(
        "  Gate input tokens computed for {} layers",
        gate_top_tokens_per_layer.len()
    );

    eprintln!("  Reading down_meta.jsonl for offset directions...");
    let down_path = output_dir.join(DOWN_META_JSONL);
    let down_file = std::fs::File::open(&down_path)?;
    let reader = std::io::BufReader::new(down_file);
    let mut count = 0usize;
    for line in std::io::BufRead::lines(reader) {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let obj: serde_json::Value =
            serde_json::from_str(line).map_err(|e| VindexError::Parse(e.to_string()))?;
        if obj.get("_header").is_some() {
            continue;
        }

        let layer = obj.get("l").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let feat = obj.get("f").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let top_token_id = obj.get("i").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

        let is_knowledge_layer = knowledge_layers
            .map(|(start, end)| layer >= start && layer < end)
            .unwrap_or(false);
        if is_knowledge_layer && top_token_id >= FIRST_CONTENT_TOKEN_ID && top_token_id < vocab_size
        {
            if let Some(gate_tokens) = gate_top_tokens_per_layer.get(&layer) {
                if feat < gate_tokens.len() {
                    let gate_tok = &gate_tokens[feat];
                    if let Some(offset) = compute_offset_direction(
                        gate_tok,
                        top_token_id,
                        weights,
                        tokenizer,
                        hidden_size,
                        vocab_size,
                    ) {
                        cluster_directions.extend_from_slice(&offset);
                        cluster_features.push((layer, feat));
                        let all_tokens: Vec<String> = obj
                            .get("k")
                            .and_then(|v| v.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|e| {
                                        e.get("t").and_then(|t| t.as_str()).map(|s| s.to_string())
                                    })
                                    .collect()
                            })
                            .unwrap_or_default();
                        cluster_top_tokens.push(all_tokens.join("|"));
                        let out_str = obj
                            .get("t")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        cluster_input_tokens.push(gate_tok.clone());
                        cluster_output_tokens.push(out_str);
                    }
                }
            }
        }
        count += 1;
        if count.is_multiple_of(50000) {
            eprint!("\r  Read {} features...", count);
        }
    }
    eprintln!(
        "\r  Read {} features, {} in knowledge layers",
        count,
        cluster_features.len()
    );

    run_clustering_pipeline(
        ClusterData {
            directions: cluster_directions,
            features: cluster_features,
            top_tokens: cluster_top_tokens,
            input_tokens: cluster_input_tokens,
            output_tokens: cluster_output_tokens,
        },
        hidden_size,
        weights,
        tokenizer,
        output_dir,
        callbacks,
    )?;

    callbacks.on_stage(STAGE_TOKENIZER);
    let tokenizer_json = tokenizer
        .to_string(true)
        .map_err(|e| VindexError::Parse(format!("tokenizer serialize: {e}")))?;
    std::fs::write(output_dir.join(TOKENIZER_JSON), tokenizer_json)?;
    callbacks.on_stage_done(STAGE_TOKENIZER, 0.0);

    let down_top_k = DEFAULT_DOWN_TOP_K;
    let mut config = VindexConfig {
        version: 2,
        model: model_name.to_string(),
        family: family.clone(),
        num_layers,
        hidden_size,
        intermediate_size,
        vocab_size,
        embed_scale,
        layers: layer_infos,
        down_top_k,
        has_model_weights: output_dir.join(MODEL_WEIGHTS_BIN).exists(),
        source: Some(crate::VindexSource {
            huggingface_repo: Some(model_name.to_string()),
            huggingface_revision: None,
            safetensors_sha256: None,
            extracted_at: chrono_now(),
            larql_version: env!("CARGO_PKG_VERSION").to_string(),
        }),
        checksums: None,
        extract_level: crate::ExtractLevel::Browse,
        dtype: crate::config::dtype::StorageDtype::F32,
        quant: crate::QuantFormat::None,
        layer_bands: crate::LayerBands::for_family(&family, num_layers),
        model_config: {
            let cfg = weights.arch.config();
            Some(VindexModelConfig {
                model_type: cfg.model_type.clone(),
                head_dim: weights.head_dim,
                num_q_heads: weights.num_q_heads,
                num_kv_heads: weights.num_kv_heads,
                rope_base: weights.rope_base,
                sliding_window: cfg.sliding_window,
                moe: if weights.arch.is_moe() {
                    Some(crate::MoeConfig {
                        num_experts: weights.arch.num_experts(),
                        top_k: weights.arch.num_experts_per_token(),
                        shared_expert: weights.arch.num_shared_experts() > 0,
                        router_type: weights.arch.moe_router_type().to_string(),
                        moe_intermediate_size: if weights.arch.moe_intermediate_size() > 0 {
                            Some(weights.arch.moe_intermediate_size())
                        } else {
                            None
                        },
                        hybrid: weights.arch.is_hybrid_moe(),
                    })
                } else {
                    None
                },
                global_head_dim: cfg.global_head_dim,
                num_global_kv_heads: cfg.num_global_kv_heads,
                partial_rotary_factor: cfg.partial_rotary_factor,
                sliding_window_pattern: cfg.sliding_window_pattern,
                layer_types: cfg.layer_types.clone(),
                attention_k_eq_v: cfg.attention_k_eq_v,
                num_kv_shared_layers: cfg.num_kv_shared_layers,
                per_layer_embed_dim: cfg.per_layer_embed_dim,
                rope_local_base: cfg.rope_local_base,
                query_pre_attn_scalar: cfg.query_pre_attn_scalar,
                final_logit_softcapping: cfg.final_logit_softcapping,
            })
        },
        fp4: None,
        ffn_layout: None,
    };

    config.checksums = crate::format::checksums::compute_checksums(output_dir).ok();

    let config_json =
        serde_json::to_string_pretty(&config).map_err(|e| VindexError::Parse(e.to_string()))?;
    std::fs::write(output_dir.join(INDEX_JSON), config_json)?;

    Ok(())
}
