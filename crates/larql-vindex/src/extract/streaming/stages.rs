//! Stage methods for `StreamingContext`. Each method here corresponds
//! to a labelled stage in the streaming-extract pipeline.

use std::io::{BufWriter, Write};

use ndarray::Array2;

use crate::config::dtype::write_floats;
use crate::config::types::QuantFormat;
use crate::config::{VindexConfig, VindexLayerInfo, VindexModelConfig};
use crate::error::VindexError;
use crate::extract::constants::FEATURE_PROJECTION_BATCH;
use crate::extract::stage_labels::*;
use crate::format::filenames::*;

use super::context::StreamingContext;
use super::tensor_io::{get_tensor_f32, normalize_key, GateSink};

impl<'a> StreamingContext<'a> {
    /// Stage 1 — gate vectors (streaming, one layer at a time).
    ///
    /// If `drop_gate_vectors` is set we still walk every layer to build
    /// `layer_infos` (num_features per layer is part of `index.json`)
    /// but redirect writes to `/dev/null` (`io::sink`). The gate bytes
    /// are recoverable from `interleaved_q4k.bin` at load time.
    pub(super) fn write_gate_vectors(&mut self) -> Result<(), VindexError> {
        self.callbacks.on_stage(STAGE_GATE_VECTORS);
        let gate_path = self.output_dir.join(GATE_VECTORS_BIN);

        // Auto-resume: if a prior run finished the gate phase and saved
        // `gate_layer_infos`, reuse it and skip the gate loop entirely.
        let resumed_gate = self
            .checkpoint
            .is_complete(crate::extract::checkpoint::ExtractPhase::Gate)
            && self.checkpoint.gate_layer_infos.is_some();
        self.layer_infos = if resumed_gate {
            eprintln!(
                "  Skipping gate phase ({} layer infos restored from checkpoint; \
                 reusing existing {})",
                self.checkpoint
                    .gate_layer_infos
                    .as_ref()
                    .map(|v| v.len())
                    .unwrap_or(0),
                GATE_VECTORS_BIN,
            );
            self.callbacks.on_stage_done(STAGE_GATE_VECTORS, 0.0);
            self.checkpoint.gate_layer_infos.clone().unwrap_or_default()
        } else {
            Vec::new()
        };

        // Only allocate the writer + run the loop when the phase isn't
        // already done.
        let mut gate_file: GateSink = if resumed_gate || self.drop_gate_vectors {
            GateSink::Discard(std::io::sink())
        } else {
            GateSink::File(BufWriter::new(std::fs::File::create(&gate_path)?))
        };
        let mut offset: u64 = 0;
        let prefixes: Vec<&str> = self.prefixes.iter().map(|s| s.as_str()).collect();

        // Skip the per-layer gate loop entirely on resume.
        let layer_count_for_loop = if resumed_gate { 0 } else { self.num_layers };
        for layer in 0..layer_count_for_loop {
            self.callbacks
                .on_layer_start(COMP_GATE, layer, self.num_layers);
            let start = std::time::Instant::now();

            if self.expert_format == larql_models::ExpertFormat::PackedMxfp4 {
                // MXFP4 packed experts: dequantize gate_up_proj_blocks per layer
                // The fused tensor is [num_experts, 2*intermediate, groups, 16]
                // First half of output features = gate, second half = up
                let blocks_key = self
                    .arch
                    .packed_gate_up_blocks_key(layer)
                    .unwrap_or_default();
                let scales_key = self
                    .arch
                    .packed_gate_up_scales_key(layer)
                    .unwrap_or_default();

                if let (Some(blocks_info), Some(scales_info)) = (
                    self.tensor_index.get(&blocks_key),
                    self.tensor_index.get(&scales_key),
                ) {
                    let blocks_st = safetensors::SafeTensors::deserialize(
                        &self.shard_mmaps[blocks_info.0].mmap,
                    )
                    .map_err(|e| VindexError::Parse(e.to_string()))?;
                    let scales_st = safetensors::SafeTensors::deserialize(
                        &self.shard_mmaps[scales_info.0].mmap,
                    )
                    .map_err(|e| VindexError::Parse(e.to_string()))?;

                    let blocks_view = blocks_st
                        .tensor(&blocks_info.1)
                        .map_err(|e| VindexError::Parse(e.to_string()))?;
                    let scales_view = scales_st
                        .tensor(&scales_info.1)
                        .map_err(|e| VindexError::Parse(e.to_string()))?;

                    let shape = blocks_view.shape();
                    let n_exp = shape[0];
                    let out_features = shape[1]; // 2 * intermediate (fused gate+up)
                    let groups = shape[2];
                    let in_features = groups * 32;
                    let half = out_features / 2; // gate portion

                    let experts = crate::format::quant::mxfp4::dequantize_all_experts(
                        blocks_view.data(),
                        scales_view.data(),
                        n_exp,
                        out_features,
                        groups,
                    )?;

                    let mut total_features = 0usize;
                    let mut layer_bytes = 0u64;

                    for expert_data in &experts {
                        // Extract gate portion (first half rows)
                        let gate_data = &expert_data[..half * in_features];
                        layer_bytes += write_floats(&mut gate_file, gate_data, self.dtype)?;
                        total_features += half;
                    }

                    if total_features > 0 {
                        self.layer_infos.push(VindexLayerInfo {
                            layer,
                            num_features: total_features,
                            offset,
                            length: layer_bytes,
                            num_experts: Some(n_exp),
                            num_features_per_expert: Some(half),
                        });
                        offset += layer_bytes;
                    }
                }
            } else if self.expert_format == larql_models::ExpertFormat::PackedBF16 && self.is_moe {
                // Hybrid MoE (Gemma 4 26B A4B): packed experts stored separately.
                // gate_vectors.bin uses the dense FFN gate for KNN walk routing.
                let gate_key = normalize_key(&self.arch.ffn_gate_key(layer), &prefixes);
                if let Some(tensor) =
                    get_tensor_f32(&self.shard_mmaps, &self.tensor_index, &gate_key)?
                {
                    let num_features = tensor.shape()[0];
                    let data = tensor.as_slice().unwrap();
                    let length = write_floats(&mut gate_file, data, self.dtype)?;
                    self.layer_infos.push(VindexLayerInfo {
                        layer,
                        num_features,
                        offset,
                        length,
                        num_experts: None,
                        num_features_per_expert: None,
                    });
                    offset += length;
                }
            } else if self.is_moe && self.n_experts > 0 {
                // Standard MoE (Mixtral): per-expert gate tensors
                let mut total_features = 0usize;
                let mut layer_bytes = 0u64;
                let mut features_per_expert = 0usize;

                for expert in 0..self.n_experts {
                    let gate_key = match self.arch.expert_ffn_gate_key(layer, expert) {
                        Some(k) => normalize_key(&k, &prefixes),
                        None => continue,
                    };

                    if let Some(tensor) =
                        get_tensor_f32(&self.shard_mmaps, &self.tensor_index, &gate_key)?
                    {
                        features_per_expert = tensor.shape()[0];
                        total_features += features_per_expert;
                        let data = tensor.as_slice().unwrap();
                        layer_bytes += write_floats(&mut gate_file, data, self.dtype)?;
                    }
                }

                if total_features > 0 {
                    self.layer_infos.push(VindexLayerInfo {
                        layer,
                        num_features: total_features,
                        offset,
                        length: layer_bytes,
                        num_experts: Some(self.n_experts),
                        num_features_per_expert: Some(features_per_expert),
                    });
                    offset += layer_bytes;
                }
            } else {
                // Dense: single gate matrix per layer
                let gate_key = normalize_key(&self.arch.ffn_gate_key(layer), &prefixes);
                if let Some(tensor) =
                    get_tensor_f32(&self.shard_mmaps, &self.tensor_index, &gate_key)?
                {
                    let num_features = tensor.shape()[0];
                    let data = tensor.as_slice().unwrap();
                    let length = write_floats(&mut gate_file, data, self.dtype)?;
                    self.layer_infos.push(VindexLayerInfo {
                        layer,
                        num_features,
                        offset,
                        length,
                        num_experts: None,
                        num_features_per_expert: None,
                    });
                    offset += length;
                }
            }

            self.callbacks
                .on_layer_done(COMP_GATE, layer, start.elapsed().as_secs_f64() * 1000.0);
        }
        gate_file.flush()?;
        // If we were only sinking bytes, don't leave a zero-byte
        // gate_vectors.bin behind for the loader to trip over.
        drop(gate_file);
        if self.drop_gate_vectors && gate_path.exists() && !resumed_gate {
            let _ = std::fs::remove_file(&gate_path);
        }
        if !resumed_gate {
            self.callbacks.on_stage_done(STAGE_GATE_VECTORS, 0.0);
            self.checkpoint
                .mark_gate_complete(self.layer_infos.clone(), self.output_dir)?;
        }
        Ok(())
    }

    /// Stage 1b — router weights (MoE models only).
    pub(super) fn write_router_weights(&mut self) -> Result<(), VindexError> {
        if !self.is_moe {
            return Ok(());
        }
        self.callbacks.on_stage(STAGE_ROUTER_WEIGHTS);
        let router_path = self.output_dir.join(ROUTER_WEIGHTS_BIN);
        let mut router_file = BufWriter::new(std::fs::File::create(&router_path)?);
        let prefixes: Vec<&str> = self.prefixes.iter().map(|s| s.as_str()).collect();

        for layer in 0..self.num_layers {
            let router_key = self
                .arch
                .moe_router_key(layer)
                .map(|k| normalize_key(&k, &prefixes))
                .unwrap_or_default();

            if let Some(tensor) =
                get_tensor_f32(&self.shard_mmaps, &self.tensor_index, &router_key)?
            {
                let data = tensor.as_slice().unwrap();
                let bytes = crate::config::dtype::encode_floats(data, self.dtype);
                router_file.write_all(&bytes)?;
            }

            // Also try router bias
            let bias_key = router_key.replace(".weight", ".bias");
            if let Some(tensor) = get_tensor_f32(&self.shard_mmaps, &self.tensor_index, &bias_key)?
            {
                let data = tensor.as_slice().unwrap();
                let bytes = crate::config::dtype::encode_floats(data, self.dtype);
                // Write bias after weight for each layer
                router_file.write_all(&bytes)?;
            }
        }
        router_file.flush()?;
        self.callbacks.on_stage_done(STAGE_ROUTER_WEIGHTS, 0.0);
        Ok(())
    }

    /// Stage 2 — embeddings.
    pub(super) fn write_embeddings(&mut self) -> Result<(), VindexError> {
        self.callbacks.on_stage(STAGE_EMBEDDINGS);
        let prefixes: Vec<&str> = self.prefixes.iter().map(|s| s.as_str()).collect();
        let embed_key = normalize_key(self.arch.embed_key(), &prefixes);
        let embed = get_tensor_f32(&self.shard_mmaps, &self.tensor_index, &embed_key)?
            .ok_or_else(|| VindexError::MissingTensor(embed_key.clone()))?;
        self.vocab_size = embed.shape()[0];
        let embed_data = embed.as_slice().unwrap();
        let embed_bytes = crate::config::dtype::encode_floats(embed_data, self.dtype);
        std::fs::write(self.output_dir.join(EMBEDDINGS_BIN), &embed_bytes)?;
        self.embed = Some(embed);
        self.callbacks.on_stage_done(STAGE_EMBEDDINGS, 0.0);
        Ok(())
    }

    /// Stage 3 — down meta (streaming).
    ///
    /// Auto-resume: skip the entire down-meta phase if the prior run
    /// already wrote `down_meta.bin`. The file is opaque to us here
    /// (we don't reload it), but the loader at the end uses it
    /// directly off disk via `mmap`, and the config-write doesn't
    /// need any per-layer state from this phase — so a clean skip is
    /// safe.
    pub(super) fn write_down_meta(&mut self) -> Result<(), VindexError> {
        let resumed_down = self
            .checkpoint
            .is_complete(crate::extract::checkpoint::ExtractPhase::DownMeta);
        self.callbacks.on_stage(STAGE_DOWN_META);
        if resumed_down {
            eprintln!(
                "  Skipping down_meta phase (reusing existing {})",
                DOWN_META_BIN,
            );
        }
        let mut all_down_meta: Vec<Option<Vec<Option<crate::FeatureMeta>>>> =
            vec![None; self.num_layers];

        let embed = self
            .embed
            .as_ref()
            .expect("embeddings stage must run before down_meta stage");

        // Build whole-word vocab once
        let (_ww_ids, _ww_embed) = crate::extract::build_helpers::build_whole_word_vocab(
            self.tokenizer,
            embed,
            self.vocab_size,
            self.hidden_size,
        );

        let prefixes: Vec<&str> = self.prefixes.iter().map(|s| s.as_str()).collect();
        let down_layer_count = if resumed_down { 0 } else { self.num_layers };
        for (layer, layer_down_meta) in all_down_meta.iter_mut().enumerate().take(down_layer_count)
        {
            self.callbacks
                .on_layer_start(COMP_DOWN, layer, self.num_layers);
            let start = std::time::Instant::now();

            // Get down matrices for this layer
            let down_matrices: Vec<Array2<f32>> = if self.expert_format
                == larql_models::ExpertFormat::PackedMxfp4
            {
                // MXFP4: dequantize down_proj_blocks
                let blocks_key = self.arch.packed_down_blocks_key(layer).unwrap_or_default();
                let scales_key = self.arch.packed_down_scales_key(layer).unwrap_or_default();
                if let (Some(bi), Some(si)) = (
                    self.tensor_index.get(&blocks_key),
                    self.tensor_index.get(&scales_key),
                ) {
                    let bst = safetensors::SafeTensors::deserialize(&self.shard_mmaps[bi.0].mmap)
                        .map_err(|e| VindexError::Parse(e.to_string()))?;
                    let sst = safetensors::SafeTensors::deserialize(&self.shard_mmaps[si.0].mmap)
                        .map_err(|e| VindexError::Parse(e.to_string()))?;
                    let bv = bst
                        .tensor(&bi.1)
                        .map_err(|e| VindexError::Parse(e.to_string()))?;
                    let sv = sst
                        .tensor(&si.1)
                        .map_err(|e| VindexError::Parse(e.to_string()))?;
                    let shape = bv.shape();
                    let n_exp = shape[0];
                    let out_features = shape[1];
                    let groups = shape[2];
                    let in_features = groups * 32;
                    let experts = crate::format::quant::mxfp4::dequantize_all_experts(
                        bv.data(),
                        sv.data(),
                        n_exp,
                        out_features,
                        groups,
                    )?;
                    experts
                        .into_iter()
                        .map(|data| {
                            Array2::from_shape_vec((out_features, in_features), data).unwrap()
                        })
                        .collect()
                } else {
                    self.callbacks.on_layer_done(COMP_DOWN, layer, 0.0);
                    continue;
                }
            } else if self.expert_format == larql_models::ExpertFormat::PackedBF16 && self.is_moe {
                // Hybrid MoE (Gemma 4 26B A4B): use dense FFN down for down_meta.
                // Expert down matrices live per-layer at `layers/layer_{L:02}.weights`
                // (Q4_K), written by the q4k weight writer.
                let down_key = normalize_key(&self.arch.ffn_down_key(layer), &prefixes);
                match get_tensor_f32(&self.shard_mmaps, &self.tensor_index, &down_key)? {
                    Some(t) => vec![t],
                    None => {
                        self.callbacks.on_layer_done(COMP_DOWN, layer, 0.0);
                        continue;
                    }
                }
            } else if self.is_moe && self.n_experts > 0 {
                let mut mats = Vec::new();
                for expert in 0..self.n_experts {
                    if let Some(key) = self.arch.expert_ffn_down_key(layer, expert) {
                        let nk = normalize_key(&key, &prefixes);
                        if let Some(t) = get_tensor_f32(&self.shard_mmaps, &self.tensor_index, &nk)?
                        {
                            mats.push(t);
                        }
                    }
                }
                mats
            } else {
                let down_key = normalize_key(&self.arch.ffn_down_key(layer), &prefixes);
                match get_tensor_f32(&self.shard_mmaps, &self.tensor_index, &down_key)? {
                    Some(t) => vec![t],
                    None => {
                        self.callbacks.on_layer_done(COMP_DOWN, layer, 0.0);
                        continue;
                    }
                }
            };

            if down_matrices.is_empty() {
                self.callbacks.on_layer_done(COMP_DOWN, layer, 0.0);
                continue;
            }

            let mut feature_offset = 0usize;
            for w_down in &down_matrices {
                let num_features = w_down.shape()[1];
                let batch_size = FEATURE_PROJECTION_BATCH;

                for batch_start in (0..num_features).step_by(batch_size) {
                    let batch_end = (batch_start + batch_size).min(num_features);
                    self.callbacks.on_feature_progress(
                        "down",
                        layer,
                        feature_offset + batch_start,
                        down_matrices.iter().map(|m| m.shape()[1]).sum(),
                    );

                    let w_chunk = w_down
                        .slice(ndarray::s![.., batch_start..batch_end])
                        .to_owned();
                    let cpu = larql_compute::CpuBackend;
                    use larql_compute::MatMul;
                    let chunk_logits = cpu.matmul(embed.view(), w_chunk.view());

                    for feat in batch_start..batch_end {
                        let col = chunk_logits.column(feat - batch_start);
                        let mut scores: Vec<(usize, f32)> =
                            col.iter().copied().enumerate().collect();
                        let k = self.down_top_k.min(scores.len());
                        if k > 0 && k < scores.len() {
                            scores.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
                        }
                        scores.truncate(k);
                        scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                        let top_k_entries: Vec<larql_models::TopKEntry> = scores
                            .into_iter()
                            .filter_map(|(idx, logit)| {
                                self.tokenizer
                                    .decode(&[idx as u32], true)
                                    .ok()
                                    .map(|s| s.trim().to_string())
                                    .filter(|s| !s.is_empty())
                                    .map(|token| larql_models::TopKEntry {
                                        token,
                                        token_id: idx as u32,
                                        logit,
                                    })
                            })
                            .collect();

                        let (top_token, top_token_id, c_score) =
                            if let Some(first) = top_k_entries.first() {
                                (first.token.clone(), first.token_id, first.logit)
                            } else {
                                (String::new(), 0, 0.0)
                            };

                        let feat_idx = feature_offset + feat;
                        if layer_down_meta.is_none() {
                            *layer_down_meta = Some(Vec::new());
                        }
                        if let Some(ref mut metas) = layer_down_meta {
                            while metas.len() <= feat_idx {
                                metas.push(None);
                            }
                            metas[feat_idx] = Some(crate::FeatureMeta {
                                top_token,
                                top_token_id,
                                c_score,
                                top_k: top_k_entries,
                            });
                        }
                    }
                }
                feature_offset += num_features;
            }

            self.callbacks
                .on_layer_done(COMP_DOWN, layer, start.elapsed().as_secs_f64() * 1000.0);
        }

        if !resumed_down {
            crate::format::down_meta::write_binary(
                self.output_dir,
                &all_down_meta,
                self.down_top_k,
            )?;
            self.callbacks.on_stage_done(STAGE_DOWN_META, 0.0);
            self.checkpoint.mark(
                crate::extract::checkpoint::ExtractPhase::DownMeta,
                self.output_dir,
            )?;
        }
        Ok(())
    }

    /// Stage 4 — tokenizer.
    pub(super) fn write_tokenizer(&mut self) -> Result<(), VindexError> {
        self.callbacks.on_stage(STAGE_TOKENIZER);
        let tokenizer_json = self
            .tokenizer
            .to_string(true)
            .map_err(|e| VindexError::Parse(format!("tokenizer serialize: {e}")))?;
        std::fs::write(self.output_dir.join(TOKENIZER_JSON), tokenizer_json)?;
        self.callbacks.on_stage_done(STAGE_TOKENIZER, 0.0);
        Ok(())
    }

    /// Stage 5 — assemble + write `index.json` (preliminary; checksums
    /// added later in `finalize`).
    pub(super) fn write_index_json(&mut self) -> Result<(), VindexError> {
        let cfg = self.arch.config();
        let family = self.arch.family().to_string();
        let layer_infos = std::mem::take(&mut self.layer_infos);
        let config = VindexConfig {
            version: 2,
            model: self.model_name.to_string(),
            family: family.clone(),
            num_layers: self.num_layers,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            embed_scale: self.embed_scale,
            layers: layer_infos,
            down_top_k: self.down_top_k,
            has_model_weights: false,
            source: Some(crate::VindexSource {
                huggingface_repo: Some(self.model_name.to_string()),
                huggingface_revision: None,
                safetensors_sha256: None,
                extracted_at: crate::extract::build_helpers::chrono_now(),
                larql_version: env!("CARGO_PKG_VERSION").to_string(),
            }),
            checksums: None,
            extract_level: self.extract_level,
            dtype: self.dtype,
            quant: self.quant,
            layer_bands: crate::LayerBands::for_family(&family, self.num_layers),
            model_config: Some(VindexModelConfig {
                model_type: cfg.model_type.clone(),
                head_dim: cfg.head_dim,
                num_q_heads: cfg.num_q_heads,
                num_kv_heads: cfg.num_kv_heads,
                rope_base: cfg.rope_base,
                sliding_window: cfg.sliding_window,
                moe: if self.is_moe {
                    Some(crate::MoeConfig {
                        num_experts: self.n_experts,
                        top_k: self.arch.num_experts_per_token(),
                        shared_expert: self.arch.num_shared_experts() > 0,
                        router_type: self.arch.moe_router_type().to_string(),
                        moe_intermediate_size: if self.arch.moe_intermediate_size() > 0 {
                            Some(self.arch.moe_intermediate_size())
                        } else {
                            None
                        },
                        hybrid: self.arch.is_hybrid_moe(),
                    })
                } else {
                    None
                },
                // Per-layer geometry (Gemma 4)
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
            }),
            fp4: None,
            ffn_layout: None,
        };

        // Write preliminary index.json (needed by write_model_weights which reads dtype from it).
        let config_json =
            serde_json::to_string_pretty(&config).map_err(|e| VindexError::Parse(e.to_string()))?;
        std::fs::write(self.output_dir.join(INDEX_JSON), config_json)?;
        Ok(())
    }

    /// Stage 6 — model weights (if extract level requires them).
    ///
    /// With quant=q4k we always materialise weights regardless of the
    /// declared level — the Q4_K writer emits all of attn, FFN, norms,
    /// lm_head in one pass and makes `--level browse --quant q4k`
    /// incoherent, so q4k implicitly promotes to "all".
    pub(super) fn maybe_write_model_weights(&mut self) -> Result<(), VindexError> {
        let needs_weights = self.extract_level.writes_attn() || self.quant != QuantFormat::None;
        if !needs_weights {
            return Ok(());
        }
        let shard_refs: Vec<&[u8]> = self.shard_mmaps.iter().map(|s| s.mmap.as_ref()).collect();
        let streaming_source = crate::format::weights::StreamingWeights {
            shard_mmaps: &shard_refs,
            tensor_index: &self.tensor_index,
            arch: &*self.arch,
            num_layers: self.num_layers,
        };
        // Thread the extract level into the write options so the
        // writer can skip attn/FFN/lm_head sections per tier.
        let mut level_opts = self.weight_opts;
        level_opts.level = self.extract_level;
        match self.quant {
            QuantFormat::None => {
                crate::format::weights::write_model_weights_with_opts(
                    &streaming_source,
                    self.output_dir,
                    self.callbacks,
                    level_opts,
                )?;
            }
            QuantFormat::Q4K => {
                // Q4K doesn't write `up_weights.bin` / `down_weights.bin`
                // at all — the FFN weights live in `interleaved_q4k.bin`.
                // `ffn_compact` is a no-op here by construction. Level
                // gating for Q4K is a future refinement (today Q4K
                // always writes the full set).
                crate::format::weights::write_model_weights_q4k_with_opts(
                    &streaming_source,
                    self.output_dir,
                    self.callbacks,
                    self.q4k_opts,
                )?;
            }
        }
        Ok(())
    }
}
