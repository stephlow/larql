use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use larql_inference::attention::{run_attention_block_with_pre_o, SharedKV};
use larql_inference::forward::ple::precompute_per_layer_inputs;
use larql_inference::forward::{
    dot_proj, embed_tokens_pub, run_layer_with_ffn, run_layer_with_replaced_head_residual_delta,
    run_layer_with_replaced_pre_o_head, run_layer_with_subtracted_pre_o_heads,
};
use larql_inference::{encode_prompt, hidden_to_raw_logits, WeightFfn};
use larql_vindex::{
    load_model_weights_q4k, load_vindex_tokenizer, SilentLoadCallbacks, VectorIndex,
};
use ndarray::{s, Array2};

use super::input::{load_prompts, parse_head_spec};
use super::metrics::{kl_logp, log_softmax, max_abs_diff, mean};
use super::reports::{SanityCheckReport, SanityHeadReport, SanityPromptReport};
use super::runtime::{insert_q4k_layer_tensors, remove_layer_tensors};
use super::types::HeadId;
use super::zero_ablate::forward_q4k_zero_pre_o_head;

#[derive(Args)]
pub(super) struct SanityCheckArgs {
    /// Self-contained Q4K vindex directory.
    #[arg(long)]
    index: PathBuf,

    /// JSONL prompt file. Each line must include at least {"prompt": "..."}.
    #[arg(long)]
    prompts: PathBuf,

    /// Output directory.
    #[arg(long)]
    out: PathBuf,

    /// Explicit heads as layer:head comma list, e.g. 0:4,0:6.
    #[arg(long)]
    heads: String,

    /// Limit prompts for bounded sanity runs.
    #[arg(long)]
    max_prompts: Option<usize>,
}

#[derive(Debug)]
struct SanityHeadAccumulator {
    prompts: Vec<SanityPromptReport>,
}

impl SanityHeadAccumulator {
    fn new() -> Self {
        Self {
            prompts: Vec::new(),
        }
    }

    fn add(&mut self, prompt: SanityPromptReport) {
        self.prompts.push(prompt);
    }

    fn finish(self, head: HeadId) -> SanityHeadReport {
        let noop_kls: Vec<f64> = self.prompts.iter().map(|p| p.noop_kl).collect();
        let residual_delta_noop_kls: Vec<f64> = self
            .prompts
            .iter()
            .map(|p| p.residual_delta_noop_kl)
            .collect();
        let zero_subtract_kls: Vec<f64> = self.prompts.iter().map(|p| p.zero_subtract_kl).collect();
        SanityHeadReport {
            layer: head.layer,
            head: head.head,
            prompts: self.prompts.len(),
            noop_mean_kl: mean(&noop_kls),
            noop_max_kl: noop_kls.iter().copied().fold(0.0, f64::max),
            noop_max_abs_logit_diff: self
                .prompts
                .iter()
                .map(|p| p.noop_max_abs_logit_diff)
                .fold(0.0, f64::max),
            residual_delta_noop_mean_kl: mean(&residual_delta_noop_kls),
            residual_delta_noop_max_kl: residual_delta_noop_kls.iter().copied().fold(0.0, f64::max),
            residual_delta_noop_max_abs_logit_diff: self
                .prompts
                .iter()
                .map(|p| p.residual_delta_noop_max_abs_logit_diff)
                .fold(0.0, f64::max),
            zero_subtract_mean_kl: mean(&zero_subtract_kls),
            zero_subtract_max_kl: zero_subtract_kls.iter().copied().fold(0.0, f64::max),
            zero_subtract_max_abs_logit_diff: self
                .prompts
                .iter()
                .map(|p| p.zero_subtract_max_abs_logit_diff)
                .fold(0.0, f64::max),
            per_prompt: self.prompts,
        }
    }
}

pub(super) fn run_sanity_check(args: SanityCheckArgs) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(&args.out)?;

    eprintln!("Loading vindex: {}", args.index.display());
    let start = Instant::now();
    let mut cb = SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex(&args.index, &mut cb)?;
    index.load_attn_q4k(&args.index)?;
    index.load_interleaved_q4k(&args.index)?;
    let mut weights = load_model_weights_q4k(&args.index, &mut cb)?;
    let tokenizer = load_vindex_tokenizer(&args.index)?;
    if weights.arch.is_hybrid_moe() {
        return Err("ov-rd sanity-check currently supports dense FFN vindexes only".into());
    }
    eprintln!(
        "  {} layers, hidden_size={}, q_heads={}, head_dim={} ({:.1}s)",
        weights.num_layers,
        weights.hidden_size,
        weights.num_q_heads,
        weights.head_dim,
        start.elapsed().as_secs_f64()
    );

    let selected_heads = parse_head_spec(&args.heads)?;
    if selected_heads.is_empty() {
        return Err("no heads selected for sanity check".into());
    }
    let prompts = load_prompts(&args.prompts, args.max_prompts)?;
    eprintln!("Selected heads: {:?}", selected_heads);
    eprintln!("Prompts: {}", prompts.len());

    let mut accumulators: Vec<SanityHeadAccumulator> = selected_heads
        .iter()
        .map(|_| SanityHeadAccumulator::new())
        .collect();

    for (prompt_idx, record) in prompts.iter().enumerate() {
        let label = record
            .id
            .as_deref()
            .or(record.stratum.as_deref())
            .unwrap_or("prompt");
        eprintln!("  [{}/{}] {}", prompt_idx + 1, prompts.len(), label);

        let token_ids = encode_prompt(&tokenizer, &*weights.arch, &record.prompt)?;
        if token_ids.is_empty() {
            continue;
        }
        let stratum = record.stratum.as_deref().unwrap_or("unknown");

        let baseline_hidden =
            larql_inference::vindex::predict_q4k_hidden(&mut weights, &token_ids, &index, None);
        let baseline_logits = final_logits(&weights, &baseline_hidden);
        let baseline_logp = log_softmax(&baseline_logits);

        for (idx, head) in selected_heads.iter().copied().enumerate() {
            let noop_hidden =
                forward_q4k_noop_replace_pre_o_head(&mut weights, &token_ids, &index, head)?;
            let noop_logits = final_logits(&weights, &noop_hidden);
            let noop_logp = log_softmax(&noop_logits);

            let residual_delta_noop_hidden = forward_q4k_noop_replace_head_residual_delta(
                &mut weights,
                &token_ids,
                &index,
                head,
            )?;
            let residual_delta_noop_logits = final_logits(&weights, &residual_delta_noop_hidden);
            let residual_delta_noop_logp = log_softmax(&residual_delta_noop_logits);

            let zero_hidden = forward_q4k_zero_pre_o_head(&mut weights, &token_ids, &index, head)?;
            let zero_logits = final_logits(&weights, &zero_hidden);
            let zero_logp = log_softmax(&zero_logits);

            let subtract_hidden =
                forward_q4k_subtract_pre_o_head(&mut weights, &token_ids, &index, head)?;
            let subtract_logits = final_logits(&weights, &subtract_hidden);
            let subtract_logp = log_softmax(&subtract_logits);

            accumulators[idx].add(SanityPromptReport {
                id: label.to_string(),
                stratum: stratum.to_string(),
                noop_kl: kl_logp(&baseline_logp, &noop_logp),
                noop_max_abs_logit_diff: max_abs_diff(&baseline_logits, &noop_logits),
                residual_delta_noop_kl: kl_logp(&baseline_logp, &residual_delta_noop_logp),
                residual_delta_noop_max_abs_logit_diff: max_abs_diff(
                    &baseline_logits,
                    &residual_delta_noop_logits,
                ),
                zero_subtract_kl: kl_logp(&zero_logp, &subtract_logp),
                zero_subtract_max_abs_logit_diff: max_abs_diff(&zero_logits, &subtract_logits),
            });
        }
    }

    let heads = selected_heads
        .iter()
        .copied()
        .zip(accumulators)
        .map(|(head, acc)| acc.finish(head))
        .collect();
    let report = SanityCheckReport {
        index: args.index.display().to_string(),
        prompt_file: args.prompts.display().to_string(),
        prompts_seen: prompts.len(),
        selected_heads,
        heads,
    };

    let out_path = args.out.join("sanity_check.json");
    let file = std::fs::File::create(&out_path)?;
    serde_json::to_writer_pretty(file, &report)?;
    eprintln!("Wrote {}", out_path.display());

    Ok(())
}

fn forward_q4k_noop_replace_pre_o_head(
    weights: &mut larql_inference::ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    head: HeadId,
) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    let mut h = embed_tokens_pub(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
    let mut kv_cache: HashMap<usize, SharedKV> = HashMap::new();

    for layer in 0..weights.num_layers {
        let inserted = insert_q4k_layer_tensors(weights, index, layer)?;
        let step = {
            let shared_kv = weights
                .arch
                .kv_shared_source_layer(layer)
                .and_then(|src| kv_cache.get(&src));
            let ffn = WeightFfn { weights };
            if layer == head.layer {
                let (_, pre_o) = run_attention_block_with_pre_o(weights, &h, layer)
                    .ok_or_else(|| format!("pre-W_O capture failed at layer {layer}"))?;
                let head_dim = weights.arch.head_dim_for_layer(layer);
                let start = head.head * head_dim;
                let end = start + head_dim;
                let replacement = pre_o.slice(s![.., start..end]).to_owned();
                run_layer_with_replaced_pre_o_head(
                    weights,
                    &h,
                    layer,
                    &ffn,
                    head.head,
                    &replacement,
                    ple_inputs.get(layer),
                    shared_kv,
                )
            } else {
                run_layer_with_ffn(
                    weights,
                    &h,
                    layer,
                    &ffn,
                    false,
                    ple_inputs.get(layer),
                    shared_kv,
                )
                .map(|(h_new, _, kv_out)| (h_new, kv_out))
            }
        };

        if let Some((h_new, kv_out)) = step {
            h = h_new;
            if let Some(kv) = kv_out {
                kv_cache.insert(layer, kv);
            }
        } else {
            remove_layer_tensors(weights, inserted);
            return Err(format!(
                "forward failed at layer {layer} during no-op replacement L{} H{}",
                head.layer, head.head
            )
            .into());
        }

        remove_layer_tensors(weights, inserted);
    }

    Ok(h)
}

fn forward_q4k_subtract_pre_o_head(
    weights: &mut larql_inference::ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    head: HeadId,
) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    let mut h = embed_tokens_pub(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
    let mut kv_cache: HashMap<usize, SharedKV> = HashMap::new();

    for layer in 0..weights.num_layers {
        let inserted = insert_q4k_layer_tensors(weights, index, layer)?;
        let step = {
            let shared_kv = weights
                .arch
                .kv_shared_source_layer(layer)
                .and_then(|src| kv_cache.get(&src));
            let ffn = WeightFfn { weights };
            if layer == head.layer {
                run_layer_with_subtracted_pre_o_heads(
                    weights,
                    &h,
                    layer,
                    &ffn,
                    &[head.head],
                    ple_inputs.get(layer),
                    shared_kv,
                )
            } else {
                run_layer_with_ffn(
                    weights,
                    &h,
                    layer,
                    &ffn,
                    false,
                    ple_inputs.get(layer),
                    shared_kv,
                )
                .map(|(h_new, _, kv_out)| (h_new, kv_out))
            }
        };

        if let Some((h_new, kv_out)) = step {
            h = h_new;
            if let Some(kv) = kv_out {
                kv_cache.insert(layer, kv);
            }
        } else {
            remove_layer_tensors(weights, inserted);
            return Err(format!(
                "forward failed at layer {layer} during subtract check L{} H{}",
                head.layer, head.head
            )
            .into());
        }

        remove_layer_tensors(weights, inserted);
    }

    Ok(h)
}

fn forward_q4k_noop_replace_head_residual_delta(
    weights: &mut larql_inference::ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    head: HeadId,
) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    let mut h = embed_tokens_pub(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
    let mut kv_cache: HashMap<usize, SharedKV> = HashMap::new();

    for layer in 0..weights.num_layers {
        let inserted = insert_q4k_layer_tensors(weights, index, layer)?;
        let step = {
            let shared_kv = weights
                .arch
                .kv_shared_source_layer(layer)
                .and_then(|src| kv_cache.get(&src));
            let ffn = WeightFfn { weights };
            if layer == head.layer {
                let (_, pre_o) = run_attention_block_with_pre_o(weights, &h, layer)
                    .ok_or_else(|| format!("pre-W_O capture failed at layer {layer}"))?;
                let head_dim = weights.arch.head_dim_for_layer(layer);
                let start = head.head * head_dim;
                let end = start + head_dim;
                let head_out = pre_o.slice(s![.., start..end]);
                let w_o = weights
                    .tensors
                    .get(&weights.arch.attn_o_key(layer))
                    .ok_or_else(|| format!("missing W_O tensor at layer {layer}"))?;
                let w_o_head = w_o.slice(s![.., start..end]);
                let replacement_delta = dot_proj(&head_out, &w_o_head);
                run_layer_with_replaced_head_residual_delta(
                    weights,
                    &h,
                    layer,
                    &ffn,
                    head.head,
                    &replacement_delta,
                    ple_inputs.get(layer),
                    shared_kv,
                )
            } else {
                run_layer_with_ffn(
                    weights,
                    &h,
                    layer,
                    &ffn,
                    false,
                    ple_inputs.get(layer),
                    shared_kv,
                )
                .map(|(h_new, _, kv_out)| (h_new, kv_out))
            }
        };

        if let Some((h_new, kv_out)) = step {
            h = h_new;
            if let Some(kv) = kv_out {
                kv_cache.insert(layer, kv);
            }
        } else {
            remove_layer_tensors(weights, inserted);
            return Err(format!(
                "forward failed at layer {layer} during residual-delta no-op L{} H{}",
                head.layer, head.head
            )
            .into());
        }

        remove_layer_tensors(weights, inserted);
    }

    Ok(h)
}

fn final_logits(weights: &larql_inference::ModelWeights, h: &Array2<f32>) -> Vec<f32> {
    let last = h.nrows().saturating_sub(1);
    let h_last = h.slice(s![last..last + 1, ..]).to_owned();
    hidden_to_raw_logits(weights, &h_last)
}
