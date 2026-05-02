use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use larql_inference::attention::SharedKV;
use larql_inference::forward::ple::precompute_per_layer_inputs;
use larql_inference::forward::{
    embed_tokens_pub, run_layer_with_ffn, run_layer_with_zeroed_pre_o_heads,
};
use larql_inference::{encode_prompt, hidden_to_raw_logits, WeightFfn};
use larql_vindex::{
    load_model_weights_q4k, load_vindex_tokenizer, SilentLoadCallbacks, VectorIndex,
};
use ndarray::{s, Array2};

use super::input::{load_prompts, parse_head_spec};
use super::metrics::{argmax, bool_rate, kl_logp, log_softmax, mean, percentile, top_k_indices};
use super::reports::{
    CaptureReport, ZeroAblationReport, ZeroHeadReport, ZeroPromptReport, ZeroStratumReport,
};
use super::runtime::{insert_q4k_layer_tensors, remove_layer_tensors};
use super::types::HeadId;

#[derive(Args)]
pub(super) struct ZeroAblateArgs {
    /// Self-contained Q4K vindex directory.
    #[arg(long)]
    index: PathBuf,

    /// JSONL prompt file. Each line must include at least {"prompt": "..."}.
    #[arg(long)]
    prompts: PathBuf,

    /// Output directory.
    #[arg(long)]
    out: PathBuf,

    /// Explicit heads as layer:head comma list, e.g. 11:3,11:0,0:4.
    #[arg(long)]
    heads: Option<String>,

    /// Stage-0 stats JSON. Used with --top-heads when --heads is absent.
    #[arg(long)]
    stage0: Option<PathBuf>,

    /// Number of highest-variance Stage-0 heads to test.
    #[arg(long, default_value_t = 8)]
    top_heads: usize,

    /// Limit prompts for bounded gate runs.
    #[arg(long)]
    max_prompts: Option<usize>,
}

#[derive(Debug)]
struct ZeroHeadAccumulator {
    prompts: Vec<ZeroPromptReport>,
    by_stratum: HashMap<String, Vec<ZeroPromptReport>>,
}

impl ZeroHeadAccumulator {
    fn new() -> Self {
        Self {
            prompts: Vec::new(),
            by_stratum: HashMap::new(),
        }
    }

    fn add(&mut self, prompt: ZeroPromptReport) {
        let stratum = prompt.stratum.clone();
        self.prompts.push(prompt.clone());
        self.by_stratum.entry(stratum).or_default().push(prompt);
    }

    fn finish(self, head: HeadId) -> ZeroHeadReport {
        let prompts_len = self.prompts.len();
        let kl_values: Vec<f64> = self.prompts.iter().map(|p| p.kl).collect();
        let mean_kl = mean(&kl_values);
        let p95_kl = percentile(kl_values.clone(), 0.95);
        let max_kl = kl_values.iter().copied().fold(0.0, f64::max);
        let mean_delta_cross_entropy_bits = mean(
            &self
                .prompts
                .iter()
                .map(|p| p.delta_cross_entropy_bits)
                .collect::<Vec<_>>(),
        );
        let top1_agreement = bool_rate(self.prompts.iter().map(|p| p.top1_agree));
        let top5_contains_baseline_top1 =
            bool_rate(self.prompts.iter().map(|p| p.baseline_top1_in_ablated_top5));
        let mut worst_examples = self.prompts.clone();
        worst_examples.sort_by(|a, b| b.kl.partial_cmp(&a.kl).unwrap_or(std::cmp::Ordering::Equal));
        worst_examples.truncate(10);

        let mut strata: Vec<_> = self
            .by_stratum
            .into_iter()
            .map(|(stratum, prompts)| {
                let values: Vec<f64> = prompts.iter().map(|p| p.kl).collect();
                ZeroStratumReport {
                    stratum,
                    prompts: prompts.len(),
                    mean_kl: mean(&values),
                    max_kl: values.iter().copied().fold(0.0, f64::max),
                    top1_agreement: bool_rate(prompts.iter().map(|p| p.top1_agree)),
                    top5_contains_baseline_top1: bool_rate(
                        prompts.iter().map(|p| p.baseline_top1_in_ablated_top5),
                    ),
                }
            })
            .collect();
        strata.sort_by(|a, b| a.stratum.cmp(&b.stratum));
        ZeroHeadReport {
            layer: head.layer,
            head: head.head,
            ablation_kind: "zero_pre_wo".to_string(),
            patch_location: "before_W_O".to_string(),
            preserved_components: vec![
                "FFN".to_string(),
                "PLE".to_string(),
                "layer_scalar".to_string(),
            ],
            bounded_vocab_size: None,
            prompts: prompts_len,
            mean_kl,
            p95_kl,
            max_kl,
            mean_delta_cross_entropy_bits,
            top1_agreement,
            top5_contains_baseline_top1,
            strata,
            worst_examples,
            per_prompt: self.prompts,
        }
    }
}

pub(super) fn run_zero_ablate(args: ZeroAblateArgs) -> Result<(), Box<dyn std::error::Error>> {
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
        return Err("ov-rd zero-ablate currently supports dense FFN vindexes only".into());
    }
    eprintln!(
        "  {} layers, hidden_size={}, q_heads={}, head_dim={} ({:.1}s)",
        weights.num_layers,
        weights.hidden_size,
        weights.num_q_heads,
        weights.head_dim,
        start.elapsed().as_secs_f64()
    );

    let selected_heads = select_zero_ablation_heads(&args)?;
    if selected_heads.is_empty() {
        return Err("no heads selected for zero-ablation".into());
    }
    eprintln!("Selected heads: {:?}", selected_heads);

    let prompts = load_prompts(&args.prompts, args.max_prompts)?;
    eprintln!("Prompts: {}", prompts.len());

    let mut accumulators: Vec<ZeroHeadAccumulator> = selected_heads
        .iter()
        .map(|_| ZeroHeadAccumulator::new())
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
        let baseline_top1 = argmax(&baseline_logits);

        for (idx, head) in selected_heads.iter().copied().enumerate() {
            let ablated_hidden =
                forward_q4k_zero_pre_o_head(&mut weights, &token_ids, &index, head)?;
            let ablated_logits = final_logits(&weights, &ablated_hidden);
            let ablated_logp = log_softmax(&ablated_logits);
            let kl = kl_logp(&baseline_logp, &ablated_logp);
            let ablated_top1 = argmax(&ablated_logits);
            let ablated_top5 = top_k_indices(&ablated_logits, 5);
            accumulators[idx].add(ZeroPromptReport {
                id: label.to_string(),
                stratum: stratum.to_string(),
                kl,
                delta_cross_entropy_bits: kl / std::f64::consts::LN_2,
                baseline_top1,
                ablated_top1,
                top1_agree: baseline_top1 == ablated_top1,
                baseline_top1_in_ablated_top5: ablated_top5.contains(&baseline_top1),
            });
        }
    }

    let head_reports = selected_heads
        .iter()
        .copied()
        .zip(accumulators)
        .map(|(head, acc)| acc.finish(head))
        .collect();

    let report = ZeroAblationReport {
        index: args.index.display().to_string(),
        prompt_file: args.prompts.display().to_string(),
        prompts_seen: prompts.len(),
        selected_heads,
        heads: head_reports,
    };

    let out_path = args.out.join("gate1_zero_ablation.json");
    let file = std::fs::File::create(&out_path)?;
    serde_json::to_writer_pretty(file, &report)?;
    eprintln!("Wrote {}", out_path.display());

    Ok(())
}

fn select_zero_ablation_heads(
    args: &ZeroAblateArgs,
) -> Result<Vec<HeadId>, Box<dyn std::error::Error>> {
    let mut heads = if let Some(spec) = &args.heads {
        parse_head_spec(spec)?
    } else {
        let stage0_path = args
            .stage0
            .as_ref()
            .ok_or("--heads or --stage0 must be provided")?;
        let file = std::fs::File::open(stage0_path)?;
        let report: CaptureReport = serde_json::from_reader(file)?;
        let mut candidates = report.heads;
        candidates.sort_by(|a, b| {
            b.stats
                .variance
                .partial_cmp(&a.stats.variance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates
            .into_iter()
            .take(args.top_heads)
            .map(|h| HeadId {
                layer: h.layer,
                head: h.head,
            })
            .collect()
    };

    heads.sort_by_key(|h| (h.layer, h.head));
    heads.dedup();
    Ok(heads)
}

pub(super) fn forward_q4k_zero_pre_o_head(
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
                run_layer_with_zeroed_pre_o_heads(
                    weights,
                    &h,
                    layer,
                    &ffn,
                    &[head.head],
                    ple_inputs.get(layer),
                    shared_kv,
                )
                .map(|(h_new, kv_out)| (h_new, kv_out))
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
                "forward failed at layer {layer} while ablating L{} H{}",
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
