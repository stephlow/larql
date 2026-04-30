use std::path::PathBuf;
use std::time::Instant;

use clap::{Args, Subcommand};
use larql_inference::attention::run_attention_block_with_pre_o;
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
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Args)]
pub struct OvRdArgs {
    #[command(subcommand)]
    command: OvRdCommand,
}

#[derive(Subcommand)]
enum OvRdCommand {
    /// Capture pre-W_O OV output statistics from a Q4K vindex.
    Capture(CaptureArgs),

    /// Gate 1: zero selected pre-W_O heads and measure final-logit KL.
    ZeroAblate(ZeroAblateArgs),
}

#[derive(Args)]
struct CaptureArgs {
    /// Self-contained Q4K vindex directory.
    #[arg(long)]
    index: PathBuf,

    /// JSONL prompt file. Each line must include at least {"prompt": "..."}.
    #[arg(long)]
    prompts: PathBuf,

    /// Output directory.
    #[arg(long)]
    out: PathBuf,

    /// Layers to capture. Comma-separated or range. Default: all.
    #[arg(long)]
    layers: Option<String>,

    /// Limit prompts for smoke runs.
    #[arg(long)]
    max_prompts: Option<usize>,

    /// Limit token positions per prompt for smoke runs.
    #[arg(long)]
    max_positions: Option<usize>,
}

#[derive(Args)]
struct ZeroAblateArgs {
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

#[derive(Debug, Deserialize)]
struct PromptRecord {
    id: Option<String>,
    stratum: Option<String>,
    prompt: String,
}

#[derive(Debug)]
struct RunningHeadStats {
    count: u64,
    sum: Vec<f64>,
    sum_sq_norm: f64,
}

impl RunningHeadStats {
    fn new(head_dim: usize) -> Self {
        Self {
            count: 0,
            sum: vec![0.0; head_dim],
            sum_sq_norm: 0.0,
        }
    }

    fn add(&mut self, values: &[f32]) {
        self.count += 1;
        let mut sq = 0.0f64;
        for (dst, &v) in self.sum.iter_mut().zip(values.iter()) {
            let vf = v as f64;
            *dst += vf;
            sq += vf * vf;
        }
        self.sum_sq_norm += sq;
    }

    fn finish(&self) -> FinishedHeadStats {
        if self.count == 0 {
            return FinishedHeadStats {
                count: 0,
                mean_norm_sq: 0.0,
                second_moment: 0.0,
                variance: 0.0,
                rms_norm: 0.0,
            };
        }
        let n = self.count as f64;
        let mean_norm_sq = self
            .sum
            .iter()
            .map(|v| {
                let m = *v / n;
                m * m
            })
            .sum::<f64>();
        let second_moment = self.sum_sq_norm / n;
        let variance = (second_moment - mean_norm_sq).max(0.0);
        FinishedHeadStats {
            count: self.count,
            mean_norm_sq,
            second_moment,
            variance,
            rms_norm: second_moment.sqrt(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct FinishedHeadStats {
    count: u64,
    mean_norm_sq: f64,
    second_moment: f64,
    variance: f64,
    rms_norm: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct HeadReport {
    layer: usize,
    head: usize,
    head_dim: usize,
    stats: FinishedHeadStats,
}

#[derive(Debug, Serialize, Deserialize)]
struct CaptureReport {
    index: String,
    prompt_file: String,
    prompts_seen: usize,
    layers: Vec<usize>,
    max_positions: Option<usize>,
    heads: Vec<HeadReport>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct HeadId {
    layer: usize,
    head: usize,
}

#[derive(Debug, Serialize)]
struct ZeroStratumReport {
    stratum: String,
    prompts: usize,
    mean_kl: f64,
    max_kl: f64,
    top1_agreement: f64,
    top5_contains_baseline_top1: f64,
}

#[derive(Debug, Clone, Serialize)]
struct ZeroPromptReport {
    id: String,
    stratum: String,
    kl: f64,
    delta_cross_entropy_bits: f64,
    baseline_top1: u32,
    ablated_top1: u32,
    top1_agree: bool,
    baseline_top1_in_ablated_top5: bool,
}

#[derive(Debug, Serialize)]
struct ZeroHeadReport {
    layer: usize,
    head: usize,
    ablation_kind: String,
    patch_location: String,
    preserved_components: Vec<String>,
    bounded_vocab_size: Option<usize>,
    prompts: usize,
    mean_kl: f64,
    p95_kl: f64,
    max_kl: f64,
    mean_delta_cross_entropy_bits: f64,
    top1_agreement: f64,
    top5_contains_baseline_top1: f64,
    strata: Vec<ZeroStratumReport>,
    worst_examples: Vec<ZeroPromptReport>,
    per_prompt: Vec<ZeroPromptReport>,
}

#[derive(Debug, Serialize)]
struct ZeroAblationReport {
    index: String,
    prompt_file: String,
    prompts_seen: usize,
    selected_heads: Vec<HeadId>,
    heads: Vec<ZeroHeadReport>,
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

pub fn run(args: OvRdArgs) -> Result<(), Box<dyn std::error::Error>> {
    match args.command {
        OvRdCommand::Capture(capture) => run_capture(capture),
        OvRdCommand::ZeroAblate(zero) => run_zero_ablate(zero),
    }
}

fn run_capture(args: CaptureArgs) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(&args.out)?;

    eprintln!("Loading vindex: {}", args.index.display());
    let start = Instant::now();
    let mut cb = SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex(&args.index, &mut cb)?;
    index.load_attn_q4k(&args.index)?;
    index.load_interleaved_q4k(&args.index)?;
    let mut weights = load_model_weights_q4k(&args.index, &mut cb)?;
    let tokenizer = load_vindex_tokenizer(&args.index)?;
    eprintln!(
        "  {} layers, hidden_size={}, q_heads={}, head_dim={} ({:.1}s)",
        weights.num_layers,
        weights.hidden_size,
        weights.num_q_heads,
        weights.head_dim,
        start.elapsed().as_secs_f64()
    );

    let layers: Vec<usize> = match &args.layers {
        Some(spec) => parse_layer_spec(spec)?,
        None => (0..weights.num_layers).collect(),
    };
    let capture_layer = |layer: usize| layers.contains(&layer);

    let prompts = load_prompts(&args.prompts, args.max_prompts)?;
    eprintln!("Prompts: {}", prompts.len());
    eprintln!("Layers: {:?}", layers);

    let mut stats: Vec<Vec<RunningHeadStats>> = (0..weights.num_layers)
        .map(|layer| {
            let heads = weights.arch.num_q_heads_for_layer(layer);
            let head_dim = weights.arch.head_dim_for_layer(layer);
            (0..heads)
                .map(|_| RunningHeadStats::new(head_dim))
                .collect()
        })
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

        let mut h = embed_tokens_pub(&weights, &token_ids);
        let ple_inputs = precompute_per_layer_inputs(&weights, &h, &token_ids);

        for layer in 0..weights.num_layers {
            let inserted = insert_q4k_layer_tensors(&mut weights, &index, layer)?;

            if capture_layer(layer) {
                let (_, pre_o) = run_attention_block_with_pre_o(&weights, &h, layer)
                    .ok_or_else(|| format!("pre-W_O capture failed at layer {layer}"))?;
                add_pre_o_stats(
                    &mut stats[layer],
                    &pre_o,
                    weights.arch.num_q_heads_for_layer(layer),
                    weights.arch.head_dim_for_layer(layer),
                    args.max_positions,
                );
            }

            {
                let ffn = WeightFfn { weights: &weights };
                if let Some((h_new, _, _)) = run_layer_with_ffn(
                    &weights,
                    &h,
                    layer,
                    &ffn,
                    false,
                    ple_inputs.get(layer),
                    None,
                ) {
                    h = h_new;
                }
            }

            remove_layer_tensors(&mut weights, inserted);
        }
    }

    let mut heads = Vec::new();
    for &layer in &layers {
        let head_dim = weights.arch.head_dim_for_layer(layer);
        for (head, stat) in stats[layer].iter().enumerate() {
            heads.push(HeadReport {
                layer,
                head,
                head_dim,
                stats: stat.finish(),
            });
        }
    }

    let report = CaptureReport {
        index: args.index.display().to_string(),
        prompt_file: args.prompts.display().to_string(),
        prompts_seen: prompts.len(),
        layers,
        max_positions: args.max_positions,
        heads,
    };

    let out_path = args.out.join("stage0_pre_o_stats.json");
    let file = std::fs::File::create(&out_path)?;
    serde_json::to_writer_pretty(file, &report)?;
    eprintln!("Wrote {}", out_path.display());

    Ok(())
}

fn run_zero_ablate(args: ZeroAblateArgs) -> Result<(), Box<dyn std::error::Error>> {
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

fn add_pre_o_stats(
    stats: &mut [RunningHeadStats],
    pre_o: &Array2<f32>,
    num_heads: usize,
    head_dim: usize,
    max_positions: Option<usize>,
) {
    let positions = max_positions
        .map(|n| n.min(pre_o.nrows()))
        .unwrap_or_else(|| pre_o.nrows());
    for pos in 0..positions {
        for head in 0..num_heads {
            let start = head * head_dim;
            let end = start + head_dim;
            let row = pre_o.slice(s![pos, start..end]);
            if let Some(values) = row.as_slice() {
                stats[head].add(values);
            }
        }
    }
}

fn load_prompts(
    path: &PathBuf,
    max_prompts: Option<usize>,
) -> Result<Vec<PromptRecord>, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    let mut prompts = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        prompts.push(serde_json::from_str::<PromptRecord>(line)?);
        if max_prompts.is_some_and(|n| prompts.len() >= n) {
            break;
        }
    }
    Ok(prompts)
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

fn parse_head_spec(spec: &str) -> Result<Vec<HeadId>, Box<dyn std::error::Error>> {
    let mut heads = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let (layer, head) = part
            .split_once(':')
            .ok_or_else(|| format!("invalid head spec '{part}', expected layer:head"))?;
        heads.push(HeadId {
            layer: layer.parse()?,
            head: head.parse()?,
        });
    }
    Ok(heads)
}

fn forward_q4k_zero_pre_o_head(
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

fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let max_logit = logits
        .iter()
        .map(|&v| v as f64)
        .fold(f64::NEG_INFINITY, f64::max);
    let sum_exp = logits
        .iter()
        .map(|&v| ((v as f64) - max_logit).exp())
        .sum::<f64>();
    let log_z = max_logit + sum_exp.ln();
    logits.iter().map(|&v| (v as f64) - log_z).collect()
}

fn kl_logp(p_logp: &[f64], q_logp: &[f64]) -> f64 {
    p_logp
        .iter()
        .zip(q_logp.iter())
        .map(|(&lp, &lq)| {
            let p = lp.exp();
            p * (lp - lq)
        })
        .sum()
}

fn argmax(values: &[f32]) -> u32 {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0)
}

fn top_k_indices(values: &[f32], k: usize) -> Vec<u32> {
    let mut pairs: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    let take = k.min(pairs.len());
    pairs.select_nth_unstable_by(take.saturating_sub(1), |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    pairs.truncate(take);
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    pairs.into_iter().map(|(idx, _)| idx as u32).collect()
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn bool_rate(values: impl Iterator<Item = bool>) -> f64 {
    let mut total = 0usize;
    let mut hits = 0usize;
    for value in values {
        total += 1;
        if value {
            hits += 1;
        }
    }
    if total == 0 {
        0.0
    } else {
        hits as f64 / total as f64
    }
}

fn percentile(mut values: Vec<f64>, p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let rank = ((values.len() - 1) as f64 * p).ceil() as usize;
    values[rank.min(values.len() - 1)]
}

fn insert_q4k_layer_tensors(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    layer: usize,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let attn = index
        .attn_q4k_layer_data(layer)
        .ok_or_else(|| format!("attn Q4K slices missing for layer {layer}"))?;
    let ffn = index
        .interleaved_q4k_layer_data(layer)
        .ok_or_else(|| format!("ffn Q4K slices missing for layer {layer}"))?;

    let arch = &*weights.arch;
    let hidden = weights.hidden_size;
    let num_q = arch.num_q_heads_for_layer(layer);
    let num_kv = arch.num_kv_heads_for_layer(layer);
    let head_dim = arch.head_dim_for_layer(layer);
    let q_dim = num_q * head_dim;
    let kv_dim = num_kv * head_dim;
    let intermediate = index.num_features(layer);

    let q_key = arch.attn_q_key(layer);
    let k_key = arch.attn_k_key(layer);
    let v_key = arch.attn_v_key(layer);
    let o_key = arch.attn_o_key(layer);
    let gate_key = arch.ffn_gate_key(layer);
    let up_key = arch.ffn_up_key(layer);
    let down_key = arch.ffn_down_key(layer);

    weights.tensors.insert(
        q_key.clone(),
        dequantize_matrix(attn[0].0, attn[0].1, q_dim, hidden).into_shared(),
    );
    weights.tensors.insert(
        k_key.clone(),
        dequantize_matrix(attn[1].0, attn[1].1, kv_dim, hidden).into_shared(),
    );
    weights.tensors.insert(
        v_key.clone(),
        dequantize_matrix(attn[2].0, attn[2].1, kv_dim, hidden).into_shared(),
    );
    weights.tensors.insert(
        o_key.clone(),
        dequantize_matrix(attn[3].0, attn[3].1, hidden, q_dim).into_shared(),
    );
    weights.tensors.insert(
        gate_key.clone(),
        dequantize_matrix(ffn[0].0, ffn[0].1, intermediate, hidden).into_shared(),
    );
    weights.tensors.insert(
        up_key.clone(),
        dequantize_matrix(ffn[1].0, ffn[1].1, intermediate, hidden).into_shared(),
    );

    let inter_padded = intermediate.div_ceil(larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS)
        * larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS;
    let w_down = if inter_padded != intermediate {
        let w = dequantize_matrix(ffn[2].0, ffn[2].1, hidden, inter_padded);
        w.slice(s![.., ..intermediate]).to_owned()
    } else {
        dequantize_matrix(ffn[2].0, ffn[2].1, hidden, intermediate)
    };
    weights
        .tensors
        .insert(down_key.clone(), w_down.into_shared());

    Ok(vec![q_key, k_key, v_key, o_key, gate_key, up_key, down_key])
}

fn remove_layer_tensors(weights: &mut larql_inference::ModelWeights, keys: Vec<String>) {
    for key in keys {
        weights.tensors.remove(&key);
    }
}

fn dequantize_matrix(bytes: &[u8], format: &str, rows: usize, cols: usize) -> Array2<f32> {
    let n = rows * cols;
    let block = larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS;
    let padded = n.div_ceil(block) * block;
    let info = larql_vindex::quant::registry::lookup(format)
        .unwrap_or_else(|| panic!("unsupported quant format in vindex: {format}"));
    let floats =
        (info.dequantize)(bytes, padded).unwrap_or_else(|e| panic!("{format} dequant failed: {e}"));
    let truncated = if floats.len() > n {
        floats[..n].to_vec()
    } else {
        floats
    };
    Array2::from_shape_vec((rows, cols), truncated).expect("shape mismatch dequantising matrix")
}

fn parse_layer_spec(spec: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let mut layers = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.contains('-') {
            let (a, b) = part
                .split_once('-')
                .ok_or_else(|| format!("invalid range: {part}"))?;
            let start: usize = a.parse()?;
            let end: usize = b.parse()?;
            layers.extend(start..=end);
        } else {
            layers.push(part.parse()?);
        }
    }
    Ok(layers)
}
