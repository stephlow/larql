use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use larql_inference::attention::run_attention_block_with_pre_o;
use larql_inference::attention::SharedKV;
use larql_inference::forward::ple::precompute_per_layer_inputs;
use larql_inference::forward::{
    embed_tokens_pub, run_layer_with_ffn, run_layer_with_replaced_head_residual_delta,
    run_layer_with_replaced_pre_o_head,
};
use larql_inference::{encode_prompt, hidden_to_raw_logits, WeightFfn};
use larql_vindex::{
    load_model_weights_q4k, load_vindex_tokenizer, SilentLoadCallbacks, VectorIndex,
};
use ndarray::{s, Array2};
use std::collections::HashMap;

use super::address::*;
use super::basis::*;
use super::input::*;
use super::metrics::*;
use super::pq::*;
use super::reports::*;
use super::runtime::*;
use super::static_replace::fit_static_means;
use super::stats::*;
use super::types::*;

#[derive(Args)]
pub(super) struct OraclePqArgs {
    /// Self-contained Q4K vindex directory.
    #[arg(long)]
    index: PathBuf,

    /// JSONL prompt file. Each line must include at least {"prompt": "..."}.
    #[arg(long)]
    prompts: PathBuf,

    /// Output directory.
    #[arg(long)]
    out: PathBuf,

    /// Explicit heads as layer:head comma list, e.g. 0:6.
    #[arg(long)]
    heads: String,

    /// Comma-separated PQ configs as K:groups:bits, e.g. 128:16:4,192:24:4.
    #[arg(long)]
    configs: String,

    /// Relative singular value cutoff for retained W_O-visible directions.
    #[arg(long, default_value_t = 1e-6)]
    sigma_rel_cutoff: f64,

    /// Lloyd iterations per product-codebook group.
    #[arg(long, default_value_t = 25)]
    pq_iters: usize,

    /// Also materialize residual-space additive tables and compare Mode D injection.
    #[arg(long)]
    mode_d_check: bool,

    /// Fit and evaluate graph-native discrete address probes.
    ///
    /// The probes use only prompt metadata and token ids, not residual vectors.
    /// Requires --mode-d-check because predicted addresses are evaluated through
    /// the materialized residual-space tables.
    #[arg(long)]
    address_probes: bool,

    /// Add a mixed simple-key address probe that picks the best discrete key
    /// independently for each PQ group on the training split.
    #[arg(long)]
    address_mixed_key_probe: bool,

    /// Evaluate simple discrete keys on selected PQ groups only. Selected
    /// groups are predicted from each key; unselected groups are evaluated as
    /// either oracle-correct or majority/default.
    #[arg(long)]
    address_key_group_probe: bool,

    /// Comma-separated PQ groups for --address-key-group-probe.
    #[arg(long, default_value = "0")]
    address_key_groups: String,

    /// Evaluate how sensitive Mode D is to address corruption.
    ///
    /// This keeps a prefix of oracle PQ groups and replaces the rest with
    /// per-group majority codes learned from the training split. It estimates
    /// how many groups must be addressed correctly before predicted addressing
    /// can pass the KL gate.
    #[arg(long)]
    address_corruption_sweep: bool,

    /// Evaluate one-group-at-a-time address importance by replacing each group
    /// with its train-set majority code while all other groups remain oracle.
    #[arg(long)]
    address_group_importance: bool,

    /// Fit and evaluate fixed random-hyperplane LSH probes for selected PQ
    /// groups. The selected groups are predicted from the residual entering the
    /// target layer; other groups are evaluated both oracle-correct and
    /// majority/default.
    #[arg(long)]
    address_lsh_group_probe: bool,

    /// Comma-separated PQ groups for --address-lsh-group-probe.
    #[arg(long, default_value = "0")]
    address_lsh_groups: String,

    /// Number of LSH bits per selected group. For a 4-bit PQ group, 4 LSH bits
    /// creates 16 buckets.
    #[arg(long, default_value_t = 4)]
    address_lsh_bits: usize,

    /// Number of deterministic random-hyperplane seeds to try per selected
    /// group. The best seed is selected by train code accuracy.
    #[arg(long, default_value_t = 32)]
    address_lsh_seeds: usize,

    /// Fit and evaluate supervised binary-hyperplane address probes for
    /// selected PQ groups. The selected groups are predicted from the residual
    /// entering the target layer; other groups are evaluated both
    /// oracle-correct and majority/default.
    #[arg(long)]
    address_supervised_group_probe: bool,

    /// Comma-separated PQ groups for --address-supervised-group-probe.
    #[arg(long, default_value = "0")]
    address_supervised_groups: String,

    /// SGD epochs for supervised binary-hyperplane group address probes.
    #[arg(long, default_value_t = 16)]
    address_supervised_epochs: usize,

    /// SGD learning rate for supervised binary-hyperplane group address probes.
    #[arg(long, default_value_t = 0.05)]
    address_supervised_lr: f32,

    /// L2 weight decay for supervised binary-hyperplane group address probes.
    #[arg(long, default_value_t = 1e-4)]
    address_supervised_l2: f32,

    /// Report train/eval PQ code distribution stability for selected groups.
    #[arg(long)]
    address_code_stability: bool,

    /// Comma-separated PQ groups for --address-code-stability.
    #[arg(long, default_value = "0")]
    address_code_stability_groups: String,

    /// Comma-separated PQ groups whose centroids are fit separately per
    /// prompt stratum. This is a codebook-layout diagnostic for cases where a
    /// single global PQ group carries a hard prose/structured tail.
    #[arg(long, default_value = "")]
    stratum_conditioned_pq_groups: String,

    /// Limit prompts for bounded oracle runs.
    #[arg(long)]
    max_prompts: Option<usize>,

    /// Keep at most N prompts per stratum after loading. Useful for balanced
    /// held-out smoke runs from a larger ordered corpus.
    #[arg(long)]
    max_per_stratum: Option<usize>,

    /// Evaluate only prompts where prompt_index % eval_mod == eval_offset.
    /// The remaining prompts are used to fit static means, PCA, and PQ.
    #[arg(long)]
    eval_mod: Option<usize>,

    /// Held-out modulo offset used with --eval-mod.
    #[arg(long, default_value_t = 0)]
    eval_offset: usize,
}

#[derive(Debug)]
struct OraclePqPointAccumulator {
    prompts: Vec<OraclePqPromptReport>,
    address_probe_accumulators: HashMap<String, AddressProbeAccumulator>,
    address_corruption_accumulators: HashMap<usize, AddressProbeAccumulator>,
    address_group_importance_accumulators: HashMap<usize, AddressProbeAccumulator>,
}

impl OraclePqPointAccumulator {
    fn new() -> Self {
        Self {
            prompts: Vec::new(),
            address_probe_accumulators: HashMap::new(),
            address_corruption_accumulators: HashMap::new(),
            address_group_importance_accumulators: HashMap::new(),
        }
    }

    fn add(&mut self, prompt: OraclePqPromptReport) {
        self.prompts.push(prompt);
    }

    fn add_address_probe(
        &mut self,
        name: &str,
        selected_group_keys: &[String],
        prompt: AddressProbePromptReport,
    ) {
        self.address_probe_accumulators
            .entry(name.to_string())
            .or_insert_with(|| AddressProbeAccumulator::new_with_keys(name, selected_group_keys))
            .add(prompt);
    }

    fn add_address_corruption(
        &mut self,
        oracle_groups_kept: usize,
        prompt: AddressProbePromptReport,
    ) {
        self.address_corruption_accumulators
            .entry(oracle_groups_kept)
            .or_insert_with(|| {
                AddressProbeAccumulator::new(&format!("oracle_groups_kept_{oracle_groups_kept}"))
            })
            .add(prompt);
    }

    fn add_address_group_importance(
        &mut self,
        replaced_group: usize,
        prompt: AddressProbePromptReport,
    ) {
        self.address_group_importance_accumulators
            .entry(replaced_group)
            .or_insert_with(|| {
                AddressProbeAccumulator::new(&format!("replaced_group_{replaced_group}"))
            })
            .add(prompt);
    }

    fn finish(
        self,
        config: PqConfig,
        hidden_dim: usize,
        code_stability: Vec<CodeStabilityReport>,
    ) -> OraclePqPointReport {
        let kls: Vec<f64> = self.prompts.iter().map(|p| p.kl).collect();
        let levels = 1usize << config.bits_per_group;
        let mode_d_kls = self
            .prompts
            .iter()
            .filter_map(|p| p.mode_d_kl)
            .collect::<Vec<_>>();
        let coeff_mode_d_diffs = self
            .prompts
            .iter()
            .filter_map(|p| p.coeff_mode_d_max_abs_logit_diff)
            .collect::<Vec<_>>();
        OraclePqPointReport {
            k: config.k,
            groups: config.groups,
            bits_per_group: config.bits_per_group,
            oracle_address_bits: config.groups * config.bits_per_group,
            coefficient_codebook_bytes_f32: config.groups
                * levels
                * (config.k / config.groups)
                * std::mem::size_of::<f32>(),
            mode_d_residual_table_bytes_bf16: config.groups * levels * hidden_dim * 2,
            prompts: self.prompts.len(),
            mean_kl: mean(&kls),
            p95_kl: percentile(kls.clone(), 0.95),
            max_kl: kls.iter().copied().fold(0.0, f64::max),
            mean_delta_cross_entropy_bits: mean(
                &self
                    .prompts
                    .iter()
                    .map(|p| p.delta_cross_entropy_bits)
                    .collect::<Vec<_>>(),
            ),
            top1_agreement: bool_rate(self.prompts.iter().map(|p| p.top1_agree)),
            top5_contains_baseline_top1: bool_rate(
                self.prompts.iter().map(|p| p.baseline_top1_in_pq_top5),
            ),
            mean_baseline_top1_prob: mean(
                &self
                    .prompts
                    .iter()
                    .map(|p| p.baseline_top1_prob)
                    .collect::<Vec<_>>(),
            ),
            mean_pq_prob_of_baseline_top1: mean(
                &self
                    .prompts
                    .iter()
                    .map(|p| p.pq_prob_of_baseline_top1)
                    .collect::<Vec<_>>(),
            ),
            mean_baseline_top1_margin: mean(
                &self
                    .prompts
                    .iter()
                    .map(|p| p.baseline_top1_margin)
                    .collect::<Vec<_>>(),
            ),
            mode_d_mean_kl: if mode_d_kls.is_empty() {
                None
            } else {
                Some(mean(&mode_d_kls))
            },
            mode_d_p95_kl: if mode_d_kls.is_empty() {
                None
            } else {
                Some(percentile(mode_d_kls.clone(), 0.95))
            },
            mode_d_max_kl: if mode_d_kls.is_empty() {
                None
            } else {
                Some(mode_d_kls.iter().copied().fold(0.0, f64::max))
            },
            mode_d_top1_agreement: if mode_d_kls.is_empty() {
                None
            } else {
                Some(bool_rate(
                    self.prompts.iter().filter_map(|p| p.mode_d_top1_agree),
                ))
            },
            mode_d_top5_contains_baseline_top1: if mode_d_kls.is_empty() {
                None
            } else {
                Some(bool_rate(
                    self.prompts
                        .iter()
                        .filter_map(|p| p.baseline_top1_in_mode_d_top5),
                ))
            },
            coeff_mode_d_max_abs_logit_diff: if coeff_mode_d_diffs.is_empty() {
                None
            } else {
                Some(coeff_mode_d_diffs.iter().copied().fold(0.0, f64::max))
            },
            address_probes: self
                .address_probe_accumulators
                .into_values()
                .map(|acc| acc.finish())
                .collect(),
            address_corruption_sweep: self
                .address_corruption_accumulators
                .into_iter()
                .map(|(oracle_groups_kept, acc)| acc.finish_corruption(oracle_groups_kept))
                .collect(),
            address_group_importance: self
                .address_group_importance_accumulators
                .into_iter()
                .map(|(replaced_group, acc)| acc.finish_group_importance(replaced_group))
                .collect(),
            code_stability,
            mean_pre_wo_l2: mean(&self.prompts.iter().map(|p| p.pre_wo_l2).collect::<Vec<_>>()),
            mean_wo_visible_l2: mean(
                &self
                    .prompts
                    .iter()
                    .map(|p| p.wo_visible_l2)
                    .collect::<Vec<_>>(),
            ),
            per_prompt: self.prompts,
        }
    }
}

#[derive(Debug)]
struct AddressProbeAccumulator {
    name: String,
    selected_group_keys: Vec<String>,
    prompts: Vec<AddressProbePromptReport>,
}

impl AddressProbeAccumulator {
    fn new(name: &str) -> Self {
        Self::new_with_keys(name, &[])
    }

    fn new_with_keys(name: &str, selected_group_keys: &[String]) -> Self {
        Self {
            name: name.to_string(),
            selected_group_keys: selected_group_keys.to_vec(),
            prompts: Vec::new(),
        }
    }

    fn add(&mut self, prompt: AddressProbePromptReport) {
        self.prompts.push(prompt);
    }

    fn finish(mut self) -> AddressProbeReport {
        let kls = self.prompts.iter().map(|p| p.kl).collect::<Vec<_>>();
        let positions = self.prompts.iter().map(|p| p.positions).sum::<usize>();
        let total_groups = self
            .prompts
            .iter()
            .map(|p| p.groups_total)
            .sum::<usize>()
            .max(1);
        let correct_groups = self.prompts.iter().map(|p| p.groups_correct).sum::<usize>();
        self.prompts
            .sort_by(|a, b| b.kl.partial_cmp(&a.kl).unwrap_or(std::cmp::Ordering::Equal));
        AddressProbeReport {
            name: self.name,
            selected_group_keys: self.selected_group_keys,
            prompts: self.prompts.len(),
            positions,
            group_accuracy: correct_groups as f64 / total_groups as f64,
            exact_address_accuracy: bool_rate(self.prompts.iter().map(|p| p.exact_address_match)),
            mean_groups_correct_per_sequence: mean(
                &self
                    .prompts
                    .iter()
                    .map(|p| p.groups_correct as f64)
                    .collect::<Vec<_>>(),
            ),
            mean_groups_correct_per_position: correct_groups as f64 / positions.max(1) as f64,
            mean_kl: mean(&kls),
            p95_kl: percentile(kls.clone(), 0.95),
            max_kl: kls.iter().copied().fold(0.0, f64::max),
            top1_agreement: bool_rate(self.prompts.iter().map(|p| p.top1_agree)),
            top5_contains_baseline_top1: bool_rate(
                self.prompts
                    .iter()
                    .map(|p| p.baseline_top1_in_predicted_top5),
            ),
            worst_examples: self.prompts.into_iter().take(8).collect(),
        }
    }

    fn finish_corruption(mut self, oracle_groups_kept: usize) -> AddressCorruptionReport {
        let kls = self.prompts.iter().map(|p| p.kl).collect::<Vec<_>>();
        let positions = self.prompts.iter().map(|p| p.positions).sum::<usize>();
        let total_groups = self
            .prompts
            .iter()
            .map(|p| p.groups_total)
            .sum::<usize>()
            .max(1);
        let correct_groups = self.prompts.iter().map(|p| p.groups_correct).sum::<usize>();
        self.prompts
            .sort_by(|a, b| b.kl.partial_cmp(&a.kl).unwrap_or(std::cmp::Ordering::Equal));
        AddressCorruptionReport {
            label: self.name,
            oracle_groups_kept,
            prompts: self.prompts.len(),
            positions,
            group_accuracy: correct_groups as f64 / total_groups as f64,
            exact_address_accuracy: bool_rate(self.prompts.iter().map(|p| p.exact_address_match)),
            mean_kl: mean(&kls),
            p95_kl: percentile(kls.clone(), 0.95),
            max_kl: kls.iter().copied().fold(0.0, f64::max),
            top1_agreement: bool_rate(self.prompts.iter().map(|p| p.top1_agree)),
            top5_contains_baseline_top1: bool_rate(
                self.prompts
                    .iter()
                    .map(|p| p.baseline_top1_in_predicted_top5),
            ),
            worst_examples: self.prompts.into_iter().take(8).collect(),
        }
    }

    fn finish_group_importance(mut self, replaced_group: usize) -> AddressGroupImportanceReport {
        let kls = self.prompts.iter().map(|p| p.kl).collect::<Vec<_>>();
        let positions = self.prompts.iter().map(|p| p.positions).sum::<usize>();
        let total_groups = self
            .prompts
            .iter()
            .map(|p| p.groups_total)
            .sum::<usize>()
            .max(1);
        let correct_groups = self.prompts.iter().map(|p| p.groups_correct).sum::<usize>();
        self.prompts
            .sort_by(|a, b| b.kl.partial_cmp(&a.kl).unwrap_or(std::cmp::Ordering::Equal));
        AddressGroupImportanceReport {
            replaced_group,
            prompts: self.prompts.len(),
            positions,
            group_accuracy: correct_groups as f64 / total_groups as f64,
            exact_address_accuracy: bool_rate(self.prompts.iter().map(|p| p.exact_address_match)),
            mean_kl: mean(&kls),
            p95_kl: percentile(kls.clone(), 0.95),
            max_kl: kls.iter().copied().fold(0.0, f64::max),
            top1_agreement: bool_rate(self.prompts.iter().map(|p| p.top1_agree)),
            top5_contains_baseline_top1: bool_rate(
                self.prompts
                    .iter()
                    .map(|p| p.baseline_top1_in_predicted_top5),
            ),
            worst_examples: self.prompts.into_iter().take(8).collect(),
        }
    }
}

pub(super) fn run_oracle_pq(args: OraclePqArgs) -> Result<(), Box<dyn std::error::Error>> {
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
        return Err("ov-rd oracle-pq currently supports dense FFN vindexes only".into());
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
        return Err("no heads selected for oracle PQ".into());
    }
    let configs = parse_pq_configs(&args.configs)?;
    if configs.is_empty() {
        return Err("no PQ configs selected".into());
    }
    let mut key_groups = parse_usize_list(&args.address_key_groups)?;
    key_groups.sort_unstable();
    key_groups.dedup();
    if args.address_key_group_probe {
        if key_groups.is_empty() {
            return Err(
                "--address-key-group-probe requires at least one --address-key-groups value".into(),
            );
        }
        for config in &configs {
            for &group in &key_groups {
                if group >= config.groups {
                    return Err(format!(
                        "--address-key-groups includes group {group}, but config {:?} has only {} groups",
                        config, config.groups
                    )
                    .into());
                }
            }
        }
    }
    let mut lsh_groups = parse_usize_list(&args.address_lsh_groups)?;
    lsh_groups.sort_unstable();
    lsh_groups.dedup();
    if args.address_lsh_group_probe {
        if lsh_groups.is_empty() {
            return Err(
                "--address-lsh-group-probe requires at least one --address-lsh-groups value".into(),
            );
        }
        if args.address_lsh_bits == 0 {
            return Err("--address-lsh-bits must be greater than zero".into());
        }
        if args.address_lsh_bits > 16 {
            return Err("--address-lsh-bits is capped at 16 for bounded diagnostics".into());
        }
        if args.address_lsh_seeds == 0 {
            return Err("--address-lsh-seeds must be greater than zero".into());
        }
        for config in &configs {
            for &group in &lsh_groups {
                if group >= config.groups {
                    return Err(format!(
                        "--address-lsh-groups includes group {group}, but config {:?} has only {} groups",
                        config, config.groups
                    )
                    .into());
                }
            }
        }
    }
    let mut supervised_groups = parse_usize_list(&args.address_supervised_groups)?;
    supervised_groups.sort_unstable();
    supervised_groups.dedup();
    if args.address_supervised_group_probe {
        if supervised_groups.is_empty() {
            return Err(
                "--address-supervised-group-probe requires at least one --address-supervised-groups value".into(),
            );
        }
        if args.address_supervised_epochs == 0 {
            return Err("--address-supervised-epochs must be greater than zero".into());
        }
        if args.address_supervised_lr <= 0.0 {
            return Err("--address-supervised-lr must be greater than zero".into());
        }
        if args.address_supervised_l2 < 0.0 {
            return Err("--address-supervised-l2 must be non-negative".into());
        }
        for config in &configs {
            for &group in &supervised_groups {
                if group >= config.groups {
                    return Err(format!(
                        "--address-supervised-groups includes group {group}, but config {:?} has only {} groups",
                        config, config.groups
                    )
                    .into());
                }
            }
        }
    }
    let mut code_stability_groups = parse_usize_list(&args.address_code_stability_groups)?;
    code_stability_groups.sort_unstable();
    code_stability_groups.dedup();
    if args.address_code_stability {
        if code_stability_groups.is_empty() {
            return Err(
                "--address-code-stability requires at least one --address-code-stability-groups value"
                    .into(),
            );
        }
        for config in &configs {
            for &group in &code_stability_groups {
                if group >= config.groups {
                    return Err(format!(
                        "--address-code-stability-groups includes group {group}, but config {:?} has only {} groups",
                        config, config.groups
                    )
                    .into());
                }
            }
        }
    }
    let mut stratum_conditioned_pq_groups = parse_usize_list(&args.stratum_conditioned_pq_groups)?;
    stratum_conditioned_pq_groups.sort_unstable();
    stratum_conditioned_pq_groups.dedup();
    for config in &configs {
        for &group in &stratum_conditioned_pq_groups {
            if group >= config.groups {
                return Err(format!(
                    "--stratum-conditioned-pq-groups includes group {group}, but config {:?} has only {} groups",
                    config, config.groups
                )
                .into());
            }
        }
    }
    let mut prompts = load_prompts(&args.prompts, args.max_prompts)?;
    if let Some(max_per_stratum) = args.max_per_stratum {
        prompts = limit_prompts_per_stratum(prompts, max_per_stratum);
    }
    eprintln!("Selected heads: {:?}", selected_heads);
    eprintln!("PQ configs: {:?}", configs);
    eprintln!("Prompts: {}", prompts.len());
    let (fit_prompts, eval_prompts): (Vec<PromptRecord>, Vec<PromptRecord>) =
        if let Some(eval_mod) = args.eval_mod {
            split_prompt_records(&prompts, eval_mod, args.eval_offset)?
        } else {
            (prompts.clone(), prompts.clone())
        };
    eprintln!(
        "Oracle PQ split: fit_prompts={}, eval_prompts={}",
        fit_prompts.len(),
        eval_prompts.len()
    );

    eprintln!("Fitting position-mean static bases");
    let means = fit_static_means(
        &mut weights,
        &index,
        &tokenizer,
        &fit_prompts,
        &selected_heads,
    )?;

    eprintln!("Building W_O-visible bases");
    let bases =
        build_roundtrip_bases(&mut weights, &index, &selected_heads, args.sigma_rel_cutoff)?;

    eprintln!("Fitting empirical z-space PCA bases");
    let pca_bases = fit_z_pca_bases(
        &mut weights,
        &index,
        &tokenizer,
        &fit_prompts,
        &selected_heads,
        &bases,
        &means,
    )?;

    eprintln!("Fitting product quantizers");
    let codebooks = fit_pq_codebooks(
        &mut weights,
        &index,
        &tokenizer,
        &fit_prompts,
        &selected_heads,
        &bases,
        &means,
        &pca_bases,
        &configs,
        args.pq_iters,
        &stratum_conditioned_pq_groups,
    )?;
    let mode_d_tables = if args.mode_d_check {
        eprintln!("Materializing Mode D residual-space tables");
        materialize_mode_d_tables(
            &mut weights,
            &index,
            &selected_heads,
            &bases,
            &means,
            &pca_bases,
            &codebooks,
            &stratum_conditioned_pq_groups,
        )?
    } else {
        HashMap::new()
    };
    let run_address_probes =
        args.address_probes || args.address_mixed_key_probe || args.address_key_group_probe;
    let address_probe_models = if run_address_probes {
        if !args.mode_d_check {
            return Err(
                "--address-probes/--address-mixed-key-probe requires --mode-d-check".into(),
            );
        }
        eprintln!("Fitting graph-native address probes");
        fit_address_probe_models(
            &mut weights,
            &index,
            &tokenizer,
            &fit_prompts,
            &selected_heads,
            &bases,
            &means,
            &pca_bases,
            &codebooks,
            args.address_mixed_key_probe,
        )?
    } else {
        HashMap::new()
    };
    let address_lsh_models = if args.address_lsh_group_probe {
        if !args.mode_d_check {
            return Err("--address-lsh-group-probe requires --mode-d-check".into());
        }
        eprintln!(
            "Fitting LSH group address probes for groups {:?} (bits={}, seeds={})",
            lsh_groups, args.address_lsh_bits, args.address_lsh_seeds
        );
        fit_address_lsh_group_models(
            &mut weights,
            &index,
            &tokenizer,
            &fit_prompts,
            &selected_heads,
            &bases,
            &means,
            &pca_bases,
            &codebooks,
            &lsh_groups,
            args.address_lsh_bits,
            args.address_lsh_seeds,
        )?
    } else {
        HashMap::new()
    };
    let address_supervised_models = if args.address_supervised_group_probe {
        if !args.mode_d_check {
            return Err("--address-supervised-group-probe requires --mode-d-check".into());
        }
        eprintln!(
            "Fitting supervised group address probes for groups {:?} (epochs={}, lr={}, l2={})",
            supervised_groups,
            args.address_supervised_epochs,
            args.address_supervised_lr,
            args.address_supervised_l2
        );
        fit_address_supervised_group_models(
            &mut weights,
            &index,
            &tokenizer,
            &fit_prompts,
            &selected_heads,
            &bases,
            &means,
            &pca_bases,
            &codebooks,
            &supervised_groups,
            args.address_supervised_epochs,
            args.address_supervised_lr,
            args.address_supervised_l2,
        )?
    } else {
        HashMap::new()
    };
    if args.address_corruption_sweep && !args.mode_d_check {
        return Err("--address-corruption-sweep requires --mode-d-check".into());
    }
    if args.address_group_importance && !args.mode_d_check {
        return Err("--address-group-importance requires --mode-d-check".into());
    }
    let majority_codes = if args.address_corruption_sweep
        || args.address_group_importance
        || args.address_lsh_group_probe
        || args.address_supervised_group_probe
        || args.address_key_group_probe
    {
        eprintln!("Fitting per-group majority codes for address diagnostics");
        fit_majority_codes_for_codebooks(
            &mut weights,
            &index,
            &tokenizer,
            &fit_prompts,
            &selected_heads,
            &bases,
            &means,
            &pca_bases,
            &codebooks,
        )?
    } else {
        HashMap::new()
    };
    let code_stability = if args.address_code_stability {
        eprintln!(
            "Measuring PQ code stability for groups {:?}",
            code_stability_groups
        );
        measure_code_stability(
            &mut weights,
            &index,
            &tokenizer,
            &fit_prompts,
            &eval_prompts,
            &selected_heads,
            &bases,
            &means,
            &pca_bases,
            &codebooks,
            &code_stability_groups,
        )?
    } else {
        HashMap::new()
    };

    let mut accumulators: HashMap<(HeadId, PqConfig), OraclePqPointAccumulator> = HashMap::new();
    for head in &selected_heads {
        for &config in &configs {
            accumulators.insert((*head, config), OraclePqPointAccumulator::new());
        }
    }

    for (prompt_idx, record) in eval_prompts.iter().enumerate() {
        let label = record
            .id
            .as_deref()
            .or(record.stratum.as_deref())
            .unwrap_or("prompt");
        eprintln!("  [{}/{}] {}", prompt_idx + 1, eval_prompts.len(), label);

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
        let baseline_top2 = top_k_indices(&baseline_logits, 2);
        let baseline_top2_token = baseline_top2.get(1).copied().unwrap_or(baseline_top1);
        let baseline_top1_prob = token_prob(&baseline_logp, baseline_top1);
        let baseline_top2_prob = token_prob(&baseline_logp, baseline_top2_token);
        let baseline_top1_margin = baseline_top1_prob - baseline_top2_prob;

        for head in &selected_heads {
            let basis = bases.get(head).ok_or_else(|| {
                format!("missing basis for oracle PQ L{} H{}", head.layer, head.head)
            })?;
            let head_means = means.get(head).ok_or_else(|| {
                format!(
                    "missing position means for oracle PQ L{} H{}",
                    head.layer, head.head
                )
            })?;
            let pca_basis = pca_bases.get(head).ok_or_else(|| {
                format!(
                    "missing empirical PCA basis for oracle PQ L{} H{}",
                    head.layer, head.head
                )
            })?;
            for &config in &configs {
                let codebook = codebooks.get(&(*head, config)).ok_or_else(|| {
                    format!("missing PQ codebook for L{} H{}", head.layer, head.head)
                })?;
                let (pq_hidden, metrics, oracle_codes_by_position) = forward_q4k_oracle_pq_head(
                    &mut weights,
                    &token_ids,
                    &index,
                    *head,
                    basis,
                    pca_basis,
                    head_means,
                    codebook,
                    stratum,
                )?;
                let pq_logits = final_logits(&weights, &pq_hidden);
                let pq_logp = log_softmax(&pq_logits);
                let kl = kl_logp(&baseline_logp, &pq_logp);
                let pq_top1 = argmax(&pq_logits);
                let pq_top5 = top_k_indices(&pq_logits, 5);
                let pq_top2 = top_k_indices(&pq_logits, 2);
                let pq_top2_token = pq_top2.get(1).copied().unwrap_or(pq_top1);
                let pq_top1_prob = token_prob(&pq_logp, pq_top1);
                let pq_top2_prob = token_prob(&pq_logp, pq_top2_token);
                let pq_top1_margin = pq_top1_prob - pq_top2_prob;
                let pq_prob_of_baseline_top1 = token_prob(&pq_logp, baseline_top1);

                let (
                    mode_d_kl,
                    mode_d_top1,
                    mode_d_top1_agree,
                    baseline_top1_in_mode_d_top5,
                    coeff_mode_d_max_abs_logit_diff,
                ) = if args.mode_d_check {
                    let mode_d_table = mode_d_tables.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing Mode D table for L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let mode_d_hidden = forward_q4k_oracle_pq_mode_d_head(
                        &mut weights,
                        &token_ids,
                        &index,
                        *head,
                        basis,
                        pca_basis,
                        head_means,
                        codebook,
                        mode_d_table,
                        stratum,
                    )?;
                    let mode_d_logits = final_logits(&weights, &mode_d_hidden);
                    let mode_d_logp = log_softmax(&mode_d_logits);
                    let mode_d_top1 = argmax(&mode_d_logits);
                    let mode_d_top5 = top_k_indices(&mode_d_logits, 5);
                    (
                        Some(kl_logp(&baseline_logp, &mode_d_logp)),
                        Some(mode_d_top1),
                        Some(baseline_top1 == mode_d_top1),
                        Some(mode_d_top5.contains(&baseline_top1)),
                        Some(max_abs_diff(&pq_logits, &mode_d_logits)),
                    )
                } else {
                    (None, None, None, None, None)
                };

                if run_address_probes {
                    let mode_d_table = mode_d_tables.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing Mode D table for address probes L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let probe_models =
                        address_probe_models.get(&(*head, config)).ok_or_else(|| {
                            format!(
                                "missing address probe models for L{} H{} {:?}",
                                head.layer, head.head, config
                            )
                        })?;
                    for probe_model in probe_models {
                        let full_probe_enabled =
                            args.address_probes || probe_model.name == "mixed_best_simple_key";
                        if full_probe_enabled {
                            let predicted_codes_by_position = (0..token_ids.len())
                                .map(|pos| probe_model.predict_codes(&token_ids, stratum, pos))
                                .collect::<Vec<_>>();
                            let address_match = address_match_report(
                                &oracle_codes_by_position,
                                &predicted_codes_by_position,
                            );
                            let predicted_hidden = forward_q4k_predicted_address_mode_d_head(
                                &mut weights,
                                &token_ids,
                                &index,
                                *head,
                                mode_d_table,
                                &predicted_codes_by_position,
                                stratum,
                            )?;
                            let predicted_logits = final_logits(&weights, &predicted_hidden);
                            let predicted_logp = log_softmax(&predicted_logits);
                            let predicted_top1 = argmax(&predicted_logits);
                            let predicted_top5 = top_k_indices(&predicted_logits, 5);
                            accumulators
                                .get_mut(&(*head, config))
                                .expect("oracle PQ accumulator missing")
                                .add_address_probe(
                                    &probe_model.name,
                                    &probe_model.selected_group_keys,
                                    AddressProbePromptReport {
                                        id: label.to_string(),
                                        stratum: stratum.to_string(),
                                        kl: kl_logp(&baseline_logp, &predicted_logp),
                                        positions: oracle_codes_by_position.len(),
                                        groups_correct: address_match.groups_correct,
                                        groups_total: address_match.groups_total,
                                        exact_address_match: address_match.exact_address_match,
                                        top1_agree: baseline_top1 == predicted_top1,
                                        baseline_top1_in_predicted_top5: predicted_top5
                                            .contains(&baseline_top1),
                                    },
                                );
                        }
                        if args.address_key_group_probe {
                            let group_majority =
                                majority_codes.get(&(*head, config)).ok_or_else(|| {
                                    format!(
                                        "missing majority codes for key group probe L{} H{} {:?}",
                                        head.layer, head.head, config
                                    )
                                })?;
                            for (probe_name, use_oracle_rest) in [
                                (
                                    format!(
                                        "{}_groups_{:?}_oracle_rest",
                                        probe_model.name, key_groups
                                    ),
                                    true,
                                ),
                                (
                                    format!(
                                        "{}_groups_{:?}_majority_rest",
                                        probe_model.name, key_groups
                                    ),
                                    false,
                                ),
                            ] {
                                let predicted_codes_by_position = oracle_codes_by_position
                                    .iter()
                                    .enumerate()
                                    .map(|(pos, oracle_codes)| {
                                        let mut codes = if use_oracle_rest {
                                            oracle_codes.clone()
                                        } else {
                                            group_majority.clone()
                                        };
                                        let probe_codes =
                                            probe_model.predict_codes(&token_ids, stratum, pos);
                                        for &group in &key_groups {
                                            codes[group] = probe_codes[group];
                                        }
                                        codes
                                    })
                                    .collect::<Vec<_>>();
                                let address_match = address_match_report(
                                    &oracle_codes_by_position,
                                    &predicted_codes_by_position,
                                );
                                let predicted_hidden = forward_q4k_predicted_address_mode_d_head(
                                    &mut weights,
                                    &token_ids,
                                    &index,
                                    *head,
                                    mode_d_table,
                                    &predicted_codes_by_position,
                                    stratum,
                                )?;
                                let predicted_logits = final_logits(&weights, &predicted_hidden);
                                let predicted_logp = log_softmax(&predicted_logits);
                                let predicted_top1 = argmax(&predicted_logits);
                                let predicted_top5 = top_k_indices(&predicted_logits, 5);
                                accumulators
                                    .get_mut(&(*head, config))
                                    .expect("oracle PQ accumulator missing")
                                    .add_address_probe(
                                        &probe_name,
                                        &probe_model.selected_group_keys,
                                        AddressProbePromptReport {
                                            id: label.to_string(),
                                            stratum: stratum.to_string(),
                                            kl: kl_logp(&baseline_logp, &predicted_logp),
                                            positions: oracle_codes_by_position.len(),
                                            groups_correct: address_match.groups_correct,
                                            groups_total: address_match.groups_total,
                                            exact_address_match: address_match.exact_address_match,
                                            top1_agree: baseline_top1 == predicted_top1,
                                            baseline_top1_in_predicted_top5: predicted_top5
                                                .contains(&baseline_top1),
                                        },
                                    );
                            }
                        }
                    }
                }

                if args.address_group_importance {
                    let mode_d_table = mode_d_tables.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing Mode D table for address group importance L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let group_majority = majority_codes.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing majority codes for address group importance L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    for replaced_group in 0..config.groups {
                        let predicted_codes_by_position = oracle_codes_by_position
                            .iter()
                            .map(|codes| {
                                codes
                                    .iter()
                                    .enumerate()
                                    .map(|(group, &code)| {
                                        if group == replaced_group {
                                            group_majority[group]
                                        } else {
                                            code
                                        }
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        let address_match = address_match_report(
                            &oracle_codes_by_position,
                            &predicted_codes_by_position,
                        );
                        let predicted_hidden = forward_q4k_predicted_address_mode_d_head(
                            &mut weights,
                            &token_ids,
                            &index,
                            *head,
                            mode_d_table,
                            &predicted_codes_by_position,
                            stratum,
                        )?;
                        let predicted_logits = final_logits(&weights, &predicted_hidden);
                        let predicted_logp = log_softmax(&predicted_logits);
                        let predicted_top1 = argmax(&predicted_logits);
                        let predicted_top5 = top_k_indices(&predicted_logits, 5);
                        accumulators
                            .get_mut(&(*head, config))
                            .expect("oracle PQ accumulator missing")
                            .add_address_group_importance(
                                replaced_group,
                                AddressProbePromptReport {
                                    id: label.to_string(),
                                    stratum: stratum.to_string(),
                                    kl: kl_logp(&baseline_logp, &predicted_logp),
                                    positions: oracle_codes_by_position.len(),
                                    groups_correct: address_match.groups_correct,
                                    groups_total: address_match.groups_total,
                                    exact_address_match: address_match.exact_address_match,
                                    top1_agree: baseline_top1 == predicted_top1,
                                    baseline_top1_in_predicted_top5: predicted_top5
                                        .contains(&baseline_top1),
                                },
                            );
                    }
                }

                if args.address_lsh_group_probe {
                    let mode_d_table = mode_d_tables.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing Mode D table for LSH group probe L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let lsh_model = address_lsh_models.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing LSH group probe model for L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let group_majority = majority_codes.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing majority codes for LSH group probe L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let layer_input =
                        capture_layer_input_hidden(&mut weights, &token_ids, &index, head.layer)?;
                    let selected_group_keys = lsh_model.selected_group_keys();
                    for (probe_name, use_oracle_rest) in [
                        (
                            format!("lsh_groups_{:?}_oracle_rest", lsh_model.groups),
                            true,
                        ),
                        (
                            format!("lsh_groups_{:?}_majority_rest", lsh_model.groups),
                            false,
                        ),
                    ] {
                        let predicted_codes_by_position = oracle_codes_by_position
                            .iter()
                            .enumerate()
                            .map(|(pos, oracle_codes)| {
                                let base_codes = if use_oracle_rest {
                                    oracle_codes.as_slice()
                                } else {
                                    group_majority.as_slice()
                                };
                                lsh_model.predict_selected_groups(&layer_input, pos, base_codes)
                            })
                            .collect::<Vec<_>>();
                        let address_match = address_match_report(
                            &oracle_codes_by_position,
                            &predicted_codes_by_position,
                        );
                        let predicted_hidden = forward_q4k_predicted_address_mode_d_head(
                            &mut weights,
                            &token_ids,
                            &index,
                            *head,
                            mode_d_table,
                            &predicted_codes_by_position,
                            stratum,
                        )?;
                        let predicted_logits = final_logits(&weights, &predicted_hidden);
                        let predicted_logp = log_softmax(&predicted_logits);
                        let predicted_top1 = argmax(&predicted_logits);
                        let predicted_top5 = top_k_indices(&predicted_logits, 5);
                        accumulators
                            .get_mut(&(*head, config))
                            .expect("oracle PQ accumulator missing")
                            .add_address_probe(
                                &probe_name,
                                &selected_group_keys,
                                AddressProbePromptReport {
                                    id: label.to_string(),
                                    stratum: stratum.to_string(),
                                    kl: kl_logp(&baseline_logp, &predicted_logp),
                                    positions: oracle_codes_by_position.len(),
                                    groups_correct: address_match.groups_correct,
                                    groups_total: address_match.groups_total,
                                    exact_address_match: address_match.exact_address_match,
                                    top1_agree: baseline_top1 == predicted_top1,
                                    baseline_top1_in_predicted_top5: predicted_top5
                                        .contains(&baseline_top1),
                                },
                            );
                    }
                }

                if args.address_supervised_group_probe {
                    let mode_d_table = mode_d_tables.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing Mode D table for supervised group probe L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let supervised_model = address_supervised_models
                        .get(&(*head, config))
                        .ok_or_else(|| {
                            format!(
                                "missing supervised group probe model for L{} H{} {:?}",
                                head.layer, head.head, config
                            )
                        })?;
                    let group_majority = majority_codes.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing majority codes for supervised group probe L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let layer_input =
                        capture_layer_input_hidden(&mut weights, &token_ids, &index, head.layer)?;
                    let selected_group_keys = supervised_model.selected_group_keys();
                    for (probe_name, use_oracle_rest) in [
                        (
                            format!(
                                "supervised_hyperplane_groups_{:?}_oracle_rest",
                                supervised_model.groups
                            ),
                            true,
                        ),
                        (
                            format!(
                                "supervised_hyperplane_groups_{:?}_majority_rest",
                                supervised_model.groups
                            ),
                            false,
                        ),
                    ] {
                        let predicted_codes_by_position = oracle_codes_by_position
                            .iter()
                            .enumerate()
                            .map(|(pos, oracle_codes)| {
                                let base_codes = if use_oracle_rest {
                                    oracle_codes.as_slice()
                                } else {
                                    group_majority.as_slice()
                                };
                                supervised_model.predict_selected_groups(
                                    &layer_input,
                                    pos,
                                    base_codes,
                                )
                            })
                            .collect::<Vec<_>>();
                        let address_match = address_match_report(
                            &oracle_codes_by_position,
                            &predicted_codes_by_position,
                        );
                        let predicted_hidden = forward_q4k_predicted_address_mode_d_head(
                            &mut weights,
                            &token_ids,
                            &index,
                            *head,
                            mode_d_table,
                            &predicted_codes_by_position,
                            stratum,
                        )?;
                        let predicted_logits = final_logits(&weights, &predicted_hidden);
                        let predicted_logp = log_softmax(&predicted_logits);
                        let predicted_top1 = argmax(&predicted_logits);
                        let predicted_top5 = top_k_indices(&predicted_logits, 5);
                        accumulators
                            .get_mut(&(*head, config))
                            .expect("oracle PQ accumulator missing")
                            .add_address_probe(
                                &probe_name,
                                &selected_group_keys,
                                AddressProbePromptReport {
                                    id: label.to_string(),
                                    stratum: stratum.to_string(),
                                    kl: kl_logp(&baseline_logp, &predicted_logp),
                                    positions: oracle_codes_by_position.len(),
                                    groups_correct: address_match.groups_correct,
                                    groups_total: address_match.groups_total,
                                    exact_address_match: address_match.exact_address_match,
                                    top1_agree: baseline_top1 == predicted_top1,
                                    baseline_top1_in_predicted_top5: predicted_top5
                                        .contains(&baseline_top1),
                                },
                            );
                    }
                }

                if args.address_corruption_sweep {
                    let mode_d_table = mode_d_tables.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing Mode D table for address corruption L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let group_majority = majority_codes.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing majority codes for address corruption L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let keep_values = corruption_keep_values(config.groups);
                    for oracle_groups_kept in keep_values {
                        let predicted_codes_by_position = oracle_codes_by_position
                            .iter()
                            .map(|codes| {
                                codes
                                    .iter()
                                    .enumerate()
                                    .map(|(group, &code)| {
                                        if group < oracle_groups_kept {
                                            code
                                        } else {
                                            group_majority[group]
                                        }
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        let address_match = address_match_report(
                            &oracle_codes_by_position,
                            &predicted_codes_by_position,
                        );
                        let predicted_hidden = forward_q4k_predicted_address_mode_d_head(
                            &mut weights,
                            &token_ids,
                            &index,
                            *head,
                            mode_d_table,
                            &predicted_codes_by_position,
                            stratum,
                        )?;
                        let predicted_logits = final_logits(&weights, &predicted_hidden);
                        let predicted_logp = log_softmax(&predicted_logits);
                        let predicted_top1 = argmax(&predicted_logits);
                        let predicted_top5 = top_k_indices(&predicted_logits, 5);
                        accumulators
                            .get_mut(&(*head, config))
                            .expect("oracle PQ accumulator missing")
                            .add_address_corruption(
                                oracle_groups_kept,
                                AddressProbePromptReport {
                                    id: label.to_string(),
                                    stratum: stratum.to_string(),
                                    kl: kl_logp(&baseline_logp, &predicted_logp),
                                    positions: oracle_codes_by_position.len(),
                                    groups_correct: address_match.groups_correct,
                                    groups_total: address_match.groups_total,
                                    exact_address_match: address_match.exact_address_match,
                                    top1_agree: baseline_top1 == predicted_top1,
                                    baseline_top1_in_predicted_top5: predicted_top5
                                        .contains(&baseline_top1),
                                },
                            );
                    }
                }

                accumulators
                    .get_mut(&(*head, config))
                    .expect("oracle PQ accumulator missing")
                    .add(OraclePqPromptReport {
                        id: label.to_string(),
                        stratum: stratum.to_string(),
                        kl,
                        delta_cross_entropy_bits: kl / std::f64::consts::LN_2,
                        baseline_top1,
                        pq_top1,
                        top1_agree: baseline_top1 == pq_top1,
                        baseline_top1_in_pq_top5: pq_top5.contains(&baseline_top1),
                        baseline_top1_prob,
                        baseline_top2: baseline_top2_token,
                        baseline_top2_prob,
                        baseline_top1_margin,
                        pq_top1_prob,
                        pq_prob_of_baseline_top1,
                        pq_top1_margin,
                        mode_d_kl,
                        mode_d_top1,
                        mode_d_top1_agree,
                        baseline_top1_in_mode_d_top5,
                        coeff_mode_d_max_abs_logit_diff,
                        pre_wo_l2: metrics.pre_wo_l2,
                        wo_visible_l2: metrics.wo_visible_l2,
                    });
            }
        }
    }

    let mut head_reports = Vec::new();
    for head in &selected_heads {
        let basis = bases
            .get(head)
            .ok_or_else(|| format!("missing basis for L{} H{}", head.layer, head.head))?;
        let pca_basis = pca_bases
            .get(head)
            .ok_or_else(|| format!("missing PCA basis for L{} H{}", head.layer, head.head))?;
        let static_train_samples = means.get(head).map(|m| m.count).unwrap_or(0);
        let mut points = Vec::new();
        for &config in &configs {
            let acc = accumulators
                .remove(&(*head, config))
                .expect("oracle PQ accumulator missing at finish");
            let stability = code_stability
                .get(&(*head, config))
                .cloned()
                .unwrap_or_default();
            points.push(acc.finish(config, weights.hidden_size, stability));
        }
        head_reports.push(OraclePqHeadReport {
            layer: head.layer,
            head: head.head,
            head_dim: basis.head_dim,
            rank_retained: basis.rank_retained(),
            empirical_rank: pca_basis.rank(),
            sigma_max: basis.sigma_max,
            sigma_min_retained: basis.sigma_min_retained,
            static_train_samples,
            points,
        });
    }

    let report = OraclePqReport {
        index: args.index.display().to_string(),
        prompt_file: args.prompts.display().to_string(),
        prompts_seen: prompts.len(),
        train_prompts_seen: fit_prompts.len(),
        eval_prompts_seen: eval_prompts.len(),
        max_per_stratum: args.max_per_stratum,
        eval_mod: args.eval_mod,
        eval_offset: args.eval_offset,
        static_base: "position_mean".to_string(),
        configs,
        sigma_rel_cutoff: args.sigma_rel_cutoff,
        pq_iters: args.pq_iters,
        mode_d_check: args.mode_d_check,
        address_probes: args.address_probes,
        address_mixed_key_probe: args.address_mixed_key_probe,
        address_key_group_probe: args.address_key_group_probe,
        address_key_groups: if args.address_key_group_probe {
            key_groups
        } else {
            Vec::new()
        },
        address_corruption_sweep: args.address_corruption_sweep,
        address_group_importance: args.address_group_importance,
        address_lsh_group_probe: args.address_lsh_group_probe,
        address_lsh_groups: if args.address_lsh_group_probe {
            lsh_groups
        } else {
            Vec::new()
        },
        address_lsh_bits: args.address_lsh_bits,
        address_lsh_seeds: args.address_lsh_seeds,
        address_supervised_group_probe: args.address_supervised_group_probe,
        address_supervised_groups: if args.address_supervised_group_probe {
            supervised_groups
        } else {
            Vec::new()
        },
        address_supervised_epochs: args.address_supervised_epochs,
        address_supervised_lr: args.address_supervised_lr,
        address_supervised_l2: args.address_supervised_l2,
        address_code_stability: args.address_code_stability,
        address_code_stability_groups: if args.address_code_stability {
            code_stability_groups
        } else {
            Vec::new()
        },
        stratum_conditioned_pq_groups,
        selected_heads,
        heads: head_reports,
    };

    let out_path = args.out.join("oracle_pq.json");
    let file = std::fs::File::create(&out_path)?;
    serde_json::to_writer_pretty(file, &report)?;
    eprintln!("Wrote {}", out_path.display());

    Ok(())
}

fn fit_pq_codebooks(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    tokenizer: &tokenizers::Tokenizer,
    prompts: &[PromptRecord],
    heads: &[HeadId],
    bases: &HashMap<HeadId, WoRoundtripBasis>,
    means: &HashMap<HeadId, StaticHeadMeans>,
    pca_bases: &HashMap<HeadId, ZPcaBasis>,
    configs: &[PqConfig],
    iterations: usize,
    stratum_conditioned_groups: &[usize],
) -> Result<HashMap<(HeadId, PqConfig), PqCodebook>, Box<dyn std::error::Error>> {
    let max_k = configs.iter().map(|c| c.k).max().unwrap_or(0);
    let mut heads_by_layer: HashMap<usize, Vec<HeadId>> = HashMap::new();
    for head in heads {
        heads_by_layer.entry(head.layer).or_default().push(*head);
    }

    let mut samples: HashMap<HeadId, Vec<Vec<f64>>> = HashMap::new();
    let mut samples_by_stratum: HashMap<(HeadId, String), Vec<Vec<f64>>> = HashMap::new();
    for head in heads {
        samples.insert(*head, Vec::new());
    }

    for (prompt_idx, record) in prompts.iter().enumerate() {
        let label = record
            .id
            .as_deref()
            .or(record.stratum.as_deref())
            .unwrap_or("prompt");
        eprintln!("  pq-fit [{}/{}] {}", prompt_idx + 1, prompts.len(), label);
        let token_ids = encode_prompt(tokenizer, &*weights.arch, &record.prompt)?;
        if token_ids.is_empty() {
            continue;
        }
        let stratum = record.stratum.as_deref().unwrap_or("unknown");
        let mut h = embed_tokens_pub(weights, &token_ids);
        let ple_inputs = precompute_per_layer_inputs(weights, &h, &token_ids);

        for layer in 0..weights.num_layers {
            let inserted = insert_q4k_layer_tensors(weights, index, layer)?;
            if let Some(layer_heads) = heads_by_layer.get(&layer) {
                let (_, pre_o) = run_attention_block_with_pre_o(weights, &h, layer)
                    .ok_or_else(|| format!("pre-W_O capture failed at layer {layer}"))?;
                let head_dim = weights.arch.head_dim_for_layer(layer);
                for head in layer_heads {
                    let basis = bases.get(head).expect("basis pre-created for PQ fit");
                    let head_means = means.get(head).expect("means pre-created for PQ fit");
                    let pca_basis = pca_bases.get(head).expect("PCA pre-created for PQ fit");
                    if pca_basis.rank() < max_k {
                        return Err(format!(
                            "PCA rank {} is below requested K {} for L{}H{}",
                            pca_basis.rank(),
                            max_k,
                            head.layer,
                            head.head
                        )
                        .into());
                    }
                    let start = head.head * head_dim;
                    let end = start + head_dim;
                    let head_samples = samples.get_mut(head).expect("PQ samples missing");
                    for pos in 0..pre_o.nrows() {
                        let row = pre_o.slice(s![pos, start..end]);
                        let values = row
                            .as_slice()
                            .ok_or("pre-W_O head row was not contiguous during PQ fit")?;
                        let base = head_means.positions.get(pos).unwrap_or(&head_means.global);
                        let residual = values
                            .iter()
                            .zip(base.iter())
                            .map(|(&yi, &bi)| yi - bi)
                            .collect::<Vec<_>>();
                        let z = basis.residual_to_z(&residual);
                        let coords = pca_basis.coordinates_with_rank(&z, max_k);
                        head_samples.push(coords.clone());
                        if !stratum_conditioned_groups.is_empty() {
                            samples_by_stratum
                                .entry((*head, stratum.to_string()))
                                .or_default()
                                .push(coords);
                        }
                    }
                }
            }

            {
                let ffn = WeightFfn { weights };
                if let Some((h_new, _, _)) =
                    run_layer_with_ffn(weights, &h, layer, &ffn, false, ple_inputs.get(layer), None)
                {
                    h = h_new;
                }
            }
            remove_layer_tensors(weights, inserted);
        }
    }

    let mut codebooks = HashMap::new();
    for head in heads {
        let head_samples = samples
            .get(head)
            .ok_or_else(|| format!("missing PQ samples for L{}H{}", head.layer, head.head))?;
        for &config in configs {
            let levels = 1usize << config.bits_per_group;
            let group_dim = config.k / config.groups;
            let mut centroids = Vec::with_capacity(config.groups);
            for group in 0..config.groups {
                let start = group * group_dim;
                let group_samples = head_samples
                    .iter()
                    .map(|sample| sample[start..start + group_dim].to_vec())
                    .collect::<Vec<_>>();
                centroids.push(kmeans_centroids(&group_samples, levels, iterations));
            }
            let mut stratum_centroids: HashMap<String, HashMap<usize, Vec<Vec<f64>>>> =
                HashMap::new();
            for &group in stratum_conditioned_groups {
                let start = group * group_dim;
                for ((sample_head, stratum), stratum_samples) in samples_by_stratum.iter() {
                    if sample_head != head {
                        continue;
                    }
                    let group_samples = stratum_samples
                        .iter()
                        .map(|sample| sample[start..start + group_dim].to_vec())
                        .collect::<Vec<_>>();
                    stratum_centroids
                        .entry(stratum.clone())
                        .or_default()
                        .insert(group, kmeans_centroids(&group_samples, levels, iterations));
                }
            }
            codebooks.insert(
                (*head, config),
                PqCodebook {
                    config,
                    centroids,
                    stratum_centroids,
                },
            );
        }
    }

    Ok(codebooks)
}

fn fit_address_probe_models(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    tokenizer: &tokenizers::Tokenizer,
    prompts: &[PromptRecord],
    heads: &[HeadId],
    bases: &HashMap<HeadId, WoRoundtripBasis>,
    means: &HashMap<HeadId, StaticHeadMeans>,
    pca_bases: &HashMap<HeadId, ZPcaBasis>,
    codebooks: &HashMap<(HeadId, PqConfig), PqCodebook>,
    include_mixed_key_probe: bool,
) -> Result<HashMap<(HeadId, PqConfig), Vec<AddressProbeModel>>, Box<dyn std::error::Error>> {
    let names = address_probe_names();
    let mut heads_by_layer: HashMap<usize, Vec<HeadId>> = HashMap::new();
    for head in heads {
        heads_by_layer.entry(head.layer).or_default().push(*head);
    }

    let mut key_counts: HashMap<(HeadId, PqConfig, String, usize, String), Vec<usize>> =
        HashMap::new();
    let mut majority_counts: HashMap<(HeadId, PqConfig, usize), Vec<usize>> = HashMap::new();

    for (prompt_idx, record) in prompts.iter().enumerate() {
        let label = record
            .id
            .as_deref()
            .or(record.stratum.as_deref())
            .unwrap_or("prompt");
        eprintln!(
            "  address-fit [{}/{}] {}",
            prompt_idx + 1,
            prompts.len(),
            label
        );
        let token_ids = encode_prompt(tokenizer, &*weights.arch, &record.prompt)?;
        if token_ids.is_empty() {
            continue;
        }
        let stratum = record.stratum.as_deref().unwrap_or("unknown");
        let mut h = embed_tokens_pub(weights, &token_ids);
        let ple_inputs = precompute_per_layer_inputs(weights, &h, &token_ids);

        for layer in 0..weights.num_layers {
            let inserted = insert_q4k_layer_tensors(weights, index, layer)?;
            if let Some(layer_heads) = heads_by_layer.get(&layer) {
                let (_, pre_o) = run_attention_block_with_pre_o(weights, &h, layer)
                    .ok_or_else(|| format!("pre-W_O capture failed at layer {layer}"))?;
                let head_dim = weights.arch.head_dim_for_layer(layer);
                for head in layer_heads {
                    let basis = bases.get(head).ok_or_else(|| {
                        format!("missing basis for L{}H{}", head.layer, head.head)
                    })?;
                    let head_means = means.get(head).ok_or_else(|| {
                        format!("missing means for L{}H{}", head.layer, head.head)
                    })?;
                    let pca_basis = pca_bases.get(head).ok_or_else(|| {
                        format!("missing PCA basis for L{}H{}", head.layer, head.head)
                    })?;
                    let start = head.head * head_dim;
                    let end = start + head_dim;
                    let head_codebooks = codebooks
                        .iter()
                        .filter(|((codebook_head, _), _)| codebook_head == head)
                        .collect::<Vec<_>>();
                    for pos in 0..pre_o.nrows() {
                        let row = pre_o.slice(s![pos, start..end]);
                        let values = row.as_slice().ok_or(
                            "pre-W_O head row was not contiguous during address probe fit",
                        )?;
                        let base = head_means.positions.get(pos).unwrap_or(&head_means.global);
                        let residual = values
                            .iter()
                            .zip(base.iter())
                            .map(|(&yi, &bi)| yi - bi)
                            .collect::<Vec<_>>();
                        let z = basis.residual_to_z(&residual);
                        for ((_, config), codebook) in &head_codebooks {
                            let coords = pca_basis.coordinates_with_rank(&z, config.k);
                            let codes = codebook.quantize_indices_for_stratum(&coords, stratum);
                            for (group, &code) in codes.iter().enumerate() {
                                let levels = 1usize << config.bits_per_group;
                                let counts = majority_counts
                                    .entry((*head, *config, group))
                                    .or_insert_with(|| vec![0; levels]);
                                counts[code] += 1;
                                for name in &names {
                                    let key = address_feature_key(name, &token_ids, stratum, pos);
                                    let counts = key_counts
                                        .entry((*head, *config, (*name).to_string(), group, key))
                                        .or_insert_with(|| vec![0; levels]);
                                    counts[code] += 1;
                                }
                            }
                        }
                    }
                }
            }

            {
                let ffn = WeightFfn { weights };
                if let Some((h_new, _, _)) =
                    run_layer_with_ffn(weights, &h, layer, &ffn, false, ple_inputs.get(layer), None)
                {
                    h = h_new;
                }
            }
            remove_layer_tensors(weights, inserted);
        }
    }

    let mut models = HashMap::new();
    for ((head, config), _) in codebooks {
        let mut probe_models = Vec::new();
        for name in &names {
            let mut group_majority = Vec::with_capacity(config.groups);
            let mut group_maps = Vec::with_capacity(config.groups);
            let mut group_train_accuracy = Vec::with_capacity(config.groups);
            for group in 0..config.groups {
                let majority = majority_counts
                    .get(&(*head, *config, group))
                    .map(|counts| argmax_usize(counts))
                    .unwrap_or(0);
                group_majority.push(majority);
                let mut map = HashMap::new();
                let mut correct = 0usize;
                let mut total = 0usize;
                for ((map_head, map_config, map_name, map_group, key), counts) in key_counts.iter()
                {
                    if map_head == head
                        && map_config == config
                        && map_name == name
                        && *map_group == group
                    {
                        let best = argmax_usize(counts);
                        correct += counts[best];
                        total += counts.iter().sum::<usize>();
                        map.insert(key.clone(), best);
                    }
                }
                group_maps.push(map);
                group_train_accuracy.push(if total == 0 {
                    0.0
                } else {
                    correct as f64 / total as f64
                });
            }
            probe_models.push(AddressProbeModel {
                name: (*name).to_string(),
                group_majority,
                group_maps,
                group_train_accuracy,
                selected_group_keys: Vec::new(),
            });
        }
        if include_mixed_key_probe && !probe_models.is_empty() {
            let mut group_majority = Vec::with_capacity(config.groups);
            let mut group_maps = Vec::with_capacity(config.groups);
            let mut group_train_accuracy = Vec::with_capacity(config.groups);
            let mut selected_group_keys = Vec::with_capacity(config.groups);
            for group in 0..config.groups {
                let best_idx = probe_models
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        a.group_train_accuracy[group]
                            .partial_cmp(&b.group_train_accuracy[group])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                let best = &probe_models[best_idx];
                group_majority.push(best.group_majority[group]);
                group_maps.push(best.group_maps[group].clone());
                group_train_accuracy.push(best.group_train_accuracy[group]);
                selected_group_keys.push(best.name.clone());
            }
            probe_models.push(AddressProbeModel {
                name: "mixed_best_simple_key".to_string(),
                group_majority,
                group_maps,
                group_train_accuracy,
                selected_group_keys,
            });
        }
        models.insert((*head, *config), probe_models);
    }

    Ok(models)
}

fn fit_address_lsh_group_models(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    tokenizer: &tokenizers::Tokenizer,
    prompts: &[PromptRecord],
    heads: &[HeadId],
    bases: &HashMap<HeadId, WoRoundtripBasis>,
    means: &HashMap<HeadId, StaticHeadMeans>,
    pca_bases: &HashMap<HeadId, ZPcaBasis>,
    codebooks: &HashMap<(HeadId, PqConfig), PqCodebook>,
    selected_groups: &[usize],
    bits: usize,
    seeds: usize,
) -> Result<HashMap<(HeadId, PqConfig), AddressLshGroupModel>, Box<dyn std::error::Error>> {
    let mut heads_by_layer: HashMap<usize, Vec<HeadId>> = HashMap::new();
    for head in heads {
        heads_by_layer.entry(head.layer).or_default().push(*head);
    }

    let mut majority_counts: HashMap<(HeadId, PqConfig, usize), Vec<usize>> = HashMap::new();
    let mut bucket_counts: HashMap<(HeadId, PqConfig, usize, u64, usize), Vec<usize>> =
        HashMap::new();

    for (prompt_idx, record) in prompts.iter().enumerate() {
        let label = record
            .id
            .as_deref()
            .or(record.stratum.as_deref())
            .unwrap_or("prompt");
        eprintln!("  lsh-fit [{}/{}] {}", prompt_idx + 1, prompts.len(), label);
        let token_ids = encode_prompt(tokenizer, &*weights.arch, &record.prompt)?;
        if token_ids.is_empty() {
            continue;
        }
        let stratum = record.stratum.as_deref().unwrap_or("unknown");
        let mut h = embed_tokens_pub(weights, &token_ids);
        let ple_inputs = precompute_per_layer_inputs(weights, &h, &token_ids);

        for layer in 0..weights.num_layers {
            let inserted = insert_q4k_layer_tensors(weights, index, layer)?;
            if let Some(layer_heads) = heads_by_layer.get(&layer) {
                let layer_input = h.clone();
                let (_, pre_o) = run_attention_block_with_pre_o(weights, &h, layer)
                    .ok_or_else(|| format!("pre-W_O capture failed at layer {layer}"))?;
                let head_dim = weights.arch.head_dim_for_layer(layer);
                for head in layer_heads {
                    let basis = bases.get(head).ok_or_else(|| {
                        format!("missing basis for L{}H{}", head.layer, head.head)
                    })?;
                    let head_means = means.get(head).ok_or_else(|| {
                        format!("missing means for L{}H{}", head.layer, head.head)
                    })?;
                    let pca_basis = pca_bases.get(head).ok_or_else(|| {
                        format!("missing PCA basis for L{}H{}", head.layer, head.head)
                    })?;
                    let start = head.head * head_dim;
                    let end = start + head_dim;
                    let head_codebooks = codebooks
                        .iter()
                        .filter(|((codebook_head, _), _)| codebook_head == head)
                        .collect::<Vec<_>>();
                    for pos in 0..pre_o.nrows() {
                        let row = pre_o.slice(s![pos, start..end]);
                        let values = row
                            .as_slice()
                            .ok_or("pre-W_O head row was not contiguous during LSH address fit")?;
                        let base = head_means.positions.get(pos).unwrap_or(&head_means.global);
                        let residual = values
                            .iter()
                            .zip(base.iter())
                            .map(|(&yi, &bi)| yi - bi)
                            .collect::<Vec<_>>();
                        let z = basis.residual_to_z(&residual);
                        let input_row = layer_input.row(pos);
                        for ((_, config), codebook) in &head_codebooks {
                            let coords = pca_basis.coordinates_with_rank(&z, config.k);
                            let codes = codebook.quantize_indices_for_stratum(&coords, stratum);
                            let levels = 1usize << config.bits_per_group;
                            for (group, &code) in codes.iter().enumerate() {
                                let counts = majority_counts
                                    .entry((*head, *config, group))
                                    .or_insert_with(|| vec![0; levels]);
                                counts[code] += 1;
                            }
                            for &group in selected_groups {
                                let code = codes[group];
                                for seed in 0..seeds {
                                    let bucket = lsh_bucket(input_row, seed as u64, bits);
                                    let counts = bucket_counts
                                        .entry((*head, *config, group, seed as u64, bucket))
                                        .or_insert_with(|| vec![0; levels]);
                                    counts[code] += 1;
                                }
                            }
                        }
                    }
                }
            }

            {
                let ffn = WeightFfn { weights };
                if let Some((h_new, _, _)) =
                    run_layer_with_ffn(weights, &h, layer, &ffn, false, ple_inputs.get(layer), None)
                {
                    h = h_new;
                }
            }
            remove_layer_tensors(weights, inserted);
        }
    }

    let mut models = HashMap::new();
    for ((head, config), _) in codebooks {
        let mut group_majority = Vec::with_capacity(config.groups);
        for group in 0..config.groups {
            let majority = majority_counts
                .get(&(*head, *config, group))
                .map(|counts| argmax_usize(counts))
                .unwrap_or(0);
            group_majority.push(majority);
        }

        let mut group_maps = vec![HashMap::new(); config.groups];
        let mut group_seeds = vec![0_u64; config.groups];
        let mut group_train_accuracy = vec![0.0; config.groups];
        for &group in selected_groups {
            let mut best_seed = 0_u64;
            let mut best_accuracy = -1.0_f64;
            let mut best_map = HashMap::new();
            for seed in 0..seeds {
                let seed = seed as u64;
                let mut map = HashMap::new();
                let mut correct = 0usize;
                let mut total = 0usize;
                for ((map_head, map_config, map_group, map_seed, bucket), counts) in
                    bucket_counts.iter()
                {
                    if map_head == head
                        && map_config == config
                        && *map_group == group
                        && *map_seed == seed
                    {
                        let best = argmax_usize(counts);
                        correct += counts[best];
                        total += counts.iter().sum::<usize>();
                        map.insert(*bucket, best);
                    }
                }
                let accuracy = if total == 0 {
                    0.0
                } else {
                    correct as f64 / total as f64
                };
                if accuracy > best_accuracy {
                    best_accuracy = accuracy;
                    best_seed = seed;
                    best_map = map;
                }
            }
            group_maps[group] = best_map;
            group_seeds[group] = best_seed;
            group_train_accuracy[group] = best_accuracy.max(0.0);
        }

        models.insert(
            (*head, *config),
            AddressLshGroupModel {
                groups: selected_groups.to_vec(),
                bits,
                group_majority,
                group_maps,
                group_seeds,
                group_train_accuracy,
            },
        );
    }

    Ok(models)
}

fn fit_address_supervised_group_models(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    tokenizer: &tokenizers::Tokenizer,
    prompts: &[PromptRecord],
    heads: &[HeadId],
    bases: &HashMap<HeadId, WoRoundtripBasis>,
    means: &HashMap<HeadId, StaticHeadMeans>,
    pca_bases: &HashMap<HeadId, ZPcaBasis>,
    codebooks: &HashMap<(HeadId, PqConfig), PqCodebook>,
    selected_groups: &[usize],
    epochs: usize,
    lr: f32,
    l2: f32,
) -> Result<HashMap<(HeadId, PqConfig), AddressSupervisedGroupModel>, Box<dyn std::error::Error>> {
    let mut heads_by_layer: HashMap<usize, Vec<HeadId>> = HashMap::new();
    for head in heads {
        heads_by_layer.entry(head.layer).or_default().push(*head);
    }

    let mut majority_counts: HashMap<(HeadId, PqConfig, usize), Vec<usize>> = HashMap::new();
    let mut samples: HashMap<(HeadId, PqConfig), Vec<(Vec<f32>, Vec<usize>)>> = HashMap::new();

    for (prompt_idx, record) in prompts.iter().enumerate() {
        let label = record
            .id
            .as_deref()
            .or(record.stratum.as_deref())
            .unwrap_or("prompt");
        eprintln!(
            "  supervised-fit [{}/{}] {}",
            prompt_idx + 1,
            prompts.len(),
            label
        );
        let token_ids = encode_prompt(tokenizer, &*weights.arch, &record.prompt)?;
        if token_ids.is_empty() {
            continue;
        }
        let stratum = record.stratum.as_deref().unwrap_or("unknown");
        let mut h = embed_tokens_pub(weights, &token_ids);
        let ple_inputs = precompute_per_layer_inputs(weights, &h, &token_ids);

        for layer in 0..weights.num_layers {
            let inserted = insert_q4k_layer_tensors(weights, index, layer)?;
            if let Some(layer_heads) = heads_by_layer.get(&layer) {
                let layer_input = h.clone();
                let (_, pre_o) = run_attention_block_with_pre_o(weights, &h, layer)
                    .ok_or_else(|| format!("pre-W_O capture failed at layer {layer}"))?;
                let head_dim = weights.arch.head_dim_for_layer(layer);
                for head in layer_heads {
                    let basis = bases.get(head).ok_or_else(|| {
                        format!("missing basis for L{}H{}", head.layer, head.head)
                    })?;
                    let head_means = means.get(head).ok_or_else(|| {
                        format!("missing means for L{}H{}", head.layer, head.head)
                    })?;
                    let pca_basis = pca_bases.get(head).ok_or_else(|| {
                        format!("missing PCA basis for L{}H{}", head.layer, head.head)
                    })?;
                    let start = head.head * head_dim;
                    let end = start + head_dim;
                    let head_codebooks = codebooks
                        .iter()
                        .filter(|((codebook_head, _), _)| codebook_head == head)
                        .collect::<Vec<_>>();
                    for pos in 0..pre_o.nrows() {
                        let row = pre_o.slice(s![pos, start..end]);
                        let values = row.as_slice().ok_or(
                            "pre-W_O head row was not contiguous during supervised address fit",
                        )?;
                        let base = head_means.positions.get(pos).unwrap_or(&head_means.global);
                        let residual = values
                            .iter()
                            .zip(base.iter())
                            .map(|(&yi, &bi)| yi - bi)
                            .collect::<Vec<_>>();
                        let z = basis.residual_to_z(&residual);
                        let input_row = layer_input.row(pos).to_vec();
                        for ((_, config), codebook) in &head_codebooks {
                            let coords = pca_basis.coordinates_with_rank(&z, config.k);
                            let codes = codebook.quantize_indices_for_stratum(&coords, stratum);
                            let levels = 1usize << config.bits_per_group;
                            for (group, &code) in codes.iter().enumerate() {
                                let counts = majority_counts
                                    .entry((*head, *config, group))
                                    .or_insert_with(|| vec![0; levels]);
                                counts[code] += 1;
                            }
                            samples
                                .entry((*head, *config))
                                .or_default()
                                .push((input_row.clone(), codes));
                        }
                    }
                }
            }

            {
                let ffn = WeightFfn { weights };
                if let Some((h_new, _, _)) =
                    run_layer_with_ffn(weights, &h, layer, &ffn, false, ple_inputs.get(layer), None)
                {
                    h = h_new;
                }
            }
            remove_layer_tensors(weights, inserted);
        }
    }

    let mut models = HashMap::new();
    for ((head, config), _) in codebooks {
        let train_samples = samples.get(&(*head, *config)).cloned().unwrap_or_default();
        let dim = train_samples.first().map(|(row, _)| row.len()).unwrap_or(0);
        let mut group_majority = Vec::with_capacity(config.groups);
        for group in 0..config.groups {
            let majority = majority_counts
                .get(&(*head, *config, group))
                .map(|counts| argmax_usize(counts))
                .unwrap_or(0);
            group_majority.push(majority);
        }

        let mut group_hyperplanes = vec![Vec::new(); config.groups];
        let mut group_train_accuracy = vec![0.0; config.groups];
        for &group in selected_groups {
            let mut bit_planes = Vec::with_capacity(config.bits_per_group);
            for bit in 0..config.bits_per_group {
                let labels = train_samples
                    .iter()
                    .map(|(_, codes)| ((codes[group] >> bit) & 1) != 0)
                    .collect::<Vec<_>>();
                let rows = train_samples
                    .iter()
                    .map(|(row, _)| row.as_slice())
                    .collect::<Vec<_>>();
                bit_planes.push(train_binary_hyperplane(&rows, &labels, dim, epochs, lr, l2));
            }

            let mut correct = 0usize;
            for (row, codes) in &train_samples {
                let predicted = predict_code_from_hyperplanes(row, &bit_planes);
                if predicted == codes[group] {
                    correct += 1;
                }
            }
            group_train_accuracy[group] = if train_samples.is_empty() {
                0.0
            } else {
                correct as f64 / train_samples.len() as f64
            };
            group_hyperplanes[group] = bit_planes;
        }

        models.insert(
            (*head, *config),
            AddressSupervisedGroupModel {
                groups: selected_groups.to_vec(),
                bits_per_group: config.bits_per_group,
                epochs,
                lr,
                l2,
                group_majority,
                group_hyperplanes,
                group_train_accuracy,
            },
        );
    }

    Ok(models)
}

#[derive(Debug, Clone)]
struct CodeDistributionCounts {
    group_counts: HashMap<usize, Vec<usize>>,
    stratum_group_counts: HashMap<String, HashMap<usize, Vec<usize>>>,
}

impl CodeDistributionCounts {
    fn new(selected_groups: &[usize], levels: usize) -> Self {
        Self {
            group_counts: selected_groups
                .iter()
                .map(|&group| (group, vec![0; levels]))
                .collect(),
            stratum_group_counts: HashMap::new(),
        }
    }

    fn add(&mut self, group: usize, code: usize, stratum: &str, levels: usize) {
        if let Some(counts) = self.group_counts.get_mut(&group) {
            counts[code] += 1;
        }
        self.stratum_group_counts
            .entry(stratum.to_string())
            .or_default()
            .entry(group)
            .or_insert_with(|| vec![0; levels])[code] += 1;
    }
}

fn measure_code_stability(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    tokenizer: &tokenizers::Tokenizer,
    train_prompts: &[PromptRecord],
    eval_prompts: &[PromptRecord],
    heads: &[HeadId],
    bases: &HashMap<HeadId, WoRoundtripBasis>,
    means: &HashMap<HeadId, StaticHeadMeans>,
    pca_bases: &HashMap<HeadId, ZPcaBasis>,
    codebooks: &HashMap<(HeadId, PqConfig), PqCodebook>,
    selected_groups: &[usize],
) -> Result<HashMap<(HeadId, PqConfig), Vec<CodeStabilityReport>>, Box<dyn std::error::Error>> {
    let train = collect_code_distribution_counts(
        weights,
        index,
        tokenizer,
        train_prompts,
        heads,
        bases,
        means,
        pca_bases,
        codebooks,
        selected_groups,
        "code-stability-train",
    )?;
    let eval = collect_code_distribution_counts(
        weights,
        index,
        tokenizer,
        eval_prompts,
        heads,
        bases,
        means,
        pca_bases,
        codebooks,
        selected_groups,
        "code-stability-eval",
    )?;

    let mut reports = HashMap::new();
    for ((head, config), _) in codebooks {
        let levels = 1usize << config.bits_per_group;
        let empty_counts = CodeDistributionCounts::new(selected_groups, levels);
        let train_counts = train.get(&(*head, *config)).unwrap_or(&empty_counts);
        let eval_counts = eval.get(&(*head, *config)).unwrap_or(&empty_counts);
        let mut group_reports = Vec::new();
        for &group in selected_groups {
            let train_group = train_counts
                .group_counts
                .get(&group)
                .cloned()
                .unwrap_or_else(|| vec![0; levels]);
            let eval_group = eval_counts
                .group_counts
                .get(&group)
                .cloned()
                .unwrap_or_else(|| vec![0; levels]);
            let train_top = argmax_usize(&train_group);
            let eval_top = argmax_usize(&eval_group);
            let mut stratum_names = train_counts
                .stratum_group_counts
                .keys()
                .chain(eval_counts.stratum_group_counts.keys())
                .cloned()
                .collect::<Vec<_>>();
            stratum_names.sort();
            stratum_names.dedup();
            let by_stratum = stratum_names
                .into_iter()
                .map(|stratum| {
                    let train_s = train_counts
                        .stratum_group_counts
                        .get(&stratum)
                        .and_then(|groups| groups.get(&group))
                        .cloned()
                        .unwrap_or_else(|| vec![0; levels]);
                    let eval_s = eval_counts
                        .stratum_group_counts
                        .get(&stratum)
                        .and_then(|groups| groups.get(&group))
                        .cloned()
                        .unwrap_or_else(|| vec![0; levels]);
                    let train_s_top = argmax_usize(&train_s);
                    let eval_s_top = argmax_usize(&eval_s);
                    CodeStabilityStratumReport {
                        stratum,
                        train_positions: train_s.iter().sum(),
                        eval_positions: eval_s.iter().sum(),
                        train_entropy_bits: entropy_bits(&train_s),
                        eval_entropy_bits: entropy_bits(&eval_s),
                        train_top_code: train_s_top,
                        train_top_code_mass: code_mass(&train_s, train_s_top),
                        eval_top_code: eval_s_top,
                        eval_top_code_mass: code_mass(&eval_s, eval_s_top),
                        train_eval_js_bits: js_divergence_bits(&train_s, &eval_s),
                    }
                })
                .collect();
            group_reports.push(CodeStabilityReport {
                group,
                train_positions: train_group.iter().sum(),
                eval_positions: eval_group.iter().sum(),
                train_entropy_bits: entropy_bits(&train_group),
                eval_entropy_bits: entropy_bits(&eval_group),
                train_top_code: train_top,
                train_top_code_mass: code_mass(&train_group, train_top),
                eval_top_code: eval_top,
                eval_top_code_mass: code_mass(&eval_group, eval_top),
                train_eval_js_bits: js_divergence_bits(&train_group, &eval_group),
                by_stratum,
            });
        }
        reports.insert((*head, *config), group_reports);
    }

    Ok(reports)
}

fn collect_code_distribution_counts(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    tokenizer: &tokenizers::Tokenizer,
    prompts: &[PromptRecord],
    heads: &[HeadId],
    bases: &HashMap<HeadId, WoRoundtripBasis>,
    means: &HashMap<HeadId, StaticHeadMeans>,
    pca_bases: &HashMap<HeadId, ZPcaBasis>,
    codebooks: &HashMap<(HeadId, PqConfig), PqCodebook>,
    selected_groups: &[usize],
    label_prefix: &str,
) -> Result<HashMap<(HeadId, PqConfig), CodeDistributionCounts>, Box<dyn std::error::Error>> {
    let mut heads_by_layer: HashMap<usize, Vec<HeadId>> = HashMap::new();
    for head in heads {
        heads_by_layer.entry(head.layer).or_default().push(*head);
    }
    let mut counts = HashMap::new();
    for ((head, config), _) in codebooks {
        counts.insert(
            (*head, *config),
            CodeDistributionCounts::new(selected_groups, 1usize << config.bits_per_group),
        );
    }

    for (prompt_idx, record) in prompts.iter().enumerate() {
        let label = record
            .id
            .as_deref()
            .or(record.stratum.as_deref())
            .unwrap_or("prompt");
        eprintln!(
            "  {label_prefix} [{}/{}] {}",
            prompt_idx + 1,
            prompts.len(),
            label
        );
        let token_ids = encode_prompt(tokenizer, &*weights.arch, &record.prompt)?;
        if token_ids.is_empty() {
            continue;
        }
        let stratum = record.stratum.as_deref().unwrap_or("unknown");
        let mut h = embed_tokens_pub(weights, &token_ids);
        let ple_inputs = precompute_per_layer_inputs(weights, &h, &token_ids);

        for layer in 0..weights.num_layers {
            let inserted = insert_q4k_layer_tensors(weights, index, layer)?;
            if let Some(layer_heads) = heads_by_layer.get(&layer) {
                let (_, pre_o) = run_attention_block_with_pre_o(weights, &h, layer)
                    .ok_or_else(|| format!("pre-W_O capture failed at layer {layer}"))?;
                let head_dim = weights.arch.head_dim_for_layer(layer);
                for head in layer_heads {
                    let basis = bases.get(head).ok_or_else(|| {
                        format!("missing basis for L{}H{}", head.layer, head.head)
                    })?;
                    let head_means = means.get(head).ok_or_else(|| {
                        format!("missing means for L{}H{}", head.layer, head.head)
                    })?;
                    let pca_basis = pca_bases.get(head).ok_or_else(|| {
                        format!("missing PCA basis for L{}H{}", head.layer, head.head)
                    })?;
                    let start = head.head * head_dim;
                    let end = start + head_dim;
                    let head_codebooks = codebooks
                        .iter()
                        .filter(|((codebook_head, _), _)| codebook_head == head)
                        .collect::<Vec<_>>();
                    for pos in 0..pre_o.nrows() {
                        let row = pre_o.slice(s![pos, start..end]);
                        let values = row
                            .as_slice()
                            .ok_or("pre-W_O head row was not contiguous during code stability")?;
                        let base = head_means.positions.get(pos).unwrap_or(&head_means.global);
                        let residual = values
                            .iter()
                            .zip(base.iter())
                            .map(|(&yi, &bi)| yi - bi)
                            .collect::<Vec<_>>();
                        let z = basis.residual_to_z(&residual);
                        for ((_, config), codebook) in &head_codebooks {
                            let coords = pca_basis.coordinates_with_rank(&z, config.k);
                            let codes = codebook.quantize_indices_for_stratum(&coords, stratum);
                            let levels = 1usize << config.bits_per_group;
                            let point_counts =
                                counts.get_mut(&(*head, *config)).ok_or_else(|| {
                                    format!(
                                        "missing code stability counts for L{}H{} {:?}",
                                        head.layer, head.head, config
                                    )
                                })?;
                            for &group in selected_groups {
                                point_counts.add(group, codes[group], stratum, levels);
                            }
                        }
                    }
                }
            }

            {
                let ffn = WeightFfn { weights };
                if let Some((h_new, _, _)) =
                    run_layer_with_ffn(weights, &h, layer, &ffn, false, ple_inputs.get(layer), None)
                {
                    h = h_new;
                }
            }
            remove_layer_tensors(weights, inserted);
        }
    }

    Ok(counts)
}

fn fit_majority_codes_for_codebooks(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    tokenizer: &tokenizers::Tokenizer,
    prompts: &[PromptRecord],
    heads: &[HeadId],
    bases: &HashMap<HeadId, WoRoundtripBasis>,
    means: &HashMap<HeadId, StaticHeadMeans>,
    pca_bases: &HashMap<HeadId, ZPcaBasis>,
    codebooks: &HashMap<(HeadId, PqConfig), PqCodebook>,
) -> Result<HashMap<(HeadId, PqConfig), Vec<usize>>, Box<dyn std::error::Error>> {
    let mut heads_by_layer: HashMap<usize, Vec<HeadId>> = HashMap::new();
    for head in heads {
        heads_by_layer.entry(head.layer).or_default().push(*head);
    }

    let mut majority_counts: HashMap<(HeadId, PqConfig, usize), Vec<usize>> = HashMap::new();

    for (prompt_idx, record) in prompts.iter().enumerate() {
        let label = record
            .id
            .as_deref()
            .or(record.stratum.as_deref())
            .unwrap_or("prompt");
        eprintln!(
            "  majority-fit [{}/{}] {}",
            prompt_idx + 1,
            prompts.len(),
            label
        );
        let token_ids = encode_prompt(tokenizer, &*weights.arch, &record.prompt)?;
        if token_ids.is_empty() {
            continue;
        }
        let stratum = record.stratum.as_deref().unwrap_or("unknown");
        let mut h = embed_tokens_pub(weights, &token_ids);
        let ple_inputs = precompute_per_layer_inputs(weights, &h, &token_ids);

        for layer in 0..weights.num_layers {
            let inserted = insert_q4k_layer_tensors(weights, index, layer)?;
            if let Some(layer_heads) = heads_by_layer.get(&layer) {
                let (_, pre_o) = run_attention_block_with_pre_o(weights, &h, layer)
                    .ok_or_else(|| format!("pre-W_O capture failed at layer {layer}"))?;
                let head_dim = weights.arch.head_dim_for_layer(layer);
                for head in layer_heads {
                    let basis = bases.get(head).ok_or_else(|| {
                        format!("missing basis for L{}H{}", head.layer, head.head)
                    })?;
                    let head_means = means.get(head).ok_or_else(|| {
                        format!("missing means for L{}H{}", head.layer, head.head)
                    })?;
                    let pca_basis = pca_bases.get(head).ok_or_else(|| {
                        format!("missing PCA basis for L{}H{}", head.layer, head.head)
                    })?;
                    let start = head.head * head_dim;
                    let end = start + head_dim;
                    let head_codebooks = codebooks
                        .iter()
                        .filter(|((codebook_head, _), _)| codebook_head == head)
                        .collect::<Vec<_>>();
                    for pos in 0..pre_o.nrows() {
                        let row = pre_o.slice(s![pos, start..end]);
                        let values = row.as_slice().ok_or(
                            "pre-W_O head row was not contiguous during majority code fit",
                        )?;
                        let base = head_means.positions.get(pos).unwrap_or(&head_means.global);
                        let residual = values
                            .iter()
                            .zip(base.iter())
                            .map(|(&yi, &bi)| yi - bi)
                            .collect::<Vec<_>>();
                        let z = basis.residual_to_z(&residual);
                        for ((_, config), codebook) in &head_codebooks {
                            let coords = pca_basis.coordinates_with_rank(&z, config.k);
                            let codes = codebook.quantize_indices_for_stratum(&coords, stratum);
                            for (group, &code) in codes.iter().enumerate() {
                                let levels = 1usize << config.bits_per_group;
                                let counts = majority_counts
                                    .entry((*head, *config, group))
                                    .or_insert_with(|| vec![0; levels]);
                                counts[code] += 1;
                            }
                        }
                    }
                }
            }

            {
                let ffn = WeightFfn { weights };
                if let Some((h_new, _, _)) =
                    run_layer_with_ffn(weights, &h, layer, &ffn, false, ple_inputs.get(layer), None)
                {
                    h = h_new;
                }
            }
            remove_layer_tensors(weights, inserted);
        }
    }

    let mut out = HashMap::new();
    for ((head, config), _) in codebooks {
        let mut group_majority = Vec::with_capacity(config.groups);
        for group in 0..config.groups {
            group_majority.push(
                majority_counts
                    .get(&(*head, *config, group))
                    .map(|counts| argmax_usize(counts))
                    .unwrap_or(0),
            );
        }
        out.insert((*head, *config), group_majority);
    }
    Ok(out)
}

fn corruption_keep_values(groups: usize) -> Vec<usize> {
    [0usize, 4, 8, 12, 16, 24, 32, 40, groups]
        .into_iter()
        .filter(|value| *value <= groups)
        .collect()
}

fn materialize_mode_d_tables(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    heads: &[HeadId],
    bases: &HashMap<HeadId, WoRoundtripBasis>,
    means: &HashMap<HeadId, StaticHeadMeans>,
    pca_bases: &HashMap<HeadId, ZPcaBasis>,
    codebooks: &HashMap<(HeadId, PqConfig), PqCodebook>,
    stratum_conditioned_groups: &[usize],
) -> Result<HashMap<(HeadId, PqConfig), ModeDTable>, Box<dyn std::error::Error>> {
    let mut heads_by_layer: HashMap<usize, Vec<HeadId>> = HashMap::new();
    for head in heads {
        heads_by_layer.entry(head.layer).or_default().push(*head);
    }

    let mut tables = HashMap::new();
    for (layer, layer_heads) in heads_by_layer {
        let inserted = insert_q4k_layer_tensors(weights, index, layer)?;
        let w_o = weights
            .tensors
            .get(&weights.arch.attn_o_key(layer))
            .ok_or_else(|| format!("missing W_O tensor at layer {layer}"))?;
        let head_dim = weights.arch.head_dim_for_layer(layer);
        for head in layer_heads {
            let start = head.head * head_dim;
            let end = start + head_dim;
            let w_o_head = w_o.slice(s![.., start..end]);
            let head_means = means
                .get(&head)
                .ok_or_else(|| format!("missing means for L{}H{}", head.layer, head.head))?;
            let static_global_delta = project_head_vector_to_hidden(&w_o_head, &head_means.global);
            let static_delta_by_position = head_means
                .positions
                .iter()
                .map(|mean| project_head_vector_to_hidden(&w_o_head, mean))
                .collect::<Vec<_>>();
            let basis = bases
                .get(&head)
                .ok_or_else(|| format!("missing W_O basis for L{}H{}", head.layer, head.head))?;
            let pca_basis = pca_bases
                .get(&head)
                .ok_or_else(|| format!("missing PCA basis for L{}H{}", head.layer, head.head))?;

            for ((codebook_head, config), codebook) in codebooks.iter() {
                if *codebook_head != head {
                    continue;
                }
                let group_dim = config.k / config.groups;
                let mut group_tables = Vec::with_capacity(config.groups);
                for group in 0..config.groups {
                    let mut table = Vec::with_capacity(codebook.centroids[group].len());
                    for centroid in &codebook.centroids[group] {
                        let mut coords = vec![0.0; config.k];
                        let start_coord = group * group_dim;
                        coords[start_coord..start_coord + group_dim].copy_from_slice(centroid);
                        let z_part = pca_basis.reconstruct_from_coordinates(&coords);
                        let residual_part = basis.z_to_residual(&z_part);
                        table.push(project_head_vector_to_hidden(&w_o_head, &residual_part));
                    }
                    group_tables.push(table);
                }
                let mut stratum_group_tables: HashMap<String, HashMap<usize, Vec<Vec<f32>>>> =
                    HashMap::new();
                for (stratum, groups) in &codebook.stratum_centroids {
                    for &group in stratum_conditioned_groups {
                        let Some(centroids) = groups.get(&group) else {
                            continue;
                        };
                        let mut table = Vec::with_capacity(centroids.len());
                        for centroid in centroids {
                            let mut coords = vec![0.0; config.k];
                            let start_coord = group * group_dim;
                            coords[start_coord..start_coord + group_dim].copy_from_slice(centroid);
                            let z_part = pca_basis.reconstruct_from_coordinates(&coords);
                            let residual_part = basis.z_to_residual(&z_part);
                            table.push(project_head_vector_to_hidden(&w_o_head, &residual_part));
                        }
                        stratum_group_tables
                            .entry(stratum.clone())
                            .or_default()
                            .insert(group, table);
                    }
                }
                tables.insert(
                    (head, *config),
                    ModeDTable {
                        static_delta_by_position: static_delta_by_position.clone(),
                        static_global_delta: static_global_delta.clone(),
                        group_tables,
                        stratum_group_tables,
                    },
                );
            }
        }
        remove_layer_tensors(weights, inserted);
    }
    Ok(tables)
}

fn project_head_vector_to_hidden(
    w_o_head: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,
    values: &[f32],
) -> Vec<f32> {
    let mut out = vec![0.0f32; w_o_head.nrows()];
    for row in 0..w_o_head.nrows() {
        let mut sum = 0.0f32;
        for col in 0..w_o_head.ncols() {
            sum += values[col] * w_o_head[[row, col]];
        }
        out[row] = sum;
    }
    out
}

fn forward_q4k_oracle_pq_head(
    weights: &mut larql_inference::ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    head: HeadId,
    basis: &WoRoundtripBasis,
    pca_basis: &ZPcaBasis,
    means: &StaticHeadMeans,
    codebook: &PqCodebook,
    stratum: &str,
) -> Result<(Array2<f32>, RoundtripPatchMetrics, Vec<Vec<usize>>), Box<dyn std::error::Error>> {
    let mut h = embed_tokens_pub(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
    let mut kv_cache: HashMap<usize, SharedKV> = HashMap::new();
    let mut metrics = None;
    let mut oracle_codes = Vec::new();

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
                let mut replacement = Vec::with_capacity(pre_o.nrows() * head_dim);
                let mut pre_sq = 0.0;
                let mut visible_sq = 0.0;
                let mut count = 0usize;
                for pos in 0..pre_o.nrows() {
                    let row = pre_o.slice(s![pos, start..end]);
                    let values = row
                        .as_slice()
                        .ok_or("pre-W_O head row was not contiguous during PQ")?;
                    let base = means.positions.get(pos).unwrap_or(&means.global);
                    let residual = values
                        .iter()
                        .zip(base.iter())
                        .map(|(&yi, &bi)| yi - bi)
                        .collect::<Vec<_>>();
                    let z = basis.residual_to_z(&residual);
                    let coords = pca_basis.coordinates_with_rank(&z, codebook.config.k);
                    let codes = codebook.quantize_indices_for_stratum(&coords, stratum);
                    let quantized_coords =
                        codebook.quantize_from_indices_for_stratum(&codes, stratum);
                    oracle_codes.push(codes);
                    let z_projected = pca_basis.reconstruct_from_coordinates(&quantized_coords);
                    let residual_projected = basis.z_to_residual(&z_projected);
                    let projected = residual_projected
                        .into_iter()
                        .zip(base.iter())
                        .map(|(ri, &bi)| ri + bi)
                        .collect::<Vec<_>>();
                    for (&original, &recon) in values.iter().zip(projected.iter()) {
                        let delta = original as f64 - recon as f64;
                        pre_sq += delta * delta;
                    }
                    let delta = values
                        .iter()
                        .zip(projected.iter())
                        .map(|(&original, &recon)| original as f64 - recon as f64)
                        .collect::<Vec<_>>();
                    visible_sq += basis.visible_sq_norm(&delta);
                    count += 1;
                    replacement.extend_from_slice(&projected);
                }
                metrics = Some(RoundtripPatchMetrics {
                    pre_wo_l2: (pre_sq / count.max(1) as f64).sqrt(),
                    wo_visible_l2: (visible_sq / count.max(1) as f64).sqrt(),
                });
                let replacement = Array2::from_shape_vec((pre_o.nrows(), head_dim), replacement)?;
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
                "forward failed at layer {layer} during oracle PQ L{} H{} K={} groups={} bits={}",
                head.layer,
                head.head,
                codebook.config.k,
                codebook.config.groups,
                codebook.config.bits_per_group
            )
            .into());
        }

        remove_layer_tensors(weights, inserted);
    }

    Ok((
        h,
        metrics.ok_or("oracle PQ did not visit target layer")?,
        oracle_codes,
    ))
}

fn forward_q4k_oracle_pq_mode_d_head(
    weights: &mut larql_inference::ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    head: HeadId,
    basis: &WoRoundtripBasis,
    pca_basis: &ZPcaBasis,
    means: &StaticHeadMeans,
    codebook: &PqCodebook,
    mode_d_table: &ModeDTable,
    stratum: &str,
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
                let mut replacement_delta = Vec::with_capacity(pre_o.nrows() * weights.hidden_size);
                for pos in 0..pre_o.nrows() {
                    let row = pre_o.slice(s![pos, start..end]);
                    let values = row
                        .as_slice()
                        .ok_or("pre-W_O head row was not contiguous during Mode D PQ")?;
                    let base = means.positions.get(pos).unwrap_or(&means.global);
                    let residual = values
                        .iter()
                        .zip(base.iter())
                        .map(|(&yi, &bi)| yi - bi)
                        .collect::<Vec<_>>();
                    let z = basis.residual_to_z(&residual);
                    let coords = pca_basis.coordinates_with_rank(&z, codebook.config.k);
                    let codes = codebook.quantize_indices_for_stratum(&coords, stratum);
                    let delta =
                        mode_d_table.delta_for_position_codes_with_stratum(pos, &codes, stratum);
                    replacement_delta.extend_from_slice(&delta);
                }
                let replacement_delta = Array2::from_shape_vec(
                    (pre_o.nrows(), weights.hidden_size),
                    replacement_delta,
                )?;
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
                "forward failed at layer {layer} during Mode D oracle PQ L{} H{} K={} groups={} bits={}",
                head.layer,
                head.head,
                codebook.config.k,
                codebook.config.groups,
                codebook.config.bits_per_group
            )
            .into());
        }

        remove_layer_tensors(weights, inserted);
    }

    Ok(h)
}

fn forward_q4k_predicted_address_mode_d_head(
    weights: &mut larql_inference::ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    head: HeadId,
    mode_d_table: &ModeDTable,
    predicted_codes_by_position: &[Vec<usize>],
    stratum: &str,
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
                let mut replacement_delta = Vec::with_capacity(h.nrows() * weights.hidden_size);
                for pos in 0..h.nrows() {
                    let codes = predicted_codes_by_position
                        .get(pos)
                        .ok_or("missing predicted address for sequence position")?;
                    let delta =
                        mode_d_table.delta_for_position_codes_with_stratum(pos, codes, stratum);
                    replacement_delta.extend_from_slice(&delta);
                }
                let replacement_delta =
                    Array2::from_shape_vec((h.nrows(), weights.hidden_size), replacement_delta)?;
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
                "forward failed at layer {layer} during predicted-address Mode D L{} H{}",
                head.layer, head.head
            )
            .into());
        }

        remove_layer_tensors(weights, inserted);
    }

    Ok(h)
}

fn capture_layer_input_hidden(
    weights: &mut larql_inference::ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    target_layer: usize,
) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    let mut h = embed_tokens_pub(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
    let mut kv_cache: HashMap<usize, SharedKV> = HashMap::new();

    for layer in 0..target_layer {
        let inserted = insert_q4k_layer_tensors(weights, index, layer)?;
        let step = {
            let shared_kv = weights
                .arch
                .kv_shared_source_layer(layer)
                .and_then(|src| kv_cache.get(&src));
            let ffn = WeightFfn { weights };
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
        };
        if let Some((h_new, kv_out)) = step {
            h = h_new;
            if let Some(kv) = kv_out {
                kv_cache.insert(layer, kv);
            }
        } else {
            remove_layer_tensors(weights, inserted);
            return Err(format!("layer {layer} returned no output").into());
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
