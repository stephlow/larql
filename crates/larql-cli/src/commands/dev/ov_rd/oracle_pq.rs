use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use larql_inference::encode_prompt;
use larql_vindex::{
    load_model_weights_q4k, load_vindex_tokenizer, SilentLoadCallbacks, VectorIndex,
};
use std::collections::HashMap;

use super::basis::*;
use super::input::*;
use super::metrics::*;
use super::oracle_pq_address::{
    fit_address_lsh_group_models, fit_address_probe_models, fit_address_supervised_group_models,
    fit_majority_codes_for_codebooks,
};
use super::oracle_pq_eval::evaluate_predicted_address;
use super::oracle_pq_fit::fit_pq_codebooks;
use super::oracle_pq_forward::{
    capture_layer_input_hidden, final_logits, forward_q4k_oracle_pq_head,
    forward_q4k_oracle_pq_mode_d_head,
};
use super::oracle_pq_mode_d::{corruption_keep_values, materialize_mode_d_tables};
use super::oracle_pq_reports::OraclePqPointAccumulator;
use super::oracle_pq_stability::measure_code_stability;
use super::reports::*;
use super::static_replace::fit_static_means;
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
                            let prompt_report = evaluate_predicted_address(
                                &mut weights,
                                &token_ids,
                                &index,
                                *head,
                                mode_d_table,
                                &predicted_codes_by_position,
                                stratum,
                                label,
                                &baseline_logp,
                                baseline_top1,
                                &oracle_codes_by_position,
                            )?;
                            accumulators
                                .get_mut(&(*head, config))
                                .expect("oracle PQ accumulator missing")
                                .add_address_probe(
                                    &probe_model.name,
                                    &probe_model.selected_group_keys,
                                    prompt_report,
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
                                let prompt_report = evaluate_predicted_address(
                                    &mut weights,
                                    &token_ids,
                                    &index,
                                    *head,
                                    mode_d_table,
                                    &predicted_codes_by_position,
                                    stratum,
                                    label,
                                    &baseline_logp,
                                    baseline_top1,
                                    &oracle_codes_by_position,
                                )?;
                                accumulators
                                    .get_mut(&(*head, config))
                                    .expect("oracle PQ accumulator missing")
                                    .add_address_probe(
                                        &probe_name,
                                        &probe_model.selected_group_keys,
                                        prompt_report,
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
                        let prompt_report = evaluate_predicted_address(
                            &mut weights,
                            &token_ids,
                            &index,
                            *head,
                            mode_d_table,
                            &predicted_codes_by_position,
                            stratum,
                            label,
                            &baseline_logp,
                            baseline_top1,
                            &oracle_codes_by_position,
                        )?;
                        accumulators
                            .get_mut(&(*head, config))
                            .expect("oracle PQ accumulator missing")
                            .add_address_group_importance(replaced_group, prompt_report);
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
                        let prompt_report = evaluate_predicted_address(
                            &mut weights,
                            &token_ids,
                            &index,
                            *head,
                            mode_d_table,
                            &predicted_codes_by_position,
                            stratum,
                            label,
                            &baseline_logp,
                            baseline_top1,
                            &oracle_codes_by_position,
                        )?;
                        accumulators
                            .get_mut(&(*head, config))
                            .expect("oracle PQ accumulator missing")
                            .add_address_probe(&probe_name, &selected_group_keys, prompt_report);
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
                        let prompt_report = evaluate_predicted_address(
                            &mut weights,
                            &token_ids,
                            &index,
                            *head,
                            mode_d_table,
                            &predicted_codes_by_position,
                            stratum,
                            label,
                            &baseline_logp,
                            baseline_top1,
                            &oracle_codes_by_position,
                        )?;
                        accumulators
                            .get_mut(&(*head, config))
                            .expect("oracle PQ accumulator missing")
                            .add_address_probe(&probe_name, &selected_group_keys, prompt_report);
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
                        let prompt_report = evaluate_predicted_address(
                            &mut weights,
                            &token_ids,
                            &index,
                            *head,
                            mode_d_table,
                            &predicted_codes_by_position,
                            stratum,
                            label,
                            &baseline_logp,
                            baseline_top1,
                            &oracle_codes_by_position,
                        )?;
                        accumulators
                            .get_mut(&(*head, config))
                            .expect("oracle PQ accumulator missing")
                            .add_address_corruption(oracle_groups_kept, prompt_report);
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
