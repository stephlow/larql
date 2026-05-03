use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use larql_inference::attention::run_attention_block_with_pre_o;
use larql_inference::forward::ple::precompute_per_layer_inputs;
use larql_inference::forward::{embed_tokens_pub, run_layer_with_ffn};
use larql_inference::{encode_prompt, WeightFfn};
use larql_vindex::{
    load_model_weights_q4k, load_vindex_tokenizer, SilentLoadCallbacks, VectorIndex,
};
use ndarray::{s, Array2};

use super::basis::{build_roundtrip_bases, fit_z_pca_bases, WoRoundtripBasis, ZPcaBasis};
use super::input::{
    limit_prompts_per_stratum, load_prompts, parse_head_spec, parse_pq_configs, parse_usize_list,
    split_prompt_records,
};
use super::metrics::{
    argmax, bool_rate, kl_logp, log_softmax, mean, percentile, token_prob, top_k_indices,
};
use super::oracle_pq_fit::fit_pq_codebooks;
use super::oracle_pq_forward::{final_logits, forward_q4k_oracle_pq_mode_d_head};
use super::oracle_pq_mode_d::materialize_mode_d_tables;
use super::pq::{kmeans_centroids, nearest_centroid_index, ModeDTable, PqCodebook};
use super::reports::{
    OraclePqExceptionHeadReport, OraclePqExceptionPointReport, OraclePqExceptionPromptReport,
    OraclePqExceptionReport,
};
use super::runtime::{insert_q4k_layer_tensors, remove_layer_tensors};
use super::static_replace::fit_static_means;
use super::stats::StaticHeadMeans;
use super::types::{HeadId, PqConfig, PromptRecord};

#[derive(Args)]
pub(super) struct OraclePqExceptionArgs {
    /// Self-contained Q4K vindex directory.
    #[arg(long)]
    index: PathBuf,

    /// JSONL prompt file. Each line must include at least {"prompt": "..."}.
    #[arg(long)]
    prompts: PathBuf,

    /// Output directory.
    #[arg(long)]
    out: PathBuf,

    /// Explicit heads as layer:head comma list, e.g. 20:6.
    #[arg(long)]
    heads: String,

    /// Base PQ config as K:groups:bits, e.g. 192:48:4.
    #[arg(long)]
    base_config: String,

    /// Comma-separated exception edit counts.
    #[arg(long, default_value = "4,8,16,32")]
    exception_edits: String,

    /// Comma-separated top-error fractions used to fit exception edits.
    #[arg(long, default_value = "1.0,0.25,0.1")]
    tail_fracs: String,

    /// Training-position selector for exception fitting: residual-error or prompt-kl.
    #[arg(long, default_value = "residual-error")]
    tail_selector: String,

    /// Relative singular value cutoff for retained W_O-visible directions.
    #[arg(long, default_value_t = 1e-6)]
    sigma_rel_cutoff: f64,

    /// Lloyd iterations for the base PQ codebook.
    #[arg(long, default_value_t = 25)]
    pq_iters: usize,

    /// Lloyd iterations for exception residual catalogues.
    #[arg(long, default_value_t = 25)]
    exception_iters: usize,

    /// Limit prompts for bounded oracle runs.
    #[arg(long)]
    max_prompts: Option<usize>,

    /// Keep at most N prompts per stratum after loading.
    #[arg(long)]
    max_per_stratum: Option<usize>,

    /// Evaluate only prompts where prompt_index % eval_mod == eval_offset.
    /// The remaining prompts are used to fit static means, PCA, PQ, and exceptions.
    #[arg(long)]
    eval_mod: Option<usize>,

    /// Held-out modulo offset used with --eval-mod.
    #[arg(long, default_value_t = 0)]
    eval_offset: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ExceptionKey {
    head: HeadId,
    edits: usize,
    tail_frac_key: u64,
}

#[derive(Debug, Clone)]
struct ExceptionCatalog {
    edits: usize,
    tail_frac: f64,
    train_error_samples: usize,
    train_error_samples_used: usize,
    centroids: Vec<Vec<f64>>,
}

#[derive(Debug, Clone)]
struct ErrorSample {
    score: f64,
    sq_norm: f64,
    values: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TailSelector {
    ResidualError,
    PromptKl,
}

impl TailSelector {
    fn parse(value: &str) -> Result<Self, Box<dyn std::error::Error>> {
        match value {
            "residual-error" => Ok(Self::ResidualError),
            "prompt-kl" => Ok(Self::PromptKl),
            other => Err(format!(
                "invalid --tail-selector '{other}', expected residual-error or prompt-kl"
            )
            .into()),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::ResidualError => "residual-error",
            Self::PromptKl => "prompt-kl",
        }
    }
}

pub(super) fn run_oracle_pq_exception(
    args: OraclePqExceptionArgs,
) -> Result<(), Box<dyn std::error::Error>> {
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
        return Err("ov-rd oracle-pq-exception currently supports dense FFN vindexes only".into());
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
        return Err("no heads selected for oracle PQ exception".into());
    }
    let mut base_configs = parse_pq_configs(&args.base_config)?;
    if base_configs.len() != 1 {
        return Err("--base-config must contain exactly one K:groups:bits config".into());
    }
    let base_config = base_configs.remove(0);
    let mut exception_edits = parse_usize_list(&args.exception_edits)?;
    exception_edits.sort_unstable();
    exception_edits.dedup();
    if exception_edits.is_empty() || exception_edits.iter().any(|&edits| edits == 0) {
        return Err("--exception-edits values must be greater than zero".into());
    }
    let mut tail_fracs = parse_f64_list(&args.tail_fracs)?;
    tail_fracs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    tail_fracs.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON);
    if tail_fracs.is_empty()
        || tail_fracs
            .iter()
            .any(|&frac| !(frac.is_finite() && frac > 0.0 && frac <= 1.0))
    {
        return Err("--tail-fracs values must be finite and in (0, 1]".into());
    }
    let tail_selector = TailSelector::parse(&args.tail_selector)?;

    let mut prompts = load_prompts(&args.prompts, args.max_prompts)?;
    if let Some(max_per_stratum) = args.max_per_stratum {
        prompts = limit_prompts_per_stratum(prompts, max_per_stratum);
    }
    let prompts_seen = prompts.len();
    let (fit_prompts, eval_prompts) = if let Some(eval_mod) = args.eval_mod {
        split_prompt_records(&prompts, eval_mod, args.eval_offset)?
    } else {
        (prompts.clone(), prompts)
    };
    eprintln!("Selected heads: {:?}", selected_heads);
    eprintln!("Base PQ config: {:?}", base_config);
    eprintln!("Exception edits: {:?}", exception_edits);
    eprintln!("Tail fractions: {:?}", tail_fracs);
    eprintln!("Tail selector: {}", tail_selector.as_str());
    eprintln!("Prompts: {}", prompts_seen);

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

    eprintln!("Fitting base product quantizer");
    let base_codebooks = fit_pq_codebooks(
        &mut weights,
        &index,
        &tokenizer,
        &fit_prompts,
        &selected_heads,
        &bases,
        &means,
        &pca_bases,
        &[base_config],
        args.pq_iters,
        &[],
    )?;

    eprintln!("Materializing base Mode D tables");
    let base_tables = materialize_mode_d_tables(
        &mut weights,
        &index,
        &selected_heads,
        &bases,
        &means,
        &pca_bases,
        &base_codebooks,
        &[],
    )?;
    let w_o_heads = copy_w_o_heads(&mut weights, &index, &selected_heads)?;
    let prompt_scores = if tail_selector == TailSelector::PromptKl {
        eprintln!("Measuring fit-prompt base-PQ KL for exception selection");
        measure_fit_prompt_base_pq_kl(
            &mut weights,
            &index,
            &tokenizer,
            &fit_prompts,
            &selected_heads,
            &bases,
            &means,
            &pca_bases,
            &base_codebooks,
            &base_tables,
            base_config,
        )?
    } else {
        HashMap::new()
    };

    eprintln!("Fitting exception residual catalogues");
    let exception_catalogs = fit_exception_catalogs(
        &mut weights,
        &index,
        &tokenizer,
        &fit_prompts,
        &selected_heads,
        &bases,
        &means,
        &pca_bases,
        &base_codebooks,
        &base_tables,
        &w_o_heads,
        base_config,
        &exception_edits,
        &tail_fracs,
        tail_selector,
        &prompt_scores,
        args.exception_iters,
    )?;

    let mut accumulators: HashMap<ExceptionKey, PqExceptionAccumulator> = HashMap::new();
    for head in &selected_heads {
        for &edits in &exception_edits {
            for &tail_frac in &tail_fracs {
                accumulators.insert(
                    ExceptionKey {
                        head: *head,
                        edits,
                        tail_frac_key: tail_frac_key(tail_frac),
                    },
                    PqExceptionAccumulator::new(),
                );
            }
        }
    }

    for (prompt_idx, record) in eval_prompts.iter().enumerate() {
        let label = prompt_label(record);
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
            let basis = bases
                .get(head)
                .ok_or_else(|| format!("missing basis for L{}H{}", head.layer, head.head))?;
            let pca_basis = pca_bases
                .get(head)
                .ok_or_else(|| format!("missing PCA basis for L{}H{}", head.layer, head.head))?;
            let head_means = means
                .get(head)
                .ok_or_else(|| format!("missing means for L{}H{}", head.layer, head.head))?;
            let codebook = base_codebooks.get(&(*head, base_config)).ok_or_else(|| {
                format!("missing base codebook for L{}H{}", head.layer, head.head)
            })?;
            let table = base_tables
                .get(&(*head, base_config))
                .ok_or_else(|| format!("missing base table for L{}H{}", head.layer, head.head))?;
            let w_o_head = w_o_heads
                .get(head)
                .ok_or_else(|| format!("missing W_O head for L{}H{}", head.layer, head.head))?;
            for &edits in &exception_edits {
                for &tail_frac in &tail_fracs {
                    let key = ExceptionKey {
                        head: *head,
                        edits,
                        tail_frac_key: tail_frac_key(tail_frac),
                    };
                    let catalog = exception_catalogs.get(&key).ok_or_else(|| {
                        format!(
                            "missing exception catalog for L{}H{} edits={} tail={}",
                            head.layer, head.head, edits, tail_frac
                        )
                    })?;
                    let exception_hidden = forward_q4k_oracle_pq_exception_head(
                        &mut weights,
                        &token_ids,
                        &index,
                        *head,
                        basis,
                        pca_basis,
                        head_means,
                        codebook,
                        table,
                        w_o_head,
                        catalog,
                        stratum,
                    )?;
                    let exception_logits = final_logits(&weights, &exception_hidden);
                    let exception_logp = log_softmax(&exception_logits);
                    let kl = kl_logp(&baseline_logp, &exception_logp);
                    let exception_top1 = argmax(&exception_logits);
                    let exception_top5 = top_k_indices(&exception_logits, 5);
                    let exception_top2 = top_k_indices(&exception_logits, 2);
                    let exception_top2_token =
                        exception_top2.get(1).copied().unwrap_or(exception_top1);
                    let exception_top1_prob = token_prob(&exception_logp, exception_top1);
                    let exception_top2_prob = token_prob(&exception_logp, exception_top2_token);
                    let exception_top1_margin = exception_top1_prob - exception_top2_prob;
                    let exception_prob_of_baseline_top1 =
                        token_prob(&exception_logp, baseline_top1);
                    accumulators
                        .get_mut(&key)
                        .expect("exception accumulator missing")
                        .add(OraclePqExceptionPromptReport {
                            id: label.to_string(),
                            stratum: stratum.to_string(),
                            kl,
                            delta_cross_entropy_bits: kl / std::f64::consts::LN_2,
                            baseline_top1,
                            exception_top1,
                            top1_agree: baseline_top1 == exception_top1,
                            baseline_top1_in_exception_top5: exception_top5
                                .contains(&baseline_top1),
                            baseline_top1_prob,
                            baseline_top2: baseline_top2_token,
                            baseline_top2_prob,
                            baseline_top1_margin,
                            exception_top1_prob,
                            exception_prob_of_baseline_top1,
                            exception_top1_margin,
                        });
                }
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
        let mut points = Vec::new();
        for &edits in &exception_edits {
            for &tail_frac in &tail_fracs {
                let key = ExceptionKey {
                    head: *head,
                    edits,
                    tail_frac_key: tail_frac_key(tail_frac),
                };
                let acc = accumulators
                    .remove(&key)
                    .expect("exception accumulator missing at finish");
                let catalog = exception_catalogs
                    .get(&key)
                    .expect("exception catalog missing at finish");
                points.push(acc.finish(base_config, catalog, weights.hidden_size));
            }
        }
        let static_train_samples = means.get(head).map(|m| m.count).unwrap_or(0);
        head_reports.push(OraclePqExceptionHeadReport {
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

    let report = OraclePqExceptionReport {
        index: args.index.display().to_string(),
        prompt_file: args.prompts.display().to_string(),
        prompts_seen,
        train_prompts_seen: fit_prompts.len(),
        eval_prompts_seen: eval_prompts.len(),
        max_per_stratum: args.max_per_stratum,
        eval_mod: args.eval_mod,
        eval_offset: args.eval_offset,
        static_base: "position_mean".to_string(),
        base_config,
        exception_edits,
        tail_fracs,
        tail_selector: tail_selector.as_str().to_string(),
        sigma_rel_cutoff: args.sigma_rel_cutoff,
        pq_iters: args.pq_iters,
        exception_iters: args.exception_iters,
        selected_heads,
        heads: head_reports,
    };

    let out_path = args.out.join("oracle_pq_exception.json");
    let file = std::fs::File::create(&out_path)?;
    serde_json::to_writer_pretty(file, &report)?;
    eprintln!("Wrote {}", out_path.display());

    Ok(())
}

fn fit_exception_catalogs(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    tokenizer: &tokenizers::Tokenizer,
    prompts: &[PromptRecord],
    heads: &[HeadId],
    bases: &HashMap<HeadId, WoRoundtripBasis>,
    means: &HashMap<HeadId, StaticHeadMeans>,
    pca_bases: &HashMap<HeadId, ZPcaBasis>,
    codebooks: &HashMap<(HeadId, PqConfig), PqCodebook>,
    tables: &HashMap<(HeadId, PqConfig), ModeDTable>,
    w_o_heads: &HashMap<HeadId, Vec<Vec<f32>>>,
    base_config: PqConfig,
    exception_edits: &[usize],
    tail_fracs: &[f64],
    tail_selector: TailSelector,
    prompt_scores: &HashMap<(HeadId, usize), f64>,
    iterations: usize,
) -> Result<HashMap<ExceptionKey, ExceptionCatalog>, Box<dyn std::error::Error>> {
    let mut heads_by_layer: HashMap<usize, Vec<HeadId>> = HashMap::new();
    for head in heads {
        heads_by_layer.entry(head.layer).or_default().push(*head);
    }
    let mut samples: HashMap<HeadId, Vec<ErrorSample>> = HashMap::new();
    for head in heads {
        samples.insert(*head, Vec::new());
    }

    for (prompt_idx, record) in prompts.iter().enumerate() {
        let label = prompt_label(record);
        eprintln!(
            "  exception-fit [{}/{}] {}",
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
                    let basis = bases.get(head).expect("basis pre-created");
                    let pca_basis = pca_bases.get(head).expect("PCA pre-created");
                    let head_means = means.get(head).expect("means pre-created");
                    let codebook = codebooks
                        .get(&(*head, base_config))
                        .expect("base codebook pre-created");
                    let table = tables
                        .get(&(*head, base_config))
                        .expect("base Mode D table pre-created");
                    let w_o_head = w_o_heads.get(head).expect("W_O head pre-copied");
                    let start = head.head * head_dim;
                    let end = start + head_dim;
                    for pos in 0..pre_o.nrows() {
                        let row = pre_o.slice(s![pos, start..end]);
                        let values = row
                            .as_slice()
                            .ok_or("pre-W_O head row was not contiguous during exception fit")?;
                        let base_delta = base_pq_delta(
                            values, basis, pca_basis, head_means, codebook, table, pos, stratum,
                        );
                        let true_delta = project_head_vector_to_hidden(w_o_head, values);
                        let error = true_delta
                            .iter()
                            .zip(base_delta.iter())
                            .map(|(&true_value, &base_value)| true_value as f64 - base_value as f64)
                            .collect::<Vec<_>>();
                        let sq_norm = error.iter().map(|value| value * value).sum::<f64>();
                        let score = match tail_selector {
                            TailSelector::ResidualError => sq_norm,
                            TailSelector::PromptKl => {
                                *prompt_scores.get(&(*head, prompt_idx)).unwrap_or(&0.0)
                            }
                        };
                        samples
                            .get_mut(head)
                            .expect("exception samples missing")
                            .push(ErrorSample {
                                score,
                                sq_norm,
                                values: error,
                            });
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

    let mut catalogs = HashMap::new();
    for head in heads {
        let mut head_samples = samples.remove(head).ok_or_else(|| {
            format!(
                "missing exception samples for L{}H{}",
                head.layer, head.head
            )
        })?;
        head_samples.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    b.sq_norm
                        .partial_cmp(&a.sq_norm)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        let total = head_samples.len();
        for &tail_frac in tail_fracs {
            let used = ((total as f64) * tail_frac).ceil() as usize;
            let used = used.clamp(1, total.max(1));
            let selected = head_samples
                .iter()
                .take(used)
                .map(|sample| sample.values.clone())
                .collect::<Vec<_>>();
            for &edits in exception_edits {
                let centroids = kmeans_centroids(&selected, edits, iterations);
                catalogs.insert(
                    ExceptionKey {
                        head: *head,
                        edits,
                        tail_frac_key: tail_frac_key(tail_frac),
                    },
                    ExceptionCatalog {
                        edits,
                        tail_frac,
                        train_error_samples: total,
                        train_error_samples_used: used,
                        centroids,
                    },
                );
            }
        }
    }

    Ok(catalogs)
}

fn measure_fit_prompt_base_pq_kl(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    tokenizer: &tokenizers::Tokenizer,
    prompts: &[PromptRecord],
    heads: &[HeadId],
    bases: &HashMap<HeadId, WoRoundtripBasis>,
    means: &HashMap<HeadId, StaticHeadMeans>,
    pca_bases: &HashMap<HeadId, ZPcaBasis>,
    codebooks: &HashMap<(HeadId, PqConfig), PqCodebook>,
    tables: &HashMap<(HeadId, PqConfig), ModeDTable>,
    base_config: PqConfig,
) -> Result<HashMap<(HeadId, usize), f64>, Box<dyn std::error::Error>> {
    let mut scores = HashMap::new();
    for (prompt_idx, record) in prompts.iter().enumerate() {
        let label = prompt_label(record);
        eprintln!(
            "  selector-fit [{}/{}] {}",
            prompt_idx + 1,
            prompts.len(),
            label
        );
        let token_ids = encode_prompt(tokenizer, &*weights.arch, &record.prompt)?;
        if token_ids.is_empty() {
            continue;
        }
        let stratum = record.stratum.as_deref().unwrap_or("unknown");
        let baseline_hidden =
            larql_inference::vindex::predict_q4k_hidden(weights, &token_ids, index, None);
        let baseline_logits = final_logits(weights, &baseline_hidden);
        let baseline_logp = log_softmax(&baseline_logits);
        for head in heads {
            let basis = bases
                .get(head)
                .ok_or_else(|| format!("missing basis for L{}H{}", head.layer, head.head))?;
            let pca_basis = pca_bases
                .get(head)
                .ok_or_else(|| format!("missing PCA basis for L{}H{}", head.layer, head.head))?;
            let head_means = means
                .get(head)
                .ok_or_else(|| format!("missing means for L{}H{}", head.layer, head.head))?;
            let codebook = codebooks.get(&(*head, base_config)).ok_or_else(|| {
                format!("missing base codebook for L{}H{}", head.layer, head.head)
            })?;
            let table = tables
                .get(&(*head, base_config))
                .ok_or_else(|| format!("missing base table for L{}H{}", head.layer, head.head))?;
            let pq_hidden = forward_q4k_oracle_pq_mode_d_head(
                weights, &token_ids, index, *head, basis, pca_basis, head_means, codebook, table,
                stratum,
            )?;
            let pq_logits = final_logits(weights, &pq_hidden);
            let pq_logp = log_softmax(&pq_logits);
            scores.insert((*head, prompt_idx), kl_logp(&baseline_logp, &pq_logp));
        }
    }
    Ok(scores)
}

fn forward_q4k_oracle_pq_exception_head(
    weights: &mut larql_inference::ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    head: HeadId,
    basis: &WoRoundtripBasis,
    pca_basis: &ZPcaBasis,
    means: &StaticHeadMeans,
    codebook: &PqCodebook,
    table: &ModeDTable,
    w_o_head: &[Vec<f32>],
    catalog: &ExceptionCatalog,
    stratum: &str,
) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    let hidden_size = weights.hidden_size;
    larql_inference::vindex::predict_q4k_hidden_with_mapped_head_residual_delta(
        weights,
        token_ids,
        index,
        head.layer,
        head.head,
        |original_head| {
            let mut replacement_delta = Vec::with_capacity(original_head.nrows() * hidden_size);
            for pos in 0..original_head.nrows() {
                let row = original_head.row(pos);
                let values = row
                    .as_slice()
                    .ok_or("pre-W_O head row was not contiguous during exception eval")?;
                let base_delta = base_pq_delta(
                    values, basis, pca_basis, means, codebook, table, pos, stratum,
                );
                let true_delta = project_head_vector_to_hidden(w_o_head, values);
                let error = true_delta
                    .iter()
                    .zip(base_delta.iter())
                    .map(|(&true_value, &base_value)| true_value as f64 - base_value as f64)
                    .collect::<Vec<_>>();
                let code = nearest_centroid_index(&error, &catalog.centroids);
                let exception = &catalog.centroids[code];
                for (&base, &extra) in base_delta.iter().zip(exception.iter()) {
                    replacement_delta.push(base + extra as f32);
                }
            }
            Array2::from_shape_vec((original_head.nrows(), hidden_size), replacement_delta)
                .map_err(|err| err.to_string())
        },
    )
    .map_err(Into::into)
}

fn base_pq_delta(
    values: &[f32],
    basis: &WoRoundtripBasis,
    pca_basis: &ZPcaBasis,
    means: &StaticHeadMeans,
    codebook: &PqCodebook,
    table: &ModeDTable,
    position: usize,
    stratum: &str,
) -> Vec<f32> {
    let base = means.positions.get(position).unwrap_or(&means.global);
    let residual = values
        .iter()
        .zip(base.iter())
        .map(|(&value, &mean)| value - mean)
        .collect::<Vec<_>>();
    let z = basis.residual_to_z(&residual);
    let coords = pca_basis.coordinates_with_rank(&z, codebook.config.k);
    let codes = codebook.quantize_indices_for_stratum(&coords, stratum);
    table.delta_for_position_codes_with_stratum(position, &codes, stratum)
}

fn copy_w_o_heads(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    heads: &[HeadId],
) -> Result<HashMap<HeadId, Vec<Vec<f32>>>, Box<dyn std::error::Error>> {
    let mut heads_by_layer: HashMap<usize, Vec<HeadId>> = HashMap::new();
    for head in heads {
        heads_by_layer.entry(head.layer).or_default().push(*head);
    }
    let mut out = HashMap::new();
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
            let rows = (0..w_o_head.nrows())
                .map(|row| {
                    (0..w_o_head.ncols())
                        .map(|col| w_o_head[[row, col]])
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            out.insert(head, rows);
        }
        remove_layer_tensors(weights, inserted);
    }
    Ok(out)
}

fn project_head_vector_to_hidden(w_o_head: &[Vec<f32>], values: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; w_o_head.len()];
    for (row_idx, row) in w_o_head.iter().enumerate() {
        let mut sum = 0.0f32;
        for (&value, &weight) in values.iter().zip(row.iter()) {
            sum += value * weight;
        }
        out[row_idx] = sum;
    }
    out
}

#[derive(Debug)]
struct PqExceptionAccumulator {
    prompts: Vec<OraclePqExceptionPromptReport>,
}

impl PqExceptionAccumulator {
    fn new() -> Self {
        Self {
            prompts: Vec::new(),
        }
    }

    fn add(&mut self, prompt: OraclePqExceptionPromptReport) {
        self.prompts.push(prompt);
    }

    fn finish(
        self,
        base_config: PqConfig,
        catalog: &ExceptionCatalog,
        hidden_dim: usize,
    ) -> OraclePqExceptionPointReport {
        let kls = self.prompts.iter().map(|p| p.kl).collect::<Vec<_>>();
        let levels = 1usize << base_config.bits_per_group;
        let base_bytes = base_config.groups * levels * hidden_dim * 2;
        let exception_bytes = catalog.edits * hidden_dim * 2;
        let exception_bits = catalog.edits.next_power_of_two().trailing_zeros() as usize;
        let base_bits = base_config.groups * base_config.bits_per_group;
        OraclePqExceptionPointReport {
            exception_edits: catalog.edits,
            tail_frac: catalog.tail_frac,
            train_error_samples: catalog.train_error_samples,
            train_error_samples_used: catalog.train_error_samples_used,
            base_address_bits: base_bits,
            exception_address_bits: exception_bits,
            total_address_bits: base_bits + exception_bits,
            base_table_bytes_bf16: base_bytes,
            exception_table_bytes_bf16: exception_bytes,
            total_table_bytes_bf16: base_bytes + exception_bytes,
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
                self.prompts
                    .iter()
                    .map(|p| p.baseline_top1_in_exception_top5),
            ),
            mean_baseline_top1_prob: mean(
                &self
                    .prompts
                    .iter()
                    .map(|p| p.baseline_top1_prob)
                    .collect::<Vec<_>>(),
            ),
            mean_exception_prob_of_baseline_top1: mean(
                &self
                    .prompts
                    .iter()
                    .map(|p| p.exception_prob_of_baseline_top1)
                    .collect::<Vec<_>>(),
            ),
            mean_baseline_top1_margin: mean(
                &self
                    .prompts
                    .iter()
                    .map(|p| p.baseline_top1_margin)
                    .collect::<Vec<_>>(),
            ),
            per_prompt: self.prompts,
        }
    }
}

fn parse_f64_list(spec: &str) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let mut values = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        values.push(part.parse()?);
    }
    Ok(values)
}

fn tail_frac_key(tail_frac: f64) -> u64 {
    (tail_frac * 1_000_000.0).round() as u64
}

fn prompt_label(record: &PromptRecord) -> &str {
    record
        .id
        .as_deref()
        .or(record.stratum.as_deref())
        .unwrap_or("prompt")
}
