use std::collections::HashMap;

use larql_inference::attention::run_attention_block_with_pre_o;
use larql_inference::forward::ple::precompute_per_layer_inputs;
use larql_inference::forward::{embed_tokens_pub, run_layer_with_ffn};
use larql_inference::{encode_prompt, WeightFfn};
use larql_vindex::VectorIndex;
use ndarray::{s, ArrayView1};

use super::address::{
    address_feature_key, address_probe_names, lsh_bucket, predict_code_from_hyperplanes,
    train_binary_hyperplane, AddressLshGroupModel, AddressProbeModel, AddressSupervisedGroupModel,
};
use super::basis::{WoRoundtripBasis, ZPcaBasis};
use super::metrics::argmax_usize;
use super::pq::PqCodebook;
use super::runtime::{insert_q4k_layer_tensors, remove_layer_tensors};
use super::stats::StaticHeadMeans;
use super::types::{HeadId, PqConfig, PromptRecord};

type SampleVisitResult = Result<(), Box<dyn std::error::Error>>;

pub(super) fn fit_address_probe_models(
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
    let mut key_counts: HashMap<(HeadId, PqConfig, String, usize, String), Vec<usize>> =
        HashMap::new();
    let mut majority_counts: HashMap<(HeadId, PqConfig, usize), Vec<usize>> = HashMap::new();

    visit_code_samples(
        weights,
        index,
        tokenizer,
        prompts,
        heads,
        bases,
        means,
        pca_bases,
        codebooks,
        "address-fit",
        false,
        |head, config, pos, codes, token_ids, stratum, _| {
            for (group, &code) in codes.iter().enumerate() {
                let levels = 1usize << config.bits_per_group;
                let counts = majority_counts
                    .entry((head, config, group))
                    .or_insert_with(|| vec![0; levels]);
                counts[code] += 1;
                for name in &names {
                    let key = address_feature_key(name, token_ids, stratum, pos);
                    let counts = key_counts
                        .entry((head, config, (*name).to_string(), group, key))
                        .or_insert_with(|| vec![0; levels]);
                    counts[code] += 1;
                }
            }
            Ok(())
        },
    )?;

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

pub(super) fn fit_address_lsh_group_models(
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
    let mut majority_counts: HashMap<(HeadId, PqConfig, usize), Vec<usize>> = HashMap::new();
    let mut bucket_counts: HashMap<(HeadId, PqConfig, usize, u64, usize), Vec<usize>> =
        HashMap::new();

    visit_code_samples(
        weights,
        index,
        tokenizer,
        prompts,
        heads,
        bases,
        means,
        pca_bases,
        codebooks,
        "lsh-fit",
        true,
        |head, config, _pos, codes, _token_ids, _stratum, input_row| {
            let input_row = input_row.ok_or("missing layer-input row during LSH address fit")?;
            for (group, &code) in codes.iter().enumerate() {
                let levels = 1usize << config.bits_per_group;
                let counts = majority_counts
                    .entry((head, config, group))
                    .or_insert_with(|| vec![0; levels]);
                counts[code] += 1;
            }
            for &group in selected_groups {
                let code = codes[group];
                for seed in 0..seeds {
                    let bucket = lsh_bucket(ArrayView1::from(input_row), seed as u64, bits);
                    let levels = 1usize << config.bits_per_group;
                    let counts = bucket_counts
                        .entry((head, config, group, seed as u64, bucket))
                        .or_insert_with(|| vec![0; levels]);
                    counts[code] += 1;
                }
            }
            Ok(())
        },
    )?;

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

pub(super) fn fit_address_supervised_group_models(
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
    let mut majority_counts: HashMap<(HeadId, PqConfig, usize), Vec<usize>> = HashMap::new();
    let mut samples: HashMap<(HeadId, PqConfig), Vec<(Vec<f32>, Vec<usize>)>> = HashMap::new();

    visit_code_samples(
        weights,
        index,
        tokenizer,
        prompts,
        heads,
        bases,
        means,
        pca_bases,
        codebooks,
        "supervised-fit",
        true,
        |head, config, _pos, codes, _token_ids, _stratum, input_row| {
            let input_row =
                input_row.ok_or("missing layer-input row during supervised address fit")?;
            for (group, &code) in codes.iter().enumerate() {
                let levels = 1usize << config.bits_per_group;
                let counts = majority_counts
                    .entry((head, config, group))
                    .or_insert_with(|| vec![0; levels]);
                counts[code] += 1;
            }
            samples
                .entry((head, config))
                .or_default()
                .push((input_row.to_vec(), codes.to_vec()));
            Ok(())
        },
    )?;

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

pub(super) fn fit_majority_codes_for_codebooks(
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
    let mut majority_counts: HashMap<(HeadId, PqConfig, usize), Vec<usize>> = HashMap::new();

    visit_code_samples(
        weights,
        index,
        tokenizer,
        prompts,
        heads,
        bases,
        means,
        pca_bases,
        codebooks,
        "majority-fit",
        false,
        |head, config, _pos, codes, _token_ids, _stratum, _| {
            for (group, &code) in codes.iter().enumerate() {
                let levels = 1usize << config.bits_per_group;
                let counts = majority_counts
                    .entry((head, config, group))
                    .or_insert_with(|| vec![0; levels]);
                counts[code] += 1;
            }
            Ok(())
        },
    )?;

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

fn visit_code_samples<F>(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    tokenizer: &tokenizers::Tokenizer,
    prompts: &[PromptRecord],
    heads: &[HeadId],
    bases: &HashMap<HeadId, WoRoundtripBasis>,
    means: &HashMap<HeadId, StaticHeadMeans>,
    pca_bases: &HashMap<HeadId, ZPcaBasis>,
    codebooks: &HashMap<(HeadId, PqConfig), PqCodebook>,
    label_prefix: &str,
    with_layer_input: bool,
    mut visit: F,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: FnMut(HeadId, PqConfig, usize, &[usize], &[u32], &str, Option<&[f32]>) -> SampleVisitResult,
{
    let mut heads_by_layer: HashMap<usize, Vec<HeadId>> = HashMap::new();
    for head in heads {
        heads_by_layer.entry(head.layer).or_default().push(*head);
    }

    for (prompt_idx, record) in prompts.iter().enumerate() {
        let label = record
            .id
            .as_deref()
            .or(record.stratum.as_deref())
            .unwrap_or("prompt");
        eprintln!(
            "  {} [{}/{}] {}",
            label_prefix,
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
                let layer_input = if with_layer_input {
                    Some(h.clone())
                } else {
                    None
                };
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
                            .ok_or("pre-W_O head row was not contiguous during address fit")?;
                        let base = head_means.positions.get(pos).unwrap_or(&head_means.global);
                        let residual = values
                            .iter()
                            .zip(base.iter())
                            .map(|(&yi, &bi)| yi - bi)
                            .collect::<Vec<_>>();
                        let z = basis.residual_to_z(&residual);
                        let input_row = layer_input.as_ref().map(|input| input.row(pos).to_vec());
                        for ((_, config), codebook) in &head_codebooks {
                            let coords = pca_basis.coordinates_with_rank(&z, config.k);
                            let codes = codebook.quantize_indices_for_stratum(&coords, stratum);
                            visit(
                                *head,
                                *config,
                                pos,
                                &codes,
                                &token_ids,
                                stratum,
                                input_row.as_deref(),
                            )?;
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

    Ok(())
}
