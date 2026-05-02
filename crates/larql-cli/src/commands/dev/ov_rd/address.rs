use std::collections::HashMap;

use ndarray::{Array2, ArrayView1};

#[derive(Debug, Clone)]
pub(super) struct AddressProbeModel {
    pub(super) name: String,
    pub(super) group_majority: Vec<usize>,
    pub(super) group_maps: Vec<HashMap<String, usize>>,
    pub(super) group_train_accuracy: Vec<f64>,
    pub(super) selected_group_keys: Vec<String>,
}

impl AddressProbeModel {
    pub(super) fn predict_codes(
        &self,
        token_ids: &[u32],
        stratum: &str,
        position: usize,
    ) -> Vec<usize> {
        let key = address_feature_key(&self.name, token_ids, stratum, position);
        self.group_maps
            .iter()
            .enumerate()
            .map(|(group, map)| {
                map.get(&key)
                    .copied()
                    .unwrap_or_else(|| self.group_majority[group])
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
pub(super) struct AddressLshGroupModel {
    pub(super) groups: Vec<usize>,
    pub(super) bits: usize,
    pub(super) group_majority: Vec<usize>,
    pub(super) group_maps: Vec<HashMap<usize, usize>>,
    pub(super) group_seeds: Vec<u64>,
    pub(super) group_train_accuracy: Vec<f64>,
}

impl AddressLshGroupModel {
    pub(super) fn selected_group_keys(&self) -> Vec<String> {
        (0..self.group_majority.len())
            .map(|group| {
                if self.groups.contains(&group) {
                    format!(
                        "lsh{}bits_seed{}_train_acc_{:.3}",
                        self.bits, self.group_seeds[group], self.group_train_accuracy[group]
                    )
                } else {
                    "majority".to_string()
                }
            })
            .collect()
    }

    pub(super) fn predict_selected_groups(
        &self,
        layer_input: &Array2<f32>,
        position: usize,
        base_codes: &[usize],
    ) -> Vec<usize> {
        let mut codes = base_codes.to_vec();
        let row = layer_input.row(position);
        for &group in &self.groups {
            let bucket = lsh_bucket(row, self.group_seeds[group], self.bits);
            codes[group] = self.group_maps[group]
                .get(&bucket)
                .copied()
                .unwrap_or(self.group_majority[group]);
        }
        codes
    }
}

#[derive(Debug, Clone)]
pub(super) struct BinaryHyperplane {
    pub(super) weights: Vec<f32>,
    pub(super) bias: f32,
}

impl BinaryHyperplane {
    fn predict_bit(&self, row: ArrayView1<'_, f32>) -> bool {
        normalized_hyperplane_logit(row, &self.weights, self.bias) >= 0.0
    }
}

#[derive(Debug, Clone)]
pub(super) struct AddressSupervisedGroupModel {
    pub(super) groups: Vec<usize>,
    pub(super) bits_per_group: usize,
    pub(super) epochs: usize,
    pub(super) lr: f32,
    pub(super) l2: f32,
    pub(super) group_majority: Vec<usize>,
    pub(super) group_hyperplanes: Vec<Vec<BinaryHyperplane>>,
    pub(super) group_train_accuracy: Vec<f64>,
}

impl AddressSupervisedGroupModel {
    pub(super) fn selected_group_keys(&self) -> Vec<String> {
        (0..self.group_majority.len())
            .map(|group| {
                if self.groups.contains(&group) {
                    format!(
                        "supervised{}bit_train_acc_{:.3}_epochs{}_lr{:.3}_l2_{:.1e}",
                        self.bits_per_group,
                        self.group_train_accuracy[group],
                        self.epochs,
                        self.lr,
                        self.l2
                    )
                } else {
                    "majority".to_string()
                }
            })
            .collect()
    }

    pub(super) fn predict_selected_groups(
        &self,
        layer_input: &Array2<f32>,
        position: usize,
        base_codes: &[usize],
    ) -> Vec<usize> {
        let mut codes = base_codes.to_vec();
        let row = layer_input.row(position);
        for &group in &self.groups {
            let mut code = 0usize;
            for (bit, hyperplane) in self.group_hyperplanes[group].iter().enumerate() {
                if hyperplane.predict_bit(row) {
                    code |= 1usize << bit;
                }
            }
            codes[group] = code;
        }
        codes
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct AddressMatchSummary {
    pub(super) groups_correct: usize,
    pub(super) groups_total: usize,
    pub(super) exact_address_match: bool,
}

pub(super) fn address_probe_names() -> Vec<&'static str> {
    vec![
        "position",
        "stratum",
        "position_stratum",
        "token_id",
        "prev_token_id",
        "token_bigram",
        "position_stratum_token",
    ]
}

pub(super) fn address_feature_key(
    name: &str,
    token_ids: &[u32],
    stratum: &str,
    position: usize,
) -> String {
    let token = token_ids.get(position).copied().unwrap_or(0);
    let prev = if position == 0 {
        u32::MAX
    } else {
        token_ids.get(position - 1).copied().unwrap_or(0)
    };
    match name {
        "position" => format!("p:{position}"),
        "stratum" => format!("s:{stratum}"),
        "position_stratum" => format!("p:{position}|s:{stratum}"),
        "token_id" => format!("t:{token}"),
        "prev_token_id" => format!("pt:{prev}"),
        "token_bigram" => format!("pt:{prev}|t:{token}"),
        "position_stratum_token" => format!("p:{position}|s:{stratum}|t:{token}"),
        _ => format!("p:{position}"),
    }
}

pub(super) fn lsh_bucket(row: ArrayView1<'_, f32>, seed: u64, bits: usize) -> usize {
    let mut bucket = 0usize;
    for bit in 0..bits {
        let mut sum = 0.0_f64;
        for (dim, &value) in row.iter().enumerate() {
            let hash = splitmix64(
                seed ^ ((bit as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                    ^ ((dim as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)),
            );
            let sign = if hash & 1 == 0 { -1.0 } else { 1.0 };
            sum += value as f64 * sign;
        }
        if sum >= 0.0 {
            bucket |= 1usize << bit;
        }
    }
    bucket
}

pub(super) fn train_binary_hyperplane(
    rows: &[&[f32]],
    labels: &[bool],
    dim: usize,
    epochs: usize,
    lr: f32,
    l2: f32,
) -> BinaryHyperplane {
    let mut weights = vec![0.0_f32; dim];
    let positives = labels.iter().filter(|&&label| label).count();
    let negatives = labels.len().saturating_sub(positives);
    let mut bias = if positives == 0 {
        -4.0
    } else if negatives == 0 {
        4.0
    } else {
        ((positives as f32 + 0.5) / (negatives as f32 + 0.5)).ln()
    };

    for _ in 0..epochs {
        for (row, &label) in rows.iter().zip(labels.iter()) {
            let scale = normalized_row_scale_slice(row);
            let dot = row
                .iter()
                .zip(weights.iter())
                .map(|(&x, &w)| (x / scale) * w)
                .sum::<f32>();
            let logit = (bias + dot).clamp(-30.0, 30.0);
            let prob = 1.0 / (1.0 + (-logit).exp());
            let target = if label { 1.0 } else { 0.0 };
            let grad = prob - target;
            for (w, &x) in weights.iter_mut().zip(row.iter()) {
                *w -= lr * (grad * (x / scale) + l2 * *w);
            }
            bias -= lr * grad;
        }
    }

    BinaryHyperplane { weights, bias }
}

pub(super) fn predict_code_from_hyperplanes(
    row: &[f32],
    hyperplanes: &[BinaryHyperplane],
) -> usize {
    let scale = normalized_row_scale_slice(row);
    let mut code = 0usize;
    for (bit, hyperplane) in hyperplanes.iter().enumerate() {
        let dot = row
            .iter()
            .zip(hyperplane.weights.iter())
            .map(|(&x, &w)| (x / scale) * w)
            .sum::<f32>();
        if hyperplane.bias + dot >= 0.0 {
            code |= 1usize << bit;
        }
    }
    code
}

pub(super) fn address_match_report(
    oracle_codes_by_position: &[Vec<usize>],
    predicted_codes_by_position: &[Vec<usize>],
) -> AddressMatchSummary {
    let mut groups_correct = 0usize;
    let mut groups_total = 0usize;
    let mut exact_address_match = true;
    for (oracle, predicted) in oracle_codes_by_position
        .iter()
        .zip(predicted_codes_by_position.iter())
    {
        if oracle != predicted {
            exact_address_match = false;
        }
        for (&oracle_code, &predicted_code) in oracle.iter().zip(predicted.iter()) {
            groups_total += 1;
            if oracle_code == predicted_code {
                groups_correct += 1;
            }
        }
    }
    AddressMatchSummary {
        groups_correct,
        groups_total,
        exact_address_match,
    }
}

fn normalized_row_scale_slice(row: &[f32]) -> f32 {
    let mean_square = if row.is_empty() {
        0.0
    } else {
        row.iter()
            .map(|&value| (value as f64) * (value as f64))
            .sum::<f64>()
            / row.len() as f64
    };
    (mean_square.sqrt() as f32).max(1e-6)
}

fn normalized_row_scale_view(row: ArrayView1<'_, f32>) -> f32 {
    let mean_square = if row.is_empty() {
        0.0
    } else {
        row.iter()
            .map(|&value| (value as f64) * (value as f64))
            .sum::<f64>()
            / row.len() as f64
    };
    (mean_square.sqrt() as f32).max(1e-6)
}

fn normalized_hyperplane_logit(row: ArrayView1<'_, f32>, weights: &[f32], bias: f32) -> f32 {
    let scale = normalized_row_scale_view(row);
    let dot = row
        .iter()
        .zip(weights.iter())
        .map(|(&x, &w)| (x / scale) * w)
        .sum::<f32>();
    bias + dot
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
