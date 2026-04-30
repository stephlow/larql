#![cfg(feature = "metal")]

extern crate blas_src;

#[path = "common/mod.rs"]
mod common;

use common::{cos_sim, get_metal, max_diff};
use larql_compute::cpu::ops::moe::run_single_expert;
use larql_compute::{Activation, MoeScratch, QuantFormat};

fn synth_values(len: usize, seed: f32, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let a = (seed + i as f32 * 0.0017).sin();
            let b = (seed * 0.37 + (i >> 7) as f32 * 0.019).cos();
            (a + 0.25 * b) * scale
        })
        .collect()
}

fn pad_rows_to_256(data: &[f32], rows: usize, cols: usize) -> (Vec<f32>, usize) {
    let padded_cols = cols.div_ceil(256) * 256;
    if padded_cols == cols {
        return (data.to_vec(), cols);
    }
    let mut out = vec![0.0f32; rows * padded_cols];
    for r in 0..rows {
        out[r * padded_cols..r * padded_cols + cols]
            .copy_from_slice(&data[r * cols..(r + 1) * cols]);
    }
    (out, padded_cols)
}

fn make_q4k_experts(hidden: usize, inter: usize, top_k: usize) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let mut gate_up = Vec::with_capacity(top_k);
    let mut down = Vec::with_capacity(top_k);
    for e in 0..top_k {
        let gate = synth_values(inter * hidden, 0.11 + e as f32 * 0.13, 0.18);
        let up = synth_values(inter * hidden, 0.41 + e as f32 * 0.17, 0.16);
        let mut gu = Vec::with_capacity(2 * inter * hidden);
        gu.extend_from_slice(&gate);
        gu.extend_from_slice(&up);
        gate_up.push(larql_compute::cpu::ops::q4_common::quantize_q4_k(&gu));

        let raw_down = synth_values(hidden * inter, 0.73 + e as f32 * 0.07, 0.11);
        let (down_padded, _) = pad_rows_to_256(&raw_down, hidden, inter);
        down.push(larql_compute::cpu::ops::q4_common::quantize_q4_k(
            &down_padded,
        ));
    }
    (gate_up, down)
}

fn assert_preselected_dispatch_matches_cpu(label: &str, hidden: usize, inter: usize, top_k: usize) {
    let metal = get_metal();
    let h_norm = synth_values(hidden, 1.23, 0.35);
    let expert_ids: Vec<usize> = (0..top_k).collect();
    let expert_weights: Vec<f32> = (0..top_k)
        .map(|i| (i as f32 + 1.0) / (top_k as f32 * (top_k as f32 + 1.0) * 0.5))
        .collect();
    let (gate_up, down) = make_q4k_experts(hidden, inter, top_k);

    let mut expected = vec![0.0f32; hidden];
    for e in 0..top_k {
        let out = run_single_expert(
            &h_norm,
            &gate_up[e],
            &down[e],
            inter,
            QuantFormat::Q4_K,
            Activation::GeluTanh,
        );
        for (acc, &v) in expected.iter_mut().zip(&out) {
            *acc += v * expert_weights[e];
        }
    }

    let scratch = MoeScratch::new_public(&metal, top_k, hidden, inter);
    let got = metal.run_experts_preselected_metal(
        &h_norm,
        &expert_ids,
        &expert_weights,
        &scratch,
        |eid| Some((gate_up[eid].as_slice(), down[eid].as_slice())),
    );

    let diff = max_diff(&expected, &got);
    let cos = cos_sim(&expected, &got);
    assert!(
        diff < 1.0 && cos > 0.995,
        "{label}: Metal MoE expert dispatch diverged from CPU: max_abs={diff:.3e} cos={cos:.6}"
    );
}

#[test]
fn metal_moe_preselected_small_q4k_matches_cpu() {
    assert_preselected_dispatch_matches_cpu("small q4k moe", 256, 256, 2);
}

#[test]
#[ignore = "known open Metal MoE issue at Gemma 4 26B-A4B shape; run explicitly while debugging"]
fn metal_moe_preselected_gemma4_26b_a4b_shape_matches_cpu() {
    assert_preselected_dispatch_matches_cpu("gemma4-26b-a4b moe", 2816, 704, 8);
}
