extern crate blas_src;

use larql_compute::{cpu_backend, default_backend, Activation};
use larql_compute::cpu::ops::moe::cpu_moe_forward;
use larql_compute::MoeLayerWeights;

// ── lib.rs entry points ──────────────────────────────────────────────────────

#[test]
fn cpu_backend_name_is_nonempty() {
    assert!(!cpu_backend().name().is_empty());
}

#[test]
fn cpu_backend_device_info_is_nonempty() {
    assert!(!cpu_backend().device_info().is_empty());
}

#[test]
fn default_backend_name_is_nonempty() {
    assert!(!default_backend().name().is_empty());
}

#[test]
fn cpu_backend_is_dyn_compatible() {
    let _: Box<dyn larql_compute::ComputeBackend> = cpu_backend();
}

// ── MoE forward — router norm variants ──────────────────────────────────────

fn bf16_fill(len: usize, val: f32) -> Vec<u8> {
    let hi = (val.to_bits() >> 16) as u16;
    let b = hi.to_le_bytes();
    let mut v = vec![0u8; len * 2];
    for i in 0..len { v[i * 2] = b[0]; v[i * 2 + 1] = b[1]; }
    v
}

fn make_moe_weights<'a>(
    _hidden: usize, inter: usize, num_experts: usize, top_k: usize,
    gate_up: &'a [u8], down: &'a [u8], router: &'a [f32],
    router_norm: &'a [f32], router_norm_parameter_free: bool,
) -> MoeLayerWeights<'a> {
    MoeLayerWeights {
        experts_gate_up: gate_up,
        experts_down: down,
        router_proj: router,
        router_scale: &[],
        router_per_expert_scale: &[],
        router_norm,
        router_norm_parameter_free,
        router_input_scalar: 1.0,
        pre_experts_norm: &[],
        post_ffn1_norm: &[],
        post_experts_norm: &[],
        num_experts,
        top_k,
        intermediate_size: inter,
        activation: Activation::Silu,
    }
}

#[test]
fn moe_parameter_free_router_norm_runs_without_panic() {
    // Exercises the `rms_norm_no_weight` code path in forward.rs
    let hidden = 8;
    let inter = 4;
    let num_experts = 4;
    let top_k = 2;

    let gate_up = bf16_fill(num_experts * 2 * inter * hidden, 1.0);
    let down = bf16_fill(num_experts * hidden * inter, 1.0);
    // Non-zero router so experts can be selected
    let router: Vec<f32> = (0..num_experts * hidden)
        .map(|i| if i < hidden { 1.0 } else { 0.1 })
        .collect();

    let moe = make_moe_weights(
        hidden, inter, num_experts, top_k,
        &gate_up, &down, &router,
        &[],  // empty router_norm → triggers parameter_free path
        true, // router_norm_parameter_free = true
    );
    let h = vec![1.0f32; hidden];
    let out = cpu_moe_forward(&h, &moe, 0.0, 1e-6);
    assert_eq!(out.len(), hidden);
}

#[test]
fn moe_learned_router_norm_runs_without_panic() {
    // Exercises the learned `router_norm` code path (non-empty router_norm slice)
    let hidden = 8;
    let inter = 4;
    let num_experts = 4;
    let top_k = 2;

    let gate_up = bf16_fill(num_experts * 2 * inter * hidden, 1.0);
    let down = bf16_fill(num_experts * hidden * inter, 1.0);
    let router: Vec<f32> = (0..num_experts * hidden)
        .map(|i| if i < hidden { 1.0 } else { 0.1 })
        .collect();
    let router_norm = vec![1.0f32; hidden];

    let moe = make_moe_weights(
        hidden, inter, num_experts, top_k,
        &gate_up, &down, &router,
        &router_norm, false,
    );
    let h = vec![1.0f32; hidden];
    let out = cpu_moe_forward(&h, &moe, 0.0, 1e-6);
    assert_eq!(out.len(), hidden);
}

#[test]
fn moe_per_expert_scale_applied() {
    // Verify that per_expert_scale changes the output magnitude.
    let hidden = 8;
    let inter = 4;
    let num_experts = 4;
    let top_k = 1;

    let gate_up = bf16_fill(num_experts * 2 * inter * hidden, 1.0);
    let down = bf16_fill(num_experts * hidden * inter, 1.0);
    let router: Vec<f32> = (0..num_experts * hidden)
        .map(|i| if i < hidden { 1.0 } else { 0.0 })
        .collect();
    let h = vec![1.0f32; hidden];

    // Without per-expert scale
    let moe_no_scale = MoeLayerWeights {
        experts_gate_up: &gate_up, experts_down: &down,
        router_proj: &router,
        router_scale: &[], router_per_expert_scale: &[],
        router_norm: &[], router_norm_parameter_free: false,
        router_input_scalar: 1.0, pre_experts_norm: &[],
        post_ffn1_norm: &[], post_experts_norm: &[],
        num_experts, top_k, intermediate_size: inter,
        activation: Activation::Silu,
    };
    let out_no_scale = cpu_moe_forward(&h, &moe_no_scale, 0.0, 1e-6);

    // With per-expert scale = [2.0, 1.0, 1.0, 1.0] (expert 0 gets 2× weight)
    let per_expert_scale = vec![2.0f32, 1.0, 1.0, 1.0];
    let moe_scaled = MoeLayerWeights {
        experts_gate_up: &gate_up, experts_down: &down,
        router_proj: &router,
        router_scale: &[], router_per_expert_scale: &per_expert_scale,
        router_norm: &[], router_norm_parameter_free: false,
        router_input_scalar: 1.0, pre_experts_norm: &[],
        post_ffn1_norm: &[], post_experts_norm: &[],
        num_experts, top_k, intermediate_size: inter,
        activation: Activation::Silu,
    };
    let out_scaled = cpu_moe_forward(&h, &moe_scaled, 0.0, 1e-6);

    assert_eq!(out_no_scale.len(), hidden);
    assert_eq!(out_scaled.len(), hidden);
    // Scaled output should differ from unscaled (expert 0 weight doubled)
    let max_diff: f32 = out_no_scale.iter().zip(&out_scaled)
        .map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    assert!(max_diff > 1e-6, "per_expert_scale should change output; max_diff={max_diff}");
}

#[test]
fn moe_router_scale_vector_applied() {
    // Exercises the `!moe.router_scale.is_empty()` branch in forward.rs
    let hidden = 8;
    let inter = 4;
    let num_experts = 4;
    let top_k = 1;

    let gate_up = bf16_fill(num_experts * 2 * inter * hidden, 1.0);
    let down = bf16_fill(num_experts * hidden * inter, 1.0);
    let router: Vec<f32> = (0..num_experts * hidden)
        .map(|i| if i < hidden { 1.0 } else { 0.0 })
        .collect();
    let router_scale = vec![1.0f32; hidden]; // scale each hidden dim by 1 (neutral)
    let h = vec![1.0f32; hidden];

    let moe = MoeLayerWeights {
        experts_gate_up: &gate_up, experts_down: &down,
        router_proj: &router,
        router_scale: &router_scale,   // non-empty → enters the scale branch
        router_per_expert_scale: &[],
        router_norm: &[], router_norm_parameter_free: false,
        router_input_scalar: 1.0, pre_experts_norm: &[],
        post_ffn1_norm: &[], post_experts_norm: &[],
        num_experts, top_k, intermediate_size: inter,
        activation: Activation::Silu,
    };
    let out = cpu_moe_forward(&h, &moe, 0.0, 1e-6);
    assert_eq!(out.len(), hidden);
}

#[test]
fn moe_router_input_scalar_nonunit() {
    // Exercises the `router_input_scalar != 1.0 && != 0.0` branch in forward.rs
    let hidden = 8;
    let inter = 4;
    let num_experts = 4;
    let top_k = 1;

    let gate_up = bf16_fill(num_experts * 2 * inter * hidden, 1.0);
    let down = bf16_fill(num_experts * hidden * inter, 1.0);
    let router: Vec<f32> = (0..num_experts * hidden)
        .map(|i| if i < hidden { 1.0 } else { 0.0 })
        .collect();
    let h = vec![1.0f32; hidden];

    // scalar = 0.5 → router input scaled down before projection
    let moe_scalar = MoeLayerWeights {
        experts_gate_up: &gate_up, experts_down: &down,
        router_proj: &router,
        router_scale: &[], router_per_expert_scale: &[],
        router_norm: &[], router_norm_parameter_free: false,
        router_input_scalar: 0.5,   // non-unit → enters the scaling branch
        pre_experts_norm: &[],
        post_ffn1_norm: &[], post_experts_norm: &[],
        num_experts, top_k, intermediate_size: inter,
        activation: Activation::Silu,
    };
    let out = cpu_moe_forward(&h, &moe_scalar, 0.0, 1e-6);
    assert_eq!(out.len(), hidden);
}

#[test]
fn moe_empty_router_proj_returns_zeros() {
    let hidden = 8;
    let moe = MoeLayerWeights {
        experts_gate_up: &[], experts_down: &[],
        router_proj: &[], // empty → early return
        router_scale: &[], router_per_expert_scale: &[],
        router_norm: &[], router_norm_parameter_free: false,
        router_input_scalar: 1.0, pre_experts_norm: &[],
        post_ffn1_norm: &[], post_experts_norm: &[],
        num_experts: 4, top_k: 2, intermediate_size: 4,
        activation: Activation::Silu,
    };
    let h = vec![1.0f32; hidden];
    let out = cpu_moe_forward(&h, &moe, 0.0, 1e-6);
    assert_eq!(out.len(), hidden);
    assert!(out.iter().all(|v| *v == 0.0), "empty router_proj should produce all-zero output");
}

#[test]
fn moe_zero_num_experts_returns_zeros() {
    // Exercises the num_experts == 0 early-return in forward.rs line 41.
    let hidden = 8;
    let moe = MoeLayerWeights {
        experts_gate_up: &[], experts_down: &[],
        router_proj: &[1.0f32], // non-empty so we don't hit that guard
        router_scale: &[], router_per_expert_scale: &[],
        router_norm: &[], router_norm_parameter_free: false,
        router_input_scalar: 1.0, pre_experts_norm: &[],
        post_ffn1_norm: &[], post_experts_norm: &[],
        num_experts: 0,  // triggers the early return
        top_k: 2, intermediate_size: 4,
        activation: Activation::Silu,
    };
    let h = vec![1.0f32; hidden];
    let out = cpu_moe_forward(&h, &moe, 0.0, 1e-6);
    assert_eq!(out, vec![0.0f32; hidden]);
}

#[test]
fn moe_gelu_tanh_activation_in_forward() {
    // Exercises the GeluTanh arm of the match in the rayon closure (forward.rs line 157).
    let hidden = 8;
    let inter = 4;
    let num_experts = 4;
    let top_k = 1;

    let gate_up = bf16_fill(num_experts * 2 * inter * hidden, 1.0);
    let down = bf16_fill(num_experts * hidden * inter, 1.0);
    let router: Vec<f32> = (0..num_experts * hidden)
        .map(|i| if i < hidden { 1.0 } else { 0.0 })
        .collect();

    let moe = MoeLayerWeights {
        experts_gate_up: &gate_up, experts_down: &down,
        router_proj: &router,
        router_scale: &[], router_per_expert_scale: &[],
        router_norm: &[], router_norm_parameter_free: false,
        router_input_scalar: 1.0, pre_experts_norm: &[],
        post_ffn1_norm: &[], post_experts_norm: &[],
        num_experts, top_k, intermediate_size: inter,
        activation: Activation::GeluTanh,  // exercises the GeluTanh arm
    };
    let h = vec![1.0f32; hidden];
    let out = cpu_moe_forward(&h, &moe, 0.0, 1e-6);
    assert_eq!(out.len(), hidden);
    assert!(out.iter().any(|v| v.abs() > 1e-4), "GeluTanh forward should produce nonzero output");
}
