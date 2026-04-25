//! Dense FFN backend — full matrix multiply, architecture-correct.
//! This is the ground truth: identical to model inference.

use ndarray::Array2;
use larql_compute::{ComputeBackend, dot_proj_gpu};

use crate::forward::add_bias;
use crate::model::ModelWeights;
use super::{sigmoid, gelu_tanh, silu_gate_up, gelu_tanh_gate_up, FfnBackend};

/// Dense FFN: follows the model architecture exactly (CPU BLAS).
/// Gated: activation(x @ gate.T) * (x @ up.T) @ down.T + bias
/// Non-gated: activation(x @ up.T + bias) @ down.T + bias
pub struct WeightFfn<'a> {
    pub weights: &'a ModelWeights,
}

impl<'a> FfnBackend for WeightFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        dense_ffn_forward(self.weights, layer, x).0
    }

    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        dense_ffn_forward(self.weights, layer, x)
    }

    fn name(&self) -> &str { "weights" }
}

/// Backend-dispatched dense FFN. Matmuls route through `ComputeBackend` when
/// `backend` is `Some` — useful for prefill on Metal where gate/up/down
/// projections are the dominant cost.
pub struct BackendFfn<'a, 'b> {
    pub weights: &'a ModelWeights,
    pub backend: &'b dyn ComputeBackend,
}

impl<'a, 'b> FfnBackend for BackendFfn<'a, 'b> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        dense_ffn_forward_backend(self.weights, layer, x, Some(self.backend)).0
    }

    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        dense_ffn_forward_backend(self.weights, layer, x, Some(self.backend))
    }

    fn name(&self) -> &str { "weights+backend" }
}

/// Architecture-correct dense FFN — CPU BLAS path.
pub fn dense_ffn_forward(
    weights: &ModelWeights,
    layer: usize,
    x: &Array2<f32>,
) -> (Array2<f32>, Array2<f32>) {
    dense_ffn_forward_backend(weights, layer, x, None)
}

/// Architecture-correct dense FFN with optional backend dispatch.
/// `backend = None` → plain ndarray BLAS (same as `dense_ffn_forward`).
/// `backend = Some(be)` → gate/up/down matmuls through `be.matmul_transb`.
pub fn dense_ffn_forward_backend(
    weights: &ModelWeights,
    layer: usize,
    x: &Array2<f32>,
    backend: Option<&dyn ComputeBackend>,
) -> (Array2<f32>, Array2<f32>) {
    let arch = &*weights.arch;
    let compact_hint = "FFN weight tensor missing — this is a `--compact` \
        vindex. Use `WalkFfn` instead of `WeightFfn` for inference \
        (or re-extract without `--compact` if you need dense matmul).";

    let w_up = weights
        .tensors
        .get(&arch.ffn_up_key(layer))
        .unwrap_or_else(|| panic!("{compact_hint} (key: {})", arch.ffn_up_key(layer)));
    let w_down = weights
        .tensors
        .get(&arch.ffn_down_key(layer))
        .unwrap_or_else(|| panic!("{compact_hint} (key: {})", arch.ffn_down_key(layer)));

    let activation = if arch.ffn_type() == larql_models::FfnType::Gated {
        let w_gate = weights
            .tensors
            .get(&arch.ffn_gate_key(layer))
            .unwrap_or_else(|| panic!("{compact_hint} (key: {})", arch.ffn_gate_key(layer)));
        let gate = dot_proj_gpu(x, w_gate, backend);
        let up   = dot_proj_gpu(x, w_up, backend);
        match arch.activation() {
            larql_models::Activation::GeluTanh => gelu_tanh_gate_up(&gate, &up),
            _ => silu_gate_up(&gate, &up),
        }
    } else {
        let mut projected = dot_proj_gpu(x, w_up, backend);
        if let Some(bias) = arch.ffn_up_bias_key(layer).and_then(|k| weights.vectors.get(&k)) {
            add_bias(&mut projected, bias);
        }
        match arch.activation() {
            larql_models::Activation::GeluTanh | larql_models::Activation::Gelu => projected.mapv(gelu_tanh),
            _ => projected.mapv(|v| v * sigmoid(v)),
        }
    };

    let mut out = dot_proj_gpu(&activation, w_down, backend);
    if let Some(bias) = arch.ffn_down_bias_key(layer).and_then(|k| weights.vectors.get(&k)) {
        add_bias(&mut out, bias);
    }


    (out, activation)
}
