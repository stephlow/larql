use larql_vindex::VectorIndex;
use ndarray::Array2;

use super::dequant::dequantize_matrix;

/// Run one layer's FFN forward on a Q4_K vindex, dequantising gate/up/down
/// for just this layer and applying the architecture's activation gate.
pub fn q4k_ffn_forward_layer(
    arch: &dyn larql_models::ModelArchitecture,
    index: &VectorIndex,
    layer: usize,
    x: &Array2<f32>,
) -> Array2<f32> {
    use crate::ffn::{gelu_tanh_gate_up, silu_gate_up};
    use crate::forward::dot_proj;

    let hidden = x.shape()[1];
    let intermediate = index.num_features(layer);

    let ffn = index.interleaved_q4k_layer_data(layer).unwrap_or_else(|| {
        panic!(
            "interleaved_q4k layer data missing for layer {layer} - \
             server must call `load_interleaved_q4k` before serving walk-ffn"
        )
    });

    let w_gate = dequantize_matrix(ffn[0].0, ffn[0].1, intermediate, hidden);
    let w_up = dequantize_matrix(ffn[1].0, ffn[1].1, intermediate, hidden);
    let inter_padded = intermediate.div_ceil(larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS)
        * larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS;
    let w_down = if inter_padded != intermediate {
        let w = dequantize_matrix(ffn[2].0, ffn[2].1, hidden, inter_padded);
        w.slice(ndarray::s![.., ..intermediate]).to_owned()
    } else {
        dequantize_matrix(ffn[2].0, ffn[2].1, hidden, intermediate)
    };

    let gate = dot_proj(x, &w_gate);
    let up = dot_proj(x, &w_up);
    let activation = match arch.activation() {
        larql_models::Activation::GeluTanh | larql_models::Activation::Gelu => {
            gelu_tanh_gate_up(&gate, &up)
        }
        _ => silu_gate_up(&gate, &up),
    };
    dot_proj(&activation, &w_down)
}
