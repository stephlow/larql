//! Q4K dequant walk — dequantises gate/up/down from `interleaved_q4k.bin`
//! for the given layer, then runs the standard dense GEGLU forward.
//!
//! Used by the INFER pipeline on Q4K vindexes without a GPU backend.
//! Peak memory is one layer's worth of dequantised f32 matrices;
//! cheap on 4B (120 MB), tight on 31B (1.8 GB).

use ndarray::Array2;


use super::WalkFfn;

impl<'a> WalkFfn<'a> {
    pub(super) fn walk_ffn_q4k_dequant(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> Option<(Array2<f32>, Array2<f32>)> {
        let ffn = self.index.interleaved_q4k_layer_data(layer)?;
        let arch = &*self.weights.arch;
        let intermediate = self.index.num_features(layer);
        if intermediate == 0 {
            return None;
        }
        let hidden = x.shape()[1];

        let dequant = |bytes: &[u8], fmt: &str, rows: usize, cols: usize| -> Array2<f32> {
            let padded = rows * cols;
            let flat = match fmt {
                "Q6_K" => larql_models::quant::ggml::dequantize_q6_k(bytes, padded)
                    .expect("q6k dequant"),
                _ => larql_models::quant::ggml::dequantize_q4_k(bytes, padded)
                    .expect("q4k dequant"),
            };
            Array2::from_shape_vec((rows, cols), flat[..rows * cols].to_vec())
                .expect("dequant shape mismatch")
        };

        let w_gate = dequant(ffn[0].0, ffn[0].1, intermediate, hidden);
        let w_up = dequant(ffn[1].0, ffn[1].1, intermediate, hidden);
        let w_down = dequant(ffn[2].0, ffn[2].1, hidden, intermediate);

        let use_gelu = matches!(
            arch.activation(),
            larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
        );
        let gate = crate::forward::dot_proj(x, &w_gate);
        let up = crate::forward::dot_proj(x, &w_up);
        let activation = if use_gelu {
            crate::ffn::gelu_tanh_gate_up(&gate, &up)
        } else {
            crate::ffn::silu_gate_up(&gate, &up)
        };
        let out = crate::forward::dot_proj(&activation, &w_down);
        self.trace_path(layer, "interleaved_q4k:dequant");
        Some((out, activation))
    }
}
