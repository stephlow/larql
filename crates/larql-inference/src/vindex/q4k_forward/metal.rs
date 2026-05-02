use larql_models::ModelWeights;
use larql_vindex::VectorIndex;
use tokenizers::Tokenizer;

use crate::forward::PredictResult;

/// End-to-end predict on a Q4_K vindex driven by a Metal (or any Q4-capable)
/// `ComputeBackend`.
pub fn predict_q4k_metal(
    weights: &ModelWeights,
    tokenizer: &Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    index: &VectorIndex,
    backend: &dyn larql_compute::ComputeBackend,
) -> PredictResult {
    use crate::layer_graph::pipeline_layer::{build_arch_params, resolve_attn_weights};
    use larql_compute::QuantFormat;

    let arch = &*weights.arch;
    let num_layers = weights.num_layers;

    let layers: Vec<_> = (0..num_layers)
        .map(|layer| {
            let (wq, wk, wv, wo) =
                resolve_attn_weights(index, layer).expect("attn Q4K slices missing for layer");
            let [(gate_bytes, gate_fmt), (up_bytes, up_fmt), (down_bytes, down_fmt)] = index
                .interleaved_q4k_layer_data(layer)
                .expect("ffn Q4K slices missing for layer");
            fn to_format(s: &str) -> QuantFormat {
                match s {
                    "Q4_K" => QuantFormat::Q4_K,
                    "Q6_K" => QuantFormat::Q6_K,
                    other => panic!(
                        "q4k_forward: registry tag {other:?} has no compute::QuantFormat mapping"
                    ),
                }
            }
            let gate = larql_compute::QuantWeight {
                data: gate_bytes,
                scales: None,
                format: to_format(gate_fmt),
            };
            let up = larql_compute::QuantWeight {
                data: up_bytes,
                scales: None,
                format: to_format(up_fmt),
            };
            let down = larql_compute::QuantWeight {
                data: down_bytes,
                scales: None,
                format: to_format(down_fmt),
            };
            build_arch_params(weights, layer, wq, wk, wv, wo, gate, up, down)
        })
        .collect();

    let max_seq = token_ids.len().max(64);
    let shapes: Vec<(usize, usize)> = layers
        .iter()
        .map(|l| (l.num_kv_heads, l.head_dim))
        .collect();
    backend.preallocate_kv_cache_per_layer(&shapes, max_seq);
    backend.reset_kv_cache();

    let hidden = weights.hidden_size;
    let embed = &weights.embed;
    let embed_scale = arch.embed_scale();

    let q_dim_first = layers[0].num_q_heads * layers[0].head_dim;
    let kv_dim_first = layers[0].num_kv_heads * layers[0].head_dim;
    let softcap = arch.attn_logit_softcapping().unwrap_or(0.0);
    let qk_norm = arch.attn_q_norm_key(0).is_some();

    let _ = (q_dim_first, kv_dim_first, qk_norm, softcap);

    let dims_q = layers[0].num_q_heads * layers[0].head_dim;
    let dims_kv = layers[0].num_kv_heads * layers[0].head_dim;

    let mut h_vec: Vec<f32> = Vec::with_capacity(hidden);
    for &tok in token_ids {
        let row = embed.row(tok as usize);
        let x: Vec<f32> = row.iter().map(|v| v * embed_scale).collect();

        let out = backend
            .decode_token(
                &layers,
                &x,
                hidden,
                weights.intermediate_size,
                dims_q,
                dims_kv,
                layers[0].num_q_heads,
                layers[0].num_kv_heads,
                layers[0].head_dim,
                layers[0].rope_base,
            )
            .expect("backend doesn't support decode_token - need Metal with Q4 kernels");
        h_vec = out;
    }

    let h_last = ndarray::Array2::from_shape_vec((1, hidden), h_vec).expect("residual shape");
    crate::forward::predict::logits_to_predictions_pub(weights, &h_last, tokenizer, top_k, 1.0)
}
