//! CPU forward pass driven by a Q4_K / Q6_K vindex.
//!
//! The normal CPU path reads attention Q/K/V/O and FFN gate/up/down from
//! `weights.tensors` as f32 matrices. For a Q4 vindex those tensors were
//! never loaded (expanding 31B to f32 is ~127 GB and won't fit on a 96 GB
//! machine), so this module dequantises one layer's worth of weights into
//! `weights.tensors`, runs the existing `run_layer_with_ffn` against it,
//! then removes the entries before moving to the next layer. Peak f32 heap
//! stays around 1.8 GB per layer (the 31B down_proj) — the rest of the
//! model lives on disk through `VectorIndex` mmaps.
//!
//! The forward path reuses every attention / QK-norm / RoPE / GQA /
//! GEGLU routine from the f32 code, so Gemma 2/3/4 model families all
//! work. A future optimisation would call
//! `larql_compute::cpu::ops::q4k_matvec` directly to avoid the per-layer
//! dequant, but that would mean re-implementing the whole attention
//! block.
//!
//! ## Gemma 4 E2B specifics
//!
//! Getting E2B green required four fixes on top of the baseline 31B
//! path:
//!
//! - **Cross-layer KV sharing** — `num_kv_shared_layers=20` means layers
//!   15-34 reuse K/V computed by the last unshared sliding / full layer.
//!   We thread a `kv_cache: HashMap<usize, SharedKV>` through the loop
//!   (mirrors `predict_with_temperature`).
//! - **Per-Layer Embeddings (PLE)** — extraction writes the global PLE
//!   tensors (`per_layer_model_projection`, `embed_tokens_per_layer`)
//!   and the per-layer `per_layer_input_gate` / `per_layer_projection`
//!   into `ple_weights.bin` at **f16** (NOT Q4_K — the super-block
//!   calibration zeroes out embedding-style tensors). Load populates
//!   `weights.tensors` so `precompute_per_layer_inputs` and
//!   `apply_per_layer_embedding` can read them directly.
//! - **Double-wide MLP** — `use_double_wide_mlp=True` gives some layers
//!   `intermediate=12288` while the model-wide config reports 6144. Use
//!   `index.num_features(layer)` per-layer to size the FFN dequant;
//!   `weights.intermediate_size` is wrong for wide layers.
//! - **Final-logit softcap** — `final_logit_softcapping=30.0` must
//!   survive extract → vindex → load. Without it `logits_to_predictions`
//!   peaks on the wrong token; the cos-sim 0.99 uncapped distribution
//!   on E2B happened to argmax on "hyperparameters".
//!
//! Wire-in point: `walk --predict --index <q4 vindex>` in
//! `larql-cli/src/commands/extraction/walk_cmd.rs`.

use std::collections::HashMap;

use ndarray::Array2;
use tokenizers::Tokenizer;

use larql_models::ModelWeights;
use larql_vindex::VectorIndex;

use crate::attention::SharedKV;
use crate::forward::embed_tokens_pub;
use crate::forward::ple::precompute_per_layer_inputs;
use crate::forward::run_layer_with_ffn;
use crate::forward::PredictResult;

/// Compute the final hidden state for `token_ids` against a Q4_K/Q6_K
/// vindex, dequantising attn + FFN one layer at a time. Returns the
/// `[seq_len, hidden]` array — caller owns the lm_head step (top-k
/// predictions, raw logits, masking, etc.).
///
/// Shared by [`predict_q4k`] and [`generate_q4k_cpu_constrained`].
pub fn predict_q4k_hidden(
    weights: &mut ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    moe_remote: Option<&crate::ffn::RemoteMoeBackend>,
) -> ndarray::Array2<f32> {
    let num_layers = weights.num_layers;
    let hidden = weights.hidden_size;
    // NOTE: don't use `weights.intermediate_size` — Gemma 4 E2B has
    // `use_double_wide_mlp=True`, so half the layers (15-34) actually
    // ship with intermediate=12288 while `weights.intermediate_size`
    // reports the baseline 6144. Ask the index per layer instead.

    let mut h = embed_tokens_pub(weights, token_ids);

    // Per-Layer Embeddings + cross-layer KV-sharing — both used by
    // Gemma 4 E2B (PLE + last-20 layers reuse K/V from the preceding
    // unshared sliding/global layer). Mirrors `predict_with_temperature`.
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
    let mut kv_cache: HashMap<usize, SharedKV> = HashMap::new();
    let dump_dir = std::env::var("LARQL_CPU_DUMP_LAYERS").ok();
    if let Some(ref dir) = dump_dir {
        let slice = h.as_slice().unwrap_or(&[]);
        let bytes: Vec<u8> = slice.iter().flat_map(|v| v.to_le_bytes()).collect();
        let _ = std::fs::write(format!("{dir}/cpu_h_embed.f32"), &bytes);
    }

    for layer in 0..num_layers {
        let attn = index
            .attn_q4k_layer_data(layer)
            .unwrap_or_else(|| panic!("attn Q4K slices missing for layer {layer}"));
        let ffn = index
            .interleaved_q4k_layer_data(layer)
            .unwrap_or_else(|| panic!("ffn Q4K slices missing for layer {layer}"));

        let arch = &*weights.arch;
        let num_q = arch.num_q_heads_for_layer(layer);
        let num_kv = arch.num_kv_heads_for_layer(layer);
        let head_dim = arch.head_dim_for_layer(layer);
        let q_dim = num_q * head_dim;
        let kv_dim = num_kv * head_dim;
        let intermediate = index.num_features(layer);

        let q_key = arch.attn_q_key(layer);
        let k_key = arch.attn_k_key(layer);
        let v_key = arch.attn_v_key(layer);
        let o_key = arch.attn_o_key(layer);
        let gate_key = arch.ffn_gate_key(layer);
        let up_key = arch.ffn_up_key(layer);
        let down_key = arch.ffn_down_key(layer);

        let w_q = dequantize_matrix(attn[0].0, attn[0].1, q_dim, hidden);
        let w_k = dequantize_matrix(attn[1].0, attn[1].1, kv_dim, hidden);
        let w_v = dequantize_matrix(attn[2].0, attn[2].1, kv_dim, hidden);
        let w_o = dequantize_matrix(attn[3].0, attn[3].1, hidden, q_dim);

        let w_gate = dequantize_matrix(ffn[0].0, ffn[0].1, intermediate, hidden);
        let w_up = dequantize_matrix(ffn[1].0, ffn[1].1, intermediate, hidden);
        // down_proj: stored at the Q6_K-padded column width (inter_padded).
        // Reading with `intermediate` as the column count gives the wrong row
        // stride when intermediate is not a multiple of K_QUANT_BLOCK_ELEMS
        // (e.g., 2112 → padded 2304 for Gemma 4 26B-A4B dense FFN). Dequantize
        // at the padded width, then slice off the padding columns.
        let inter_padded = intermediate.div_ceil(larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS)
            * larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS;
        let w_down = if inter_padded != intermediate {
            let w = dequantize_matrix(ffn[2].0, ffn[2].1, hidden, inter_padded);
            w.slice(ndarray::s![.., ..intermediate]).to_owned()
        } else {
            dequantize_matrix(ffn[2].0, ffn[2].1, hidden, intermediate)
        };

        weights.tensors.insert(q_key.clone(), w_q.into_shared());
        weights.tensors.insert(k_key.clone(), w_k.into_shared());
        weights.tensors.insert(v_key.clone(), w_v.into_shared());
        weights.tensors.insert(o_key.clone(), w_o.into_shared());
        weights
            .tensors
            .insert(gate_key.clone(), w_gate.into_shared());
        weights.tensors.insert(up_key.clone(), w_up.into_shared());
        weights
            .tensors
            .insert(down_key.clone(), w_down.into_shared());

        let shared_kv = weights
            .arch
            .kv_shared_source_layer(layer)
            .and_then(|src| kv_cache.get(&src));
        let is_moe_layer = weights.arch.is_hybrid_moe();
        let ffn_backend = crate::ffn::WeightFfn { weights };
        if is_moe_layer {
            // Gemma 4 hybrid-MoE layer: dense FFN (h1) + CPU MoE (h2),
            // combined under the outer post-FFN norm, then PLE + layer_scalar.
            if let Some((h_new, kv_out)) = run_moe_layer_cpu(
                weights,
                &h,
                layer,
                &ffn_backend,
                ple_inputs.get(layer),
                shared_kv,
                moe_remote,
            ) {
                h = h_new;
                if let Some(kv) = kv_out {
                    kv_cache.insert(layer, kv);
                }
            }
        } else if let Some((h_new, _, kv_out)) = run_layer_with_ffn(
            weights,
            &h,
            layer,
            &ffn_backend,
            false,
            ple_inputs.get(layer),
            shared_kv,
        ) {
            h = h_new;
            if let Some(kv) = kv_out {
                kv_cache.insert(layer, kv);
            }
        }

        weights.tensors.remove(&q_key);
        weights.tensors.remove(&k_key);
        weights.tensors.remove(&v_key);
        weights.tensors.remove(&o_key);
        weights.tensors.remove(&gate_key);
        weights.tensors.remove(&up_key);
        weights.tensors.remove(&down_key);

        if let Some(ref dir) = dump_dir {
            let slice = h.as_slice().unwrap_or(&[]);
            let bytes: Vec<u8> = slice.iter().flat_map(|v| v.to_le_bytes()).collect();
            let path = format!("{dir}/cpu_layer_{layer:02}.f32");
            if let Err(e) = std::fs::write(&path, &bytes) {
                eprintln!("[dump] failed to write {path}: {e}");
            }
        }
    }

    h
}

/// Build `MoeRouterWeights` for a single layer from the model's vector store.
///
/// Mirrors the inline construction in `layer_graph/grid.rs` so remote dispatch
/// uses the same routing math as the Metal path. Returns `None` if the required
/// router projection is absent (non-MoE layer or weights not loaded).
fn build_moe_router_weights<'a>(
    weights: &'a larql_models::ModelWeights,
    arch: &dyn larql_models::ModelArchitecture,
    layer: usize,
) -> Option<crate::ffn::MoeRouterWeights<'a>> {
    let router_key = arch.moe_router_key(layer)?;
    let router_proj = weights.vectors.get(&router_key)?.as_slice();
    let sl = |k: Option<String>| -> &'a [f32] {
        k.and_then(|k| weights.vectors.get(&k))
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    };
    Some(crate::ffn::MoeRouterWeights {
        router_proj,
        router_scale: sl(arch.moe_router_scale_key(layer)),
        router_per_expert_scale: sl(arch.moe_router_per_expert_scale_key(layer)),
        router_norm: sl(arch.moe_router_norm_key(layer)),
        router_norm_parameter_free: arch.moe_router_norm_parameter_free(),
        router_input_scalar: arch.moe_router_input_scalar().unwrap_or(1.0),
        pre_experts_norm: sl(arch.moe_pre_experts_norm_key(layer)),
        post_experts_norm: sl(arch.moe_post_experts_norm_key(layer)),
        num_experts: arch.num_experts(),
        top_k: arch.num_experts_per_token(),
    })
}

/// CPU forward for one hybrid-MoE layer (Gemma 4 26B A4B).
///
/// Matches HF's `Gemma4TextDecoderLayer.forward` for MoE-enabled layers:
///
/// 1. `h_post_attn = h + attn_out`
/// 2. Dense branch: `h1 = post_ffn_norm_1(dense_mlp(pre_norm(h_post_attn)))`
/// 3. MoE branch:   `h2 = post_ffn_norm_2(moe_block(h_post_attn))`
///    (the MoE block itself applies `pre_experts_norm`, runs
///    router + top-k + experts, and applies `post_experts_norm_2`)
/// 4. Combine:      `h_out = h_post_attn + outer_post_ffn_norm(h1 + h2)`
/// 5. Per-layer embedding contribution (PLE)
/// 6. `h_out *= layer_scalar`
///
/// Mirrors the Metal decode interleave in
/// `larql-compute/src/metal/decode/mod.rs` and `moe_combine.rs` so that CPU
/// and GPU paths produce the same hidden state (verified against HF bf16
/// via residual-cosine diff in the Metal `diag.rs` dumps).
fn run_moe_layer_cpu(
    weights: &ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    ffn: &dyn crate::ffn::FfnBackend,
    ple_input: Option<&Array2<f32>>,
    shared_kv: Option<&SharedKV>,
    moe_remote: Option<&crate::ffn::RemoteMoeBackend>,
) -> Option<(Array2<f32>, Option<SharedKV>)> {
    let arch = &*weights.arch;
    let norm_offset = arch.norm_weight_offset();
    let eps = arch.norm_eps();
    let hidden = h.ncols();

    // ── 1. Attention (with or without shared K/V) ─────────────────────────
    let (h_post_attn, kv_out) = if let Some(shared) = shared_kv {
        let (h_pa, _, _) =
            crate::attention::run_attention_block_shared(weights, h, layer, false, Some(shared))?;
        (h_pa, None)
    } else {
        let (h_pa, _, _, k_rope, v_final) =
            crate::attention::run_attention_block_with_kv_out(weights, h, layer, false, None)?;
        (h_pa, Some((k_rope, v_final)))
    };

    // Dump h_post_attn for layer-by-layer parity vs Metal
    // (LARQL_DUMP_RESIDUALS). Same hook the dense path uses in
    // `forward/layer.rs:182`; missing here means the MoE-bisect tools
    // can't tell whether attention or the FFN-side is off.
    if let Ok(dir) = std::env::var("LARQL_CPU_DUMP_LAYERS") {
        let slice = h_post_attn.as_slice().unwrap_or(&[]);
        let bytes: Vec<u8> = slice.iter().flat_map(|v| v.to_le_bytes()).collect();
        let path = format!("{dir}/cpu_layer_{layer:02}_h_post_attn.f32");
        let _ = std::fs::write(&path, &bytes);
    }

    // ── 2. Dense FFN branch (h1). `run_ffn` returns `h_post_attn + _1(dense)`
    //     plus residual; subtract h_post_attn to isolate `_1(dense) = h1`.
    let (h_post_ffn_dense, _) = crate::forward::run_ffn(weights, &h_post_attn, layer, ffn, false);
    let h1 = &h_post_ffn_dense - &h_post_attn;

    // ── 3. MoE branch (h2).
    //
    // Remote path: router runs locally, top-K expert matmuls are dispatched
    // to the warm mini-processes via POST /v1/expert/batch.
    //
    // Local path: router + expert matmuls run on CPU (the original path).
    let seq_len = h_post_attn.nrows();
    let mut h2 = Array2::<f32>::zeros((seq_len, hidden));

    if let Some(remote) = moe_remote {
        // Remote dispatch: one batch call per shard per layer across ALL
        // positions. forward_moe_seq replaces the per-position loop,
        // reducing HTTP round trips from seq_len×shards to shards.
        if let Some(router) = build_moe_router_weights(weights, arch, layer) {
            match remote.forward_moe_seq(layer, &h_post_attn, &router, norm_offset, eps) {
                Ok(out) => h2 = out,
                Err(e) => eprintln!("[run_moe_layer_cpu] remote dispatch error L{layer}: {e}"),
            }
        }
        // If router weights unavailable, h2 stays zero (dense-only degradation).
    } else {
        // Local CPU path.
        let moe_weights =
            crate::layer_graph::pipeline_layer::build_moe_weights(weights, arch, layer);
        if let Some(ref moe) = moe_weights {
            for pos in 0..seq_len {
                let row: Vec<f32> = h_post_attn.row(pos).to_vec();
                let moe_out =
                    larql_compute::cpu::ops::moe::cpu_moe_forward(&row, moe, norm_offset, eps);
                for (dst, src) in h2.row_mut(pos).iter_mut().zip(moe_out.iter()) {
                    *dst = *src;
                }
            }
        } else {
            // Arch says hybrid-MoE but weights unavailable — dense-only fallback.
            let mut out = h_post_ffn_dense;
            let mut h_ple =
                crate::forward::ple::apply_per_layer_embedding(weights, &out, layer, ple_input);
            crate::forward::layer::apply_layer_scalar(weights, &mut h_ple, layer);
            out = h_ple;
            return Some((out, kv_out));
        }
    }

    // ── 4. Combine via outer post-FFN norm + residual + layer_scalar.
    //
    // Routed through `larql_compute::cpu::ops::outer_combine` so this
    // CPU path and Metal's `apply_outer_combine` share a single
    // implementation of the math. Earlier the two backends had
    // independent transcriptions of the same formula and silently
    // drifted (CPU used f64 RMS / fell back to identity-scale norm;
    // Metal used f32 RMS / skipped the norm entirely on missing
    // weights), producing cos=0.63 layer-output divergence on
    // Gemma 4 26B-A4B even though h_post_attn matched at cos=1.0.
    let combined = &h1 + &h2;

    // Layer-0 stage dumps (LARQL_CPU_STAGE_DUMP=<dir>) for hybrid-MoE
    // bisection vs Metal's `metal/decode/moe_combine.rs`.
    let l0_stage_dump = if layer == 0 {
        std::env::var("LARQL_CPU_STAGE_DUMP").ok()
    } else {
        None
    };
    let dump_l0_arr = |name: &str, arr: &Array2<f32>| {
        if let Some(ref dir) = l0_stage_dump {
            let slice = arr.as_slice().unwrap_or(&[]);
            let bytes: Vec<u8> = slice.iter().flat_map(|v| v.to_le_bytes()).collect();
            let _ = std::fs::write(format!("{dir}/cpu_L0_{name}.f32"), &bytes);
        }
    };
    dump_l0_arr("h1_dense_norm1", &h1);
    dump_l0_arr("h2_moe_norm2", &h2);
    dump_l0_arr("combined_h1_plus_h2", &combined);

    // Resolve the outer norm weight the same way Metal does:
    // `moe_outer_post_norm` first, fall back to the dense-branch
    // `post_ffn_norm` (the `_1` variant on Gemma 4). `None` means
    // the vindex didn't ship either; the helper then skips the norm
    // entirely instead of silently applying an identity scale.
    let outer_w_vec: Option<&Vec<f32>> = if arch.moe_has_combined_output_norm() {
        arch.moe_post_outer_norm_key(layer)
            .or_else(|| arch.post_feedforward_layernorm_key(layer))
            .and_then(|k| weights.vectors.get(&k))
    } else {
        None
    };

    let seq = combined.nrows();
    let mut out_buf = Array2::<f32>::zeros((seq, hidden));
    for pos in 0..seq {
        let h_post_attn_row = h_post_attn.row(pos);
        let combined_row = combined.row(pos);
        let combined_normed = larql_compute::cpu::ops::outer_combine::outer_post_norm_residual(
            h_post_attn_row.as_slice().expect("contiguous row"),
            combined_row.as_slice().expect("contiguous row"),
            outer_w_vec.map(|v| v.as_slice()),
            norm_offset,
            eps,
        );
        for (dst, src) in out_buf.row_mut(pos).iter_mut().zip(combined_normed.iter()) {
            *dst = *src;
        }
    }
    dump_l0_arr("h_out_pre_layer_scalar", &out_buf);

    // ── 5 + 6. PLE then whole-layer `layer_scalar`.
    let mut h_out =
        crate::forward::ple::apply_per_layer_embedding(weights, &out_buf, layer, ple_input);
    if let Some(scalar_key) = arch.layer_scalar_key(layer) {
        if let Some(scalars) = weights.vectors.get(&scalar_key) {
            if let Some(&scalar) = scalars.first() {
                let flat = h_out.as_slice_mut().expect("contiguous out_buf");
                larql_compute::cpu::ops::outer_combine::apply_layer_scalar_in_place(flat, scalar);
            }
        }
    }

    Some((h_out, kv_out))
}

/// End-to-end predict on a Q4_K/Q6_K vindex.
///
/// `weights` must carry norms + embed + lm_head but is allowed — and
/// expected — to have empty attn / FFN tensor entries; this function
/// fills them in per layer from the vindex. Returns the top-k next-token
/// predictions in the same shape as `larql_inference::predict`.
pub fn predict_q4k(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    index: &VectorIndex,
) -> PredictResult {
    let h = predict_q4k_hidden(weights, token_ids, index, None);
    crate::forward::predict::logits_to_predictions_pub(weights, &h, tokenizer, top_k, 1.0)
}

/// Common end-of-turn / EOS markers across Gemma, Llama, Mistral, ChatML.
///
/// Used by [`generate_q4k_cpu`] to halt generation when the model emits any
/// of these. Catches a wider set than the raw EOS token id because chat
/// templates tend to use family-specific terminators.
pub fn is_end_of_turn(token: &str) -> bool {
    matches!(
        token,
        "<eos>"
            | "</s>"
            | "<|endoftext|>"
            | "<|im_end|>"
            | "<|end_of_turn|>"
            | "<end_of_turn>"
            | "<|eot_id|>"
    )
}

/// CPU autoregressive generation against a Q4_K / Q6_K vindex.
///
/// Loops [`predict_q4k`] one token at a time. Stops on `max_tokens` or when
/// the produced token text matches [`is_end_of_turn`]. Per-step cost is
/// O(N²) in context length (no KV cache) — the same trade-off
/// `larql dev walk --predict --max-tokens N` makes for the CPU path. For
/// long outputs use the Metal backend instead via
/// [`crate::layer_graph::generate`].
///
/// Returns `(token_text, token_id)` pairs in generation order.
pub fn generate_q4k_cpu(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    max_tokens: usize,
    index: &VectorIndex,
) -> Vec<(String, u32)> {
    let mut ids = prompt_ids.to_vec();
    let mut out: Vec<(String, u32)> = Vec::with_capacity(max_tokens);
    for _ in 0..max_tokens {
        let result = predict_q4k(weights, tokenizer, &ids, 1, index);
        let next_id = match result.token_ids.first() {
            Some(&id) => id,
            None => break,
        };
        let tok = result
            .predictions
            .first()
            .map(|p| p.0.clone())
            .unwrap_or_default();
        let stop = is_end_of_turn(&tok);
        out.push((tok, next_id));
        ids.push(next_id);
        if stop {
            break;
        }
    }
    out
}

/// Like [`generate_q4k_cpu`] but dispatches MoE expert matmuls to remote
/// shard servers via [`crate::ffn::RemoteMoeBackend`].
///
/// The client holds attention weights, dense-FFN weights, norms, and router
/// weights (loaded via [`larql_vindex::load_model_weights_q4k`] — no expert
/// bytes needed locally). Expert bytes live on the mini-processes launched
/// with `larql serve --experts START-END`.
///
/// Router runs locally per layer; the top-K expert residuals are dispatched
/// in parallel to the owning shard(s) via `POST /v1/expert/batch`; the
/// client assembles the weighted sum.
pub fn generate_q4k_cpu_remote(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    max_tokens: usize,
    index: &VectorIndex,
    moe_remote: &crate::ffn::RemoteMoeBackend,
) -> Vec<(String, u32)> {
    let mut ids = prompt_ids.to_vec();
    let mut out: Vec<(String, u32)> = Vec::with_capacity(max_tokens);
    for _ in 0..max_tokens {
        let h = predict_q4k_hidden(weights, &ids, index, Some(moe_remote));
        // Extract last-position hidden state then compute lm_head logits.
        // predict_q4k_hidden returns [seq_len, hidden]; next-token prediction
        // uses only the last row (the most recent token's output state).
        let last = h.nrows().saturating_sub(1);
        let h_last = h.slice(ndarray::s![last..last + 1, ..]).to_owned();
        let logits = crate::forward::hidden_to_raw_logits(weights, &h_last);
        let next_id = logits
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        let tok = tokenizer.decode(&[next_id], true).unwrap_or_default();
        let stop = is_end_of_turn(&tok);
        out.push((tok, next_id));
        ids.push(next_id);
        if stop {
            break;
        }
    }
    out
}

/// Constrained variant of [`generate_q4k_cpu`].
///
/// Computes raw logits at each step, calls `mask_fn(generated_ids, &mut logits)`
/// to let the caller mask invalid token ids to `f32::NEG_INFINITY`, then takes
/// the masked argmax. Returns the same `(token_text, token_id)` shape so it's
/// drop-in interchangeable with the unconstrained loop.
///
/// The mask callback receives only the *generated* tokens (excluding prompt),
/// so its grammar state is consistent across decode paths.
pub fn generate_q4k_cpu_constrained<M>(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    max_tokens: usize,
    index: &VectorIndex,
    mut mask_fn: M,
) -> Vec<(String, u32)>
where
    M: FnMut(&[u32], &mut Vec<f32>),
{
    let mut ids = prompt_ids.to_vec();
    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut out: Vec<(String, u32)> = Vec::with_capacity(max_tokens);

    for _ in 0..max_tokens {
        // Forward pass to the final hidden state.
        let h = predict_q4k_hidden(weights, &ids, index, None);
        let last_hidden = h.row(h.nrows().saturating_sub(1)).to_owned();
        let last_2d = ndarray::Array2::from_shape_vec((1, last_hidden.len()), last_hidden.to_vec())
            .expect("shape");

        // Raw logits over vocab → mask → argmax.
        let mut logits = crate::forward::hidden_to_raw_logits(weights, &last_2d);
        mask_fn(&generated, &mut logits);

        let (id, idx_score) = logits
            .iter()
            .enumerate()
            .filter(|(_, v)| !v.is_nan() && v.is_finite())
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, &s)| (i as u32, s))
            .unwrap_or((0, f32::NEG_INFINITY));
        if !idx_score.is_finite() {
            break;
        }
        let tok = tokenizer.decode(&[id], true).unwrap_or_default();

        let stop = is_end_of_turn(&tok);
        out.push((tok, id));
        ids.push(id);
        generated.push(id);
        if stop {
            break;
        }
    }
    out
}

/// End-to-end predict on a Q4_K vindex with the FFN served by an external
/// [`FfnBackend`] — typically [`crate::ffn::RemoteWalkBackend`] for the
/// dense-remote demo where attention runs locally and each layer's FFN is
/// one HTTP round trip to an `larql serve --ffn-only` server.
///
/// Mirrors [`predict_q4k`] except: only attention Q/K/V/O are dequantised
/// per layer (FFN weights are never loaded client-side), and the per-layer
/// FFN step is delegated to the passed backend rather than `WeightFfn`.
/// Peak f32 heap drops from ~1.8 GB/layer to ~0.4 GB/layer on 31B.
pub fn predict_q4k_with_ffn(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    index: &VectorIndex,
    ffn_backend: &dyn crate::ffn::FfnBackend,
) -> PredictResult {
    let num_layers = weights.num_layers;
    let hidden = weights.hidden_size;

    let mut h = embed_tokens_pub(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
    let mut kv_cache: HashMap<usize, SharedKV> = HashMap::new();

    for layer in 0..num_layers {
        // Attention Q/K/V/O only — FFN lives on the remote server.
        let attn = index
            .attn_q4k_layer_data(layer)
            .unwrap_or_else(|| panic!("attn Q4K slices missing for layer {layer}"));

        let arch = &*weights.arch;
        let num_q = arch.num_q_heads_for_layer(layer);
        let num_kv = arch.num_kv_heads_for_layer(layer);
        let head_dim = arch.head_dim_for_layer(layer);
        let q_dim = num_q * head_dim;
        let kv_dim = num_kv * head_dim;

        let q_key = arch.attn_q_key(layer);
        let k_key = arch.attn_k_key(layer);
        let v_key = arch.attn_v_key(layer);
        let o_key = arch.attn_o_key(layer);

        let w_q = dequantize_matrix(attn[0].0, attn[0].1, q_dim, hidden);
        let w_k = dequantize_matrix(attn[1].0, attn[1].1, kv_dim, hidden);
        let w_v = dequantize_matrix(attn[2].0, attn[2].1, kv_dim, hidden);
        let w_o = dequantize_matrix(attn[3].0, attn[3].1, hidden, q_dim);

        weights.tensors.insert(q_key.clone(), w_q.into_shared());
        weights.tensors.insert(k_key.clone(), w_k.into_shared());
        weights.tensors.insert(v_key.clone(), w_v.into_shared());
        weights.tensors.insert(o_key.clone(), w_o.into_shared());

        let shared_kv = weights
            .arch
            .kv_shared_source_layer(layer)
            .and_then(|src| kv_cache.get(&src));
        if let Some((h_new, _, kv_out)) = run_layer_with_ffn(
            weights,
            &h,
            layer,
            ffn_backend,
            false,
            ple_inputs.get(layer),
            shared_kv,
        ) {
            h = h_new;
            if let Some(kv) = kv_out {
                kv_cache.insert(layer, kv);
            }
        }

        weights.tensors.remove(&q_key);
        weights.tensors.remove(&k_key);
        weights.tensors.remove(&v_key);
        weights.tensors.remove(&o_key);
    }

    crate::forward::predict::logits_to_predictions_pub(weights, &h, tokenizer, top_k, 1.0)
}

/// End-to-end hidden-state forward on a Q4_K vindex with the FFN served by an
/// external [`FfnBackend`].
///
/// This mirrors [`predict_q4k_with_ffn`] but returns the final hidden states
/// before the lm-head step. Callers that need exact probabilities for a small
/// set of target tokens can project the last row through
/// `forward::hidden_to_raw_logits` and avoid top-k truncation.
pub fn predict_q4k_hidden_with_ffn(
    weights: &mut ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    ffn_backend: &dyn crate::ffn::FfnBackend,
) -> ndarray::Array2<f32> {
    let num_layers = weights.num_layers;
    let hidden = weights.hidden_size;

    let mut h = embed_tokens_pub(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
    let mut kv_cache: HashMap<usize, SharedKV> = HashMap::new();

    for layer in 0..num_layers {
        let attn = index
            .attn_q4k_layer_data(layer)
            .unwrap_or_else(|| panic!("attn Q4K slices missing for layer {layer}"));

        let arch = &*weights.arch;
        let num_q = arch.num_q_heads_for_layer(layer);
        let num_kv = arch.num_kv_heads_for_layer(layer);
        let head_dim = arch.head_dim_for_layer(layer);
        let q_dim = num_q * head_dim;
        let kv_dim = num_kv * head_dim;

        let q_key = arch.attn_q_key(layer);
        let k_key = arch.attn_k_key(layer);
        let v_key = arch.attn_v_key(layer);
        let o_key = arch.attn_o_key(layer);

        let w_q = dequantize_matrix(attn[0].0, attn[0].1, q_dim, hidden);
        let w_k = dequantize_matrix(attn[1].0, attn[1].1, kv_dim, hidden);
        let w_v = dequantize_matrix(attn[2].0, attn[2].1, kv_dim, hidden);
        let w_o = dequantize_matrix(attn[3].0, attn[3].1, hidden, q_dim);

        weights.tensors.insert(q_key.clone(), w_q.into_shared());
        weights.tensors.insert(k_key.clone(), w_k.into_shared());
        weights.tensors.insert(v_key.clone(), w_v.into_shared());
        weights.tensors.insert(o_key.clone(), w_o.into_shared());

        let shared_kv = weights
            .arch
            .kv_shared_source_layer(layer)
            .and_then(|src| kv_cache.get(&src));
        if let Some((h_new, _, kv_out)) = run_layer_with_ffn(
            weights,
            &h,
            layer,
            ffn_backend,
            false,
            ple_inputs.get(layer),
            shared_kv,
        ) {
            h = h_new;
            if let Some(kv) = kv_out {
                kv_cache.insert(layer, kv);
            }
        }

        weights.tensors.remove(&q_key);
        weights.tensors.remove(&k_key);
        weights.tensors.remove(&v_key);
        weights.tensors.remove(&o_key);
    }

    h
}

/// End-to-end predict on a Q4_K vindex driven by a Metal (or any Q4-capable)
/// `ComputeBackend`. Prompt tokens are fed through `backend.decode_token` one
/// position at a time — each call reads the token's embedding, appends its K/V
/// to the per-layer cache, attends causally against positions 0..=pos, and
/// returns the post-residual hidden state. Logits come from the final
/// post-prompt position via the standard final-norm + lm_head path.
///
/// Gemma 4 31B's asymmetric geometry (sliding 16×256 / global 4×512) is
/// handled by calling `backend.preallocate_kv_cache_per_layer` with the
/// exact per-layer `(num_kv_heads, head_dim)` shapes before the first decode.
/// Without that preallocation the backend would lazily size the cache from
/// the first layer's dims and the global layers would read off the end of
/// under-sized buffers.
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

    // ── Build FullPipelineLayer per layer ──
    // FFN weights come from interleaved_q4k_layer_data (manifest-driven
    // per-matrix layout). Attn weights come from resolve_attn_weights which
    // prefers the Q4K manifest. Norms/layer_scalar/etc come from the arch
    // + weights.vectors map populated by load_model_weights_q4k.
    let layers: Vec<_> = (0..num_layers)
        .map(|layer| {
            let (wq, wk, wv, wo) =
                resolve_attn_weights(index, layer).expect("attn Q4K slices missing for layer");
            let [(gate_bytes, gate_fmt), (up_bytes, up_fmt), (down_bytes, down_fmt)] = index
                .interleaved_q4k_layer_data(layer)
                .expect("ffn Q4K slices missing for layer");
            // Translate registry tag → `larql_compute::QuantFormat`. Two
            // enum systems cross here (vindex registry vs compute pipeline),
            // and the previous `_ => Q4_K` default silently hid every
            // other format. Be explicit.
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

    // ── Preallocate KV cache with correct per-layer shapes ──
    let max_seq = token_ids.len().max(64);
    let shapes: Vec<(usize, usize)> = layers
        .iter()
        .map(|l| (l.num_kv_heads, l.head_dim))
        .collect();
    backend.preallocate_kv_cache_per_layer(&shapes, max_seq);
    backend.reset_kv_cache();

    // ── Run decode one token at a time, building up KV cache ──
    let hidden = weights.hidden_size;
    let embed = &weights.embed;
    let embed_scale = arch.embed_scale();

    let q_dim_first = layers[0].num_q_heads * layers[0].head_dim;
    let kv_dim_first = layers[0].num_kv_heads * layers[0].head_dim;
    let softcap = arch.attn_logit_softcapping().unwrap_or(0.0);
    let qk_norm = arch.attn_q_norm_key(0).is_some();

    let _ = (q_dim_first, kv_dim_first, qk_norm, softcap); // reserved for a future prefill path

    // decode_token processes one token position at a time, appending its K/V
    // to the per-layer cache and attending causally against positions 0..=pos.
    // We feed the prompt tokens through it one by one to build the cache, then
    // the final residual is the prediction-time hidden state.
    //
    // Each decode_token call takes the FIRST layer's dims as the outer
    // scalar shape; the per-layer FullPipelineLayer inside drives the actual
    // geometry. This works even on Gemma 4 31B because the scratch buffers
    // inside decode_token are now sized to max(layer.q_dim) / max(layer.kv_dim).
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
            .expect("backend doesn't support decode_token — need Metal with Q4 kernels");
        h_vec = out;
    }

    // ── Final norm + lm_head over the last position's residual ──
    let h_last = ndarray::Array2::from_shape_vec((1, hidden), h_vec).expect("residual shape");
    crate::forward::predict::logits_to_predictions_pub(weights, &h_last, tokenizer, top_k, 1.0)
}

/// Run one layer's FFN forward on a Q4_K vindex — dequantise gate/up/down
/// for just this layer and apply the architecture's activation gate.
///
/// Used by `larql-server`'s `/v1/walk-ffn` (full_output mode) when serving
/// a Q4_K vindex: the FFN weights aren't materialised into `ModelWeights.tensors`
/// at startup (would cost ~120 GB f32 on 31B), so we dequantise per-request
/// per-layer. Working-set is ~3 GB on 31B (one layer's gate+up+down f32).
///
/// Requires `index.load_interleaved_q4k()` to have been called; panics
/// otherwise.
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
            "interleaved_q4k layer data missing for layer {layer} — \
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

/// Dequantise a row-major Q4_K or Q6_K matrix into a dense f32 `Array2`.
///
/// The on-disk layout (`rows × cols` elements) must be stored contiguously
/// row-major and padded to a multiple of 256 elements per the k-quant
/// super-block size. Unknown formats panic — callers have already
/// dispatched on format via `larql_vindex::quant::registry`, so the
/// `None` arm is unreachable in well-formed inputs.
fn dequantize_matrix(bytes: &[u8], format: &str, rows: usize, cols: usize) -> Array2<f32> {
    let n = rows * cols;
    // Q4_K and Q6_K quantise in K_QUANT_BLOCK_ELEMS-sized super-blocks; the
    // vindex writer pads up to that boundary (e.g. moe_intermediate=704 →
    // 768 padded). Use the canonical constant rather than re-hardcoding 256.
    let block = larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS;
    let padded = n.div_ceil(block) * block;
    let info = larql_vindex::quant::registry::lookup(format)
        .unwrap_or_else(|| panic!("unsupported quant format in vindex: {format}"));
    let floats =
        (info.dequantize)(bytes, padded).unwrap_or_else(|e| panic!("{format} dequant failed: {e}"));
    let truncated = if floats.len() > n {
        floats[..n].to_vec()
    } else {
        floats
    };
    Array2::from_shape_vec((rows, cols), truncated).expect("shape mismatch dequantising Q4K matrix")
}

#[cfg(test)]
mod tests {
    use super::is_end_of_turn;

    #[test]
    fn is_end_of_turn_recognises_known_terminators() {
        for t in [
            "<eos>",
            "</s>",
            "<|endoftext|>",
            "<|im_end|>",
            "<|end_of_turn|>",
            "<end_of_turn>",
            "<|eot_id|>",
        ] {
            assert!(is_end_of_turn(t), "expected {t:?} to be a terminator");
        }
    }

    #[test]
    fn is_end_of_turn_rejects_arbitrary_tokens() {
        for t in ["", " ", "the", "<eos", "eos>", "<EOS>", "<|im_start|>"] {
            assert!(
                !is_end_of_turn(t),
                "did not expect {t:?} to be a terminator"
            );
        }
    }
}
