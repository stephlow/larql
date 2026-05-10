//! GPU-side prefill phase for [`super::generate_streaming`].
//!
//! Three branches:
//!
//! 1. **PLE** (Gemma 4 E2B) — token-by-token via `decode_token` so the
//!    Metal backend can apply per-layer-input embeddings inside each
//!    layer block. Only compiled when `--features metal`; without it the
//!    branch is unreachable (no `MetalBackend` to downcast to).
//! 2. **Per-layer Q4_K MoE** — token-by-token via `decode_token_q4k_moe`.
//!    The standard `prefill_q4` path calls `cpu_moe_forward` which expects
//!    BF16 blobs and would panic on Q4_K expert bytes; token-by-token is
//!    correct and builds the KV cache identically.
//! 3. **Standard** — the batched fused `prefill_q4_prompt` path.
//!
//! Returns the post-prefill `h_vec` (`seq_len × hidden` floats; only the
//! last position is meaningful for the subsequent first-token sample).

use crate::layer_graph::generate::gpu_setup::prefill_q4_prompt;
use crate::layer_graph::generate::types::GenerateError;
use crate::layer_graph::pipeline_layer::AttentionGeometry;
use larql_compute::FullPipelineLayer;
use crate::model::ModelWeights;
use larql_compute::prelude::*;

/// Run the prefill phase for streaming Q4 generation.
///
/// `metal_ple_backend` is `Some(metal)` only when (a) the model uses
/// per-layer embeddings AND (b) `LARQL_METAL_PLE=1` AND (c) we're on
/// macOS+metal. The PLE-upload closure is invoked once per prompt token.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn prefill_for_streaming(
    weights: &ModelWeights,
    backend: &dyn ComputeBackend,
    layers: &[FullPipelineLayer],
    attention: AttentionGeometry,
    hidden: usize,
    intermediate: usize,
    token_ids: &[u32],
    x: &[f32],
    qk_norm_val: bool,
    softcap_val: f32,
    metal_ple_backend: Option<&larql_compute::metal::MetalBackend>,
    upload_ple: &dyn Fn(&larql_compute::metal::MetalBackend, u32, &[f32]),
) -> Result<Vec<f32>, GenerateError> {
    let seq_len = token_ids.len();

    // Branch 1: Per-Layer Embeddings (Metal-only).
    if let Some(metal) = metal_ple_backend {
        let mut last_h = vec![0.0f32; hidden];
        for pos in 0..seq_len {
            let x_pos: Vec<f32> = x[pos * hidden..(pos + 1) * hidden].to_vec();
            upload_ple(metal, token_ids[pos], &x_pos);
            last_h = backend
                .decode_token(
                    layers,
                    &x_pos,
                    hidden,
                    intermediate,
                    attention.q_dim,
                    attention.kv_dim,
                    attention.num_q_heads,
                    attention.num_kv_heads,
                    attention.head_dim,
                    attention.rope_base,
                )
                .unwrap_or_else(|| vec![0.0f32; hidden]);
        }
        let mut out = vec![0.0f32; seq_len * hidden];
        out[(seq_len - 1) * hidden..].copy_from_slice(&last_h);
        return Ok(out);
    }

    // Branch 2: per-layer Q4_K MoE format.
    if weights.has_per_layer_ffn() {
        return prefill_q4k_moe(
            weights,
            backend,
            layers,
            attention,
            hidden,
            intermediate,
            token_ids,
            x,
        );
    }

    // Branch 3: standard fused prefill.
    prefill_q4_prompt(
        backend,
        layers,
        x,
        hidden,
        intermediate,
        attention,
        seq_len,
        qk_norm_val,
        softcap_val,
        "GPU Q4 prefill returned no output",
    )
}

/// Non-metal build: PLE branch is unreachable (no `MetalBackend`).
#[cfg(not(all(feature = "metal", target_os = "macos")))]
#[allow(clippy::too_many_arguments)]
pub(super) fn prefill_for_streaming(
    weights: &ModelWeights,
    backend: &dyn ComputeBackend,
    layers: &[FullPipelineLayer],
    attention: AttentionGeometry,
    hidden: usize,
    intermediate: usize,
    token_ids: &[u32],
    x: &[f32],
    qk_norm_val: bool,
    softcap_val: f32,
) -> Result<Vec<f32>, GenerateError> {
    let seq_len = token_ids.len();

    if weights.has_per_layer_ffn() {
        return prefill_q4k_moe(
            weights,
            backend,
            layers,
            attention,
            hidden,
            intermediate,
            token_ids,
            x,
        );
    }

    prefill_q4_prompt(
        backend,
        layers,
        x,
        hidden,
        intermediate,
        attention,
        seq_len,
        qk_norm_val,
        softcap_val,
        "GPU Q4 prefill returned no output",
    )
}

/// Per-layer Q4_K MoE prefill: route on CPU, dispatch experts on GPU
/// via `decode_token_q4k_moe` per token. Returns the last-position hidden
/// padded to a `seq_len × hidden` buffer to match the batched-prefill shape.
#[allow(clippy::too_many_arguments)]
fn prefill_q4k_moe(
    weights: &ModelWeights,
    backend: &dyn ComputeBackend,
    layers: &[FullPipelineLayer],
    attention: AttentionGeometry,
    hidden: usize,
    intermediate: usize,
    token_ids: &[u32],
    x: &[f32],
) -> Result<Vec<f32>, GenerateError> {
    if !backend.supports(Capability::DecodeQ4KMoe) {
        return Err(GenerateError::unsupported_backend(
            "per-layer Q4K expert generation requires backend Q4K MoE decode support",
        ));
    }
    let seq_len = token_ids.len();
    let norm_eps = weights.arch.norm_eps();
    let mut last_h = vec![0.0f32; hidden];
    for pos in 0..seq_len {
        let x_pos: Vec<f32> = x[pos * hidden..(pos + 1) * hidden].to_vec();
        let get_expert =
            |layer_idx, expert_idx| weights.get_layer_entry_bytes(layer_idx, expert_idx);
        last_h = backend
            .decode_token_q4k_moe(
                layers,
                &x_pos,
                hidden,
                intermediate,
                attention.q_dim,
                attention.kv_dim,
                attention.num_q_heads,
                attention.num_kv_heads,
                attention.head_dim,
                attention.rope_base,
                norm_eps,
                &get_expert,
            )
            .unwrap_or_else(|| vec![0.0f32; hidden]);
    }
    let mut out = vec![0.0f32; seq_len * hidden];
    out[(seq_len - 1) * hidden..].copy_from_slice(&last_h);
    Ok(out)
}
