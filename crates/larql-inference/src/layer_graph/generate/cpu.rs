//! CPU Q4K generate path — used when the active backend does not support the
//! fused Q4 prefill + KV-cached decode pipeline (today: CpuBackend).

use super::types::{GenerateResult, StageTimings};
use crate::model::ModelWeights;
use larql_compute::prelude::*;

// ── Backend capability probe + CPU Q4K delegation ────────────────────────────
//
// `generate` / `generate_constrained` assume the backend implements the fused
// Q4 prefill + KV-cached decode pipeline (currently: Metal). Backends that
// lack it (CpuBackend) delegate to the per-layer CPU Q4K dequant path
// (`predict_q4k_hidden`), which mutates `weights.tensors` per layer — that's
// the single reason these functions take `&mut ModelWeights`.

/// True when the backend can handle the fused Q4 prefill + decode pipeline
/// directly. Metal: yes. Pure CPU: no — that path produces correct forward
/// results via the vindex Q4K dequant loop in `crate::vindex::q4k_forward`.
pub(super) fn backend_supports_fused_q4_pipeline(backend: &dyn ComputeBackend) -> bool {
    // CpuBackend reports `has_q4() == true` (it has Q4 matvecs) but does not
    // override `prefill_q4` — the trait default returns None. A zero-arg
    // probe would allocate; probe the backend name instead, which is stable
    // and cheap. Metal's CpuBackend is labelled "cpu (...)".
    let name = backend.name();
    !name.starts_with("cpu")
}

/// CPU Q4K generate path: loops `predict_q4k` one step at a time. O(N²) in
/// context length (no KV cache), but correct across all supported
/// architectures including hybrid MoE (if wired — see
/// `crate::vindex::q4k_forward::predict_q4k_hidden`).
pub(super) fn generate_via_cpu_q4k(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    max_tokens: usize,
    index: &larql_vindex::VectorIndex,
) -> GenerateResult {
    let prefill_start = std::time::Instant::now();
    // First-token pass covers the prompt — that's our "prefill" here.
    let first = crate::vindex::predict_q4k(weights, tokenizer, token_ids, 5, index);
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

    let mut tokens: Vec<(String, f64)> = Vec::with_capacity(max_tokens);
    let mut decode_ms = Vec::with_capacity(max_tokens);
    let mut t_gpu = 0.0f64;

    let mut ids = token_ids.to_vec();
    // Seed with the first predicted token from the prefill pass.
    if let (Some(&id), Some(first_pred)) = (first.token_ids.first(), first.predictions.first()) {
        tokens.push((first_pred.0.clone(), 1.0));
        let stop = crate::vindex::is_end_of_turn(first_pred.0.trim());
        ids.push(id);
        if stop {
            return GenerateResult {
                tokens,
                prefill_ms,
                decode_ms,
                stage_timings: StageTimings::default(),
            };
        }
    } else {
        return GenerateResult {
            tokens,
            prefill_ms,
            decode_ms,
            stage_timings: StageTimings::default(),
        };
    }

    for _step in 1..max_tokens {
        let t0 = std::time::Instant::now();
        let result = crate::vindex::predict_q4k(weights, tokenizer, &ids, 5, index);
        let step_ms = t0.elapsed().as_secs_f64() * 1000.0;
        decode_ms.push(step_ms);
        t_gpu += step_ms;

        match result.token_ids.first() {
            Some(&id) => {
                let tok = result
                    .predictions
                    .first()
                    .map(|p| p.0.clone())
                    .unwrap_or_default();
                let stop = crate::vindex::is_end_of_turn(tok.trim());
                tokens.push((tok, 1.0));
                ids.push(id);
                if stop {
                    break;
                }
            }
            None => break,
        }
    }

    GenerateResult {
        tokens,
        prefill_ms,
        decode_ms,
        stage_timings: StageTimings {
            embed_ms_total: 0.0,
            gpu_ms_total: t_gpu,
            gate_up_ms_total: 0.0,
            down_ms_total: 0.0,
            norm_ms_total: 0.0,
            lm_head_ms_total: 0.0,
            detok_ms_total: 0.0,
        },
    }
}

/// Constrained variant of [`generate_via_cpu_q4k`]. Thin wrapper over
/// `vindex::q4k_forward::generate_q4k_cpu_constrained` that adapts the
/// result shape into `GenerateResult`.
pub(super) fn generate_constrained_via_cpu_q4k<M>(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    max_tokens: usize,
    index: &larql_vindex::VectorIndex,
    mask_fn: M,
) -> GenerateResult
where
    M: FnMut(&[u32], &mut Vec<f32>),
{
    generate_constrained_via_cpu_q4k_streaming(
        weights,
        tokenizer,
        token_ids,
        max_tokens,
        index,
        mask_fn,
        |_, _, _| {},
    )
}

/// Streaming variant of [`generate_constrained_via_cpu_q4k`]. Greedy
/// under the mask; for sampling under mask see
/// [`generate_constrained_via_cpu_q4k_streaming_sampled`].
pub(super) fn generate_constrained_via_cpu_q4k_streaming<M, F>(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    max_tokens: usize,
    index: &larql_vindex::VectorIndex,
    mask_fn: M,
    on_token: F,
) -> GenerateResult
where
    M: FnMut(&[u32], &mut Vec<f32>),
    F: FnMut(u32, &str, f64),
{
    generate_constrained_via_cpu_q4k_streaming_sampled(
        weights,
        tokenizer,
        token_ids,
        max_tokens,
        index,
        mask_fn,
        on_token,
        super::sampling::SamplingConfig::greedy(),
    )
}

/// Sampling-aware bridge to the CPU Q4_K constrained decoder. Threads
/// the caller's `SamplingConfig` (temperature/top_p/seed/penalties)
/// through to token selection over the masked logits.
#[allow(clippy::too_many_arguments)]
pub(super) fn generate_constrained_via_cpu_q4k_streaming_sampled<M, F>(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    max_tokens: usize,
    index: &larql_vindex::VectorIndex,
    mask_fn: M,
    on_token: F,
    sampling: super::sampling::SamplingConfig,
) -> GenerateResult
where
    M: FnMut(&[u32], &mut Vec<f32>),
    F: FnMut(u32, &str, f64),
{
    let prefill_start = std::time::Instant::now();
    let out = crate::vindex::generate_q4k_cpu_constrained_streaming_sampled(
        weights, tokenizer, token_ids, max_tokens, index, mask_fn, on_token, sampling,
    );
    let total_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
    // Heuristic split: attribute the first token to prefill, the rest to
    // decode. Matches the semantics of the GPU path closely enough for
    // bench-report purposes without tracking per-step timing inside the
    // constrained CPU loop.
    let n = out.len();
    let (prefill_ms, decode_ms_each) = if n == 0 {
        (total_ms, 0.0)
    } else {
        let avg = total_ms / n as f64;
        (avg, avg)
    };
    let tokens: Vec<(String, f64)> = out.into_iter().map(|(t, _)| (t, 1.0)).collect();
    let decode_ms = (1..tokens.len()).map(|_| decode_ms_each).collect();
    GenerateResult {
        tokens,
        prefill_ms,
        decode_ms,
        stage_timings: StageTimings::default(),
    }
}
