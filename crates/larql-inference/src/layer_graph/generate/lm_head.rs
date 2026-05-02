//! LM-head top-K helpers and constrained-decode token sampling.

use crate::model::ModelWeights;
use larql_compute::prelude::*;
use larql_compute::CpuBackend;

/// Top-K logits lookup that transparently handles models with tied
/// input/output embeddings (Gemma 2/3/4) whose vindex has no dedicated
/// `lm_head.bin` / `lm_head_q4.bin`.
///
/// Resolution order:
/// 1. Vindex-native KNN (`lm_head_knn_backend`) — fastest, used when the
///    vindex was built with a separate lm_head.
/// 2. CPU gemv against `weights.lm_head` — the loader fills this from
///    `embed.clone()` for tied-embedding models, so it's always populated
///    even when no lm_head file is present.
///
/// The second path is O(vocab * hidden) floats through the CPU, but that's
/// a one-shot matvec per generated token — negligible compared to the
/// per-layer attention + FFN. It lets every model generate tokens through
/// the Metal pipeline regardless of how its vindex was packaged.
pub fn lm_head_topk(
    index: &larql_vindex::VectorIndex,
    weights: &ModelWeights,
    query: &ndarray::Array1<f32>,
    top_k: usize,
    backend: &dyn ComputeBackend,
) -> Vec<(u32, f32)> {
    // Default route: `lm_head_knn_backend` — Metal `q4k_matvec` first
    // (1.85 ms/tok on Gemma 3 4B, was 2.95 ms via the stride-32 workaround
    // before the 2026-05-02 dispatch-geometry fix), f16 GEMV fallback for
    // vindexes lacking Q4_K lm_head bytes, f32 BLAS as last resort.
    //
    // `LARQL_LM_HEAD_SKIP_Q4K=1` routes through `_skip_q4k` instead
    // (stride-32 Q4_K → f16 → f32) for diagnostic A/B against the Q4_K
    // path. See `crates/larql-compute/PERFORMANCE.md` "Decision: lm_head
    // dispatch order" for the full root-cause history.
    let skip_q4k = matches!(
        std::env::var("LARQL_LM_HEAD_SKIP_Q4K").as_deref(),
        Ok("1") | Ok("true") | Ok("on") | Ok("yes")
    );
    let is_metal_backend = backend.as_any().type_id() != std::any::TypeId::of::<CpuBackend>();
    if skip_q4k && is_metal_backend {
        // Diagnostic path: skip the Q4_K Metal matvec and use stride-32
        // Q4_K (or f16 GEMV / f32 BLAS) instead. Useful for verifying
        // top-1 stability against a known-stable reduction tree, or for
        // vindexes where the Q4_K lm_head bytes aren't populated.
        let hits = index.lm_head_knn_backend_skip_q4k(query, top_k, backend);
        let all_zero = !hits.is_empty() && hits.iter().all(|(_, s)| *s == 0.0 || s.is_nan());
        if !hits.is_empty() && !all_zero {
            return hits;
        }
        return backend_lm_head_topk(weights, query, top_k, backend);
    }
    let hits = index.lm_head_knn_backend(query, top_k, backend);
    // Workaround for the prefill→decode boundary: on the first decode
    // step, the Metal `q4k_matvec` / `f16_gemv` for lm_head occasionally
    // returns an all-zeros score vector even though the query is healthy
    // (verified rms ≈ 4, max_abs ≈ 60). The underlying cause appears to
    // be a GPU command-buffer ordering edge case after the first
    // `decode_token_with_moe` call. Falling back to the CPU/backend
    // gemv path produces correct scores immediately.
    let all_zero = !hits.is_empty() && hits.iter().all(|(_, s)| *s == 0.0 || s.is_nan());
    if !hits.is_empty() && !all_zero {
        return hits;
    }
    backend_lm_head_topk(weights, query, top_k, backend)
}

/// LM-head top-K via the active ComputeBackend.
///
/// Performs a single gemv `scores[vocab] = lm_head[vocab, hidden] · query[hidden]`
/// by dispatching `matmul_transb(query[1, hidden], lm_head[vocab, hidden])`.
/// On Metal this is a GPU f32 gemv (under Apple Silicon unified memory the
/// 2.68 GB `weights.lm_head` is shared, not copied). On CPU it's the
/// BLAS fallback via the same trait method. Either way this replaces the
/// previous unconditional CPU `ndarray::dot`, which was ~26 ms/tok on
/// Gemma 3 4B — the dominant cost of real-vindex decode.
pub(super) fn backend_lm_head_topk(
    weights: &ModelWeights,
    query: &ndarray::Array1<f32>,
    top_k: usize,
    backend: &dyn ComputeBackend,
) -> Vec<(u32, f32)> {
    let lm = &weights.lm_head;
    if lm.is_empty() || query.is_empty() {
        return Vec::new();
    }
    let vocab = lm.shape()[0];
    let hidden = lm.shape()[1];
    if hidden != query.len() {
        return Vec::new();
    }

    let query_slice = match query.as_slice() {
        Some(s) => s,
        None => &query.to_vec(),
    };

    // Fast path for top-1 (greedy decode): GPU gemv + GPU argmax
    // reads back only 8 KB partial results instead of 1 MB, saving ~0.33ms.
    if top_k == 1 {
        if let Some((idx, score)) = backend.f32_gemv_topk1(lm.view(), query_slice) {
            return vec![(idx, score)];
        }
    }

    // General path: GPU gemv → full Vec<f32> → CPU top-k.
    let scores_vec: Vec<f32> = if let Some(s) = backend.f32_gemv(lm.view(), query_slice) {
        s
    } else {
        let q_row = match query.view().into_shape_with_order((1, hidden)) {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };
        backend.matmul_transb(q_row, lm.view()).row(0).to_vec()
    };

    // Fast path for greedy decode (top_k=1): a single linear scan avoids
    // allocating the full 262K×8=2MB indexed Vec and the select_nth pass.
    if top_k == 1 {
        let best = scores_vec
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, s)| s.is_finite())
            .fold(None::<(usize, f32)>, |acc, (i, s)| {
                Some(match acc {
                    None => (i, s),
                    Some((bi, bs)) => {
                        if s > bs {
                            (i, s)
                        } else {
                            (bi, bs)
                        }
                    }
                })
            });
        let _ = vocab;
        return match best {
            Some((i, s)) => vec![(i as u32, s)],
            None => vec![],
        };
    }

    // Min-heap of size k: O(k) space, O(N log k) time.
    // Avoids allocating the full 262K×8=2MB indexed Vec.
    let k = top_k.min(vocab);
    let _ = vocab;
    let mut heap: Vec<(f32, u32)> = Vec::with_capacity(k + 1);

    // sift-down to maintain min-heap property (smallest score at index 0).
    fn sift_down(h: &mut [(f32, u32)], mut i: usize) {
        let n = h.len();
        loop {
            let mut smallest = i;
            let l = 2 * i + 1;
            let r = 2 * i + 2;
            if l < n && h[l].0 < h[smallest].0 {
                smallest = l;
            }
            if r < n && h[r].0 < h[smallest].0 {
                smallest = r;
            }
            if smallest == i {
                break;
            }
            h.swap(i, smallest);
            i = smallest;
        }
    }

    for (i, &s) in scores_vec.iter().enumerate() {
        if !s.is_finite() {
            continue;
        }
        if heap.len() < k {
            heap.push((s, i as u32));
            if heap.len() == k {
                // Build min-heap in O(k)
                for j in (0..k / 2).rev() {
                    sift_down(&mut heap, j);
                }
            }
        } else if s > heap[0].0 {
            heap[0] = (s, i as u32);
            sift_down(&mut heap, 0);
        }
    }
    // If we gathered fewer than k finite values, still heapify.
    if heap.len() < k && heap.len() > 1 {
        for j in (0..heap.len() / 2).rev() {
            sift_down(&mut heap, j);
        }
    }

    heap.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    heap.into_iter().map(|(s, i)| (i, s)).collect()
}

/// Kept for the `LARQL_METAL_COMPARE_CPU=1` diagnostic mode which wants a
/// known-good CPU reference. Not used in the hot path.
#[allow(dead_code)]
pub(super) fn cpu_lm_head_topk(
    weights: &ModelWeights,
    query: &ndarray::Array1<f32>,
    top_k: usize,
) -> Vec<(u32, f32)> {
    backend_lm_head_topk(weights, query, top_k, &larql_compute::CpuBackend)
}

/// Dense LM-head: full `Vec<f32>` of vocabulary scores. Required for
/// constrained decoding — the sparse vindex KNN can't apply an arbitrary
/// vocabulary mask because masked-out tokens might fall outside the top-K.
/// Same compute kernel as [`backend_lm_head_topk`], just no truncation.
pub(super) fn backend_lm_head_scores(
    weights: &ModelWeights,
    query: &ndarray::Array1<f32>,
    backend: &dyn ComputeBackend,
) -> Vec<f32> {
    let lm = &weights.lm_head;
    if lm.is_empty() || query.is_empty() {
        return Vec::new();
    }
    let hidden = lm.shape()[1];
    if hidden != query.len() {
        return Vec::new();
    }
    let query_slice = match query.as_slice() {
        Some(s) => s,
        None => &query.to_vec(),
    };
    if let Some(s) = backend.f32_gemv(lm.view(), query_slice) {
        s
    } else {
        let q_row = match query.view().into_shape_with_order((1, hidden)) {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };
        backend.matmul_transb(q_row, lm.view()).row(0).to_vec()
    }
}

/// Apply `mask_fn` to dense logits, then return the argmax `(id, score)`
/// over finite (post-mask) entries. Returns `None` if every entry is NaN
/// or `-inf`. Greedy under mask (no sampler).
pub(super) fn pick_next_token_masked<M>(
    weights: &ModelWeights,
    h_1d: &ndarray::Array1<f32>,
    generated: &[u32],
    backend: &dyn ComputeBackend,
    mask_fn: &mut M,
) -> Option<(u32, f32)>
where
    M: FnMut(&[u32], &mut Vec<f32>),
{
    let mut logits = backend_lm_head_scores(weights, h_1d, backend);
    if logits.is_empty() {
        return None;
    }
    mask_fn(generated, &mut logits);
    logits
        .iter()
        .enumerate()
        .filter(|(_, v)| !v.is_nan() && v.is_finite())
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, &s)| (i as u32, s))
}

/// Sampling-under-mask variant. Runs the dense LM head, applies the
/// mask, then defers token selection to the caller-supplied
/// [`Sampler`]. Repetition penalties on the sampler are applied as
/// usual via the `generated` history.
///
/// Returns `(id, raw_post_mask_score)` so callers that record per-token
/// probability still get the masked logit for the picked id (even
/// though the multinomial draw used the softmaxed distribution).
pub(super) fn pick_next_token_masked_sampled<M>(
    weights: &ModelWeights,
    h_1d: &ndarray::Array1<f32>,
    generated: &[u32],
    backend: &dyn ComputeBackend,
    mask_fn: &mut M,
    sampler: &mut super::sampling::Sampler,
) -> Option<(u32, f32)>
where
    M: FnMut(&[u32], &mut Vec<f32>),
{
    let mut logits = backend_lm_head_scores(weights, h_1d, backend);
    if logits.is_empty() {
        return None;
    }
    mask_fn(generated, &mut logits);
    let id = sampler.sample_with_history(&logits, generated)?;
    let score = *logits.get(id as usize)?;
    Some((id, score))
}
