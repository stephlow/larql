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
    // Metal q4k_matvec on the lm_head produces sub-percent logit drift
    // vs CPU q4k_matvec. Each row of the 262K-vocab × 2560-hidden matvec
    // is reduced across a 32-lane simdgroup with a 2-way inter-superblock
    // split (`q4k_matvec.rs::ix = lane & 1u`); CPU runs the same dot
    // product as a sequential per-element accumulator. Both paths use
    // f32 throughout but the reduction trees differ, and that's enough
    // to flip top-1 on close-call tokens. End-to-end symptom on Gemma 3
    // 4B: prompt "The capital of France is" continues with " Capital"
    // (capital C, no answer) on Metal vs " capital ... **Paris**" on
    // CPU; per-layer hidden parity holds at cos≥0.99995 across all 34
    // layers (`test_decode_consistency_gemma3_4b_2steps`), so the drift
    // is fully concentrated in the lm_head matvec.
    //
    // Default-route the lm_head through `CpuBackend` whenever the
    // active compute backend isn't already CPU; opt back into Metal
    // with `LARQL_METAL_LM_HEAD=1` (~1ms/tok faster but token-flip risk
    // on close-ranking pairs). Same correctness-over-speed pattern
    // shipped for the Metal MoE expert path.
    let prefer_cpu = std::env::var("LARQL_METAL_LM_HEAD").is_err();
    let is_metal_backend = backend.as_any().type_id() != std::any::TypeId::of::<CpuBackend>();
    if prefer_cpu && is_metal_backend {
        // Route to `lm_head_knn_backend_skip_q4k` — the same dispatch
        // chain as `lm_head_knn_backend` but starting at the stable f16
        // GEMV path instead of the production Q4_K matvec path.
        //
        // Why: Metal's `q4k_matvec` 32-lane simdgroup reduction drifts
        // ~1e-3 vs CPU's sequential accumulator (different reduction
        // tree, same f32 precision). On the 262K × 2560 lm_head matvec
        // that's enough to flip top-1 on close-call tokens (e.g.
        // " Capital" vs " capital" at decode step 1 of Gemma 3 4B).
        // Metal's `f16_gemv` shader uses a tighter reduction tree and
        // keeps top-1 stable end-to-end. Reads 1.3 GB of f16 weights
        // per token vs 2.6 GB for f32, and avoids the extra stride-32
        // Q4 correctness path now that tied-embedding f16 is available
        // in the hot path.
        //
        // For models where the f16 mmap isn't populated (no tied embed
        // / no f16 lm_head), this falls back to stride-32 Q4_K, then
        // `lm_head_knn` (f32 BLAS). The Q4_K Metal path stays opt-in
        // via `LARQL_METAL_LM_HEAD=1` for runs where the speed margin
        // matters more than top-1 stability; the stride-32 fallback can
        // be forced first with `LARQL_LM_HEAD_STRIDE32=1`.
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
/// or `-inf`.
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
