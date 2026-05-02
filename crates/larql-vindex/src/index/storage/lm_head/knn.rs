//! LM-head KNN dispatch — Q4_K, f16, and f32 backend paths plus the
//! shared `top_k_sorted` reduce.
//!
//! `lm_head_knn_backend` picks the cheapest available format; the
//! `_skip_q4k` variant exists for backends whose Q4_K matvec has
//! reduction-tree drift on close-call tokens. Both paths share
//! `top_k_sorted` for the K-largest extraction so a future tweak (e.g.
//! widening the argmax fast path) lands in one place.

use crate::index::core::VectorIndex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stride32Mode {
    Disabled,
    Fallback,
    First,
}

fn lm_head_stride32_mode() -> Stride32Mode {
    match std::env::var("LARQL_LM_HEAD_STRIDE32") {
        Ok(v) if matches!(v.as_str(), "1" | "true" | "on" | "yes") => Stride32Mode::First,
        Ok(v) if matches!(v.as_str(), "0" | "false" | "off" | "no") => Stride32Mode::Disabled,
        _ => Stride32Mode::Fallback,
    }
}

impl VectorIndex {
    /// KNN against lm_head via a ComputeBackend. Tries paths in order:
    ///   1. Q4 matvec on `lm_head_q4.bin` (when present and backend has q4).
    ///   2. f16 gemv on the mmap'd embeddings (tied-embed models only).
    ///   3. f32 BLAS fallback via `lm_head_knn`.
    ///
    /// `top_k == 1` uses the GPU-argmax fast paths on backends that
    /// implement them, returning a single `(token_id, score)` without
    /// the 1MB scores readback + 262K-element CPU sort that the general
    /// path requires. Bench (greedy decode) takes this path.
    pub fn lm_head_knn_backend(
        &self,
        query: &ndarray::Array1<f32>,
        top_k: usize,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Vec<(u32, f32)> {
        // 1. Q4_K path — ~1 ms on Metal (mmap file or synthesized from f16 embeddings).
        //
        // The on-disk `lm_head_q4.bin` is written by `format/weights/write_q4k`
        // as **Q4_K** (144 bytes per 256 elements with sub-block scales/mins).
        // Earlier code dispatched `q4_matvec` (which is Q4_0 — 18 bytes per 32
        // elements with one f16 scale): the byte-rate happens to match
        // (0.5625 B/element) so file size was identical, but the kernel read
        // Q4_K bytes as Q4_0 scales/quants and silently produced garbage
        // logits. Symptom: multilingual gibberish under `--metal` on any
        // vindex with a fresh `lm_head_q4.bin` (e.g. gemma3-4b-v2 extracted
        // 2026-04-27). Routing through `q4k_matvec` (which takes raw f32 x,
        // no Q8 step) restores the format match.
        if backend.has_q4() {
            let q4_bytes: Option<&[u8]> = self
                .projections
                .lm_head_q4_mmap
                .as_ref()
                .map(|m| m.as_ref() as &[u8])
                .or_else(|| {
                    self.projections
                        .lm_head_q4_synth
                        .as_ref()
                        .map(|v| v.as_slice())
                });
            if let Some(q4_data) = q4_bytes {
                let vocab = self.vocab_size;
                let hidden = self.hidden_size;
                if vocab > 0 {
                    if let Some(x) = query.as_slice() {
                        if let Some(scores_vec) = backend.q4k_matvec(q4_data, x, vocab, hidden) {
                            return Self::top_k_sorted(scores_vec, top_k);
                        }
                    }
                }
            }
        }
        // 2. f16 path — tied-embed Gemma, ~2× the bandwidth of Q4 but still
        //    half of f32 and avoids a 5.6 GB heap allocation on 31B.
        if let Some(hits) = self.lm_head_f16_backend_hits(query, top_k, backend) {
            return hits;
        }
        // 3. f32 BLAS fallback.
        self.lm_head_knn(query, top_k)
    }

    /// Diagnostic alternative to `lm_head_knn_backend` — skips the
    /// production `q4k_matvec` path and tries stable-reduction
    /// alternatives in this order:
    ///
    ///   1. **Stride-32 Q4_K matvec** (`backend.q4k_matvec_stride32`) on
    ///      the same Q4_K bytes — same bandwidth as production
    ///      `q4k_matvec` (~327 MB/token), but with `f16_gemv`'s
    ///      reduction tree. ~2.95 ms/tok lm_head on Gemma 3 4B v2.
    ///   2. f16 GEMV on `embeddings.bin` mmap (tied-embed only).
    ///      Fallback when Q4_K bytes aren't populated. ~3.88 ms/tok.
    ///   3. f32 BLAS fallback (`lm_head_knn`).
    ///
    /// **History:** before 2026-05-02 this was the production default,
    /// because `lm_head_knn_backend` (which calls `q4k_matvec`) was
    /// producing argmax drift on close-call tokens. Root cause turned
    /// out to be a dispatch geometry mismatch in `MetalBackend::q4k_matvec`,
    /// not a kernel-level reduction-tree drift. With the dispatch fix,
    /// `q4k_matvec` is correct AND ~1.10 ms/tok faster than stride-32,
    /// so the canonical chain is now the default and this path is
    /// reachable via `LARQL_LM_HEAD_SKIP_Q4K=1` as a diagnostic A/B.
    /// See `PERFORMANCE.md` "Decision: lm_head dispatch order" for
    /// the full root-cause write-up.
    ///
    /// Env-var overrides (within this fallback chain):
    ///   - `LARQL_LM_HEAD_STRIDE32=0` — disable stride-32 entirely; go
    ///     straight to f16 (then f32). Used to A/B the stride-32 win.
    ///
    /// `lm_head_topk` in `larql-inference::layer_graph::generate::lm_head`
    /// routes here only when `LARQL_LM_HEAD_SKIP_Q4K=1` is set on a
    /// non-CPU backend; the canonical path is `lm_head_knn_backend`.
    pub fn lm_head_knn_backend_skip_q4k(
        &self,
        query: &ndarray::Array1<f32>,
        top_k: usize,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Vec<(u32, f32)> {
        let stride32_mode = lm_head_stride32_mode();

        // 1. Default stable path: stride-32 Q4_K matvec — Q4_K bandwidth
        //    win + f16_gemv's stable reduction tree. Skipped when
        //    `LARQL_LM_HEAD_STRIDE32=0`.
        if stride32_mode != Stride32Mode::Disabled {
            if let Some(hits) = self.lm_head_stride32_backend_hits(query, top_k, backend) {
                return hits;
            }
        }

        // 2. f16 GEMV fallback for vindexes lacking Q4_K lm_head bytes.
        if let Some(hits) = self.lm_head_f16_backend_hits(query, top_k, backend) {
            return hits;
        }

        // 3. f32 BLAS last resort.
        self.lm_head_knn(query, top_k)
    }

    fn lm_head_f16_backend_hits(
        &self,
        query: &ndarray::Array1<f32>,
        top_k: usize,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Option<Vec<(u32, f32)>> {
        if let Some(ref f16_mmap) = self.projections.lm_head_f16_mmap {
            let vocab = self.vocab_size;
            let hidden = self.hidden_size;
            if vocab > 0 {
                let expected = vocab * hidden * 2;
                if f16_mmap.len() >= expected {
                    if let Some(x) = query.as_slice() {
                        if top_k == 1 {
                            if let Some((idx, score)) =
                                backend.f16_gemv_topk1(&f16_mmap[..expected], x, vocab, hidden)
                            {
                                return Some(vec![(idx, score)]);
                            }
                        } else if let Some(hits) =
                            backend.f16_gemv_topk(&f16_mmap[..expected], x, vocab, hidden, top_k)
                        {
                            if !hits.is_empty() {
                                return Some(hits);
                            }
                        }
                        if let Some(scores_vec) =
                            backend.f16_gemv(&f16_mmap[..expected], x, vocab, hidden)
                        {
                            return Some(Self::top_k_sorted(scores_vec, top_k));
                        }
                    }
                }
            }
        }
        None
    }

    fn lm_head_stride32_backend_hits(
        &self,
        query: &ndarray::Array1<f32>,
        top_k: usize,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Option<Vec<(u32, f32)>> {
        if !backend.has_q4() {
            return None;
        }
        let q4_bytes: Option<&[u8]> = self
            .projections
            .lm_head_q4_mmap
            .as_ref()
            .map(|m| m.as_ref() as &[u8])
            .or_else(|| {
                self.projections
                    .lm_head_q4_synth
                    .as_ref()
                    .map(|v| v.as_slice())
            });
        let q4_data = q4_bytes?;
        let vocab = self.vocab_size;
        let hidden = self.hidden_size;
        if vocab == 0 {
            return None;
        }
        let x = query.as_slice()?;
        backend
            .q4k_matvec_stride32(q4_data, x, vocab, hidden)
            .map(|scores_vec| Self::top_k_sorted(scores_vec, top_k))
    }

    /// Sort `scores` by descending value and keep the top `top_k`. Shared
    /// by the Q4 / f16 / f32 paths above.
    ///
    /// Uses a size-K min-heap instead of `select_nth_unstable_by` so we
    /// don't materialise a 2MB `Vec<(u32, f32)>` for a 262K-vocab lm_head
    /// only to throw away 262K-K of it. For typical K=1..5 on Gemma 3 4B
    /// this drops the CPU portion of lm_head from ~0.5ms to ~50µs.
    ///
    /// Visibility note: `pub(super)` so the `mod tests` in `lm_head/mod.rs`
    /// can keep its existing `VectorIndex::top_k_sorted(...)` call sites
    /// after the M9 file split.
    pub(super) fn top_k_sorted(scores: Vec<f32>, top_k: usize) -> Vec<(u32, f32)> {
        if scores.is_empty() || top_k == 0 {
            return Vec::new();
        }
        let k = top_k.min(scores.len());

        // Argmax fast path — no heap, single linear scan.
        if k == 1 {
            let mut best_i: u32 = 0;
            let mut best_v = f32::NEG_INFINITY;
            for (i, &s) in scores.iter().enumerate() {
                if s.is_finite() && s > best_v {
                    best_v = s;
                    best_i = i as u32;
                }
            }
            if best_v == f32::NEG_INFINITY {
                return Vec::new();
            }
            return vec![(best_i, best_v)];
        }

        // Min-heap of size K, smallest score at index 0. We push until full,
        // then replace-and-sift-down whenever we see something larger than
        // the current min.
        let mut heap: Vec<(f32, u32)> = Vec::with_capacity(k + 1);

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

        for (i, &s) in scores.iter().enumerate() {
            if !s.is_finite() {
                continue;
            }
            if heap.len() < k {
                heap.push((s, i as u32));
                if heap.len() == k {
                    for j in (0..k / 2).rev() {
                        sift_down(&mut heap, j);
                    }
                }
            } else if s > heap[0].0 {
                heap[0] = (s, i as u32);
                sift_down(&mut heap, 0);
            }
        }
        if heap.len() < k && heap.len() > 1 {
            for j in (0..heap.len() / 2).rev() {
                sift_down(&mut heap, j);
            }
        }

        heap.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        heap.into_iter().map(|(s, i)| (i, s)).collect()
    }

    /// KNN against lm_head: find top-K tokens by dot product with query vector.
    /// Single BLAS gemv: query[1, hidden] @ lm_head[vocab, hidden]^T → [1, vocab].
    /// Then top-K selection. Returns (token_id, score) sorted by score descending.
    pub fn lm_head_knn(&self, query: &ndarray::Array1<f32>, top_k: usize) -> Vec<(u32, f32)> {
        let mmap = match self.projections.lm_head_mmap.as_ref() {
            Some(m) => m,
            None => return vec![],
        };
        let vocab = self.vocab_size;
        let hidden = self.hidden_size;
        if vocab == 0 {
            return vec![];
        }

        let expected = vocab * hidden * 4;
        if mmap.len() < expected {
            return vec![];
        }

        // Zero-copy: reinterpret mmap as [vocab, hidden] f32 matrix
        let data = unsafe {
            let ptr = mmap.as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, vocab * hidden)
        };
        let lm_view = ndarray::ArrayView2::from_shape((vocab, hidden), data).unwrap();

        // gemv via larql-compute: scores = query @ lm_head^T → [1, vocab]
        let hidden = self.hidden_size;
        let x = query.view().into_shape_with_order((1, hidden)).unwrap();
        let cpu = larql_compute::CpuBackend;
        use larql_compute::MatMul;
        let result = cpu.matmul_transb(x, lm_view); // [1, hidden] @ [vocab, hidden]^T → [1, vocab]
        let scores = ndarray::Array1::from_vec(result.into_raw_vec_and_offset().0);

        // Top-K selection
        let mut indexed: Vec<(u32, f32)> = scores
            .iter()
            .copied()
            .enumerate()
            .map(|(i, s)| (i as u32, s))
            .collect();
        let k = top_k.min(indexed.len());
        if k > 0 && k < indexed.len() {
            indexed.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
            indexed.truncate(k);
        }
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed
    }
}
