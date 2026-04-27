//! LM-head loaders + KNN.
//!
//! Loads the output projection (vocab × hidden) in one of three formats:
//!
//! - **Q4_K** (`lm_head_q4.bin`): GPU Q4 matvec, ~1 ms on Metal.
//! - **f16**: adopted from the vindex's `embeddings.bin` when that file
//!   is IEEE-half (tied-embedding Gemma / Llama). Drives Metal's
//!   `f16_gemv` shader — half the memory-bandwidth of f32 without the
//!   5.6 GB heap clone that a dequantised lm_head would need on 31B.
//! - **f32** (`lm_head.bin` or cloned from `embed`): CPU BLAS fallback.
//!
//! `lm_head_knn_backend` dispatches in the order above, using the
//! cheapest available backend path for the loaded lm_head representation.
//! Sibling to `super::walk` (FFN) and `super::attn` (attention).

use std::sync::Arc;

use larql_models::quant::ggml::{
    K_QUANT_BLOCK_ELEMS, Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMS, Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS,
};

use crate::error::VindexError;
use crate::format::filenames::*;
use crate::mmap_util::mmap_optimized;

use crate::index::core::VectorIndex;

/// Numerator/denominator used to back-derive `vocab_size` from a Q4-packed
/// lm_head file's byte length. Q4_K (144 B / 256 elems) and Q4_0 (18 B / 32
/// elems) both rate at 0.5625 B/element, i.e. `9/16`. Knowing only the file
/// size and `hidden_size`, the inverse is `vocab = bytes * 16 / (hidden * 9)`.
const Q4_BYTES_PER_ELEM_NUM: usize = 9;
const Q4_BYTES_PER_ELEM_DEN: usize = 16;

// Compile-time invariants — if either constant ever changes, this assertion
// catches the byte-rate calc immediately rather than producing silent vocab
// inference drift.
const _: () = assert!(
    Q4_K_BLOCK_BYTES * Q4_BYTES_PER_ELEM_DEN == Q4_K_BLOCK_ELEMS * Q4_BYTES_PER_ELEM_NUM,
    "Q4_K byte rate drift: 144/256 must equal 9/16",
);
const _: () = assert!(
    Q4_0_BLOCK_BYTES * Q4_BYTES_PER_ELEM_DEN == Q4_0_BLOCK_ELEMS * Q4_BYTES_PER_ELEM_NUM,
    "Q4_0 byte rate drift: 18/32 must equal 9/16",
);

/// Read the manifest entry for `lm_head.weight` from `weight_manifest.json`,
/// if the manifest exists and contains an entry for that key. Returns `None`
/// when the manifest is absent (older vindexes) or doesn't list lm_head.
///
/// Used by `load_lm_head_q4` to assert the on-disk file matches the format
/// the reader is about to dispatch. The Q4_K-vs-Q4_0 byte-rate collision
/// (0.5625 B/elem in both formats) made silent format mismatches invisible
/// to file-size validation; checking the manifest's `kind` discriminator
/// catches the mismatch at load-time.
fn read_lm_head_manifest_kind(dir: &std::path::Path) -> Option<String> {
    let manifest_path = dir.join(WEIGHT_MANIFEST_JSON);
    let text = std::fs::read_to_string(&manifest_path).ok()?;
    let entries: Vec<crate::format::weights::write_f32::WeightEntry> =
        serde_json::from_str(&text).ok()?;
    entries
        .into_iter()
        .find(|e| e.key == "lm_head.weight")
        .map(|e| e.kind)
}

impl VectorIndex {
    /// Load Q4 lm_head for GPU logits (replaces CPU f32 lm_head KNN).
    ///
    /// When `weight_manifest.json` is present and lists `lm_head.weight`, the
    /// entry's `kind` must be `kind::TENSOR_Q4K` — anything else is treated
    /// as a writer/reader contract violation and rejected, since the matvec
    /// kernel dispatched here (`q4k_matvec` via `lm_head_knn_backend`) is
    /// Q4_K-specific. This blocks the regression where a Q4_0 file shipped
    /// under the Q4_K filename produced silent garbage logits.
    ///
    /// Older vindexes without a manifest entry for lm_head still load (the
    /// extractor wrote the file directly), but no format check happens.
    pub fn load_lm_head_q4(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join(LM_HEAD_Q4_BIN);
        if !path.exists() {
            return Err(VindexError::Parse("lm_head_q4.bin not found".into()));
        }
        if let Some(manifest_kind) = read_lm_head_manifest_kind(dir) {
            if manifest_kind != crate::format::weights::write_f32::kind::TENSOR_Q4K {
                return Err(VindexError::Parse(format!(
                    "lm_head_q4.bin manifest mismatch: expected kind \"{}\", \
                     found \"{}\". This indicates the vindex was extracted with \
                     a writer that disagrees with the Q4_K matvec dispatch path \
                     — refusing to load to avoid silent garbage logits.",
                    crate::format::weights::write_f32::kind::TENSOR_Q4K,
                    manifest_kind
                )));
            }
        }
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { mmap_optimized(&file)? };
        // Derive `vocab_size` from the file size when it's still 0. Q4_K and
        // Q4_0 share the 9/16 byte-rate (`Q4_BYTES_PER_ELEM_*`), so the same
        // divisor handles both formats. Mirrors the pattern in `load_lm_head`
        // for f32 lm_head files.
        if self.vocab_size == 0 && self.hidden_size > 0 {
            let bytes = mmap.len();
            let denom = self.hidden_size * Q4_BYTES_PER_ELEM_NUM;
            if denom > 0 {
                let vocab = (bytes * Q4_BYTES_PER_ELEM_DEN) / denom;
                if vocab > 0 {
                    self.vocab_size = vocab;
                }
            }
        }
        self.projections.lm_head_q4_mmap = Some(Arc::new(mmap));
        Ok(())
    }

    /// Whether Q4 lm_head is loaded (from file or synthesized from f16 embeddings).
    pub fn has_lm_head_q4(&self) -> bool {
        self.projections.lm_head_q4_mmap.is_some() || self.projections.lm_head_q4_synth.is_some()
    }

    /// Synthesize Q4_0 lm_head in RAM from the f16 embeddings mmap.
    /// No-op if a Q4 source already exists or preconditions are not met.
    pub fn synthesize_lm_head_q4(&mut self) {
        if self.projections.lm_head_q4_mmap.is_some() || self.projections.lm_head_q4_synth.is_some()
        {
            return;
        }
        let vocab = self.vocab_size;
        let hidden = self.hidden_size;
        // Q4_K quantises in `K_QUANT_BLOCK_ELEMS`-element super-blocks, so
        // `hidden` must be a multiple of that (matches the on-disk
        // `lm_head_q4.bin` writer in `format/weights/write_q4k/mod.rs`).
        // Earlier code used Q4_0 (32-element blocks) here but
        // `lm_head_knn_backend` dispatches `q4k_matvec` for both the mmap and
        // synth paths — keeping the synth bytes in Q4_K avoids the format-
        // collision bug that broke gemma3-4b-v2.vindex (writer Q4_K vs reader
        // Q4_0).
        if vocab == 0 || hidden == 0 || !hidden.is_multiple_of(K_QUANT_BLOCK_ELEMS) {
            return;
        }
        let f16_mmap = match self.projections.lm_head_f16_mmap.as_ref() {
            Some(m) => m.clone(),
            None => return,
        };
        let expected = vocab * hidden * 2;
        if f16_mmap.len() < expected {
            return;
        }
        // Decode the whole f16 mmap to f32 in one pass, then Q4_K-quantise
        // the flat `[vocab, hidden]` row-major data. Q4_K's 256-element
        // super-blocks fit cleanly into one row when `hidden` is a multiple
        // of 256, so a flat call gives the same row-by-row layout the
        // matvec kernel expects.
        let mut all_f32 = vec![0.0f32; vocab * hidden];
        for (i, slot) in all_f32.iter_mut().enumerate() {
            let off = i * 2;
            let bits = u16::from_le_bytes([f16_mmap[off], f16_mmap[off + 1]]);
            *slot = larql_models::quant::half::f16_to_f32(bits);
        }
        let q4k = larql_compute::cpu::ops::q4_common::quantize_q4_k(&all_f32);
        self.projections.lm_head_q4_synth = Some(Arc::new(q4k));
    }

    /// Adopt the vindex's f16 `embeddings.bin` mmap as an f16 view of the
    /// LM head. Safe only for tied-embedding models (Gemma 2/3/4, Llama
    /// when `tie_word_embeddings=true`) — the loader is responsible for
    /// gating. Caller must have already populated `vocab_size`.
    ///
    /// When set, `lm_head_knn_backend` prefers `ComputeBackend::f16_gemv`
    /// on the mmap'd bytes, avoiding the 5.6 GB f32 clone on Gemma 4 31B.
    pub fn set_lm_head_f16_mmap(&mut self, mmap: Arc<memmap2::Mmap>) {
        self.projections.lm_head_f16_mmap = Some(mmap);
    }

    /// Whether an f16 mmap view of the LM head is available.
    pub fn has_lm_head_f16(&self) -> bool {
        self.projections.lm_head_f16_mmap.is_some() && self.vocab_size > 0
    }

    // ── LM head (output projection) for vindex logits ──

    /// Load lm_head from lm_head.bin for KNN logit lookup.
    pub fn load_lm_head(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join(LM_HEAD_BIN);
        if !path.exists() {
            return Err(VindexError::Parse("lm_head.bin not found".into()));
        }
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { mmap_optimized(&file)? };
        // Detect vocab size from file size: vocab = file_bytes / (hidden_size * 4)
        let vocab = mmap.len() / (self.hidden_size * 4);
        self.vocab_size = vocab;
        self.projections.lm_head_mmap = Some(Arc::new(mmap));
        Ok(())
    }

    /// Whether lm_head is loaded for vindex logits.
    pub fn has_lm_head(&self) -> bool {
        self.projections.lm_head_mmap.is_some() && self.vocab_size > 0
    }

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
                                return vec![(idx, score)];
                            }
                        } else if let Some(hits) =
                            backend.f16_gemv_topk(&f16_mmap[..expected], x, vocab, hidden, top_k)
                        {
                            if !hits.is_empty() {
                                return hits;
                            }
                        }
                        if let Some(scores_vec) =
                            backend.f16_gemv(&f16_mmap[..expected], x, vocab, hidden)
                        {
                            return Self::top_k_sorted(scores_vec, top_k);
                        }
                    }
                }
            }
        }
        // 3. f32 BLAS fallback.
        self.lm_head_knn(query, top_k)
    }

    /// Sort `scores` by descending value and keep the top `top_k`. Shared
    /// by the Q4 / f16 / f32 paths above.
    ///
    /// Uses a size-K min-heap instead of `select_nth_unstable_by` so we
    /// don't materialise a 2MB `Vec<(u32, f32)>` for a 262K-vocab lm_head
    /// only to throw away 262K-K of it. For typical K=1..5 on Gemma 3 4B
    /// this drops the CPU portion of lm_head from ~0.5ms to ~50µs.
    fn top_k_sorted(scores: Vec<f32>, top_k: usize) -> Vec<(u32, f32)> {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// `top_k_sorted` is the shared reduce used by Q4 / f16 / f32 paths.
    /// Pin the contract: descending by score, capped at `top_k`.
    #[test]
    fn top_k_sorted_descending_and_capped() {
        let scores = vec![0.5f32, 0.1, 0.9, 0.3, 0.7];
        let top3 = VectorIndex::top_k_sorted(scores.clone(), 3);
        let tokens: Vec<u32> = top3.iter().map(|(t, _)| *t).collect();
        let probs: Vec<f32> = top3.iter().map(|(_, s)| *s).collect();
        assert_eq!(
            tokens,
            vec![2, 4, 0],
            "expect descending-by-score token order"
        );
        assert!(probs[0] > probs[1] && probs[1] > probs[2]);

        // top_k larger than input → no truncation, but still sorted.
        let all = VectorIndex::top_k_sorted(scores, 99);
        assert_eq!(all.len(), 5);
        let probs: Vec<f32> = all.iter().map(|(_, s)| *s).collect();
        assert!(probs.windows(2).all(|w| w[0] >= w[1]));
    }

    /// `top_k = 0` returns an empty Vec, never the input.
    #[test]
    fn top_k_sorted_zero_returns_empty() {
        let scores = vec![0.5f32, 0.1, 0.9];
        let out = VectorIndex::top_k_sorted(scores, 0);
        assert!(out.is_empty());
    }

    /// Empty score vector → empty output (no panic).
    #[test]
    fn top_k_sorted_empty_input_returns_empty() {
        let out = VectorIndex::top_k_sorted(Vec::new(), 5);
        assert!(out.is_empty());
    }

    /// `top_k = 1` takes the argmax fast path. Filter is `is_finite()` —
    /// NaN, +∞ and -∞ are all skipped (matching `backend_lm_head_topk` in
    /// the inference crate). Test pins this contract: the highest finite
    /// score wins, regardless of any ±∞ entries.
    #[test]
    fn top_k_sorted_k1_argmax_skips_non_finite() {
        let scores = vec![0.2f32, f32::NAN, 0.9, f32::NEG_INFINITY, 0.5, f32::INFINITY];
        let out = VectorIndex::top_k_sorted(scores, 1);
        assert_eq!(out.len(), 1, "expected one finite winner");
        assert_eq!(out[0].0, 2, "highest finite score is 0.9 at idx 2");
        assert!((out[0].1 - 0.9).abs() < 1e-6);
    }

    /// All-NaN scores yield an empty argmax (no garbage token id).
    #[test]
    fn top_k_sorted_k1_all_nan_returns_empty() {
        let scores = vec![f32::NAN; 10];
        let out = VectorIndex::top_k_sorted(scores, 1);
        assert!(out.is_empty());
    }

    /// Heap path (k=3) skips non-finite values and returns sorted descending.
    #[test]
    fn top_k_sorted_heap_skips_non_finite() {
        let scores = vec![0.1f32, f32::NAN, 0.9, 0.5, f32::NEG_INFINITY, 0.3];
        let out = VectorIndex::top_k_sorted(scores, 3);
        let tokens: Vec<u32> = out.iter().map(|(t, _)| *t).collect();
        assert_eq!(tokens, vec![2, 3, 5]);
    }

    /// Fewer finite values than k → return only the finite ones, sorted.
    #[test]
    fn top_k_sorted_heap_fewer_finite_than_k() {
        let scores = vec![0.7f32, f32::NAN, 0.3, f32::NAN, f32::NAN];
        let out = VectorIndex::top_k_sorted(scores, 5);
        let tokens: Vec<u32> = out.iter().map(|(t, _)| *t).collect();
        assert_eq!(tokens, vec![0, 2]);
    }

    /// Tied scores: return is descending by score; tied tokens are still
    /// distinct (no duplicate index). Stability of which tied index wins
    /// is implementation-defined.
    #[test]
    fn top_k_sorted_handles_ties() {
        let scores = vec![0.5f32, 0.7, 0.5, 0.7, 0.1];
        let out = VectorIndex::top_k_sorted(scores, 3);
        assert_eq!(out.len(), 3);
        let probs: Vec<f32> = out.iter().map(|(_, s)| *s).collect();
        assert!(probs.windows(2).all(|w| w[0] >= w[1]));
        let tokens: std::collections::HashSet<u32> = out.iter().map(|(t, _)| *t).collect();
        assert_eq!(tokens.len(), 3, "no duplicate token ids in top-k output");
    }

    /// `synthesize_lm_head_q4` converts f16 embeddings to Q4_0 in RAM.
    ///
    /// Invariants:
    ///   - `has_lm_head_q4` false before synthesis, true after.
    ///   - Output byte length = vocab × (hidden/32 × 18).
    ///   - Re-quantizing a row via CPU path gives dot-product scores that rank
    ///     the matching row first (round-trip correctness).
    #[test]
    fn synthesize_lm_head_q4_produces_correct_bytes() {
        use std::sync::Arc;

        let vocab: usize = 16;
        // Q4_K uses 256-element super-blocks; the synth path now matches
        // the on-disk `lm_head_q4.bin` writer (Q4_K) so hidden must be a
        // multiple of 256. Earlier this used 64 (Q4_0's 32-elem blocks)
        // and the synth emitted Q4_0, which silently corrupted logits
        // when `lm_head_knn_backend` dispatched `q4k_matvec` on it.
        let hidden: usize = 256;

        // Build a synthetic f16 embedding table: row i = constant (i+1) * 0.01
        let mut f16_bytes = vec![0u8; vocab * hidden * 2];
        for row in 0..vocab {
            let val = (row as f32 + 1.0) * 0.01;
            let bits = larql_models::quant::half::f32_to_f16(val);
            for col in 0..hidden {
                let off = (row * hidden + col) * 2;
                let b = bits.to_le_bytes();
                f16_bytes[off] = b[0];
                f16_bytes[off + 1] = b[1];
            }
        }

        // Minimal VectorIndex with the f16 mmap and known dims.
        let mmap = Arc::new({
            let mem = memmap2::MmapMut::map_anon(f16_bytes.len()).unwrap();
            let mut mem = mem;
            mem.copy_from_slice(&f16_bytes);
            mem.make_read_only().unwrap()
        });

        let mut index =
            crate::index::core::VectorIndex::new(vec![None; 1], vec![None; 1], 1, hidden);
        index.vocab_size = vocab;
        index.set_lm_head_f16_mmap(mmap);

        assert!(
            !index.has_lm_head_q4(),
            "should not have Q4 before synthesis"
        );
        index.synthesize_lm_head_q4();
        assert!(index.has_lm_head_q4(), "should have Q4 after synthesis");

        // Byte length check uses canonical Q4_K block geometry from
        // `larql-models::quant::ggml` so the test fails immediately if the
        // writer ever switches blocks under us.
        let synth = index.projections.lm_head_q4_synth.as_ref().unwrap();
        let super_blocks = (vocab * hidden) / Q4_K_BLOCK_ELEMS;
        assert_eq!(
            synth.len(),
            super_blocks * Q4_K_BLOCK_BYTES,
            "synthesized Q4_K byte length should be \
             (vocab × hidden / Q4_K_BLOCK_ELEMS) × Q4_K_BLOCK_BYTES — \
             a different rate (e.g. /Q4_0_BLOCK_ELEMS × Q4_0_BLOCK_BYTES) means \
             the synth path has drifted from the on-disk Q4_K writer and \
             `q4k_matvec` will read it as garbage. Same byte rate (0.5625 \
             B/elem) makes this regression silent without an explicit \
             super-block count check."
        );

        // Calling again should be a no-op (idempotent).
        let ptr_before = synth.as_ptr();
        index.synthesize_lm_head_q4();
        let ptr_after = index
            .projections
            .lm_head_q4_synth
            .as_ref()
            .unwrap()
            .as_ptr();
        assert_eq!(ptr_before, ptr_after, "second call should not reallocate");
    }

    /// Regression: a vindex shipping `lm_head_q4.bin` but no `lm_head.bin`
    /// (the post-2026-04-26 Q4_K writer's default) used to leave
    /// `vocab_size = 0`. The Q4 lm_head fast path then silently bailed
    /// (`if vocab > 0`), forcing a 4× slower fallback through the f32
    /// BLAS gemv on `weights.lm_head`. This test pins the fix:
    /// `load_lm_head_q4` must populate `vocab_size` from the file size
    /// when no other source has set it.
    #[test]
    fn load_lm_head_q4_sets_vocab_size_from_file_size() {
        // Q4_K and Q4_0 both rate at `Q4_BYTES_PER_ELEM_NUM /
        // Q4_BYTES_PER_ELEM_DEN` (= 9/16 = 0.5625 B/elem), so the same
        // formula handles both. vocab=256 × hidden=128 → 18432 bytes.
        let hidden = 128usize;
        let vocab = 256usize;
        let bytes = vocab * hidden * Q4_BYTES_PER_ELEM_NUM / Q4_BYTES_PER_ELEM_DEN;
        let payload = vec![0u8; bytes];

        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join(LM_HEAD_Q4_BIN), &payload).unwrap();

        // Build a minimal index — vocab_size starts at 0.
        let mut index = VectorIndex::empty(1, hidden);
        assert_eq!(index.vocab_size, 0);

        index.load_lm_head_q4(tmp.path()).expect("load lm_head_q4");

        assert_eq!(
            index.vocab_size, vocab,
            "load_lm_head_q4 must derive vocab_size from file size when it's 0"
        );
    }

    /// Companion: when `vocab_size` is *already* set (by index.json or
    /// `load_lm_head`), `load_lm_head_q4` must not clobber it.
    #[test]
    fn load_lm_head_q4_does_not_overwrite_existing_vocab_size() {
        let hidden = 128usize;
        let bytes = 256 * hidden * Q4_BYTES_PER_ELEM_NUM / Q4_BYTES_PER_ELEM_DEN;
        let payload = vec![0u8; bytes];
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join(LM_HEAD_Q4_BIN), &payload).unwrap();

        let mut index = VectorIndex::empty(1, hidden);
        index.vocab_size = 999; // pretend index.json already set this
        index.load_lm_head_q4(tmp.path()).unwrap();

        assert_eq!(index.vocab_size, 999, "must not clobber preset vocab_size");
    }

    /// Companion: `load_lm_head_q4` is a no-op for vocab_size when the
    /// hidden_size is 0 (avoid div-by-zero / nonsense vocab).
    #[test]
    fn load_lm_head_q4_skips_vocab_inference_when_hidden_size_zero() {
        let payload = vec![0u8; 100];
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join(LM_HEAD_Q4_BIN), &payload).unwrap();

        let mut index = VectorIndex::empty(1, 0);
        index.load_lm_head_q4(tmp.path()).unwrap();
        assert_eq!(index.vocab_size, 0, "no inference possible without hidden_size");
    }

    /// Regression test for the gemma3-4b-v2 garbage-output bug (2026-04-27):
    /// `format/weights/write_q4k::write_model_weights_q4k` writes
    /// `lm_head_q4.bin` as **Q4_K** (144 B / 256 elems with sub-block
    /// scales/mins). `lm_head_knn_backend` previously dispatched
    /// `backend.q4_matvec` which is **Q4_0** (18 B / 32 elems with one f16
    /// scale): same byte rate, completely different layout, silent garbage.
    ///
    /// This pins the contract that the two ends of the pipeline agree on
    /// the format. Round-trip a known matrix through the writer's
    /// quantiser, run it through `lm_head_knn_backend`, and assert the
    /// top-1 token matches the f32 dot-product reference.
    #[test]
    fn lm_head_q4k_writer_reader_format_round_trip() {
        // Q4_K constraint: hidden must be a multiple of 256, vocab*hidden
        // must be a multiple of 256. 256×256 satisfies both with cheap
        // numerical work for a unit test.
        let vocab = 256usize;
        let hidden = 256usize;

        // Build a deterministic, well-conditioned [vocab, hidden] matrix.
        // Each row has a peak at one column so the f32 reference has an
        // unambiguous top-1 answer for any one-hot-ish query, while
        // sub-block scales/mins are non-trivial (Q4_K is structure-aware).
        let mut lm_head = vec![0.0f32; vocab * hidden];
        for v in 0..vocab {
            for h in 0..hidden {
                // Peak shaped like a smooth Gaussian centred at column v%hidden,
                // with a small ramp for off-diagonal values.
                let dist = ((h as f32) - (v as f32 % hidden as f32)).abs();
                lm_head[v * hidden + h] = (-dist * 0.05).exp() + 0.001 * (h as f32);
            }
        }

        // Quantise via the SAME writer the production extractor uses.
        let q4k_bytes = larql_compute::cpu::ops::q4_common::quantize_q4_k(&lm_head);
        // Sanity: byte count matches the canonical Q4_K rate.
        assert_eq!(
            q4k_bytes.len(),
            vocab * hidden / Q4_K_BLOCK_ELEMS * Q4_K_BLOCK_BYTES,
            "Q4_K quant should produce Q4_K_BLOCK_BYTES per Q4_K_BLOCK_ELEMS-element super-block"
        );

        // Inject into a synthetic VectorIndex via the synth path.
        let mut index = VectorIndex::empty(1, hidden);
        index.vocab_size = vocab;
        index.projections.lm_head_q4_synth = Some(Arc::new(q4k_bytes));

        // Pick a query that points at a known peak — token 42's row peaks
        // at column 42, so the dot product is highest at row 42.
        let target_token = 42u32;
        let mut query = ndarray::Array1::<f32>::zeros(hidden);
        query[target_token as usize] = 1.0;

        // f32 reference: dot product of `query` against every row of `lm_head`.
        let ref_scores: Vec<f32> = (0..vocab)
            .map(|v| (0..hidden).map(|h| lm_head[v * hidden + h] * query[h]).sum())
            .collect();
        let ref_top1 = ref_scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        assert_eq!(
            ref_top1, target_token,
            "fixture sanity: f32 reference must pick token 42"
        );

        // Run through the production dispatch with a CPU backend.
        let cpu = larql_compute::CpuBackend;
        let hits = index.lm_head_knn_backend(&query, 5, &cpu);
        assert!(
            !hits.is_empty(),
            "lm_head_knn_backend returned empty — Q4_K dispatch silently failed; \
             this is exactly the format-collision bug the test exists to catch"
        );
        let (top_token, _) = hits[0];
        assert_eq!(
            top_token, target_token,
            "Q4_K-quantised lm_head must select the same top-1 token as the \
             f32 reference (within Q4_K noise on a Gaussian-peak fixture). \
             A mismatch here means the writer and reader disagree on the \
             quantisation format — most likely a regression of the \
             Q4_K-vs-Q4_0 dispatch confusion fixed in 2026-04-27. \
             ref_top1={ref_top1}, got={top_token}"
        );

        // Stronger: top-5 must include the target (ranking can shift by
        // ±1 from Q4_K noise on the smooth fixture, but not by hundreds).
        let top5_tokens: Vec<u32> = hits.iter().map(|(t, _)| *t).collect();
        assert!(
            top5_tokens.contains(&target_token),
            "top-5 must contain target token {target_token}, got {top5_tokens:?}"
        );
    }

    /// Companion: the synth path (`synthesize_lm_head_q4`) must produce
    /// the same Q4_K format as the on-disk writer. Earlier the synth path
    /// emitted Q4_0 while the writer emitted Q4_K — both ended up routed
    /// through `q4k_matvec` after the dispatch fix, so a Q4_0 synth would
    /// silently corrupt logits for tied-embedding models that take the
    /// synth branch.
    #[test]
    fn synth_q4_lm_head_uses_q4k_format() {
        let vocab = 256usize;
        let hidden = 256usize;

        // Build an f16 mmap-shaped buffer (vocab × hidden × 2 bytes).
        // Use simple values so f16 conversion round-trips cleanly.
        let mut f16_buf = vec![0u8; vocab * hidden * 2];
        for v in 0..vocab {
            for h in 0..hidden {
                let val = if h == v { 1.0f32 } else { 0.01 };
                let bits = larql_models::quant::half::f32_to_f16(val);
                let off = (v * hidden + h) * 2;
                f16_buf[off] = (bits & 0xff) as u8;
                f16_buf[off + 1] = ((bits >> 8) & 0xff) as u8;
            }
        }

        let mut index = VectorIndex::empty(1, hidden);
        index.vocab_size = vocab;
        index.set_lm_head_f16_mmap(Arc::new(memmap_from_bytes(&f16_buf)));
        index.synthesize_lm_head_q4();

        let synth = index
            .projections
            .lm_head_q4_synth
            .as_ref()
            .expect("synth must populate lm_head_q4_synth");
        // Q4_K size invariant: Q4_K_BLOCK_BYTES per Q4_K_BLOCK_ELEMS-element super-block.
        assert_eq!(
            synth.len(),
            vocab * hidden / Q4_K_BLOCK_ELEMS * Q4_K_BLOCK_BYTES,
            "synth must produce Q4_K-sized bytes \
             (Q4_K_BLOCK_BYTES B / Q4_K_BLOCK_ELEMS elems), not Q4_0-sized \
             (Q4_0_BLOCK_BYTES B / Q4_0_BLOCK_ELEMS elems). Same byte rate \
             per element makes this regression silent without this assert."
        );

        // Functional check: top-1 against an indicator query points at the
        // expected diagonal token.
        let target = 17u32;
        let mut query = ndarray::Array1::<f32>::zeros(hidden);
        query[target as usize] = 1.0;
        let cpu = larql_compute::CpuBackend;
        let hits = index.lm_head_knn_backend(&query, 5, &cpu);
        let top: Vec<u32> = hits.iter().map(|(t, _)| *t).collect();
        assert!(
            top.contains(&target),
            "synth Q4_K lm_head must rank target token {target} in top-5 \
             of an indicator query; got {top:?}"
        );
    }

    /// Helper: build a memmap2::Mmap-shaped byte source for tests. Writes
    /// to a tempfile and mmaps it back — the synth function holds an
    /// `Arc<Mmap>` so we can't fake it inline.
    fn memmap_from_bytes(bytes: &[u8]) -> memmap2::Mmap {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), bytes).unwrap();
        let f = std::fs::File::open(tmp.path()).unwrap();
        unsafe { memmap2::Mmap::map(&f).unwrap() }
    }

    /// Architectural regression test: when `weight_manifest.json` lists
    /// `lm_head.weight` with `kind != tensor_q4k`, `load_lm_head_q4` must
    /// refuse to load. This is the bug class that produced silent garbage
    /// logits in gemma3-4b-v2.vindex (writer Q4_K, reader Q4_0 dispatch).
    #[test]
    fn load_lm_head_q4_rejects_manifest_kind_mismatch() {
        let hidden = 128usize;
        let vocab = 256usize;
        let bytes = vocab * hidden * Q4_BYTES_PER_ELEM_NUM / Q4_BYTES_PER_ELEM_DEN;

        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join(LM_HEAD_Q4_BIN), vec![0u8; bytes]).unwrap();

        // Manifest claims lm_head is f16 — incompatible with Q4_K dispatch.
        let manifest = serde_json::json!([{
            "key": "lm_head.weight",
            "kind": crate::format::weights::write_f32::kind::TENSOR_F16,
            "shape": [vocab, hidden],
            "offset": 0,
            "length": bytes,
            "file": "lm_head_q4.bin",
        }]);
        std::fs::write(
            tmp.path().join(WEIGHT_MANIFEST_JSON),
            serde_json::to_string(&manifest).unwrap(),
        )
        .unwrap();

        let mut index = VectorIndex::empty(1, hidden);
        let result = index.load_lm_head_q4(tmp.path());
        assert!(
            result.is_err(),
            "load_lm_head_q4 must reject when manifest kind disagrees with TENSOR_Q4K"
        );
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("manifest mismatch"),
            "error must explain the mismatch, got: {err_msg}"
        );
    }

    /// Companion: when the manifest correctly tags lm_head as TENSOR_Q4K,
    /// loading proceeds normally.
    #[test]
    fn load_lm_head_q4_accepts_correct_manifest_kind() {
        let hidden = 128usize;
        let vocab = 256usize;
        let bytes = vocab * hidden * Q4_BYTES_PER_ELEM_NUM / Q4_BYTES_PER_ELEM_DEN;

        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join(LM_HEAD_Q4_BIN), vec![0u8; bytes]).unwrap();

        let manifest = serde_json::json!([{
            "key": "lm_head.weight",
            "kind": crate::format::weights::write_f32::kind::TENSOR_Q4K,
            "shape": [vocab, hidden],
            "offset": 0,
            "length": bytes,
            "file": "lm_head_q4.bin",
        }]);
        std::fs::write(
            tmp.path().join(WEIGHT_MANIFEST_JSON),
            serde_json::to_string(&manifest).unwrap(),
        )
        .unwrap();

        let mut index = VectorIndex::empty(1, hidden);
        index
            .load_lm_head_q4(tmp.path())
            .expect("matching manifest kind should load");
        assert_eq!(index.vocab_size, vocab);
    }
}
