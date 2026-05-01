//! POST /v1/expert/{layer}/{expert_id} — remote expert endpoint for MoE inference.
//!
//! A shard server started with `--experts START-END` owns a contiguous range of
//! experts. The inference client routes individual expert calls to the right
//! shard rather than running all experts locally.
//!
//! # Single expert
//!   POST /v1/expert/{layer}/{expert_id}
//!   Body: {"residual": [f32...]}
//!   Response: {"output": [f32...], "latency_ms": f64}
//!
//! # Batch (multiple experts in one round-trip)
//!   POST /v1/expert/batch
//!   Body: {"requests": [{"layer": usize, "expert_id": usize, "residual": [f32...]}, ...]}
//!   Response: {"results": [{"layer": usize, "expert_id": usize, "output": [f32...]}, ...], "latency_ms": f64}

use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::header;
use axum::response::Response;
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::error::ServerError;
use crate::state::AppState;
use larql_inference;
use larql_inference::ffn::moe_remote::{
    decode_expert_request, decode_layer_batch_request, decode_layer_batch_request_f16,
    encode_expert_response, encode_layer_batch_response, encode_layer_batch_response_f16,
    ExpertCallItem, ExpertResultItem, EXPERT_BINARY_CONTENT_TYPE, LAYER_BATCH_CONTENT_TYPE,
    LAYER_BATCH_F16_CONTENT_TYPE,
};

// ── Request / response types ──────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct SingleExpertRequest {
    pub residual: Vec<f32>,
}

#[derive(Serialize)]
pub struct SingleExpertResponse {
    pub output: Vec<f32>,
    pub latency_ms: f64,
}

#[derive(Deserialize)]
pub struct BatchExpertItem {
    pub layer: usize,
    pub expert_id: usize,
    pub residual: Vec<f32>,
}

#[derive(Deserialize)]
pub struct BatchExpertRequest {
    pub requests: Vec<BatchExpertItem>,
}

#[derive(Serialize)]
pub struct BatchExpertResult {
    pub layer: usize,
    pub expert_id: usize,
    pub output: Vec<f32>,
}

#[derive(Serialize)]
pub struct BatchExpertResponse {
    pub results: Vec<BatchExpertResult>,
    pub latency_ms: f64,
}

// ── Core computation ──────────────────────────────────────────────────────────

/// CPU expert dispatch with pre_norm hoisted out of the per-expert loop and
/// allocation-free per-expert compute via `ExpertScratch`.  Replaces the old
/// `expert_ids.par_iter().filter_map(|&eid| run_expert(...)).collect()` pattern
/// where every expert call:
///   1. re-applied `pre_experts_norm` to the same residual (K× wasted work),
///   2. re-allocated three Vec<f32> per matmul (3 × K × num_layers per token).
///
/// New flow:
///   - apply `pre_experts_norm` once per frame, store h_norm
///   - rayon par_iter over K experts; each rayon worker reuses a thread-local
///     `ExpertScratch` for matmul outputs and activation
///   - weighted sum of partials into the result
///
/// Returns the same `Vec<f32>` (length = hidden) as the old code path —
/// callers see no behavioural change, only fewer allocations and no
/// redundant rms_norm work.
pub fn run_experts_cpu_batch(
    state: &AppState,
    layer: usize,
    h_post_attn: &[f32],
    expert_ids: &[usize],
    expert_weights: &[f32],
) -> Result<Vec<f32>, ServerError> {
    use larql_compute::cpu::ops::moe::{
        pre_experts_norm, quantize_h_norm_for_q4k, run_single_expert_into,
        run_single_expert_q4k_q8k_into, ExpertScratch,
    };
    use std::time::Instant;
    let timing_enabled = std::env::var("LARQL_MOE_TIMING").is_ok();
    let t_start = Instant::now();

    let model = state.model_or_err(None)?;
    let weights = model
        .get_or_load_weights()
        .map_err(ServerError::InferenceUnavailable)?;
    let arch = &*weights.arch;
    let hidden = h_post_attn.len();
    if hidden == 0 || expert_ids.is_empty() {
        return Ok(vec![0.0f32; hidden]);
    }
    let inter = arch.moe_intermediate_size();
    let activation = larql_inference::activation_from_arch(arch);
    let inter_padded = if let Some(per_layer) = weights.has_per_layer_ffn().then_some(()) {
        let _ = per_layer;
        let block = larql_models::quant::ggml::Q4_K_BLOCK_ELEMS;
        inter.div_ceil(block) * block
    } else {
        inter
    };
    let t_arch = t_start.elapsed();

    // Hoist pre_experts_norm: same input residual for all K experts; rms_norm
    // is invariant of the expert id, so doing it once per frame saves K-1
    // redundant passes per layer.
    let t_norm_start = Instant::now();
    let pre_norm_slice: &[f32] = arch
        .moe_pre_experts_norm_key(layer)
        .and_then(|key| weights.vectors.get(&key))
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let h_norm = pre_experts_norm(
        h_post_attn,
        pre_norm_slice,
        arch.norm_weight_offset(),
        arch.norm_eps(),
    );
    let t_norm = t_norm_start.elapsed();

    // Per-rayon-thread scratch.  16 cores on M3 Max → up to 16 instances live
    // for the lifetime of the worker thread; replaces the old code's 3 fresh
    // Vec<f32> heap allocations per expert call.
    thread_local! {
        static SCRATCH: std::cell::RefCell<Option<ExpertScratch>> =
            const { std::cell::RefCell::new(None) };
    }

    let format = if weights.has_per_layer_ffn() {
        larql_inference::QuantFormat::Q4_K
    } else {
        larql_inference::QuantFormat::BF16
    };

    // For Q4_K weights, quantise h_norm to Q8_K once per layer (shared
    // across all K active experts).  Enables the SDOT-based direct-Q4K
    // matvec kernel — bypasses the f32 dequant cache entirely.  Default-on
    // when format is Q4_K and the activation length is divisible by 256
    // (always true for production hidden sizes); set
    // `LARQL_DISABLE_Q4K_DIRECT=1` to fall back to the BLAS-on-cached-f32
    // path (e.g. for kernel-debug A/B comparison).
    let q4k_direct = matches!(format, larql_inference::QuantFormat::Q4_K)
        && std::env::var("LARQL_DISABLE_Q4K_DIRECT").is_err();
    let h_norm_q8k = if q4k_direct {
        quantize_h_norm_for_q4k(&h_norm)
    } else {
        None
    };

    // Resolve (gate_up, down) bytes for one expert.  Pulled out of the
    // rayon closure so the closure body is small and the legacy BF16 path
    // doesn't fight the borrow checker on `weights` / `arch`.
    let resolve_bytes = |eid: usize| -> Option<(&[u8], &[u8])> {
        if weights.has_per_layer_ffn() {
            weights.get_layer_entry_bytes(layer, eid)
        } else {
            let gu_key = arch.packed_experts_gate_up_key(layer)?;
            let dn_key = arch.packed_experts_down_key(layer)?;
            let gu_all = weights.get_packed_bytes(&gu_key)?;
            let dn_all = weights.get_packed_bytes(&dn_key)?;
            let gu_stride = 2 * inter * hidden * 2; // BF16 = 2 bytes
            let dn_stride = hidden * inter * 2;
            let gu_start = eid * gu_stride;
            let dn_start = eid * dn_stride;
            if gu_start + gu_stride > gu_all.len() || dn_start + dn_stride > dn_all.len() {
                return None;
            }
            Some((
                &gu_all[gu_start..gu_start + gu_stride],
                &dn_all[dn_start..dn_start + dn_stride],
            ))
        }
    };

    // Fold the K experts directly into a per-worker hidden-sized accumulator,
    // then reduce across workers.  Replaces the prior pattern of collecting
    // K (Vec<f32>, weight) partials and serially summing them — that path
    // forced an 11 KB Vec allocation per expert per layer (≈2.7 MB/token at
    // 30 MoE layers × top-K=8) and serialized the final accumulation on one
    // thread.
    use rayon::prelude::*;
    let out = expert_ids
        .par_iter()
        .zip(expert_weights.par_iter())
        .filter(|(_, &w)| w != 0.0)
        .fold(
            || vec![0.0f32; hidden],
            |mut acc, (&eid, &w)| {
                let Some((gu_bytes, dn_bytes)) = resolve_bytes(eid) else {
                    return acc;
                };
                SCRATCH.with(|cell| {
                    let mut borrow = cell.borrow_mut();
                    let scratch = borrow
                        .get_or_insert_with(|| ExpertScratch::new(hidden, inter, inter_padded));
                    // Resize-on-shape-change: a single server might host multiple
                    // models with different shapes (rare, but cheap to handle).
                    if scratch.gate_out.len() != inter
                        || scratch.act.len() != inter_padded
                        || scratch.out.len() != hidden
                    {
                        *scratch = ExpertScratch::new(hidden, inter, inter_padded);
                    }
                    let h2 = if let Some(q8k) = h_norm_q8k.as_ref() {
                        run_single_expert_q4k_q8k_into(
                            scratch, q8k, gu_bytes, dn_bytes, inter, activation,
                        )
                    } else {
                        run_single_expert_into(
                            scratch, &h_norm, gu_bytes, dn_bytes, inter, format, activation,
                        )
                    };
                    for (a, &v) in acc.iter_mut().zip(h2.iter()) {
                        *a += w * v;
                    }
                });
                acc
            },
        )
        .reduce(
            || vec![0.0f32; hidden],
            |mut a, b| {
                for (x, &y) in a.iter_mut().zip(b.iter()) {
                    *x += y;
                }
                a
            },
        );

    let t_par = t_norm_start.elapsed() - t_norm;
    if timing_enabled {
        eprintln!(
            "[run_experts_cpu] layer={layer} K={} arch={:.2}ms norm={:.2}ms \
             par_fold={:.2}ms total={:.2}ms",
            expert_ids.len(),
            t_arch.as_secs_f32() * 1000.0,
            t_norm.as_secs_f32() * 1000.0,
            t_par.as_secs_f32() * 1000.0,
            t_start.elapsed().as_secs_f32() * 1000.0,
        );
    }
    Ok(out)
}

/// Eager warmup of the per-(layer, expert) HNSW unit cache for **walk** /
/// interpretability KNN queries.  Iterates every `(layer, expert)` this
/// shard owns and pre-builds an HNSW index over that expert's gate slice
/// (`moe_intermediate_size` vectors per unit, vs `num_experts ×
/// moe_intermediate_size` for the layer-level index).
///
/// Independent of the Metal expert cache: this is for the gate-KNN code
/// path (`gate_knn_expert`), not the MoE forward pass.  Skipped when
/// `LARQL_NO_WARMUP=1`.  Requires `--hnsw` to actually be useful at query
/// time, but the cache is populated regardless so flipping the toggle on
/// later doesn't pay a build burst.
///
/// Returns `(units_built, num_layers, experts_per_shard)` so the caller
/// can log a one-line summary.  All builds happen in parallel via rayon.
pub fn warmup_hnsw_unit_cache(
    model: &crate::state::LoadedModel,
) -> Result<(usize, usize, usize), String> {
    if std::env::var("LARQL_NO_WARMUP").is_ok() {
        return Ok((0, 0, 0));
    }
    let weights = model.get_or_load_weights()?;
    let arch = &*weights.arch;
    if !arch.is_hybrid_moe() {
        return Ok((0, 0, 0));
    }
    let num_layers = model.config.num_layers;
    let num_experts = arch.num_experts();
    let moe_inter = arch.moe_intermediate_size();
    if num_layers == 0 || moe_inter == 0 {
        return Ok((0, 0, 0));
    }
    // Resolve the (layer, expert_id) ownership set for this shard.
    // Priority: `--units` manifest (`unit_filter`) → `--experts START-END`
    // (`expert_filter`, layer-uniform) → all experts on every layer.
    let owned_units: Vec<(usize, usize)> = if let Some(units) = model.unit_filter.as_ref() {
        let mut v: Vec<(usize, usize)> = units.iter().copied().collect();
        v.sort_unstable();
        v
    } else {
        let (start, end_excl) = model.expert_filter.unwrap_or((0, num_experts));
        (0..num_layers)
            .flat_map(|l| (start..end_excl).map(move |e| (l, e)))
            .collect()
    };
    let n_experts_owned = if let Some(units) = model.unit_filter.as_ref() {
        units
            .iter()
            .map(|(_, e)| *e)
            .collect::<std::collections::HashSet<_>>()
            .len()
    } else {
        let (start, end_excl) = model.expert_filter.unwrap_or((0, num_experts));
        end_excl.saturating_sub(start)
    };

    // Build the (layer, feat_start, feat_end) triples for every owned unit.
    // feat_start_for_expert_e = e * moe_intermediate_size — same layout the
    // gate_knn_expert callers use.
    let mut units: Vec<(usize, usize, usize)> = Vec::with_capacity(owned_units.len());
    for (layer, eid) in owned_units {
        let fs = eid * moe_inter;
        let fe = (eid + 1) * moe_inter;
        units.push((layer, fs, fe));
    }

    // We need a `&VectorIndex` to call `warmup_hnsw_units`.  The patched
    // overlay's `blocking_read` exposes that synchronously — fine here
    // because this runs inside a `spawn_blocking` job during startup.
    let patched = model.patched.blocking_read();
    let n_built = patched.base().warmup_hnsw_units(&units);
    drop(patched);
    Ok((n_built, num_layers, n_experts_owned))
}

/// Eager warmup of the Metal expert buffer cache.
///
/// Iterates every `(layer, expert_id)` owned by this shard and calls
/// `cached_buffer_for_bytes` on the expert's gate_up + down mmap regions,
/// populating `BufferCache` so that subsequent RPC calls hit instantly
/// instead of paying the first-touch ~10–28ms Metal-buffer allocation.
///
/// Returns the number of (gate_up, down) buffer pairs staged.
///
/// Skipped when `LARQL_NO_WARMUP=1` (useful in low-RSS dev setups; warmup
/// allocates ~10MB × experts_owned × num_layers of Metal-resident memory).
#[cfg(feature = "metal-experts")]
pub fn warmup_metal_expert_cache(model: &crate::state::LoadedModel) -> Result<usize, String> {
    use larql_compute::MetalBackend;

    if std::env::var("LARQL_NO_WARMUP").is_ok() {
        return Ok(0);
    }

    let weights = model.get_or_load_weights()?;
    let arch = &*weights.arch;
    if !arch.is_hybrid_moe() || !weights.has_per_layer_ffn() {
        return Ok(0);
    }

    let backend_slot = model.metal_backend.get_or_init(MetalBackend::new);
    let Some(backend) = backend_slot.as_ref() else {
        return Ok(0);
    };

    let num_layers = model.config.num_layers;
    let num_experts = arch.num_experts();

    // Same ownership-resolution pattern as warmup_hnsw_unit_cache:
    // unit_filter > expert_filter > all.  See that function for rationale.
    let owned_units: Vec<(usize, usize)> = if let Some(units) = model.unit_filter.as_ref() {
        let mut v: Vec<(usize, usize)> = units.iter().copied().collect();
        v.sort_unstable();
        v
    } else {
        let (start, end_excl) = model.expert_filter.unwrap_or((0, num_experts));
        (0..num_layers)
            .flat_map(|l| (start..end_excl).map(move |e| (l, e)))
            .collect()
    };

    let mut staged = 0usize;
    for (layer, eid) in owned_units {
        if let Some((gu, dn)) = weights.get_layer_entry_bytes(layer, eid) {
            // Each call returns a cached Buffer; first call pays the
            // mmap → Metal allocation/copy, subsequent calls are O(1)
            // hash lookups.  We discard the returned Buffer here — the
            // cache holds it for the server's lifetime.
            let _ = backend.cached_buffer_for_bytes(gu);
            let _ = backend.cached_buffer_for_bytes(dn);
            staged += 1;
        }
    }
    Ok(staged)
}

/// Run a layer's pre-selected experts on the Metal GPU and return the weighted
/// sum of their outputs.  Returns `Ok(None)` when Metal is unavailable, the
/// model is not hybrid-MoE, or per-layer Q4_K weights are missing — caller
/// should fall back to the per-expert CPU path.
///
/// `h_post_attn` is the residual the streaming RPC carries (pre-norm not yet
/// applied).  `expert_ids` and `expert_weights` are already client-routed (no
/// router run on the server).  Returns the weighted sum WITHOUT post-experts
/// norm; the client applies post-norm once after summing across shards.
#[cfg(feature = "metal-experts")]
pub fn run_experts_metal_batch(
    state: &AppState,
    layer: usize,
    h_post_attn: &[f32],
    expert_ids: &[usize],
    expert_weights: &[f32],
) -> Result<Option<Vec<f32>>, ServerError> {
    use larql_compute::{MetalBackend, MoeScratch};
    use std::time::Instant;
    let timing_enabled = std::env::var("LARQL_MOE_TIMING").is_ok();
    // 2026-04-30 ACCURACY ISSUE: the Metal MoE expert dispatch (both
    // `run_experts_preselected_metal` and `run_experts_prestaged_metal`,
    // and the in-process `gpu_moe_dispatch_with_scratch`) produces
    // numerically wrong expert outputs for Gemma 4 26B-A4B-it (inter=704,
    // hidden=2816, top_k=8). Cosine similarity vs CPU reference ≈ 0.7;
    // |metal| consistently ~70% of |cpu|. Same model produces "Paris"
    // via CPU experts and "answer is in the context of France" via Metal
    // experts. Bug appears to be in the q4k_ffn_gate_up + GELU + q4k_matvec
    // chain when applied to the 704-wide intermediate dim — the same
    // shaders work correctly for dense FFN at inter=2560/10240/21504.
    // Until the kernel is fixed, default to CPU expert dispatch even on
    // a build that linked the Metal backend.  Set LARQL_USE_METAL_EXPERTS=1
    // to opt back in (e.g. for kernel-debugging runs).
    let use_metal = std::env::var("LARQL_USE_METAL_EXPERTS").is_ok();
    if !use_metal || std::env::var("LARQL_DISABLE_METAL_EXPERTS").is_ok() {
        return Ok(None);
    }
    let t_start = Instant::now();

    let model = state.model_or_err(None)?;
    let weights = model
        .get_or_load_weights()
        .map_err(ServerError::InferenceUnavailable)?;
    let arch = &*weights.arch;
    let t_state = t_start.elapsed();

    if !arch.is_hybrid_moe() || !weights.has_per_layer_ffn() {
        return Ok(None);
    }

    // Lazy-init the Metal backend.  `MetalBackend::new()` returns None when
    // Metal is unavailable on this build/host (e.g. cross-compile, no GPU).
    let backend_slot = model.metal_backend.get_or_init(MetalBackend::new);
    let Some(backend) = backend_slot.as_ref() else {
        return Ok(None);
    };

    let hidden = model.config.hidden_size;
    if h_post_attn.len() != hidden {
        return Err(ServerError::BadRequest(format!(
            "residual length {} != hidden_size {hidden}",
            h_post_attn.len()
        )));
    }
    let inter = arch.moe_intermediate_size();
    let top_k = arch.num_experts_per_token();

    let t_pre = Instant::now();
    // Apply pre_experts_norm on CPU (cheap; matches the per-expert CPU path's
    // behaviour in `run_single_expert_with_norm`).
    //   out[i] = h[i] / sqrt(mean(h²) + eps) * (norm[i] + norm_offset)
    let h_norm: Vec<f32> = if let Some(norm_key) = arch.moe_pre_experts_norm_key(layer) {
        if let Some(pre_norm) = weights.vectors.get(&norm_key) {
            let eps = arch.norm_eps();
            let norm_offset = arch.norm_weight_offset();
            let pre_norm = pre_norm.as_slice();
            let rms = (h_post_attn.iter().map(|v| v * v).sum::<f32>() / hidden as f32 + eps).sqrt();
            h_post_attn
                .iter()
                .zip(pre_norm.iter())
                .map(|(x, w)| x / rms * (w + norm_offset))
                .collect()
        } else {
            h_post_attn.to_vec()
        }
    } else {
        h_post_attn.to_vec()
    };
    let t_norm = t_pre.elapsed();

    // get_expert_bytes maps expert_id → (gate_up_bytes, down_bytes) mmap slices.
    let get_expert_bytes =
        |eid: usize| -> Option<(&[u8], &[u8])> { weights.get_layer_entry_bytes(layer, eid) };

    // Pre-stage per-expert weights as cache-backed Metal buffers.  First
    // call for each (layer, expert_id) pays a memcpy (when bytes aren't
    // page-aligned for zero-copy aliasing); subsequent calls hit the
    // BufferCache and return instantly.  By the time the model is warm
    // (a handful of decode tokens), every owned expert has been staged.
    let t_buf_start = Instant::now();
    let mut expert_bufs: Vec<(larql_compute::MetalBuffer, larql_compute::MetalBuffer)> =
        Vec::with_capacity(expert_ids.len());
    let mut filtered_weights: Vec<f32> = Vec::with_capacity(expert_ids.len());
    for (i, &eid) in expert_ids.iter().enumerate() {
        if let Some((gu, dn)) = weights.get_layer_entry_bytes(layer, eid) {
            expert_bufs.push((
                backend.cached_buffer_for_bytes(gu),
                backend.cached_buffer_for_bytes(dn),
            ));
            filtered_weights.push(expert_weights[i]);
        }
    }
    let t_bufs = t_buf_start.elapsed();

    // Look up (or create + cache) the MoE scratch for this layer's shape.
    //
    // MoeScratch owns mutable Metal staging/output buffers. Keep the cache lock
    // held across the dispatch so concurrent RPCs cannot overwrite each other's
    // scratch contents. This path is opt-in while the Metal expert accuracy bug
    // is being debugged; replace with a scratch pool if parallel GPU expert
    // dispatch becomes a production requirement.
    let t_scratch_start = Instant::now();
    let scratch_key = (top_k, hidden, inter);
    let mut scratch_cache = model.moe_scratches.lock().expect("moe_scratches poisoned");
    let scratch = scratch_cache
        .entry(scratch_key)
        .or_insert_with(|| Arc::new(MoeScratch::new_public(backend, top_k, hidden, inter)));
    let t_scratch = t_scratch_start.elapsed();

    let t_gpu_start = Instant::now();
    // 2026-04-30: switched from `run_experts_prestaged_metal` (per-expert
    // pre-cached buffers, per-expert dispatch) back to
    // `run_experts_preselected_metal` (byte-copy into shared scratch,
    // ONE big dispatch for all K experts). The prestaged variant produces
    // numerically wrong expert outputs (cos≈0.7 vs CPU reference, |metal|
    // consistently ~70% of |cpu|) — the per-expert dispatch loop in
    // `q4k_ffn_gate_up` apparently doesn't see the per-expert bound buffers
    // / output offsets the way the all-experts-at-once dispatch does. The
    // preselected variant matches the in-process `gpu_moe_dispatch_with_scratch`
    // dispatch pattern that's been proven correct end-to-end. Speedup over
    // CPU is preserved; we lose only the per-call memcpy elimination.
    let _ = (&expert_bufs, &filtered_weights);
    let result = backend.run_experts_preselected_metal(
        &h_norm,
        expert_ids,
        expert_weights,
        scratch.as_ref(),
        get_expert_bytes,
    );
    let t_gpu = t_gpu_start.elapsed();

    // LARQL_METAL_VS_CPU_DEBUG=1 — recompute via CPU and print element-wise
    // max diff. Used to localise the metal-experts accuracy bug. Slow
    // (every layer × every token does both paths), so opt-in only.
    if std::env::var("LARQL_METAL_VS_CPU_DEBUG").is_ok() {
        // Run the same K experts via the CPU pooled path against the same
        // residual + weights so we get a direct apples-to-apples diff.
        match run_experts_cpu_batch(state, layer, h_post_attn, expert_ids, expert_weights) {
            Ok(cpu_out) => {
                let max_abs_diff = result
                    .iter()
                    .zip(cpu_out.iter())
                    .fold(0.0f32, |acc, (m, c)| acc.max((m - c).abs()));
                let metal_norm = (result.iter().map(|v| v * v).sum::<f32>() / hidden as f32).sqrt();
                let cpu_norm = (cpu_out.iter().map(|v| v * v).sum::<f32>() / hidden as f32).sqrt();
                let cos = {
                    let dot: f32 = result.iter().zip(cpu_out.iter()).map(|(a, b)| a * b).sum();
                    let na: f32 = result.iter().map(|v| v * v).sum::<f32>().sqrt();
                    let nb: f32 = cpu_out.iter().map(|v| v * v).sum::<f32>().sqrt();
                    if na > 0.0 && nb > 0.0 {
                        dot / (na * nb)
                    } else {
                        f32::NAN
                    }
                };
                eprintln!(
                    "[metal-vs-cpu] L{layer:02} K={} max|Δ|={max_abs_diff:.4e} \
                     |metal|={metal_norm:.4} |cpu|={cpu_norm:.4} cos={cos:.6}",
                    expert_ids.len()
                );
            }
            Err(e) => {
                eprintln!("[metal-vs-cpu] L{layer:02} cpu reference failed: {e}");
            }
        }
    }

    if timing_enabled {
        eprintln!(
            "[expert_metal_batch] layer={layer} experts={} state={:.2}ms norm={:.2}ms \
             scratch={:.2}ms bufs={:.2}ms gpu={:.2}ms total={:.2}ms",
            expert_ids.len(),
            t_state.as_secs_f32() * 1000.0,
            t_norm.as_secs_f32() * 1000.0,
            t_scratch.as_secs_f32() * 1000.0,
            t_bufs.as_secs_f32() * 1000.0,
            t_gpu.as_secs_f32() * 1000.0,
            t_start.elapsed().as_secs_f32() * 1000.0,
        );
    }

    Ok(Some(result))
}

pub fn run_expert(
    state: &AppState,
    layer: usize,
    expert_id: usize,
    residual: &[f32],
) -> Result<Vec<f32>, ServerError> {
    let model = state.model_or_err(None)?;

    // Ownership check.  When `unit_filter` is set (`--units` JSON manifest),
    // the per-(layer, expert) ownership set takes precedence over the
    // layer-uniform `expert_filter` range.  The two flags are mutually
    // exclusive at the CLI parse layer, but check both in priority order
    // so misconfiguration fails loudly at request time rather than silently
    // accepting a request the shard doesn't own.
    if let Some(units) = model.unit_filter.as_ref() {
        if !units.contains(&(layer, expert_id)) {
            return Err(ServerError::BadRequest(format!(
                "(layer={layer}, expert={expert_id}) not owned by this shard \
                 (--units manifest defines its ownership set)"
            )));
        }
    } else if let Some((start, end_excl)) = model.expert_filter {
        if expert_id < start || expert_id >= end_excl {
            let end_inclusive = end_excl.saturating_sub(1);
            return Err(ServerError::BadRequest(format!(
                "expert {expert_id} not owned by this shard (owns {start}–{end_inclusive})"
            )));
        }
    }

    let weights = model
        .get_or_load_weights()
        .map_err(ServerError::InferenceUnavailable)?;

    let arch = &*weights.arch;

    if !arch.is_hybrid_moe() {
        return Err(ServerError::BadRequest(
            "model is not a hybrid MoE — no expert endpoints available".into(),
        ));
    }

    let hidden = model.config.hidden_size;
    if residual.len() != hidden {
        return Err(ServerError::BadRequest(format!(
            "residual length {} != hidden_size {hidden}",
            residual.len()
        )));
    }

    let inter = arch.moe_intermediate_size();
    let hidden = model.config.hidden_size;
    let activation = larql_inference::activation_from_arch(arch);

    // Resolve this expert's per-expert byte slice. Per-layer Q4_K vindexes
    // expose entries at `layers/{layer}/{expert}/...`; legacy BF16 vindexes
    // expose a monolithic `packed_experts_{gate_up,down}_key` blob that we
    // slice by stride. Either way we feed `run_single_expert*` exactly one
    // expert's bytes — no monolith arithmetic in the compute path.
    let (gate_up_bytes, down_bytes, format) = if weights.has_per_layer_ffn() {
        let (gu, dn) = weights
            .get_layer_entry_bytes(layer, expert_id)
            .ok_or_else(|| {
                ServerError::Internal(format!(
                    "per-layer entry missing for layer {layer} expert {expert_id}"
                ))
            })?;
        (gu, dn, larql_inference::QuantFormat::Q4_K)
    } else {
        let gate_up_key = arch.packed_experts_gate_up_key(layer).ok_or_else(|| {
            ServerError::BadRequest(format!("no MoE gate/up weights for layer {layer}"))
        })?;
        let down_key = arch.packed_experts_down_key(layer).ok_or_else(|| {
            ServerError::BadRequest(format!("no MoE down weights for layer {layer}"))
        })?;
        let gu_all = weights.get_packed_bytes(&gate_up_key).ok_or_else(|| {
            ServerError::Internal(format!("gate_up bytes missing for layer {layer}"))
        })?;
        let dn_all = weights.get_packed_bytes(&down_key).ok_or_else(|| {
            ServerError::Internal(format!("down bytes missing for layer {layer}"))
        })?;
        let gu_stride = 2 * inter * hidden * 2; // BF16 = 2 bytes
        let dn_stride = hidden * inter * 2;
        let gu_start = expert_id * gu_stride;
        let dn_start = expert_id * dn_stride;
        if gu_start + gu_stride > gu_all.len() || dn_start + dn_stride > dn_all.len() {
            return Err(ServerError::Internal(format!(
                "expert {expert_id} byte range out of bounds for layer {layer}"
            )));
        }
        (
            &gu_all[gu_start..gu_start + gu_stride],
            &dn_all[dn_start..dn_start + dn_stride],
            larql_inference::QuantFormat::BF16,
        )
    };

    let output = if let Some(norm_key) = arch.moe_pre_experts_norm_key(layer) {
        let pre_experts_norm = weights
            .vectors
            .get(&norm_key)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        larql_inference::run_single_expert_with_norm(
            residual,
            gate_up_bytes,
            down_bytes,
            inter,
            pre_experts_norm,
            arch.norm_weight_offset(),
            arch.norm_eps(),
            format,
            activation,
        )
    } else {
        larql_inference::run_single_expert(
            residual,
            gate_up_bytes,
            down_bytes,
            inter,
            format,
            activation,
        )
    };

    Ok(output)
}

// ── HTTP handlers ─────────────────────────────────────────────────────────────

pub async fn handle_expert(
    State(state): State<Arc<AppState>>,
    Path((layer, expert_id)): Path<(usize, usize)>,
    Json(req): Json<SingleExpertRequest>,
) -> Result<Json<SingleExpertResponse>, ServerError> {
    state.bump_requests();
    let start = std::time::Instant::now();

    let output =
        tokio::task::spawn_blocking(move || run_expert(&state, layer, expert_id, &req.residual))
            .await
            .map_err(|e| ServerError::Internal(e.to_string()))??;

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok(Json(SingleExpertResponse { output, latency_ms }))
}

pub async fn handle_expert_batch(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    body: Bytes,
) -> Result<Response, ServerError> {
    state.bump_requests();
    let start = std::time::Instant::now();

    // Accept both binary (application/x-larql-expert) and JSON.
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let binary = content_type.contains(EXPERT_BINARY_CONTENT_TYPE);

    // Decode request items from either wire format.
    let items: Vec<ExpertCallItem> = if binary {
        decode_expert_request(&body)
            .ok_or_else(|| ServerError::BadRequest("binary expert request truncated".into()))?
    } else {
        let req: BatchExpertRequest = serde_json::from_slice(&body)
            .map_err(|e| ServerError::BadRequest(format!("JSON parse: {e}")))?;
        req.requests
            .into_iter()
            .map(|r| ExpertCallItem {
                layer: r.layer,
                expert_id: r.expert_id,
                residual: r.residual,
            })
            .collect()
    };

    let result_items = tokio::task::spawn_blocking(move || {
        use rayon::prelude::*;
        items
            .par_iter()
            .map(|item| {
                run_expert(&state, item.layer, item.expert_id, &item.residual).map(|output| {
                    ExpertResultItem {
                        layer: item.layer,
                        expert_id: item.expert_id,
                        output,
                    }
                })
            })
            .collect::<Result<Vec<ExpertResultItem>, ServerError>>()
    })
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))??;

    let latency_ms = (start.elapsed().as_secs_f64() * 1000.0) as f32;

    // Respond in the same wire format the client requested.
    let response = if binary {
        let body = encode_expert_response(&result_items, latency_ms);
        Response::builder()
            .header(header::CONTENT_TYPE, EXPERT_BINARY_CONTENT_TYPE)
            .body(axum::body::Body::from(body))
            .map_err(|e| ServerError::Internal(e.to_string()))?
    } else {
        let resp = BatchExpertResponse {
            results: result_items
                .into_iter()
                .map(|r| BatchExpertResult {
                    layer: r.layer,
                    expert_id: r.expert_id,
                    output: r.output,
                })
                .collect(),
            latency_ms: latency_ms as f64,
        };
        Response::builder()
            .header(header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from(
                serde_json::to_vec(&resp).map_err(|e| ServerError::Internal(e.to_string()))?,
            ))
            .map_err(|e| ServerError::Internal(e.to_string()))?
    };

    Ok(response)
}

/// `POST /v1/experts/layer-batch` — single residual + K (expert_id, weight)
/// pairs for one layer.  Server applies pre_experts_norm once, quantises
/// h_norm to Q8_K once, fans out the K expert kernels with the shared
/// activation via `run_experts_cpu_batch`, returns the router-weighted sum.
///
/// Wire format documented in `larql_inference::ffn::moe_remote` next to
/// `LAYER_BATCH_CONTENT_TYPE`.  Replaces the K-residual-copies pattern of
/// `/v1/expert/batch` for the common-case forward_moe call where every
/// expert in the layer's top-K shares the same residual.
pub async fn handle_experts_layer_batch(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> Result<Response, ServerError> {
    state.bump_requests();
    // Per-stage timing for HTTP-overhead diagnosis.  Enable with
    // `LARQL_HTTP_TIMING=1`.  Cached in TLS to avoid syscalls per call.
    thread_local! {
        static HTTP_TIMING: bool = std::env::var("LARQL_HTTP_TIMING").is_ok();
    }
    let timing = HTTP_TIMING.with(|t| *t);
    let t_start = std::time::Instant::now();

    let (layer, residual, expert_ids_u32, expert_weights) = decode_layer_batch_request(&body)
        .ok_or_else(|| ServerError::BadRequest("layer-batch request truncated".into()))?;
    let t_decode = if timing {
        Some(t_start.elapsed())
    } else {
        None
    };

    // Convert expert_ids u32 → usize for the existing run_experts_cpu_batch
    // signature.  Cheap; expert_ids is small (K=8 typical).
    let expert_ids: Vec<usize> = expert_ids_u32.iter().map(|&e| e as usize).collect();

    let t_spawn_in = std::time::Instant::now();
    // `spawn_blocking` (vs `block_in_place`): we want the compute on the
    // dedicated blocking thread pool so tokio's worker threads stay free
    // for the hot HTTP path.  Tried block_in_place (2026-05-01): saved
    // the ~25 µs transition server-side but made sweep ~0.3 ms slower
    // because tokio kept spawning replacement OS workers when every
    // request blocked the worker.  spawn_blocking's pool reuses threads
    // and works better for the hot-path-blocks-every-call pattern.
    let (weighted_sum, t_spawn_internal) = tokio::task::spawn_blocking(move || {
        let t_in = std::time::Instant::now();
        let r = run_experts_cpu_batch(&state, layer, &residual, &expert_ids, &expert_weights);
        let t_internal = t_in.elapsed();
        (r, t_internal)
    })
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))?;
    let weighted_sum = weighted_sum?;
    let t_total_compute = t_spawn_in.elapsed();
    let t_spawn_overhead = t_total_compute.saturating_sub(t_spawn_internal);

    let t_encode_in = std::time::Instant::now();
    let latency_ms = (t_start.elapsed().as_secs_f64() * 1000.0) as f32;
    let body = encode_layer_batch_response(&weighted_sum, latency_ms);
    let t_encode = t_encode_in.elapsed();

    let resp = Response::builder()
        .header(header::CONTENT_TYPE, LAYER_BATCH_CONTENT_TYPE)
        .body(axum::body::Body::from(body))
        .map_err(|e| ServerError::Internal(e.to_string()))?;

    if timing {
        eprintln!(
            "[handle_layer_batch] layer={layer} K={} decode={:.0}us \
             spawn_overhead={:.0}us compute={:.0}us encode={:.0}us total={:.0}us",
            expert_ids_u32.len(),
            t_decode.unwrap().as_secs_f64() * 1e6,
            t_spawn_overhead.as_secs_f64() * 1e6,
            t_spawn_internal.as_secs_f64() * 1e6,
            t_encode.as_secs_f64() * 1e6,
            t_start.elapsed().as_secs_f64() * 1e6,
        );
    }

    Ok(resp)
}

/// `POST /v1/experts/layer-batch-f16` — same semantics as the f32 layer-batch
/// endpoint but residual + response use IEEE-754 binary16.  Halves the wire
/// bytes (~5.5 KB residual + 5.5 KB response vs 11+11 KB f32).  f16 quant
/// noise on activations is well below the Q8_K activation quant the SDOT
/// kernel already applies, so end-to-end accuracy is unchanged.
pub async fn handle_experts_layer_batch_f16(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> Result<Response, ServerError> {
    state.bump_requests();
    thread_local! {
        static HTTP_TIMING: bool = std::env::var("LARQL_HTTP_TIMING").is_ok();
    }
    let timing = HTTP_TIMING.with(|t| *t);
    let t_start = std::time::Instant::now();

    let (layer, residual, expert_ids_u32, expert_weights) =
        decode_layer_batch_request_f16(&body)
            .ok_or_else(|| ServerError::BadRequest("layer-batch-f16 request truncated".into()))?;
    let t_decode = if timing {
        Some(t_start.elapsed())
    } else {
        None
    };

    let expert_ids: Vec<usize> = expert_ids_u32.iter().map(|&e| e as usize).collect();

    let t_spawn_in = std::time::Instant::now();
    let (weighted_sum, t_spawn_internal) = tokio::task::spawn_blocking(move || {
        let t_in = std::time::Instant::now();
        let r = run_experts_cpu_batch(&state, layer, &residual, &expert_ids, &expert_weights);
        let t_internal = t_in.elapsed();
        (r, t_internal)
    })
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))?;
    let weighted_sum = weighted_sum?;
    let t_total_compute = t_spawn_in.elapsed();
    let t_spawn_overhead = t_total_compute.saturating_sub(t_spawn_internal);

    let t_encode_in = std::time::Instant::now();
    let latency_ms = (t_start.elapsed().as_secs_f64() * 1000.0) as f32;
    let body = encode_layer_batch_response_f16(&weighted_sum, latency_ms);
    let t_encode = t_encode_in.elapsed();

    let resp = Response::builder()
        .header(header::CONTENT_TYPE, LAYER_BATCH_F16_CONTENT_TYPE)
        .body(axum::body::Body::from(body))
        .map_err(|e| ServerError::Internal(e.to_string()))?;

    if timing {
        eprintln!(
            "[handle_layer_batch_f16] layer={layer} K={} decode={:.0}us \
             spawn_overhead={:.0}us compute={:.0}us encode={:.0}us total={:.0}us",
            expert_ids_u32.len(),
            t_decode.unwrap().as_secs_f64() * 1e6,
            t_spawn_overhead.as_secs_f64() * 1e6,
            t_spawn_internal.as_secs_f64() * 1e6,
            t_encode.as_secs_f64() * 1e6,
            t_start.elapsed().as_secs_f64() * 1e6,
        );
    }

    Ok(resp)
}
