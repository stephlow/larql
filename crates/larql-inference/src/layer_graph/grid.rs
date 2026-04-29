//! Autoregressive generation via a sharded expert grid.
//!
//! Uses the Metal pipeline for attention + dense FFN (same as normal `generate`),
//! but intercepts the MoE expert block per layer via a callback that dispatches
//! to remote shards over HTTP instead of calling `cpu_moe_forward` locally.
//!
//! The hook: `ComputeBackend::decode_token_with_moe(layers, x, ..., moe_fn)`
//! where `moe_fn(layer, h_post_attn) -> Vec<f32>` calls
//! `RemoteMoeBackend::forward_moe`.
//!
//! # Diagnostics
//!
//! Set `SKIP_MOE=1` to zero out the expert block on every decode step.
//! This isolates whether errors come from remote dispatch vs. dense FFN.

use larql_compute::prelude::*;
use larql_models::ModelWeights;
use larql_vindex::VectorIndex;

use crate::ffn::moe_remote::{InflightMoe, MoeRouterWeights, RemoteMoeError, ShardStream};
use crate::ffn::RemoteMoeBackend;
use crate::forward::{apply_norm, embed_tokens_pub};
use crate::layer_graph::generate::lm_head_topk as lm_topk;
use crate::layer_graph::pipeline_layer::build_pipeline_layers;

// ── Bottleneck diagnostic ────────────────────────────────────────────────────
//
// Activated by `LARQL_MOE_TIMING=1`.  The streaming path swaps
// `forward_moe_stream` for an explicit fire/collect_with_timing pair so we can
// see, for every MoE layer of every decoded token:
//
//   - `total_ms`:        wall-clock time inside the moe_fn closure
//   - `route_fire_ms`:   CPU routing + non-blocking fire
//   - `collect_ms`:      condvar-blocking wait for all shards' h2 frames
//   - per-shard `(wall_ms, server_compute_ms)` so `network_ms` is derivable
//     as `wall_ms − server_compute_ms`
//
// Everything is per-MoE-layer; the GPU side (attention + dense FFN) is timed
// independently by `LARQL_GPU_TIMING=1` in the metal backend.

#[derive(Clone, Debug)]
struct LayerTiming {
    layer: usize,
    total_ms: f32,
    route_fire_ms: f32,
    collect_ms: f32,
    /// One entry per shard: `(wall_collect_ms, server_compute_ms)`.
    per_shard: Vec<(f32, f32)>,
}

/// Sum of per-shard wall times — gives the inner-loop's collect wait.  Note
/// shards collect sequentially today (loop in `forward_moe_stream_collect`),
/// so this matches `collect_ms` to within microseconds.
fn shard_wall_sum(t: &LayerTiming) -> f32 {
    t.per_shard.iter().map(|(w, _)| *w).sum()
}

fn shard_compute_max(t: &LayerTiming) -> f32 {
    t.per_shard.iter().map(|(_, c)| *c).fold(0.0, f32::max)
}

fn print_token_breakdown(label: &str, tok_idx: usize, timings: &[LayerTiming]) {
    if timings.is_empty() {
        return;
    }
    let n = timings.len();
    let total: f32 = timings.iter().map(|t| t.total_ms).sum();
    let route: f32 = timings.iter().map(|t| t.route_fire_ms).sum();
    let collect: f32 = timings.iter().map(|t| t.collect_ms).sum();
    let server_max: f32 = timings.iter().map(shard_compute_max).sum();
    let network = (collect - server_max).max(0.0);
    eprintln!(
        "[moe-timing] {label} tok={tok_idx} layers={n} \
         moe_total={total:.1}ms (route+fire={route:.1}ms collect={collect:.1}ms \
         | server_compute≈{server_max:.1}ms network≈{network:.1}ms)"
    );
}

fn print_run_summary(label: &str, per_token: &[Vec<LayerTiming>]) {
    if per_token.is_empty() {
        return;
    }
    let n_tokens = per_token.len();
    let layers_per_tok = per_token.iter().map(|v| v.len()).max().unwrap_or(0);

    // Per-token aggregates.
    let mut tot_total = 0.0f32;
    let mut tot_route = 0.0f32;
    let mut tot_collect = 0.0f32;
    let mut tot_server = 0.0f32;
    for tok in per_token {
        tot_total += tok.iter().map(|t| t.total_ms).sum::<f32>();
        tot_route += tok.iter().map(|t| t.route_fire_ms).sum::<f32>();
        tot_collect += tok.iter().map(|t| t.collect_ms).sum::<f32>();
        tot_server += tok.iter().map(shard_compute_max).sum::<f32>();
    }
    let avg_total = tot_total / n_tokens as f32;
    let avg_route = tot_route / n_tokens as f32;
    let avg_collect = tot_collect / n_tokens as f32;
    let avg_server = tot_server / n_tokens as f32;
    let avg_net = (avg_collect - avg_server).max(0.0);

    eprintln!("[moe-timing] {label} SUMMARY ({n_tokens} tokens, {layers_per_tok} MoE layers/token)");
    eprintln!(
        "[moe-timing]   per-token avg: moe_total={avg_total:.1}ms \
         (route+fire={avg_route:.1}ms collect={avg_collect:.1}ms \
         | server_compute≈{avg_server:.1}ms network≈{avg_net:.1}ms)"
    );
    if layers_per_tok > 0 {
        let avg_per_layer_total = avg_total / layers_per_tok as f32;
        let avg_per_layer_collect = avg_collect / layers_per_tok as f32;
        let avg_per_layer_server = avg_server / layers_per_tok as f32;
        let avg_per_layer_net = (avg_per_layer_collect - avg_per_layer_server).max(0.0);
        eprintln!(
            "[moe-timing]   per-layer avg: total={avg_per_layer_total:.2}ms \
             collect={avg_per_layer_collect:.2}ms \
             (server≈{avg_per_layer_server:.2}ms net≈{avg_per_layer_net:.2}ms)"
        );
    }
    // Bottleneck attribution: collect dominates when remote round-trip dwarfs
    // local routing.  The "X% of MoE time" framing is what the operator wants
    // — it's the actionable lever (move shards closer / use batch mode / …).
    if avg_total > 0.0 {
        let collect_pct = 100.0 * avg_collect / avg_total;
        let server_pct = 100.0 * avg_server / avg_total;
        let net_pct = 100.0 * avg_net / avg_total;
        let route_pct = 100.0 * avg_route / avg_total;
        eprintln!(
            "[moe-timing]   bottleneck: collect={collect_pct:.0}% \
             (of which server≈{server_pct:.0}%, network≈{net_pct:.0}%) \
             route+fire={route_pct:.0}%"
        );
    }
}

/// Inner moe call with optional timing capture.  Returns the h2 vec.  When
/// `timing.is_some()`, splits the call into fire + collect_with_timing so we
/// can record per-shard wall + server-compute breakdown.
fn moe_call_timed(
    remote: &RemoteMoeBackend,
    layer: usize,
    h_post_attn: &[f32],
    router: &MoeRouterWeights<'_>,
    streams: &mut [ShardStream],
    norm_offset: f32,
    eps: f32,
    timing: Option<&mut Vec<LayerTiming>>,
) -> Result<Vec<f32>, RemoteMoeError> {
    if streams.is_empty() {
        return remote.forward_moe(layer, h_post_attn, router, norm_offset, eps);
    }
    let Some(timing) = timing else {
        return remote.forward_moe_stream(layer, h_post_attn, router, streams, norm_offset, eps);
    };
    let t_total = std::time::Instant::now();
    let t_fire = std::time::Instant::now();
    let inflight =
        remote.forward_moe_stream_fire(layer, h_post_attn, router, streams, norm_offset, eps)?;
    let route_fire_ms = t_fire.elapsed().as_secs_f32() * 1000.0;
    let t_collect = std::time::Instant::now();
    let (h2, per_shard) =
        remote.forward_moe_stream_collect_with_timing(streams, inflight)?;
    let collect_ms = t_collect.elapsed().as_secs_f32() * 1000.0;
    let total_ms = t_total.elapsed().as_secs_f32() * 1000.0;
    timing.push(LayerTiming {
        layer,
        total_ms,
        route_fire_ms,
        collect_ms,
        per_shard,
    });
    Ok(h2)
}

/// Build `MoeRouterWeights` for one layer from the model's vector store.
/// Returns None if the required router projection is absent.
///
/// `LARQL_MOE_TOP_K=<N>` overrides the architecture-default top_k at runtime
/// (clamped to `[1, arch_top_k]`).  Cheap accuracy/speed knob — Gemma 4 ships
/// with top_k=8; testing top_k=4 cuts active experts in half for a roughly
/// 2× server-compute speedup at the cost of some routing fidelity.
fn build_router<'a>(
    weights: &'a ModelWeights,
    arch: &dyn larql_models::ModelArchitecture,
    layer: usize,
) -> Option<MoeRouterWeights<'a>> {
    let router_proj_key = arch.moe_router_key(layer)?;
    let router_proj = weights.vectors.get(&router_proj_key)?.as_slice();
    let sl = |k: Option<String>| -> &'a [f32] {
        k.and_then(|k| weights.vectors.get(&k))
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    };
    let arch_top_k = arch.num_experts_per_token();
    let top_k = std::env::var("LARQL_MOE_TOP_K")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .map(|k| k.clamp(1, arch_top_k))
        .unwrap_or(arch_top_k);
    Some(MoeRouterWeights {
        router_proj,
        router_scale:            sl(arch.moe_router_scale_key(layer)),
        router_per_expert_scale: sl(arch.moe_router_per_expert_scale_key(layer)),
        router_norm:             sl(arch.moe_router_norm_key(layer)),
        router_norm_parameter_free: arch.moe_router_norm_parameter_free(),
        router_input_scalar: arch.moe_router_input_scalar().unwrap_or(1.0),
        pre_experts_norm:  sl(arch.moe_pre_experts_norm_key(layer)),
        post_experts_norm: sl(arch.moe_post_experts_norm_key(layer)),
        num_experts: arch.num_experts(),
        top_k,
    })
}

#[derive(Debug)]
pub struct GridGenerateResult {
    pub tokens: Vec<String>,
    pub decode_ms: Vec<f64>,
}

/// Greedy autoregressive generation through a remote-expert grid.
///
/// Requires a Metal (or Q4-capable) backend — attention and dense FFN run on
/// the GPU exactly as in the normal `generate()` path.  Expert blocks are
/// dispatched to `remote` instead of running locally.
pub fn generate_with_remote_moe(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    prompt_ids: Vec<u32>,
    max_tokens: usize,
    index: &VectorIndex,
    remote: &RemoteMoeBackend,
    backend: &dyn ComputeBackend,
) -> Result<GridGenerateResult, RemoteMoeError> {
    let arch = &*weights.arch;
    let norm_offset = arch.norm_weight_offset();
    let eps = arch.norm_eps();
    let hidden = weights.hidden_size;
    let num_layers = weights.num_layers;

    // ── Build pipeline layers (same as generate()) ────────────────────────────
    let gate_index: &dyn larql_vindex::GateIndex = index;
    let q4_ffn = gate_index
        .interleaved_q4k_mmap_ref()
        .or_else(|| gate_index.interleaved_q4_mmap_ref())
        .ok_or_else(|| {
            RemoteMoeError::BadResponse("no interleaved Q4 FFN mmap in vindex".into())
        })?;
    let ffn_is_q4k = gate_index.interleaved_q4k_mmap_ref().is_some();

    let intermediate = gate_index.num_features(0);
    let q4_ffn_per_matrix = if ffn_is_q4k {
        (intermediate * hidden).div_ceil(256) * 144
    } else {
        intermediate * hidden / 32 * 18
    };
    let ffn_format = if ffn_is_q4k {
        larql_compute::QuantFormat::Q4_K
    } else {
        larql_compute::QuantFormat::Q4_0
    };

    let layers = build_pipeline_layers(
        weights,
        index,
        0..num_layers,
        q4_ffn,
        q4_ffn_per_matrix,
        ffn_format,
    );

    let q_dim = weights.num_q_heads * weights.head_dim;
    let kv_dim = weights.num_kv_heads * weights.head_dim;
    let rope = arch.rope_base_for_layer(0) as f32;

    // ── Open gRPC streams (one pair for the entire generation) ───────────────
    //
    // For gRPC shards (`grpc://` URLs), we open one bidirectional stream per
    // shard and reuse it for every layer of every token (prefill + decode).
    // This eliminates the per-layer connection setup cost: each layer pays only
    // the cost of one proto frame exchange on an existing HTTP/2 connection
    // (~0.5ms) instead of ~12ms for a new unary call.
    //
    // For HTTP shards, `open_streams` returns an empty vec and we fall back to
    // `forward_moe` (per-layer HTTP calls, as before).
    let mut streams: Vec<crate::ffn::moe_remote::ShardStream> =
        if remote.has_grpc_shards() {
            remote.open_streams().unwrap_or_default()
        } else {
            vec![]
        };

    // ── Prefill ───────────────────────────────────────────────────────────────
    //
    // Run one `decode_token_with_moe` per prompt token rather than `prefill_q4`.
    // `prefill_q4` does not correctly apply MoE experts for hybrid-MoE post-norm
    // models (Gemma 4 26B-A4B), so the first-token prediction and subsequent KV
    // cache entries are wrong.  Sequential decode builds the KV cache correctly
    // — each token processes with the proper remote expert contribution.
    backend.reset_kv_cache();
    {
        let kv_shapes: Vec<(usize, usize)> = (0..num_layers)
            .map(|l| (arch.num_kv_heads_for_layer(l), arch.head_dim_for_layer(l)))
            .collect();
        backend.preallocate_kv_cache_per_layer(&kv_shapes, 4096);
    }

    let skip_moe = std::env::var("SKIP_MOE").is_ok();
    let timing_enabled = std::env::var("LARQL_MOE_TIMING").is_ok();
    let mut per_token_timings: Vec<Vec<LayerTiming>> = Vec::new();
    let mut last_hidden_vec: Vec<f32> = vec![0.0f32; hidden];
    let mut current_ids = prompt_ids.clone();

    for (prefill_idx, &tok_id) in prompt_ids.iter().enumerate() {
        let tok_embed = embed_tokens_pub(weights, &[tok_id]);
        let x_tok: Vec<f32> = tok_embed.as_slice().unwrap_or(&[]).to_vec();

        let mut step_error: Option<RemoteMoeError> = None;
        let mut tok_timings: Vec<LayerTiming> = Vec::new();
        let mut moe_fn = |layer: usize, h_post_attn: &[f32]| -> Vec<f32> {
            if skip_moe { return vec![0.0f32; hidden]; }
            if step_error.is_some() { return vec![0.0f32; hidden]; }
            let router = match build_router(weights, arch, layer) {
                Some(r) => r,
                None => return vec![0.0f32; hidden],
            };
            let timing_slot = if timing_enabled { Some(&mut tok_timings) } else { None };
            match moe_call_timed(
                remote, layer, h_post_attn, &router, &mut streams, norm_offset, eps, timing_slot,
            ) {
                Ok(out) => out,
                Err(e) => { step_error = Some(e); vec![0.0f32; hidden] }
            }
        };

        let h = backend.decode_token_with_moe(
            &layers, &x_tok, hidden, intermediate, q_dim, kv_dim,
            weights.num_q_heads, weights.num_kv_heads, weights.head_dim, rope, &mut moe_fn,
        );
        if let Some(err) = step_error { return Err(err); }
        last_hidden_vec = h.ok_or_else(|| {
            RemoteMoeError::BadResponse("decode_token_with_moe returned None during prefill".into())
        })?;
        if timing_enabled {
            print_token_breakdown("prefill", prefill_idx, &tok_timings);
            per_token_timings.push(tok_timings);
        }
    }

    // ── Decode loop ───────────────────────────────────────────────────────────
    let mut tokens = Vec::new();
    let mut decode_ms = Vec::new();

    // First token from the (correct) prefill output.
    let prefill_h_arr = ndarray::Array2::from_shape_vec((1, hidden), last_hidden_vec.clone())
        .map_err(|e| RemoteMoeError::BadResponse(e.to_string()))?;
    let h_norm0 = apply_norm(weights, &prefill_h_arr, arch.final_norm_key(), norm_offset);
    let last0 = h_norm0.row(0).to_owned();
    let first_id = lm_topk(index, weights, &last0, 1, backend)
        .into_iter()
        .next()
        .map(|(id, _)| id)
        .unwrap_or(0);

    let first_tok = crate::tokenizer::decode_token(tokenizer, first_id)
        .unwrap_or_else(|| format!("<{first_id}>"));
    tokens.push(first_tok);
    current_ids.push(first_id);
    let first_is_eos = crate::vindex::is_end_of_turn(
        crate::tokenizer::decode_token(tokenizer, first_id)
            .unwrap_or_default()
            .trim(),
    );
    if first_is_eos || tokens.len() >= max_tokens {
        return Ok(GridGenerateResult {
            tokens,
            decode_ms: vec![0.0],
        });
    }

    for step in 0..max_tokens.saturating_sub(1) {
        let t0 = std::time::Instant::now();
        let next_input_id = *current_ids.last().unwrap();

        // Embed next token.
        let tok_embed = embed_tokens_pub(weights, &[next_input_id]);
        let x_tok: Vec<f32> = tok_embed.as_slice().unwrap_or(&[]).to_vec();

        let mut step_error: Option<RemoteMoeError> = None;
        let mut tok_timings: Vec<LayerTiming> = Vec::new();

        // Two paths:
        //   - LARQL_MOE_SPLIT=1 + streams (gRPC) → split fire/collect so dense
        //     FFN overlaps with the remote round trip.  Only beneficial when
        //     the shard servers are on **separate physical GPUs** from the
        //     client; on a single-GPU dev box client + server contend for the
        //     device and overlap regresses (measured: 20 → 4 tok/s on M3 Max
        //     with one local shard).  Off by default.
        //   - otherwise → existing unary HTTP / synchronous moe_fn (used for
        //     both HTTP shards and the loopback gRPC dev case).
        let split_enabled = std::env::var("LARQL_MOE_SPLIT").is_ok();
        let result = if streams.is_empty() || !split_enabled {
            let mut moe_fn = |layer: usize, h_post_attn: &[f32]| -> Vec<f32> {
                if skip_moe { return vec![0.0f32; hidden]; }
                if step_error.is_some() { return vec![0.0f32; hidden]; }
                let router = match build_router(weights, arch, layer) {
                    Some(r) => r,
                    None => return vec![0.0f32; hidden],
                };
                let timing_slot = if timing_enabled { Some(&mut tok_timings) } else { None };
                match moe_call_timed(
                    remote, layer, h_post_attn, &router, &mut streams, norm_offset, eps, timing_slot,
                ) {
                    Ok(out) => out,
                    Err(e) => { step_error = Some(e); vec![0.0f32; hidden] }
                }
            };
            backend.decode_token_with_moe(
                &layers,
                &x_tok,
                hidden,
                intermediate,
                q_dim,
                kv_dim,
                weights.num_q_heads,
                weights.num_kv_heads,
                weights.head_dim,
                rope,
                &mut moe_fn,
            )
        } else {
            // Split path: shared inflight handle + step_error via RefCell
            // because both closures capture them and can't both have unique
            // mut borrows.  Closures are still called strictly sequentially
            // by the metal decode loop so RefCell never panics in practice.
            use std::cell::RefCell;
            let inflight: RefCell<Option<(InflightMoe, std::time::Instant)>> = RefCell::new(None);
            let step_err_cell: RefCell<Option<RemoteMoeError>> = RefCell::new(None);
            let tok_timings_cell: RefCell<Vec<LayerTiming>> = RefCell::new(Vec::new());

            let mut fire_fn = |layer: usize, h_post_attn: &[f32]| {
                if skip_moe { return; }
                if step_err_cell.borrow().is_some() { return; }
                let router = match build_router(weights, arch, layer) {
                    Some(r) => r,
                    None => return,
                };
                let t_start = std::time::Instant::now();
                match remote.forward_moe_stream_fire(
                    layer, h_post_attn, &router, &streams, norm_offset, eps,
                ) {
                    Ok(inf) => { *inflight.borrow_mut() = Some((inf, t_start)); }
                    Err(e) => { *step_err_cell.borrow_mut() = Some(e); }
                }
            };
            let mut collect_fn = |layer: usize| -> Vec<f32> {
                if skip_moe { return vec![0.0f32; hidden]; }
                if step_err_cell.borrow().is_some() { return vec![0.0f32; hidden]; }
                let Some((inf, t_start)) = inflight.borrow_mut().take() else {
                    return vec![0.0f32; hidden];
                };
                match remote.forward_moe_stream_collect_with_timing(&streams, inf) {
                    Ok((h2, per_shard)) => {
                        if timing_enabled {
                            let total_ms = t_start.elapsed().as_secs_f32() * 1000.0;
                            tok_timings_cell.borrow_mut().push(LayerTiming {
                                layer,
                                total_ms,
                                route_fire_ms: 0.0,
                                collect_ms: total_ms,
                                per_shard,
                            });
                        }
                        h2
                    }
                    Err(e) => {
                        *step_err_cell.borrow_mut() = Some(e);
                        vec![0.0f32; hidden]
                    }
                }
            };
            let r = backend.decode_token_with_moe_split(
                &layers,
                &x_tok,
                hidden,
                intermediate,
                q_dim,
                kv_dim,
                weights.num_q_heads,
                weights.num_kv_heads,
                weights.head_dim,
                rope,
                &mut fire_fn,
                &mut collect_fn,
            );
            // Propagate any error captured by the closures.
            step_error = step_err_cell.into_inner();
            tok_timings = tok_timings_cell.into_inner();
            r
        };

        if let Some(err) = step_error {
            return Err(err);
        }

        let h_vec = result.ok_or_else(|| {
            RemoteMoeError::BadResponse("decode_token_with_moe returned None".into())
        })?;

        last_hidden_vec = h_vec;

        let h_arr = ndarray::Array2::from_shape_vec((1, hidden), last_hidden_vec.clone())
            .map_err(|e| RemoteMoeError::BadResponse(e.to_string()))?;
        let h_normed = apply_norm(weights, &h_arr, arch.final_norm_key(), norm_offset);
        let last_hidden = h_normed.row(0).to_owned();
        let next_id = lm_topk(index, weights, &last_hidden, 1, backend)
            .into_iter()
            .next()
            .map(|(id, _)| id)
            .unwrap_or(0);

        let token_wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
        decode_ms.push(token_wall_ms);
        if timing_enabled {
            print_token_breakdown("decode", step, &tok_timings);
            let moe_total: f32 = tok_timings.iter().map(|t| t.total_ms).sum();
            let other = (token_wall_ms as f32 - moe_total).max(0.0);
            eprintln!(
                "[moe-timing] decode tok={step} wall={token_wall_ms:.1}ms \
                 moe={moe_total:.1}ms other(gpu+sample)={other:.1}ms"
            );
            per_token_timings.push(tok_timings);
        }
        let tok_str = crate::tokenizer::decode_token(tokenizer, next_id)
            .unwrap_or_else(|| format!("<{next_id}>"));
        let is_eos = crate::vindex::is_end_of_turn(tok_str.trim());
        tokens.push(tok_str);
        current_ids.push(next_id);
        if is_eos {
            break;
        }
    }

    if timing_enabled {
        print_run_summary("generate", &per_token_timings);
    }

    Ok(GridGenerateResult { tokens, decode_ms })
}

/// Batch pre-dispatch variant of [`generate_with_remote_moe`].
///
/// Each decode step runs two Metal passes:
///   1. **SKIP_MOE pass**: Metal runs attention + dense FFN with zero expert
///      contributions, capturing `h_post_attn` at each of the 30 MoE layers.
///   2. **Batch dispatch**: ONE gRPC `ExpertBatch` call per shard (parallel),
///      carrying all 30 layers' expert inputs.  The server processes all 120
///      expert matmuls concurrently with `join_all(spawn_blocking)`.
///   3. **Apply pass**: Metal runs the same 30 layers, but `moe_fn` now returns
///      the pre-computed h2 instead of calling remote shards per-layer.
///
/// **Trade-off vs streaming**: streaming is exact (each layer's `h_post_attn`
/// includes all previous layers' expert contributions). Batch uses the
/// SKIP_MOE pass `h_post_attn` as an approximation — the error is small for
/// well-trained models and typically produces the same top-1 token.
///
/// **Speed**: streaming makes 30 sequential round-trips per token (each paying
/// ~3.5ms server compute + condvar overhead).  Batch makes ONE round-trip whose
/// server-side cost is max(N_experts / N_cores) × t_expert — much less than
/// 30 × t_expert when the server has enough parallel cores.
pub fn generate_with_remote_moe_batch(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    prompt_ids: Vec<u32>,
    max_tokens: usize,
    index: &VectorIndex,
    remote: &RemoteMoeBackend,
    backend: &dyn ComputeBackend,
) -> Result<GridGenerateResult, RemoteMoeError> {
    let arch = &*weights.arch;
    let norm_offset = arch.norm_weight_offset();
    let eps = arch.norm_eps();
    let hidden = weights.hidden_size;
    let num_layers = weights.num_layers;

    let gate_index: &dyn larql_vindex::GateIndex = index;
    let q4_ffn = gate_index
        .interleaved_q4k_mmap_ref()
        .or_else(|| gate_index.interleaved_q4_mmap_ref())
        .ok_or_else(|| RemoteMoeError::BadResponse("no interleaved Q4 FFN mmap".into()))?;
    let ffn_is_q4k = gate_index.interleaved_q4k_mmap_ref().is_some();
    let intermediate = gate_index.num_features(0);
    let q4_ffn_per_matrix = if ffn_is_q4k {
        (intermediate * hidden).div_ceil(256) * 144
    } else {
        intermediate * hidden / 32 * 18
    };
    let ffn_format = if ffn_is_q4k {
        larql_compute::QuantFormat::Q4_K
    } else {
        larql_compute::QuantFormat::Q4_0
    };
    let layers = crate::layer_graph::pipeline_layer::build_pipeline_layers(
        weights, index, 0..num_layers, q4_ffn, q4_ffn_per_matrix, ffn_format,
    );

    let q_dim = weights.num_q_heads * weights.head_dim;
    let kv_dim = weights.num_kv_heads * weights.head_dim;
    let rope = arch.rope_base_for_layer(0) as f32;

    // Prefill: sequential decode_token_with_moe (same as streaming variant).
    backend.reset_kv_cache();
    {
        let kv_shapes: Vec<(usize, usize)> = (0..num_layers)
            .map(|l| (arch.num_kv_heads_for_layer(l), arch.head_dim_for_layer(l)))
            .collect();
        backend.preallocate_kv_cache_per_layer(&kv_shapes, 4096);
    }

    let skip_moe = std::env::var("SKIP_MOE").is_ok();
    let mut last_hidden_vec: Vec<f32> = vec![0.0f32; hidden];
    let mut current_ids = prompt_ids.clone();

    for &tok_id in &prompt_ids {
        let tok_embed = embed_tokens_pub(weights, &[tok_id]);
        let x_tok: Vec<f32> = tok_embed.as_slice().unwrap_or(&[]).to_vec();
        let mut step_error: Option<RemoteMoeError> = None;
        let mut h_capture: Vec<Vec<f32>> = Vec::with_capacity(num_layers);
        let mut moe_fn_pass1 = |layer: usize, h: &[f32]| -> Vec<f32> {
            if h_capture.len() == layer { h_capture.push(h.to_vec()); }
            vec![0.0f32; hidden]
        };
        let h = backend.decode_token_with_moe(
            &layers, &x_tok, hidden, intermediate, q_dim, kv_dim,
            weights.num_q_heads, weights.num_kv_heads, weights.head_dim, rope, &mut moe_fn_pass1,
        );
        // Dispatch captured layers
        let routers: Vec<_> = (0..h_capture.len())
            .filter_map(|l| build_router(weights, arch, l))
            .collect();
        let h2_per_layer = if skip_moe || h_capture.is_empty() {
            vec![vec![0.0f32; hidden]; num_layers]
        } else {
            remote.forward_moe_predispatch(&h_capture, &routers, norm_offset, eps)
                .unwrap_or_else(|_| vec![vec![0.0f32; hidden]; num_layers])
        };
        // Pass 2: apply h2
        let mut li2 = 0usize;
        let mut moe_fn_pass2 = |layer: usize, _h: &[f32]| -> Vec<f32> {
            li2 = layer;
            if layer < h2_per_layer.len() { h2_per_layer[layer].clone() }
            else { vec![0.0f32; hidden] }
        };
        let h2 = backend.decode_token_with_moe(
            &layers, &x_tok, hidden, intermediate, q_dim, kv_dim,
            weights.num_q_heads, weights.num_kv_heads, weights.head_dim, rope, &mut moe_fn_pass2,
        );
        if let Some(e) = step_error { return Err(e); }
        last_hidden_vec = h2.or(h).ok_or_else(|| {
            RemoteMoeError::BadResponse("decode returned None during prefill".into())
        })?;
    }

    // First token from prefill.
    let mut tokens = Vec::new();
    let mut decode_ms = Vec::new();
    let pfa = ndarray::Array2::from_shape_vec((1, hidden), last_hidden_vec.clone())
        .map_err(|e| RemoteMoeError::BadResponse(e.to_string()))?;
    let pfn = apply_norm(weights, &pfa, arch.final_norm_key(), norm_offset);
    let first_id = lm_topk(index, weights, &pfn.row(0).to_owned(), 1, backend)
        .into_iter().next().map(|(id, _)| id).unwrap_or(0);
    let first_tok = crate::tokenizer::decode_token(tokenizer, first_id)
        .unwrap_or_else(|| format!("<{first_id}>"));
    let first_eos = crate::vindex::is_end_of_turn(first_tok.trim());
    tokens.push(first_tok);
    current_ids.push(first_id);
    if first_eos || tokens.len() >= max_tokens {
        return Ok(GridGenerateResult { tokens, decode_ms: vec![0.0] });
    }

    // Decode loop — two Metal passes per token + ONE batch dispatch.
    for _step in 0..max_tokens.saturating_sub(1) {
        let t0 = std::time::Instant::now();
        let next_id = *current_ids.last().unwrap();
        let tok_embed = embed_tokens_pub(weights, &[next_id]);
        let x_tok: Vec<f32> = tok_embed.as_slice().unwrap_or(&[]).to_vec();

        // ── Pass 1: SKIP_MOE, capture h_post_attn at every MoE layer ───────
        let mut h_capture: Vec<Vec<f32>> = Vec::with_capacity(num_layers);
        let mut moe_pass1 = |layer: usize, h: &[f32]| -> Vec<f32> {
            if skip_moe || h_capture.len() == layer { h_capture.push(h.to_vec()); }
            vec![0.0f32; hidden]
        };
        backend.decode_token_with_moe(
            &layers, &x_tok, hidden, intermediate, q_dim, kv_dim,
            weights.num_q_heads, weights.num_kv_heads, weights.head_dim, rope, &mut moe_pass1,
        );

        // ── Batch dispatch: ONE call per shard, all 30 layers ───────────────
        let routers: Vec<_> = (0..h_capture.len())
            .filter_map(|l| build_router(weights, arch, l))
            .collect();
        let h2_per_layer = if skip_moe || h_capture.is_empty() {
            vec![vec![0.0f32; hidden]; num_layers]
        } else {
            match remote.forward_moe_predispatch(&h_capture, &routers, norm_offset, eps) {
                Ok(h2) => h2,
                Err(e) => return Err(e),
            }
        };

        // ── Pass 2: apply pre-computed h2 ───────────────────────────────────
        let mut moe_pass2 = |layer: usize, _h: &[f32]| -> Vec<f32> {
            if layer < h2_per_layer.len() { h2_per_layer[layer].clone() }
            else { vec![0.0f32; hidden] }
        };
        let h_out = backend.decode_token_with_moe(
            &layers, &x_tok, hidden, intermediate, q_dim, kv_dim,
            weights.num_q_heads, weights.num_kv_heads, weights.head_dim, rope, &mut moe_pass2,
        ).ok_or_else(|| RemoteMoeError::BadResponse("pass2 returned None".into()))?;

        // Pick next token.
        let h_arr = ndarray::Array2::from_shape_vec((1, hidden), h_out.clone())
            .map_err(|e| RemoteMoeError::BadResponse(e.to_string()))?;
        let h_normed = apply_norm(weights, &h_arr, arch.final_norm_key(), norm_offset);
        let next_tok_id = lm_topk(index, weights, &h_normed.row(0).to_owned(), 1, backend)
            .into_iter().next().map(|(id, _)| id).unwrap_or(0);

        decode_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        let tok_str = crate::tokenizer::decode_token(tokenizer, next_tok_id)
            .unwrap_or_else(|| format!("<{next_tok_id}>"));
        let is_eos = crate::vindex::is_end_of_turn(tok_str.trim());
        tokens.push(tok_str);
        current_ids.push(next_tok_id);
        if is_eos { break; }
    }

    Ok(GridGenerateResult { tokens, decode_ms })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::test_utils::{make_test_tokenizer, make_test_vindex, make_test_weights};
    use crate::ffn::moe_remote::RemoteMoeBackend;
    use larql_compute::CpuBackend;

    // ── generate_with_remote_moe — error path ────────────────────────────────

    #[test]
    fn errors_when_vindex_has_no_q4k_mmap() {
        let weights = make_test_weights();
        let idx = make_test_vindex(&weights);
        let tokenizer = make_test_tokenizer(weights.vocab_size);

        // make_test_vindex has no interleaved Q4K or Q4 mmap.
        // The function should fail at the mmap guard, before any GPU or shard call.
        let remote = RemoteMoeBackend::new_disconnected();
        let result = generate_with_remote_moe(
            &weights,
            &tokenizer,
            vec![0u32],
            1,
            &idx,
            &remote,
            &CpuBackend,
        );
        match result {
            Err(RemoteMoeError::BadResponse(msg)) => {
                assert!(
                    msg.contains("no interleaved Q4 FFN mmap"),
                    "unexpected error message: {msg}"
                );
            }
            other => panic!("expected BadResponse, got: {other:?}"),
        }
    }
}
