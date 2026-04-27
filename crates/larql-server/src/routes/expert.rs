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
    decode_expert_request, encode_expert_response, ExpertCallItem, ExpertResultItem,
    EXPERT_BINARY_CONTENT_TYPE,
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

fn run_expert(
    state: &AppState,
    layer: usize,
    expert_id: usize,
    residual: &[f32],
) -> Result<Vec<f32>, ServerError> {
    let model = state.model_or_err(None)?;

    // Ownership check: reject if this shard doesn't own this expert.
    // `expert_filter` uses the half-open `[start, end_exclusive)` convention
    // returned by `parse_layer_range`, so the upper bound is exclusive.
    // Display the inclusive bound in the error message to match the CLI flag.
    if let Some((start, end_excl)) = model.expert_filter {
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
        let (gu, dn) = weights.get_layer_entry_bytes(layer, expert_id).ok_or_else(|| {
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
        items
            .iter()
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
                serde_json::to_vec(&resp)
                    .map_err(|e| ServerError::Internal(e.to_string()))?,
            ))
            .map_err(|e| ServerError::Internal(e.to_string()))?
    };

    Ok(response)
}
