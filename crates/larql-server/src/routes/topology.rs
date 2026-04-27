//! `GET /v1/expert/topology` — advertise this shard's expert ownership range.
//!
//! Returns the expert ID range `[owned_start, owned_end]` (inclusive) that
//! this server was launched with via `--experts START-END`. Clients use this
//! to build the shard map dynamically instead of having it baked into the
//! `--moe-shards` flag.
//!
//! Returns HTTP 404 when the server was not launched with `--experts` (i.e.,
//! it owns all experts or is not operating as an expert shard).

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use serde::Serialize;

use crate::state::AppState;

#[derive(Serialize)]
pub struct TopologyResponse {
    /// Model identifier (e.g. `"google/gemma-4-26B-A4B-it"`).
    pub model_id: String,
    /// Total number of experts in the model (0 for non-MoE models).
    pub num_experts: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// First expert ID owned by this shard (inclusive).
    pub owned_start: usize,
    /// Last expert ID owned by this shard (inclusive).
    pub owned_end: usize,
}

pub async fn handle_topology(
    State(state): State<Arc<AppState>>,
) -> Result<Json<TopologyResponse>, StatusCode> {
    let model = state.model_or_err(None).map_err(|_| StatusCode::NOT_FOUND)?;

    // 404 if this server was not launched with --experts (no shard filter set).
    let (start, end_excl) = model.expert_filter.ok_or(StatusCode::NOT_FOUND)?;

    let num_experts = model
        .config
        .model_config
        .as_ref()
        .and_then(|m| m.moe.as_ref())
        .map(|m| m.num_experts)
        .unwrap_or(0);

    Ok(Json(TopologyResponse {
        model_id: model.id.clone(),
        num_experts,
        num_layers: model.config.num_layers,
        owned_start: start,
        owned_end: end_excl.saturating_sub(1), // convert exclusive→inclusive for display
    }))
}
