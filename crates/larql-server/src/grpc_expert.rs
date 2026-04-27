//! gRPC `ExpertService` implementation.
//!
//! One persistent HTTP/2 stream per shard: all expert matmuls for a decode step
//! arrive in a single `ExpertBatch` RPC call rather than 30 per-layer HTTP POSTs.
//! The gRPC server shares the same listening port as the VindexService (both are
//! registered on the tonic `Server`).

use std::sync::Arc;
use std::time::Instant;

use tonic::{Request, Response, Status};

use larql_router_protocol::{
    ExpertBatchRequest, ExpertBatchResponse, ExpertBatchResult, ExpertService,
};

use crate::state::AppState;

pub struct ExpertGrpcService {
    pub state: Arc<AppState>,
}

#[tonic::async_trait]
impl ExpertService for ExpertGrpcService {
    async fn expert_batch(
        &self,
        request: Request<ExpertBatchRequest>,
    ) -> Result<Response<ExpertBatchResponse>, Status> {
        self.state.bump_requests();
        let start = Instant::now();
        let req = request.into_inner();
        let state = Arc::clone(&self.state);

        let results = tokio::task::spawn_blocking(move || {
            let model = state
                .model(None)
                .ok_or_else(|| Status::not_found("no model loaded"))?;

            req.items
                .iter()
                .map(|item| {
                    let layer = item.layer as usize;
                    let expert_id = item.expert_id as usize;

                    // Decode bytes → f32 residual.
                    if item.residual.len() % 4 != 0 {
                        return Err(Status::invalid_argument(format!(
                            "residual byte length {} not divisible by 4",
                            item.residual.len()
                        )));
                    }
                    let residual: Vec<f32> = item
                        .residual
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                        .collect();

                    // Run expert (same logic as HTTP handle_expert_batch).
                    let output =
                        crate::routes::expert::run_expert(&model, layer, expert_id, &residual)
                            .map_err(|e| Status::internal(e.to_string()))?;

                    // Encode f32 output → bytes.
                    let output_bytes: Vec<u8> = output
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect();

                    Ok(ExpertBatchResult {
                        layer: item.layer,
                        expert_id: item.expert_id,
                        output: output_bytes,
                    })
                })
                .collect::<Result<Vec<_>, Status>>()
        })
        .await
        .map_err(|e| Status::internal(e.to_string()))??;

        let latency_ms = start.elapsed().as_secs_f32() * 1000.0;
        Ok(Response::new(ExpertBatchResponse {
            results,
            latency_ms,
        }))
    }
}
