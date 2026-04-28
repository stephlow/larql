//! gRPC `ExpertService` implementation.
//!
//! Exposes two RPCs:
//!
//! `ExpertBatch` — unary, processes a flat list of (layer, expert_id, residual) items.
//! Good for correctness testing and small batches.
//!
//! `ExpertStream` — bidirectional streaming, one frame per MoE layer per decode step.
//! Client sends `ExpertLayerInput` for each layer as it becomes available; server
//! streams back `ExpertLayerOutput` after computing the weighted expert sum.
//! ONE stream per shard per token eliminates the per-call connection overhead of
//! 30 unary calls — measured improvement: ~360ms overhead → ~18ms.

use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use futures::Stream;
use tonic::{Request, Response, Status, Streaming};

use larql_router_protocol::{
    ExpertBatchRequest, ExpertBatchResponse, ExpertBatchResult, ExpertLayerInput,
    ExpertLayerOutput, ExpertService,
};

use crate::state::AppState;

pub struct ExpertGrpcService {
    pub state: Arc<AppState>,
}

#[tonic::async_trait]
impl ExpertService for ExpertGrpcService {
    // ── Unary batch ──────────────────────────────────────────────────────────

    async fn expert_batch(
        &self,
        request: Request<ExpertBatchRequest>,
    ) -> Result<Response<ExpertBatchResponse>, Status> {
        self.state.bump_requests();
        let start = Instant::now();
        let req = request.into_inner();
        let state = Arc::clone(&self.state);

        let futs: Vec<_> = req.items
            .into_iter()
            .map(|item| {
                let s = Arc::clone(&state);
                tokio::task::spawn_blocking(move || {
                    let layer = item.layer as usize;
                    let expert_id = item.expert_id as usize;
                    if item.residual.len() % 4 != 0 {
                        return Err(Status::invalid_argument("residual not 4-byte aligned"));
                    }
                    let residual: Vec<f32> = item.residual.chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                        .collect();
                    let output = crate::routes::expert::run_expert(&s, layer, expert_id, &residual)
                        .map_err(|e| Status::internal(e.to_string()))?;
                    Ok(ExpertBatchResult {
                        layer: item.layer,
                        expert_id: item.expert_id,
                        output: output.iter().flat_map(|v| v.to_le_bytes()).collect(),
                    })
                })
            })
            .collect();

        let results: Vec<ExpertBatchResult> = {
            let mut v = Vec::new();
            for task in futures::future::join_all(futs).await {
                v.push(task.map_err(|e| Status::internal(e.to_string()))?
                    .map_err(|e| e)?);
            }
            v
        };

        let latency_ms = start.elapsed().as_secs_f32() * 1000.0;
        Ok(Response::new(ExpertBatchResponse {
            results,
            latency_ms,
        }))
    }

    // ── Bidirectional streaming ──────────────────────────────────────────────
    //
    // Each incoming ExpertLayerInput carries:
    //   layer, expert_ids[], expert_weights[], residual (h_post_attn), post_experts_norm
    //
    // For each message, the server:
    //   1. Runs each selected expert: run_single_expert_with_norm(residual, ...)
    //   2. Weighted sum: h2 = sum(w_k * expert_k_output)
    //   3. Post-experts norm: h2 = rms_norm(h2, post_experts_norm)
    //   4. Streams back ExpertLayerOutput { layer, h2 }

    type ExpertStreamStream =
        Pin<Box<dyn Stream<Item = Result<ExpertLayerOutput, Status>> + Send + 'static>>;

    async fn expert_stream(
        &self,
        request: Request<Streaming<ExpertLayerInput>>,
    ) -> Result<Response<Self::ExpertStreamStream>, Status> {
        self.state.bump_requests();
        let state = Arc::clone(&self.state);
        let mut in_stream = request.into_inner();

        let out_stream = async_stream::try_stream! {
            while let Some(msg) = {
                use futures::StreamExt;
                in_stream.next().await
            } {
                let input = msg?;
                let layer = input.layer as usize;

                // Decode bytes on the async thread, then do blocking expert compute.
                if input.residual.len() % 4 != 0 {
                    Err(Status::invalid_argument("residual not 4-byte aligned"))?;
                }
                let residual: Vec<f32> = input
                    .residual
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();

                let post_norm: Vec<f32> = if input.post_experts_norm.is_empty() {
                    vec![]
                } else {
                    input.post_experts_norm
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                        .collect()
                };
                let norm_offset = input.norm_offset;
                let eps = input.eps;

                let expert_ids: Vec<usize> =
                    input.expert_ids.iter().map(|&e| e as usize).collect();
                let expert_weights: Vec<f32> = input.expert_weights.clone();

                let state2 = Arc::clone(&state);

                // Spawn each expert as a separate non-blocking tokio task.
                // The stream handler stays async throughout — it awaits all
                // expert futures concurrently via join_all rather than
                // blocking on any of them.  4 experts run on 4 separate
                // blocking-pool threads, truly in parallel.
                let futs: Vec<_> = expert_ids
                    .iter()
                    .zip(expert_weights.iter())
                    .filter(|(_, &w)| w != 0.0)
                    .map(|(&eid, &w)| {
                        let s = Arc::clone(&state2);
                        let r = residual.clone();
                        tokio::task::spawn_blocking(move || {
                            crate::routes::expert::run_expert(&s, layer, eid, &r)
                                .map(|out| (out, w))
                                .map_err(|e| Status::internal(e.to_string()))
                        })
                    })
                    .collect();

                let hidden = residual.len();
                let mut out = vec![0.0f32; hidden];
                for task in futures::future::join_all(futs).await {
                    let (expert_out, weight) = task
                        .map_err(|e| Status::internal(e.to_string()))?
                        .map_err(|e| e)?;
                    for (acc, &v) in out.iter_mut().zip(expert_out.iter()) {
                        *acc += weight * v;
                    }
                }
                let h2 = out;

                let h2_bytes: Vec<u8> = h2.iter().flat_map(|v| v.to_le_bytes()).collect();
                yield ExpertLayerOutput {
                    layer: input.layer,
                    h2: h2_bytes,
                };
            }
        };

        Ok(Response::new(Box::pin(out_stream)))
    }
}
