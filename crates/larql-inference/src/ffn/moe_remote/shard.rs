use std::sync::Arc;
use std::time::Duration;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::config::ShardConfig;
use super::error::RemoteMoeError;
use super::multi_layer_wire::{
    decode_multi_layer_response, encode_multi_layer_request, encode_multi_layer_request_q8k,
    MultiLayerResult, MultiLayerTask, MultiLayerTaskQ8K, MULTI_LAYER_BATCH_CONTENT_TYPE,
    MULTI_LAYER_BATCH_Q8K_CONTENT_TYPE,
};
use super::router::{rms_norm, MoeRouterWeights};
use super::stream::{InflightMoe, ShardStream};
use super::wire::{
    decode_expert_response, decode_layer_batch_response, decode_layer_batch_response_f16,
    encode_expert_request, encode_layer_batch_request, encode_layer_batch_request_f16,
    ExpertCallItem, ExpertResultItem, EXPERT_BINARY_CONTENT_TYPE, LAYER_BATCH_CONTENT_TYPE,
    LAYER_BATCH_F16_CONTENT_TYPE,
};

// ── Internal shard state ──────────────────────────────────────────────────────

pub(super) struct GrpcState {
    runtime: std::sync::Arc<tokio::runtime::Runtime>,
    client: larql_router_protocol::ExpertServiceClient<tonic::transport::Channel>,
}

pub(super) enum ShardTransport {
    Http(reqwest::blocking::Client),
    Grpc(std::sync::Arc<GrpcState>),
    /// Unix domain socket transport for same-host shards.  Holds one
    /// persistent stream per shard behind a `Mutex` (per-shard calls
    /// are sequential within a `forward_moe`, and across `forward_moe`
    /// calls in chat mode).  Manual HTTP/1.1 framing keeps the wire
    /// protocol identical to the TCP `Http` variant — server-side it's
    /// the same axum router on a `UnixListener`.
    ///
    /// Saves ~50 µs/call on loopback by skipping the kernel TCP stack
    /// (no Nagle, no delayed ACK, no socket buffer copies through the
    /// network stack).  Most of the saving is on the response path
    /// (server flushes complete writes immediately).
    Uds(UdsState),
}

struct UdsState {
    /// Filesystem path of the socket.  Used in error messages.
    path: std::path::PathBuf,
    /// Persistent stream behind a mutex.  Reconnect lazily on disconnect.
    stream: std::sync::Mutex<Option<std::os::unix::net::UnixStream>>,
}

pub(super) struct Shard {
    pub(super) config: ShardConfig,
    pub(super) transport: ShardTransport,
}

impl Shard {
    pub(super) fn connect(config: ShardConfig) -> Result<Self, RemoteMoeError> {
        // URL scheme dispatch:
        //   `grpc://host:port` → tonic gRPC over HTTP/2 persistent channel.
        //   `unix:///path/to/sock` → manual HTTP/1.1 over a Unix domain
        //     socket (same-host fast path; ~50 µs/call faster than TCP
        //     loopback).
        //   `http://host:port` → reqwest blocking HTTP/1.1 (default).
        let transport = if let Some(uds_path) = config
            .url
            .strip_prefix("unix://")
            .or_else(|| config.url.strip_prefix("unix:"))
        {
            // Strip the leading `///` of `unix:///abs/path` (the third `/`
            // is part of the path).  `unix:relative/path` also accepted.
            let path = std::path::PathBuf::from(uds_path);
            // Open + health check.
            let stream = std::os::unix::net::UnixStream::connect(&path).map_err(|e| {
                RemoteMoeError::Unreachable {
                    url: format!("unix://{}", path.display()),
                    cause: e.to_string(),
                }
            })?;
            // Apply the configured timeout to read/write so a stuck shard
            // doesn't wedge the client forever.
            let _ = stream.set_read_timeout(Some(config.timeout));
            let _ = stream.set_write_timeout(Some(config.timeout));
            ShardTransport::Uds(UdsState {
                path,
                stream: std::sync::Mutex::new(Some(stream)),
            })
        } else if config.url.starts_with("grpc://") || config.url.starts_with("grpcs://") {
            let use_tls = config.url.starts_with("grpcs://");
            let grpc_endpoint = if use_tls {
                config.url.replacen("grpcs://", "https://", 1)
            } else {
                config.url.replacen("grpc://", "http://", 1)
            };
            let rt = std::sync::Arc::new(
                tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .build()
                    .map_err(|e| RemoteMoeError::Client(e.to_string()))?,
            );
            let client = if use_tls {
                let endpoint = tonic::transport::Channel::from_shared(grpc_endpoint.clone())
                    .map_err(|e| RemoteMoeError::Client(e.to_string()))?
                    .tls_config(tonic::transport::ClientTlsConfig::new().with_webpki_roots())
                    .map_err(|e| RemoteMoeError::Client(e.to_string()))?;
                let channel =
                    rt.block_on(endpoint.connect())
                        .map_err(|e| RemoteMoeError::Unreachable {
                            url: grpc_endpoint,
                            cause: e.to_string(),
                        })?;
                larql_router_protocol::ExpertServiceClient::new(channel)
            } else {
                rt.block_on(larql_router_protocol::ExpertServiceClient::connect(
                    grpc_endpoint.clone(),
                ))
                .map_err(|e| RemoteMoeError::Unreachable {
                    url: grpc_endpoint,
                    cause: e.to_string(),
                })?
            };
            ShardTransport::Grpc(std::sync::Arc::new(GrpcState {
                runtime: rt,
                client,
            }))
        } else {
            let http = reqwest::blocking::Client::builder()
                .timeout(config.timeout)
                .pool_max_idle_per_host(64)
                .build()
                .map_err(|e| RemoteMoeError::Client(e.to_string()))?;
            // Health check on HTTP shards only (gRPC connect already verifies).
            let health_url = format!("{}/v1/health", config.url);
            let resp = http
                .get(&health_url)
                .send()
                .map_err(|e| RemoteMoeError::Unreachable {
                    url: health_url.clone(),
                    cause: e.to_string(),
                })?;
            if !resp.status().is_success() {
                return Err(RemoteMoeError::ServerError {
                    status: resp.status().as_u16(),
                    body: resp.text().unwrap_or_default(),
                });
            }
            ShardTransport::Http(http)
        };

        Ok(Self { config, transport })
    }

    /// Layer-uniform ownership check (legacy `--moe-shards "S-E=URL"` path).
    /// Used by routing call sites that don't know the layer — keep returning
    /// `false` for fine-grained shards so the layer-aware `owns_unit` is
    /// always preferred when the layer is in scope.
    pub(super) fn owns(&self, expert_id: usize) -> bool {
        if self.config.unit_set.is_some() {
            // Fine-grained shards never claim ownership without a layer
            // context — forces callers to use `owns_unit` instead.
            return false;
        }
        expert_id >= self.config.start && expert_id <= self.config.end
    }

    /// Layer-aware ownership check.  When the shard's `unit_set` is set
    /// (`--moe-units-manifest`), checks the explicit `(layer, expert_id)`
    /// membership; otherwise falls back to the layer-uniform range so
    /// existing `--moe-shards "0-63=URL"` configs keep working unchanged.
    pub(super) fn owns_unit(&self, layer: usize, expert_id: usize) -> bool {
        if let Some(units) = self.config.unit_set.as_ref() {
            return units.contains(&(layer, expert_id));
        }
        expert_id >= self.config.start && expert_id <= self.config.end
    }

    /// True if this shard uses gRPC transport (not HTTP or UDS).
    /// Used by `backend.rs` to decide whether to use the multi-layer fast path.
    pub(super) fn is_grpc(&self) -> bool {
        matches!(self.transport, ShardTransport::Grpc(_))
    }

    /// Open a bidirectional gRPC stream for one decode step.
    ///
    /// Spawns a dedicated async tokio task that:
    ///   1. Reads work inputs from `work_rx` (async channel — no thread wakeup)
    ///   2. Sends them on the gRPC stream via `await` (no block_on)
    ///   3. Awaits the server's response (async)
    ///   4. Puts the decoded result in `result_tx` (sync mpsc — condvar wakeup)
    ///
    /// The sync Metal thread communicates via `work_tx.send` (non-blocking) and
    /// `result_rx.recv()` (condvar, ~0.1ms) — no tokio Runtime::block_on anywhere.
    pub(super) fn open_stream(&self) -> Result<ShardStream, RemoteMoeError> {
        match &self.transport {
            ShardTransport::Grpc(grpc) => {
                let rt = std::sync::Arc::clone(&grpc.runtime);
                let mut client = grpc.client.clone();

                // Work channel: Metal thread → async task (non-blocking send)
                let (work_tx, mut work_rx) = tokio::sync::mpsc::unbounded_channel::<
                    larql_router_protocol::ExpertLayerInput,
                >();

                // Result channel: async task → Metal thread (condvar recv).
                // The f32 carries `compute_ms` from the server (0.0 when the
                // server isn't recording timing) so the client can decompose
                // its wall-clock collect time into network vs server compute.
                let (result_tx, result_rx) =
                    std::sync::mpsc::channel::<Result<(Vec<f32>, f32), RemoteMoeError>>();

                // Open the gRPC stream + spawn the dispatch task in one block_on.
                // This is the ONLY block_on — one-time stream setup, not per-layer.
                rt.block_on(async {
                    // Channel for feeding the gRPC request stream.
                    let (grpc_input_tx, mut grpc_input_rx) = tokio::sync::mpsc::unbounded_channel::<
                        larql_router_protocol::ExpertLayerInput,
                    >();

                    let req_stream = async_stream::stream! {
                        while let Some(msg) = grpc_input_rx.recv().await { yield msg; }
                    };
                    let mut grpc_output = client
                        .expert_stream(tonic::Request::new(req_stream))
                        .await
                        .map(|r| r.into_inner())
                        .map_err(|e| RemoteMoeError::ServerError {
                            status: e.code() as u16,
                            body: e.message().to_string(),
                        })?;

                    // Spawn the async dispatch loop.
                    tokio::spawn(async move {
                        use futures::StreamExt;
                        while let Some(input) = work_rx.recv().await {
                            // Forward input to gRPC stream.
                            if grpc_input_tx.send(input).is_err() {
                                break;
                            }
                            // Await server response (pure async, no block_on).
                            let result = match grpc_output.next().await {
                                Some(Ok(out)) => {
                                    if out.h2.len() % 4 != 0 {
                                        Err(RemoteMoeError::BadResponse("h2 unaligned".into()))
                                    } else {
                                        let h2: Vec<f32> = out
                                            .h2
                                            .chunks_exact(4)
                                            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                                            .collect();
                                        Ok((h2, out.compute_ms))
                                    }
                                }
                                Some(Err(e)) => Err(RemoteMoeError::ServerError {
                                    status: e.code() as u16,
                                    body: e.message().to_string(),
                                }),
                                None => Err(RemoteMoeError::BadResponse("stream ended".into())),
                            };
                            // Wake the Metal thread via condvar (much cheaper than block_on).
                            if result_tx.send(result).is_err() {
                                break;
                            }
                        }
                    });

                    Ok::<(), RemoteMoeError>(())
                })?;

                Ok(ShardStream {
                    work_tx,
                    result_rx: std::sync::Mutex::new(result_rx),
                    _runtime: rt,
                })
            }
            ShardTransport::Http(_) | ShardTransport::Uds(_) => Err(RemoteMoeError::Client(
                "open_stream requires grpc:// shards".into(),
            )),
        }
    }

    /// Send a batch of expert calls to this shard.
    ///
    /// Dispatches via gRPC (persistent HTTP/2) when the shard URL starts with
    /// `grpc://`, otherwise falls back to binary HTTP.
    pub(super) fn call_batch(
        &self,
        requests: &[ExpertCallItem],
    ) -> Result<Vec<ExpertResultItem>, RemoteMoeError> {
        match &self.transport {
            ShardTransport::Grpc(grpc) => {
                // Build protobuf items — raw bytes for residuals avoids varint overhead.
                let items: Vec<larql_router_protocol::ExpertBatchItem> = requests
                    .iter()
                    .map(|r| larql_router_protocol::ExpertBatchItem {
                        layer: r.layer as u32,
                        expert_id: r.expert_id as u32,
                        residual: r.residual.iter().flat_map(|v| v.to_le_bytes()).collect(),
                    })
                    .collect();

                let grpc_req = larql_router_protocol::ExpertBatchRequest { items };
                // Block on the async gRPC call from this sync context.
                let mut client = grpc.client.clone();
                let t_call = std::time::Instant::now();
                let resp = grpc
                    .runtime
                    .block_on(client.expert_batch(tonic::Request::new(grpc_req)))
                    .map_err(|e| RemoteMoeError::ServerError {
                        status: e.code() as u16,
                        body: e.message().to_string(),
                    })?
                    .into_inner();

                eprintln!(
                    "[call_batch/grpc] n={} block_on={:.1}ms",
                    requests.len(),
                    t_call.elapsed().as_secs_f64() * 1000.0
                );
                // Decode proto results back to ExpertResultItem.
                resp.results
                    .into_iter()
                    .map(|r| {
                        if r.output.len() % 4 != 0 {
                            return Err(RemoteMoeError::BadResponse(
                                "output bytes not divisible by 4".into(),
                            ));
                        }
                        let output: Vec<f32> = r
                            .output
                            .chunks_exact(4)
                            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                            .collect();
                        Ok(ExpertResultItem {
                            layer: r.layer as usize,
                            expert_id: r.expert_id as usize,
                            output,
                        })
                    })
                    .collect()
            }

            ShardTransport::Http(client) => {
                // Binary HTTP fallback (application/x-larql-expert).
                let url = format!("{}/v1/expert/batch", self.config.url);
                let body = encode_expert_request(requests);
                let resp = client
                    .post(&url)
                    .header("Content-Type", EXPERT_BINARY_CONTENT_TYPE)
                    .header("Accept", EXPERT_BINARY_CONTENT_TYPE)
                    .body(body)
                    .send()
                    .map_err(|e| RemoteMoeError::Unreachable {
                        url: url.clone(),
                        cause: e.to_string(),
                    })?;

                if !resp.status().is_success() {
                    return Err(RemoteMoeError::ServerError {
                        status: resp.status().as_u16(),
                        body: resp.text().unwrap_or_default(),
                    });
                }

                let bytes = resp
                    .bytes()
                    .map_err(|e| RemoteMoeError::BadResponse(e.to_string()))?;
                decode_expert_response(&bytes)
                    .ok_or_else(|| RemoteMoeError::BadResponse("binary response truncated".into()))
            }
            ShardTransport::Uds(uds) => {
                // Same wire body as the HTTP path; UDS framing is identical
                // to TCP HTTP/1.1 — only the transport differs.
                let body = encode_expert_request(requests);
                let resp_bytes =
                    uds_call(uds, "/v1/expert/batch", EXPERT_BINARY_CONTENT_TYPE, &body)?;
                decode_expert_response(&resp_bytes).ok_or_else(|| {
                    RemoteMoeError::BadResponse("UDS expert/batch response truncated".into())
                })
            }
        }
    }

    /// Send a layer-batch request: ONE residual + K (expert_id, weight) pairs.
    /// Returns the router-weighted sum across the K experts owned by this
    /// shard.  Eliminates the K-1 redundant residual copies on the wire and
    /// the K-1 redundant `pre_experts_norm` + Q8_K quantisations on the
    /// server (the server applies them once and shares across the K experts).
    ///
    /// HTTP-only for now (gRPC variant TODO).  Falls back to `call_batch` if
    /// the shard transport is gRPC.
    pub(super) fn call_layer_batch(
        &self,
        layer: usize,
        residual: &[f32],
        expert_ids: &[u32],
        expert_weights: &[f32],
    ) -> Result<Vec<f32>, RemoteMoeError> {
        match &self.transport {
            ShardTransport::Grpc(_) => {
                // TODO: gRPC variant.  For now, encode-and-fall-back to
                // call_batch with K identical residuals.
                let items: Vec<ExpertCallItem> = expert_ids
                    .iter()
                    .map(|&eid| ExpertCallItem {
                        layer,
                        expert_id: eid as usize,
                        residual: residual.to_vec(),
                    })
                    .collect();
                let results = self.call_batch(&items)?;
                // Apply weights and sum on the client (mirrors the server's
                // run_experts_cpu_batch behaviour for the http path).
                let hidden = residual.len();
                let mut out = vec![0.0f32; hidden];
                for (i, item) in results.iter().enumerate() {
                    let w = expert_weights[i];
                    for (a, &v) in out.iter_mut().zip(item.output.iter()) {
                        *a += w * v;
                    }
                }
                Ok(out)
            }
            ShardTransport::Http(client) => {
                // Per-stage client-side timing (`LARQL_HTTP_TIMING=1`).
                thread_local! {
                    static HTTP_TIMING: bool =
                        std::env::var("LARQL_HTTP_TIMING").is_ok();
                }
                let timing = HTTP_TIMING.with(|t| *t);

                // Wire format selection.  Default f32 (loopback / same-host
                // grids — TCP buffer/copy costs dominate, f16 conversion
                // CPU cost cancels the wire-bytes saving).  Set
                // `LARQL_MOE_WIRE_F16=1` for LAN deployments where the
                // 5 KB/call wire saving matters more than the 9 µs/call
                // f32↔f16 conversion CPU.  Bench (M3 Max loopback,
                // 2026-05-01): f16 was 0.5-1% slower (within noise) on
                // 100-token poem; expected to invert on >100 µs RTT links.
                thread_local! {
                    static USE_F16_WIRE: bool =
                        std::env::var("LARQL_MOE_WIRE_F16").is_ok();
                }
                let use_f16 = USE_F16_WIRE.with(|v| *v);

                let url = if use_f16 {
                    format!("{}/v1/experts/layer-batch-f16", self.config.url)
                } else {
                    format!("{}/v1/experts/layer-batch", self.config.url)
                };
                let ct = if use_f16 {
                    LAYER_BATCH_F16_CONTENT_TYPE
                } else {
                    LAYER_BATCH_CONTENT_TYPE
                };

                let t_encode_in = std::time::Instant::now();
                let body = if use_f16 {
                    encode_layer_batch_request_f16(layer, residual, expert_ids, expert_weights)
                } else {
                    encode_layer_batch_request(layer, residual, expert_ids, expert_weights)
                };
                let t_encode = t_encode_in.elapsed();

                let t_send_in = std::time::Instant::now();
                let resp = client
                    .post(&url)
                    .header("Content-Type", ct)
                    .header("Accept", ct)
                    .body(body)
                    .send()
                    .map_err(|e| RemoteMoeError::Unreachable {
                        url: url.clone(),
                        cause: e.to_string(),
                    })?;
                let t_send = t_send_in.elapsed();

                if !resp.status().is_success() {
                    return Err(RemoteMoeError::ServerError {
                        status: resp.status().as_u16(),
                        body: resp.text().unwrap_or_default(),
                    });
                }

                let t_recv_in = std::time::Instant::now();
                let bytes = resp
                    .bytes()
                    .map_err(|e| RemoteMoeError::BadResponse(e.to_string()))?;
                let t_recv = t_recv_in.elapsed();

                let t_decode_in = std::time::Instant::now();
                let out = if use_f16 {
                    decode_layer_batch_response_f16(&bytes)
                } else {
                    decode_layer_batch_response(&bytes)
                }
                .ok_or_else(|| {
                    RemoteMoeError::BadResponse("layer-batch response truncated".into())
                });
                let t_decode = t_decode_in.elapsed();

                if timing {
                    eprintln!(
                        "[shard.call_layer_batch] layer={layer} K={} wire={} \
                         encode={:.0}us send_total={:.0}us recv_body={:.0}us decode={:.0}us",
                        expert_ids.len(),
                        if use_f16 { "f16" } else { "f32" },
                        t_encode.as_secs_f64() * 1e6,
                        t_send.as_secs_f64() * 1e6,
                        t_recv.as_secs_f64() * 1e6,
                        t_decode.as_secs_f64() * 1e6,
                    );
                }

                out
            }
            ShardTransport::Uds(uds) => {
                // Manual HTTP/1.1 over UnixStream — same wire format as
                // the TCP `Http` variant, just no TCP stack.  The server
                // is the same axum router on a `UnixListener`; from the
                // handler's perspective it can't tell.
                thread_local! {
                    static HTTP_TIMING: bool =
                        std::env::var("LARQL_HTTP_TIMING").is_ok();
                    static USE_F16_WIRE: bool =
                        std::env::var("LARQL_MOE_WIRE_F16").is_ok();
                }
                let timing = HTTP_TIMING.with(|t| *t);
                let use_f16 = USE_F16_WIRE.with(|v| *v);

                let path = if use_f16 {
                    "/v1/experts/layer-batch-f16"
                } else {
                    "/v1/experts/layer-batch"
                };
                let ct = if use_f16 {
                    LAYER_BATCH_F16_CONTENT_TYPE
                } else {
                    LAYER_BATCH_CONTENT_TYPE
                };

                let t_encode_in = std::time::Instant::now();
                let body = if use_f16 {
                    encode_layer_batch_request_f16(layer, residual, expert_ids, expert_weights)
                } else {
                    encode_layer_batch_request(layer, residual, expert_ids, expert_weights)
                };
                let t_encode = t_encode_in.elapsed();

                let t_send_in = std::time::Instant::now();
                let resp_bytes = uds_call(uds, path, ct, &body)?;
                let t_send = t_send_in.elapsed();

                let t_decode_in = std::time::Instant::now();
                let out = if use_f16 {
                    decode_layer_batch_response_f16(&resp_bytes)
                } else {
                    decode_layer_batch_response(&resp_bytes)
                }
                .ok_or_else(|| {
                    RemoteMoeError::BadResponse("layer-batch response truncated (uds)".into())
                });
                let t_decode = t_decode_in.elapsed();

                if timing {
                    eprintln!(
                        "[shard.call_layer_batch] layer={layer} K={} wire={} \
                         transport=uds encode={:.0}us send_total={:.0}us decode={:.0}us",
                        expert_ids.len(),
                        if use_f16 { "f16" } else { "f32" },
                        t_encode.as_secs_f64() * 1e6,
                        t_send.as_secs_f64() * 1e6,
                        t_decode.as_secs_f64() * 1e6,
                    );
                }
                out
            }
        }
    }

    /// Send all layers' routing decisions in one request, receive all h2 values.
    ///
    /// HTTP and UDS only.  The sequential server-side loop eliminates rayon
    /// oversubscription; each task gets the full thread pool.
    pub(super) fn call_multi_layer_batch(
        &self,
        tasks: &[MultiLayerTask],
    ) -> Result<Vec<MultiLayerResult>, RemoteMoeError> {
        let body = encode_multi_layer_request(tasks);
        match &self.transport {
            ShardTransport::Http(client) => {
                let url = format!("{}/v1/experts/multi-layer-batch", self.config.url);
                let resp = client
                    .post(&url)
                    .header("Content-Type", MULTI_LAYER_BATCH_CONTENT_TYPE)
                    .header("Accept", MULTI_LAYER_BATCH_CONTENT_TYPE)
                    .body(body)
                    .send()
                    .map_err(|e| RemoteMoeError::Unreachable {
                        url: url.clone(),
                        cause: e.to_string(),
                    })?;
                if !resp.status().is_success() {
                    return Err(RemoteMoeError::ServerError {
                        status: resp.status().as_u16(),
                        body: resp.text().unwrap_or_default(),
                    });
                }
                let bytes = resp
                    .bytes()
                    .map_err(|e| RemoteMoeError::BadResponse(e.to_string()))?;
                decode_multi_layer_response(&bytes).ok_or_else(|| {
                    RemoteMoeError::BadResponse("multi-layer-batch response truncated".into())
                })
            }
            ShardTransport::Uds(uds) => {
                let resp_bytes = uds_call(
                    uds,
                    "/v1/experts/multi-layer-batch",
                    MULTI_LAYER_BATCH_CONTENT_TYPE,
                    &body,
                )?;
                decode_multi_layer_response(&resp_bytes).ok_or_else(|| {
                    RemoteMoeError::BadResponse("UDS multi-layer-batch response truncated".into())
                })
            }
            ShardTransport::Grpc(_) => Err(RemoteMoeError::Client(
                "call_multi_layer_batch unavailable for gRPC shards".into(),
            )),
        }
    }

    /// Q8K-prenormed variant: client sends pre-quantised h_norm instead of
    /// the raw residual.  4× smaller upload; server skips pre_experts_norm
    /// + Q8K quantisation and calls the matvec directly.
    pub(super) fn call_multi_layer_batch_q8k(
        &self,
        tasks: &[MultiLayerTaskQ8K],
    ) -> Result<Vec<MultiLayerResult>, RemoteMoeError> {
        let body = encode_multi_layer_request_q8k(tasks);
        match &self.transport {
            ShardTransport::Http(client) => {
                let url = format!("{}/v1/experts/multi-layer-batch-q8k", self.config.url);
                let resp = client
                    .post(&url)
                    .header("Content-Type", MULTI_LAYER_BATCH_Q8K_CONTENT_TYPE)
                    .header("Accept", MULTI_LAYER_BATCH_CONTENT_TYPE)
                    .body(body)
                    .send()
                    .map_err(|e| RemoteMoeError::Unreachable {
                        url: url.clone(),
                        cause: e.to_string(),
                    })?;
                if !resp.status().is_success() {
                    return Err(RemoteMoeError::ServerError {
                        status: resp.status().as_u16(),
                        body: resp.text().unwrap_or_default(),
                    });
                }
                let bytes = resp
                    .bytes()
                    .map_err(|e| RemoteMoeError::BadResponse(e.to_string()))?;
                decode_multi_layer_response(&bytes).ok_or_else(|| {
                    RemoteMoeError::BadResponse("multi-layer-batch-q8k response truncated".into())
                })
            }
            ShardTransport::Uds(uds) => {
                let resp_bytes = uds_call(
                    uds,
                    "/v1/experts/multi-layer-batch-q8k",
                    MULTI_LAYER_BATCH_Q8K_CONTENT_TYPE,
                    &body,
                )?;
                decode_multi_layer_response(&resp_bytes).ok_or_else(|| {
                    RemoteMoeError::BadResponse(
                        "UDS multi-layer-batch-q8k response truncated".into(),
                    )
                })
            }
            ShardTransport::Grpc(_) => Err(RemoteMoeError::Client(
                "call_multi_layer_batch_q8k unavailable for gRPC shards".into(),
            )),
        }
    }
}

// ── UDS HTTP/1.1 helpers ──────────────────────────────────────────────────────
//
// Hand-rolled because reqwest doesn't natively expose UDS, and pulling in
// hyperlocal + hyper for one request type would be heavier than the wire
// protocol itself.  We control both ends so framing is fixed:
//
//   POST <path> HTTP/1.1\r\n
//   Host: localhost\r\n
//   Content-Type: <ct>\r\n
//   Content-Length: <N>\r\n
//   Connection: keep-alive\r\n
//   \r\n
//   <body bytes>
//
// Response:
//   HTTP/1.1 200 OK\r\n
//   Content-Type: <ct>\r\n
//   Content-Length: <M>\r\n
//   ...other headers...
//   \r\n
//   <body bytes>
//
// Connections are persistent and reused across calls (the server's axum
// hyper accept loop honours keep-alive by default).

/// Send a single POST + read the response body via the persistent UDS
/// stream.  Reconnects on broken-pipe / read errors.
fn uds_call(
    uds: &UdsState,
    path: &str,
    content_type: &str,
    body: &[u8],
) -> Result<Vec<u8>, RemoteMoeError> {
    use std::io::{Read, Write};

    let mut guard = uds
        .stream
        .lock()
        .map_err(|_| RemoteMoeError::Client("UDS stream mutex poisoned".into()))?;

    // Try once; on transport error, reconnect and retry once.
    for attempt in 0..2 {
        // Establish the stream lazily / after disconnect.
        if guard.is_none() {
            let s = std::os::unix::net::UnixStream::connect(&uds.path).map_err(|e| {
                RemoteMoeError::Unreachable {
                    url: format!("unix://{}", uds.path.display()),
                    cause: e.to_string(),
                }
            })?;
            *guard = Some(s);
        }
        let stream = guard.as_mut().expect("just populated");

        // Build request header in a small Vec so the kernel sees one syscall
        // for the header (write_vectored could split header/body but for
        // small headers the difference is negligible; the bench result is
        // dominated by the body bytes).
        let mut req = Vec::with_capacity(160 + body.len());
        req.extend_from_slice(b"POST ");
        req.extend_from_slice(path.as_bytes());
        req.extend_from_slice(b" HTTP/1.1\r\n");
        req.extend_from_slice(b"Host: localhost\r\n");
        req.extend_from_slice(b"Content-Type: ");
        req.extend_from_slice(content_type.as_bytes());
        req.extend_from_slice(b"\r\n");
        req.extend_from_slice(format!("Content-Length: {}\r\n", body.len()).as_bytes());
        req.extend_from_slice(b"Connection: keep-alive\r\n\r\n");
        req.extend_from_slice(body);

        // Send request.
        if let Err(e) = stream.write_all(&req) {
            if attempt == 0 {
                *guard = None; // force reconnect
                continue;
            }
            return Err(RemoteMoeError::Unreachable {
                url: format!("unix://{}", uds.path.display()),
                cause: format!("write: {e}"),
            });
        }

        // Read response: parse headers, find Content-Length, then read N bytes.
        let mut buf = Vec::with_capacity(8 * 1024);
        let mut tmp = [0u8; 4096];
        let body_start;
        let content_length;
        loop {
            match stream.read(&mut tmp) {
                Ok(0) => {
                    // Server closed; reconnect on next attempt.
                    if attempt == 0 {
                        *guard = None;
                    }
                    return Err(RemoteMoeError::BadResponse(
                        "UDS server closed connection mid-response".into(),
                    ));
                }
                Ok(n) => buf.extend_from_slice(&tmp[..n]),
                Err(e) => {
                    if attempt == 0 {
                        *guard = None;
                    }
                    return Err(RemoteMoeError::BadResponse(format!("UDS read: {e}")));
                }
            }
            // Look for end-of-headers (\r\n\r\n).
            if let Some(idx) = find_header_end(&buf) {
                body_start = idx + 4;
                content_length = parse_content_length(&buf[..idx])?;
                break;
            }
            if buf.len() > 64 * 1024 {
                return Err(RemoteMoeError::BadResponse(
                    "UDS response headers exceed 64 KB — refusing to read further".into(),
                ));
            }
        }

        // Check status line — first 12 bytes are "HTTP/1.1 XXX".
        if buf.len() < 12 || &buf[..9] != b"HTTP/1.1 " {
            return Err(RemoteMoeError::BadResponse(
                "UDS response missing HTTP/1.1 status line".into(),
            ));
        }
        let status = std::str::from_utf8(&buf[9..12])
            .ok()
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(0);
        if !(200..300).contains(&status) {
            // Read body for the error message but cap to keep memory bounded.
            let body_end = (body_start + content_length).min(buf.len());
            let body_slice = &buf[body_start..body_end];
            return Err(RemoteMoeError::ServerError {
                status,
                body: String::from_utf8_lossy(body_slice).into_owned(),
            });
        }

        // Read remaining body bytes.
        let already_have = buf.len() - body_start;
        if already_have < content_length {
            let mut body_buf = vec![0u8; content_length - already_have];
            if let Err(e) = stream.read_exact(&mut body_buf) {
                return Err(RemoteMoeError::BadResponse(format!("UDS body read: {e}")));
            }
            buf.extend_from_slice(&body_buf);
        }

        return Ok(buf[body_start..body_start + content_length].to_vec());
    }
    Err(RemoteMoeError::Client("UDS retry exhausted".into()))
}

fn find_header_end(buf: &[u8]) -> Option<usize> {
    if buf.len() < 4 {
        return None;
    }
    for i in 0..=buf.len() - 4 {
        if &buf[i..i + 4] == b"\r\n\r\n" {
            return Some(i);
        }
    }
    None
}

fn parse_content_length(headers: &[u8]) -> Result<usize, RemoteMoeError> {
    // Headers look like:
    //   HTTP/1.1 200 OK\r\nContent-Type: ...\r\nContent-Length: 11264\r\n
    // Search case-insensitively for "content-length:".
    let lower = headers
        .iter()
        .map(|&b| b.to_ascii_lowercase())
        .collect::<Vec<u8>>();
    let needle = b"content-length:";
    let pos = lower
        .windows(needle.len())
        .position(|w| w == needle)
        .ok_or_else(|| {
            RemoteMoeError::BadResponse("UDS response missing Content-Length header".into())
        })?;
    let mut start = pos + needle.len();
    while start < headers.len() && (headers[start] == b' ' || headers[start] == b'\t') {
        start += 1;
    }
    let mut end = start;
    while end < headers.len() && headers[end].is_ascii_digit() {
        end += 1;
    }
    let s = std::str::from_utf8(&headers[start..end])
        .map_err(|_| RemoteMoeError::BadResponse("UDS Content-Length value not UTF-8".into()))?;
    s.parse::<usize>()
        .map_err(|_| RemoteMoeError::BadResponse(format!("UDS Content-Length not a number: {s:?}")))
}
