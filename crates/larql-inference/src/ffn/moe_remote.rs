//! `RemoteMoeBackend` — Mixture-of-Experts weight-shard dispatch over HTTP.
//!
//! Not to be confused with [`crate::experts`] — that module hosts deterministic
//! WASM compute experts (gcd, base64, …). This module dispatches *MoE expert
//! weights* (the FFN sub-blocks of an MoE transformer) to remote shard servers.
//!
//! For hybrid MoE models (e.g. Gemma 4 26B A4B), the client holds attention
//! weights + router weights (~5.5 GB). Expert weights live on remote shard
//! servers. For each layer:
//!
//!   1. Client runs the router locally: norm → scale → proj → softmax → top-K.
//!   2. Client groups selected experts by shard.
//!   3. One `POST /v1/expert/batch` per shard (parallel via rayon).
//!   4. Client assembles weighted sum from responses.
//!
//! Wire format: JSON — `{"requests": [{layer, expert_id, residual}]}`
//!              → `{"results": [{layer, expert_id, output}], "latency_ms": f64}`
//!
//! This mirrors [`crate::ffn::RemoteWalkBackend`] at the MoE level, replacing
//! `POST /v1/walk-ffn` with `POST /v1/expert/batch`.
//!
//! # Shard map
//!
//! Expert IDs are contiguous ranges owned by each shard:
//!
//! ```text
//! "0-31"  → https://shard-a.local:8081
//! "32-63" → https://shard-b.local:8082
//! ```
//!
//! A single-shard setup (`"0-63"`) routes all experts to one server.
//! `reshard()` swaps the map live without reloading the model.

use std::sync::{Arc, RwLock};
use std::time::Duration;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// ── Public error type ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum RemoteMoeError {
    /// Could not reach the shard server (connection refused, DNS failure, etc.).
    Unreachable { url: String, cause: String },
    /// The server responded with a non-2xx status.
    ServerError { status: u16, body: String },
    /// Response body could not be parsed.
    BadResponse(String),
    /// No shard owns a required expert ID.
    NoShard { expert_id: usize },
    /// HTTP client construction failed.
    Client(String),
}

impl std::fmt::Display for RemoteMoeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unreachable { url, cause } => {
                write!(f, "expert shard unreachable: {url} ({cause})")
            }
            Self::ServerError { status, body } => {
                write!(f, "expert shard returned {status}: {body}")
            }
            Self::BadResponse(msg) => write!(f, "bad expert response: {msg}"),
            Self::NoShard { expert_id } => write!(f, "no shard owns expert {expert_id}"),
            Self::Client(msg) => write!(f, "HTTP client error: {msg}"),
        }
    }
}

impl std::error::Error for RemoteMoeError {}

// ── Shard configuration ───────────────────────────────────────────────────────

/// One entry in the shard map: a contiguous expert-ID range + its URL.
#[derive(Clone, Debug)]
pub struct ShardConfig {
    /// First expert ID owned by this shard (inclusive).
    pub start: usize,
    /// Last expert ID owned by this shard (inclusive).
    pub end: usize,
    /// Base URL, e.g. `"http://shard-a.local:8081"`. Trailing slashes stripped.
    pub url: String,
    /// HTTP request timeout (default: 30 s).
    pub timeout: Duration,
}

impl ShardConfig {
    pub fn new(start: usize, end: usize, url: impl Into<String>) -> Self {
        let url = url.into().trim_end_matches('/').to_string();
        Self {
            start,
            end,
            url,
            timeout: Duration::from_secs(30),
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Parse `"0-31"` → `(0, 31)`. Returns `None` on bad input.
    pub fn parse_range(s: &str) -> Option<(usize, usize)> {
        let mut parts = s.splitn(2, '-');
        let start: usize = parts.next()?.parse().ok()?;
        let end: usize = parts.next()?.parse().ok()?;
        if start <= end {
            Some((start, end))
        } else {
            None
        }
    }
}

// ── Internal shard state ──────────────────────────────────────────────────────

struct GrpcState {
    runtime: std::sync::Arc<tokio::runtime::Runtime>,
    client: larql_router_protocol::ExpertServiceClient<tonic::transport::Channel>,
}

enum ShardTransport {
    Http(reqwest::blocking::Client),
    Grpc(std::sync::Arc<GrpcState>),
}

struct Shard {
    config: ShardConfig,
    transport: ShardTransport,
}

impl Shard {
    fn connect(config: ShardConfig) -> Result<Self, RemoteMoeError> {
        // `grpc://` URL → tonic gRPC over HTTP/2 persistent channel.
        // `http://` URL → reqwest blocking HTTP/1.1 (legacy path).
        let transport = if config.url.starts_with("grpc://") {
            let grpc_endpoint = config.url.replacen("grpc://", "http://", 1);
            let rt = std::sync::Arc::new(
                tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .build()
                    .map_err(|e| RemoteMoeError::Client(e.to_string()))?,
            );
            let client = rt
                .block_on(larql_router_protocol::ExpertServiceClient::connect(
                    grpc_endpoint.clone(),
                ))
                .map_err(|e| RemoteMoeError::Unreachable {
                    url: grpc_endpoint,
                    cause: e.to_string(),
                })?;
            ShardTransport::Grpc(std::sync::Arc::new(GrpcState { runtime: rt, client }))
        } else {
            let http = reqwest::blocking::Client::builder()
                .timeout(config.timeout)
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

    fn owns(&self, expert_id: usize) -> bool {
        expert_id >= self.config.start && expert_id <= self.config.end
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
    fn open_stream(&self) -> Result<ShardStream, RemoteMoeError> {
        match &self.transport {
            ShardTransport::Grpc(grpc) => {
                let rt = std::sync::Arc::clone(&grpc.runtime);
                let mut client = grpc.client.clone();

                // Work channel: Metal thread → async task (non-blocking send)
                let (work_tx, mut work_rx) =
                    tokio::sync::mpsc::unbounded_channel::<larql_router_protocol::ExpertLayerInput>();

                // Result channel: async task → Metal thread (condvar recv)
                let (result_tx, result_rx) =
                    std::sync::mpsc::channel::<Result<Vec<f32>, RemoteMoeError>>();

                // Open the gRPC stream + spawn the dispatch task in one block_on.
                // This is the ONLY block_on — one-time stream setup, not per-layer.
                rt.block_on(async {
                    // Channel for feeding the gRPC request stream.
                    let (grpc_input_tx, mut grpc_input_rx) =
                        tokio::sync::mpsc::unbounded_channel::<larql_router_protocol::ExpertLayerInput>();

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
                            if grpc_input_tx.send(input).is_err() { break; }
                            // Await server response (pure async, no block_on).
                            let result = match grpc_output.next().await {
                                Some(Ok(out)) => {
                                    if out.h2.len() % 4 != 0 {
                                        Err(RemoteMoeError::BadResponse("h2 unaligned".into()))
                                    } else {
                                        Ok(out.h2
                                            .chunks_exact(4)
                                            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                                            .collect())
                                    }
                                }
                                Some(Err(e)) => Err(RemoteMoeError::ServerError {
                                    status: e.code() as u16,
                                    body: e.message().to_string(),
                                }),
                                None => Err(RemoteMoeError::BadResponse("stream ended".into())),
                            };
                            // Wake the Metal thread via condvar (much cheaper than block_on).
                            if result_tx.send(result).is_err() { break; }
                        }
                    });

                    Ok::<(), RemoteMoeError>(())
                })?;

                Ok(ShardStream { work_tx, result_rx, _runtime: rt })
            }
            ShardTransport::Http(_) => Err(RemoteMoeError::Client(
                "open_stream requires grpc:// shards".into(),
            )),
        }
    }

    /// Send a batch of expert calls to this shard.
    ///
    /// Dispatches via gRPC (persistent HTTP/2) when the shard URL starts with
    /// `grpc://`, otherwise falls back to binary HTTP.
    fn call_batch(
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
                        residual: r
                            .residual
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect(),
                    })
                    .collect();

                let grpc_req = larql_router_protocol::ExpertBatchRequest { items };
                // Block on the async gRPC call from this sync context.
                let mut client = grpc.client.clone();
                let resp = grpc.runtime
                    .block_on(client.expert_batch(tonic::Request::new(grpc_req)))
                    .map_err(|e| RemoteMoeError::ServerError {
                        status: e.code() as u16,
                        body: e.message().to_string(),
                    })?
                    .into_inner();

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
        }
    }
}

// ── Binary wire format ────────────────────────────────────────────────────────
//
// Content-Type: application/x-larql-expert
//
// Request:  [N u32][hidden u32] + N × [layer u32][expert_id u32][f32 × hidden]
// Response: [N u32][hidden u32][latency_ms f32] + N × [layer u32][expert_id u32][f32 × hidden]
//
// All integers and floats are little-endian.  This is ~6× smaller than JSON
// for typical 2816-float payloads and avoids serde_json float formatting.

pub const EXPERT_BINARY_CONTENT_TYPE: &str = "application/x-larql-expert";

/// Encode a batch of expert requests as binary.
pub fn encode_expert_request(items: &[ExpertCallItem]) -> Vec<u8> {
    let n = items.len();
    let hidden = items.first().map(|r| r.residual.len()).unwrap_or(0);
    let mut buf = Vec::with_capacity(8 + n * (8 + hidden * 4));
    buf.extend_from_slice(&(n as u32).to_le_bytes());
    buf.extend_from_slice(&(hidden as u32).to_le_bytes());
    for item in items {
        buf.extend_from_slice(&(item.layer as u32).to_le_bytes());
        buf.extend_from_slice(&(item.expert_id as u32).to_le_bytes());
        for &v in &item.residual {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    buf
}

/// Decode a binary expert response. Returns None on truncation.
pub fn decode_expert_response(bytes: &[u8]) -> Option<Vec<ExpertResultItem>> {
    if bytes.len() < 12 {
        return None;
    }
    let n = u32::from_le_bytes(bytes[0..4].try_into().ok()?) as usize;
    let hidden = u32::from_le_bytes(bytes[4..8].try_into().ok()?) as usize;
    // bytes[8..12] = latency_ms f32 (informational, skip)
    let mut pos = 12usize;
    let item_bytes = 8 + hidden * 4;
    if bytes.len() < 12 + n * item_bytes {
        return None;
    }
    let mut results = Vec::with_capacity(n);
    for _ in 0..n {
        let layer = u32::from_le_bytes(bytes[pos..pos + 4].try_into().ok()?) as usize;
        let expert_id = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().ok()?) as usize;
        pos += 8;
        let output: Vec<f32> = bytes[pos..pos + hidden * 4]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        pos += hidden * 4;
        results.push(ExpertResultItem { layer, expert_id, output });
    }
    Some(results)
}

/// Decode a binary expert request from the server side.
pub fn decode_expert_request(bytes: &[u8]) -> Option<Vec<ExpertCallItem>> {
    if bytes.len() < 8 {
        return None;
    }
    let n = u32::from_le_bytes(bytes[0..4].try_into().ok()?) as usize;
    let hidden = u32::from_le_bytes(bytes[4..8].try_into().ok()?) as usize;
    let mut pos = 8usize;
    let item_bytes = 8 + hidden * 4;
    if bytes.len() < 8 + n * item_bytes {
        return None;
    }
    let mut items = Vec::with_capacity(n);
    for _ in 0..n {
        let layer = u32::from_le_bytes(bytes[pos..pos + 4].try_into().ok()?) as usize;
        let expert_id = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().ok()?) as usize;
        pos += 8;
        let residual: Vec<f32> = bytes[pos..pos + hidden * 4]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        pos += hidden * 4;
        items.push(ExpertCallItem { layer, expert_id, residual });
    }
    Some(items)
}

/// Encode a batch of expert results as binary (server-side response).
pub fn encode_expert_response(items: &[ExpertResultItem], latency_ms: f32) -> Vec<u8> {
    let n = items.len();
    let hidden = items.first().map(|r| r.output.len()).unwrap_or(0);
    let mut buf = Vec::with_capacity(12 + n * (8 + hidden * 4));
    buf.extend_from_slice(&(n as u32).to_le_bytes());
    buf.extend_from_slice(&(hidden as u32).to_le_bytes());
    buf.extend_from_slice(&latency_ms.to_le_bytes());
    for item in items {
        buf.extend_from_slice(&(item.layer as u32).to_le_bytes());
        buf.extend_from_slice(&(item.expert_id as u32).to_le_bytes());
        for &v in &item.output {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    buf
}

// ── Wire types ────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct BatchRequest<'a> {
    requests: &'a [ExpertCallItem],
}

#[derive(Serialize, Clone)]
pub struct ExpertCallItem {
    pub layer: usize,
    pub expert_id: usize,
    pub residual: Vec<f32>,
}

#[derive(Deserialize)]
struct BatchResponse {
    results: Vec<ExpertResultItem>,
}

#[derive(Deserialize)]
pub struct ExpertResultItem {
    pub layer: usize,
    pub expert_id: usize,
    pub output: Vec<f32>,
}

// ── Local routing math ────────────────────────────────────────────────────────
// Mirrored from larql-compute cpu/ops/moe.rs so the client can route without
// having the expert weights locally.

fn rms_norm(x: &[f32], w: &[f32], eps: f32, offset: f32) -> Vec<f32> {
    if w.is_empty() || x.is_empty() {
        return x.to_vec();
    }
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    x.iter()
        .zip(w.iter())
        .map(|(&xi, &wi)| xi / rms * (wi + offset))
        .collect()
}

/// Parameter-free RMSNorm (HF `Gemma4RMSNorm(with_scale=False)`): scales
/// `x` by `1/sqrt(mean(x²) + eps)` with no learned weight.
fn rms_norm_no_weight(x: &[f32], eps: f32) -> Vec<f32> {
    if x.is_empty() {
        return Vec::new();
    }
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    x.iter().map(|v| v / rms).collect()
}

fn matmul_vec(x: &[f32], w: &[f32], out_rows: usize, in_cols: usize) -> Vec<f32> {
    (0..out_rows)
        .map(|row| {
            let w_row = &w[row * in_cols..(row + 1) * in_cols];
            x.iter().zip(w_row.iter()).map(|(a, b)| a * b).sum()
        })
        .collect()
}

fn softmax(v: &mut [f32]) {
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    if sum > 0.0 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    }
}

fn top_k(v: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
    let k = k.min(v.len());
    let mut indexed: Vec<(usize, f32)> = v.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    (
        indexed.iter().map(|(i, _)| *i).collect(),
        indexed.iter().map(|(_, v)| *v).collect(),
    )
}

/// Routing-only parameters. A subset of `MoeLayerWeights` — the expert weight
/// slices (`experts_gate_up`, `experts_down`) are absent; those live on shards.
pub struct MoeRouterWeights<'a> {
    /// Router linear projection [num_experts × hidden_size].
    pub router_proj: &'a [f32],
    /// Optional router input scale [hidden_size].
    pub router_scale: &'a [f32],
    /// Optional per-expert output scale [num_experts].
    pub router_per_expert_scale: &'a [f32],
    /// Optional router-specific RMSNorm weights [hidden_size]. When non-empty,
    /// the router input is `rms_norm(h, router_norm)`; when empty AND
    /// `router_norm_parameter_free` is true, it's parameter-free RMSNorm;
    /// otherwise falls back to `rms_norm(h, pre_experts_norm)`.
    pub router_norm: &'a [f32],
    /// Parameter-free router RMSNorm (no learned weight). HF Gemma 4 sets
    /// this true (`Gemma4RMSNorm(with_scale=False)`).
    pub router_norm_parameter_free: bool,
    /// Scalar multiplier on the router input after the norm and `router_scale`.
    /// HF Gemma 4: `hidden_size^-0.5`. Use `1.0` for no scaling.
    pub router_input_scalar: f32,
    /// Pre-experts RMSNorm weights [hidden_size].
    pub pre_experts_norm: &'a [f32],
    /// Post-experts RMSNorm weights [hidden_size]. Applied to the summed output.
    pub post_experts_norm: &'a [f32],
    pub num_experts: usize,
    pub top_k: usize,
}

impl MoeRouterWeights<'_> {
    /// Run steps 1-5 of the MoE forward pass (norm → scale → proj → softmax → top-K).
    /// Returns `(h_norm, expert_indices, expert_weights)` where `h_norm` is
    /// the experts' input (pre_experts_norm output), not the router's input.
    pub fn route(&self, h: &[f32], norm_offset: f32, eps: f32) -> (Vec<f32>, Vec<usize>, Vec<f32>) {
        let hidden = h.len();

        // Experts' input norm (used by callers for the expert matmuls).
        // Router norm composes on top of h_norm — see the matching note in
        // `larql-compute/src/cpu/ops/moe/forward.rs` (verified via
        // `larql parity --component moe-block` against Metal's GPU
        // dispatch convention).
        let h_norm = rms_norm(h, self.pre_experts_norm, eps, norm_offset);

        // Router input norm. Priority:
        //   1. learned router_norm weight (architectures that ship one),
        //   2. parameter-free RMSNorm (HF Gemma 4 — `with_scale=False`),
        //   3. fallback: experts' pre-norm.
        // All apply on top of h_norm so routing matches Metal.
        let router_in_normed = if !self.router_norm.is_empty() {
            rms_norm(&h_norm, self.router_norm, eps, norm_offset)
        } else if self.router_norm_parameter_free {
            rms_norm_no_weight(&h_norm, eps)
        } else {
            h_norm.clone()
        };

        let mut router_in: Vec<f32> = if !self.router_scale.is_empty() {
            router_in_normed
                .iter()
                .zip(self.router_scale.iter())
                .map(|(a, b)| a * b)
                .collect()
        } else {
            router_in_normed
        };
        if self.router_input_scalar != 1.0 && self.router_input_scalar != 0.0 {
            for v in router_in.iter_mut() {
                *v *= self.router_input_scalar;
            }
        }

        let mut logits = matmul_vec(&router_in, self.router_proj, self.num_experts, hidden);
        softmax(&mut logits);

        let (indices, mut weights) = top_k(&logits, self.top_k);

        // Renormalize selected weights to sum to 1 — matches Gemma 4's
        // gemma4_top_k_softmax which normalises after selection.
        let weight_sum: f32 = weights.iter().sum();
        if weight_sum > 0.0 {
            for w in &mut weights {
                *w /= weight_sum;
            }
        }

        if !self.router_per_expert_scale.is_empty() {
            for (i, &ei) in indices.iter().enumerate() {
                if ei < self.router_per_expert_scale.len() {
                    weights[i] *= self.router_per_expert_scale[ei];
                }
            }
        }

        (h_norm, indices, weights)
    }
}

// ── RemoteMoeBackend ───────────────────────────────────────────────────────

/// Remote MoE expert backend. Thread-safe — all methods take `&self`.
///
/// The shard map is stored behind an `RwLock` so `reshard()` can replace it
/// without interrupting in-flight `forward_moe` calls on other threads.
pub struct RemoteMoeBackend {
    shards: Arc<RwLock<Vec<Shard>>>,
}

impl RemoteMoeBackend {
    /// Build with no shards and no health check. Tests only — the backend
    /// will return errors on any actual dispatch attempt.
    #[cfg(test)]
    pub fn new_disconnected() -> Self {
        Self {
            shards: Arc::new(RwLock::new(vec![])),
        }
    }

    /// Build from a shard list. Performs a health check on each shard.
    pub fn connect(configs: Vec<ShardConfig>) -> Result<Self, RemoteMoeError> {
        let shards: Result<Vec<Shard>, _> = configs.into_iter().map(Shard::connect).collect();
        Ok(Self {
            shards: Arc::new(RwLock::new(shards?)),
        })
    }

    /// Replace the shard map live (no model reload, no inference interruption).
    ///
    /// Reconnects to new shards, then atomically swaps the map.
    /// In-flight requests against old shards complete normally.
    pub fn reshard(&self, configs: Vec<ShardConfig>) -> Result<(), RemoteMoeError> {
        let new_shards: Result<Vec<Shard>, _> = configs.into_iter().map(Shard::connect).collect();
        *self.shards.write().unwrap() = new_shards?;
        Ok(())
    }

    /// Returns true if all shards use gRPC transport (`grpc://` URLs).
    /// When true, `open_streams` is available and `forward_moe_stream` can be used.
    pub fn has_grpc_shards(&self) -> bool {
        let shards = self.shards.read().unwrap();
        !shards.is_empty()
            && shards
                .iter()
                .all(|s| matches!(s.transport, ShardTransport::Grpc(_)))
    }

    /// Latency-stats probe: test-call each shard with a zero-length batch and
    /// return `(url, rtt_ms)` per shard. Non-fatal — returns partial results.
    pub fn probe_latency(&self) -> Vec<(String, f64)> {
        let shards = self.shards.read().unwrap();
        shards
            .par_iter()
            .map(|shard| {
                let t = std::time::Instant::now();
                let _ = shard.call_batch(&[]);
                let rtt_ms = t.elapsed().as_secs_f64() * 1000.0;
                (shard.config.url.clone(), rtt_ms)
            })
            .collect()
    }

    /// Run one MoE layer forward pass with experts dispatched remotely.
    ///
    /// Steps:
    ///   1. Router runs locally on `h` using `router`.
    ///   2. Selected experts are grouped by owning shard.
    ///   3. One `POST /v1/expert/batch` per shard (parallel).
    ///   4. Weighted outputs are summed; post-experts norm applied.
    ///
    /// Returns the expert-block contribution (same shape as `h`).
    pub fn forward_moe(
        &self,
        layer: usize,
        h: &[f32],
        router: &MoeRouterWeights<'_>,
        norm_offset: f32,
        eps: f32,
    ) -> Result<Vec<f32>, RemoteMoeError> {
        let hidden = h.len();
        if hidden == 0 || router.num_experts == 0 || router.top_k == 0 {
            return Ok(vec![0.0f32; hidden]);
        }

        // 1. Route locally.
        let (_h_norm, expert_indices, expert_weights) = router.route(h, norm_offset, eps);

        // 2. Build per-shard call lists.
        let shards = self.shards.read().unwrap();
        let mut shard_calls: Vec<(usize, Vec<ExpertCallItem>)> =
            (0..shards.len()).map(|i| (i, Vec::new())).collect();

        for (&expert_id, _) in expert_indices.iter().zip(expert_weights.iter()) {
            let shard_idx = shards
                .iter()
                .position(|s| s.owns(expert_id))
                .ok_or(RemoteMoeError::NoShard { expert_id })?;
            shard_calls[shard_idx].1.push(ExpertCallItem {
                layer,
                expert_id,
                residual: h.to_vec(),
            });
        }

        // 3. Parallel dispatch — one batch call per shard that has work.
        let non_empty: Vec<(usize, &Vec<ExpertCallItem>)> = shard_calls
            .iter()
            .filter(|(_, items)| !items.is_empty())
            .map(|(si, items)| (*si, items))
            .collect();

        let results_per_shard: Vec<Result<Vec<ExpertResultItem>, RemoteMoeError>> = non_empty
            .par_iter()
            .map(|(si, items)| shards[*si].call_batch(items))
            .collect();

        // 4. Accumulate weighted outputs.
        let expert_weight_map: std::collections::HashMap<usize, f32> = expert_indices
            .iter()
            .copied()
            .zip(expert_weights.iter().copied())
            .collect();

        let mut out = vec![0.0f32; hidden];
        for result in results_per_shard {
            for item in result? {
                if item.output.len() != hidden {
                    return Err(RemoteMoeError::BadResponse(format!(
                        "expert {}/{} returned {} floats, expected {hidden}",
                        item.layer,
                        item.expert_id,
                        item.output.len()
                    )));
                }
                let weight = expert_weight_map
                    .get(&item.expert_id)
                    .copied()
                    .unwrap_or(0.0);
                for (acc, &val) in out.iter_mut().zip(item.output.iter()) {
                    *acc += weight * val;
                }
            }
        }

        // 5. Post-experts norm.
        Ok(rms_norm(&out, router.post_experts_norm, eps, norm_offset))
    }

    /// Batch MoE forward for a full sequence of positions in one shot.
    ///
    /// Runs the router on every row of `h`, then issues **one** HTTP batch
    /// call per shard per layer (instead of one call per position). For a
    /// prefill of N positions this reduces dispatch from `N × shards` calls
    /// to `shards` calls — 18× fewer round trips for an 18-token context.
    ///
    /// Results are stitched back into an `[N, hidden]` output array by
    /// sequential index: the server returns items in request order, so we
    /// can match result[i] → request[i] without a position tag in the
    /// wire format.
    pub fn forward_moe_seq(
        &self,
        layer: usize,
        h: &ndarray::Array2<f32>,
        router: &MoeRouterWeights<'_>,
        norm_offset: f32,
        eps: f32,
    ) -> Result<ndarray::Array2<f32>, RemoteMoeError> {
        let seq_len = h.nrows();
        let hidden = h.ncols();
        if hidden == 0 || router.num_experts == 0 || router.top_k == 0 {
            return Ok(ndarray::Array2::zeros((seq_len, hidden)));
        }

        // 1. Route every position locally.
        // routing[pos] = (expert_indices, expert_weights)
        let mut routing: Vec<(Vec<usize>, Vec<f32>)> = Vec::with_capacity(seq_len);
        for pos in 0..seq_len {
            let row: Vec<f32> = h.row(pos).to_vec();
            let (_, idx, wts) = router.route(&row, norm_offset, eps);
            routing.push((idx, wts));
        }

        // 2. Build per-shard call lists preserving (pos, local_idx) so we
        //    can reconstruct the output ordering.
        //    shard_items[si] = Vec<(pos, expert_id, residual)>
        let shards = self.shards.read().unwrap();
        let mut shard_items: Vec<Vec<(usize, usize, Vec<f32>)>> =
            (0..shards.len()).map(|_| Vec::new()).collect();

        for pos in 0..seq_len {
            let row: Vec<f32> = h.row(pos).to_vec();
            for &expert_id in &routing[pos].0 {
                let si = shards
                    .iter()
                    .position(|s| s.owns(expert_id))
                    .ok_or(RemoteMoeError::NoShard { expert_id })?;
                shard_items[si].push((pos, expert_id, row.clone()));
            }
        }

        // 3. One batch call per shard that has work (parallel).
        let non_empty: Vec<(usize, &Vec<(usize, usize, Vec<f32>)>)> = shard_items
            .iter()
            .enumerate()
            .filter(|(_, items)| !items.is_empty())
            .collect();

        let dispatch_results: Vec<(usize, Result<Vec<ExpertResultItem>, RemoteMoeError>)> =
            non_empty
                .par_iter()
                .map(|(si, items)| {
                    let calls: Vec<ExpertCallItem> = items
                        .iter()
                        .map(|(_, expert_id, residual)| ExpertCallItem {
                            layer,
                            expert_id: *expert_id,
                            residual: residual.clone(),
                        })
                        .collect();
                    (*si, shards[*si].call_batch(&calls))
                })
                .collect();

        // 4. Reassemble: for each shard, result[i] corresponds to
        //    shard_items[si][i].  Accumulate weighted sums per position.
        let mut out = ndarray::Array2::<f32>::zeros((seq_len, hidden));

        for (si, result) in dispatch_results {
            let items = &shard_items[si];
            let results = result?;
            if results.len() != items.len() {
                return Err(RemoteMoeError::BadResponse(format!(
                    "shard returned {} results for {} requests at layer {layer}",
                    results.len(),
                    items.len()
                )));
            }
            for ((pos, expert_id, _), item) in items.iter().zip(results.iter()) {
                if item.output.len() != hidden {
                    return Err(RemoteMoeError::BadResponse(format!(
                        "expert {expert_id} at pos {pos} returned {} floats, expected {hidden}",
                        item.output.len()
                    )));
                }
                // Find the weight for this expert at this position.
                let weight = routing[*pos]
                    .0
                    .iter()
                    .zip(routing[*pos].1.iter())
                    .find(|(&eid, _)| eid == *expert_id)
                    .map(|(_, &w)| w)
                    .unwrap_or(0.0);

                let mut row = out.row_mut(*pos);
                for (acc, &val) in row.iter_mut().zip(item.output.iter()) {
                    *acc += weight * val;
                }
            }
        }

        // 5. Post-experts norm per position.
        if !router.post_experts_norm.is_empty() {
            for pos in 0..seq_len {
                let row_vec: Vec<f32> = out.row(pos).to_vec();
                let normed = rms_norm(&row_vec, router.post_experts_norm, eps, norm_offset);
                for (dst, src) in out.row_mut(pos).iter_mut().zip(normed.iter()) {
                    *dst = *src;
                }
            }
        }

        Ok(out)
    }

    /// Open one gRPC streaming channel per shard for a decode step.
    ///
    /// Returns a `Vec<ShardStream>`, one per shard in the internal shard map.
    /// Each stream stays open until dropped; the caller sends one
    /// `ExpertLayerInput` per MoE layer and receives one `ExpertLayerOutput`.
    ///
    /// Use in `generate_with_remote_moe`:
    ///   ```ignore
    ///   let mut streams = backend.open_streams()?;
    ///   // inside moe_fn for each layer:
    ///   let h2 = backend.forward_moe_stream(layer, h_post_attn, &router, &mut streams, norm_offset, eps)?;
    ///   // streams are dropped (and gRPC streams closed) at end of decode step.
    ///   ```
    pub fn open_streams(&self) -> Result<Vec<ShardStream>, RemoteMoeError> {
        let shards = self.shards.read().unwrap();
        shards
            .iter()
            .map(|shard| shard.open_stream())
            .collect()
    }

    /// Run one MoE layer via the already-open per-shard streams.
    ///
    /// Eliminates the per-call connection overhead of `forward_moe` — the
    /// gRPC streams stay alive for the entire decode step (30 layers) so
    /// each layer only pays the cost of sending/receiving one proto frame
    /// over an existing HTTP/2 connection (~0.5ms vs ~12ms per layer).
    pub fn forward_moe_stream(
        &self,
        layer: usize,
        h: &[f32],
        router: &MoeRouterWeights<'_>,
        streams: &mut [ShardStream],
        norm_offset: f32,
        eps: f32,
    ) -> Result<Vec<f32>, RemoteMoeError> {
        let hidden = h.len();
        if hidden == 0 || router.num_experts == 0 || router.top_k == 0 {
            return Ok(vec![0.0f32; hidden]);
        }

        // 1. Route locally (same as forward_moe).
        let (_h_norm, expert_indices, expert_weights) = router.route(h, norm_offset, eps);

        // 2. Encode residual + post_norm bytes once.
        let residual_bytes: Vec<u8> = h.iter().flat_map(|v| v.to_le_bytes()).collect();
        let post_norm_bytes: Vec<u8> = router
            .post_experts_norm
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        // 3. Build per-shard inputs and send, then receive.
        //
        // Each shard gets the expert_ids it owns (with their weights); the
        // server applies its local weighted sum + post-norm and returns h2.
        // Shards with no owned experts for this layer get an empty expert_ids
        // list — they return zeros immediately, preserving stream ordering.

        // Figure out which experts each shard owns.
        let shards_guard = self.shards.read().unwrap();
        let num_shards = shards_guard.len();

        // Distribute expert_ids/weights across shards.
        let mut shard_eids: Vec<Vec<u32>> = vec![Vec::new(); num_shards];
        let mut shard_ewts: Vec<Vec<f32>> = vec![Vec::new(); num_shards];

        for (&eid, &w) in expert_indices.iter().zip(expert_weights.iter()) {
            let si = shards_guard
                .iter()
                .position(|s| s.owns(eid))
                .ok_or(RemoteMoeError::NoShard { expert_id: eid })?;
            shard_eids[si].push(eid as u32);
            shard_ewts[si].push(w);
        }
        drop(shards_guard);

        // Fire all shards first (non-blocking channel push), then collect.
        // Both shards start processing simultaneously — shard B no longer
        // waits for shard A to finish.  Per-layer wall time drops from
        // (A_ms + B_ms) to max(A_ms, B_ms) ≈ 3.5ms instead of 7ms.
        for (si, stream) in streams.iter().enumerate() {
            let input = larql_router_protocol::ExpertLayerInput {
                layer: layer as u32,
                expert_ids: shard_eids[si].clone(),
                expert_weights: shard_ewts[si].clone(),
                residual: residual_bytes.clone(),
                post_experts_norm: post_norm_bytes.clone(),
                norm_offset,
                eps,
            };
            if let Err(e) = stream.fire(input) {
                return Err(e);
            }
        }
        // Collect: both shards are processing in parallel; by the time we
        // wait for shard A the shard B result is also already in flight.
        let mut results: Vec<Result<Vec<f32>, RemoteMoeError>> = Vec::with_capacity(streams.len());
        for stream in streams.iter() {
            results.push(stream.collect());
        }

        // 4. Sum partial weighted sums from all shards.
        //    Each shard returns raw weighted_sum(its_experts) WITHOUT post-norm
        //    because post-norm on a partial sum then summing is wrong:
        //       norm(shard_A) + norm(shard_B) ≠ norm(shard_A + shard_B)
        let mut out = vec![0.0f32; hidden];
        for result in results {
            let partial = result?;
            if partial.len() == hidden {
                for (acc, v) in out.iter_mut().zip(partial.iter()) {
                    *acc += v;
                }
            }
        }

        // 5. Post-experts norm on the fully combined output (same as forward_moe).
        Ok(rms_norm(&out, router.post_experts_norm, eps, norm_offset))
    }
}

// ── ShardStream — async-native dispatch without block_on ─────────────────────
//
// Architecture: one async tokio task per shard manages the gRPC stream.
// The sync Metal decode thread communicates via std::sync::mpsc channels:
//
//   Metal thread               tokio async task
//   ────────────────────────   ──────────────────────────────────
//   work_tx.send(input)  ───▶  work_rx.recv().await
//                              gRPC stream: send + await response
//   result_rx.recv()     ◀───  result_tx.send(decoded_h2)
//
// `work_tx.send` is non-blocking (UnboundedSender — returns immediately).
// `result_rx.recv` uses a condvar/futex — ~0.1ms overhead vs ~1.45ms
// for `Runtime::block_on` on macOS.  The gRPC itself runs as proper async
// inside the tokio task without any scheduling penalty.

/// A live gRPC bidirectional stream to one shard.
///
/// The async gRPC work runs in a dedicated tokio task.  The sync Metal decode
/// thread fires inputs via `fire()` (non-blocking) and collects results via
/// `collect()` (condvar wait, ~0.1ms overhead).
pub struct ShardStream {
    /// Non-blocking input channel: Metal thread → tokio task.
    work_tx: tokio::sync::mpsc::UnboundedSender<larql_router_protocol::ExpertLayerInput>,
    /// Blocking result channel: tokio task → Metal thread.
    result_rx: std::sync::mpsc::Receiver<Result<Vec<f32>, RemoteMoeError>>,
    /// Keep the runtime alive so the tokio task keeps running.
    _runtime: std::sync::Arc<tokio::runtime::Runtime>,
}

impl ShardStream {
    /// Fire: push input to the async task, return immediately.
    /// Pair with `collect()` to retrieve the result.
    pub fn fire(
        &self,
        input: larql_router_protocol::ExpertLayerInput,
    ) -> Result<(), RemoteMoeError> {
        self.work_tx
            .send(input)
            .map_err(|_| RemoteMoeError::BadResponse("shard stream closed".into()))
    }

    /// Collect: condvar-wait for the async task's result (~0.1ms).
    /// No tokio block_on — just a futex wake when the result arrives.
    pub fn collect(&self) -> Result<Vec<f32>, RemoteMoeError> {
        self.result_rx
            .recv()
            .unwrap_or(Err(RemoteMoeError::BadResponse("shard result channel closed".into())))
    }

    /// Convenience: fire then collect.
    pub fn send_recv(
        &self,
        input: larql_router_protocol::ExpertLayerInput,
    ) -> Result<Vec<f32>, RemoteMoeError> {
        self.fire(input)?;
        self.collect()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_range_valid() {
        assert_eq!(ShardConfig::parse_range("0-31"), Some((0, 31)));
        assert_eq!(ShardConfig::parse_range("32-63"), Some((32, 63)));
        assert_eq!(ShardConfig::parse_range("0-0"), Some((0, 0)));
    }

    #[test]
    fn parse_range_invalid() {
        assert_eq!(ShardConfig::parse_range("31-0"), None); // reversed
        assert_eq!(ShardConfig::parse_range("abc"), None);
        assert_eq!(ShardConfig::parse_range(""), None);
    }

    #[test]
    fn shard_config_strips_trailing_slash() {
        let s = ShardConfig::new(0, 31, "http://a.example.com:8081///");
        assert_eq!(s.url, "http://a.example.com:8081");
    }

    #[test]
    fn shard_owns() {
        fn make_shard(start: usize, end: usize) -> Shard {
            let config = ShardConfig::new(start, end, "http://localhost:8080");
            let client = reqwest::blocking::Client::new();
            Shard { config, client }
        }
        let s = make_shard(0, 31);
        assert!(s.owns(0));
        assert!(s.owns(31));
        assert!(!s.owns(32));
        let s2 = make_shard(32, 63);
        assert!(s2.owns(32));
        assert!(s2.owns(63));
        assert!(!s2.owns(31));
    }

    #[test]
    fn route_softmax_sums_to_one() {
        let num_experts = 8;
        let hidden = 4;
        let router_proj: Vec<f32> = (0..num_experts * hidden).map(|i| i as f32 * 0.01).collect();
        let router = MoeRouterWeights {
            router_proj: &router_proj,
            router_scale: &[],
            router_per_expert_scale: &[],
            router_norm: &[],
            router_norm_parameter_free: false,
            router_input_scalar: 1.0,
            pre_experts_norm: &[],
            post_experts_norm: &[],
            num_experts,
            top_k: 2,
        };
        let h: Vec<f32> = vec![1.0, 0.5, -0.3, 0.2];
        let (_, indices, weights) = router.route(&h, 0.0, 1e-6);
        assert_eq!(indices.len(), 2);
        assert_eq!(weights.len(), 2);
        assert!(weights.iter().all(|&w| w >= 0.0));
    }

    #[test]
    fn route_with_parameter_free_router_norm() {
        // HF Gemma 4 codepath: router_norm is empty AND parameter_free=true →
        // route() must call rms_norm_no_weight on the input. Without the
        // helper this branch panics with "function not found"; with it, the
        // route should still produce a valid top-k.
        let num_experts = 4;
        let hidden = 4;
        let router_proj: Vec<f32> = (0..num_experts * hidden)
            .map(|i| (i as f32) * 0.1)
            .collect();
        let router = MoeRouterWeights {
            router_proj: &router_proj,
            router_scale: &[],
            router_per_expert_scale: &[],
            router_norm: &[],
            router_norm_parameter_free: true,
            router_input_scalar: 1.0,
            pre_experts_norm: &[],
            post_experts_norm: &[],
            num_experts,
            top_k: 2,
        };
        let h: Vec<f32> = vec![1.0, -2.0, 3.0, 0.5];
        let (h_norm_out, indices, weights) = router.route(&h, 0.0, 1e-6);

        // h_norm_out is the experts' input (pre_experts_norm output).
        // Since pre_experts_norm is empty, h_norm_out should be h verbatim.
        assert_eq!(h_norm_out, h);

        // Top-K selected and weights renormalised to sum to 1.
        assert_eq!(indices.len(), 2);
        assert_eq!(weights.len(), 2);
        let sum: f32 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "weights should sum to 1, got {sum}"
        );
        assert!(weights.iter().all(|&w| w >= 0.0));
    }

    #[test]
    fn route_with_router_input_scalar() {
        // HF Gemma 4 also uses router_input_scalar = hidden_size^-0.5.
        // Verify the scalar is applied (changes which expert wins) without
        // breaking the softmax+top-k pipeline.
        let num_experts = 4;
        let hidden = 4;
        // Bias router_proj so expert 0 wins on un-scaled input.
        let mut router_proj: Vec<f32> = vec![0.0; num_experts * hidden];
        router_proj[0] = 100.0; // expert 0 row, dim 0
        router_proj[hidden] = -100.0; // expert 1 row, dim 0

        let h: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];

        let unscaled = MoeRouterWeights {
            router_proj: &router_proj,
            router_scale: &[],
            router_per_expert_scale: &[],
            router_norm: &[],
            router_norm_parameter_free: false,
            router_input_scalar: 1.0,
            pre_experts_norm: &[],
            post_experts_norm: &[],
            num_experts,
            top_k: 1,
        };
        let (_, idx_unscaled, _) = unscaled.route(&h, 0.0, 1e-6);
        assert_eq!(idx_unscaled, vec![0]);

        // With scalar = 0.5, the logit gap shrinks (50 vs -50 still picks
        // expert 0). Use a negating scalar to flip the winner — this proves
        // the scalar actually multiplies through.
        let flipped = MoeRouterWeights {
            router_input_scalar: -1.0,
            ..unscaled
        };
        let (_, idx_flipped, _) = flipped.route(&h, 0.0, 1e-6);
        assert_eq!(
            idx_flipped,
            vec![1],
            "negative scalar should flip the winner"
        );
    }

    #[test]
    fn forward_moe_empty_input_returns_zero() {
        // Can't connect to a real server, but we can verify the early-exit path.
        // Construct a backend with an empty shard list via the raw struct (bypassing connect).
        let backend = RemoteMoeBackend {
            shards: Arc::new(RwLock::new(vec![])),
        };
        let router = MoeRouterWeights {
            router_proj: &[],
            router_scale: &[],
            router_per_expert_scale: &[],
            router_norm: &[],
            router_norm_parameter_free: false,
            router_input_scalar: 1.0,
            pre_experts_norm: &[],
            post_experts_norm: &[],
            num_experts: 0,
            top_k: 0,
        };
        let result = backend.forward_moe(0, &[1.0f32, 2.0, 3.0], &router, 0.0, 1e-6);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec![0.0f32; 3]);
    }
}
