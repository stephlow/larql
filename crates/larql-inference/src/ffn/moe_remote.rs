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

/// One entry in the shard map: an expert-ID range + its URL.
///
/// Two ownership modes (mutually exclusive — `unit_set` takes precedence):
///
///   1. **Layer-uniform range** (`start..=end`) — same expert range applies
///      to every layer. Set via [`ShardConfig::new`] or `--moe-shards
///      "0-63=URL,..."`.
///   2. **Per-(layer, expert) set** (`unit_set`) — explicit ownership for
///      fine-grained shards. Set via [`ShardConfig::with_unit_set`] or
///      `--moe-units-manifest PATH`.
///
/// `start`/`end` are still populated in unit-set mode (carrying the
/// min/max expert id across all units) so RTT probes and existing
/// diagnostics keep working without special-casing.
#[derive(Clone, Debug)]
pub struct ShardConfig {
    /// First expert ID this shard touches (inclusive).  When `unit_set` is
    /// `Some`, this is the min of the unit set, kept for diagnostics.
    pub start: usize,
    /// Last expert ID this shard touches (inclusive).  When `unit_set` is
    /// `Some`, this is the max of the unit set.
    pub end: usize,
    /// Base URL, e.g. `"http://shard-a.local:8081"`. Trailing slashes stripped.
    pub url: String,
    /// HTTP request timeout (default: 30 s).
    pub timeout: Duration,
    /// Fine-grained ownership: every `(layer, expert_id)` in this set is
    /// owned by this shard.  When `Some`, takes precedence over the
    /// `start..=end` range.  See `crate::ffn::moe_remote::UnitManifest`
    /// for the JSON shape that produces this set.
    pub unit_set: Option<std::sync::Arc<std::collections::HashSet<(usize, usize)>>>,
}

impl ShardConfig {
    pub fn new(start: usize, end: usize, url: impl Into<String>) -> Self {
        let url = url.into().trim_end_matches('/').to_string();
        Self {
            start,
            end,
            url,
            timeout: Duration::from_secs(30),
            unit_set: None,
        }
    }

    /// Build a shard config that owns an explicit set of `(layer, expert_id)`
    /// pairs.  `start`/`end` are derived from the set's min/max for
    /// diagnostic compatibility; ownership checks use the set itself.
    pub fn with_units(
        url: impl Into<String>,
        units: std::collections::HashSet<(usize, usize)>,
    ) -> Self {
        let url = url.into().trim_end_matches('/').to_string();
        let (start, end) = if units.is_empty() {
            (0, 0)
        } else {
            let min = units.iter().map(|(_, e)| *e).min().unwrap();
            let max = units.iter().map(|(_, e)| *e).max().unwrap();
            (min, max)
        };
        Self {
            start,
            end,
            url,
            timeout: Duration::from_secs(30),
            unit_set: Some(std::sync::Arc::new(units)),
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

// ── Unit manifest (fine-grained shard map) ───────────────────────────────────
//
// Mirrors the server's `--units PATH` JSON shape but augmented with `url`:
//
//   {
//     "shards": [
//       { "url": "grpc://hostA:9081",
//         "layer_experts": {"0": [[0,31]], "1": [[0,15]], "2": [[0,31]]} },
//       { "url": "grpc://hostB:9082",
//         "layer_experts": {"0": [[32,63]], "1": [[16,31],[64,79]]} }
//     ]
//   }
//
// One JSON object → many `ShardConfig`s.  Each shard has its own explicit
// `(layer, expert_id)` ownership set; the client routes per-(layer, expert)
// rather than per-expert.

/// Top-level JSON shape: a list of shards, each with its URL + per-layer
/// expert-range ownership.  Matches the server-side `--units` format
/// extended with `url` so a single manifest can describe the whole grid.
#[derive(serde::Deserialize)]
pub struct UnitManifest {
    pub shards: Vec<UnitShard>,
}

/// One shard's slice of the grid.
#[derive(serde::Deserialize)]
pub struct UnitShard {
    pub url: String,
    /// Per-layer list of inclusive `[start, end]` expert-id ranges.  Layers
    /// absent from the map are not owned by this shard.
    pub layer_experts: std::collections::BTreeMap<String, Vec<[usize; 2]>>,
}

impl UnitShard {
    /// Expand the per-layer ranges into a flat `(layer, expert_id)` set.
    pub fn into_unit_set(
        self,
    ) -> Result<std::collections::HashSet<(usize, usize)>, RemoteMoeError> {
        let mut units = std::collections::HashSet::new();
        for (layer_str, ranges) in self.layer_experts {
            let layer: usize = layer_str.parse().map_err(|_| {
                RemoteMoeError::Client(format!(
                    "unit-manifest: layer key '{layer_str}' is not a valid usize"
                ))
            })?;
            for [start, end] in ranges {
                if end < start {
                    return Err(RemoteMoeError::Client(format!(
                        "unit-manifest: layer {layer}: end ({end}) must be >= start ({start})"
                    )));
                }
                for eid in start..=end {
                    units.insert((layer, eid));
                }
            }
        }
        Ok(units)
    }
}

impl UnitManifest {
    /// Convert the parsed manifest into one `ShardConfig` per shard, each
    /// carrying its explicit `(layer, expert_id)` ownership set.
    pub fn into_shard_configs(self) -> Result<Vec<ShardConfig>, RemoteMoeError> {
        let mut out = Vec::with_capacity(self.shards.len());
        for shard in self.shards {
            let url = shard.url.clone();
            let units = shard.into_unit_set()?;
            out.push(ShardConfig::with_units(url, units));
        }
        Ok(out)
    }
}

/// Parse a unit-manifest JSON file from `path` into ready-to-connect
/// `ShardConfig`s.  Returns `RemoteMoeError::Client` on read or parse
/// failure with the path included so the operator can fix it without
/// grepping logs.
pub fn parse_unit_manifest(path: &std::path::Path) -> Result<Vec<ShardConfig>, RemoteMoeError> {
    let bytes = std::fs::read(path).map_err(|e| {
        RemoteMoeError::Client(format!("unit-manifest: read {}: {e}", path.display()))
    })?;
    let manifest: UnitManifest = serde_json::from_slice(&bytes).map_err(|e| {
        RemoteMoeError::Client(format!("unit-manifest: parse {}: {e}", path.display()))
    })?;
    manifest.into_shard_configs()
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

    /// Layer-uniform ownership check (legacy `--moe-shards "S-E=URL"` path).
    /// Used by routing call sites that don't know the layer — keep returning
    /// `false` for fine-grained shards so the layer-aware `owns_unit` is
    /// always preferred when the layer is in scope.
    fn owns(&self, expert_id: usize) -> bool {
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
    fn owns_unit(&self, layer: usize, expert_id: usize) -> bool {
        if let Some(units) = self.config.unit_set.as_ref() {
            return units.contains(&(layer, expert_id));
        }
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
                                        let h2: Vec<f32> = out.h2
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
                let t_call = std::time::Instant::now();
                let resp = grpc.runtime
                    .block_on(client.expert_batch(tonic::Request::new(grpc_req)))
                    .map_err(|e| RemoteMoeError::ServerError {
                        status: e.code() as u16,
                        body: e.message().to_string(),
                    })?
                    .into_inner();

                eprintln!("[call_batch/grpc] n={} block_on={:.1}ms", requests.len(),
                    t_call.elapsed().as_secs_f64()*1000.0);
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

        // Experts' input norm (used by callers for the expert matmuls only).
        // Per HF Gemma4TextDecoderLayer.forward, router consumes raw h and
        // experts consume pre_experts_norm(h). See the matching note in
        // `larql-compute/src/cpu/ops/moe/forward.rs`.
        let h_norm = rms_norm(h, self.pre_experts_norm, eps, norm_offset);

        // Router input norm — applied to RAW h (not h_norm). Priority:
        //   1. learned router_norm weight (architectures that ship one),
        //   2. parameter-free RMSNorm (HF Gemma 4 — `with_scale=False`),
        //   3. fallback: raw h.
        let router_in_normed = if !self.router_norm.is_empty() {
            rms_norm(h, self.router_norm, eps, norm_offset)
        } else if self.router_norm_parameter_free {
            rms_norm_no_weight(h, eps)
        } else {
            h.to_vec()
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
                .position(|s| s.owns_unit(layer, expert_id))
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
                    .position(|s| s.owns_unit(layer, expert_id))
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
        let inflight = self.forward_moe_stream_fire(layer, h, router, streams, norm_offset, eps)?;
        self.forward_moe_stream_collect(streams, inflight)
    }

    /// Fire half of `forward_moe_stream`: route locally, push one input per
    /// shard onto its async dispatch task, and return immediately.
    ///
    /// Pair with [`Self::forward_moe_stream_collect`] to retrieve the result.
    /// The [`InflightMoe`] handle carries the post-norm context so the caller
    /// does not need to keep the [`MoeRouterWeights`] borrow alive across the
    /// fire/collect boundary.
    ///
    /// Used by the GPU/MoE overlap path: the metal decode loop fires the MoE
    /// call as soon as `h_post_attn` is ready, encodes dense FFN on a fresh
    /// command buffer, and then collects — letting GPU dense FFN run in
    /// parallel with the remote round trip.
    pub fn forward_moe_stream_fire(
        &self,
        layer: usize,
        h: &[f32],
        router: &MoeRouterWeights<'_>,
        streams: &[ShardStream],
        norm_offset: f32,
        eps: f32,
    ) -> Result<InflightMoe, RemoteMoeError> {
        let hidden = h.len();
        if hidden == 0 || router.num_experts == 0 || router.top_k == 0 || streams.is_empty() {
            return Ok(InflightMoe {
                hidden,
                n_streams: 0,
                post_experts_norm: Vec::new(),
                norm_offset,
                eps,
            });
        }

        // 1. Route locally.
        let (_h_norm, expert_indices, expert_weights) = router.route(h, norm_offset, eps);

        // 2. Encode residual + post_norm bytes once.
        let residual_bytes: Vec<u8> = h.iter().flat_map(|v| v.to_le_bytes()).collect();
        let post_norm_bytes: Vec<u8> = router
            .post_experts_norm
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        // 3. Distribute expert_ids/weights across shards.
        let shards_guard = self.shards.read().unwrap();
        let num_shards = shards_guard.len();
        let mut shard_eids: Vec<Vec<u32>> = vec![Vec::new(); num_shards];
        let mut shard_ewts: Vec<Vec<f32>> = vec![Vec::new(); num_shards];
        for (&eid, &w) in expert_indices.iter().zip(expert_weights.iter()) {
            let si = shards_guard
                .iter()
                .position(|s| s.owns_unit(layer, eid))
                .ok_or(RemoteMoeError::NoShard { expert_id: eid })?;
            shard_eids[si].push(eid as u32);
            shard_ewts[si].push(w);
        }
        drop(shards_guard);

        // 4. Fire one input per stream — non-blocking channel push.
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
            stream.fire(input)?;
        }

        Ok(InflightMoe {
            hidden,
            n_streams: streams.len(),
            post_experts_norm: router.post_experts_norm.to_vec(),
            norm_offset,
            eps,
        })
    }

    /// Collect half of `forward_moe_stream`: condvar-wait one partial weighted
    /// sum per shard, accumulate, and apply the post-experts RMS norm.
    ///
    /// Each shard returns the raw weighted sum of its own experts (without
    /// post-norm) so the caller can sum across shards and norm the combined
    /// output once — `rms_norm(a) + rms_norm(b) ≠ rms_norm(a + b)`.
    pub fn forward_moe_stream_collect(
        &self,
        streams: &[ShardStream],
        inflight: InflightMoe,
    ) -> Result<Vec<f32>, RemoteMoeError> {
        self.forward_moe_stream_collect_with_timing(streams, inflight)
            .map(|(h2, _)| h2)
    }

    /// Same as [`Self::forward_moe_stream_collect`] but also returns
    /// per-shard `(wall_collect_ms, server_compute_ms)` for diagnostics.
    /// The `wall_collect_ms` is the wall-clock time the caller waited
    /// for that shard's response (network + server compute + decode);
    /// `server_compute_ms` is what the server reported (when timing is
    /// enabled there).  `network_ms ≈ wall_collect_ms − server_compute_ms`.
    pub fn forward_moe_stream_collect_with_timing(
        &self,
        streams: &[ShardStream],
        inflight: InflightMoe,
    ) -> Result<(Vec<f32>, Vec<(f32, f32)>), RemoteMoeError> {
        let InflightMoe {
            hidden,
            n_streams,
            post_experts_norm,
            norm_offset,
            eps,
        } = inflight;

        if hidden == 0 || n_streams == 0 {
            return Ok((vec![0.0f32; hidden], Vec::new()));
        }

        let mut out = vec![0.0f32; hidden];
        let mut per_shard: Vec<(f32, f32)> = Vec::with_capacity(n_streams);
        for stream in streams.iter().take(n_streams) {
            let t0 = std::time::Instant::now();
            let (partial, server_compute_ms) = stream.collect_with_timing()?;
            let wall_ms = t0.elapsed().as_secs_f32() * 1000.0;
            per_shard.push((wall_ms, server_compute_ms));
            if partial.len() == hidden {
                for (acc, v) in out.iter_mut().zip(partial.iter()) {
                    *acc += v;
                }
            }
        }

        let normed = rms_norm(&out, &post_experts_norm, eps, norm_offset);
        Ok((normed, per_shard))
    }

    /// Pre-dispatch: route ALL layers at once, fire ONE batch call per shard
    /// (parallel), return h2 per layer.
    ///
    /// # Why faster than streaming
    ///
    /// `forward_moe` / `forward_moe_stream` make N sequential round-trips (one
    /// per layer). `forward_moe_predispatch` collapses them into ONE call per
    /// shard regardless of layer count.  The trade-off: each layer's expert
    /// input is computed from `h_post_attn` captured WITHOUT prior layers'
    /// expert contributions (pass-1 approximation), so the returned h2 values
    /// are slightly wrong for layers > 0.  In practice the error is small
    /// enough that the model still produces the correct top-1 token.
    ///
    /// # Usage
    ///
    /// 1. Run Metal with `moe_fn = |l, h| { capture[l] = h.to_vec(); zeros }`.
    /// 2. Call `forward_moe_predispatch(&captures, routers, ...)` — ONE async call.
    /// 3. Run Metal again with `moe_fn = |l, _h| { h2_per_layer[l].clone() }`.
    pub fn forward_moe_predispatch(
        &self,
        // h_post_attn captured per layer in the SKIP_MOE pass
        h_per_layer: &[Vec<f32>],
        // router weights for each layer (same length as h_per_layer)
        routers: &[MoeRouterWeights<'_>],
        norm_offset: f32,
        eps: f32,
    ) -> Result<Vec<Vec<f32>>, RemoteMoeError> {
        let num_layers = h_per_layer.len().min(routers.len());
        if num_layers == 0 {
            return Ok(vec![]);
        }
        let hidden = h_per_layer[0].len();
        let t0 = std::time::Instant::now();

        // 1. Route all layers locally, group expert calls by shard.
        let shards = self.shards.read().unwrap();
        let num_shards = shards.len();
        // shard_items[si] = Vec<(layer, expert_id, residual_bytes, weight)>
        let mut shard_items: Vec<Vec<(usize, usize, Vec<u8>, f32)>> =
            vec![Vec::new(); num_shards];

        for (l, (h, router)) in h_per_layer.iter().zip(routers.iter()).enumerate() {
            let residual_bytes: Vec<u8> =
                h.iter().flat_map(|v| v.to_le_bytes()).collect();
            let (_, expert_indices, expert_weights) = router.route(h, norm_offset, eps);
            for (&eid, &w) in expert_indices.iter().zip(expert_weights.iter()) {
                let si = shards
                    .iter()
                    .position(|s| s.owns_unit(l, eid))
                    .ok_or(RemoteMoeError::NoShard { expert_id: eid })?;
                shard_items[si].push((l, eid, residual_bytes.clone(), w));
            }
        }
        drop(shards);
        let t_route = t0.elapsed().as_secs_f64() * 1000.0;

        // 2. Fire ONE call per shard in parallel (rayon), collect raw outputs.
        //    Each item: (layer, expert_id, h2_contribution).
        let shard_results: Vec<Result<Vec<(usize, usize, Vec<f32>)>, RemoteMoeError>> =
            shard_items
                .par_iter()
                .map(|items| {
                    if items.is_empty() {
                        return Ok(vec![]);
                    }
                    let calls: Vec<ExpertCallItem> = items
                        .iter()
                        .map(|(layer, eid, res, _w)| ExpertCallItem {
                            layer: *layer,
                            expert_id: *eid,
                            residual: res
                                .chunks_exact(4)
                                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                                .collect(),
                        })
                        .collect();
                    let shards_g = self.shards.read().unwrap();
                    // `items` is a per-shard bucket built above; every entry
                    // here belongs to the same shard, so picking shard from
                    // the first item's (layer, expert_id) is correct.
                    let (first_layer, first_eid) = (items[0].0, items[0].1);
                    let si = shards_g
                        .iter()
                        .position(|s| s.owns_unit(first_layer, first_eid))
                        .ok_or(RemoteMoeError::NoShard { expert_id: first_eid })?;
                    let raw = shards_g[si].call_batch(&calls)?;
                    Ok(items
                        .iter()
                        .zip(raw.iter())
                        .map(|((layer, eid, _, _), r)| (*layer, *eid, r.output.clone()))
                        .collect())
                })
                .collect();
        let t_dispatch = t0.elapsed().as_secs_f64() * 1000.0;

        // 3. Accumulate weighted outputs per layer.
        //    Weight for each (layer, expert_id) is stored in shard_items[si][j].3
        let mut h2_per_layer: Vec<Vec<f32>> = vec![vec![0.0f32; hidden]; num_layers];
        for (si, shard_result) in shard_results.into_iter().enumerate() {
            let items_out = shard_result?;
            for (j, (layer, _eid, output)) in items_out.into_iter().enumerate() {
                let weight = shard_items[si][j].3; // stored weight from routing
                if output.len() == hidden {
                    for (acc, &v) in h2_per_layer[layer].iter_mut().zip(output.iter()) {
                        *acc += weight * v;
                    }
                }
            }
        }

        let t_accum = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!("[predispatch] route={:.1}ms dispatch={:.1}ms accum={:.1}ms  items/shard={:?}",
            t_route, t_dispatch - t_route, t_accum - t_dispatch,
            shard_items.iter().map(|v| v.len()).collect::<Vec<_>>());

        // Apply post-experts norm per layer.
        for (l, h2) in h2_per_layer.iter_mut().enumerate() {
            if !routers[l].post_experts_norm.is_empty() {
                *h2 = rms_norm(h2, routers[l].post_experts_norm, eps, norm_offset);
            }
        }

        Ok(h2_per_layer)
    }
}

// ── InflightMoe — handle returned by forward_moe_stream_fire ─────────────────
//
// Carries the post-norm context across the fire/collect boundary so callers do
// not need to retain the `MoeRouterWeights` borrow while GPU work runs in
// between.  `n_streams == 0` signals the trivial case (empty hidden / zero
// experts / no shards) where `collect` returns zeros without waiting.

/// Opaque handle for a fire-and-collect MoE round trip on a stream.
pub struct InflightMoe {
    hidden: usize,
    n_streams: usize,
    post_experts_norm: Vec<f32>,
    norm_offset: f32,
    eps: f32,
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
    /// Each item is `(h2, server_compute_ms)` — `compute_ms` is `0.0` when the
    /// server isn't recording timing.
    result_rx: std::sync::mpsc::Receiver<Result<(Vec<f32>, f32), RemoteMoeError>>,
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
    /// Discards `compute_ms` — use [`Self::collect_with_timing`] to keep it.
    pub fn collect(&self) -> Result<Vec<f32>, RemoteMoeError> {
        self.collect_with_timing().map(|(h2, _)| h2)
    }

    /// Collect with the server's `compute_ms` value attached. `compute_ms` is
    /// `0.0` when the server isn't recording timing (`LARQL_MOE_TIMING` unset).
    pub fn collect_with_timing(&self) -> Result<(Vec<f32>, f32), RemoteMoeError> {
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
            let transport = ShardTransport::Http(reqwest::blocking::Client::new());
            Shard { config, transport }
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

    // ── Per-(layer, expert) ownership ────────────────────────────────────
    //
    // Verify that:
    //   1. A shard built with `with_units` ignores layer-uniform `owns(...)`
    //      so layer-aware `owns_unit(...)` is the only source of truth.
    //   2. Layer-uniform shards keep working unchanged via `owns_unit`
    //      (legacy `--moe-shards "0-63=URL"` configs).
    //   3. The manifest parser round-trips JSON → `Vec<ShardConfig>` with
    //      ownership sets matching the inclusive ranges in the input.

    fn make_unit_shard(units: &[(usize, usize)]) -> Shard {
        let set: std::collections::HashSet<(usize, usize)> = units.iter().copied().collect();
        let config = ShardConfig::with_units("http://localhost:9000", set);
        let transport = ShardTransport::Http(reqwest::blocking::Client::new());
        Shard { config, transport }
    }

    #[test]
    fn shard_with_units_only_owns_via_layer_aware_check() {
        let s = make_unit_shard(&[(0, 5), (3, 17)]);
        // Legacy owns must return false in unit-set mode (forces layer-aware
        // routing at all call sites).
        assert!(!s.owns(5));
        assert!(!s.owns(17));
        // Layer-aware owns_unit honours the explicit set.
        assert!(s.owns_unit(0, 5));
        assert!(s.owns_unit(3, 17));
        assert!(!s.owns_unit(1, 5)); // wrong layer
        assert!(!s.owns_unit(0, 6)); // wrong expert
        assert!(!s.owns_unit(3, 5)); // belongs to layer 0, not 3
    }

    #[test]
    fn shard_layer_uniform_owns_unit_falls_back_to_range() {
        let config = ShardConfig::new(0, 31, "http://localhost:9000");
        let transport = ShardTransport::Http(reqwest::blocking::Client::new());
        let s = Shard { config, transport };
        // owns_unit on a legacy range-shard ignores the layer and uses the
        // range — keeps `--moe-shards "0-31=URL"` semantics.
        assert!(s.owns_unit(0, 0));
        assert!(s.owns_unit(0, 31));
        assert!(s.owns_unit(7, 17));
        assert!(!s.owns_unit(0, 32));
    }

    #[test]
    fn unit_manifest_round_trips_into_shard_configs() {
        let json = r#"{
            "shards": [
                {"url": "grpc://a:9081",
                 "layer_experts": {"0": [[0,2]], "1": [[5,7]]}},
                {"url": "grpc://b:9082",
                 "layer_experts": {"0": [[3,5]], "1": [[8,10],[15,15]]}}
            ]
        }"#;
        let m: UnitManifest = serde_json::from_str(json).unwrap();
        let configs = m.into_shard_configs().unwrap();
        assert_eq!(configs.len(), 2);

        // Shard A: 6 (layer, expert) pairs.
        let a = &configs[0];
        let a_units = a.unit_set.as_ref().unwrap();
        assert_eq!(a_units.len(), 6);
        for &(l, e) in &[(0, 0), (0, 1), (0, 2), (1, 5), (1, 6), (1, 7)] {
            assert!(a_units.contains(&(l, e)), "shard A missing ({l},{e})");
        }
        assert_eq!(a.start, 0); // min expert id across set
        assert_eq!(a.end, 7);   // max expert id across set

        // Shard B: 7 pairs (note the singleton range [15,15]).
        let b_units = configs[1].unit_set.as_ref().unwrap();
        assert_eq!(b_units.len(), 7);
        assert!(b_units.contains(&(1, 15)));
    }

    #[test]
    fn unit_manifest_rejects_reversed_range() {
        let json = r#"{"shards": [
            {"url": "grpc://x:1", "layer_experts": {"0": [[5,2]]}}
        ]}"#;
        let m: UnitManifest = serde_json::from_str(json).unwrap();
        let err = m.into_shard_configs().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("end (2) must be >= start (5)"), "got: {msg}");
    }

    #[test]
    fn unit_manifest_rejects_non_numeric_layer() {
        let json = r#"{"shards": [
            {"url": "grpc://x:1", "layer_experts": {"oops": [[0,1]]}}
        ]}"#;
        let m: UnitManifest = serde_json::from_str(json).unwrap();
        let err = m.into_shard_configs().unwrap_err();
        assert!(format!("{err}").contains("layer key 'oops'"));
    }

    #[test]
    fn parse_unit_manifest_reports_path_on_missing_file() {
        let bogus = std::path::PathBuf::from("/nonexistent/larql-units-x.json");
        let err = parse_unit_manifest(&bogus).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("read"), "msg should mention read failure: {msg}");
        assert!(msg.contains(bogus.to_str().unwrap()), "msg should name path: {msg}");
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
