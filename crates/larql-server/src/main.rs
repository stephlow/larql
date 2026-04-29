//! larql-server — HTTP server for vindex knowledge queries.

use std::path::PathBuf;
use std::sync::Arc;

use axum::middleware;
use clap::Parser;
use tracing::{info, warn};

use larql_server::bootstrap::{
    discover_vindexes, load_single_vindex, normalize_serve_alias, parse_layer_range, BoxError,
    LoadVindexOptions,
};
use larql_server::cache::DescribeCache;
use larql_server::session::SessionManager;
use larql_server::state::{AppState, LoadedModel};
use larql_server::{announce, auth, grpc, grpc_expert, ratelimit, routes};

#[derive(Parser)]
#[command(
    name = "larql-server",
    version,
    about = "HTTP server for vindex knowledge queries and inference"
)]
struct Cli {
    /// Path to a .vindex directory (or hf:// path).
    #[arg(value_name = "VINDEX_PATH")]
    vindex_path: Option<String>,

    /// Serve all .vindex directories in this folder.
    #[arg(long)]
    dir: Option<PathBuf>,

    /// Listen port.
    #[arg(long, default_value = "8080")]
    port: u16,

    /// Bind address.
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Disable INFER endpoint (browse-only, reduces memory).
    #[arg(long)]
    no_infer: bool,

    /// Run as an FFN-service endpoint for remote `RemoteWalkBackend`
    /// clients. Disables `/v1/infer` (like `--no-infer`) and advertises
    /// `mode: ffn-service` in `/v1/stats`. This is Act 2 of the demo —
    /// the server holds the FFN weights, clients hold attention.
    ///
    /// Also skips the f16→f32 gate-vector warmup, which is the largest
    /// eager cost on startup (~2x the gate_vectors.bin size). Gate
    /// decode happens lazily per layer on first request instead.
    #[arg(long)]
    ffn_only: bool,

    /// Run as an embed-service endpoint.
    ///
    /// Loads only embeddings.bin, lm_head, and the tokenizer — skips all
    /// FFN and attention weights. Advertises `mode: embed-service` in
    /// `/v1/stats`. Enables `/v1/embed`, `/v1/logits`, and `/v1/token/*`.
    ///
    /// Use this to offload the static embedding + lm_head lookup from
    /// attention-only clients (ADR-0007). The embed slice is ~2-5% of the
    /// full model weight — a minimal VPS can host it independently.
    #[arg(long)]
    embed_only: bool,

    /// Only load and serve layers in this range (inclusive, e.g. "0-19").
    /// Layers outside the range are not dequantized and their mmap pages are
    /// never touched, keeping RSS proportional to the shard size.
    /// Requests for out-of-range layers are rejected with HTTP 400.
    #[arg(long)]
    layers: Option<String>,

    /// Cap the number of decoded f16 gate layers held in the lazy cache.
    /// 0 = unlimited (default; matches historical behaviour). Each decoded
    /// layer is roughly `intermediate × hidden × 4 bytes` — on 31B that's
    /// ~433 MB per layer, so a 60-layer model fully decoded is ~26 GB.
    /// Set to N to cap at N layers via LRU eviction.
    ///
    /// Use when RSS headroom matters (e.g. co-hosting multiple models) at
    /// the cost of re-decode when evicted layers are re-accessed.
    #[arg(long, default_value = "0")]
    max_gate_cache_layers: usize,

    /// Cap the number of layers held in the Q4_K/Q6_K FFN dequant cache.
    /// 0 = unlimited (default). Only fires on the CPU per-position
    /// fallback in walk_ffn — Metal full-K decode does not populate
    /// this cache. Each cached layer holds up to gate+up+down
    /// dequantised to f32 (`intermediate × hidden × 4 bytes` per
    /// component). On Gemma 3 4B that's ~105 MB/component — set to
    /// 8 for ~840 MB ceiling on the down leg.
    #[arg(long, default_value = "0")]
    max_q4k_cache_layers: usize,

    /// Use HNSW for gate KNN instead of brute-force matmul. Indexes
    /// are built lazily per layer on first query. Approximate (recall
    /// drops from 100% to 80–95% depending on `--hnsw-ef-search`); the
    /// retrieval ranks by |dot| like the brute path, but oversamples
    /// HNSW and re-ranks at the seam. Wins for high-feature MoE
    /// (64-expert ≈ 230 → 60 ms/layer); break-even or net loss for
    /// dense ≤ 10K-feature models.
    #[arg(long)]
    hnsw: bool,

    /// HNSW beam width. Higher = better recall, slower search. 50 is
    /// the floor; 200 is the default; 400 is the practical ceiling.
    #[arg(long, default_value = "200")]
    hnsw_ef_search: usize,

    /// Eager-build the HNSW index for every owned layer at startup
    /// (rayon-parallel across layers). One-shot; trades ~700 ms of boot
    /// time for first-query latency that would otherwise pay ~76 ms /
    /// layer × N lazy builds spread across the first request volume.
    /// Recommended when this server will see traffic on every layer
    /// (e.g. `larql-router` shards behind a steady-state interp pipeline).
    /// Requires `--hnsw`.
    #[arg(long, requires = "hnsw")]
    warmup_hnsw: bool,

    /// Pre-load inference weights and prefetch every owned layer's
    /// Q4K mmap pages at boot. Cuts first-`walk-ffn` latency from
    /// ~1.3 s + 17 ms / cold layer down to the warm baseline
    /// (~0.3 ms / layer) at the cost of a ~1–2 s startup delay and
    /// ~3 GB pre-allocated f32 gate cache. Recommended for grid
    /// shards under a steady-state load — operators can also fire
    /// `POST /v1/warmup` later without a restart.
    #[arg(long)]
    warmup_walk_ffn: bool,

    /// Ask the kernel to drop resident mmap pages after each walk-ffn
    /// request (calls `madvise(MADV_DONTNEED)` on every mapping). On
    /// Linux RSS drops immediately; on Darwin the kernel may defer.
    /// Pairs with `--max-gate-cache-layers` to enforce a hard bound.
    ///
    /// Prefer `--layers START-END` for real deployments — sharding
    /// prevents out-of-range pages from ever being touched. This flag
    /// is for the single-shard-holds-everything demo topology.
    #[arg(long)]
    release_mmap_after_request: bool,

    /// Only load and serve experts in this range (inclusive, e.g. "0-31").
    /// Requests for out-of-range expert IDs are rejected with HTTP 400.
    /// Used to shard the expert bank across multiple servers.
    /// Layer-uniform: same expert range applies to every layer.
    #[arg(long)]
    experts: Option<String>,

    /// Path to a JSON manifest specifying per-(layer, expert) ownership for
    /// fine-grained shards.  Format:
    /// ```json
    /// { "layer_experts": { "0": [[0,31]], "1": [[0,15],[64,79]], ... } }
    /// ```
    /// Each value is a list of inclusive `[start, end]` expert-id ranges.
    /// Layers absent from the map own no experts on this shard.
    ///
    /// When set, overrides `--experts` and switches `run_expert` ownership
    /// checks to per-(layer, expert) lookups.  Designed for the architecture
    /// where each shard hosts a tight set of (layer, expert) units rather
    /// than a contiguous expert range across all layers.
    #[arg(long, value_name = "PATH")]
    units: Option<std::path::PathBuf>,

    /// Enable CORS for browser access.
    #[arg(long)]
    cors: bool,

    /// API key for authentication (clients send Authorization: Bearer <key>).
    #[arg(long)]
    api_key: Option<String>,

    /// Rate limit per IP (e.g., "100/min", "10/sec").
    #[arg(long)]
    rate_limit: Option<String>,

    /// Trust X-Forwarded-For when rate limiting.
    ///
    /// Enable only when the server is behind a trusted reverse proxy that
    /// strips untrusted client-supplied forwarding headers.
    #[arg(long)]
    trust_forwarded_for: bool,

    /// Max concurrent requests.
    #[arg(long, default_value = "100")]
    max_concurrent: usize,

    /// Cache TTL for DESCRIBE results in seconds (0 = disabled).
    #[arg(long, default_value = "0")]
    cache_ttl: u64,

    /// Logging level.
    #[arg(long, default_value = "info")]
    log_level: String,

    /// gRPC port (enables gRPC server alongside HTTP).
    #[arg(long)]
    grpc_port: Option<u16>,

    /// TLS certificate path for HTTPS.
    #[arg(long)]
    tls_cert: Option<PathBuf>,

    /// TLS private key path for HTTPS.
    #[arg(long)]
    tls_key: Option<PathBuf>,

    /// Join one or more router grids (comma-separated gRPC addresses).
    /// Example: "http://router-a:50052,http://router-b:50052"
    /// Each router gets an independent announce stream — stateless fan-out.
    /// Requires --public-url so routers know where to send clients.
    #[arg(long)]
    join: Option<String>,

    /// Public HTTP URL clients should use to reach this server.
    /// Used when announcing to the grid with --join.
    /// Example: "http://server-a:8080"
    #[arg(long)]
    public_url: Option<String>,

    /// Shared secret matching the router's --grid-key.
    /// Required when the router enforces grid authentication.
    #[arg(long, env = "LARQL_GRID_KEY")]
    grid_key: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    // Accept both `larql-server <path>` and `larql-server serve <path>`.
    let cli = Cli::parse_from(normalize_serve_alias(std::env::args().collect()));

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&cli.log_level)),
        )
        .init();

    info!("larql-server v{}", env!("CARGO_PKG_VERSION"));

    let mut models: Vec<Arc<LoadedModel>> = Vec::new();

    let layer_range = cli.layers.as_deref().map(parse_layer_range).transpose()?;
    let expert_filter = cli.experts.as_deref().map(parse_layer_range).transpose()?;
    // --units PATH (per-(layer, expert) ownership manifest) takes precedence
    // over --experts START-END; the two are mutually exclusive at parse time
    // so the operator gets a clear error rather than silently picking one.
    if cli.units.is_some() && cli.experts.is_some() {
        return Err(
            "--units and --experts are mutually exclusive — \
             use --experts for layer-uniform ranges, --units for fine-grained ownership"
                .into(),
        );
    }
    let unit_filter = cli
        .units
        .as_deref()
        .map(larql_server::bootstrap::parse_unit_manifest)
        .transpose()?
        .map(Arc::new);
    if let Some(ref u) = unit_filter {
        info!(
            "  Units (--units): {} ({}, {}) pairs across {} layers",
            u.len(),
            "layer",
            "expert",
            u.iter().map(|(l, _)| *l).collect::<std::collections::HashSet<_>>().len(),
        );
    }
    let load_opts = LoadVindexOptions {
        no_infer: cli.no_infer,
        ffn_only: cli.ffn_only,
        embed_only: cli.embed_only,
        layer_range,
        max_gate_cache_layers: cli.max_gate_cache_layers,
        max_q4k_cache_layers: cli.max_q4k_cache_layers,
        hnsw: if cli.hnsw {
            Some(cli.hnsw_ef_search)
        } else {
            None
        },
        warmup_hnsw: cli.warmup_hnsw,
        release_mmap_after_request: cli.release_mmap_after_request,
        expert_filter,
        unit_filter,
    };

    if let Some(ref dir) = cli.dir {
        let paths = discover_vindexes(dir);
        if paths.is_empty() {
            return Err(format!("no .vindex directories found in {}", dir.display()).into());
        }
        info!("Found {} vindexes in {}", paths.len(), dir.display());
        for p in &paths {
            // `LoadVindexOptions` is `Clone` (was `Copy` until `unit_filter`
            // added an `Arc<HashSet<...>>` field) — clone per iteration so
            // the loop owns each call's argument.
            match load_single_vindex(&p.to_string_lossy(), load_opts.clone()) {
                Ok(m) => models.push(Arc::new(m)),
                Err(e) => warn!("  Skipping {}: {}", p.display(), e),
            }
        }
    } else if let Some(ref vindex_path) = cli.vindex_path {
        let m = load_single_vindex(vindex_path, load_opts)?;
        models.push(Arc::new(m));
    } else {
        return Err("must provide a vindex path or --dir".into());
    }

    if models.is_empty() {
        return Err("no vindexes loaded".into());
    }

    // Parse rate limiter if specified.
    let rate_limiter =
        cli.rate_limit
            .as_ref()
            .and_then(|spec| match ratelimit::RateLimiter::parse(spec) {
                Some(rl) => {
                    info!("Rate limit: {}", spec);
                    Some(Arc::new(rl))
                }
                None => {
                    warn!(
                        "Invalid rate limit format: {} (expected e.g. '100/min')",
                        spec
                    );
                    None
                }
            });

    let state = Arc::new(AppState {
        models: models.clone(),
        started_at: std::time::Instant::now(),
        requests_served: std::sync::atomic::AtomicU64::new(0),
        api_key: cli.api_key.clone(),
        sessions: SessionManager::new(3600),
        describe_cache: DescribeCache::new(cli.cache_ttl),
    });

    if cli.cache_ttl > 0 {
        info!("DESCRIBE cache: {}s TTL", cli.cache_ttl);
    }

    let is_multi = state.is_multi_model();
    let mut app = if is_multi {
        info!("Multi-model mode ({} models)", state.models.len());
        for m in &state.models {
            info!("  /v1/{}/...", m.id);
        }
        routes::multi_model_router(Arc::clone(&state))
    } else {
        let m = &models[0];
        info!("Single-model mode: {}", m.config.model);
        routes::single_model_router(Arc::clone(&state))
    };

    // `--warmup-walk-ffn` — pre-load inference weights + prefetch every
    // owned layer's Q4K mmap so the first `/v1/walk-ffn` doesn't pay
    // the ~1.3 s lazy weight load + ~17 ms / cold layer (see
    // ROADMAP G1 / G2). Same code path as `POST /v1/warmup`. Goes
    // through `warmup_model_async` (which uses `spawn_blocking`)
    // because we're inside the tokio runtime here and the patched
    // RwLock is async — `blocking_read` would panic.
    if cli.warmup_walk_ffn {
        for m in &state.models {
            // walk-ffn needs the inference weights (gate-f32 cache,
            // norms, lm_head) regardless of `--no-infer` (which only
            // disables the `/v1/infer` route). Always load.
            let req = routes::warmup::WarmupRequest {
                layers: None,
                skip_weights: false,
                warmup_hnsw: false,
            };
            let r = routes::warmup::warmup_model_async(Arc::clone(m), req).await;
            info!(
                "  Warmup walk-ffn[{}]: weights={} ({}ms), prefetched {} layers ({}ms), total {}ms",
                r.model,
                r.weights_loaded,
                r.weights_load_ms,
                r.layers_prefetched,
                r.prefetch_ms,
                r.total_ms,
            );
        }
    }

    // Per-(layer, expert) HNSW unit warmup.  Each shard pre-builds an
    // HNSW index over each owned expert's gate slice (~704 vectors per
    // unit on Gemma 4 26B-A4B, vs ~90k for the layer-level index).
    // Used by walk / interp KNN queries (`gate_knn_expert`); not on the
    // MoE forward path.  Skipped when `LARQL_NO_WARMUP=1`.
    for m in &state.models {
        if m.expert_filter.is_none() && !cli.warmup_walk_ffn {
            // No shard filter and operator didn't ask for walk-ffn warmup
            // → skip the cost; whoever queries first will lazy-build.
            continue;
        }
        let model = Arc::clone(m);
        let model_id = model.id.clone();
        let t0 = std::time::Instant::now();
        let result = tokio::task::spawn_blocking(move || {
            crate::routes::expert::warmup_hnsw_unit_cache(&model)
        })
        .await;
        match result {
            Ok(Ok((built, n_layers, n_owned))) if built > 0 => {
                let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
                info!(
                    "  Warmup hnsw-units[{model_id}]: built {built} units \
                     ({n_layers} layers × {n_owned} experts/shard) in {elapsed_ms:.0}ms"
                );
            }
            Ok(Ok(_)) => {} // No units built (non-MoE / opted-out / nothing owned).
            Ok(Err(e)) => {
                tracing::warn!("Warmup hnsw-units[{model_id}] failed: {e}");
            }
            Err(e) => {
                tracing::warn!("Warmup hnsw-units[{model_id}] join failed: {e}");
            }
        }
    }

    // Metal expert cache warmup (cfg=metal-experts only).  For shard
    // servers, eagerly populate the BufferCache for every expert this
    // shard owns so the first decode token sees the steady-state ~20
    // tok/s instead of the cold-call 4–8 tok/s ramp.  Skipped when
    // LARQL_NO_WARMUP=1 is set.
    #[cfg(feature = "metal-experts")]
    for m in &state.models {
        // Only run for shard servers (have an expert_filter).  Models
        // without --experts are running the whole MoE locally and use a
        // different code path.
        if m.expert_filter.is_none() {
            continue;
        }
        let model = Arc::clone(m);
        let model_id = model.id.clone();
        let t0 = std::time::Instant::now();
        let result = tokio::task::spawn_blocking(move || {
            crate::routes::expert::warmup_metal_expert_cache(&model)
        })
        .await;
        match result {
            Ok(Ok(staged)) => {
                let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
                if staged > 0 {
                    info!(
                        "  Warmup metal-experts[{model_id}]: staged {staged} \
                         (gate_up, down) buffer pairs in {elapsed_ms:.0}ms"
                    );
                }
            }
            Ok(Err(e)) => {
                tracing::warn!("Warmup metal-experts[{model_id}] failed: {e}");
            }
            Err(e) => {
                tracing::warn!("Warmup metal-experts[{model_id}] join failed: {e}");
            }
        }
    }

    // Rate limiting middleware.
    if let Some(ref rl) = rate_limiter {
        let rate_state = Arc::new(ratelimit::RateLimitState {
            limiter: Arc::clone(rl),
            trust_forwarded_for: cli.trust_forwarded_for,
        });
        app = app.layer(middleware::from_fn_with_state(
            rate_state,
            ratelimit::rate_limit_middleware,
        ));
        if cli.trust_forwarded_for {
            info!("Rate limit: trusting X-Forwarded-For");
        }
    }

    // Auth middleware (if --api-key set).
    if cli.api_key.is_some() {
        app = app.layer(middleware::from_fn_with_state(
            Arc::clone(&state),
            auth::auth_middleware,
        ));
        info!("Auth: API key required");
    }

    // CORS middleware.
    if cli.cors {
        use tower_http::cors::CorsLayer;
        app = app.layer(CorsLayer::permissive());
        info!("CORS: enabled");
    }

    // Concurrency limit.
    app = app.layer(tower::limit::ConcurrencyLimitLayer::new(cli.max_concurrent));
    info!("Max concurrent: {}", cli.max_concurrent);

    // Trace middleware.
    app = app.layer(tower_http::trace::TraceLayer::new_for_http());

    // gRPC server (if --grpc-port set).
    if let Some(grpc_port) = cli.grpc_port {
        let grpc_addr = format!("{}:{}", cli.host, grpc_port).parse()?;
        let grpc_state = Arc::clone(&state);
        info!("gRPC: listening on {}", grpc_addr);
        tokio::spawn(async move {
            let vindex_svc = grpc::VindexGrpcService { state: Arc::clone(&grpc_state) };
            let expert_svc = grpc_expert::ExpertGrpcService { state: Arc::clone(&grpc_state) };
            if let Err(e) = tonic::transport::Server::builder()
                .add_service(grpc::proto::vindex_service_server::VindexServiceServer::new(vindex_svc))
                .add_service(larql_router_protocol::ExpertServiceServer::new(expert_svc))
                .serve(grpc_addr)
                .await
            {
                tracing::error!("gRPC server error: {}", e);
            }
        });
    }

    let addr = format!("{}:{}", cli.host, cli.port);

    // Grid announce (if --join provided).
    if let Some(join_spec) = cli.join.clone() {
        let listen_url = cli.public_url.clone().unwrap_or_else(|| {
            let host = if cli.host == "0.0.0.0" {
                "127.0.0.1"
            } else {
                &cli.host
            };
            format!("http://{}:{}", host, cli.port)
        });
        let join_urls: Vec<String> = join_spec
            .split(',')
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty())
            .collect();
        if join_urls.len() > 1 {
            info!("Joining {} routers (stateless fan-out)", join_urls.len());
        }
        for m in &models {
            let (layer_start, layer_end) = match layer_range {
                Some((s, e)) => (s as u32, (e - 1) as u32),
                None => (0, (m.config.num_layers.saturating_sub(1)) as u32),
            };
            let vhash = announce::vindex_identity_hash(&m.id, m.config.num_layers);
            for join_url in &join_urls {
                announce::run_announce(announce::AnnounceConfig {
                    join_url: join_url.clone(),
                    model_id: m.id.clone(),
                    layer_start,
                    layer_end,
                    listen_url: listen_url.clone(),
                    ram_bytes: 0,
                    grid_key: cli.grid_key.clone(),
                    vindex_hash: vhash.clone(),
                });
            }
        }
    }

    // TLS or plain HTTP.
    if let (Some(cert_path), Some(key_path)) = (&cli.tls_cert, &cli.tls_key) {
        info!(
            "TLS: enabled ({}, {})",
            cert_path.display(),
            key_path.display()
        );
        info!("Listening: https://{}", addr);

        let tls_config =
            axum_server::tls_rustls::RustlsConfig::from_pem_file(cert_path, key_path).await?;

        axum_server::bind_rustls(addr.parse()?, tls_config)
            .serve(app.into_make_service())
            .await?;
    } else {
        info!("Listening: http://{}", addr);
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        axum::serve(listener, app).await?;
    }

    Ok(())
}
