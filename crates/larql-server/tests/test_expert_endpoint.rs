//! End-to-end parity test for `POST /v1/expert/batch`.
//!
//! Spins up the real `larql-server` router (same axum app that ships in
//! production) on an OS-assigned port, then drives it through the real
//! `RemoteMoeBackend` from `larql-inference`.  Output is compared
//! bit-for-bit against a local `cpu_moe_forward` call on the same weights.
//!
//! What this proves:
//!   1. `routes::expert` handlers are correctly wired and reachable.
//!   2. JSON wire format serialises/deserialises cleanly end-to-end.
//!   3. `RemoteMoeBackend` routes, dispatches, and accumulates outputs
//!      identically to the local path.
//!   4. Two-shard split gives the same result as a single shard.
//!   5. `reshard()` swaps endpoints live without breaking the next call.
//!   6. Expert owned by a different shard → `NoShard` error.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use tokio::net::TcpListener;

use larql_inference::{
    cpu_moe_forward, ndarray::ArcArray2, MoeLayerWeights, MoeRouterWeights, RemoteMoeBackend,
    RemoteMoeError, ShardConfig,
};
use larql_models::weights::ModelWeights;
use larql_models::{ModelArchitecture, ModelConfig};
use larql_vindex::{
    ndarray::Array2, ExtractLevel, LayerBands, PatchedVindex, QuantFormat, VectorIndex,
    VindexConfig, VindexLayerInfo,
};

use larql_server::{
    cache::DescribeCache,
    ffn_l2_cache::FfnL2Cache,
    routes::single_model_router,
    session::SessionManager,
    state::{AppState, LoadedModel},
};

// ── Synthetic weight dimensions ───────────────────────────────────────────────

const NUM_EXPERTS: usize = 4;
const TOP_K: usize = 2;
const HIDDEN: usize = 8;
const INTER: usize = 6;
const VOCAB: usize = 8;

// ── Minimal test ModelArchitecture ────────────────────────────────────────────

struct TestMoeArch {
    cfg: ModelConfig,
}

impl TestMoeArch {
    fn new() -> Self {
        Self {
            cfg: ModelConfig {
                model_type: "test-moe".to_string(),
                num_layers: 1,
                hidden_size: HIDDEN,
                intermediate_size: 16,
                head_dim: 4,
                num_q_heads: 2,
                num_kv_heads: 2,
                vocab_size: Some(VOCAB),
                rope_base: 10000.0,
                rope_local_base: None,
                sliding_window: None,
                num_experts: Some(NUM_EXPERTS),
                num_experts_per_token: None,
                num_shared_experts: None,
                enable_moe_block: true,
                top_k_experts: Some(TOP_K),
                moe_intermediate_size: Some(INTER),
                kv_lora_rank: None,
                q_lora_rank: None,
                rope_scaling: None,
                attn_logit_softcapping: None,
                final_logit_softcapping: None,
                query_pre_attn_scalar: None,
                embedding_multiplier: None,
                residual_multiplier: None,
                attention_multiplier: None,
                logits_scaling: None,
                global_head_dim: None,
                num_global_kv_heads: None,
                partial_rotary_factor: None,
                sliding_window_pattern: None,
                layer_types: None,
                attention_k_eq_v: false,
                per_layer_embed_dim: None,
                num_kv_shared_layers: None,
            },
        }
    }
}

impl ModelArchitecture for TestMoeArch {
    fn family(&self) -> &str {
        "test-moe"
    }
    fn config(&self) -> &ModelConfig {
        &self.cfg
    }
    fn is_hybrid_moe(&self) -> bool {
        true
    }
    fn num_experts(&self) -> usize {
        NUM_EXPERTS
    }
    fn num_experts_per_token(&self) -> usize {
        TOP_K
    }
    fn moe_intermediate_size(&self) -> usize {
        INTER
    }
    fn norm_eps(&self) -> f32 {
        1e-6
    }
    fn packed_experts_gate_up_key(&self, _: usize) -> Option<String> {
        Some("test.gate_up".into())
    }
    fn packed_experts_down_key(&self, _: usize) -> Option<String> {
        Some("test.down".into())
    }
    fn moe_router_key(&self, _: usize) -> Option<String> {
        Some("test.router".into())
    }
    fn moe_pre_experts_norm_key(&self, _: usize) -> Option<String> {
        Some("test.pre_norm".into())
    }
}

// ── BF16 packing helper ───────────────────────────────────────────────────────

fn pack_bf16(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 2);
    for &v in data {
        let bits = v.to_bits();
        let bf16 = (bits >> 16) as u16;
        out.extend_from_slice(&bf16.to_le_bytes());
    }
    out
}

// ── Synthetic weight factories ────────────────────────────────────────────────

fn make_gate_up_bytes() -> Vec<u8> {
    let f32s: Vec<f32> = (0..NUM_EXPERTS * 2 * INTER * HIDDEN)
        .map(|i| (i as f32 * 0.01 - 0.5).clamp(-0.25, 0.25))
        .collect();
    pack_bf16(&f32s)
}

fn make_down_bytes() -> Vec<u8> {
    let f32s: Vec<f32> = (0..NUM_EXPERTS * HIDDEN * INTER)
        .map(|i| (i as f32 * 0.01 - 0.5).clamp(-0.25, 0.25))
        .collect();
    pack_bf16(&f32s)
}

fn make_router_proj() -> Vec<f32> {
    (0..NUM_EXPERTS * HIDDEN)
        .map(|i| (i as f32 + 1.0) * 0.05)
        .collect()
}

fn make_pre_norm() -> Vec<f32> {
    vec![1.0f32; HIDDEN]
}

fn make_input() -> Vec<f32> {
    (0..HIDDEN).map(|i| (i as f32 + 1.0) * 0.1).collect()
}

// ── LoadedModel factory ───────────────────────────────────────────────────────

fn make_loaded_model(
    gate_up: Vec<u8>,
    down: Vec<u8>,
    router_proj: Vec<f32>,
    pre_norm: Vec<f32>,
) -> LoadedModel {
    let gate = Array2::<f32>::zeros((2, HIDDEN));
    let index = VectorIndex::new(vec![Some(gate)], vec![None], 1, HIDDEN);
    let patched = PatchedVindex::new(index);

    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json).unwrap();

    let config = VindexConfig {
        version: 2,
        model: "test-moe".to_string(),
        family: "test".to_string(),
        source: None,
        checksums: None,
        num_layers: 1,
        hidden_size: HIDDEN,
        intermediate_size: 16,
        vocab_size: VOCAB,
        embed_scale: 1.0,
        extract_level: ExtractLevel::Browse,
        dtype: larql_vindex::StorageDtype::default(),
        quant: QuantFormat::None,
        layer_bands: Some(LayerBands {
            syntax: (0, 0),
            knowledge: (0, 0),
            output: (0, 0),
        }),
        layers: vec![VindexLayerInfo {
            layer: 0,
            num_features: 2,
            offset: 0,
            length: 32,
            num_experts: None,
            num_features_per_expert: None,
        }],
        down_top_k: 1,
        has_model_weights: false,
        model_config: None,
        fp4: None,
        ffn_layout: None,
    };

    // Build ModelWeights with expert data in raw_bytes (no mmap needed).
    let mut raw_bytes = HashMap::new();
    raw_bytes.insert("test.gate_up".to_string(), gate_up);
    raw_bytes.insert("test.down".to_string(), down);

    let mut vectors = HashMap::new();
    vectors.insert("test.router".to_string(), router_proj);
    vectors.insert("test.pre_norm".to_string(), pre_norm);

    let embed: ArcArray2<f32> = ArcArray2::zeros((VOCAB, HIDDEN));
    let weights = ModelWeights {
        tensors: HashMap::new(),
        vectors,
        raw_bytes,
        skipped_tensors: Vec::new(),
        packed_mmaps: HashMap::new(),
        packed_byte_ranges: HashMap::new(),
        embed: embed.clone(),
        lm_head: embed,
        arch: Box::new(TestMoeArch::new()),
        num_layers: 1,
        hidden_size: HIDDEN,
        intermediate_size: 16,
        vocab_size: VOCAB,
        head_dim: 4,
        num_q_heads: 2,
        num_kv_heads: 2,
        rope_base: 10000.0,
    };

    let lock = OnceLock::new();
    lock.set(weights).ok();

    LoadedModel {
        id: "test-moe".into(),
        path: PathBuf::from("/nonexistent"),
        config,
        patched: tokio::sync::RwLock::new(patched),
        embeddings: Array2::zeros((VOCAB, HIDDEN)),
        embed_scale: 1.0,
        tokenizer,
        infer_disabled: true,
        ffn_only: false,
        embed_only: false,
        embed_store: None,
        release_mmap_after_request: false,
        weights: lock,
        probe_labels: HashMap::new(),
        ffn_l2_cache: FfnL2Cache::new(1),
        expert_filter: None,
    }
}

/// Variant that sets `expert_filter` on the returned model. Used to test
/// `--experts START-END` ownership enforcement.
fn make_loaded_model_with_filter(
    gate_up: Vec<u8>,
    down: Vec<u8>,
    router_proj: Vec<f32>,
    pre_norm: Vec<f32>,
    filter: (usize, usize),
) -> LoadedModel {
    let mut m = make_loaded_model(gate_up, down, router_proj, pre_norm);
    m.expert_filter = Some(filter);
    m
}

// ── Server helper ─────────────────────────────────────────────────────────────

async fn spawn_server_with_model(model: LoadedModel) -> String {
    let state = Arc::new(AppState {
        models: vec![Arc::new(model)],
        started_at: std::time::Instant::now(),
        requests_served: std::sync::atomic::AtomicU64::new(0),
        api_key: None,
        sessions: SessionManager::new(3600),
        describe_cache: DescribeCache::new(60),
    });

    let router = single_model_router(state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    format!("http://{addr}")
}

// ── Reference output ──────────────────────────────────────────────────────────

fn local_output(
    h: &[f32],
    gate_up: &[u8],
    down: &[u8],
    router_proj: &[f32],
    pre_norm: &[f32],
) -> Vec<f32> {
    // Synthetic test fixtures store BF16 monolith. Slice into per-expert
    // tables for the new MoeLayerWeights API.
    let gu_stride = 2 * INTER * HIDDEN * 2;
    let dn_stride = HIDDEN * INTER * 2;
    let experts_gate_up: Vec<&[u8]> = (0..NUM_EXPERTS)
        .map(|e| &gate_up[e * gu_stride..(e + 1) * gu_stride])
        .collect();
    let experts_down: Vec<&[u8]> = (0..NUM_EXPERTS)
        .map(|e| &down[e * dn_stride..(e + 1) * dn_stride])
        .collect();
    cpu_moe_forward(
        h,
        &MoeLayerWeights {
            experts_gate_up,
            experts_down,
            router_proj,
            router_scale: &[],
            router_per_expert_scale: &[],
            router_norm: &[],
            router_norm_parameter_free: false,
            router_input_scalar: 1.0,
            pre_experts_norm: pre_norm,
            post_ffn1_norm: &[],
            post_experts_norm: &[],
            num_experts: NUM_EXPERTS,
            top_k: TOP_K,
            intermediate_size: INTER,
            activation: larql_compute::Activation::Silu,
            expert_data_format: larql_compute::QuantFormat::BF16,
        },
        0.0,
        1e-6,
    )
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn expert_endpoint_single_shard_parity() {
    let gate_up = make_gate_up_bytes();
    let down = make_down_bytes();
    let router_proj = make_router_proj();
    let pre_norm = make_pre_norm();
    let h = make_input();

    let url = spawn_server_with_model(make_loaded_model(
        gate_up.clone(),
        down.clone(),
        router_proj.clone(),
        pre_norm.clone(),
    ))
    .await;

    tokio::time::sleep(std::time::Duration::from_millis(10)).await;

    let url_c = url.clone();
    let backend = tokio::task::spawn_blocking(move || {
        RemoteMoeBackend::connect(vec![ShardConfig::new(0, NUM_EXPERTS - 1, url_c)])
            .expect("connect")
    })
    .await
    .unwrap();

    let rp = router_proj.clone();
    let pn = pre_norm.clone();
    let h_c = h.clone();
    let remote_out = tokio::task::spawn_blocking(move || {
        let router = MoeRouterWeights {
            router_proj: &rp,
            router_scale: &[],
            router_per_expert_scale: &[],
            router_norm: &[],
            router_norm_parameter_free: false,
            router_input_scalar: 1.0,
            pre_experts_norm: &pn,
            post_experts_norm: &[],
            num_experts: NUM_EXPERTS,
            top_k: TOP_K,
        };
        backend.forward_moe(0, &h_c, &router, 0.0, 1e-6)
    })
    .await
    .unwrap()
    .expect("forward_moe");

    let expected = local_output(&h, &gate_up, &down, &router_proj, &pre_norm);

    assert_eq!(remote_out.len(), expected.len());
    for (i, (&got, &exp)) in remote_out.iter().zip(expected.iter()).enumerate() {
        let diff = (got - exp).abs();
        assert!(
            diff < 1e-4,
            "output[{i}]: remote={got} local={exp} diff={diff:.2e}"
        );
    }
}

#[tokio::test]
async fn expert_endpoint_two_shard_parity() {
    let gate_up = make_gate_up_bytes();
    let down = make_down_bytes();
    let router_proj = make_router_proj();
    let pre_norm = make_pre_norm();
    let h = make_input();

    // Two separate server instances, each with all expert weights.
    // Shard A owns experts 0-1, shard B owns experts 2-3.
    let url_a = spawn_server_with_model(make_loaded_model(
        gate_up.clone(),
        down.clone(),
        router_proj.clone(),
        pre_norm.clone(),
    ))
    .await;
    let url_b = spawn_server_with_model(make_loaded_model(
        gate_up.clone(),
        down.clone(),
        router_proj.clone(),
        pre_norm.clone(),
    ))
    .await;
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;

    let backend = tokio::task::spawn_blocking(move || {
        RemoteMoeBackend::connect(vec![
            ShardConfig::new(0, 1, url_a),
            ShardConfig::new(2, 3, url_b),
        ])
        .expect("connect two shards")
    })
    .await
    .unwrap();

    let rp = router_proj.clone();
    let pn = pre_norm.clone();
    let h_c = h.clone();
    let remote_out = tokio::task::spawn_blocking(move || {
        let router = MoeRouterWeights {
            router_proj: &rp,
            router_scale: &[],
            router_per_expert_scale: &[],
            router_norm: &[],
            router_norm_parameter_free: false,
            router_input_scalar: 1.0,
            pre_experts_norm: &pn,
            post_experts_norm: &[],
            num_experts: NUM_EXPERTS,
            top_k: TOP_K,
        };
        backend.forward_moe(0, &h_c, &router, 0.0, 1e-6)
    })
    .await
    .unwrap()
    .expect("forward_moe two shards");

    let expected = local_output(&h, &gate_up, &down, &router_proj, &pre_norm);

    assert_eq!(remote_out.len(), expected.len());
    for (i, (&got, &exp)) in remote_out.iter().zip(expected.iter()).enumerate() {
        let diff = (got - exp).abs();
        assert!(
            diff < 1e-4,
            "output[{i}]: remote={got} local={exp} diff={diff:.2e}"
        );
    }
}

#[tokio::test]
async fn expert_endpoint_reshard_same_output() {
    let gate_up = make_gate_up_bytes();
    let down = make_down_bytes();
    let router_proj = make_router_proj();
    let pre_norm = make_pre_norm();
    let h = make_input();

    let url_a = spawn_server_with_model(make_loaded_model(
        gate_up.clone(),
        down.clone(),
        router_proj.clone(),
        pre_norm.clone(),
    ))
    .await;
    let url_b = spawn_server_with_model(make_loaded_model(
        gate_up.clone(),
        down.clone(),
        router_proj.clone(),
        pre_norm.clone(),
    ))
    .await;
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;

    let url_a_c = url_a.clone();
    let backend = Arc::new(
        tokio::task::spawn_blocking(move || {
            RemoteMoeBackend::connect(vec![ShardConfig::new(0, NUM_EXPERTS - 1, url_a_c)])
                .expect("connect A")
        })
        .await
        .unwrap(),
    );

    // First call on shard A.
    let rp = router_proj.clone();
    let pn = pre_norm.clone();
    let h_c = h.clone();
    let b = backend.clone();
    let out_a = tokio::task::spawn_blocking(move || {
        let router = MoeRouterWeights {
            router_proj: &rp,
            router_scale: &[],
            router_per_expert_scale: &[],
            router_norm: &[],
            router_norm_parameter_free: false,
            router_input_scalar: 1.0,
            pre_experts_norm: &pn,
            post_experts_norm: &[],
            num_experts: NUM_EXPERTS,
            top_k: TOP_K,
        };
        b.forward_moe(0, &h_c, &router, 0.0, 1e-6)
    })
    .await
    .unwrap()
    .expect("call on A");

    // Reshard to shard B.
    let b = backend.clone();
    tokio::task::spawn_blocking(move || {
        b.reshard(vec![ShardConfig::new(0, NUM_EXPERTS - 1, url_b)])
            .expect("reshard")
    })
    .await
    .unwrap();

    // Second call on shard B — same weights, must produce same output.
    let rp = router_proj.clone();
    let pn = pre_norm.clone();
    let h_c = h.clone();
    let b = backend.clone();
    let out_b = tokio::task::spawn_blocking(move || {
        let router = MoeRouterWeights {
            router_proj: &rp,
            router_scale: &[],
            router_per_expert_scale: &[],
            router_norm: &[],
            router_norm_parameter_free: false,
            router_input_scalar: 1.0,
            pre_experts_norm: &pn,
            post_experts_norm: &[],
            num_experts: NUM_EXPERTS,
            top_k: TOP_K,
        };
        b.forward_moe(0, &h_c, &router, 0.0, 1e-6)
    })
    .await
    .unwrap()
    .expect("call on B");

    assert_eq!(out_a.len(), out_b.len());
    for (i, (&a, &b)) in out_a.iter().zip(out_b.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "output[{i}] changed after reshard: {a} vs {b}"
        );
    }
}

#[tokio::test]
async fn expert_endpoint_no_shard_error() {
    let gate_up = make_gate_up_bytes();
    let down = make_down_bytes();
    let pre_norm = make_pre_norm();

    let url = spawn_server_with_model(make_loaded_model(
        gate_up,
        down,
        make_router_proj(),
        pre_norm.clone(),
    ))
    .await;
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;

    let url_c = url.clone();
    let backend = tokio::task::spawn_blocking(move || {
        // This shard only owns experts 0-1.
        RemoteMoeBackend::connect(vec![ShardConfig::new(0, 1, url_c)]).expect("connect")
    })
    .await
    .unwrap();

    // Router projection that makes expert 3 win overwhelmingly.
    let mut router_proj = vec![0.01f32; NUM_EXPERTS * HIDDEN];
    for j in 0..HIDDEN {
        router_proj[3 * HIDDEN + j] = 10.0;
    }

    let rp = router_proj.clone();
    let h = make_input();
    let err = tokio::task::spawn_blocking(move || {
        let router = MoeRouterWeights {
            router_proj: &rp,
            router_scale: &[],
            router_per_expert_scale: &[],
            router_norm: &[],
            router_norm_parameter_free: false,
            router_input_scalar: 1.0,
            pre_experts_norm: &pre_norm,
            post_experts_norm: &[],
            num_experts: NUM_EXPERTS,
            top_k: TOP_K,
        };
        backend.forward_moe(0, &h, &router, 0.0, 1e-6)
    })
    .await
    .unwrap();

    assert!(
        matches!(err, Err(RemoteMoeError::NoShard { expert_id: 3 })),
        "expected NoShard(3), got {err:?}"
    );
}

/// Regression test: `--experts 0-1` (CLI) → `expert_filter = (0, 2)` (the
/// half-open range `parse_layer_range` produces). The route handler must
/// REJECT expert 2 even though it's at the half-open upper bound — earlier
/// the inclusive `>` check let `expert_id == end` slip through, exposing a
/// neighbour shard's experts. Test covers boundaries: 0 (in), 1 (in, last),
/// 2 (out, off-by-one), 3 (out, far).
#[tokio::test]
async fn expert_filter_rejects_at_upper_bound() {
    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use larql_server::{
        cache::DescribeCache, routes::single_model_router, session::SessionManager, state::AppState,
    };
    use std::sync::atomic::AtomicU64;
    use tower::ServiceExt as _;

    let gate_up = make_gate_up_bytes();
    let down = make_down_bytes();
    let router_proj = make_router_proj();
    let pre_norm = make_pre_norm();
    let h = make_input();

    // Filter: (0, 2) = inclusive 0-1 = `--experts 0-1`.
    let model = make_loaded_model_with_filter(gate_up, down, router_proj, pre_norm, (0, 2));
    let state = Arc::new(AppState {
        models: vec![Arc::new(model)],
        started_at: std::time::Instant::now(),
        requests_served: AtomicU64::new(0),
        api_key: None,
        sessions: SessionManager::new(3600),
        describe_cache: DescribeCache::new(60),
    });
    let app = single_model_router(state);

    async fn call(app: axum::Router, h: &[f32], id: usize) -> (StatusCode, String) {
        let body_str = serde_json::json!({ "residual": h }).to_string();
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/v1/expert/0/{id}"))
                    .header("content-type", "application/json")
                    .body(Body::from(body_str))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let text = String::from_utf8_lossy(&bytes).to_string();
        (status, text)
    }

    let (s0, _) = call(app.clone(), &h, 0).await;
    assert_eq!(s0, StatusCode::OK, "expert 0 (in-range) must succeed");

    let (s1, _) = call(app.clone(), &h, 1).await;
    assert_eq!(s1, StatusCode::OK, "expert 1 (last in-range) must succeed");

    let (s2, body2) = call(app.clone(), &h, 2).await;
    assert_eq!(
        s2,
        StatusCode::BAD_REQUEST,
        "expert 2 (one past the inclusive end) MUST be rejected — \
         this catches the half-open vs inclusive off-by-one. body: {body2}"
    );
    assert!(
        body2.contains("not owned"),
        "rejection body must explain ownership: {body2}"
    );
    // Error message displays the inclusive bound, not the half-open one.
    assert!(
        body2.contains("0–1") || body2.contains("0-1"),
        "error message must show inclusive range 0–1, not 0–2; got: {body2}"
    );

    let (s3, _) = call(app, &h, 3).await;
    assert_eq!(
        s3,
        StatusCode::BAD_REQUEST,
        "expert 3 (well out of range) must be rejected"
    );
}
