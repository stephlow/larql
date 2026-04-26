//! HTTP-level integration tests for larql-server.
//!
//! Uses axum's tower::ServiceExt::oneshot pattern — requests are dispatched
//! in-process to the full router with no network socket. Every test builds a
//! synthetic in-memory VectorIndex (1 layer, 3 features, hidden=4).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::middleware;
use axum::response::IntoResponse;
use larql_server::auth::auth_middleware;
use larql_server::cache::DescribeCache;
use larql_server::error::ServerError;
use larql_server::ffn_l2_cache::FfnL2Cache;
use larql_server::routes::{multi_model_router, single_model_router};
use larql_server::session::SessionManager;
use larql_server::state::{AppState, LoadedModel};
use larql_vindex::{
    ndarray::Array2, ExtractLevel, FeatureMeta, LayerBands, PatchedVindex, QuantFormat,
    VectorIndex, VindexConfig, VindexLayerInfo,
};
use tower::ServiceExt;

// ══════════════════════════════════════════════════════════════
// Shared test infrastructure
// ══════════════════════════════════════════════════════════════

fn make_feature(token: &str, id: u32, score: f32) -> FeatureMeta {
    FeatureMeta {
        top_token: token.to_string(),
        top_token_id: id,
        c_score: score,
        top_k: vec![
            larql_models::TopKEntry { token: token.to_string(), token_id: id, logit: score },
            larql_models::TopKEntry { token: "also".into(), token_id: id + 1, logit: score * 0.5 },
        ],
    }
}

fn test_index() -> VectorIndex {
    let hidden = 4;
    let mut gate = Array2::<f32>::zeros((3, hidden));
    gate[[0, 0]] = 1.0; // Paris  → dim 0
    gate[[1, 1]] = 1.0; // French → dim 1
    gate[[2, 2]] = 1.0; // Europe → dim 2

    let meta: Vec<Option<FeatureMeta>> = vec![
        Some(make_feature("Paris",  100, 0.95)),
        Some(make_feature("French", 101, 0.88)),
        Some(make_feature("Europe", 102, 0.75)),
    ];

    VectorIndex::new(vec![Some(gate)], vec![Some(meta)], 1, hidden)
}

fn test_config() -> VindexConfig {
    VindexConfig {
        version: 2,
        model: "test/model-4".to_string(),
        family: "test".to_string(),
        source: None,
        checksums: None,
        num_layers: 1,
        hidden_size: 4,
        intermediate_size: 12,
        vocab_size: 8,
        embed_scale: 1.0,
        extract_level: ExtractLevel::Browse,
        dtype: larql_vindex::StorageDtype::default(),
        quant: QuantFormat::None,
        layer_bands: Some(LayerBands { syntax: (0, 0), knowledge: (0, 0), output: (0, 0) }),
        layers: vec![VindexLayerInfo {
            layer: 0, num_features: 3, offset: 0, length: 48,
            num_experts: None, num_features_per_expert: None,
        }],
        down_top_k: 5,
        has_model_weights: false,
        model_config: None,
        fp4: None,
    }
}

fn empty_tokenizer() -> larql_vindex::tokenizers::Tokenizer {
    let json = r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    larql_vindex::tokenizers::Tokenizer::from_bytes(json).unwrap()
}

struct ModelBuilder {
    id: String,
    ffn_only: bool,
    embed_only: bool,
    probe_labels: HashMap<(usize, usize), String>,
    config: VindexConfig,
}

impl ModelBuilder {
    fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            ffn_only: false,
            embed_only: false,
            probe_labels: HashMap::new(),
            config: test_config(),
        }
    }
    fn ffn_only(mut self) -> Self { self.ffn_only = true; self }
    fn embed_only(mut self) -> Self { self.embed_only = true; self }
    fn with_labels(mut self, labels: HashMap<(usize, usize), String>) -> Self {
        self.probe_labels = labels;
        self
    }
    fn build(self) -> Arc<LoadedModel> {
        Arc::new(LoadedModel {
            id: self.id,
            path: PathBuf::from("/nonexistent"),
            config: self.config,
            patched: tokio::sync::RwLock::new(PatchedVindex::new(test_index())),
            embeddings: {
                let mut e = Array2::<f32>::zeros((8, 4));
                e[[0, 0]] = 1.0;
                e[[1, 1]] = 1.0;
                e[[2, 2]] = 1.0;
                e[[3, 3]] = 1.0;
                e
            },
            embed_scale: 1.0,
            tokenizer: empty_tokenizer(),
            infer_disabled: true,
            ffn_only: self.ffn_only,
            embed_only: self.embed_only,
            embed_store: None,
            release_mmap_after_request: false,
            weights: std::sync::OnceLock::new(),
            probe_labels: self.probe_labels,
            ffn_l2_cache: FfnL2Cache::new(1),
            expert_filter: None,
        })
    }
}

fn model(id: &str) -> Arc<LoadedModel> { ModelBuilder::new(id).build() }

fn state(models: Vec<Arc<LoadedModel>>) -> Arc<AppState> {
    Arc::new(AppState {
        models,
        started_at: std::time::Instant::now(),
        requests_served: AtomicU64::new(0),
        api_key: None,
        sessions: SessionManager::new(3600),
        describe_cache: DescribeCache::new(0),
    })
}

fn state_with_key(models: Vec<Arc<LoadedModel>>, key: &str) -> Arc<AppState> {
    Arc::new(AppState {
        models,
        started_at: std::time::Instant::now(),
        requests_served: AtomicU64::new(0),
        api_key: Some(key.to_string()),
        sessions: SessionManager::new(3600),
        describe_cache: DescribeCache::new(0),
    })
}

async fn body_json(body: Body) -> serde_json::Value {
    let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
    serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
}

async fn get(app: axum::Router, path: &str) -> axum::http::Response<Body> {
    app.oneshot(Request::builder().method("GET").uri(path).body(Body::empty()).unwrap())
        .await.unwrap()
}

async fn get_h(app: axum::Router, path: &str, h: (&str, &str)) -> axum::http::Response<Body> {
    app.oneshot(
        Request::builder().method("GET").uri(path).header(h.0, h.1).body(Body::empty()).unwrap()
    ).await.unwrap()
}

async fn post_json(app: axum::Router, path: &str, body: serde_json::Value) -> axum::http::Response<Body> {
    app.oneshot(
        Request::builder()
            .method("POST").uri(path)
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap())).unwrap()
    ).await.unwrap()
}

async fn post_json_h(
    app: axum::Router, path: &str,
    body: serde_json::Value, h: (&str, &str),
) -> axum::http::Response<Body> {
    app.oneshot(
        Request::builder()
            .method("POST").uri(path)
            .header("content-type", "application/json")
            .header(h.0, h.1)
            .body(Body::from(serde_json::to_vec(&body).unwrap())).unwrap()
    ).await.unwrap()
}

async fn delete(app: axum::Router, path: &str) -> axum::http::Response<Body> {
    app.oneshot(Request::builder().method("DELETE").uri(path).body(Body::empty()).unwrap())
        .await.unwrap()
}

// ══════════════════════════════════════════════════════════════
// GET /v1/health
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_health_returns_200() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/health").await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn http_health_body_has_required_fields() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/health").await;
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["status"], "ok");
    assert!(body["uptime_seconds"].as_u64().is_some());
    assert!(body["requests_served"].as_u64().is_some());
}

#[tokio::test]
async fn http_health_bumps_request_counter() {
    let st = state(vec![model("test")]);
    let app = single_model_router(st.clone());
    get(app, "/v1/health").await;
    assert_eq!(st.requests_served.load(std::sync::atomic::Ordering::Relaxed), 1);
}

// ══════════════════════════════════════════════════════════════
// GET /v1/models
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_models_single_lists_one_model() {
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = get(app, "/v1/models").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let models = body["models"].as_array().unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0]["id"], "gemma");
    assert!(models[0]["features"].as_u64().is_some());
    assert_eq!(models[0]["loaded"], true);
}

#[tokio::test]
async fn http_models_single_path_is_v1() {
    let app = single_model_router(state(vec![model("m")]));
    let resp = get(app, "/v1/models").await;
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["models"][0]["path"], "/v1");
}

#[tokio::test]
async fn http_models_multi_path_includes_model_id() {
    let app = multi_model_router(state(vec![model("a"), model("b")]));
    let resp = get(app, "/v1/models").await;
    let body = body_json(resp.into_body()).await;
    let models = body["models"].as_array().unwrap();
    assert_eq!(models.len(), 2);
    // Multi-model paths are /v1/{id}
    let paths: Vec<&str> = models.iter()
        .map(|m| m["path"].as_str().unwrap()).collect();
    assert!(paths.contains(&"/v1/a"));
    assert!(paths.contains(&"/v1/b"));
}

// ══════════════════════════════════════════════════════════════
// GET /v1/stats
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_stats_returns_model_info() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/stats").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["model"], "test/model-4");
    assert_eq!(body["family"], "test");
    assert_eq!(body["layers"], 1);
    assert_eq!(body["features"], 3);
    assert_eq!(body["hidden_size"], 4);
    assert_eq!(body["vocab_size"], 8);
    assert!(body["layer_bands"].is_object());
}

#[tokio::test]
async fn http_stats_mode_full_by_default() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/stats").await;
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["mode"], "full");
    assert_eq!(body["loaded"]["ffn_service"], true);
}

#[tokio::test]
async fn http_stats_mode_ffn_service_when_ffn_only() {
    let m = ModelBuilder::new("test").ffn_only().build();
    let app = single_model_router(state(vec![m]));
    let resp = get(app, "/v1/stats").await;
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["mode"], "ffn-service");
    assert_eq!(body["loaded"]["inference"], false);
}

#[tokio::test]
async fn http_stats_mode_embed_service_when_embed_only() {
    let m = ModelBuilder::new("test").embed_only().build();
    let app = single_model_router(state(vec![m]));
    let resp = get(app, "/v1/stats").await;
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["mode"], "embed-service");
    assert_eq!(body["loaded"]["embed_service"], true);
    assert_eq!(body["loaded"]["browse"], false);
}

#[tokio::test]
async fn http_stats_layer_bands_shape() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/stats").await;
    let body = body_json(resp.into_body()).await;
    let bands = &body["layer_bands"];
    assert!(bands["syntax"].is_array());
    assert!(bands["knowledge"].is_array());
    assert!(bands["output"].is_array());
}

// ══════════════════════════════════════════════════════════════
// GET /v1/describe
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_describe_returns_200_with_entity_field() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/describe?entity=France").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["entity"], "France");
    assert!(body["edges"].is_array());
    assert!(body["latency_ms"].as_f64().is_some());
}

#[tokio::test]
async fn http_describe_empty_vocab_returns_empty_edges() {
    // Empty BPE tokenizer → empty token_ids → graceful empty response.
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/describe?entity=Germany").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["edges"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn http_describe_missing_entity_returns_400() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/describe").await; // no entity param
    // axum rejects the missing required query param
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

// ══════════════════════════════════════════════════════════════
// POST /v1/select
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_select_no_filter_returns_all_features() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(app, "/v1/select", serde_json::json!({})).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["total"], 3);
    let edges = body["edges"].as_array().unwrap();
    assert_eq!(edges.len(), 3);
    assert!(body["latency_ms"].as_f64().is_some());
}

#[tokio::test]
async fn http_select_layer_filter_returns_correct_features() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(app, "/v1/select", serde_json::json!({"layer": 0})).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["total"], 3); // 3 features at layer 0
    let edges = body["edges"].as_array().unwrap();
    for edge in edges {
        assert_eq!(edge["layer"], 0);
    }
}

#[tokio::test]
async fn http_select_entity_filter() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(app, "/v1/select", serde_json::json!({"entity": "Par"})).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let edges = body["edges"].as_array().unwrap();
    // Only "Paris" matches "Par" (case-insensitive substring).
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0]["target"].as_str().unwrap().trim(), "Paris");
}

#[tokio::test]
async fn http_select_min_confidence_filter() {
    let app = single_model_router(state(vec![model("test")]));
    // Only Paris (0.95) and French (0.88) pass min_confidence=0.85.
    let resp = post_json(app, "/v1/select", serde_json::json!({"min_confidence": 0.85})).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let edges = body["edges"].as_array().unwrap();
    assert_eq!(edges.len(), 2);
    for edge in edges {
        assert!(edge["c_score"].as_f64().unwrap() >= 0.85);
    }
}

#[tokio::test]
async fn http_select_limit_truncates_results() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(app, "/v1/select", serde_json::json!({"limit": 2})).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let edges = body["edges"].as_array().unwrap();
    assert_eq!(edges.len(), 2);
    assert_eq!(body["total"], 3); // total still 3, but truncated to 2
}

#[tokio::test]
async fn http_select_order_asc_returns_lowest_confidence_first() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(app, "/v1/select",
        serde_json::json!({"order_by": "confidence", "order": "asc"})).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let edges = body["edges"].as_array().unwrap();
    let scores: Vec<f64> = edges.iter().map(|e| e["c_score"].as_f64().unwrap()).collect();
    // Should be ascending.
    for i in 1..scores.len() {
        assert!(scores[i] >= scores[i - 1], "expected ascending: {:?}", scores);
    }
}

#[tokio::test]
async fn http_select_order_desc_returns_highest_confidence_first() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(app, "/v1/select",
        serde_json::json!({"order_by": "confidence", "order": "desc"})).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let edges = body["edges"].as_array().unwrap();
    let scores: Vec<f64> = edges.iter().map(|e| e["c_score"].as_f64().unwrap()).collect();
    for i in 1..scores.len() {
        assert!(scores[i] <= scores[i - 1], "expected descending: {:?}", scores);
    }
}

#[tokio::test]
async fn http_select_relation_filter_returns_labelled_features() {
    let mut labels = HashMap::new();
    labels.insert((0usize, 0usize), "capital".to_string());
    labels.insert((0usize, 1usize), "language".to_string());
    let m = ModelBuilder::new("test").with_labels(labels).build();
    let app = single_model_router(state(vec![m]));
    let resp = post_json(app, "/v1/select", serde_json::json!({"relation": "capital"})).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let edges = body["edges"].as_array().unwrap();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0]["relation"], "capital");
    assert_eq!(edges[0]["target"].as_str().unwrap().trim(), "Paris");
}

#[tokio::test]
async fn http_select_order_by_layer_asc() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(app, "/v1/select",
        serde_json::json!({"order_by": "layer", "order": "asc"})).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    // All features are at layer 0 in our 1-layer test index; ordering should succeed.
    assert!(body["edges"].is_array());
}

// ══════════════════════════════════════════════════════════════
// GET /v1/relations
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_relations_returns_json_structure() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/relations").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert!(body["relations"].is_array());
    assert!(body["probe_relations"].is_array());
    assert!(body["total"].as_u64().is_some());
    assert!(body["probe_count"].as_u64().is_some());
    assert!(body["latency_ms"].as_f64().is_some());
}

#[tokio::test]
async fn http_relations_probe_count_reflects_labels() {
    let mut labels = HashMap::new();
    labels.insert((0usize, 0usize), "capital".to_string());
    labels.insert((0usize, 1usize), "language".to_string());
    let m = ModelBuilder::new("test").with_labels(labels).build();
    let app = single_model_router(state(vec![m]));
    let resp = get(app, "/v1/relations").await;
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["probe_count"], 2);
    let probe_rels = body["probe_relations"].as_array().unwrap();
    let names: Vec<&str> = probe_rels.iter().map(|r| r["name"].as_str().unwrap()).collect();
    assert!(names.contains(&"capital"));
    assert!(names.contains(&"language"));
}

// ══════════════════════════════════════════════════════════════
// GET /v1/patches
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_patches_list_empty_returns_empty_array() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/patches").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let patches = body["patches"].as_array().unwrap();
    assert!(patches.is_empty());
}

#[tokio::test]
async fn http_patches_delete_nonexistent_returns_404() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = delete(app, "/v1/patches/nonexistent-patch").await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn http_patches_session_list_returns_session_field() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get_h(app, "/v1/patches", ("x-session-id", "sess-abc")).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["session"], "sess-abc");
    assert!(body["patches"].as_array().unwrap().is_empty());
}

// ══════════════════════════════════════════════════════════════
// MULTI-MODEL ROUTES (/v1/{model_id}/...)
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_multi_health_returns_200() {
    let app = multi_model_router(state(vec![model("a"), model("b")]));
    let resp = get(app, "/v1/health").await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn http_multi_models_lists_both() {
    let app = multi_model_router(state(vec![model("a"), model("b")]));
    let resp = get(app, "/v1/models").await;
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["models"].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn http_multi_stats_valid_model_returns_200() {
    let app = multi_model_router(state(vec![model("alpha"), model("beta")]));
    let resp = get(app, "/v1/alpha/stats").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["model"], "test/model-4");
}

#[tokio::test]
async fn http_multi_stats_unknown_model_returns_404() {
    let app = multi_model_router(state(vec![model("a")]));
    let resp = get(app, "/v1/unknown/stats").await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn http_multi_select_all_features() {
    let app = multi_model_router(state(vec![model("m1"), model("m2")]));
    let resp = post_json(app, "/v1/m1/select", serde_json::json!({})).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["total"], 3);
}

#[tokio::test]
async fn http_multi_describe_returns_entity() {
    let app = multi_model_router(state(vec![model("mymodel")]));
    let resp = get(app, "/v1/mymodel/describe?entity=France").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["entity"], "France");
}

// ══════════════════════════════════════════════════════════════
// AUTH MIDDLEWARE
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_auth_no_api_key_configured_allows_all() {
    // No api_key in state → middleware passes everything.
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/stats").await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn http_auth_correct_bearer_returns_200() {
    let st = state_with_key(vec![model("test")], "secret123");
    let app = single_model_router(st.clone())
        .layer(middleware::from_fn_with_state(st, auth_middleware));
    let resp = get_h(app, "/v1/stats", ("authorization", "Bearer secret123")).await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn http_auth_wrong_bearer_returns_401() {
    let st = state_with_key(vec![model("test")], "secret123");
    let app = single_model_router(st.clone())
        .layer(middleware::from_fn_with_state(st, auth_middleware));
    let resp = get_h(app, "/v1/stats", ("authorization", "Bearer wrongkey")).await;
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn http_auth_missing_header_returns_401() {
    let st = state_with_key(vec![model("test")], "secret123");
    let app = single_model_router(st.clone())
        .layer(middleware::from_fn_with_state(st, auth_middleware));
    let resp = get(app, "/v1/stats").await; // no auth header
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn http_auth_health_exempt_without_key() {
    let st = state_with_key(vec![model("test")], "secret123");
    let app = single_model_router(st.clone())
        .layer(middleware::from_fn_with_state(st, auth_middleware));
    // /v1/health must be reachable even without auth.
    let resp = get(app, "/v1/health").await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn http_auth_non_bearer_format_rejected() {
    let st = state_with_key(vec![model("test")], "secret123");
    let app = single_model_router(st.clone())
        .layer(middleware::from_fn_with_state(st, auth_middleware));
    let resp = get_h(app, "/v1/stats", ("authorization", "Token secret123")).await;
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

// ══════════════════════════════════════════════════════════════
// POST /v1/embed
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_embed_valid_token_ids_returns_200() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(app, "/v1/embed", serde_json::json!({"token_ids": [0, 1, 2]})).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["seq_len"], 3);
    assert_eq!(body["hidden_size"], 4);
    assert!(body["residual"].is_array());
}

#[tokio::test]
async fn http_embed_empty_token_ids_returns_400() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(app, "/v1/embed", serde_json::json!({"token_ids": []})).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_embed_out_of_range_token_returns_400() {
    // vocab_size=8, token_id=100 is out of range.
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(app, "/v1/embed", serde_json::json!({"token_ids": [100]})).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_embed_single_token_returns_correct_shape() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(app, "/v1/embed", serde_json::json!({"token_ids": [0]})).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    // seq_len=1, hidden_size=4 → residual[0] has 4 values.
    let row = body["residual"][0].as_array().unwrap();
    assert_eq!(row.len(), 4);
}

// ══════════════════════════════════════════════════════════════
// GET /v1/token/decode
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_token_decode_empty_ids_returns_200() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/token/decode?ids=").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert!(body["token_ids"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn http_token_decode_invalid_id_returns_400() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/token/decode?ids=notanumber").await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_token_decode_missing_ids_param_returns_400() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/token/decode").await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

// ══════════════════════════════════════════════════════════════
// GET /v1/token/encode
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_token_encode_returns_200() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/token/encode?text=hello").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["text"], "hello");
    assert!(body["token_ids"].is_array());
}

#[tokio::test]
async fn http_token_encode_missing_text_returns_400() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/token/encode").await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

// ══════════════════════════════════════════════════════════════
// GET /v1/embed/{token_id}  (single-token lookup)
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_embed_single_get_returns_200() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/embed/0").await;
    assert_eq!(resp.status(), StatusCode::OK);
}

// ══════════════════════════════════════════════════════════════
// ASYNC STATE / SESSION MANAGER TESTS
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn session_manager_list_empty_for_unknown_session() {
    let sm = SessionManager::new(3600);
    let patches = sm.list_patches("session-xyz").await;
    assert!(patches.is_empty());
}

#[tokio::test]
async fn session_manager_apply_patch_and_list() {
    let sm = SessionManager::new(3600);
    let m = model("test");

    // Pre-create the session with get_or_create (uses read().await, safe in async).
    // apply_patch's or_insert_with calls blocking_read only when the session doesn't
    // exist, so we must create it first.
    sm.get_or_create("sess-1", &m).await;

    let patch = larql_vindex::VindexPatch {
        version: 1,
        base_model: "test".into(),
        base_checksum: None,
        created_at: "2026-04-26".into(),
        description: Some("my-patch".into()),
        author: None,
        tags: vec![],
        operations: vec![larql_vindex::PatchOp::Delete { layer: 0, feature: 0, reason: None }],
    };

    let (op_count, active) = sm.apply_patch("sess-1", &m, patch).await;
    assert_eq!(op_count, 1);
    assert_eq!(active, 1);

    let list = sm.list_patches("sess-1").await;
    assert_eq!(list.len(), 1);
    assert_eq!(list[0]["name"], "my-patch");
}

#[tokio::test]
async fn session_manager_remove_nonexistent_patch_returns_err() {
    let sm = SessionManager::new(3600);
    let m = model("test");
    // Pre-create the session, then apply one patch.
    sm.get_or_create("sess-1", &m).await;
    let patch = larql_vindex::VindexPatch {
        version: 1,
        base_model: "test".into(),
        base_checksum: None,
        created_at: "2026-04-26".into(),
        description: Some("my-patch".into()),
        author: None,
        tags: vec![],
        operations: vec![larql_vindex::PatchOp::Delete { layer: 0, feature: 0, reason: None }],
    };
    sm.apply_patch("sess-1", &m, patch).await;

    let err = sm.remove_patch("sess-1", "nonexistent").await;
    assert!(err.is_err());
    assert!(err.unwrap_err().contains("not found"));
}

#[tokio::test]
async fn session_manager_remove_patch_by_name() {
    let sm = SessionManager::new(3600);
    let m = model("test");

    // Pre-create session, then apply two patches.
    sm.get_or_create("sess-2", &m).await;
    for name in &["patch-a", "patch-b"] {
        let patch = larql_vindex::VindexPatch {
            version: 1,
            base_model: "test".into(),
            base_checksum: None,
            created_at: "2026-04-26".into(),
            description: Some((*name).into()),
            author: None,
            tags: vec![],
            operations: vec![larql_vindex::PatchOp::Delete { layer: 0, feature: 1, reason: None }],
        };
        sm.apply_patch("sess-2", &m, patch).await;
    }

    let remaining = sm.remove_patch("sess-2", "patch-a").await.unwrap();
    assert_eq!(remaining, 1);

    let list = sm.list_patches("sess-2").await;
    assert_eq!(list.len(), 1);
    assert_eq!(list[0]["name"], "patch-b");
}

#[tokio::test]
async fn session_manager_remove_from_unknown_session_returns_err() {
    let sm = SessionManager::new(3600);
    let err = sm.remove_patch("no-such-session", "any-patch").await;
    assert!(err.is_err());
    assert!(err.unwrap_err().contains("not found"));
}

// ══════════════════════════════════════════════════════════════
// SERVER ERROR → HTTP RESPONSE (async body read)
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_server_error_not_found_body_has_error_key() {
    let resp = ServerError::NotFound("entity not found".into()).into_response();
    let status = resp.status();
    let body = body_json(resp.into_body()).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(body["error"].as_str().unwrap().contains("entity not found"));
}

#[tokio::test]
async fn http_server_error_bad_request_body_has_error_key() {
    let resp = ServerError::BadRequest("invalid param".into()).into_response();
    let status = resp.status();
    let body = body_json(resp.into_body()).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"].as_str().unwrap().contains("invalid param"));
}

#[tokio::test]
async fn http_server_error_internal_body_has_error_key() {
    let resp = ServerError::Internal("disk failure".into()).into_response();
    let status = resp.status();
    let body = body_json(resp.into_body()).await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert!(body["error"].as_str().unwrap().contains("disk failure"));
}

#[tokio::test]
async fn http_server_error_unavailable_body_has_error_key() {
    let resp = ServerError::InferenceUnavailable("no weights loaded".into()).into_response();
    let status = resp.status();
    let body = body_json(resp.into_body()).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert!(body["error"].as_str().unwrap().contains("no weights loaded"));
}

// ══════════════════════════════════════════════════════════════
// REQUEST COUNTER (ensure all routes bump it)
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_requests_served_increments_per_request() {
    let st = state(vec![model("test")]);
    let before = st.requests_served.load(std::sync::atomic::Ordering::Relaxed);

    let app = single_model_router(st.clone());
    get(app, "/v1/health").await;

    let after = st.requests_served.load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(after, before + 1);
}

#[tokio::test]
async fn http_select_increments_request_counter() {
    let st = state(vec![model("test")]);
    let app = single_model_router(st.clone());
    post_json(app, "/v1/select", serde_json::json!({})).await;
    assert_eq!(st.requests_served.load(std::sync::atomic::Ordering::Relaxed), 1);
}

// ══════════════════════════════════════════════════════════════
// LOAD PROBE LABELS (async round-trip via file I/O)
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_load_probe_labels_roundtrip() {
    use larql_server::state::load_probe_labels;
    let dir = std::env::temp_dir().join("larql_http_labels_01");
    tokio::fs::create_dir_all(&dir).await.unwrap();
    let json = r#"{"L0_F0":"capital","L1_F2":"language"}"#;
    tokio::fs::write(dir.join("feature_labels.json"), json).await.unwrap();

    let labels = load_probe_labels(&dir);
    assert_eq!(labels.get(&(0, 0)), Some(&"capital".to_string()));
    assert_eq!(labels.get(&(1, 2)), Some(&"language".to_string()));

    let _ = tokio::fs::remove_dir_all(&dir).await;
}
