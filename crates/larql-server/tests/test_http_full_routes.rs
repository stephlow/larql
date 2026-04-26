//! HTTP integration tests using the functional tokenizer.
//!
//! These tests cover routes that need real tokenization to return
//! non-empty results: walk, describe (with edges), and insert.
//! The empty BPE tokenizer in the default model() helper produces no
//! token IDs, causing walk to return 400 and describe to return empty edges.
//! model_functional() uses a WordLevel tokenizer with a small vocabulary,
//! so "France" → token 0, which maps to the [1,0,0,0] embedding row and
//! matches gate feature 0 ("Paris").

mod common;
use common::*;

use axum::http::StatusCode;

// ══════════════════════════════════════════════════════════════
// GET /v1/walk — functional tokenizer
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_walk_functional_returns_hits() {
    let app = single_model_router(state(vec![model_functional("test")]));
    let resp = get(app, "/v1/walk?prompt=France").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert!(body["hits"].is_array(), "response must have a 'hits' array");
}

#[tokio::test]
async fn http_walk_functional_hits_contain_paris() {
    let app = single_model_router(state(vec![model_functional("test")]));
    let resp = get(app, "/v1/walk?prompt=France").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let hits = body["hits"].as_array().unwrap();
    assert!(!hits.is_empty(), "expected at least one hit for 'France'");
    // The top hit should be "Paris" (feature 0, gate [1,0,0,0] matches embed row 0)
    let targets: Vec<&str> = hits.iter()
        .filter_map(|h| h["target"].as_str())
        .collect();
    assert!(
        targets.contains(&"Paris"),
        "expected 'Paris' in walk hits, got: {:?}", targets
    );
}

#[tokio::test]
async fn http_walk_functional_with_layer_range() {
    let app = single_model_router(state(vec![model_functional("test")]));
    let resp = get(app, "/v1/walk?prompt=France&layers=0-0").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert!(body["hits"].is_array());
}

#[tokio::test]
async fn http_walk_functional_with_layer_list() {
    let app = single_model_router(state(vec![model_functional("test")]));
    let resp = get(app, "/v1/walk?prompt=France&layers=0").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert!(body["hits"].is_array());
}

#[tokio::test]
async fn http_walk_functional_with_oob_layer() {
    // Layer 99 doesn't exist (only layer 0 loaded) — hits should be empty
    let app = single_model_router(state(vec![model_functional("test")]));
    let resp = get(app, "/v1/walk?prompt=France&layers=99").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let hits = body["hits"].as_array().unwrap();
    assert!(hits.is_empty(), "out-of-range layer should return empty hits");
}

#[tokio::test]
async fn http_walk_functional_multi_model() {
    let app = multi_model_router(state(vec![model_functional("a"), model_functional("b")]));
    let resp = get(app, "/v1/a/walk?prompt=France").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert!(body["hits"].is_array());
}

#[tokio::test]
async fn http_walk_multi_model_not_found() {
    let app = multi_model_router(state(vec![model_functional("a")]));
    let resp = get(app, "/v1/nosuchmodel/walk?prompt=France").await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// ══════════════════════════════════════════════════════════════
// GET /v1/describe — functional tokenizer (min_score=0 bypasses 5.0 default)
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_describe_functional_returns_edges() {
    let app = single_model_router(state(vec![model_functional("test")]));
    let resp = get(app, "/v1/describe?entity=France&min_score=0").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let edges = body["edges"].as_array().unwrap();
    assert!(!edges.is_empty(), "expected non-empty edges for 'France' with min_score=0");
}

#[tokio::test]
async fn http_describe_functional_paris_edge() {
    let app = single_model_router(state(vec![model_functional("test")]));
    let resp = get(app, "/v1/describe?entity=France&min_score=0").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let edges = body["edges"].as_array().unwrap();
    let targets: Vec<&str> = edges.iter()
        .filter_map(|e| e["target"].as_str())
        .collect();
    assert!(
        targets.contains(&"Paris"),
        "expected 'Paris' in describe edges, got: {:?}", targets
    );
}

#[tokio::test]
async fn http_describe_functional_band_syntax() {
    let app = single_model_router(state(vec![model_functional("test")]));
    let resp = get(app, "/v1/describe?entity=France&band=syntax&min_score=0").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert!(body["edges"].is_array());
}

#[tokio::test]
async fn http_describe_functional_band_output() {
    let app = single_model_router(state(vec![model_functional("test")]));
    let resp = get(app, "/v1/describe?entity=France&band=output&min_score=0").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert!(body["edges"].is_array());
}

#[tokio::test]
async fn http_describe_functional_band_all() {
    let app = single_model_router(state(vec![model_functional("test")]));
    let resp = get(app, "/v1/describe?entity=France&band=all&min_score=0").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert!(body["edges"].is_array());
}

#[tokio::test]
async fn http_describe_functional_verbose() {
    let app = single_model_router(state(vec![model_functional("test")]));
    let resp = get(app, "/v1/describe?entity=France&verbose=true&min_score=0").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let edges = body["edges"].as_array().unwrap();
    // With verbose=true each edge should have a "count" field
    if !edges.is_empty() {
        assert!(
            edges[0]["count"].as_u64().is_some(),
            "verbose mode should include 'count' field in each edge"
        );
    }
}

#[tokio::test]
async fn http_describe_functional_min_score_filter() {
    // min_score=100 is far above any gate score (max 0.95 in test_index)
    let app = single_model_router(state(vec![model_functional("test")]));
    let resp = get(app, "/v1/describe?entity=France&min_score=100").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let edges = body["edges"].as_array().unwrap();
    assert!(edges.is_empty(), "min_score=100 should filter all edges (max score is 0.95)");
}

#[tokio::test]
async fn http_describe_functional_self_ref_filtered() {
    // The describe handler filters out edges where the target == the entity
    // "Paris" as entity: gate feature 0 is "Paris", which should be filtered out
    let app = single_model_router(state(vec![model_functional("test")]));
    let resp = get(app, "/v1/describe?entity=Paris&min_score=0").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let edges = body["edges"].as_array().unwrap();
    let targets: Vec<&str> = edges.iter()
        .filter_map(|e| e["target"].as_str())
        .collect();
    assert!(
        !targets.iter().any(|t| t.to_lowercase() == "paris"),
        "self-reference 'Paris' should be filtered from describe results"
    );
}

#[tokio::test]
async fn http_describe_functional_multi_model() {
    let app = multi_model_router(state(vec![model_functional("a"), model_functional("b")]));
    let resp = get(app, "/v1/a/describe?entity=France&min_score=0").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["entity"], "France");
    assert!(body["edges"].is_array());
}

// ══════════════════════════════════════════════════════════════
// POST /v1/insert — functional tokenizer
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_insert_functional_with_tokenizer() {
    // Insert still works (embedding fallback) with the functional tokenizer
    let app = single_model_router(state(vec![model_functional("test")]));
    let resp = post_json(app, "/v1/insert", serde_json::json!({
        "entity": "France",
        "relation": "capital",
        "target": "Paris"
    })).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["entity"], "France");
    assert_eq!(body["target"], "Paris");
    assert!(body["inserted"].as_u64().is_some());
}

// ══════════════════════════════════════════════════════════════
// GET /v1/walk — prompt field in response
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_walk_functional_response_has_prompt_field() {
    let app = single_model_router(state(vec![model_functional("test")]));
    let resp = get(app, "/v1/walk?prompt=France").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["prompt"], "France");
    assert!(body["latency_ms"].as_f64().is_some());
}
