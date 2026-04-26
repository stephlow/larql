//! HTTP integration tests: embed, logits, token encode/decode (single + multi).

mod common;
use common::*;

use axum::http::StatusCode;

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
// GET /v1/embed/{token_id}  (single-token lookup)
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_embed_single_get_returns_200() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/embed/0").await;
    assert_eq!(resp.status(), StatusCode::OK);
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
