//! HTTP integration tests: embed, logits, token encode/decode (single + multi).

mod common;
use common::*;

use axum::body::Body;
use axum::http::Request;
use axum::http::StatusCode;
use larql_server::http::BINARY_FFN_CONTENT_TYPE;
use tower::ServiceExt;

fn binary_embed_body(token_ids: &[u32]) -> Vec<u8> {
    let mut body = Vec::with_capacity(4 + token_ids.len() * 4);
    body.extend_from_slice(&(token_ids.len() as u32).to_le_bytes());
    for &token_id in token_ids {
        body.extend_from_slice(&token_id.to_le_bytes());
    }
    body
}

fn binary_logits_body(values: &[f32]) -> Vec<u8> {
    let mut body = Vec::with_capacity(values.len() * 4);
    for &value in values {
        body.extend_from_slice(&value.to_le_bytes());
    }
    body
}

async fn post_binary(app: axum::Router, path: &str, body: Vec<u8>) -> axum::http::Response<Body> {
    app.oneshot(
        Request::builder()
            .method("POST")
            .uri(path)
            .header("content-type", BINARY_FFN_CONTENT_TYPE)
            .body(Body::from(body))
            .unwrap(),
    )
    .await
    .unwrap()
}

// ══════════════════════════════════════════════════════════════
// POST /v1/embed
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_embed_valid_token_ids_returns_200() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(
        app,
        "/v1/embed",
        serde_json::json!({"token_ids": [0, 1, 2]}),
    )
    .await;
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

#[tokio::test]
async fn http_embed_invalid_json_returns_400() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embed")
                .header("content-type", "application/json")
                .body(Body::from("{not json"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_embed_no_model_returns_404() {
    let app = single_model_router(state(vec![]));
    let resp = post_json(app, "/v1/embed", serde_json::json!({"token_ids": [0]})).await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn http_embed_binary_returns_binary_response() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_binary(app, "/v1/embed", binary_embed_body(&[0, 1])).await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok()),
        Some(BINARY_FFN_CONTENT_TYPE)
    );
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 2);
    assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 4);
    assert_eq!(bytes.len(), 8 + 2 * 4 * 4);
}

#[tokio::test]
async fn http_embed_binary_truncated_returns_400() {
    let app = single_model_router(state(vec![model("test")]));
    let mut body = Vec::new();
    body.extend_from_slice(&2u32.to_le_bytes());
    body.extend_from_slice(&0u32.to_le_bytes());
    let resp = post_binary(app, "/v1/embed", body).await;
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

#[tokio::test]
async fn http_embed_single_get_json_accept_returns_json() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get_h(app, "/v1/embed/0", ("accept", "application/json")).await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers()
            .get("cache-control")
            .and_then(|v| v.to_str().ok()),
        Some("public, max-age=31536000, immutable")
    );
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["token_id"], 0);
    assert_eq!(body["hidden_size"], 4);
}

#[tokio::test]
async fn http_embed_single_get_out_of_range_returns_400() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/embed/100").await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_multi_embed_single_get_unknown_model_returns_404() {
    let app = multi_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/missing/embed/0").await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// ══════════════════════════════════════════════════════════════
// POST /v1/logits
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_logits_invalid_json_returns_400() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/logits")
                .header("content-type", "application/json")
                .body(Body::from("{bad"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_logits_binary_odd_length_returns_400() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_binary(app, "/v1/logits", vec![0, 1, 2, 3, 4]).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_logits_hidden_mismatch_returns_400() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(
        app,
        "/v1/logits",
        serde_json::json!({"residual": [1.0, 2.0], "top_k": 2}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_logits_binary_hidden_mismatch_returns_400() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_binary(app, "/v1/logits", binary_logits_body(&[1.0, 2.0])).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_logits_no_model_returns_404() {
    let app = single_model_router(state(vec![]));
    let resp = post_json(
        app,
        "/v1/logits",
        serde_json::json!({"residual": [0.0, 0.0, 0.0, 0.0]}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
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
