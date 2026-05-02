//! `POST /v1/embeddings` — OpenAI-compatible embeddings endpoint (N0.4).
//!
//! Implements the [OpenAI Embeddings API](https://platform.openai.com/docs/api-reference/embeddings/create)
//! shape so existing `openai` SDKs work unmodified:
//!
//! ```python
//! from openai import OpenAI
//! client = OpenAI(base_url="http://larql:8080/v1", api_key="sk-...")
//! resp = client.embeddings.create(model="gemma-3-4b", input="hello world")
//! ```
//!
//! ## Pooling semantics
//!
//! OpenAI's text-embedding models output one pooled vector per input.
//! This endpoint emulates that by **mean-pooling** the per-token static
//! embeddings (`embeddings.bin` row lookup) over the input sequence.
//! Static embeddings are not the same as a contrastively-trained sentence
//! encoder — clients should treat results as "lookup-pooled" rather than
//! "semantic" embeddings until a dedicated embedding head is added.
//!
//! For per-token embeddings (no pooling), use the native `/v1/embed`
//! endpoint instead.
//!
//! ## Input variants supported
//!
//! - `string` — one input
//! - `string[]` — batched inputs
//! - `int[]` — one pre-tokenised input
//! - `int[][]` — batched pre-tokenised inputs
//!
//! ## Encoding format
//!
//! `encoding_format: "float"` (default) is supported. `"base64"` returns
//! HTTP 400 — follow-up. For now, clients should request floats.

use std::sync::Arc;

use axum::extract::State;
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::error::ServerError;
use crate::state::{AppState, LoadedModel};

use crate::routes::embed::embed_tokens;

const EMBEDDING_OBJECT: &str = "embedding";
const LIST_OBJECT: &str = "list";

#[derive(Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Batch(Vec<String>),
    SingleTokens(Vec<u32>),
    BatchTokens(Vec<Vec<u32>>),
}

#[derive(Deserialize)]
pub struct EmbeddingsRequest {
    /// Model id (matched against the loaded model's id; ignored in
    /// single-model mode).
    pub model: Option<String>,
    pub input: EmbeddingInput,
    /// Only `"float"` is currently supported. `"base64"` returns 400.
    #[serde(default)]
    pub encoding_format: Option<String>,
    /// Optional caller-supplied dimensionality. Larql ignores this — the
    /// vector size is the model's `hidden_size`. Logged but not enforced.
    #[serde(default)]
    pub dimensions: Option<usize>,
    /// Optional end-user id (OpenAI tracks this for abuse monitoring;
    /// larql logs it via tracing if set, otherwise no-op).
    #[serde(default)]
    pub user: Option<String>,
}

#[derive(Serialize)]
pub struct EmbeddingObject {
    pub object: &'static str,
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Serialize)]
pub struct EmbeddingsUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Serialize)]
pub struct EmbeddingsResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    pub usage: EmbeddingsUsage,
}

pub async fn handle_embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbeddingsRequest>,
) -> Result<Json<EmbeddingsResponse>, ServerError> {
    state.bump_requests();

    if let Some(fmt) = req.encoding_format.as_deref() {
        if fmt != "float" {
            return Err(ServerError::BadRequest(format!(
                "encoding_format='{fmt}' not supported yet (only 'float'); base64 follow-up"
            )));
        }
    }

    let model = state.model_or_err(req.model.as_deref())?;

    // Resolve input to one or more token-id sequences. Strings get
    // tokenised; pre-tokenised inputs pass through.
    let model_ref: &LoadedModel = model.as_ref();
    let token_seqs: Vec<Vec<u32>> = match req.input {
        EmbeddingInput::Single(s) => vec![tokenize_one(model_ref, &s)?],
        EmbeddingInput::Batch(strs) => strs
            .iter()
            .map(|s| tokenize_one(model_ref, s))
            .collect::<Result<_, _>>()?,
        EmbeddingInput::SingleTokens(ids) => vec![ids],
        EmbeddingInput::BatchTokens(idses) => idses,
    };

    if token_seqs.iter().all(|s| s.is_empty()) {
        return Err(ServerError::BadRequest("input is empty".into()));
    }

    let mut data = Vec::with_capacity(token_seqs.len());
    let mut total_tokens = 0usize;
    for (idx, ids) in token_seqs.iter().enumerate() {
        if ids.is_empty() {
            return Err(ServerError::BadRequest(format!(
                "input[{idx}] is empty — every input must have ≥1 token"
            )));
        }
        let h = embed_tokens(model_ref, ids)?;
        let pooled = mean_pool(&h);
        total_tokens += ids.len();
        data.push(EmbeddingObject {
            object: EMBEDDING_OBJECT,
            embedding: pooled,
            index: idx,
        });
    }

    Ok(Json(EmbeddingsResponse {
        object: LIST_OBJECT,
        data,
        model: model.id.clone(),
        usage: EmbeddingsUsage {
            prompt_tokens: total_tokens,
            total_tokens,
        },
    }))
}

fn tokenize_one(model: &LoadedModel, text: &str) -> Result<Vec<u32>, ServerError> {
    let enc = model
        .tokenizer
        .encode(text, false)
        .map_err(|e| ServerError::Internal(format!("tokenize: {e}")))?;
    Ok(enc.get_ids().to_vec())
}

/// Mean pool a `[seq_len × hidden]` matrix to a `[hidden]` vector.
/// Returns zeros for empty sequences (caller should reject upstream).
fn mean_pool(h: &larql_vindex::ndarray::Array2<f32>) -> Vec<f32> {
    let seq_len = h.shape()[0];
    let hidden = h.shape()[1];
    if seq_len == 0 {
        return vec![0.0; hidden];
    }
    let mut out = vec![0.0f32; hidden];
    for row in h.rows() {
        for (a, &v) in out.iter_mut().zip(row.iter()) {
            *a += v;
        }
    }
    let inv_n = 1.0 / seq_len as f32;
    for v in out.iter_mut() {
        *v *= inv_n;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_vindex::ndarray::array;

    #[test]
    fn mean_pool_single_row_returns_row() {
        let h = array![[1.0f32, 2.0, 3.0]];
        let pooled = mean_pool(&h);
        assert_eq!(pooled, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn mean_pool_two_rows_averages_per_column() {
        let h = array![[1.0f32, 4.0], [3.0, 6.0]];
        let pooled = mean_pool(&h);
        assert_eq!(pooled, vec![2.0, 5.0]);
    }

    #[test]
    fn mean_pool_empty_sequence_returns_zero_vector() {
        let h: larql_vindex::ndarray::Array2<f32> = larql_vindex::ndarray::Array2::zeros((0, 4));
        let pooled = mean_pool(&h);
        assert_eq!(pooled, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn embedding_input_deserializes_single_string() {
        let json = serde_json::json!({"input": "hello"});
        let req: EmbeddingsRequest = serde_json::from_value(json).unwrap();
        match req.input {
            EmbeddingInput::Single(s) => assert_eq!(s, "hello"),
            _ => panic!("expected Single"),
        }
    }

    #[test]
    fn embedding_input_deserializes_string_batch() {
        let json = serde_json::json!({"input": ["a", "b"]});
        let req: EmbeddingsRequest = serde_json::from_value(json).unwrap();
        match req.input {
            EmbeddingInput::Batch(v) => assert_eq!(v, vec!["a", "b"]),
            _ => panic!("expected Batch"),
        }
    }

    #[test]
    fn embedding_input_deserializes_pretokenised_single() {
        let json = serde_json::json!({"input": [1, 2, 3]});
        let req: EmbeddingsRequest = serde_json::from_value(json).unwrap();
        match req.input {
            EmbeddingInput::SingleTokens(v) => assert_eq!(v, vec![1, 2, 3]),
            other => panic!(
                "expected SingleTokens, got {:?}",
                std::mem::discriminant(&other)
            ),
        }
    }

    #[test]
    fn embedding_input_deserializes_pretokenised_batch() {
        let json = serde_json::json!({"input": [[1, 2], [3, 4]]});
        let req: EmbeddingsRequest = serde_json::from_value(json).unwrap();
        match req.input {
            EmbeddingInput::BatchTokens(v) => assert_eq!(v, vec![vec![1, 2], vec![3, 4]]),
            _ => panic!("expected BatchTokens"),
        }
    }
}
