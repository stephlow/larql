//! `POST /v1/completions` — OpenAI-compatible legacy text completions (N0.2).
//!
//! Implements the [OpenAI Completions API](https://platform.openai.com/docs/api-reference/completions/create)
//! shape so existing `openai` SDKs and eval harnesses work unmodified:
//!
//! ```python
//! from openai import OpenAI
//! client = OpenAI(base_url="http://larql:8080/v1", api_key="sk-...")
//! resp = client.completions.create(
//!     model="gemma-3-4b",
//!     prompt="The capital of France is",
//!     max_tokens=10,
//! )
//! ```
//!
//! ## Generation path (slice 1 + 3)
//!
//! Slice 1 runs an **un-KV-cached generation loop** —
//! `larql_inference::forward::predict_with_temperature` is invoked
//! once per generated token, re-running the full forward pass each
//! step. Cost is O(N²) in context length. Functional and immutable
//! (`&ModelWeights`-only), so it serializes cleanly with concurrent
//! `/v1/infer` traffic.
//!
//! The fast KV-cached path (`larql_inference::layer_graph::generate`)
//! requires `&mut ModelWeights` for the per-layer Q4_K dequant cache.
//! Wiring that into `LoadedModel` requires putting `ModelWeights` behind
//! a `RwLock`; roadmap'd as N0.2-fast.
//!
//! ## Streaming (slice 3)
//!
//! `stream: true` returns an SSE response — `text/event-stream` with
//! one `data: {chunk}\n\n` event per generated token, terminated by
//! `data: [DONE]\n\n`. Each chunk's shape mirrors the OpenAI
//! Completions stream: `{id, object: "text_completion", created,
//! model, choices: [{text, index, finish_reason, logprobs: null}]}`.
//! The final chunk before `[DONE]` carries `finish_reason: "stop" |
//! "length"`.
//!
//! Generation runs on the blocking pool; the stream channel is
//! capacity-bounded so the producer back-pressures naturally on slow
//! clients. Client disconnect cleans up early on the next
//! `blocking_send` failure.
//!
//! ## Logprobs
//!
//! `logprobs: int` returns `null` in the response. Top-k log-probabilities
//! over the lm_head distribution land in F18.

use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt as _;

use crate::error::ServerError;
use crate::state::{AppState, LoadedModel};

use super::util::{contains_any, error_chunk, new_id_suffix, trim_at_stop, unix_now, StopSpec};

const TEXT_COMPLETION_OBJECT: &str = "text_completion";
const DEFAULT_MAX_TOKENS: usize = 16;
const DEFAULT_TEMPERATURE: f32 = 1.0;

/// One generated token slot — used internally by the loop, not exposed.
struct Generated {
    text: String,
    eos: bool,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum CompletionPrompt {
    Single(String),
    Batch(Vec<String>),
}

#[derive(Deserialize)]
pub struct CompletionsRequest {
    pub model: Option<String>,
    pub prompt: CompletionPrompt,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Top-p (nucleus) sampling — accepted for shape-compat but ignored
    /// in this slice (greedy/temperature only). See N0.2-fast.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Streaming via SSE — returns 501 in this slice (N0.1 SSE follow-up).
    #[serde(default)]
    pub stream: Option<bool>,
    /// Number of completions per prompt — only `n=1` supported; values
    /// >1 return 501.
    #[serde(default)]
    pub n: Option<usize>,
    /// Stop strings — accepted; first match halts generation.
    #[serde(default)]
    pub stop: Option<StopSpec>,
    /// Echo the prompt in the completion text (OpenAI legacy behaviour).
    #[serde(default)]
    pub echo: Option<bool>,
    /// Top-k log-probs — returns `null` in the response (F18 follow-up).
    #[serde(default)]
    pub logprobs: Option<usize>,
    /// Best-of — accepted, ignored (treats as 1).
    #[serde(default)]
    pub best_of: Option<usize>,
    /// Seed for reproducible sampling — accepted, ignored in greedy mode.
    #[serde(default)]
    pub seed: Option<u64>,
    /// End-user id — logged via tracing if set, otherwise no-op.
    #[serde(default)]
    pub user: Option<String>,
}

#[derive(Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: usize,
    pub finish_reason: &'static str,
    /// Always `null` in this slice (logprobs F18).
    pub logprobs: Option<()>,
}

#[derive(Serialize)]
pub struct CompletionsUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Serialize)]
pub struct CompletionsResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: CompletionsUsage,
}

pub async fn handle_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionsRequest>,
) -> Result<Response, ServerError> {
    state.bump_requests();

    if req.n.unwrap_or(1) > 1 {
        return Err(ServerError::BadRequest(
            "n>1 not yet supported; only n=1 (single completion per prompt)".into(),
        ));
    }

    let model = state.model_or_err(req.model.as_deref())?;
    if model.infer_disabled {
        return Err(ServerError::InferenceUnavailable(
            "inference disabled (--no-infer / --embed-only / --ffn-only)".into(),
        ));
    }

    let prompts: Vec<String> = match req.prompt {
        CompletionPrompt::Single(s) => vec![s],
        CompletionPrompt::Batch(v) => v,
    };
    if prompts.is_empty() {
        return Err(ServerError::BadRequest("prompt is empty".into()));
    }

    let max_tokens = req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    let temperature = req.temperature.unwrap_or(DEFAULT_TEMPERATURE).max(0.0);
    let stop_strings: Vec<String> = req
        .stop
        .as_ref()
        .map(|s| s.as_slice().to_vec())
        .unwrap_or_default();
    let echo = req.echo.unwrap_or(false);

    // Model id for the response (matches the request when given,
    // otherwise the loaded model's id).
    let model_id = req.model.clone().unwrap_or_else(|| model.id.clone());
    let model_arc = model.clone();

    if req.stream.unwrap_or(false) {
        // Streaming mode: SSE response. `echo` and batched prompts are
        // not supported in stream mode (OpenAI's stream contract is
        // one prompt → one stream of chunks).
        if echo {
            return Err(ServerError::BadRequest(
                "echo=true is not supported with stream=true".into(),
            ));
        }
        if prompts.len() > 1 {
            return Err(ServerError::BadRequest(
                "batched prompts (prompt: [...]) are not supported with stream=true; \
                 send one prompt per request"
                    .into(),
            ));
        }
        let prompt = prompts.into_iter().next().unwrap();
        return Ok(stream_completions(
            model_arc,
            prompt,
            max_tokens,
            temperature,
            stop_strings,
            model_id,
        )
        .into_response());
    }

    // Non-streaming: the existing buffered path.
    let (choices, prompt_tokens, completion_tokens) =
        tokio::task::spawn_blocking(move || -> Result<_, ServerError> {
            run_completions_loop(
                &model_arc,
                &prompts,
                max_tokens,
                temperature,
                &stop_strings,
                echo,
            )
        })
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))??;

    Ok(Json(CompletionsResponse {
        id: format!("cmpl-{}", new_id_suffix()),
        object: TEXT_COMPLETION_OBJECT,
        created: unix_now(),
        model: model_id,
        choices,
        usage: CompletionsUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    })
    .into_response())
}

/// Build an SSE response that streams one chunk per generated token.
/// Final chunk carries `finish_reason`; the stream terminates with
/// `data: [DONE]\n\n`.
fn stream_completions(
    model: Arc<LoadedModel>,
    prompt: String,
    max_tokens: usize,
    temperature: f32,
    stop_strings: Vec<String>,
    model_id: String,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<String>(64);
    let cmpl_id = format!("cmpl-{}", new_id_suffix());

    tokio::task::spawn_blocking(move || {
        let weights = match model.get_or_load_weights() {
            Ok(w) => w,
            Err(e) => {
                let _ = tx.blocking_send(error_chunk(&e));
                return;
            }
        };
        let encoding = match model.tokenizer.encode(prompt.as_str(), true) {
            Ok(e) => e,
            Err(e) => {
                let _ = tx.blocking_send(error_chunk(&format!("tokenize: {e}")));
                return;
            }
        };
        let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
        if prompt_ids.is_empty() {
            let _ = tx.blocking_send(error_chunk("prompt tokenises to empty"));
            return;
        }

        // Take a read guard on the patched vindex for the full
        // generation. WalkFfn does gate-KNN through the (possibly Q4_K)
        // index for every layer; holding the read guard for the
        // generation duration keeps the index pinned and keeps the
        // dequant cache warm across decode steps.
        let patched = model.patched.blocking_read();
        let walk_ffn = larql_inference::WalkFfn::new_unlimited(weights, &*patched);

        let mut ids = prompt_ids;
        let mut completion_text = String::new();
        let mut finish_reason: &'static str = "length";

        for _ in 0..max_tokens {
            let pred =
                larql_inference::predict_with_ffn(weights, &model.tokenizer, &ids, 1, &walk_ffn);
            let next_id = match pred.token_ids.first() {
                Some(&id) => id,
                None => {
                    finish_reason = "stop";
                    break;
                }
            };
            let next_text = pred
                .predictions
                .first()
                .map(|(t, _)| t.clone())
                .unwrap_or_default();
            let is_eos = larql_inference::vindex::is_end_of_turn(&next_text);
            let _ = temperature; // accepted; not consumed by the WalkFfn path.

            let chunk = build_text_completion_chunk(&cmpl_id, &model_id, Some(&next_text), None);
            if tx.blocking_send(chunk).is_err() {
                // Client disconnected.
                return;
            }
            completion_text.push_str(&next_text);
            ids.push(next_id);

            if is_eos {
                finish_reason = "stop";
                break;
            }
            if !stop_strings.is_empty() && contains_any(&completion_text, &stop_strings) {
                finish_reason = "stop";
                break;
            }
        }

        // Final chunk: finish_reason, no text.
        let final_chunk =
            build_text_completion_chunk(&cmpl_id, &model_id, None, Some(finish_reason));
        let _ = tx.blocking_send(final_chunk);
    });

    let stream = ReceiverStream::new(rx)
        .map(|data| Event::default().data(data))
        .chain(tokio_stream::once(Event::default().data("[DONE]")))
        .map(Ok::<_, Infallible>);

    Sse::new(stream).keep_alive(KeepAlive::default())
}

fn build_text_completion_chunk(
    id: &str,
    model: &str,
    text: Option<&str>,
    finish_reason: Option<&'static str>,
) -> String {
    let chunk = serde_json::json!({
        "id": id,
        "object": TEXT_COMPLETION_OBJECT,
        "created": unix_now(),
        "model": model,
        "choices": [{
            "text": text.unwrap_or(""),
            "index": 0,
            "logprobs": serde_json::Value::Null,
            "finish_reason": match finish_reason {
                Some(r) => serde_json::Value::String(r.to_string()),
                None => serde_json::Value::Null,
            },
        }]
    });
    chunk.to_string()
}

/// Generate completions for every prompt. Returns
/// `(choices, prompt_tokens_sum, completion_tokens_sum)`.
fn run_completions_loop(
    model: &LoadedModel,
    prompts: &[String],
    max_tokens: usize,
    temperature: f32,
    stop_strings: &[String],
    echo: bool,
) -> Result<(Vec<CompletionChoice>, usize, usize), ServerError> {
    let weights = model
        .get_or_load_weights()
        .map_err(ServerError::InferenceUnavailable)?;
    // Hold the read guard + WalkFfn for the lifetime of the loop:
    // gate-KNN through the (possibly Q4_K) index gives correct dense
    // FFN output without needing f32 dense FFN weights resident.
    let patched = model.patched.blocking_read();
    let walk_ffn = larql_inference::WalkFfn::new_unlimited(weights, &*patched);
    let _ = temperature; // accepted; WalkFfn path is greedy by construction.

    let mut choices = Vec::with_capacity(prompts.len());
    let mut total_prompt_tokens = 0usize;
    let mut total_completion_tokens = 0usize;

    for (idx, prompt) in prompts.iter().enumerate() {
        let encoding = model
            .tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|e| ServerError::Internal(format!("tokenize: {e}")))?;
        let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
        if prompt_ids.is_empty() {
            return Err(ServerError::BadRequest(format!(
                "prompt[{idx}] tokenises to empty"
            )));
        }
        total_prompt_tokens += prompt_ids.len();

        let mut ids = prompt_ids.clone();
        let mut completion_text = String::new();
        let mut completion_token_count = 0usize;
        let mut finish_reason = "length";

        for _ in 0..max_tokens {
            let pred = larql_inference::predict_with_ffn(
                weights,
                &model.tokenizer,
                &ids,
                1,
                &walk_ffn,
            );
            let next_id = match pred.token_ids.first() {
                Some(&id) => id,
                None => {
                    finish_reason = "stop";
                    break;
                }
            };
            let next_text = pred
                .predictions
                .first()
                .map(|(t, _)| t.clone())
                .unwrap_or_default();
            let gen = Generated {
                text: next_text.clone(),
                eos: larql_inference::vindex::is_end_of_turn(&next_text),
            };
            completion_text.push_str(&gen.text);
            completion_token_count += 1;
            ids.push(next_id);

            if gen.eos {
                finish_reason = "stop";
                break;
            }
            if !stop_strings.is_empty() && contains_any(&completion_text, stop_strings) {
                // Trim at the matched stop so it isn't included in the output.
                completion_text = trim_at_stop(&completion_text, stop_strings);
                finish_reason = "stop";
                break;
            }
        }

        total_completion_tokens += completion_token_count;

        let text_out = if echo {
            format!("{prompt}{completion_text}")
        } else {
            completion_text
        };

        choices.push(CompletionChoice {
            text: text_out,
            index: idx,
            finish_reason,
            logprobs: None,
        });
    }

    Ok((choices, total_prompt_tokens, total_completion_tokens))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_single_string_prompt() {
        let json = serde_json::json!({"prompt": "hello"});
        let req: CompletionsRequest = serde_json::from_value(json).unwrap();
        match req.prompt {
            CompletionPrompt::Single(s) => assert_eq!(s, "hello"),
            _ => panic!(),
        }
    }

    #[test]
    fn deserialize_string_array_prompt() {
        let json = serde_json::json!({"prompt": ["a", "b"]});
        let req: CompletionsRequest = serde_json::from_value(json).unwrap();
        match req.prompt {
            CompletionPrompt::Batch(v) => assert_eq!(v, vec!["a", "b"]),
            _ => panic!(),
        }
    }
}
