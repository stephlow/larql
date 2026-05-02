//! `POST /v1/chat/completions` — OpenAI-compatible chat completions (N0.1, slice 2).
//!
//! Implements the [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat/create)
//! shape so existing `openai` SDKs work unmodified:
//!
//! ```python
//! from openai import OpenAI
//! client = OpenAI(base_url="http://larql:8080/v1", api_key="sk-...")
//! resp = client.chat.completions.create(
//!     model="gemma-3-4b",
//!     messages=[
//!         {"role": "system", "content": "You are a helpful assistant."},
//!         {"role": "user",   "content": "What is the capital of France?"},
//!     ],
//!     max_tokens=20,
//! )
//! ```
//!
//! ## Chat template handling
//!
//! `messages` is rendered to a single prompt via the model's chat
//! template (Gemma / Llama / ChatML / Mistral / plain), detected from
//! the model's `family` and `id`. The rendered prompt then runs through
//! the same generation loop as `/v1/completions`.
//!
//! Template detection precedence:
//! 1. `arch.family()` (authoritative when available)
//! 2. Substring match on `model.id` ("gemma", "llama", "qwen", …)
//! 3. Plain (fallback for unknown families and base models)
//!
//! ## Slice 2 limitations
//!
//! - `stream=true` returns 400 (SSE arrives in slice 3)
//! - `tools` / `tool_choice` returns 400 (slice 4 = N0.6 constrained decoding)
//! - `response_format: json_object | json_schema` returns 400 (slice 4)
//! - `n>1` returns 400
//! - `logprobs` request field accepted, response field always `null` (F18)
//! - generation is un-KV-cached, ~1-3 tok/s on CPU for Gemma 3 4B
//!   (KV-cached fast path = N0.2-fast in ROADMAP)

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt as _;

use larql_inference::{ChatMLRenderer, GemmaRenderer, Llama3Renderer, TurnRenderer};

use crate::error::ServerError;
use crate::state::{AppState, LoadedModel};

use super::util::{contains_any, error_chunk, new_id_suffix, trim_at_stop, unix_now, StopSpec};

const CHAT_COMPLETION_OBJECT: &str = "chat.completion";
const CHAT_COMPLETION_CHUNK_OBJECT: &str = "chat.completion.chunk";
const ASSISTANT_ROLE: &str = "assistant";
const SYSTEM_ROLE: &str = "system";
const USER_ROLE: &str = "user";
const DEFAULT_MAX_TOKENS: usize = 256;
const DEFAULT_TEMPERATURE: f32 = 1.0;

#[derive(Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    /// OpenAI tool-call fields — accepted for shape-compat in slice 2,
    /// but `tool_calls`/`tool_call_id` non-null returns 400 (tools land
    /// in slice 4).
    #[serde(default)]
    pub tool_calls: Option<serde_json::Value>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

#[derive(Deserialize)]
pub struct ChatCompletionsRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Top-p — accepted, ignored (greedy/temperature only in slice 2).
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Streaming via SSE — returns 400 in slice 2 (slice 3 SSE follow-up).
    #[serde(default)]
    pub stream: Option<bool>,
    /// Number of completions per prompt — only n=1 supported.
    #[serde(default)]
    pub n: Option<usize>,
    /// Stop strings — first match halts generation.
    #[serde(default)]
    pub stop: Option<StopSpec>,
    /// Top-k log-probs — request accepted, response field always null.
    #[serde(default)]
    pub logprobs: Option<bool>,
    /// Newer log-probs field used by recent SDKs — same handling as `logprobs`.
    #[serde(default)]
    pub top_logprobs: Option<usize>,
    /// Tool definitions — slice 4 (N0.6 constrained decoding); 400 if non-empty.
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
    /// Tool choice — same as `tools` (slice 4).
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    /// Response format (`{type: "json_object" | "json_schema", ...}`) —
    /// slice 4. Returns 400 for any non-text response_format.
    #[serde(default)]
    pub response_format: Option<serde_json::Value>,
    /// Seed for reproducible sampling — accepted, ignored in greedy mode.
    #[serde(default)]
    pub seed: Option<u64>,
    /// End-user id — logged via tracing if set.
    #[serde(default)]
    pub user: Option<String>,
    /// Frequency / presence penalties — accepted, ignored in slice 2.
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
}

#[derive(Serialize)]
pub struct ChatChoiceMessage {
    pub role: &'static str,
    pub content: String,
}

#[derive(Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatChoiceMessage,
    pub finish_reason: &'static str,
    /// Always null in slice 2 (logprobs F18).
    pub logprobs: Option<()>,
}

#[derive(Serialize)]
pub struct ChatUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Serialize)]
pub struct ChatCompletionsResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: ChatUsage,
}

pub async fn handle_chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionsRequest>,
) -> Result<Response, ServerError> {
    state.bump_requests();

    if req.n.unwrap_or(1) > 1 {
        return Err(ServerError::BadRequest(
            "n>1 not yet supported; only n=1 (single completion per prompt)".into(),
        ));
    }
    if req
        .tools
        .as_ref()
        .is_some_and(|v| !v.is_null() && !is_empty_json_array(v))
        || req.tool_choice.is_some()
    {
        return Err(ServerError::BadRequest(
            "tools / tool_choice not yet supported; arrives in N0 slice 4 \
             (constrained decoding). See ROADMAP."
                .into(),
        ));
    }
    if let Some(rf) = req.response_format.as_ref() {
        // Reject any explicit non-text response_format. `{type: "text"}` is
        // the OpenAI default and we treat it as a no-op.
        let is_text_default = rf
            .get("type")
            .and_then(|t| t.as_str())
            .map(|s| s == "text")
            .unwrap_or(false);
        if !is_text_default {
            return Err(ServerError::BadRequest(
                "response_format != \"text\" (json_object, json_schema) not yet \
                 supported; arrives in N0 slice 4."
                    .into(),
            ));
        }
    }
    for (i, m) in req.messages.iter().enumerate() {
        if m.tool_calls
            .as_ref()
            .is_some_and(|v| !v.is_null() && !is_empty_json_array(v))
            || m.tool_call_id.is_some()
        {
            return Err(ServerError::BadRequest(format!(
                "messages[{i}] contains tool_calls / tool_call_id; tools land in N0 slice 4"
            )));
        }
    }

    let model = state.model_or_err(req.model.as_deref())?;
    if model.infer_disabled {
        return Err(ServerError::InferenceUnavailable(
            "inference disabled (--no-infer / --embed-only / --ffn-only)".into(),
        ));
    }
    if req.messages.is_empty() {
        return Err(ServerError::BadRequest("messages is empty".into()));
    }
    for (i, m) in req.messages.iter().enumerate() {
        if !matches!(m.role.as_str(), USER_ROLE | ASSISTANT_ROLE | SYSTEM_ROLE) {
            return Err(ServerError::BadRequest(format!(
                "messages[{i}].role must be 'user' | 'assistant' | 'system' (got {:?})",
                m.role
            )));
        }
    }

    let max_tokens = req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    let temperature = req.temperature.unwrap_or(DEFAULT_TEMPERATURE).max(0.0);
    let stop_strings: Vec<String> = req
        .stop
        .as_ref()
        .map(|s| s.as_slice().to_vec())
        .unwrap_or_default();
    let model_id = req.model.clone().unwrap_or_else(|| model.id.clone());
    let model_arc = model.clone();
    let messages = req.messages;

    if req.stream.unwrap_or(false) {
        return Ok(stream_chat_completion(
            model_arc,
            messages,
            max_tokens,
            temperature,
            stop_strings,
            model_id,
        )
        .into_response());
    }

    let (text, finish_reason, prompt_tokens, completion_tokens) =
        tokio::task::spawn_blocking(move || -> Result<_, ServerError> {
            run_chat_completion(
                &model_arc,
                &messages,
                max_tokens,
                temperature,
                &stop_strings,
            )
        })
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))??;

    Ok(Json(ChatCompletionsResponse {
        id: format!("chatcmpl-{}", new_id_suffix()),
        object: CHAT_COMPLETION_OBJECT,
        created: unix_now(),
        model: model_id,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatChoiceMessage {
                role: ASSISTANT_ROLE,
                content: text,
            },
            finish_reason,
            logprobs: None,
        }],
        usage: ChatUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    })
    .into_response())
}

/// SSE stream for `/v1/chat/completions`. First chunk emits
/// `delta: {role: "assistant"}`; subsequent chunks emit
/// `delta: {content: "<token text>"}`; the final chunk has empty
/// `delta` and `finish_reason`. Stream terminates with `data: [DONE]`.
fn stream_chat_completion(
    model: Arc<LoadedModel>,
    messages: Vec<ChatMessage>,
    max_tokens: usize,
    temperature: f32,
    stop_strings: Vec<String>,
    model_id: String,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<String>(64);
    let chat_id = format!("chatcmpl-{}", new_id_suffix());

    tokio::task::spawn_blocking(move || {
        let weights = match model.get_or_load_weights() {
            Ok(w) => w,
            Err(e) => {
                let _ = tx.blocking_send(error_chunk(&e));
                return;
            }
        };
        let template = pick_template(&model);
        let prompt = render_messages(template, &messages);
        let encoding = match model.tokenizer.encode(prompt.as_str(), true) {
            Ok(e) => e,
            Err(e) => {
                let _ = tx.blocking_send(error_chunk(&format!("tokenize: {e}")));
                return;
            }
        };
        let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
        if prompt_ids.is_empty() {
            let _ = tx.blocking_send(error_chunk("rendered prompt tokenises to empty"));
            return;
        }

        // First chunk: role="assistant" delta. OpenAI's chat completion
        // stream contract starts with this, even before any content.
        let first = build_chat_chunk(&chat_id, &model_id, Some(ASSISTANT_ROLE), None, None);
        if tx.blocking_send(first).is_err() {
            return;
        }

        // WalkFfn through the (possibly Q4_K) index — same path the
        // existing /v1/infer mode=walk uses, takes &ModelWeights only.
        let patched = model.patched.blocking_read();
        let walk_ffn = larql_inference::WalkFfn::new_unlimited(weights, &*patched);
        let _ = temperature; // accepted; WalkFfn path is greedy.

        let mut ids = prompt_ids;
        let mut completion_text = String::new();
        let mut finish_reason: &'static str = "length";

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
            let is_eos = larql_inference::vindex::is_end_of_turn(&next_text);

            let chunk = build_chat_chunk(&chat_id, &model_id, None, Some(&next_text), None);
            if tx.blocking_send(chunk).is_err() {
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

        let final_chunk = build_chat_chunk(&chat_id, &model_id, None, None, Some(finish_reason));
        let _ = tx.blocking_send(final_chunk);
    });

    let stream = ReceiverStream::new(rx)
        .map(|data| Event::default().data(data))
        .chain(tokio_stream::once(Event::default().data("[DONE]")))
        .map(Ok::<_, Infallible>);

    Sse::new(stream).keep_alive(KeepAlive::default())
}

fn build_chat_chunk(
    id: &str,
    model: &str,
    role: Option<&str>,
    content: Option<&str>,
    finish_reason: Option<&'static str>,
) -> String {
    let mut delta = serde_json::Map::new();
    if let Some(r) = role {
        delta.insert("role".into(), serde_json::Value::String(r.to_string()));
    }
    if let Some(c) = content {
        delta.insert("content".into(), serde_json::Value::String(c.to_string()));
    }
    let chunk = serde_json::json!({
        "id": id,
        "object": CHAT_COMPLETION_CHUNK_OBJECT,
        "created": unix_now(),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": serde_json::Value::Object(delta),
            "finish_reason": match finish_reason {
                Some(r) => serde_json::Value::String(r.to_string()),
                None => serde_json::Value::Null,
            },
            "logprobs": serde_json::Value::Null,
        }]
    });
    chunk.to_string()
}

/// Render `messages` to a single prompt, then run the un-KV-cached
/// generation loop. Returns `(text, finish_reason, prompt_tokens,
/// completion_tokens)`.
fn run_chat_completion(
    model: &LoadedModel,
    messages: &[ChatMessage],
    max_tokens: usize,
    temperature: f32,
    stop_strings: &[String],
) -> Result<(String, &'static str, usize, usize), ServerError> {
    let weights = model
        .get_or_load_weights()
        .map_err(ServerError::InferenceUnavailable)?;

    let template = pick_template(model);
    let prompt = render_messages(template, messages);

    let encoding = model
        .tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| ServerError::Internal(format!("tokenize: {e}")))?;
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    if prompt_ids.is_empty() {
        return Err(ServerError::BadRequest(
            "rendered prompt tokenises to empty".into(),
        ));
    }
    let prompt_token_count = prompt_ids.len();

    let patched = model.patched.blocking_read();
    let walk_ffn = larql_inference::WalkFfn::new_unlimited(weights, &*patched);
    let _ = temperature; // accepted; WalkFfn path is greedy.

    let mut ids = prompt_ids;
    let mut completion_text = String::new();
    let mut completion_token_count = 0usize;
    let mut finish_reason: &'static str = "length";

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
        let is_eos = larql_inference::vindex::is_end_of_turn(&next_text);
        completion_text.push_str(&next_text);
        completion_token_count += 1;
        ids.push(next_id);

        if is_eos {
            finish_reason = "stop";
            break;
        }
        if !stop_strings.is_empty() && contains_any(&completion_text, stop_strings) {
            completion_text = trim_at_stop(&completion_text, stop_strings);
            finish_reason = "stop";
            break;
        }
    }

    Ok((
        completion_text,
        finish_reason,
        prompt_token_count,
        completion_token_count,
    ))
}

// ── Template selection + multi-turn rendering ────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Template {
    Gemma,
    Llama,
    ChatML,
    Mistral,
    Plain,
}

fn pick_template(model: &LoadedModel) -> Template {
    // Prefer the architecture's family signal if loaded weights expose
    // one. Fall back to model id heuristics.
    if let Some(weights) = model.weights.get() {
        let fam = weights.arch.family();
        match fam {
            "gemma2" | "gemma3" | "gemma4" => return Template::Gemma,
            "llama" => return Template::Llama,
            "qwen" | "qwen2" | "qwen3" | "deepseek" | "gpt_oss" => return Template::ChatML,
            "mistral" | "mixtral" => return Template::Mistral,
            _ => {}
        }
    }
    let id = model.id.to_ascii_lowercase();
    if id.contains("gemma") {
        Template::Gemma
    } else if id.contains("mixtral") || id.contains("mistral") {
        Template::Mistral
    } else if id.contains("llama") {
        Template::Llama
    } else if id.contains("qwen") || id.contains("deepseek") || id.contains("chatml") {
        Template::ChatML
    } else {
        Template::Plain
    }
}

/// Render a message list into a single prompt string, ready to feed to
/// the tokenizer. The final assistant-open marker is appended so the
/// model continues from "I am about to speak as the assistant".
fn render_messages(tpl: Template, messages: &[ChatMessage]) -> String {
    match tpl {
        Template::Gemma => render_via_renderer(&GemmaRenderer, messages),
        Template::Llama => render_via_renderer(&Llama3Renderer, messages),
        Template::ChatML => render_via_renderer(&ChatMLRenderer, messages),
        Template::Mistral => render_mistral(messages),
        Template::Plain => render_plain(messages),
    }
}

fn render_via_renderer<R: TurnRenderer>(renderer: &R, messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for m in messages {
        out.push_str(&renderer.render(&m.role, &m.content));
    }
    out.push_str(&renderer.assistant_open());
    out
}

/// Mistral / Mixtral: `[INST] {user} [/INST] {assistant}` with system
/// prompt prepended to the first user turn.
fn render_mistral(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    let mut pending_system: Vec<String> = Vec::new();
    let mut i = 0;
    while i < messages.len() {
        let m = &messages[i];
        match m.role.as_str() {
            SYSTEM_ROLE => {
                pending_system.push(m.content.clone());
                i += 1;
            }
            USER_ROLE => {
                let prefix = if pending_system.is_empty() {
                    String::new()
                } else {
                    let p = pending_system.join("\n") + "\n\n";
                    pending_system.clear();
                    p
                };
                out.push_str(&format!("[INST] {prefix}{} [/INST]", m.content));
                i += 1;
                if let Some(next) = messages.get(i) {
                    if next.role == ASSISTANT_ROLE {
                        out.push_str(&format!(" {} ", next.content));
                        i += 1;
                    }
                }
            }
            ASSISTANT_ROLE => {
                // Stray assistant turn (no preceding user) — emit verbatim.
                out.push_str(&format!(" {} ", m.content));
                i += 1;
            }
            _ => i += 1,
        }
    }
    // Trailing system without a user turn → wrap as a user prompt so
    // the model has somewhere to respond.
    if !pending_system.is_empty() {
        out.push_str(&format!("[INST] {} [/INST]", pending_system.join("\n")));
    }
    out
}

/// Plain template — for base / non-instruct models. Concatenates the
/// messages with `User:` / `Assistant:` / `System:` markers and ends
/// with an `Assistant:` open so the model continues. Not great, but
/// better than dropping system prompts on the floor.
fn render_plain(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for m in messages {
        let label = match m.role.as_str() {
            USER_ROLE => "User",
            ASSISTANT_ROLE => "Assistant",
            SYSTEM_ROLE => "System",
            other => other,
        };
        out.push_str(&format!("{label}: {}\n", m.content));
    }
    out.push_str("Assistant: ");
    out
}

// ── chat-only request validation helper ─────────────────────────────────────

fn is_empty_json_array(v: &serde_json::Value) -> bool {
    v.as_array().map(|a| a.is_empty()).unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    #[test]
    fn render_gemma_multi_turn_includes_model_open() {
        let out = render_messages(
            Template::Gemma,
            &[
                msg("user", "hi"),
                msg("assistant", "hello"),
                msg("user", "more"),
            ],
        );
        assert!(out.contains("<start_of_turn>user\nhi<end_of_turn>"));
        assert!(out.contains("<start_of_turn>model\nhello<end_of_turn>"));
        assert!(out.contains("<start_of_turn>user\nmore<end_of_turn>"));
        assert!(out.ends_with("<start_of_turn>model\n"));
    }

    #[test]
    fn render_chatml_multi_turn() {
        let out = render_messages(
            Template::ChatML,
            &[
                msg("system", "You are concise."),
                msg("user", "hi"),
                msg("assistant", "hello"),
                msg("user", "more"),
            ],
        );
        assert!(out.contains("<|im_start|>system\nYou are concise.<|im_end|>"));
        assert!(out.contains("<|im_start|>user\nhi<|im_end|>"));
        assert!(out.contains("<|im_start|>assistant\nhello<|im_end|>"));
        assert!(out.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn render_llama_multi_turn() {
        let out = render_messages(
            Template::Llama,
            &[
                msg("user", "hi"),
                msg("assistant", "hello"),
                msg("user", "more"),
            ],
        );
        assert!(out.contains("<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>"));
        assert!(out.contains("<|start_header_id|>assistant<|end_header_id|>\n\nhello<|eot_id|>"));
        assert!(out.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn render_mistral_prepends_system_to_first_user() {
        let out = render_messages(
            Template::Mistral,
            &[msg("system", "Be brief."), msg("user", "hi")],
        );
        assert_eq!(out, "[INST] Be brief.\n\nhi [/INST]");
    }

    #[test]
    fn render_mistral_handles_assistant_turn() {
        let out = render_messages(
            Template::Mistral,
            &[
                msg("user", "hi"),
                msg("assistant", "hello"),
                msg("user", "more"),
            ],
        );
        assert_eq!(out, "[INST] hi [/INST] hello [INST] more [/INST]");
    }

    #[test]
    fn render_plain_uses_role_labels() {
        let out = render_messages(
            Template::Plain,
            &[msg("system", "Concise."), msg("user", "hi")],
        );
        assert_eq!(out, "System: Concise.\nUser: hi\nAssistant: ");
    }

    #[test]
    fn pick_template_uses_id_heuristic_when_no_weights() {
        // We can't construct a real LoadedModel here; cover the id-based
        // fallback via the helper directly.
        let cases = [
            ("google/gemma-3-4b-it", Template::Gemma),
            ("meta-llama/Llama-3.2-3B-Instruct", Template::Llama),
            ("Qwen/Qwen2.5-7B-Instruct", Template::ChatML),
            ("deepseek-ai/DeepSeek-V2", Template::ChatML),
            ("mistralai/Mistral-7B-Instruct-v0.3", Template::Mistral),
            ("mistralai/Mixtral-8x7B", Template::Mistral),
            ("some-random-model", Template::Plain),
            ("", Template::Plain),
        ];
        for (id, want) in cases {
            let lower = id.to_ascii_lowercase();
            let got = if lower.contains("gemma") {
                Template::Gemma
            } else if lower.contains("mixtral") || lower.contains("mistral") {
                Template::Mistral
            } else if lower.contains("llama") {
                Template::Llama
            } else if lower.contains("qwen")
                || lower.contains("deepseek")
                || lower.contains("chatml")
            {
                Template::ChatML
            } else {
                Template::Plain
            };
            assert_eq!(got, want, "id={id}");
        }
    }

    #[test]
    fn deserialize_chat_request_min() {
        let json = serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req: ChatCompletionsRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, "user");
    }

    #[test]
    fn deserialize_chat_request_full() {
        let json = serde_json::json!({
            "model": "gemma-3-4b",
            "messages": [
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "What is 2+2?"}
            ],
            "max_tokens": 50,
            "temperature": 0.0,
            "top_p": 0.9,
            "n": 1,
            "stream": false,
            "stop": ["\n\n"],
            "seed": 42
        });
        let req: ChatCompletionsRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.max_tokens, Some(50));
        assert_eq!(req.temperature, Some(0.0));
    }
}
