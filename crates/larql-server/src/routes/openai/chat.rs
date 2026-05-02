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
//! ## Generation path
//!
//! Buffered + SSE streaming both call
//! `larql_inference::layer_graph::generate{,_streaming}` which is KV-
//! cached on f16 vindexes (and falls back to a per-step Q4_K decode
//! when the backend is CPU + Q4K). Generation acquires an exclusive
//! write guard on `LoadedModel.weights` for the duration; concurrent
//! reads block but other endpoints are unaffected in steady state.
//!
//! ## Slice 2-3 limitations
//!
//! - `tools` / `tool_choice` returns 400 (slice 4 = N0.6 constrained decoding)
//! - `response_format: json_object | json_schema` returns 400 (slice 4)
//! - `n>1` returns 400
//! - `logprobs` request field accepted, response field always `null` (F18)

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

use crate::error::ServerError;
use crate::state::{AppState, LoadedModel};

use super::schema::{ObjectSchema, Schema};
use super::util::{contains_any, error_chunk, new_id_suffix, trim_at_stop, unix_now, StopSpec};

const CHAT_COMPLETION_OBJECT: &str = "chat.completion";
const CHAT_COMPLETION_CHUNK_OBJECT: &str = "chat.completion.chunk";
const ASSISTANT_ROLE: &str = "assistant";
const SYSTEM_ROLE: &str = "system";
const USER_ROLE: &str = "user";
const DEFAULT_MAX_TOKENS: usize = 256;

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
    /// Nucleus (top-p) filter applied after temperature scaling. Only
    /// honoured when `temperature > 0`; for greedy decoding it's a no-op.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Streaming via SSE — emits one `chat.completion.chunk` per token,
    /// terminated by `data: [DONE]\n\n`.
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
    /// Seed for reproducible sampling. Same seed + same temperature +
    /// same prompt produces the same tokens. No-op for greedy mode
    /// (greedy is already deterministic on argmax).
    #[serde(default)]
    pub seed: Option<u64>,
    /// End-user id — logged via tracing if set.
    #[serde(default)]
    pub user: Option<String>,
    /// Frequency / presence penalties — accepted for shape compat;
    /// the sampler does not yet apply repetition penalties (F19).
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
    let constrained_schema = schema_for_response_format(req.response_format.as_ref())?;
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
    let stop_strings: Vec<String> = req
        .stop
        .as_ref()
        .map(|s| s.as_slice().to_vec())
        .unwrap_or_default();
    let temperature = req.temperature;
    let top_p = req.top_p;
    let seed = req.seed;
    let model_id = req.model.clone().unwrap_or_else(|| model.id.clone());
    let model_arc = model.clone();
    let messages = req.messages;

    if req.stream.unwrap_or(false) {
        return Ok(stream_chat_completion(
            model_arc,
            messages,
            max_tokens,
            temperature,
            top_p,
            seed,
            stop_strings,
            constrained_schema,
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
                top_p,
                seed,
                &stop_strings,
                constrained_schema,
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
#[allow(clippy::too_many_arguments)]
fn stream_chat_completion(
    model: Arc<LoadedModel>,
    messages: Vec<ChatMessage>,
    max_tokens: usize,
    temperature: Option<f32>,
    top_p: Option<f32>,
    seed: Option<u64>,
    stop_strings: Vec<String>,
    constrained_schema: Option<Schema>,
    model_id: String,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<String>(64);
    let chat_id = format!("chatcmpl-{}", new_id_suffix());

    tokio::task::spawn_blocking(move || {
        let mut weights_guard = match model.lock_weights_for_gen() {
            Ok(w) => w,
            Err(e) => {
                let _ = tx.blocking_send(error_chunk(&e));
                return;
            }
        };
        let weights: &mut larql_inference::ModelWeights = &mut weights_guard;
        let template = pick_template(&model);
        let prompt = render(template, &messages);
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

        let patched = model.patched.blocking_read();
        let index = patched.base();
        let backend = larql_compute::default_backend();
        let cached_layers = larql_inference::CachedLayerGraph::from_residuals(Vec::new());
        let num_layers = weights.num_layers;

        // Per-token callback used by both the sampling and the
        // constrained streaming paths. Pushes one SSE chunk per token
        // and tracks completion text so client-supplied stop strings
        // can halt generation early. `early_stop` is shared with the
        // post-loop finish-reason check via Rc<Cell<bool>> — ergonomic
        // single-threaded mutable state, since the whole spawn_blocking
        // body runs on one thread.
        let chat_id_cb = chat_id.clone();
        let model_id_cb = model_id.clone();
        let tx_cb = tx.clone();
        let stop_strings_cb = stop_strings.clone();
        let early_stop = std::rc::Rc::new(std::cell::Cell::new(false));
        let early_stop_cb = early_stop.clone();
        let mut completion_text = String::new();
        let on_token = move |_id: u32, text: &str, _prob: f64| {
            if early_stop_cb.get() {
                return;
            }
            let chunk = build_chat_chunk(&chat_id_cb, &model_id_cb, None, Some(text), None);
            if tx_cb.blocking_send(chunk).is_err() {
                early_stop_cb.set(true);
                return;
            }
            completion_text.push_str(text);
            if !stop_strings_cb.is_empty() && contains_any(&completion_text, &stop_strings_cb) {
                early_stop_cb.set(true);
            }
        };

        let result = if let Some(schema) = constrained_schema {
            let _ = (temperature, top_p, seed); // accepted but no-op for constrained
            let mask = build_constrained_mask(&model.tokenizer, schema);
            larql_inference::layer_graph::generate_constrained_streaming(
                weights,
                &model.tokenizer,
                &prompt_ids,
                max_tokens,
                index,
                &*backend,
                &cached_layers,
                0..num_layers,
                mask,
                on_token,
            )
        } else {
            let (sampling, eos) =
                super::util::build_sampling_eos(temperature, top_p, seed, &stop_strings);
            larql_inference::layer_graph::generate_streaming(
                weights,
                &model.tokenizer,
                &prompt_ids,
                max_tokens,
                index,
                &*backend,
                &cached_layers,
                0..num_layers,
                sampling,
                &eos,
                on_token,
            )
        };

        // Final-chunk finish reason: layer_graph::generate halts on
        // EOS internally; tokens.len() < max_tokens implies stop.
        let finish_reason: &'static str = if early_stop.get() || result.tokens.len() < max_tokens {
            "stop"
        } else {
            "length"
        };
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

/// Render `messages` to a single prompt, then run the generation loop.
/// Returns `(text, finish_reason, prompt_tokens, completion_tokens)`.
///
/// Branches on `constrained_schema`:
/// - `None` → sampling path (`generate_with_sampling`).
/// - `Some(schema)` → grammar-mask path (`generate_constrained`).
///   Sampling fields (temperature/top_p/seed) are accepted but ignored
///   in this slice — constrained decoding is greedy by design so JSON /
///   structured output is deterministic.
#[allow(clippy::too_many_arguments)]
fn run_chat_completion(
    model: &LoadedModel,
    messages: &[ChatMessage],
    max_tokens: usize,
    temperature: Option<f32>,
    top_p: Option<f32>,
    seed: Option<u64>,
    stop_strings: &[String],
    constrained_schema: Option<Schema>,
) -> Result<(String, &'static str, usize, usize), ServerError> {
    // Take an exclusive write guard on the weights for the duration
    // of generation. `larql_inference::layer_graph::generate` mutates
    // `weights.tensors` (the per-layer Q4_K dequant cache), so other
    // read paths block while one chat completion runs.
    let mut weights_guard = model
        .lock_weights_for_gen()
        .map_err(ServerError::InferenceUnavailable)?;
    let weights: &mut larql_inference::ModelWeights = &mut weights_guard;

    let template = pick_template(model);
    let prompt = render(template, messages);

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
    let index = patched.base();
    let backend = larql_compute::default_backend();
    let cached_layers = larql_inference::CachedLayerGraph::from_residuals(Vec::new());
    let num_layers = weights.num_layers;

    let result = if let Some(schema) = constrained_schema {
        let _ = (temperature, top_p, seed); // accepted but no-op for constrained
        let mask = build_constrained_mask(&model.tokenizer, schema);
        larql_inference::layer_graph::generate_constrained(
            weights,
            &model.tokenizer,
            &prompt_ids,
            max_tokens,
            index,
            &*backend,
            &cached_layers,
            0..num_layers,
            mask,
        )
    } else {
        let (sampling, eos) =
            super::util::build_sampling_eos(temperature, top_p, seed, stop_strings);
        larql_inference::layer_graph::generate_with_sampling(
            weights,
            &model.tokenizer,
            &prompt_ids,
            max_tokens,
            index,
            &*backend,
            &cached_layers,
            0..num_layers,
            sampling,
            &eos,
        )
    };

    let mut completion_text = String::new();
    let mut completion_token_count = 0usize;
    let mut finish_reason: &'static str = "length";
    for (text, _prob) in &result.tokens {
        completion_text.push_str(text);
        completion_token_count += 1;
        if larql_inference::vindex::is_end_of_turn(text) {
            finish_reason = "stop";
            break;
        }
    }
    if !stop_strings.is_empty() && contains_any(&completion_text, stop_strings) {
        completion_text = trim_at_stop(&completion_text, stop_strings);
        finish_reason = "stop";
    }

    Ok((
        completion_text,
        finish_reason,
        prompt_token_count,
        completion_token_count,
    ))
}

// ── Template selection ───────────────────────────────────────────────────────
//
// The multi-turn rendering itself lives in
// `larql_inference::prompt::ChatTemplate::render_messages`. This handler
// only needs to pick the right template variant for the loaded model.

fn pick_template(model: &LoadedModel) -> larql_inference::prompt::ChatTemplate {
    use larql_inference::prompt::ChatTemplate;
    // Prefer the architecture's family signal when weights are loaded;
    // fall back to id heuristics when weights haven't been touched yet.
    if let Some(cell) = model.weights.get() {
        if let Ok(weights) = cell.read() {
            return ChatTemplate::for_family(weights.arch.family());
        }
    }
    ChatTemplate::for_model_id(&model.id)
}

/// Adapter: convert our wire `ChatMessage` list to the `(role, content)`
/// shape `ChatTemplate::render_messages` accepts. Pure plumbing — no
/// rendering logic lives here any more.
fn render(template: larql_inference::prompt::ChatTemplate, messages: &[ChatMessage]) -> String {
    template.render_messages(
        messages
            .iter()
            .map(|m| (m.role.as_str(), m.content.as_str())),
    )
}

// ── chat-only request validation helper ─────────────────────────────────────

fn is_empty_json_array(v: &serde_json::Value) -> bool {
    v.as_array().map(|a| a.is_empty()).unwrap_or(false)
}

/// Map an OpenAI `response_format` field to the `Schema` the FSM
/// should enforce. `None` (or `{type: "text"}`) means "no constrained
/// decoding" — fall through to the sampling path.
///
/// `json_object` compiles to `Schema::Object(any)`. `json_schema`
/// reaches into `json_schema.schema` and runs the JSON Schema parser
/// with `strict: true` when the `strict` field is set (matching
/// OpenAI's structured-outputs contract).
fn schema_for_response_format(
    rf: Option<&serde_json::Value>,
) -> Result<Option<Schema>, ServerError> {
    let Some(rf) = rf else {
        return Ok(None);
    };
    let kind = rf.get("type").and_then(|t| t.as_str()).unwrap_or("text");
    match kind {
        "text" => Ok(None),
        "json_object" => Ok(Some(Schema::object(ObjectSchema::any()))),
        "json_schema" => {
            let js = rf.get("json_schema").ok_or_else(|| {
                ServerError::BadRequest(
                    "response_format.type=json_schema requires a json_schema field".into(),
                )
            })?;
            let schema_value = js.get("schema").ok_or_else(|| {
                ServerError::BadRequest("response_format.json_schema.schema is required".into())
            })?;
            // OpenAI's `strict: true` flips the additionalProperties default
            // to false. Default is `false` here so non-strict callers can
            // still send extra keys.
            let strict = js.get("strict").and_then(|v| v.as_bool()).unwrap_or(false);
            let opts = super::schema::ParseOptions { strict };
            let parsed = super::schema::parse_schema_with(schema_value, opts)
                .map_err(|e| ServerError::BadRequest(format!("invalid json_schema: {e}")))?;
            Ok(Some(parsed))
        }
        other => Err(ServerError::BadRequest(format!(
            "response_format.type {other:?} is not supported (expected \
             \"text\" | \"json_object\" | \"json_schema\")"
        ))),
    }
}

/// Resolve common end-of-turn token ids for the loaded model. The
/// constrained-mask uses these to gate EOS — the model can't truncate
/// while the FSM is mid-structure, but once the FSM is complete the
/// EOS tokens become legal again.
///
/// Looks up a small set of well-known special markers
/// (`<end_of_turn>`, `<|im_end|>`, `<eos>`, `</s>`, etc.) via
/// `tokenizer.token_to_id` and ignores any that aren't present in the
/// vocab.
fn resolve_eos_token_ids(
    tokenizer: &larql_inference::tokenizers::Tokenizer,
) -> std::collections::HashSet<u32> {
    let mut ids = std::collections::HashSet::new();
    for tok in [
        "<end_of_turn>",
        "<|end_of_turn|>",
        "<|im_end|>",
        "<|eot_id|>",
        "<|eom_id|>",
        "<|endoftext|>",
        "<|end_of_text|>",
        "<eos>",
        "</s>",
    ] {
        if let Some(id) = tokenizer.token_to_id(tok) {
            ids.insert(id);
        }
    }
    ids
}

/// Build the masked-vocab callback the constrained generator expects.
/// Wraps the tokenizer in `Arc` (the schema mask caches surface forms
/// per id), seeds a fresh FSM from `schema`, and includes the model's
/// EOS marker ids so structured output can terminate cleanly once the
/// FSM hits `is_complete()`.
fn build_constrained_mask(
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    schema: Schema,
) -> impl FnMut(&[u32], &mut Vec<f32>) {
    let eos_ids = resolve_eos_token_ids(tokenizer);
    let tk: std::sync::Arc<larql_inference::tokenizers::Tokenizer> =
        std::sync::Arc::new(tokenizer.clone());
    super::schema::build_mask(tk, super::schema::Fsm::new(schema), String::new(), eos_ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Multi-turn template rendering is tested in
    // `larql_inference::prompt::render_messages_tests` (Gemma, ChatML,
    // Llama, Mistral, Plain). This handler only marshals JSON to the
    // inference helper, so our tests focus on the request-validation
    // surface and shape decisions specific to the OpenAI wire.

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
