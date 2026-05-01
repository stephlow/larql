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
//! ## Implementation note (slice 1)
//!
//! This first slice runs an **un-KV-cached generation loop** —
//! `larql_inference::predict_with_temperature` is invoked once per
//! generated token, re-running the full forward pass each step. Cost is
//! O(N²) in context length. Functional and immutable
//! (`&ModelWeights`-only), so it serializes cleanly with concurrent
//! `/v1/infer` traffic.
//!
//! The fast KV-cached path (`larql_inference::layer_graph::generate`)
//! requires `&mut ModelWeights` for the per-layer Q4_K dequant cache.
//! Wiring that into `LoadedModel` requires putting `ModelWeights` behind
//! a `RwLock` (every existing `&ModelWeights` reader becomes a read-guard
//! holder); roadmap'd as N0.2-fast.
//!
//! ## Streaming
//!
//! `stream: true` returns 501 in this slice. SSE arrives in N0.1 streaming
//! along with `/v1/chat/completions/stream`.
//!
//! ## Logprobs
//!
//! `logprobs: int` returns `null` in the response. Top-k log-probabilities
//! over the lm_head distribution land in F18.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::error::ServerError;
use crate::state::{AppState, LoadedModel};

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

#[derive(Deserialize)]
#[serde(untagged)]
pub enum StopSpec {
    Single(String),
    Multi(Vec<String>),
}

impl StopSpec {
    fn as_slice(&self) -> &[String] {
        match self {
            StopSpec::Single(s) => std::slice::from_ref(s),
            StopSpec::Multi(v) => v.as_slice(),
        }
    }
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
) -> Result<Json<CompletionsResponse>, ServerError> {
    state.bump_requests();

    if req.stream.unwrap_or(false) {
        return Err(ServerError::BadRequest(
            "stream=true not yet supported on /v1/completions; SSE arrives in N0.1 \
             (see ROADMAP). Use stream=false."
                .into(),
        ));
    }
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
    let model_id = req
        .model
        .clone()
        .unwrap_or_else(|| model.id.clone());
    let model_arc = Arc::clone(&model);

    // Run the generation loop on the blocking pool so the tokio runtime
    // stays responsive to other requests.
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
        created: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        model: model_id,
        choices,
        usage: CompletionsUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    }))
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
            let pred = larql_inference::forward::predict_with_temperature(
                weights,
                &model.tokenizer,
                &ids,
                1,
                temperature,
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

fn contains_any(haystack: &str, needles: &[String]) -> bool {
    needles.iter().any(|n| !n.is_empty() && haystack.contains(n.as_str()))
}

fn trim_at_stop(haystack: &str, needles: &[String]) -> String {
    let mut earliest: Option<usize> = None;
    for n in needles {
        if n.is_empty() {
            continue;
        }
        if let Some(idx) = haystack.find(n.as_str()) {
            earliest = Some(earliest.map_or(idx, |e| e.min(idx)));
        }
    }
    match earliest {
        Some(i) => haystack[..i].to_string(),
        None => haystack.to_string(),
    }
}

/// Generate a short hex id suffix for `cmpl-...`. Not cryptographically
/// strong; uniqueness across one server lifetime is sufficient.
fn new_id_suffix() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let now_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    format!("{:016x}{:08x}", now_ns, n)
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

    #[test]
    fn stop_spec_single_or_multi() {
        let single: StopSpec = serde_json::from_value(serde_json::json!("\\n")).unwrap();
        assert_eq!(single.as_slice(), &["\\n".to_string()]);
        let multi: StopSpec = serde_json::from_value(serde_json::json!(["a", "b"])).unwrap();
        assert_eq!(multi.as_slice(), &["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn trim_at_stop_finds_earliest() {
        let s = "hello world stop here";
        let stops = vec!["stop".to_string(), "world".to_string()];
        assert_eq!(trim_at_stop(s, &stops), "hello ");
    }

    #[test]
    fn contains_any_matches_substring() {
        let stops = vec!["END".to_string()];
        assert!(contains_any("text END more", &stops));
        assert!(!contains_any("text only", &stops));
    }

    #[test]
    fn new_id_suffix_is_unique_within_thread() {
        let a = new_id_suffix();
        let b = new_id_suffix();
        assert_ne!(a, b);
        assert_eq!(a.len(), b.len());
    }
}
