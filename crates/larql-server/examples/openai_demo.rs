//! Live OpenAI-compat demo — boots an in-process larql server and
//! exercises `/v1/models`, `/v1/embeddings`, `/v1/completions`,
//! `/v1/chat/completions` end-to-end against the loaded vindex.
//!
//! Usage:
//!   cargo run -p larql-server --example openai_demo -- <vindex_path>
//!
//! ## Vindex compatibility
//!
//! Both **f16** and **Q4_K** vindexes produce correct, intelligible
//! output now that the KV-cached generation path is wired up
//! (e.g. "The capital of France is" → " Paris.").
//!
//! ```bash
//! # f16 (fastest, KV-cached):
//! cargo run --release -p larql-server --example openai_demo -- \
//!   output/gemma3-4b-f16.vindex
//!
//! # Q4_K (correct output; CPU per-step Q4_K decode is O(N²) so
//! # high `max_tokens` runs are slow):
//! cargo run --release -p larql-server --example openai_demo -- \
//!   output/gemma3-4b-q4k-streaming.vindex
//! ```
//!
//! Pattern mirrors `bench_embed_server` / `bench_expert_server`: build
//! the router via `tower::ServiceExt::oneshot`, no port binding, no
//! external HTTP client. The wire shapes are real — captured from the
//! same router that the production binary uses.

use std::path::PathBuf;
use std::sync::{atomic::AtomicU64, Arc};
use std::time::Instant;

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use axum::Router;
use serde_json::Value;
use tower::ServiceExt;

use larql_server::{
    bootstrap::{load_single_vindex, LoadVindexOptions},
    cache::DescribeCache,
    routes::single_model_router,
    session::SessionManager,
    state::{AppState, LoadedModel},
};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn section(title: &str) {
    println!("\n══ {title} ══");
}

fn pretty(value: &Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_else(|_| "<serialize error>".into())
}

/// Trim large arrays (embeddings) to the first N + "...total: K" so the
/// printed JSON stays readable. Recursive.
fn trim_arrays_for_print(v: &Value, head: usize) -> Value {
    match v {
        Value::Array(a) if a.len() > head + 2 => {
            let mut head_vals: Vec<Value> = a.iter().take(head).cloned().collect();
            head_vals.push(Value::String(format!(
                "...{} more elements (total: {})",
                a.len() - head,
                a.len()
            )));
            Value::Array(head_vals)
        }
        Value::Array(a) => Value::Array(a.iter().map(|x| trim_arrays_for_print(x, head)).collect()),
        Value::Object(m) => Value::Object(
            m.iter()
                .map(|(k, x)| (k.clone(), trim_arrays_for_print(x, head)))
                .collect(),
        ),
        other => other.clone(),
    }
}

async fn get_json(app: &Router, path: &str) -> (StatusCode, Value) {
    let resp = app
        .clone()
        .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
        .await
        .expect("oneshot get");
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024 * 1024)
        .await
        .expect("read body");
    let json: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, json)
}

async fn post_json(app: &Router, path: &str, body: &Value) -> (StatusCode, Value) {
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(path)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(body).unwrap()))
                .unwrap(),
        )
        .await
        .expect("oneshot post");
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024 * 1024)
        .await
        .expect("read body");
    let json: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, json)
}

// ── Server boot ───────────────────────────────────────────────────────────────

fn make_app_state(model: LoadedModel) -> Arc<AppState> {
    Arc::new(AppState {
        models: vec![Arc::new(model)],
        started_at: Instant::now(),
        requests_served: AtomicU64::new(0),
        api_key: None,
        sessions: SessionManager::new(3600),
        describe_cache: DescribeCache::new(60),
    })
}

fn load_default(path: &str) -> Result<LoadedModel, Box<dyn std::error::Error + Send + Sync>> {
    let opts = LoadVindexOptions {
        no_infer: false,
        ffn_only: false,
        embed_only: false,
        layer_range: None,
        max_gate_cache_layers: 0,
        max_q4k_cache_layers: 0,
        hnsw: None,
        warmup_hnsw: false,
        release_mmap_after_request: false,
        expert_filter: None,
        unit_filter: None,
    };
    Ok(load_single_vindex(path, opts)?)
}

// ── Demos ─────────────────────────────────────────────────────────────────────

async fn demo_models(app: &Router) {
    section("GET /v1/models");
    let t = Instant::now();
    let (status, body) = get_json(app, "/v1/models").await;
    println!("Status: {status}  ({} ms)", t.elapsed().as_millis());
    println!("{}", pretty(&body));
    println!(
        "\nNote: `id`, `object`, `created`, `owned_by` are the OpenAI required\n\
         fields. `path`, `features`, `loaded` are larql-specific extras —\n\
         OpenAI SDKs ignore unknown fields."
    );
}

async fn demo_embeddings(app: &Router, model_id: &str) {
    section("POST /v1/embeddings — single string");
    let req = serde_json::json!({"model": model_id, "input": "France"});
    println!("Request body:\n{}", pretty(&req));
    let t = Instant::now();
    let (status, body) = post_json(app, "/v1/embeddings", &req).await;
    println!("\nStatus: {status}  ({} ms)", t.elapsed().as_millis());
    println!("{}", pretty(&trim_arrays_for_print(&body, 4)));
    let dim = body
        .get("data")
        .and_then(|d| d.as_array())
        .and_then(|a| a.first())
        .and_then(|e| e.get("embedding"))
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    if dim > 0 {
        println!("\n→ {dim}-dim mean-pooled lookup vector");
    }

    section("POST /v1/embeddings — string array");
    let req = serde_json::json!({"model": model_id, "input": ["France", "Germany", "Japan"]});
    println!("Request body:\n{}", pretty(&req));
    let t = Instant::now();
    let (status, body) = post_json(app, "/v1/embeddings", &req).await;
    println!("\nStatus: {status}  ({} ms)", t.elapsed().as_millis());
    println!("{}", pretty(&trim_arrays_for_print(&body, 3)));

    section("POST /v1/embeddings — base64 (returns 400 in slice 1)");
    let req = serde_json::json!({"model": model_id, "input": "x", "encoding_format": "base64"});
    let t = Instant::now();
    let (status, body) = post_json(app, "/v1/embeddings", &req).await;
    println!("Status: {status}  ({} ms)", t.elapsed().as_millis());
    println!("{}", pretty(&body));
}

async fn demo_completions(app: &Router, model_id: &str) {
    section("POST /v1/completions — non-streaming");
    let req = serde_json::json!({
        "model": model_id,
        "prompt": "The capital of France is",
        "max_tokens": 5,
        "temperature": 0.0
    });
    println!("Request body:\n{}", pretty(&req));
    let t = Instant::now();
    let (status, body) = post_json(app, "/v1/completions", &req).await;
    println!("\nStatus: {status}  ({} ms)", t.elapsed().as_millis());
    println!("{}", pretty(&body));
    println!(
        "\nNote: generation runs through the KV-cached path\n\
         (`larql_inference::layer_graph::generate_with_sampling`) on\n\
         f16 vindexes, with a per-step Q4_K fallback on CPU+Q4_K\n\
         vindexes. Output text quality depends on the base model."
    );

    section("POST /v1/completions — temperature + top_p + seed (reproducible)");
    let req = serde_json::json!({
        "model": model_id,
        "prompt": "Once upon a time",
        "max_tokens": 6,
        "temperature": 0.8,
        "top_p": 0.9,
        "seed": 42
    });
    println!("Request body:\n{}", pretty(&req));
    let t = Instant::now();
    let (status, body) = post_json(app, "/v1/completions", &req).await;
    println!("\nStatus: {status}  ({} ms)", t.elapsed().as_millis());
    println!("{}", pretty(&body));
    println!(
        "\nNote: seed=42 + temperature>0 makes output reproducible —\n\
         re-running with the same prompt and seed yields the same\n\
         tokens. Drop the seed to get a fresh sample each call."
    );

    section("POST /v1/completions — n=3 (returns 400)");
    let req = serde_json::json!({
        "model": model_id,
        "prompt": "x",
        "max_tokens": 1,
        "n": 3
    });
    let t = Instant::now();
    let (status, body) = post_json(app, "/v1/completions", &req).await;
    println!("Status: {status}  ({} ms)", t.elapsed().as_millis());
    println!("{}", pretty(&body));
}

async fn demo_chat_completions(app: &Router, model_id: &str) {
    section("POST /v1/chat/completions — non-streaming (slice 2)");
    let req = serde_json::json!({
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You are concise."},
            {"role": "user",   "content": "What is the capital of France?"}
        ],
        "max_tokens": 8,
        "temperature": 0.0
    });
    println!("Request body:\n{}", pretty(&req));
    let t = Instant::now();
    let (status, body) = post_json(app, "/v1/chat/completions", &req).await;
    println!("\nStatus: {status}  ({} ms)", t.elapsed().as_millis());
    println!("{}", pretty(&body));
    println!(
        "\nNote: messages render through the model's chat template\n\
         (Gemma / Llama / ChatML / Mistral / Plain) before going into\n\
         the KV-cached generation loop. Sampling fields\n\
         (temperature, top_p, seed, stop) plumb through the same way\n\
         /v1/completions wires them."
    );

    section("POST /v1/chat/completions — tools field (returns 400 in slice 2)");
    let req = serde_json::json!({
        "model": model_id,
        "messages": [{"role": "user", "content": "x"}],
        "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
        "max_tokens": 1
    });
    let t = Instant::now();
    let (status, body) = post_json(app, "/v1/chat/completions", &req).await;
    println!("Status: {status}  ({} ms)", t.elapsed().as_millis());
    println!("{}", pretty(&body));

    section("POST /v1/chat/completions — response_format json_schema (returns 400)");
    let req = serde_json::json!({
        "model": model_id,
        "messages": [{"role": "user", "content": "x"}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}},
        "max_tokens": 1
    });
    let t = Instant::now();
    let (status, body) = post_json(app, "/v1/chat/completions", &req).await;
    println!("Status: {status}  ({} ms)", t.elapsed().as_millis());
    println!("{}", pretty(&body));
}

fn print_client_snippets(model_id: &str) {
    section("Equivalent client code");
    println!(
        "Python (openai SDK):\n\
         \n\
             from openai import OpenAI\n\
             client = OpenAI(\n\
                 base_url=\"http://localhost:8080/v1\",\n\
                 api_key=\"sk-anything\",  # required by SDK; matched against --api-key if set\n\
             )\n\
             # /v1/models\n\
             models = client.models.list()\n\
             # /v1/embeddings\n\
             emb = client.embeddings.create(\n\
                 model=\"{model_id}\",\n\
                 input=\"France\",\n\
             )\n\
             # /v1/completions\n\
             resp = client.completions.create(\n\
                 model=\"{model_id}\",\n\
                 prompt=\"The capital of France is\",\n\
                 max_tokens=10,\n\
             )\n\
             # /v1/chat/completions\n\
             chat = client.chat.completions.create(\n\
                 model=\"{model_id}\",\n\
                 messages=[\n\
                     {{\"role\": \"system\", \"content\": \"You are concise.\"}},\n\
                     {{\"role\": \"user\",   \"content\": \"Capital of France?\"}},\n\
                 ],\n\
                 max_tokens=10,\n\
             )\n\
             print(chat.choices[0].message.content)\n\
         \n\
         curl:\n\
         \n\
             curl http://localhost:8080/v1/models\n\
             curl -X POST http://localhost:8080/v1/embeddings \\\n\
                  -H 'Content-Type: application/json' \\\n\
                  -d '{{\"model\": \"{model_id}\", \"input\": \"France\"}}'\n\
             curl -X POST http://localhost:8080/v1/completions \\\n\
                  -H 'Content-Type: application/json' \\\n\
                  -d '{{\"model\": \"{model_id}\", \"prompt\": \"The capital of France is\", \"max_tokens\": 5}}'\n\
             curl -X POST http://localhost:8080/v1/chat/completions \\\n\
                  -H 'Content-Type: application/json' \\\n\
                  -d '{{\"model\": \"{model_id}\", \"messages\": [{{\"role\": \"user\", \"content\": \"Capital of France?\"}}], \"max_tokens\": 5}}'"
    );
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn,larql_server=info")),
        )
        .with_target(false)
        .try_init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: openai_demo <vindex_path>\n\n\
             Boots an in-process larql server (no port binding, no external\n\
             HTTP client) and exercises the OpenAI-compat endpoints end-to-\n\
             end against the loaded vindex.\n\n\
             Examples:\n\
               cargo run --release -p larql-server --example openai_demo -- \\\n\
                 output/gemma3-4b-q4k-streaming.vindex"
        );
        std::process::exit(1);
    }
    let vindex_path = PathBuf::from(&args[1]);

    println!("── larql-server OpenAI-compat live demo ──");
    println!("Vindex: {}", vindex_path.display());

    let t = Instant::now();
    let model = load_default(&args[1])?;
    let model_id = model.id.clone();
    let hidden = model.config.hidden_size;
    let num_layers = model.config.num_layers;
    println!(
        "Loaded {} ({} layers, hidden={}) in {} ms",
        model_id,
        num_layers,
        hidden,
        t.elapsed().as_millis(),
    );

    let state = make_app_state(model);
    let app = single_model_router(state);

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    runtime.block_on(async {
        demo_models(&app).await;
        demo_embeddings(&app, &model_id).await;
        demo_completions(&app, &model_id).await;
        demo_chat_completions(&app, &model_id).await;
    });

    print_client_snippets(&model_id);

    section("Done");
    println!(
        "Boot a public server with the same vindex and the same endpoints\n\
         become reachable from any OpenAI SDK:\n\
         \n\
           larql-server {} --port 8080\n\
         \n\
         Then point `base_url=\"http://localhost:8080/v1\"` and your\n\
         existing OpenAI Python or JS client works unmodified.",
        vindex_path.display()
    );
    Ok(())
}
