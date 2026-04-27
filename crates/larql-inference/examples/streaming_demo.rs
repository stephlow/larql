//! Streaming demo — print each token as the model emits it.
//!
//! Demonstrates [`generate_streaming`]'s `on_token` callback. Each token
//! is printed live with stdout flushed after every write so the user sees
//! the response unfold rather than appearing all at once at the end.
//!
//! Compare against `bench_generate.rs` which collects the full result
//! before printing — the buffered version completes faster wall-clock
//! but the streaming version delivers visible tokens with the same
//! latency profile as Ollama / llama.cpp.
//!
//! Usage:
//!   cargo run --release --features metal -p larql-inference \
//!     --example streaming_demo -- --vindex output/gemma3-4b-q4k-v2.vindex
//!
//! Optional flags:
//!   --prompt "<text>"   (default: "The capital of France is")
//!   --max-tokens N      (default: 32)
//!   --temperature F     (default: 0.0 = greedy)
//!   --top-p F           (default: not applied)
//!   --top-k N           (default: not applied)
//!   --seed N            (default: 42 if any sampling flag is set)

use std::io::Write;
use std::time::Instant;

use larql_inference::{
    default_backend, generate_streaming, open_inference_vindex, CachedLayerGraph, EosConfig,
    InferenceModel, SamplingConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut vindex_path = std::path::PathBuf::from("output/gemma3-4b-q4k-v2.vindex");
    let mut prompt = "The capital of France is".to_string();
    let mut max_tokens = 32usize;
    let mut temperature: f32 = 0.0;
    let mut top_p: Option<f32> = None;
    let mut top_k: Option<usize> = None;
    let mut seed: u64 = 42;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--vindex" => {
                i += 1;
                vindex_path = std::path::PathBuf::from(&args[i]);
            }
            "--prompt" => {
                i += 1;
                prompt = args[i].clone();
            }
            "--max-tokens" => {
                i += 1;
                max_tokens = args[i].parse()?;
            }
            "--temperature" => {
                i += 1;
                temperature = args[i].parse()?;
            }
            "--top-p" => {
                i += 1;
                top_p = Some(args[i].parse()?);
            }
            "--top-k" => {
                i += 1;
                top_k = Some(args[i].parse()?);
            }
            "--seed" => {
                i += 1;
                seed = args[i].parse()?;
            }
            _ => {}
        }
        i += 1;
    }

    let mut sampling = SamplingConfig::temperature(temperature);
    if let Some(p) = top_p {
        sampling = sampling.with_top_p(p);
    }
    if let Some(k) = top_k {
        sampling = sampling.with_top_k(k);
    }
    if !sampling.is_greedy() {
        sampling = sampling.with_seed(seed);
    }

    let mut model = InferenceModel::load("google/gemma-3-4b-it")?;
    let num_layers = model.weights().num_layers;
    let tokenizer = model.tokenizer().clone();

    let index = open_inference_vindex(&vindex_path)?;

    let gpu_be = default_backend();
    let encoding = tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| format!("{e}"))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    // No precomputed cache — stream the full transformer end-to-end. The
    // earlier `CachedLayerGraph::build` over `(0..=12)` + generate range
    // `13..num_layers` is invalid for any model whose layers 0-12 contribute
    // anything beyond a dense FFN (hybrid-MoE in particular: the cache built
    // from `WeightFfn` would skip every MoE expert block in those layers and
    // produce multilingual gibberish). Match the convention used by
    // `walk_cmd` and `bench_generate`: empty cache, full layer range.
    let cache = CachedLayerGraph::from_residuals(Vec::new());
    let eos = EosConfig::from_vindex_dir(&vindex_path);

    println!("=== larql-inference: Streaming Demo ===\n");
    println!("Prompt:      \"{prompt}\"");
    println!("Sampling:    {sampling:?}");
    println!("Max tokens:  {max_tokens}");
    println!("Backend:     {}\n", gpu_be.name());
    print!("Output:      ");
    std::io::stdout().flush().ok();

    let start = Instant::now();
    let weights = model.weights_mut();
    let result = generate_streaming(
        weights,
        &tokenizer,
        &token_ids,
        max_tokens,
        &index,
        &*gpu_be,
        &cache,
        0..num_layers,
        sampling,
        &eos,
        |_id, text, _prob| {
            print!("{text}");
            std::io::stdout().flush().ok();
        },
    );
    let wall = start.elapsed().as_secs_f64();
    println!("\n");
    println!("(buffered text: \"{}\")", result.text());
    println!("Tokens emitted: {}", result.tokens.len());
    println!(
        "Decode rate:    {:.1} tok/s ({:.1} ms/tok)",
        result.decode_tok_s(),
        result.avg_decode_ms()
    );
    println!("Wall time:      {wall:.2}s (prefill {:.0}ms)", result.prefill_ms);

    Ok(())
}
