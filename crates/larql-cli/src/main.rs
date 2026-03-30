use clap::{Parser, Subcommand};

mod commands;
mod formatting;
mod utils;

use commands::extraction::*;
use commands::query::*;
use commands::surreal::*;

#[derive(Parser)]
#[command(
    name = "larql",
    version,
    about = "LARQL knowledge graph extraction and querying"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    // ── Extraction ──
    /// Extract edges from FFN weights. Zero forward passes.
    WeightExtract(weight_walk_cmd::WeightWalkArgs),

    /// Extract routing edges from attention OV circuits. Zero forward passes.
    AttentionExtract(attention_walk_cmd::AttentionWalkArgs),

    /// Extract full vectors from model weights to NDJSON files.
    VectorExtract(vector_extract_cmd::VectorExtractArgs),

    /// Capture residual stream vectors for entities via forward passes.
    Residuals(residuals_cmd::ResidualsArgs),

    /// Run full forward pass and predict next token.
    Predict(predict_cmd::PredictArgs),

    /// Build gate index for graph-based FFN (offline, run once per model).
    IndexGates(index_gates_cmd::IndexGatesArgs),

    /// Extract attention routing patterns from forward passes.
    ExtractRoutes(extract_routes_cmd::ExtractRoutesArgs),

    /// Walk the model as a local vector index — gate KNN + down token lookup.
    Walk(walk_cmd::WalkArgs),

    /// Capture and compare attention patterns across prompts.
    AttentionCapture(attention_capture_cmd::AttentionCaptureArgs),

    /// Extract attention template circuits from QK weight decomposition.
    QkTemplates(qk_templates_cmd::QkTemplatesArgs),

    /// Map attention OV circuits to FFN gate features (what each head activates).
    OvGate(ov_gate_cmd::OvGateArgs),

    /// Discover attention→FFN circuits from weight decomposition. No forward passes.
    CircuitDiscover(circuit_discover_cmd::CircuitDiscoverArgs),

    /// Build a .vindex — the model decompiled to a standalone vector index.
    ExtractIndex(extract_index_cmd::ExtractIndexArgs),

    /// BFS extraction from a model endpoint.
    Bfs(bfs_cmd::BfsArgs),

    // ── SurrealDB ──
    /// Load vectors into SurrealDB with HNSW indexes (small tables, HTTP).
    VectorLoad(vector_load_cmd::VectorLoadArgs),

    /// Import vectors into SurrealDB via batched `surreal import` (large tables).
    VectorImport(vector_import_cmd::VectorImportArgs),

    /// Export vectors to .surql files for manual import.
    VectorExportSurql(vector_export_surql_cmd::VectorExportSurqlArgs),

    /// Load OV→gate coupling edges into SurrealDB for circuit discovery.
    CouplingLoad(coupling_load_cmd::CouplingLoadArgs),

    // ── Query ──
    /// Query a graph for facts.
    Query(query_cmd::QueryArgs),

    /// Describe an entity (all edges).
    Describe(describe_cmd::DescribeArgs),

    /// Show graph statistics.
    Stats(stats_cmd::StatsArgs),

    /// Validate a graph file.
    Validate(validate_cmd::ValidateArgs),

    /// Merge multiple graph files.
    Merge(merge_cmd::MergeArgs),
}

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        // Extraction
        Commands::WeightExtract(args) => weight_walk_cmd::run(args),
        Commands::AttentionExtract(args) => attention_walk_cmd::run(args),
        Commands::VectorExtract(args) => vector_extract_cmd::run(args),
        Commands::Residuals(args) => residuals_cmd::run(args),
        Commands::Predict(args) => predict_cmd::run(args),
        Commands::IndexGates(args) => index_gates_cmd::run(args),
        Commands::AttentionCapture(args) => attention_capture_cmd::run(args),
        Commands::QkTemplates(args) => qk_templates_cmd::run(args),
        Commands::OvGate(args) => ov_gate_cmd::run(args),
        Commands::CircuitDiscover(args) => circuit_discover_cmd::run(args),
        Commands::ExtractRoutes(args) => extract_routes_cmd::run(args),
        Commands::Walk(args) => walk_cmd::run(args),
        Commands::ExtractIndex(args) => extract_index_cmd::run(args),
        Commands::Bfs(args) => bfs_cmd::run(args),
        // SurrealDB
        Commands::VectorLoad(args) => vector_load_cmd::run(args),
        Commands::VectorImport(args) => vector_import_cmd::run(args),
        Commands::VectorExportSurql(args) => vector_export_surql_cmd::run(args),
        Commands::CouplingLoad(args) => coupling_load_cmd::run(args),
        // Query
        Commands::Query(args) => query_cmd::run(args),
        Commands::Describe(args) => describe_cmd::run(args),
        Commands::Stats(args) => stats_cmd::run(args),
        Commands::Validate(args) => validate_cmd::run(args),
        Commands::Merge(args) => merge_cmd::run(args),
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
