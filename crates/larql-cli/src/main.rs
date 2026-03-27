use clap::{Parser, Subcommand};

mod attention_walk_cmd;
mod bfs_cmd;
mod describe_cmd;
mod formatting;
mod query_cmd;
mod stats_cmd;
mod validate_cmd;
mod vector_extract_cmd;
mod weight_walk_cmd;

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
    /// BFS extraction from a model endpoint.
    Bfs(bfs_cmd::BfsArgs),

    /// Show graph statistics.
    Stats(stats_cmd::StatsArgs),

    /// Validate a .larql.json file.
    Validate(validate_cmd::ValidateArgs),

    /// Query a graph for facts.
    Query(query_cmd::QueryArgs),

    /// Describe an entity.
    Describe(describe_cmd::DescribeArgs),

    /// Walk FFN weights and extract edges. Zero forward passes.
    WeightWalk(weight_walk_cmd::WeightWalkArgs),

    /// Walk attention OV circuits and extract routing edges.
    AttentionWalk(attention_walk_cmd::AttentionWalkArgs),

    /// Extract full vectors from model weights to intermediate NDJSON files.
    VectorExtract(vector_extract_cmd::VectorExtractArgs),
}

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Bfs(args) => bfs_cmd::run(args),
        Commands::Stats(args) => stats_cmd::run(args),
        Commands::Validate(args) => validate_cmd::run(args),
        Commands::Query(args) => query_cmd::run(args),
        Commands::Describe(args) => describe_cmd::run(args),
        Commands::WeightWalk(args) => weight_walk_cmd::run(args),
        Commands::AttentionWalk(args) => attention_walk_cmd::run(args),
        Commands::VectorExtract(args) => vector_extract_cmd::run(args),
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
