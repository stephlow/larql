use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use larql_surreal::loader::{self, CouplingReader};
use larql_surreal::SurrealClient;

#[derive(Args)]
pub struct CouplingLoadArgs {
    /// Path to the OV→gate coupling NDJSON file from `ov-gate --output ndjson`.
    input: PathBuf,

    /// SurrealDB endpoint (e.g. http://localhost:8000).
    #[arg(long, default_value = "http://localhost:8000")]
    surreal: String,

    /// SurrealDB namespace.
    #[arg(long)]
    ns: String,

    /// SurrealDB database.
    #[arg(long)]
    db: String,

    /// SurrealDB username.
    #[arg(long, default_value = "root")]
    user: String,

    /// SurrealDB password.
    #[arg(long, default_value = "root")]
    pass: String,

    /// Batch size for INSERT transactions.
    #[arg(long, default_value = "200")]
    batch_size: usize,

    /// Minimum coupling threshold to load (filters weak edges).
    #[arg(long)]
    min_coupling: Option<f32>,

    /// Create schema only (no data load).
    #[arg(long)]
    schema_only: bool,
}

pub fn run(args: CouplingLoadArgs) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Connecting to SurrealDB: {}", args.surreal);
    eprintln!("  ns={}, db={}", args.ns, args.db);

    let client = SurrealClient::new(&args.surreal, &args.ns, &args.db, &args.user, &args.pass);

    // Setup namespace/database
    let setup_sql = loader::setup_sql(&args.ns, &args.db);
    client.exec(&setup_sql)?;

    // Create ov_gate_coupling schema
    let schema = loader::ov_gate_coupling_schema_sql();
    client.exec(&schema)?;
    eprintln!("  schema: ov_gate_coupling ready");

    if args.schema_only {
        eprintln!("\nSchema created. No data loaded (--schema-only).");
        return Ok(());
    }

    // Stream records from NDJSON
    eprintln!("Loading coupling edges from: {}", args.input.display());
    let start = Instant::now();
    let mut reader = CouplingReader::open(&args.input)?;

    let mut batch = Vec::new();
    let mut total_loaded = 0usize;
    let mut total_skipped = 0usize;

    while let Some(record) = reader.next_record()? {
        // Apply coupling threshold filter
        if let Some(min) = args.min_coupling {
            if record.coupling < min {
                total_skipped += 1;
                continue;
            }
        }

        batch.push(record);

        if batch.len() >= args.batch_size {
            let sql = loader::coupling_batch_insert_sql(&batch);
            client.exec(&sql)?;
            total_loaded += batch.len();
            batch.clear();

            if total_loaded % 2000 == 0 {
                eprint!("\r  {} edges loaded...", total_loaded);
            }
        }
    }

    // Flush remaining batch
    if !batch.is_empty() {
        let sql = loader::coupling_batch_insert_sql(&batch);
        client.exec(&sql)?;
        total_loaded += batch.len();
        batch.clear();
    }

    let elapsed = start.elapsed();
    eprintln!(
        "\r  Loaded {} coupling edges ({} skipped) in {:.1}s",
        total_loaded,
        total_skipped,
        elapsed.as_secs_f64(),
    );

    // Verify count
    let count_resp = client.query(&loader::count_sql("ov_gate_coupling"))?;
    eprintln!("  Verify: {:?}", count_resp);

    Ok(())
}
