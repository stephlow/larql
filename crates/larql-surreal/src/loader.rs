//! Read intermediate vector NDJSON files and generate SurrealDB SQL for loading.
//!
//! Pure functions — no database I/O. The CLI module handles HTTP communication
//! to SurrealDB. This module provides:
//! - Streaming NDJSON reader for VectorRecord files
//! - Schema DDL generation per component type
//! - Batch INSERT SQL generation

use std::collections::HashSet;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use larql_models::{VectorFileHeader, VectorRecord, ALL_COMPONENTS};

use crate::error::SurrealError;

/// Configuration for the vector loader.
pub struct LoadConfig {
    pub tables: Option<Vec<String>>,
    pub layers: Option<Vec<usize>>,
    pub batch_size: usize,
}

impl Default for LoadConfig {
    fn default() -> Self {
        Self {
            tables: None,
            layers: None,
            batch_size: 500,
        }
    }
}

/// Callbacks for load progress.
pub trait LoadCallbacks {
    fn on_table_start(&mut self, _table: &str, _total_records: usize) {}
    fn on_batch_done(&mut self, _table: &str, _batch_num: usize, _records_loaded: usize) {}
    fn on_table_done(&mut self, _table: &str, _total_loaded: usize, _elapsed_ms: f64) {}
}

pub struct SilentLoadCallbacks;
impl LoadCallbacks for SilentLoadCallbacks {}

/// Summary of a load run.
pub struct LoadSummary {
    pub tables: Vec<TableSummary>,
    pub total_records: usize,
    pub elapsed_secs: f64,
}

pub struct TableSummary {
    pub table: String,
    pub records_loaded: usize,
    pub elapsed_secs: f64,
}

// ═══════════════════════════════════════════════════
// NDJSON Reader
// ═══════════════════════════════════════════════════

/// Streaming reader for .vectors.jsonl files.
pub struct VectorReader {
    reader: BufReader<std::fs::File>,
    header: VectorFileHeader,
    line_buf: String,
}

impl VectorReader {
    /// Open a .vectors.jsonl file and read its header.
    pub fn open(path: &Path) -> Result<Self, SurrealError> {
        let file = std::fs::File::open(path)?;
        let mut reader = BufReader::new(file);

        let mut header_line = String::new();
        reader
            .read_line(&mut header_line)
            .map_err(|e| SurrealError::Parse(format!("failed to read header: {e}")))?;

        let header: VectorFileHeader = serde_json::from_str(&header_line)
            .map_err(|e| SurrealError::Parse(format!("invalid header: {e}")))?;

        Ok(Self {
            reader,
            header,
            line_buf: String::new(),
        })
    }

    pub fn header(&self) -> &VectorFileHeader {
        &self.header
    }

    /// Read the next vector record. Returns None at EOF.
    pub fn next_record(&mut self) -> Result<Option<VectorRecord>, SurrealError> {
        self.line_buf.clear();
        let bytes = self.reader.read_line(&mut self.line_buf)?;
        if bytes == 0 {
            return Ok(None);
        }
        let trimmed = self.line_buf.trim();
        if trimmed.is_empty() {
            return Ok(None);
        }
        let record: VectorRecord = serde_json::from_str(trimmed)
            .map_err(|e| SurrealError::Parse(format!("invalid record: {e}")))?;
        Ok(Some(record))
    }

    /// Collect all records, optionally filtered by layer.
    pub fn read_all(
        &mut self,
        layer_filter: Option<&HashSet<usize>>,
    ) -> Result<Vec<VectorRecord>, SurrealError> {
        let mut records = Vec::new();
        while let Some(record) = self.next_record()? {
            if let Some(layers) = layer_filter {
                if !layers.contains(&record.layer) {
                    continue;
                }
            }
            records.push(record);
        }
        Ok(records)
    }
}

/// Discover .vectors.jsonl files in a directory.
pub fn discover_vector_files(dir: &Path) -> Result<Vec<(String, PathBuf)>, SurrealError> {
    if !dir.is_dir() {
        return Err(SurrealError::NoVectorFiles(dir.to_path_buf()));
    }

    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if name.ends_with(".vectors.jsonl") {
                let component = name.trim_end_matches(".vectors.jsonl").to_string();
                if ALL_COMPONENTS.contains(&component.as_str()) {
                    files.push((component, path));
                }
            }
        }
    }

    files.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(files)
}

// ═══════════════════════════════════════════════════
// Schema DDL Generation
// ═══════════════════════════════════════════════════

const SETUP_TEMPLATE: &str = include_str!("../surql/setup.surql");
const SCHEMA_TEMPLATE: &str = include_str!("../surql/component_schema.surql");
const PROGRESS_TEMPLATE: &str = include_str!("../surql/load_progress.surql");
const OV_GATE_COUPLING_TEMPLATE: &str = include_str!("../surql/ov_gate_coupling_schema.surql");

/// Generate the full schema DDL for a namespace + database setup.
pub fn setup_sql(ns: &str, db: &str) -> String {
    SETUP_TEMPLATE.replace("{ns}", ns).replace("{db}", db)
}

/// Generate schema DDL for a single component table.
pub fn schema_sql(component: &str, dimension: usize) -> Result<String, SurrealError> {
    if !ALL_COMPONENTS.contains(&component) {
        return Err(SurrealError::UnknownComponent(component.to_string()));
    }

    Ok(SCHEMA_TEMPLATE
        .replace("{table}", component)
        .replace("{dimension}", &dimension.to_string()))
}

/// Generate load_progress table schema.
pub fn progress_table_sql() -> String {
    PROGRESS_TEMPLATE.to_string()
}

// ═══════════════════════════════════════════════════
// INSERT SQL Generation
// ═══════════════════════════════════════════════════

/// Generate a batch INSERT transaction for a slice of vector records.
pub fn batch_insert_sql(table: &str, records: &[VectorRecord]) -> String {
    let mut sql = String::from("BEGIN TRANSACTION;\n");

    for record in records {
        let content = serde_json::json!({
            "layer": record.layer,
            "feature": record.feature,
            "vector": record.vector,
            "top_token": record.top_token,
            "top_token_id": record.top_token_id,
            "c_score": record.c_score,
            "top_k": record.top_k,
        });

        sql.push_str(&format!(
            "CREATE {table}:{id} CONTENT {json};\n",
            id = record.id,
            json = content,
        ));
    }

    sql.push_str("COMMIT TRANSACTION;");
    sql
}

/// Generate a single-record INSERT SQL.
pub fn single_insert_sql(table: &str, record: &VectorRecord) -> String {
    let content = serde_json::json!({
        "layer": record.layer,
        "feature": record.feature,
        "vector": record.vector,
        "top_token": record.top_token,
        "top_token_id": record.top_token_id,
        "c_score": record.c_score,
        "top_k": record.top_k,
    });

    format!(
        "CREATE {table}:{id} CONTENT {json};",
        id = record.id,
        json = content,
    )
}

/// Generate SQL to mark a layer as completed in load_progress.
pub fn mark_layer_done_sql(table: &str, layer: usize, count: usize) -> String {
    format!(
        "CREATE load_progress:{table}_L{layer} CONTENT {{\
            \"table_name\": \"{table}\",\
            \"layer\": {layer},\
            \"vectors_loaded\": {count},\
            \"completed\": true,\
            \"timestamp\": time::now()\
        }};"
    )
}

/// Generate SQL to query completed layers for a table.
pub fn completed_layers_sql(table: &str) -> String {
    format!("SELECT layer FROM load_progress WHERE table_name = '{table}' AND completed = true;")
}

/// Count records in a table.
pub fn count_sql(table: &str) -> String {
    format!("SELECT count() FROM {table} GROUP ALL;")
}

// ═══════════════════════════════════════════════════
// OV→Gate Coupling Edge Loading
// ═══════════════════════════════════════════════════

/// Generate schema DDL for the ov_gate_coupling table.
pub fn ov_gate_coupling_schema_sql() -> String {
    OV_GATE_COUPLING_TEMPLATE.to_string()
}

/// A single OV→gate coupling record from NDJSON.
#[derive(serde::Deserialize)]
pub struct CouplingRecord {
    pub head: String,
    pub layer: usize,
    pub head_idx: usize,
    pub feature: String,
    pub feature_idx: usize,
    pub feature_layer: usize,
    pub target_token: String,
    pub coupling: f32,
    pub total_coupling: f32,
}

/// Generate INSERT SQL for a single coupling record.
pub fn coupling_insert_sql(record: &CouplingRecord) -> String {
    let id = format!("{}_{}", record.head, record.feature);
    let content = serde_json::json!({
        "head": record.head,
        "layer": record.layer,
        "head_idx": record.head_idx,
        "feature": record.feature,
        "feature_idx": record.feature_idx,
        "feature_layer": record.feature_layer,
        "target_token": record.target_token,
        "coupling": record.coupling,
        "total_coupling": record.total_coupling,
    });

    format!("CREATE ov_gate_coupling:{id} CONTENT {json};", json = content)
}

/// Generate a batch INSERT transaction for coupling records.
pub fn coupling_batch_insert_sql(records: &[CouplingRecord]) -> String {
    let mut sql = String::from("BEGIN TRANSACTION;\n");

    for record in records {
        sql.push_str(&coupling_insert_sql(record));
        sql.push('\n');
    }

    sql.push_str("COMMIT TRANSACTION;");
    sql
}

/// Streaming reader for coupling NDJSON files.
pub struct CouplingReader {
    reader: std::io::BufReader<std::fs::File>,
    line_buf: String,
}

impl CouplingReader {
    /// Open a coupling NDJSON file and skip the header.
    pub fn open(path: &Path) -> Result<Self, SurrealError> {
        let file = std::fs::File::open(path)?;
        let mut reader = std::io::BufReader::new(file);

        // Read and validate header
        let mut header_line = String::new();
        reader
            .read_line(&mut header_line)
            .map_err(|e| SurrealError::Parse(format!("failed to read header: {e}")))?;

        let header: serde_json::Value = serde_json::from_str(&header_line)
            .map_err(|e| SurrealError::Parse(format!("invalid header: {e}")))?;

        if header.get("type").and_then(|v| v.as_str()) != Some("ov_gate_coupling") {
            return Err(SurrealError::Parse(
                "not an ov_gate_coupling file (wrong header type)".to_string(),
            ));
        }

        Ok(Self {
            reader,
            line_buf: String::new(),
        })
    }

    /// Read the next coupling record. Returns None at EOF.
    pub fn next_record(&mut self) -> Result<Option<CouplingRecord>, SurrealError> {
        self.line_buf.clear();
        let bytes = self.reader.read_line(&mut self.line_buf)?;
        if bytes == 0 {
            return Ok(None);
        }
        let trimmed = self.line_buf.trim();
        if trimmed.is_empty() {
            return Ok(None);
        }
        let record: CouplingRecord = serde_json::from_str(trimmed)
            .map_err(|e| SurrealError::Parse(format!("invalid coupling record: {e}")))?;
        Ok(Some(record))
    }
}
