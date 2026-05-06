use super::super::pq::ModeDTable;
use super::super::types::{HeadId, PqConfig};

/// Per-eval-prompt captured data. Built once from the fitted codebook;
/// reused across every proposal evaluation without re-running oracle PQ.
pub struct PromptCapture {
    pub id: String,
    pub stratum: String,
    pub token_ids: Vec<u32>,
    /// Oracle codes per position per group.
    pub oracle_codes: Vec<Vec<usize>>,
    /// Attention row per position (over all key positions, causal).
    pub attention_rows: Vec<Vec<f32>>,
    /// Log-softmax of baseline logits (full vocab).
    pub baseline_logp: Vec<f64>,
    pub baseline_top1: u32,
}

/// Everything needed to evaluate proposals without refitting.
pub struct FitContext {
    pub head: HeadId,
    pub group: usize,
    pub config: PqConfig,
    pub mode_d_table: ModeDTable,
    pub captures: Vec<PromptCapture>,
    pub codebook_fingerprint: Option<String>,
}

impl FitContext {
    pub fn num_codes(&self) -> usize {
        1 << self.config.bits_per_group
    }
}
