//! Pluggable KV-cache engines.
//!
//! Each engine implements the full prefill + autoregressive decode loop but
//! manages its persistent inference state differently. Engines are selected
//! via [`EngineKind`] and bench via `larql bench --engine`.
//!
//! Correctness contract: `prefill` and `decode_step` return the pre-lm_head
//! hidden state (shape `[1, hidden_dim]`). The caller applies `final_norm +
//! lm_head` to get logits — see `larql_inference::forward::hidden_to_raw_logits`.

pub mod markov_residual;
pub mod unlimited_context;

use ndarray::Array2;
use crate::model::ModelWeights;

/// Runtime diagnostics reported by each engine.
#[derive(Debug, Clone)]
pub struct EngineInfo {
    /// Short engine name (e.g. `"markov-rs"`).
    pub name: String,
    /// Human-readable description of the engine's state management strategy.
    pub description: String,
    /// Hardware backend: `"cpu"`, `"metal"`, etc.
    pub backend: String,
    /// Key config parameters (e.g. `"window=512"`), empty if unconfigured.
    pub config: String,
}

impl EngineInfo {
    pub fn summary(&self) -> String {
        if self.config.is_empty() {
            format!("{} [{}]  {}", self.name, self.backend, self.description)
        } else {
            format!("{} [{}] ({})  {}", self.name, self.backend, self.config, self.description)
        }
    }
}

/// Common interface shared by all KV-cache engines.
pub trait KvEngine: Send {
    fn name(&self) -> &str;

    /// Runtime diagnostics: engine name, backend, config, description.
    fn info(&self) -> EngineInfo;

    /// Run the prefill forward pass over all prompt tokens.
    /// Returns the hidden state at the final token position (shape [1, hidden_dim]).
    fn prefill(&mut self, weights: &ModelWeights, token_ids: &[u32]) -> Option<Array2<f32>>;

    /// Run one autoregressive decode step for a single new token.
    /// Returns the hidden state (shape [1, hidden_dim]).
    fn decode_step(&mut self, weights: &ModelWeights, token_id: u32) -> Option<Array2<f32>>;

    /// Bytes of persistent engine state (excludes model weights).
    fn memory_bytes(&self) -> usize;
}

/// Engine selector. Parse with [`EngineKind::from_name`]; build with [`EngineKind::build`].
#[derive(Debug, Clone)]
pub enum EngineKind {
    MarkovResidual { window_size: Option<usize> },
    UnlimitedContext { window_size: usize },
}

impl EngineKind {
    /// Parse a CLI name into an `EngineKind`. Accepted names:
    /// - `markov-rs`, `markov-residual` → [`EngineKind::MarkovResidual`]
    /// - `unlimited`, `unlimited-context` → [`EngineKind::UnlimitedContext`]
    pub fn from_name(s: &str) -> Option<Self> {
        match s {
            "markov-rs" | "markov_rs" | "markov-residual" | "markov_residual" => {
                Some(EngineKind::MarkovResidual { window_size: None })
            }
            "unlimited" | "unlimited-context" | "unlimited_context" => {
                Some(EngineKind::UnlimitedContext { window_size: 512 })
            }
            _ => None,
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            EngineKind::MarkovResidual { .. } => "markov-rs",
            EngineKind::UnlimitedContext { .. } => "unlimited-context",
        }
    }

    pub fn build(self) -> Box<dyn KvEngine> {
        match self {
            EngineKind::MarkovResidual { window_size } => {
                Box::new(markov_residual::MarkovResidualEngine::new(window_size))
            }
            EngineKind::UnlimitedContext { window_size } => {
                Box::new(unlimited_context::UnlimitedContextEngine::new(window_size))
            }
        }
    }
}
