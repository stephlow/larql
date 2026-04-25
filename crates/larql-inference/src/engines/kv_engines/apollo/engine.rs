//! ApolloEngine — retrieval-augmented generation via vec_inject.
//!
//! At prefill: routes the prompt through the RoutingIndex, retrieves the
//! most relevant VecInjectEntry records, computes a combined injection delta
//! (scaled token embeddings), then runs the forward pass on the context
//! (window_tokens ++ query_tokens) with the delta injected at `crystal_layer`.
//!
//! At decode: extends the context by one token per step and re-runs the
//! forward pass with the same injection delta. Generation is O(N) per step —
//! there is no K/V cache; accuracy comes from the injection residual.
//!
//! Memory: ~2.8 MB for 176 windows × 3,585 entries on the Apollo 11 corpus,
//! vs ~25.8 GB Standard KV at 370K tokens (~20,000× compression).
//!
//! Simplifications vs the full Python pipeline:
//! - Injection is at the last token position only (Python does per-entry
//!   `position_in_window`).
//! - Routing uses tf-idf-lite on raw token IDs (no stemming/stopwords).
//! - Boundary-residual replay not yet wired (`prefill_to_layer` is a TODO).

use ndarray::{s, Array1, Array2};
use thiserror::Error;

use super::entry::{InjectionConfig, VecInjectEntry};
use super::routing::{RoutingIndex, RoutingQuery};
use super::store::ApolloStore;
use crate::model::ModelWeights;
use crate::forward::{embed_tokens_pub, forward_raw_logits};
use crate::engines::{EngineInfo, KvEngine};

// ─── Error ────────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum ApolloError {
    #[error("store not loaded")]
    StoreNotLoaded,
    #[error("routing index not built — call build_routing_index() first")]
    RoutingNotBuilt,
    #[error("invalid window id: {0}")]
    InvalidWindowId(u16),
    #[error("forward pass failed")]
    Forward,
    #[error("no windows matched query (routing returned empty)")]
    NoMatch,
}

// ─── Trace types ─────────────────────────────────────────────────────────────

/// Summary of a single query answered by the engine.
#[derive(Debug, Clone)]
pub struct QueryTrace {
    pub routed_windows: Vec<u16>,
    pub injected_entries: Vec<VecInjectEntry>,
    pub context_tokens: usize,
    pub top1_token_id: u32,
    pub top1_logit: f32,
}

// ─── Engine struct ────────────────────────────────────────────────────────────

pub struct ApolloEngine {
    pub store: Option<ApolloStore>,
    pub routing: RoutingIndex,
    pub config: InjectionConfig,
    /// State maintained between prefill and decode steps.
    context_tokens: Vec<u32>,
    injection_delta: Option<Array1<f32>>,
}

impl ApolloEngine {
    pub fn new(config: InjectionConfig) -> Self {
        Self {
            store: None,
            routing: RoutingIndex::new(),
            config,
            context_tokens: Vec::new(),
            injection_delta: None,
        }
    }

    pub fn with_store(mut self, store: ApolloStore) -> Self {
        self.store = Some(store);
        self
    }

    pub fn build_routing_index(&mut self) -> Result<(), ApolloError> {
        let store = self.store.as_ref().ok_or(ApolloError::StoreNotLoaded)?;
        self.routing = RoutingIndex::from_store(store);
        Ok(())
    }

    pub fn config(&self) -> &InjectionConfig { &self.config }
    pub fn has_store(&self) -> bool { self.store.is_some() }
    pub fn store(&self) -> Option<&ApolloStore> { self.store.as_ref() }
    pub fn routing(&self) -> &RoutingIndex { &self.routing }

    /// Return the top-k entries most relevant to `query_token_ids`,
    /// scoped to `candidate_windows`. Uses seed + proximity + fact-group +
    /// backfill ranking.
    pub fn retrieve_entries(
        &self,
        query_token_ids: &[u32],
        candidate_windows: &[u16],
    ) -> Result<Vec<VecInjectEntry>, ApolloError> {
        const PROXIMITY_RADIUS: u16 = 10;
        let store = self.store.as_ref().ok_or(ApolloError::StoreNotLoaded)?;
        if query_token_ids.is_empty() { return Ok(vec![]); }
        let qset: std::collections::HashSet<u32> = query_token_ids.iter().copied().collect();
        let wset: std::collections::HashSet<u16> = candidate_windows.iter().copied().collect();
        let in_candidate = |e: &VecInjectEntry| wset.is_empty() || wset.contains(&e.window_id);
        let entry_key = |e: &VecInjectEntry| (e.window_id, e.position_in_window, e.token_id, e.fact_id);

        let seeds: Vec<&VecInjectEntry> = store.entries.iter()
            .filter(|e| in_candidate(e) && qset.contains(&e.token_id))
            .collect();

        if seeds.is_empty() {
            let mut scored: Vec<(VecInjectEntry, f32)> = store.entries.iter()
                .filter(|e| in_candidate(e))
                .map(|e| (*e, e.coefficient))
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(self.config.top_k);
            return Ok(scored.into_iter().map(|(e, _)| e).collect());
        }

        let seed_facts: std::collections::HashSet<u16> = seeds.iter().map(|e| e.fact_id).collect();
        let seed_positions: std::collections::HashSet<(u16, u16)> = seeds.iter()
            .map(|e| (e.window_id, e.position_in_window))
            .collect();

        let mut scored: Vec<(VecInjectEntry, f32)> = Vec::new();
        let mut seen: std::collections::HashSet<(u16, u16, u32, u16)> = std::collections::HashSet::new();

        for e in &seeds {
            scored.push((**e, e.coefficient));
            seen.insert(entry_key(e));
        }
        for e in store.entries.iter().filter(|e| in_candidate(e)) {
            let k = entry_key(e);
            if seen.contains(&k) { continue; }
            let near = seed_positions.iter().any(|(w, p)| {
                *w == e.window_id && (e.position_in_window as i32 - *p as i32).abs() <= PROXIMITY_RADIUS as i32
            });
            if near { scored.push((*e, e.coefficient * 1.3)); seen.insert(k); }
        }
        for e in store.entries.iter().filter(|e| in_candidate(e) && seed_facts.contains(&e.fact_id)) {
            let k = entry_key(e);
            if !seen.contains(&k) { scored.push((*e, e.coefficient * 1.3)); seen.insert(k); }
        }
        if scored.len() < self.config.top_k {
            let mut pool: Vec<&VecInjectEntry> = store.entries.iter()
                .filter(|e| in_candidate(e) && !seen.contains(&entry_key(e)))
                .collect();
            pool.sort_by(|a, b| b.coefficient.partial_cmp(&a.coefficient).unwrap_or(std::cmp::Ordering::Equal));
            for e in pool.into_iter().take(self.config.top_k - scored.len()) {
                scored.push((*e, e.coefficient * 0.8));
            }
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(self.config.top_k);
        Ok(scored.into_iter().map(|(e, _)| e).collect())
    }

    /// Build the injection delta and initial context for a set of query tokens.
    /// Returns `(context_tokens, injection_delta)`.
    fn prepare_injection(
        &self,
        weights: &ModelWeights,
        query_ids: &[u32],
    ) -> Option<(Vec<u32>, Array1<f32>)> {
        let store = self.store.as_ref()?;
        let q = RoutingQuery { token_ids: query_ids.to_vec() };
        let routed = self.routing.resolve(&q, 3);
        let top_window = *routed.first()?;

        let entries = self.retrieve_entries(query_ids, &[top_window]).ok()?;
        let window_tokens = store.window_tokens.get(top_window as usize)?;

        // Context = window_tokens ++ query_tokens (drop leading BOS if present)
        let mut context: Vec<u32> = window_tokens.clone();
        let skip = if !query_ids.is_empty() && query_ids[0] == 2 { 1 } else { 0 }; // BOS=2 for Gemma
        context.extend_from_slice(&query_ids[skip..]);

        // Injection delta: sum of answer-side entry embeddings (not question-side echoes)
        let hidden = weights.hidden_size;
        let mut delta = vec![0.0f32; hidden];
        let qset: std::collections::HashSet<u32> = query_ids.iter().copied().collect();
        for e in &entries {
            if qset.contains(&e.token_id) { continue; }
            let emb = embed_tokens_pub(weights, &[e.token_id]);
            let scale = e.coefficient * self.config.inject_coefficient;
            for (i, v) in emb.row(0).iter().enumerate() {
                delta[i] += v * scale;
            }
        }

        Some((context, Array1::from(delta)))
    }

    /// One-shot query: route → retrieve → inject → forward. For diagnostics.
    pub fn query_greedy(
        &self,
        weights: &ModelWeights,
        query_ids: &[u32],
    ) -> Option<QueryTrace> {
        let (context, delta) = self.prepare_injection(weights, query_ids)?;
        let perturb = Some((self.config.injection_layer, delta.view()));
        let raw = forward_raw_logits(weights, &context, perturb);
        let (top1_id, top1_logit) = raw.logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, &v)| (i as u32, v))?;
        let q = RoutingQuery { token_ids: query_ids.to_vec() };
        let routed = self.routing.resolve(&q, 3);
        let entries = self.retrieve_entries(query_ids, &routed.get(..1).unwrap_or(&[])).unwrap_or_default();
        Some(QueryTrace {
            routed_windows: routed,
            injected_entries: entries,
            context_tokens: context.len(),
            top1_token_id: top1_id,
            top1_logit,
        })
    }
}

// ─── KvEngine impl ────────────────────────────────────────────────────────────

impl KvEngine for ApolloEngine {
    fn name(&self) -> &str { "apollo" }

    fn info(&self) -> EngineInfo {
        let windows = self.store.as_ref().map_or(0, |s| s.window_tokens.len());
        let entries = self.store.as_ref().map_or(0, |s| s.entries.len());
        let store_kb = self.store.as_ref().map_or(0, |s| s.total_bytes()) / 1024;
        EngineInfo {
            name: "apollo".into(),
            description: format!(
                "retrieval+injection: {windows} windows, {entries} entries, store={store_kb}KB",
            ),
            backend: "cpu".into(),
            config: format!("layer={}, coef={}, top_k={}",
                self.config.injection_layer,
                self.config.inject_coefficient,
                self.config.top_k,
            ),
        }
    }

    /// Prefill routes the token_ids, builds the injection delta and context,
    /// runs the initial forward pass with injection, and caches state for
    /// subsequent decode steps.
    fn prefill(&mut self, weights: &ModelWeights, token_ids: &[u32]) -> Option<Array2<f32>> {
        if self.routing.is_empty() {
            // Auto-build routing index if store is loaded but index is stale.
            let store = self.store.as_ref()?;
            self.routing = RoutingIndex::from_store(store);
        }

        let (context, delta) = self.prepare_injection(weights, token_ids)?;
        let perturb = Some((self.config.injection_layer, delta.view()));
        let raw = forward_raw_logits(weights, &context, perturb);

        // Cache state for decode steps.
        self.context_tokens = context;
        self.injection_delta = Some(delta);

        let last = raw.h_pre_norm.shape()[0] - 1;
        Some(raw.h_pre_norm.slice(s![last..=last, ..]).to_owned())
    }

    /// Extend context by one token and re-run the forward pass with the
    /// same injection delta. O(N) per step (full re-forward, no K/V cache).
    fn decode_step(&mut self, weights: &ModelWeights, token_id: u32) -> Option<Array2<f32>> {
        self.context_tokens.push(token_id);
        let delta = self.injection_delta.as_ref()?;
        let perturb = Some((self.config.injection_layer, delta.view()));
        let raw = forward_raw_logits(weights, &self.context_tokens, perturb);
        let last = raw.h_pre_norm.shape()[0] - 1;
        Some(raw.h_pre_norm.slice(s![last..=last, ..]).to_owned())
    }

    fn memory_bytes(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.total_bytes())
    }
}
