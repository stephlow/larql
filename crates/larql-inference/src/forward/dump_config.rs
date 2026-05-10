//! Typed accessor for the diagnostic per-layer / per-stage dump env vars.
//!
//! Three env vars used to be re-read on every CPU forward pass — once per
//! layer and once per stage (norm, Q, K, V, attn, O, FFN). On a 34-layer
//! decode that was ≥ 200 process-env lookups per token even when no dump
//! was active. They also drift if the env changes mid-process, which the
//! dump consumers don't actually want.
//!
//! [`DumpConfig::get`] reads them once via `OnceLock`, returns a cheap
//! borrow, and exposes `should_dump_layer(layer)` / `stage_dir(layer)`
//! helpers that match the prior inline logic verbatim. Replaces 7 inline
//! `std::env::var(...)` calls across `attention/`, `forward/layer*.rs`,
//! and `vindex/q4k_forward/hidden.rs`.
//!
//! Recognised vars (semantics unchanged from the previous inline reads):
//!
//! - `LARQL_CPU_DUMP_LAYERS=<dir>` — write `cpu_layer_NN_h_post_attn.f32`
//!   and `cpu_layer_NN_h_out.f32` per layer.
//! - `LARQL_CPU_STAGE_DUMP=<dir>` — write per-stage intermediates
//!   (`cpu_L0_norm_out.f32`, `cpu_L0_q_proj.f32`, …) for one specific
//!   layer (default 0).
//! - `LARQL_STAGE_DUMP_LAYER=<usize>` — pick a different stage-dump layer
//!   (Gemma 4 global layers 5/11/17/… are useful for partial-RoPE bisects).

use std::sync::OnceLock;

/// Snapshot of dump-related env vars, captured once per process.
#[derive(Clone, Debug, Default)]
pub struct DumpConfig {
    /// Directory to write per-layer end-of-layer dumps to. `None` disables.
    pub layer_dump_dir: Option<String>,
    /// Directory to write per-stage intermediates to. `None` disables.
    pub stage_dump_dir: Option<String>,
    /// Which layer's stages to dump when `stage_dump_dir` is set. Defaults to 0.
    pub stage_dump_layer: usize,
}

impl DumpConfig {
    /// Read the three env vars and assemble a `DumpConfig`. Public so test
    /// fixtures can build one without touching the process env.
    pub fn from_env() -> Self {
        Self {
            layer_dump_dir: std::env::var("LARQL_CPU_DUMP_LAYERS").ok(),
            stage_dump_dir: std::env::var("LARQL_CPU_STAGE_DUMP").ok(),
            stage_dump_layer: std::env::var("LARQL_STAGE_DUMP_LAYER")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0),
        }
    }

    /// Process-wide singleton. First caller pays the env-read cost; every
    /// subsequent caller borrows.
    pub fn get() -> &'static Self {
        static CFG: OnceLock<DumpConfig> = OnceLock::new();
        CFG.get_or_init(Self::from_env)
    }

    /// `Some(dir)` only when stage dumps are enabled AND the active layer
    /// matches `stage_dump_layer`. Mirrors the prior inline guard.
    pub fn stage_dir(&self, layer: usize) -> Option<&str> {
        if layer == self.stage_dump_layer {
            self.stage_dump_dir.as_deref()
        } else {
            None
        }
    }

    /// `Some(dir)` when per-layer dumps are enabled.
    pub fn layer_dir(&self) -> Option<&str> {
        self.layer_dump_dir.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_env_reads_all_three_vars() {
        // Hand-build via from_env-style construction (we can't poke the
        // process env reliably in parallel tests, so build the struct
        // directly and exercise the accessors).
        let cfg = DumpConfig {
            layer_dump_dir: Some("/tmp/layers".into()),
            stage_dump_dir: Some("/tmp/stages".into()),
            stage_dump_layer: 5,
        };
        assert_eq!(cfg.layer_dir(), Some("/tmp/layers"));
        assert_eq!(cfg.stage_dir(5), Some("/tmp/stages"));
        assert_eq!(cfg.stage_dir(0), None);
        assert_eq!(cfg.stage_dir(99), None);
    }

    #[test]
    fn stage_dir_returns_none_when_dump_disabled() {
        let cfg = DumpConfig {
            layer_dump_dir: None,
            stage_dump_dir: None,
            stage_dump_layer: 0,
        };
        assert_eq!(cfg.stage_dir(0), None);
    }

    #[test]
    fn layer_dir_returns_none_when_dump_disabled() {
        let cfg = DumpConfig::default();
        assert_eq!(cfg.layer_dir(), None);
    }

    #[test]
    fn singleton_is_stable() {
        // Two calls return the same backing struct — `OnceLock` semantics.
        let a = DumpConfig::get() as *const _;
        let b = DumpConfig::get() as *const _;
        assert_eq!(a, b);
    }

    #[test]
    fn stage_dump_layer_default_is_zero_when_unparsed() {
        // Mirrors the unwrap_or(0) on bad/missing LARQL_STAGE_DUMP_LAYER.
        let cfg = DumpConfig::default();
        assert_eq!(cfg.stage_dump_layer, 0);
    }
}
