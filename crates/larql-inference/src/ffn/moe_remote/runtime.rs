//! Process-wide runtime toggles for the remote-MoE backend.
//!
//! Replaces four scattered `std::env::var(..)` reads (some via per-thread
//! `thread_local!` caches) with a single typed config read once via
//! `OnceLock`. Centralises the diagnostic surface so adding a new toggle
//! is a single edit instead of a hunt across `shard.rs` / `backend.rs`.
//!
//! Recognised vars (semantics unchanged from prior inline reads):
//!
//! - `LARQL_HTTP_TIMING=1` — print per-stage HTTP/UDS client-side
//!   timings on every shard call.
//! - `LARQL_MOE_WIRE_F16=1` — encode residual as f16 on the wire (LAN
//!   deployments where wire bytes matter more than f32↔f16 conversion CPU).
//!   Default off — loopback / same-host grids prefer f32.
//! - `LARQL_DISABLE_Q8K_WIRE=1` — disable Q8K residual upload (4×
//!   smaller). Default enabled; flag exists for debugging.
//! - `LARQL_VERBOSE=1` — emit dispatch-step timings for the multi-layer
//!   gRPC path (route / dispatch / accum breakdown per call).

use std::sync::OnceLock;

/// Snapshot of the four MoE runtime toggles, captured once per process.
#[derive(Clone, Debug, Default)]
pub struct RemoteMoeRuntime {
    /// Print per-stage HTTP/UDS client timings for each shard call.
    pub http_timing: bool,
    /// Use f16 residual on the wire (off by default — only wins on LAN).
    pub wire_f16: bool,
    /// Disable Q8K residual upload (debug flag — default enabled).
    pub q8k_disabled: bool,
    /// Emit gRPC predispatch / multi-layer timing breakdown.
    pub verbose: bool,
}

impl RemoteMoeRuntime {
    /// Read the four env vars and assemble the runtime config.
    pub fn from_env() -> Self {
        Self {
            http_timing: std::env::var("LARQL_HTTP_TIMING").is_ok(),
            wire_f16: std::env::var("LARQL_MOE_WIRE_F16").is_ok(),
            q8k_disabled: std::env::var("LARQL_DISABLE_Q8K_WIRE").is_ok(),
            verbose: std::env::var("LARQL_VERBOSE").is_ok(),
        }
    }

    /// Process-wide singleton. First caller pays the env-read cost; every
    /// subsequent caller returns a cheap `&'static`.
    pub fn get() -> &'static Self {
        static CFG: OnceLock<RemoteMoeRuntime> = OnceLock::new();
        CFG.get_or_init(Self::from_env)
    }

    /// Inverse of `q8k_disabled` — what most callers actually want to
    /// branch on. Q8K is on by default; the env var is an opt-out.
    pub fn q8k_enabled(&self) -> bool {
        !self.q8k_disabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_documented_off_state() {
        let cfg = RemoteMoeRuntime::default();
        assert!(!cfg.http_timing);
        assert!(!cfg.wire_f16);
        assert!(!cfg.q8k_disabled);
        assert!(cfg.q8k_enabled(), "q8k is enabled by default");
        assert!(!cfg.verbose);
    }

    #[test]
    fn q8k_enabled_inverts_q8k_disabled() {
        let cfg = RemoteMoeRuntime {
            q8k_disabled: true,
            ..Default::default()
        };
        assert!(!cfg.q8k_enabled());
    }

    #[test]
    fn singleton_is_stable() {
        let a = RemoteMoeRuntime::get() as *const _;
        let b = RemoteMoeRuntime::get() as *const _;
        assert_eq!(a, b);
    }
}
