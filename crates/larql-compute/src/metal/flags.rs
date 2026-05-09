//! Decode hot-path env flags, captured once at backend construction.
//!
//! Decode reads these per-layer per-token; resolving them via `getenv`
//! would cost ~12 syscalls × 34 layers = ~400/token on Gemma 3 4B
//! (worse on Gemma 4 26B-A4B's 60-layer split path). Settling to a
//! single load at `MetalBackend::new()` time keeps the hot path free
//! of syscalls.
//!
//! ## Contract for callers
//!
//! Once a backend is constructed, its flags are fixed. To reflect an
//! env-var flip, build a fresh backend (`MetalBackend::new()`). Tests
//! that toggle these vars must do this — see the env-mutating tests
//! in `tests/test_metal_decode_synthetic.rs` and the `ENV_TEST_LOCK`
//! that serialises them.
//!
//! ## What lives here vs what doesn't
//!
//! Cached: flags consulted from `decode_token_with_moe_split_fn` and
//! its `encode_*` callees. Off the hot path (debug dumps, timing
//! diagnostics, prefill-only paths) intentionally read env per-call so
//! they can be flipped without restarting — see for example
//! `LARQL_DEBUG_NAN_LAYERS`, `LARQL_GPU_TIMING`, `LARQL_DECODE_DUMP_LAYERS`.

use crate::options::{
    env_flag, env_not_zero_or_default, env_opt_in, env_opt_out, ENV_F16_ACC, ENV_FUSED_ATTN,
    ENV_FUSED_DOWN, ENV_FUSED_KV_APPEND_ATTEND, ENV_FUSED_POST_ATTN_NORM, ENV_FUSED_POST_FFN_NORM,
    ENV_FUSED_PRELAYER_NORM, ENV_FUSED_Q6K_DOWN, ENV_FUSED_QK_NORM_ROPE, ENV_GATE_UP_8SG,
    ENV_GATE_UP_COOP, ENV_QKV_FUSED,
};

/// Decode-path flag snapshot captured at backend startup.
#[derive(Copy, Clone, Debug)]
pub struct DecodeFlags {
    /// `LARQL_GATE_UP_COOP=1` — cooperative-scale-load Q4_K gate+up
    /// variant. Opt-in (default false).
    pub gate_up_coop: bool,
    /// `LARQL_GATE_UP_8SG=0` opts OUT of the default 8sg Q4_K gate+up
    /// back to the older 4sg kernel. Stored as `use_4sg` so the
    /// reading site doesn't have to re-invert.
    pub gate_up_use_4sg: bool,
    /// `LARQL_F16_ACC=1` — f16 accumulator on the legacy 4sg gate+up.
    /// Opt-in (default false).
    pub f16_acc: bool,
    /// `LARQL_FUSED_Q6K_DOWN=1` — opt into the cached-activation Q6_K
    /// fused down. Currently no-op pending kernel parity (see
    /// `encode_ffn.rs` block doc).
    pub fused_q6k_down: bool,
    /// `LARQL_FUSED_DOWN` — fused Q4_K GEGLU+down on the decode hot
    /// path. Default true; opt out with `=0`. (The prefill path in
    /// `stages/ffn.rs` reads the same env var with default OFF — that
    /// asymmetry is intentional and not cached here; prefill is
    /// allocation-bound, the per-call env read is in the noise.)
    pub fused_down: bool,
    /// `LARQL_FUSED_PRELAYER_NORM=1` — D-RMS-FUSE Phase 1: fold the
    /// post-FFN residual_add into the next-layer input rms_norm. Opt-in.
    pub fused_prelayer_norm: bool,
    /// `LARQL_QKV_FUSED=1` — opt back IN to the (now-defaulted-off)
    /// `q4k_q6k_qkv_proj_normed` shader. Opt-in (ADR-016 flipped the
    /// default to defused on 2026-05-09).
    pub qkv_fused: bool,
    /// `LARQL_FUSED_ATTN=1` — opt into the triple-fused QK-norm + RoPE
    /// + KV-append + attend kernel. Opt-in.
    pub fused_attn: bool,
    /// `LARQL_FUSED_QK_NORM_ROPE` — default true; opt out with `=0`.
    pub fused_qk_norm_rope: bool,
    /// `LARQL_FUSED_KV_APPEND_ATTEND` — default true; opt out with `=0`.
    pub fused_kv_append_attend: bool,
    /// `LARQL_FUSED_POST_ATTN_NORM` — default true; opt out with `=0`.
    pub fused_post_attn_norm: bool,
    /// `LARQL_FUSED_POST_FFN_NORM` — default true; opt out with `=0`.
    pub fused_post_ffn_norm: bool,
}

impl DecodeFlags {
    /// Snapshot the env. Call once per `MetalBackend::new()`.
    pub fn from_env() -> Self {
        Self {
            gate_up_coop: env_opt_in(ENV_GATE_UP_COOP),
            gate_up_use_4sg: env_opt_out(ENV_GATE_UP_8SG),
            f16_acc: env_flag(ENV_F16_ACC),
            fused_q6k_down: env_flag(ENV_FUSED_Q6K_DOWN),
            fused_down: env_not_zero_or_default(ENV_FUSED_DOWN, true),
            fused_prelayer_norm: env_opt_in(ENV_FUSED_PRELAYER_NORM),
            qkv_fused: env_opt_in(ENV_QKV_FUSED),
            fused_attn: env_opt_in(ENV_FUSED_ATTN),
            fused_qk_norm_rope: !env_opt_out(ENV_FUSED_QK_NORM_ROPE),
            fused_kv_append_attend: !env_opt_out(ENV_FUSED_KV_APPEND_ATTEND),
            fused_post_attn_norm: !env_opt_out(ENV_FUSED_POST_ATTN_NORM),
            fused_post_ffn_norm: !env_opt_out(ENV_FUSED_POST_FFN_NORM),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pin the defaults: every flag must resolve to a known value with
    /// no env vars set. Captures the contract that
    /// `MetalBackend::new()` on a clean process is deterministic.
    #[test]
    fn from_env_defaults_when_no_vars_set() {
        // ENV vars may leak in from the running process. To make the
        // test robust we explicitly clear ours, snapshot, and restore.
        // Held under a static lock so this doesn't race other tests in
        // the file.
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _g = LOCK.lock().expect("flag-test lock poisoned");

        let names = [
            ENV_GATE_UP_COOP,
            ENV_GATE_UP_8SG,
            ENV_F16_ACC,
            ENV_FUSED_Q6K_DOWN,
            ENV_FUSED_DOWN,
            ENV_FUSED_PRELAYER_NORM,
            ENV_QKV_FUSED,
            ENV_FUSED_ATTN,
            ENV_FUSED_QK_NORM_ROPE,
            ENV_FUSED_KV_APPEND_ATTEND,
            ENV_FUSED_POST_ATTN_NORM,
            ENV_FUSED_POST_FFN_NORM,
        ];
        let prior: Vec<_> = names
            .iter()
            .map(|n| (*n, std::env::var_os(n)))
            .collect();
        for n in names {
            std::env::remove_var(n);
        }

        let flags = DecodeFlags::from_env();

        // Opt-in flags (default false).
        assert!(!flags.gate_up_coop);
        assert!(!flags.gate_up_use_4sg); // opt-OUT of 8sg, so default false → use 8sg
        assert!(!flags.f16_acc);
        assert!(!flags.fused_q6k_down);
        assert!(!flags.fused_prelayer_norm);
        assert!(!flags.qkv_fused);
        assert!(!flags.fused_attn);

        // Default-on flags.
        assert!(flags.fused_down);
        assert!(flags.fused_qk_norm_rope);
        assert!(flags.fused_kv_append_attend);
        assert!(flags.fused_post_attn_norm);
        assert!(flags.fused_post_ffn_norm);

        for (n, v) in prior {
            match v {
                Some(val) => std::env::set_var(n, val),
                None => std::env::remove_var(n),
            }
        }
    }
}
