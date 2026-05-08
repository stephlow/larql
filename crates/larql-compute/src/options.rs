//! Runtime options and environment-variable names for compute backends.
//!
//! Keep process-global debug and experiment toggles here instead of spelling
//! string literals through hot paths. Most callers should eventually pass an
//! explicit options struct; this module is the compatibility bridge while those
//! APIs are split out.

/// Enable timing around the full CPU MoE forward pass.
pub const ENV_MOE_FWD_TIMING: &str = "LARQL_MOE_FWD_TIMING";
/// Enable timing around one CPU MoE expert.
pub const ENV_MOE_EXPERT_TIMING: &str = "LARQL_MOE_EXPERT_TIMING";
/// Enable timing inside the direct Q4_K expert kernel.
pub const ENV_KERNEL_TIMING: &str = "LARQL_KERNEL_TIMING";
/// Disable the direct Q4_K/Q8_K CPU MoE path.
pub const ENV_DISABLE_Q4K_DIRECT: &str = "LARQL_DISABLE_Q4K_DIRECT";
/// Opt in to the older scalar Q4_K direct path in `run_single_expert_into`.
pub const ENV_Q4K_DIRECT: &str = "LARQL_Q4K_DIRECT";
/// Max entries in the dequantised MoE expert cache.
pub const ENV_MOE_CACHE_ENTRIES: &str = "LARQL_MOE_CACHE_ENTRIES";
/// Namespaced MoE bypass toggle.
pub const ENV_SKIP_MOE: &str = "LARQL_SKIP_MOE";
/// Legacy MoE bypass toggle. Prefer [`ENV_SKIP_MOE`] in new scripts.
pub const ENV_SKIP_MOE_LEGACY: &str = "SKIP_MOE";
/// Namespaced MoE route/debug output toggle.
pub const ENV_MOE_DEBUG: &str = "LARQL_MOE_DEBUG";
/// Legacy MoE route/debug output toggle. Prefer [`ENV_MOE_DEBUG`].
pub const ENV_MOE_DEBUG_LEGACY: &str = "MOE_DEBUG";

pub(crate) fn env_flag(name: &str) -> bool {
    std::env::var_os(name).is_some()
}

pub(crate) fn env_flag_any(names: &[&str]) -> bool {
    names.iter().any(|name| env_flag(name))
}

pub(crate) fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.parse().ok()
}

pub(crate) fn moe_debug_enabled() -> bool {
    env_flag_any(&[ENV_MOE_DEBUG, ENV_MOE_DEBUG_LEGACY])
}

pub(crate) fn skip_moe_enabled() -> bool {
    env_flag_any(&[ENV_SKIP_MOE, ENV_SKIP_MOE_LEGACY])
}
