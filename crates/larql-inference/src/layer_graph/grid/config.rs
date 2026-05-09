use crate::layer_graph::generate::policy::TokenSelectionPolicy;

#[derive(Clone, Debug)]
pub(super) struct GridRuntimeConfig {
    pub moe_top_k_override: Option<usize>,
    pub skip_moe: bool,
    pub timing_enabled: bool,
    pub split_disabled: bool,
    pub token_policy: TokenSelectionPolicy,
}

impl GridRuntimeConfig {
    pub fn from_env() -> Self {
        Self {
            moe_top_k_override: std::env::var("LARQL_MOE_TOP_K")
                .ok()
                .and_then(|s| s.parse::<usize>().ok()),
            skip_moe: std::env::var("SKIP_MOE").is_ok(),
            timing_enabled: std::env::var("LARQL_MOE_TIMING").is_ok(),
            split_disabled: std::env::var("LARQL_MOE_NO_SPLIT").is_ok(),
            token_policy: TokenSelectionPolicy::from_env(),
        }
    }

    pub fn moe_top_k(&self, arch_top_k: usize) -> usize {
        self.moe_top_k_override
            .map(|k| k.clamp(1, arch_top_k))
            .unwrap_or(arch_top_k)
    }
}
