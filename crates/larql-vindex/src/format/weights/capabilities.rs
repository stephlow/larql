use crate::error::VindexError;

pub(super) const SURFACE_F32_WEIGHT_WRITER: &str = "f32 weight writer";
pub(super) const SURFACE_Q4K_WEIGHT_WRITER: &str = "q4k weight writer";

const FEATURE_MLA: &str = "multi-head latent attention (MLA)";

/// Ensure the current vindex weight layout can represent this architecture's
/// attention tensors.
///
/// The existing f32 and Q4K manifests store standard decoder attention as
/// Q/K/V/O tensors. Architectures such as DeepSeek MLA expose a different
/// tensor contract (`mla_*`) and must be implemented explicitly before the
/// writer accepts them.
pub(super) fn ensure_standard_attention_supported(
    arch: &dyn larql_models::ModelArchitecture,
    surface: &'static str,
) -> Result<(), VindexError> {
    if arch.uses_mla() {
        return Err(VindexError::UnsupportedArchitecture {
            family: arch.family().to_string(),
            feature: FEATURE_MLA.into(),
            surface: surface.into(),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SURFACE: &str = "test";
    const TEST_Q4K_SURFACE: &str = SURFACE_Q4K_WEIGHT_WRITER;
    const MODEL_TYPE_LLAMA: &str = "llama";
    const MODEL_TYPE_DEEPSEEK_V2: &str = "deepseek_v2";
    const HIDDEN_SIZE_LLAMA_7B: usize = 4096;
    const HIDDEN_SIZE_TEST: usize = 4096;
    const INTERMEDIATE_SIZE_TEST: usize = 12288;
    const NUM_LAYERS_LLAMA_7B: usize = 32;
    const NUM_LAYERS_TEST: usize = 4;
    const NUM_ATTENTION_HEADS_LLAMA_7B: usize = 32;
    const NUM_ATTENTION_HEADS_TEST: usize = 32;
    const NUM_KV_HEADS_TEST: usize = 32;
    const HEAD_DIM_TEST: usize = 128;
    const KV_LORA_RANK_TEST: usize = 512;
    const Q_LORA_RANK_TEST: usize = 1536;

    #[test]
    fn standard_attention_accepts_llama() {
        let arch = larql_models::detect_from_json(&serde_json::json!({
            "model_type": MODEL_TYPE_LLAMA,
            "hidden_size": HIDDEN_SIZE_LLAMA_7B,
            "num_hidden_layers": NUM_LAYERS_LLAMA_7B,
            "num_attention_heads": NUM_ATTENTION_HEADS_LLAMA_7B
        }));

        assert!(ensure_standard_attention_supported(&*arch, TEST_SURFACE).is_ok());
    }

    #[test]
    fn mla_architecture_is_rejected() {
        let arch = larql_models::detect_from_json(&serde_json::json!({
            "model_type": MODEL_TYPE_DEEPSEEK_V2,
            "hidden_size": HIDDEN_SIZE_TEST,
            "intermediate_size": INTERMEDIATE_SIZE_TEST,
            "num_hidden_layers": NUM_LAYERS_TEST,
            "num_attention_heads": NUM_ATTENTION_HEADS_TEST,
            "num_key_value_heads": NUM_KV_HEADS_TEST,
            "head_dim": HEAD_DIM_TEST,
            "kv_lora_rank": KV_LORA_RANK_TEST,
            "q_lora_rank": Q_LORA_RANK_TEST
        }));

        let err = ensure_standard_attention_supported(&*arch, TEST_Q4K_SURFACE)
            .expect_err("MLA must not be accepted by standard Q/K/V/O writers");
        let msg = err.to_string();
        assert!(msg.contains(arch.family()), "{msg}");
        assert!(msg.contains(FEATURE_MLA), "{msg}");
        assert!(msg.contains(TEST_Q4K_SURFACE), "{msg}");
    }
}
