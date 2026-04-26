//! End-of-sequence detection.
//!
//! Resolves stop tokens from `generation_config.json::eos_token_id` /
//! `stop_strings` plus a built-in list of family-specific terminators
//! (Gemma `<end_of_turn>`, ChatML `<|im_end|>`, Llama-3 `<|eot_id|>`).
//!
//! Centralises the check that previously lived in four places with subtly
//! different lists — `gpu.rs` had only `<eos>`, `</s>`, `<|endoftext|>`
//! (Gemma 4 ran to `--max-tokens` because `<end_of_turn>` was missing);
//! `vindex::is_end_of_turn` had a longer list; `forward::kv_generate` had
//! a third superset including Llama-3 markers.

use std::collections::HashSet;
use std::path::Path;

/// Token strings that always terminate generation across model families.
///
/// Built-in fallback when `generation_config.json` is missing or doesn't
/// list a family-specific marker. Gemma 4 in particular puts
/// `<end_of_turn>` only in `stop_strings`, not in `eos_token_id`.
pub const BUILTIN_STOP_STRINGS: &[&str] = &[
    "<eos>",
    "</s>",
    "<|endoftext|>",
    "<|im_end|>",
    "<|end_of_turn|>",
    "<end_of_turn>",
    "<|eot_id|>",
    "<|eom_id|>",
    "<|end_of_text|>",
];

/// Filename inside a vindex containing default sampling + stop config.
pub const GENERATION_CONFIG_FILENAME: &str = "generation_config.json";

/// JSON keys read from `generation_config.json`.
pub const KEY_EOS_TOKEN_ID: &str = "eos_token_id";
pub const KEY_STOP_STRINGS: &str = "stop_strings";

/// Configuration for EOS detection.
#[derive(Debug, Clone, Default)]
pub struct EosConfig {
    pub eos_token_ids: HashSet<u32>,
    pub stop_strings: Vec<String>,
}

impl EosConfig {
    /// Empty config (greedy decode never stops on its own).
    pub fn empty() -> Self {
        Self::default()
    }

    /// Built-in stop strings, no EOS IDs. Use as a baseline before merging
    /// in `generation_config.json` overrides.
    pub fn builtin() -> Self {
        Self {
            eos_token_ids: HashSet::new(),
            stop_strings: BUILTIN_STOP_STRINGS.iter().map(|s| s.to_string()).collect(),
        }
    }

    pub fn with_eos_id(mut self, id: u32) -> Self {
        self.eos_token_ids.insert(id);
        self
    }

    pub fn with_stop_string(mut self, s: impl Into<String>) -> Self {
        let s = s.into();
        if !self.stop_strings.iter().any(|existing| existing == &s) {
            self.stop_strings.push(s);
        }
        self
    }

    /// Build from a parsed `generation_config.json` value, layered on top
    /// of [`Self::builtin`]. Both `eos_token_id: 1` and `eos_token_id: [1, 2]`
    /// shapes are handled.
    pub fn from_generation_config(json: &serde_json::Value) -> Self {
        let mut cfg = Self::builtin();
        match json.get(KEY_EOS_TOKEN_ID) {
            Some(serde_json::Value::Number(n)) => {
                if let Some(id) = n.as_u64() {
                    cfg.eos_token_ids.insert(id as u32);
                }
            }
            Some(serde_json::Value::Array(arr)) => {
                for v in arr {
                    if let Some(id) = v.as_u64() {
                        cfg.eos_token_ids.insert(id as u32);
                    }
                }
            }
            _ => {}
        }
        if let Some(stops) = json.get(KEY_STOP_STRINGS).and_then(|v| v.as_array()) {
            for s in stops {
                if let Some(s) = s.as_str() {
                    cfg = cfg.with_stop_string(s);
                }
            }
        }
        cfg
    }

    /// Convenience: read `<vindex_dir>/generation_config.json` and apply
    /// it. Missing file falls back to [`Self::builtin`].
    pub fn from_vindex_dir(vindex_dir: &Path) -> Self {
        let path = vindex_dir.join(GENERATION_CONFIG_FILENAME);
        if !path.is_file() {
            return Self::builtin();
        }
        match std::fs::read(&path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        {
            Some(v) => Self::from_generation_config(&v),
            None => Self::builtin(),
        }
    }

    /// Halt generation when this token id or its decoded surface form
    /// matches any configured stop. Surface-form match is whitespace
    /// trimmed since the tokenizer often emits leading-space variants.
    pub fn is_eos(&self, id: u32, decoded: &str) -> bool {
        if self.eos_token_ids.contains(&id) {
            return true;
        }
        let trimmed = decoded.trim();
        if trimmed.is_empty() {
            return false;
        }
        self.stop_strings.iter().any(|s| s == trimmed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_recognises_gemma_end_of_turn() {
        let cfg = EosConfig::builtin();
        assert!(cfg.is_eos(0, "<end_of_turn>"));
        assert!(cfg.is_eos(0, "<|end_of_turn|>"));
    }

    #[test]
    fn builtin_recognises_chatml_and_llama() {
        let cfg = EosConfig::builtin();
        assert!(cfg.is_eos(0, "<|im_end|>"));
        assert!(cfg.is_eos(0, "<|eot_id|>"));
        assert!(cfg.is_eos(0, "<|eom_id|>"));
    }

    #[test]
    fn empty_never_stops() {
        let cfg = EosConfig::empty();
        assert!(!cfg.is_eos(1, "<eos>"));
        assert!(!cfg.is_eos(0, ""));
    }

    #[test]
    fn surface_form_trimmed() {
        let cfg = EosConfig::builtin();
        assert!(cfg.is_eos(0, "  <end_of_turn>  "));
        assert!(cfg.is_eos(0, "\n<eos>\n"));
    }

    #[test]
    fn empty_decoded_does_not_match() {
        // A purely-whitespace decode shouldn't trigger every stop string.
        let cfg = EosConfig::builtin();
        assert!(!cfg.is_eos(0, ""));
        assert!(!cfg.is_eos(0, "   "));
    }

    #[test]
    fn eos_id_match_independent_of_string() {
        let cfg = EosConfig::empty().with_eos_id(2);
        assert!(cfg.is_eos(2, "anything"));
        assert!(!cfg.is_eos(3, "anything"));
    }

    #[test]
    fn from_generation_config_scalar_eos_id() {
        let json: serde_json::Value = serde_json::from_str(r#"{"eos_token_id": 7}"#).unwrap();
        let cfg = EosConfig::from_generation_config(&json);
        assert!(cfg.is_eos(7, "noise"));
        assert!(!cfg.is_eos(8, "noise"));
    }

    #[test]
    fn from_generation_config_array_eos_id() {
        let json: serde_json::Value =
            serde_json::from_str(r#"{"eos_token_id": [1, 107, 106]}"#).unwrap();
        let cfg = EosConfig::from_generation_config(&json);
        for id in [1u32, 107, 106] {
            assert!(cfg.is_eos(id, ""), "{id} should be EOS");
        }
    }

    #[test]
    fn from_generation_config_stop_strings_merged() {
        // Gemma 4 actually ships this combination — `<end_of_turn>` only via stop_strings.
        let json: serde_json::Value =
            serde_json::from_str(r#"{"eos_token_id": 1, "stop_strings": ["<end_of_turn>"]}"#)
                .unwrap();
        let cfg = EosConfig::from_generation_config(&json);
        assert!(cfg.is_eos(1, ""));
        assert!(cfg.is_eos(0, "<end_of_turn>"));
    }

    #[test]
    fn duplicate_stop_string_not_added_twice() {
        // `<end_of_turn>` is in BUILTIN_STOP_STRINGS already.
        let cfg = EosConfig::builtin().with_stop_string("<end_of_turn>");
        let count = cfg
            .stop_strings
            .iter()
            .filter(|s| s.as_str() == "<end_of_turn>")
            .count();
        assert_eq!(count, 1);
    }

    #[test]
    fn from_vindex_dir_missing_file_falls_back_to_builtin() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = EosConfig::from_vindex_dir(tmp.path());
        assert!(cfg.is_eos(0, "<eos>"));
    }

    #[test]
    fn from_vindex_dir_reads_file() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join(GENERATION_CONFIG_FILENAME),
            r#"{"eos_token_id": [42]}"#,
        )
        .unwrap();
        let cfg = EosConfig::from_vindex_dir(tmp.path());
        assert!(cfg.is_eos(42, ""));
        // builtin still applies
        assert!(cfg.is_eos(0, "<eos>"));
    }
}
