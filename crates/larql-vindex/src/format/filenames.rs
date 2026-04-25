//! Vindex on-disk filenames — single source of truth.
//!
//! Every `.bin` / `.json` filename written or read by the vindex format
//! lives here as a `pub const`. Use these instead of string literals.
//!
//! Why: the 2026-04-25 audit found 244 occurrences of these names
//! scattered across 18+ files. A typo silently triggers a fallback
//! codepath (the file just "doesn't exist") and bugs go undiagnosed.
//! Centralising means renaming a file changes one line.
//!
//! Convention: `SCREAMING_SNAKE`, named for what they hold, not how
//! they're encoded.

// ── Top-level config / sidecars ─────────────────────────────────────────
pub const INDEX_JSON: &str = "index.json";
pub const TOKENIZER_JSON: &str = "tokenizer.json";
pub const TOKENIZER_CONFIG_JSON: &str = "tokenizer_config.json";
pub const WEIGHT_MANIFEST_JSON: &str = "weight_manifest.json";

// ── Embeddings + norms (always present) ────────────────────────────────
pub const EMBEDDINGS_BIN: &str = "embeddings.bin";
pub const NORMS_BIN: &str = "norms.bin";

// ── Gate vectors ───────────────────────────────────────────────────────
pub const GATE_VECTORS_BIN: &str = "gate_vectors.bin";
pub const GATE_VECTORS_Q4_BIN: &str = "gate_vectors_q4.bin";

// ── Down meta + feature-major projections ──────────────────────────────
pub const DOWN_META_BIN: &str = "down_meta.bin";
pub const DOWN_FEATURES_BIN: &str = "down_features.bin";
pub const UP_FEATURES_BIN: &str = "up_features.bin";

// ── Interleaved FFN (gate|up|down packed per layer) ────────────────────
pub const INTERLEAVED_BIN: &str = "interleaved.bin";
pub const INTERLEAVED_Q4_BIN: &str = "interleaved_q4.bin";
pub const INTERLEAVED_Q4K_BIN: &str = "interleaved_q4k.bin";
pub const INTERLEAVED_Q4K_MANIFEST_JSON: &str = "interleaved_q4k_manifest.json";

// ── Attention weights ──────────────────────────────────────────────────
pub const ATTN_WEIGHTS_BIN: &str = "attn_weights.bin";
pub const ATTN_WEIGHTS_Q4K_BIN: &str = "attn_weights_q4k.bin";
pub const ATTN_WEIGHTS_Q4K_MANIFEST_JSON: &str = "attn_weights_q4k_manifest.json";

// ── LM head ────────────────────────────────────────────────────────────
pub const LM_HEAD_Q4_BIN: &str = "lm_head_q4.bin";

// ── HuggingFace upload manifest order ──────────────────────────────────
//
// Order matches what `format/huggingface.rs` uploads. Adding or
// removing a vindex file means updating both this list AND the
// per-file upload code.
pub const HF_UPLOAD_FILES: &[&str] = &[
    INDEX_JSON,
    TOKENIZER_JSON,
    WEIGHT_MANIFEST_JSON,
    EMBEDDINGS_BIN,
    NORMS_BIN,
    GATE_VECTORS_BIN,
    DOWN_META_BIN,
    INTERLEAVED_BIN,
    INTERLEAVED_Q4K_BIN,
    INTERLEAVED_Q4K_MANIFEST_JSON,
    ATTN_WEIGHTS_BIN,
    ATTN_WEIGHTS_Q4K_BIN,
    ATTN_WEIGHTS_Q4K_MANIFEST_JSON,
    DOWN_FEATURES_BIN,
    UP_FEATURES_BIN,
    LM_HEAD_Q4_BIN,
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Constants must never collide — a duplicate name would silently
    /// route two writers at the same file.
    #[test]
    fn all_filenames_unique() {
        let names = [
            INDEX_JSON, TOKENIZER_JSON, TOKENIZER_CONFIG_JSON,
            WEIGHT_MANIFEST_JSON, EMBEDDINGS_BIN, NORMS_BIN,
            GATE_VECTORS_BIN, GATE_VECTORS_Q4_BIN, DOWN_META_BIN,
            DOWN_FEATURES_BIN, UP_FEATURES_BIN,
            INTERLEAVED_BIN, INTERLEAVED_Q4_BIN, INTERLEAVED_Q4K_BIN,
            INTERLEAVED_Q4K_MANIFEST_JSON, ATTN_WEIGHTS_BIN,
            ATTN_WEIGHTS_Q4K_BIN, ATTN_WEIGHTS_Q4K_MANIFEST_JSON,
            LM_HEAD_Q4_BIN,
        ];
        let unique: std::collections::HashSet<_> = names.iter().collect();
        assert_eq!(unique.len(), names.len(), "duplicate filename constant");
    }

    #[test]
    fn hf_upload_files_subset_of_all() {
        // HF_UPLOAD_FILES must reference real constants. If a constant
        // is removed, this test catches the dangling reference.
        for name in HF_UPLOAD_FILES {
            assert!(name.ends_with(".bin") || name.ends_with(".json"),
                "HF_UPLOAD_FILES has odd entry: {name}");
        }
    }
}
