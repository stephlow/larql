//! Serialization types for the .vindex format.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Metadata stored in index.json inside a .vindex directory.
///
/// All fields implement `Default`. Prefer
/// `VindexConfig { version: 2, model: "...".into(), ..Default::default() }`
/// over listing every field explicitly — optional additions (like `fp4`)
/// don't then propagate to every construction site.
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct VindexConfig {
    /// Format version.
    pub version: u32,
    /// Original model name (e.g., "google/gemma-3-4b-it").
    pub model: String,
    /// Model family (e.g., "gemma3", "llama").
    pub family: String,
    /// Provenance: which model checkpoint this vindex was built from.
    #[serde(default)]
    pub source: Option<VindexSource>,
    /// SHA256 checksums of each binary file for integrity verification.
    #[serde(default)]
    pub checksums: Option<HashMap<String, String>>,
    /// Number of layers.
    pub num_layers: usize,
    /// Hidden dimension.
    pub hidden_size: usize,
    /// Intermediate (FFN) size.
    pub intermediate_size: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Embedding scale factor.
    pub embed_scale: f32,
    /// What level of weights are included.
    #[serde(default)]
    pub extract_level: ExtractLevel,
    /// Storage precision (f32 or f16).
    #[serde(default)]
    pub dtype: crate::config::dtype::StorageDtype,
    /// Quantisation format of the model weights written alongside this
    /// vindex. `None` means float storage controlled by `dtype`;
    /// `Q4k` means Q4_K/Q6_K blocks in `attn_weights_q4k.bin` +
    /// `interleaved_q4k.bin`. Loaders dispatch on this field so they
    /// don't have to sniff filenames.
    #[serde(default)]
    pub quant: QuantFormat,
    /// Model-specific layer band boundaries for DESCRIBE and label matching.
    #[serde(default)]
    pub layer_bands: Option<LayerBands>,
    /// Per-layer info for gate_vectors.bin layout.
    pub layers: Vec<VindexLayerInfo>,
    /// Top-K tokens stored per feature in down metadata.
    pub down_top_k: usize,
    /// Whether model_weights.bin is present (legacy, use extract_level).
    #[serde(default)]
    pub has_model_weights: bool,
    /// Model config for architecture reconstruction.
    #[serde(default)]
    pub model_config: Option<VindexModelConfig>,
    /// Optional FP4/FP8 block-storage manifest. Set when one or more FFN
    /// projections are stored in the block-quantised format described
    /// in `docs/specs/vindex-format-spec.md` §5.10 and
    /// `docs/specs/fp4-format-spec.md`.
    /// Absent or null → legacy f16/f32 projection files are
    /// authoritative and loaders use the legacy codepath.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fp4: Option<Fp4Config>,
}

/// Provenance: which model checkpoint this vindex was built from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VindexSource {
    #[serde(default)]
    pub huggingface_repo: Option<String>,
    #[serde(default)]
    pub huggingface_revision: Option<String>,
    #[serde(default)]
    pub safetensors_sha256: Option<String>,
    /// ISO 8601 timestamp of extraction.
    pub extracted_at: String,
    /// Version of larql used for extraction.
    pub larql_version: String,
}

/// What components are included in the vindex. Strictly increasing —
/// each tier is a superset of the previous.
///
/// | Tier        | Adds                                   | Enables                                |
/// |-------------|----------------------------------------|----------------------------------------|
/// | `browse`    | gate, embed, down_meta, tokenizer      | WALK / DESCRIBE / SELECT               |
/// | `attention` | + attention + norms                    | client-side of `run --ffn URL` (Act 2) |
/// | `inference` | + FFN up/down                          | full local forward pass (INFER)        |
/// | `all`       | + lm_head + any COMPILE extras         | COMPILE                                |
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum ExtractLevel {
    /// Gate + embed + down_meta + tokenizer. Enables WALK, DESCRIBE,
    /// SELECT. No forward pass possible.
    #[default]
    Browse,
    /// + attention + norms. Enables the client-side half of
    /// `larql run --ffn URL` (Act 2 of the Gemma 4 MoE demo). Cannot
    /// run a forward pass alone — FFN must live somewhere else.
    Attention,
    /// + FFN up/down weights. Enables full local INFER.
    Inference,
    /// + lm_head (when not tied to embed) + anything else future
    /// COMPILE passes need. Enables COMPILE.
    All,
}

impl ExtractLevel {
    /// Whether this tier includes attention weights + norms.
    /// True for Attention, Inference, All.
    pub fn writes_attn(self) -> bool {
        self >= Self::Attention
    }

    /// Whether this tier includes FFN up/down weight files (the full
    /// compute weights, not just the gate used by KNN).
    /// True for Inference, All.
    pub fn writes_ffn(self) -> bool {
        self >= Self::Inference
    }

    /// Whether this tier writes lm_head. When the model ties
    /// embeddings (embed_tokens shares weights with lm_head), the
    /// writer may still skip it — this is the intent flag.
    /// True for Inference, All.
    pub fn writes_lm_head(self) -> bool {
        self >= Self::Inference
    }
}

impl std::fmt::Display for ExtractLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Browse => write!(f, "browse"),
            Self::Attention => write!(f, "attention"),
            Self::Inference => write!(f, "inference"),
            Self::All => write!(f, "all"),
        }
    }
}

/// Quantization format for the model weights written to a vindex.
///
/// `None` = float weights (dtype controlled separately by `StorageDtype`).
/// `Q4K`  = Q4_K for Q/K/O/gate/up + Q6_K for V/down, Ollama-compatible.
///          Skips the f32 intermediate entirely — quantisation happens in
///          the streaming extract loop straight from bf16 safetensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum QuantFormat {
    #[default]
    None,
    Q4k,
}

impl std::fmt::Display for QuantFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Q4k => write!(f, "q4k"),
        }
    }
}

/// Per-projection storage precision tag for FP4 vindexes.
///
/// Legal values for `Fp4Config.projections.{gate,up,down}.precision`.
/// Readers MUST dispatch on this tag and MUST NOT sniff filenames.
/// Unrecognised values should produce an explicit error rather than
/// silently downgrade — future tags (e.g. `fp6`, `nf4`) will require
/// a code-path addition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    /// FP4 E2M1 values + FP8 E4M3 sub-block scales + FP8 E4M3 block scale.
    Fp4,
    /// FP8 E4M3 values + FP8 E4M3 block scale. No sub-block scales.
    Fp8,
    /// Legacy IEEE half-precision. Uses the non-suffixed filename.
    F16,
    /// Legacy f32. Uses the non-suffixed filename.
    F32,
}

impl std::fmt::Display for Precision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fp4 => write!(f, "fp4"),
            Self::Fp8 => write!(f, "fp8"),
            Self::F16 => write!(f, "f16"),
            Self::F32 => write!(f, "f32"),
        }
    }
}

/// One projection's storage descriptor in the FP4 manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionFormat {
    pub precision: Precision,
    /// Filename relative to the vindex directory. Readers open this
    /// file directly. Must be the legacy name (e.g. `gate_vectors.bin`)
    /// when `precision` is `f16`/`f32`, and the suffixed name (e.g.
    /// `gate_vectors_fp4.bin`) when `precision` is `fp4`/`fp8`.
    pub file: String,
}

/// The three FFN projection tags covered by FP4 storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Projections {
    pub gate: ProjectionFormat,
    pub up: ProjectionFormat,
    pub down: ProjectionFormat,
}

/// Self-policing gate applied at extract time. When a projection's Q1
/// compliance falls below `min_compliant_fraction` at `threshold_ratio`,
/// the extractor downgrades that projection to `fallback_precision`
/// rather than committing a vindex that silently violates the contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceGate {
    pub threshold_ratio: f32,
    pub min_compliant_fraction: f32,
    pub fallback_precision: Precision,
}

/// FP4 vindex manifest — the inline block that lives under `index.json.fp4`
/// when any FFN projection is stored in FP4 or FP8.
///
/// `fp4_format_version` is independent of `VindexConfig.version`. It
/// bumps only when the on-disk byte layout of blocks themselves
/// changes; schema additions (new precision tags, new optional fields)
/// are non-breaking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fp4Config {
    pub fp4_format_version: u32,
    /// Elements per FP4/FP8 block. v1 pins this at 256 (the largest
    /// size that divides every model family LARQL currently ships).
    pub block_elements: u32,
    /// Elements per sub-block. v1 pins this at 32 (matches OCP MXFP4).
    pub sub_block_elements: u32,
    /// Scale dtype for the 8 per-sub-block scales inside each FP4 block.
    /// v1: `"fp8_e4m3"`.
    pub sub_block_scale_dtype: String,
    /// Scale dtype for the per-block scale (both FP4 and FP8 blocks).
    /// v1: `"fp8_e4m3"`.
    pub block_scale_dtype: String,
    /// Encoding identifier for the FP4 4-bit values themselves.
    /// v1: `"fp4_e2m1_mxfp4_nibble_order"`.
    pub value_encoding: String,
    /// Per-projection precision + filename.
    pub projections: Projections,
    /// Compliance policy applied by the extractor.
    pub compliance_gate: ComplianceGate,
    /// Filename of the compliance sidecar (relative to vindex dir).
    /// v1 default: `"fp4_compliance.json"`.
    pub compliance_report: String,
}

impl Fp4Config {
    /// The v1 default: 256-element blocks, 32-element sub-blocks,
    /// FP4 E2M1 values with FP8 E4M3 two-level scales, MXFP4 nibble order.
    /// `projections` is filled by the caller.
    pub fn v1_defaults(projections: Projections) -> Self {
        Self {
            fp4_format_version: 1,
            block_elements: 256,
            sub_block_elements: 32,
            sub_block_scale_dtype: "fp8_e4m3".into(),
            block_scale_dtype: "fp8_e4m3".into(),
            value_encoding: "fp4_e2m1_mxfp4_nibble_order".into(),
            projections,
            compliance_gate: ComplianceGate {
                threshold_ratio: 16.0,
                min_compliant_fraction: 0.99,
                fallback_precision: Precision::Fp8,
            },
            compliance_report: "fp4_compliance.json".into(),
        }
    }

    /// Option B default: FP4 gate + FP4 up + FP8 down.
    pub fn option_b_default() -> Self {
        Self::v1_defaults(Projections {
            gate: ProjectionFormat { precision: Precision::Fp4, file: "gate_vectors_fp4.bin".into() },
            up:   ProjectionFormat { precision: Precision::Fp4, file: "up_features_fp4.bin".into() },
            down: ProjectionFormat { precision: Precision::Fp8, file: "down_features_fp8.bin".into() },
        })
    }
}

/// Model-specific layer band boundaries.
/// Computed during EXTRACT, stored in index.json, used by DESCRIBE and label matching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerBands {
    /// Syntax/morphological band (e.g., [0, 13] for Gemma 3 4B).
    pub syntax: (usize, usize),
    /// Knowledge/factual band (e.g., [14, 27] for Gemma 3 4B).
    pub knowledge: (usize, usize),
    /// Output/formatting band (e.g., [28, 33] for Gemma 3 4B).
    pub output: (usize, usize),
}

impl LayerBands {
    /// Known-good layer bands for supported model families.
    /// Returns None if the family isn't recognised — caller should fall back
    /// to treating all layers as a single band.
    pub fn for_family(family: &str, num_layers: usize) -> Option<Self> {
        let last = num_layers.saturating_sub(1);
        match (family, num_layers) {
            // Gemma family — validated via probe analysis
            ("gemma3", 34) => Some(Self { syntax: (0, 13), knowledge: (14, 27), output: (28, 33) }),
            ("gemma3", 42) => Some(Self { syntax: (0, 16), knowledge: (17, 34), output: (35, 41) }),
            ("gemma2", 26) => Some(Self { syntax: (0, 10), knowledge: (11, 20), output: (21, 25) }),
            ("gemma2", 42) => Some(Self { syntax: (0, 16), knowledge: (17, 34), output: (35, 41) }),
            ("gemma2", 46) => Some(Self { syntax: (0, 18), knowledge: (19, 37), output: (38, 45) }),

            // Gemma 4 family
            ("gemma4", 30) => Some(Self { syntax: (0, 11), knowledge: (12, 23), output: (24, 29) }),
            ("gemma4", 36) => Some(Self { syntax: (0, 14), knowledge: (15, 28), output: (29, 35) }),
            ("gemma4", 35) => Some(Self { syntax: (0, 13), knowledge: (14, 27), output: (28, 34) }),
            ("gemma4", 60) => Some(Self { syntax: (0, 23), knowledge: (24, 47), output: (48, 59) }),

            // Llama family
            ("llama", 32) => Some(Self { syntax: (0, 12), knowledge: (13, 25), output: (26, 31) }),
            ("llama", 40) => Some(Self { syntax: (0, 15), knowledge: (16, 32), output: (33, 39) }),
            ("llama", 80) => Some(Self { syntax: (0, 31), knowledge: (32, 63), output: (64, 79) }),

            // Mistral / Mixtral
            ("mistral", 32) => Some(Self { syntax: (0, 12), knowledge: (13, 25), output: (26, 31) }),
            ("mixtral", 32) => Some(Self { syntax: (0, 12), knowledge: (13, 25), output: (26, 31) }),

            // Qwen
            ("qwen2", 28) => Some(Self { syntax: (0, 10), knowledge: (11, 22), output: (23, 27) }),
            ("qwen2", 32) => Some(Self { syntax: (0, 12), knowledge: (13, 25), output: (26, 31) }),
            ("qwen2", 40) => Some(Self { syntax: (0, 15), knowledge: (16, 32), output: (33, 39) }),
            ("qwen2", 64) => Some(Self { syntax: (0, 25), knowledge: (26, 51), output: (52, 63) }),
            ("qwen2", 80) => Some(Self { syntax: (0, 31), knowledge: (32, 63), output: (64, 79) }),

            // Phi
            ("phi", 32) => Some(Self { syntax: (0, 12), knowledge: (13, 25), output: (26, 31) }),
            ("phi", 40) => Some(Self { syntax: (0, 15), knowledge: (16, 32), output: (33, 39) }),

            // GPT-2 (smaller, denser)
            ("gpt2", 12) => Some(Self { syntax: (0, 4), knowledge: (5, 9), output: (10, 11) }),
            ("gpt2", 24) => Some(Self { syntax: (0, 9), knowledge: (10, 19), output: (20, 23) }),
            ("gpt2", 36) => Some(Self { syntax: (0, 14), knowledge: (15, 28), output: (29, 35) }),
            ("gpt2", 48) => Some(Self { syntax: (0, 19), knowledge: (20, 38), output: (39, 47) }),

            // Fallback: estimate from layer count
            // ~40% syntax, ~40% knowledge, ~20% output
            _ if num_layers >= 8 => {
                let syntax_end = num_layers * 2 / 5;
                let knowledge_end = num_layers * 4 / 5;
                Some(Self {
                    syntax: (0, syntax_end.saturating_sub(1)),
                    knowledge: (syntax_end, knowledge_end.saturating_sub(1)),
                    output: (knowledge_end, last),
                })
            }

            // Too few layers to band meaningfully
            _ => None,
        }
    }

    /// Check which band a layer belongs to.
    pub fn band_for_layer(&self, layer: usize) -> &'static str {
        if layer >= self.syntax.0 && layer <= self.syntax.1 {
            "syntax"
        } else if layer >= self.knowledge.0 && layer <= self.knowledge.1 {
            "knowledge"
        } else if layer >= self.output.0 && layer <= self.output.1 {
            "output"
        } else {
            "unknown"
        }
    }
}

/// Model configuration stored in the vindex for architecture reconstruction.
/// All fields are serialized to index.json so the model architecture can be
/// reconstructed without the original config.json.
#[derive(Serialize, Deserialize, Clone)]
pub struct VindexModelConfig {
    pub model_type: String,
    pub head_dim: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub rope_base: f64,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    /// MoE configuration (None for dense models).
    #[serde(default)]
    pub moe: Option<MoeConfig>,

    // ── Gemma 4 per-layer attention geometry ──
    // All optional for backward compatibility with existing vindexes.

    /// Head dimension for global (full) attention layers. If None, all layers use head_dim.
    /// Gemma 4: 512 for global layers, head_dim (256) for sliding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub global_head_dim: Option<usize>,
    /// Number of KV heads for global attention layers. If None, all layers use num_kv_heads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_global_kv_heads: Option<usize>,
    /// Fraction of head_dim to apply RoPE to (0.0–1.0). If None, full rotation.
    /// Gemma 4 global layers: 0.25.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partial_rotary_factor: Option<f64>,
    /// Sliding window pattern: every Nth layer is full attention.
    /// Gemma 4: 6 (layers 5, 11, 17, ... are full).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sliding_window_pattern: Option<usize>,
    /// Explicit per-layer type array (e.g., ["sliding_attention", "full_attention", ...]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_types: Option<Vec<String>>,
    /// Whether value projection shares key projection (K=V).
    #[serde(default)]
    pub attention_k_eq_v: bool,
    /// Number of layers at the end that share KV from earlier layers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_kv_shared_layers: Option<usize>,
    /// Per-layer embedding dimension (PLE). 0 or None = no PLE.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub per_layer_embed_dim: Option<usize>,
    /// RoPE base for local/sliding window layers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_local_base: Option<f64>,
    /// Query pre-attention scalar (overrides 1/sqrt(head_dim)).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_pre_attn_scalar: Option<f64>,
    /// Final-logit tanh softcap (Gemma 2/3/4: 30.0). Applied to logits
    /// immediately before softmax in `logits_to_predictions`. Omitting it
    /// leaves logits uncapped — on E2B this peaked the softmax on the
    /// wrong token (observed: "Paris" → "hyperparameters").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_logit_softcapping: Option<f64>,
}

/// MoE (Mixture of Experts) configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoeConfig {
    /// Number of experts per layer.
    pub num_experts: usize,
    /// Number of experts selected per token (top-K routing).
    pub top_k: usize,
    /// Whether there's a shared expert always active (DeepSeek V2/V3).
    #[serde(default)]
    pub shared_expert: bool,
    /// Router type (e.g., "top_k_softmax", "gemma4_top_k_softmax").
    #[serde(default = "default_router_type")]
    pub router_type: String,
    /// Per-expert intermediate (hidden) dimension.
    /// Differs from the dense FFN intermediate_size in hybrid models (Gemma 4 A4B).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moe_intermediate_size: Option<usize>,
    /// Hybrid MoE: dense MLP and expert block coexist in each layer, outputs summed.
    /// True for Gemma 4 A4B. False for pure MoE (Mixtral, DeepSeek).
    #[serde(default)]
    pub hybrid: bool,
}

fn default_router_type() -> String {
    "top_k_softmax".to_string()
}

/// Per-layer info for gate_vectors.bin layout.
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct VindexLayerInfo {
    pub layer: usize,
    pub num_features: usize,
    /// Byte offset into gate_vectors.bin.
    pub offset: u64,
    /// Byte length of this layer's gate data.
    pub length: u64,
    /// Number of experts at this layer (None or absent for dense models).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_experts: Option<usize>,
    /// Features per expert (None or absent for dense models).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_features_per_expert: Option<usize>,
}

/// Down metadata entry in the NDJSON file (compact, no vectors).
#[derive(Serialize, Deserialize)]
pub struct DownMetaRecord {
    #[serde(rename = "l")]
    pub layer: usize,
    #[serde(rename = "f")]
    pub feature: usize,
    #[serde(rename = "t")]
    pub top_token: String,
    #[serde(rename = "i")]
    pub top_token_id: u32,
    #[serde(rename = "c")]
    pub c_score: f32,
    #[serde(rename = "k")]
    pub top_k: Vec<DownMetaTopK>,
}

#[derive(Serialize, Deserialize)]
pub struct DownMetaTopK {
    #[serde(rename = "t")]
    pub token: String,
    #[serde(rename = "i")]
    pub token_id: u32,
    #[serde(rename = "s")]
    pub logit: f32,
}

#[cfg(test)]
mod fp4_schema_tests {
    use super::*;

    #[test]
    fn option_b_default_shape() {
        let cfg = Fp4Config::option_b_default();
        assert_eq!(cfg.fp4_format_version, 1);
        assert_eq!(cfg.block_elements, 256);
        assert_eq!(cfg.sub_block_elements, 32);
        assert_eq!(cfg.sub_block_scale_dtype, "fp8_e4m3");
        assert_eq!(cfg.block_scale_dtype, "fp8_e4m3");
        assert_eq!(cfg.value_encoding, "fp4_e2m1_mxfp4_nibble_order");
        assert!(matches!(cfg.projections.gate.precision, Precision::Fp4));
        assert!(matches!(cfg.projections.up.precision, Precision::Fp4));
        assert!(matches!(cfg.projections.down.precision, Precision::Fp8));
        assert_eq!(cfg.projections.gate.file, "gate_vectors_fp4.bin");
        assert_eq!(cfg.projections.down.file, "down_features_fp8.bin");
        assert_eq!(cfg.compliance_gate.threshold_ratio, 16.0);
        assert_eq!(cfg.compliance_gate.min_compliant_fraction, 0.99);
        assert!(matches!(cfg.compliance_gate.fallback_precision, Precision::Fp8));
        assert_eq!(cfg.compliance_report, "fp4_compliance.json");
    }

    #[test]
    fn fp4_config_serde_round_trip() {
        let cfg = Fp4Config::option_b_default();
        let json = serde_json::to_string(&cfg).unwrap();
        let back: Fp4Config = serde_json::from_str(&json).unwrap();
        assert_eq!(back.fp4_format_version, cfg.fp4_format_version);
        assert_eq!(back.block_elements, cfg.block_elements);
        assert_eq!(back.projections.gate.file, cfg.projections.gate.file);
    }

    #[test]
    fn precision_json_is_snake_case() {
        let cfg = Fp4Config::option_b_default();
        let json = serde_json::to_string(&cfg).unwrap();
        // The JSON surface must use the stable tags the format spec pins.
        assert!(json.contains("\"fp4\""));
        assert!(json.contains("\"fp8\""));
        assert!(!json.contains("\"Fp4\""), "camel/title case leaked: {json}");
    }

    #[test]
    fn vindex_config_without_fp4_serialises_without_key() {
        // Verify the `skip_serializing_if = "Option::is_none"` path so a
        // legacy vindex's index.json is byte-stable after a round trip.
        let cfg = VindexConfig {
            version: 2,
            model: "x".into(),
            family: "gemma3".into(),
            source: None,
            checksums: None,
            num_layers: 1,
            hidden_size: 256,
            intermediate_size: 1024,
            vocab_size: 32,
            embed_scale: 1.0,
            extract_level: ExtractLevel::default(),
            dtype: Default::default(),
            quant: QuantFormat::None,
            layer_bands: None,
            layers: vec![],
            down_top_k: 10,
            has_model_weights: false,
            model_config: None,
            fp4: None,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(!json.contains("\"fp4\""), "legacy config leaked fp4 field: {json}");

        // And still deserialises when the key is absent (default).
        let parsed: VindexConfig = serde_json::from_str(&json).unwrap();
        assert!(parsed.fp4.is_none());
    }

    #[test]
    fn vindex_config_with_fp4_round_trips() {
        let cfg = VindexConfig {
            version: 2,
            model: "x".into(),
            family: "gemma3".into(),
            source: None,
            checksums: None,
            num_layers: 1,
            hidden_size: 256,
            intermediate_size: 1024,
            vocab_size: 32,
            embed_scale: 1.0,
            extract_level: ExtractLevel::default(),
            dtype: Default::default(),
            quant: QuantFormat::None,
            layer_bands: None,
            layers: vec![],
            down_top_k: 10,
            has_model_weights: false,
            model_config: None,
            fp4: Some(Fp4Config::option_b_default()),
        };
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(json.contains("\"fp4\""));
        let parsed: VindexConfig = serde_json::from_str(&json).unwrap();
        let fp4 = parsed.fp4.expect("round trip kept fp4");
        assert!(matches!(fp4.projections.down.precision, Precision::Fp8));
    }
}
