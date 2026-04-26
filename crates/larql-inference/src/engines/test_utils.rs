//! Synthetic test fixtures for engine and layer-graph unit tests.
//!
//! Three helpers:
//! - `make_test_weights()` — fully functional 2-layer ModelWeights (no disk I/O)
//! - `make_test_vindex(weights)` — in-memory VectorIndex with random gate vectors
//! - `make_test_tokenizer(vocab_size)` — WordLevel tokenizer mapping token N to "[N]"
//!
//! Dimensions: vocab=32, hidden=16, intermediate=32, 2 q-heads, 1 kv-head,
//! head_dim=8, 2 layers. Forward pass ≈ 10 ms on CPU.

use std::collections::HashMap;
use ndarray::Array2;
use larql_models::{ModelWeights, WeightArray, detect_from_json};

/// Build a synthetic `ModelWeights` with all tensors populated.
/// Uses `TinyModelArch` key conventions (e.g. `"0.attn.q_proj.weight"`).
pub fn make_test_weights() -> ModelWeights {
    const VOCAB: usize = 32;
    const HIDDEN: usize = 16;
    const INTER: usize = 32;
    const NUM_Q: usize = 2;
    const NUM_KV: usize = 1;
    const HEAD_DIM: usize = 8;
    const NUM_LAYERS: usize = 2;

    let arch_json = serde_json::json!({
        "model_type": "tinymodel",
        "hidden_size": HIDDEN,
        "num_hidden_layers": NUM_LAYERS,
        "intermediate_size": INTER,
        "head_dim": HEAD_DIM,
        "num_attention_heads": NUM_Q,
        "num_key_value_heads": NUM_KV,
        "vocab_size": VOCAB,
    });
    let arch = detect_from_json(&arch_json);

    let mut tensors: HashMap<String, WeightArray> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut rng_state = 0xdeadbeef_u64;

    // LCG giving values in [-scale, +scale]
    let mut rand_mat = |rows: usize, cols: usize, scale: f32| -> WeightArray {
        let data: Vec<f32> = (0..rows * cols)
            .map(|_| {
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (rng_state as u32) as f32 / u32::MAX as f32 * 2.0 * scale - scale
            })
            .collect();
        Array2::from_shape_vec((rows, cols), data).unwrap().into_shared()
    };

    // Embed + lm_head
    let embed = rand_mat(VOCAB, HIDDEN, 0.1);
    let lm_head = rand_mat(VOCAB, HIDDEN, 0.1);
    tensors.insert(arch.embed_key().to_string(), embed.clone());

    // Final norm (ones → valid unweighted RMSNorm fallback)
    vectors.insert(arch.final_norm_key().to_string(), vec![1.0; HIDDEN]);

    let q_dim = NUM_Q * HEAD_DIM;
    let kv_dim = NUM_KV * HEAD_DIM;

    for layer in 0..NUM_LAYERS {
        // Attention projections
        tensors.insert(arch.attn_q_key(layer), rand_mat(q_dim, HIDDEN, 0.1));
        tensors.insert(arch.attn_k_key(layer), rand_mat(kv_dim, HIDDEN, 0.1));
        tensors.insert(arch.attn_v_key(layer), rand_mat(kv_dim, HIDDEN, 0.1));
        tensors.insert(arch.attn_o_key(layer), rand_mat(HIDDEN, q_dim, 0.1));
        // FFN — missing tensors cause panic, so always provide them
        tensors.insert(arch.ffn_gate_key(layer), rand_mat(INTER, HIDDEN, 0.1));
        tensors.insert(arch.ffn_up_key(layer), rand_mat(INTER, HIDDEN, 0.1));
        tensors.insert(arch.ffn_down_key(layer), rand_mat(HIDDEN, INTER, 0.1));
        // Layer norms
        vectors.insert(arch.input_layernorm_key(layer), vec![1.0; HIDDEN]);
        vectors.insert(arch.post_attention_layernorm_key(layer), vec![1.0; HIDDEN]);
    }

    ModelWeights {
        tensors,
        vectors,
        raw_bytes: HashMap::new(),
        packed_mmaps: HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_byte_ranges: HashMap::new(),
        embed,
        lm_head,
        arch,
        num_layers: NUM_LAYERS,
        hidden_size: HIDDEN,
        intermediate_size: INTER,
        vocab_size: VOCAB,
        head_dim: HEAD_DIM,
        num_q_heads: NUM_Q,
        num_kv_heads: NUM_KV,
        rope_base: 10_000.0,
    }
}

/// Build an in-memory `VectorIndex` with random gate vectors per layer.
/// The VectorIndex has no Q4K or interleaved data — `predict_honest` falls
/// through to the CPU path, and `WalkFfn` routes through the sparse fallback
/// that uses `weights.tensors`.
pub fn make_test_vindex(weights: &ModelWeights) -> larql_vindex::VectorIndex {
    let n_features = weights.intermediate_size;
    let hidden = weights.hidden_size;

    // Each layer gets an independent LCG seed so gate matrices are distinct.
    let gate_vectors: Vec<Option<Array2<f32>>> = (0..weights.num_layers)
        .map(|l| {
            let mut state = 0xabcdef_u64.wrapping_add(l as u64 * 0x9e3779b97f4a7c15);
            let data: Vec<f32> = (0..n_features * hidden).map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (state as u32) as f32 / u32::MAX as f32 * 0.1 - 0.05
            }).collect();
            Some(Array2::from_shape_vec((n_features, hidden), data).unwrap())
        })
        .collect();

    let down_meta = vec![None; weights.num_layers];
    larql_vindex::VectorIndex::new(gate_vectors, down_meta, weights.num_layers, hidden)
}

/// Build a `tokenizers::Tokenizer` with a vocabulary of `vocab_size` tokens.
/// Token N decodes to `"[N]"`, so token IDs from `make_test_weights()` all
/// decode to valid (if meaningless) strings.
pub fn make_test_tokenizer(vocab_size: usize) -> tokenizers::Tokenizer {
    // WordLevel::builder().vocab() requires an AHashMap.
    // Build a simple BPE-less tokenizer via JSON serialization instead.
    let mut vocab_json = serde_json::Map::new();
    for i in 0..vocab_size as u64 {
        vocab_json.insert(format!("[{i}]"), serde_json::Value::Number(i.into()));
    }
    // Add UNK token at the end
    vocab_json.insert("[UNK]".into(), serde_json::Value::Number(vocab_size.into()));

    let tokenizer_json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": { "type": "Whitespace" },
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": vocab_json,
            "unk_token": "[UNK]"
        }
    });

    let bytes = serde_json::to_vec(&tokenizer_json).expect("JSON serialization failed");
    tokenizers::Tokenizer::from_bytes(&bytes).expect("synthetic tokenizer construction failed")
}

/// All three synthetic fixtures bundled together. Build once per test module
/// via `OnceLock`; each field is cheaply borrowed.
pub struct TestFixtures {
    pub weights: ModelWeights,
    pub tokenizer: tokenizers::Tokenizer,
    pub index: larql_vindex::VectorIndex,
}

impl TestFixtures {
    pub fn build() -> Self {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let index = make_test_vindex(&weights);
        Self { weights, tokenizer, index }
    }
}
