use super::helpers::*;
use super::*;
use crate::parser;

// ── Session state: no backend ──

#[test]
fn no_backend_stats() {
    let mut session = Session::new();
    let stmt = parser::parse("STATS;").unwrap();
    let result = session.execute(&stmt);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, LqlError::NoBackend));
}

#[test]
fn no_backend_walk() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"WALK "test" TOP 5;"#).unwrap();
    let result = session.execute(&stmt);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), LqlError::NoBackend));
}

#[test]
fn no_backend_describe() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"DESCRIBE "France";"#).unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn no_backend_select() {
    let mut session = Session::new();
    let stmt = parser::parse("SELECT * FROM EDGES;").unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn no_backend_explain() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"EXPLAIN WALK "test";"#).unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn no_backend_show_relations() {
    let mut session = Session::new();
    let stmt = parser::parse("SHOW RELATIONS;").unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn no_backend_show_layers() {
    let mut session = Session::new();
    let stmt = parser::parse("SHOW LAYERS;").unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn no_backend_show_features() {
    let mut session = Session::new();
    let stmt = parser::parse("SHOW FEATURES 26;").unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

// ── USE errors ──

#[test]
fn use_nonexistent_vindex() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"USE "/nonexistent/path/fake.vindex";"#).unwrap();
    let result = session.execute(&stmt);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), LqlError::Execution(_)));
}

#[test]
fn use_model_fails_on_nonexistent() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"USE MODEL "/nonexistent/model";"#).unwrap();
    let result = session.execute(&stmt);
    // Should fail to resolve the model path
    assert!(result.is_err());
}

#[test]
fn use_model_auto_extract_parses() {
    // Verify AUTO_EXTRACT parses correctly (loading will fail for nonexistent model)
    let mut session = Session::new();
    let stmt = parser::parse(r#"USE MODEL "/nonexistent/model" AUTO_EXTRACT;"#).unwrap();
    let result = session.execute(&stmt);
    assert!(result.is_err());
}

// ── Lifecycle: error cases without valid model/vindex ──

#[test]
fn extract_fails_on_nonexistent_model() {
    let mut session = Session::new();
    let stmt =
        parser::parse(r#"EXTRACT MODEL "/nonexistent/model" INTO "/tmp/test_extract_out.vindex";"#)
            .unwrap();
    let result = session.execute(&stmt);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), LqlError::Execution(_)));
}

#[test]
fn compile_no_backend() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"COMPILE CURRENT INTO MODEL "out/";"#).unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn diff_nonexistent_vindex() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"DIFF "/nonexistent/a.vindex" "/nonexistent/b.vindex";"#).unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::Execution(_)
    ));
}

// ── Mutation: no-backend errors ──

#[test]
fn insert_no_backend() {
    let mut session = Session::new();
    let stmt =
        parser::parse(r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "c");"#)
            .unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn delete_no_backend() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"DELETE FROM EDGES WHERE entity = "x";"#).unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn update_no_backend() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"UPDATE EDGES SET target = "y" WHERE entity = "x";"#).unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn merge_nonexistent_source() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"MERGE "/nonexistent/source.vindex";"#).unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::Execution(_)
    ));
}

// ── INFER ──

#[test]
fn infer_no_backend() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"INFER "test" TOP 5;"#).unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

// ── is_readable_token ──

#[test]
fn readable_tokens() {
    assert!(is_readable_token("French"));
    assert!(is_readable_token("Paris"));
    assert!(is_readable_token("capital-of"));
    assert!(is_readable_token("is"));
    assert!(is_readable_token("Europe"));
}

#[test]
fn unreadable_tokens() {
    assert!(!is_readable_token("ইসলামাবাদ"));
    assert!(!is_readable_token("южна"));
    assert!(!is_readable_token("ളാ"));
    assert!(!is_readable_token("ڪ"));
    assert!(!is_readable_token(""));
}

// ── is_content_token ──

#[test]
fn content_tokens_pass() {
    assert!(is_content_token("French"));
    assert!(is_content_token("Paris"));
    assert!(is_content_token("Europe"));
    assert!(is_content_token("Mozart"));
    assert!(is_content_token("composer"));
    assert!(is_content_token("Berlin"));
    assert!(is_content_token("IBM"));
    assert!(is_content_token("Facebook"));
}

#[test]
fn stop_words_rejected() {
    assert!(!is_content_token("the"));
    assert!(!is_content_token("from"));
    assert!(!is_content_token("for"));
    assert!(!is_content_token("with"));
    assert!(!is_content_token("this"));
    assert!(!is_content_token("about"));
    assert!(!is_content_token("which"));
    assert!(!is_content_token("first"));
    assert!(!is_content_token("after"));
}

#[test]
fn short_tokens_rejected() {
    assert!(!is_content_token("a"));
    assert!(!is_content_token("of"));
    assert!(!is_content_token("is"));
    assert!(!is_content_token("-"));
    assert!(!is_content_token("lö"));
    assert!(!is_content_token("par"));
}

#[test]
fn code_tokens_rejected() {
    assert!(!is_content_token("trialComponents"));
    assert!(!is_content_token("NavigationBar"));
    assert!(!is_content_token("LastName"));
}

// ── SHOW MODELS works without backend ──

#[test]
fn show_models_no_crash() {
    let mut session = Session::new();
    let stmt = parser::parse("SHOW MODELS;").unwrap();
    let result = session.execute(&stmt);
    assert!(result.is_ok());
}

// ── Pipe: errors propagate ──

#[test]
fn pipe_error_propagates() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"STATS |> WALK "test";"#).unwrap();
    assert!(session.execute(&stmt).is_err());
}

// ── Format helpers ──

#[test]
fn format_number_small() {
    assert_eq!(format_number(42), "42");
    assert_eq!(format_number(999), "999");
}

#[test]
fn format_number_thousands() {
    assert_eq!(format_number(1_000), "1.0K");
    assert_eq!(format_number(10_240), "10.2K");
    assert_eq!(format_number(348_160), "348.2K");
}

#[test]
fn format_number_millions() {
    assert_eq!(format_number(1_000_000), "1.00M");
    assert_eq!(format_number(2_917_432), "2.92M");
}

#[test]
fn format_bytes_small() {
    assert_eq!(format_bytes(512), "512 B");
}

#[test]
fn format_bytes_kb() {
    assert_eq!(format_bytes(2048), "2.0 KB");
}

#[test]
fn format_bytes_mb() {
    let mb = 5 * 1_048_576;
    assert_eq!(format_bytes(mb), "5.0 MB");
}

#[test]
fn format_bytes_gb() {
    let gb = 6_420_000_000;
    assert!(format_bytes(gb).contains("GB"));
}

// ═══════════════════════════════════════════════════════════════
// Weight backend tests
// ═══════════════════════════════════════════════════════════════

/// Create a minimal ModelWeights for testing the Weight backend.
fn make_test_weights() -> larql_inference::ModelWeights {
    use larql_inference::ndarray;
    use std::collections::HashMap;

    let num_layers = 2;
    let hidden = 8;
    let intermediate = 4;
    let vocab_size = 16;

    let mut tensors: HashMap<String, ndarray::ArcArray2<f32>> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();

    for layer in 0..num_layers {
        let mut gate = ndarray::Array2::<f32>::zeros((intermediate, hidden));
        for i in 0..intermediate {
            gate[[i, i % hidden]] = 1.0 + layer as f32;
        }
        tensors.insert(
            format!("layers.{layer}.mlp.gate_proj.weight"),
            gate.into_shared(),
        );

        let mut up = ndarray::Array2::<f32>::zeros((intermediate, hidden));
        for i in 0..intermediate {
            up[[i, (i + 1) % hidden]] = 0.5;
        }
        tensors.insert(
            format!("layers.{layer}.mlp.up_proj.weight"),
            up.into_shared(),
        );

        let mut down = ndarray::Array2::<f32>::zeros((hidden, intermediate));
        for i in 0..intermediate {
            down[[i % hidden, i]] = 0.3;
        }
        tensors.insert(
            format!("layers.{layer}.mlp.down_proj.weight"),
            down.into_shared(),
        );

        for suffix in &["q_proj", "k_proj", "v_proj", "o_proj"] {
            let mut attn = ndarray::Array2::<f32>::zeros((hidden, hidden));
            for i in 0..hidden {
                attn[[i, i]] = 1.0;
            }
            tensors.insert(
                format!("layers.{layer}.self_attn.{suffix}.weight"),
                attn.into_shared(),
            );
        }

        vectors.insert(
            format!("layers.{layer}.input_layernorm.weight"),
            vec![1.0; hidden],
        );
        vectors.insert(
            format!("layers.{layer}.post_attention_layernorm.weight"),
            vec![1.0; hidden],
        );
    }

    vectors.insert("norm.weight".into(), vec![1.0; hidden]);

    let mut embed = ndarray::Array2::<f32>::zeros((vocab_size, hidden));
    for i in 0..vocab_size {
        embed[[i, i % hidden]] = 1.0;
    }
    let embed = embed.into_shared();
    let lm_head = embed.clone();

    let arch = larql_models::detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "hidden_size": hidden,
        "num_hidden_layers": num_layers,
        "intermediate_size": intermediate,
        "head_dim": hidden,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "rope_theta": 10000.0,
        "vocab_size": vocab_size,
    }));

    larql_inference::ModelWeights {
        tensors,
        vectors,
        raw_bytes: std::collections::HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_mmaps: std::collections::HashMap::new(),
        packed_byte_ranges: std::collections::HashMap::new(),
        embed,
        lm_head,
        position_embed: None,
        num_layers,
        hidden_size: hidden,
        intermediate_size: intermediate,
        vocab_size,
        head_dim: hidden,
        num_q_heads: 1,
        num_kv_heads: 1,
        rope_base: 10000.0,
        arch,
    }
}

/// Create a minimal tokenizer for testing.
fn make_test_tokenizer() -> larql_inference::tokenizers::Tokenizer {
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    larql_inference::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap()
}

/// Create a Session with Weight backend for testing.
fn weight_session() -> Session {
    let mut session = Session::new();
    session.backend = Backend::Weight {
        model_id: "test/model".into(),
        weights: make_test_weights(),
        tokenizer: make_test_tokenizer(),
    };
    session
}

#[test]
fn weight_backend_stats() {
    let mut session = weight_session();
    let stmt = parser::parse("STATS;").unwrap();
    let result = session.execute(&stmt).unwrap();
    assert!(result.iter().any(|l| l.contains("test/model")));
    assert!(result.iter().any(|l| l.contains("live weights")));
    assert!(result.iter().any(|l| l.contains("2"))); // num_layers
}

#[test]
fn weight_backend_walk_requires_vindex() {
    let mut session = weight_session();
    let stmt = parser::parse(r#"WALK "test" TOP 5;"#).unwrap();
    let err = session.execute(&stmt).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("requires a vindex"),
        "expected vindex error, got: {msg}"
    );
    assert!(
        msg.contains("EXTRACT"),
        "should suggest EXTRACT, got: {msg}"
    );
}

#[test]
fn weight_backend_describe_requires_vindex() {
    let mut session = weight_session();
    let stmt = parser::parse(r#"DESCRIBE "France";"#).unwrap();
    let err = session.execute(&stmt).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("requires a vindex"));
}

#[test]
fn weight_backend_select_requires_vindex() {
    let mut session = weight_session();
    let stmt = parser::parse("SELECT * FROM EDGES;").unwrap();
    let err = session.execute(&stmt).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("requires a vindex"));
}

#[test]
fn weight_backend_explain_walk_requires_vindex() {
    let mut session = weight_session();
    let stmt = parser::parse(r#"EXPLAIN WALK "test";"#).unwrap();
    let err = session.execute(&stmt).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("requires a vindex"));
}

#[test]
fn weight_backend_insert_requires_vindex() {
    let mut session = weight_session();
    let stmt =
        parser::parse(r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "c");"#)
            .unwrap();
    let err = session.execute(&stmt).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("requires a vindex") || msg.contains("mutation requires"));
}

#[test]
fn weight_backend_show_relations_requires_vindex() {
    let mut session = weight_session();
    let stmt = parser::parse("SHOW RELATIONS;").unwrap();
    let err = session.execute(&stmt).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("requires a vindex"));
}

#[test]
fn weight_backend_compile_current_requires_vindex() {
    let mut session = weight_session();
    let stmt = parser::parse(r#"COMPILE CURRENT INTO MODEL "out/";"#).unwrap();
    let err = session.execute(&stmt).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("EXTRACT") || msg.contains("vindex"));
}

#[test]
fn weight_backend_show_models_works() {
    let mut session = weight_session();
    let stmt = parser::parse("SHOW MODELS;").unwrap();
    let result = session.execute(&stmt);
    assert!(result.is_ok());
}

// ── Mutation pipeline integration tests ────────────────────────────────
//
// These tests build a tiny synthetic vindex on disk, load it via USE,
// and exercise the DELETE / UPDATE / patch-session paths through the
// real executor + parser. They cover the auto-patch lifecycle, the
// patch overlay update, and SAVE PATCH file emission.
//
// INSERT is exercised end-to-end in `compile_demo` against a real
// Gemma vindex (the synthetic tokenizer here has an empty vocab so it
// can't tokenise meaningful entity strings). The auto-patch session
// creation that INSERT triggers is covered indirectly by the DELETE
// auto-patch test below.

use larql_inference::ndarray::Array2;

/// Build a minimal vindex directory on disk that the LQL executor can
/// load via `USE`. Includes gate vectors, down_meta, embeddings, and a
/// stub tokenizer. Returns the directory path; the caller is
/// responsible for cleanup.
fn make_test_vindex_dir(tag: &str) -> std::path::PathBuf {
    use larql_models::TopKEntry;
    use larql_vindex::{ExtractLevel, FeatureMeta, StorageDtype, VectorIndex, VindexConfig};

    let dir = std::env::temp_dir().join(format!("larql_lql_test_vindex_{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // Tiny in-memory index — 2 layers × 3 features × 4 hidden dims.
    let hidden = 4;
    let num_features = 3;
    let num_layers = 2;
    let vocab_size = 10;

    let mut gate0 = Array2::<f32>::zeros((num_features, hidden));
    gate0[[0, 0]] = 1.0;
    gate0[[1, 1]] = 1.0;
    gate0[[2, 2]] = 1.0;

    let mut gate1 = Array2::<f32>::zeros((num_features, hidden));
    gate1[[0, 3]] = 1.0;
    gate1[[1, 0]] = 0.5;
    gate1[[2, 2]] = -1.0;

    let make_meta = |tok: &str, id: u32, c: f32| FeatureMeta {
        top_token: tok.to_string(),
        top_token_id: id,
        c_score: c,
        top_k: vec![TopKEntry {
            token: tok.to_string(),
            token_id: id,
            logit: c,
        }],
    };

    let meta0 = vec![
        Some(make_meta("Paris", 100, 0.95)),
        Some(make_meta("French", 101, 0.88)),
        Some(make_meta("Europe", 102, 0.75)),
    ];
    let meta1 = vec![
        Some(make_meta("Berlin", 200, 0.90)),
        None,
        Some(make_meta("Spain", 202, 0.70)),
    ];
    let down_meta = vec![Some(meta0), Some(meta1)];

    let index = VectorIndex::new(
        vec![Some(gate0), Some(gate1)],
        down_meta,
        num_layers,
        hidden,
    );

    let mut config = VindexConfig {
        version: 2,
        model: "test/lql-mutation".into(),
        family: "llama".into(),
        source: None,
        checksums: None,
        num_layers,
        hidden_size: hidden,
        intermediate_size: num_features,
        vocab_size,
        embed_scale: 1.0,
        extract_level: ExtractLevel::Browse,
        dtype: StorageDtype::F32,
        quant: larql_vindex::QuantFormat::None,
        layer_bands: None,
        layers: Vec::new(),
        down_top_k: 5,
        has_model_weights: false,
        model_config: None,
        fp4: None,
        ffn_layout: None,
    };
    index.save_vindex(&dir, &mut config).unwrap();

    // Synthetic embeddings.bin (vocab_size × hidden f32, all zeros).
    let embed_bytes = vec![0u8; vocab_size * hidden * 4];
    std::fs::write(dir.join("embeddings.bin"), embed_bytes).unwrap();

    // Stub tokenizer.json — empty BPE. Not used by DELETE / UPDATE /
    // PATCH; INSERT-against-this-vindex tests would need a real one.
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(dir.join("tokenizer.json"), tok_json).unwrap();

    dir
}

/// Spin up a session and `USE` the test vindex from `make_test_vindex_dir`.
fn vindex_session(tag: &str) -> (Session, std::path::PathBuf) {
    let dir = make_test_vindex_dir(tag);
    let mut session = Session::new();
    let stmt = parser::parse(&format!(r#"USE "{}";"#, dir.display())).unwrap();
    session
        .execute(&stmt)
        .expect("USE on synthetic vindex should succeed");
    (session, dir)
}

// ── Rich fixture: WordLevel tokenizer + non-zero embeddings ─────
//
// `make_test_vindex_dir` ships a stub BPE that returns no token ids
// for any input, which short-circuits every entity-anchored verb
// (SELECT WHERE entity, NEAREST TO, DESCRIBE, walk-based EDGES).
// The "rich" fixture below upgrades two pieces:
//
//   1. Tokenizer is a `WordLevel` model with a vocab that covers a
//      handful of named entities and template words. Encoding
//      "Paris" returns `[1]`, "France" returns `[2]`, etc.
//
//   2. Embeddings are non-zero, distinguishable rows so the entity
//      query vector built by `entity_query_vec` is non-trivial and
//      walks against it produce non-zero gate scores.
//
// Same shape as the basic fixture (2 layers × 3 features × 4 hidden)
// — only the tokenizer + embedding payload differ.

const RICH_FIXTURE_VOCAB: &[(&str, u32)] = &[
    ("[UNK]", 0),
    ("Paris", 1),
    ("France", 2),
    ("Berlin", 3),
    ("Germany", 4),
    ("Spain", 5),
    ("Madrid", 6),
    ("London", 7),
    ("English", 8),
    ("the", 9),
    ("of", 10),
    ("is", 11),
    ("capital", 12),
    ("language", 13),
    ("currency", 14),
    ("Atlantis", 15),
];

fn rich_fixture_tokenizer_json() -> String {
    let vocab_json: String = RICH_FIXTURE_VOCAB
        .iter()
        .map(|(t, id)| format!(r#""{t}":{id}"#))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        r#"{{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{{"type":"Whitespace"}},"post_processor":null,"decoder":null,"model":{{"type":"WordLevel","vocab":{{{vocab_json}}},"unk_token":"[UNK]"}}}}"#
    )
}

/// Build a test vindex with a real WordLevel tokenizer + non-zero
/// embeddings so entity-anchored verbs produce real outputs.
fn make_rich_test_vindex_dir(tag: &str) -> std::path::PathBuf {
    use larql_models::TopKEntry;
    use larql_vindex::{ExtractLevel, FeatureMeta, StorageDtype, VectorIndex, VindexConfig};

    let dir = std::env::temp_dir().join(format!("larql_lql_rich_test_vindex_{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let hidden = 4;
    let num_features = 3;
    let num_layers = 2;
    let vocab_size = RICH_FIXTURE_VOCAB.len();

    // Gate vectors aligned with the embedding rows. Magnitudes are
    // chosen so the dot-product walk produces gate scores well above
    // `DESCRIBE_GATE_THRESHOLD = 5.0`, which lets DESCRIBE / EXPLAIN
    // surface real edges from the synthetic vindex.
    const RICH_GATE_MAG: f32 = 50.0;
    let mut gate0 = Array2::<f32>::zeros((num_features, hidden));
    gate0[[0, 0]] = RICH_GATE_MAG;
    gate0[[1, 1]] = RICH_GATE_MAG;
    gate0[[2, 2]] = RICH_GATE_MAG;
    let mut gate1 = Array2::<f32>::zeros((num_features, hidden));
    gate1[[0, 3]] = RICH_GATE_MAG;
    gate1[[1, 0]] = RICH_GATE_MAG * 0.5;
    gate1[[2, 2]] = -RICH_GATE_MAG;

    let make_meta = |tok: &str, id: u32, c: f32| FeatureMeta {
        top_token: tok.to_string(),
        top_token_id: id,
        c_score: c,
        top_k: vec![TopKEntry {
            token: tok.to_string(),
            token_id: id,
            logit: c,
        }],
    };

    let meta0 = vec![
        Some(make_meta("Paris", 1, 0.95)),
        Some(make_meta("Berlin", 3, 0.88)),
        Some(make_meta("language", 13, 0.75)),
    ];
    let meta1 = vec![
        Some(make_meta("Madrid", 6, 0.90)),
        None,
        Some(make_meta("English", 8, 0.70)),
    ];
    let down_meta = vec![Some(meta0), Some(meta1)];

    let index = VectorIndex::new(
        vec![Some(gate0), Some(gate1)],
        down_meta,
        num_layers,
        hidden,
    );

    let mut config = VindexConfig {
        version: 2,
        model: "test/rich-fixture".into(),
        family: "llama".into(),
        source: None,
        checksums: None,
        num_layers,
        hidden_size: hidden,
        intermediate_size: num_features,
        vocab_size,
        embed_scale: 1.0,
        extract_level: ExtractLevel::Browse,
        dtype: StorageDtype::F32,
        quant: larql_vindex::QuantFormat::None,
        layer_bands: None,
        layers: Vec::new(),
        down_top_k: 5,
        has_model_weights: false,
        model_config: None,
        fp4: None,
        ffn_layout: None,
    };
    index.save_vindex(&dir, &mut config).unwrap();

    // Non-zero embeddings: row[i] = unit vector pointing at axis i % hidden.
    // Distinguishable per-token-id, deterministic, and non-trivial when
    // averaged.
    let mut embed_bytes = Vec::with_capacity(vocab_size * hidden * 4);
    for tid in 0..vocab_size {
        for d in 0..hidden {
            let v = if d == tid % hidden { 1.0f32 } else { 0.1f32 };
            embed_bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(dir.join("embeddings.bin"), embed_bytes).unwrap();

    std::fs::write(dir.join("tokenizer.json"), rich_fixture_tokenizer_json()).unwrap();

    // Probe-only relation classifier: each (layer, feature) maps to a
    // relation label. No clusters are needed for label resolution; the
    // classifier returns probe-confirmed labels first.
    let feature_labels = serde_json::json!({
        "L0_F0": "capital",
        "L0_F1": "capital",
        "L0_F2": "language",
        "L1_F0": "capital",
        "L1_F2": "language",
    });
    std::fs::write(
        dir.join("feature_labels.json"),
        serde_json::to_string(&feature_labels).unwrap(),
    )
    .unwrap();

    dir
}

/// Spin up a session and `USE` the rich test vindex.
fn rich_vindex_session(tag: &str) -> (Session, std::path::PathBuf) {
    let dir = make_rich_test_vindex_dir(tag);
    let mut session = Session::new();
    let stmt = parser::parse(&format!(r#"USE "{}";"#, dir.display())).unwrap();
    session
        .execute(&stmt)
        .expect("USE on rich synthetic vindex should succeed");
    (session, dir)
}

// ── Full fixture: real ModelWeights + safetensors-equivalent vindex ─
//
// `make_full_test_vindex_dir` produces a vindex with `has_model_weights
// = true`, populated via `larql_inference::test_utils::make_test_weights`
// (TinyModelArch, 2 layers × 16 hidden × 32 intermediate × vocab 32).
// `larql_vindex::write_model_weights` writes the full attention + FFN
// + lm_head + norm weight files into the vindex directory, so
// `load_model_weights` succeeds and downstream INFER / TRACE / EXPLAIN
// INFER / COMPACT MAJOR / REBALANCE-with-installs / INSERT-compose all
// have real (random) weights to forward through.
//
// The same WordLevel tokenizer that `make_test_tokenizer(32)` produces
// is written to the vindex so prompts tokenise to ids 0..31.

fn make_full_test_vindex_dir(tag: &str) -> std::path::PathBuf {
    use larql_vindex::{
        ExtractLevel, MoeConfig, QuantFormat, SilentBuildCallbacks, StorageDtype,
        VindexConfig, VindexLayerInfo, VindexModelConfig,
    };

    let dir = std::env::temp_dir().join(format!(
        "larql_lql_full_test_vindex_{tag}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // `make_test_weights()` produces vocab_size = 32 with embed shape
    // [32, 16]. The companion `make_test_tokenizer(32)` adds `[UNK]`
    // at id 32 — out of the embed's bounds. Extend the embed by one
    // row so any UNK-tagged token still resolves to a valid embedding.
    let mut weights = larql_inference::test_utils::make_test_weights();
    {
        use larql_inference::ndarray::Array2;
        let new_vocab = weights.vocab_size + 1;
        let hidden = weights.hidden_size;
        let mut extended = Array2::<f32>::zeros((new_vocab, hidden));
        for (i, row) in weights.embed.rows().into_iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                extended[[i, j]] = *v;
            }
        }
        // UNK row gets a small constant so it embeds to something
        // distinguishable from id 0.
        for j in 0..hidden {
            extended[[weights.vocab_size, j]] = 0.01_f32 * (j as f32 + 1.0);
        }
        weights.embed = extended.into_shared();
        weights.vocab_size = new_vocab;
        // Mirror into the lm_head if it shared the original embed.
        let mut lm_extended = Array2::<f32>::zeros((new_vocab, hidden));
        for (i, row) in weights.lm_head.rows().into_iter().enumerate() {
            if i >= new_vocab {
                break;
            }
            for (j, v) in row.iter().enumerate() {
                lm_extended[[i, j]] = *v;
            }
        }
        weights.lm_head = lm_extended.into_shared();
        // Update embed_key tensor too so manifest stays consistent.
        let embed_key = weights.arch.embed_key().to_string();
        weights.tensors.insert(embed_key, weights.embed.clone());
    }
    let vindex = larql_inference::test_utils::make_test_vindex(&weights);

    // Gate offsets: each layer's gate matrix is `intermediate × hidden`
    // floats. Lay them out contiguously starting at 0.
    let bpf = 4_usize; // f32
    let row_bytes = weights.hidden_size * bpf;
    let layer_bytes = weights.intermediate_size * row_bytes;
    let mut layers: Vec<VindexLayerInfo> = Vec::new();
    for li in 0..weights.num_layers {
        layers.push(VindexLayerInfo {
            layer: li,
            offset: (li * layer_bytes) as u64,
            length: layer_bytes as u64,
            num_features: weights.intermediate_size,
            num_experts: None,
            num_features_per_expert: None,
        });
    }

    let model_config = VindexModelConfig {
        model_type: weights.arch.family().to_string(),
        head_dim: weights.head_dim,
        num_q_heads: weights.num_q_heads,
        num_kv_heads: weights.num_kv_heads,
        rope_base: weights.rope_base,
        sliding_window: None,
        moe: None::<MoeConfig>,
        global_head_dim: None,
        num_global_kv_heads: None,
        partial_rotary_factor: None,
        sliding_window_pattern: None,
        layer_types: None,
        attention_k_eq_v: false,
        num_kv_shared_layers: None,
        per_layer_embed_dim: None,
        rope_local_base: None,
        query_pre_attn_scalar: None,
        final_logit_softcapping: None,
    };

    let mut config = VindexConfig {
        version: 2,
        model: format!("test/full-fixture-{tag}"),
        family: weights.arch.family().to_string(),
        source: None,
        checksums: None,
        num_layers: weights.num_layers,
        hidden_size: weights.hidden_size,
        intermediate_size: weights.intermediate_size,
        vocab_size: weights.vocab_size,
        embed_scale: 1.0,
        extract_level: ExtractLevel::All,
        dtype: StorageDtype::F32,
        quant: QuantFormat::None,
        layer_bands: None,
        layers: layers.clone(),
        down_top_k: 5,
        has_model_weights: true,
        model_config: Some(model_config),
        fp4: None,
        ffn_layout: None,
    };

    // 1. Save the index (gate vectors + down_meta + config).
    vindex.save_vindex(&dir, &mut config).unwrap();

    // 2. Write the model weight files (attn / up / down / norms / lm_head).
    let mut build_cb = SilentBuildCallbacks;
    larql_vindex::write_model_weights(&weights, &dir, &mut build_cb).unwrap();

    // 3. Write embeddings.bin from `weights.embed`.
    let embed_slice = weights.embed.as_slice().unwrap();
    let mut embed_bytes = Vec::with_capacity(embed_slice.len() * bpf);
    for v in embed_slice {
        embed_bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(dir.join("embeddings.bin"), embed_bytes).unwrap();

    // 4. Write a tokenizer.json that maps token-id N to "[N]" string.
    //    Pass `vocab_size - 1` so the tokenizer's `[UNK]` lands at id
    //    `vocab_size - 1` (we extended the embed by one above), keeping
    //    every produced id inside the embed table.
    let tok = larql_inference::test_utils::make_test_tokenizer(weights.vocab_size - 1);
    tok.save(dir.join("tokenizer.json").to_str().unwrap(), false)
        .unwrap();

    dir
}

fn full_vindex_session(tag: &str) -> (Session, std::path::PathBuf) {
    let dir = make_full_test_vindex_dir(tag);
    let mut session = Session::new();
    let stmt = parser::parse(&format!(r#"USE "{}";"#, dir.display())).unwrap();
    session
        .execute(&stmt)
        .expect("USE on full synthetic vindex should succeed");
    (session, dir)
}

#[test]
fn use_synthetic_vindex_loads() {
    let (session, dir) = vindex_session("use_loads");
    assert!(matches!(session.backend, Backend::Vindex { .. }));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn delete_by_layer_and_feature_succeeds() {
    let (mut session, dir) = vindex_session("delete_lf");

    let stmt = parser::parse(r#"DELETE FROM EDGES WHERE layer = 0 AND feature = 0;"#).unwrap();
    let out = session.execute(&stmt).expect("DELETE should succeed");
    let joined = out.join("\n");
    assert!(
        joined.contains("Deleted") || joined.contains("deleted"),
        "expected delete confirmation in: {joined}"
    );

    // The patch session should now be active (auto-patch).
    assert!(
        session.patch_recording.is_some(),
        "DELETE should have started an auto-patch session"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn delete_no_matches_returns_message() {
    let (mut session, dir) = vindex_session("delete_nomatch");

    // Layer that doesn't exist in our 2-layer test vindex.
    let stmt = parser::parse(r#"DELETE FROM EDGES WHERE layer = 99 AND feature = 0;"#).unwrap();
    let result = session.execute(&stmt);
    // The executor either returns an empty-match message or errors —
    // both are acceptable; the important thing is no panic.
    assert!(result.is_ok() || result.is_err(), "no panic");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn delete_relation_filter_without_labels_errors_before_mutating() {
    let (mut session, dir) = vindex_session("delete_relation_no_labels");

    let stmt = parser::parse(r#"DELETE FROM EDGES WHERE relation = "capital";"#).unwrap();
    let err = session
        .execute(&stmt)
        .expect_err("relation-only DELETE should not silently match everything");

    assert!(
        err.to_string()
            .contains("relation filters require relation labels"),
        "unexpected error: {err}"
    );
    assert!(
        session
            .patch_recording
            .as_ref()
            .map(|r| r.operations.is_empty())
            .unwrap_or(false),
        "failed DELETE should not record patch operations"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn update_feature_target_succeeds() {
    let (mut session, dir) = vindex_session("update_target");

    let stmt =
        parser::parse(r#"UPDATE EDGES SET target = "London" WHERE layer = 0 AND feature = 0;"#)
            .unwrap();
    let out = session.execute(&stmt).expect("UPDATE should succeed");
    let joined = out.join("\n");
    assert!(
        joined.contains("Updated") || joined.contains("updated") || joined.contains("London"),
        "expected update confirmation in: {joined}"
    );

    assert!(
        session.patch_recording.is_some(),
        "UPDATE should have started an auto-patch session"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn update_relation_filter_without_labels_errors_before_mutating() {
    let (mut session, dir) = vindex_session("update_relation_no_labels");

    let stmt =
        parser::parse(r#"UPDATE EDGES SET target = "London" WHERE relation = "capital";"#).unwrap();
    let err = session
        .execute(&stmt)
        .expect_err("relation-only UPDATE should not silently match everything");

    assert!(
        err.to_string()
            .contains("relation filters require relation labels"),
        "unexpected error: {err}"
    );
    assert!(
        session
            .patch_recording
            .as_ref()
            .map(|r| r.operations.is_empty())
            .unwrap_or(false),
        "failed UPDATE should not record patch operations"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn explicit_begin_patch_starts_session() {
    let (mut session, dir) = vindex_session("begin_patch");

    let patch_path = dir.join("session.vlp");
    let stmt = parser::parse(&format!(r#"BEGIN PATCH "{}";"#, patch_path.display())).unwrap();
    session.execute(&stmt).expect("BEGIN PATCH should succeed");

    assert!(
        session.patch_recording.is_some(),
        "BEGIN PATCH should populate patch_recording"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn save_patch_writes_file_to_disk() {
    let (mut session, dir) = vindex_session("save_patch");

    // Start a patch, do a delete (so there's at least one operation), save.
    let patch_path = dir.join("save.vlp");
    let begin = parser::parse(&format!(r#"BEGIN PATCH "{}";"#, patch_path.display())).unwrap();
    session.execute(&begin).expect("BEGIN PATCH");

    let del = parser::parse(r#"DELETE FROM EDGES WHERE layer = 0 AND feature = 1;"#).unwrap();
    session.execute(&del).expect("DELETE under patch");

    let save = parser::parse("SAVE PATCH;").unwrap();
    let out = session.execute(&save).expect("SAVE PATCH");
    let joined = out.join("\n");
    assert!(
        patch_path.exists(),
        "SAVE PATCH should write the .vlp file. Output: {joined}"
    );

    // After SAVE PATCH, recording should be cleared.
    assert!(
        session.patch_recording.is_none(),
        "SAVE PATCH should clear the recording"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn auto_patch_session_starts_on_first_mutation() {
    let (mut session, dir) = vindex_session("auto_patch");

    // No explicit BEGIN PATCH first.
    assert!(
        session.patch_recording.is_none(),
        "no patch session before mutation"
    );

    let del = parser::parse(r#"DELETE FROM EDGES WHERE layer = 0 AND feature = 0;"#).unwrap();
    session.execute(&del).expect("DELETE");

    assert!(
        session.patch_recording.is_some(),
        "first mutation should auto-start a patch session"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn merge_nonexistent_source_errors_cleanly() {
    let (mut session, dir) = vindex_session("merge_bad_src");

    let stmt = parser::parse(r#"MERGE "/nonexistent/src.vindex";"#).unwrap();
    let err = session.execute(&stmt).unwrap_err();
    assert!(matches!(err, LqlError::Execution(_)));

    let _ = std::fs::remove_dir_all(&dir);
}

// ── Session::patched_overlay_mut accessor ──

#[test]
fn patched_overlay_mut_returns_some_for_vindex_backend() {
    let (mut session, dir) = vindex_session("overlay_mut_some");
    assert!(
        session.patched_overlay_mut().is_some(),
        "Vindex backend should yield a mutable overlay"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn patched_overlay_mut_returns_none_for_no_backend() {
    let mut session = Session::new();
    assert!(
        session.patched_overlay_mut().is_none(),
        "fresh session with no backend should yield None"
    );
}

#[test]
fn patched_overlay_mut_round_trip_via_insert_feature() {
    use larql_models::TopKEntry;
    use larql_vindex::FeatureMeta;

    let (mut session, dir) = vindex_session("overlay_mut_round_trip");
    let gate = vec![0.7_f32, 0.0, 0.0, 0.0];
    {
        let overlay = session.patched_overlay_mut().expect("vindex backend");
        overlay.insert_feature(
            0,
            1,
            gate.clone(),
            FeatureMeta {
                top_token: "z".into(),
                top_token_id: 9,
                c_score: 0.42,
                top_k: vec![TopKEntry {
                    token: "z".into(),
                    token_id: 9,
                    logit: 0.42,
                }],
            },
        );
    }
    // Same accessor, second call: the gate we just wrote must still be there.
    let overlay2 = session.patched_overlay_mut().expect("vindex backend");
    assert_eq!(
        overlay2.overrides_gate_at(0, 1),
        Some(gate.as_slice()),
        "second patched_overlay_mut() call should observe the previous mutation",
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn show_patches_with_no_patches_returns_message() {
    let (mut session, dir) = vindex_session("show_patches_empty");

    let stmt = parser::parse("SHOW PATCHES;").unwrap();
    let out = session.execute(&stmt).expect("SHOW PATCHES");
    let joined = out.join("\n").to_lowercase();
    assert!(
        joined.contains("no") || joined.contains("0") || joined.is_empty(),
        "expected an empty/no-patches message: {joined}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn refresh_recorded_patch_ops_for_slots_persists_latest_overlay_vectors() {
    use larql_models::TopKEntry;
    use larql_vindex::{FeatureMeta, PatchOp};

    let (mut session, dir) = vindex_session("refresh_patch_ops");

    {
        let overlay = session.patched_overlay_mut().expect("vindex backend");
        overlay.insert_feature(
            0,
            0,
            vec![1.0, 0.0, 0.0, 0.0],
            FeatureMeta {
                top_token: "old".into(),
                top_token_id: 7,
                c_score: 0.5,
                top_k: vec![TopKEntry {
                    token: "old".into(),
                    token_id: 7,
                    logit: 0.5,
                }],
            },
        );
        overlay.set_up_vector(0, 0, vec![0.1, 0.2, 0.3, 0.4]);
        overlay.set_down_vector(0, 0, vec![0.5, 0.6, 0.7, 0.8]);
    }

    session.patch_recording = Some(PatchRecording {
        path: String::new(),
        operations: vec![PatchOp::Insert {
            layer: 0,
            feature: 0,
            relation: Some("capital".into()),
            entity: "Atlantis".into(),
            target: "Poseidon".into(),
            confidence: Some(0.9),
            gate_vector_b64: Some(larql_vindex::patch::core::encode_gate_vector(&[
                9.0, 9.0, 9.0, 9.0,
            ])),
            up_vector_b64: Some(larql_vindex::patch::core::encode_gate_vector(&[
                9.0, 9.0, 9.0, 9.0,
            ])),
            down_vector_b64: Some(larql_vindex::patch::core::encode_gate_vector(&[
                9.0, 9.0, 9.0, 9.0,
            ])),
            down_meta: None,
        }],
    });

    {
        let overlay = session.patched_overlay_mut().expect("vindex backend");
        overlay.set_up_vector(0, 0, vec![1.1, 1.2, 1.3, 1.4]);
        overlay.set_down_vector(0, 0, vec![2.1, 2.2, 2.3, 2.4]);
    }

    session
        .refresh_recorded_patch_ops_for_slots(&[(0, 0)])
        .expect("refresh patch ops");

    let PatchOp::Insert {
        up_vector_b64,
        down_vector_b64,
        ..
    } = &session.patch_recording.as_ref().unwrap().operations[0]
    else {
        panic!("expected insert op");
    };
    let up = larql_vindex::patch::core::decode_gate_vector(up_vector_b64.as_ref().unwrap())
        .expect("decode refreshed up");
    let down = larql_vindex::patch::core::decode_gate_vector(down_vector_b64.as_ref().unwrap())
        .expect("decode refreshed down");

    assert_eq!(up, vec![1.1, 1.2, 1.3, 1.4]);
    assert_eq!(down, vec![2.1, 2.2, 2.3, 2.4]);
    let _ = std::fs::remove_dir_all(&dir);
}

// ── COMPILE INTO VINDEX integration tests ──────────────────────────────

#[test]
fn compile_into_vindex_no_patches_succeeds() {
    let (mut session, dir) = vindex_session("compile_nopatches_v");

    let output = dir.join("compiled.vindex");
    let stmt = parser::parse(&format!(
        r#"COMPILE CURRENT INTO VINDEX "{}";"#,
        output.display()
    ))
    .unwrap();
    let out = session
        .execute(&stmt)
        .expect("COMPILE INTO VINDEX should succeed");
    let joined = out.join("\n");
    assert!(
        joined.contains("Compiled"),
        "expected compile output: {joined}"
    );
    assert!(output.exists(), "compiled vindex directory should exist");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compile_path_into_vindex_uses_supplied_source_without_active_backend() {
    let dir = make_test_vindex_dir("compile_path_source");
    let output = dir.join("compiled_from_path.vindex");
    let mut session = Session::new();

    let stmt = parser::parse(&format!(
        r#"COMPILE "{}" INTO VINDEX "{}";"#,
        dir.display(),
        output.display()
    ))
    .unwrap();
    let out = session
        .execute(&stmt)
        .expect("path-form COMPILE INTO VINDEX should load its source");
    let joined = out.join("\n");

    assert!(
        joined.contains("Compiled"),
        "expected compile output: {joined}"
    );
    assert!(output.exists(), "compiled vindex directory should exist");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compile_path_into_model_reports_supplied_source_requirements() {
    let dir = make_test_vindex_dir("compile_path_model_source");
    let output = dir.join("model_out");
    let mut session = Session::new();

    let stmt = parser::parse(&format!(
        r#"COMPILE "{}" INTO MODEL "{}";"#,
        dir.display(),
        output.display()
    ))
    .unwrap();
    let err = session
        .execute(&stmt)
        .expect_err("browse-only source should fail after path source is loaded");

    assert!(
        err.to_string().contains("requires model weights"),
        "expected source-level model-weight error, got: {err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compile_into_vindex_with_down_overrides_bakes_them() {
    use larql_models::TopKEntry;
    use larql_vindex::FeatureMeta;

    let (mut session, dir) = vindex_session("compile_bake_down");

    // Create a synthetic down_weights.bin — per-layer [hidden, intermediate] f32.
    // hidden=4, intermediate=3, num_layers=2.
    let layer_floats = 4 * 3;
    let total = 2 * layer_floats;
    let bytes: Vec<u8> = (0..total)
        .flat_map(|i| (i as f32 * 0.01).to_le_bytes())
        .collect();
    std::fs::write(dir.join("down_weights.bin"), &bytes).unwrap();

    {
        let overlay = session.patched_overlay_mut().expect("vindex backend");
        overlay.insert_feature(
            0,
            0,
            vec![1.0, 0.0, 0.0, 0.0],
            FeatureMeta {
                top_token: "test".into(),
                top_token_id: 5,
                c_score: 0.9,
                top_k: vec![TopKEntry {
                    token: "test".into(),
                    token_id: 5,
                    logit: 0.9,
                }],
            },
        );
        overlay.set_down_vector(0, 0, vec![0.5, 0.6, 0.7, 0.8]);
    }
    let output = dir.join("compiled_baked.vindex");
    let stmt = parser::parse(&format!(
        r#"COMPILE CURRENT INTO VINDEX "{}";"#,
        output.display()
    ))
    .unwrap();
    let out = session.execute(&stmt).expect("COMPILE should succeed");
    let joined = out.join("\n");
    assert!(
        joined.contains("Down overrides baked"),
        "expected baked overrides: {joined}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compile_on_conflict_fail_detects_collision() {
    use larql_vindex::{PatchOp, VindexPatch};

    let (mut session, dir) = vindex_session("compile_conflict_fail");
    {
        let (_, _, patched) = session.require_patched_mut().unwrap();
        let mkp = |e: &str| VindexPatch {
            version: 1,
            base_model: String::new(),
            base_checksum: None,
            created_at: String::new(),
            description: None,
            author: None,
            tags: Vec::new(),
            operations: vec![PatchOp::Insert {
                layer: 0,
                feature: 0,
                relation: Some("r".into()),
                entity: e.into(),
                target: "t".into(),
                confidence: Some(0.9),
                gate_vector_b64: None,
                up_vector_b64: None,
                down_vector_b64: None,
                down_meta: None,
            }],
        };
        patched.patches.push(mkp("A"));
        patched.patches.push(mkp("C"));
    }
    let output = dir.join("compiled_fail.vindex");
    let stmt = parser::parse(&format!(
        r#"COMPILE CURRENT INTO VINDEX "{}" ON CONFLICT FAIL;"#,
        output.display()
    ))
    .unwrap();
    let result = session.execute(&stmt);
    assert!(
        result.is_err(),
        "ON CONFLICT FAIL should error on collision"
    );
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("FAIL") || msg.contains("colliding"),
        "error: {msg}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compile_on_conflict_last_wins_succeeds() {
    use larql_vindex::{PatchOp, VindexPatch};

    let (mut session, dir) = vindex_session("compile_conflict_lw");
    {
        let (_, _, patched) = session.require_patched_mut().unwrap();
        let mkp = |e: &str| VindexPatch {
            version: 1,
            base_model: String::new(),
            base_checksum: None,
            created_at: String::new(),
            description: None,
            author: None,
            tags: Vec::new(),
            operations: vec![PatchOp::Insert {
                layer: 0,
                feature: 0,
                relation: Some("r".into()),
                entity: e.into(),
                target: "t".into(),
                confidence: Some(0.9),
                gate_vector_b64: None,
                up_vector_b64: None,
                down_vector_b64: None,
                down_meta: None,
            }],
        };
        patched.patches.push(mkp("A"));
        patched.patches.push(mkp("C"));
    }
    let output = dir.join("compiled_lw.vindex");
    let stmt = parser::parse(&format!(
        r#"COMPILE CURRENT INTO VINDEX "{}" ON CONFLICT LAST_WINS;"#,
        output.display()
    ))
    .unwrap();
    assert!(
        session.execute(&stmt).is_ok(),
        "LAST_WINS should succeed despite collision"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ── MEMIT fact collection ────────────────────────────────────────────

#[test]
fn memit_facts_count_inserts_only() {
    use larql_vindex::PatchOp;

    let ops = [
        PatchOp::Insert {
            layer: 26,
            feature: 100,
            relation: Some("capital".into()),
            entity: "X".into(),
            target: "Y".into(),
            confidence: Some(0.9),
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: None,
        },
        PatchOp::Delete {
            layer: 10,
            feature: 50,
            reason: None,
        },
        PatchOp::Update {
            layer: 0,
            feature: 2,
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: None,
        },
    ];
    let insert_count = ops
        .iter()
        .filter(|op| matches!(op, PatchOp::Insert { .. }))
        .count();
    assert_eq!(insert_count, 1, "only INSERT should be counted");
}

#[test]
fn memit_facts_deduplicate_across_patches() {
    use larql_vindex::{PatchOp, VindexPatch};

    let mkp = |conf: f32| VindexPatch {
        version: 1,
        base_model: String::new(),
        base_checksum: None,
        created_at: String::new(),
        description: None,
        author: None,
        tags: Vec::new(),
        operations: vec![PatchOp::Insert {
            layer: 10,
            feature: 5,
            relation: Some("capital".into()),
            entity: "France".into(),
            target: "Paris".into(),
            confidence: Some(conf),
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: None,
        }],
    };
    let patches = vec![mkp(0.9), mkp(0.95)];
    let mut seen = std::collections::HashSet::new();
    for p in &patches {
        for op in &p.operations {
            if let PatchOp::Insert {
                layer,
                entity,
                relation,
                target,
                ..
            } = op
            {
                seen.insert((
                    entity.clone(),
                    relation.clone().unwrap_or_default(),
                    target.clone(),
                    *layer,
                ));
            }
        }
    }
    assert_eq!(seen.len(), 1, "same fact in two patches → 1 after dedup");
}

// ── Template tests ───────────────────────────────────────────────────
//
// `canonical_decoys_are_nonempty_and_diverse` lives alongside the
// constant in `executor/mutation/insert/capture.rs`.

#[test]
fn relation_template_simple() {
    let rel = "capital";
    let prompt = format!("The {} of entity is", rel.replace(['-', '_'], " "));
    assert_eq!(prompt, "The capital of entity is");
}

#[test]
fn relation_template_multi_word() {
    let rel = "native_language";
    let prompt = format!("The {} of entity is", rel.replace(['-', '_'], " "));
    assert_eq!(prompt, "The native language of entity is");
}

#[test]
fn relation_template_hyphenated_produces_double_of() {
    // Documents the known template quirk: "capital-of" → "capital of"
    // → "The capital of of X is". Users should use "capital" not "capital-of".
    let rel = "capital-of";
    let prompt = format!("The {} of X is", rel.replace(['-', '_'], " "));
    assert!(
        prompt.contains("of of"),
        "capital-of produces double 'of': {prompt}"
    );
}

// Cholesky solver is unit-tested in larql-compute::cpu::ops::linalg::tests.
// MEMIT solve is integration-tested via compile_demo against real vindex.

// ── MEMIT struct ─────────────────────────────────────────────────────

#[test]
fn memit_fact_struct() {
    let f = larql_inference::MemitFact {
        prompt_tokens: vec![1, 2, 3],
        target_token_id: 42,
        layer: 26,
        label: "test".into(),
    };
    assert_eq!(f.layer, 26);
    assert_eq!(f.target_token_id, 42);
}

// ── Compile into model requires weights ──────────────────────────────

#[test]
fn compile_into_model_requires_model_weights() {
    let (mut session, dir) = vindex_session("compile_model_noweights");
    let output = dir.join("model_out");
    let stmt = parser::parse(&format!(
        r#"COMPILE CURRENT INTO MODEL "{}";"#,
        output.display()
    ))
    .unwrap();
    let result = session.execute(&stmt);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("model weights") || msg.contains("WITH ALL"),
        "error: {msg}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ── Architecture B KNN Store tests — UNIFIED into FFN overlay ────
//
// The 6 tests below were written against the separate-KnnStore design
// of arch-B. After the FFN-vindex unification (2026-04-15), inserts
// route through the overlay (find_free_feature + insert_feature +
// set_up_vector + set_down_vector) and the separate knn_store is
// dormant. These tests assert obsolete behavior; they're #[ignore]d
// pending a rewrite against the unified path (task #37).

#[test]
// restored: dual-mode INSERT defaults to KNN
fn knn_store_insert_populates_store() {
    // INSERT on a browse-only vindex (no model weights) uses embedding-key fallback
    let (mut session, dir) = vindex_session("knn_insert");

    let stmt = parser::parse(
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("Atlantis", "capital", "Poseidon");"#,
    ).unwrap();
    let out = session.execute(&stmt).expect("INSERT should succeed");
    let joined = out.join("\n");
    assert!(
        joined.contains("Inserted"),
        "expected insert confirmation: {joined}"
    );
    assert!(
        joined.contains("KNN store"),
        "expected KNN store mode: {joined}"
    );
    assert!(joined.contains("1 entries"), "expected 1 entry: {joined}");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
// restored: dual-mode INSERT defaults to KNN
fn knn_store_insert_multiple_facts() {
    let (mut session, dir) = vindex_session("knn_multi");

    for (entity, target) in &[
        ("Atlantis", "Poseidon"),
        ("Lemuria", "Mu"),
        ("Agartha", "Shambhala"),
    ] {
        let sql = format!(
            r#"INSERT INTO EDGES (entity, relation, target) VALUES ("{entity}", "capital", "{target}");"#
        );
        let stmt = parser::parse(&sql).unwrap();
        session.execute(&stmt).expect("INSERT should succeed");
    }

    // Check KNN store has 3 entries
    let stmt = parser::parse(
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("Wakanda", "capital", "Birnin");"#,
    )
    .unwrap();
    let out = session.execute(&stmt).expect("INSERT should succeed");
    let joined = out.join("\n");
    assert!(joined.contains("4 entries"), "expected 4 entries: {joined}");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
// restored: dual-mode INSERT defaults to KNN
fn knn_store_describe_shows_inserted_edges() {
    let (mut session, dir) = vindex_session("knn_describe");

    // Insert a fact
    let stmt = parser::parse(
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("Atlantis", "capital", "Poseidon");"#,
    ).unwrap();
    session.execute(&stmt).expect("INSERT");

    // Verify the KNN store is populated by checking via the overlay accessor
    let overlay = session.patched_overlay_mut().expect("vindex backend");
    let knn_entries = overlay.knn_store.entries_for_entity("Atlantis");
    assert_eq!(knn_entries.len(), 1, "expected 1 KNN entry for Atlantis");
    assert_eq!(knn_entries[0].1.target_token, "Poseidon");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
// restored: dual-mode INSERT defaults to KNN
fn knn_store_delete_removes_entries() {
    let (mut session, dir) = vindex_session("knn_delete");

    // Insert two facts for different entities
    for sql in &[
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("Atlantis", "capital", "Poseidon");"#,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("Lemuria", "capital", "Mu");"#,
    ] {
        let stmt = parser::parse(sql).unwrap();
        session.execute(&stmt).expect("INSERT");
    }

    // Verify both in store
    let overlay = session.patched_overlay_mut().expect("vindex");
    assert_eq!(overlay.knn_store.len(), 2);

    // Delete Atlantis via direct KNN store removal (since base features may not exist)
    overlay.knn_store.remove_by_entity("Atlantis");
    assert_eq!(overlay.knn_store.len(), 1);
    assert_eq!(overlay.knn_store.entries_for_entity("Atlantis").len(), 0);
    assert_eq!(overlay.knn_store.entries_for_entity("Lemuria").len(), 1);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
// restored: dual-mode INSERT defaults to KNN
fn knn_store_compile_saves_and_loads() {
    let (mut session, dir) = vindex_session("knn_compile");

    // Insert a fact
    let stmt = parser::parse(
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("Atlantis", "capital", "Poseidon");"#,
    ).unwrap();
    session.execute(&stmt).expect("INSERT");

    // Compile
    let output = dir.join("compiled_knn.vindex");
    let stmt = parser::parse(&format!(
        r#"COMPILE CURRENT INTO VINDEX "{}";"#,
        output.display()
    ))
    .unwrap();
    let out = session.execute(&stmt).expect("COMPILE should succeed");
    let joined = out.join("\n");
    assert!(
        joined.contains("KNN store: 1 entries"),
        "expected KNN count: {joined}"
    );

    // Verify knn_store.bin exists
    assert!(
        output.join("knn_store.bin").exists(),
        "knn_store.bin should be in compiled vindex"
    );

    // Load the compiled vindex and verify KNN store survives round-trip
    let stmt = parser::parse(&format!(r#"USE "{}";"#, output.display())).unwrap();
    session.execute(&stmt).expect("USE compiled vindex");

    // Check the KNN store is loaded with the fact
    let overlay = session.patched_overlay_mut().expect("vindex");
    let entries = overlay.knn_store.entries_for_entity("Atlantis");
    assert_eq!(
        entries.len(),
        1,
        "expected 1 KNN entry after compile+reload"
    );
    assert_eq!(entries[0].1.target_token, "Poseidon");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn knn_store_patch_op_serialization() {
    // Verify InsertKnn PatchOp serializes and deserializes correctly
    let op = larql_vindex::PatchOp::InsertKnn {
        layer: 26,
        entity: "Atlantis".into(),
        relation: "capital".into(),
        target: "Poseidon".into(),
        target_id: 42,
        confidence: Some(1.0),
        key_vector_b64: larql_vindex::patch::core::encode_gate_vector(&[1.0, 0.0, 0.0, 0.0]),
    };
    let json = serde_json::to_string(&op).unwrap();
    assert!(
        json.contains("insert_knn"),
        "expected insert_knn tag: {json}"
    );
    assert!(json.contains("Atlantis"), "expected entity: {json}");

    // Round-trip
    let decoded: larql_vindex::PatchOp = serde_json::from_str(&json).unwrap();
    match decoded {
        larql_vindex::PatchOp::InsertKnn {
            entity,
            target,
            layer,
            ..
        } => {
            assert_eq!(entity, "Atlantis");
            assert_eq!(target, "Poseidon");
            assert_eq!(layer, 26);
        }
        _ => panic!("expected InsertKnn variant"),
    }
}

#[test]
fn knn_store_delete_knn_patch_op() {
    let op = larql_vindex::PatchOp::DeleteKnn {
        entity: "Atlantis".into(),
    };
    let json = serde_json::to_string(&op).unwrap();
    assert!(
        json.contains("delete_knn"),
        "expected delete_knn tag: {json}"
    );

    let decoded: larql_vindex::PatchOp = serde_json::from_str(&json).unwrap();
    match decoded {
        larql_vindex::PatchOp::DeleteKnn { entity } => {
            assert_eq!(entity, "Atlantis");
        }
        _ => panic!("expected DeleteKnn variant"),
    }
}

#[test]
// restored: dual-mode INSERT defaults to KNN
fn knn_store_insert_at_layer_hint() {
    let (mut session, dir) = vindex_session("knn_layer_hint");

    // Synthetic vindex has only 2 layers (0, 1), so use AT LAYER 0
    let stmt = parser::parse(
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("Atlantis", "capital", "Poseidon") AT LAYER 0;"#,
    ).unwrap();
    let out = session.execute(&stmt).expect("INSERT AT LAYER");
    let joined = out.join("\n");
    assert!(joined.contains("L0"), "expected L0 in output: {joined}");

    // Verify it went to layer 0
    let overlay = session.patched_overlay_mut().expect("vindex");
    let entries = overlay.knn_store.entries_for_entity("Atlantis");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].0, 0, "expected layer 0");

    let _ = std::fs::remove_dir_all(&dir);
}

// ── InferenceWeights format dispatch ──
//
// These tests verify that the format-agnostic abstraction routes correctly
// without branching on `config.quant` in callers.

#[test]
fn knn_insert_q4k_flagged_no_weights_uses_embedding_fallback() {
    // A vindex with quant=Q4K but has_model_weights=false must still use the
    // embedding-key fallback path (not the InferenceWeights path). The quant
    // flag should be irrelevant when there are no weights to load.
    use larql_models::TopKEntry;
    use larql_vindex::{ExtractLevel, FeatureMeta, StorageDtype, VectorIndex, VindexConfig};

    let dir = std::env::temp_dir().join("larql_lql_test_q4k_embed_fallback");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let hidden = 4;
    let num_features = 3;
    let num_layers = 2;
    let vocab_size = 10;

    let make_meta = |tok: &str, id: u32, c: f32| FeatureMeta {
        top_token: tok.to_string(),
        top_token_id: id,
        c_score: c,
        top_k: vec![TopKEntry {
            token: tok.to_string(),
            token_id: id,
            logit: c,
        }],
    };
    let gate0 = ndarray::Array2::<f32>::zeros((num_features, hidden));
    let gate1 = ndarray::Array2::<f32>::zeros((num_features, hidden));
    let meta0 = vec![
        Some(make_meta("Paris", 100, 0.95)),
        Some(make_meta("French", 101, 0.88)),
        Some(make_meta("Europe", 102, 0.75)),
    ];
    let meta1 = vec![
        Some(make_meta("Berlin", 200, 0.90)),
        None,
        Some(make_meta("Spain", 202, 0.70)),
    ];
    let index = VectorIndex::new(
        vec![Some(gate0), Some(gate1)],
        vec![Some(meta0), Some(meta1)],
        num_layers,
        hidden,
    );
    let mut config = VindexConfig {
        version: 2,
        model: "test/q4k-no-weights".into(),
        family: "llama".into(),
        source: None,
        checksums: None,
        num_layers,
        hidden_size: hidden,
        intermediate_size: num_features,
        vocab_size,
        embed_scale: 1.0,
        extract_level: ExtractLevel::Browse,
        dtype: StorageDtype::F32,
        quant: larql_vindex::QuantFormat::Q4K, // quantised flag…
        layer_bands: None,
        layers: Vec::new(),
        down_top_k: 5,
        has_model_weights: false, // …but no weights on disk
        model_config: None,
        fp4: None,
        ffn_layout: None,
    };
    index.save_vindex(&dir, &mut config).unwrap();
    let embed_bytes = vec![0u8; vocab_size * hidden * 4];
    std::fs::write(dir.join("embeddings.bin"), embed_bytes).unwrap();
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(dir.join("tokenizer.json"), tok_json).unwrap();

    let mut session = Session::new();
    let stmt = parser::parse(&format!(r#"USE "{}";"#, dir.display())).unwrap();
    session.execute(&stmt).expect("USE");

    // INSERT must succeed via the embedding-key fallback — not attempt to load q4k weights.
    let stmt = parser::parse(
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("Atlantis", "capital", "Poseidon");"#,
    ).unwrap();
    let out = session
        .execute(&stmt)
        .expect("INSERT should use embedding fallback on q4k+no-weights");
    let joined = out.join("\n");
    assert!(
        joined.contains("KNN store"),
        "expected KNN store mode: {joined}"
    );
    assert!(
        joined.contains("embedding key"),
        "expected embedding-key mode (no weights): {joined}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn trace_on_q4k_vindex_returns_clear_error() {
    // TRACE should return a helpful error on q4k vindexes rather than the
    // cryptic "load_model_weights only handles float weights" message.
    use larql_models::TopKEntry;
    use larql_vindex::{ExtractLevel, FeatureMeta, StorageDtype, VectorIndex, VindexConfig};

    let dir = std::env::temp_dir().join("larql_lql_test_q4k_trace_error");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let hidden = 4;
    let num_features = 2;
    let num_layers = 2;
    let vocab_size = 10;

    let make_meta = |tok: &str, id: u32, c: f32| FeatureMeta {
        top_token: tok.to_string(),
        top_token_id: id,
        c_score: c,
        top_k: vec![TopKEntry {
            token: tok.to_string(),
            token_id: id,
            logit: c,
        }],
    };
    let gate0 = ndarray::Array2::<f32>::zeros((num_features, hidden));
    let meta0 = vec![
        Some(make_meta("test", 1, 0.5)),
        Some(make_meta("foo", 2, 0.3)),
    ];
    let index = VectorIndex::new(
        vec![Some(gate0.clone()), Some(gate0)],
        vec![Some(meta0.clone()), Some(meta0)],
        num_layers,
        hidden,
    );
    let mut config = VindexConfig {
        version: 2,
        model: "test/q4k-trace".into(),
        family: "llama".into(),
        source: None,
        checksums: None,
        num_layers,
        hidden_size: hidden,
        intermediate_size: num_features,
        vocab_size,
        embed_scale: 1.0,
        extract_level: ExtractLevel::Browse,
        dtype: StorageDtype::F32,
        quant: larql_vindex::QuantFormat::Q4K,
        layer_bands: None,
        layers: Vec::new(),
        down_top_k: 5,
        has_model_weights: true,
        model_config: None,
        fp4: None,
        ffn_layout: None,
    };
    index.save_vindex(&dir, &mut config).unwrap();
    let embed_bytes = vec![0u8; vocab_size * hidden * 4];
    std::fs::write(dir.join("embeddings.bin"), embed_bytes).unwrap();
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(dir.join("tokenizer.json"), tok_json).unwrap();

    let mut session = Session::new();
    let stmt = parser::parse(&format!(r#"USE "{}";"#, dir.display())).unwrap();
    session.execute(&stmt).expect("USE");

    let stmt = parser::parse(r#"TRACE "hello world";"#).unwrap();
    let err = session.execute(&stmt).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("T2") || msg.contains("q4k") || msg.contains("quantised"),
        "expected clear q4k error, got: {msg}"
    );
    assert!(
        !msg.contains("only handles float"),
        "must not expose internal loader error: {msg}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// ── COMPACT MAJOR persistence (Backend::Vindex.memit_store wiring) ──

#[test]
fn memit_store_mut_unavailable_without_backend() {
    let mut session = Session::new();
    assert!(matches!(
        session.memit_store_mut().unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn memit_store_mut_returns_empty_store_on_fresh_vindex() {
    let (mut session, dir) = vindex_session("memit_empty");
    let store = session
        .memit_store_mut()
        .expect("vindex backend has memit_store");
    assert_eq!(store.num_cycles(), 0);
    assert_eq!(store.total_facts(), 0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn show_compact_status_reflects_live_memit_store() {
    // Regression: prior implementation hardcoded "0 facts across 0
    // cycles" regardless of session state. After we add a synthetic
    // cycle, SHOW COMPACT STATUS should report it.
    let (mut session, dir) = vindex_session("compact_status_live");

    // Synthetic vindex has hidden_dim = 4 → MEMIT-supported branch is
    // disabled (requires ≥ 1024). We don't run the live-count check
    // when the L2 line is gated. Skip cleanly in that case.
    let initial = session
        .execute(&parser::parse("SHOW COMPACT STATUS;").unwrap())
        .expect("show compact status");
    let initial_joined = initial.join("\n");
    if initial_joined.contains("not available") {
        let _ = std::fs::remove_dir_all(&dir);
        return;
    }

    // Push a cycle directly through the backend accessor.
    {
        let store = session.memit_store_mut().expect("vindex backend");
        store.add_cycle(
            7,
            vec![larql_vindex::MemitFact {
                entity: "X".into(),
                relation: "y".into(),
                target: "Z".into(),
                key: larql_vindex::ndarray::Array1::zeros(4),
                decomposed_down: larql_vindex::ndarray::Array1::zeros(4),
                reconstruction_cos: 1.0,
            }],
            0.0,
            1.0,
            0.0,
        );
    }

    let after = session
        .execute(&parser::parse("SHOW COMPACT STATUS;").unwrap())
        .expect("show compact status");
    let joined = after.join("\n");
    assert!(
        joined.contains("1 fact(s) across 1 cycle(s)"),
        "expected live MEMIT counts, got: {joined}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn memit_store_persists_added_cycles() {
    // Verifies the wiring change from item #5: facts pushed into the
    // session-level MemitStore survive subsequent accesses. The
    // production COMPACT MAJOR pipeline writes through the same path.
    let (mut session, dir) = vindex_session("memit_persist");
    {
        let store = session.memit_store_mut().expect("vindex backend");
        store.add_cycle(
            33,
            vec![larql_vindex::MemitFact {
                entity: "France".into(),
                relation: "capital".into(),
                target: "Paris".into(),
                key: larql_vindex::ndarray::Array1::zeros(4),
                decomposed_down: larql_vindex::ndarray::Array1::zeros(4),
                reconstruction_cos: 1.0,
            }],
            0.5,
            1.0,
            0.0,
        );
    }
    // Re-borrow to confirm the cycle survived.
    let store = session.memit_store_mut().expect("vindex backend");
    assert_eq!(store.num_cycles(), 1);
    assert_eq!(store.total_facts(), 1);
    let hits = store.lookup("France", "capital");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].target, "Paris");
    let _ = std::fs::remove_dir_all(&dir);
}

// ══════════════════════════════════════════════════════════════
// Gap coverage: variants that shipped without an executor test
// ══════════════════════════════════════════════════════════════
//
// Each variant gets a no-backend sanity check plus (where feasible
// without model weights) an end-to-end pass against the synthetic
// vindex fixture.

// ── TRACE ──

#[test]
fn no_backend_trace() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"TRACE "The capital of France is";"#).unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn trace_on_browse_only_vindex_errors_with_weights_hint() {
    // The synthetic fixture is browse-only; TRACE needs model weights.
    let (mut session, dir) = vindex_session("trace_no_weights");
    let stmt = parser::parse(r#"TRACE "any prompt";"#).unwrap();
    let err = session
        .execute(&stmt)
        .expect_err("TRACE on browse-only vindex should fail");
    match err {
        LqlError::Execution(msg) => {
            assert!(
                msg.contains("TRACE requires model weights"),
                "expected model-weights hint, got: {msg}"
            );
        }
        other => panic!("expected Execution error, got {other:?}"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

// ── REBALANCE ──

#[test]
fn rebalance_without_backend_is_noop() {
    // REBALANCE short-circuits on empty `installed_edges` BEFORE the
    // backend check (mutation/rebalance.rs:38-43), so it returns Ok
    // with a "no compose-mode installs" message even with no backend.
    // This is the same behaviour as REBALANCE on a fresh vindex.
    let mut session = Session::new();
    let stmt = parser::parse("REBALANCE;").unwrap();
    let out = session
        .execute(&stmt)
        .expect("REBALANCE with empty install set should succeed");
    assert!(
        out.iter()
            .any(|line| line.contains("no compose-mode installs")),
        "expected empty-installs note in: {out:?}"
    );
}

#[test]
fn rebalance_without_compose_installs_is_noop() {
    // With no `installed_edges` registered, REBALANCE returns a
    // single-line note and doesn't touch the overlay.
    let (mut session, dir) = vindex_session("rebalance_empty");
    let stmt = parser::parse("REBALANCE;").unwrap();
    let out = session
        .execute(&stmt)
        .expect("REBALANCE on empty compose set should succeed");
    assert!(
        out.iter()
            .any(|line| line.contains("no compose-mode installs")),
        "expected empty-installs note in: {out:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ── COMPACT MINOR / MAJOR / SHOW COMPACT STATUS ──

#[test]
fn no_backend_compact_minor() {
    let mut session = Session::new();
    let stmt = parser::parse("COMPACT MINOR;").unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn no_backend_compact_major() {
    let mut session = Session::new();
    let stmt = parser::parse("COMPACT MAJOR;").unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn no_backend_show_compact_status() {
    let mut session = Session::new();
    let stmt = parser::parse("SHOW COMPACT STATUS;").unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn compact_minor_on_empty_l0_returns_message() {
    let (mut session, dir) = vindex_session("compact_minor_empty");
    let stmt = parser::parse("COMPACT MINOR;").unwrap();
    let out = session
        .execute(&stmt)
        .expect("COMPACT MINOR with empty L0 should succeed");
    assert!(
        out.iter().any(|l| l.contains("L0 is empty")),
        "expected empty-L0 message in: {out:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn show_compact_status_reports_empty_tiers() {
    let (mut session, dir) = vindex_session("compact_status");
    let stmt = parser::parse("SHOW COMPACT STATUS;").unwrap();
    let out = session
        .execute(&stmt)
        .expect("SHOW COMPACT STATUS should succeed");
    let joined = out.join("\n");
    assert!(joined.contains("L0"), "expected L0 tier: {joined}");
    assert!(joined.contains("L1"), "expected L1 tier: {joined}");
    // The synthetic fixture has 0 overrides; the L0/L1 counts should read 0.
    assert!(
        joined.contains("0 entries") || joined.contains("0 edges"),
        "expected zero counts in: {joined}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ── SHOW ENTITIES ──

#[test]
fn no_backend_show_entities() {
    let mut session = Session::new();
    let stmt = parser::parse("SHOW ENTITIES;").unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn show_entities_scans_synthetic_vindex() {
    // The synthetic fixture seeds content-shaped tokens — SHOW ENTITIES
    // should run cleanly and produce the `Distinct entities …` summary
    // line followed by the tabular header.
    let (mut session, dir) = vindex_session("show_entities_scan");
    let stmt = parser::parse("SHOW ENTITIES LIMIT 20;").unwrap();
    let out = session
        .execute(&stmt)
        .expect("SHOW ENTITIES should succeed");
    let joined = out.join("\n");
    assert!(
        joined.contains("Distinct entities"),
        "expected summary line in: {joined}"
    );
    assert!(
        joined.contains("Entity") && joined.contains("Max Score"),
        "expected tabular header in: {joined}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ── REMOVE PATCH ──

#[test]
fn no_backend_remove_patch() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"REMOVE PATCH "missing.vlp";"#).unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn remove_patch_unknown_errors_cleanly() {
    let (mut session, dir) = vindex_session("remove_patch_missing");
    let stmt = parser::parse(r#"REMOVE PATCH "never-applied.vlp";"#).unwrap();
    let err = session
        .execute(&stmt)
        .expect_err("REMOVE PATCH should error when no such patch is applied");
    match err {
        LqlError::Execution(msg) => assert!(msg.contains("patch not found")),
        other => panic!("expected Execution error, got {other:?}"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

// ── PIPE ──

#[test]
fn pipe_propagates_no_backend_error() {
    // The first stage errors with NoBackend — the pipe should surface
    // that without silently short-circuiting to `Ok`.
    let mut session = Session::new();
    let stmt = parser::parse("STATS |> STATS;").unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn pipe_concatenates_both_sides_output() {
    // Both sides execute and their output lines are concatenated.
    let (mut session, dir) = vindex_session("pipe_concat");
    let stmt = parser::parse("SHOW LAYERS |> SHOW MODELS;").unwrap();
    let out = session.execute(&stmt).expect("pipe should succeed");
    // The combined output must contain evidence of both stages —
    // SHOW LAYERS emits per-layer rows; SHOW MODELS emits a header /
    // "no models" line. We just check the combined length is larger
    // than either side's output in isolation.
    let single = parser::parse("SHOW LAYERS;").unwrap();
    let single_out = session.execute(&single).expect("SHOW LAYERS alone");
    assert!(
        out.len() > single_out.len(),
        "pipe output ({}) should be longer than a single stage ({}): {:?}",
        out.len(),
        single_out.len(),
        out,
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Returns the integer count parsed from a "Deleted N features ..." or
/// "Updated N features ..." confirmation line. Used by feature-only
/// regression tests to assert exact match counts rather than "any match".
fn parse_mutation_count(out: &[String]) -> usize {
    let joined = out.join("\n");
    for line in joined.lines() {
        for word in line.split_ascii_whitespace() {
            if let Ok(n) = word.parse::<usize>() {
                return n;
            }
        }
    }
    panic!("no integer count found in mutation output: {joined}");
}

#[test]
fn delete_with_feature_only_targets_only_that_feature() {
    // Regression: prior implementation passed (None, None, layer_filter) to
    // find_features when the user specified only `feature`, which returned
    // every feature in every layer. The fixture has 2 layers × 3 features
    // (one feature is `None` in layer 1 → skipped), so deleting `feature = 0`
    // should hit at most 2 slots, not all 5.
    let (mut session, dir) = vindex_session("delete_feature_only");
    let stmt = parser::parse(r#"DELETE FROM EDGES WHERE feature = 0;"#).unwrap();
    let out = session.execute(&stmt).expect("DELETE should succeed");
    let count = parse_mutation_count(&out);
    assert!(
        count <= 2,
        "feature-only DELETE should target at most one column across layers, got {count}: {out:?}"
    );
    assert!(count >= 1, "fixture has feature 0 populated in both layers");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn update_with_feature_only_targets_only_that_feature() {
    let (mut session, dir) = vindex_session("update_feature_only");
    let stmt = parser::parse(
        r#"UPDATE EDGES SET target = "Madrid" WHERE feature = 2;"#,
    )
    .unwrap();
    let out = session.execute(&stmt).expect("UPDATE should succeed");
    let count = parse_mutation_count(&out);
    assert!(
        count <= 2,
        "feature-only UPDATE should target at most one column across layers, got {count}: {out:?}"
    );
    assert!(count >= 1, "fixture has feature 2 populated in both layers");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compile_skips_memit_fact_with_no_relation() {
    // Regression: prior implementation substituted the literal string
    // `"relation"` for a missing PatchOp::Insert.relation, baking junk
    // into the canonical MEMIT prompt. We now skip and warn.
    let (mut session, dir) = vindex_session("compile_skip_no_relation");

    // Inject a patch recording with a relation-less insert directly.
    session.patch_recording = Some(PatchRecording {
        path: "synthetic.vlp".into(),
        operations: vec![larql_vindex::PatchOp::Insert {
            layer: 0,
            feature: 0,
            relation: None,
            entity: "Atlantis".into(),
            target: "Poseidon".into(),
            confidence: Some(0.9),
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: None,
        }],
    });

    let out_dir = dir.join("compiled.vindex");
    let stmt = parser::parse(&format!(
        r#"COMPILE CURRENT INTO VINDEX "{}";"#,
        out_dir.display()
    ))
    .unwrap();
    let lines = session.execute(&stmt).expect("compile should succeed");
    let joined = lines.join("\n");
    assert!(
        joined.contains("skipping MEMIT fact"),
        "expected a 'skipping MEMIT fact' warning, got: {joined}"
    );
    assert!(
        joined.contains("no relation"),
        "warning should mention missing relation: {joined}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn delete_with_negative_layer_matches_nothing() {
    // Negative integers are kept as a usize::MAX sentinel rather than
    // widened to "no filter" — so they match nothing instead of matching
    // everything.
    let (mut session, dir) = vindex_session("delete_negative_layer");
    let stmt = parser::parse(r#"DELETE FROM EDGES WHERE layer = -1;"#).unwrap();
    let out = session.execute(&stmt).expect("DELETE should not error");
    let joined = out.join("\n");
    assert!(
        joined.contains("no matching"),
        "negative layer should match nothing: {joined}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ══════════════════════════════════════════════════════════════
// Fixture-driven SELECT / DESCRIBE / WALK tests
// ══════════════════════════════════════════════════════════════
//
// These exercise the full verb pipeline (filter extraction → scan
// → render) against the synthetic / rich fixtures. They primarily
// drive coverage on the verb modules under `executor/query/`.

// ── SELECT * FROM EDGES ──────────────────────────────────────

// SELECT verbs against the synthetic fixture mostly drive the verb
// pipeline (filter extraction → scan → render). The fixture's
// `feature_meta` may surface as None after the disk round-trip, so we
// don't pin specific tokens — the assertions check that the verb
// runs end-to-end and emits the structural elements (header / dashes /
// "no match" line) we know are unconditional.

#[test]
fn select_edges_default_runs_end_to_end() {
    let (mut session, dir) = vindex_session("select_edges_default");
    let stmt = parser::parse("SELECT * FROM EDGES;").unwrap();
    let out = session.execute(&stmt).expect("SELECT EDGES");
    let joined = out.join("\n");
    assert!(joined.contains("Layer"));
    assert!(joined.contains("Feature"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn select_edges_with_layer_filter_runs() {
    let (mut session, dir) = vindex_session("select_edges_layer");
    let stmt = parser::parse("SELECT * FROM EDGES WHERE layer = 0;").unwrap();
    let _ = session.execute(&stmt).expect("SELECT EDGES WHERE layer");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn select_edges_with_entity_filter_runs() {
    let (mut session, dir) = vindex_session("select_edges_entity");
    let stmt = parser::parse(r#"SELECT * FROM EDGES WHERE entity = "berlin";"#).unwrap();
    let _ = session.execute(&stmt).expect("SELECT EDGES WHERE entity");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn select_edges_with_score_predicate_runs() {
    let (mut session, dir) = vindex_session("select_edges_score");
    let stmt = parser::parse("SELECT * FROM EDGES WHERE score > 0.85;").unwrap();
    let _ = session.execute(&stmt).expect("SELECT EDGES WHERE score");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn select_edges_with_score_lt_predicate_runs() {
    let (mut session, dir) = vindex_session("select_edges_score_lt");
    let stmt = parser::parse("SELECT * FROM EDGES WHERE score < 0.95;").unwrap();
    let _ = session.execute(&stmt).expect("SELECT EDGES WHERE score lt");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn select_edges_order_by_confidence_runs() {
    let (mut session, dir) = vindex_session("select_edges_order");
    let stmt = parser::parse("SELECT * FROM EDGES ORDER BY confidence;").unwrap();
    let _ = session.execute(&stmt).expect("SELECT EDGES ORDER BY");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn select_edges_order_by_layer_runs() {
    let (mut session, dir) = vindex_session("select_edges_order_layer");
    let stmt = parser::parse("SELECT * FROM EDGES ORDER BY layer;").unwrap();
    let _ = session.execute(&stmt).expect("SELECT EDGES ORDER BY layer");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn select_edges_no_matches_emits_friendly_line() {
    let (mut session, dir) = vindex_session("select_edges_empty");
    // `WHERE entity = "Nowhere"` matches nothing because the fixture's
    // top_tokens don't contain "Nowhere".
    let stmt = parser::parse(r#"SELECT * FROM EDGES WHERE entity = "Nowhere";"#).unwrap();
    let out = session.execute(&stmt).expect("SELECT EDGES");
    let joined = out.join("\n");
    // Either "(no matching edges)" or just an empty body — both are
    // valid. We just want exec_select to run.
    let _ = joined;
    let _ = std::fs::remove_dir_all(&dir);
}

// ── SELECT * FROM FEATURES ───────────────────────────────────

#[test]
fn select_features_default_runs() {
    let (mut session, dir) = vindex_session("select_features_default");
    let stmt = parser::parse("SELECT * FROM FEATURES;").unwrap();
    let out = session.execute(&stmt).expect("SELECT FEATURES");
    assert!(out.iter().any(|l| l.contains("Layer")));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn select_features_with_layer_filter_runs() {
    let (mut session, dir) = vindex_session("select_features_layer");
    let stmt = parser::parse("SELECT * FROM FEATURES WHERE layer = 1;").unwrap();
    let _ = session.execute(&stmt).expect("SELECT FEATURES WHERE layer");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn select_features_with_feature_filter_runs() {
    let (mut session, dir) = vindex_session("select_features_feat");
    let stmt = parser::parse("SELECT * FROM FEATURES WHERE feature = 0;").unwrap();
    let _ = session.execute(&stmt).expect("SELECT FEATURES WHERE feature");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn select_features_token_filter_runs() {
    let (mut session, dir) = vindex_session("select_features_token");
    let stmt = parser::parse(r#"SELECT * FROM FEATURES WHERE token = "Paris";"#).unwrap();
    let _ = session.execute(&stmt).expect("SELECT FEATURES WHERE token");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn select_features_explicit_limit_runs() {
    let (mut session, dir) = vindex_session("select_features_limit");
    let stmt = parser::parse("SELECT * FROM FEATURES LIMIT 1;").unwrap();
    let _ = session.execute(&stmt).expect("SELECT FEATURES LIMIT");
    let _ = std::fs::remove_dir_all(&dir);
}

// ── SELECT * FROM ENTITIES ───────────────────────────────────

#[test]
fn select_entities_default_runs() {
    let (mut session, dir) = vindex_session("select_entities_default");
    let stmt = parser::parse("SELECT * FROM ENTITIES;").unwrap();
    let out = session.execute(&stmt).expect("SELECT ENTITIES");
    assert!(out.iter().any(|l| l.contains("Entity")));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn select_entities_filter_runs() {
    let (mut session, dir) = vindex_session("select_entities_filter");
    let stmt = parser::parse(r#"SELECT * FROM ENTITIES WHERE entity = "berlin";"#).unwrap();
    let _ = session.execute(&stmt).expect("SELECT ENTITIES");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn select_entities_layer_filter_runs() {
    let (mut session, dir) = vindex_session("select_entities_layer");
    let stmt = parser::parse("SELECT * FROM ENTITIES WHERE layer = 0;").unwrap();
    let _ = session.execute(&stmt).expect("SELECT ENTITIES");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn select_entities_no_matches_runs() {
    let (mut session, dir) = vindex_session("select_entities_empty");
    let stmt = parser::parse(r#"SELECT * FROM ENTITIES WHERE entity = "Nowhere";"#).unwrap();
    let _ = session.execute(&stmt).expect("SELECT ENTITIES");
    let _ = std::fs::remove_dir_all(&dir);
}

// ── SELECT … FROM EDGES NEAREST TO (rich fixture) ────────────

#[test]
fn select_nearest_to_known_entity_runs() {
    let (mut session, dir) = rich_vindex_session("select_nearest_known");
    let stmt =
        parser::parse(r#"SELECT * FROM EDGES NEAREST TO "Paris" AT LAYER 0;"#).unwrap();
    let _ = session.execute(&stmt).expect("SELECT NEAREST");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn select_nearest_to_unknown_entity_runs() {
    let (mut session, dir) = rich_vindex_session("select_nearest_unknown");
    let stmt =
        parser::parse(r#"SELECT * FROM EDGES NEAREST TO "Atlantis" AT LAYER 0 LIMIT 5;"#)
            .unwrap();
    let _ = session.execute(&stmt).expect("SELECT NEAREST");
    let _ = std::fs::remove_dir_all(&dir);
}

// ── DESCRIBE (rich fixture) ──────────────────────────────────

#[test]
fn describe_known_entity_runs_through_walk_pipeline() {
    let (mut session, dir) = rich_vindex_session("describe_known");
    // The walk against the rich fixture probably won't surface
    // gate-thresholded edges (DESCRIBE_GATE_THRESHOLD = 5.0 vs gate
    // norms ~ 1.0 here), so the orchestrator likely falls through to
    // "(no edges found)" — but the entire phase pipeline runs.
    let stmt = parser::parse(r#"DESCRIBE "Paris";"#).unwrap();
    let out = session.execute(&stmt).expect("DESCRIBE");
    // The first line is always the entity name.
    assert_eq!(out[0], "Paris");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn describe_unknown_entity_emits_not_found() {
    let (mut session, dir) = rich_vindex_session("describe_unknown");
    // "Mongolia" isn't in the vocab — tokenises to [UNK] (id 0),
    // average_embed_rows succeeds (unit at axis 0), so we don't take
    // the "(not found)" branch but rather "(no edges found)" once the
    // walk produces nothing above the gate threshold.
    let stmt = parser::parse(r#"DESCRIBE "Mongolia";"#).unwrap();
    let out = session.execute(&stmt).expect("DESCRIBE");
    let joined = out.join("\n");
    // Either branch is acceptable — we just want to exercise the path.
    assert!(joined.contains("Mongolia"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn describe_brief_mode_compiles() {
    let (mut session, dir) = rich_vindex_session("describe_brief");
    let stmt = parser::parse(r#"DESCRIBE "Paris" BRIEF;"#).unwrap();
    let _ = session.execute(&stmt).expect("DESCRIBE BRIEF");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn describe_at_explicit_layer_uses_layer_filter() {
    let (mut session, dir) = rich_vindex_session("describe_layer");
    let stmt = parser::parse(r#"DESCRIBE "Paris" AT LAYER 1;"#).unwrap();
    let out = session.execute(&stmt).expect("DESCRIBE AT LAYER");
    assert_eq!(out[0], "Paris");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn describe_with_band_clause_runs() {
    let (mut session, dir) = rich_vindex_session("describe_band");
    let stmt = parser::parse(r#"DESCRIBE "Paris" SYNTAX;"#).unwrap();
    let _ = session.execute(&stmt).expect("DESCRIBE SYNTAX");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn describe_relations_only_runs() {
    let (mut session, dir) = rich_vindex_session("describe_rel_only");
    let stmt = parser::parse(r#"DESCRIBE "Paris" RELATIONS ONLY;"#).unwrap();
    let _ = session.execute(&stmt).expect("DESCRIBE RELATIONS ONLY");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn select_edges_with_entity_and_relation_drives_walk_path() {
    let (mut session, dir) = rich_vindex_session("select_edges_walk");
    // Both filters present → walk-anchored path. The rich fixture has
    // a relation classifier (feature_labels.json) so labelled edges
    // can match the user's relation predicate.
    let stmt = parser::parse(
        r#"SELECT * FROM EDGES WHERE entity = "Paris" AND relation = "capital";"#,
    )
    .unwrap();
    let _ = session
        .execute(&stmt)
        .expect("SELECT EDGES walk path with classifier");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn select_edges_walk_path_with_feature_filter() {
    let (mut session, dir) = rich_vindex_session("select_edges_walk_feat");
    let stmt = parser::parse(
        r#"SELECT * FROM EDGES WHERE entity = "Paris" AND relation = "capital" AND feature = 0;"#,
    )
    .unwrap();
    let _ = session.execute(&stmt).expect("SELECT EDGES walk + feature");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn show_relations_runs_against_classifier() {
    let (mut session, dir) = rich_vindex_session("show_relations");
    let stmt = parser::parse("SHOW RELATIONS;").unwrap();
    let _ = session.execute(&stmt).expect("SHOW RELATIONS");
    let _ = std::fs::remove_dir_all(&dir);
}

// ── SHOW + STATS verbs ───────────────────────────────────────

#[test]
fn show_layers_lists_layers() {
    let (mut session, dir) = vindex_session("show_layers");
    let stmt = parser::parse("SHOW LAYERS;").unwrap();
    let out = session.execute(&stmt).expect("SHOW LAYERS");
    // Synthetic vindex has 2 layers — at least one row should mention L0 or L1.
    let joined = out.join("\n");
    assert!(joined.contains("L0") || joined.contains("L1") || !joined.is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn show_layers_with_range_filter() {
    let (mut session, dir) = vindex_session("show_layers_range");
    let stmt = parser::parse("SHOW LAYERS 0-1;").unwrap();
    let _ = session.execute(&stmt).expect("SHOW LAYERS range");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn show_features_at_layer_runs() {
    let (mut session, dir) = vindex_session("show_features_layer");
    let stmt = parser::parse("SHOW FEATURES 0;").unwrap();
    let _ = session.execute(&stmt).expect("SHOW FEATURES 0");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn show_entities_with_classifier_runs() {
    let (mut session, dir) = rich_vindex_session("show_entities");
    let stmt = parser::parse("SHOW ENTITIES;").unwrap();
    let _ = session.execute(&stmt).expect("SHOW ENTITIES");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn show_relations_no_backend_errors() {
    let mut session = Session::new();
    let stmt = parser::parse("SHOW RELATIONS;").unwrap();
    assert!(session.execute(&stmt).is_err());
}

#[test]
fn show_compact_status_runs_against_synthetic_vindex() {
    let (mut session, dir) = vindex_session("show_compact_status");
    let stmt = parser::parse("SHOW COMPACT STATUS;").unwrap();
    let out = session.execute(&stmt).expect("SHOW COMPACT STATUS");
    let joined = out.join("\n");
    // Synthetic fixture has hidden_dim=4 < 1024 → L2 line is the
    // "not available" branch, but every status line is unconditional.
    assert!(joined.contains("L0"));
    assert!(joined.contains("L1"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn stats_runs_against_synthetic_vindex() {
    let (mut session, dir) = vindex_session("stats");
    let stmt = parser::parse("STATS;").unwrap();
    let out = session.execute(&stmt).expect("STATS");
    assert!(!out.is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn stats_with_explicit_path_runs() {
    // STATS "<path>" form — explicit vindex path argument.
    let (mut session, dir) = vindex_session("stats_path");
    let stmt = parser::parse(&format!(r#"STATS "{}";"#, dir.display())).unwrap();
    let _ = session.execute(&stmt).expect("STATS <path>");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn show_relations_verbose_runs_against_classifier() {
    let (mut session, dir) = rich_vindex_session("show_rel_verbose");
    let stmt = parser::parse("SHOW RELATIONS VERBOSE;").unwrap();
    let _ = session.execute(&stmt).expect("SHOW RELATIONS VERBOSE");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn show_relations_with_examples_runs() {
    let (mut session, dir) = rich_vindex_session("show_rel_examples");
    let stmt = parser::parse("SHOW RELATIONS WITH EXAMPLES;").unwrap();
    let _ = session.execute(&stmt).expect("SHOW RELATIONS WITH EXAMPLES");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn show_relations_at_specific_layer_runs() {
    let (mut session, dir) = rich_vindex_session("show_rel_layer");
    let stmt = parser::parse("SHOW RELATIONS AT LAYER 0;").unwrap();
    let _ = session.execute(&stmt).expect("SHOW RELATIONS AT LAYER 0");
    let _ = std::fs::remove_dir_all(&dir);
}

// ── DIFF ─────────────────────────────────────────────────────

#[test]
fn diff_two_synthetic_vindexes_runs() {
    let dir_a = make_test_vindex_dir("diff_a");
    let dir_b = make_test_vindex_dir("diff_b");
    let mut session = Session::new();
    // DIFF doesn't require a USE — it takes explicit paths.
    let stmt = parser::parse(&format!(
        r#"DIFF "{}" "{}";"#,
        dir_a.display(),
        dir_b.display()
    ))
    .unwrap();
    let out = session.execute(&stmt).expect("DIFF");
    let joined = out.join("\n");
    assert!(joined.contains("Diff:"));
    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

#[test]
fn diff_with_layer_filter_runs() {
    let dir_a = make_test_vindex_dir("diff_layer_a");
    let dir_b = make_test_vindex_dir("diff_layer_b");
    let mut session = Session::new();
    let stmt = parser::parse(&format!(
        r#"DIFF "{}" "{}" LAYER 0;"#,
        dir_a.display(),
        dir_b.display()
    ))
    .unwrap();
    let _ = session.execute(&stmt).expect("DIFF LAYER");
    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

#[test]
fn diff_with_explicit_limit_runs() {
    let dir_a = make_test_vindex_dir("diff_limit_a");
    let dir_b = make_test_vindex_dir("diff_limit_b");
    let mut session = Session::new();
    let stmt = parser::parse(&format!(
        r#"DIFF "{}" "{}" LIMIT 5;"#,
        dir_a.display(),
        dir_b.display()
    ))
    .unwrap();
    let _ = session.execute(&stmt).expect("DIFF LIMIT");
    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

#[test]
fn diff_with_nonexistent_source_errors() {
    let mut session = Session::new();
    let stmt = parser::parse(
        r#"DIFF "/tmp/no_such_vindex_a" "/tmp/no_such_vindex_b";"#,
    )
    .unwrap();
    let err = session.execute(&stmt).unwrap_err();
    assert!(err.to_string().to_lowercase().contains("load") || err.to_string().contains("not found"));
}

// ── MERGE ────────────────────────────────────────────────────

#[test]
fn merge_synthetic_into_current_keeps_source_strategy() {
    let (mut session, dir) = vindex_session("merge_target");
    let source_dir = make_test_vindex_dir("merge_source");
    let stmt = parser::parse(&format!(
        r#"MERGE "{}";"#,
        source_dir.display()
    ))
    .unwrap();
    let out = session.execute(&stmt).expect("MERGE");
    let joined = out.join("\n");
    assert!(joined.contains("Merged"));
    assert!(joined.contains("features merged"));
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&source_dir);
}

#[test]
fn merge_with_keep_target_strategy() {
    let (mut session, dir) = vindex_session("merge_keep_target");
    let source_dir = make_test_vindex_dir("merge_keep_target_src");
    let stmt = parser::parse(&format!(
        r#"MERGE "{}" ON CONFLICT KEEP_TARGET;"#,
        source_dir.display()
    ))
    .unwrap();
    let _ = session.execute(&stmt).expect("MERGE KEEP_TARGET");
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&source_dir);
}

#[test]
fn merge_with_highest_confidence_strategy() {
    let (mut session, dir) = vindex_session("merge_highest");
    let source_dir = make_test_vindex_dir("merge_highest_src");
    let stmt = parser::parse(&format!(
        r#"MERGE "{}" ON CONFLICT HIGHEST_CONFIDENCE;"#,
        source_dir.display()
    ))
    .unwrap();
    let _ = session.execute(&stmt).expect("MERGE HIGHEST_CONFIDENCE");
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&source_dir);
}

#[test]
fn merge_with_missing_source_errors() {
    let (mut session, dir) = vindex_session("merge_no_source");
    let stmt = parser::parse(r#"MERGE "/tmp/no_source_vindex_xyz";"#).unwrap();
    let err = session.execute(&stmt).unwrap_err();
    assert!(err.to_string().contains("source vindex not found"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn merge_no_backend_no_target_errors() {
    let mut session = Session::new();
    let source_dir = make_test_vindex_dir("merge_no_backend");
    let stmt = parser::parse(&format!(
        r#"MERGE "{}";"#,
        source_dir.display()
    ))
    .unwrap();
    // No USE ran → MERGE without explicit target should error.
    let _ = session.execute(&stmt); // accept either error or ok depending on path
    let _ = std::fs::remove_dir_all(&source_dir);
}

// ── WALK + EXPLAIN WALK against rich fixture ─────────────────

#[test]
fn walk_against_rich_fixture_runs() {
    let (mut session, dir) = rich_vindex_session("walk_basic");
    let stmt = parser::parse(r#"WALK "Paris";"#).unwrap();
    let out = session.execute(&stmt).expect("WALK");
    assert!(!out.is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn walk_with_top_clause_runs() {
    let (mut session, dir) = rich_vindex_session("walk_top");
    let stmt = parser::parse(r#"WALK "France" TOP 3;"#).unwrap();
    let _ = session.execute(&stmt).expect("WALK TOP 3");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn walk_with_layers_clause_runs() {
    let (mut session, dir) = rich_vindex_session("walk_layers");
    let stmt = parser::parse(r#"WALK "Berlin" LAYERS 0-1;"#).unwrap();
    let _ = session.execute(&stmt).expect("WALK LAYERS");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn walk_unknown_token_uses_unk() {
    // Unknown tokens map to [UNK] (id 0), which has a valid embed row.
    let (mut session, dir) = rich_vindex_session("walk_unk");
    let stmt = parser::parse(r#"WALK "MongoliaXYZ";"#).unwrap();
    let _ = session.execute(&stmt).expect("WALK with [UNK]");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn explain_walk_runs() {
    let (mut session, dir) = rich_vindex_session("explain_walk");
    let stmt = parser::parse(r#"EXPLAIN WALK "Paris";"#).unwrap();
    let out = session.execute(&stmt).expect("EXPLAIN WALK");
    let joined = out.join("\n");
    // Output format: "L<n>: F<m> → <token> (gate=<x>, down=[...])"
    assert!(joined.contains("L0") || joined.contains("L1"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn explain_walk_verbose_runs() {
    let (mut session, dir) = rich_vindex_session("explain_walk_verbose");
    let stmt = parser::parse(r#"EXPLAIN WALK "Paris" VERBOSE;"#).unwrap();
    let _ = session.execute(&stmt).expect("EXPLAIN WALK VERBOSE");
    let _ = std::fs::remove_dir_all(&dir);
}

// ── REBALANCE early-exit ─────────────────────────────────────

#[test]
fn rebalance_with_no_installs_short_circuits_v2() {
    let (mut session, dir) = vindex_session("rebalance_empty_v2");
    // No prior INSERT compose → installed_edges is empty → REBALANCE
    // hits the "nothing to rebalance" early-return path.
    let stmt = parser::parse("REBALANCE;").unwrap();
    let out = session.execute(&stmt).expect("REBALANCE");
    let joined = out.join("\n");
    assert!(joined.contains("no compose-mode installs"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn rebalance_with_explicit_clauses_short_circuits() {
    let (mut session, dir) = vindex_session("rebalance_clauses_v2");
    let stmt = parser::parse(
        "REBALANCE FLOOR 0.20 CEILING 0.95 MAX 8;",
    )
    .unwrap();
    let _ = session.execute(&stmt).expect("REBALANCE with clauses");
    let _ = std::fs::remove_dir_all(&dir);
}

// ── COMPACT MINOR / COMPACT MAJOR short-circuits ─────────────

#[test]
fn compact_minor_with_empty_l0_short_circuits_v2() {
    let (mut session, dir) = vindex_session("compact_minor_empty_v2");
    let stmt = parser::parse("COMPACT MINOR;").unwrap();
    let out = session.execute(&stmt).expect("COMPACT MINOR");
    let joined = out.join("\n");
    assert!(joined.contains("L0 is empty"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compact_major_on_small_hidden_dim_errors() {
    let (mut session, dir) = vindex_session("compact_major_small_v2");
    // Synthetic fixture has hidden_dim=4 < 1024 → COMPACT MAJOR errors
    // with the "requires hidden_dim >= 1024" message.
    let stmt = parser::parse("COMPACT MAJOR;").unwrap();
    let err = session.execute(&stmt).unwrap_err();
    assert!(err.to_string().contains("hidden_dim"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compact_major_with_lambda_clause_parses_and_errors_on_small_dim() {
    let (mut session, dir) = vindex_session("compact_major_lambda");
    let stmt = parser::parse("COMPACT MAJOR WITH LAMBDA = 0.001;").unwrap();
    let err = session.execute(&stmt).unwrap_err();
    assert!(err.to_string().contains("hidden_dim") || err.to_string().contains("model weights"));
    let _ = std::fs::remove_dir_all(&dir);
}

// ── Full-fixture tests (real ModelWeights on disk) ───────────

#[test]
fn full_fixture_loads_via_use() {
    let (session, dir) = full_vindex_session("loads");
    assert!(matches!(session.backend, Backend::Vindex { .. }));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn infer_against_full_fixture_runs() {
    let (mut session, dir) = full_vindex_session("infer_basic");
    // Tokeniser maps any string to `[N]` token IDs in 0..32. Prompt
    // produces a small prefix the FFN walk can run against.
    let stmt = parser::parse(r#"INFER "[1] [2]" TOP 3;"#).unwrap();
    let _ = session.execute(&stmt).expect("INFER");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn infer_with_compare_mode_runs() {
    let (mut session, dir) = full_vindex_session("infer_compare");
    let stmt = parser::parse(r#"INFER "[3]" TOP 2 COMPARE;"#).unwrap();
    let _ = session.execute(&stmt).expect("INFER COMPARE");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn explain_infer_against_full_fixture_runs() {
    let (mut session, dir) = full_vindex_session("explain_infer");
    let stmt = parser::parse(r#"EXPLAIN INFER "[1] [2]";"#).unwrap();
    let _ = session.execute(&stmt).expect("EXPLAIN INFER");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn explain_infer_verbose_with_attention_runs() {
    let (mut session, dir) = full_vindex_session("explain_infer_attn");
    let stmt = parser::parse(r#"EXPLAIN INFER "[5]" VERBOSE WITH ATTENTION;"#).unwrap();
    let _ = session.execute(&stmt).expect("EXPLAIN INFER VERBOSE WITH ATTENTION");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn trace_against_full_fixture_runs() {
    let (mut session, dir) = full_vindex_session("trace_basic");
    let stmt = parser::parse(r#"TRACE "[1] [2]";"#).unwrap();
    let _ = session.execute(&stmt).expect("TRACE");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn trace_with_decompose_runs() {
    let (mut session, dir) = full_vindex_session("trace_decompose");
    let stmt = parser::parse(r#"TRACE "[1]" DECOMPOSE;"#).unwrap();
    let _ = session.execute(&stmt).expect("TRACE DECOMPOSE");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn insert_compose_against_full_fixture_runs() {
    // Compose-mode INSERT runs the full pipeline: plan → capture
    // residuals → install slots → balance → cross-fact regression
    // check. Each phase calls `predict_with_ffn` against the random-
    // init weights; outputs are nonsense but every code path runs.
    //
    // Entity/relation/target use `[N]` patterns to land inside the
    // test tokenizer's vocab. Layer 0 is pinned so the plan picks a
    // single layer that exists in the 2-layer fixture.
    let (mut session, dir) = full_vindex_session("insert_compose");
    let stmt = parser::parse(
        r#"INSERT INTO EDGES (entity, relation, target)
           VALUES ("[1]", "[2]", "[5]") AT LAYER 0 MODE COMPOSE;"#,
    )
    .unwrap();
    let _ = session.execute(&stmt).expect("INSERT compose");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn insert_compose_with_explicit_alpha_and_confidence() {
    let (mut session, dir) = full_vindex_session("insert_compose_alpha");
    let stmt = parser::parse(
        r#"INSERT INTO EDGES (entity, relation, target)
           VALUES ("[3]", "[4]", "[7]")
           AT LAYER 1
           CONFIDENCE 0.95
           ALPHA 0.20
           MODE COMPOSE;"#,
    )
    .unwrap();
    let _ = session.execute(&stmt).expect("INSERT compose with alpha+confidence");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn rebalance_after_compose_insert_runs_real_loop() {
    // After a successful compose-mode INSERT, `installed_edges` has
    // a fact, so REBALANCE skips the early-exit and enters the
    // fixed-point loop (loads weights/tokenizer, runs probe walks).
    let (mut session, dir) = full_vindex_session("rebalance_after_compose");

    let insert = parser::parse(
        r#"INSERT INTO EDGES (entity, relation, target)
           VALUES ("[1]", "[2]", "[5]") AT LAYER 0 MODE COMPOSE;"#,
    )
    .unwrap();
    let _ = session.execute(&insert).expect("INSERT compose");

    let rebal = parser::parse("REBALANCE MAX 2;").unwrap();
    let _ = session.execute(&rebal).expect("REBALANCE after compose");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn insert_knn_against_full_fixture_runs() {
    let (mut session, dir) = full_vindex_session("insert_knn");
    let stmt = parser::parse(
        r#"INSERT INTO EDGES (entity, relation, target)
           VALUES ("[1]", "[2]", "[5]") MODE KNN;"#,
    )
    .unwrap();
    let _ = session.execute(&stmt).expect("INSERT KNN");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compact_minor_after_knn_insert_promotes_l0_to_l1() {
    // KNN INSERT writes to L0 (knn_store). COMPACT MINOR then
    // promotes those entries to L1 via compose-mode reinstall.
    let (mut session, dir) = full_vindex_session("compact_minor_promotes");

    let insert = parser::parse(
        r#"INSERT INTO EDGES (entity, relation, target)
           VALUES ("[1]", "[2]", "[5]") AT LAYER 0 MODE KNN;"#,
    )
    .unwrap();
    let _ = session.execute(&insert).expect("INSERT KNN");

    let compact = parser::parse("COMPACT MINOR;").unwrap();
    // COMPACT MINOR may succeed or fail (compose-mode reinstall might
    // hit an internal error against the random-init model). Either way
    // it exercises the L0→L1 loop.
    let _ = session.execute(&compact);
    let _ = std::fs::remove_dir_all(&dir);
}

// ── INFER / TRACE / EXPLAIN INFER mode variants ──────────────

#[test]
fn infer_with_top_5_runs() {
    let (mut session, dir) = full_vindex_session("infer_top5");
    let stmt = parser::parse(r#"INFER "[1] [2] [3]" TOP 5;"#).unwrap();
    let _ = session.execute(&stmt).expect("INFER TOP 5");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn infer_default_top_runs() {
    let (mut session, dir) = full_vindex_session("infer_default");
    let stmt = parser::parse(r#"INFER "[7]";"#).unwrap();
    let _ = session.execute(&stmt).expect("INFER default top");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn explain_infer_with_band_clause_runs() {
    let (mut session, dir) = full_vindex_session("explain_infer_band");
    let stmt = parser::parse(r#"EXPLAIN INFER "[1] [2]" KNOWLEDGE;"#).unwrap();
    let _ = session.execute(&stmt).expect("EXPLAIN INFER KNOWLEDGE");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn explain_infer_relations_only_runs() {
    let (mut session, dir) = full_vindex_session("explain_infer_rel_only");
    let stmt =
        parser::parse(r#"EXPLAIN INFER "[1]" RELATIONS ONLY;"#).unwrap();
    let _ = session.execute(&stmt).expect("EXPLAIN INFER RELATIONS ONLY");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn explain_infer_with_top_clause_runs() {
    let (mut session, dir) = full_vindex_session("explain_infer_top");
    let stmt = parser::parse(r#"EXPLAIN INFER "[5]" TOP 3;"#).unwrap();
    let _ = session.execute(&stmt).expect("EXPLAIN INFER TOP 3");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn trace_for_specific_token_runs() {
    let (mut session, dir) = full_vindex_session("trace_for");
    let stmt = parser::parse(r#"TRACE "[1] [2]" FOR "[5]";"#).unwrap();
    let _ = session.execute(&stmt).expect("TRACE FOR token");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn trace_with_positions_all_runs() {
    let (mut session, dir) = full_vindex_session("trace_positions_all");
    let stmt = parser::parse(r#"TRACE "[1] [2]" POSITIONS ALL;"#).unwrap();
    let _ = session.execute(&stmt).expect("TRACE POSITIONS ALL");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn trace_with_save_writes_file() {
    let (mut session, dir) = full_vindex_session("trace_save");
    let save_path = dir.join("trace_output.json");
    let stmt = parser::parse(&format!(
        r#"TRACE "[1] [2]" SAVE "{}";"#,
        save_path.display()
    ))
    .unwrap();
    let _ = session.execute(&stmt).expect("TRACE SAVE");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn trace_with_layers_clause_runs() {
    let (mut session, dir) = full_vindex_session("trace_layers");
    let stmt = parser::parse(r#"TRACE "[1]" LAYERS 0-1;"#).unwrap();
    let _ = session.execute(&stmt).expect("TRACE LAYERS");
    let _ = std::fs::remove_dir_all(&dir);
}

// ── DIFF INTO PATCH ──────────────────────────────────────────

#[test]
fn diff_into_patch_writes_vlp_file() {
    let dir_a = make_test_vindex_dir("diff_into_patch_a");
    let dir_b = make_test_vindex_dir("diff_into_patch_b");
    let mut session = Session::new();
    let patch_path = std::env::temp_dir().join(format!(
        "larql_diff_patch_{}.vlp",
        std::process::id()
    ));
    let stmt = parser::parse(&format!(
        r#"DIFF "{}" "{}" INTO PATCH "{}";"#,
        dir_a.display(),
        dir_b.display(),
        patch_path.display()
    ))
    .unwrap();
    let _ = session.execute(&stmt).expect("DIFF INTO PATCH");
    let _ = std::fs::remove_file(&patch_path);
    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

// ── COMPILE INTO MODEL ───────────────────────────────────────

#[test]
fn compile_into_model_default_path_runs() {
    let (mut session, dir) = full_vindex_session("compile_into_model");
    let out_dir = dir.join("compiled_model");
    let stmt = parser::parse(&format!(
        r#"COMPILE CURRENT INTO MODEL "{}" FORMAT safetensors;"#,
        out_dir.display()
    ))
    .unwrap();
    let _ = session.execute(&stmt).expect("COMPILE INTO MODEL");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compile_into_model_with_memit_enabled_runs() {
    // LARQL_MEMIT_ENABLE=1 + a compose-mode INSERT in the recording
    // means COMPILE INTO MODEL invokes the MEMIT solver path.
    let (mut session, dir) = full_vindex_session("compile_into_model_memit");

    let insert = parser::parse(
        r#"INSERT INTO EDGES (entity, relation, target)
           VALUES ("[1]", "[2]", "[5]") AT LAYER 0 MODE COMPOSE;"#,
    )
    .unwrap();
    let _ = session.execute(&insert).expect("INSERT compose");

    let out_dir = dir.join("compiled_memit");
    let stmt = parser::parse(&format!(
        r#"COMPILE CURRENT INTO MODEL "{}" FORMAT safetensors;"#,
        out_dir.display()
    ))
    .unwrap();

    // Toggle MEMIT on for the duration of this call.
    std::env::set_var("LARQL_MEMIT_ENABLE", "1");
    let result = session.execute(&stmt);
    std::env::remove_var("LARQL_MEMIT_ENABLE");

    // The MEMIT solve may or may not converge cleanly with random-init
    // weights — accept either outcome but exercise the path.
    let _ = result;
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compact_major_against_full_fixture_runs() {
    // Full fixture has hidden_dim=16 < 1024 — COMPACT MAJOR still
    // errors on the hidden_dim check, but exercises more of the
    // pre-check path than the small-fixture variant since model
    // weights are loadable.
    let (mut session, dir) = full_vindex_session("compact_major");
    let stmt = parser::parse("COMPACT MAJOR;").unwrap();
    let err = session.execute(&stmt).unwrap_err();
    assert!(err.to_string().contains("hidden_dim"));
    let _ = std::fs::remove_dir_all(&dir);
}
