//! Helpers for the `build_vindex` extraction pipeline.
//!
//! Each function is a discrete pipeline stage or utility used by
//! `super::build::build_vindex`:
//!
//! - `chrono_now`              — ISO-8601 timestamp without `chrono`.
//! - `build_whole_word_vocab`  — reduce the vocab to whole-word tokens
//!                               + matching embedding rows.
//! - `compute_gate_top_tokens` — per-feature top whole-word token (the
//!                               "what activates this feature" label).
//! - `compute_offset_direction`— normalised `embed[output] - embed[input]`
//!                               direction; the relation vector for
//!                               clustering.
//! - `ClusterData`             — collected cluster inputs.
//! - `run_clustering_pipeline` — k-means + label + write
//!                               `relation_clusters.json` /
//!                               `feature_clusters.jsonl`.

use std::io::{BufWriter, Write};
use std::path::Path;

use crate::extract::constants::{
    FIRST_CONTENT_TOKEN_ID, GATE_TOP_TOKEN_BATCH, MAX_RELATION_CLUSTERS, RELATION_KMEANS_ITERS,
};
use crate::extract::stage_labels::STAGE_RELATION_CLUSTERS;
use crate::format::filenames::{FEATURE_CLUSTERS_JSONL, RELATION_CLUSTERS_JSON};

use larql_models::{FfnType, ModelWeights};
use ndarray::Array2;

use crate::error::VindexError;
use crate::extract::callbacks::IndexBuildCallbacks;

// ── Timestamp ──────────────────────────────────────────────────────────

/// Simple ISO 8601 timestamp without chrono dependency.
pub(crate) fn chrono_now() -> String {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = d.as_secs();
    // Rough UTC timestamp — good enough for provenance
    let days = secs / 86400;
    let years_approx = 1970 + days / 365;
    let remainder_days = days % 365;
    let months = remainder_days / 30 + 1;
    let day = remainder_days % 30 + 1;
    let hour = (secs % 86400) / 3600;
    let min = (secs % 3600) / 60;
    let sec = secs % 60;
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        years_approx,
        months.min(12),
        day.min(31),
        hour,
        min,
        sec
    )
}

// ── Whole-word vocab ───────────────────────────────────────────────────

/// Build the whole-word vocabulary: tokens that decode as 3+ char alphabetic words.
/// Returns (token_ids, reduced_embedding_matrix).
pub(crate) fn build_whole_word_vocab(
    tokenizer: &tokenizers::Tokenizer,
    embed: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,
    vocab_size: usize,
    hidden_size: usize,
) -> (Vec<usize>, Array2<f32>) {
    let mut ww_ids: Vec<usize> = Vec::new();
    for id in 0..vocab_size {
        if let Ok(tok) = tokenizer.decode(&[id as u32], true) {
            let tok = tok.trim();
            if tok.len() >= 3
                && tok
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '\'')
            {
                ww_ids.push(id);
            }
        }
    }

    let ww_count = ww_ids.len();
    let mut ww_embed = Array2::<f32>::zeros((ww_count, hidden_size));
    for (i, &id) in ww_ids.iter().enumerate() {
        ww_embed.row_mut(i).assign(&embed.row(id));
    }

    eprintln!(
        "    Whole-word vocab: {} tokens (of {})",
        ww_count, vocab_size
    );
    (ww_ids, ww_embed)
}

// ── Gate top tokens ────────────────────────────────────────────────────

/// Compute gate top tokens for features at a layer using whole-word embeddings.
/// Returns a Vec<String> of decoded whole-word tokens, one per feature.
pub(super) fn compute_gate_top_tokens(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    layer: usize,
    num_features: usize,
    ww_ids: &[usize],
    ww_embed: &Array2<f32>,
) -> Vec<String> {
    // Gated FFN routes through `ffn_gate`; non-gated FFN (GPT-2, StarCoder2)
    // reuses `ffn_up` for the same per-feature input direction.
    let gate_key = match weights.arch.ffn_type() {
        FfnType::Gated => weights.arch.ffn_gate_key(layer),
        FfnType::Standard => weights.arch.ffn_up_key(layer),
    };
    let w_gate = match weights.tensors.get(&gate_key) {
        Some(w) => w,
        None => return vec![String::new(); num_features],
    };

    let mut tokens = vec![String::new(); num_features];
    let gbatch = GATE_TOP_TOKEN_BATCH;
    for gstart in (0..num_features).step_by(gbatch) {
        let gend = (gstart + gbatch).min(num_features);
        let chunk = w_gate.slice(ndarray::s![gstart..gend, ..]);
        let cpu = larql_compute::CpuBackend;
        use larql_compute::MatMul;
        let proj = cpu.matmul_transb(ww_embed.view(), chunk.view());
        for f in 0..(gend - gstart) {
            let col = proj.column(f);
            let mut best_idx = 0;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &val) in col.iter().enumerate() {
                if val > best_val {
                    best_val = val;
                    best_idx = i;
                }
            }
            let tok_id = ww_ids[best_idx];
            tokens[gstart + f] = tokenizer
                .decode(&[tok_id as u32], true)
                .unwrap_or_default()
                .trim()
                .to_string();
        }
    }
    tokens
}

// ── Offset direction ───────────────────────────────────────────────────

/// Compute the offset direction for a gate→down feature pair.
/// Returns normalized(output_embed - input_embed) or None if invalid.
pub(super) fn compute_offset_direction(
    gate_token: &str,
    output_token_id: usize,
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    hidden_size: usize,
    vocab_size: usize,
) -> Option<Vec<f32>> {
    if gate_token.is_empty()
        || output_token_id < FIRST_CONTENT_TOKEN_ID
        || output_token_id >= vocab_size
    {
        return None;
    }

    // Get gate token embedding (may be multi-subword)
    let enc = tokenizer.encode(gate_token, false).ok()?;
    let ids = enc.get_ids();
    let valid: Vec<usize> = ids
        .iter()
        .filter(|&&id| id as usize >= FIRST_CONTENT_TOKEN_ID)
        .map(|&id| id as usize)
        .filter(|&id| id < vocab_size)
        .collect();
    if valid.is_empty() {
        return None;
    }

    let mut input_avg = vec![0.0f32; hidden_size];
    for &id in &valid {
        for (j, &v) in weights.embed.row(id).iter().enumerate() {
            input_avg[j] += v;
        }
    }
    let n = valid.len() as f32;
    for v in &mut input_avg {
        *v /= n;
    }

    let output_embed = weights.embed.row(output_token_id);
    let offset: Vec<f32> = output_embed
        .iter()
        .zip(input_avg.iter())
        .map(|(o, i)| o - i)
        .collect();
    let norm: f32 = offset.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 1e-8 {
        Some(offset.iter().map(|v| v / norm).collect())
    } else {
        None
    }
}

// ── Clustering ─────────────────────────────────────────────────────────

/// Collected data for relation clustering.
pub(super) struct ClusterData {
    pub directions: Vec<f32>,
    pub features: Vec<(usize, usize)>,
    pub top_tokens: Vec<String>,
    #[allow(dead_code)]
    pub input_tokens: Vec<String>,
    pub output_tokens: Vec<String>,
}

/// Run the clustering and labeling pipeline on collected cluster data.
/// Writes relation_clusters.json and feature_clusters.jsonl.
pub(super) fn run_clustering_pipeline(
    data: ClusterData,
    hidden_size: usize,
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    output_dir: &Path,
    callbacks: &mut dyn IndexBuildCallbacks,
) -> Result<(), VindexError> {
    if data.directions.is_empty() {
        return Ok(());
    }

    callbacks.on_stage(STAGE_RELATION_CLUSTERS);

    let n_features = data.features.len();
    let matrix = ndarray::Array2::from_shape_vec((n_features, hidden_size), data.directions)
        .map_err(|e| VindexError::Parse(format!("cluster data shape: {e}")))?;

    let optimal_k = MAX_RELATION_CLUSTERS.min(n_features);

    let (centres, assignments, _distances) =
        crate::clustering::kmeans(&matrix, optimal_k, RELATION_KMEANS_ITERS);

    // Load reference databases
    let ref_dbs = crate::clustering::load_reference_databases();

    // Tier 1: output-only matching — Wikidata ONLY for L14-27 features.
    // WordNet is for L0-13 (linguistic). Wikidata is for L14-27 (factual).
    // They don't compete — each database matches its own layer range.
    let wikidata_refs: Vec<&crate::clustering::pair_matching::RelationDatabase> =
        ref_dbs.wikidata.iter().collect();
    let output_labels = if !wikidata_refs.is_empty() {
        crate::clustering::pair_matching::label_clusters_from_outputs(
            &assignments,
            &data.output_tokens,
            optimal_k,
            &wikidata_refs,
        )
    } else {
        vec![None; optimal_k]
    };

    let output_labeled = output_labels.iter().filter(|l| l.is_some()).count();
    eprintln!(
        "  Wikidata output matching: {}/{} clusters labeled",
        output_labeled, optimal_k
    );

    // Tier 2+3: embedding projection + pattern detection
    let (embed_labels, top_tokens_per_cluster) =
        crate::clustering::auto_label_clusters_from_embeddings(
            &centres,
            &weights.embed,
            tokenizer,
            &assignments,
            &data.top_tokens,
            optimal_k,
        );

    // Merge: Wikidata output labels > embedding/pattern labels
    let labels: Vec<String> = (0..optimal_k)
        .map(|c| {
            output_labels[c]
                .clone()
                .unwrap_or_else(|| embed_labels[c].clone())
        })
        .collect();

    let mut counts = vec![0usize; optimal_k];
    for &a in &assignments {
        if a < optimal_k {
            counts[a] += 1;
        }
    }

    // Write relation_clusters.json
    let cluster_result = crate::clustering::ClusterResult {
        k: optimal_k,
        centres: centres.rows().into_iter().map(|r| r.to_vec()).collect(),
        labels,
        counts,
        top_tokens: top_tokens_per_cluster,
    };

    let clusters_json = serde_json::to_string_pretty(&cluster_result)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    std::fs::write(output_dir.join(RELATION_CLUSTERS_JSON), clusters_json)?;

    // Write per-feature cluster assignments
    let assign_path = output_dir.join(FEATURE_CLUSTERS_JSONL);
    let mut assign_file = BufWriter::new(std::fs::File::create(&assign_path)?);
    for (i, &(layer, feat)) in data.features.iter().enumerate() {
        let record = serde_json::json!({ "l": layer, "f": feat, "c": assignments[i] });
        serde_json::to_writer(&mut assign_file, &record)
            .map_err(|e| VindexError::Parse(e.to_string()))?;
        assign_file.write_all(b"\n")?;
    }
    assign_file.flush()?;

    callbacks.on_stage_done(
        &format!(
            "relation_clusters (k={}, {} features)",
            optimal_k, n_features
        ),
        0.0,
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    /// Build a WordLevel tokenizer with a fixed vocab via JSON. Only
    /// the ids listed exist; everything else falls back to `[UNK]`
    /// (id 0). JSON form avoids the `AHashMap` dependency that
    /// `WordLevel::builder().vocab()` requires.
    fn vocab_tokenizer(words: &[&str]) -> tokenizers::Tokenizer {
        let mut entries = vec![("[UNK]".to_string(), 0u32)];
        for (i, w) in words.iter().enumerate() {
            entries.push((w.to_string(), (i + 1) as u32));
        }
        let vocab_json: String = entries
            .iter()
            .map(|(w, id)| {
                format!(
                    "\"{}\": {}",
                    w.replace('\\', "\\\\").replace('"', "\\\""),
                    id
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        let json = format!(
            r#"{{
                "version": "1.0",
                "model": {{"type": "WordLevel", "vocab": {{{vocab_json}}}, "unk_token": "[UNK]"}},
                "pre_tokenizer": {{"type": "Whitespace"}},
                "added_tokens": []
            }}"#
        );
        tokenizers::Tokenizer::from_bytes(json.as_bytes()).unwrap()
    }

    // ── chrono_now ──────────────────────────────────────────────────

    #[test]
    fn chrono_now_returns_iso8601_z_format() {
        let s = chrono_now();
        // YYYY-MM-DDTHH:MM:SSZ
        assert_eq!(s.len(), 20);
        assert_eq!(&s[10..11], "T");
        assert_eq!(&s[19..20], "Z");
        assert_eq!(&s[4..5], "-");
        assert_eq!(&s[7..8], "-");
        assert_eq!(&s[13..14], ":");
        assert_eq!(&s[16..17], ":");
    }

    #[test]
    fn chrono_now_year_above_1970() {
        // The function uses SystemTime::now(); we can't pin an exact
        // year, but we can verify it's at least past 2020 (some
        // sanity floor on system clocks).
        let s = chrono_now();
        let year: u32 = s[..4].parse().expect("year parses");
        assert!(year >= 2020, "year {year} is too old");
    }

    #[test]
    fn chrono_now_clamps_month_and_day() {
        // The approximation can over-shoot 12 months / 31 days; the
        // formatter clamps with `.min(12)` and `.min(31)`. Since we
        // can't force a specific input, just verify the output stays
        // within bounds.
        let s = chrono_now();
        let month: u32 = s[5..7].parse().unwrap();
        let day: u32 = s[8..10].parse().unwrap();
        assert!((1..=12).contains(&month), "month {month} out of range");
        assert!((1..=31).contains(&day), "day {day} out of range");
    }

    // ── build_whole_word_vocab ──────────────────────────────────────

    #[test]
    fn build_whole_word_vocab_keeps_alphabetic_tokens_3plus_chars() {
        // Vocab has 5 valid words + [UNK] (decodes to "[UNK]" — 5 chars,
        // alphanumeric, not just alphabetic). Plus a 2-char word "hi"
        // that should be filtered (length < 3) and an unsafe "no!"
        // with a non-alphanumeric character.
        let toks = vocab_tokenizer(&["hello", "world", "hi", "no!", "foo123"]);
        // vocab_size must include id 0..=5 (UNK + 5 words).
        let vocab_size = 6;
        let hidden = 4;
        let embed = ndarray::Array2::<f32>::from_shape_fn((vocab_size, hidden), |(i, j)| {
            (i * 100 + j) as f32
        });

        let (ids, ww_embed) = build_whole_word_vocab(&toks, &embed, vocab_size, hidden);
        // Whole-words: "hello"(1), "world"(2), "foo123"(5), and "[UNK]"(0).
        // Filtered: "hi"(3, len<3), "no!"(4, has '!').
        assert!(ids.contains(&1), "hello kept");
        assert!(ids.contains(&2), "world kept");
        assert!(ids.contains(&5), "foo123 kept (alphanumeric)");
        assert!(!ids.contains(&3), "hi filtered (len<3)");
        assert!(!ids.contains(&4), "'no!' filtered (special char)");

        assert_eq!(ww_embed.shape(), &[ids.len(), hidden]);
        // First row matches embed[ids[0]].
        for (j, &v) in ww_embed.row(0).iter().enumerate() {
            assert_eq!(v, (ids[0] * 100 + j) as f32);
        }
    }

    #[test]
    fn build_whole_word_vocab_empty_vocab_returns_empty() {
        let toks = vocab_tokenizer(&[]);
        let embed = ndarray::Array2::<f32>::zeros((1, 4));
        let (ids, ww_embed) = build_whole_word_vocab(&toks, &embed, 1, 4);
        // Only [UNK] exists. "[UNK]" with brackets isn't alphanumeric
        // (the `[` and `]` chars), so the filter rejects it.
        assert!(ids.is_empty());
        assert_eq!(ww_embed.shape(), &[0, 4]);
    }

    // ── compute_offset_direction ────────────────────────────────────

    /// Build a `ModelWeights` with just enough fields to test
    /// `compute_offset_direction` — only `embed` and `arch` are read.
    fn weights_with_embed(embed: ndarray::Array2<f32>, vocab_size: usize) -> ModelWeights {
        let arch = larql_models::detect_from_json(&serde_json::json!({
            "model_type": "llama",
            "hidden_size": embed.shape()[1],
            "num_hidden_layers": 1,
            "intermediate_size": embed.shape()[1] * 2,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": embed.shape()[1],
            "rope_theta": 10000.0,
            "vocab_size": vocab_size,
        }));
        let cfg = arch.config();
        let lm_head = embed.clone();
        ModelWeights {
            tensors: HashMap::new(),
            vectors: HashMap::new(),
            raw_bytes: HashMap::new(),
            skipped_tensors: Vec::new(),
            packed_mmaps: HashMap::new(),
            packed_byte_ranges: HashMap::new(),
            embed: embed.into_shared(),
            lm_head: lm_head.into_shared(),
            position_embed: None,
            num_layers: cfg.num_layers,
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
            vocab_size,
            head_dim: cfg.head_dim,
            num_q_heads: cfg.num_q_heads,
            num_kv_heads: cfg.num_kv_heads,
            rope_base: cfg.rope_base,
            arch,
        }
    }

    #[test]
    fn compute_offset_direction_returns_normalised_vector() {
        // Vocab: [UNK]=0, "Paris"=1, "France"=2.
        // FIRST_CONTENT_TOKEN_ID is 3 in extract::constants, so set
        // up vocab so the tokens land at IDs ≥ 3. Use 5 vocab slots
        // with words at indices 3 and 4.
        // Build a tokenizer where "Paris" lands at id 3 and "France"
        // at id 4 (both ≥ FIRST_CONTENT_TOKEN_ID = 3).
        let json = r#"{
            "version": "1.0",
            "model": {
                "type": "WordLevel",
                "vocab": {"[UNK]": 0, "[PAD]": 1, "[BOS]": 2, "Paris": 3, "France": 4},
                "unk_token": "[UNK]"
            },
            "pre_tokenizer": {"type": "Whitespace"},
            "added_tokens": []
        }"#;
        let toks = tokenizers::Tokenizer::from_bytes(json.as_bytes()).unwrap();

        // 5 × 4 embed: rows 3 and 4 carry distinct directions.
        let mut embed = ndarray::Array2::<f32>::zeros((5, 4));
        // input ("France") = id 4 → embed row 4
        embed
            .row_mut(4)
            .assign(&ndarray::array![1.0, 0.0, 0.0, 0.0]);
        // output ("Paris") = id 3 → embed row 3
        embed
            .row_mut(3)
            .assign(&ndarray::array![0.0, 1.0, 0.0, 0.0]);
        let weights = weights_with_embed(embed, 5);

        let dir =
            compute_offset_direction("France", 3, &weights, &toks, 4, 5).expect("offset computed");
        // output - input = [-1, 1, 0, 0]; norm = sqrt(2). Normalised
        // direction = [-1/√2, 1/√2, 0, 0].
        let expected_neg = -1.0_f32 / 2.0_f32.sqrt();
        let expected_pos = 1.0_f32 / 2.0_f32.sqrt();
        assert!((dir[0] - expected_neg).abs() < 1e-6);
        assert!((dir[1] - expected_pos).abs() < 1e-6);
        assert!(dir[2].abs() < 1e-6);
        assert!(dir[3].abs() < 1e-6);
    }

    #[test]
    fn compute_offset_direction_returns_none_for_empty_gate_token() {
        let toks = vocab_tokenizer(&["x"]);
        let embed = ndarray::Array2::<f32>::zeros((2, 4));
        let weights = weights_with_embed(embed, 2);
        assert!(compute_offset_direction("", 3, &weights, &toks, 4, 5).is_none());
    }

    #[test]
    fn compute_offset_direction_returns_none_for_special_token_output() {
        // output_token_id < FIRST_CONTENT_TOKEN_ID (=3) → reject.
        let toks = vocab_tokenizer(&["hello"]);
        let embed = ndarray::Array2::<f32>::zeros((5, 4));
        let weights = weights_with_embed(embed, 5);
        for special_id in 0..3 {
            assert!(
                compute_offset_direction("hello", special_id, &weights, &toks, 4, 5).is_none(),
                "id {special_id} must be rejected"
            );
        }
    }

    #[test]
    fn compute_offset_direction_returns_none_for_oob_output_id() {
        let toks = vocab_tokenizer(&["hello"]);
        let embed = ndarray::Array2::<f32>::zeros((5, 4));
        let weights = weights_with_embed(embed, 5);
        // output_token_id >= vocab_size (5)
        assert!(compute_offset_direction("hello", 99, &weights, &toks, 4, 5).is_none());
    }

    #[test]
    fn compute_offset_direction_returns_none_when_gate_decodes_to_unk() {
        // "unknown_word" tokenizes to [UNK] (id 0) which is special →
        // valid IDs filter drops it → empty → None.
        let toks = vocab_tokenizer(&["hello"]);
        let embed = ndarray::Array2::<f32>::zeros((5, 4));
        let weights = weights_with_embed(embed, 5);
        assert!(compute_offset_direction("unknown_word", 3, &weights, &toks, 4, 5).is_none());
    }

    #[test]
    fn compute_offset_direction_returns_none_for_zero_offset() {
        // input embedding == output embedding → offset = 0 → norm < 1e-8 → None.
        let toks = vocab_tokenizer(&["[UNK]", "[PAD]", "[BOS]", "hello"]);
        // hello → id 4
        let mut embed = ndarray::Array2::<f32>::zeros((5, 4));
        embed
            .row_mut(3)
            .assign(&ndarray::array![1.0, 0.0, 0.0, 0.0]);
        embed
            .row_mut(4)
            .assign(&ndarray::array![1.0, 0.0, 0.0, 0.0]); // same as 3
        let weights = weights_with_embed(embed, 5);
        // input ("hello"=4) and output id=3 share the same embedding → norm 0.
        assert!(compute_offset_direction("hello", 3, &weights, &toks, 4, 5).is_none());
    }

    // ── compute_gate_top_tokens ─────────────────────────────────────

    #[test]
    fn compute_gate_top_tokens_returns_empty_strings_when_no_gate_tensor() {
        // weights.tensors doesn't contain ffn_gate_key → return Vec of
        // empty strings, one per feature.
        let toks = vocab_tokenizer(&["x"]);
        let embed = ndarray::Array2::<f32>::zeros((2, 4));
        let weights = weights_with_embed(embed.clone(), 2);
        let ww_ids = vec![0usize];
        let result = compute_gate_top_tokens(&weights, &toks, 0, 5, &ww_ids, &embed);
        assert_eq!(result.len(), 5);
        assert!(result.iter().all(|s| s.is_empty()));
    }

    // ── ClusterData + run_clustering_pipeline (early return) ────────

    #[test]
    fn run_clustering_pipeline_skips_empty_directions() {
        // empty directions → early-return Ok without touching disk.
        let toks = vocab_tokenizer(&[]);
        let embed = ndarray::Array2::<f32>::zeros((1, 4));
        let weights = weights_with_embed(embed, 1);
        let tmp = tempfile::tempdir().unwrap();
        let mut cb = crate::extract::callbacks::SilentBuildCallbacks;
        let result = run_clustering_pipeline(
            ClusterData {
                directions: Vec::new(),
                features: Vec::new(),
                top_tokens: Vec::new(),
                input_tokens: Vec::new(),
                output_tokens: Vec::new(),
            },
            4,
            &weights,
            &toks,
            tmp.path(),
            &mut cb,
        );
        assert!(result.is_ok());
        // No files written.
        assert!(!tmp.path().join(RELATION_CLUSTERS_JSON).exists());
        assert!(!tmp.path().join(FEATURE_CLUSTERS_JSONL).exists());
    }

    // ── Heavy paths ────────────────────────────────────────────────

    /// Insert a tensor `key → array` into `weights.tensors`. Used to
    /// satisfy `compute_gate_top_tokens`'s `weights.tensors.get(&gate_key)`
    /// lookup without standing up the full safetensors load path.
    fn insert_tensor(weights: &mut ModelWeights, key: &str, array: ndarray::Array2<f32>) {
        weights.tensors.insert(key.to_string(), array.into_shared());
    }

    #[test]
    fn compute_gate_top_tokens_picks_argmax_word_per_feature() {
        // Build a synthetic case:
        //   - hidden = 4
        //   - vocab has 4 whole-word tokens at ids 1..=4 plus [UNK] at 0
        //   - embed[i] = standard basis e_{i-1} (so word `i-1` aligns with axis i-1)
        //   - gate has 4 features; feature f has weight 1 at position f, 0 elsewhere
        //
        // With ww_embed @ gate^T, score[i, f] = 1 iff i == f, else 0.
        // → feature f's argmax word is the f-th whole-word id.
        let words = ["alpha", "beta", "gamma", "delta"]; // 4 whole-words
        let toks = vocab_tokenizer(&words);
        let hidden = 4;
        let vocab_size = 5; // [UNK]=0 + 4 words

        // embed[id] = e_{id-1} for ids 1..=4; embed[0] = 0.
        let mut embed = ndarray::Array2::<f32>::zeros((vocab_size, hidden));
        for i in 0..hidden {
            embed[[i + 1, i]] = 1.0;
        }
        let mut weights = weights_with_embed(embed.clone(), vocab_size);

        // ww_embed = the 4 whole-word rows (ids 1..=4 in vocab order).
        let (ww_ids, ww_embed) = build_whole_word_vocab(&toks, &embed, vocab_size, hidden);
        assert_eq!(ww_ids.len(), 4, "expected 4 whole-words for the test");

        // Gate: feature f has weight 1 at position f.
        let mut gate = ndarray::Array2::<f32>::zeros((4, hidden));
        for f in 0..4 {
            gate[[f, f]] = 1.0;
        }
        let arch = &weights.arch;
        let gate_key = arch.ffn_gate_key(0);
        insert_tensor(&mut weights, &gate_key, gate);

        let result = compute_gate_top_tokens(&weights, &toks, 0, 4, &ww_ids, &ww_embed);
        assert_eq!(result.len(), 4);

        // Decode each ww_id back to its word and verify the argmax
        // produced the expected word per feature.
        let decoded: Vec<String> = ww_ids
            .iter()
            .map(|&id| {
                toks.decode(&[id as u32], true)
                    .unwrap_or_default()
                    .trim()
                    .to_string()
            })
            .collect();
        for (f, top) in result.iter().enumerate() {
            assert_eq!(top, &decoded[f], "feature {f} should pick word {f}");
        }
    }

    #[test]
    fn compute_gate_top_tokens_iterates_in_batches() {
        // num_features > GATE_TOP_TOKEN_BATCH forces the function to
        // run multiple batch chunks. Pin that all features still get
        // labels (no off-by-one in the chunk loop).
        use crate::extract::constants::GATE_TOP_TOKEN_BATCH;
        let num_features = GATE_TOP_TOKEN_BATCH + 5;

        let words = ["alpha"];
        let toks = vocab_tokenizer(&words);
        let hidden = 4;
        let vocab_size = 2;
        let mut embed = ndarray::Array2::<f32>::zeros((vocab_size, hidden));
        embed[[1, 0]] = 1.0; // "alpha" → e_0
        let mut weights = weights_with_embed(embed.clone(), vocab_size);

        let (ww_ids, ww_embed) = build_whole_word_vocab(&toks, &embed, vocab_size, hidden);
        assert_eq!(ww_ids.len(), 1);

        // Every gate row aligned with axis 0 → argmax is always the
        // single whole-word "alpha".
        let mut gate = ndarray::Array2::<f32>::zeros((num_features, hidden));
        for f in 0..num_features {
            gate[[f, 0]] = 1.0;
        }
        let arch = &weights.arch;
        let gate_key = arch.ffn_gate_key(0);
        insert_tensor(&mut weights, &gate_key, gate);

        let result = compute_gate_top_tokens(&weights, &toks, 0, num_features, &ww_ids, &ww_embed);
        assert_eq!(result.len(), num_features);
        assert!(
            result.iter().all(|s| s == "alpha"),
            "all features should pick the only whole-word"
        );
    }

    #[test]
    fn run_clustering_pipeline_writes_outputs_for_non_empty_data() {
        // Synthetic two-cluster setup:
        //   - 6 directions in a 4-D space, 3 along [1,0,0,0] and 3 along [0,1,0,0]
        //   - k-means should split them cleanly into 2 clusters
        //   - The function writes relation_clusters.json + feature_clusters.jsonl
        //
        // We don't assert on cluster *labels* — those depend on
        // the auto-labelling reaching `data/wikidata_triples.json` /
        // `data/wordnet_relations.json` files which may or may not
        // exist in the test cwd. We only verify the file shapes.
        let words = ["paris", "france", "berlin", "germany"];
        let toks = vocab_tokenizer(&words);
        let hidden = 4;
        let vocab_size = 5;
        let mut embed = ndarray::Array2::<f32>::zeros((vocab_size, hidden));
        for i in 0..4 {
            embed[[i + 1, i]] = 1.0;
        }
        let weights = weights_with_embed(embed, vocab_size);

        // 6 features × 4-D directions: half along axis 0, half along axis 1.
        let mut directions = Vec::with_capacity(6 * hidden);
        for _ in 0..3 {
            directions.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
        }
        for _ in 0..3 {
            directions.extend_from_slice(&[0.0, 1.0, 0.0, 0.0]);
        }
        let features = (0..6).map(|f| (0usize, f)).collect();
        let top_tokens = vec![
            "paris".into(),
            "france".into(),
            "berlin".into(),
            "germany".into(),
            "rome".into(),
            "italy".into(),
        ];
        let output_tokens = top_tokens.clone();

        let tmp = tempfile::tempdir().unwrap();
        let mut cb = crate::extract::callbacks::SilentBuildCallbacks;
        let result = run_clustering_pipeline(
            ClusterData {
                directions,
                features,
                top_tokens,
                input_tokens: Vec::new(),
                output_tokens,
            },
            hidden,
            &weights,
            &toks,
            tmp.path(),
            &mut cb,
        );
        assert!(result.is_ok(), "pipeline returned: {result:?}");

        // relation_clusters.json must exist and parse as a ClusterResult.
        let json_path = tmp.path().join(RELATION_CLUSTERS_JSON);
        assert!(json_path.exists(), "relation_clusters.json written");
        let json = std::fs::read_to_string(&json_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.get("k").is_some(), "k field present");
        assert!(parsed.get("centres").is_some(), "centres field present");
        assert!(parsed.get("labels").is_some(), "labels field present");
        assert!(parsed.get("counts").is_some(), "counts field present");

        // feature_clusters.jsonl must have one line per feature.
        let assign_path = tmp.path().join(FEATURE_CLUSTERS_JSONL);
        assert!(assign_path.exists(), "feature_clusters.jsonl written");
        let jsonl = std::fs::read_to_string(&assign_path).unwrap();
        let lines: Vec<_> = jsonl.lines().collect();
        assert_eq!(lines.len(), 6, "one line per feature");
        for line in lines {
            let r: serde_json::Value = serde_json::from_str(line).unwrap();
            assert!(r.get("l").is_some());
            assert!(r.get("f").is_some());
            assert!(r.get("c").is_some());
        }
    }

    #[test]
    fn run_clustering_pipeline_errors_on_shape_mismatch() {
        // directions.len() != n_features × hidden_size → from_shape_vec
        // returns Err → we wrap as VindexError::Parse.
        let toks = vocab_tokenizer(&["x"]);
        let embed = ndarray::Array2::<f32>::zeros((2, 4));
        let weights = weights_with_embed(embed, 2);
        let tmp = tempfile::tempdir().unwrap();
        let mut cb = crate::extract::callbacks::SilentBuildCallbacks;

        let result = run_clustering_pipeline(
            ClusterData {
                // 3 features but only 8 floats → should be 12 (3 × 4).
                directions: vec![1.0; 8],
                features: vec![(0, 0), (0, 1), (0, 2)],
                top_tokens: vec!["a".into(), "b".into(), "c".into()],
                input_tokens: Vec::new(),
                output_tokens: vec!["a".into(), "b".into(), "c".into()],
            },
            4,
            &weights,
            &toks,
            tmp.path(),
            &mut cb,
        );
        assert!(result.is_err(), "shape mismatch must error");
        let err = result.unwrap_err();
        assert!(err.to_string().contains("cluster data shape"));
    }
}
