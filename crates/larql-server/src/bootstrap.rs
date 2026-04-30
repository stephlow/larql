//! Server bootstrap and vindex loading helpers.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use larql_vindex::format::filenames::*;
use larql_vindex::{
    load_vindex_config, load_vindex_embeddings, load_vindex_tokenizer, PatchedVindex,
    SilentLoadCallbacks, VectorIndex,
};
use tokio::sync::RwLock;
use tracing::info;

use crate::state::{load_probe_labels, model_id_from_name, LoadedModel};

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

pub fn parse_layer_range(s: &str) -> Result<(usize, usize), BoxError> {
    let parts: Vec<&str> = s.splitn(2, '-').collect();
    if parts.len() != 2 {
        return Err(format!("--layers: expected 'START-END' (e.g. '0-19'), got '{s}'").into());
    }
    let start: usize = parts[0]
        .trim()
        .parse()
        .map_err(|_| format!("--layers: invalid start '{}'", parts[0]))?;
    let end: usize = parts[1]
        .trim()
        .parse()
        .map_err(|_| format!("--layers: invalid end '{}'", parts[1]))?;
    if end < start {
        return Err(format!("--layers: end ({end}) must be >= start ({start})").into());
    }
    Ok((start, end + 1))
}

#[derive(Clone)]
pub struct LoadVindexOptions {
    pub no_infer: bool,
    pub ffn_only: bool,
    pub embed_only: bool,
    pub layer_range: Option<(usize, usize)>,
    pub max_gate_cache_layers: usize,
    pub max_q4k_cache_layers: usize,
    pub hnsw: Option<usize>,
    pub warmup_hnsw: bool,
    pub release_mmap_after_request: bool,
    pub expert_filter: Option<(usize, usize)>,
    /// Fine-grained per-(layer, expert) ownership.  When `Some`, takes
    /// precedence over `expert_filter` for `run_expert`'s ownership check
    /// and for the HNSW / Metal warmup loops.  Loaded from `--units` JSON.
    pub unit_filter: Option<Arc<std::collections::HashSet<(usize, usize)>>>,
}

/// JSON layout for the `--units` manifest.  Each value is a list of inclusive
/// `[start, end]` expert-id ranges, keyed by layer index (as a string for
/// JSON-object compatibility).
#[derive(serde::Deserialize)]
pub struct UnitManifest {
    pub layer_experts: std::collections::BTreeMap<String, Vec<[usize; 2]>>,
}

impl UnitManifest {
    /// Expand the per-layer range list into the flat `(layer, expert_id)`
    /// set used by ownership checks.  Reports the first malformed entry in
    /// the error path so the operator can fix it without grepping.
    pub fn into_unit_set(self) -> Result<std::collections::HashSet<(usize, usize)>, BoxError> {
        let mut units = std::collections::HashSet::new();
        for (layer_str, ranges) in self.layer_experts {
            let layer: usize = layer_str.parse().map_err(|_| -> BoxError {
                format!("--units: layer key '{layer_str}' is not a valid usize").into()
            })?;
            for [start, end] in ranges {
                if end < start {
                    return Err(format!(
                        "--units: layer {layer}: end ({end}) must be >= start ({start})"
                    )
                    .into());
                }
                for eid in start..=end {
                    units.insert((layer, eid));
                }
            }
        }
        Ok(units)
    }
}

/// Parse `--units PATH` into the canonical `(layer, expert_id)` ownership set.
pub fn parse_unit_manifest(
    path: &Path,
) -> Result<std::collections::HashSet<(usize, usize)>, BoxError> {
    let bytes = std::fs::read(path)
        .map_err(|e| -> BoxError { format!("--units: read {}: {e}", path.display()).into() })?;
    let manifest: UnitManifest = serde_json::from_slice(&bytes)
        .map_err(|e| -> BoxError { format!("--units: parse {}: {e}", path.display()).into() })?;
    manifest.into_unit_set()
}

pub fn load_single_vindex(
    path_str: &str,
    opts: LoadVindexOptions,
) -> Result<LoadedModel, BoxError> {
    let path = if larql_vindex::is_hf_path(path_str) {
        info!("Resolving HuggingFace path: {}", path_str);
        larql_vindex::resolve_hf_vindex(path_str)?
    } else {
        PathBuf::from(path_str)
    };

    info!("Loading: {}", path.display());

    let config = load_vindex_config(&path)?;
    let model_name = config.model.clone();
    let id = model_id_from_name(&model_name);

    let mut cb = SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex_with_range(&path, &mut cb, opts.layer_range)?;
    if opts.max_gate_cache_layers > 0 {
        index.set_gate_cache_max_layers(opts.max_gate_cache_layers);
        info!(
            "  Gate cache: LRU, max {} layers",
            opts.max_gate_cache_layers
        );
    }
    if opts.max_q4k_cache_layers > 0 {
        index.set_q4k_ffn_cache_max_layers(opts.max_q4k_cache_layers);
        info!(
            "  Q4K FFN cache: LRU, max {} layers",
            opts.max_q4k_cache_layers
        );
    }
    if let Some(ef) = opts.hnsw {
        index.enable_hnsw(ef);
        info!("  HNSW gate KNN: enabled (ef_search={ef})");
        if opts.warmup_hnsw {
            let t0 = std::time::Instant::now();
            index.warmup_hnsw_all_layers();
            let owned = match opts.layer_range {
                Some((s, e)) => e - s,
                None => config.num_layers,
            };
            info!(
                "  HNSW warmup: built {} owned layer(s) in {:.2?}",
                owned,
                t0.elapsed()
            );
        }
    }
    let total_features: usize = config.layers.iter().map(|l| l.num_features).sum();

    let has_weights = config.has_model_weights
        || config.extract_level == larql_vindex::ExtractLevel::Inference
        || config.extract_level == larql_vindex::ExtractLevel::All;

    if let Some((start, end)) = opts.layer_range {
        info!("  Layers: {start}–{} (of {})", end - 1, config.num_layers);
    }
    info!(
        "  Model: {} ({} layers, {} features)",
        model_name, config.num_layers, total_features
    );

    if !opts.embed_only {
        match index.load_down_features(&path) {
            Ok(()) => info!("  Down features: loaded (mmap walk enabled)"),
            Err(_) => info!("  Down features: not available"),
        }
        if let Ok(()) = index.load_up_features(&path) {
            info!("  Up features: loaded (full mmap FFN)")
        }
        if index.has_down_features_q4k() {
            info!(
                "  Down features Q4K: loaded (W2 — per-feature decode skips q4k_ffn_layer cache)"
            );
        }
    }

    if opts.ffn_only || opts.embed_only {
        let reason = if opts.embed_only {
            "--embed-only"
        } else {
            "--ffn-only"
        };
        info!("  Warmup: skipped ({reason})");
    } else {
        index.warmup();
        info!("  Warmup: done");
    }

    let (embeddings, embed_scale) = load_vindex_embeddings(&path)?;
    info!(
        "  Embeddings: {}x{}",
        embeddings.shape()[0],
        embeddings.shape()[1]
    );

    let embed_store = if opts.embed_only {
        match crate::embed_store::EmbedStoreF16::open(
            &path,
            embed_scale,
            config.vocab_size,
            config.hidden_size,
            5_000,
        ) {
            Ok(store) => {
                let f16_bytes = config.vocab_size * config.hidden_size * 2;
                info!(
                    "  Embed store: f16 mmap ({:.1} GB, L1 cap 5000 tokens)",
                    f16_bytes as f64 / 1e9
                );
                Some(Arc::new(store))
            }
            Err(e) => {
                info!("  Embed store: f16 mmap unavailable ({e}), using f32 heap");
                None
            }
        }
    } else {
        None
    };

    let tokenizer = load_vindex_tokenizer(&path)?;
    let patched = PatchedVindex::new(index);

    let probe_labels = load_probe_labels(&path);
    if !probe_labels.is_empty() {
        info!("  Labels: {} probe-confirmed", probe_labels.len());
    }

    let infer_disabled = opts.no_infer || opts.ffn_only || opts.embed_only;
    if opts.embed_only {
        info!("  Mode: embed-service (--embed-only)");
        info!("  Infer: disabled (embed-service mode)");
    } else if opts.ffn_only {
        info!("  Mode: ffn-service (--ffn-only)");
        info!("  Infer: disabled (FFN-service mode)");
    } else if opts.no_infer {
        info!("  Infer: disabled (--no-infer)");
    } else if has_weights {
        info!("  Infer: available (weights detected, will lazy-load on first request)");
    } else {
        info!("  Infer: not available (no model weights in vindex)");
    }

    if opts.release_mmap_after_request {
        info!("  Mmap release: enabled (MADV_DONTNEED after each walk-ffn request)");
    }

    if let Some((start, end)) = opts.expert_filter {
        info!("  Experts: {start}–{end} (shard filter)");
    }

    let num_layers = config.num_layers;
    Ok(LoadedModel {
        id,
        path,
        config,
        patched: RwLock::new(patched),
        embeddings,
        embed_scale,
        tokenizer,
        infer_disabled,
        ffn_only: opts.ffn_only,
        embed_only: opts.embed_only,
        embed_store,
        release_mmap_after_request: opts.release_mmap_after_request,
        weights: std::sync::OnceLock::new(),
        probe_labels,
        ffn_l2_cache: crate::ffn_l2_cache::FfnL2Cache::new(num_layers),
        expert_filter: opts.expert_filter,
        unit_filter: opts.unit_filter.clone(),
        #[cfg(feature = "metal-experts")]
        metal_backend: std::sync::OnceLock::new(),
        #[cfg(feature = "metal-experts")]
        moe_scratches: std::sync::Mutex::new(std::collections::HashMap::new()),
    })
}

pub fn discover_vindexes(dir: &Path) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() && p.join(INDEX_JSON).exists() {
                paths.push(p);
            }
        }
    }
    paths.sort();
    paths
}

pub fn normalize_serve_alias(args: Vec<String>) -> Vec<String> {
    if args.len() > 1 && args[1] == "serve" {
        std::iter::once(args[0].clone())
            .chain(args[2..].iter().cloned())
            .collect()
    } else {
        args
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_temp_dir(name: &str) -> PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "larql-server-bootstrap-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    // ── Unit-manifest parser ─────────────────────────────────────────────
    //
    // The JSON shape the operator hands the server must round-trip through
    // `parse_unit_manifest` into a deterministic ownership set.  Tests
    // cover: well-formed multi-range manifest, bad layer key, reversed
    // range, missing file.  The data shape is exercised end-to-end here so
    // ownership-check and warmup loops can rely on it without having to
    // re-validate.

    fn write_units_file(dir: &Path, body: &str) -> PathBuf {
        let path = dir.join("units.json");
        std::fs::write(&path, body).unwrap();
        path
    }

    #[test]
    fn parse_unit_manifest_round_trips_per_layer_ranges() {
        let dir = unique_temp_dir("units-ok");
        let path = write_units_file(
            &dir,
            r#"{"layer_experts": {"0": [[0,2]], "3": [[5,7],[10,10]]}}"#,
        );
        let units = parse_unit_manifest(&path).unwrap();
        // Layer 0: experts 0..=2 → (0,0), (0,1), (0,2)
        // Layer 3: experts 5..=7 + 10 → (3,5), (3,6), (3,7), (3,10)
        let expected: std::collections::HashSet<(usize, usize)> =
            [(0, 0), (0, 1), (0, 2), (3, 5), (3, 6), (3, 7), (3, 10)]
                .into_iter()
                .collect();
        assert_eq!(units, expected);
    }

    #[test]
    fn parse_unit_manifest_rejects_non_numeric_layer_key() {
        let dir = unique_temp_dir("units-bad-layer");
        let path = write_units_file(&dir, r#"{"layer_experts": {"oops": [[0,2]]}}"#);
        let err = parse_unit_manifest(&path).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("layer key 'oops'"), "got: {msg}");
    }

    #[test]
    fn parse_unit_manifest_rejects_reversed_range() {
        let dir = unique_temp_dir("units-bad-range");
        let path = write_units_file(&dir, r#"{"layer_experts": {"0": [[5,2]]}}"#);
        let err = parse_unit_manifest(&path).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("end (2) must be >= start (5)"), "got: {msg}");
    }

    #[test]
    fn parse_unit_manifest_missing_file_reports_path() {
        let bogus = PathBuf::from("/nonexistent/larql-units-not-here.json");
        let err = parse_unit_manifest(&bogus).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("read"),
            "msg should mention read failure: {msg}"
        );
        assert!(
            msg.contains(bogus.to_str().unwrap()),
            "msg should name path: {msg}"
        );
    }

    #[test]
    fn parse_unit_manifest_accepts_empty_object() {
        // Operator may want to test the wiring without owning any units —
        // empty manifest should yield an empty set, not error.
        let dir = unique_temp_dir("units-empty");
        let path = write_units_file(&dir, r#"{"layer_experts": {}}"#);
        let units = parse_unit_manifest(&path).unwrap();
        assert!(units.is_empty());
    }

    #[test]
    fn parse_layer_range_accepts_inclusive_cli_range() {
        assert_eq!(parse_layer_range("0-19").unwrap(), (0, 20));
        assert_eq!(parse_layer_range(" 2 - 2 ").unwrap(), (2, 3));
    }

    #[test]
    fn parse_layer_range_rejects_bad_shapes() {
        assert!(parse_layer_range("0").is_err());
        assert!(parse_layer_range("x-2").is_err());
        assert!(parse_layer_range("2-x").is_err());
        assert!(parse_layer_range("3-2").is_err());
    }

    #[test]
    fn normalize_serve_alias_removes_subcommand() {
        let filtered = normalize_serve_alias(vec![
            "larql-server".into(),
            "serve".into(),
            "model.vindex".into(),
        ]);
        assert_eq!(filtered, vec!["larql-server", "model.vindex"]);
    }

    #[test]
    fn normalize_serve_alias_leaves_non_alias_args_unchanged() {
        let args = vec!["larql-server".into(), "model.vindex".into()];
        assert_eq!(normalize_serve_alias(args.clone()), args);
    }

    #[test]
    fn discover_vindexes_returns_sorted_dirs_with_index_json() {
        let dir = unique_temp_dir("discover");
        let b = dir.join("b.vindex");
        let a = dir.join("a.vindex");
        let ignored = dir.join("ignored.vindex");
        std::fs::create_dir_all(&b).unwrap();
        std::fs::create_dir_all(&a).unwrap();
        std::fs::create_dir_all(&ignored).unwrap();
        std::fs::write(b.join(INDEX_JSON), "{}").unwrap();
        std::fs::write(a.join(INDEX_JSON), "{}").unwrap();

        let paths = discover_vindexes(&dir);
        assert_eq!(paths, vec![a, b]);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn load_options_are_copyable() {
        let opts = LoadVindexOptions {
            no_infer: true,
            ffn_only: false,
            embed_only: false,
            layer_range: Some((0, 2)),
            max_gate_cache_layers: 1,
            max_q4k_cache_layers: 2,
            hnsw: Some(200),
            warmup_hnsw: true,
            release_mmap_after_request: true,
            expert_filter: Some((3, 4)),
            unit_filter: None,
        };
        let copied = opts.clone();
        assert!(copied.no_infer);
        assert_eq!(copied.layer_range, Some((0, 2)));
        assert_eq!(copied.expert_filter, Some((3, 4)));
    }
}
