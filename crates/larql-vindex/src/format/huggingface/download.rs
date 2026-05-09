//! HuggingFace download path — `hf://` resolution, snapshot cache
//! traversal, conditional ETag-based fetch.
//!
//! Carved out of the monolithic `huggingface.rs` in the 2026-04-25
//! reorg. See `super::mod.rs` for the module map.

use std::path::PathBuf;

use crate::error::VindexError;
use crate::format::filenames::*;

use super::publish::get_hf_token;
use super::{VINDEX_CORE_FILES, VINDEX_WEIGHT_FILES};

/// Which side of the HF API a repo lives on. Datasets are how vindexes
/// are stored; Models is the canonical home of safetensors / GGUF / etc.
/// Both share the same blob-cache layout but differ in the URL prefix
/// and the `{datasets,models}--` cache-dir prefix.
#[derive(Clone, Copy)]
enum RepoKind {
    Dataset,
    Model,
}

impl RepoKind {
    fn url_segment(self) -> &'static str {
        match self {
            RepoKind::Dataset => "datasets/",
            RepoKind::Model => "",
        }
    }

    fn cache_prefix(self) -> &'static str {
        match self {
            RepoKind::Dataset => "datasets--",
            RepoKind::Model => "models--",
        }
    }
}

/// Resolve an `hf://` path to a local directory, downloading if needed.
///
/// Supports:
/// - `hf://user/repo` — downloads the full dataset repo
/// - `hf://user/repo@revision` — specific revision/tag
///
/// Files are cached in the HuggingFace cache directory (~/.cache/huggingface/).
/// Only downloads files that don't already exist locally.
pub fn resolve_hf_vindex(hf_path: &str) -> Result<PathBuf, VindexError> {
    let path = hf_path
        .strip_prefix("hf://")
        .ok_or_else(|| VindexError::Parse(format!("not an hf:// path: {hf_path}")))?;

    // Parse repo and optional revision
    let (repo_id, revision) = if let Some((repo, rev)) = path.split_once('@') {
        (repo.to_string(), Some(rev.to_string()))
    } else {
        (path.to_string(), None)
    };

    // Use hf-hub to download
    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| VindexError::Parse(format!("HuggingFace API init failed: {e}")))?;

    let repo = if let Some(ref rev) = revision {
        api.repo(hf_hub::Repo::with_revision(
            repo_id.clone(),
            hf_hub::RepoType::Dataset,
            rev.clone(),
        ))
    } else {
        api.repo(hf_hub::Repo::new(
            repo_id.clone(),
            hf_hub::RepoType::Dataset,
        ))
    };

    // Download index.json first (small, tells us what we need)
    let index_path = repo.get(INDEX_JSON).map_err(|e| {
        VindexError::Parse(format!(
            "failed to download index.json from hf://{}: {e}",
            repo_id
        ))
    })?;

    let vindex_dir = index_path
        .parent()
        .ok_or_else(|| VindexError::Parse("cannot determine vindex directory".into()))?
        .to_path_buf();

    // Download core files (needed for browse)
    for filename in VINDEX_CORE_FILES {
        if *filename == INDEX_JSON {
            continue; // already downloaded
        }
        let _ = repo.get(filename); // optional file, skip if missing
    }

    Ok(vindex_dir)
}

/// Download additional weight files for inference/compile.
/// Called lazily when INFER or COMPILE is first used.
pub fn download_hf_weights(hf_path: &str) -> Result<(), VindexError> {
    let path = hf_path
        .strip_prefix("hf://")
        .ok_or_else(|| VindexError::Parse(format!("not an hf:// path: {hf_path}")))?;

    let (repo_id, revision) = if let Some((repo, rev)) = path.split_once('@') {
        (repo.to_string(), Some(rev.to_string()))
    } else {
        (path.to_string(), None)
    };

    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| VindexError::Parse(format!("HuggingFace API init failed: {e}")))?;

    let repo = if let Some(ref rev) = revision {
        api.repo(hf_hub::Repo::with_revision(
            repo_id.clone(),
            hf_hub::RepoType::Dataset,
            rev.clone(),
        ))
    } else {
        api.repo(hf_hub::Repo::new(
            repo_id.clone(),
            hf_hub::RepoType::Dataset,
        ))
    };

    for filename in VINDEX_WEIGHT_FILES {
        let _ = repo.get(filename); // optional, skip if not in repo
    }

    Ok(())
}

/// Re-exported from hf-hub 0.5 so callers don't have to depend on
/// `hf_hub` directly. Implement this trait on an `indicatif::ProgressBar`
/// wrapper (or similar) to get per-file progress + resume behaviour out
/// of [`resolve_hf_vindex_with_progress`].
pub use hf_hub::api::Progress as DownloadProgress;

/// Check hf-hub's on-disk cache for `filename` and return `(path, size)`
/// iff a ready-to-use copy exists whose content hash matches what HF
/// reports on the remote.
///
/// hf-hub 0.5 lays the cache out as:
///
///   ```text
///   ~/.cache/huggingface/hub/datasets--{owner}--{name}/
///     ├── blobs/<etag>            actual file bytes
///     └── snapshots/<commit>/     symlinks → blobs
///         └── <filename>
///   ```
///
/// The etag is HF's content identifier: for LFS-tracked files it's the
/// SHA-256 oid; for git-tracked small files it's the git blob SHA-1.
/// Either way it uniquely identifies the bytes — so if `blobs/<etag>`
/// exists locally, the content matches the remote and we can skip the
/// download. This is stronger than the old size-only check: if the
/// remote file changes (new commit rewriting the same filename), the
/// etag changes, the cache probe misses, and we re-download.
///
/// The cost is one HEAD request per file. On a 10-file vindex that's a
/// few hundred ms vs the GB we'd re-download otherwise — cheap.
///
/// Returns `None` on any failure (HEAD error, cache missing, etag
/// absent, etc.); the caller falls back to `download_with_progress`.
fn cached_snapshot_file(
    kind: RepoKind,
    repo_id: &str,
    revision: Option<&str>,
    filename: &str,
) -> Option<(PathBuf, u64)> {
    let (etag, size) = head_etag_and_size(kind, repo_id, revision, filename)?;
    let repo_dir = hf_cache_repo_dir(kind, repo_id)?;
    let blob_path = repo_dir.join("blobs").join(&etag);
    let meta = std::fs::metadata(&blob_path).ok()?;
    if !meta.is_file() {
        return None;
    }
    // Size mismatch shouldn't happen if the etag matched, but treat it
    // as cache-miss defensively.
    if meta.len() != size {
        return None;
    }

    // Return the snapshot path (symlink → blob) if the repo has one,
    // otherwise the blob path itself. Either works — the caller only
    // needs a file it can open.
    let snapshots = repo_dir.join("snapshots");
    if let Ok(entries) = std::fs::read_dir(&snapshots) {
        for entry in entries.flatten() {
            let snap_file = entry.path().join(filename);
            if snap_file.exists() {
                return Some((snap_file, size));
            }
        }
    }
    // Fall back to the pinned revision (if any) even if the symlink is
    // missing — the blob still has the bytes.
    if let Some(rev) = revision {
        let snap_file = snapshots.join(rev).join(filename);
        if snap_file.exists() {
            return Some((snap_file, size));
        }
    }
    Some((blob_path, size))
}

/// Issue a HEAD against HF's file-resolve endpoint for this repo+file
/// and return `(etag, size)` from the response headers. HF redirects
/// LFS files to S3 which also returns an etag, so we must follow
/// redirects. Returns `None` for any failure: bad status, missing
/// headers, malformed size, etc.
fn head_etag_and_size(
    kind: RepoKind,
    repo_id: &str,
    revision: Option<&str>,
    filename: &str,
) -> Option<(String, u64)> {
    let rev = revision.unwrap_or("main");
    let url = format!(
        "https://huggingface.co/{}{repo_id}/resolve/{rev}/{filename}",
        kind.url_segment()
    );
    let token = get_hf_token().ok();

    // **No redirects.** HF LFS files 302 → S3, and `X-Linked-Etag` +
    // `X-Linked-Size` (the stable LFS oid + content length) only exist
    // on HF's own first response. Following the redirect would lose
    // those headers and leave us with S3's multipart ETag, which is
    // MD5-based and doesn't match how hf-hub names blob files.
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .ok()?;
    let mut req = client.head(&url);
    if let Some(t) = token {
        req = req.header("Authorization", format!("Bearer {t}"));
    }
    let resp = req.send().ok()?;
    // Accept both 2xx (git-tracked small files stay on HF) and 3xx
    // (LFS files redirect to S3; the 302 carries the linked-etag we want).
    let status = resp.status();
    if !status.is_success() && !status.is_redirection() {
        return None;
    }

    // Prefer `X-Linked-Etag` when present (LFS oid = SHA256, stable).
    // Fall back to `ETag` for git-tracked files.
    let raw_etag = resp
        .headers()
        .get("X-Linked-Etag")
        .or_else(|| resp.headers().get("ETag"))
        .and_then(|v| v.to_str().ok())?;
    let etag = strip_etag_quoting(raw_etag);
    let size_hdr = resp
        .headers()
        .get("X-Linked-Size")
        .or_else(|| resp.headers().get("Content-Length"))
        .and_then(|v| v.to_str().ok())?;
    let size: u64 = size_hdr.parse().ok()?;
    Some((etag, size))
}

/// Normalise an HTTP ETag header to the raw content hash hf-hub uses
/// as blob filenames. Handles:
///   * strong etag: `"abc123"` → `abc123`
///   * weak etag:   `W/"abc123"` → `abc123`
fn strip_etag_quoting(raw: &str) -> String {
    let trimmed = raw.trim();
    let no_weak = trimmed.strip_prefix("W/").unwrap_or(trimmed);
    no_weak.trim_matches('"').to_string()
}

/// Resolve the hf-hub cache directory for a repo: the root of
/// `~/.cache/huggingface/hub/{datasets,models}--{owner}--{name}/`. Honours
/// `HF_HOME` and `HUGGINGFACE_HUB_CACHE` env overrides that hf-hub itself
/// respects.
fn hf_cache_repo_dir(kind: RepoKind, repo_id: &str) -> Option<PathBuf> {
    let hub_root = if let Ok(hub) = std::env::var("HUGGINGFACE_HUB_CACHE") {
        PathBuf::from(hub)
    } else if let Ok(hf_home) = std::env::var("HF_HOME") {
        PathBuf::from(hf_home).join("hub")
    } else {
        let home = std::env::var("HOME").ok()?;
        PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("hub")
    };
    let safe = repo_id.replace('/', "--");
    Some(hub_root.join(format!("{}{safe}", kind.cache_prefix())))
}

/// Like [`resolve_hf_vindex`], but drives a progress reporter per file.
/// hf-hub handles `.incomplete` partial-file resume internally — if the
/// download is interrupted, the next call picks up from where it left off.
///
/// Also honours the local cache: before each file, we check the
/// `snapshots/` tree for an already-downloaded copy whose size matches
/// the remote. Matches fire `init → update(size) → finish` on the
/// progress reporter with no HTTP traffic, so cached pulls complete in
/// milliseconds and the bar snaps to 100 %.
///
/// `progress` is a factory: called once per file with the filename.
/// Return a fresh `DownloadProgress` — typically an
/// `indicatif::ProgressBar` fetched from a `MultiProgress`.
pub fn resolve_hf_vindex_with_progress<F, P>(
    hf_path: &str,
    mut progress: F,
) -> Result<PathBuf, VindexError>
where
    F: FnMut(&str) -> P,
    P: DownloadProgress,
{
    let path = hf_path
        .strip_prefix("hf://")
        .ok_or_else(|| VindexError::Parse(format!("not an hf:// path: {hf_path}")))?;

    let (repo_id, revision) = if let Some((repo, rev)) = path.split_once('@') {
        (repo.to_string(), Some(rev.to_string()))
    } else {
        (path.to_string(), None)
    };

    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| VindexError::Parse(format!("HuggingFace API init failed: {e}")))?;

    let repo = if let Some(ref rev) = revision {
        api.repo(hf_hub::Repo::with_revision(
            repo_id.clone(),
            hf_hub::RepoType::Dataset,
            rev.clone(),
        ))
    } else {
        api.repo(hf_hub::Repo::new(
            repo_id.clone(),
            hf_hub::RepoType::Dataset,
        ))
    };

    // Helper: one file, with cache short-circuit. Returns the resolved
    // on-disk path. The cache check fires the progress reporter so the
    // bar shows a filled-to-100% track tagged with the filename — users
    // see that the file was served from cache, not re-downloaded.
    let mut fetch = |filename: &str, label: &str| -> Option<PathBuf> {
        if let Some((cached_path, size)) =
            cached_snapshot_file(RepoKind::Dataset, &repo_id, revision.as_deref(), filename)
        {
            // Tag the progress message so the bar visibly distinguishes
            // "cached" from "just downloaded very fast". Callers rendering
            // the bar see the prefix at init time and can restyle.
            let mut p = progress(label);
            let tagged = format!("{filename} [cached]");
            p.init(size as usize, &tagged);
            p.update(size as usize);
            p.finish();
            return Some(cached_path);
        }
        repo.download_with_progress(filename, progress(label)).ok()
    };

    // index.json drives everything — we need its snapshot dir to know
    // where the rest of the files live. Cache-hit or download.
    let index_path = fetch(INDEX_JSON, INDEX_JSON).ok_or_else(|| {
        VindexError::Parse(format!("failed to fetch index.json from hf://{repo_id}"))
    })?;
    let vindex_dir = index_path
        .parent()
        .ok_or_else(|| VindexError::Parse("cannot determine vindex directory".into()))?
        .to_path_buf();

    for filename in VINDEX_CORE_FILES {
        if *filename == INDEX_JSON {
            continue;
        }
        // Optional files — ignore failures (missing from repo is fine).
        let _ = fetch(filename, filename);
    }
    Ok(vindex_dir)
}

/// Filenames that we never want to pull from a model repo even when they're
/// listed in the siblings response. PyTorch `.bin` weights are skipped when
/// safetensors are present (the standard HF mirror has both); image and
/// metadata files are noise; .gguf is a different acquisition path.
fn want_model_file(name: &str) -> bool {
    let lower = name.to_lowercase();
    // Junk we never want.
    if lower.ends_with(".png")
        || lower.ends_with(".jpg")
        || lower.ends_with(".jpeg")
        || lower.ends_with(".gif")
        || lower.ends_with(".svg")
        || lower.ends_with(".gguf")
        || lower.ends_with(".onnx")
        || lower == ".gitattributes"
        || lower.starts_with("readme")
        || lower.starts_with("license")
    {
        return false;
    }
    // Skip pickle/torch — we load via safetensors. Keeping these would
    // double the download size on most HF model repos.
    if lower.ends_with(".bin") || lower.ends_with(".pt") || lower.ends_with(".pth") {
        return false;
    }
    true
}

/// Resolve an `hf://` model repo path to a local snapshot directory,
/// downloading the safetensors + tokenizer + config sidecar files needed
/// for `larql convert safetensors-to-vindex`. Mirrors
/// [`resolve_hf_vindex_with_progress`] but talks to the model side of the
/// HF API (`models/...`) and enumerates files via the repo `info()` call
/// instead of a fixed list, so sharded checkpoints (Qwen3 4B/27B) Just Work.
///
/// Skips PyTorch `.bin` shards when safetensors are also present in the
/// repo (`want_model_file`) — saves several GB on the typical mirror.
pub fn resolve_hf_model_with_progress<F, P>(
    hf_path: &str,
    mut progress: F,
) -> Result<PathBuf, VindexError>
where
    F: FnMut(&str) -> P,
    P: DownloadProgress,
{
    let path = hf_path
        .strip_prefix("hf://")
        .ok_or_else(|| VindexError::Parse(format!("not an hf:// path: {hf_path}")))?;

    let (repo_id, revision) = if let Some((repo, rev)) = path.split_once('@') {
        (repo.to_string(), Some(rev.to_string()))
    } else {
        (path.to_string(), None)
    };

    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| VindexError::Parse(format!("HuggingFace API init failed: {e}")))?;

    let repo = if let Some(ref rev) = revision {
        api.repo(hf_hub::Repo::with_revision(
            repo_id.clone(),
            hf_hub::RepoType::Model,
            rev.clone(),
        ))
    } else {
        api.repo(hf_hub::Repo::new(repo_id.clone(), hf_hub::RepoType::Model))
    };

    let info = repo
        .info()
        .map_err(|e| VindexError::Parse(format!("HF info failed for {hf_path}: {e}")))?;

    let mut wanted: Vec<&str> = info
        .siblings
        .iter()
        .map(|s| s.rfilename.as_str())
        .filter(|n| want_model_file(n))
        .collect();
    wanted.sort();

    if wanted.is_empty() {
        return Err(VindexError::Parse(format!(
            "no usable model files in {hf_path} (siblings: {})",
            info.siblings.len()
        )));
    }

    let mut snapshot_dir: Option<PathBuf> = None;
    let mut fetch = |filename: &str| -> Option<PathBuf> {
        if let Some((cached_path, size)) =
            cached_snapshot_file(RepoKind::Model, &repo_id, revision.as_deref(), filename)
        {
            let mut p = progress(filename);
            let tagged = format!("{filename} [cached]");
            p.init(size as usize, &tagged);
            p.update(size as usize);
            p.finish();
            return Some(cached_path);
        }
        repo.download_with_progress(filename, progress(filename))
            .ok()
    };

    for filename in &wanted {
        if let Some(p) = fetch(filename) {
            if snapshot_dir.is_none() {
                snapshot_dir = p.parent().map(|d| d.to_path_buf());
            }
        }
    }

    snapshot_dir.ok_or_else(|| {
        VindexError::Parse(format!(
            "downloaded zero files from {hf_path} — check repo access"
        ))
    })
}

#[cfg(test)]
mod tests {
    //! Unit tests for the pure helpers in this module — `strip_etag_quoting`,
    //! `want_model_file`, and `hf_cache_repo_dir`. The hf_hub-bound functions
    //! (`resolve_hf_vindex`, `download_hf_weights`, the *_with_progress
    //! variants) need an HF_ENDPOINT-mocking harness; that's tracked as a
    //! separate follow-up because the routing path in
    //! `head_etag_and_size` currently hardcodes `https://huggingface.co`.
    use super::*;
    use serial_test::serial;

    // ─── strip_etag_quoting ────────────────────────────────────────────

    #[test]
    fn strip_etag_quoting_unquotes_strong_etag() {
        assert_eq!(strip_etag_quoting("\"abc123\""), "abc123");
    }

    #[test]
    fn strip_etag_quoting_handles_weak_etag() {
        // `W/"abc123"` — weak prefix dropped, inner quoted hash returned.
        assert_eq!(strip_etag_quoting("W/\"abc123\""), "abc123");
    }

    #[test]
    fn strip_etag_quoting_trims_surrounding_whitespace() {
        assert_eq!(strip_etag_quoting("  \"abc\"  "), "abc");
    }

    #[test]
    fn strip_etag_quoting_handles_unquoted_input() {
        // Defensive: HF should always quote, but if it doesn't, return
        // the input unchanged after trimming.
        assert_eq!(strip_etag_quoting("plainhash"), "plainhash");
    }

    #[test]
    fn strip_etag_quoting_handles_empty_string() {
        assert_eq!(strip_etag_quoting(""), "");
    }

    #[test]
    fn strip_etag_quoting_handles_weak_unquoted() {
        // Edge case: weak prefix without quotes — strip the prefix only.
        assert_eq!(strip_etag_quoting("W/abc"), "abc");
    }

    // ─── want_model_file ───────────────────────────────────────────────

    #[test]
    fn want_model_file_accepts_safetensors() {
        assert!(want_model_file("model.safetensors"));
        assert!(want_model_file("model-00001-of-00002.safetensors"));
    }

    #[test]
    fn want_model_file_accepts_config_and_tokenizer() {
        assert!(want_model_file("config.json"));
        assert!(want_model_file("tokenizer.json"));
        assert!(want_model_file("tokenizer_config.json"));
        assert!(want_model_file("special_tokens_map.json"));
    }

    #[test]
    fn want_model_file_rejects_pickle_torch_shards() {
        // We always go through safetensors; .bin/.pt/.pth would double
        // download size on the typical HF mirror.
        assert!(!want_model_file("pytorch_model.bin"));
        assert!(!want_model_file("pytorch_model-00001-of-00002.bin"));
        assert!(!want_model_file("model.pt"));
        assert!(!want_model_file("model.pth"));
    }

    #[test]
    fn want_model_file_rejects_repo_metadata_and_media() {
        for name in [
            "README.md",
            "README",
            "readme.txt",
            "LICENSE",
            "license.txt",
            ".gitattributes",
            "preview.png",
            "logo.JPG",
            "banner.jpeg",
            "demo.gif",
            "diagram.svg",
            "model.onnx",
            "weights.gguf",
        ] {
            assert!(!want_model_file(name), "should reject {name}");
        }
    }

    #[test]
    fn want_model_file_case_insensitive() {
        // The lowercase pre-pass means uppercased extensions also reject.
        assert!(!want_model_file("LOGO.PNG"));
        assert!(!want_model_file("Model.GGUF"));
    }

    #[test]
    fn want_model_file_accepts_unknown_supporting_files() {
        // Anything that isn't on the reject list is kept — the wanted
        // set is open-ended (model configs vary by family).
        assert!(want_model_file("generation_config.json"));
        assert!(want_model_file("chat_template.jinja"));
    }

    // ─── hf_cache_repo_dir ─────────────────────────────────────────────
    //
    // Env-var driven; serialised to avoid races on HUGGINGFACE_HUB_CACHE /
    // HF_HOME / HOME between parallel test threads.

    /// RAII guard for `(key, value)` pairs in std::env. Restores the
    /// original values on drop so neighbouring tests aren't affected.
    struct EnvSet {
        keys: Vec<(String, Option<String>)>,
    }

    impl EnvSet {
        fn new(pairs: &[(&str, Option<&str>)]) -> Self {
            let mut keys = Vec::new();
            for (k, v) in pairs {
                let prev = std::env::var(*k).ok();
                match v {
                    Some(val) => std::env::set_var(*k, val),
                    None => std::env::remove_var(*k),
                }
                keys.push((k.to_string(), prev));
            }
            Self { keys }
        }
    }

    impl Drop for EnvSet {
        fn drop(&mut self) {
            for (k, prev) in self.keys.drain(..) {
                match prev {
                    Some(v) => std::env::set_var(&k, v),
                    None => std::env::remove_var(&k),
                }
            }
        }
    }

    #[test]
    #[serial]
    fn hf_cache_repo_dir_uses_huggingface_hub_cache_when_set() {
        let _e = EnvSet::new(&[
            ("HUGGINGFACE_HUB_CACHE", Some("/tmp/test-hub")),
            ("HF_HOME", None),
        ]);
        let dir = hf_cache_repo_dir(RepoKind::Dataset, "owner/name").unwrap();
        assert_eq!(dir.to_string_lossy(), "/tmp/test-hub/datasets--owner--name");
    }

    #[test]
    #[serial]
    fn hf_cache_repo_dir_uses_hf_home_when_hub_unset() {
        let _e = EnvSet::new(&[
            ("HUGGINGFACE_HUB_CACHE", None),
            ("HF_HOME", Some("/tmp/hf-home")),
        ]);
        let dir = hf_cache_repo_dir(RepoKind::Model, "owner/name").unwrap();
        assert_eq!(
            dir.to_string_lossy(),
            "/tmp/hf-home/hub/models--owner--name"
        );
    }

    #[test]
    #[serial]
    fn hf_cache_repo_dir_falls_back_to_home_default() {
        let _e = EnvSet::new(&[
            ("HUGGINGFACE_HUB_CACHE", None),
            ("HF_HOME", None),
            ("HOME", Some("/tmp/fallback-home")),
        ]);
        let dir = hf_cache_repo_dir(RepoKind::Dataset, "owner/repo").unwrap();
        assert_eq!(
            dir.to_string_lossy(),
            "/tmp/fallback-home/.cache/huggingface/hub/datasets--owner--repo"
        );
    }

    #[test]
    #[serial]
    fn hf_cache_repo_dir_returns_none_when_home_missing() {
        let _e = EnvSet::new(&[
            ("HUGGINGFACE_HUB_CACHE", None),
            ("HF_HOME", None),
            ("HOME", None),
        ]);
        assert!(hf_cache_repo_dir(RepoKind::Model, "owner/name").is_none());
    }

    #[test]
    #[serial]
    fn hf_cache_repo_dir_replaces_slash_in_repo_id() {
        // HF cache uses `--` as the path separator; every `/` in the
        // repo ID maps to a literal `--` in the cache directory name.
        let _e = EnvSet::new(&[
            ("HUGGINGFACE_HUB_CACHE", Some("/tmp/x")),
            ("HF_HOME", None),
        ]);
        let dir = hf_cache_repo_dir(RepoKind::Model, "complex/owner/name").unwrap();
        assert!(dir.to_string_lossy().ends_with("models--complex--owner--name"));
    }

    #[test]
    #[serial]
    fn hf_cache_repo_dir_distinguishes_dataset_from_model() {
        let _e = EnvSet::new(&[
            ("HUGGINGFACE_HUB_CACHE", Some("/tmp/y")),
            ("HF_HOME", None),
        ]);
        let ds = hf_cache_repo_dir(RepoKind::Dataset, "x/y").unwrap();
        let md = hf_cache_repo_dir(RepoKind::Model, "x/y").unwrap();
        assert_ne!(ds, md, "RepoKind must produce distinct cache dirs");
        assert!(ds.to_string_lossy().contains("datasets--"));
        assert!(md.to_string_lossy().contains("models--"));
    }

    // ─── hf_hub-bound functions: not-an-hf-path early return ────────────
    //
    // These four functions all share the same `hf://` strip_prefix +
    // `@revision` parsing + `Api::new()` setup head. Pin the early-return
    // path that fires when the input doesn't start with `hf://`. No HTTP
    // mocking needed — the error fires before any network call.

    #[test]
    fn resolve_hf_vindex_rejects_non_hf_path() {
        let err = resolve_hf_vindex("/local/path").expect_err("must reject local paths");
        assert!(err.to_string().contains("not an hf://"));
    }

    #[test]
    fn resolve_hf_vindex_rejects_https_url() {
        let err = resolve_hf_vindex("https://huggingface.co/owner/repo").expect_err("must reject");
        assert!(err.to_string().contains("not an hf://"));
    }

    #[test]
    fn download_hf_weights_rejects_non_hf_path() {
        let err = download_hf_weights("./relative").expect_err("must reject");
        assert!(err.to_string().contains("not an hf://"));
    }

    #[test]
    fn download_hf_weights_rejects_empty_string() {
        let err = download_hf_weights("").expect_err("must reject empty");
        assert!(err.to_string().contains("not an hf://"));
    }

    /// Stub `DownloadProgress` for the *_with_progress tests. We only need
    /// the trait to exist so the function type-checks; the stub is never
    /// invoked because we hit the early-return path.
    struct NoOpProgress;
    impl DownloadProgress for NoOpProgress {
        fn init(&mut self, _size: usize, _filename: &str) {}
        fn update(&mut self, _size: usize) {}
        fn finish(&mut self) {}
    }

    #[test]
    fn resolve_hf_vindex_with_progress_rejects_non_hf_path() {
        let err = resolve_hf_vindex_with_progress("/tmp/foo", |_| NoOpProgress)
            .expect_err("must reject");
        assert!(err.to_string().contains("not an hf://"));
    }

    #[test]
    fn resolve_hf_model_with_progress_rejects_non_hf_path() {
        let err =
            resolve_hf_model_with_progress("./local-model", |_| NoOpProgress).expect_err("must reject");
        assert!(err.to_string().contains("not an hf://"));
    }

    // ─── hf_hub-bound: revision parsing covered by error path ──────────
    //
    // The `@revision` split happens after the `hf://` prefix strip but
    // before any network call. The functions then do `Api::new()` which
    // (with HF_ENDPOINT pointing at a non-existent server) fails fast.
    // That path covers the revision-vs-no-revision branches.

    /// RAII guard for HF_ENDPOINT + HF_HOME + a tempdir cache.
    /// Restores prior values on drop.
    struct HfTestEnv {
        prev_endpoint: Option<String>,
        prev_home: Option<String>,
        prev_hub: Option<String>,
        prev_token: Option<String>,
        // Hold the tempdir so it lives as long as the guard.
        _tmp: tempfile::TempDir,
    }
    impl HfTestEnv {
        fn new(endpoint: &str) -> Self {
            let prev_endpoint = std::env::var("HF_ENDPOINT").ok();
            let prev_home = std::env::var("HF_HOME").ok();
            let prev_hub = std::env::var("HUGGINGFACE_HUB_CACHE").ok();
            let prev_token = std::env::var("HF_TOKEN").ok();

            let tmp = tempfile::tempdir().unwrap();
            std::env::set_var("HF_ENDPOINT", endpoint);
            std::env::set_var("HF_HOME", tmp.path());
            // Clear HUGGINGFACE_HUB_CACHE so HF_HOME wins; clear token
            // so we don't accidentally hit a real auth header.
            std::env::remove_var("HUGGINGFACE_HUB_CACHE");
            std::env::remove_var("HF_TOKEN");

            Self {
                prev_endpoint,
                prev_home,
                prev_hub,
                prev_token,
                _tmp: tmp,
            }
        }
    }
    impl Drop for HfTestEnv {
        fn drop(&mut self) {
            for (k, prev) in [
                ("HF_ENDPOINT", self.prev_endpoint.take()),
                ("HF_HOME", self.prev_home.take()),
                ("HUGGINGFACE_HUB_CACHE", self.prev_hub.take()),
                ("HF_TOKEN", self.prev_token.take()),
            ] {
                match prev {
                    Some(v) => std::env::set_var(k, v),
                    None => std::env::remove_var(k),
                }
            }
        }
    }

    #[test]
    #[serial]
    fn resolve_hf_vindex_errors_on_404_index_json() {
        // mockito returns 404 for /datasets/owner/repo/resolve/main/index.json
        // → repo.get(INDEX_JSON) errors → resolve_hf_vindex returns
        // the wrapped "failed to download index.json" error. Exercises:
        // hf:// strip, no-revision branch, Api::new(), repo.get error path.
        let mut server = mockito::Server::new();
        let _g = HfTestEnv::new(&server.url());
        let _m = server
            .mock(
                "GET",
                mockito::Matcher::Regex(r"/datasets/owner/repo/resolve/.*/index\.json".into()),
            )
            .with_status(404)
            .create();

        let err = resolve_hf_vindex("hf://owner/repo").expect_err("404 must error");
        assert!(
            err.to_string().contains("failed to download index.json"),
            "got: {err}"
        );
    }

    #[test]
    #[serial]
    fn resolve_hf_vindex_errors_with_revision_pinned() {
        // Same as above but with `@v2.0` revision. The split path takes
        // a different `repo` constructor (with_revision) — verify both
        // branches by exercising them with the same 404 mock.
        let mut server = mockito::Server::new();
        let _g = HfTestEnv::new(&server.url());
        let _m = server
            .mock(
                "GET",
                mockito::Matcher::Regex(r"/datasets/owner/repo/resolve/v2\.0/index\.json".into()),
            )
            .with_status(404)
            .create();

        let err = resolve_hf_vindex("hf://owner/repo@v2.0").expect_err("404 must error");
        assert!(
            err.to_string().contains("owner/repo"),
            "error must mention repo: {err}"
        );
    }

    #[test]
    #[serial]
    fn download_hf_weights_silently_skips_missing_files() {
        // download_hf_weights iterates VINDEX_WEIGHT_FILES with `let _ =
        // repo.get(filename)` — every miss is silenced. Pin that contract:
        // even when every file 404s, the function returns Ok(()).
        let mut server = mockito::Server::new();
        let _g = HfTestEnv::new(&server.url());
        let _m = server
            .mock("GET", mockito::Matcher::Any)
            .with_status(404)
            .create();

        download_hf_weights("hf://owner/repo").expect("missing files are non-fatal");
    }

    #[test]
    #[serial]
    fn resolve_hf_model_with_progress_errors_when_info_fails() {
        // The model-side variant calls `repo.info()` first (which hits
        // /api/models/{repo}/revision/{rev}). A 500 there propagates as
        // `HF info failed for {hf_path}`.
        let mut server = mockito::Server::new();
        let _g = HfTestEnv::new(&server.url());
        let _m = server
            .mock(
                "GET",
                mockito::Matcher::Regex(r"/api/models/owner/repo.*".into()),
            )
            .with_status(500)
            .with_body(r#"{"error": "boom"}"#)
            .create();

        let err = resolve_hf_model_with_progress("hf://owner/repo", |_| NoOpProgress)
            .expect_err("info failure must surface");
        assert!(
            err.to_string().contains("HF info failed"),
            "expected 'HF info failed' wrapper, got: {err}"
        );
    }

    #[test]
    #[serial]
    fn resolve_hf_vindex_with_progress_errors_when_index_json_404s() {
        // The progress variant fetches index.json first; when it's
        // missing the `ok_or_else` clause produces a clear error.
        let mut server = mockito::Server::new();
        let _g = HfTestEnv::new(&server.url());
        let _m = server
            .mock("GET", mockito::Matcher::Any)
            .with_status(404)
            .create();

        let err = resolve_hf_vindex_with_progress("hf://owner/repo", |_| NoOpProgress)
            .expect_err("404 on index.json must error");
        assert!(err.to_string().contains("failed to fetch index.json"));
    }
}
