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
