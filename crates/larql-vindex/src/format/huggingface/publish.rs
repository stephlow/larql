//! HuggingFace publish path — repo creation + per-file upload + LFS
//! pointer/upload protocol + callback hooks.
//!
//! Carved out of the monolithic `huggingface.rs` in the 2026-04-25
//! reorg. See `super::mod.rs` for the module map.

use std::path::{Path, PathBuf};

use crate::error::VindexError;
use crate::format::filenames::*;

/// Options controlling [`publish_vindex_with_opts`]. Kept as a struct so
/// the signature can grow without breaking callers.
#[derive(Clone, Debug)]
pub struct PublishOptions {
    /// When true, skip uploading LFS-tracked files whose local SHA256
    /// already matches the remote `lfs.oid`. Small files (git-tracked
    /// json / manifest) are always re-uploaded — their text is tiny and
    /// the git blob SHA-1 format isn't directly derivable from the file
    /// content SHA256 without a separate hash.
    pub skip_unchanged: bool,
    /// HuggingFace repo type: `"model"` (default) or `"dataset"`.
    pub repo_type: String,
}

impl Default for PublishOptions {
    fn default() -> Self {
        Self {
            skip_unchanged: false,
            repo_type: "model".into(),
        }
    }
}

impl PublishOptions {
    pub fn skip_unchanged() -> Self {
        Self {
            skip_unchanged: true,
            ..Self::default()
        }
    }
}

/// Returns the HF API base URL for a repo: `https://huggingface.co/api/{models|datasets}/{repo_id}`.
#[allow(dead_code)]
fn hf_api_url(repo_type: &str, repo_id: &str, path: &str) -> String {
    let plural = if repo_type == "dataset" {
        "datasets"
    } else {
        "models"
    };
    format!("https://huggingface.co/api/{plural}/{repo_id}/{path}")
}

/// Returns the web / git base URL for a repo.
/// Models: `https://huggingface.co/{repo_id}`, datasets: `https://huggingface.co/datasets/{repo_id}`.
fn hf_repo_url(repo_type: &str, repo_id: &str) -> String {
    if repo_type == "dataset" {
        format!("https://huggingface.co/datasets/{repo_id}")
    } else {
        format!("https://huggingface.co/{repo_id}")
    }
}

/// Upload a local vindex directory to HuggingFace as a dataset repo.
///
/// Equivalent to `publish_vindex_with_opts(dir, repo_id, &PublishOptions::default(), cb)`.
/// Requires HF_TOKEN environment variable or ~/.huggingface/token.
pub fn publish_vindex(
    vindex_dir: &Path,
    repo_id: &str,
    callbacks: &mut dyn PublishCallbacks,
) -> Result<String, VindexError> {
    publish_vindex_with_opts(vindex_dir, repo_id, &PublishOptions::default(), callbacks)
}

/// Upload a vindex directory with explicit options. See [`PublishOptions`].
pub fn publish_vindex_with_opts(
    vindex_dir: &Path,
    repo_id: &str,
    opts: &PublishOptions,
    callbacks: &mut dyn PublishCallbacks,
) -> Result<String, VindexError> {
    if !vindex_dir.is_dir() {
        return Err(VindexError::NotADirectory(vindex_dir.to_path_buf()));
    }
    let index_path = vindex_dir.join(INDEX_JSON);
    if !index_path.exists() {
        return Err(VindexError::Parse(format!(
            "not a vindex directory (no index.json): {}",
            vindex_dir.display()
        )));
    }

    let token = get_hf_token()?;
    let repo_type = opts.repo_type.as_str();
    callbacks.on_start(repo_id);
    create_hf_repo(repo_id, &token, repo_type)?;

    // Pull remote LFS index so we can skip unchanged files. Non-fatal
    // if the tree API errors (brand-new repo returns 404 here) — we just
    // fall back to "upload everything".
    let remote_lfs: std::collections::HashMap<String, String> = if opts.skip_unchanged {
        fetch_remote_lfs_oids(repo_id, &token, repo_type).unwrap_or_default()
    } else {
        std::collections::HashMap::new()
    };

    // Collect files from the root and any immediate subdirectories (e.g. layers/).
    let mut files: Vec<(PathBuf, String)> = Vec::new(); // (abs_path, repo_path)
    for entry in std::fs::read_dir(vindex_dir)?.filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_file() {
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            files.push((path, name));
        } else if path.is_dir() {
            let dir_name = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            for sub in std::fs::read_dir(&path)
                .ok()
                .into_iter()
                .flatten()
                .filter_map(|e| e.ok())
            {
                let sub_path = sub.path();
                if sub_path.is_file() {
                    let sub_name = sub_path
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default();
                    files.push((sub_path, format!("{dir_name}/{sub_name}")));
                }
            }
        }
    }
    files.sort_by(|a, b| a.1.cmp(&b.1));

    for (file_path, filename) in &files {
        let size = std::fs::metadata(file_path).map(|m| m.len()).unwrap_or(0);

        // Skip-if-unchanged: compare local SHA256 against remote lfs.oid.
        if opts.skip_unchanged {
            if let Some(remote_sha) = remote_lfs.get(filename) {
                if let Ok(local_sha) = crate::format::checksums::sha256_file(file_path) {
                    if local_sha == *remote_sha {
                        callbacks.on_file_skipped(filename, size, remote_sha);
                        continue;
                    }
                }
            }
        }

        callbacks.on_file_start(filename, size);
        upload_file_to_hf(repo_id, &token, file_path, filename, callbacks, repo_type)?;
        callbacks.on_file_done(filename);
    }

    let url = hf_repo_url(repo_type, repo_id);
    callbacks.on_complete(&url);
    Ok(url)
}

/// List remote files and return `filename → lfs.oid` for every LFS-tracked
/// file at the repo root. Files without an `lfs.oid` (git-tracked small
/// text) are omitted; callers skip only what's in the map.
fn fetch_remote_lfs_oids(
    repo_id: &str,
    token: &str,
    repo_type: &str,
) -> Result<std::collections::HashMap<String, String>, VindexError> {
    let plural = if repo_type == "dataset" {
        "datasets"
    } else {
        "models"
    };
    let url = format!("https://huggingface.co/api/{plural}/{repo_id}/tree/main?recursive=true");
    let client = reqwest::blocking::Client::new();
    let resp = client
        .get(&url)
        .header("Authorization", format!("Bearer {token}"))
        .send()
        .map_err(|e| VindexError::Parse(format!("HF tree fetch failed: {e}")))?;

    if !resp.status().is_success() {
        // 404 on a fresh repo → no remote files, can't skip anything.
        return Ok(std::collections::HashMap::new());
    }

    let body: serde_json::Value = resp
        .json()
        .map_err(|e| VindexError::Parse(format!("HF tree JSON: {e}")))?;
    let arr = match body.as_array() {
        Some(a) => a,
        None => return Ok(std::collections::HashMap::new()),
    };

    let mut out = std::collections::HashMap::new();
    for entry in arr {
        if entry.get("type").and_then(|v| v.as_str()) != Some("file") {
            continue;
        }
        let path = match entry.get("path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => continue,
        };
        if let Some(lfs_oid) = entry
            .get("lfs")
            .and_then(|v| v.get("oid"))
            .and_then(|v| v.as_str())
        {
            out.insert(path.to_string(), lfs_oid.to_string());
        }
    }
    Ok(out)
}

/// Callbacks for publish progress.
pub trait PublishCallbacks {
    fn on_start(&mut self, _repo: &str) {}
    fn on_file_start(&mut self, _filename: &str, _size: u64) {}
    /// Fired periodically during the upload with cumulative bytes sent
    /// for the current file. Default no-op. Implement to render a live
    /// progress bar; indicatif wrappers live in the CLI layer to stay
    /// version-agnostic here.
    fn on_file_progress(&mut self, _filename: &str, _bytes_sent: u64, _total_bytes: u64) {}
    fn on_file_done(&mut self, _filename: &str) {}
    /// Fired when [`PublishOptions::skip_unchanged`] matches the remote
    /// `lfs.oid` and the upload is skipped. Default no-op so existing
    /// callbacks don't need to change.
    fn on_file_skipped(&mut self, _filename: &str, _size: u64, _sha256: &str) {}
    fn on_complete(&mut self, _url: &str) {}
}

pub struct SilentPublishCallbacks;
impl PublishCallbacks for SilentPublishCallbacks {}

// ═══════════════════════════════════════════════════════════════
// HuggingFace HTTP API helpers
// ═══════════════════════════════════════════════════════════════

pub(super) fn get_hf_token() -> Result<String, VindexError> {
    // Try environment variable first
    if let Ok(token) = std::env::var("HF_TOKEN") {
        return Ok(token);
    }

    // Try token file
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    let token_path = PathBuf::from(&home).join(".huggingface").join("token");
    if token_path.exists() {
        let token = std::fs::read_to_string(&token_path)?;
        return Ok(token.trim().to_string());
    }

    // Try newer cache location
    let token_path = PathBuf::from(&home)
        .join(".cache")
        .join("huggingface")
        .join("token");
    if token_path.exists() {
        let token = std::fs::read_to_string(&token_path)?;
        return Ok(token.trim().to_string());
    }

    Err(VindexError::Parse(
        "HuggingFace token not found. Set HF_TOKEN or run `huggingface-cli login`.".into(),
    ))
}

fn create_hf_repo(repo_id: &str, token: &str, repo_type: &str) -> Result<(), VindexError> {
    let client = reqwest::blocking::Client::new();
    let resp = client
        .post("https://huggingface.co/api/repos/create")
        .header("Authorization", format!("Bearer {token}"))
        .json(&serde_json::json!({
            "name": repo_id.split('/').next_back().unwrap_or(repo_id),
            "type": repo_type,
            "private": false,
        }))
        .send()
        .map_err(|e| VindexError::Parse(format!("HF API error: {e}")))?;

    // 409 = already exists, that's fine
    if resp.status().is_success() || resp.status().as_u16() == 409 {
        Ok(())
    } else {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        Err(VindexError::Parse(format!(
            "HF repo create failed ({status}): {body}"
        )))
    }
}

/// Counting `Read` adapter — increments a shared atomic on every read so
/// a poll thread can report upload progress without per-chunk syscalls.
struct CountingReader<R: std::io::Read> {
    inner: R,
    counter: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl<R: std::io::Read> std::io::Read for CountingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.counter
            .fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(n)
    }
}

/// Upload a single file to a HuggingFace dataset repo via the real HF
/// protocol:
///
///   1. **Preupload** — `POST /api/datasets/{repo}/preupload/main` with a
///      base64 sample of the first 512 bytes. HF decides `lfs` vs `regular`
///      based on size + `.gitattributes`.
///   2. **LFS batch** (LFS path only) — `POST {repo}.git/info/lfs/objects/batch`
///      returns a signed upload URL or tells us the file is already there.
///   3. **Streaming PUT** to the signed URL, ticking `on_file_progress` as
///      bytes flow. `CountingReader` + worker thread keeps the main thread
///      free to poll.
///   4. **Verify** — `POST {verify.href}` with `{oid, size}`.
///   5. **Commit** — `POST /api/datasets/{repo}/commit/main` as NDJSON with
///      a `lfsFile` (LFS) or `file` (regular, base64-inline) operation.
///
/// The old single-PUT "upload endpoint" this replaced was fictional — HF
/// never exposed `PUT /api/datasets/{repo}/upload/main/{file}`. Requests
/// to it 404 after the first few megabytes of body, which was the bug
/// that triggered this rewrite.
fn upload_file_to_hf(
    repo_id: &str,
    token: &str,
    local_path: &Path,
    remote_filename: &str,
    callbacks: &mut dyn PublishCallbacks,
    repo_type: &str,
) -> Result<(), VindexError> {
    let size = std::fs::metadata(local_path)?.len();
    let sha256 = crate::format::checksums::sha256_file(local_path)?;

    let decision = preupload_decide(repo_id, token, remote_filename, local_path, size, repo_type)?;

    if decision.should_ignore {
        // HF's preupload told us the server would ignore this path
        // (matches `.gitignore` / similar). Skip silently.
        return Ok(());
    }

    match decision.mode.as_str() {
        "lfs" => upload_lfs(
            repo_id,
            token,
            local_path,
            remote_filename,
            size,
            &sha256,
            callbacks,
            repo_type,
        ),
        "regular" => upload_regular(
            repo_id,
            token,
            local_path,
            remote_filename,
            size,
            callbacks,
            repo_type,
        ),
        other => Err(VindexError::Parse(format!(
            "HF preupload returned unknown mode `{other}` for {remote_filename}"
        ))),
    }
}

struct PreuploadDecision {
    mode: String,
    should_ignore: bool,
}

/// Call `POST /api/datasets/{repo}/preupload/main` for a single file and
/// return whether HF wants it uploaded via LFS or inlined in a regular
/// commit. HF requires a base64 sample of the first ~512 bytes so it
/// can sniff the file's format (text vs binary, etc.).
fn preupload_decide(
    repo_id: &str,
    token: &str,
    remote_filename: &str,
    local_path: &Path,
    size: u64,
    repo_type: &str,
) -> Result<PreuploadDecision, VindexError> {
    use base64::Engine;
    use std::io::Read;

    // Read up to 512 bytes for the format-sniff sample. HF accepts a
    // smaller sample for small files without complaint.
    let mut sample_buf = vec![0u8; 512.min(size as usize)];
    if !sample_buf.is_empty() {
        let mut file = std::fs::File::open(local_path)?;
        file.read_exact(&mut sample_buf)?;
    }
    let sample_b64 = base64::prelude::BASE64_STANDARD.encode(&sample_buf);

    let plural = if repo_type == "dataset" {
        "datasets"
    } else {
        "models"
    };
    let url = format!("https://huggingface.co/api/{plural}/{repo_id}/preupload/main");
    let body = serde_json::json!({
        "files": [{
            "path":   remote_filename,
            "sample": sample_b64,
            "size":   size,
        }],
    });
    let client = reqwest::blocking::Client::new();
    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {token}"))
        .json(&body)
        .send()
        .map_err(|e| VindexError::Parse(format!("preupload failed: {e}")))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        return Err(VindexError::Parse(format!(
            "preupload ({status}) for {remote_filename}: {body}"
        )));
    }
    let json: serde_json::Value = resp
        .json()
        .map_err(|e| VindexError::Parse(format!("preupload JSON: {e}")))?;
    let files = json
        .get("files")
        .and_then(|v| v.as_array())
        .ok_or_else(|| VindexError::Parse("preupload response missing `files`".into()))?;
    let entry = files
        .first()
        .ok_or_else(|| VindexError::Parse("preupload response files[] empty".into()))?;
    let mode = entry
        .get("uploadMode")
        .and_then(|v| v.as_str())
        .unwrap_or("lfs")
        .to_string();
    let should_ignore = entry
        .get("shouldIgnore")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    Ok(PreuploadDecision {
        mode,
        should_ignore,
    })
}

/// LFS-mode upload: batch → PUT to signed URL → verify → commit pointer.
#[allow(clippy::too_many_arguments)]
fn upload_lfs(
    repo_id: &str,
    token: &str,
    local_path: &Path,
    remote_filename: &str,
    size: u64,
    sha256: &str,
    callbacks: &mut dyn PublishCallbacks,
    repo_type: &str,
) -> Result<(), VindexError> {
    let batch = lfs_batch_upload(repo_id, token, sha256, size, repo_type)?;

    // If the response has no upload action, the object is already present
    // on the LFS server — skip to verify (if present) + commit.
    if let Some(ref upload) = batch.upload {
        stream_put_with_progress(
            &upload.href,
            &upload.header,
            local_path,
            size,
            remote_filename,
            callbacks,
        )?;
    } else {
        // Still tick the bar to 100% so the UX matches the upload path.
        callbacks.on_file_progress(remote_filename, size, size);
    }

    if let Some(ref verify) = batch.verify {
        lfs_verify(&verify.href, &verify.header, token, sha256, size)?;
    }

    commit_lfs_file(repo_id, token, remote_filename, sha256, size, repo_type)
}

/// Small-file path: commit directly with the content inlined as base64
/// in the NDJSON commit body. HF's preupload flags tiny text files for
/// this path.
fn upload_regular(
    repo_id: &str,
    token: &str,
    local_path: &Path,
    remote_filename: &str,
    size: u64,
    callbacks: &mut dyn PublishCallbacks,
    repo_type: &str,
) -> Result<(), VindexError> {
    use base64::Engine;
    let data = std::fs::read(local_path)?;
    // Fire start+end of the progress bar even though we don't stream —
    // keeps the UX consistent across file sizes.
    callbacks.on_file_progress(remote_filename, 0, size);
    let encoded = base64::prelude::BASE64_STANDARD.encode(&data);

    let plural = if repo_type == "dataset" {
        "datasets"
    } else {
        "models"
    };
    let url = format!("https://huggingface.co/api/{plural}/{repo_id}/commit/main");
    let mut ndjson = String::new();
    ndjson.push_str(
        &serde_json::to_string(&serde_json::json!({
            "key": "header",
            "value": {
                "summary": format!("Upload {remote_filename}"),
            },
        }))
        .unwrap(),
    );
    ndjson.push('\n');
    ndjson.push_str(
        &serde_json::to_string(&serde_json::json!({
            "key": "file",
            "value": {
                "path":     remote_filename,
                "encoding": "base64",
                "content":  encoded,
            },
        }))
        .unwrap(),
    );
    ndjson.push('\n');

    let client = reqwest::blocking::Client::new();
    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {token}"))
        .header("Content-Type", "application/x-ndjson")
        .body(ndjson)
        .send()
        .map_err(|e| VindexError::Parse(format!("commit (regular) failed: {e}")))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        return Err(VindexError::Parse(format!(
            "commit (regular) {remote_filename} ({status}): {body}"
        )));
    }
    callbacks.on_file_progress(remote_filename, size, size);
    Ok(())
}

#[derive(Debug)]
struct LfsAction {
    href: String,
    header: std::collections::HashMap<String, String>,
}

#[derive(Debug)]
struct LfsBatchResponse {
    upload: Option<LfsAction>,
    verify: Option<LfsAction>,
}

/// POST to the LFS batch endpoint asking for an upload URL for one
/// object. Returns the upload + verify actions (either or both may be
/// absent — an absent `upload` means the object is already stored).
fn lfs_batch_upload(
    repo_id: &str,
    token: &str,
    sha256: &str,
    size: u64,
    repo_type: &str,
) -> Result<LfsBatchResponse, VindexError> {
    let url = format!(
        "{}.git/info/lfs/objects/batch",
        hf_repo_url(repo_type, repo_id)
    );
    let body = serde_json::json!({
        "operation":  "upload",
        "transfers":  ["basic"],
        "hash_algo":  "sha256",
        "objects":    [{"oid": sha256, "size": size}],
    });
    let client = reqwest::blocking::Client::new();
    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {token}"))
        .header("Accept", "application/vnd.git-lfs+json")
        .header("Content-Type", "application/vnd.git-lfs+json")
        .json(&body)
        .send()
        .map_err(|e| VindexError::Parse(format!("LFS batch failed: {e}")))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        return Err(VindexError::Parse(format!("LFS batch ({status}): {body}")));
    }
    let json: serde_json::Value = resp
        .json()
        .map_err(|e| VindexError::Parse(format!("LFS batch JSON: {e}")))?;
    let objects = json
        .get("objects")
        .and_then(|v| v.as_array())
        .ok_or_else(|| VindexError::Parse("LFS batch response missing `objects`".into()))?;
    let obj = objects
        .first()
        .ok_or_else(|| VindexError::Parse("LFS batch objects[] empty".into()))?;

    // Per-object error surfaced in-line rather than as an HTTP status.
    if let Some(err) = obj.get("error") {
        return Err(VindexError::Parse(format!("LFS batch object error: {err}")));
    }

    let actions = obj.get("actions");
    let parse_action = |key: &str| -> Option<LfsAction> {
        let a = actions?.get(key)?;
        let href = a.get("href").and_then(|v| v.as_str())?.to_string();
        let header: std::collections::HashMap<String, String> = a
            .get("header")
            .and_then(|v| v.as_object())
            .map(|m| {
                m.iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect()
            })
            .unwrap_or_default();
        Some(LfsAction { href, header })
    };
    Ok(LfsBatchResponse {
        upload: parse_action("upload"),
        verify: parse_action("verify"),
    })
}

/// PUT the file contents to the signed LFS URL, streaming through a
/// `CountingReader` so the worker thread can report progress.
fn stream_put_with_progress(
    href: &str,
    extra_headers: &std::collections::HashMap<String, String>,
    local_path: &Path,
    size: u64,
    remote_filename: &str,
    callbacks: &mut dyn PublishCallbacks,
) -> Result<(), VindexError> {
    use std::sync::atomic::Ordering;
    use std::sync::mpsc::TryRecvError;
    use std::time::Duration;

    let file = std::fs::File::open(local_path)?;
    let counter = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let reader = CountingReader {
        inner: file,
        counter: counter.clone(),
    };
    let body = reqwest::blocking::Body::sized(reader, size);

    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(3600))
        .build()
        .map_err(|e| VindexError::Parse(format!("HTTP client error: {e}")))?;

    // Build the request on the worker thread (reqwest's Body needs to
    // travel there). Include any signature headers the LFS server
    // requested — on AWS-backed buckets these carry the AWS sigv4 bits.
    let href_owned = href.to_string();
    let headers_owned: Vec<(String, String)> = extra_headers
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    let (tx, rx) = std::sync::mpsc::channel();
    let handle = std::thread::spawn(move || {
        let mut req = client.put(&href_owned);
        for (k, v) in &headers_owned {
            req = req.header(k.as_str(), v.as_str());
        }
        let result = req.body(body).send();
        let _ = tx.send(result);
    });

    loop {
        match rx.try_recv() {
            Ok(resp) => {
                let _ = handle.join();
                let resp = resp.map_err(|e| VindexError::Parse(format!("LFS PUT failed: {e}")))?;
                if resp.status().is_success() {
                    callbacks.on_file_progress(remote_filename, size, size);
                    return Ok(());
                }
                let status = resp.status();
                let body = resp.text().unwrap_or_default();
                return Err(VindexError::Parse(format!(
                    "LFS PUT {remote_filename} ({status}): {body}"
                )));
            }
            Err(TryRecvError::Empty) => {
                let sent = counter.load(Ordering::Relaxed);
                callbacks.on_file_progress(remote_filename, sent, size);
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(TryRecvError::Disconnected) => {
                let _ = handle.join();
                return Err(VindexError::Parse(
                    "upload worker terminated unexpectedly".into(),
                ));
            }
        }
    }
}

/// POST `{oid, size}` to the verify URL the LFS batch returned. HF uses
/// this to confirm the object made it to storage intact before the
/// commit references it.
fn lfs_verify(
    href: &str,
    extra_headers: &std::collections::HashMap<String, String>,
    token: &str,
    sha256: &str,
    size: u64,
) -> Result<(), VindexError> {
    let body = serde_json::json!({"oid": sha256, "size": size});
    let client = reqwest::blocking::Client::new();
    let mut req = client
        .post(href)
        .header("Authorization", format!("Bearer {token}"))
        .header("Accept", "application/vnd.git-lfs+json")
        .header("Content-Type", "application/vnd.git-lfs+json");
    for (k, v) in extra_headers {
        req = req.header(k.as_str(), v.as_str());
    }
    let resp = req
        .json(&body)
        .send()
        .map_err(|e| VindexError::Parse(format!("LFS verify failed: {e}")))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        return Err(VindexError::Parse(format!("LFS verify ({status}): {body}")));
    }
    Ok(())
}

/// Commit a single LFS pointer into the repo via NDJSON. HF's commit
/// API is one request per change set; we commit per file for simplicity
/// (batching every file into one commit is a future optimisation).
fn commit_lfs_file(
    repo_id: &str,
    token: &str,
    remote_filename: &str,
    sha256: &str,
    size: u64,
    repo_type: &str,
) -> Result<(), VindexError> {
    let plural = if repo_type == "dataset" {
        "datasets"
    } else {
        "models"
    };
    let url = format!("https://huggingface.co/api/{plural}/{repo_id}/commit/main");
    let mut ndjson = String::new();
    ndjson.push_str(
        &serde_json::to_string(&serde_json::json!({
            "key": "header",
            "value": {"summary": format!("Upload {remote_filename}")},
        }))
        .unwrap(),
    );
    ndjson.push('\n');
    ndjson.push_str(
        &serde_json::to_string(&serde_json::json!({
            "key": "lfsFile",
            "value": {
                "path": remote_filename,
                "algo": "sha256",
                "oid":  sha256,
                "size": size,
            },
        }))
        .unwrap(),
    );
    ndjson.push('\n');

    let client = reqwest::blocking::Client::new();
    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {token}"))
        .header("Content-Type", "application/x-ndjson")
        .body(ndjson)
        .send()
        .map_err(|e| VindexError::Parse(format!("commit (LFS) failed: {e}")))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        return Err(VindexError::Parse(format!(
            "commit (LFS) {remote_filename} ({status}): {body}"
        )));
    }
    Ok(())
}
