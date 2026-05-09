//! HuggingFace LFS protocol primitives — batch endpoint, signed-URL
//! streaming PUT, verify, commit. Internal-only siblings of
//! [`super::upload::upload_file_to_hf`]; see that file for the
//! orchestration shape.

use std::collections::HashMap;
use std::path::Path;

use crate::error::VindexError;

use super::hf_repo_url;
use super::protocol::{
    repo_type_plural, CONTENT_TYPE_LFS_JSON, CONTENT_TYPE_NDJSON, HASH_ALGO_SHA256, LFS_OP_UPLOAD,
    LFS_OP_VERIFY, LFS_PUT_TIMEOUT, LFS_TRANSFER_BASIC, UPLOAD_PROGRESS_POLL_INTERVAL,
};
use super::PublishCallbacks;

/// Counting `Read` adapter — increments a shared atomic on every read so
/// a poll thread can report upload progress without per-chunk syscalls.
pub(super) struct CountingReader<R: std::io::Read> {
    pub(super) inner: R,
    pub(super) counter: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl<R: std::io::Read> std::io::Read for CountingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.counter
            .fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(n)
    }
}

#[derive(Debug)]
pub(super) struct LfsAction {
    pub(super) href: String,
    pub(super) header: HashMap<String, String>,
}

#[derive(Debug)]
pub(super) struct LfsBatchResponse {
    pub(super) upload: Option<LfsAction>,
    pub(super) verify: Option<LfsAction>,
}

/// LFS-mode upload: batch → PUT to signed URL → verify → commit pointer.
#[allow(clippy::too_many_arguments)]
pub(super) fn upload_lfs(
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
        "operation":  LFS_OP_UPLOAD,
        "transfers":  [LFS_TRANSFER_BASIC],
        "hash_algo":  HASH_ALGO_SHA256,
        "objects":    [{"oid": sha256, "size": size}],
    });
    let client = reqwest::blocking::Client::new();
    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {token}"))
        .header("Accept", CONTENT_TYPE_LFS_JSON)
        .header("Content-Type", CONTENT_TYPE_LFS_JSON)
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
    parse_lfs_batch_response(&json)
}

/// Parse the JSON body of an LFS batch response into the upload/verify
/// actions our caller dispatches on. Pulled out as a pure helper so the
/// JSON contract can be unit-tested without an HTTP server.
fn parse_lfs_batch_response(json: &serde_json::Value) -> Result<LfsBatchResponse, VindexError> {
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
        let header: HashMap<String, String> = a
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
        upload: parse_action(LFS_OP_UPLOAD),
        verify: parse_action(LFS_OP_VERIFY),
    })
}

/// PUT the file contents to the signed LFS URL, streaming through a
/// `CountingReader` so the worker thread can report progress.
fn stream_put_with_progress(
    href: &str,
    extra_headers: &HashMap<String, String>,
    local_path: &Path,
    size: u64,
    remote_filename: &str,
    callbacks: &mut dyn PublishCallbacks,
) -> Result<(), VindexError> {
    use std::sync::atomic::Ordering;
    use std::sync::mpsc::TryRecvError;

    let file = std::fs::File::open(local_path)?;
    let counter = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let reader = CountingReader {
        inner: file,
        counter: counter.clone(),
    };
    let body = reqwest::blocking::Body::sized(reader, size);

    let client = reqwest::blocking::Client::builder()
        .timeout(LFS_PUT_TIMEOUT)
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
                std::thread::sleep(UPLOAD_PROGRESS_POLL_INTERVAL);
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
    extra_headers: &HashMap<String, String>,
    token: &str,
    sha256: &str,
    size: u64,
) -> Result<(), VindexError> {
    let body = serde_json::json!({"oid": sha256, "size": size});
    let client = reqwest::blocking::Client::new();
    let mut req = client
        .post(href)
        .header("Authorization", format!("Bearer {token}"))
        .header("Accept", CONTENT_TYPE_LFS_JSON)
        .header("Content-Type", CONTENT_TYPE_LFS_JSON);
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
    let plural = repo_type_plural(repo_type);
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
                "algo": HASH_ALGO_SHA256,
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
        .header("Content-Type", CONTENT_TYPE_NDJSON)
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::sync::atomic::Ordering;

    // ─── CountingReader ────────────────────────────────────────────

    #[test]
    fn counting_reader_counts_bytes_read() {
        let bytes = b"hello world".to_vec();
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let mut reader = CountingReader {
            inner: bytes.as_slice(),
            counter: counter.clone(),
        };
        let mut buf = [0u8; 5];
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"hello");
        assert_eq!(counter.load(Ordering::Relaxed), 5);

        let mut rest = Vec::new();
        reader.read_to_end(&mut rest).unwrap();
        assert_eq!(rest, b" world");
        assert_eq!(counter.load(Ordering::Relaxed), 11);
    }

    #[test]
    fn counting_reader_counter_starts_at_zero() {
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let mut reader = CountingReader {
            inner: std::io::empty(),
            counter: counter.clone(),
        };
        let mut buf = [0u8; 16];
        let n = reader.read(&mut buf).unwrap();
        assert_eq!(n, 0);
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    // ─── parse_lfs_batch_response ──────────────────────────────────

    #[test]
    fn parse_lfs_batch_with_upload_and_verify_actions() {
        let json = serde_json::json!({
            "objects": [{
                "actions": {
                    "upload": {
                        "href": "https://lfs.example/upload",
                        "header": {"Authorization": "Bearer x", "X-Amz": "sig"}
                    },
                    "verify": {
                        "href": "https://lfs.example/verify",
                        "header": {}
                    }
                }
            }]
        });
        let parsed = parse_lfs_batch_response(&json).unwrap();
        let upload = parsed.upload.expect("upload action present");
        assert_eq!(upload.href, "https://lfs.example/upload");
        assert_eq!(
            upload.header.get("Authorization").map(|s| s.as_str()),
            Some("Bearer x")
        );
        assert_eq!(upload.header.get("X-Amz").map(|s| s.as_str()), Some("sig"));
        let verify = parsed.verify.expect("verify action present");
        assert_eq!(verify.href, "https://lfs.example/verify");
        assert!(verify.header.is_empty());
    }

    #[test]
    fn parse_lfs_batch_with_no_upload_means_object_already_present() {
        // HF returns no `actions.upload` when the LFS object is already
        // stored — the caller skips straight to verify + commit. Only
        // verify present.
        let json = serde_json::json!({
            "objects": [{
                "actions": {
                    "verify": {"href": "https://lfs.example/verify", "header": {}}
                }
            }]
        });
        let parsed = parse_lfs_batch_response(&json).unwrap();
        assert!(parsed.upload.is_none(), "upload action absent");
        assert!(parsed.verify.is_some(), "verify action present");
    }

    #[test]
    fn parse_lfs_batch_with_no_actions_returns_both_none() {
        let json = serde_json::json!({"objects": [{}]});
        let parsed = parse_lfs_batch_response(&json).unwrap();
        assert!(parsed.upload.is_none());
        assert!(parsed.verify.is_none());
    }

    #[test]
    fn parse_lfs_batch_missing_objects_array_errors() {
        let json = serde_json::json!({});
        let err = parse_lfs_batch_response(&json).expect_err("objects[] missing must error");
        let msg = err.to_string();
        assert!(msg.contains("objects"), "{msg}");
    }

    #[test]
    fn parse_lfs_batch_empty_objects_array_errors() {
        let json = serde_json::json!({"objects": []});
        let err = parse_lfs_batch_response(&json).expect_err("empty objects[] must error");
        let msg = err.to_string();
        assert!(msg.contains("empty"), "{msg}");
    }

    #[test]
    fn parse_lfs_batch_per_object_error_surfaces() {
        // Per-object errors come back inline rather than as an HTTP
        // status code. Make sure they propagate.
        let json = serde_json::json!({
            "objects": [{
                "error": {"code": 422, "message": "object too large"}
            }]
        });
        let err = parse_lfs_batch_response(&json).expect_err("inline object error");
        let msg = err.to_string();
        assert!(msg.contains("LFS batch object error"), "{msg}");
        assert!(msg.contains("too large"), "{msg}");
    }

    #[test]
    fn parse_lfs_batch_action_without_href_is_skipped() {
        // Defensive: an action object without `href` is malformed but
        // shouldn't panic; treat it as absent.
        let json = serde_json::json!({
            "objects": [{
                "actions": {
                    "upload": {"header": {}}
                }
            }]
        });
        let parsed = parse_lfs_batch_response(&json).unwrap();
        assert!(parsed.upload.is_none());
    }
}
