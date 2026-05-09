//! Per-file upload pipeline. Decides between LFS and inline-base64
//! commits based on HuggingFace's preupload response, then dispatches
//! into [`super::lfs`] or runs the regular commit inline.

use std::path::Path;

use crate::error::VindexError;

use super::lfs::upload_lfs;
use super::protocol::{repo_type_plural, CONTENT_TYPE_NDJSON, HF_PREUPLOAD_SAMPLE_BYTES};
use super::PublishCallbacks;

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
pub(super) fn upload_file_to_hf(
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

#[derive(Debug)]
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

    // Read up to the configured byte count for the format-sniff sample. HF accepts a
    // smaller sample for small files without complaint.
    let mut sample_buf = vec![0u8; HF_PREUPLOAD_SAMPLE_BYTES.min(size as usize)];
    if !sample_buf.is_empty() {
        let mut file = std::fs::File::open(local_path)?;
        file.read_exact(&mut sample_buf)?;
    }
    let sample_b64 = base64::prelude::BASE64_STANDARD.encode(&sample_buf);

    let plural = repo_type_plural(repo_type);
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
    parse_preupload_response(&json)
}

/// Parse the JSON body of an HF preupload response into our routing
/// decision. Pulled out as a pure helper so the JSON contract can be
/// unit-tested without an HTTP server.
fn parse_preupload_response(json: &serde_json::Value) -> Result<PreuploadDecision, VindexError> {
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

    let plural = repo_type_plural(repo_type);
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
        .header("Content-Type", CONTENT_TYPE_NDJSON)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_preupload_lfs_mode() {
        let json = serde_json::json!({
            "files": [{"uploadMode": "lfs"}]
        });
        let d = parse_preupload_response(&json).unwrap();
        assert_eq!(d.mode, "lfs");
        assert!(!d.should_ignore);
    }

    #[test]
    fn parse_preupload_regular_mode() {
        let json = serde_json::json!({
            "files": [{"uploadMode": "regular"}]
        });
        let d = parse_preupload_response(&json).unwrap();
        assert_eq!(d.mode, "regular");
    }

    #[test]
    fn parse_preupload_should_ignore_true() {
        // HF flags files matching `.gitignore` patterns; caller skips them.
        let json = serde_json::json!({
            "files": [{"uploadMode": "regular", "shouldIgnore": true}]
        });
        let d = parse_preupload_response(&json).unwrap();
        assert!(d.should_ignore);
    }

    #[test]
    fn parse_preupload_missing_uploadmode_defaults_to_lfs() {
        // Defensive default: if HF doesn't tell us, route to LFS (the
        // safer choice — small files commit-fail with a clearer error
        // than corruption from misrouting binary as base64-text).
        let json = serde_json::json!({"files": [{}]});
        let d = parse_preupload_response(&json).unwrap();
        assert_eq!(d.mode, "lfs");
        assert!(!d.should_ignore);
    }

    #[test]
    fn parse_preupload_missing_files_array_errors() {
        let json = serde_json::json!({});
        let err = parse_preupload_response(&json).expect_err("missing files[] errors");
        assert!(err.to_string().contains("files"));
    }

    #[test]
    fn parse_preupload_empty_files_array_errors() {
        let json = serde_json::json!({"files": []});
        let err = parse_preupload_response(&json).expect_err("empty files[] errors");
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn parse_preupload_unknown_mode_passes_through() {
        // Unknown modes flow through to the upload_file_to_hf dispatch
        // which surfaces them with a targeted error. Don't reject here.
        let json = serde_json::json!({
            "files": [{"uploadMode": "future-mode"}]
        });
        let d = parse_preupload_response(&json).unwrap();
        assert_eq!(d.mode, "future-mode");
    }
}
