//! HuggingFace API helpers for repo-level state — remote LFS index lookup
//! and repo creation. Both are blocking HTTP calls used by the publish
//! orchestrator before any per-file upload runs.

use std::collections::HashMap;

use crate::error::VindexError;

use super::protocol::{repo_type_plural, HTTP_STATUS_CONFLICT};

/// List remote files and return `filename → lfs.oid` for every LFS-tracked
/// file at the repo root. Files without an `lfs.oid` (git-tracked small
/// text) are omitted; callers skip only what's in the map.
pub(super) fn fetch_remote_lfs_oids(
    repo_id: &str,
    token: &str,
    repo_type: &str,
) -> Result<HashMap<String, String>, VindexError> {
    let plural = repo_type_plural(repo_type);
    let url = format!("https://huggingface.co/api/{plural}/{repo_id}/tree/main?recursive=true");
    let client = reqwest::blocking::Client::new();
    let resp = client
        .get(&url)
        .header("Authorization", format!("Bearer {token}"))
        .send()
        .map_err(|e| VindexError::Parse(format!("HF tree fetch failed: {e}")))?;

    if !resp.status().is_success() {
        // 404 on a fresh repo → no remote files, can't skip anything.
        return Ok(HashMap::new());
    }

    let body: serde_json::Value = resp
        .json()
        .map_err(|e| VindexError::Parse(format!("HF tree JSON: {e}")))?;
    Ok(parse_lfs_oid_index(&body))
}

/// Walk the HF tree-listing JSON and return `filename → lfs.oid` for
/// every LFS-tracked file. Files without an `lfs.oid` (small text /
/// directories) are omitted. Pulled out as a pure helper so the JSON
/// contract can be unit-tested without an HTTP server.
fn parse_lfs_oid_index(body: &serde_json::Value) -> HashMap<String, String> {
    let arr = match body.as_array() {
        Some(a) => a,
        None => return HashMap::new(),
    };

    let mut out = HashMap::new();
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
    out
}

pub(super) fn create_hf_repo(
    repo_id: &str,
    token: &str,
    repo_type: &str,
) -> Result<(), VindexError> {
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

    // 409 Conflict = already exists, that's fine
    if resp.status().is_success() || resp.status().as_u16() == HTTP_STATUS_CONFLICT {
        Ok(())
    } else {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        Err(VindexError::Parse(format!(
            "HF repo create failed ({status}): {body}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_oid_index_extracts_lfs_files_only() {
        // Mixed entries: LFS file, non-LFS file (small text), directory.
        // Only the LFS entry should appear in the map.
        let body = serde_json::json!([
            {
                "type": "file",
                "path": "weights.bin",
                "lfs": {"oid": "abc123", "size": 1024}
            },
            {
                "type": "file",
                "path": "index.json"
                // no lfs key — git-tracked small text
            },
            {
                "type": "directory",
                "path": "layers",
                "lfs": {"oid": "should-not-appear"}
            }
        ]);
        let map = parse_lfs_oid_index(&body);
        assert_eq!(map.len(), 1);
        assert_eq!(map.get("weights.bin").map(|s| s.as_str()), Some("abc123"));
    }

    #[test]
    fn parse_oid_index_handles_subdir_paths() {
        // HF returns paths like "layers/layer_00.weights" — they should
        // round-trip through the map verbatim (the publish code uses
        // them as filename keys).
        let body = serde_json::json!([
            {
                "type": "file",
                "path": "layers/layer_00.weights",
                "lfs": {"oid": "deadbeef"}
            }
        ]);
        let map = parse_lfs_oid_index(&body);
        assert_eq!(
            map.get("layers/layer_00.weights").map(|s| s.as_str()),
            Some("deadbeef"),
        );
    }

    #[test]
    fn parse_oid_index_non_array_body_yields_empty_map() {
        // Fresh repo / unauth → HF can return a non-array body. Non-fatal:
        // caller falls back to "upload everything".
        let body = serde_json::json!({"error": "not found"});
        assert!(parse_lfs_oid_index(&body).is_empty());
    }

    #[test]
    fn parse_oid_index_empty_array_yields_empty_map() {
        let body = serde_json::json!([]);
        assert!(parse_lfs_oid_index(&body).is_empty());
    }

    #[test]
    fn parse_oid_index_missing_path_skips_entry() {
        // Defensive: malformed entries don't poison the whole walk.
        let body = serde_json::json!([
            {"type": "file", "lfs": {"oid": "x"}},
            {
                "type": "file",
                "path": "good.bin",
                "lfs": {"oid": "y"}
            }
        ]);
        let map = parse_lfs_oid_index(&body);
        assert_eq!(map.len(), 1);
        assert_eq!(map.get("good.bin").map(|s| s.as_str()), Some("y"));
    }
}
