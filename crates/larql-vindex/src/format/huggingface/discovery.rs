//! HuggingFace collection / repo discovery — listing + existence
//! probes used by the CLI to wire vindexes into HF collections.
//!
//! Carved out of the monolithic `huggingface.rs` in the 2026-04-25
//! reorg. See `super::mod.rs` for the module map.

use crate::error::VindexError;

use super::publish::get_hf_token;

// ═══════════════════════════════════════════════════════════════
// Collections
// ═══════════════════════════════════════════════════════════════

/// One repo in a collection.
#[derive(Clone, Debug)]
pub struct CollectionItem {
    /// Repo id (`owner/name`). Full form including namespace.
    pub repo_id: String,
    /// `"model"` (vindex repos, default) or `"dataset"`.
    pub repo_type: String,
    /// Optional short note rendered on the collection card.
    pub note: Option<String>,
}

/// Ensure a collection titled `title` exists in `namespace`, then add
/// every item to it. Idempotent: re-runs reuse the slug (matched by
/// case-insensitive title) and treat HTTP 409 on add-item as success.
/// Returns the collection URL on success.
pub fn ensure_collection(
    namespace: &str,
    title: &str,
    description: Option<&str>,
    items: &[CollectionItem],
) -> Result<String, VindexError> {
    let token = get_hf_token()?;
    let slug = match find_collection_slug(namespace, title, &token)? {
        Some(existing) => existing,
        None => create_collection(namespace, title, description, &token)?,
    };
    for item in items {
        add_collection_item(&slug, item, &token)?;
    }
    Ok(format!("https://huggingface.co/collections/{slug}"))
}

fn find_collection_slug(
    namespace: &str,
    title: &str,
    token: &str,
) -> Result<Option<String>, VindexError> {
    let client = reqwest::blocking::Client::new();
    let url = format!("https://huggingface.co/api/users/{namespace}/collections?limit=100");
    let resp = client
        .get(&url)
        .header("Authorization", format!("Bearer {token}"))
        .send()
        .map_err(|e| VindexError::Parse(format!("HF collections list failed: {e}")))?;
    if !resp.status().is_success() {
        if resp.status().as_u16() == 404 {
            return Ok(None);
        }
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        return Err(VindexError::Parse(format!(
            "HF collections list ({status}): {body}"
        )));
    }
    let body: serde_json::Value = resp
        .json()
        .map_err(|e| VindexError::Parse(format!("HF collections JSON: {e}")))?;
    let arr = match body.as_array() {
        Some(a) => a,
        None => return Ok(None),
    };
    let target = title.to_ascii_lowercase();
    for entry in arr {
        let entry_title = entry.get("title").and_then(|v| v.as_str()).unwrap_or("");
        if entry_title.to_ascii_lowercase() == target {
            if let Some(slug) = entry.get("slug").and_then(|v| v.as_str()) {
                return Ok(Some(slug.to_string()));
            }
        }
    }
    Ok(None)
}

fn create_collection(
    namespace: &str,
    title: &str,
    description: Option<&str>,
    token: &str,
) -> Result<String, VindexError> {
    let client = reqwest::blocking::Client::new();
    let mut body = serde_json::json!({
        "title": title,
        "namespace": namespace,
        "private": false,
    });
    if let Some(desc) = description {
        body["description"] = serde_json::Value::String(desc.to_string());
    }
    let resp = client
        .post("https://huggingface.co/api/collections")
        .header("Authorization", format!("Bearer {token}"))
        .json(&body)
        .send()
        .map_err(|e| VindexError::Parse(format!("HF collection create failed: {e}")))?;

    let status = resp.status();
    let body_text = resp.text().unwrap_or_default();

    // Happy path — new collection created.
    if status.is_success() {
        let json: serde_json::Value = serde_json::from_str(&body_text)
            .map_err(|e| VindexError::Parse(format!("HF collection JSON: {e}")))?;
        let slug = json
            .get("slug")
            .and_then(|v| v.as_str())
            .ok_or_else(|| VindexError::Parse("HF collection response missing slug".into()))?;
        return Ok(slug.to_string());
    }

    // 409 Conflict — collection already exists. HF returns the existing
    // slug in the error body. We hit this when `find_collection_slug`
    // failed to find it (e.g. auth scope / list pagination issues) but
    // the collection does exist. Short-circuiting here is the robust
    // path regardless of why find missed it.
    if status.as_u16() == 409 {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body_text) {
            if let Some(slug) = json.get("slug").and_then(|v| v.as_str()) {
                return Ok(slug.to_string());
            }
        }
    }

    Err(VindexError::Parse(format!(
        "HF collection create ({status}): {body_text}"
    )))
}

pub fn add_collection_item(
    slug: &str,
    item: &CollectionItem,
    token: &str,
) -> Result<(), VindexError> {
    let client = reqwest::blocking::Client::new();
    // HF's collection API uses `/items` (plural) for POST-to-append.
    // The singular form is only valid as `PATCH/DELETE
    // /api/collections/{slug}/item/{item_id}` for editing an existing
    // entry. Got caught by this on the first real publish — the add
    // failed with 404 after the four repos had already uploaded fine.
    let url = format!("https://huggingface.co/api/collections/{slug}/items");
    let mut body = serde_json::json!({
        "item": {
            "type": item.repo_type,
            "id": item.repo_id,
        },
    });
    if let Some(note) = &item.note {
        body["note"] = serde_json::Value::String(note.clone());
    }
    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {token}"))
        .json(&body)
        .send()
        .map_err(|e| VindexError::Parse(format!("HF collection add-item failed: {e}")))?;
    if resp.status().is_success() || resp.status().as_u16() == 409 {
        Ok(())
    } else {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        Err(VindexError::Parse(format!(
            "HF collection add-item ({status}): {body}"
        )))
    }
}

/// Cheap HEAD probe — returns `Ok(true)` if the dataset repo exists and
/// is readable, `Ok(false)` on 404, `Err` on other failures. Auth is
/// optional; pass-through when available (lets callers see private
/// repos they own).
pub fn dataset_repo_exists(repo_id: &str) -> Result<bool, VindexError> {
    repo_exists(repo_id, "model")
}

pub fn repo_exists(repo_id: &str, repo_type: &str) -> Result<bool, VindexError> {
    let token = get_hf_token().ok();
    let plural = if repo_type == "dataset" {
        "datasets"
    } else {
        "models"
    };
    let url = format!("https://huggingface.co/api/{plural}/{repo_id}");
    let client = reqwest::blocking::Client::new();
    let mut req = client.head(&url);
    if let Some(t) = token {
        req = req.header("Authorization", format!("Bearer {t}"));
    }
    let resp = req
        .send()
        .map_err(|e| VindexError::Parse(format!("HF HEAD failed: {e}")))?;
    if resp.status().is_success() {
        Ok(true)
    } else if resp.status().as_u16() == 404 {
        Ok(false)
    } else {
        Err(VindexError::Parse(format!(
            "HF HEAD {repo_id}: {}",
            resp.status()
        )))
    }
}

/// Fetch a collection by slug (or full collection URL) and return its
/// items as `(type, id)` pairs — typically `("dataset", "owner/name")`.
pub fn fetch_collection_items(slug_or_url: &str) -> Result<Vec<(String, String)>, VindexError> {
    let slug = slug_or_url
        .trim_start_matches("https://huggingface.co/collections/")
        .trim_start_matches("http://huggingface.co/collections/")
        .trim_start_matches("hf://collections/")
        .trim_start_matches('/');
    let token = get_hf_token().ok();
    let url = format!("https://huggingface.co/api/collections/{slug}");
    let client = reqwest::blocking::Client::new();
    let mut req = client.get(&url);
    if let Some(t) = token {
        req = req.header("Authorization", format!("Bearer {t}"));
    }
    let resp = req
        .send()
        .map_err(|e| VindexError::Parse(format!("HF collection fetch failed: {e}")))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        return Err(VindexError::Parse(format!(
            "HF collection fetch ({status}): {body}"
        )));
    }
    let body: serde_json::Value = resp
        .json()
        .map_err(|e| VindexError::Parse(format!("HF collection JSON: {e}")))?;
    let items = body
        .get("items")
        .and_then(|v| v.as_array())
        .ok_or_else(|| VindexError::Parse("collection response missing items".into()))?;
    let mut out = Vec::new();
    for item in items {
        let kind = match item.get("type").and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        let id = match item.get("id").and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        out.push((kind, id));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::super::is_hf_path;

    #[test]
    fn test_is_hf_path() {
        assert!(is_hf_path("hf://chrishayuk/gemma-3-4b-it-vindex"));
        assert!(is_hf_path("hf://user/repo@v1.0"));
        assert!(!is_hf_path("./local.vindex"));
        assert!(!is_hf_path("/absolute/path"));
    }

    #[test]
    fn test_parse_hf_path() {
        let path = "hf://chrishayuk/gemma-3-4b-it-vindex@v2.0";
        let stripped = path.strip_prefix("hf://").unwrap();
        let (repo, rev) = stripped.split_once('@').unwrap();
        assert_eq!(repo, "chrishayuk/gemma-3-4b-it-vindex");
        assert_eq!(rev, "v2.0");
    }
}
