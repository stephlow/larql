//! HuggingFace collection / repo discovery — listing + existence
//! probes used by the CLI to wire vindexes into HF collections.
//!
//! Carved out of the monolithic `huggingface.rs` in the 2026-04-25
//! reorg. See `super::mod.rs` for the module map.

use crate::error::VindexError;

use super::publish::get_hf_token;
use super::publish::protocol::hf_base;

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
    Ok(format!("{}/collections/{slug}", hf_base()))
}

fn find_collection_slug(
    namespace: &str,
    title: &str,
    token: &str,
) -> Result<Option<String>, VindexError> {
    let client = reqwest::blocking::Client::new();
    let url = format!("{}/api/users/{namespace}/collections?limit=100", hf_base());
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
    let url = format!("{}/api/collections", hf_base());
    let resp = client
        .post(&url)
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
    let url = format!("{}/api/collections/{slug}/items", hf_base());
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
    let url = format!("{}/api/{plural}/{repo_id}", hf_base());
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
    let url = format!("{}/api/collections/{slug}", hf_base());
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
    use super::*;
    use crate::format::huggingface::publish::protocol::TEST_BASE_ENV;
    use serial_test::serial;

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

    // ─── HTTP-mocked integration tests ─────────────────────────────

    /// RAII env-var override for `LARQL_HF_TEST_BASE`, plus a fake
    /// HF_TOKEN so the discovery functions don't try to read
    /// `~/.huggingface/token` during the test. Restored on drop.
    struct TestEnvGuard {
        prev_base: Option<String>,
        prev_token: Option<String>,
    }

    impl TestEnvGuard {
        fn new(base: &str) -> Self {
            let prev_base = std::env::var(TEST_BASE_ENV).ok();
            let prev_token = std::env::var("HF_TOKEN").ok();
            std::env::set_var(TEST_BASE_ENV, base);
            std::env::set_var("HF_TOKEN", "test-token");
            Self {
                prev_base,
                prev_token,
            }
        }
    }

    impl Drop for TestEnvGuard {
        fn drop(&mut self) {
            match self.prev_base.take() {
                Some(v) => std::env::set_var(TEST_BASE_ENV, v),
                None => std::env::remove_var(TEST_BASE_ENV),
            }
            match self.prev_token.take() {
                Some(v) => std::env::set_var("HF_TOKEN", v),
                None => std::env::remove_var("HF_TOKEN"),
            }
        }
    }

    // ── repo_exists / dataset_repo_exists ──

    #[test]
    #[serial]
    fn repo_exists_returns_true_on_200() {
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let mock = server
            .mock("HEAD", "/api/models/org/repo")
            .with_status(200)
            .create();

        assert!(repo_exists("org/repo", "model").unwrap());
        mock.assert();
    }

    #[test]
    #[serial]
    fn repo_exists_returns_false_on_404() {
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let mock = server
            .mock("HEAD", "/api/models/missing/repo")
            .with_status(404)
            .create();

        assert!(!repo_exists("missing/repo", "model").unwrap());
        mock.assert();
    }

    #[test]
    #[serial]
    fn repo_exists_dataset_uses_datasets_path() {
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let mock = server
            .mock("HEAD", "/api/datasets/org/repo")
            .with_status(200)
            .create();

        assert!(repo_exists("org/repo", "dataset").unwrap());
        mock.assert();
    }

    #[test]
    #[serial]
    fn repo_exists_other_status_propagates_error() {
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let mock = server
            .mock("HEAD", "/api/models/org/repo")
            .with_status(500)
            .create();

        let err = repo_exists("org/repo", "model").expect_err("500 must error");
        mock.assert();
        assert!(err.to_string().contains("500"));
    }

    #[test]
    #[serial]
    fn dataset_repo_exists_is_thin_wrapper_over_repo_exists() {
        // dataset_repo_exists hits the `model` HEAD path (the function
        // body passes "model" — the misnomer is a known wart). Pin the
        // wire contract via the model endpoint.
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let mock = server
            .mock("HEAD", "/api/models/org/repo")
            .with_status(200)
            .create();

        assert!(dataset_repo_exists("org/repo").unwrap());
        mock.assert();
    }

    // ── fetch_collection_items ──

    #[test]
    #[serial]
    fn fetch_collection_items_parses_type_id_pairs() {
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let mock = server
            .mock("GET", "/api/collections/org/title-abc")
            .with_status(200)
            .with_body(
                serde_json::json!({
                    "items": [
                        {"type": "dataset", "id": "owner/repo-a"},
                        {"type": "model", "id": "owner/repo-b"},
                    ]
                })
                .to_string(),
            )
            .create();

        let items = fetch_collection_items("org/title-abc").unwrap();
        mock.assert();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0], ("dataset".into(), "owner/repo-a".into()));
        assert_eq!(items[1], ("model".into(), "owner/repo-b".into()));
    }

    #[test]
    #[serial]
    fn fetch_collection_items_skips_malformed_entries() {
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let mock = server
            .mock("GET", "/api/collections/org/title-abc")
            .with_status(200)
            .with_body(
                serde_json::json!({
                    "items": [
                        {"type": "dataset"},
                        {"id": "owner/no-type"},
                        {"type": "model", "id": "owner/good"},
                    ]
                })
                .to_string(),
            )
            .create();

        let items = fetch_collection_items("org/title-abc").unwrap();
        mock.assert();
        assert_eq!(items, vec![("model".into(), "owner/good".into())]);
    }

    #[test]
    #[serial]
    fn fetch_collection_items_strips_url_prefix() {
        // Callers can pass a full collection URL — the function strips
        // the prefix and treats the rest as a slug.
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let mock = server
            .mock("GET", "/api/collections/org/title-abc")
            .with_status(200)
            .with_body(r#"{"items":[]}"#)
            .create();

        let items =
            fetch_collection_items("https://huggingface.co/collections/org/title-abc").unwrap();
        mock.assert();
        assert!(items.is_empty());
    }

    #[test]
    #[serial]
    fn fetch_collection_items_strips_hf_uri_scheme() {
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let mock = server
            .mock("GET", "/api/collections/org/title-abc")
            .with_status(200)
            .with_body(r#"{"items":[]}"#)
            .create();

        let items = fetch_collection_items("hf://collections/org/title-abc").unwrap();
        mock.assert();
        assert!(items.is_empty());
    }

    #[test]
    #[serial]
    fn fetch_collection_items_http_error_propagates() {
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let mock = server
            .mock("GET", "/api/collections/org/title-abc")
            .with_status(404)
            .with_body("not found")
            .create();

        let err = fetch_collection_items("org/title-abc").expect_err("404 errors");
        mock.assert();
        assert!(err.to_string().contains("404"));
    }

    // ── add_collection_item ──

    #[test]
    #[serial]
    fn add_collection_item_sends_typed_repo_payload() {
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let mock = server
            .mock("POST", "/api/collections/org/title-abc/items")
            .match_header("authorization", "Bearer t")
            .match_body(mockito::Matcher::PartialJson(serde_json::json!({
                "item": {"type": "dataset", "id": "org/data-repo"}
            })))
            .with_status(200)
            .create();

        let item = CollectionItem {
            repo_id: "org/data-repo".into(),
            repo_type: "dataset".into(),
            note: None,
        };
        add_collection_item("org/title-abc", &item, "t").unwrap();
        mock.assert();
    }

    #[test]
    #[serial]
    fn add_collection_item_with_note_sends_note_field() {
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let mock = server
            .mock("POST", "/api/collections/org/title-abc/items")
            .match_body(mockito::Matcher::Regex(
                r#""note":\s*"why this repo""#.into(),
            ))
            .with_status(200)
            .create();

        let item = CollectionItem {
            repo_id: "org/data-repo".into(),
            repo_type: "dataset".into(),
            note: Some("why this repo".into()),
        };
        add_collection_item("org/title-abc", &item, "t").unwrap();
        mock.assert();
    }

    #[test]
    #[serial]
    fn add_collection_item_409_is_ok() {
        // 409 = item already in the collection. ensure_collection's
        // idempotency depends on this not erroring.
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let mock = server
            .mock("POST", "/api/collections/org/title-abc/items")
            .with_status(409)
            .with_body("already in collection")
            .create();

        let item = CollectionItem {
            repo_id: "org/data-repo".into(),
            repo_type: "dataset".into(),
            note: None,
        };
        add_collection_item("org/title-abc", &item, "t").unwrap();
        mock.assert();
    }

    #[test]
    #[serial]
    fn add_collection_item_other_error_propagates() {
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let mock = server
            .mock("POST", "/api/collections/org/title-abc/items")
            .with_status(403)
            .with_body("denied")
            .create();

        let item = CollectionItem {
            repo_id: "org/data-repo".into(),
            repo_type: "dataset".into(),
            note: None,
        };
        let err = add_collection_item("org/title-abc", &item, "t").expect_err("403 errors");
        mock.assert();
        assert!(err.to_string().contains("403"));
    }

    // ── ensure_collection orchestrator ──

    #[test]
    #[serial]
    fn ensure_collection_creates_when_missing() {
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        // List returns an empty array — collection doesn't exist yet.
        let list_mock = server
            .mock("GET", "/api/users/org/collections?limit=100")
            .with_status(200)
            .with_body("[]")
            .create();
        // Create succeeds.
        let create_mock = server
            .mock("POST", "/api/collections")
            .with_status(200)
            .with_body(r#"{"slug": "org/new-title-abc"}"#)
            .create();
        // Add-item succeeds.
        let add_mock = server
            .mock("POST", "/api/collections/org/new-title-abc/items")
            .with_status(200)
            .create();

        let item = CollectionItem {
            repo_id: "org/data".into(),
            repo_type: "dataset".into(),
            note: None,
        };
        let url = ensure_collection("org", "New Title", None, &[item]).unwrap();
        list_mock.assert();
        create_mock.assert();
        add_mock.assert();
        assert!(url.ends_with("/collections/org/new-title-abc"));
    }

    #[test]
    #[serial]
    fn ensure_collection_reuses_existing_slug_case_insensitive() {
        // Title match is case-insensitive — `find_collection_slug`
        // returns the existing slug and `ensure_collection` skips
        // the create endpoint entirely.
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let list_mock = server
            .mock("GET", "/api/users/org/collections?limit=100")
            .with_status(200)
            .with_body(
                serde_json::json!([
                    {"title": "MY EXISTING TITLE", "slug": "org/existing-abc"}
                ])
                .to_string(),
            )
            .create();
        // No `create` mock — if we hit it, the test fails.
        let add_mock = server
            .mock("POST", "/api/collections/org/existing-abc/items")
            .with_status(200)
            .create();

        let item = CollectionItem {
            repo_id: "org/data".into(),
            repo_type: "dataset".into(),
            note: None,
        };
        let url = ensure_collection("org", "my existing title", None, &[item]).unwrap();
        list_mock.assert();
        add_mock.assert();
        assert!(url.ends_with("/collections/org/existing-abc"));
    }

    #[test]
    #[serial]
    fn ensure_collection_uses_409_slug_when_create_conflicts() {
        // Race: list returned no match, but create returns 409 with the
        // existing slug in the body. We trust the 409 slug and proceed.
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let list_mock = server
            .mock("GET", "/api/users/org/collections?limit=100")
            .with_status(200)
            .with_body("[]")
            .create();
        let create_mock = server
            .mock("POST", "/api/collections")
            .with_status(409)
            .with_body(r#"{"slug": "org/raced-abc"}"#)
            .create();
        let add_mock = server
            .mock("POST", "/api/collections/org/raced-abc/items")
            .with_status(200)
            .create();

        let item = CollectionItem {
            repo_id: "org/data".into(),
            repo_type: "dataset".into(),
            note: None,
        };
        ensure_collection("org", "Raced Title", None, &[item]).unwrap();
        list_mock.assert();
        create_mock.assert();
        add_mock.assert();
    }

    #[test]
    #[serial]
    fn ensure_collection_create_response_missing_slug_errors() {
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let list_mock = server
            .mock("GET", "/api/users/org/collections?limit=100")
            .with_status(200)
            .with_body("[]")
            .create();
        let create_mock = server
            .mock("POST", "/api/collections")
            .with_status(200)
            .with_body(r#"{"oops": "no slug"}"#)
            .create();

        let err = ensure_collection("org", "Bad", None, &[]).expect_err("missing slug errors");
        list_mock.assert();
        create_mock.assert();
        assert!(err.to_string().contains("slug"));
    }

    #[test]
    #[serial]
    fn find_collection_slug_handles_404_as_no_match() {
        // 404 list → returns Ok(None) (function falls through to create).
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let list_mock = server
            .mock("GET", "/api/users/org/collections?limit=100")
            .with_status(404)
            .create();
        // Falls through to create on the missing-list path.
        let create_mock = server
            .mock("POST", "/api/collections")
            .with_status(200)
            .with_body(r#"{"slug": "org/created-abc"}"#)
            .create();
        let add_mock = server
            .mock("POST", "/api/collections/org/created-abc/items")
            .with_status(200)
            .create();

        let item = CollectionItem {
            repo_id: "org/data".into(),
            repo_type: "dataset".into(),
            note: None,
        };
        ensure_collection("org", "Title", None, &[item]).unwrap();
        list_mock.assert();
        create_mock.assert();
        add_mock.assert();
    }

    #[test]
    #[serial]
    fn find_collection_slug_other_list_error_propagates() {
        // 500 on list is a real failure — `ensure_collection` propagates
        // the error rather than falling through to create.
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let list_mock = server
            .mock("GET", "/api/users/org/collections?limit=100")
            .with_status(500)
            .with_body("server error")
            .create();

        let err = ensure_collection("org", "Title", None, &[]).expect_err("500 errors");
        list_mock.assert();
        assert!(err.to_string().contains("500"));
    }

    #[test]
    #[serial]
    fn create_collection_includes_description_when_provided() {
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let list_mock = server
            .mock("GET", "/api/users/org/collections?limit=100")
            .with_status(200)
            .with_body("[]")
            .create();
        let create_mock = server
            .mock("POST", "/api/collections")
            .match_body(mockito::Matcher::PartialJson(serde_json::json!({
                "description": "my collection"
            })))
            .with_status(200)
            .with_body(r#"{"slug": "org/with-desc"}"#)
            .create();
        let add_mock = server
            .mock("POST", "/api/collections/org/with-desc/items")
            .with_status(200)
            .create();

        ensure_collection("org", "Titled", Some("my collection"), &[]).unwrap();
        list_mock.assert();
        create_mock.assert();
        let _ = add_mock; // no items → never invoked
    }
}
