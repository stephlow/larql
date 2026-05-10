//! Remote executor — forwards LQL queries to a `larql-server` over HTTP.
//!
//! Layout:
//!
//!   - `mod.rs` (this file): connection plumbing (`exec_use_remote`,
//!     `is_remote`, `require_remote`), generic HTTP helpers, and
//!     centralised endpoint / header constants.
//!   - `query.rs`: read-side verbs (DESCRIBE, WALK, INFER, EXPLAIN
//!     INFER, STATS, SHOW RELATIONS, SELECT) plus shared `knn_override`
//!     formatters.
//!   - `mutation.rs`: write-side verbs (INSERT, DELETE, UPDATE) plus
//!     local-patch management (APPLY/SHOW/REMOVE PATCH).

use super::Backend;
use super::Session;
use crate::ast::{Condition, Value};
use crate::error::LqlError;

mod mutation;
mod query;

// ── Endpoint constants ───────────────────────────────────────────
//
// Every remote forwarder used to embed its `/v1/foo` path inline. A
// stale or moved endpoint would silently 404 across multiple call
// sites; centralising means a server-side rename is a one-line patch.

pub(super) const ENDPOINT_STATS: &str = "/v1/stats";
pub(super) const ENDPOINT_DESCRIBE: &str = "/v1/describe";
pub(super) const ENDPOINT_WALK: &str = "/v1/walk";
pub(super) const ENDPOINT_INFER: &str = "/v1/infer";
pub(super) const ENDPOINT_EXPLAIN_INFER: &str = "/v1/explain-infer";
pub(super) const ENDPOINT_RELATIONS: &str = "/v1/relations";
pub(super) const ENDPOINT_SELECT: &str = "/v1/select";
pub(super) const ENDPOINT_INSERT: &str = "/v1/insert";
pub(super) const ENDPOINT_PATCHES_APPLY: &str = "/v1/patches/apply";

/// HTTP header used to identify the per-session patch queue on the
/// server. Sent on POSTs that mutate session state.
pub(super) const SESSION_HEADER: &str = "x-session-id";

/// Default request timeout. Long enough to absorb a deep-walk INFER
/// on a 4B-class model; short enough that a hung server doesn't
/// freeze the REPL indefinitely.
pub(super) const DEFAULT_REMOTE_TIMEOUT_SECS: u64 = 30;

// ── WHERE-clause helpers ─────────────────────────────────────────
//
// Used by remote DELETE / UPDATE: both verbs require an explicit
// `layer = N AND feature = M` predicate; missing fields used to
// silently default to `0` and would clobber `L0 F0` on the server.

/// Require an explicit `layer = N AND feature = M` from a remote
/// DELETE / UPDATE WHERE clause. Errors with a verb-tagged message
/// when either is missing or non-positive-integer.
pub(super) fn require_layer_feature(
    conditions: &[Condition],
    verb: &str,
) -> Result<(usize, usize), LqlError> {
    let layer = lookup_usize_condition(conditions, "layer").ok_or_else(|| {
        LqlError::Execution(format!(
            "remote {verb} requires `layer = <int>` in WHERE clause"
        ))
    })?;
    let feature = lookup_usize_condition(conditions, "feature").ok_or_else(|| {
        LqlError::Execution(format!(
            "remote {verb} requires `feature = <int>` in WHERE clause"
        ))
    })?;
    Ok((layer, feature))
}

pub(super) fn lookup_usize_condition(
    conditions: &[Condition],
    field: &str,
) -> Option<usize> {
    conditions
        .iter()
        .find(|c| c.field == field)
        .and_then(|c| match &c.value {
            Value::Integer(n) if *n >= 0 => Some(*n as usize),
            _ => None,
        })
}

impl Session {
    /// Connect to a remote larql-server.
    pub(crate) fn exec_use_remote(&mut self, url: &str) -> Result<Vec<String>, LqlError> {
        let url = url.trim_end_matches('/').to_string();

        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(DEFAULT_REMOTE_TIMEOUT_SECS))
            .build()
            .map_err(|e| LqlError::exec("failed to create HTTP client", e))?;

        // Verify the server is reachable by hitting /v1/stats.
        let stats_url = format!("{url}{ENDPOINT_STATS}");
        let resp = client
            .get(&stats_url)
            .send()
            .map_err(|e| LqlError::exec(format!("failed to connect to {url}"), e))?;

        if !resp.status().is_success() {
            return Err(LqlError::Execution(format!(
                "server returned {}: {}",
                resp.status(),
                resp.text().unwrap_or_default()
            )));
        }

        let stats: serde_json::Value = resp
            .json()
            .map_err(|e| LqlError::exec("invalid response from server", e))?;

        let model = stats["model"].as_str().unwrap_or("unknown");
        let layers = stats["layers"].as_u64().unwrap_or(0);
        let features = stats["features"].as_u64().unwrap_or(0);

        // Per-session id for server-side patch queue. PID + millis is
        // weak entropy but acceptable for the localhost dev model;
        // production deployments should authenticate before this is
        // load-bearing.
        let session_id = format!(
            "larql-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        );

        self.backend = Backend::Remote {
            url: url.clone(),
            client,
            local_patches: Vec::new(),
            session_id,
        };
        self.patch_recording = None;
        self.auto_patch = false;

        Ok(vec![format!(
            "Connected: {model} ({layers} layers, {features} features)\n  Remote: {url}"
        )])
    }

    /// True iff the active backend is the Remote variant.
    pub(crate) fn is_remote(&self) -> bool {
        matches!(&self.backend, Backend::Remote { .. })
    }

    /// Get the remote URL, client, and session ID, or error.
    pub(super) fn require_remote(
        &self,
    ) -> Result<(&str, &reqwest::blocking::Client, &str), LqlError> {
        match &self.backend {
            Backend::Remote {
                url,
                client,
                session_id,
                ..
            } => Ok((url, client, session_id)),
            _ => Err(LqlError::Execution(
                "not connected to a remote server".into(),
            )),
        }
    }

    // ── Generic HTTP forwarding helpers ──

    /// GET `{remote_url}{endpoint}` with optional query parameters,
    /// check the response status, and parse the body as JSON.
    pub(super) fn remote_get_json(
        &self,
        endpoint: &str,
        query: &[(&str, &str)],
    ) -> Result<serde_json::Value, LqlError> {
        let (url, client, _sid) = self.require_remote()?;
        let resp = client
            .get(format!("{url}{endpoint}"))
            .query(query)
            .send()
            .map_err(|e| LqlError::exec("request failed", e))?;
        Self::check_and_parse(endpoint, resp)
    }

    /// POST `{remote_url}{endpoint}` with a JSON body. When
    /// `with_session` is true, attaches the `x-session-id` header so
    /// the server can route this request to the correct session-side
    /// patch queue.
    pub(super) fn remote_post_json(
        &self,
        endpoint: &str,
        body: &serde_json::Value,
        with_session: bool,
    ) -> Result<serde_json::Value, LqlError> {
        let (url, client, sid) = self.require_remote()?;
        let mut req = client.post(format!("{url}{endpoint}")).json(body);
        if with_session {
            req = req.header(SESSION_HEADER, sid);
        }
        let resp = req
            .send()
            .map_err(|e| LqlError::exec("request failed", e))?;
        Self::check_and_parse(endpoint, resp)
    }

    fn check_and_parse(
        endpoint: &str,
        resp: reqwest::blocking::Response,
    ) -> Result<serde_json::Value, LqlError> {
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().unwrap_or_default();
            return Err(LqlError::Execution(format!(
                "{endpoint} failed ({status}): {text}"
            )));
        }
        resp.json::<serde_json::Value>()
            .map_err(|e| LqlError::exec("invalid response", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::CompareOp;

    fn cond(field: &str, value: Value) -> Condition {
        Condition {
            field: field.into(),
            op: CompareOp::Eq,
            value,
        }
    }

    #[test]
    fn require_layer_feature_accepts_explicit_pair() {
        let conds = vec![
            cond("layer", Value::Integer(7)),
            cond("feature", Value::Integer(42)),
        ];
        let (l, f) = require_layer_feature(&conds, "DELETE").unwrap();
        assert_eq!((l, f), (7, 42));
    }

    #[test]
    fn require_layer_feature_errors_when_layer_missing() {
        // Regression: prior shape silently coerced missing fields to 0,
        // turning `WHERE foo = 1` into a destructive `DELETE L0 F0`.
        let conds = vec![cond("feature", Value::Integer(1))];
        let err = require_layer_feature(&conds, "DELETE").unwrap_err();
        assert!(err.to_string().contains("layer"));
        assert!(err.to_string().contains("DELETE"));
    }

    #[test]
    fn require_layer_feature_errors_when_feature_missing() {
        let conds = vec![cond("layer", Value::Integer(0))];
        let err = require_layer_feature(&conds, "UPDATE").unwrap_err();
        assert!(err.to_string().contains("feature"));
        assert!(err.to_string().contains("UPDATE"));
    }

    #[test]
    fn require_layer_feature_rejects_non_integer_value() {
        let conds = vec![
            cond("layer", Value::String("oops".into())),
            cond("feature", Value::Integer(1)),
        ];
        let err = require_layer_feature(&conds, "DELETE").unwrap_err();
        assert!(err.to_string().contains("layer"));
    }

    #[test]
    fn require_layer_feature_rejects_negative_value() {
        let conds = vec![
            cond("layer", Value::Integer(-1)),
            cond("feature", Value::Integer(0)),
        ];
        let err = require_layer_feature(&conds, "DELETE").unwrap_err();
        assert!(err.to_string().contains("layer"));
    }

    #[test]
    fn lookup_usize_condition_finds_field() {
        let conds = vec![cond("layer", Value::Integer(3))];
        assert_eq!(lookup_usize_condition(&conds, "layer"), Some(3));
        assert_eq!(lookup_usize_condition(&conds, "feature"), None);
    }

    #[test]
    fn endpoint_constants_share_prefix() {
        // Pinned: every endpoint hangs off `/v1/`. A future v2 server
        // would need a deliberate sweep here.
        for ep in [
            ENDPOINT_STATS,
            ENDPOINT_DESCRIBE,
            ENDPOINT_WALK,
            ENDPOINT_INFER,
            ENDPOINT_EXPLAIN_INFER,
            ENDPOINT_RELATIONS,
            ENDPOINT_SELECT,
            ENDPOINT_INSERT,
            ENDPOINT_PATCHES_APPLY,
        ] {
            assert!(
                ep.starts_with("/v1/"),
                "expected /v1/-prefixed endpoint, got {ep:?}"
            );
        }
    }

    #[test]
    fn default_timeout_is_reasonable() {
        // < 5s would race common deep-walk INFER; > 10min would let
        // the REPL freeze on a hung server.
        assert!(DEFAULT_REMOTE_TIMEOUT_SECS >= 5);
        assert!(DEFAULT_REMOTE_TIMEOUT_SECS <= 600);
    }
}
