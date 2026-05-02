//! Shared helpers used across OpenAI endpoints.
//!
//! These were originally duplicated in `chat.rs` and `completions.rs`
//! (and partly in `embeddings.rs`); centralised here so both buffered
//! and SSE paths share one source of truth for id formatting, time
//! stamping, stop-string handling, and the SSE error envelope.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::Deserialize;

/// Stop strings — accepted as either a single string or a list.
/// OpenAI's `stop` field allows both forms.
#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum StopSpec {
    Single(String),
    Multi(Vec<String>),
}

impl StopSpec {
    pub fn as_slice(&self) -> &[String] {
        match self {
            StopSpec::Single(s) => std::slice::from_ref(s),
            StopSpec::Multi(v) => v.as_slice(),
        }
    }
}

/// Unix epoch seconds — used as the OpenAI `created` field on every
/// response and stream chunk.
pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Generate a short hex id suffix for `cmpl-...` / `chatcmpl-...`.
/// Not cryptographically strong; uniqueness across one server lifetime
/// is sufficient.
pub fn new_id_suffix() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let now_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    format!("{now_ns:016x}{n:08x}")
}

/// Returns true if any non-empty needle appears as a substring of
/// `haystack`. Used to halt generation on stop strings.
pub fn contains_any(haystack: &str, needles: &[String]) -> bool {
    needles
        .iter()
        .any(|n| !n.is_empty() && haystack.contains(n.as_str()))
}

/// Trim `haystack` at the first occurrence of any (non-empty) needle.
/// Used by the buffered `/v1/completions` path to chop the matched
/// stop string off the returned text.
pub fn trim_at_stop(haystack: &str, needles: &[String]) -> String {
    let mut earliest: Option<usize> = None;
    for n in needles {
        if n.is_empty() {
            continue;
        }
        if let Some(idx) = haystack.find(n.as_str()) {
            earliest = Some(earliest.map_or(idx, |e| e.min(idx)));
        }
    }
    match earliest {
        Some(i) => haystack[..i].to_string(),
        None => haystack.to_string(),
    }
}

/// Format a JSON error chunk for SSE error paths. Wraps in OpenAI's
/// `{error: {message, type}}` envelope so clients see a structured
/// failure mid-stream rather than a truncated success response.
pub fn error_chunk(msg: &str) -> String {
    serde_json::json!({"error": {"message": msg, "type": "server_error"}}).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_spec_single_or_multi() {
        let single: StopSpec = serde_json::from_value(serde_json::json!("\\n")).unwrap();
        assert_eq!(single.as_slice(), &["\\n".to_string()]);
        let multi: StopSpec = serde_json::from_value(serde_json::json!(["a", "b"])).unwrap();
        assert_eq!(multi.as_slice(), &["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn trim_at_stop_finds_earliest() {
        let s = "hello world stop here";
        let stops = vec!["stop".to_string(), "world".to_string()];
        assert_eq!(trim_at_stop(s, &stops), "hello ");
    }

    #[test]
    fn trim_at_stop_no_match_returns_input() {
        let s = "hello world";
        let stops = vec!["xx".to_string()];
        assert_eq!(trim_at_stop(s, &stops), s);
    }

    #[test]
    fn contains_any_matches_substring() {
        let stops = vec!["END".to_string()];
        assert!(contains_any("text END more", &stops));
        assert!(!contains_any("text only", &stops));
    }

    #[test]
    fn contains_any_skips_empty_needles() {
        let stops = vec!["".to_string()];
        assert!(!contains_any("text", &stops));
    }

    #[test]
    fn new_id_suffix_is_unique_within_thread() {
        let a = new_id_suffix();
        let b = new_id_suffix();
        assert_ne!(a, b);
        assert_eq!(a.len(), b.len());
    }

    #[test]
    fn unix_now_is_recent() {
        let now = unix_now();
        // 1 Jan 2024 in unix seconds = 1704067200; safety margin against
        // a clock badly out of sync.
        assert!(now > 1_700_000_000);
    }

    #[test]
    fn error_chunk_returns_openai_shape() {
        let chunk = error_chunk("oops");
        let v: serde_json::Value = serde_json::from_str(&chunk).unwrap();
        assert_eq!(v["error"]["message"], "oops");
        assert_eq!(v["error"]["type"], "server_error");
    }
}
