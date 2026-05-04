//! HTTP wire-format helpers shared by routes that accept both binary and
//! JSON request bodies (walk-ffn, embed, expert/batch).
//!
//! The detection uses `contains` rather than `starts_with` so that
//! parameterised types (`application/json; charset=utf-8`,
//! `application/x-larql-ffn; v=2`) match. The binary content types we
//! advertise (`application/x-larql-ffn`, `application/x-larql-expert`)
//! are unique enough that no ambiguity arises.

use axum::http::header;
use axum::http::HeaderMap;

/// Returns `true` when the `Content-Type` header on `headers` contains the
/// substring `expected` (e.g. an `application/x-larql-ffn` binary type).
pub fn has_content_type(headers: &HeaderMap, expected: &str) -> bool {
    headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.contains(expected))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    fn hm(ct: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(header::CONTENT_TYPE, HeaderValue::from_str(ct).unwrap());
        h
    }

    #[test]
    fn matches_exact_type() {
        assert!(has_content_type(
            &hm("application/x-larql-ffn"),
            "application/x-larql-ffn"
        ));
    }

    #[test]
    fn matches_with_parameters() {
        assert!(has_content_type(
            &hm("application/json; charset=utf-8"),
            "application/json"
        ));
    }

    #[test]
    fn does_not_match_other_type() {
        assert!(!has_content_type(
            &hm("application/json"),
            "application/x-larql-ffn"
        ));
    }

    #[test]
    fn missing_header_does_not_match() {
        assert!(!has_content_type(&HeaderMap::new(), "application/json"));
    }
}
