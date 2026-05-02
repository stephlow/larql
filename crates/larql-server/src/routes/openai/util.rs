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

/// Build the sampling + EOS config from OpenAI request parameters.
///
/// - `temperature`: 0.0 (or `None`) → greedy. Otherwise temperature
///   sampling. OpenAI default is 1.0, but we default to greedy when
///   the field is omitted so existing tests / curl one-liners stay
///   deterministic.
/// - `top_p`: nucleus filter; only applied when temperature > 0.
/// - `seed`: deterministic RNG. Same seed + same inputs = same tokens.
/// - `stop`: extends the model's built-in EOS stop strings; first
///   match halts generation mid-stream (not post-trimmed).
pub fn build_sampling_eos(
    temperature: Option<f32>,
    top_p: Option<f32>,
    seed: Option<u64>,
    stop_strings: &[String],
) -> (larql_inference::SamplingConfig, larql_inference::EosConfig) {
    let temp = temperature.unwrap_or(0.0).max(0.0);
    let mut sampling = if temp > 0.0 {
        larql_inference::SamplingConfig::temperature(temp)
    } else {
        larql_inference::SamplingConfig::greedy()
    };
    if let Some(p) = top_p {
        // Only honour top_p when sampling is on; for greedy it's a no-op.
        if temp > 0.0 && (0.0..=1.0).contains(&p) {
            sampling = sampling.with_top_p(p);
        }
    }
    if let Some(s) = seed {
        sampling = sampling.with_seed(s);
    }
    let mut eos = larql_inference::EosConfig::builtin();
    for s in stop_strings {
        if !s.is_empty() {
            eos = eos.with_stop_string(s.clone());
        }
    }
    (sampling, eos)
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

    #[test]
    fn build_sampling_eos_defaults_to_greedy() {
        let (sampling, _eos) = build_sampling_eos(None, None, None, &[]);
        assert!(sampling.is_greedy());
    }

    #[test]
    fn build_sampling_eos_zero_temperature_is_greedy() {
        let (sampling, _eos) = build_sampling_eos(Some(0.0), Some(0.9), Some(7), &[]);
        // Zero temperature collapses to greedy regardless of top_p / seed.
        assert!(sampling.is_greedy());
    }

    #[test]
    fn build_sampling_eos_temperature_enables_sampling() {
        let (sampling, _eos) = build_sampling_eos(Some(0.7), None, None, &[]);
        assert!(!sampling.is_greedy());
        assert!((sampling.temperature - 0.7).abs() < 1e-6);
        assert!(sampling.top_p.is_none());
        assert!(sampling.seed.is_none());
    }

    #[test]
    fn build_sampling_eos_top_p_only_with_temperature() {
        // top_p with temperature > 0 → applied.
        let (sampling, _eos) = build_sampling_eos(Some(0.8), Some(0.9), None, &[]);
        assert_eq!(sampling.top_p, Some(0.9));

        // top_p with temperature == 0 → ignored (greedy can't nucleus).
        let (sampling, _eos) = build_sampling_eos(Some(0.0), Some(0.9), None, &[]);
        assert!(sampling.top_p.is_none());
    }

    #[test]
    fn build_sampling_eos_top_p_out_of_range_dropped() {
        // OpenAI rejects top_p > 1.0; we silently drop instead of erroring.
        let (sampling, _eos) = build_sampling_eos(Some(0.8), Some(1.5), None, &[]);
        assert!(sampling.top_p.is_none());
        let (sampling, _eos) = build_sampling_eos(Some(0.8), Some(-0.1), None, &[]);
        assert!(sampling.top_p.is_none());
    }

    #[test]
    fn build_sampling_eos_seed_carried_through() {
        let (sampling, _eos) = build_sampling_eos(Some(0.7), None, Some(42), &[]);
        assert_eq!(sampling.seed, Some(42));
    }

    #[test]
    fn build_sampling_eos_negative_temperature_clamped() {
        let (sampling, _eos) = build_sampling_eos(Some(-0.5), None, None, &[]);
        assert!(sampling.is_greedy());
    }

    #[test]
    fn build_sampling_eos_stop_strings_added() {
        let (_, eos_baseline) = build_sampling_eos(None, None, None, &[]);
        let (_, eos) = build_sampling_eos(None, None, None, &["\n\n".into(), "STOP".into()]);
        // Caller's stop strings are appended on top of the baseline.
        assert_eq!(eos.stop_strings.len(), eos_baseline.stop_strings.len() + 2);
        assert!(eos.stop_strings.iter().any(|s| s == "\n\n"));
        assert!(eos.stop_strings.iter().any(|s| s == "STOP"));
    }

    #[test]
    fn build_sampling_eos_empty_stop_strings_skipped() {
        let (_, eos_baseline) = build_sampling_eos(None, None, None, &[]);
        let (_, eos) = build_sampling_eos(None, None, None, &["".into(), "x".into()]);
        // Empty needles are skipped; only "x" should be added.
        assert_eq!(eos.stop_strings.len(), eos_baseline.stop_strings.len() + 1);
        assert!(eos.stop_strings.iter().any(|s| s == "x"));
        assert!(!eos.stop_strings.iter().any(|s| s.is_empty()));
    }
}
