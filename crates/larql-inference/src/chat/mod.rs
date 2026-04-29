//! Chat-template prompt wrapping, driven by the template that ships with
//! the model.
//!
//! **How it works.** The extractor snapshots the template source files
//! (`tokenizer_config.json`, `chat_template.jinja`, …) from the HF source
//! directory into the vindex — see [`larql_vindex::snapshot_hf_metadata`].
//! At runtime the [`source`] layer resolves a template string, the
//! [`render`] layer evaluates it with `minijinja` against a single user
//! turn (`add_generation_prompt=True` — same call shape as HF's
//! `apply_chat_template`), and the [`fallback`] layer kicks in for
//! instruct families whose upstream configs don't publish a template.
//!
//! **Public API is stable**: callers use [`wrap_chat_prompt`] or the
//! simpler [`wrap_with_vindex_template`] and inspect [`ChatWrap`].
//! Internal modules are `pub(crate)` only for tests — everything useful
//! is re-exported here.
//!
//! **Fallbacks.** Any failure path (no template found, render error,
//! unknown family) returns the raw prompt unchanged with an explanatory
//! `note` on [`ChatWrap`]. A broken template must never brick generation.

pub(crate) mod fallback;
pub(crate) mod render;
pub(crate) mod source;

/// Re-export of the multi-message renderer for diagnostic CLI flags
/// (`--system`, `--thinking`) and external callers that need richer
/// chat shapes than the single-turn `wrap_prompt_raw` exposes.
pub use render::render_chat_template_multi;

use std::path::Path;

use serde_json::Value;

use fallback::fallback_template_for;
use source::try_hf_template;

/// Outcome of applying (or not applying) a chat template to the user's
/// prompt. Returned wholesale so callers can both use the rendered string
/// and surface a note (`"rendered from chat_template.jinja"`,
/// `"no tokenizer_config.json in vindex"`, `"render failed: …"`).
#[derive(Debug, Clone)]
pub struct ChatWrap {
    /// The prompt to pass to `encode_prompt`. Equals the input prompt
    /// verbatim when [`ChatWrap::applied`] is false.
    pub prompt: String,
    /// True when a template was loaded and rendered successfully; false
    /// when we passed through (missing template, render error, etc.).
    pub applied: bool,
    /// Human-readable trail of where the template came from (or why we
    /// skipped). Surface in CLI/benchmark output so users can see
    /// whether their prompt was wrapped.
    pub note: String,
}

/// Simple form: resolves and renders the template stored in
/// `<vindex_dir>/…` against a single user turn. No hardcoded fallbacks.
/// Returns raw prompt with `applied=false` on any failure.
pub fn wrap_with_vindex_template(vindex_dir: &Path, user_prompt: &str) -> ChatWrap {
    wrap_chat_prompt(vindex_dir, None, user_prompt)
}

/// Full form: primary path is the HF template in the vindex; secondary is
/// a small hardcoded-template fallback keyed on a `model_hint` string
/// (e.g. the `cfg.model` field from the vindex —
/// `"meta-llama/Llama-2-7b-chat-hf"`, `"mistralai/Mistral-7B-Instruct-v0.3"`)
/// for families whose upstream configs don't publish the template directly.
///
/// Tries, in order:
/// 1. `<vindex_dir>/chat_template.jinja` (newer standalone-file convention —
///    Gemma 4, Qwen3, etc.).
/// 2. `<vindex_dir>/tokenizer_config.json::chat_template` (older embedded
///    convention — Gemma 2/3, Llama-3, …).
/// 3. A hardcoded template matched on `model_hint` + family heuristics,
///    when the hint clearly names an instruct/chat variant we recognise.
/// 4. Raw passthrough.
///
/// Base models ("…-hf", "…-v0.1" without `-Instruct` / `-chat`) skip step 3
/// and stay on raw prompts — wrapping them in `[INST]` markers would be
/// wrong since they weren't trained to see those tokens.
pub fn wrap_chat_prompt(
    vindex_dir: &Path,
    model_hint: Option<&str>,
    user_prompt: &str,
) -> ChatWrap {
    match try_hf_template(vindex_dir, user_prompt) {
        Ok(wrap) if wrap.applied => wrap,
        Ok(passthrough) => try_fallback(model_hint, user_prompt).unwrap_or(passthrough),
        // Render/parse error on the HF template: still try a hardcoded
        // fallback before giving up. The `Err` branch keeps the failure
        // note on `passthrough` in case the fallback also misses.
        Err(passthrough) => try_fallback(model_hint, user_prompt).unwrap_or(passthrough),
    }
}

/// Try the hardcoded instruct-family fallback (Llama-2-chat,
/// Mistral-Instruct). Returns `None` when the hint doesn't match or
/// `model_hint` was `None`.
fn try_fallback(model_hint: Option<&str>, user_prompt: &str) -> Option<ChatWrap> {
    let hint = model_hint?;
    let (family_label, template_str) = fallback_template_for(hint)?;
    let cfg = Value::Object(Default::default());
    match render::render_chat_template(template_str, &cfg, user_prompt) {
        Ok(s) => Some(ChatWrap {
            prompt: s,
            applied: true,
            note: format!("hardcoded {family_label} fallback"),
        }),
        Err(e) => {
            eprintln!("[chat] {family_label} fallback render failed: {e}");
            None
        }
    }
}

/// Render `template_str` (Jinja2) against a single user turn. Exposed so
/// callers that already have the template text in memory (remote API, test
/// fixture, in-memory generation) can reuse the render machinery without
/// touching the filesystem.
pub fn wrap_prompt_raw(
    template_str: &str,
    cfg: &Value,
    user_prompt: &str,
) -> Result<String, String> {
    render::render_chat_template(template_str, cfg, user_prompt).map_err(|e| e.to_string())
}

/// Back-compat shim — used by older callers that just want a pass-through.
/// Returns `user_prompt` unchanged.
pub fn passthrough(user_prompt: &str) -> String {
    user_prompt.to_string()
}

#[cfg(test)]
mod integration_tests {
    //! High-level tests that exercise the full `wrap_chat_prompt` pipeline
    //! across its three fallback layers. Module-local logic (JSON shape
    //! handling, Jinja edge cases, per-family patterns) is covered in the
    //! tests adjacent to [`source`], [`render`], and [`fallback`].

    use super::*;

    #[test]
    fn hf_template_wins_over_fallback_when_both_exist() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = r#"{"chat_template":"HF:{{ messages[0].content }}"}"#;
        std::fs::write(tmp.path().join("tokenizer_config.json"), cfg).unwrap();
        let w = wrap_chat_prompt(tmp.path(), Some("meta-llama/Llama-2-7b-chat-hf"), "hi");
        assert!(w.applied);
        // Primary path wins — we get the HF template, not `[INST]`.
        assert_eq!(w.prompt, "HF:hi");
    }

    #[test]
    fn full_passthrough_when_nothing_matches() {
        let tmp = tempfile::tempdir().unwrap();
        // No vindex metadata, model hint is a base model — every layer
        // declines; we expect the raw prompt back with `applied=false`.
        let w = wrap_chat_prompt(tmp.path(), Some("meta-llama/Llama-2-7b-hf"), "hi");
        assert!(!w.applied);
        assert_eq!(w.prompt, "hi");
    }

    #[test]
    fn standalone_jinja_file_beats_tokenizer_config() {
        // When both sources are present, `chat_template.jinja` wins
        // (matches the lookup order documented on `wrap_chat_prompt`).
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("chat_template.jinja"),
            "JINJA:{{ messages[0].content }}",
        )
        .unwrap();
        std::fs::write(
            tmp.path().join("tokenizer_config.json"),
            r#"{"chat_template":"TC:{{ messages[0].content }}"}"#,
        )
        .unwrap();
        let w = wrap_with_vindex_template(tmp.path(), "hi");
        assert!(w.applied);
        assert_eq!(w.prompt, "JINJA:hi");
        assert!(w.note.contains("chat_template.jinja"), "note={}", w.note);
    }
}
