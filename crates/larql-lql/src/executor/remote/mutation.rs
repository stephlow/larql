//! Write-side remote forwarders (INSERT, DELETE, UPDATE) plus
//! client-side local patch management (APPLY / SHOW / REMOVE PATCH).

use crate::ast::{Assignment, Condition, Value};
use crate::error::LqlError;
use crate::executor::{Backend, Session};

use super::{require_layer_feature, ENDPOINT_INSERT, ENDPOINT_PATCHES_APPLY};

/// Default `confidence` for INSERT/UPDATE when the user doesn't
/// pass one. 0.9 lands well above the retrieval floor without
/// dominating template-matched siblings.
const REMOTE_DEFAULT_CONFIDENCE: f32 = 0.9;

impl Session {
    pub(crate) fn remote_insert(
        &self,
        entity: &str,
        relation: &str,
        target: &str,
        layer: Option<u32>,
        confidence: Option<f32>,
    ) -> Result<Vec<String>, LqlError> {
        let request = serde_json::json!({
            "entity": entity,
            "relation": relation,
            "target": target,
            "layer": layer,
            "confidence": confidence.unwrap_or(REMOTE_DEFAULT_CONFIDENCE),
        });

        let result = self.remote_post_json(ENDPOINT_INSERT, &request, true)?;

        let inserted = result["inserted"].as_u64().unwrap_or(0);
        let mode = result["mode"].as_str().unwrap_or("unknown");
        let ms = result["latency_ms"].as_f64().unwrap_or(0.0);

        Ok(vec![
            format!("Inserted: {entity} —[{relation}]→ {target} ({inserted} layers, mode: {mode})"),
            format!("{ms:.0}ms (remote)"),
        ])
    }

    pub(crate) fn remote_delete(
        &self,
        conditions: &[Condition],
    ) -> Result<Vec<String>, LqlError> {
        let (layer, feature) = require_layer_feature(conditions, "DELETE")?;

        let ops = vec![larql_vindex::PatchOp::Delete {
            layer,
            feature,
            reason: Some("remote DELETE".into()),
        }];

        let patch = larql_vindex::VindexPatch {
            version: 1,
            base_model: String::new(),
            base_checksum: None,
            created_at: String::new(),
            description: Some(format!("DELETE L{layer} F{feature}")),
            author: None,
            tags: vec![],
            operations: ops,
        };

        let _ = self.remote_post_json(
            ENDPOINT_PATCHES_APPLY,
            &serde_json::json!({"patch": patch}),
            false,
        )?;

        Ok(vec![format!(
            "Deleted: L{layer} F{feature} → remote server"
        )])
    }

    pub(crate) fn remote_update(
        &self,
        set: &[Assignment],
        conditions: &[Condition],
    ) -> Result<Vec<String>, LqlError> {
        let (layer, feature) = require_layer_feature(conditions, "UPDATE")?;

        let target = set
            .iter()
            .find(|a| a.field == "target" || a.field == "top_token")
            .and_then(|a| match &a.value {
                Value::String(s) => Some(s.clone()),
                _ => None,
            });
        let confidence = set
            .iter()
            .find(|a| a.field == "confidence" || a.field == "c_score")
            .and_then(|a| match &a.value {
                Value::Number(n) => Some(*n as f32),
                Value::Integer(n) => Some(*n as f32),
                _ => None,
            });

        let down_meta = target
            .as_ref()
            .map(|t| larql_vindex::patch::core::PatchDownMeta {
                top_token: t.clone(),
                top_token_id: 0,
                c_score: confidence.unwrap_or(REMOTE_DEFAULT_CONFIDENCE),
            });

        let op = larql_vindex::PatchOp::Update {
            layer,
            feature,
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta,
        };

        let patch = larql_vindex::VindexPatch {
            version: 1,
            base_model: String::new(),
            base_checksum: None,
            created_at: String::new(),
            description: Some(format!("UPDATE L{layer} F{feature}")),
            author: None,
            tags: vec![],
            operations: vec![op],
        };

        let _ = self.remote_post_json(
            ENDPOINT_PATCHES_APPLY,
            &serde_json::json!({"patch": patch}),
            false,
        )?;

        let desc = target
            .as_deref()
            .map(|t| format!(" target={t}"))
            .unwrap_or_default();
        Ok(vec![format!(
            "Updated: L{layer} F{feature}{desc} → remote server"
        )])
    }

    // ── Local patch management (client-side overlay) ─────────────

    pub(crate) fn remote_apply_local_patch(
        &mut self,
        path: &str,
    ) -> Result<Vec<String>, LqlError> {
        let patch_path = std::path::PathBuf::from(path);
        if !patch_path.exists() {
            return Err(LqlError::Execution(format!("patch not found: {path}")));
        }

        let patch = larql_vindex::VindexPatch::load(&patch_path)
            .map_err(|e| LqlError::exec("failed to load patch", e))?;

        let (ins, upd, del) = patch.counts();
        let total = patch.len();

        match &mut self.backend {
            Backend::Remote { local_patches, .. } => {
                local_patches.push(patch);
                Ok(vec![format!(
                    "Applied locally: {path} ({total} ops: {ins} ins, {upd} upd, {del} del)\n\
                     Patch stays client-side — server never sees it."
                )])
            }
            _ => Err(LqlError::Execution(
                "not connected to a remote server".into(),
            )),
        }
    }

    pub(crate) fn remote_show_patches(&self) -> Result<Vec<String>, LqlError> {
        let local_patches = match &self.backend {
            Backend::Remote { local_patches, .. } => local_patches,
            _ => {
                return Err(LqlError::Execution(
                    "not connected to a remote server".into(),
                ))
            }
        };

        let mut out = Vec::new();
        if local_patches.is_empty() {
            out.push("  (no local patches)".into());
        } else {
            out.push("Local patches (client-side only):".into());
            for (i, patch) in local_patches.iter().enumerate() {
                let (ins, upd, del) = patch.counts();
                let name = patch.description.as_deref().unwrap_or("(unnamed)");
                out.push(format!(
                    "  {}. {:<40} {} ops ({} ins, {} upd, {} del)",
                    i + 1,
                    name,
                    patch.len(),
                    ins,
                    upd,
                    del,
                ));
            }
        }
        Ok(out)
    }

    pub(crate) fn remote_remove_local_patch(
        &mut self,
        name: &str,
    ) -> Result<Vec<String>, LqlError> {
        let local_patches = match &mut self.backend {
            Backend::Remote { local_patches, .. } => local_patches,
            _ => {
                return Err(LqlError::Execution(
                    "not connected to a remote server".into(),
                ))
            }
        };

        let pos = local_patches
            .iter()
            .position(|p| p.description.as_deref().unwrap_or("unnamed") == name);

        match pos {
            Some(i) => {
                local_patches.remove(i);
                Ok(vec![format!("Removed local patch: {name}")])
            }
            None => Err(LqlError::Execution(format!(
                "local patch not found: {name}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_default_confidence_is_in_band() {
        // Pinned: must land between PROB_FLOOR and PROB_CEILING from
        // `executor::tuning` so newly-inserted facts come out of the
        // gate already inside the per-INSERT balance band.
        use crate::executor::tuning::{PROB_CEILING, PROB_FLOOR};
        assert!((REMOTE_DEFAULT_CONFIDENCE as f64) >= PROB_FLOOR);
        assert!((REMOTE_DEFAULT_CONFIDENCE as f64) <= PROB_CEILING);
    }

    #[test]
    fn remote_default_confidence_is_finite() {
        assert!(REMOTE_DEFAULT_CONFIDENCE.is_finite());
        assert!(REMOTE_DEFAULT_CONFIDENCE > 0.0);
    }
}
