//! `DELETE FROM EDGES WHERE ...` — remove features via the patch overlay.

use crate::ast::{Condition, Value};
use crate::error::LqlError;
use crate::executor::Session;

use super::{relation_filter_matches, string_condition};

impl Session {
    pub(crate) fn exec_delete(
        &mut self,
        conditions: &[Condition],
    ) -> Result<Vec<String>, LqlError> {
        let layer_filter = conditions
            .iter()
            .find(|c| c.field == "layer")
            .and_then(|c| {
                if let Value::Integer(n) = c.value {
                    Some(n as usize)
                } else {
                    None
                }
            });
        let feature_filter = conditions
            .iter()
            .find(|c| c.field == "feature")
            .and_then(|c| {
                if let Value::Integer(n) = c.value {
                    Some(n as usize)
                } else {
                    None
                }
            });
        let entity_filter = conditions
            .iter()
            .find(|c| c.field == "entity")
            .and_then(|c| {
                if let Value::String(ref s) = c.value {
                    Some(s.as_str())
                } else {
                    None
                }
            });
        let relation_filter = string_condition(conditions, "relation");

        // Collect candidates with a readonly borrow before mutating the
        // patch overlay, so relation predicates cannot be dropped silently.
        let deletes = {
            let (_path, _config, patched) = self.require_vindex()?;
            let candidates: Vec<(usize, usize)> =
                if let (Some(layer), Some(feature)) = (layer_filter, feature_filter) {
                    vec![(layer, feature)]
                } else {
                    patched
                        .base()
                        .find_features(entity_filter, None, layer_filter)
                };

            let mut matches = Vec::new();
            for (layer, feature) in candidates {
                if relation_filter_matches(
                    self.relation_classifier(),
                    relation_filter,
                    layer,
                    feature,
                )? {
                    matches.push((layer, feature));
                }
            }
            matches
        };

        if deletes.is_empty() {
            return Ok(vec!["  (no matching features found)".into()]);
        }

        {
            let (_path, _config, patched) = self.require_patched_mut()?;
            for &(layer, feature) in &deletes {
                patched.delete_feature(layer, feature);
            }
        }

        // Record to patch session
        for &(layer, feature) in &deletes {
            if let Some(ref mut recording) = self.patch_recording {
                recording.operations.push(larql_vindex::PatchOp::Delete {
                    layer,
                    feature,
                    reason: None,
                });
            }
        }

        Ok(vec![format!(
            "Deleted {} features (patch overlay)",
            deletes.len()
        )])
    }
}
