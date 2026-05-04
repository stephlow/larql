//! Mutation executor: INSERT, DELETE, UPDATE, MERGE, REBALANCE.
//!
//! All mutations go through the `PatchedVindex` overlay — base vindex
//! files on disk are never modified.

mod delete;
mod insert;
mod merge;
mod rebalance;
mod update;

use std::collections::HashMap;

use crate::ast::{CompareOp, Condition, Value};
use crate::error::LqlError;
use crate::executor::Session;
use crate::relations::RelationClassifier;

type PatchVectorSnapshot = (Option<String>, Option<String>, Option<String>);

pub(super) fn string_condition<'a>(
    conditions: &'a [Condition],
    field: &str,
) -> Option<(&'a CompareOp, &'a str)> {
    conditions
        .iter()
        .find(|c| c.field == field)
        .and_then(|c| match &c.value {
            Value::String(s) => Some((&c.op, s.as_str())),
            _ => None,
        })
}

pub(super) fn relation_filter_matches(
    classifier: Option<&RelationClassifier>,
    relation_filter: Option<(&CompareOp, &str)>,
    layer: usize,
    feature: usize,
) -> Result<bool, LqlError> {
    let Some((op, wanted)) = relation_filter else {
        return Ok(true);
    };
    let Some(classifier) = classifier else {
        return Err(LqlError::Execution(
            "relation filters require relation labels for the active vindex; \
             target by layer/feature or omit relation"
                .into(),
        ));
    };
    let label = classifier.label_for_feature(layer, feature).unwrap_or("");
    Ok(match op {
        CompareOp::Eq => relation_eq(label, wanted),
        CompareOp::Neq => !label.is_empty() && !relation_eq(label, wanted),
        CompareOp::Like => relation_like(label, wanted),
        _ => {
            return Err(LqlError::Execution(format!(
                "unsupported relation predicate operator: {:?}",
                op
            )))
        }
    })
}

fn relation_eq(label: &str, wanted: &str) -> bool {
    let label = label.to_lowercase();
    let wanted = wanted.to_lowercase();
    !label.is_empty() && (label.contains(&wanted) || wanted.contains(&label))
}

fn relation_like(label: &str, pattern: &str) -> bool {
    let label = label.to_lowercase();
    let pattern = pattern.to_lowercase();
    if pattern == "%" {
        return !label.is_empty();
    }
    let needle = pattern.trim_matches('%');
    if needle.is_empty() {
        return !label.is_empty();
    }
    match (pattern.starts_with('%'), pattern.ends_with('%')) {
        (true, true) => label.contains(needle),
        (true, false) => label.ends_with(needle),
        (false, true) => label.starts_with(needle),
        (false, false) => label == needle,
    }
}

impl Session {
    pub(crate) fn refresh_recorded_patch_ops_for_slots(
        &mut self,
        slots: &[(usize, usize)],
    ) -> Result<(), LqlError> {
        if slots.is_empty() || self.patch_recording.is_none() {
            return Ok(());
        }

        let mut snapshots: HashMap<(usize, usize), PatchVectorSnapshot> = HashMap::new();
        {
            let (_, _, patched) = self.require_vindex()?;
            for &(layer, feature) in slots {
                let gate = patched.overrides_gate_at(layer, feature).map(encode_vector);
                let up = patched.up_override_at(layer, feature).map(encode_vector);
                let down = patched.down_override_at(layer, feature).map(encode_vector);
                snapshots.insert((layer, feature), (gate, up, down));
            }
        }

        let Some(recording) = self.patch_recording.as_mut() else {
            return Ok(());
        };
        for op in &mut recording.operations {
            if let larql_vindex::PatchOp::Insert {
                layer,
                feature,
                gate_vector_b64,
                up_vector_b64,
                down_vector_b64,
                ..
            } = op
            {
                if let Some((gate, up, down)) = snapshots.get(&(*layer, *feature)) {
                    if let Some(gate) = gate {
                        *gate_vector_b64 = Some(gate.clone());
                    }
                    if let Some(up) = up {
                        *up_vector_b64 = Some(up.clone());
                    }
                    if let Some(down) = down {
                        *down_vector_b64 = Some(down.clone());
                    }
                }
            }
        }

        Ok(())
    }
}

fn encode_vector(vec: &[f32]) -> String {
    larql_vindex::patch::core::encode_gate_vector(vec)
}
