-- ═══════════════════════════════════════════════════════════════════
-- Circuit Discovery: bipartite graph from OV→gate coupling edges
-- ═══════════════════════════════════════════════════════════════════
--
-- Prerequisite: ov_gate_coupling table loaded via `coupling-load`.
-- Prerequisite: ffn_gate and ffn_down tables loaded via `vector-load`.
--
-- The bipartite graph:
--   Left:  attention heads (ov_gate_coupling.head)
--   Right: FFN gate features (ov_gate_coupling.feature → ffn_gate records)
--   Edges: coupling strength
--
-- Circuit = connected component = heads sharing gate features.

-- ───────────────────────────────────────────────────────────────────
-- Query 1: Full circuit trace for a single head
-- "Head L33_H5 → which features → which entities?"
-- ───────────────────────────────────────────────────────────────────

SELECT
  ov.head,
  ov.feature,
  ov.coupling,
  ov.target_token,
  down.top_token AS down_token,
  down.c_score AS down_confidence
FROM ov_gate_coupling AS ov
LEFT JOIN ffn_down AS down
  ON down.layer = ov.feature_layer AND down.feature = ov.feature_idx
WHERE ov.head = "L33_H5"
ORDER BY ov.coupling DESC
LIMIT 20;


-- ───────────────────────────────────────────────────────────────────
-- Query 2: Head co-activation matrix
-- Which heads share FFN feature targets?
-- Heads with high shared feature counts are in the same circuit.
-- ───────────────────────────────────────────────────────────────────

SELECT
  ov1.head AS head_a,
  ov2.head AS head_b,
  count() AS shared_features,
  math::mean(ov1.coupling) AS avg_coupling_a,
  math::mean(ov2.coupling) AS avg_coupling_b
FROM ov_gate_coupling AS ov1
INNER JOIN ov_gate_coupling AS ov2
  ON ov1.feature = ov2.feature
  AND ov1.head != ov2.head
WHERE ov1.coupling > 1.0
  AND ov2.coupling > 1.0
GROUP BY head_a, head_b
HAVING shared_features > 3
ORDER BY shared_features DESC
LIMIT 50;


-- ───────────────────────────────────────────────────────────────────
-- Query 3: Feature hub analysis
-- Which FFN features are targeted by the most attention heads?
-- These are the "hub" features — key routing targets.
-- ───────────────────────────────────────────────────────────────────

SELECT
  feature,
  target_token,
  count() AS head_count,
  math::mean(coupling) AS avg_coupling,
  array::group(head) AS heads
FROM ov_gate_coupling
WHERE coupling > 1.0
GROUP BY feature, target_token
HAVING head_count > 2
ORDER BY head_count DESC
LIMIT 30;


-- ───────────────────────────────────────────────────────────────────
-- Query 4: Head → feature → entity path (three-hop)
-- Full attention-to-knowledge trace.
-- ───────────────────────────────────────────────────────────────────

SELECT
  ov.head,
  ov.feature,
  ov.coupling,
  gate.top_token AS gate_hears,
  gate.c_score AS gate_confidence,
  down.top_token AS down_produces,
  down.c_score AS down_confidence
FROM ov_gate_coupling AS ov
LEFT JOIN ffn_gate AS gate
  ON gate.layer = ov.feature_layer AND gate.feature = ov.feature_idx
LEFT JOIN ffn_down AS down
  ON down.layer = ov.feature_layer AND down.feature = ov.feature_idx
WHERE ov.head = "L33_H5"
  AND ov.coupling > 0.5
ORDER BY ov.coupling DESC
LIMIT 30;


-- ───────────────────────────────────────────────────────────────────
-- Query 5: Per-layer head role summary
-- For each layer, show what functional role each head plays
-- (based on the dominant token targets of its top features).
-- ───────────────────────────────────────────────────────────────────

SELECT
  layer,
  head,
  total_coupling,
  count() AS feature_count,
  array::group(target_token) AS target_tokens
FROM ov_gate_coupling
WHERE coupling > 1.5
GROUP BY layer, head, total_coupling
ORDER BY layer, total_coupling DESC;


-- ───────────────────────────────────────────────────────────────────
-- Query 6: Cross-layer feature sharing
-- Do the same FFN features get targeted by heads at different layers?
-- This reveals multi-layer circuits.
-- ───────────────────────────────────────────────────────────────────

SELECT
  ov1.feature,
  ov1.target_token,
  ov1.head AS head_a,
  ov1.layer AS layer_a,
  ov2.head AS head_b,
  ov2.layer AS layer_b,
  ov1.coupling AS coupling_a,
  ov2.coupling AS coupling_b
FROM ov_gate_coupling AS ov1
INNER JOIN ov_gate_coupling AS ov2
  ON ov1.feature = ov2.feature
  AND ov1.layer != ov2.layer
WHERE ov1.coupling > 1.5
  AND ov2.coupling > 1.5
ORDER BY ov1.feature, ov1.layer
LIMIT 50;
