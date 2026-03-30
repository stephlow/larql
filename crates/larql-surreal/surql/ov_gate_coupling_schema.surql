-- Schema for OV→gate coupling edges.
-- Links attention heads to the FFN gate features they activate.
-- One record per (head, feature) pair above the coupling threshold.

DEFINE TABLE IF NOT EXISTS ov_gate_coupling SCHEMAFULL;

-- Attention head identity
DEFINE FIELD IF NOT EXISTS head        ON ov_gate_coupling TYPE string;  -- "L25_H6"
DEFINE FIELD IF NOT EXISTS layer       ON ov_gate_coupling TYPE int;
DEFINE FIELD IF NOT EXISTS head_idx    ON ov_gate_coupling TYPE int;

-- Target FFN feature
DEFINE FIELD IF NOT EXISTS feature     ON ov_gate_coupling TYPE string;  -- "L25_F6841"
DEFINE FIELD IF NOT EXISTS feature_idx ON ov_gate_coupling TYPE int;
DEFINE FIELD IF NOT EXISTS feature_layer ON ov_gate_coupling TYPE int;

-- Coupling metadata
DEFINE FIELD IF NOT EXISTS target_token ON ov_gate_coupling TYPE string; -- what the gate feature responds to
DEFINE FIELD IF NOT EXISTS coupling    ON ov_gate_coupling TYPE float;   -- coupling strength (L2 norm)
DEFINE FIELD IF NOT EXISTS total_coupling ON ov_gate_coupling TYPE float; -- head's total coupling across all features

-- Indexes for circuit discovery queries
DEFINE INDEX IF NOT EXISTS idx_head     ON ov_gate_coupling FIELDS head;
DEFINE INDEX IF NOT EXISTS idx_feature  ON ov_gate_coupling FIELDS feature;
DEFINE INDEX IF NOT EXISTS idx_layer    ON ov_gate_coupling FIELDS layer;
DEFINE INDEX IF NOT EXISTS idx_coupling ON ov_gate_coupling FIELDS coupling;
DEFINE INDEX IF NOT EXISTS idx_head_feature ON ov_gate_coupling FIELDS head, feature UNIQUE;
