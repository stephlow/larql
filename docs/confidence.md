# LARQL Confidence Scoring

## Overview

Every edge extracted by `weight-walk` carries a confidence score derived from the raw logit magnitudes of the FFN feature that produced it. Confidence separates `(France, L26-F9298, Paris)` at 0.89 from `(France, L3-F2041, crawl)` at 0.002.

Extraction is always complete — all edges are stored regardless of confidence. Filtering by confidence happens at query time or as a post-processing step.

## How confidence is computed

Each FFN feature `i` at layer `L` has two projections:

**Input side (W_gate):** `embed @ W_gate.T` — projects the embedding matrix through the gate weights. The top score for feature `i` is `c_in`: how specifically this feature responds to one trigger token vs many. High `c_in` = entity-selective.

**Output side (W_down):** `embed @ W_down` — projects the embedding matrix through the down weights. The top score for feature `i` is `c_out`: how strongly this feature pushes toward one answer token. High `c_out` = strong writer.

**Raw product:** `c_in × c_out` — a feature that fires specifically for "France" AND writes strongly toward "Paris" has a high raw product. A feature that fires vaguely AND writes weakly is noise.

**Per-layer normalization:** After all features in a layer are walked:

```
c = (c_in × c_out) / max(c_in × c_out across this layer)
```

This gives confidence in [0, 1] normalized within each layer.

## Why per-layer normalization

Different layers serve different functions in the transformer:

| Layer range | Role | Signal type |
|---|---|---|
| L0–L14 | Dark accumulation | Structural, low factual confidence |
| L14–L25 | Relation differentiation | Mixed, relations emerging |
| L26 | Fact explosion | Highest factual confidence |
| L27–L33 | Refinement | Copy, format, consolidation |

A confidence of 0.8 at L26 means "strong factual edge." A confidence of 0.8 at L3 means "strong structural edge." Both are valid but serve different purposes. Per-layer normalization keeps scores comparable within their function. The `layer` field lets you weight across layers at query time.

## Edge schema

```json
{
  "s": "France",
  "r": "L26-F9298",
  "o": "Paris",
  "c": 0.89,
  "src": "parametric",
  "meta": {
    "layer": 26,
    "feature": 9298,
    "c_in": 8.7,
    "c_out": 12.4
  }
}
```

| Field | Description |
|---|---|
| `c` | Normalized confidence [0, 1] — per-layer scaled |
| `c_in` | Raw input selectivity (gate projection magnitude) |
| `c_out` | Raw output strength (down projection magnitude) |
| `layer` | Source transformer layer |
| `feature` | Source FFN feature index |

## Filtering at query time

Extraction stores everything. Filtering happens when you load or query:

```bash
# Extract everything (always complete)
larql weight-walk google/gemma-3-4b-it -o knowledge.larql.json

# Query with implicit threshold in your application
# The graph engine supports filtering by confidence at load time
```

From the library:

```rust
// Load and filter
let graph = load("knowledge.larql.json")?;
let high_conf: Vec<&Edge> = graph.edges()
    .iter()
    .filter(|e| e.confidence >= 0.1)
    .collect();
```

## Layer statistics

The `--stats` flag writes per-layer statistics for validation:

```bash
larql weight-walk google/gemma-3-4b-it \
    -o knowledge.larql.json \
    --stats stats.json
```

Stats file contains per-layer:

| Field | Description |
|---|---|
| `mean_confidence` | Average normalized confidence across all edges in this layer |
| `max_confidence` | Highest confidence edge in this layer |
| `min_confidence` | Lowest confidence edge in this layer |
| `mean_c_in` | Average raw input selectivity |
| `mean_c_out` | Average raw output strength |
| `edges_found` | Total edges extracted from this layer |
| `features_scanned` | Number of FFN features walked |

**Validation:** L26 (or the primary factual layer) should show the highest `mean_confidence`. If it doesn't, the extraction or normalization has an issue.

## Expected scale

For Gemma 3-4B-IT (34 layers, 10240 features/layer):

| Metric | Approximate value |
|---|---|
| Total edges | ~8M |
| Edges at c >= 0.1 | ~500K–1M |
| Edges at c >= 0.5 | ~30K–50K |
| JSON file (complete) | ~1.5 GB |
| JSON file (c >= 0.1) | ~200 MB |
| MessagePack (complete) | ~700 MB |
