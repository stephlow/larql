# LARQL Graph Format Specification

Version: 0.1.0

The `.larql.json` format is the interchange format between Rust and Python. MessagePack (`.larql.bin`) is a binary-identical encoding of the same structure.

## File structure

```json
{
  "larql_version": "0.1.0",
  "metadata": { ... },
  "schema": { ... },
  "edges": [ ... ]
}
```

### `larql_version`

String. Currently `"0.1.0"`.

### `metadata`

Free-form object. Stores extraction provenance.

```json
{
  "model": "google/gemma-3-4b-it",
  "method": "weight-walk",
  "extraction_date": "2026-03-27"
}
```

### `schema`

Optional. Defines relation metadata and type inference rules.

```json
{
  "relations": [
    {
      "name": "capital-of",
      "subject_types": ["country"],
      "object_types": ["city"],
      "reversible": true,
      "reverse_name": null
    }
  ],
  "type_rules": [
    {
      "node_type": "country",
      "outgoing": ["capital-of", "language-of", "currency"],
      "incoming": []
    }
  ]
}
```

**`relations`** — array of relation metadata. All fields optional except `name`.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | string | required | Relation identifier |
| `subject_types` | string[] | `[]` | Expected subject node types |
| `object_types` | string[] | `[]` | Expected object node types |
| `reversible` | bool | `true` | Whether the relation has a meaningful reverse |
| `reverse_name` | string? | `null` | Name of the reverse relation |

**`type_rules`** — array of inference rules. If a node has any of the listed outgoing or incoming relations, it's assigned the given type. First match wins. If no rule matches, the node type is `"unknown"`.

### `edges`

Array of compact edge objects.

#### Compact edge format

```json
{"s": "France", "r": "capital-of", "o": "Paris", "c": 0.89, "src": "parametric"}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `s` | string | yes | — | Subject (trigger entity) |
| `r` | string | yes | — | Relation |
| `o` | string | yes | — | Object (answer entity) |
| `c` | f64 | no | `1.0` | Confidence score [0, 1] |
| `src` | string? | no | omitted | Source type. Omitted when `"unknown"` |
| `meta` | object? | no | omitted | Arbitrary metadata. Omitted when empty |
| `inj` | [int, f64]? | no | omitted | Injection point. Omitted when absent |

**Source types:** `"parametric"`, `"document"`, `"installed"`, `"wikidata"`, `"manual"`, `"unknown"`

#### Weight walk edges

Edges from `weight-walk` include confidence scoring metadata:

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

| Meta field | Type | Description |
|---|---|---|
| `layer` | int | Transformer layer index (0-based) |
| `feature` | int | FFN feature index within the layer |
| `c_in` | f64 | Raw input selectivity — W_gate projection magnitude |
| `c_out` | f64 | Raw output strength — W_down projection magnitude |

**Confidence computation:**
- `c_in × c_out` = raw product (how strongly this feature maps input → output)
- `c` = raw product / max(raw product) across the layer, giving [0, 1] per-layer normalized confidence

#### Attention walk edges

Edges from `attention-walk` include OV circuit metadata:

```json
{
  "s": "machine",
  "r": "L12-H3",
  "o": "learning",
  "c": 0.45,
  "src": "parametric",
  "meta": {
    "layer": 12,
    "head": 3,
    "circuit": "OV"
  }
}
```

| Meta field | Type | Description |
|---|---|---|
| `layer` | int | Transformer layer index |
| `head` | int | Attention head index |
| `circuit` | string | Circuit type (currently always `"OV"`) |

## Serialization formats

| Extension | Format | Notes |
|---|---|---|
| `.larql.json`, `.json` | JSON (pretty-printed) | Human-readable. Python interop. |
| `.larql.bin`, `.bin`, `.msgpack` | MessagePack | Binary. ~53% smaller. ~10% faster I/O. |

Both formats encode the same structure. Format is auto-detected from the file extension on load.

## Identity and equality

Edge identity is based on the `(s, r, o)` triple only. Confidence, source, and metadata do not affect equality or deduplication. Adding an edge with the same triple as an existing edge is silently skipped.

## Nodes

Nodes are not stored in the file. They are derived from edges at load time. Each unique string appearing as a subject or object becomes a node. Node types are inferred from schema type rules.
