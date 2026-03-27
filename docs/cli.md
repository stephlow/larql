# LARQL CLI Reference

```
larql <COMMAND> [OPTIONS]
```

## Extraction commands

### `larql weight-walk`

Extract edges from FFN weight matrices. Zero forward passes. Pure matrix multiplication.

```
larql weight-walk <MODEL> --output <OUTPUT> [OPTIONS]
```

| Argument/Flag | Description |
|---|---|
| `<MODEL>` | Model path or HuggingFace model ID (e.g. `google/gemma-3-4b-it`) |
| `-o, --output <OUTPUT>` | Output file (`.larql.json` or `.larql.bin`) |
| `-l, --layer <LAYER>` | Single layer to walk. Default: all layers |
| `--top-k <TOP_K>` | Top-k tokens per feature [default: 5] |
| `--min-score <MIN_SCORE>` | Minimum raw activation score for top-k selection [default: 0.02] |
| `--min-confidence <MIN_CONFIDENCE>` | Minimum normalized confidence [0-1] to keep an edge [default: 0.0] |
| `--stats <STATS>` | Write layer statistics to a separate JSON file |

**Model resolution:** Accepts a local directory path or a HuggingFace model ID. Model IDs are resolved from the HuggingFace cache at `~/.cache/huggingface/hub/` (or `$HF_HOME/hub/`).

**Resume:** If the output file already exists, completed layers are detected from edge metadata and skipped. Saves after each layer. Safe to interrupt and re-run.

**Confidence scoring:** Each edge gets a normalized confidence `c` in [0, 1], computed as `(c_in × c_out) / max(c_in × c_out)` per layer. Raw scores `c_in` (input selectivity) and `c_out` (output strength) are stored in metadata.

**Examples:**

```bash
# Full extraction
larql weight-walk google/gemma-3-4b-it -o knowledge.larql.json

# Single layer test
larql weight-walk google/gemma-3-4b-it --layer 26 -o L26.larql.json

# Filtered extraction with stats
larql weight-walk google/gemma-3-4b-it \
    -o knowledge.larql.json \
    --min-confidence 0.1 \
    --stats stats.json

# MessagePack output (smaller, faster)
larql weight-walk google/gemma-3-4b-it -o knowledge.larql.bin
```

### `larql attention-walk`

Extract routing edges from attention OV circuits. Zero forward passes.

```
larql attention-walk <MODEL> --output <OUTPUT> [OPTIONS]
```

| Argument/Flag | Description |
|---|---|
| `<MODEL>` | Model path or HuggingFace model ID |
| `-o, --output <OUTPUT>` | Output file (`.larql.json` or `.larql.bin`) |
| `-l, --layer <LAYER>` | Single layer to walk. Default: all layers |
| `--top-k <TOP_K>` | Top-k tokens per head [default: 3] |
| `--min-score <MIN_SCORE>` | Minimum score [default: 0.0] |

**How it works:** For each attention head, computes the OV circuit (`O_h @ V_h`), projects all vocab tokens through it, finds the most amplified inputs, and decodes what output tokens each produces.

**Resume:** Same as `weight-walk` — detects completed layers and skips them.

**Examples:**

```bash
larql attention-walk google/gemma-3-4b-it -o attention.larql.json
larql attention-walk google/gemma-3-4b-it --layer 12 -o attention-L12.larql.json
```

### `larql bfs`

BFS extraction from a running model endpoint.

```
larql bfs --seeds <SEEDS> --templates <TEMPLATES> --output <OUTPUT> [OPTIONS]
```

| Flag | Description |
|---|---|
| `-s, --seeds <SEEDS>` | Comma-separated seed entities |
| `-t, --templates <TEMPLATES>` | Path to templates JSON file |
| `-o, --output <OUTPUT>` | Output file (`.larql.json` or `.larql.bin`) |
| `-e, --endpoint <ENDPOINT>` | Model endpoint URL [default: `http://localhost:11434/v1`] |
| `-m, --model <MODEL>` | Model name for the endpoint |
| `--mock` | Use mock provider instead of HTTP |
| `--mock-knowledge <PATH>` | Path to mock knowledge JSON (with `--mock`) |
| `--max-depth <N>` | Maximum BFS depth [default: 3] |
| `--max-entities <N>` | Maximum entities to probe [default: 1000] |
| `--min-confidence <F>` | Minimum edge confidence [default: 0.3] |
| `--resume <PATH>` | Resume from a checkpoint file |

**Requires:** Templates JSON file defining prompt templates for each relation. See [format.md](format.md) for template format.

**Examples:**

```bash
# Against Ollama
larql bfs \
    --seeds "France,Germany,Japan" \
    --templates templates.json \
    --endpoint http://localhost:11434/v1 \
    --model gemma3:4b-it \
    -o knowledge.larql.json

# With mock provider
larql bfs \
    --seeds "France,Germany" \
    --templates templates.json \
    --mock --mock-knowledge mock.json \
    -o knowledge.larql.json
```

## Query commands

### `larql query`

Select edges from a subject, optionally filtered by relation.

```
larql query --graph <GRAPH> <SUBJECT> [RELATION]
```

```bash
larql query --graph knowledge.larql.json France
larql query --graph knowledge.larql.json France capital-of
```

### `larql describe`

Show all outgoing and incoming edges for an entity.

```
larql describe --graph <GRAPH> <ENTITY>
```

```bash
larql describe --graph knowledge.larql.json France
```

### `larql stats`

Show graph statistics: entity count, edge count, relation count, connected components, average degree, average confidence, source distribution.

```
larql stats <GRAPH>
```

```bash
larql stats knowledge.larql.json
```

### `larql validate`

Check a graph file for issues: zero-confidence edges, self-loops, empty subjects/objects.

```
larql validate <GRAPH>
```

```bash
larql validate knowledge.larql.json
```

## Templates format

Used by `larql bfs`. A JSON array of prompt templates:

```json
[
  {
    "relation": "capital-of",
    "template": "The capital of {subject} is",
    "multi_token": true,
    "stop_tokens": [".", "\n", ",", ";"]
  }
]
```

| Field | Type | Description |
|---|---|---|
| `relation` | string | Relation name for edges produced by this template |
| `template` | string | Prompt text. `{subject}` is replaced with the entity name |
| `multi_token` | bool | Chain multiple forward passes for multi-token answers |
| `reverse_template` | string? | Optional reverse probe (`{object}` placeholder) |
| `stop_tokens` | char[] | Characters that terminate multi-token chaining |

## Mock knowledge format

Used by `larql bfs --mock`. A JSON array:

```json
[
  {"prompt": "The capital of France is", "answer": "Paris", "probability": 0.89}
]
```
