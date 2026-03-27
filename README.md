# chuk-larql-rs

Knowledge graphs extracted from neural network weights. One library, one binary, one graph format.

LARQL extracts knowledge from language models in two ways:

- **Weight walking** — reads FFN weight matrices directly from safetensors files. Zero forward passes. Pure matrix multiplication. Each edge carries a confidence score derived from input selectivity and output strength.
- **BFS probing** — sends structured prompts to a running model endpoint, chains next-token predictions into multi-token answers, and assembles them into edges.

Both methods produce the same graph format. The graph can be queried, walked, merged, and serialized to JSON or MessagePack.

## Quick start

```bash
# Build
make release

# Extract from weights — accepts a HuggingFace model ID
larql weight-walk google/gemma-3-4b-it -o knowledge.larql.json

# Single layer test
larql weight-walk google/gemma-3-4b-it --layer 26 -o L26.larql.json

# With layer statistics
larql weight-walk google/gemma-3-4b-it -o knowledge.larql.json --stats stats.json

# Query
larql query --graph knowledge.larql.json France
larql describe --graph knowledge.larql.json Mozart
larql stats knowledge.larql.json
```

## Documentation

| Doc | Description |
|---|---|
| [docs/cli.md](docs/cli.md) | Full CLI reference — all commands, flags, examples |
| [docs/format.md](docs/format.md) | Graph file format specification — JSON and MessagePack |
| [docs/confidence.md](docs/confidence.md) | Confidence scoring — how c, c_in, c_out are computed |

## Two extraction methods

### Weight walking (recommended)

Reads safetensors files directly. No model server needed. Uses Apple Accelerate BLAS on macOS. Extracts edges from every FFN feature in every layer with per-layer normalized confidence scores.

```bash
larql weight-walk google/gemma-3-4b-it -o knowledge.larql.json
larql weight-walk ./path/to/model -o knowledge.larql.bin
```

- Accepts HuggingFace model IDs or local paths
- Auto-resumes on re-run (saves after each layer, detects completed layers)
- Per-feature progress bar within each layer
- Handles f32, f16, and bf16 weight formats
- Confidence scored: `c = (c_in × c_out) / max` per layer. See [docs/confidence.md](docs/confidence.md)

### Attention walking

Extracts routing edges from attention OV circuits. Each head is a routing rule: when it fires on input X, it copies output Y.

```bash
larql attention-walk google/gemma-3-4b-it -o attention.larql.json
```

### BFS probing

Sends prompts to an OpenAI-compatible endpoint (Ollama, vLLM, llama.cpp, LM Studio):

```bash
larql bfs \
    --seeds "France,Germany,Japan" \
    --templates examples/templates.json \
    --endpoint http://localhost:11434/v1 \
    --model gemma3:4b-it \
    -o knowledge.larql.json
```

## Querying

```bash
larql query --graph knowledge.larql.json France capital-of
larql describe --graph knowledge.larql.json France
larql stats knowledge.larql.json
larql validate knowledge.larql.json
```

See [docs/cli.md](docs/cli.md) for full reference.

## Workspace structure

```
chuk-larql-rs/
├── crates/
│   ├── larql-core/       Library — graph engine, walker, providers, I/O
│   ├── larql-cli/        Binary — CLI over larql-core
│   └── larql-python/     PyO3 binding — native Python extension
├── docs/
│   ├── cli.md            CLI reference
│   ├── format.md         Graph format specification
│   └── confidence.md     Confidence scoring spec
├── examples/
│   ├── templates.json
│   ├── mock_knowledge.json
│   └── gemma_4b_knowledge.json
├── Makefile
└── README.md
```

## Python integration

The `chuk-larql` Python package uses this Rust engine as its native backend. Core types (Graph, Edge, Node) and I/O are Rust. Python wraps them transparently.

```python
from chuk_larql import Graph, Edge

g = Graph()
g.add_edge(Edge("France", "capital-of", "Paris", 0.89, "parametric"))

from _larql_core import weight_walk
g = weight_walk("google/gemma-3-4b-it")
```

See [docs/cli.md](docs/cli.md) for the full Python/Rust API mapping.

## Building

```bash
make build          # debug build
make release        # optimized build
make test           # run all tests
make check          # check all crates (including Python binding)
make python-build   # build Python extension (requires virtualenv)
```

### Feature flags

| Feature | Default | Description |
|---|---|---|
| `http` | yes | HTTP model provider (adds reqwest) |
| `msgpack` | yes | MessagePack serialization (adds rmp-serde) |
| `walker` | yes | Weight walking from safetensors (adds safetensors, ndarray, tokenizers, blas) |

## Status

### What's working

- **Weight walker** — 8.2M edges from Gemma 3-4B in 40 minutes. BLAS-accelerated. Per-layer confidence scoring. Resume. Progress reporting.
- **Attention walker** — OV circuit extraction from attention heads.
- **Core graph engine** — insert, remove, deduplicate, select, reverse select, describe, walk, search, subgraph, count, node, stats, components.
- **BFS extraction** — template-based probing, multi-token chaining, checkpoint callbacks.
- **Serialization** — JSON and MessagePack with format auto-detection.
- **CLI** — seven commands: `weight-walk`, `attention-walk`, `bfs`, `stats`, `query`, `describe`, `validate`.
- **PyO3 binding** — full Python API parity. `chuk-larql` uses Rust as native backend.
- **Test suite** — 102 tests across 10 test files, all passing.

### What's next

- CI / GitHub Actions
- `larql filter` command (post-extraction confidence filtering)
- Packed binary edge format for runtime graphs
- Crate publishing

## License

Apache-2.0
