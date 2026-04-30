# larql-core Roadmap

`larql-core` owns the in-memory graph model, graph algorithms, lightweight
model-provider extraction helpers, and portable graph serialization formats.
It should stay independent of vindex storage and inference internals: higher
crates can depend on it, but this crate should remain a small, reusable graph
engine.

---

## Current state

- `Graph` is an indexed directed multigraph over `(subject, relation, object)`
  facts with confidence, source, metadata, and optional injection hints.
- Query indexes exist for outgoing edges, incoming edges, exact triples, and
  keyword search.
- Algorithms include shortest path/A*, PageRank, BFS/DFS, components, walks,
  filtering, merging, and diffing.
- Serialization supports JSON, MessagePack, packed binary, CSV, and append-only
  checkpoint logs.
- LLM extraction utilities are provider-agnostic through `ModelProvider`,
  `TemplateRegistry`, `chain_tokens`, and BFS extraction.
- Baseline verification: `cargo test -p larql-core` passes.

---

## P0 - Correctness and robustness

Status: shipped. Keep this section as a record of the hardening pass and the
regressions now covered by tests.

| Item | Area | Status |
|---|---|---|
| Store exact path edges in shortest path | `algo::shortest_path` | Done. Dijkstra/A* predecessor state now stores the selected edge, so multiedge paths and costs agree. |
| Harden packed binary decoding | `io::packed` | Done. Decoder validates flags, offsets, record bounds, string indexes, checked arithmetic, and metadata ranges. |
| Replace ad hoc CSV parsing/writing | `io::csv` | Done. CSV supports quoted commas, escaped quotes, CRLF/LF records, and multiline quoted fields. |
| Diff all edge attributes | `algo::diff` | Done. Same-triple changes now include confidence, source, metadata, and injection. |
| Clarify traversal edge semantics | `algo::traversal` | Done. `TraversalResult.edges` means edges actually traversed to newly discovered nodes. |

---

## P1 - API polish

| Item | Area | Detail |
|---|---|---|
| Deterministic ordered accessors | `core::graph` | `list_entities`, `list_relations`, `nodes`, `search`, and component enumeration often come from hash maps/sets. Add sorted variants or make current outputs deterministic where caller-facing tests and CLI output rely on order. |
| Fallible graph mutation API | `core::graph` | `add_edge` silently drops duplicate triples. Add `try_add_edge` or `insert_edge` returning `Inserted`, `Duplicate`, or `Replaced`, while keeping `add_edge` as the convenient legacy path. |
| Explicit multiedge lookup | `core::graph` | Add helpers for exact triple lookup returning `Option<&Edge>` and subject/object relation iteration that do not require callers to scan `select()` results. |
| Configurable keyword tokenizer | `core::graph` | Search lowercases and splits on whitespace/hyphen only. Add a small tokenizer abstraction or normalization options for punctuation, relation aliases, and case/diacritic handling. |
| Error types per subsystem | `core::graph`, `io`, `engine` | `GraphError::Deserialize(String)` is too broad. Split parse, format, unsupported-version, corrupt-offset, and IO context enough for CLI/server diagnostics. |

---

## P2 - Graph features

| Item | Area | Detail |
|---|---|---|
| Relation-aware subgraph extraction | `core::graph`, `algo` | Extend `subgraph` and traversal APIs with relation allow/deny lists, direction modes (`out`, `in`, `both`), confidence thresholds, and source filters. |
| Weighted traversal and path queries | `algo` | Add path APIs for `k_shortest_paths`, all simple paths with bounded depth, and relation-constrained shortest path. These map well to LQL path queries. |
| Stronger graph diff/patch model | `algo::diff` | Provide a stable diff format that can be applied to a graph, serialized, and surfaced as added/removed/updated triples with attribute-level changes. |
| Graph validation | `core::schema` | Validate edges against schema relation metadata: allowed subject/object types, reversible relation declarations, confidence ranges, required metadata keys, and unknown relation warnings. |
| Provenance utilities | `core::edge`, `algo` | Add merge and filter helpers that preserve source precedence, collect source counts per relation, and expose provenance summaries for DESCRIBE/SELECT callers. |
| Graph sampling | `algo` | Add deterministic sampling utilities for large graphs: top confidence per relation, stratified source sampling, random walk sampling with seed control. |

---

## P3 - Performance and scale

| Item | Area | Detail |
|---|---|---|
| Incremental index updates | `core::graph` | `remove_edge` and replacement flows rebuild all indexes. Add index-slot invalidation or swap-remove bookkeeping before large mutation workloads rely on this crate. |
| Memory-efficient string storage | `core::graph` | Edges and indexes clone strings heavily. Consider optional string interning for large graphs while preserving ergonomic `String` APIs. |
| Streaming readers/writers | `io` | JSON and packed paths operate on whole buffers. Add streaming load/save where format allows, especially for checkpoint compaction and large interchange files. |
| Packed format versioning plan | `io::packed` | Add explicit flags handling, forward-compatible unknown flag rejection, metadata/injection section lengths, and upgrade tests before `.larql.pak` becomes a durable format. |
| Bench regression harness | `examples`, benches | Turn README benchmark claims into repeatable `cargo bench` or example-driven measurements with fixed graph generators. |

---

## P4 - LLM extraction extensions

| Item | Area | Detail |
|---|---|---|
| Stop-token support in BFS extraction | `engine::bfs` | `PromptTemplate.stop_tokens` exists but `extract_bfs` currently passes `None` to `chain_tokens`. Use template-specific stop tokens. |
| Better multi-token mock provider | `engine::mock_provider` | The mock currently returns only the first token, which under-tests chaining behavior. Add scripted token sequences for realistic multi-pass extraction tests. |
| Provider capability metadata | `engine::provider` | Add optional capability reporting for logprobs, token IDs, timeout behavior, and max top-k so extraction code can fail clearly when a backend cannot supply confidence. |
| Extraction normalization hooks | `engine::bfs` | Add answer cleanup hooks for trimming articles, punctuation, casing, aliases, and entity rejection rules without hardcoding domain policy in BFS. |
| Async provider option | `engine` | Keep blocking APIs for simple callers, but consider an async provider trait behind a feature for server-side extraction and concurrent probing. |

---

## P0 regression coverage

- Shortest path with two `A -> B` edges where the cheaper edge is not the first
  inserted edge; returned path edge and cost must agree.
- Packed files with invalid `string_table_offset`, truncated edge records,
  out-of-range string indexes, unsupported flags, and invalid metadata ranges.
- CSV roundtrip with commas, quotes, and newlines in subject/object fields.
- Diff where confidence is unchanged but `source`, `metadata`, or `injection`
  changes.
- BFS/DFS with `max_depth = 0`, confirming no traversed edges are returned.

---

## Non-goals

- Do not add dependencies on `larql-vindex`, `larql-inference`, or CLI/server
  crates.
- Do not make this crate responsible for mmap vindex storage or tensor patching.
- Do not introduce model-family-specific extraction rules here; keep those in
  higher-level crates or external configuration.
