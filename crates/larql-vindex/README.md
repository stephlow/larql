# larql-vindex

The queryable model format. Decompile, browse, edit, and recompile neural networks.

## What is a Vindex?

A vindex (vector index) is a directory containing a transformer model's weights reorganised for queryability. The model IS the database — each weight matrix is stored once in its optimal format.

```rust
use larql_vindex::*;

// Load
let index = VectorIndex::load_vindex(&path, &mut SilentLoadCallbacks)?;

// Query — which features fire for "France"?
let hits = index.gate_knn(layer, &query, 10);  // 0.008ms/layer

// Walk — multi-layer feature scan
let trace = index.walk(&query, &layers, 10);

// Mutate — insert a fact
index.set_gate_vector(layer, feature, &gate_vec);
index.set_feature_meta(layer, feature, meta);
index.save_down_meta(&path)?;

// Patch — lightweight knowledge diff
let patch = VindexPatch::load("medical.vlp")?;
let mut patched = PatchedVindex::new(index);
patched.apply_patch(patch);
```

## Features

- **Extract** from safetensors models (any gated FFN architecture)
- **Gate KNN** via BLAS matmul — 0.008ms per layer
- **Walk** across all layers with down-meta annotation
- **Mutate** — insert, delete, update features with disk persistence
- **Patches** — stackable, reversible knowledge diffs (.vlp files)
- **Split weight files** — gate, up, down, attn, norms, lm_head (no duplication)
- **Binary down_meta** — 5x smaller than JSONL
- **f16 storage** — halves file sizes with negligible accuracy loss
- **MoE support** — Mixtral, DeepSeek (experts as contiguous features)
- **Layer bands** — per-family boundaries (Gemma, Llama, Qwen, etc.)
- **Checksums** — SHA256 integrity verification
- **Provenance** — model source, timestamp, version tracking

## Crate Structure

```
larql-vindex/src/
│
├── lib.rs                      Crate root + re-exports
├── error.rs                    VindexError
├── describe.rs                 DescribeEdge, LabelSource
│
├── config/                     Configuration types
│   ├── types.rs                VindexConfig, ExtractLevel, LayerBands, MoeConfig
│   └── dtype.rs                StorageDtype (f32/f16), conversion utilities
│
├── index/                      In-memory KNN engine
│   ├── core.rs                 VectorIndex, FeatureMeta, gate_knn, walk
│   └── mutate.rs               set/delete features, find_free_feature, save to disk
│
├── format/                     File I/O
│   ├── load.rs                 load_vindex, load_embeddings, load_tokenizer, load_config
│   ├── loader.rs               safetensors → ModelWeights (model loading)
│   ├── down_meta.rs            Binary down_meta read/write
│   ├── weights.rs              Split weight files (attn, up, down, norms, lm_head)
│   └── checksums.rs            SHA256 computation + verification
│
├── extract/                    Build pipeline (model → vindex)
│   ├── callbacks.rs            IndexBuildCallbacks trait
│   ├── build.rs                build_vindex (full extraction + clustering)
│   └── build_from_vectors.rs   Build from pre-extracted NDJSON vectors
│
├── patch/                      Patch system
│   └── core.rs                 VindexPatch, PatchOp, PatchedVindex, base64 gate encoding
│
└── clustering/                 Relation discovery
    ├── kmeans.rs               k-means clustering
    ├── labeling.rs             Pattern detection, TF-IDF labels
    ├── categories.rs           Entity category word lists
    ├── pair_matching.rs        Wikidata/WordNet output matching
    └── probe.rs                Probe label loading
```

## Supported Architectures

| Family | Models | FFN Type |
|--------|--------|----------|
| Gemma | Gemma 2/3 (2B-27B) | Gated (GeGLU) |
| Llama | Llama 2/3 (7B-405B) | Gated (SiLU) |
| Mistral | Mistral 7B | Gated (SiLU) |
| Mixtral | Mixtral 8x7B | MoE (8 experts) |
| Qwen | Qwen 2/2.5 | Gated (SiLU) |
| Phi | Phi 2/3 | Gated |
| DeepSeek | DeepSeek V2/V3 | MoE (shared + routed) |
| GPT-2 | GPT-2 | Dense (GELU) |

## File Layout

```
model.vindex/
├── gate_vectors.bin        W_gate per layer (KNN index)
├── embeddings.bin          W_embed matrix
├── down_meta.bin           Per-feature output metadata (binary)
├── down_meta.jsonl         Per-feature output metadata (JSONL, compat)
├── attn_weights.bin        Q, K, V, O per layer
├── up_weights.bin          W_up per layer
├── down_weights.bin        W_down per layer
├── norms.bin               LayerNorm parameters
├── lm_head.bin             Output projection
├── index.json              Config, layer bands, provenance, checksums
├── tokenizer.json          Tokenizer
├── relation_clusters.json  Discovered relation types
├── feature_labels.json     Probe-confirmed labels
└── weight_manifest.json    Weight file → offset mapping
```

## Extract Levels

| Level | Size (f16) | Enables |
|-------|-----------|---------|
| Browse | ~3 GB | DESCRIBE, WALK, SELECT |
| Inference | ~6 GB | + INFER |
| All | ~8.5 GB | + COMPILE |

## Testing

```bash
cargo test -p larql-vindex                                  # 102 tests
cargo run -p larql-vindex --example vindex_demo              # Feature showcase
cargo run -p larql-vindex --example vindex_bench --release    # Benchmarks
```

## Benchmarks

| Operation | Latency |
|---|---|
| Gate KNN (per layer) | 0.008ms |
| Walk (8 layers) | 0.088ms |
| Feature lookup | <1ns |
| Save gates (8 MB) | 1.1ms |
| Load vindex | 8.1ms |
| Mutate (meta + gate) | 617ns |
| Checksum (SHA256) | 23ms |
| MoE 8x scaling | 6.6x (sub-linear) |

## License

Apache-2.0
