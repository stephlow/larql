# Performance — larql-models

This crate is not compute-bound — it describes models and loads weights. Performance characteristics are about loading speed and memory.

## Benchmark Suite

Run the crate-local Criterion suite with:

```bash
cargo bench -p larql-models --bench models
```

The suite measures config detection and validation, architecture tensor-key
generation, FFN tensor classification, synthetic safetensors loading, and GGML
Q4_0/Q8_0/Q4_K/Q6_K dequantization. The synthetic loading case uses an
in-benchmark safetensors model so CI and local runs do not need external model
downloads.

Current local baseline from 2026-04-26:

| Benchmark | Median |
|-----------|--------|
| `config_detection/detect/llama` | ~590 ns |
| `config_detection/detect_validated/llama` | ~605 ns |
| `config_detection/detect/gemma4` | ~2.48 µs |
| `config_detection/detect_validated/gemma4` | ~2.58 µs |
| `config_detection/detect_validated/gpt_oss` | ~609 ns |
| `tensor_keys/gemma4_all_layer_hot_keys` | ~24.3 µs |
| `tensor_classification/is_ffn_tensor_key_set` | ~6.15 µs |
| `weight_loading/load_synthetic_safetensors_validated` | ~156 µs |

| Quant Decode | Median | Throughput |
|--------------|--------|------------|
| `quant_decode/q4_0` | ~4.43 µs | ~1.85 Gelem/s |
| `quant_decode/q8_0` | ~3.76 µs | ~2.18 Gelem/s |
| `quant_decode/q4_k` | ~2.40 µs | ~3.42 Gelem/s |
| `quant_decode/q6_k` | ~6.51 µs | ~1.26 Gelem/s |

## Weight Loading (M3 Max, NVMe SSD)

| Model | Format | Shards | Tensors | Load Time | Peak RAM | Notes |
|-------|--------|--------|---------|-----------|----------|-------|
| Gemma 3 4B | safetensors | 2 | ~270 | ~2s | ~16.6GB | f16 → f32 conversion |
| Gemma 3 4B | safetensors (mmap) | 2 | ~270 | ~0.8s | ~8.3GB | Zero-copy where possible |
| Llama 3 8B | safetensors | 4 | ~290 | ~4s | ~32GB | f16 → f32 |
| Gemma 3 4B | GGUF Q4_K | 1 | ~270 | ~3s | ~16.6GB | Dequant Q4_K → f32 |

### Where Time Goes

| Phase | % of Load | Notes |
|-------|-----------|-------|
| mmap file(s) | 5% | OS page cache makes repeated loads fast |
| Parse safetensors index | 1% | JSON header with tensor offsets |
| dtype conversion (f16→f32) | 70% | Vectorized but still touches every byte |
| Prefix stripping + key mapping | 1% | String operations on ~270 keys |
| Architecture detection | <1% | JSON parse + match |
| Config validation | <1% | O(num_layers) checks when callers opt in |
| GGUF dequantization | 80% | Block-by-block decode (when using GGUF) |

### Memory: Walk-only filtering and drop_ffn_weights

Walk-only mode skips FFN tensors during loading where possible. Safetensors keys
are filtered before dtype conversion, GGUF keys are normalized and filtered
before dequantization, and GPT-OSS packed MXFP4 experts are not expanded when
their generated expert keys are filtered. `drop_ffn_weights()` remains available
for already-loaded `ModelWeights`.

| Model | Before | After | Freed | Savings |
|-------|--------|-------|-------|---------|
| Gemma 3 4B (f32) | 16.6GB | 3.5GB | 13.1GB | 79% |
| Llama 3 8B (f32) | 32GB | 6.5GB | 25.5GB | 80% |

FFN weights (gate + up + down projections) are ~80% of total model weight. When using vindex walk mode, these are served from mmap'd index files instead.

## Architecture Detection

Detection is essentially instant — JSON parse + string match:

```
detect_from_json: <1μs (no I/O)
detect_architecture: ~50μs (read config.json + parse + detect)
```

## Config Parsing

`parse_model_config` handles ~30 fields from config.json. All fields use `.as_u64()` / `.as_f64()` with defaults and detection remains permissive. `ModelArchitecture::validate()` is an explicit O(num_layers) caller check, not part of detection by default.

Gemma 4 adds precomputed vectors in `from_config`:
- `global_layers: Vec<bool>` — O(num_layers) allocation, computed once
- `kv_sources: Vec<Option<usize>>` — O(num_layers), computed once

These avoid per-call branching in hot-path trait methods like `head_dim_for_layer()`.

## Quantization Format Performance

Encode/decode throughput (single-threaded, M3 Max):

| Format | Operation | Throughput | Notes |
|--------|-----------|------------|-------|
| f16 | encode (f32→f16) | ~2 GB/s | Bit manipulation, no SIMD |
| f16 | decode (f16→f32) | ~2 GB/s | Bit manipulation |
| bf16 | decode (bf16→f32) | ~2 GB/s | Shift + mask |
| Q4_0 | dequantize (32-block) | ~500 MB/s | Scale × nibble lookup |
| Q8_0 | dequantize (32-block) | ~800 MB/s | Scale × int8, simpler |
| MXFP4 | dequantize (32-block) | ~400 MB/s | e8m0 scale decode + 4-bit lookup |

These are data format operations only. For compute-path quantized operations (GPU matvec at 57 GB/s), see `larql-compute/PERFORMANCE.md`.
