# Roadmap — larql-cli

## Current state

Primary verbs: `run`, `chat`, `pull`, `list`, `show`, `rm`, `link`, `serve`, `bench`.
490 tests passing across the workspace. Legacy research commands gated under
`larql dev <subcmd>` for backwards-compat. Dual cache (HuggingFace hub +
`~/.cache/larql/local/`) with shorthand resolution (`larql run gemma3-4b-it-vindex`).

---

## P0: Generation UX (blocks demo)

### `larql parity` — backend parity diagnostic
**Status**: Designed 2026-04-27, not started.
**Files**: new `src/commands/diagnostics/parity.rs` and a `Subcommand::Parity`
  variant in `src/main.rs`. Trace-point infrastructure lives in
  `larql-inference/src/diagnostics/` (new module).

Cross-backend numerical diff tool. Catches "I refactored quantization /
activation / norm and silently broke something" regressions that latency
benches and synthetic-weight unit tests miss. Today's specific motivation:
the CPU MoE path on Gemma 4 26B-A4B produces incoherent text while Metal
produces "Paris." (See `larql-server/ROADMAP.md` P0 F0.)

**Shape:**
```bash
larql parity <vindex> --component <C> [--prompt "..."] [--seed N]
                                       [--layer N] [--expert M]
                                       [--backends cpu,metal,hf]
                                       [--tolerance 1e-3] [--verbose]
```

**Components (in order of build priority):**
| Component | What it diffs | When it lands |
|---|---|---|
| `moe-expert` | Single expert forward (gate matmul, up matmul, gelu_tanh, down matmul) | v1 |
| `moe-block` | Full MoE block, one layer (router → top-K → K experts → weighted sum → post-norm) | v1 — finds today's bug |
| `attention` | Single attention block (Q/K/V proj, RoPE, softmax, O proj) | v2 |
| `dense-ffn` | Dense FFN layer (gate, up, act, down) | v2 |
| `layer` | Full transformer layer end-to-end | v2 |
| `forward` | Full forward pass; per-layer divergence trace | v3 |

**Backends (in order of build priority):**
| Backend | Source of truth | When |
|---|---|---|
| `reference` | Slow naive triple-loop CPU; f64 accumulators; no BLAS, no padding tricks. The bedrock other backends compare against. | v1 |
| `cpu` | Production `cpu_moe_forward` / `predict_q4k` paths | v1 |
| `metal` | `gpu_moe_dispatch` / Metal `predict_q4k_metal`. Requires exposing public entry points or adding `gpu_dispatch_one_<component>` shims. | v2 |
| `hf` | HuggingFace `transformers` reference loaded from a sidecar dump. Python script (`tools/hf_capture.py`) runs `model.forward` with intermediate captures, writes `.safetensors`; Rust harness loads and compares. | v3 |

**Architecture:**
- Trace points at well-defined checkpoints (`post_pre_norm`, `post_router_softmax`,
  `post_gate_matmul`, `post_activation`, `post_down_matmul`, `post_combine`,
  `post_post_norm`). Each checkpoint emits `(name: &str, &[f32])` to a
  registered `TraceSink`.
- One sink per backend. The diagnostic runs the same input through each
  backend with its sink attached, then walks the merged traces and prints
  the **first divergence** beyond `--tolerance` along with magnitude, index,
  and surrounding context.
- Trace points are zero-overhead in release builds (gated on a `diagnostics`
  feature flag in `larql-inference`). When the feature is off, sinks are no-ops
  and the compiler optimises them away.

**v1 has already been validated as a one-shot prototype** (deleted after
proving the approach): a slow naive reference matches `cpu_moe_forward`
bit-identically (max diff 4.3e-6) on layer 0, expert 0 of the 26B-A4B vindex
— so today's bug is **not** in per-expert compute. It must be in routing or
expert combination, which v1's `moe-block` component will catch.

**Testing strategy:**
- `cargo test -p larql-cli --test test_parity_smoke`: synthetic 4-expert
  MoE built from known weights; reference and CPU must agree to fp32 noise.
- `cargo run -p larql-cli -- parity <real-vindex> --component moe-block`
  in CI on a representative MoE vindex once we have one in the test fleet.

**Open scoping decisions:**
- Output format: human-readable table by default, `--json` for CI consumption?
- Should `larql parity` accept `--from-recording <path>` to replay a previously
  captured trace (avoids loading the model twice for repeated diffs)? Probably
  yes for v3 once HF sidecar exists.
- Tolerance per-component: `forward` after 30 layers will accumulate to
  ~1e-2 even for "correct" backends; need component-specific defaults.

### Chat template — CLI side
**Status**: Not started  
**Files**: `src/commands/run_cmd.rs`  
Instruction-tuned models need the prompt wrapped in the model's turn format before
tokenisation. `larql chat` should always apply the template; `larql run` exposes
`--no-chat-template` to skip it on base models. The inference-side Jinja parsing
is tracked in `larql-inference/ROADMAP.md`; this item is only the flag wiring and
auto-detect logic in `run_cmd.rs`.

### Streaming display
**Status**: Not started  
**Files**: `src/commands/run_cmd.rs`  
Once `generate.rs` emits an `on_token` callback (see larql-inference P0), the CLI
side is: print each token to stdout and `flush()` immediately. One-liner in the
callback closure; without it the terminal is silent for the full `--max-tokens` run.

---

## P1: Usability

### Sampling flags
**Status**: Not started  
**Files**: `src/commands/run_cmd.rs`  
Add `--temperature F`, `--top-p F`, `--top-k N`, `--repetition-penalty F` to
the `run` / `chat` subcommands. Values are threaded through to `generate.rs`
logit post-processing (tracked in larql-inference P0).

### `--max-context N`
**Status**: Not started  
**Files**: `src/commands/run_cmd.rs`  
Expose `--max-context N` (default 8192). Thread through to `KVCache::new_per_layer`
in `generate.rs`. `larql chat` should also respect this for multi-turn state.

### Auto-extract on `larql run hf://`
**Status**: Not started  
**Files**: `src/cache/resolve_model.rs` (or equivalent resolver)  
If the shorthand looks like `hf://owner/name` and no cached vindex is found, offer
to run `larql extract` inline (confirm prompt or `--yes`). Collapses the three-step
`extract → link → run` flow to one command.

### OpenAI-compatible surface — CLI side
**Status**: Not started  
**Files**: `src/commands/run_cmd.rs`  
After the server-side `/v1/chat/completions` endpoint lands (larql-server P0),
expose `larql run --openai-url URL` to send prompts to any OpenAI-compatible
endpoint (including the local `larql serve` instance). Useful for round-trip
testing without a client library.

---

## P2: MoE / expert routing

### `--experts` flag
**Status**: Not started  
**Files**: `src/commands/run_cmd.rs`, `src/commands/serve_cmd.rs`  
`larql run --experts '0-31=http://host1,32-63=http://host2'` — MoE counterpart
to `--ffn URL`. Maps expert ID ranges to remote URLs; passed through to
`RemoteExpertBackend` in larql-inference. See also `larql-lql/ROADMAP.md` Phase 3
for the LQL grammar surface.

---

## Shipped — 2026-04-30

| What | Notes |
|------|-------|
| `larql parity --component layer` extended to dense models | Was MoE-only via `LARQL_DUMP_RESIDUALS`; now also handles dense by setting `LARQL_METAL_DUMP_LAYERS` and reading per-layer `metal_layer_NN_h_out.f32` / `metal_layer_NN_h_post_attn.f32`. Used to confirm Gemma 4 31B dense matches between CPU and Metal at every layer (cos ≥ 0.9999), which localised the bug to chat-template / sampling rather than the math |
| `larql parity --component lm-head` works on dense vindexes | The MoE-only gate (`is_hybrid_moe()` check) only fires for `moe-expert` / `moe-block` now; `lm-head` is backend-agnostic (Q4_K matvec vs f32 reference) and works on any vindex with an lm_head |
| Dense Metal path applies chat templates | `walk_cmd::run_predict_q4k` was sending the raw user prompt to `encode_prompt`; chat-template wrapping only happened for the `--moe-shards` / `--moe-units-manifest` paths. Both paths now go through `larql_inference::chat::render_user_prompt`. Fixes "The answer is:" looping on Gemma 4 31B dense and the "more questions instead of answers" frame on Gemma 3 |
| Auto-injected default system prompt for Gemma 4 (all variants) | Gemma 4 needs a system prompt to enter answer mode; `LARQL_NO_DEFAULT_SYSTEM=1` opts out, `LARQL_SYSTEM=<text>` overrides |
