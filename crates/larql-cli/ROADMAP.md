# Roadmap — larql-cli

## Current state

Primary verbs: `run`, `chat`, `pull`, `list`, `show`, `rm`, `link`, `serve`, `bench`.
490 tests passing across the workspace. Legacy research commands gated under
`larql dev <subcmd>` for backwards-compat. Dual cache (HuggingFace hub +
`~/.cache/larql/local/`) with shorthand resolution (`larql run gemma3-4b-it-vindex`).

---

## P0: Generation UX (blocks demo)

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
