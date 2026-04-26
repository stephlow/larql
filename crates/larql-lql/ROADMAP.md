# Roadmap — larql-lql

## Current state

INSERT/SELECT/USE/COMPILE/TRACE grammar fully parsed. 317 tests passing
(146 parser, 93+ executor integration, 17 in-module unit tests). INSERT
supports `MODE KNN` (residual retrieval override, validated at 25K edges)
and `MODE COMPOSE` (FFN-overlay, ~5–10 facts/layer). `COMPILE INTO VINDEX`
bakes patches into canonical `down_weights.bin`. `COMPILE INTO MODEL` applies
MEMIT (opt-in via `LARQL_MEMIT_ENABLE=1`). `WITH alpha/gate_scale/refine_rounds/mode`
clauses accepted; `refine_rounds` implementation is a TODO (see P1 below).

---

## P0: Phase 3 — Expert routing grammar

### `USE "..." WALK ONLY WITH EXPERTS REMOTE { ... }` grammar
**Status**: Not started  
**Files**: `src/parser/lifecycle.rs`, `src/executor/lifecycle/use_cmd.rs`  
New clause on the `USE` statement that attaches a remote expert map before
any `WALK` or `INFER` call. Syntax:
```sql
USE "gemma4-26b.vindex" WALK ONLY WITH EXPERTS REMOTE {
  "0-31":  "http://host1:8080",
  "32-63": "http://host2:8080"
};
```
Parser extension: parse the JSON-like expert map into `HashMap<ExpertRange, Url>`.
Executor: store the map on the `Session`; wire into `RemoteExpertBackend` in
larql-inference before the next `WALK` / `INFER`.

### `RESHARD EXPERTS { ... }` statement
**Status**: Not started  
**Files**: `src/parser/mutation.rs` (or new `src/parser/expert.rs`), `src/executor/`  
Allows live redistribution of experts across servers without a `USE` restart.
Useful for the demo "kill one shard, rewire on the fly" proof shot:
```sql
RESHARD EXPERTS { "0-63": "http://new-host:8080" };
```
Updates the `Session`'s expert map in place; subsequent WALK/INFER calls use
the new routing immediately.

---

## P1: INSERT quality

### Refinement rounds — `WITH refine_rounds = N`
**Status**: TODO in `mutation/insert/compose.rs`  
The `INSERT INTO EDGES … WITH refine_rounds = N` clause is parsed and stored
but the executor ignores `N` and always runs the cliff-breaker single-round
refine. Implement the loop: after the initial slot install, run up to `N`
additional refine passes that re-capture residuals under the live install
and re-orthogonalise, lifting `self_scores` when the first pass undershoots.
Validated manually in Python (`compile_facts.py refine(rounds=2)` lifts 5/5);
needs to be wired into the Rust executor path.
