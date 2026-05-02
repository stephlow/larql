# OV/RD Dev Command

`larql dev ov-rd` is the experimental harness for attention output-vector
rate-distortion work. It is deliberately a `dev` command, not a production
extraction command.

The core question is whether an attention head's pre-`W_O` output can be
replaced by a compact table:

```text
runtime state -> address -> residual-space lookup/add
```

For the current L0H6 line of work, the stable findings are:

```text
oracle table exists
Mode D residual-table materialization works
held-out mean/p95 can pass
the current dominant group-0 code is not addressable from shallow state
```

## What Belongs Here

Keep Rust code here when it needs exact model/vindex behavior:

- Q4K vindex loading and layer tensor insertion
- attention `pre_W_O` capture
- `W_O`-visible projection and roundtrip checks
- oracle low-rank and PQ reconstruction
- Mode D residual-delta table materialization
- final-logit KL/top-k evaluation through the real forward path
- canonical JSON artifacts that other tools consume

The command should remain an orchestrator plus faithful runtime validator. It
should not become the place where every new probe, plot, or clustering variant
lives.

## What Should Move To Python

Use Python over exported artifacts for fast-changing analysis:

- code stability tables
- plotting and report tables
- window hashes, bag-of-token hashes, shingling, MinHash
- decision trees, nearest-centroid variants, and classifier sweeps
- feature/code correlation scans

If a Python probe becomes a serious runtime candidate, reimplement only that
candidate in Rust after its artifact contract is clear.

Small summary diagnostics that are part of the canonical JSON schema can stay in
Rust. For example, entropy/JS divergence helpers belong in `metrics.rs` if they
are emitted by `oracle-pq`, while broader exploratory scans should use Python
against exported artifacts.

## Artifact Contract

Rust should export enough canonical state that Python can iterate without
rerunning full model forward passes for every idea:

```text
prompt id / stratum / tokens
layer-input residual rows
captured pre-W_O head rows
oracle PQ codes by position
baseline and replacement logits or metrics
per-prompt KL/top-k summaries
```

Prefer compact binary arrays plus JSON metadata for large matrices. JSON alone
is fine for summaries and small diagnostics.

## Documentation Boundary

Use `experiments/38_ov_rate_distortion/RESULTS.md` as the lab notebook: commands
run, artifacts written, negative results, and interpretation.

When a result becomes architectural rather than experimental, promote it to a
short stable doc under `docs/`, for example:

```text
docs/attention-tableability.md
```

The experiment log should stay detailed and chronological. The docs should be
short, curated, and claim-focused.

## Current Refactor Direction

This directory replaced the old single-file
`commands/extraction/ov_rd_cmd.rs`. The command is now under `dev` because these
runs are experimental probes, not stable vindex extraction verbs.

Current split:

```text
cmd.rs             CLI dispatch only
address.rs         address predictor models and address-match helpers
basis.rs           W_O roundtrip basis, z-space PCA fitting, and eigensolver
capture.rs         stage-0 pre-W_O capture and head statistics
input.rs           prompt loading, held-out splits, and CLI string parsers
metrics.rs         KL, entropy, top-k, and distribution helpers
oracle.rs          roundtrip and low-rank oracle checks
oracle_pq.rs       PQ fitting, Mode D validation, and address probes
pq.rs              PQ codebooks, Mode D tables, and k-means mechanics
reports.rs         JSON artifact schemas
runtime.rs         Q4K tensor insertion/removal and local dequantization
sanity.rs          no-op/subtract/residual-delta equivalence checks
static_replace.rs  static mean replacement gate and shared static fitting
stats.rs           running head stats and static mean accumulators
types.rs           shared input/config identifiers
zero_ablate.rs     zero pre-W_O ablation gate
```

The next cleanup should continue splitting by experiment role:

```text
forward.rs  shared full-model forward replacement helpers, if more reuse appears
```

Do this incrementally. The first invariant is that existing `larql dev ov-rd`
commands keep their behavior and artifact schema unless a schema change is
intentional and documented in the experiment results.
