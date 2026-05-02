# ADR-015: Isolated kernel speedup ≠ end-to-end win when batched throughput is already saturated

**Status**: Accepted (recurring pattern, four confirmed instances)
**Date**: 2026-05-02 (initial; updated with NR2 then `q4k_matvec` lm_head)
**Context**: A pattern that has now reproduced across three independent kernel
optimisation attempts on Gemma 3 4B decode. Future kernel work needs to budget
benchmark cost against this prior — the isolated `diag_profile_kernels` number
is necessary but not sufficient evidence to promote a new shader.

## The pattern

A candidate kernel shows a meaningful speedup in the **isolated** profiler
measurement (one commit+wait per call, includes ~20 µs GPU spin-up) but
either matches or *regresses* the **batched** measurement (n_layers
dispatches in one cmd buffer, single commit+wait — matches the real decode
pipeline).

End-to-end decode benchmarks then track the batched number, not the isolated
one. The isolated win was real — it just was not load-bearing under the
production workload.

## Four confirmed instances

| Kernel | Isolated speedup | Batched delta | End-to-end | Outcome |
|---|---|---|---|---|
| `q4k_ffn_gate_up_f16acc` (2026-04-28) | 1.79× (0.607 → 0.340 ms) | within noise | parity on quiet GPU | opt-in only (`LARQL_F16_ACC=1`) |
| `attn_fused` (2026-05-01) | merged 2 kernels into 1 | TGs collapse 12 → 8 | **−1.45 ms regression** | opt-in only (`LARQL_FUSED_ATTN=1`) |
| `q4k_ffn_gate_up_nr2` (2026-05-02) | 1.47× (0.591 → 0.401 ms iso) | 279 → 267 GB/s (−4%) | **−0.62 ms regression on GPU fwd** | not promoted; opt-in `LARQL_GATE_UP_NR2=1` |
| **`q4k_matvec` lm_head** (broken-fast) | n/a — different category | 1.47 ms vs stride-32's 2.95 ms | **+10 tok/s but FAILS smoke** ("Capital" / truncated) | opt-in only (`LARQL_METAL_LM_HEAD=1`); production stays on stride-32 |

The mechanisms differ but the symptom is identical at the perf level — a
candidate that looks like a strict win at one measurement granularity and
loses (or breaks) at the actual production granularity.

- **f16 acc**: the kernel was already at 274 GB/s = 74% of LPDDR5X peak.
  Freed ALU cycles got absorbed by surrounding kernels' bandwidth contention
  rather than translating to wall-clock reduction.
- **attn_fused**: dispatch fusion saved ~30 µs of cmd-buffer overhead but the
  fused kernel's larger register footprint forced 8 TGs/dispatch instead of
  the unfused path's 12. Parallelism loss dwarfed the dispatch saving.
- **NR2**: the isolated measurement caught dispatch-overhead amortisation
  that disappears once n_layers calls share one cmd buffer. The batched
  geometry is the production geometry, and NR2 is *worse* there.
- **`q4k_matvec` lm_head**: the broken-fast variant. Same Q4_K bandwidth as
  `q4k_matvec_stride32` (327 MB/token), but the 32-lane simdgroup
  reduction tree drifts ~1e-3 vs CPU's sequential dot product. On a
  262K-vocab × 2560-hidden matvec that's enough to flip top-1 on close-
  call tokens — the canonical smoke ("The capital of France is **Paris**")
  fails as "The Capital of France is: **" (capitalised and truncated).
  Same family as the historical 2026-04-26 "81–84 tok/s on broken Q4_K
  dispatch" trap (pre-fix `q4k_matvec` writing through `q4_matvec` and
  leaving 75% of output rows unwritten). Listed here because it
  surfaces under the same diagnostic — the kernel-level number looks
  great, end-to-end fails. **The broken-fast number is never a baseline.**

## Diagnostic test before promoting any new kernel

Run all three measurements before deciding:

1. **Isolated** (`diag_shader_bench` `iso_ms` column): cheap, fastest signal.
   A regression here is enough to drop the candidate. A win here is
   necessary but not sufficient.
2. **Batched** (`diag_shader_bench` `bat_ms` / `GB/s` columns): the
   production geometry. **This is the number that predicts end-to-end.**
   If batched regresses or is within noise, the candidate is not a win.
3. **End-to-end bench A/B** (`larql bench --warmup 8 -n 30 --profile`):
   final confirmation, with correctness smoke (`larql run "The capital of
   France is" -n 8 --metal` should still emit Paris).

Steps 1 and 2 take ~30 s total. Step 3 takes another minute. Skipping step 2
and going straight from isolated → end-to-end has burned three sessions; do
not skip it.

### Mechanised flow (2026-05-02)

`diag_shader_bench --profile gemma3` with `--json` and `--compare` automates
steps 1 and 2 against a saved baseline. The full save-then-compare command
pair lives in `crates/larql-compute/PERFORMANCE.md` under "How to A/B a
shader candidate" — that is the canonical promotion gate. Use
`--threshold N` to set the percent regression considered a real loss
(default 5%).

## When the pattern does NOT apply

The 8sg geometry rollout (2026-04-28) showed when isolated wins *do* carry
end-to-end: `q4k_matvec_8sg` at 55% LPDDR5X utilisation gave +5.2% end-to-end;
gate+up at 68% gave +2.1%; q6k_matvec at 84% gave 0% (regressed). The
predictor is **bandwidth headroom under the batched measurement**: kernels
below ~75% of LPDDR5X peak have room to convert isolated wins into batched
wins. Kernels above ~80% don't.

## Decision

1. ADR pinned. New shader work follows the three-step diagnostic above.
2. The lesson lives in three places so it's findable from each entry point:
   this ADR (canonical), `PERFORMANCE.md` current-state data, and
   `PERFORMANCE.md` recent-changes table per instance.

## Consequences

- New candidates that look hot in `diag_profile_kernels` isolated column do
  not justify a session of end-to-end measurement on their own.
- Kernels that pass the batched test (e.g. fused QK norm + RoPE; the
  May 2026 fusion wave that landed −1.5 ms cumulatively) are the
  evidence-based bar for promotion.
- Decode at ~76 tok/s on this hardware is closer to the parallelism /
  bandwidth ceiling than headline isolated numbers suggest. Closing the
  remaining 1.30× to ollama needs work that *changes the batched
  measurement*, not work that just makes a single kernel faster in
  isolation.

## Related

- ADR-008 (Q4_K kernel optimization findings) — predecessor pattern at the
  matvec level.
- `crates/larql-compute/PERFORMANCE.md` recent-changes table.
- `crates/larql-compute/ROADMAP.md` "P0: Production gap closers" — multi-TG
  `attn_fused` retry is the next target that explicitly works *with* this
  pattern (preserve TG count while fusing).
