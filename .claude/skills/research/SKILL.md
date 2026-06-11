---
name: research
description: Triggered, in-session strategy-research workflow. Hand it a markdown brief and it designs a parsimonious StrategySpec, runs it through the existing pipeline + Nautilus execution sim, iterates up to 3 times if the analyst flags a fixable structural flaw, and produces a research memo for a human approve/reject. Invoke as `/research <path-to-brief>`. Manual only — no scheduling, no auto accept/reject gates.
---

> **SUPERSEDED** — The primary research workflow is now the **frontend workbench** (`frontend/`):
> build/edit a `StrategySpec`, run it, and inspect results in the UI. Load `/research-infra` for
> the mandated workflow and data-access context. This skill remains available for legacy agent-driven
> runs but is no longer the recommended path.

# /research — agent-driven strategy research (orchestrator)

This skill is a **thin loader and orchestrator**, not the methodology. The methodology lives in
`docs/library/Strategy_research/` (the README + five docs) — one source of truth. Read those;
do not re-derive decisions already made there.

> **Manual, in-session — no scheduling.** A run starts when the user types `/research <brief>` and
> ends when they read the memo. The two stopping points are **the human at the keyboard**, never a
> numerical gate. (See `docs/library/Strategy_research/agent_harness.md`.)

## Inputs

`$ARGUMENTS` is the path to a markdown **brief** for a strategy idea (e.g.
`research/briefs/turnaround_tuesday.md`). If no path is given, ask the user for one.

## What already exists (call it, don't rebuild)

- `research/spec/strategy_spec.py` — the `StrategySpec` dataclass + validation (build item #1).
- `research/spec/adapter.py` — `to_feature_config` / `to_portfolio_config` / `to_pnl_engine` /
  `validate` (build item #2). The adapter builds **fresh** config objects; it never mutates the
  canonical configs.
- The full pipeline + `research/portfolio/pnl/nautilus_engine.py` (the realistic-execution lane).
- Entry points: `research.feature.in_sample.run_is`, `research.portfolio.run_portfolio_test`.

A `PreToolUse` guardrail (`.claude/hooks/guard_research_writes.py`) **denies** any `Edit`/`Write`
to `research/feature/config.py`, `research/portfolio/config.py`, and the vault trees — so the
constraints are mechanically enforced, not just hoped for.

## Workspace (one per experiment)

```
research/agent_experiments/{YYYY-MM-DD}_{slug}/
  brief.md        # copy of the input brief
  hypothesis.md   # designer output — rationale, category, parsimony justification
  spec.py         # the StrategySpec instance (updated in-place on each iteration)
  run_1/          # iteration 1 pipeline outputs + logs
  run_2/          # iteration 2 (if triggered)
  run_3/          # iteration 3 (if triggered)
  bundle_1.json   # metrics bundle for iteration 1
  bundle_2.json   # metrics bundle for iteration 2 (if triggered)
  bundle_3.json   # metrics bundle for iteration 3 (if triggered)
  memo.md         # analyst output — final deliverable (covers all iterations)
```

An invented bias node goes in `nodes/experimental/{slug}.py` so the pipeline can import it.

## The flow

1. **Set up the workspace.** Copy the brief to `research/agent_experiments/{date}_{slug}/brief.md`.

2. **Design** — invoke the `strategy-designer` subagent with the brief path and workspace path.
   It classifies the category (per `strategy_engineering.md`), studies existing `nodes/` and vault
   sleeves for reuse/redundancy, and writes `spec.py` + `hypothesis.md`. It returns the spec summary.

   Show the user a one-line spec summary (module, combos, direction, sleeve) — then immediately
   continue. Do not stop and wait.

3. **Execute (iteration 1)** — invoke the `backtest-executor` subagent with:
   - workspace path
   - `run_dir = "run_1"`
   - prior bundles: none
   It loads `spec.py`, calls the adapter, runs `run_is`, assembles `run_1/` + `bundle_1.json`.
   Returns the bundle path.

4. **Analyze (iteration 1)** — invoke the `research-analyst` subagent with:
   - `bundle_1.json` path
   - workspace path (to read `hypothesis.md`)
   - prior bundles: none
   It writes `memo.md` via the six lenses and closes with an **iteration signal** block
   (see §Iteration loop). Returns the `ITERATE:` verdict.

### Iteration loop (automatic, max 3 rounds, no checkpoint between rounds)

After each analyst step, scan `memo.md` for the `## Iteration recommendation` section:

- **`ITERATE: NO`** (or iteration count = 3): skip to CHECKPOINT 2.
- **`ITERATE: YES`** and `iteration_count < 3`:
  1. Extract the `DIAGNOSIS` and `NEXT_SPEC_CHANGE` lines. Show the user a one-liner for
     transparency (do **not** stop and wait): e.g. _"Iteration 2: analyst flagged [diagnosis] →
     trying [change]."_
  2. Call `strategy-designer` in **reformulation mode** — pass:
     - workspace dir
     - current `spec.py` path
     - `memo.md` path (the analyst's diagnosis is in the iteration block)
     - iteration number N
     The designer reads the diagnosis, edits `spec.py` in-place (targeted change only), and appends
     an iteration entry to `hypothesis.md`. Returns "Iteration N: changed X to Y."
  3. Increment iteration counter. Call `backtest-executor` with `run_dir = "run_N"` and all prior
     bundle paths. It writes `run_N/` + `bundle_N.json`.
  4. Call `research-analyst` with the new bundle + all prior bundles (to compare across iterations).
     The analyst overwrites `memo.md`, which now spans all iterations.
  5. Go to top of loop.

5. **Deliver.** Show the user the contents of `memo.md` in full. The run is complete.
   If they want to promote the strategy to the vault, they run a separate explicit command:
   `/research-promote <experiment-dir>` — the only path permitted to write to the vault.
   Never auto-promote.

## Hard rules (enforced by the docs + the guardrail; restate to the subagents)

- No automatic accept/reject gates anywhere — metrics are advisory; the human decides.
- The test window is **locked** — never used for selection.
- Parsimony: signal grid ≤ 300 combos (the spec enforces this at construction; aim < ~50).
- Never edit the canonical configs; the adapter builds fresh objects.
- Engine B (bracket/scalping state machine) is deferred — do not build the node-with-intrabar-stops middle.
- Iteration targets **structural** flaws (wrong exit, bad threshold, mismatched direction), not noise.
  If the analyst sets `ITERATE: YES` with no clear mechanical fix, override it — do not iterate.
