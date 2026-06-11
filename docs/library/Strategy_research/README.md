# Agent-driven research pipeline — overview & implementation handoff

> **⚠️ Superseded (2026-06-07).** The day-to-day workflow is now the **frontend research app**
> ([`frontend/`](../../../frontend/), see CLAUDE.md → "Research Frontend"): build/edit a
> `StrategySpec` and examine results in the UI, with an agent helping author the spec JSON in
> [`research/specs/`](../../../research/specs/). The **`StrategySpec` + adapter** below remain the
> config contract; the auto **`/research` orchestrator** (skill + subagents + memo loop) is retired
> in favour of the app. The methodology docs here (engineering, spec, execution, report lenses)
> are still the design reference.

**Start here.** This is the entry point for building the agent-driven strategy research pipeline.
The five docs in this folder are the **source of truth**; this README states the goal, the
cross-cutting principles, the architecture in brief, what already exists, and the build backlog in
order. Read the linked docs for the detail — do not re-derive decisions already made here.

> **Status:** fully built. The primary day-to-day workflow is now the **frontend research app**
> (`frontend/`): build/edit a `StrategySpec` in the UI (or have an agent author the JSON
> directly in `research/specs/`), launch exploration or validation, and inspect results.
> The `/research` agent orchestrator (`.claude/skills/research`, subagents, memo loop) is
> **superseded** but left in place; it is no longer the recommended path.

---

## The goal

A **triggered** (not autonomous) research workflow. The user (or an agent co-authoring
`research/specs/*.json`) builds a `StrategySpec`, then:

1. **Designs** a parsimonious strategy (classify category, study what already works, pick the
   simplest construct, minimize parameters) — using the frontend Spec Builder or by hand-writing
   the JSON.
2. **`StrategySpec`** — one flat, self-validating object. A thin **adapter** translates it into
   the *existing* research/backtest machinery. The agent/UI never edits the canonical configs.
3. **Runs** it through the pipeline via the frontend (`POST /api/runs`, phase `exploration` or
   `validation`).
4. **Examines results** in the frontend (plateau / equity / grid / headline views via Plotly), or
   an agent reads the raw CSV/JSON artifacts and writes a research memo.
5. The **human reads results and decides.**

This replaced two messy things: the sprawling `research/feature/config.py` (too complex for an
agent to drive) and the old ~10k-line research dashboard (replaced by the React frontend app).

---

## Cross-cutting principles (respect these everywhere)

1. **Metrics guide, the human decides — no automatic accept/reject gates.** Existing gate
   thresholds become *advisory reference lines*, not verdicts.
2. **Pipeline emits raw data; the agent judges holistically.** Not a numerical pass/fail pipeline.
3. **Code computes numbers, the agent interprets them** — never hallucinate a Sharpe/correlation.
4. **Parsimony is the front-line overfit defense** — few parameters, simplest construct, fix what
   needn't be optimized, small grids.
5. **Keep the existing pipeline unchanged.** Bias nodes (single-tf, signal-only, `{-1,0,+1}`),
   `DiversifiedEnsemble`, `WeightLayer`, `TFPortfolio`/`GlobalPortfolio`, `PositionSizer` are not
   modified. The agent writes a `StrategySpec`; the adapter builds fresh config objects.
6. **The test window is locked** — scored once, never used for selection.
7. **Param grid ≤ 300 combos** (aim < ~50), validated at spec construction.

---

## The five docs (read in this order)

1. **[[Strategy_research/strategy_engineering]]** — design a parsimonious strategy from a brief.
   Category taxonomy mapping `mean-reversion / breakout / trend / seasonal / flow / pairs / buy-hold`
   → node folders (`nodes/…`) → vault sleeves. Parsimony rules.
2. **[[Strategy_research/strategy_spec]]** — the formal `StrategySpec` object: every field, the
   per-mode window defaults, vol scaling, execution, vault target, and the adapter mapping table.
3. **[[Strategy_research/execution_architecture]]** — the **two engines**; Engine A (what we run)
   and Engine B (deferred), plus portfolio-level swap avoidance.
4. **[[Strategy_research/research_report]]** — the report contract: raw data → agent memo, the six
   analytical lenses, the phase-structured memo template.
5. **[[Strategy_research/agent_harness]]** — the Claude Code operational layer (superseded, kept
   for reference): the `research` skill, three subagents, guardrail hooks, scripts, workspace, and
   the two checkpoints. **Current workflow:** use the frontend app instead.
6. **[[Data/feed_and_execution_decision]]** — *existing* decision of record the above lean on
   (CFD-native research, swap overlay, open-to-close convention).

---

## Architecture in brief

**`StrategySpec`** (flat frozen dataclass the agent writes). Fields: identity (name, hypothesis,
author); universe (`tickers`, `data_feed` = norgate_futures | darwinex_cfd); `mode` (daily =
D/W/M | intraday = H1/M15…); `windows` (per-mode defaults, test locked); `signal` (`module_name` +
`param_grid` ≤300); `direction`; `vol_scaling` (off | blended | long_only); `vol_scaling_model`
(inherit_daily | intraday_custom); `risk` (target_vol τ, forecast_cap, max_position_pct, buffer);
`account`; `execution`; `vault` (sleeve + ensemble_name). **Metrics are NOT fields** — they're
computed and reported.

**Adapter** — pure translation `StrategySpec` → `ResearchConfig` / `PortfolioResearchConfig` /
`NautilusPnLEngine`. Validates (≤300 combos, windows ordered, sleeve ∈ the 13, fill_feed
consistency). Never mutates the canonical configs.

**Vol scaling** — EWSD is a **70/30 blend** (32-day EWMA short + 2520-bar long), `F = signal·τ/σ`
capped ±2. Intraday **inherits daily σ** by default (σ from daily returns, *not* the spans on
intraday bars).

**Execution — two engines:**
- **Engine A (the signal book, what we run):** level signal → vol-scaled → `position_fraction` →
  executed **market or passive-limit**, **no per-trade stops**. Per-leg `entry_policy` /
  `exit_policy` (market_on_open | limit_at_touch | limit_improve); `unfilled_limit`
  (cross_after | carry); `fill_feed` derived (fine M1/tick iff any leg uses a limit). The node
  **never tracks fills** — the signal always dictates the target; no latch, no autonomous exit.
- **Engine B (deferred, walled off):** bracket/scalping state machine on **partitioned**
  instruments — NOT a bias node, requires tick data, kept out of the vault. Build only for fast
  mean-reversion. **Do not build the middle** (a node that also needs intrabar stops).

**Holding vs swap** — `holding: overnight | intraday` is the **alpha intent** (in the spec).
**Swap avoidance** (rollover flatten/re-enter) is a **portfolio-level** cost overlay on the *net*
position (NOT in the spec; already a decision of record + implemented in
`execution/rollover_overlay.py`). Research reports **two bounds** — close-to-close vs
rollover-bounded open-to-close — to size the forgone overnight gap vs the swap cost.

**Research report** — pipeline emits raw data (CSV/JSON + a run manifest); the agent reasons over
six lenses (A rationale & simplicity, B parameter plateau/sensitivity, C overfit, D IS→val
persistence, E degradation, F diversification) and writes a phase-structured memo
(in-sample → validation → portfolio-addition → holdout) with a verdict, ranked concerns, and ≤2
artifact links. **No auto-gates.**

---

## What is built

- **`StrategySpec`** dataclass + sub-objects + construction-time validation
  (`research/spec/strategy_spec.py`). Tests: `tests/unit-tests/spec/`.
- **Adapter** `StrategySpec` → `ResearchConfig` / `NautilusPnLEngine`
  (`research/spec/adapter.py`, `research/spec/serialization.py`).
- **Frontend app** — FastAPI (`frontend/api/`) + React/Vite/Mantine (`frontend/web/`). Spec
  Library, Spec Builder, Runs & results (plateau/equity/grid/headline via Plotly), Portfolio
  research, Vault archive. Run: `.\.venv\Scripts\python.exe -m uvicorn frontend.api.server:app
  --reload --port 5057` + `cd frontend\web && npm run dev` (port 5173).
- **`research/specs/`** — round-trippable JSON; agent and UI share the same files.
- **Per-leg `entry_policy` / `exit_policy` + `CARRY`** in `ExecutionSpec` (spec + adapter built;
  NautilusEngine side is the remaining open item — see doc 3).
- The full pipeline: bias nodes (142, in `nodes/<category>/`), `DiversifiedEnsemble`,
  `WeightLayer` (9 methods + FDM), `TFPortfolio`/`GlobalPortfolio` (IDM), `PositionSizer`.
- `research/portfolio/pnl/nautilus_engine.py` — `NautilusPnLEngine`, `TargetRebalanceStrategy`,
  `FillDiagnostic`, `ExecutionWindowPolicy`, `ExecutionPolicy`, `CrossAfterPolicy`.
- Passive-limit execution + synth M1-spread quotes (CFD migration).
- `execution/rollover_overlay.py` swap overlay + the open-to-close research convention.
- Phases emit CSV/JSON + workspace manifest (`research/feature/ui/`).
- **`research/validation/`** — final-validation lane (`validate_candidate.py`); lookahead-free
  check before promotion. See [[Strategy_research/pnl_lanes_and_validation]].
- **`research/rollover_cost/`** — rollover-cost execution study (decision-grade; swap overlay
  policy encoded in `execution/rollover_overlay.py`). See README in that folder.
- **`research/workspace/`** (was `tools/research_workspace/`) — workspace discovery + panels.

## Open items

- **Per-leg policy in `NautilusPnLEngine`** (today uses one policy for both legs; spec models
  `entry_policy` / `exit_policy` separately). (Doc 3.)
- **Two-bounds reporting** (c2c vs rollover-bounded o2c) as a standard research diagnostic.
  (Doc 3.)
- **`INTRADAY_CUSTOM`** vol-scaling model; intraday `TimeFrame` enum members (currently D/W/M).
- **Engine B** (bracket/scalping state machine) — deferred.

---

## Hard constraints for the implementer

- **Do not** mutate `research/feature/config.py` or `research/portfolio/config.py` — the adapter
  builds fresh config objects from the spec.
- **Do not** change the `BiasNode` interface or the `{-1,0,+1}` signal contract.
- **No** automatic accept/reject gates anywhere; metrics are advisory.
- The **test window is never** used for selection.
- Enforce **parsimony**: validate the grid is small; fixed params are stated, not swept.
- **Engine B is deferred**; do not build the node-with-intrabar-stops middle.
- Windows: daily `train 2000–2018 / val 2019–2022 / test 2023+`; intraday
  `train 2018–2022 / val 2023–2024 / test 2025+`.
- venv: `.\.venv\Scripts\python.exe`; the 13 vault sleeves are in `ensemble.vault.constants`.

> _Authored 2026-06-06. Updated 2026-06-07 to reflect built state._

> _Verified against current code via CodeGraph on 2026-06-07._
