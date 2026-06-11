---
name: research-infra
description: >-
  Context primer + workflow for researching and backtesting trading strategies in THIS repo's
  infrastructure. Load this BEFORE starting any strategy-research or backtest task — it replaces
  re-gathering infra context every chat. Covers the mandated workflow (vectorized frictionless
  proof-of-concept FIRST, then iterate into Nautilus), the codebase-hygiene rules (no single-use
  scripts in the packages; experiments isolated in research/experiments/; delegate new architecture
  to a sub-agent), the spec→adapter→gate→vault pipeline, the data feeds, and the critical gotchas.
  Trigger: researching/backtesting a strategy idea, designing a signal, or any question about how
  the research/backtest infra works. Invoke as /research-infra.
---

# /research-infra — how we research & backtest strategies here

This is the **standing context** for strategy research in this repo. Read it, then act — do **not**
re-run a multi-agent infra sweep or re-derive the map below; it's already done. Drop into
`codegraph_explore` / `graphify query` only for a *specific* symbol this doc points you to.

Most of the territory is the **`research/`** folder. Methodology source-of-truth is
**`docs/library/Strategy_research/`** (strategy_spec.md, pnl_lanes_and_validation.md,
strategy_engineering.md, agent_harness.md, …).

---

## The mandated workflow (do these in order)

> **Default for almost every idea: prove the edge with a vectorized, frictionless backtest BEFORE
> spending any effort on Nautilus, costs, or execution realism.** If the edge doesn't survive a
> clean signal×return test, it will not survive spread/swap/fills — stop early and cheaply.

**Phase 0 — Vectorized frictionless proof-of-concept (ALWAYS FIRST).**
- A cheap pandas test: load candles, build the signal, `shift(1)` so weights are known at *t‑1*
  and returns are earned at *t*, sum P&L, report Sharpe / maxDD / ann%. **No costs, no spread,
  no Nautilus.** Just: *does a signal exist at all?*
- **Lookahead discipline is mandatory** even here: the signal at a decision time may use only data
  ≤ that time; weights must be `.shift(1)` before multiplying returns; independently re‑derive the
  P&L a second way and assert they match (see the verification blocks in past studies). A POC that
  silently looks ahead wastes the whole investigation.
- Two ways to run Phase 0, pick by what the signal is:
  - **(a) Ad‑hoc study** — the signal isn't (yet) one of our nodes → a self‑contained script in
    `research/experiments/<slug>/` (see *Codebase hygiene*). Fastest iteration; the right place for
    a brand‑new idea.
  - **(b) Formal exploration** — the signal maps to a bias node + a `StrategySpec` → the
    `exploration` phase, which is frictionless by construction (it *asserts* the realistic Nautilus
    lane is unreachable). Use this once the idea is node‑shaped and you want the EDA / robustness /
    permutation machinery.

**Phase 1 — Iterate & add realism (only after Phase 0 shows promise).**
- Now bring in costs and event‑driven fills via the Nautilus lanes: modeled spread/swap, the
  rollover‑flatten overlay, the validation lane, the sandbox sim. This is where most *apparent*
  edges die (e.g. a gross‑positive intraday MR that goes net‑negative on spread). Expect it.

**Phase 2 — Formalize & productionize (only for survivors).**
- Express the strategy as a `StrategySpec` (`research/specs/<slug>.json`) → run `exploration` →
  `validation` (fires the **portfolio‑addition gate**) → human reads metrics → `/research-promote`
  writes it to the vault. Metrics are **advisory**; a human decides accept/reject — there are no
  auto accept/reject gates.

**Record the outcome regardless of result** in `research/experiments/<slug>/FINDINGS.md`. A killed
idea documented is worth as much as a survivor — it stops us re‑testing it. (`commodity_cross_mr`
is the model FINDINGS.md.)

---

## Codebase hygiene (hard rules — the user cares about this)

1. **No single‑use scripts in the packages.** Never drop a one‑off experiment script into
   `nodes/`, `ensemble/`, `features/`, `lib/`, `execution/`, `data_platform/`, `cache/`, etc.
   Those packages are the installable, acyclic core — keep them clean.
2. **Experiments are isolated in `research/experiments/<slug>/`.** Everything for a single study —
   scratch scripts, intermediate parquet, plots — lives there and nowhere else. Keep charts/CSVs
   worth keeping in `research/experiments/<slug>/outputs/`.
3. **The durable artifact is `FINDINGS.md`, not the scripts.** Once a finding is captured in
   `FINDINGS.md` (+ key `outputs/`), the exploratory scratch scripts are **removed**. The house
   pattern, verbatim from `commodity_cross_mr/FINDINGS.md`: *"All research scripts were exploratory
   scratch and have been removed; this memo is the record."* Don't leave dead scratch behind.
4. **Reuse before you build.** There are 50+ bias nodes (`nodes/`) and a full spec→pipeline→PnL
   stack. Check for an existing node/sleeve/lane before inventing anything (`codegraph_search`,
   `graphify query`). Parsimony: a `SignalSpec` grid is capped at 300 combos — aim for < ~50.
5. **Delegate new architecture to a sub‑agent — build it cleanly, don't inline a hack.** Whenever
   the task genuinely needs *new infra* — a new bias node, a new data adapter, a new engine/lane
   wiring, a new spec field — **delegate it to a sub‑agent** (e.g. `feature-dev:code-architect`
   to design the blueprint against existing patterns, then a build agent; or `strategy-designer`
   for spec authoring). The sub‑agent builds it into the infra **following house conventions**
   (`.cursor/rules/` 001‑004: frozen dataclasses, Protocols, 100% type hints, no raw loops, the
   one‑way package DAG `lib/core → data_platform → nodes → cache → features → ensemble → execution
   → analysis → deployment → research/scripts`) **with tests** — not as throwaway code in the main
   chat. A new node subclasses the `BiasNode` ABC (`nodes/base.py`), emits a signed signal, and is
   wireable via `create_base_model_from_config` (`ensemble/ensemble_utils.py:257`,
   `model_type='signed_signal'`).
6. **Never hand‑edit the guarded surfaces** (a PreToolUse hook, `.claude/hooks/guard_research_writes.py`,
   hard‑blocks Edit/Write with exit 2): `research/feature/config.py`, `research/portfolio/config.py`,
   and the vault trees `vault/`, `vault_personal/`, `vault_cfd_prop/`. The adapter builds **fresh**
   config objects from your spec; the only sanctioned vault write is `/research-promote` (audited
   Bash save) or `POST /api/vault/commit`.

---

## Infra map (jump straight to these — don't re‑discover)

### The signal pipeline (alpha core)
`Candles (OHLCV) → BiasNode → BaseModel (signed_signal) → DiversifiedEnsemble (F = τ/σ, cap 2.0)
→ [per‑TF] TFPortfolio (instrument weights + IDM) → [cross‑TF] GlobalPortfolio (WeightLayer + FDM,
global IDM, position cap) → PositionSizer (contracts)`. Final frame:
`['ticker','datetime','forecast_score','position_fraction']`.
- `ensemble/diversified_ensemble.py` (F=τ/σ, cap 2.0) · `ensemble/portfolio_impl/tf_portfolio.py`
  (IDM) · `ensemble/portfolio_impl/global_portfolio_impl.py` (`.fit`/`.predict`, cross‑TF combine)
  · `ensemble/weight_layer.py` (FDM = `min(√(1/(mean_corr+0.01)), 2.0)`, positive‑clipped *signal*
  corr) · `execution/position_sizer.py` (`contracts = position_fraction·capital/(price·mult·fx)`).
- Build a wired portfolio from the vault: `build_global_portfolio_from_ensemble_dirs`
  (`ensemble/portfolio_impl/vault_portfolio_loader.py:53`).
- **IDM ≠ FDM**: IDM (cap 2.5, instrument *return* corr) vs FDM (cap 2.0, *signal* corr). The
  TF‑level WeightLayer is **bypassed** — WeightLayer/FDM runs once, cross‑TF, in GlobalPortfolio.
- `daily_volatility_df` (cols `['datetime','ticker','ewsd_annual_vol']`) is required everywhere or
  `predict` raises.

### The strategy spec (the config contract)
- `research/spec/strategy_spec.py` — `StrategySpec` (frozen, kw‑only) + `SignalSpec`
  (`module_name` + cartesian `param_grid`, ≤ `MAX_GRID_COMBOS=300`), `ResearchWindows`
  (train < validation < **locked test**), `RiskSpec` (`F=target_vol/σ`, cap), `ExecutionSpec`,
  `AccountSpec`, `VaultTarget` (one of 13 vault sleeves). Self‑validates in `__post_init__`.
- `research/spec/serialization.py` — `save_spec`/`load_spec`/`spec_to_dict`/`spec_from_dict`
  (BOM‑tolerant JSON). One file per idea: `research/specs/<slug>.json`. Shared by agent ↔ UI.
- `research/spec/adapter.py` — **pure** translation: `to_feature_config` → fresh `ResearchConfig`,
  `to_portfolio_config` → fresh `PortfolioResearchConfig`, `to_pnl_engine` → `NautilusPnLEngine`.
  Plus `exploration_feed_for`, `apply_vol_scaling`, `validate`. **Never** mutates canonical configs.

### The three research phases + the gate
- **exploration** (frictionless vectorized EDA → robustness → vector‑shuffle permutation):
  `research/feature/exploration/orchestrate.py::execute_exploration_phase` (asserts Nautilus lane
  unreachable). Entry: `research/feature/in_sample/run_is.py`.
- **validation** (walkforward) → fires the **portfolio‑addition gate**:
  `research/feature/pipelines/_shared.py:377` → `research/feature/portfolio_addition/gate_runner.py`.
  The gate scores the candidate against the **portfolio baseline** (`research/portfolio/config.py::load_config`
  — universe ES/NQ/GC/CL/SI, `hierarchy_equal`, split train ≤2018‑12‑31 / val 2019‑2022 / test
  2023+), **not** the feature config. **It MUST run `n_jobs=1`** (loky workers don't inherit the
  process‑global feed/EWSD blend).
- A passing gate is the only thing that unblocks the vault write.

### The PnL / backtest lanes (Phase 1 realism)
- **Vectorized (frozen parity baseline):** `research/portfolio/pnl/pnl_engine.py::VectorizedPnLEngine`
  — never overwrite it; it's the reconciliation anchor.
- **Research realistic (Nautilus):** `make_pnl_engine('nautilus', window_policy=ROLLOVER_FLATTEN_REENTER,
  measure_spread=True)`. Parity check: `scripts/nautilus_reconciliation.py --ticker NQ`
  (PASS = corr > 0.99 AND |vol_ratio−1| < 0.02).
- **Final‑validation lane (run once before promoting, lookahead‑free):**
  `research/validation/validation_lane.py::run_validation_backtest` — the *real* live
  `VaultRebalanceStrategy` + `VaultForecastEngine.evaluate(as_of=sim_clock)`. CLI:
  `scripts/validate_candidate.py`.
- **Sandbox sim (Darwinex signal + modeled FTMO spread/swap):** `scripts/run_sandbox_sim.py`.

### Data feeds & cache (prerequisite for any run)
- `data_platform/loaders.py` (re‑exported lazily as `lib.core.helpers.load_data`) loads candles;
  `codegraph_explore` the exact signature before writing a POC loader.
- Feed selection is **process‑global** via `lib/core/research_feed.py::set_research_feed`. ⚠️ The
  module default is `'futures'` (LEGACY), **not** `'cfd'` — pipelines flip to CFD only by calling
  `set_research_feed` at entry. Set `$env:RESEARCH_DATA_FEED='cfd'` (or `futures`/`futures_ratio`)
  to pin it. Cache is feed‑namespaced: futures → `.cache/trading_algo/central_cache` (what LIVE
  runs on); others → `central_cache/_feeds/{feed}`.
- Bootstrap cache: `python -m cache.runtime.bootstrap_source_candles --timeframes D W M`.

### Experiment harness — use research/toolkit/ (mandated)
- **`research/toolkit/`** (`bars.py`, `sessions.py`, `metrics.py`, `verify.py`, `sizing.py`,
  `costs.py`) is the **mandated** helper layer for experiment scripts. Import from there rather
  than re‑inventing bars loading or metric computation in scratch scripts.

### Data metadata lookups — use the registry CLI + reader (mandated)
- **`data/registry.db`** exists and is the queryable metadata index over all stores, instruments,
  research runs, vault entries, and live‑trading records. It is rebuildable with
  `python -m data_platform.registry rebuild`.
- **Mandated metadata lookup commands** (all read-only, safe to run anytime):
  ```powershell
  # What stores exist and how stale are they?
  python -m data_platform.registry coverage
  python -m data_platform.registry freshness
  # What research runs are in the DB?
  python -m data_platform.registry runs [--kind exploration|validation] [--limit N]
  # Vault lineage for a feature or run:
  python -m data_platform.registry lineage <feature_name_or_run_id>
  # Fill / slippage records:
  python -m data_platform.registry fills [--broker darwinex]
  # Full store inventory (Markdown):
  python -m data_platform.registry report
  # Spread / cost surface:
  python -m data_platform.registry costs [--refresh] [--instrument NDX]
  ```
- **`data_platform.registry.reader`** is the Python API for the same queries (used by
  `frontend/api/data.py` and toolkit modules). Always use `db.connect_readonly()` for reads.
- **Nautilus catalog** for the 16 research instruments (ES, NQ, GC, CL, SI, EURUSD, GBPUSD,
  USDJPY, XAUUSD, NDX, SP500, XTIUSD, XAGUSD, GDAXI, UK100, WS30) with M1 bars back to 2018+
  is already persisted under `data/nautilus_catalog/`.
- `spec.data_feed` is a **vestigial label** — it does not drive feed selection.

### Metrics
- Canonical scalar metrics: `features/validation/objective_metrics.py` (`metric_sharpe/sortino/
  calmar/t_stat/profit_factor`), annualized via `TimeFrame.bars_per_year` (D=252). Drawdown/equity
  helpers in `analysis/`. Frontend reads per‑run CSVs (`headline/grid/plateau/equity`) → Plotly.
- Ad‑hoc `research/experiments/*` study scripts hardcode their own `sqrt(252)` stats — fine for a
  POC, but **not** the canonical metric path; don't conflate them.

---

## Commands (Windows PowerShell, repo root — ALWAYS the venv interpreter)

Use `.\.venv\Scripts\python.exe` (or `.\venv\...` if that's the folder). Bare `python` pulls
system Python → `ModuleNotFoundError: nautilus`.

```powershell
# Frontend research workbench (recommended human-driven path)
.\.venv\Scripts\python.exe -m uvicorn frontend.api.server:app --reload --port 5057
cd frontend\web; npm run dev        # 5173, proxies /api -> 5057

# Feature-research CLI phases (idea expressed as a node/spec)
.\.venv\Scripts\python.exe -m feature_research exploration   # aliases: in-sample, eda
.\.venv\Scripts\python.exe -m feature_research validation    # fires portfolio-addition gate (n_jobs=1)

# Nautilus / realism lanes
.\.venv\Scripts\python.exe scripts\validate_candidate.py --vault-root vault --start 2024-01-01 --end 2025-01-01 --tickers ES NQ GC CL SI
.\.venv\Scripts\python.exe -m scripts.run_sandbox_sim --days 20
.\.venv\Scripts\python.exe scripts\nautilus_reconciliation.py --ticker NQ

# Cache bootstrap (prereq)
.\.venv\Scripts\python.exe -m cache.runtime.bootstrap_source_candles --timeframes D W M
```

Vault promotion is **human‑gated only**: `/research-promote <experiment-dir>` or
`POST /api/vault/commit`. Verify the exact raw save module via codegraph at the time (docstrings
cite stale pre‑reorg paths).

The two agent‑authoring skills already exist — `/research <brief.md>` (designs a spec, runs the
pipeline + Nautilus sim, iterates ≤3×, emits a memo) and `/research-promote`. Per CLAUDE.md the
frontend app supersedes the `/research` orchestrator, but both are present.

---

## Critical gotchas (the ones that silently corrupt a run)

- **Lookahead, twice over.** (1) Hand POCs: weights `.shift(1)`, signal uses only past data,
  re‑derive to confirm. (2) Pipeline: the Nautilus lane needs `shift_positions_to_holding` (decided
  day *t* held *t+1*) — without it parity corr drops ~0.99→~0.90; the final‑validation lane is
  lookahead‑free only because `VaultForecastEngine.evaluate(as_of=sim_clock)` caps the window.
- **Feed defaults to `futures` (legacy), not `cfd`.** Verify the entry `set_research_feed` fired or
  pin `$env:RESEARCH_DATA_FEED` before trusting any run.
- **Portfolio‑addition gate must be `n_jobs=1`.** Don't "optimize" it back to parallel.
- **Exploration is frictionless by design and asserts it** — never import the realistic Nautilus
  engine into exploration. Permutation there is a *feature/vector* shuffle (fixed target), not a
  target shuffle.
- **Test window is locked** — scored once, never used for selection. The gate baseline is
  `research/portfolio/config.py`; changing that universe/weights moves every feature's bar.
- **Feature JSON is strict:** exactly one `signed_signal` base model, no grid params at save
  (freeze one combo first), no legacy binning keys. `create_base_model_from_config` is the only
  active base‑model path (and has no covering tests — verify after edits).
- **MT5 timestamps are broker EET/EEST mislabelled as UTC** (ET = stored − 7h); run M1 timestamps
  through the broker‑tz conversion or the decision clock fires at the wrong instant
  (`docs/library/Data/mt5_timezones.md`).

---

## Quick decision flow

```
New strategy idea
  └─ existing node/sleeve covers the signal?
        yes → StrategySpec → exploration (frictionless)         [Phase 0b]
        no  → ad-hoc vectorized POC in research/experiments/<slug>/  [Phase 0a]
  └─ edge survives frictionless?  no → write FINDINGS.md (killed), stop.
        yes → add costs / Nautilus realism                       [Phase 1]
  └─ survives realism?  no → FINDINGS.md (killed-on-cost), stop.
        yes → StrategySpec → validation + portfolio-addition gate [Phase 2]
  └─ gate passes + human approves → /research-promote → vault
  └─ need NEW infra at any step? → delegate to a sub-agent to build it cleanly (+tests).
```
