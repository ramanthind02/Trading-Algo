# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. It combines **behavioral guidelines** (how to work) with **project-specific instructions** (what this repo is and how it is structured).

## Behavioral Guidelines

Guidelines to reduce common LLM coding mistakes. These bias toward caution over speed; for trivial tasks, use judgment.

### 1. Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.
- Ask: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

This complements the coding conventions in `.cursor/rules/` — follow those patterns, but don't add layers the task didn't require.

### 3. Surgical Changes

Touch only what you must. Clean up only your own mess.

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that **your** changes made unused.
- Don't remove pre-existing dead code unless asked.

**Test:** Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

- "Add validation" → write tests for invalid inputs, then make them pass.
- "Fix the bug" → write a test that reproduces it, then make it pass.
- "Refactor X" → ensure tests pass before and after.

For multi-step tasks, state a brief plan:

1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

### 5. Delegate Implementation to Cheap Subagents

The main (expensive) agent does architecture, specs, orchestration, and review — **always
delegate plan implementation to cheaper grunt subagents** (Agent tool with `model: "sonnet"`
for code edits, `model: "haiku"` for purely mechanical work like file moves/renames/doc fixes).

- Write a precise, self-contained spec per delegation: exact files, exact changes, the test
  command, the expected outcome, and what NOT to touch.
- Run independent delegations in parallel (background agents) when their files don't overlap;
  serialize them when they would run pytest over the same area simultaneously.
- Keep in the main agent: architecture decisions, parity-sensitive or live-trading-adjacent
  changes, and final review.
- Verify subagent work by running the tests yourself before marking a task complete.

These guidelines are working if: fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Project Overview

Systematic trading framework implementing Robert Carver's methodology. Generates trading signals from technical indicators (bias nodes), combines them through ensemble learning with diversification multipliers, and converts signals to tradeable positions.

## Environment Setup

**CRITICAL**: This repository uses a **shared virtual environment** located at the repository root.

- The venv is shared across all git worktrees
- **NEVER** create new virtual environments in worktrees or subdirectories
- **Linux/macOS**: activate with an absolute path, e.g. `source /home/raman/repos/Trading-Algo/venv/bin/activate`
- **Windows PowerShell**: do not use `source venv/bin/activate` (that is a Unix shell pattern). Prefer invoking the venv interpreter directly so you never rely on PATH or activation:
  - `.\.venv\Scripts\python.exe` or `.\venv\Scripts\python.exe` (use whichever folder exists at the repo root)
- **Agents / IDE**: if `pytest` appears missing, the wrong interpreter is selected (often system Python). Use the venv’s `python.exe` with `python -m pytest`, not `pytest` on PATH.

## Commands

```bash
# Activate venv (REQUIRED for all Python operations)
# Use absolute path - works from any worktree
source /home/raman/repos/Trading-Algo/venv/bin/activate

# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_integration.py -v

# Run a specific test class or method
pytest tests/test_integration.py::TestFormulaVerification -v

# Compile Cython extensions (optional, for performance)
python lib/compute/cython/setup_cython.py build_ext --inplace
```

## Commands (Windows PowerShell, repo root)

Prefer the venv interpreter explicitly; `python -m pytest` avoids needing `pytest` on PATH.

```powershell
# Example: one file
.\.venv\Scripts\python.exe -m pytest tests\unit-tests\feature_research\test_permutation_pipeline.py -v

# In-sample research (same as: python -m research.feature.in_sample.run_is)
.\.venv\Scripts\python.exe -m research.feature.in_sample.run_is

# Feature–vault correlation CSV (runs research.feature OOS once; enable FeatureVaultCorrelationConfig in research.portfolio.config.load_config; unset vault_root scans prop vault, set Path for vault_personal)
.\.venv\Scripts\python.exe -m research.portfolio.run_feature_vault_correlation
```

If your venv directory is named `venv` instead of `.venv`, use `.\venv\Scripts\python.exe` in place of `.\.venv\Scripts\python.exe`.

## Visualization Policy

- Keep research artifacts CSV-first: pipelines write tabular or JSON/CSV outputs, and
  Matplotlib is the preferred plotting consumer for research charts.
- Keep QuantStats tearsheets, prop-firm HTML reports, and Norgate migration QA plots unless the task explicitly says otherwise.

## Research Frontend (StrategySpec workbench)

The primary research workflow is the **frontend app** in [`frontend/`](frontend/): build/edit a
`StrategySpec`, run it, and examine results — instead of hand-editing `research/feature/config.py`.
An **agent** (in a chat) and the **UI** share the same round-trippable JSON in
[`research/specs/`](research/specs/), so you can have an agent design a strategy and then run +
inspect it in the app.

- **Config object:** `StrategySpec` ([`research/spec/strategy_spec.py`](research/spec/strategy_spec.py)) +
  JSON (de)serialization ([`research/spec/serialization.py`](research/spec/serialization.py)). The
  thin adapter ([`research/spec/adapter.py`](research/spec/adapter.py)) translates a spec into the
  existing pipeline configs — it never edits the canonical configs.
- **Backend:** FastAPI JSON API in [`frontend/api/`](frontend/api/) (spec CRUD/validate, node
  catalog, runs (`exploration` + `validation`), results, artifacts, vault preview/commit, portfolio
  research, vault browser). Run from the repo root:
  `.\.venv\Scripts\python.exe -m uvicorn frontend.api.server:app --reload --port 5057`
- **Web app:** React + Vite + TS + Mantine in [`frontend/web/`](frontend/web/). Dev:
  `cd frontend\web; npm run dev` (port 5173, proxies `/api` → 5057). Prod: `npm run build` emits
  `dist/`, which the FastAPI app serves at `/`.
- **Sections:** Spec Library · Spec Builder (live-validated form) · Runs & results (plateau / equity /
  grid / headline via Plotly) · Portfolio research · Vault archive (browse vault features, re-research).
- **Feature↔portfolio dependency:** the feature **portfolio-addition gate** (a `validation`-phase run)
  scores a candidate against the **portfolio baseline** (canonical `research/portfolio/config.py`).
  A passing gate unblocks vault save. The gate is forced **`n_jobs=1`** in the spec-driven path
  because the runtime-set research feed / EWSD blend are process-global and don't propagate to loky
  workers (a parallel gate fits baseline ensembles under the wrong feed and fails).
- **The old `/research` agent orchestrator** (`.claude/skills/research`, the 3 subagents, the guard
  hook) is **superseded** by this app; it is left in place but no longer the workflow.
- Tests: `tests/unit-tests/spec/` (StrategySpec + serialization) and `tests/unit-tests/frontend_api/`.

## Testing Boundaries

- Keep a strict separation between **unit** and **integration** tests:
  - Unit tests: isolated logic, synthetic fixtures/mocks allowed.
  - Integration tests: real end-to-end pipeline behavior with repository data/cache.
- Never place synthetic/mock-heavy tests under `tests/integration/`.
- Integration tests must use persisted data from repo-backed sources (for example `data/ohlc_data`) and real pipeline entrypoints (`extract_features_for_bias_node`, `BaseModel`, validators, etc.).
- For integration tests, default to cache-backed execution (`USE_CACHE=True`); if cache does not exist, populate through `bootstrap_source_candles(...)` + `CacheManager.ensure_bias_cache_coverage(...)` (or `extract_features_for_bias_node(..., populate_on_miss=True)`) or skip with an explicit message.
- Integration test data config (tickers, date range, bias spec, cache policy) must be explicit in the test and driven by user requirements for that task.

## Git Workflow

**IMPORTANT**: When merging branches into main, ALWAYS use squash merge to maintain a clean commit history:

```bash
# Squash and merge workflow
git checkout main
git pull origin main
git merge --squash <branch-name>
git commit -m "feat: descriptive summary of changes"
git push origin main
```

This keeps the main branch history linear and readable, with each merge representing a complete feature or fix.

## Architecture

### Pipeline (data flows left to right)

The pipeline has two levels: per-timeframe stacks and a cross-timeframe global combine.

```
Per TF:  Candles (OHLCV) → Bias Nodes → Base Models → DiversifiedEnsemble → TFPortfolio (D / W / M)
         (nodes/)          (features/models/)         (ensemble/)            (ensemble/portfolio_impl/)

Global:  TFPortfolio forecast streams → WeightLayer (cross-TF) → GlobalPortfolio → PositionSizer
                                        (ensemble/weight_layer.py)          (ensemble/)        (execution/)
```

Non-daily forecasts are forward-filled to a daily grid before `WeightLayer` runs. `WeightLayer` is configured on `GlobalPortfolio` (default `equal_signal`).

**Bias Nodes** (`nodes/`): 50+ technical indicators (RSI, ATR, EWMAC, etc.). Each produces one feature column per instrument. Naming convention: `{module}_{feature}_{timeframe}_{param}_{value}` (e.g., `rsi_signal_D_lookback_14`).

**Base Models** (`features/models/`): The alpha unit per timeframe. The legacy binning ABCs (`BinningModelBase`/`ContinuousBinningModel`/`RuleBasedModel`) are retired/deleted; the active path is `create_base_model_from_config` (in `ensemble/ensemble_utils.py`), which builds thin `BaseModel` instances backed by native `signed_signal` node specs.

**DiversifiedEnsemble** (`ensemble/diversified_ensemble.py`): Owns base models, generates per-model volatility-scaled forecasts. Formula: `F = τ / σ`, capped at `2.0` (the all-in/all-out signal means `h_i = 1`, so the `√h_i` term drops out). Configured via JSON control files.

**WeightLayer** (`ensemble/weight_layer.py`): Combines encoded global forecast streams. Exposes 9 weighting methods: `equal_signal`, `inverse_avg_pairwise_corr`, `hierarchy_equal`, `inverse_corr_hierarchy`, `ledoit_wolf_min_corr`, `risk_parity_corr`, `hierarchy_theme_inv_corr`, `hierarchy_theme_ledoit`, `ledoit_wolf_hierarchy_within` (manual hierarchy nested tree in `ensemble/weight_hierarchy.py`). Applies FDM (Forecast Diversification Multiplier) from positive-clipped signal correlation: `FDM = min(√(1 / (mean_corr + 0.01)), fdm_max)` (default `fdm_max = 2.0`). The genuinely-removed legacy methods are `hrp_cluster_equal`, `hrp_classic`, and `optimize_sortino_capped`.

**TFPortfolio** (`ensemble/portfolio.py` / `ensemble/portfolio_impl/tf_portfolio.py`): Per-timeframe portfolio. Applies instrument weights and IDM (Instrument Diversification Multiplier): `IDM = min(√(1 / (mean_corr + 0.01)), 2.5)`. `Portfolio` is a backward-compatible alias for `TFPortfolio`.

**GlobalPortfolio** (`ensemble/portfolio.py` / `ensemble/portfolio_impl/global_portfolio_impl.py`): Owns multiple `TFPortfolio` instances and the cross-timeframe `WeightLayer`. Orchestrates fit/predict across timeframes and produces the final `['ticker', 'datetime', 'forecast_score', 'position_fraction']` output.

**PositionSizer** (`execution/position_sizer.py`): Converts position fractions to contract quantities: `contracts = (position_fraction × capital) / (price × multiplier × fx_rate)`.

### Key Data Models

- `lib/core/models.py`: `Candle` (Pydantic model with OHLCV + ticker + timeframe)
- `lib/core/enums.py`: `TimeFrame` (D/W/M), `Ticker` (ES, NQ, CL, GC, etc.), `Bias`, `Direction`, `PositionMode`

### Supporting Systems

- **Cache** (`cache/`, `cache/runtime/cache_manager.py`): Stores computed bias node outputs per node type
- **Vault:** default prop tree `vault/`, personal `vault_personal/` (env overrides in `lib/core/vault_paths.py`; see `docs/library/Vault/vault.md`). Validated feature storage by timeframe under `<vault_root>/D|W|M/`. Working ensembles use a **nested** layout `<vault_root>/<TF>/<weight_hierarchy_group>/<ensemble_leaf>/` (the 13 `VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES`: `mean_reversion_indices`, `buy_hold`, `es_tlt`, `seasonal`, `momentum`, `trend_following`, `momentum_gc`, `crude_oil_mr`, `gc_breakout`, `cl_breakout`, `breakout`, `silver_mr`, `silver_trend`) so on-disk folders match the manual global weight hierarchy; feature JSONs carry `weight_hierarchy_group`. Legacy flat `<vault_root>/<TF>/<ensemble_leaf>/` is still supported for discovery and cache preflight.
- **Deployment** (`deployment/`): REST forecast server, production training pipeline, MT5 connector, live runtime (`deployment/live/`), and ops/scheduler scripts (`deployment/ops/`, was `deploy/`). The Telegram notifier now lives in `lib/core/notify.py`.
- **Live Trading**: Interactive Brokers integration via `scripts/enigma_live_forecast.py`

### Package layering (installable, acyclic)

The repo is an installable package (`pyproject.toml`; `pip install -e . --no-deps --config-settings editable_mode=compat`). Top-level packages form a one-way dependency DAG — `research/` and `scripts/` sit on top and import downward; nothing imports them back:

```
lib/core (+ lib/compute) → data_platform → nodes → cache →
features → ensemble/portfolio → execution → analysis → deployment → research / scripts
```

`lib/core` is the pure foundation (imports nothing upward; the one runtime reach, `helpers.load_data`, is a lazy `__getattr__` re-export of `data_platform.loaders`). Notable homes after the reorg: `cache/` (was `lib/cache`), `analysis/` (metrics + plotting, was `lib/metrics` / `lib/plotting`), `lib/core/notify.py` (`TelegramNotifier`), `lib/core/runtime_bootstrap.py` (`.env` / UTF-8 bootstrap — library code uses this, never `scripts._bootstrap`), `nodes/base.py` (the `BiasNode` ABC; `nodes/__init__.py` is a thin re-export), `research/workspace/` (was `tools/research_workspace`). The `utils/` and `tools/` grab-bags were dissolved.

### Control Files (JSON)

Ensembles are configured/persisted via JSON control files containing metadata, base model configs, fitted state, and ensemble weights. The `is_fit` flag tracks whether the ensemble has been trained.

## NautilusTrader Reference

Local docs mirror at `docs/nautilustrader/` — fetched from the GitHub source repo (raw Markdown, not the rendered site). Refresh with:

```powershell
.\.venv\Scripts\python.exe scripts\scrape_nautilus_docs.py --force
```

Key pages for the architecture refactor (read these before writing Nautilus wrappers):

| File | What it covers |
|---|---|
| `concepts/architecture.md` | NautilusKernel, MessageBus, threading model, environment contexts |
| `concepts/strategies.md` | Strategy ABC, lifecycle hooks, signal generation |
| `concepts/actors.md` | Actor pattern (base of Strategy), subscriptions, handlers |
| `concepts/data.md` | Data pipeline, subscriptions, bar/quote/trade types |
| `concepts/execution.md` | Order lifecycle, execution engine, routing |
| `concepts/orders/index.md` | Order types overview |
| `concepts/cache.md` | In-memory cache API (instruments, orders, positions) |
| `concepts/message_bus.md` | Pub/Sub, Req/Rep, custom topics |
| `concepts/backtesting.md` | BacktestEngine vs BacktestNode, data loading |
| `concepts/live.md` | TradingNode, live adapter lifecycle |
| `concepts/configuration.md` | Config system, environment variables |
| `concepts/continuous_futures.md` | Continuous contract roll logic |
| `concepts/portfolio.md` | Portfolio component, P&L tracking |
| `concepts/positions.md` | Position model, netting vs hedging |
| `integrations/ib.md` | Interactive Brokers adapter (our current live broker) |
| `getting_started/installation.md` | Install + quickstart |

## Live Trading Runtime (broker clock — the host clock is NOT trustworthy)

The live vault runtime (`deployment/live/`) trades the vault on MT5 demo accounts, **aligned to
the financing rollover** (not a fixed cash-close clock). Operator entrypoint + dashboard:

```powershell
# Arm a REAL-demo-order node (per broker; --live required for demo/live tiers):
.\.venv\Scripts\python.exe -m deployment.live.run_vault_sandbox --broker ftmo     --exec-tier demo --arm --live
.\.venv\Scripts\python.exe -m deployment.live.run_vault_sandbox --broker darwinex --exec-tier demo --arm --live --skip-refresh
# (omit --exec-tier/--live for the zero-broker-risk sandbox tier; --dry-build validates wiring offline)

# Live monitor dashboard (FastAPI serves the built web dist at / and the JSON API at /api/live/*):
.\.venv\Scripts\python.exe -m uvicorn frontend.api.server:app --host 127.0.0.1 --port 5057   # → http://127.0.0.1:5057/live
```

One node per broker, each bound to its own terminal via `MT5Config.path` (`brokers.terminal_path`).
Signals are broker-agnostic: every node reads ONE shared Darwinex signal cache
(`data/broker_cache/darwinex/`); execution (quotes, orders, sizing) stays per-broker.

**The development machine's system clock is unreliable (observed ~7h fast) — NEVER use host time
for any trading-time reasoning. Always derive "now" from a fresh MT5 tick:**

- **Broker wall-clock** = `deployment/live/runtime/rollover_market.broker_now()` — reads `EURUSD`
  `tick.time` (broker EET/EEST encoded as a Unix epoch) and returns `epoch + tick.time` with NO tz
  conversion. This is the schedule clock for the rollover EXIT/ENTRY decisions.
- The **rollover schedule** (`runtime/rollover_schedule.py`) fires entirely off `broker_now()`:
  EXIT window `[00:00 − exit_lead_min, 00:00)` broker (default T-15 = 23:45 broker), ENTRY at each
  symbol's reopen (most CFDs 01:00 broker) + per-leg settle. So exit/flat/reenter timing is immune
  to the host clock. The 00:00–01:00 broker dead-zone is held flat (no ticks).
- Quote-freshness is also host-clock-immune: the vendored MT5 data client
  (`deployment/nautilus_mt5/.../mt5connect/data.py::_compute_utc_offset`) measures a one-time
  broker→UTC offset at connect (`round(host_utc − tick.time)`, snapped to ½h) and bakes it into
  every tick's `ts_event`, so `quote_age = host_now − ts_event` stays correct however wrong the host
  clock is. (Look for the log line `MT5DataClient: broker→UTC tick offset = …`.)
- Residual host-clock effects are **cosmetic only**: equity-curve x-axis labels and the daily
  risk-gauge date anchor (`vault_strategy._broker_date()` converts host time → can land on the wrong
  broker date).

**Quick broker-time check (read-only; attaches to the already-running terminal, sends no orders):**

```powershell
.\.venv\Scripts\python.exe -c "from lib.core.runtime_bootstrap import bootstrap_runtime; bootstrap_runtime(); import MetaTrader5 as mt5; from data_platform.providers.mt5 import brokers; from deployment.live.runtime import rollover_market; mt5.initialize(path=str(brokers.terminal_path('ftmo'))); print('broker now =', rollover_market.broker_now()); mt5.shutdown()"
```

Conversions for humans: **ET = broker − 7h** (EEST↔EDT); broker **00:00 rollover = 17:00 ET**,
T-15 EXIT = 23:45 broker = **16:45 ET**, reopen ENTRY ≈ 01:00 broker = **18:00 ET**.

## CodeGraph — use this first for all code exploration

This repo is indexed by **CodeGraph** (MCP server: `codegraph`). It provides a pre-built
knowledge graph of every symbol, call edge, and file — queries are sub-millisecond and
return verbatim source, so one `codegraph_explore` call replaces dozens of Grep/Glob/Read
round-trips.

**Rules:**

1. **Always call `codegraph_explore` first** for any question about how code works, where
   something is defined, what calls what, or what a symbol does. Do NOT start with Grep,
   Glob, or Read for exploration tasks.

2. **`codegraph_explore` returns verbatim source** — treat each file block it returns as
   an already-performed Read. Do NOT re-read those files with the Read tool.

3. **Only fall back to Grep/Read** for a specific line range that codegraph didn't surface,
   or to confirm a detail not covered by the response.

4. **Use `codegraph_search`** when you know a symbol name but not its file — it returns
   locations instantly without reading any code.

5. **Use `codegraph_callers` / `codegraph_callees` / `codegraph_impact`** before editing
   anything — know the blast radius first.

**Tool selection cheat-sheet:**

| Intent | Tool |
|--------|------|
| How does X work / what is X / where is X | `codegraph_explore` (PRIMARY) |
| Find a symbol by name (location only) | `codegraph_search` |
| What calls this function? | `codegraph_callers` |
| What does this function call? | `codegraph_callees` |
| What would break if I change X? | `codegraph_impact` |
| Single specific line range not in explore result | `Read` (fallback only) |
| Grep for a pattern codegraph can't match | `Grep` (fallback only) |

**Current index stats** (2026-06-04): 818 files · 12,117 nodes · 25,510 edges · Python 813 files.
The file watcher auto-syncs on save; no manual reindex needed.

## Coding Conventions

Defined in `.cursor/rules/` (001-004):

- **Functional core, imperative shell**: Pure functions for logic, I/O at edges. Separate data structures from behavior.
- **Immutability**: `@dataclass(frozen=True)` by default. Use `NewType` for domain primitives.
- **No raw loops**: Prefer comprehensions, generators, `map`/`filter`/`reduce`, `itertools`.
- **Composition over inheritance**: Use `typing.Protocol` for interfaces, dependency injection for strategies.
- **100% type hints**: No `Any`. Use `TypeVar`/`ParamSpec` for generics. Code should pass strict `mypy`/`pyright`.
- **No booleans in public APIs**: Use `Enum` instead.
- **SRP**: Functions either do work or coordinate work, never both. If the name has "And", split it.
- **Monadic error handling**: `Result[T, E]` pattern over exceptions for expected failures.
- **Decorators** must use `@functools.wraps` and preserve type hints with `ParamSpec`.

## graphify

This project has a knowledge graph at `graphify-out/` — **12,816 nodes · 37,958 edges · 443 communities** built from 925 Python/code files (commit `c3bd8a4a`).

**Current limitations:**
- Docs (`docs/`) are excluded from the graph (`.graphifyignore` skips them — they require an LLM API key to index).
- Community names are generic placeholders ("Community 0", …). Fix with: `$env:ANTHROPIC_API_KEY="sk-..."; graphify label .`
- No `graph.html` — 12,816 nodes exceeds the browser viz limit; use CLI queries instead.

**Rules:**
- For codebase questions, first run `graphify query "<question>"` when `graphify-out/graph.json` exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If `graphify-out/wiki/index.md` exists, use it for broad navigation instead of raw source browsing.
- Read `graphify-out/GRAPH_REPORT.md` only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

**Example queries (Windows PowerShell — no leading slash):**
```powershell
graphify query "how does GlobalPortfolio orchestrate TFPortfolios?"
graphify explain "WeightLayer"
graphify path "DiversifiedEnsemble" "PositionSizer"
```
