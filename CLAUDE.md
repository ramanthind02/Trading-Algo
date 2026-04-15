# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
python utils/compute/cython/setup_cython.py build_ext --inplace
```

## Commands (Windows PowerShell, repo root)

Prefer the venv interpreter explicitly; `python -m pytest` avoids needing `pytest` on PATH.

```powershell
# Example: one file
.\.venv\Scripts\python.exe -m pytest tests\unit-tests\feature_research\test_permutation_pipeline.py -v

# In-sample research (same as: python -m feature_research.in_sample.run_is)
.\.venv\Scripts\python.exe -m feature_research.in_sample.run_is

# Feature–vault correlation CSV (runs feature_research OOS once; enable FeatureVaultCorrelationConfig in portfolio_research.config.load_config)
.\.venv\Scripts\python.exe -m portfolio_research.run_feature_vault_correlation
```

If your venv directory is named `venv` instead of `.venv`, use `.\venv\Scripts\python.exe` in place of `.\.venv\Scripts\python.exe`.

## Visualization Policy

- Prefer tabular or JSON/CSV research outputs over in-repo plotting.
- Keep QuantStats tearsheets, prop-firm HTML reports, and Norgate migration QA plots unless the task explicitly says otherwise.

## Testing Boundaries

- Keep a strict separation between **unit** and **integration** tests:
  - Unit tests: isolated logic, synthetic fixtures/mocks allowed.
  - Integration tests: real end-to-end pipeline behavior with repository data/cache.
- Never place synthetic/mock-heavy tests under `tests/integration/`.
- Integration tests must use persisted data from repo-backed sources (for example `data/ohlc_data`) and real pipeline entrypoints (`extract_features_for_bias_node`, `BaseModel`, validators, etc.).
- For integration tests, default to cache-backed execution (`USE_CACHE=True`); if cache does not exist, populate through `CacheManager.populate_cache(...)` or skip with an explicit message.
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

The pipeline has two levels: a per-timeframe level and a cross-timeframe global level.

```
                                                                     ┌─ forecast_score_D ─┐
Candles (OHLCV) → Bias Nodes → Base Models → DiversifiedEnsemble → WeightLayer → TFPortfolio (D) ──┤
                  (nodes/)      (feature_      (ensemble/)           (ensemble/)   (ensemble/)       ├→ GlobalWeightLayer → GlobalPortfolio → PositionSizer
                                selection/)                                                           │  (cross-TF HRP+FDM)  (ensemble/)       (execution/)
                                                                     TFPortfolio (W) ── forecast_W ──┤
                                                                     TFPortfolio (M) ── forecast_M ──┘
```

**Bias Nodes** (`nodes/`): 50+ technical indicators (RSI, ATR, EWMAC, etc.). Each produces one feature column per instrument. Naming convention: `{module}_{feature}_{timeframe}_{param}_{value}` (e.g., `rsi_signal_D_lookback_14`).

**Base Models** (`feature_selection/base_models/`): Transform continuous features into binary signals (0/1) via binning strategies (quantile, decision tree, rule-based). ABC is `BinningModelBase` in `base_model.py`.

**DiversifiedEnsemble** (`ensemble/diversified_ensemble.py`): Owns base models, generates per-model volatility-scaled forecasts. Formula: `F_i = (τ / (σ × √h_i)) × X_i`. Configured via JSON control files.

**WeightLayer** (`ensemble/weight_layer.py`): Combines forecasts using inverse-correlation weights and applies FDM (Forecast Diversification Multiplier): `FDM = min(√(1 / (mean_corr + 0.01)), 2.0)`.

**TFPortfolio** (`ensemble/portfolio.py`): Per-timeframe portfolio. Applies instrument weights and IDM (Instrument Diversification Multiplier): `IDM = min(√(1 / (mean_corr + 0.01)), 2.5)`. `Portfolio` is a backward-compatible alias for `TFPortfolio`.

**GlobalWeightLayer** (`ensemble/global_weight_layer.py`): Combines per-timeframe `forecast_score` streams using downside HRP across timeframes, applies a cross-TF FDM (same formula, cap 2.0). Non-daily TF streams are forward-filled to a daily grid before combination. Single-TF fallback: weight=1.0, FDM=1.0.

**GlobalPortfolio** (`ensemble/portfolio.py`): Top-level class owning multiple `TFPortfolio` instances and one `GlobalWeightLayer`. Orchestrates fit/predict across all timeframes and produces the final `['ticker', 'datetime', 'forecast_score', 'position_fraction']` output.

**PositionSizer** (`execution/position_sizer.py`): Converts position fractions to contract quantities: `contracts = (position_fraction × capital) / (price × multiplier × fx_rate)`.

### Key Data Models

- `utils/models.py`: `Candle` (Pydantic model with OHLCV + ticker + timeframe)
- `utils/enums.py`: `TimeFrame` (D/W/M), `Ticker` (ES, NQ, CL, GC, etc.), `Bias`, `Direction`, `PositionMode`

### Supporting Systems

- **Cache** (`cache/`, `utils/cache_manager.py`): Stores computed bias node outputs per node type
- **Vault** (`vault/`): Validated feature storage organized by timeframe and ensemble direction, contains feature control files (JSON)
- **Deployment** (`deployment/`): REST forecast server, production training pipeline, Telegram notifier, MT5 connector
- **Live Trading**: Interactive Brokers integration via `scripts/tws_live_forecast.py`

### Control Files (JSON)

Ensembles are configured/persisted via JSON control files containing metadata, base model configs, fitted state, and ensemble weights. The `is_fit` flag tracks whether the ensemble has been trained.

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
