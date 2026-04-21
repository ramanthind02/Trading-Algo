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

# Feature–vault correlation CSV (runs feature_research OOS once; enable FeatureVaultCorrelationConfig in portfolio_research.config.load_config; unset vault_root scans prop vault, set Path for vault_personal)
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

The pipeline has two levels: per-timeframe stacks and a cross-timeframe global combine.

```
Per TF:  Candles (OHLCV) → Bias Nodes → Base Models → DiversifiedEnsemble → TFPortfolio (D / W / M)
         (nodes/)          (feature_selection/)      (ensemble/)            (ensemble/portfolio_impl/)

Global:  TFPortfolio forecast streams → WeightLayer (cross-TF) → GlobalPortfolio → PositionSizer
                                        (ensemble/weight_layer.py)          (ensemble/)        (execution/)
```

Non-daily forecasts are forward-filled to a daily grid before `WeightLayer` runs. `WeightLayer` is configured on `GlobalPortfolio` (default `equal_signal`).

**Bias Nodes** (`nodes/`): 50+ technical indicators (RSI, ATR, EWMAC, etc.). Each produces one feature column per instrument. Naming convention: `{module}_{feature}_{timeframe}_{param}_{value}` (e.g., `rsi_signal_D_lookback_14`).

**Base Models** (`feature_selection/base_models/`): Transform continuous features into binary signals (0/1) via binning strategies (quantile, decision tree, rule-based). ABC is `BinningModelBase` in `base_model.py`.

**DiversifiedEnsemble** (`ensemble/diversified_ensemble.py`): Owns base models, generates per-model volatility-scaled forecasts. Formula: `F_i = (τ / (σ × √h_i)) × X_i`. Configured via JSON control files.

**WeightLayer** (`ensemble/weight_layer.py`): Combines encoded global forecast streams. Modes include `equal_signal`, `inverse_avg_pairwise_corr`, and manual `hierarchy_equal` (nested tree in `ensemble/weight_hierarchy.py`). Applies FDM (Forecast Diversification Multiplier) from positive-clipped signal correlation: `FDM = min(√(1 / (mean_corr + 0.01)), fdm_max)` (default `fdm_max = 2.0`). Legacy HRP-based weighting methods are no longer valid config.

**TFPortfolio** (`ensemble/portfolio.py` / `ensemble/portfolio_impl/tf_portfolio.py`): Per-timeframe portfolio. Applies instrument weights and IDM (Instrument Diversification Multiplier): `IDM = min(√(1 / (mean_corr + 0.01)), 2.5)`. `Portfolio` is a backward-compatible alias for `TFPortfolio`.

**GlobalPortfolio** (`ensemble/portfolio.py` / `ensemble/portfolio_impl/global_portfolio_impl.py`): Owns multiple `TFPortfolio` instances and the cross-timeframe `WeightLayer`. Orchestrates fit/predict across timeframes and produces the final `['ticker', 'datetime', 'forecast_score', 'position_fraction']` output.

**PositionSizer** (`execution/position_sizer.py`): Converts position fractions to contract quantities: `contracts = (position_fraction × capital) / (price × multiplier × fx_rate)`.

### Key Data Models

- `utils/models.py`: `Candle` (Pydantic model with OHLCV + ticker + timeframe)
- `utils/enums.py`: `TimeFrame` (D/W/M), `Ticker` (ES, NQ, CL, GC, etc.), `Bias`, `Direction`, `PositionMode`

### Supporting Systems

- **Cache** (`cache/`, `utils/cache_manager.py`): Stores computed bias node outputs per node type
- **Vault:** default prop tree `vault/`, personal `vault_personal/` (env overrides in `utils/vault_paths.py`; see `docs/library/Vault/vault.md`). Validated feature storage by timeframe under `<vault_root>/D|W|M/`. Working ensembles use a **nested** layout `<vault_root>/<TF>/<weight_hierarchy_group>/<ensemble_leaf>/` (groups: `mean_reversion_indices`, `buy_hold`, `es_tlt`, `seasonal`, `momentum`) so on-disk folders match the manual global weight hierarchy; feature JSONs carry `weight_hierarchy_group`. Legacy flat `<vault_root>/<TF>/<ensemble_leaf>/` is still supported for discovery and cache preflight.
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
