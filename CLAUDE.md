# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Systematic trading framework implementing Robert Carver's methodology. Generates trading signals from technical indicators (bias nodes), combines them through ensemble learning with diversification multipliers, and converts signals to tradeable positions.

## Commands

```bash
# Activate venv (required for all Python operations)
source venv/bin/activate

# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_integration.py -v

# Run a specific test class or method
pytest tests/test_integration.py::TestFormulaVerification -v

# Compile Cython extensions (optional, for performance)
python utils/setup_cython.py build_ext --inplace
```

## Architecture

### Pipeline (data flows left to right)

```
Candles (OHLCV) → Bias Nodes → Base Models → DiversifiedEnsemble → WeightLayer → Portfolio → PositionSizer
                  (nodes/)      (feature_      (ensemble/)           (ensemble/)   (ensemble/) (execution/)
                                selection/)
```

**Bias Nodes** (`nodes/`): 50+ technical indicators (RSI, ATR, EWMAC, etc.). Each produces one feature column per instrument. Naming convention: `{module}_{feature}_{timeframe}_{param}_{value}` (e.g., `rsi_signal_D_lookback_14`).

**Base Models** (`feature_selection/base_models/`): Transform continuous features into binary signals (0/1) via binning strategies (quantile, decision tree, rule-based). ABC is `BinningModelBase` in `base_model.py`.

**DiversifiedEnsemble** (`ensemble/diversified_ensemble.py`): Owns base models, generates per-model volatility-scaled forecasts. Formula: `F_i = (τ / (σ × √h_i)) × X_i`. Configured via JSON control files.

**WeightLayer** (`ensemble/weight_layer.py`): Combines forecasts using inverse-correlation weights and applies FDM (Forecast Diversification Multiplier): `FDM = min(√(1 / (mean_corr + 0.01)), 2.0)`.

**Portfolio** (`ensemble/portfolio.py`): Applies instrument weights and IDM (Instrument Diversification Multiplier): `IDM = min(√(1 / (mean_corr + 0.01)), 2.5)`.

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
