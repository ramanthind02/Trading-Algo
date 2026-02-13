# Cython-Optimized Portfolio Backtest

This guide explains how to build and use the Cython-accelerated pipeline for portfolio backtesting: volatility calculation, ensemble prediction, and strategy-return computation.

## Overview

The portfolio backtest stack uses:

- **Portfolio** – fits ensembles from candles, aggregates forecasts, applies risk management (IDM, FDM, position capping).
- **PortfolioTester** – drives fit → predict → strategy returns and baseline returns from a candles DataFrame.
- **Fast volatility** – array-based EWSD-style volatility (`utils.fast_volatility`) used by `Portfolio._calculate_volatility_from_candles` (no Cython; pure NumPy).
- **Cython extensions** (optional) – used under the hood by base models and feature extraction:
  - `utils.cython_optimized` – stats (Spearman, rank, threshold optimization, MA-diff).
  - `utils.cython_nodes` – node-level kernels (ATR, EMA, RSI, ROC, Donchian, etc.).

Without building Cython, the pipeline runs using pure-Python fallbacks; building Cython improves throughput for feature extraction and base-model fitting.

---

## 1. Building Cython Extensions

From the project root, with the project virtualenv activated:

```bash
source venv/bin/activate
python utils/setup_cython.py build_ext --inplace
```

This compiles:

- `utils/cython_optimized.pyx` → `utils.cython_optimized` (used by `utils.fast_stats`)
- `utils/cython_nodes.pyx` → `utils.cython_nodes` (used by `utils.fast_nodes`)

After a successful build, imports of `utils.fast_stats` and `utils.fast_nodes` will use the Cython implementations when available. The portfolio’s volatility path uses `utils.fast_volatility` (NumPy only) and does not require Cython.

---

## 2. Running the Portfolio Backtest

### 2.1 API: Candles → Fit → Predict → Returns

1. **Candles** – DataFrame with columns: `datetime`, `open`, `high`, `low`, `close`, `volume`, `ticker`, `timeframe`.
2. **Portfolio** – built from a list of ensembles (e.g. `DiversifiedEnsemble` instances or mocks) and options such as `trading_timeframe`, `max_position_pct`.
3. **PortfolioTester** – wraps a `Portfolio` and exposes `fit`, `predict`, `calculate_strategy_returns`, `calculate_baseline_returns`.

Minimal flow:

```python
from ensemble.portfolio import Portfolio
from ensemble.portfolio_tester import PortfolioTester
from utils.enums import TimeFrame

# Build or load ensembles (see vault or mock)
ensembles = [...]  # list of DiversifiedEnsemble or objects with fit_from_candles / predict_from_candles

portfolio = Portfolio(
    ensembles=ensembles,
    trading_timeframe=TimeFrame.D,
    max_position_pct=2.0,
)
tester = PortfolioTester(portfolio=portfolio, baseline_mode="equal_weight")

# Run backtest
tester.fit(candles_df)           # fits portfolio and ensembles from candles; target = log returns
tester.predict(candles_df)      # portfolio positions (ticker, datetime, position_fraction)
tester.calculate_strategy_returns(candles_df)   # strategy returns from positions
tester.calculate_baseline_returns(candles_df)   # baseline (e.g. equal-weight) returns
```

Strategy and baseline returns are then available on the tester (e.g. for tearsheets or metrics).

### 2.2 Using Ensembles from the Vault

To use real ensembles (with control files and base models), load them via the vault manager, then pass the resulting list into `Portfolio` as above. See `docs/vault_user_guide.md` and `ensemble/vault_manager.load_ensemble_from_vault` for vault layout and loading.

### 2.3 Real-Data Benchmark (Stress Test)

To benchmark the full pipeline with real data and vault ensembles (same flow as `research/portfolio_test.ipynb`):

```bash
source venv/bin/activate
python scripts/benchmark_portfolio_backtest.py [--warmup 1] [--runs 2]
```

The script:

- Loads 8 ensembles from vault (buy_hold, rsi_bias, momentum, rsi_regime long/short, cum_rsi, rsi_5, ultimate_c) with `refit=True`.
- Loads train (2000–2020) and test (2020–2024) daily candles for ES, NQ, YM, RTY.
- Runs warmup (optional), then timed runs: each run = new portfolio+tester, then fit → predict (basic) → predict (granular) → strategy returns → baseline returns.
- Prints Cython extension status (fast_stats, fast_nodes) and mean ± std timings for each phase.

Run once without Cython, then build Cython (`python utils/setup_cython.py build_ext --inplace`) and run again to compare fit/predict timings.

#### Benchmark results (with vs without Cython)

Post-refactor (sequential ensembles, no joblib, itertuples in BaseModel). Single-run comparison on the same machine (8 vault ensembles, train 2000–2020, test 2020–2024, ES/NQ/YM/RTY, `--warmup 0 --runs 1`):

| Phase              | Without Cython | With Cython |
|--------------------|----------------|-------------|
| Load ensembles     | 0.11 s         | 0.11 s      |
| Load data          | 0.06 s         | 0.06 s      |
| **Fit**            | **71.11 s**    | **71.27 s** |
| **Predict (basic)**| **6.11 s**     | **6.15 s**  |
| Predict (granular) | 0.00 s (cache) | 0.00 s (cache) |
| Strategy returns   | 0.02 s         | 0.02 s      |
| Baseline returns   | 0.01 s         | 0.01 s      |
| **Core pipeline**  | **~77.25 s**   | **~77.45 s** |

Post-refactor, fit and predict are faster than the previous parallel/joblib setup (~77 s vs ~101–110 s core pipeline). Cython vs pure Python is within run-to-run noise (~0.2 s). For a stable comparison use `--runs 1` and run the script twice (once with Cython .so files renamed away, once with Cython built); avoid `--runs 2` for comparison because the same ensemble instances are reused and the second fit hits ensemble caches.

### 2.4 Dummy Data and Profiling

For profiling or testing without real models, use the synthetic script that builds dummy candles and mock ensembles:

```bash
source venv/bin/activate
python scripts/profile_portfolio_pipeline.py
```

This script:

- Generates synthetic daily candles for a few tickers over a short history.
- Creates mock ensembles that implement `fit_from_candles` and `predict_from_candles` with zero forecasts.
- Runs `PortfolioTester.fit`, `predict`, `calculate_strategy_returns`, and `calculate_baseline_returns`.
- Writes a profile to `profile_portfolio_pipeline.prof` and prints a cumulative-time summary (e.g. top 30 functions).

Use this to inspect hotspots in the portfolio and tester (volatility, risk management, returns) without depending on vault or feature extraction.

---

## 3. What Is Optimized

- **Portfolio volatility** – `compute_ewsd_annualized_from_closes` in `utils.fast_volatility` (array-based, no per-bar Python loops).
- **Portfolio risk and weighting** – vectorized pandas in `_apply_risk_management_to_forecasts` (merge, clip, instrument weights).
- **PortfolioTester returns** – `calculate_log_returns_from_candles`, `calculate_strategy_returns_from_positions`, and `calculate_baseline_returns` use vectorized groupby/merge/diff.
- **Base models and nodes** – when Cython is built, `utils.fast_stats` and `utils.fast_nodes` use Cython kernels for feature and model internals.

---

## 4. Why is it still slow? Why doesn’t Cython help more?

Profiling the real-data benchmark (e.g. `python scripts/benchmark_portfolio_backtest.py --profile`) shows:

- **~82% of time** is spent in `feature_base_model.add_candle` (~69 s of ~84 s for one fit + predict). That function is called **once per row** (211k+ times) and each call:
  - Builds or uses a `Candle` (Pydantic validation, `uuid4`, `from_row_fast`).
  - Dispatches to `bias_nodes[(ticker, tf)].add_candle(candle)` → `_compute_candle(candle)`.
  - Stores the result in Python dicts/lists (`_feature_values`, `_feature_datetimes`).

So the bottleneck is **per-candle Python work**, not the inner math:

1. **Per-candle Python overhead** – 211k Python function calls (BaseModel → BiasNode.add_candle → _compute_candle). Even when `_compute_candle` uses a Cython helper (e.g. RSI, EMA), every candle still pays for Python call overhead, cache key handling, and `Candle` attribute access.
2. **Vault ensembles use mixed nodes** – Some nodes have Cython kernels (RSI, EMA, ATR, MA-diff, etc.), but others are **pure Python** (e.g. **UltimateC**, **BuyHold**, and much of **RSIRegime**). The profile shows non-trivial time in `ultimate_c._compute_candle` and in Pandas/datetime (indexing, `to_datetime`, `get_loc`).
3. **Candle creation** – 211k Candles mean 211k × (Pydantic validation, `uuid4`, `from_row_fast`), which shows up as several seconds in the profile.
4. **No vectorized node API** – The design is “one row → one `add_candle` call”. Cython can only speed up the small kernel inside `_compute_candle`; it cannot remove the 211k Python calls or the 211k Candle constructions.

So **Cython does speed up the kernels** (RSI, EMA, ATR, etc.), but the **dominant cost is the per-row Python loop and object creation**. To get a large additional speedup you’d need:

- A **vectorized node API**: pass OHLC (and other) arrays into the node and get back a full feature array (one or few Cython calls per series), avoiding 211k Python calls and 211k Candles.
- **More nodes implemented in Cython** (e.g. UltimateC’s inner loop), and/or moving to array-based processing so Candle objects are not created per row.

---

