# Cython & Portfolio Backtest

> ⚠️ Slated for rewrite under the NautilusTrader migration (WP-4 live execution). See docs/refactor/nautilus/.

> [!summary] Overview
> Cython extensions accelerate inner math kernels for feature extraction and base-model fitting.
> The portfolio backtest API uses `Portfolio` + `PortfolioTester`: fit → predict → strategy returns.
> Cython is **optional** — pure-Python fallbacks exist for all operations.

---

## Building Cython Extensions

```bash
source /home/raman/repos/Trading-Algo/venv/bin/activate
python lib/compute/cython/setup_cython.py build_ext --inplace
```

Compiles two modules:

| Source | Module | Used by |
|---|---|---|
| `lib/compute/cython/cython_optimized.pyx` | `lib.compute.cython.cython_optimized` | `lib.compute.fast_stats` (Spearman, rank, threshold opt, MA-diff) |
| `lib/compute/cython/cython_nodes.pyx` | `lib.compute.cython.cython_nodes` | `lib.compute.fast_nodes` (ATR, EMA, RSI, ROC, Donchian, etc.) |

> [!note] Portfolio volatility uses `lib.compute.fast_volatility` (pure NumPy EWSD). No Cython required for that path.

---

## Portfolio Backtest API

### Minimal Flow: fit → predict → returns

```python
from ensemble.portfolio import Portfolio
from ensemble.portfolio_impl.portfolio_tester import PortfolioTester
from lib.core.enums import TimeFrame

portfolio = Portfolio(
    ensembles=ensembles,          # list of DiversifiedEnsemble or vault-loaded ensembles
    trading_timeframe=TimeFrame.D,
    max_position_pct=2.0,
)
tester = PortfolioTester(portfolio=portfolio, baseline_mode="equal_weight")

tester.fit(candles_df)                          # fits portfolio + all ensembles; target = log returns
tester.predict(candles_df)                      # returns positions: ticker, datetime, position_fraction
tester.calculate_strategy_returns(candles_df)   # strategy P&L from positions
tester.calculate_baseline_returns(candles_df)   # equal-weight baseline
```

Candles DataFrame columns: `datetime`, `open`, `high`, `low`, `close`, `volume`, `ticker`, `timeframe`.

### Benchmark Script (Real Data)

```bash
python scripts/benchmark_portfolio_backtest.py [--warmup 1] [--runs 2]
```

Uses the same config as `research/portfolio/run_portfolio_test.py`: tickers, ensemble list, and train/test dates come from `research.portfolio.config.load_config()`.

### Profiling Script (Synthetic Data)

```bash
python scripts/profile_portfolio_pipeline.py
```

Uses mock ensembles and synthetic candles. Writes `profile_portfolio_pipeline.prof`.

---

## What Is Vectorized / Optimized

- **Volatility** — `compute_ewsd_annualized_from_closes` in `lib.compute.fast_volatility` (array-based, no per-bar loops)
- **Risk management** — `_apply_risk_management_to_forecasts`: vectorized pandas merge/clip/instrument weights
- **Returns calculation** — `calculate_strategy_returns_from_positions`, baseline returns: vectorized groupby/merge/diff
- **Cython kernels** — RSI, EMA, ATR, MA-diff, Spearman rank (when built)

---

## Benchmark Results (With vs Without Cython)

8 vault ensembles, train 2000–2020, test 2020–2024, ES/NQ/YM/RTY:

| Phase | Without Cython | With Cython |
|---|---|---|
| Load ensembles | 0.11 s | 0.11 s |
| Load data | 0.06 s | 0.06 s |
| **Fit** | **71.11 s** | **71.27 s** |
| **Predict (basic)** | **6.11 s** | **6.15 s** |
| Predict (granular) | 0.00 s (cache) | 0.00 s (cache) |
| Strategy returns | 0.02 s | 0.02 s |
| **Core pipeline** | **~77.25 s** | **~77.45 s** |

> [!warning] Cython provides no measurable speedup in the current design. See below.

---

## The Bottleneck: Per-Candle Python Loop

- **~82% of total time** is in `feature_base_model.add_candle()` (~69 s of ~84 s)
- Called **211,000+ times** — once per row of the candles DataFrame

Each call pays for:
1. `Candle` object creation — Pydantic validation + `uuid4` + `from_row_fast`
2. `BiasNode.add_candle(candle)` → `_compute_candle(candle)` dispatch
3. Appending to Python `dict`/`list` (`_feature_values`, `_feature_datetimes`)

> [!important] Cython speeds the small math kernel inside `_compute_candle`, but **cannot remove 211k Python function calls or 211k Candle constructions**.

Additional costs:
- Some nodes have no Cython kernel (UltimateC, BuyHold, RSIRegime) — pure Python inner loops
- Pandas/datetime indexing per candle (`to_datetime`, `get_loc`)

---

## Path to Major Speedup (Not Yet Implemented)

A **vectorized node API** is required:
- Pass full OHLC arrays into each node → get back a complete feature array
- One (or few) Cython calls per series instead of 211k per-row calls
- Eliminates per-row Candle construction entirely

Until then, Cython's contribution is within run-to-run noise.

---

## See Also

- [[portfolio]] — Portfolio and PortfolioTester architecture

> _Verified against commit a07b6bf->197221e on 2026-06-04 (docs Phase A; WP-8 restructure repoint)._
