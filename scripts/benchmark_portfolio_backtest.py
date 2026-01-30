#!/usr/bin/env python3
"""
Benchmark the Cython-optimized portfolio backtest with real data and vault ensembles.

Runs the same flow as research/portfolio_test.ipynb:
- Load 8 ensembles from vault (refit=True)
- Load train (2000–2020) and test (2020–2024) candles for ES, NQ, YM, RTY
- Fit portfolio on train, predict (basic + granular) and compute returns on test

Reports Cython availability and timings for load, fit, predict, and returns.

Usage:
    source venv/bin/activate
    python scripts/benchmark_portfolio_backtest.py [--warmup 1] [--runs 2]
    python scripts/benchmark_portfolio_backtest.py --profile   # run once with cProfile, print hotspots
"""

import argparse
import cProfile
import os
import pstats
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ensemble.portfolio import Portfolio  # noqa: E402
from ensemble.portfolio_tester import PortfolioTester  # noqa: E402
from ensemble.vault_manager import load_ensemble_from_vault  # noqa: E402
from utils.enums import TimeFrame, Ticker  # noqa: E402
from utils import helpers  # noqa: E402


def _cython_status() -> dict[str, bool]:
    """Report whether Cython extensions are available (stats + nodes)."""
    stats_available = False
    nodes_available = False
    try:
        import utils.fast_stats as fs  # noqa: F401
        stats_available = getattr(fs, "CYTHON_AVAILABLE", False)
    except Exception:
        pass
    try:
        import utils.fast_nodes as fn  # noqa: F401
        nodes_available = getattr(fn, "CYTHON_NODES_AVAILABLE", False)
    except Exception:
        pass
    return {"fast_stats (cython_optimized)": stats_available, "fast_nodes (cython_nodes)": nodes_available}


def _load_ensembles(target_vol: float) -> list:
    """Load the same 8 vault ensembles as in portfolio_test.ipynb."""
    dirs = [
        "vault/D/buy_hold_long",
        "vault/D/rsi_bias_lookback_2_long",
        "vault/D/indices_momentum_long",
        "vault/D/rsi_regime_long",
        "vault/D/rsi_regime_short",
        "vault/D/cum_rsi_2_long",
        "vault/D/rsi_5_long",
        "vault/D/ultimate_c_2_3_3 _long",
    ]
    return [
        load_ensemble_from_vault(d, refit=True, target_volatility=target_vol)
        for d in dirs
    ]


def _load_data() -> tuple:
    """Load train (2000–2020) and test (2020–2024) candles for ES, NQ, YM, RTY."""
    tickers = [Ticker.ES, Ticker.NQ, Ticker.YM, Ticker.RTY]
    train_candles = helpers.load_data_multi_ticker(
        tickers=tickers,
        timeframe=TimeFrame.D,
        start=datetime(2000, 1, 1),
        end=datetime(2020, 1, 1),
        use_millisecond_offset=True,
    )
    test_candles = helpers.load_data_multi_ticker(
        tickers=tickers,
        timeframe=TimeFrame.D,
        start=datetime(2020, 1, 1),
        end=datetime(2024, 1, 1),
        use_millisecond_offset=True,
    )
    return train_candles, test_candles


def run_benchmark(warmup: int = 1, runs: int = 2) -> None:
    """Run full pipeline and report timings."""
    target_vol = 0.15

    # Cython status
    cython = _cython_status()
    print("Cython extensions:")
    for name, available in cython.items():
        print(f"  {name}: {'✓' if available else '✗ (pure Python)'}")
    print()

    # Load ensembles
    t0 = time.perf_counter()
    ensembles = _load_ensembles(target_vol)
    t_load_ensembles = time.perf_counter() - t0
    print(f"Loaded {len(ensembles)} ensembles in {t_load_ensembles:.2f}s")

    # Load data
    t0 = time.perf_counter()
    train_candles, test_candles = _load_data()
    t_load_data = time.perf_counter() - t0
    print(f"Loaded train: {len(train_candles)} candles, test: {len(test_candles)} candles in {t_load_data:.2f}s")
    print()

    # Build portfolio and tester
    portfolio = Portfolio(
        ensembles=ensembles,
        trading_timeframe=TimeFrame.D,
        target_volatility=target_vol,
        max_position_pct=2.5,
    )
    tester = PortfolioTester(portfolio=portfolio, baseline_mode="equal_weight")

    # Warmup: one full fit + predict to clear one-off overhead (e.g. imports)
    print("Warmup...")
    for _ in range(warmup):
        _p = Portfolio(
            ensembles=ensembles,
            trading_timeframe=TimeFrame.D,
            target_volatility=target_vol,
            max_position_pct=2.5,
        )
        _t = PortfolioTester(portfolio=_p, baseline_mode="equal_weight")
        _t.fit(train_candles)
        _t.predict(train_candles)
    print("Warmup done.\n")

    # Timed runs: each run = new portfolio+tester, then time fit -> predict -> returns
    fit_times: list[float] = []
    predict_times: list[float] = []
    predict_granular_times: list[float] = []
    strategy_returns_times: list[float] = []
    baseline_returns_times: list[float] = []

    for _ in range(runs):
        # New portfolio/tester per fit so cache doesn’t skip work
        portfolio = Portfolio(
            ensembles=ensembles,
            trading_timeframe=TimeFrame.D,
            target_volatility=target_vol,
            max_position_pct=2.5,
        )
        tester = PortfolioTester(portfolio=portfolio, baseline_mode="equal_weight")
        t0 = time.perf_counter()
        tester.fit(train_candles)
        fit_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        tester.predict(test_candles)
        predict_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        tester.predict(
            test_candles,
            return_ensemble_predictions=True,
            return_base_model_predictions=True,
        )
        predict_granular_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        tester.calculate_strategy_returns(test_candles)
        strategy_returns_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        tester.calculate_baseline_returns(test_candles)
        baseline_returns_times.append(time.perf_counter() - t0)

    # Summary
    def _mean_and_std(seq: list[float]) -> tuple[float, float]:
        a = np.array(seq)
        return float(np.mean(a)), float(np.std(a)) if len(a) > 1 else 0.0

    fit_mean, fit_std = _mean_and_std(fit_times)
    pred_mean, pred_std = _mean_and_std(predict_times)
    pred_gr_mean, pred_gr_std = _mean_and_std(predict_granular_times)
    strat_mean, strat_std = _mean_and_std(strategy_returns_times)
    base_mean, base_std = _mean_and_std(baseline_returns_times)

    print("=" * 60)
    print("Benchmark summary (real data, 8 vault ensembles)")
    print("=" * 60)
    print(f"  Load ensembles:     {t_load_ensembles:>8.2f}s  (1 run)")
    print(f"  Load data:         {t_load_data:>8.2f}s  (1 run)")
    print(f"  Fit:               {fit_mean:>8.2f}s  (±{fit_std:.2f}, n={runs})")
    print(f"  Predict (basic):   {pred_mean:>8.2f}s  (±{pred_std:.2f}, n={runs})")
    print(f"  Predict (granular):{pred_gr_mean:>8.2f}s  (±{pred_gr_std:.2f}, n={runs})")
    print(f"  Strategy returns:  {strat_mean:>8.2f}s  (±{strat_std:.2f}, n={runs})")
    print(f"  Baseline returns:  {base_mean:>8.2f}s  (±{base_std:.2f}, n={runs})")
    print("=" * 60)
    total_core = fit_mean + pred_mean + strat_mean + base_mean
    print(f"  Core pipeline (fit + predict + returns): ~{total_core:.2f}s")
    print()
    print("Sequential ensemble execution (no joblib) for Cython-friendly pipeline.")
    print("To enable Cython: python utils/setup_cython.py build_ext --inplace")
    print("Re-run this script and compare timings (fit/predict benefit most).")


def run_profile_mode() -> None:
    """Run one fit+predict under cProfile and print top hotspots (cumtime)."""
    target_vol = 0.15
    cython = _cython_status()
    print("Cython extensions:")
    for name, available in cython.items():
        print(f"  {name}: {'✓' if available else '✗ (pure Python)'}")
    print()
    print("Loading ensembles and data...")
    t0 = time.perf_counter()
    ensembles = _load_ensembles(target_vol)
    train_candles, test_candles = _load_data()
    print(f"Loaded in {time.perf_counter() - t0:.2f}s\n")

    portfolio = Portfolio(
        ensembles=ensembles,
        trading_timeframe=TimeFrame.D,
        target_volatility=target_vol,
        max_position_pct=2.5,
    )
    tester = PortfolioTester(portfolio=portfolio, baseline_mode="equal_weight")

    def _target() -> None:
        tester.fit(train_candles)
        tester.predict(test_candles)

    profile_path = _PROJECT_ROOT / "benchmark_portfolio.prof"
    print("Profiling one fit + predict (this may take ~80s)...")
    cProfile.runctx("_target()", globals(), locals(), str(profile_path))
    print(f"\nProfile written to: {profile_path}")
    print("\nWhy so slow? Most time is in feature_base_model.add_candle (per-row Python loop +")
    print("Candle creation + bias node dispatch). Cython only speeds the inner kernels; see")
    print("docs/cython_portfolio_backtest.md §4 for details.")
    print("\nTop 80 by cumulative time (where the CPU is spent):")
    print("=" * 80)
    stats = pstats.Stats(str(profile_path))
    stats.strip_dirs().sort_stats("cumtime").print_stats(80)
    print("\nTop 40 by total time (own time, excluding callees):")
    print("=" * 80)
    stats.sort_stats("tottime").print_stats(40)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark portfolio backtest with real data and vault ensembles."
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run one fit+predict under cProfile and print hotspots (no timing table).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup fit+predict cycles (default: 1)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="Number of timed runs for fit/predict/returns (default: 2)",
    )
    args = parser.parse_args()
    if args.profile:
        run_profile_mode()
    else:
        run_benchmark(warmup=args.warmup, runs=args.runs)


if __name__ == "__main__":
    main()
