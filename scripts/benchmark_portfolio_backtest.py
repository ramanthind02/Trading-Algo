#!/usr/bin/env python3
"""
Benchmark the Cython-optimized portfolio backtest with real data and vault ensembles.

Uses portfolio_research config (tickers, ensemble_dirs, train/test dates from
portfolio_research.config.load_config()), sharing a single source of truth with
python portfolio_research/run_portfolio_test.py. Fit portfolio on train, predict
(basic + granular) and compute returns on test; reports Cython availability and
timings for load, fit, predict, and returns.

Usage:
    source venv/bin/activate
    python scripts/benchmark_portfolio_backtest.py [--warmup 1] [--runs 2]
    python scripts/benchmark_portfolio_backtest.py --profile   # run once with cProfile, print hotspots
"""

import argparse
import cProfile
import pstats
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ensemble.portfolio import Portfolio  # noqa: E402
from ensemble.portfolio_tester import PortfolioTester  # noqa: E402
from ensemble.vault_manager import load_ensemble_from_vault  # noqa: E402
from ensemble.weight_layer import WeightLayer  # noqa: E402
from portfolio_research.config import load_config  # noqa: E402
from utils.compute.daily_ewsd_volatility import compute_daily_ewsd_volatility  # noqa: E402
from utils.core.enums import TimeFrame  # noqa: E402
from utils.core.helpers import load_data_multi_ticker  # noqa: E402


def _cython_status() -> dict[str, bool]:
    """Report whether Cython extensions are available (stats + nodes)."""
    stats_available = False
    nodes_available = False
    try:
        import utils.compute.fast_stats as fs  # noqa: F401
        stats_available = getattr(fs, "CYTHON_AVAILABLE", False)
    except Exception:
        pass
    try:
        import utils.compute.fast_nodes as fn  # noqa: F401
        nodes_available = getattr(fn, "CYTHON_NODES_AVAILABLE", False)
    except Exception:
        pass
    return {"fast_stats (cython_optimized)": stats_available, "fast_nodes (cython_nodes)": nodes_available}


def _slice_candles_by_date(
    candles: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Return rows where start <= datetime <= end (inclusive)."""
    if candles.empty:
        return candles
    dt = pd.to_datetime(candles["datetime"], utc=False)
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_localize(None)
    mask = (dt >= start) & (dt <= end)
    return candles.loc[mask].copy()


def _load_data_from_config(config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/test candles and daily EWSD volatility using portfolio_research bounds."""
    wf_train_start, wf_train_end, wf_test_start, wf_test_end = (
        config.walkforward_train_test_bounds()
    )
    start_ts = pd.Timestamp(wf_train_start)
    end_ts = pd.Timestamp(wf_test_end)
    full = load_data_multi_ticker(
        tickers=config.tickers,
        timeframe=config.timeframe,
        start=datetime(start_ts.year, start_ts.month, start_ts.day),
        end=datetime(end_ts.year, end_ts.month, end_ts.day),
    )
    train_candles = _slice_candles_by_date(
        full, pd.Timestamp(wf_train_start), pd.Timestamp(wf_train_end)
    )
    test_candles = _slice_candles_by_date(
        full, pd.Timestamp(wf_test_start), pd.Timestamp(wf_test_end)
    )
    if config.timeframe == TimeFrame.D:
        daily_candles = full
    else:
        daily_candles = load_data_multi_ticker(
            tickers=config.tickers,
            timeframe=TimeFrame.D,
            start=datetime(start_ts.year, start_ts.month, start_ts.day),
            end=datetime(end_ts.year, end_ts.month, end_ts.day),
        )
    daily_volatility_df = compute_daily_ewsd_volatility(daily_candles)
    return train_candles, test_candles, daily_volatility_df


def _load_ensembles_from_config(config) -> list:
    """Load vault ensembles from config.ensemble_dirs; enable cache if config.use_cache."""
    def _enable_cache(ensemble):
        ensemble.use_cache = config.use_cache
        for model in getattr(ensemble, "base_models", {}).values():
            setattr(model, "use_cache", config.use_cache)
        return ensemble

    return [
        _enable_cache(
            load_ensemble_from_vault(
                path,
                refit=True,
                target_volatility=config.target_volatility,
            )
        )
        for path in config.ensemble_dirs.values()
    ]


def _build_portfolio_and_tester(config, ensembles: list) -> tuple:
    """Build Portfolio and PortfolioTester from config (same as run_portfolio_test)."""
    weight_layer = WeightLayer(
        weight_method=config.weight_layer_method,
        **dict(config.weight_layer_kwargs),
    )
    portfolio_kw = {
        "ensembles": ensembles,
        "trading_timeframe": config.timeframe,
        "target_volatility": config.target_volatility,
        "max_position_pct": config.max_position_pct,
        "weight_layer": weight_layer,
        "use_cache": config.use_cache,
    }
    portfolio = Portfolio(**portfolio_kw)
    tester = PortfolioTester(portfolio=portfolio, baseline_mode=config.baseline_mode)
    return portfolio, tester


def run_benchmark(warmup: int = 1, runs: int = 2) -> None:
    """Run full pipeline and report timings."""
    config = load_config()
    n_ensembles = len(config.ensemble_dirs)

    # Cython status
    cython = _cython_status()
    print("Cython extensions:")
    for name, available in cython.items():
        print(f"  {name}: {'✓' if available else '✗ (pure Python)'}")
    print()

    # Load ensembles
    t0 = time.perf_counter()
    ensembles = _load_ensembles_from_config(config)
    t_load_ensembles = time.perf_counter() - t0
    print(f"Loaded {len(ensembles)} ensembles in {t_load_ensembles:.2f}s")

    # Load data
    t0 = time.perf_counter()
    train_candles, test_candles, daily_volatility_df = _load_data_from_config(config)
    t_load_data = time.perf_counter() - t0
    print(f"Loaded train: {len(train_candles)} candles, test: {len(test_candles)} candles in {t_load_data:.2f}s")
    print()

    # Build portfolio and tester
    portfolio, tester = _build_portfolio_and_tester(config, ensembles)

    # Warmup: one full fit + predict to clear one-off overhead (e.g. imports)
    print("Warmup...")
    for _ in range(warmup):
        _p, _t = _build_portfolio_and_tester(config, ensembles)
        _t.fit(train_candles)
        _t.predict(train_candles, daily_volatility_df=daily_volatility_df)
    print("Warmup done.\n")

    # Timed runs: each run = new portfolio+tester, then time fit -> predict -> returns
    fit_times: list[float] = []
    predict_times: list[float] = []
    predict_granular_times: list[float] = []
    strategy_returns_times: list[float] = []
    baseline_returns_times: list[float] = []

    for _ in range(runs):
        portfolio, tester = _build_portfolio_and_tester(config, ensembles)
        t0 = time.perf_counter()
        tester.fit(train_candles)
        fit_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        tester.predict(test_candles, daily_volatility_df=daily_volatility_df)
        predict_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        tester.predict(
            test_candles,
            daily_volatility_df=daily_volatility_df,
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
    print(f"Benchmark summary (real data, {n_ensembles} vault ensembles from portfolio_research config)")
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
    print("To enable Cython: python utils/compute/cython/setup_cython.py build_ext --inplace")
    print("Re-run this script and compare timings (fit/predict benefit most).")


def run_profile_mode() -> None:
    """Run one fit+predict under cProfile and print top hotspots (cumtime)."""
    config = load_config()
    cython = _cython_status()
    print("Cython extensions:")
    for name, available in cython.items():
        print(f"  {name}: {'✓' if available else '✗ (pure Python)'}")
    print()
    print("Loading ensembles and data (from portfolio_research config)...")
    t0 = time.perf_counter()
    ensembles = _load_ensembles_from_config(config)
    train_candles, test_candles, daily_volatility_df = _load_data_from_config(config)
    print(f"Loaded in {time.perf_counter() - t0:.2f}s\n")

    portfolio, tester = _build_portfolio_and_tester(config, ensembles)

    def _target() -> None:
        tester.fit(train_candles)
        tester.predict(test_candles, daily_volatility_df=daily_volatility_df)

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
        description="Benchmark portfolio backtest with real data and vault ensembles (config from portfolio_research)."
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
