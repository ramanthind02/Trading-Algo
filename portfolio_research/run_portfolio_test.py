"""Portfolio test script: fit on walkforward train, evaluate on walkforward test and OOS.

Uses config from portfolio_research.config (tickers, dates, ensemble_dirs, OOS window).
Two datasets: (1) in-sample walkforward bounds from compute_first_fold_bounds,
(2) OOS when config.oos_window is set.

Usage
-----
    source /home/raman/repos/Trading-Algo/venv/bin/activate
    python portfolio_research/run_portfolio_test.py

Artifacts are written to config.output_root (walkforward/ and oos/ subdirs).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def _find_repo_root(start: Path) -> Path | None:
    """Search up from start path to find repo root (pyproject.toml or .git)."""
    search_root = start if start.is_dir() else start.parent
    for parent in (search_root, *search_root.parents):
        if (parent / "pyproject.toml").exists():
            return parent
        if (parent / ".git").exists():
            return parent
    return None


_repo_root = _find_repo_root(Path(__file__).resolve())
if _repo_root is not None and str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from portfolio_research.config import PortfolioResearchConfig, load_config
from utils.core.helpers import load_data_multi_ticker


def _load_candles(
    config: PortfolioResearchConfig,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load OHLC candles for config tickers/timeframe over [start, end]. Normalize datetime."""
    from datetime import datetime

    s = start if start is not None else pd.Timestamp(config.start)
    e = end if end is not None else pd.Timestamp(config.end)
    df = load_data_multi_ticker(
        tickers=config.tickers,
        timeframe=config.timeframe,
        start=datetime(s.year, s.month, s.day),
        end=datetime(e.year, e.month, e.day),
    )
    if df.empty:
        return df
    if "datetime" not in df.columns:
        raise ValueError("Candles DataFrame must include 'datetime' column.")
    df = df.sort_values(["ticker", "datetime"])
    return df


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


def run_portfolio_test(config: PortfolioResearchConfig) -> None:
    """Load ensembles, fit portfolio on walkforward train, run tearsheets for walkforward test and OOS."""
    from ensemble.portfolio import Portfolio
    from ensemble.portfolio_tester import PortfolioTester
    from ensemble.vault_manager import load_ensemble_from_vault
    from ensemble.weight_layer import WeightLayer

    wf_train_start, wf_train_end, wf_test_start, wf_test_end = (
        config.walkforward_train_test_bounds()
    )
    wf_train_start_ts = pd.Timestamp(wf_train_start)
    wf_train_end_ts = pd.Timestamp(wf_train_end)
    wf_test_start_ts = pd.Timestamp(wf_test_start)
    wf_test_end_ts = pd.Timestamp(wf_test_end)

    print("\n" + "=" * 64)
    print("Portfolio Test")
    print(f"Tickers     : {[t.name for t in config.tickers]}")
    print(f"WF train    : {wf_train_start_ts.date()} -> {wf_train_end_ts.date()}")
    print(f"WF test     : {wf_test_start_ts.date()} -> {wf_test_end_ts.date()}")
    if config.oos_window is not None:
        oos = config.oos_window
        print(f"OOS train   : {oos.train_start.date()} -> {oos.train_end.date()}")
        print(f"OOS test    : {oos.test_start.date()} -> {oos.test_end.date()}")
    print(f"Output root : {config.output_root}")
    print("=" * 64 + "\n")

    # Load candles: walkforward period
    wf_candles = _load_candles(
        config,
        start=wf_train_start_ts,
        end=wf_test_end_ts,
    )
    wf_train_candles = _slice_candles_by_date(
        wf_candles, wf_train_start_ts, wf_train_end_ts
    )
    wf_test_candles = _slice_candles_by_date(
        wf_candles, wf_test_start_ts, wf_test_end_ts
    )
    if wf_train_candles.empty:
        raise ValueError(
            f"No candles in walkforward train window {wf_train_start_ts.date()} -> {wf_train_end_ts.date()}. "
            "Check data/ohlc_data and config start/end."
        )
    if wf_test_candles.empty:
        raise ValueError(
            f"No candles in walkforward test window {wf_test_start_ts.date()} -> {wf_test_end_ts.date()}."
        )
    print(f"Walkforward train: {len(wf_train_candles)} candles")
    print(f"Walkforward test : {len(wf_test_candles)} candles")

    oos_train_candles = pd.DataFrame()
    oos_test_candles = pd.DataFrame()
    if config.oos_window is not None:
        oos = config.oos_window
        oos_start_ts = pd.Timestamp(oos.train_start)
        oos_end_ts = pd.Timestamp(oos.test_end)
        oos_candles = _load_candles(config, start=oos_start_ts, end=oos_end_ts)
        oos_train_candles = _slice_candles_by_date(
            oos_candles,
            pd.Timestamp(oos.train_start),
            pd.Timestamp(oos.train_end),
        )
        oos_test_candles = _slice_candles_by_date(
            oos_candles,
            pd.Timestamp(oos.test_start),
            pd.Timestamp(oos.test_end),
        )
        print(f"OOS train: {len(oos_train_candles)} candles")
        print(f"OOS test : {len(oos_test_candles)} candles")

    # Load ensembles from vault
    def _enable_cache(ensemble):  # noqa: ANN001
        ensemble.use_cache = config.use_cache
        for model in getattr(ensemble, "base_models", {}).values():
            setattr(model, "use_cache", config.use_cache)
        return ensemble

    ensembles_list = [
        _enable_cache(
            load_ensemble_from_vault(
                path,
                refit=True,
                target_volatility=config.target_volatility,
            )
        )
        for _name, path in config.ensemble_dirs.items()
    ]
    print(f"Loaded {len(ensembles_list)} ensemble(s) from vault")

    # Build weight layer and portfolio
    weight_layer = WeightLayer(
        weight_method=config.weight_layer_method,
        **dict(config.weight_layer_kwargs),
    )
    portfolio = Portfolio(
        ensembles=ensembles_list,
        trading_timeframe=config.timeframe,
        target_volatility=config.target_volatility,
        max_position_pct=config.max_position_pct,
        weight_layer=weight_layer,
        use_cache=config.use_cache,
    )
    tester = PortfolioTester(portfolio, baseline_mode=config.baseline_mode)

    # Fit on walkforward train
    print("\nFitting portfolio on walkforward train...")
    tester.fit(wf_train_candles)

    # In-sample walkforward: predict on WF test, tearsheets
    wf_out = config.output_root / "walkforward"
    wf_out.mkdir(parents=True, exist_ok=True)
    print("\nEvaluating on walkforward test...")
    tester.predict(wf_test_candles)
    tester.calculate_strategy_returns(wf_test_candles)
    tester.calculate_baseline_returns(wf_test_candles)
    tester.generate_tearsheet(
        strategy_name="Portfolio Walkforward Test",
        output_dir=str(wf_out),
        mode="html",
        candles_df=wf_test_candles,
    )
    if tester.ensemble_predictions:
        tester.generate_ensemble_tearsheets(
            output_dir=str(wf_out),
            mode="html",
            candles_df=wf_test_candles,
        )
    if tester.base_model_predictions:
        tester.generate_base_model_tearsheets(
            output_dir=str(wf_out),
            mode="html",
            candles_df=wf_test_candles,
        )
    print(f"Walkforward tearsheets written to {wf_out}")

    # OOS: optional refit on OOS train, predict on OOS test, tearsheets
    if config.oos_window is not None and not oos_test_candles.empty:
        oos_out = config.output_root / "oos"
        oos_out.mkdir(parents=True, exist_ok=True)
        if not oos_train_candles.empty:
            print("\nRefitting portfolio on OOS train...")
            tester.fit(oos_train_candles)
        print("Evaluating on OOS test...")
        tester.predict(oos_test_candles)
        tester.calculate_strategy_returns(oos_test_candles)
        tester.calculate_baseline_returns(oos_test_candles)
        tester.generate_tearsheet(
            strategy_name="Portfolio OOS Test",
            output_dir=str(oos_out),
            mode="html",
            candles_df=oos_test_candles,
        )
        if tester.ensemble_predictions:
            tester.generate_ensemble_tearsheets(
                output_dir=str(oos_out),
                mode="html",
                candles_df=oos_test_candles,
            )
        if tester.base_model_predictions:
            tester.generate_base_model_tearsheets(
                output_dir=str(oos_out),
                mode="html",
                candles_df=oos_test_candles,
            )
        print(f"OOS tearsheets written to {oos_out}")

    print(f"\nDone. Artifacts written to {config.output_root}\n")


if __name__ == "__main__":
    config = load_config()
    run_portfolio_test(config)
