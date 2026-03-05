"""Portfolio test script: fit on explicit train/validation/test windows and composites.

Uses config from portfolio_research.config (tickers, dates, ensemble_dirs, windows).
Primary datasets: (1) train, (2) validation, (3) test. Two composite tearsheets:
(4) validation+test, (5) train+validation+test.

Usage
-----
    source /home/raman/repos/Trading-Algo/venv/bin/activate
    python portfolio_research/run_portfolio_test.py

Artifacts are written to config.output_root (walkforward/ and oos/ subdirs).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

from ensemble.portfolio import Portfolio
from ensemble.portfolio_tester import (
    PortfolioTester,
    aggregate_intraday_returns_to_daily,
    calculate_baseline_returns,
    calculate_strategy_returns_from_positions,
    resample_positions_to_daily,
)
from ensemble.vault_manager import load_ensemble_from_vault
from ensemble.weight_layer import WeightLayer
from metrics.plotting.graphing.quantstats_reports import generate_tearsheet
from portfolio_research.config import PortfolioResearchConfig, load_config
from utils.core.enums import TimeFrame
from utils.core.helpers import load_data_multi_ticker


def _timeframe_label(timeframe: TimeFrame) -> str:
    """Return stable lowercase labels for output filenames."""
    if timeframe == TimeFrame.D:
        return "daily"
    if timeframe == TimeFrame.W:
        return "weekly"
    if timeframe == TimeFrame.M:
        return "monthly"
    return timeframe.name.lower()


def _safe_name(name: str) -> str:
    """Sanitize report names for filesystem-safe filenames."""
    return name.replace(" ", "_").replace("::", "_").replace("/", "_")


@dataclass(frozen=True)
class PhaseResult:
    """Container for combined portfolio-level returns for a single phase."""

    name: str
    output_dir: Path
    combined_strategy_returns: pd.Series
    combined_baseline_returns: pd.Series


def _load_candles(
    config: PortfolioResearchConfig,
    timeframe: TimeFrame,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load OHLC candles for config tickers/timeframe over [start, end]. Normalize datetime."""
    from datetime import datetime

    s = start if start is not None else pd.Timestamp(config.start)
    e = end if end is not None else pd.Timestamp(config.end)
    df = load_data_multi_ticker(
        tickers=config.tickers,
        timeframe=timeframe,
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


def _group_ensembles_by_timeframe(
    named_ensembles: list[tuple[str, Any]],
) -> dict[TimeFrame, list[Any]]:
    """Group loaded ensembles by ensemble.base_tf (default TimeFrame.D when unset)."""
    grouped: dict[TimeFrame, list[Any]] = {}
    for _name, ensemble in named_ensembles:
        base_tf = getattr(ensemble, "base_tf", None)
        timeframe = base_tf if isinstance(base_tf, TimeFrame) else TimeFrame.D
        grouped.setdefault(timeframe, []).append(ensemble)
    return grouped


def _build_daily_dates_per_ticker(
    daily_candles: pd.DataFrame,
) -> dict[object, pd.DatetimeIndex]:
    """Build ticker -> sorted DatetimeIndex from daily candles."""
    if daily_candles.empty:
        return {}

    dates_df = daily_candles.loc[:, ["ticker", "datetime"]].copy()
    dates_df["datetime"] = pd.to_datetime(dates_df["datetime"]).dt.floor("s")

    return {
        ticker: pd.DatetimeIndex(series.sort_values().drop_duplicates())
        for ticker, series in dates_df.groupby("ticker", sort=False)["datetime"]
    }


def _generate_component_tearsheets(
    tester: PortfolioTester,
    candles_df: pd.DataFrame,
    output_dir: Path,
    baseline_returns: pd.Series,
    filename_prefix: str | None,
) -> None:
    """Generate ensemble/base-model tearsheets with optional filename prefix."""
    prefix = "" if filename_prefix is None else f"{filename_prefix}_"

    if tester.ensemble_predictions:
        for ensemble_name, ensemble_positions in tester.ensemble_predictions.items():
            ensemble_returns = calculate_strategy_returns_from_positions(
                ensemble_positions,
                candles_df,
            )
            ensemble_returns = aggregate_intraday_returns_to_daily(ensemble_returns)
            output_file = output_dir / f"{prefix}{ensemble_name}_tearsheet.html"
            generate_tearsheet(
                strategy_returns=ensemble_returns,
                baseline_returns=baseline_returns,
                feature_name=f"Ensemble: {ensemble_name}",
                output_file=str(output_file),
                mode="html",
            )

    if tester.base_model_predictions:
        for model_name, model_positions in tester.base_model_predictions.items():
            model_returns = calculate_strategy_returns_from_positions(
                model_positions,
                candles_df,
            )
            model_returns = aggregate_intraday_returns_to_daily(model_returns)
            safe_model_name = _safe_name(model_name)
            output_file = output_dir / f"{prefix}{safe_model_name}_tearsheet.html"
            generate_tearsheet(
                strategy_returns=model_returns,
                baseline_returns=baseline_returns,
                feature_name=f"Base Model: {model_name}",
                output_file=str(output_file),
                mode="html",
            )


def run_portfolio_test(config: PortfolioResearchConfig) -> None:
    """Load ensembles, fit portfolios by timeframe, run tearsheets for windows."""

    train_window = config.train_window
    validation_window = config.validation_window
    test_window = config.test_window

    train_start_ts = pd.Timestamp(train_window.start)
    train_end_ts = pd.Timestamp(train_window.end)
    val_start_ts = pd.Timestamp(validation_window.start)
    val_end_ts = pd.Timestamp(validation_window.end)
    test_start_ts = pd.Timestamp(test_window.start)
    test_end_ts = pd.Timestamp(test_window.end)

    print("\n" + "=" * 64)
    print("Portfolio Test")
    print(f"Tickers     : {[t.name for t in config.tickers]}")
    print(f"Train       : {train_start_ts.date()} -> {train_end_ts.date()}")
    print(f"Validation  : {val_start_ts.date()} -> {val_end_ts.date()}")
    print(f"Test        : {test_start_ts.date()} -> {test_end_ts.date()}")
    print(f"Output root : {config.output_root}")
    print("=" * 64 + "\n")

    def _enable_cache(ensemble: Any) -> Any:
        ensemble.use_cache = config.use_cache
        for model in getattr(ensemble, "base_models", {}).values():
            setattr(model, "use_cache", config.use_cache)
        return ensemble

    named_ensembles = [
        (
            name,
            _enable_cache(
                load_ensemble_from_vault(
                    path,
                    refit=True,
                    target_volatility=config.target_volatility,
                )
            ),
        )
        for name, path in config.ensemble_dirs.items()
    ]
    grouped_ensembles = _group_ensembles_by_timeframe(named_ensembles)
    unique_timeframes = sorted(grouped_ensembles.keys())
    has_multiple_timeframes = len(unique_timeframes) >= 2

    print(f"Loaded {len(named_ensembles)} ensemble(s) from vault")
    for tf in unique_timeframes:
        print(f"  {tf.name}: {len(grouped_ensembles[tf])} ensemble(s)")

    def _build_tester_for_timeframe(timeframe: TimeFrame) -> PortfolioTester:
        weight_layer = WeightLayer(
            weight_method=config.weight_layer_method,
            **dict(config.weight_layer_kwargs),
        )
        portfolio_kw: dict[str, Any] = {
            "ensembles": grouped_ensembles[timeframe],
            "trading_timeframe": timeframe,
            "target_volatility": config.target_volatility,
            "max_position_pct": config.max_position_pct,
            "weight_layer": weight_layer,
            "use_cache": config.use_cache,
        }
        if config.sector_allocation_config_path is not None:
            portfolio_kw["sector_allocation_config_path"] = config.sector_allocation_config_path
        portfolio = Portfolio(**portfolio_kw)
        return PortfolioTester(portfolio, baseline_mode=config.baseline_mode)

    def _evaluate_phase(
        phase_title: str,
        output_dir_name: str,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        test_start: pd.Timestamp,
        test_end: pd.Timestamp,
    ) -> PhaseResult:
        phase_out = config.output_root / output_dir_name
        phase_out.mkdir(parents=True, exist_ok=True)

        candles_by_timeframe: dict[TimeFrame, pd.DataFrame] = {
            TimeFrame.D: _load_candles(config, TimeFrame.D, start=train_start, end=test_end)
        }
        for timeframe in unique_timeframes:
            if timeframe == TimeFrame.D:
                continue
            candles_by_timeframe[timeframe] = _load_candles(
                config,
                timeframe,
                start=train_start,
                end=test_end,
            )

        train_candles_by_timeframe = {
            timeframe: _slice_candles_by_date(candles, train_start, train_end)
            for timeframe, candles in candles_by_timeframe.items()
        }
        test_candles_by_timeframe = {
            timeframe: _slice_candles_by_date(candles, test_start, test_end)
            for timeframe, candles in candles_by_timeframe.items()
        }

        daily_train_candles = train_candles_by_timeframe[TimeFrame.D]
        daily_test_candles = test_candles_by_timeframe[TimeFrame.D]
        if daily_train_candles.empty:
            raise ValueError(
                f"No daily candles in {phase_title} train window {train_start.date()} -> {train_end.date()}."
            )
        if daily_test_candles.empty:
            raise ValueError(
                f"No daily candles in {phase_title} test window {test_start.date()} -> {test_end.date()}."
            )

        for timeframe in unique_timeframes:
            tf_train = train_candles_by_timeframe.get(timeframe, pd.DataFrame())
            tf_test = test_candles_by_timeframe.get(timeframe, pd.DataFrame())
            if tf_train.empty:
                raise ValueError(
                    f"No candles for timeframe {timeframe.name} in {phase_title} train window."
                )
            if tf_test.empty:
                raise ValueError(
                    f"No candles for timeframe {timeframe.name} in {phase_title} test window."
                )

        print(f"\nEvaluating {phase_title} phase...")
        print(f"{phase_title} daily train: {len(daily_train_candles)} candles")
        print(f"{phase_title} daily test : {len(daily_test_candles)} candles")

        daily_dates_per_ticker = _build_daily_dates_per_ticker(daily_test_candles)
        daily_positions_by_timeframe: list[pd.DataFrame] = []
        testers_by_timeframe: dict[TimeFrame, PortfolioTester] = {}
        per_tf_strategy_returns: dict[TimeFrame, pd.Series] = {}
        per_tf_baseline_returns: dict[TimeFrame, pd.Series] = {}

        for timeframe in unique_timeframes:
            tf_train_candles = train_candles_by_timeframe[timeframe]
            tf_test_candles = test_candles_by_timeframe[timeframe]
            tf_label = _timeframe_label(timeframe)

            print(f"  Fitting {tf_label} portfolio...")
            tester = _build_tester_for_timeframe(timeframe)
            tester.fit(tf_train_candles)

            print(f"  Predicting {tf_label} portfolio...")
            tester.predict(
                tf_test_candles,
                return_ensemble_predictions=True,
                return_base_model_predictions=True,
            )
            if tester.positions_df is None:
                raise ValueError(f"No positions produced for timeframe {timeframe.name}")

            native_positions = tester.positions_df.loc[
                :, ["ticker", "datetime", "position_fraction"]
            ].copy()
            native_positions["datetime"] = pd.to_datetime(native_positions["datetime"]).dt.floor("s")

            if timeframe == TimeFrame.D:
                daily_positions = native_positions
            else:
                daily_positions = resample_positions_to_daily(
                    native_positions,
                    daily_dates_per_ticker,
                )

            daily_positions_by_timeframe.append(daily_positions)
            testers_by_timeframe[timeframe] = tester

            strategy_returns = tester.calculate_strategy_returns(
                tf_test_candles,
                positions_df=native_positions,
            )
            baseline_returns = tester.calculate_baseline_returns(tf_test_candles)
            per_tf_strategy_returns[timeframe] = aggregate_intraday_returns_to_daily(strategy_returns)
            per_tf_baseline_returns[timeframe] = aggregate_intraday_returns_to_daily(baseline_returns)

        combined_positions = (
            pd.concat(daily_positions_by_timeframe, ignore_index=True)
            .groupby(["ticker", "datetime"], as_index=False)["position_fraction"]
            .sum()
        )
        combined_positions["position_fraction"] = combined_positions["position_fraction"].clip(
            -config.max_position_pct,
            config.max_position_pct,
        )

        combined_strategy_returns = calculate_strategy_returns_from_positions(
            combined_positions,
            daily_test_candles,
        )
        combined_baseline_returns = calculate_baseline_returns(
            daily_test_candles,
            equal_weight=(config.baseline_mode == "equal_weight"),
        )
        combined_strategy_returns = aggregate_intraday_returns_to_daily(combined_strategy_returns)
        combined_baseline_returns = aggregate_intraday_returns_to_daily(combined_baseline_returns)

        combined_output_file = phase_out / f"Portfolio_{phase_title}_Test_tearsheet.html"
        generate_tearsheet(
            strategy_returns=combined_strategy_returns,
            baseline_returns=combined_baseline_returns,
            feature_name=f"Portfolio {phase_title} Test",
            output_file=str(combined_output_file),
            mode="html",
        )

        if has_multiple_timeframes:
            for timeframe in unique_timeframes:
                tf_label = _timeframe_label(timeframe)
                tf_output_file = phase_out / f"{tf_label}_Portfolio_{phase_title}_Test_tearsheet.html"
                generate_tearsheet(
                    strategy_returns=per_tf_strategy_returns[timeframe],
                    baseline_returns=per_tf_baseline_returns[timeframe],
                    feature_name=f"{tf_label.title()} Portfolio {phase_title} Test",
                    output_file=str(tf_output_file),
                    mode="html",
                )

        for timeframe in unique_timeframes:
            tf_label = _timeframe_label(timeframe)
            filename_prefix = tf_label if has_multiple_timeframes else None
            _generate_component_tearsheets(
                tester=testers_by_timeframe[timeframe],
                candles_df=test_candles_by_timeframe[timeframe],
                output_dir=phase_out,
                baseline_returns=per_tf_baseline_returns[timeframe],
                filename_prefix=filename_prefix,
            )

        print(f"{phase_title} tearsheets written to {phase_out}")

        return PhaseResult(
            name=phase_title,
            output_dir=phase_out,
            combined_strategy_returns=combined_strategy_returns,
            combined_baseline_returns=combined_baseline_returns,
        )

    # Phase 1: Train (fit and test on train window).
    train_result = _evaluate_phase(
        phase_title="Train",
        output_dir_name="train",
        train_start=train_start_ts,
        train_end=train_end_ts,
        test_start=train_start_ts,
        test_end=train_end_ts,
    )

    # Phase 2: Validation (fit on train, test on validation).
    validation_result = _evaluate_phase(
        phase_title="Validation",
        output_dir_name="validation",
        train_start=train_start_ts,
        train_end=train_end_ts,
        test_start=val_start_ts,
        test_end=val_end_ts,
    )

    # Phase 3: Test (fit on train+validation, test on test window).
    test_result = _evaluate_phase(
        phase_title="Test",
        output_dir_name="test",
        train_start=train_start_ts,
        train_end=val_end_ts,
        test_start=test_start_ts,
        test_end=test_end_ts,
    )

    # Composite 1: Validation + Test.
    combined_dir = config.output_root / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    val_test_strategy = (
        pd.concat(
            [
                validation_result.combined_strategy_returns,
                test_result.combined_strategy_returns,
            ]
        ).sort_index()
    )
    val_test_baseline = (
        pd.concat(
            [
                validation_result.combined_baseline_returns,
                test_result.combined_baseline_returns,
            ]
        ).sort_index()
    )
    val_test_output = combined_dir / "Portfolio_Validation_Test_tearsheet.html"
    generate_tearsheet(
        strategy_returns=val_test_strategy,
        baseline_returns=val_test_baseline,
        feature_name="Portfolio Validation+Test",
        output_file=str(val_test_output),
        mode="html",
    )

    # Composite 2: Train + Validation + Test.
    all_strategy = (
        pd.concat(
            [
                train_result.combined_strategy_returns,
                validation_result.combined_strategy_returns,
                test_result.combined_strategy_returns,
            ]
        ).sort_index()
    )
    all_baseline = (
        pd.concat(
            [
                train_result.combined_baseline_returns,
                validation_result.combined_baseline_returns,
                test_result.combined_baseline_returns,
            ]
        ).sort_index()
    )
    all_output = combined_dir / "Portfolio_Train_Validation_Test_tearsheet.html"
    generate_tearsheet(
        strategy_returns=all_strategy,
        baseline_returns=all_baseline,
        feature_name="Portfolio Train+Validation+Test",
        output_file=str(all_output),
        mode="html",
    )

    print(f"\nDone. Artifacts written to {config.output_root}\n")


if __name__ == "__main__":
    config = load_config()
    run_portfolio_test(config)
