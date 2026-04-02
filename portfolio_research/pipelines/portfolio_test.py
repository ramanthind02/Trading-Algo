"""Portfolio test pipeline: fit on explicit train/validation/test windows and composites.

Uses config from portfolio_research.config (tickers, dates, ensemble_dirs, windows).
Primary datasets: (1) train, (2) validation, (3) test. Two composite tearsheets:
(4) validation+test, (5) train+validation+test.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ensemble.portfolio import (
    GlobalPortfolio,
    Portfolio,
    PortfolioCacheQuery,
    PortfolioWorld,
    TFPortfolio,
    materialize_global_portfolio_predictions,
)
from ensemble.portfolio_tester import (
    PortfolioTester,
    aggregate_intraday_returns_to_daily,
    calculate_baseline_returns,
    calculate_strategy_returns_from_positions,
)
from ensemble.vault_manager import (
    get_bias_node_specs,
    get_ensemble_tickers,
    load_ensemble_from_vault,
)
from ensemble.weight_layer import WeightLayer
from metrics.plotting.graphing.quantstats_reports import generate_tearsheet
from utils.cache import (
    CentralCacheStore,
    bootstrap_source_candles,
    extract_cross_ticker_names,
)
from utils.core.enums import Ticker, TimeFrame
from utils.evaluation.walkforward.runner import _sanitize_tearsheet_name

try:
    from ensemble.vault_manager import (  # type: ignore[attr-defined]
        ensure_vault_cache_coverage,
        migrate_legacy_feature_members_schema,
    )
except ImportError:  # pragma: no cover - fallback for branches that still add these helpers
    def migrate_legacy_feature_members_schema(*_args: object, **_kwargs: object) -> None:
        raise ImportError(
            "ensemble.vault_manager.migrate_legacy_feature_members_schema is not available"
        )

    def ensure_vault_cache_coverage(*_args: object, **_kwargs: object) -> dict[str, Any]:
        raise ImportError(
            "ensemble.vault_manager.ensure_vault_cache_coverage is not available"
        )

# Tearsheet output mode (HTML is currently the only supported format)
_TEARSHEET_MODE = "html"


def _timeframe_label(timeframe: TimeFrame) -> str:
    """Return stable lowercase labels for output filenames."""
    if timeframe == TimeFrame.D:
        return "daily"
    if timeframe == TimeFrame.W:
        return "weekly"
    if timeframe == TimeFrame.M:
        return "monthly"
    return timeframe.name.lower()


@dataclass(frozen=True)
class PhaseResult:
    """Container for combined portfolio-level returns for a single phase."""

    name: str
    output_dir: Path
    combined_strategy_returns: pd.Series
    combined_baseline_returns: pd.Series


def _portfolio_cache_query(
    config: Any,
    start: datetime,
    end: datetime,
    timeframes: tuple[TimeFrame, ...],
) -> PortfolioCacheQuery:
    """Build a cache-native portfolio query from the research config."""
    return PortfolioCacheQuery(
        tickers=tuple(
            ticker.name if hasattr(ticker, "name") else str(ticker)
            for ticker in config.tickers
        ),
        start=start,
        end=end,
        timeframes=timeframes,
    )


def _coerce_ticker(ticker: object) -> Ticker:
    """Normalize ticker-like inputs to the project ticker enum."""
    if isinstance(ticker, TimeFrame):  # pragma: no cover - defensive guard
        raise TypeError("timeframe passed where ticker was expected")
    if hasattr(ticker, "name") and not isinstance(ticker, str):
        return ticker  # type: ignore[return-value]
    return Ticker[str(ticker)]


def _phase_world(output_dir_name: str) -> PortfolioWorld:
    normalized = output_dir_name.strip().lower()
    if normalized == "train":
        return PortfolioWorld.TRAIN
    if normalized == "validation":
        return PortfolioWorld.VAL
    if normalized == "test":
        return PortfolioWorld.TEST
    return PortfolioWorld.LIVE


def _load_candles(
    config: Any,
    timeframe: TimeFrame,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load candles for the requested window from the central cache."""
    store = CentralCacheStore.get_instance()
    s = pd.Timestamp(start if start is not None else config.start).to_pydatetime()
    e = pd.Timestamp(end if end is not None else config.end).to_pydatetime()

    frames = []
    for ticker in config.tickers:
        ticker_enum = _coerce_ticker(ticker)
        record = store.describe_candle(ticker_enum, timeframe)
        if record is None or record.coverage.start is None or record.coverage.end is None:
            raise ValueError(
                f"No cached candles available for {ticker_enum.name}/{timeframe.name}"
            )
        effective_start = max(pd.Timestamp(s), pd.Timestamp(record.coverage.start))
        effective_end = min(pd.Timestamp(e), pd.Timestamp(record.coverage.end))
        if effective_start > effective_end:
            raise ValueError(
                f"Requested window has no candle overlap for {ticker_enum.name}/{timeframe.name}: "
                f"{s.date()} -> {e.date()} vs coverage "
                f"{pd.Timestamp(record.coverage.start).date()} -> {pd.Timestamp(record.coverage.end).date()}"
            )
        frame = store.query_candles(
            ticker_enum,
            timeframe,
            start=effective_start.to_pydatetime(),
            end=effective_end.to_pydatetime(),
        ).reset_index()
        if "ticker" not in frame.columns:
            frame["ticker"] = ticker_enum.name
        if "timeframe" not in frame.columns:
            frame["timeframe"] = timeframe
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "datetime"]).reset_index(drop=True)


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


def _enable_cache(ensemble: Any, use_cache: bool) -> Any:
    """Enable/disable cache on ensemble and base models."""
    ensemble.use_cache = use_cache
    if use_cache and hasattr(ensemble, "retry_on_cache_miss"):
        ensemble.retry_on_cache_miss = False
    for model in getattr(ensemble, "base_models", {}).values():
        setattr(model, "use_cache", use_cache)
    return ensemble


def _build_instrument_returns(daily_candles: pd.DataFrame) -> pd.DataFrame:
    """Build a (date × ticker) instrument-returns DataFrame from daily candles.

    Used as input to ``GlobalPortfolio.fit`` for IDM and WeightLayer fitting.
    """
    if daily_candles.empty:
        return pd.DataFrame()

    df = daily_candles[["ticker", "datetime", "close"]].copy()
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.floor("s")
    pivot_close = df.pivot_table(
        index="datetime", columns="ticker", values="close", aggfunc="last"
    )
    pivot_close.columns.name = None
    log_returns = np.log(pivot_close / pivot_close.shift(1))
    return log_returns.dropna(how="all")


def _coerce_timeframe(timeframe: object) -> TimeFrame:
    """Normalize timeframe-like values to the project timeframe enum."""
    if isinstance(timeframe, TimeFrame):
        return timeframe
    if hasattr(timeframe, "name") and not isinstance(timeframe, str):
        return TimeFrame[getattr(timeframe, "name")]
    return TimeFrame[str(timeframe)]


def _collect_preflight_bootstrap_inputs(
    ensemble_dirs: Sequence[str],
    portfolio_tickers: Sequence[Ticker],
) -> tuple[tuple[Ticker, ...], tuple[TimeFrame, ...]]:
    """Collect the exact candle inputs required for portfolio cache preflight."""
    ordered_tickers: list[Ticker] = []
    seen_tickers: set[Ticker] = set()

    def add_ticker(ticker_like: object) -> None:
        ticker = _coerce_ticker(ticker_like)
        if ticker in seen_tickers:
            return
        seen_tickers.add(ticker)
        ordered_tickers.append(ticker)

    ordered_timeframes: list[TimeFrame] = []
    seen_timeframes: set[TimeFrame] = set()

    def add_timeframe(timeframe_like: object) -> None:
        timeframe = _coerce_timeframe(timeframe_like)
        if timeframe in seen_timeframes:
            return
        seen_timeframes.add(timeframe)
        ordered_timeframes.append(timeframe)

    for ticker in portfolio_tickers:
        add_ticker(ticker)
    add_timeframe(TimeFrame.D)

    for ensemble_dir in ensemble_dirs:
        for ticker in get_ensemble_tickers(ensemble_dir):
            add_ticker(ticker)

        for spec in get_bias_node_specs(ensemble_dir):
            timeframes = spec.get("timeframes") or [TimeFrame.D]
            for timeframe in timeframes:
                add_timeframe(timeframe)

            params = spec.get("params", {})
            if isinstance(params, Mapping):
                for cross_ticker_name in extract_cross_ticker_names(params):
                    add_ticker(Ticker[cross_ticker_name])

    return tuple(ordered_tickers), tuple(ordered_timeframes)


def _preflight_vault_cache(
    ensemble_dirs: dict[str, str],
    portfolio_tickers: Sequence[Ticker],
    start: datetime,
    end: datetime,
) -> dict[str, Any]:
    """Normalize vault feature schemas and ensure cache coverage before portfolio runs."""
    unique_dirs = tuple(dict.fromkeys(ensemble_dirs.values()))
    for ensemble_dir in unique_dirs:
        migrate_legacy_feature_members_schema(ensemble_dir)

    bootstrap_tickers, bootstrap_timeframes = _collect_preflight_bootstrap_inputs(
        unique_dirs,
        portfolio_tickers,
    )
    bootstrap_summary = bootstrap_source_candles(
        tickers=bootstrap_tickers,
        timeframes=bootstrap_timeframes,
        start_date=start,
        end_date=end,
    )

    refresh_summary = ensure_vault_cache_coverage(
        vault_ensemble_dirs=unique_dirs,
        start_date=start,
        end_date=end,
    )
    return {
        **refresh_summary,
        "bootstrap": bootstrap_summary,
    }


def _preflight_ready_counts(summary: dict[str, Any]) -> tuple[int, int]:
    """Normalize cache-preflight summaries across old/new result contracts."""
    if "total_tasks" in summary:
        return int(summary.get("rebuilt", 0)) + int(summary.get("validated", 0)), int(
            summary.get("total_tasks", 0)
        )
    return int(summary.get("success", 0)), int(summary.get("total", 0))


def _build_tester_for_timeframe(
    timeframe: TimeFrame,
    config: Any,
    grouped_ensembles: dict[TimeFrame, list[Any]],
) -> PortfolioTester:
    """Build PortfolioTester for a specific timeframe."""
    portfolio_kw: dict[str, Any] = {
        "ensembles": grouped_ensembles[timeframe],
        "trading_timeframe": timeframe,
        "target_volatility": config.target_volatility,
        "max_position_pct": config.max_position_pct,
        "use_cache": True,
    }
    portfolio = Portfolio(**portfolio_kw)
    return PortfolioTester(portfolio, baseline_mode=config.baseline_mode)


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
                mode=_TEARSHEET_MODE,
            )

    if tester.base_model_predictions:
        for model_name, model_positions in tester.base_model_predictions.items():
            model_returns = calculate_strategy_returns_from_positions(
                model_positions,
                candles_df,
            )
            model_returns = aggregate_intraday_returns_to_daily(model_returns)
            safe_model_name = _sanitize_tearsheet_name(model_name)
            output_file = output_dir / f"{prefix}{safe_model_name}_tearsheet.html"
            generate_tearsheet(
                strategy_returns=model_returns,
                baseline_returns=baseline_returns,
                feature_name=f"Base Model: {model_name}",
                output_file=str(output_file),
                mode=_TEARSHEET_MODE,
            )


def _generate_composite_tearsheet(
    results: list[PhaseResult],
    composite_name: str,
    output_dir: Path,
) -> None:
    """Generate tearsheet from combined phase results (e.g., Validation+Test)."""
    strategy = (
        pd.concat([r.combined_strategy_returns for r in results])
        .sort_index()
    )
    baseline = (
        pd.concat([r.combined_baseline_returns for r in results])
        .sort_index()
    )
    output_file = output_dir / f"Portfolio_{composite_name}_tearsheet.html"
    generate_tearsheet(
        strategy_returns=strategy,
        baseline_returns=baseline,
        feature_name=f"Portfolio {composite_name}",
        output_file=str(output_file),
        mode=_TEARSHEET_MODE,
    )


def _evaluate_phase(
    phase_title: str,
    output_dir_name: str,
    fit_start: pd.Timestamp,
    fit_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    config: Any,
    grouped_ensembles: dict[TimeFrame, list[Any]],
    unique_timeframes: list[TimeFrame],
) -> PhaseResult:
    """Evaluate a portfolio phase: fit on train window, test on test window.

    Portfolio tearsheet naming: phase-level combined → Portfolio_{phase}_window_tearsheet.html;
    per-timeframe → {tf_label}_Portfolio_{phase}_window_tearsheet.html. Composite (multi-window)
    tearsheets use composite_name e.g. Validation_and_Test_windows → Portfolio_{name}_tearsheet.html.
    """
    phase_out = config.output_root / output_dir_name
    phase_out.mkdir(parents=True, exist_ok=True)
    portfolio_dir = phase_out / "portfolio"
    portfolio_dir.mkdir(parents=True, exist_ok=True)

    fit_query = _portfolio_cache_query(
        config,
        fit_start.to_pydatetime(),
        fit_end.to_pydatetime(),
        tuple(unique_timeframes),
    )
    predict_query = _portfolio_cache_query(
        config,
        test_start.to_pydatetime(),
        test_end.to_pydatetime(),
        tuple(unique_timeframes),
    )

    candle_timeframes = sorted({TimeFrame.D, *unique_timeframes})
    candles_by_timeframe: dict[TimeFrame, pd.DataFrame] = {
        timeframe: _load_candles(config, timeframe, start=fit_start, end=test_end)
        for timeframe in candle_timeframes
    }

    train_candles_by_timeframe = {
        timeframe: _slice_candles_by_date(candles, fit_start, fit_end)
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
            f"No daily candles in {phase_title} train window {fit_start.date()} -> {fit_end.date()}."
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

    testers_by_timeframe: dict[TimeFrame, PortfolioTester] = {}
    per_tf_strategy_returns: dict[TimeFrame, pd.Series] = {}
    per_tf_baseline_returns: dict[TimeFrame, pd.Series] = {}

    for timeframe in unique_timeframes:
        tf_test_candles = test_candles_by_timeframe[timeframe]
        tf_label = _timeframe_label(timeframe)

        print(f"  Fitting {tf_label} portfolio...")
        tester = _build_tester_for_timeframe(timeframe, config, grouped_ensembles)
        tester.fit_from_cache(fit_query.for_timeframe(timeframe))

        print(f"  Predicting {tf_label} portfolio...")
        tester.predict_from_cache(
            predict_query.for_timeframe(timeframe),
            return_ensemble_predictions=True,
            return_base_model_predictions=True,
        )
        if tester.positions_df is None:
            raise ValueError(f"No positions produced for timeframe {timeframe.name}")

        native_positions = tester.positions_df.loc[
            :, ["ticker", "datetime", "position_fraction"]
        ].copy()
        native_positions["datetime"] = pd.to_datetime(native_positions["datetime"]).dt.floor("s")

        testers_by_timeframe[timeframe] = tester

        strategy_returns = tester.calculate_strategy_returns(
            tf_test_candles,
            positions_df=native_positions,
        )
        baseline_returns = tester.calculate_baseline_returns(tf_test_candles)
        per_tf_strategy_returns[timeframe] = aggregate_intraday_returns_to_daily(strategy_returns)
        per_tf_baseline_returns[timeframe] = aggregate_intraday_returns_to_daily(baseline_returns)

    # Always build GlobalPortfolio so the global WeightLayer is fitted
    # even for a single-timeframe run.
    tf_portfolios = [testers_by_timeframe[tf].portfolio for tf in unique_timeframes]
    global_portfolio = GlobalPortfolio(
        tf_portfolios=tf_portfolios,
        weight_layer=WeightLayer(
            weight_method=config.weight_layer_method,
            **dict(config.weight_layer_kwargs),
        ),
        max_position_pct=config.max_position_pct,
    )
    instrument_returns = _build_instrument_returns(daily_train_candles)
    global_portfolio.fit_from_cache(
        fit_query,
        instrument_returns,
    )
    portfolio_id = global_portfolio.save_to_vault(
        fit_start=fit_start.to_pydatetime(),
        fit_end=fit_end.to_pydatetime(),
    )
    global_positions_raw = global_portfolio.predict_from_cache(predict_query)
    materialize_global_portfolio_predictions(
        portfolio=global_portfolio,
        query=predict_query,
        portfolio_id=portfolio_id,
        world=_phase_world(output_dir_name),
    )
    global_positions_raw["datetime"] = pd.to_datetime(
        global_positions_raw["datetime"]
    ).dt.floor("s")
    combined_positions = global_positions_raw[
        ["ticker", "datetime", "position_fraction"]
    ].copy()
    combined_positions["position_fraction"] = combined_positions[
        "position_fraction"
    ].clip(-config.max_position_pct, config.max_position_pct)

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

    combined_output_file = portfolio_dir / f"Portfolio_{phase_title}_window_tearsheet.html"
    generate_tearsheet(
        strategy_returns=combined_strategy_returns,
        baseline_returns=combined_baseline_returns,
        feature_name=f"Portfolio {phase_title} Test",
        output_file=str(combined_output_file),
        mode="html",
    )

    if len(unique_timeframes) >= 2:
        for timeframe in unique_timeframes:
            tf_label = _timeframe_label(timeframe)
            tf_output_file = portfolio_dir / f"{tf_label}_Portfolio_{phase_title}_window_tearsheet.html"
            generate_tearsheet(
                strategy_returns=per_tf_strategy_returns[timeframe],
                baseline_returns=per_tf_baseline_returns[timeframe],
                feature_name=f"{tf_label.title()} Portfolio {phase_title} Test",
                output_file=str(tf_output_file),
                mode=_TEARSHEET_MODE,
            )

    for timeframe in unique_timeframes:
        tf_label = _timeframe_label(timeframe)
        tf_dir = phase_out / tf_label
        tf_dir.mkdir(parents=True, exist_ok=True)
        _generate_component_tearsheets(
            tester=testers_by_timeframe[timeframe],
            candles_df=test_candles_by_timeframe[timeframe],
            output_dir=tf_dir,
            baseline_returns=per_tf_baseline_returns[timeframe],
            filename_prefix=None,
        )

    print(f"{phase_title} tearsheets written to {phase_out}")

    return PhaseResult(
        name=phase_title,
        output_dir=phase_out,
        combined_strategy_returns=combined_strategy_returns,
        combined_baseline_returns=combined_baseline_returns,
    )


def run_portfolio_test_pipeline(config: Any) -> None:
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

    cache_preflight = _preflight_vault_cache(
        config.ensemble_dirs,
        config.tickers,
        train_start_ts.to_pydatetime(),
        test_end_ts.to_pydatetime(),
    )
    bootstrap_summary = cache_preflight.get("bootstrap", {})
    if isinstance(bootstrap_summary, dict) and bootstrap_summary:
        print(
            "Source candle bootstrap complete: "
            f"{int(bootstrap_summary.get('success', 0))}/"
            f"{int(bootstrap_summary.get('total', 0))} datasets loaded"
        )
    ready_count, total_count = _preflight_ready_counts(cache_preflight)
    print(
        "Vault cache preflight complete: "
        f"{ready_count}/{total_count} artifacts ready"
    )
    bootstrap_summary = cache_preflight.get("bootstrap", cache_preflight.get("ingested", {}))
    bootstrap_failures = (
        int(bootstrap_summary.get("failed", 0))
        if isinstance(bootstrap_summary, dict)
        else 0
    )
    artifact_failures = int(cache_preflight.get("failed", 0))
    if bootstrap_failures or artifact_failures:
        raise RuntimeError(
            "Vault cache preflight failed: "
            f"{artifact_failures} artifact refresh errors, "
            f"{bootstrap_failures} candle bootstrap errors."
        )

    named_ensembles = [
        (
            name,
            _enable_cache(
                load_ensemble_from_vault(
                    path,
                    refit=True,
                    target_volatility=config.target_volatility,
                ),
                True,
            ),
        )
        for name, path in config.ensemble_dirs.items()
    ]
    grouped_ensembles = _group_ensembles_by_timeframe(named_ensembles)
    unique_timeframes = sorted(grouped_ensembles.keys())

    print(f"Loaded {len(named_ensembles)} ensemble(s) from vault")
    for tf in unique_timeframes:
        print(f"  {tf.name}: {len(grouped_ensembles[tf])} ensemble(s)")

    # Phase 1: Train (fit and test on train window).
    train_result = _evaluate_phase(
        phase_title="Train",
        output_dir_name="train",
        fit_start=train_start_ts,
        fit_end=train_end_ts,
        test_start=train_start_ts,
        test_end=train_end_ts,
        config=config,
        grouped_ensembles=grouped_ensembles,
        unique_timeframes=unique_timeframes,
    )

    # Phase 2: Validation (fit on train, test on validation).
    validation_result = _evaluate_phase(
        phase_title="Validation",
        output_dir_name="validation",
        fit_start=train_start_ts,
        fit_end=train_end_ts,
        test_start=val_start_ts,
        test_end=val_end_ts,
        config=config,
        grouped_ensembles=grouped_ensembles,
        unique_timeframes=unique_timeframes,
    )

    # Phase 3: Test (fit on train+validation, test on test window).
    test_result = _evaluate_phase(
        phase_title="Test",
        output_dir_name="test",
        fit_start=train_start_ts,
        fit_end=val_end_ts,
        test_start=test_start_ts,
        test_end=test_end_ts,
        config=config,
        grouped_ensembles=grouped_ensembles,
        unique_timeframes=unique_timeframes,
    )

    # Composite: Validation+Test and Train+Validation+Test tearsheets.
    combined_dir = config.output_root / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_portfolio_dir = combined_dir / "portfolio"
    combined_portfolio_dir.mkdir(parents=True, exist_ok=True)

    _generate_composite_tearsheet(
        [validation_result, test_result],
        "Validation_and_Test_windows",
        combined_portfolio_dir,
    )

    _generate_composite_tearsheet(
        [train_result, validation_result, test_result],
        "Train_Validation_and_Test_windows",
        combined_portfolio_dir,
    )

    print(f"\nDone. Artifacts written to {config.output_root}\n")
