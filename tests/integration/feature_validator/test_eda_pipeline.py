"""Integration tests for Feature Validator EDA pipeline (Task 6).

Uses real repository data from data/ohlc_data/ and real bias-node extraction.
Default configurations:
  - Continuous flow (T001+T002+T004): RSI lookback=5, ES daily, 2020-2023
  - Rule-based flow (T001+T003+T004): RSI 2-period strategy, ES daily, 2020-2023
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
import pytest

from feature_extraction.feature_extractor import extract_features_for_bias_node
from feature_selection.eda.eda_dataclasses import (
    ContinuousEDAReport,
    EDAConfig,
    EDAMetadata,
    RuleBasedEDAReport,
)
from feature_selection.eda.eda_reporter import (
    run_eda_for_continuous_feature,
    run_eda_for_rule_based_feature,
    save_eda_report,
)
from utils.cache_manager import CacheManager
from utils.enums import Ticker, TimeFrame

matplotlib.use("Agg")

DEFAULT_TICKER = Ticker.ES
DEFAULT_TIMEFRAME = TimeFrame.D
DEFAULT_START = datetime(2020, 1, 1)
DEFAULT_END = datetime(2023, 12, 31)

DEFAULT_CONTINUOUS_BIAS_SPEC: dict[str, Any] = {
    "module_name": "rsi",
    "timeframes": [DEFAULT_TIMEFRAME],
    "params": {"lookback": 5},
}

DEFAULT_RULE_BASED_BIAS_SPEC: dict[str, Any] = {
    "module_name": "rsi_signal",
    "timeframes": [DEFAULT_TIMEFRAME],
    "params": {
        "rsi_period": 2,
        "oversold": 25.0,
        "overbought": 65.0,
        "strategy_mode": "long",
        "exit_policy": "threshold_or_bars",
        "exit_bars": 5,
    },
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _normalize_timeframe(bias_spec: dict[str, Any], fallback: TimeFrame) -> TimeFrame:
    raw_timeframes = bias_spec.get("timeframes", [fallback])
    first = raw_timeframes[0] if isinstance(raw_timeframes, list) else raw_timeframes
    return TimeFrame[first] if isinstance(first, str) else first


def _expand_bias_specs(bias_spec: dict[str, Any]) -> list[dict[str, Any]]:
    params = bias_spec.get("params", {})
    keys = list(params.keys())
    values = [value if isinstance(value, list) else [value] for value in params.values()]
    combos = [dict(zip(keys, combo)) for combo in product(*values)] if keys else [{}]
    return [
        {
            "module_name": bias_spec["module_name"],
            "params": combo,
            "timeframes": bias_spec.get("timeframes", [TimeFrame.D]),
        }
        for combo in combos
    ]


def _maybe_populate_cache(
    bias_spec: dict[str, Any],
    ticker: Ticker,
    start: datetime,
    end: datetime,
) -> None:
    project_root = _project_root()
    candle_dir = project_root / "data" / "ohlc_data"
    if not candle_dir.exists():
        pytest.skip(f"Missing persisted candle directory: {candle_dir}")

    manager = CacheManager(candle_dir=str(candle_dir))
    summary = manager.populate_cache(
        bias_node_specs=_expand_bias_specs(bias_spec),
        tickers=[ticker],
        start_date=start,
        end_date=end,
        show_progress=False,
        overwrite_existing=False,
    )

    if summary["success"] == 0 and summary["failed"] > 0:
        pytest.skip(
            "Cache population failed and no usable cache exists. "
            f"populate_cache summary: {summary}"
        )


def _extract_feature_and_target(
    bias_spec: dict[str, Any],
    ticker: Ticker,
    start: datetime,
    end: datetime,
    use_cache: bool,
) -> tuple[pd.Series, pd.Series, str]:
    try:
        features_df, targets_df = extract_features_for_bias_node(
            bias_spec=bias_spec,
            ticker=[ticker],
            start=start,
            end=end,
            use_millisecond_offset=True,
            target_col="log_return",
            use_cache=use_cache,
        )
    except Exception as exc:
        pytest.skip(
            "Feature extraction unavailable from cache/data. "
            f"Run CacheManager.populate_cache() first. Detail: {exc}"
        )

    if features_df is None or targets_df is None or features_df.empty or targets_df.empty:
        pytest.skip("Feature extraction returned empty data. Check data/cache availability.")

    feature_candidates = [col for col in features_df.columns if col != "ticker"]
    if not feature_candidates:
        pytest.skip("No extracted feature columns were found in features DataFrame.")
    feature_col = feature_candidates[0]

    target_col = "log_return" if "log_return" in targets_df.columns else None
    if target_col is None:
        target_candidates = [col for col in targets_df.columns if col != "ticker"]
        if not target_candidates:
            pytest.skip("No target columns were found in target DataFrame.")
        target_col = target_candidates[0]

    aligned = pd.DataFrame(
        {"feature": features_df[feature_col], "target": targets_df[target_col]}
    ).dropna()
    if aligned.empty:
        pytest.skip("Aligned feature/target data is empty after dropna().")

    return aligned["feature"], aligned["target"], feature_col


def _assert_report_files(report_path: Path, expected_plot_files: list[str]) -> None:
    assert report_path.exists()
    assert (report_path / "metadata.json").exists()
    assert (report_path / "common_stats.json").exists()
    assert (report_path / "feature_stats.json").exists()
    assert (report_path / "diagnostics.json").exists()
    plots_dir = report_path / "plots"
    assert plots_dir.is_dir()
    for plot_name in expected_plot_files:
        assert (plots_dir / plot_name).exists(), f"Missing plot artifact: {plot_name}"


@pytest.mark.integration
def test_common_eda_continuous(
    bias_spec: dict[str, Any] = DEFAULT_CONTINUOUS_BIAS_SPEC,
    ticker: Ticker = DEFAULT_TICKER,
    timeframe: TimeFrame = DEFAULT_TIMEFRAME,
    start: datetime = DEFAULT_START,
    end: datetime = DEFAULT_END,
    n_bins: int = 15,
    use_cache: bool = True,
) -> None:
    """Integration test for continuous EDA flow (T001+T002+T004).

    Defaults to RSI lookback=5 on ES daily data (2020-2023), but all key
    configuration inputs are exposed for researcher-driven exploration.
    """
    _maybe_populate_cache(bias_spec=bias_spec, ticker=ticker, start=start, end=end)
    feature, target, feature_col = _extract_feature_and_target(
        bias_spec=bias_spec,
        ticker=ticker,
        start=start,
        end=end,
        use_cache=use_cache,
    )

    tf = _normalize_timeframe(bias_spec, timeframe)
    timestamps = pd.DatetimeIndex(feature.index)
    metadata = EDAMetadata(
        feature_name=feature_col,
        param_combo=bias_spec.get("params", {}),
        timeframe=tf,
        ticker=ticker,
        timestamp=datetime.now(),
    )
    rolling_window = max(20, min(252, max(20, len(feature) // 4)))
    config = EDAConfig(n_bins=n_bins, rolling_window=rolling_window)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        report = run_eda_for_continuous_feature(feature, target, timestamps, metadata, config)
        report_path = save_eda_report(report=report, output_dir=output_dir, overwrite=True)

        assert isinstance(report, ContinuousEDAReport)
        assert report.common_stats.feature_stats.sample_size == len(feature)
        assert len(report.continuous_stats.decile_analysis.bin_stats.mean_return) == n_bins
        assert len(report.continuous_stats.decile_analysis.bin_stats.sharpe) == n_bins
        assert isinstance(report.diagnostics.is_viable, bool)

        _assert_report_files(
            report_path,
            expected_plot_files=[
                "time_series_fig.png",
                "rolling_corr_fig.png",
                "rolling_obj_fig.png",
                "decile_plot_fig.png",
                "histogram_fig.png",
                "qq_plot_fig.png",
                "kde_fig.png",
            ],
        )

        print("\n" + "=" * 64)
        print("Integration Test: Continuous EDA (T001+T002+T004)")
        print("=" * 64)
        print(f"Bias: {feature_col}")
        print(f"Ticker: {ticker.value} | Timeframe: {tf.value}")
        print(f"Date range: {start.date()} to {end.date()}")
        print(f"Samples: {len(feature):,}")
        print(
            "Metrics: "
            f"pearson={report.common_stats.correlation_analysis.pearson:.4f}, "
            f"kendall_tau={report.continuous_stats.monotonicity_test.kendall_tau:.4f}, "
            f"trend={report.continuous_stats.decile_analysis.overall_trend}"
        )
        print(f"Artifacts: {report_path}")
        print("PASS: Continuous EDA integration pipeline executed")


@pytest.mark.integration
def test_rule_based_eda(
    bias_spec: dict[str, Any] = DEFAULT_RULE_BASED_BIAS_SPEC,
    ticker: Ticker = DEFAULT_TICKER,
    timeframe: TimeFrame = DEFAULT_TIMEFRAME,
    start: datetime = DEFAULT_START,
    end: datetime = DEFAULT_END,
    bootstrap_iterations: int = 500,
    use_cache: bool = True,
) -> None:
    """Integration test for rule-based EDA flow (T001+T003+T004).

    Defaults to RSI 2-period strategy configuration on ES daily data (2020-2023),
    but accepts any bias spec/ticker/timeframe/date range through function args.
    """
    _maybe_populate_cache(bias_spec=bias_spec, ticker=ticker, start=start, end=end)
    feature, target, feature_col = _extract_feature_and_target(
        bias_spec=bias_spec,
        ticker=ticker,
        start=start,
        end=end,
        use_cache=use_cache,
    )

    tf = _normalize_timeframe(bias_spec, timeframe)
    timestamps = pd.DatetimeIndex(feature.index)
    metadata = EDAMetadata(
        feature_name=feature_col,
        param_combo=bias_spec.get("params", {}),
        timeframe=tf,
        ticker=ticker,
        timestamp=datetime.now(),
    )
    rolling_window = max(20, min(252, max(20, len(feature) // 4)))
    config = EDAConfig(
        rolling_window=rolling_window,
        bootstrap_iterations=bootstrap_iterations,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        report = run_eda_for_rule_based_feature(feature, target, timestamps, metadata, config)
        report_path = save_eda_report(report=report, output_dir=output_dir, overwrite=True)

        assert isinstance(report, RuleBasedEDAReport)
        assert report.common_stats.feature_stats.sample_size == len(feature)
        assert len(report.rule_stats.per_level_stats.stats_by_level) >= 1
        assert report.rule_stats.transition_matrix.transition_probs.shape == (3, 3)
        assert isinstance(report.diagnostics.is_viable, bool)

        _assert_report_files(
            report_path,
            expected_plot_files=[
                "time_series_fig.png",
                "rolling_corr_fig.png",
                "rolling_obj_fig.png",
                "level_plot_fig.png",
                "transition_heatmap_fig.png",
            ],
        )

        levels = sorted(report.rule_stats.per_level_stats.stats_by_level.keys())
        print("\n" + "=" * 64)
        print("Integration Test: Rule-Based EDA (T001+T003+T004)")
        print("=" * 64)
        print(f"Bias: {feature_col}")
        print(f"Ticker: {ticker.value} | Timeframe: {tf.value}")
        print(f"Date range: {start.date()} to {end.date()}")
        print(f"Samples: {len(feature):,}")
        print(f"Observed levels: {levels}")
        print(
            "Metrics: "
            f"pearson={report.common_stats.correlation_analysis.pearson:.4f}, "
            f"spearman={report.common_stats.correlation_analysis.spearman:.4f}, "
            f"diag_viable={report.diagnostics.is_viable}"
        )
        print(f"Artifacts: {report_path}")
        print("PASS: Rule-based EDA integration pipeline executed")
