# feature_research/continuous_binning/pipeline.py
"""Core EDA pipeline for continuous binning research.

Entry point for tests and scripts alike — import ``run_continuous_eda_pipeline``
rather than duplicating this logic.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")  # non-interactive backend (safe for scripts and tests)

if TYPE_CHECKING:
    from feature_research.continuous_binning.config import ResearchConfig

from feature_research.continuous_binning.data_loader import (
    expand_bias_specs,
    load_features_for_combo,
    param_combo_label,
    populate_cache_if_needed,
)
from feature_research.walkforward.io import write_walkforward_artifacts
from feature_research.walkforward.runner import run_walkforward_research
from feature_research.walkforward.visualization import plot_fold_timeline, plot_selection_stability
from feature_selection.eda.eda_dataclasses import EDAConfig, EDAMetadata
from feature_selection.eda.eda_reporter import run_eda_for_continuous_feature, save_eda_report
from utils.enums import TimeFrame


def _normalize_timeframe(bias_spec: dict[str, Any], fallback: TimeFrame = TimeFrame.D) -> TimeFrame:
    raw = bias_spec.get("timeframes", [fallback])
    first = raw[0] if isinstance(raw, list) else raw
    return TimeFrame[first] if isinstance(first, str) else first


def _combo_key(params: dict[str, object]) -> tuple[tuple[str, object], ...]:
    return tuple(sorted(params.items(), key=lambda item: item[0]))


def _normalize_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    datetime_index = pd.DatetimeIndex(index)
    return datetime_index.tz_localize(None) if datetime_index.tz is not None else datetime_index


def _normalize_series_datetime_index(series: pd.Series) -> pd.Series:
    if not isinstance(series.index, pd.DatetimeIndex):
        return series
    normalized_series = series.copy()
    normalized_series.index = _normalize_datetime_index(series.index)
    return normalized_series


def _build_walkforward_evaluator(
    combo_returns: dict[tuple[tuple[str, object], ...], pd.Series],
) -> Callable[[pd.DataFrame, pd.Series, dict[str, object]], pd.Series]:
    def evaluate_param_combo(
        fold_candles: pd.DataFrame,
        _fold_target: pd.Series,
        params: dict[str, object],
    ) -> pd.Series:
        returns = combo_returns[_combo_key(params)]
        fold_returns = returns.reindex(fold_candles.index).dropna()
        return fold_returns if not fold_returns.empty else pd.Series(dtype=float)

    return evaluate_param_combo


def run_continuous_eda_pipeline(
    config: "ResearchConfig",
    output_dir: Path,
) -> dict[str, Path]:
    """Run the full continuous-feature EDA pipeline for every param combo in config.

    For each param combo:
    1. Extract feature + target data (cache-backed).
    2. Build ``EDAMetadata`` and ``EDAConfig``.
    3. Run ``run_eda_for_continuous_feature`` to produce a ``ContinuousEDAReport``.
    4. Save the report under ``output_dir / param_label /``.
    5. Print a one-line summary.

    Parameters
    ----------
    config : ResearchConfig
        Researcher-defined settings (tickers, dates, bias_spec, cache flags).
    output_dir : Path
        Root directory for output reports.  Created if it does not exist.
        Each param combo writes to ``output_dir / param_label /``.

    Returns
    -------
    dict[str, Path]
        Mapping of ``param_label`` → saved report path for each successful combo.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    populate_cache_if_needed(config)

    expanded = expand_bias_specs(config.bias_spec)
    tf = _normalize_timeframe(config.bias_spec)

    results: dict[str, Path] = {}
    combo_returns: dict[tuple[tuple[str, object], ...], pd.Series] = {}
    successful_param_grid: list[dict[str, object]] = []
    reference_index: pd.DatetimeIndex | None = None

    print(f"\n{'='*64}")
    print(f"Continuous EDA Pipeline: {config.bias_spec['module_name'].upper()}")
    print(f"Tickers : {[t.name for t in config.tickers]}")
    print(f"Period  : {config.start.date()} -> {config.end.date()}")
    print(f"Target  : {config.target_col}  |  Strategy: {config.strategy}")
    print(f"Combos  : {len(expanded)}")
    print(f"Output  : {output_dir}")
    print(f"{'='*64}\n")

    for single_spec in expanded:
        combo = single_spec["params"]
        label = param_combo_label(combo)

        data = load_features_for_combo(single_spec, config)
        if data is None:
            print(f"  [{label}] SKIP -- no data")
            continue

        feature, target, feature_col = data
        paired = pd.DataFrame({"feature": feature, "target": target}).dropna()
        if paired.empty:
            print(f"  [{label}] SKIP -- aligned feature/target empty")
            continue

        feature = paired["feature"]
        target = paired["target"]
        timestamps = pd.DatetimeIndex(feature.index)

        rolling_window = max(20, min(252, len(feature) // 4))

        metadata = EDAMetadata(
            feature_name=feature_col,
            param_combo=combo,
            timeframe=tf,
            ticker=config.tickers[0],
            timestamp=datetime.now(),
        )
        eda_config = EDAConfig(n_bins=15, rolling_window=rolling_window)

        report = run_eda_for_continuous_feature(feature, target, timestamps, metadata, eda_config)

        combo_output_dir = output_dir / label
        combo_output_dir.mkdir(parents=True, exist_ok=True)

        saved_path = save_eda_report(report=report, output_dir=combo_output_dir, overwrite=True)
        results[label] = saved_path

        combo_returns[_combo_key(combo)] = _normalize_series_datetime_index(feature.mul(target))
        successful_param_grid.append(dict(combo))
        if reference_index is None:
            reference_index = _normalize_datetime_index(target.index)

        pearson = report.common_stats.correlation_analysis.pearson
        tau = report.continuous_stats.monotonicity_test.kendall_tau
        trend = report.continuous_stats.decile_analysis.overall_trend
        viable = "VIABLE" if report.diagnostics.is_viable else f"FLAGS({len(report.diagnostics.red_flags)})"
        warnings_count = len(report.diagnostics.warnings)

        print(
            f"  [{label}] n={len(feature):,}  pearson={pearson:+.3f}  "
            f"tau={tau:+.3f}  trend={trend}  {viable}  warnings={warnings_count}"
        )

    if config.walkforward.enabled and reference_index is not None and successful_param_grid:
        reference_target = pd.Series(0.0, index=reference_index, name="walkforward_target")
        reference_candles = pd.DataFrame({"close": reference_target}, index=reference_index)
        walkforward_report = run_walkforward_research(
            candles_df=reference_candles,
            target=reference_target,
            feature_type="continuous",
            module_name=str(config.bias_spec["module_name"]),
            config=config.walkforward,
            param_grid=successful_param_grid,
            evaluate_param_combo=_build_walkforward_evaluator(combo_returns),
        )
        stability_figure, _ = plot_selection_stability(
            selection_summary_df=walkforward_report.selection_summary_df,
            top_k=config.walkforward.top_k,
        )
        timeline_figure, _ = plot_fold_timeline(folds_df=walkforward_report.folds_df)
        write_walkforward_artifacts(
            report=walkforward_report,
            walkforward_stability_figure=stability_figure,
            fold_timeline_figure=timeline_figure,
            feature_type="continuous",
            module_name=str(config.bias_spec["module_name"]),
            root_dir=config.walkforward.output_root,
        )
        plt.close(stability_figure)
        plt.close(timeline_figure)

    print(f"\nDone. {len(results)}/{len(expanded)} combos succeeded -> {output_dir}\n")
    return results
