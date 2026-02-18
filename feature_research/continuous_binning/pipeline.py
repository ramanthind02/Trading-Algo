# feature_research/continuous_binning/pipeline.py
"""Core EDA pipeline for continuous binning research.

Entry point for tests and scripts alike — import ``run_continuous_eda_pipeline``
rather than duplicating this logic.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib
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
from feature_selection.eda.eda_dataclasses import EDAConfig, EDAMetadata
from feature_selection.eda.eda_reporter import run_eda_for_continuous_feature, save_eda_report
from utils.enums import TimeFrame


def _normalize_timeframe(bias_spec: dict[str, Any], fallback: TimeFrame = TimeFrame.D) -> TimeFrame:
    raw = bias_spec.get("timeframes", [fallback])
    first = raw[0] if isinstance(raw, list) else raw
    return TimeFrame[first] if isinstance(first, str) else first


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

        pearson = report.common_stats.correlation_analysis.pearson
        tau = report.continuous_stats.monotonicity_test.kendall_tau
        trend = report.continuous_stats.decile_analysis.overall_trend
        viable = "VIABLE" if report.diagnostics.is_viable else f"FLAGS({len(report.diagnostics.red_flags)})"
        warnings_count = len(report.diagnostics.warnings)

        print(
            f"  [{label}] n={len(feature):,}  pearson={pearson:+.3f}  "
            f"tau={tau:+.3f}  trend={trend}  {viable}  warnings={warnings_count}"
        )

    print(f"\nDone. {len(results)}/{len(expanded)} combos succeeded -> {output_dir}\n")
    return results
