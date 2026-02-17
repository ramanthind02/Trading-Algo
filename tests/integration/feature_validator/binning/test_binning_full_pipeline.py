"""Comprehensive integration test for binning diagnostics T005-T008."""

from __future__ import annotations

import json
from datetime import datetime
from itertools import product
from pathlib import Path

import pandas as pd
import pytest

from feature_extraction.feature_extractor import extract_features_for_bias_node
from feature_selection.base_models.continuous_binning import ContinuousBinningModel
from feature_selection.validators.binning import (
    BinningSuccessCriteria,
    display_report_summary,
    generate_binning_report,
    save_report,
)
from utils.cache_manager import CacheManager
from utils.enums import Ticker, TimeFrame

START_DATE = datetime(2000, 1, 1)
END_DATE = datetime(2024, 12, 31)
USE_CACHE = True
POPULATE_CACHE = True

ENSEMBLE_TICKERS = [Ticker.ES, Ticker.NQ, Ticker.YM, Ticker.RTY]
BIAS_SPEC = {
    "module_name": "rsi",
    "timeframes": [TimeFrame.D],
    "params": {"lookback": [5]},
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _maybe_populate_cache(project_root: Path) -> None:
    if not POPULATE_CACHE:
        return

    candle_dir = project_root / "data" / "ohlc_data"
    if not candle_dir.exists():
        pytest.skip(f"Missing persisted candle directory: {candle_dir}")

    manager = CacheManager(candle_dir=str(candle_dir))

    params = BIAS_SPEC.get("params", {})
    keys = list(params.keys())
    values = [value if isinstance(value, list) else [value] for value in params.values()]
    param_combos = [dict(zip(keys, combo)) for combo in product(*values)] if keys else [{}]

    all_specs = [
        {
            "module_name": BIAS_SPEC["module_name"],
            "params": combo,
            "timeframes": BIAS_SPEC.get("timeframes", [TimeFrame.D]),
        }
        for combo in param_combos
    ]

    result = manager.populate_cache(
        bias_node_specs=all_specs,
        tickers=ENSEMBLE_TICKERS,
        start_date=START_DATE,
        end_date=END_DATE,
        show_progress=False,
        overwrite_existing=False,
    )
    assert result["failed"] == 0, f"Cache population failures: {result}"


def _extract_rsi_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    features_df, targets_df = extract_features_for_bias_node(
        bias_spec=BIAS_SPEC,
        ticker=ENSEMBLE_TICKERS,
        start=START_DATE,
        end=END_DATE,
        use_millisecond_offset=True,
        target_col="log_return",
        use_cache=USE_CACHE,
    )
    return features_df, targets_df


@pytest.mark.integration
def test_binning_full_pipeline_integration() -> None:
    """Comprehensive test of T005-T008 binning diagnostics pipeline.

    Tests full workflow:
    1. Extract RSI features from cache
    2. Fit ContinuousBinningModel
    3. Generate comprehensive diagnostics report (T005-T008)
    4. Save plots and JSON report to outputs directory
    5. Display terminal summary for manual verification

    Manual verification:
    - Check terminal output for readable summary
    - Inspect plots in tests/integration/outputs/binning/rsi_lookback_5/
    - Verify JSON report contains all metadata
    """
    project_root = _project_root()
    _maybe_populate_cache(project_root)

    # Extract features
    features_df, targets_df = _extract_rsi_features()
    feature_col = "rsi_signal_D_lookback_5"
    assert feature_col in features_df.columns

    feature_series = features_df[feature_col].copy()
    feature_series.name = feature_col
    target_series = targets_df.loc[feature_series.index, "log_return"]

    print("\n" + "=" * 72)
    print("BINNING DIAGNOSTICS INTEGRATION TEST")
    print("=" * 72)
    print(f"Feature: {feature_col}")
    print(f"Tickers: {[ticker.name for ticker in ENSEMBLE_TICKERS]}")
    print(f"Date range: {START_DATE.date()} -> {END_DATE.date()}")
    print(f"Samples: {len(feature_series)} observations")

    # Fit binning model
    model = ContinuousBinningModel(
        n_bins=20,
        selection_metric="sharpe",
        strategy="long",
        metric_threshold=0.0,
        t_threshold=0.5,
        min_region_width=2,
    )
    model.fit(feature_series, target_series)

    # Generate comprehensive report (T005-T008)
    criteria = BinningSuccessCriteria(
        metric_threshold=0.0, t_threshold=0.5, min_region_width=2
    )
    report = generate_binning_report(
        model=model,
        feature_data=feature_series,
        criteria=criteria,
        strategy="long",
        max_regions=1,
        direction_filter="long",
    )

    # Verify report structure
    assert isinstance(report.success_verdict, bool)
    assert report.failure_mode in [
        "no_regions",
        "isolated_spikes",
        "insufficient_edge",
        "none",
    ]
    assert isinstance(report.shape_summary, dict)
    assert isinstance(report.coverage_breakdown, list)
    assert 0.0 <= report.total_coverage_pct <= 100.0
    assert "heatmap" in report.diagnostic_plots
    assert "boundaries" in report.diagnostic_plots
    assert "multiplier_curve" in report.diagnostic_plots
    assert "panel" in report.diagnostic_plots

    # Save to outputs directory
    output_dir = (
        project_root / "tests" / "integration" / "outputs" / "binning" / "rsi_lookback_5"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = save_report(report, str(output_dir))
    assert Path(report_path).exists()

    # Verify plots were saved
    plot_dir = output_dir / feature_col / "plots"
    assert plot_dir.exists()

    plot_files = [
        f"heatmap_{feature_col}.png",
        f"boundaries_{feature_col}.png",
        f"multiplier_curve_{feature_col}.png",
        f"panel_{feature_col}.png",
    ]
    for plot_file in plot_files:
        plot_path = plot_dir / plot_file
        assert plot_path.exists(), f"Missing plot: {plot_file}"

    # Display terminal summary
    display_report_summary(report)

    print(f"\nPlots saved to: {plot_dir}")
    for plot_file in plot_files:
        print(f"  ✓ {plot_file}")
    print(f"\nReport saved to: {report_path}")
    print("=" * 72)

    # Verify JSON structure
    with open(report_path) as f:
        report_json = json.load(f)

    assert "feature_column" in report_json
    assert "success_verdict" in report_json
    assert "failure_mode" in report_json
    assert "shape_summary" in report_json
    assert "total_coverage_pct" in report_json
    assert report_json["feature_column"] == feature_col


if __name__ == "__main__":
    # Allows researchers to run this file directly
    test_binning_full_pipeline_integration()
