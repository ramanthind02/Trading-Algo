"""Integration tests for binning diagnostics using persisted pipeline data/cache."""

from __future__ import annotations

from datetime import datetime
from itertools import product
from pathlib import Path

import pandas as pd
import pytest

from feature_extraction.feature_extractor import extract_features_for_bias_node
from feature_selection.base_models.continuous_binning import ContinuousBinningModel
from feature_selection.validation.binning.diagnostics import (
    BinningSuccessCriteria,
    calculate_coverage,
    detect_region_shape,
    extract_region_metadata,
    validate_binning_success,
)
from utils.cache.cache_manager import CacheManager
from utils.core.enums import Ticker, TimeFrame

START_DATE = datetime(2000, 1, 1)
END_DATE = datetime(2024, 12, 31)
PERMUTATION_REPS = 100
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
        target_col="log_return",
        use_cache=USE_CACHE,
    )
    return features_df, targets_df


def _print_run_summary(
    *,
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    feature_col: str,
    success: bool | None = None,
    coverage: float | None = None,
    region_shapes: list[str] | None = None,
) -> None:
    print("\n" + "=" * 72)
    print("RSI BINNING DIAGNOSTICS INTEGRATION SUMMARY")
    print("=" * 72)
    print(f"Tickers: {[ticker.name for ticker in ENSEMBLE_TICKERS]}")
    print(f"Date range: {START_DATE.date()} -> {END_DATE.date()}")
    print(f"USE_CACHE={USE_CACHE} POPULATE_CACHE={POPULATE_CACHE}")
    print(f"Feature rows: {len(features_df)}  Target rows: {len(targets_df)}")
    print(f"Feature column: {feature_col}")
    if success is not None:
        print(f"Validation success: {success}")
    if coverage is not None:
        print(f"Coverage: {coverage:.4f}%")
    if region_shapes is not None:
        print(f"Region shapes: {region_shapes}")
    print("=" * 72)


def test_rsi_pipeline_cache_extracts_features_and_targets() -> None:
    project_root = _project_root()
    _maybe_populate_cache(project_root)

    features_df, targets_df = _extract_rsi_features()

    rsi_cols = [col for col in features_df.columns if col.startswith("rsi_signal_D_lookback_")]
    assert rsi_cols == ["rsi_signal_D_lookback_5"]
    assert len(features_df) > 0
    assert len(targets_df) > 0
    assert "log_return" in targets_df.columns
    _print_run_summary(
        features_df=features_df,
        targets_df=targets_df,
        feature_col="rsi_signal_D_lookback_5",
    )


def test_rsi_pipeline_binning_diagnostics_end_to_end() -> None:
    project_root = _project_root()
    _maybe_populate_cache(project_root)

    features_df, targets_df = _extract_rsi_features()
    feature_col = "rsi_signal_D_lookback_5"
    assert feature_col in features_df.columns

    # features_df and targets_df share the same row-aligned index from extraction.
    # With multi-ticker data the datetime index has duplicates (one row per ticker),
    # so we cannot use .loc reindexing. Instead, build an aligned mask and select.
    valid_mask = features_df[feature_col].notna()
    feature_series = features_df.loc[valid_mask, feature_col].copy()
    feature_series.name = feature_col
    target_series = targets_df.loc[valid_mask, "log_return"].copy()

    model = ContinuousBinningModel(
        bin_counts=[10, 5, 3],  # Test grid search
        strategy="long",
    )
    model.fit(feature_series, target_series)

    # Check that grid search worked
    assert model.is_fitted_
    assert model.n_bins in [10, 5, 3]
    assert hasattr(model, "selected_bins_")
    assert len(model.selected_bins_) > 0

    # Legacy diagnostics still work (though regions may be empty with new approach)
    criteria = BinningSuccessCriteria(metric_threshold=0.0, t_threshold=0.5, min_region_width=1)
    success = validate_binning_success(model, criteria)
    regions = extract_region_metadata(model)
    coverage = calculate_coverage(regions, feature_series)
    region_shapes = [detect_region_shape(region, n_bins=model.n_bins) for region in regions]

    assert isinstance(success, bool)
    assert 0.0 <= coverage <= 100.0
    # Note: significant_regions_ is empty in new approach, so regions may be []
    assert all(shape in {"tail", "hump"} for shape in region_shapes)
    _print_run_summary(
        features_df=features_df,
        targets_df=targets_df,
        feature_col=feature_col,
        success=success,
        coverage=coverage,
        region_shapes=region_shapes,
    )


if __name__ == "__main__":
    # Allows researchers to run this file directly and inspect output outside pytest.
    test_rsi_pipeline_cache_extracts_features_and_targets()
    test_rsi_pipeline_binning_diagnostics_end_to_end()
