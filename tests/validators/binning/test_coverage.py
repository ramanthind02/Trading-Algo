import pandas as pd

from feature_selection.validators.binning.diagnostics import RegionMetadata, calculate_coverage


def test_calculate_coverage_rsi() -> None:
    feature_data = pd.Series([5.0, 12.0, 18.0, 27.0, 33.0, 41.0, 55.0, 70.0])
    regions = [
        RegionMetadata(
            start_bin=0,
            end_bin=1,
            bins=[0, 1],
            mean_sharpe=1.0,
            mean_t_stat=2.0,
            sample_count=100,
            feature_range=(10.0, 20.0),
        ),
        RegionMetadata(
            start_bin=2,
            end_bin=3,
            bins=[2, 3],
            mean_sharpe=0.8,
            mean_t_stat=2.1,
            sample_count=100,
            feature_range=(25.0, 50.0),
        ),
    ]

    assert calculate_coverage(regions, feature_data) == 62.5


def test_calculate_coverage_handles_empty_series() -> None:
    feature_data = pd.Series([None, None])

    assert calculate_coverage([], feature_data) == 0.0
