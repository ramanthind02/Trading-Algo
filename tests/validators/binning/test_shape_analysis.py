import pandas as pd
import pytest

from feature_selection.validators.binning.diagnostics import RegionMetadata
from feature_selection.validators.binning.shape_analysis import (
    AdjacencyAnalysis,
    RegionCoverage,
    ShapeClassification,
    analyze_multi_region_shapes,
    calculate_region_coverage_breakdown,
    classify_region_shape,
    detect_region_adjacency,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _region(
    start_bin: int,
    end_bin: int,
    mean_sharpe: float,
    feature_range: tuple[float, float] = (0.0, 10.0),
    sample_count: int = 100,
) -> RegionMetadata:
    bins = list(range(start_bin, end_bin + 1))
    return RegionMetadata(
        start_bin=start_bin,
        end_bin=end_bin,
        bins=bins,
        mean_sharpe=mean_sharpe,
        mean_t_stat=2.5,
        sample_count=sample_count,
        feature_range=feature_range,
    )


# ---------------------------------------------------------------------------
# classify_region_shape
# ---------------------------------------------------------------------------

def test_classify_long_tail() -> None:
    region = _region(start_bin=0, end_bin=2, mean_sharpe=0.9)
    result = classify_region_shape(region, n_bins=15)

    assert result.shape_type == "long_tail"
    assert result.touches_extreme is True
    assert result.is_monotonic is True
    assert result.direction == "long"


def test_classify_short_tail() -> None:
    region = _region(start_bin=13, end_bin=14, mean_sharpe=-0.7)
    result = classify_region_shape(region, n_bins=15)

    assert result.shape_type == "short_tail"
    assert result.touches_extreme is True
    assert result.is_monotonic is True
    assert result.direction == "short"


def test_classify_long_hump() -> None:
    region = _region(start_bin=5, end_bin=7, mean_sharpe=0.6)
    result = classify_region_shape(region, n_bins=15)

    assert result.shape_type == "long_hump"
    assert result.touches_extreme is False
    assert result.is_monotonic is False
    assert result.direction == "long"


def test_classify_short_hump() -> None:
    region = _region(start_bin=5, end_bin=7, mean_sharpe=-0.5)
    result = classify_region_shape(region, n_bins=15)

    assert result.shape_type == "short_hump"
    assert result.touches_extreme is False
    assert result.is_monotonic is False
    assert result.direction == "short"


def test_classify_tail_at_high_end() -> None:
    """Tail touching last bin with positive Sharpe is long_tail."""
    region = _region(start_bin=12, end_bin=14, mean_sharpe=1.2)
    result = classify_region_shape(region, n_bins=15)

    assert result.shape_type == "long_tail"
    assert result.touches_extreme is True


# ---------------------------------------------------------------------------
# analyze_multi_region_shapes
# ---------------------------------------------------------------------------

def test_analyze_multi_region_shapes_counts() -> None:
    regions = [
        _region(start_bin=0, end_bin=2, mean_sharpe=0.8),   # long_tail
        _region(start_bin=5, end_bin=7, mean_sharpe=-0.5),   # short_hump
        _region(start_bin=12, end_bin=14, mean_sharpe=-0.6), # short_tail
    ]
    counts = analyze_multi_region_shapes(regions, n_bins=15)

    assert counts == {"long_tail": 1, "short_hump": 1, "short_tail": 1}


def test_analyze_multi_region_shapes_empty() -> None:
    counts = analyze_multi_region_shapes([], n_bins=15)
    assert counts == {}


# ---------------------------------------------------------------------------
# calculate_region_coverage_breakdown
# ---------------------------------------------------------------------------

def test_calculate_region_coverage_breakdown_order() -> None:
    """Cumulative coverage must be monotonically increasing."""
    regions = [
        _region(start_bin=0, end_bin=1, mean_sharpe=0.5, feature_range=(10.0, 20.0)),
        _region(start_bin=5, end_bin=6, mean_sharpe=-0.3, feature_range=(50.0, 60.0)),
    ]
    feature_data = pd.Series([15.0, 18.0, 30.0, 55.0, 58.0, 70.0, 80.0, 90.0])
    coverages = calculate_region_coverage_breakdown(regions, feature_data)

    assert len(coverages) == 2
    assert coverages[0].cumulative_coverage_pct <= coverages[1].cumulative_coverage_pct


def test_calculate_region_coverage_breakdown_sum() -> None:
    """Sum of individual coverages equals total (non-overlapping regions)."""
    regions = [
        _region(start_bin=0, end_bin=1, mean_sharpe=0.5, feature_range=(10.0, 20.0)),
        _region(start_bin=5, end_bin=6, mean_sharpe=-0.3, feature_range=(50.0, 60.0)),
    ]
    feature_data = pd.Series([15.0, 18.0, 30.0, 55.0, 58.0, 70.0, 80.0, 90.0])
    coverages = calculate_region_coverage_breakdown(regions, feature_data)

    total_individual = sum(c.individual_coverage_pct for c in coverages)
    assert coverages[-1].cumulative_coverage_pct == pytest.approx(total_individual, abs=1e-9)


def test_calculate_region_coverage_breakdown_empty() -> None:
    feature_data = pd.Series([1.0, 2.0, 3.0])
    coverages = calculate_region_coverage_breakdown([], feature_data)
    assert coverages == []


def test_calculate_region_coverage_breakdown_values() -> None:
    """Known coverage values with simple data."""
    regions = [
        _region(start_bin=0, end_bin=1, mean_sharpe=0.5, feature_range=(0.0, 25.0)),
    ]
    # 4 out of 8 values fall in [0, 25]
    feature_data = pd.Series([5.0, 10.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0])
    coverages = calculate_region_coverage_breakdown(regions, feature_data)

    assert len(coverages) == 1
    assert coverages[0].individual_coverage_pct == pytest.approx(50.0)
    assert coverages[0].cumulative_coverage_pct == pytest.approx(50.0)
    assert coverages[0].region_id == 0


# ---------------------------------------------------------------------------
# detect_region_adjacency
# ---------------------------------------------------------------------------

def test_detect_region_adjacency_gaps() -> None:
    """Gap sizes match expected bin distances between regions."""
    regions = [
        _region(start_bin=0, end_bin=2, mean_sharpe=0.5),
        _region(start_bin=5, end_bin=7, mean_sharpe=-0.5),
        _region(start_bin=12, end_bin=14, mean_sharpe=0.3),
    ]
    result = detect_region_adjacency(regions)

    assert result.gap_sizes == [2, 4]


def test_detect_region_adjacency_connected() -> None:
    """All gaps <= 2 bins means connected."""
    regions = [
        _region(start_bin=0, end_bin=2, mean_sharpe=0.5),
        _region(start_bin=4, end_bin=6, mean_sharpe=-0.3),
    ]
    result = detect_region_adjacency(regions)

    assert result.gap_sizes == [1]
    assert result.is_connected is True


def test_detect_region_adjacency_isolated() -> None:
    """Large gaps mean not connected, high isolation score."""
    regions = [
        _region(start_bin=0, end_bin=1, mean_sharpe=0.5),
        _region(start_bin=12, end_bin=14, mean_sharpe=-0.3),
    ]
    result = detect_region_adjacency(regions)

    assert result.gap_sizes == [10]
    assert result.is_connected is False
    assert result.isolation_score > 0.0


def test_detect_region_adjacency_single_region() -> None:
    """Single region has no gaps."""
    regions = [_region(start_bin=3, end_bin=5, mean_sharpe=0.5)]
    result = detect_region_adjacency(regions)

    assert result.gap_sizes == []
    assert result.is_connected is True
    assert result.isolation_score == 0.0


def test_detect_region_adjacency_empty() -> None:
    result = detect_region_adjacency([])

    assert result.gap_sizes == []
    assert result.is_connected is True
    assert result.isolation_score == 0.0
