import pytest

from feature_selection.validators.binning.diagnostics import RegionMetadata, detect_region_shape


def test_detect_tail() -> None:
    region = RegionMetadata(
        start_bin=0,
        end_bin=2,
        bins=[0, 1, 2],
        mean_sharpe=0.9,
        mean_t_stat=2.5,
        sample_count=100,
        feature_range=(0.0, 20.0),
    )

    assert detect_region_shape(region, n_bins=5) == "tail"


def test_detect_hump() -> None:
    region = RegionMetadata(
        start_bin=1,
        end_bin=3,
        bins=[1, 2, 3],
        mean_sharpe=0.8,
        mean_t_stat=2.2,
        sample_count=120,
        feature_range=(20.0, 80.0),
    )

    assert detect_region_shape(region, n_bins=5) == "hump"


def test_detect_shape_rejects_invalid_n_bins() -> None:
    region = RegionMetadata(
        start_bin=0,
        end_bin=0,
        bins=[0],
        mean_sharpe=0.4,
        mean_t_stat=1.1,
        sample_count=20,
        feature_range=(0.0, 10.0),
    )

    with pytest.raises(ValueError, match="n_bins"):
        detect_region_shape(region, n_bins=0)
