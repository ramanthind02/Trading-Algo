from feature_selection.base_models.continuous_binning import ContinuousBinningModel
from feature_selection.validators.binning.diagnostics import extract_region_metadata


def _build_metadata_fixture() -> ContinuousBinningModel:
    model = ContinuousBinningModel(n_bins=5)
    model.is_fitted_ = True
    model.bin_stats_ = {
        0: {
            "selection_metric_long": 0.1,
            "selection_metric_short": -0.1,
            "t_stat": 0.5,
            "sharpe": 0.2,
            "count": 40,
            "feature_min": 0.0,
            "feature_max": 10.0,
        },
        1: {
            "selection_metric_long": 0.8,
            "selection_metric_short": -0.8,
            "t_stat": 2.0,
            "sharpe": 1.0,
            "count": 50,
            "feature_min": 10.0,
            "feature_max": 20.0,
        },
        2: {
            "selection_metric_long": 0.9,
            "selection_metric_short": -0.9,
            "t_stat": -2.4,
            "sharpe": 1.4,
            "count": 70,
            "feature_min": 20.0,
            "feature_max": 30.0,
        },
    }
    model.significant_regions_ = [{"start_bin": 1, "end_bin": 2, "bins": [1, 2]}]
    return model


def test_extract_region_metadata_rsi() -> None:
    model = _build_metadata_fixture()

    metadata = extract_region_metadata(model)

    assert len(metadata) == 1
    assert metadata[0].start_bin == 1
    assert metadata[0].end_bin == 2
    assert metadata[0].bins == [1, 2]
    assert metadata[0].mean_sharpe == 1.2
    assert metadata[0].mean_t_stat == 2.2
    assert metadata[0].sample_count == 120
    assert metadata[0].feature_range == (10.0, 30.0)
