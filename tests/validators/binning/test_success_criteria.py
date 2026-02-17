import pytest

from feature_selection.base_models.continuous_binning import ContinuousBinningModel
from feature_selection.validators.binning.diagnostics import (
    BinningSuccessCriteria,
    validate_binning_success,
)


def _build_model_with_regions() -> ContinuousBinningModel:
    model = ContinuousBinningModel(n_bins=5)
    model.is_fitted_ = True
    model.bin_stats_ = {
        0: {
            "selection_metric_long": 0.1,
            "selection_metric_short": -0.1,
            "t_stat": 0.4,
            "sharpe": 0.1,
            "count": 50,
            "feature_min": 0.0,
            "feature_max": 20.0,
        },
        1: {
            "selection_metric_long": 0.8,
            "selection_metric_short": -0.8,
            "t_stat": 2.5,
            "sharpe": 1.2,
            "count": 60,
            "feature_min": 20.0,
            "feature_max": 40.0,
        },
        2: {
            "selection_metric_long": 0.7,
            "selection_metric_short": -0.7,
            "t_stat": 2.2,
            "sharpe": 1.0,
            "count": 55,
            "feature_min": 40.0,
            "feature_max": 60.0,
        },
        3: {
            "selection_metric_long": 0.2,
            "selection_metric_short": -0.2,
            "t_stat": 0.8,
            "sharpe": 0.2,
            "count": 65,
            "feature_min": 60.0,
            "feature_max": 80.0,
        },
    }
    model.significant_regions_ = [{"start_bin": 1, "end_bin": 2, "bins": [1, 2]}]
    return model


def test_validate_binning_success_with_valid_regions() -> None:
    model = _build_model_with_regions()
    criteria = BinningSuccessCriteria(metric_threshold=0.5, t_threshold=2.0, min_region_width=2)

    assert validate_binning_success(model, criteria) is True


def test_validate_binning_success_no_regions() -> None:
    model = _build_model_with_regions()
    model.significant_regions_ = []
    criteria = BinningSuccessCriteria(metric_threshold=0.5, t_threshold=2.0, min_region_width=2)

    assert validate_binning_success(model, criteria) is False


def test_validate_binning_success_rejects_single_bin_spikes() -> None:
    model = _build_model_with_regions()
    model.significant_regions_ = [{"start_bin": 1, "end_bin": 1, "bins": [1]}]
    criteria = BinningSuccessCriteria(metric_threshold=0.5, t_threshold=2.0, min_region_width=2)

    assert validate_binning_success(model, criteria) is False


def test_validate_binning_success_requires_fitted_model() -> None:
    model = _build_model_with_regions()
    model.is_fitted_ = False
    criteria = BinningSuccessCriteria(metric_threshold=0.5, t_threshold=2.0, min_region_width=2)

    with pytest.raises(ValueError, match="fitted"):
        validate_binning_success(model, criteria)
