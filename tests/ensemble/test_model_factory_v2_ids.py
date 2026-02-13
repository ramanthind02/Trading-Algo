import pytest

from ensemble.ensemble_utils import create_base_model_from_config


def _base_config(model_type: str) -> dict:
    return {
        "name": "rsi_signal_D_lookback_14_long",
        "model_type": model_type,
        "feature_column": "rsi_signal_D_lookback_14",
        "strategy": "long",
        "constructor_params": {"n_bins": 3, "selection_metric": "sharpe"},
        "bias_node_spec": {
            "module_name": "rsi",
            "timeframes": ["D"],
            "params": {"lookback": 14},
        },
        "tickers": ["ES"],
    }


def test_factory_accepts_continuous_binning_id() -> None:
    model = create_base_model_from_config(_base_config("continuous_binning"), use_cache=False)
    assert model.binning_model.model_type == "continuous_binning"


def test_factory_rejects_old_fitted_schema() -> None:
    with pytest.raises(ValueError, match="binning_v2"):
        create_base_model_from_config(
            _base_config("continuous_binning"),
            fitted_params={"thresholds": [1.0], "bin_stats": {}},
            use_cache=False,
        )
