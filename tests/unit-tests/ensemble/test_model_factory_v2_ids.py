import pytest
import numpy as np
import pandas as pd

from feature_selection.base_models import ContinuousBinningModel
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


def test_factory_merges_bias_node_params_into_spec() -> None:
    config = _base_config("continuous_binning")
    config["bias_node_params"] = {"lookback": 7}
    model = create_base_model_from_config(config, use_cache=False)
    assert model.bias_node_spec["params"]["lookback"] == 7


def test_factory_attaches_members_from_new_schema() -> None:
    # Build a valid member fitted payload.
    member_model = ContinuousBinningModel(n_bins=3, strategy="long")
    idx = pd.date_range("2020-01-01", periods=120, freq="D")
    feature = pd.Series(np.sin(np.linspace(0, 8, len(idx))), index=idx, name="feat")
    target = pd.Series(np.random.default_rng(42).normal(0, 0.01, len(idx)), index=idx)
    member_model.fit(feature, target)
    member_payload = member_model.get_fitted_params()

    config = _base_config("continuous_binning")
    config["members"] = [
        {
            "member_name": "cb10",
            "binning_model_type": "continuous_binning",
            "binning_model_params": {"n_bins": 3, "selection_metric": "sharpe"},
            "requires_fit": True,
            "fitted_params": member_payload,
        }
    ]

    model = create_base_model_from_config(config, use_cache=False)
    assert len(model.members) == 1
    assert model.members[0][0] == "cb10"
    assert model.members[0][1].model_type == "continuous_binning"
    assert model.members[0][1].is_fitted_
