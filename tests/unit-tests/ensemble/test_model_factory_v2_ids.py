import pytest
from ensemble.ensemble_utils import create_base_model_from_config
from feature_selection.domain_discrete import DomainDiscreteMigrationError


def _base_config(model_type: str) -> dict:
    return {
        "name": "rsi_signal_D_domain_discrete",
        "model_type": model_type,
        "feature_column": "rsi_signal_D_lookback_14",
        "strategy": "long",
        "bias_node_spec": {
            "module_name": "domain_discrete",
            "timeframes": ["D"],
            "params": {
                "source_bias_node_spec": {
                    "module_name": "rsi",
                    "timeframes": ["D"],
                    "params": {"lookback": 14},
                },
                "ticker_scope": {"tickers": ["ES"], "scope_name": "ES"},
                "edges": [-0.5, 0.5],
                "n_bins": 3,
                "long_bins": [2],
                "short_bins": [],
                "direction": "long",
                "spec_version": "v1",
            },
        },
        "tickers": ["ES"],
    }


def test_factory_accepts_domain_discrete_id() -> None:
    model = create_base_model_from_config(_base_config("domain_discrete"), use_cache=False)
    assert model.model_type == "domain_discrete"
    assert model.bias_node_spec["module_name"] == "domain_discrete"
    assert model.bias_node_spec["params"]["spec_version"] == "v1"


def test_factory_rejects_old_fitted_schema() -> None:
    with pytest.raises(DomainDiscreteMigrationError, match="domain-discrete cutover"):
        create_base_model_from_config(
            _base_config("continuous_binning"),
            use_cache=False,
        )


def test_factory_rejects_legacy_fitted_params_payload() -> None:
    with pytest.raises(DomainDiscreteMigrationError, match="domain-discrete cutover"):
        create_base_model_from_config(
            _base_config("domain_discrete"),
            fitted_params={"model_version": "binning_v2"},
            use_cache=False,
        )


def test_factory_rejects_members_schema() -> None:
    config = _base_config("domain_discrete")
    config["members"] = [
        {
            "member_name": "cb10",
            "model_type": "domain_discrete",
        }
    ]
    with pytest.raises(DomainDiscreteMigrationError, match="domain-discrete cutover"):
        create_base_model_from_config(config, use_cache=False)
