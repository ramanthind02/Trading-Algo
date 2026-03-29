from __future__ import annotations

import pandas as pd
import pytest

from feature_selection.domain_discrete import load_domain_discrete_spec
from feature_selection.base_models.feature_base_model import BaseModel
from utils.core.enums import Ticker


def _frozen_spec(*, tickers: list[str] | None = None) -> dict:
    return {
        "source_bias_node_spec": {
            "module_name": "buy_hold",
            "timeframes": ["D"],
            "params": {},
        },
        "ticker_scope": {"tickers": tickers or ["ES"], "scope_name": "scope"},
        "edges": [0.5],
        "n_bins": 2,
        "long_bins": [1],
        "short_bins": [],
        "direction": "long",
        "spec_version": "v1",
    }


def _feature_config() -> dict:
    return {
        "name": "buy_hold_domain_discrete",
        "model_type": "domain_discrete",
        "feature_column": "domain_discrete_signal_D_direction_long_sourceModule_buy_hold_specVersion_v1",
        "strategy": "long",
        "bias_node_spec": {
            "module_name": "domain_discrete",
            "timeframes": ["D"],
            "params": _frozen_spec(),
        },
        "tickers": ["ES"],
    }


def test_domain_discrete_spec_rejects_non_monotone_edges() -> None:
    payload = _frozen_spec()
    payload["edges"] = [0.5, -0.5]
    payload["n_bins"] = 3
    payload["long_bins"] = [2]
    with pytest.raises(ValueError, match="strictly monotone"):
        load_domain_discrete_spec(payload)


def test_domain_discrete_spec_rejects_direction_bin_mismatch() -> None:
    payload = _frozen_spec()
    payload["direction"] = "long_short"
    with pytest.raises(ValueError, match="requires both long_bins and short_bins"):
        load_domain_discrete_spec(payload)


def test_domain_discrete_base_model_emits_signed_signal() -> None:
    model = BaseModel(feature_config=_feature_config(), tickers=[Ticker.ES], use_cache=False)
    candles = pd.DataFrame(
        [
            {"datetime": "2024-01-01", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1, "ticker": "ES", "timeframe": "D"},
            {"datetime": "2024-01-02", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1, "ticker": "ES", "timeframe": "D"},
            {"datetime": "2024-01-03", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1, "ticker": "ES", "timeframe": "D"},
        ]
    )

    signals = model.predict(candles)

    assert signals.tolist() == [0.0, 1.0, 1.0]
    assert model.feature_column == (
        "domain_discrete_signal_D_direction_long_sourceModule_buy_hold_specVersion_v1"
    )


def test_domain_discrete_base_model_enforces_ticker_scope() -> None:
    config = _feature_config()
    config["bias_node_spec"]["params"] = _frozen_spec(tickers=["ES"])
    with pytest.raises(ValueError, match="outside ticker_scope"):
        BaseModel(feature_config=config, tickers=[Ticker.NQ], use_cache=False)
