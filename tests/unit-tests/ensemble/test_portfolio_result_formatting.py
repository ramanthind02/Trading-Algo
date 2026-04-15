from __future__ import annotations

import pandas as pd

from ensemble.portfolio_impl.portfolio_result_formatting import (
    deep_copy_result,
    format_portfolio_result,
)


def test_deep_copy_result_copies_nested_dataframes() -> None:
    payload = {
        "portfolio": pd.DataFrame({"x": [1]}),
        "nested": [{"frame": pd.DataFrame({"y": [2]})}],
    }

    copied = deep_copy_result(payload)
    copied["portfolio"].loc[0, "x"] = 99
    copied["nested"][0]["frame"].loc[0, "y"] = 42

    assert payload["portfolio"].loc[0, "x"] == 1
    assert payload["nested"][0]["frame"].loc[0, "y"] == 2


def test_format_portfolio_result_respects_detail_flags() -> None:
    cached_result = {
        "portfolio": pd.DataFrame({"ticker": ["ES"]}),
        "ensembles": {"ensemble_0": pd.DataFrame({"ticker": ["ES"]})},
        "base_models": {"ensemble_0::m1": pd.DataFrame({"ticker": ["ES"]})},
    }

    portfolio_only = format_portfolio_result(
        cached_result,
        return_ensemble_predictions=False,
        return_base_model_predictions=False,
    )
    detailed = format_portfolio_result(
        cached_result,
        return_ensemble_predictions=True,
        return_base_model_predictions=True,
    )

    assert isinstance(portfolio_only, pd.DataFrame)
    assert set(detailed) == {"portfolio", "ensembles", "base_models"}
