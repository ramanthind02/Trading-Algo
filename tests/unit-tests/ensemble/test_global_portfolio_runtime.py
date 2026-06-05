from __future__ import annotations

import pandas as pd
import pytest

from ensemble.portfolio_impl.global_portfolio_runtime import (
    apply_global_position_constraints,
    build_global_returns_proxy,
    build_reference_grid_from_daily_candles,
    collect_tf_forecast_streams,
)
from lib.core.enums import TimeFrame


class _MockTFPortfolio:
    def __init__(self, timeframe: TimeFrame, vectors: pd.DataFrame) -> None:
        self.trading_timeframe = timeframe
        self._vectors = vectors

    def predict_base_model_vectors_from_candles(self, candles_df: pd.DataFrame, **kwargs):  # noqa: ANN001
        del candles_df, kwargs
        return self._vectors


def test_collect_tf_forecast_streams_raises_when_timeframe_missing() -> None:
    portfolio = _MockTFPortfolio(TimeFrame.W, pd.DataFrame())

    with pytest.raises(ValueError, match="No candles provided for timeframe W"):
        collect_tf_forecast_streams(
            [portfolio],
            candles_per_tf={TimeFrame.D: pd.DataFrame()},
            daily_volatility_df=pd.DataFrame(),
        )


def test_build_global_returns_proxy_normalizes_index() -> None:
    returns = pd.DataFrame(
        {"ES": [0.1, -0.1]},
        index=pd.to_datetime(["2024-01-01 09:30:00", "2024-01-02 16:00:00"]),
    )

    proxy = build_global_returns_proxy(returns)

    assert proxy is not None
    assert list(proxy.index) == list(pd.to_datetime(["2024-01-01", "2024-01-02"]))


def test_build_reference_grid_from_daily_candles_uses_daily_datetime() -> None:
    candles_per_tf = {
        TimeFrame.D: pd.DataFrame(
            {"datetime": pd.to_datetime(["2024-01-01 09:30:00", "2024-01-02 16:00:00"])}
        )
    }

    grid = build_reference_grid_from_daily_candles(candles_per_tf)

    assert list(grid) == list(pd.to_datetime(["2024-01-01", "2024-01-02"]))


def test_apply_global_position_constraints_applies_weights_idm_and_cap() -> None:
    combined = pd.DataFrame(
        {
            "ticker": ["ES", "NQ"],
            "datetime": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "forecast_score": [1.0, 2.0],
        }
    )

    result = apply_global_position_constraints(
        combined,
        instrument_weights={"ES": 0.75, "NQ": 0.25},
        global_idm=2.0,
        max_position_pct=0.9,
    )

    assert result["position_fraction"].tolist() == pytest.approx([0.9, 0.9])
