from __future__ import annotations

import pandas as pd
import pytest

from ensemble.portfolio import TFPortfolio
from ensemble.portfolio_postprocessing import (
    aggregate_forecast_vectors_fallback,
    apply_forecast_risk_management,
)
from utils.core.enums import TimeFrame


def test_aggregate_forecast_vectors_fallback_averages_by_ticker_and_datetime() -> None:
    vectors = [
        pd.DataFrame(
            {
                "ticker": ["ES", "ES"],
                "datetime": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "forecast": [1.0, 2.0],
            }
        ),
        pd.DataFrame(
            {
                "ticker": ["ES", "ES"],
                "datetime": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "forecast": [3.0, 4.0],
            }
        ),
    ]

    result = aggregate_forecast_vectors_fallback(vectors)

    assert result["forecast_score"].tolist() == pytest.approx([2.0, 3.0])


def test_apply_forecast_risk_management_merges_weights_idm_and_cap() -> None:
    forecasts_df = pd.DataFrame(
        {
            "ticker": ["ES", "NQ"],
            "datetime": pd.to_datetime(["2024-01-01 09:30:00", "2024-01-01 09:30:00"]),
            "forecast_score": [1.0, 3.0],
        }
    )
    candles_df = pd.DataFrame(
        {
            "ticker": ["ES", "NQ", "ES"],
            "datetime": pd.to_datetime(
                ["2024-01-01 09:30:00", "2024-01-01 09:30:00", "2024-01-02 09:30:00"]
            ),
        }
    )

    result = apply_forecast_risk_management(
        forecasts_df,
        candles_df,
        instrument_weights={"ES": 0.5, "NQ": 0.5},
        idm=2.0,
        max_position_pct=2.0,
    )

    assert result["forecast_score"].tolist() == pytest.approx([1.0, 3.0, 0.0])
    assert result["position_fraction"].tolist() == pytest.approx([1.0, 2.0, 0.0])


def test_tfportfolio_predict_from_candles_uses_postprocessing_helpers() -> None:
    class _DummyEnsemble:
        def predict_from_candles(self, candles_df, **kwargs):  # noqa: ANN001
            del kwargs
            base = candles_df[["ticker", "datetime"]].copy()
            base["forecast_score"] = 1.0
            return {
                "ensemble": base,
                "base_models": {"m1": base.copy()},
            }

    candles_df = pd.DataFrame(
        {
            "ticker": ["ES", "NQ"],
            "datetime": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [1.0, 1.0],
            "timeframe": [TimeFrame.D, TimeFrame.D],
        }
    )
    daily_volatility_df = pd.DataFrame(
        {
            "ticker": ["ES", "NQ"],
            "datetime": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "ewsd_annual_vol": [0.2, 0.2],
        }
    )

    portfolio = TFPortfolio(
        ensembles=[_DummyEnsemble()],
        trading_timeframe=TimeFrame.D,
        instrument_weights={"ES": 0.6, "NQ": 0.4},
        max_position_pct=2.0,
    )
    portfolio.idm_ = 1.5

    result = portfolio.predict_from_candles(candles_df, daily_volatility_df)

    assert result["forecast_score"].tolist() == pytest.approx([1.0, 1.0])
    assert result["position_fraction"].tolist() == pytest.approx([0.9, 0.6])
