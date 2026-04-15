"""TFPortfolio must not swallow ensemble fit/predict errors used by GlobalPortfolio."""

from __future__ import annotations

import pandas as pd
import pytest

from ensemble.portfolio_impl.tf_portfolio import TFPortfolio
from utils.core.enums import TimeFrame


def _minimal_candles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1000, 1100],
            "ticker": ["ES", "ES"],
            "timeframe": [TimeFrame.D, TimeFrame.D],
        }
    )


def _minimal_vol() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "ticker": ["ES", "ES"],
            "ewsd_annual_vol": [0.2, 0.2],
        }
    )


class _BadEnsemble:
    vault_ensemble_name = "bad_ensemble"

    def fit_from_candles(self, *_args: object, **_kwargs: object) -> None:
        raise ValueError("simulated fit failure")

    def predict_from_candles(self, *_args: object, **_kwargs: object) -> dict[str, object]:
        raise ValueError("simulated predict failure")


def test_tf_portfolio_fit_from_candles_raises_on_first_ensemble_fit_failure() -> None:
    portfolio = TFPortfolio(
        ensembles=[_BadEnsemble()],
        trading_timeframe=TimeFrame.D,
        target_volatility=0.15,
        max_position_pct=1.0,
        use_cache=False,
    )
    with pytest.raises(RuntimeError, match="TFPortfolio: ensemble fit failed"):
        portfolio.fit_from_candles(_minimal_candles(), pd.Series(dtype=float))


def test_predict_base_model_vectors_raises_when_predict_fails() -> None:
    class _FitOkPredictBad(_BadEnsemble):
        def fit_from_candles(self, *_args: object, **_kwargs: object) -> _FitOkPredictBad:
            return self

    portfolio = TFPortfolio(
        ensembles=[_FitOkPredictBad()],
        trading_timeframe=TimeFrame.D,
        target_volatility=0.15,
        max_position_pct=1.0,
        use_cache=False,
    )
    portfolio.fit_from_candles(_minimal_candles(), pd.Series(dtype=float))
    with pytest.raises(RuntimeError, match="Global base-model vector extraction failed"):
        portfolio.predict_base_model_vectors_from_candles(_minimal_candles(), _minimal_vol())
