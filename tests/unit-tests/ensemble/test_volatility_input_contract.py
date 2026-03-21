from __future__ import annotations

import pandas as pd
import pytest

from ensemble.diversified_ensemble import DiversifiedEnsemble
from ensemble.portfolio import GlobalPortfolio, TFPortfolio
from utils.core.enums import TimeFrame


def _candles_stub() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-01"]),
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1000.0],
            "ticker": ["ES"],
            "timeframe": [TimeFrame.D],
        }
    )


def test_diversified_ensemble_predict_requires_daily_volatility() -> None:
    ensemble = DiversifiedEnsemble.__new__(DiversifiedEnsemble)
    ensemble.is_fitted_ = True  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="daily_volatility_df is required"):
        ensemble.predict_from_candles(  # type: ignore[misc]
            candles_df=_candles_stub(),
            daily_volatility_df=None,
        )


def test_tfportfolio_predict_requires_daily_volatility() -> None:
    portfolio = TFPortfolio(ensembles=[object()], trading_timeframe=TimeFrame.D)

    with pytest.raises(ValueError, match="daily_volatility_df is required"):
        portfolio.predict_from_candles(
            candles_df=_candles_stub(),
            daily_volatility_df=None,  # type: ignore[arg-type]
        )


def test_global_portfolio_fit_requires_daily_volatility() -> None:
    gp = GlobalPortfolio(tf_portfolios=[])
    candles_per_tf = {TimeFrame.D: _candles_stub()}
    instrument_returns = pd.DataFrame({"ES": [0.0]}, index=pd.to_datetime(["2024-01-01"]))

    with pytest.raises(ValueError, match="daily_volatility_df is required"):
        gp.fit(
            candles_per_tf=candles_per_tf,
            instrument_returns=instrument_returns,
            daily_volatility_df=None,  # type: ignore[arg-type]
        )


def test_global_portfolio_predict_requires_daily_volatility() -> None:
    gp = GlobalPortfolio(tf_portfolios=[])
    gp.is_fitted_ = True

    with pytest.raises(ValueError, match="daily_volatility_df is required"):
        gp.predict(
            candles_per_tf={},
            daily_volatility_df=None,  # type: ignore[arg-type]
        )
