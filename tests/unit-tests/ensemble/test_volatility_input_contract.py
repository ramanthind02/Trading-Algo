from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from ensemble.portfolio import GlobalPortfolio, TFPortfolio, PortfolioCacheQuery
from cache.runtime.central_cache import CentralCacheStore
from cache.runtime.central_cache_errors import ArtifactMissingError
from cache.runtime.central_cache_models import ArtifactDescriptor, ArtifactScope
from lib.core.enums import TimeFrame, Ticker


def _candles_frame(ticker: str = "ES") -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    return pd.DataFrame(
        {
            "datetime": dates,
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [1000.0, 1001.0, 1002.0, 1003.0],
            "ticker": [ticker] * 4,
            "timeframe": [TimeFrame.D] * 4,
        }
    )


def _volatility_frame(ticker: str = "ES") -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    return pd.DataFrame(
        {
            "datetime": dates,
            "ticker": [ticker] * 4,
            "ewsd_annual_vol": [0.2, 0.21, 0.22, 0.23],
        }
    )


@pytest.fixture(autouse=True)
def _isolated_central_cache(tmp_path: pytest.TempPathFactory) -> None:
    CentralCacheStore.reset()
    CentralCacheStore._instance = CentralCacheStore(cache_dir=tmp_path)  # type: ignore[attr-defined]
    yield
    CentralCacheStore.reset()


def _seed_cache(include_volatility: bool = True) -> PortfolioCacheQuery:
    store = CentralCacheStore.get_instance()
    candles = _candles_frame().drop(columns=["ticker", "timeframe"])
    store.set_candles(Ticker.ES, TimeFrame.D, candles)
    if include_volatility:
        vol = _volatility_frame()
        store.write_artifact(
            ArtifactDescriptor(
                family="bias",
                ticker=Ticker.ES,
                timeframe=TimeFrame.D,
                module_name="ewsd",
                params={"long_run_window": 2520},
                scope=ArtifactScope.LIVE,
                artifact_name="ewsd",
            ),
            vol.set_index("datetime")[["ewsd_annual_vol"]],
            depends_on=((Ticker.ES, TimeFrame.D),),
        )
    return PortfolioCacheQuery(
        tickers=("ES",),
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 4),
        timeframes=(TimeFrame.D,),
    )


def test_tfportfolio_predict_from_cache_reads_volatility_from_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    query = _seed_cache(include_volatility=True)
    portfolio = TFPortfolio(ensembles=[object()], trading_timeframe=TimeFrame.D)

    captured: dict[str, pd.DataFrame] = {}

    def _predict_from_candles(
        candles_df: pd.DataFrame,
        daily_volatility_df: pd.DataFrame,
        **kwargs: object,
    ) -> pd.DataFrame:
        captured["candles"] = candles_df
        captured["volatility"] = daily_volatility_df
        return pd.DataFrame(
            {
                "ticker": ["ES"],
                "datetime": [datetime(2024, 1, 2)],
                "forecast_score": [0.5],
                "position_fraction": [0.5],
            }
        )

    monkeypatch.setattr(portfolio, "predict_from_candles", _predict_from_candles)

    result = portfolio.predict_from_cache(query)

    assert isinstance(result, pd.DataFrame)
    assert "volatility" in captured
    assert set(captured["volatility"].columns) >= {"datetime", "ticker", "ewsd_annual_vol"}


def test_tfportfolio_predict_from_cache_fails_without_ewsd() -> None:
    query = _seed_cache(include_volatility=False)
    portfolio = TFPortfolio(ensembles=[object()], trading_timeframe=TimeFrame.D)

    with pytest.raises(ArtifactMissingError, match="EWSD volatility"):
        portfolio.predict_from_cache(query)


def test_global_portfolio_fit_and_predict_from_cache_use_central_cache() -> None:
    query = _seed_cache(include_volatility=True)

    class _MockTFPortfolio:
        def __init__(self, timeframe: TimeFrame) -> None:
            self.trading_timeframe = timeframe
            self.is_fitted_: bool = False

        def fit_from_candles(self, candles_df: pd.DataFrame, *args: object, **kwargs: object) -> "_MockTFPortfolio":
            self.is_fitted_ = True
            return self

        def predict_base_model_vectors_from_candles(self, candles_df: pd.DataFrame, *args: object, **kwargs: object) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "ticker": ["ES"],
                    "datetime": [datetime(2024, 1, 2)],
                    "model_name": [f"{self.trading_timeframe.name}::ensemble_0::model_a"],
                    "forecast": [0.5],
                    "signal": [0.5],
                    "timeframe": [self.trading_timeframe.name],
                }
            )

    gp = GlobalPortfolio(tf_portfolios=[_MockTFPortfolio(TimeFrame.D)])
    gp.fit_from_cache(query, instrument_returns=pd.DataFrame({"ES": [0.0, 0.1]}, index=pd.date_range("2024-01-01", periods=2, freq="D")))

    result = gp.predict_from_cache(query)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
