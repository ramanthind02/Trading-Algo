from __future__ import annotations

import sys
from datetime import datetime, timedelta
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

from lib.cache.runtime.central_cache import CentralCacheStore
from lib.core.enums import Ticker, TimeFrame
from lib.core.models import Candle


def _install_forecast_server_stubs() -> None:
    if "schedule" not in sys.modules:
        schedule_module = ModuleType("schedule")

        class _Job:
            day = property(lambda self: self)
            sunday = property(lambda self: self)

            def at(self, *_args, **_kwargs) -> "_Job":
                return self

            def do(self, *_args, **_kwargs) -> None:
                return None

        schedule_module.every = lambda *_args, **_kwargs: _Job()
        sys.modules["schedule"] = schedule_module

    if "MetaTrader5" not in sys.modules:
        sys.modules["MetaTrader5"] = SimpleNamespace(
            TIMEFRAME_D1=1,
            TIMEFRAME_W1=2,
            TIMEFRAME_MN1=3,
            initialize=lambda *_args, **_kwargs: True,
            login=lambda *_args, **_kwargs: True,
            last_error=lambda: "",
        )


_install_forecast_server_stubs()

from deployment.forecast_server import ForecastServer
from deployment.forecast_prediction_runtime import ForecastPredictionRuntime, normalize_forecast_value


def _frame(start: datetime, periods: int) -> pd.DataFrame:
    dates = [start + timedelta(days=offset) for offset in range(periods)]
    return pd.DataFrame(
        {
            "datetime": dates,
            "open": [100.0 + offset for offset in range(periods)],
            "high": [101.0 + offset for offset in range(periods)],
            "low": [99.0 + offset for offset in range(periods)],
            "close": [100.5 + offset for offset in range(periods)],
            "volume": [1_000.0 + offset for offset in range(periods)],
            "ticker": [Ticker.NQ.name] * periods,
            "timeframe": [TimeFrame.D] * periods,
        }
    )


def _candle(at: datetime, close: float) -> Candle:
    return Candle(
        datetime=at,
        open=close - 0.5,
        high=close + 0.5,
        low=close - 1.0,
        close=close,
        volume=1_500.0,
        ticker=Ticker.NQ,
        tf=TimeFrame.D,
    )


def test_upsert_cross_ticker_candle_preserves_cached_history(tmp_path) -> None:
    CentralCacheStore.reset()
    store = CentralCacheStore(cache_dir=str(tmp_path))
    CentralCacheStore._instance = store  # type: ignore[attr-defined]

    server = ForecastServer.__new__(ForecastServer)
    server.candle_buffers = {}

    store.set_candles(Ticker.NQ, TimeFrame.D, _frame(datetime(2024, 1, 1), 3))
    server._upsert_cross_ticker_candle(Ticker.NQ, TimeFrame.D, _candle(datetime(2024, 1, 4), 104.5))

    updated = store.query_candles(Ticker.NQ, TimeFrame.D)

    assert len(updated) == 4
    assert list(updated.index) == list(pd.date_range("2024-01-01", periods=4, freq="D"))


def test_upsert_cross_ticker_candle_replaces_matching_timestamp(tmp_path) -> None:
    CentralCacheStore.reset()
    store = CentralCacheStore(cache_dir=str(tmp_path))
    CentralCacheStore._instance = store  # type: ignore[attr-defined]

    server = ForecastServer.__new__(ForecastServer)
    server.candle_buffers = {}

    store.set_candles(Ticker.NQ, TimeFrame.D, _frame(datetime(2024, 1, 1), 3))
    server._upsert_cross_ticker_candle(Ticker.NQ, TimeFrame.D, _candle(datetime(2024, 1, 2), 222.5))

    updated = store.query_candles(Ticker.NQ, TimeFrame.D)

    assert len(updated) == 3
    assert updated.loc[pd.Timestamp("2024-01-02"), "close"] == 222.5


def test_run_test_forecast_returns_structured_results_for_requested_timeframe() -> None:
    server = ForecastServer.__new__(ForecastServer)
    server.timeframes = [TimeFrame.D, TimeFrame.W]
    server._update_live_inputs_for_timeframe = lambda timeframe: None
    server._generate_portfolio_forecasts = (
        lambda timeframe: {"NQ": 0.75} if timeframe == TimeFrame.D else {"ES": 0.55}
    )

    result = server.run_test_forecast(timeframe=TimeFrame.D)

    assert result["errors"] == []
    assert set(result["forecasts"]) == {"D"}
    assert result["forecasts"]["D"] == {
        "count": 1,
        "predictions": {"NQ": 0.75},
    }


def test_get_cross_tickers_discovers_unique_pairs_from_ensemble_specs() -> None:
    class _FakeEnsemble:
        def __init__(self, specs: list[dict[str, object]]) -> None:
            self._specs = specs

        def get_required_bias_nodes(self) -> list[dict[str, object]]:
            return self._specs

    server = ForecastServer.__new__(ForecastServer)
    server.mt5_connector = None
    server.volatility_service = None
    server.candle_buffers = {}
    server.ml_managers = {}
    server.lookback_candles = 50
    server.tickers = []
    server.ensembles = {
        (Ticker.ES, TimeFrame.D): _FakeEnsemble(
            [
                {"params": {"cross_tickers": ["NQ"]}},
                {"params": {"cross_tickers": ["NQ", "GC"]}},
            ]
        ),
        (Ticker.NQ, TimeFrame.W): _FakeEnsemble(
            [
                {"params": {"cross_tickers": ["ES"]}},
            ]
        ),
    }

    assert server._get_cross_tickers() == {
        (Ticker.NQ, TimeFrame.D),
        (Ticker.GC, TimeFrame.D),
        (Ticker.ES, TimeFrame.W),
    }


def test_prepare_prediction_data_requires_daily_volatility() -> None:
    runtime = ForecastPredictionRuntime(
        volatility_service=SimpleNamespace(latest_volatility_map=lambda: {}),
        ensembles={},
        ml_managers={},
        candle_buffers={},
        tickers=[],
        logger=SimpleNamespace(
            warning=lambda *args, **kwargs: None,
            error=lambda *args, **kwargs: None,
            exception=lambda *args, **kwargs: None,
            info=lambda *args, **kwargs: None,
        ),
    )

    with pytest.raises(ValueError, match="Missing daily EWSD volatility"):
        runtime.prepare_prediction_data(Ticker.NQ, 1)


def test_generate_portfolio_forecasts_returns_normalized_predictions() -> None:
    class _FakeEnsemble:
        def predict(self, X, ticker, volatility):  # noqa: ANN001,N803
            assert len(X) == 1
            assert ticker.tolist() == [Ticker.NQ.value]
            assert volatility.tolist() == [0.25]
            return pd.Series([0.5]).to_numpy()

    runtime = ForecastPredictionRuntime(
        volatility_service=SimpleNamespace(
            latest_volatility_map=lambda: {Ticker.NQ.name: 0.25}
        ),
        ensembles={(Ticker.NQ, TimeFrame.D): _FakeEnsemble()},
        ml_managers={(Ticker.NQ, TimeFrame.D): SimpleNamespace(matrix_df=pd.DataFrame({"x": [1.0]}))},
        candle_buffers={(Ticker.NQ, TimeFrame.D): [object()] * 20},
        tickers=[Ticker.NQ],
        logger=SimpleNamespace(
            warning=lambda *args, **kwargs: None,
            error=lambda *args, **kwargs: None,
            exception=lambda *args, **kwargs: None,
            info=lambda *args, **kwargs: None,
        ),
    )

    result = runtime.generate_portfolio_forecasts(TimeFrame.D)

    assert result == {Ticker.NQ.name: pytest.approx(normalize_forecast_value(0.5))}
