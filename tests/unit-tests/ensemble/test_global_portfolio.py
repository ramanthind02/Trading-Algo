"""Unit tests for GlobalPortfolio (T008).

All tests use SYNTHETIC data only — no vault, no real candle files, no disk I/O.
TFPortfolio instances are lightweight mocks to isolate GlobalPortfolio orchestration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ensemble.portfolio_impl.global_weight_layer_adapter import (
    build_global_adapter_rollups,
    decode_global_weight_layer_output,
    encode_forecast_vectors_for_global_weight_layer,
)
from lib.cache.runtime.central_cache import CentralCacheStore
from lib.cache.runtime.central_cache_models import ArtifactDescriptor, ArtifactScope
from lib.core.enums import TimeFrame, Ticker
from ensemble.portfolio import GlobalPortfolio, TFPortfolio, PortfolioCacheQuery


# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------

def _make_dates(n: int = 20) -> pd.DatetimeIndex:
    return pd.date_range("2023-01-02", periods=n, freq="B")


def _make_forecast_stream(
    tickers: list[str],
    dates: pd.DatetimeIndex,
    score: float = 0.5,
) -> pd.DataFrame:
    """Return a minimal forecast stream DataFrame."""
    rows = [
        {"ticker": t, "datetime": d, "forecast_score": score}
        for d in dates
        for t in tickers
    ]
    return pd.DataFrame(rows)


def _make_instrument_returns(
    tickers: list[str],
    dates: pd.DatetimeIndex,
    corr: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a DataFrame of synthetic daily returns with controlled correlation."""
    rng = np.random.default_rng(seed)
    n = len(dates)
    k = len(tickers)
    # Build correlated returns via Cholesky of (1-corr)*I + corr*11^T
    cov = np.full((k, k), corr)
    np.fill_diagonal(cov, 1.0)
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal((n, k))
    returns = z @ L.T * 0.01  # scale to ~1% daily vol
    return pd.DataFrame(returns, index=dates, columns=tickers)


class _MockTFPortfolio:
    """Minimal mock that satisfies GlobalPortfolio's interface.

    ``fit_from_candles`` marks the mock fitted.
    ``predict_from_candles_raw`` returns a fixed forecast stream with
    columns ['ticker', 'datetime', 'forecast_score', 'position_weighted'].
    """

    def __init__(
        self,
        timeframe: TimeFrame,
        tickers: list[str],
        score: float = 0.5,
        n_dates: int = 20,
    ) -> None:
        self.trading_timeframe = timeframe
        self._tickers = tickers
        self._score = score
        self._dates = _make_dates(n_dates)
        self.is_fitted_: bool = False

    def fit_from_candles(self, candles_df: pd.DataFrame, *args, **kwargs) -> "_MockTFPortfolio":
        self.is_fitted_ = True
        return self

    def predict_from_candles_raw(
        self, candles_df: pd.DataFrame, *args, **kwargs
    ) -> pd.DataFrame:
        rows = [
            {
                "ticker": t,
                "datetime": d,
                "forecast_score": self._score,
                "position_weighted": self._score * 0.5,
            }
            for d in self._dates
            for t in self._tickers
        ]
        return pd.DataFrame(rows)

    def predict_base_model_vectors_from_candles(
        self, candles_df: pd.DataFrame, *args, **kwargs
    ) -> pd.DataFrame:
        rows = [
            {
                "ticker": t,
                "datetime": d,
                "model_name": f"{self.trading_timeframe.name}::ensemble_0::model_a",
                "forecast": self._score,
                "signal": self._score,
                "timeframe": self.trading_timeframe.name,
            }
            for d in self._dates
            for t in self._tickers
        ]
        return pd.DataFrame(rows)

    def fit_from_cache(self, query: PortfolioCacheQuery, *args, **kwargs) -> "_MockTFPortfolio":
        return self.fit_from_candles(pd.DataFrame(), *args, **kwargs)

    def predict_from_cache(self, query: PortfolioCacheQuery, *args, **kwargs):
        return self.predict_from_candles(pd.DataFrame(), *args, **kwargs)


def _make_candles_stub(tickers: list[str], n_dates: int = 20) -> pd.DataFrame:
    """Return a minimal stub candles DataFrame (not used by the mock, just satisfies type)."""
    dates = _make_dates(n_dates)
    rows = [
        {
            "datetime": d,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1000,
            "ticker": t,
            "timeframe": TimeFrame.D,
        }
        for d in dates
        for t in tickers
    ]
    return pd.DataFrame(rows)


def _make_daily_volatility_stub(tickers: list[str], n_dates: int = 20, annual_vol: float = 0.2) -> pd.DataFrame:
    dates = _make_dates(n_dates)
    rows = [
        {"datetime": d, "ticker": t, "ewsd_annual_vol": annual_vol}
        for d in dates
        for t in tickers
    ]
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _central_cache_isolated(tmp_path):
    CentralCacheStore.reset()
    CentralCacheStore._instance = CentralCacheStore(cache_dir=tmp_path)  # type: ignore[attr-defined]
    yield
    CentralCacheStore.reset()


def _build_global_portfolio(
    tickers: list[str] = None,
    n_tfs: int = 1,
    score: float = 0.5,
    n_dates: int = 20,
    idm_max: float = 2.5,
    max_position_pct: float = 2.0,
):
    """Factory: build a fitted GlobalPortfolio with mock TFPortfolios."""
    if tickers is None:
        tickers = ["ES", "NQ"]

    timeframes = [TimeFrame.D, TimeFrame.W, TimeFrame.M][:n_tfs]
    tf_portfolios = [
        _MockTFPortfolio(tf, tickers, score=score, n_dates=n_dates)
        for tf in timeframes
    ]

    gp = GlobalPortfolio(
        tf_portfolios=tf_portfolios,
        idm_max=idm_max,
        max_position_pct=max_position_pct,
    )

    dates = _make_dates(n_dates)
    instrument_returns = _make_instrument_returns(tickers, dates)
    candles_per_tf = {tf: _make_candles_stub(tickers, n_dates) for tf in timeframes}

    daily_volatility_df = _make_daily_volatility_stub(tickers, n_dates)
    store = CentralCacheStore.get_instance()
    for tf, tf_candles in candles_per_tf.items():
        for ticker_name, ticker_df in tf_candles.groupby("ticker", sort=False):
            store.set_candles(Ticker[ticker_name], tf, ticker_df)
    for ticker_name, vol_df in daily_volatility_df.groupby("ticker", sort=False):
        store.write_artifact(
            ArtifactDescriptor(
                family="bias",
                ticker=Ticker[ticker_name],
                timeframe=TimeFrame.D,
                module_name="ewsd",
                params={"long_run_window": 2520},
                scope=ArtifactScope.LIVE,
                artifact_name="ewsd",
            ),
            vol_df.set_index("datetime")[["ewsd_annual_vol"]],
            depends_on=((Ticker[ticker_name], TimeFrame.D),),
        )

    query = PortfolioCacheQuery(
        tickers=tuple(tickers),
        start=dates.min().to_pydatetime(),
        end=dates.max().to_pydatetime(),
        timeframes=tuple(timeframes),
    )
    gp.fit_from_cache(query, instrument_returns)
    return gp, query


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOutputSchemaMatchesLegacy:
    """Output must have exactly ['ticker', 'datetime', 'forecast_score', 'position_fraction']."""

    def test_output_columns(self):
        gp, query = _build_global_portfolio(tickers=["ES", "NQ"])
        result = gp.predict_from_cache(query)
        assert isinstance(result, pd.DataFrame)
        expected_cols = {"ticker", "datetime", "forecast_score", "position_fraction"}
        assert expected_cols == set(result.columns), (
            f"Unexpected columns: {set(result.columns)}"
        )

    def test_no_nan_in_position_fraction(self):
        gp, query = _build_global_portfolio(tickers=["ES", "NQ"])
        result = gp.predict_from_cache(query)
        assert not result["position_fraction"].isna().any(), (
            "position_fraction contains NaN values"
        )

    def test_no_nan_in_forecast_score(self):
        gp, query = _build_global_portfolio(tickers=["ES", "NQ"])
        result = gp.predict_from_cache(query)
        assert not result["forecast_score"].isna().any()


class TestTwoTFOutputSchema:
    """Two-TF GlobalPortfolio must produce correct schema with no NaN."""

    def test_two_tf_columns(self):
        gp, query = _build_global_portfolio(tickers=["ES", "NQ", "CL"], n_tfs=2)
        result = gp.predict_from_cache(query)
        assert set(result.columns) == {"ticker", "datetime", "forecast_score", "position_fraction"}

    def test_two_tf_no_nan(self):
        gp, query = _build_global_portfolio(tickers=["ES", "NQ", "CL"], n_tfs=2)
        result = gp.predict_from_cache(query)
        assert not result["position_fraction"].isna().any()
        assert not result["forecast_score"].isna().any()

    def test_two_tf_position_fraction_bounded(self):
        max_pos = 2.0
        gp, query = _build_global_portfolio(
            tickers=["ES", "NQ"], n_tfs=2, max_position_pct=max_pos
        )
        result = gp.predict_from_cache(query)
        assert (result["position_fraction"].abs() <= max_pos + 1e-9).all(), (
            "position_fraction exceeds max_position_pct"
        )


class TestPositionFractionBounds:
    """position_fraction must always be within [-max_position_pct, +max_position_pct]."""

    @pytest.mark.parametrize("max_pos", [0.5, 1.0, 2.0, 5.0])
    def test_position_clipped(self, max_pos: float):
        gp, query = _build_global_portfolio(
            tickers=["ES", "NQ", "GC"],
            n_tfs=1,
            score=10.0,  # large score to stress the clip
            max_position_pct=max_pos,
        )
        result = gp.predict_from_cache(query)
        assert (result["position_fraction"].abs() <= max_pos + 1e-9).all()


class TestFittedState:
    """is_fitted_ must be True after fit() and False before."""

    def test_is_fitted_after_fit(self):
        gp, _ = _build_global_portfolio()
        assert gp.is_fitted_ is True

    def test_not_fitted_raises(self):
        tf_p = _MockTFPortfolio(TimeFrame.D, ["ES"])
        gp = GlobalPortfolio(tf_portfolios=[tf_p])
        candles_per_tf = {TimeFrame.D: _make_candles_stub(["ES"])}
        daily_volatility_df = _make_daily_volatility_stub(["ES"])
        with pytest.raises(RuntimeError, match="fitted"):
            gp.predict(candles_per_tf, daily_volatility_df=daily_volatility_df)

    def test_global_idm_set_after_fit(self):
        gp, _ = _build_global_portfolio(tickers=["ES", "NQ"])
        assert gp.global_idm_ is not None
        assert gp.global_idm_ > 0

    def test_instruments_set_after_fit(self):
        gp, _ = _build_global_portfolio(tickers=["ES", "NQ", "GC"])
        assert gp.instruments_ is not None
        assert set(gp.instruments_) == {"ES", "NQ", "GC"}


class TestIDMCap:
    """global_idm_ must be capped at idm_max."""

    def test_idm_capped(self):
        idm_max = 1.2
        gp, _ = _build_global_portfolio(tickers=["ES", "NQ"], idm_max=idm_max)
        assert gp.global_idm_ <= idm_max + 1e-9

    def test_idm_at_least_one_for_single_instrument(self):
        gp, _ = _build_global_portfolio(tickers=["ES"])
        # Single instrument → IDM = 1.0 (no diversification)
        assert gp.global_idm_ == pytest.approx(1.0)


class TestAlias:
    """GlobalPortfolio is exported from both ensemble.portfolio and the ensemble package."""

    def test_global_portfolio_importable_from_portfolio_module(self):
        import ensemble.portfolio as pm
        assert hasattr(pm, "GlobalPortfolio")
        assert pm.GlobalPortfolio.__name__ == "GlobalPortfolio"

    def test_global_portfolio_importable_from_ensemble_package(self):
        import ensemble
        assert hasattr(ensemble, "GlobalPortfolio")
        assert ensemble.GlobalPortfolio.__name__ == "GlobalPortfolio"

    def test_portfolio_alias_is_tfportfolio_for_backward_compat(self):
        """Portfolio alias must remain TFPortfolio for backward compatibility."""
        import ensemble.portfolio as pm
        assert pm.Portfolio.__name__ == "TFPortfolio", (
            f"Portfolio alias must remain TFPortfolio for backward compat, got {pm.Portfolio}"
        )


class TestDiagnostics:
    """get_diagnostics() must return expected keys."""

    def test_diagnostics_keys(self):
        gp, _ = _build_global_portfolio()
        diag = gp.get_diagnostics()
        expected_keys = {
            "is_fitted",
            "global_idm",
            "mean_instrument_return_correlation",
            "instruments",
            "idm_max",
            "max_position_pct",
            "n_tf_portfolios",
            "weight_layer",
        }
        assert expected_keys.issubset(set(diag.keys()))

    def test_diagnostics_fitted_flag(self):
        gp, _ = _build_global_portfolio()
        assert gp.get_diagnostics()["is_fitted"] is True

    def test_collect_global_strategy_health_diagnostics_records_model_stats(self):
        gp = GlobalPortfolio(tf_portfolios=[])
        forecast_vectors = [
            pd.DataFrame(
                {
                    "ticker": ["ES", "ES"],
                    "datetime": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                    "model_name": ["alpha", "alpha"],
                    "forecast": [0.5, 0.5],
                    "signal": [1.0, 1.0],
                    "timeframe": ["D", "D"],
                }
            )
        ]

        eligible_vectors, diagnostics = gp._collect_global_strategy_health_diagnostics(
            forecast_vectors,
            global_returns=None,
        )

        assert eligible_vectors == forecast_vectors
        assert diagnostics["tickers"]["ES"]["n_models_before"] == 1
        assert diagnostics["tickers"]["ES"]["n_models_after"] == 1
        assert diagnostics["tickers"]["ES"]["model_stats"][0]["effective_obs"] == 2
        assert gp.global_eligible_models_by_ticker_ == {"ES": {"alpha"}}


class TestInstrumentWeights:
    """Custom instrument weights affect position_fraction proportionally."""

    def test_custom_weights_sum_to_one(self):
        """When explicit weights given, they are used (not equal weight)."""
        tickers = ["ES", "NQ", "GC"]
        custom_w = {"ES": 0.5, "NQ": 0.3, "GC": 0.2}

        timeframes = [TimeFrame.D]
        tf_portfolios = [_MockTFPortfolio(TimeFrame.D, tickers, score=0.5, n_dates=20)]
        gp = GlobalPortfolio(
            tf_portfolios=tf_portfolios,
            instrument_weights=custom_w,
        )

        dates = _make_dates(20)
        instrument_returns = _make_instrument_returns(tickers, dates)
        candles_per_tf = {TimeFrame.D: _make_candles_stub(tickers, 20)}
        daily_volatility_df = _make_daily_volatility_stub(tickers, 20)
        gp.fit(candles_per_tf, instrument_returns, daily_volatility_df=daily_volatility_df)
        result = gp.predict(candles_per_tf, daily_volatility_df=daily_volatility_df)

        assert not result.empty
        assert not result["position_fraction"].isna().any()


class TestGlobalAdapter:
    def test_global_stream_encoding_is_unique_and_decodable(self):
        data = pd.DataFrame(
            {
                "ticker": ["ES", "ES", "NQ", "NQ"],
                "datetime": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]
                ),
                "model_name": ["alpha", "alpha", "alpha", "beta"],
                "forecast": [0.2, 0.1, 0.4, 0.5],
                "signal": [1.0, 1.0, 1.0, 1.0],
                "timeframe": ["D", "D", "W", "W"],
            }
        )
        encoded_vectors, decode_map = encode_forecast_vectors_for_global_weight_layer([data])

        assert len(encoded_vectors) == 1
        encoded = encoded_vectors[0]
        assert set(encoded["ticker"]) == {"__GLOBAL__"}
        assert len(decode_map) == 3
        assert decode_map["ES::D::alpha"] == {
            "ticker": "ES",
            "timeframe": "D",
            "original_model_name": "alpha",
        }
        assert decode_map["NQ::W::alpha"] == {
            "ticker": "NQ",
            "timeframe": "W",
            "original_model_name": "alpha",
        }
        assert decode_map["NQ::W::beta"] == {
            "ticker": "NQ",
            "timeframe": "W",
            "original_model_name": "beta",
        }

    def test_decode_aggregates_streams_back_to_tickers(self):
        encoded = pd.DataFrame(
            {
                "ticker": ["__GLOBAL__", "__GLOBAL__", "__GLOBAL__", "__GLOBAL__"],
                "datetime": pd.to_datetime(
                    ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]
                ),
                "model_name": ["ES::D::m1", "NQ::W::m2", "ES::D::m1", "NQ::W::m2"],
                "forecast": [1.0, 0.5, 1.0, -0.5],
                "signal": [1.0, 1.0, 1.0, 1.0],
            }
        )
        decode_map = {
            "ES::D::m1": {
                "ticker": "ES",
                "timeframe": "D",
                "original_model_name": "m1",
            },
            "NQ::W::m2": {
                "ticker": "NQ",
                "timeframe": "W",
                "original_model_name": "m2",
            },
        }

        decoded = decode_global_weight_layer_output(
            [encoded],
            decode_map,
            global_weights=pd.Series(
                {
                    "ES::D::m1": 0.60,
                    "NQ::W::m2": 0.40,
                }
            ),
            global_fdm=1.0,
        )
        expected = pd.DataFrame(
            {
                "ticker": ["ES", "ES", "NQ", "NQ"],
                "datetime": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]
                ),
                "forecast_score": [0.6, 0.6, 0.2, -0.2],
            }
        )
        pd.testing.assert_frame_equal(
            decoded.sort_values(["ticker", "datetime"]).reset_index(drop=True),
            expected.sort_values(["ticker", "datetime"]).reset_index(drop=True),
        )

    def test_decode_falls_back_for_missing_stream_weights_and_clips_with_fdm(self):
        encoded = pd.DataFrame(
            {
                "ticker": ["__GLOBAL__", "__GLOBAL__"],
                "datetime": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                "model_name": ["ES::D::m1", "NQ::W::m2"],
                "forecast": [1.0, 1.0],
                "signal": [1.0, 1.0],
            }
        )
        decode_map = {
            "ES::D::m1": {
                "ticker": "ES",
                "timeframe": "D",
                "original_model_name": "m1",
            },
            "NQ::W::m2": {
                "ticker": "NQ",
                "timeframe": "W",
                "original_model_name": "m2",
            },
        }
        global_weights = pd.Series({"ES::D::m1": 0.60}, dtype=float)

        decoded = decode_global_weight_layer_output(
            [encoded],
            decode_map,
            global_weights=global_weights,
            global_fdm=5.0,
        )
        assert decoded.sort_values("ticker")["forecast_score"].tolist() == pytest.approx([2.0, 2.0])

        rollups = build_global_adapter_rollups(decode_map, global_weights)
        assert rollups["stream_weights"] == {"ES::D::m1": 0.6}
        assert rollups["ticker_rollups"] == {"ES": 0.6}
        assert rollups["timeframe_rollups"] == {"D": 0.6}

    def test_removed_sector_constructor_surface_rejected(self):
        with pytest.raises(TypeError):
            GlobalPortfolio(
                tf_portfolios=[],
                sector_allocation_config_path="config.json",  # type: ignore[arg-type]
            )
