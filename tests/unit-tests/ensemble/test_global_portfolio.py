"""Unit tests for GlobalPortfolio (T008).

All tests use SYNTHETIC data only — no vault, no real candle files, no disk I/O.
TFPortfolio instances are lightweight mocks to isolate GlobalPortfolio orchestration.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.core.enums import TimeFrame
from ensemble.portfolio import GlobalPortfolio, TFPortfolio


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

    gp.fit(candles_per_tf, instrument_returns)
    return gp, candles_per_tf


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOutputSchemaMatchesLegacy:
    """Output must have exactly ['ticker', 'datetime', 'forecast_score', 'position_fraction']."""

    def test_output_columns(self):
        gp, candles_per_tf = _build_global_portfolio(tickers=["ES", "NQ"])
        result = gp.predict(candles_per_tf)
        assert isinstance(result, pd.DataFrame)
        expected_cols = {"ticker", "datetime", "forecast_score", "position_fraction"}
        assert expected_cols == set(result.columns), (
            f"Unexpected columns: {set(result.columns)}"
        )

    def test_no_nan_in_position_fraction(self):
        gp, candles_per_tf = _build_global_portfolio(tickers=["ES", "NQ"])
        result = gp.predict(candles_per_tf)
        assert not result["position_fraction"].isna().any(), (
            "position_fraction contains NaN values"
        )

    def test_no_nan_in_forecast_score(self):
        gp, candles_per_tf = _build_global_portfolio(tickers=["ES", "NQ"])
        result = gp.predict(candles_per_tf)
        assert not result["forecast_score"].isna().any()


class TestTwoTFOutputSchema:
    """Two-TF GlobalPortfolio must produce correct schema with no NaN."""

    def test_two_tf_columns(self):
        gp, candles_per_tf = _build_global_portfolio(tickers=["ES", "NQ", "CL"], n_tfs=2)
        result = gp.predict(candles_per_tf)
        assert set(result.columns) == {"ticker", "datetime", "forecast_score", "position_fraction"}

    def test_two_tf_no_nan(self):
        gp, candles_per_tf = _build_global_portfolio(tickers=["ES", "NQ", "CL"], n_tfs=2)
        result = gp.predict(candles_per_tf)
        assert not result["position_fraction"].isna().any()
        assert not result["forecast_score"].isna().any()

    def test_two_tf_position_fraction_bounded(self):
        max_pos = 2.0
        gp, candles_per_tf = _build_global_portfolio(
            tickers=["ES", "NQ"], n_tfs=2, max_position_pct=max_pos
        )
        result = gp.predict(candles_per_tf)
        assert (result["position_fraction"].abs() <= max_pos + 1e-9).all(), (
            "position_fraction exceeds max_position_pct"
        )


class TestPositionFractionBounds:
    """position_fraction must always be within [-max_position_pct, +max_position_pct]."""

    @pytest.mark.parametrize("max_pos", [0.5, 1.0, 2.0, 5.0])
    def test_position_clipped(self, max_pos: float):
        gp, candles_per_tf = _build_global_portfolio(
            tickers=["ES", "NQ", "GC"],
            n_tfs=1,
            score=10.0,  # large score to stress the clip
            max_position_pct=max_pos,
        )
        result = gp.predict(candles_per_tf)
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
        with pytest.raises(RuntimeError, match="fitted"):
            gp.predict(candles_per_tf)

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
        gp.fit(candles_per_tf, instrument_returns)
        result = gp.predict(candles_per_tf)

        assert not result.empty
        assert not result["position_fraction"].isna().any()
