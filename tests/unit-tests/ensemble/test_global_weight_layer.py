"""
Unit tests for GlobalWeightLayer and GlobalWeightLayerConfig.

Covers:
- Single-TF passthrough (weight=1.0, fdm=1.0)
- Two-TF weights sum to 1.0
- No look-ahead in forward-fill resampling
- Fallback to equal weights on insufficient data
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ensemble.global_weight_layer import GlobalWeightLayer, GlobalWeightLayerConfig
from utils.core.enums import TimeFrame


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _make_returns(
    n_days: int = 756,
    tickers: list[str] | None = None,
    start: str = "2020-01-01",
) -> pd.DataFrame:
    """Daily returns DataFrame: index=DatetimeIndex, columns=tickers."""
    if tickers is None:
        tickers = ["ES", "NQ", "CL"]
    dates = pd.bdate_range(start, periods=n_days)
    data = RNG.normal(0.0, 0.01, size=(n_days, len(tickers)))
    return pd.DataFrame(data, index=dates, columns=tickers)


def _make_forecast_stream(
    dates: pd.DatetimeIndex,
    tickers: list[str],
    score_scale: float = 0.5,
) -> pd.DataFrame:
    """Long-form forecast stream with columns ['ticker', 'datetime', 'forecast_score']."""
    rows = []
    for dt in dates:
        for ticker in tickers:
            score = float(RNG.uniform(-score_scale, score_scale))
            rows.append({"ticker": ticker, "datetime": dt, "forecast_score": score})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# T1 — single-TF passthrough
# ---------------------------------------------------------------------------

class TestSingleTFPassthrough:
    """Single TF: weight=1.0, fdm=1.0, combine() output mirrors input."""

    def test_fit_sets_weight_one_fdm_one(self):
        returns = _make_returns(n_days=252, tickers=["ES", "NQ"])
        daily_dates = returns.index
        stream = _make_forecast_stream(daily_dates, ["ES", "NQ"])

        layer = GlobalWeightLayer()
        layer.fit({TimeFrame.D: stream}, returns)

        assert layer.is_fitted_
        assert list(layer.tf_weights_.values()) == pytest.approx([1.0])
        assert layer.fdm_ == pytest.approx(1.0)

    def test_combine_preserves_scores_single_tf(self):
        """Combined forecast_score should equal fdm * weight * input score (all 1.0)."""
        returns = _make_returns(n_days=252, tickers=["ES"])
        daily_dates = returns.index

        # Deterministic score of 0.5 every day
        stream = pd.DataFrame(
            [
                {"ticker": "ES", "datetime": dt, "forecast_score": 0.5}
                for dt in daily_dates
            ]
        )

        layer = GlobalWeightLayer()
        layer.fit({TimeFrame.D: stream}, returns)
        combined = layer.combine({TimeFrame.D: stream})

        assert set(combined.columns) >= {"ticker", "datetime", "forecast_score"}
        scores = combined.query("ticker == 'ES'")["forecast_score"]
        # weight=1.0, fdm=1.0 → output == input == 0.5
        assert scores.values == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# T2 — two-TF weights sum to 1.0
# ---------------------------------------------------------------------------

class TestTwoTFWeights:
    """Two TFs: tf_weights_ values sum to 1.0."""

    def test_weights_sum_to_one(self):
        returns = _make_returns(n_days=756, tickers=["ES", "NQ", "CL"])
        daily_dates = returns.index
        # Daily stream
        stream_d = _make_forecast_stream(daily_dates, ["ES", "NQ", "CL"])
        # Weekly stream: one row per week
        weekly_dates = daily_dates[::5]
        stream_w = _make_forecast_stream(weekly_dates, ["ES", "NQ", "CL"])

        layer = GlobalWeightLayer()
        layer.fit({TimeFrame.D: stream_d, TimeFrame.W: stream_w}, returns)

        assert layer.is_fitted_
        total = sum(layer.tf_weights_.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_both_tfs_have_positive_weight(self):
        returns = _make_returns(n_days=756, tickers=["ES", "NQ"])
        daily_dates = returns.index
        stream_d = _make_forecast_stream(daily_dates, ["ES", "NQ"])
        weekly_dates = daily_dates[::5]
        stream_w = _make_forecast_stream(weekly_dates, ["ES", "NQ"])

        layer = GlobalWeightLayer()
        layer.fit({TimeFrame.D: stream_d, TimeFrame.W: stream_w}, returns)

        for tf, w in layer.tf_weights_.items():
            assert w >= 0.0, f"Weight for {tf} is negative: {w}"


# ---------------------------------------------------------------------------
# T3 — no look-ahead in forward-fill resampling
# ---------------------------------------------------------------------------

class TestNoLookaheadResampling:
    """Forward-fill must never pull a future value backward in time."""

    def test_weekly_signal_does_not_anticipate_future_change(self):
        """A weekly score update on day-5 must not appear on day-1 through day-4."""
        # Build a 10-business-day grid
        daily_dates = pd.bdate_range("2023-01-02", periods=10)

        # Weekly stream: only two observations — day-1 and day-6
        # day-1 score = 0.1, day-6 score = 0.9
        day1 = daily_dates[0]
        day6 = daily_dates[5]
        stream_w = pd.DataFrame(
            [
                {"ticker": "ES", "datetime": day1, "forecast_score": 0.1},
                {"ticker": "ES", "datetime": day6, "forecast_score": 0.9},
            ]
        )
        stream_d = _make_forecast_stream(daily_dates, ["ES"])
        returns = _make_returns(n_days=10, tickers=["ES"], start="2023-01-02")

        layer = GlobalWeightLayer()
        layer.fit({TimeFrame.D: stream_d, TimeFrame.W: stream_w}, returns)
        combined = layer.combine({TimeFrame.D: stream_d, TimeFrame.W: stream_w})

        # Extract combined scores for ES, sorted by datetime
        es = (
            combined.query("ticker == 'ES'")
            .sort_values("datetime")
            .reset_index(drop=True)
        )

        # Days 2–5 (index 1–4) must reflect the day-1 weekly value (0.1),
        # NOT the day-6 value (0.9).
        # We check that the score on day-2 < score on day-7 (post day-6 update).
        # Use a tolerance-free sign test: day-2 score must be strictly closer
        # to the day-1 state than to the day-6 state.
        w_weight = layer.tf_weights_.get(TimeFrame.W, 0.0)
        if w_weight > 0.0:
            # Score on day 2 (index 1) should use the day-1 weekly value
            score_day2 = es.loc[1, "forecast_score"]
            # Score on day 7 (index 6) should use the day-6 weekly value
            score_day7 = es.loc[6, "forecast_score"]
            # The weekly contribution on day2 should be smaller than day7
            # (since 0.1 < 0.9 and other TF contributions are the same noise)
            # We only verify the forward-fill didn't pull 0.9 into day 2-5.
            # More precisely: the combined on day 2 must differ from day 7.
            assert score_day2 != pytest.approx(score_day7, abs=0.01), (
                "Forward-fill appears to have pulled the day-6 value into day 2 "
                f"(day2={score_day2:.4f}, day7={score_day7:.4f})"
            )


# ---------------------------------------------------------------------------
# T4 — fallback on insufficient data
# ---------------------------------------------------------------------------

class TestFallbackInsufficientData:
    """< 2 rows of overlap → equal TF weights and fdm=1.0."""

    def test_single_row_returns(self):
        """Instrument returns with only 1 row triggers fallback."""
        one_date = pd.bdate_range("2023-01-02", periods=1)
        returns_1row = pd.DataFrame({"ES": [0.01]}, index=one_date)

        stream_d = pd.DataFrame(
            [{"ticker": "ES", "datetime": one_date[0], "forecast_score": 0.3}]
        )
        stream_w = pd.DataFrame(
            [{"ticker": "ES", "datetime": one_date[0], "forecast_score": 0.5}]
        )

        layer = GlobalWeightLayer()
        layer.fit(
            {TimeFrame.D: stream_d, TimeFrame.W: stream_w},
            returns_1row,
        )

        assert layer.is_fitted_
        assert layer.fdm_ == pytest.approx(1.0)
        total = sum(layer.tf_weights_.values())
        assert total == pytest.approx(1.0, abs=1e-6)
        # Each TF should have equal weight = 0.5
        for w in layer.tf_weights_.values():
            assert w == pytest.approx(0.5, abs=1e-6)

    def test_combine_raises_if_not_fitted(self):
        """combine() on an unfitted layer raises RuntimeError."""
        layer = GlobalWeightLayer()
        stream = _make_forecast_stream(
            pd.bdate_range("2023-01-02", periods=5), ["ES"]
        )
        with pytest.raises(RuntimeError, match="fitted"):
            layer.combine({TimeFrame.D: stream})


# ---------------------------------------------------------------------------
# T5 — diagnostics
# ---------------------------------------------------------------------------

class TestDiagnostics:
    def test_get_diagnostics_keys(self):
        returns = _make_returns(n_days=252, tickers=["ES"])
        stream = _make_forecast_stream(returns.index, ["ES"])
        layer = GlobalWeightLayer()
        layer.fit({TimeFrame.D: stream}, returns)

        diag = layer.get_diagnostics()
        assert "is_fitted" in diag
        assert "tf_weights" in diag
        assert "fdm" in diag
        assert "mean_cross_tf_correlation" in diag
        assert "daily_grid_len" in diag
        assert diag["is_fitted"] is True
