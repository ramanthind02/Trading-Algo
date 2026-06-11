"""Unit tests for research.toolkit.metrics on synthetic trade frames.

All data is constructed inline — no file I/O, no real market data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.toolkit.metrics import (
    TRADING_DAYS,
    avg_r,
    by_year,
    daily_grid_returns,
    daily_sharpe,
    max_drawdown,
    profit_factor,
    summary,
    win_rate,
)


# ---------------------------------------------------------------------------
# daily_sharpe
# ---------------------------------------------------------------------------

class TestDailySharpe:
    def test_known_values(self) -> None:
        arr = np.array([0.01, 0.02, -0.01, 0.03])
        expected = float(arr.mean() / arr.std(ddof=1) * np.sqrt(TRADING_DAYS))
        assert abs(daily_sharpe(arr) - expected) < 1e-12

    def test_zero_variance_returns_zero(self) -> None:
        assert daily_sharpe(np.zeros(10)) == 0.0

    def test_constant_series_returns_zero(self) -> None:
        assert daily_sharpe(np.ones(10) * 0.05) == 0.0

    def test_single_element_returns_zero(self) -> None:
        # std is undefined for n=1 → should not crash
        assert daily_sharpe(np.array([0.05])) == 0.0

    def test_positive_returns_positive_sharpe(self) -> None:
        arr = np.array([0.01, 0.02, 0.015, 0.008])
        assert daily_sharpe(arr) > 0.0

    def test_all_negative_returns_negative_sharpe(self) -> None:
        arr = np.array([-0.01, -0.02, -0.015])
        assert daily_sharpe(arr) < 0.0


# ---------------------------------------------------------------------------
# profit_factor
# ---------------------------------------------------------------------------

class TestProfitFactor:
    def test_known_answer(self) -> None:
        # wins = 2.0 + 3.0 = 5.0;  losses = 1.0 + 0.5 = 1.5  →  PF = 5/1.5
        pnl = pd.Series([2.0, -1.0, 3.0, -0.5])
        assert abs(profit_factor(pnl) - 5.0 / 1.5) < 1e-12

    def test_no_losses_is_inf(self) -> None:
        assert profit_factor(pd.Series([1.0, 2.0])) == float("inf")

    def test_all_losses_is_zero(self) -> None:
        assert profit_factor(pd.Series([-1.0, -2.0])) == 0.0

    def test_break_even(self) -> None:
        assert abs(profit_factor(pd.Series([1.0, -1.0])) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# win_rate
# ---------------------------------------------------------------------------

class TestWinRate:
    def test_known_answer(self) -> None:
        pnl = pd.Series([1.0, -1.0, 1.0])
        assert abs(win_rate(pnl) - 2.0 / 3.0) < 1e-12

    def test_empty_series_returns_zero(self) -> None:
        assert win_rate(pd.Series(dtype=float)) == 0.0

    def test_all_winners(self) -> None:
        assert win_rate(pd.Series([1.0, 2.0])) == 1.0

    def test_all_losers(self) -> None:
        assert win_rate(pd.Series([-1.0, -2.0])) == 0.0


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_known_answer(self) -> None:
        # cumsum: [0.1, 0.3, -0.2, -0.1]
        # running max: [0.1, 0.3, 0.3, 0.3]
        # drawdown: [0.0, 0.0, 0.5, 0.4]  →  max = 0.5
        arr = np.array([0.1, 0.2, -0.5, 0.1])
        assert abs(max_drawdown(arr) - 0.5) < 1e-12

    def test_monotone_rising_is_zero(self) -> None:
        assert max_drawdown(np.array([0.1, 0.2, 0.3])) == 0.0

    def test_empty_is_zero(self) -> None:
        assert max_drawdown(np.array([])) == 0.0

    def test_single_negative_is_zero(self) -> None:
        # cumsum([-0.3]) = [-0.3]; running_max = [-0.3]; drawdown = [-0.3] - [-0.3] = 0
        # No peak above the single data point → drawdown is 0. Matches engine behaviour.
        assert max_drawdown(np.array([-0.3])) == 0.0


# ---------------------------------------------------------------------------
# avg_r
# ---------------------------------------------------------------------------

class TestAvgR:
    def test_known_answer(self) -> None:
        r = pd.Series([1.0, -1.0, 2.0])
        assert abs(avg_r(r) - 2.0 / 3.0) < 1e-12

    def test_empty_is_zero(self) -> None:
        assert avg_r(pd.Series(dtype=float)) == 0.0


# ---------------------------------------------------------------------------
# daily_grid_returns
# ---------------------------------------------------------------------------

class TestDailyGridReturns:
    def test_fills_grid_with_trade_days(self) -> None:
        days = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
        trades = pd.DataFrame({
            "day": pd.to_datetime(["2020-01-01", "2020-01-03"]),
            "ret": [0.02, -0.01],
        })
        arr = daily_grid_returns(trades, days.values, "ret")
        np.testing.assert_allclose(arr, [0.02, 0.0, -0.01])

    def test_flat_days_are_zero(self) -> None:
        days = pd.to_datetime(["2020-01-01", "2020-01-02"])
        trades = pd.DataFrame({"day": pd.to_datetime(["2020-01-01"]), "ret": [0.05]})
        arr = daily_grid_returns(trades, days.values, "ret")
        np.testing.assert_allclose(arr, [0.05, 0.0])

    def test_multiple_trades_per_day_summed(self) -> None:
        days = pd.to_datetime(["2020-01-01", "2020-01-02"])
        trades = pd.DataFrame({
            "day": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "ret": [0.01, 0.02],
        })
        arr = daily_grid_returns(trades, days.values, "ret")
        np.testing.assert_allclose(arr, [0.03, 0.0])

    def test_empty_trades_all_zeros(self) -> None:
        days = pd.to_datetime(["2020-01-01", "2020-01-02"])
        arr = daily_grid_returns(pd.DataFrame(columns=["day", "ret"]), days.values, "ret")
        np.testing.assert_array_equal(arr, [0.0, 0.0])


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

class TestSummary:
    def _make_trades(self) -> tuple[pd.DataFrame, np.ndarray]:
        days = pd.to_datetime([f"2020-01-{d:02d}" for d in range(1, 11)])
        trades = pd.DataFrame({
            "day":      pd.to_datetime(["2020-01-02", "2020-01-05", "2020-01-08"]),
            "ret":      [0.05, -0.02, 0.03],
            "gross_ret":[0.06, -0.01, 0.04],
            "net_pts":  [5.0, -2.0, 3.0],
            "R":        [2.5, -1.0, 1.5],
        })
        return trades, days.values

    def test_trade_count(self) -> None:
        trades, session_days = self._make_trades()
        assert summary(trades, session_days)["trades"] == 3

    def test_profit_factor(self) -> None:
        trades, session_days = self._make_trades()
        # net_pts: wins = 5+3 = 8, losses = 2 → PF = 4.0
        assert summary(trades, session_days)["PF"] == round(8.0 / 2.0, 2)

    def test_win_rate(self) -> None:
        trades, session_days = self._make_trades()
        assert summary(trades, session_days)["win%"] == round(100.0 * 2.0 / 3.0, 1)

    def test_sharpe_nonzero(self) -> None:
        trades, session_days = self._make_trades()
        assert summary(trades, session_days)["sharpe"] != 0.0

    def test_empty_trades(self) -> None:
        session_days = pd.to_datetime(["2020-01-01"]).values
        m = summary(pd.DataFrame(), session_days)
        assert m["trades"] == 0
        assert m["sharpe"] == 0.0
        assert m["PF"] == 0.0
