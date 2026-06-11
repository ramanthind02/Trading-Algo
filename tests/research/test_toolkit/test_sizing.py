"""Unit tests for research.toolkit.sizing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.toolkit.sizing import apply_vol_target


def _make_fixture(
    n_days: int = 20,
    base_price: float = 100.0,
    step: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic bars + one trade on day index 15."""
    days = pd.to_datetime([f"2020-01-{d+1:02d}" for d in range(n_days)])
    closes = base_price + np.arange(n_days) * step
    bars = pd.DataFrame({"day": days, "close": closes})
    trade_day = days[15]
    trades = pd.DataFrame(
        {
            "day":      [trade_day],
            "ret":      [0.03],
            "gross_ret":[0.04],
            "net_pts":  [3.0],
        }
    )
    return bars, trades


class TestApplyVolTarget:
    def test_adds_required_columns(self) -> None:
        bars, trades = _make_fixture()
        out = apply_vol_target(trades, bars, vol_target=0.02, vol_lookback_days=5)
        assert "lev" in out.columns
        assert "ret_sized" in out.columns
        assert "gross_ret_sized" in out.columns

    def test_ret_sized_equals_ret_times_lev(self) -> None:
        bars, trades = _make_fixture()
        out = apply_vol_target(trades, bars, vol_target=0.02, vol_lookback_days=5)
        lev = float(out["lev"].iloc[0])
        expected = float(out["ret"].iloc[0]) * lev
        assert abs(float(out["ret_sized"].iloc[0]) - expected) < 1e-12

    def test_gross_ret_sized_equals_gross_ret_times_lev(self) -> None:
        bars, trades = _make_fixture()
        out = apply_vol_target(trades, bars, vol_target=0.02, vol_lookback_days=5)
        lev = float(out["lev"].iloc[0])
        expected = float(out["gross_ret"].iloc[0]) * lev
        assert abs(float(out["gross_ret_sized"].iloc[0]) - expected) < 1e-12

    def test_lev_cap_respected(self) -> None:
        """Very low sigma → leverage formula blows up; must be capped."""
        days = pd.to_datetime([f"2020-01-{d+1:02d}" for d in range(20)])
        # Perfectly flat prices → pct_change = 0 → sigma = 0 → lev = lev_cap
        bars = pd.DataFrame({"day": days, "close": np.ones(20) * 100.0})
        trades = pd.DataFrame({
            "day": [days[15]], "ret": [0.01], "gross_ret": [0.01],
        })
        out = apply_vol_target(
            trades, bars, vol_target=0.02, vol_lookback_days=5, lev_cap=3.0
        )
        assert float(out["lev"].iloc[0]) <= 3.0 + 1e-9

    def test_zero_lev_when_insufficient_history(self) -> None:
        """If the rolling window has no full lookback, lev should be 0 (NaN mapped to 0)."""
        days = pd.to_datetime([f"2020-01-{d+1:02d}" for d in range(10)])
        bars = pd.DataFrame({"day": days, "close": 100.0 + np.arange(10)})
        # Trade on day 0: before any lookback window fills → sigma NaN → lev 0
        trades = pd.DataFrame({
            "day": [days[1]], "ret": [0.01], "gross_ret": [0.01],
        })
        out = apply_vol_target(
            trades, bars, vol_target=0.02, vol_lookback_days=8, lev_cap=4.0
        )
        assert float(out["lev"].iloc[0]) == 0.0

    def test_empty_trades_returns_empty(self) -> None:
        bars, _ = _make_fixture()
        out = apply_vol_target(pd.DataFrame(), bars, vol_target=0.02)
        assert out.empty
        assert "lev" in out.columns
        assert "ret_sized" in out.columns

    def test_known_vol_target_scaling(self) -> None:
        """Hand-computed leverage check for a deterministic return series."""
        # Daily close-to-close returns: known, so we can compute expected sigma
        # and leverage by hand.
        # Prices: 100, 110, 99, 107.8, 103.5, ... (6 days, trade on day 5)
        days = pd.to_datetime([f"2020-01-{d+1:02d}" for d in range(6)])
        prices = np.array([100.0, 110.0, 99.0, 107.8, 103.5, 105.0])
        bars = pd.DataFrame({"day": days, "close": prices})
        trades = pd.DataFrame({"day": [days[5]], "ret": [0.01], "gross_ret": [0.01]})

        # pct_change: [NaN, 0.1, -0.1, 0.0888, -0.0398, 0.0145]
        # shift(1).rolling(3): uses indices 1,2,3 for day=4 (index 4)
        pct = pd.Series(prices).pct_change()
        sigma_series = pct.shift(1).rolling(3, min_periods=3).std(ddof=1)
        sigma5 = float(sigma_series.iloc[5])  # sigma used on trade day (index 5)
        lev_cap = 4.0
        vol_target = 0.02
        expected_lev = min(lev_cap, vol_target / sigma5) if sigma5 > 0 else 0.0

        out = apply_vol_target(
            trades, bars, vol_target=vol_target, vol_lookback_days=3, lev_cap=lev_cap
        )
        assert abs(float(out["lev"].iloc[0]) - expected_lev) < 1e-8
