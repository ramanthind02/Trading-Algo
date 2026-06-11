"""Unit tests for the equity-curve regime filter (execution.equity_curve_filter)."""

import pandas as pd
import pytest

from execution.equity_curve_filter import (
    apply_equity_curve_filter,
    equity_curve_gate,
    time_in_market,
)


def test_window_validation():
    with pytest.raises(ValueError):
        equity_curve_gate(pd.Series([0.01, 0.01, 0.01]), window=1)


def test_warmup_is_on():
    r = pd.Series([0.01] * 60)
    g = equity_curve_gate(r, window=20)
    # Before the SMA exists the gate must be ON (never suppress for lack of history).
    assert (g.iloc[:20] == 1.0).all()


def test_lookahead_free():
    # Changing the return at bar t must not change the gate at or before t.
    r1 = pd.Series([0.01] * 60)
    r2 = r1.copy()
    r2.iloc[40] = -0.5
    g1 = equity_curve_gate(r1, window=10)
    g2 = equity_curve_gate(r2, window=10)
    assert g1.iloc[:41].equals(g2.iloc[:41])      # gates through bar 40 unchanged
    assert g1.iloc[41] != g2.iloc[41]             # the shock only bites from bar 41


def test_gate_off_in_drawdown_on_in_uptrend():
    r = pd.Series([0.01] * 40 + [-0.01] * 40)     # rise then fall
    g = equity_curve_gate(r, window=10)
    assert g.iloc[35] == 1.0                       # mid up-trend: invested
    assert g.iloc[-1] == 0.0                       # deep draw-down: flat


def test_apply_zeros_returns_when_flat():
    r = pd.Series([0.01] * 40 + [-0.01] * 40)
    g = equity_curve_gate(r, window=10)
    f = apply_equity_curve_filter(r, window=10)
    assert (f[g == 0.0] == 0.0).all()              # flat bars contribute nothing
    assert (f[g == 1.0] == r[g == 1.0]).all()      # invested bars pass through


def test_time_in_market_bounds():
    r = pd.Series([0.01] * 30 + [-0.01] * 30)
    frac = time_in_market(r, window=10)
    assert 0.0 <= frac <= 1.0
    assert frac < 1.0                              # the draw-down half is gated out
