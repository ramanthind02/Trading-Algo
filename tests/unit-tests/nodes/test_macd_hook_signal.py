"""Unit tests for MacdHookSignal (simplified Algomatic MACD hook)."""

from __future__ import annotations

from datetime import datetime, timedelta

from nodes.momentum.core.macd_hook_signal import (
    MacdHookSignal,
    _histogram_hook,
)
from lib.core import helpers
from lib.core.enums import Ticker, TimeFrame
from lib.core.models import Candle


def _candle(day: int, *, close: float, open_: float | None = None) -> Candle:
    o = open_ if open_ is not None else close
    return Candle(
        datetime=datetime(2020, 1, 2) + timedelta(days=day),
        open=o,
        high=max(o, close) + 0.25,
        low=min(o, close) - 0.25,
        close=close,
        volume=1.0,
        ticker=Ticker.GC,
        tf=TimeFrame.D,
    )


def test_create_fresh_bias_node_resolves() -> None:
    node = helpers.create_fresh_bias_node(
        "macd_hook_signal",
        Ticker.GC,
        TimeFrame.D,
        {
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "trend_ema_period": 15,
            "max_hold_bars": 10,
        },
    )
    assert type(node).__name__ == "MacdHookSignal"
    assert node.params["macdFast"] == 12


def test_macd_slow_must_exceed_fast() -> None:
    import pytest

    with pytest.raises(ValueError, match="macd_slow"):
        MacdHookSignal(Ticker.GC, TimeFrame.D, macd_fast=26, macd_slow=26)


def test_histogram_hook_detects_turn_after_pullback() -> None:
    assert _histogram_hook(-0.2, -0.5, -0.3) is True
    assert _histogram_hook(-0.4, -0.3, -0.5) is False
    assert _histogram_hook(-0.1, -0.2, -0.3) is False


def test_entry_when_conditions_met() -> None:
    node = MacdHookSignal(Ticker.GC, TimeFrame.D, max_hold_bars=10)
    for _ in range(node.front_bad):
        node.add_candle(_candle(0, close=100.0))

    node._n_prices = node.front_bad + 1
    node._macd_line = 0.5
    node._hist.clear()
    node._hist.extend([-0.4, -0.6, -0.3])
    node._prev_trend_ema = 100.0
    node._trend_ema = 100.5

    out = node._apply_position_rules()
    assert out == 1.0
    assert node._position == 1


def test_time_exit_clears_position() -> None:
    node = MacdHookSignal(Ticker.GC, TimeFrame.D, max_hold_bars=3)
    node._position = 1
    node._bars_in_trade = 2
    node._macd_line = -0.5
    node._hist.clear()
    node._hist.extend([0.1, 0.0, -0.1])

    out = node._apply_position_rules()
    assert out == 0.0
    assert node._position == 0
