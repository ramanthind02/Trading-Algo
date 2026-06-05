"""Unit tests for PullbackContinuationSignal."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from nodes.momentum.core.pullback_continuation_signal import PullbackContinuationSignal
from lib.core import helpers
from lib.core.enums import Ticker, TimeFrame
from lib.core.models import Candle


def _candle(
    day: int,
    *,
    close: float,
    low: float | None = None,
    high: float | None = None,
) -> Candle:
    lo = low if low is not None else close - 0.5
    hi = high if high is not None else close + 0.5
    return Candle(
        datetime=datetime(2020, 1, 2) + timedelta(days=day),
        open=close,
        high=hi,
        low=lo,
        close=close,
        volume=1.0,
        ticker=Ticker.GC,
        tf=TimeFrame.D,
    )


def test_create_fresh_bias_node_resolves() -> None:
    node = helpers.create_fresh_bias_node(
        "pullback_continuation_signal",
        Ticker.GC,
        TimeFrame.D,
        {
            "trend_ema_period": 50,
            "pullback_ema_period": 10,
            "atr_len": 5,
            "atr_mult": 2.0,
            "atr_pullback_mult": 1.0,
            "max_hold_bars": 0,
        },
    )
    assert type(node).__name__ == "PullbackContinuationSignal"


def test_trend_period_must_exceed_pullback_period() -> None:
    with pytest.raises(ValueError, match="trend_ema_period"):
        PullbackContinuationSignal(
            Ticker.GC,
            TimeFrame.D,
            trend_ema_period=20,
            pullback_ema_period=20,
        )


def test_entry_after_dip_and_reclaim() -> None:
    node = PullbackContinuationSignal(
        Ticker.GC,
        TimeFrame.D,
        trend_ema_period=10,
        pullback_ema_period=5,
        atr_len=3,
        atr_mult=2.0,
        atr_pullback_mult=0.0,
        max_hold_bars=0,
    )
    node._n_bars = node.front_bad
    node._atr_val = 1.0
    node._ema_trend = 100.0
    node._ema_pullback = 101.0
    node._pullback_armed = True

    node._apply_flat_logic(102.0, 1.0)
    node._trail_stop(102.0, 1.0)
    assert float(node._position) == 1.0
    assert node._chandelier_stop == pytest.approx(100.0)


def test_atr_stop_clears_position() -> None:
    node = PullbackContinuationSignal(
        Ticker.GC,
        TimeFrame.D,
        trend_ema_period=10,
        pullback_ema_period=5,
        atr_len=3,
        atr_mult=2.0,
    )
    node._position = 1
    node._chandelier_stop = 100.0
    node._bars_in_trade = 3

    node._apply_exits(low=99.0)
    assert node._position == 0
    assert node._chandelier_stop is None


def test_max_hold_exit_when_enabled() -> None:
    node = PullbackContinuationSignal(
        Ticker.GC,
        TimeFrame.D,
        trend_ema_period=10,
        pullback_ema_period=5,
        atr_len=3,
        atr_mult=2.0,
        max_hold_bars=3,
    )
    node._position = 1
    node._chandelier_stop = 90.0
    node._bars_in_trade = 2

    node._apply_exits(low=100.0)
    assert node._position == 0
