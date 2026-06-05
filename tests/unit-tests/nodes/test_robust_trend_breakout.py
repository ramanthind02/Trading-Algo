"""Unit tests for RobustTrendBreakout long/short modes."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from nodes.breakout.donchian.robust_trend_breakout import RobustTrendBreakout
from lib.core.enums import Ticker, TimeFrame
from lib.core.models import Candle


def _candle(day: int, close: float, *, spread: float = 2.0) -> Candle:
    return Candle(
        datetime=datetime(2010, 1, 4) + timedelta(days=day),
        open=close,
        high=close + spread,
        low=close - spread,
        close=close,
        volume=1.0,
        ticker=Ticker.CL,
        tf=TimeFrame.D,
    )


def test_long_short_emits_signed_discrete_signal() -> None:
    node = RobustTrendBreakout(
        Ticker.CL,
        TimeFrame.D,
        lookback=5,
        ema_period=10,
        atr_len=5,
        atr_mult=2.0,
        strategy_mode="long_short",
    )
    signals = [node.add_candle(_candle(day, 80.0 + (day % 11) * 0.4))[0] for day in range(500)]
    live = signals[node.front_bad :]
    assert live
    assert all(signal in {-1.0, 0.0, 1.0} for signal in live)


def test_long_only_never_emits_short() -> None:
    node = RobustTrendBreakout(
        Ticker.CL,
        TimeFrame.D,
        lookback=5,
        ema_period=10,
        atr_len=5,
        strategy_mode="long",
    )
    signals = [node.add_candle(_candle(day, 70.0 + (day % 9) * 0.6))[0] for day in range(500)]
    assert all(signal >= 0.0 for signal in signals[node.front_bad :])


def test_invalid_strategy_mode_raises() -> None:
    with pytest.raises(ValueError, match="Invalid direction"):
        RobustTrendBreakout(
            Ticker.CL,
            TimeFrame.D,
            lookback=5,
            ema_period=10,
            strategy_mode="invalid",
        )
