"""Unit tests for AlgomaticMomentumSignal."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from nodes.momentum.core.algomatic_momentum_signal import AlgomaticMomentumSignal
from utils.core import helpers
from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle


def _candle(day: int, *, c: float, h: Optional[float] = None) -> Candle:
    high = h if h is not None else c + 0.5
    return Candle(
        datetime=datetime(2020, 1, 1) + timedelta(days=day),
        open=c,
        high=high,
        low=c - 0.5,
        close=c,
        volume=1.0,
        ticker=Ticker.ES,
        tf=TimeFrame.D,
    )


def test_create_fresh_bias_node_resolves() -> None:
    node = helpers.create_fresh_bias_node(
        "algomatic_momentum_signal",
        Ticker.ES,
        TimeFrame.D,
        {
            "momentum_lookback": 2,
            "rsi_period": 2,
            "rsi_max": 90.0,
            "exit_bars": 2,
        },
    )
    assert type(node).__name__ == "AlgomaticMomentumSignal"


def test_flat_series_stays_out_until_warmup_then_zero() -> None:
    """Constant prices => zero momentum => no entries; post-warmup signal stays 0."""
    node = AlgomaticMomentumSignal(
        Ticker.ES,
        TimeFrame.D,
        momentum_lookback=2,
        rsi_period=2,
        rsi_max=90.0,
        exit_bars=2,
    )
    # front_bad = max(4, 3, 3) = 4
    signals = [node.add_candle(_candle(i, c=100.0))[0] for i in range(20)]
    assert all(s == 0.0 for s in signals[:4])
    assert all(s == 0.0 for s in signals[4:])


def test_exit_triggers_when_close_exceeds_lagged_high() -> None:
    """When long, exit when close > high from exit_bars ago."""
    node = AlgomaticMomentumSignal(
        Ticker.ES,
        TimeFrame.D,
        momentum_lookback=2,
        rsi_period=2,
        rsi_max=90.0,
        exit_bars=2,
    )
    for i in range(6):
        node.add_candle(_candle(i, c=100.0))
    node.position = 1
    node.high_buffer.clear()
    node.high_buffer.append(10.0)
    node.high_buffer.append(10.0)
    node.high_buffer.append(10.0)
    out = node.add_candle(_candle(6, c=11.0, h=11.0))[0]
    assert out == 0.0
    assert node.position == 0
