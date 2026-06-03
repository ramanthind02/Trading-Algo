"""Unit tests for PercentBSignal (Bollinger %B crosses)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from nodes.mean_reversion.bollinger.percent_b_signal import PercentBSignal
from utils.core import helpers
from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle


def _candle(day: int, close: float) -> Candle:
    return Candle(
        datetime=datetime(2020, 1, 1) + timedelta(days=day),
        open=close,
        high=close + 0.5,
        low=close - 0.5,
        close=close,
        volume=1.0,
        ticker=Ticker.ES,
        tf=TimeFrame.D,
    )


def test_create_fresh_bias_node_resolves() -> None:
    node = helpers.create_fresh_bias_node(
        "percent_b_signal",
        Ticker.ES,
        TimeFrame.D,
        {
            "period": 20,
            "std_dev": 2.0,
            "lower_threshold": 0.0,
            "upper_threshold": 1.0,
            "strategy_mode": "long",
            "exit_policy": "threshold",
            "exit_bars": 5,
        },
    )
    assert type(node).__name__ == "PercentBSignal"


def test_long_exits_on_cross_above_upper_band() -> None:
    node = PercentBSignal(
        Ticker.ES,
        TimeFrame.D,
        period=5,
        std_dev=2.0,
        lower_threshold=0.0,
        upper_threshold=1.0,
        strategy_mode="long",
        exit_policy="threshold",
        exit_bars=5,
    )
    base = 100.0
    for i in range(30):
        node.add_candle(_candle(i, base))

    # Sharp rally should eventually cross %B above 1 and flatten a long.
    long_seen = False
    flat_after_long = False
    for i in range(30, 120):
        out = node.add_candle(_candle(i, base + (i - 30) * 2.0))[0]
        if out == 1.0:
            long_seen = True
        if long_seen and out == 0.0:
            flat_after_long = True
            break
    assert long_seen
    assert flat_after_long


def test_invalid_thresholds_raise() -> None:
    with pytest.raises(ValueError, match="lower_threshold must be"):
        PercentBSignal(
            Ticker.ES,
            TimeFrame.D,
            lower_threshold=1.0,
            upper_threshold=0.5,
        )
