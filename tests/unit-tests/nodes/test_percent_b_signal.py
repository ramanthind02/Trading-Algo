"""Unit tests for PercentBSignal (Bollinger %B crosses)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from nodes.mean_reversion.bollinger.percent_b_signal import PercentBSignal
from lib.core import helpers
from lib.core.enums import Ticker, TimeFrame
from lib.core.models import Candle


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


def test_long_enters_on_dip_then_exits_on_cross_above_upper_band() -> None:
    # Long mean reversion: a sharp DIP crosses %B below the lower band (enter long),
    # then a sharp RALLY crosses %B above the upper band (exit). period=20 so a single
    # outlier bar is a genuine band breach (with period=5 one bar dominates sigma and
    # %B can never cross past 0/1).
    node = PercentBSignal(
        Ticker.ES,
        TimeFrame.D,
        period=20,
        std_dev=2.0,
        lower_threshold=0.0,
        upper_threshold=1.0,
        strategy_mode="long",
        exit_policy="threshold",
        exit_bars=5,
    )
    base = 100.0
    for i in range(20):
        node.add_candle(_candle(i, base))  # flat warmup

    # Sharp dip -> %B crosses below the lower band -> enter long.
    enter = node.add_candle(_candle(20, base - 10.0))[0]
    # Sharp rally -> %B crosses above the upper band -> exit the long.
    exit_sig = node.add_candle(_candle(21, base + 10.0))[0]

    assert enter == 1.0
    assert exit_sig == 0.0


def test_invalid_thresholds_raise() -> None:
    with pytest.raises(ValueError, match="lower_threshold must be"):
        PercentBSignal(
            Ticker.ES,
            TimeFrame.D,
            lower_threshold=1.0,
            upper_threshold=0.5,
        )
