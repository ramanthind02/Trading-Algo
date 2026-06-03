"""Unit tests for VolatilePivotReversal (Bollinger pivot ratio entry)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from nodes.mean_reversion.bollinger.volatile_pivot_reversal import VolatilePivotReversal
from utils.core import helpers
from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle


def _candle(day: int, close: float, *, spread: float = 1.0) -> Candle:
    return Candle(
        datetime=datetime(2020, 1, 1) + timedelta(days=day),
        open=close,
        high=close + spread,
        low=close - spread,
        close=close,
        volume=1.0,
        ticker=Ticker.ES,
        tf=TimeFrame.D,
    )


def test_create_fresh_bias_node_resolves() -> None:
    node = helpers.create_fresh_bias_node(
        "volatile_pivot_reversal",
        Ticker.ES,
        TimeFrame.D,
        {
            "pivot_bars_before": 2,
            "pivot_bars_after": 2,
            "min_bars_since_bottom": 3,
            "max_bars_since_bottom": 15,
            "bb_period": 10,
            "bb_deviation_narrow": 2.0,
            "bb_deviation_wide": 20.0,
            "min_percent_b": 0.4,
            "max_percent_b": 0.55,
            "bbw_threshold": 0.5,
            "bbr_wide_threshold": 0.85,
            "exit_bars": 5,
        },
    )
    assert type(node).__name__ == "VolatilePivotReversal"


def test_time_exit_flattens_after_hold() -> None:
    node = VolatilePivotReversal(
        Ticker.ES,
        TimeFrame.D,
        pivot_bars_before=2,
        pivot_bars_after=2,
        min_bars_since_bottom=3,
        max_bars_since_bottom=10,
        bb_period=8,
        bb_deviation_narrow=2.0,
        bb_deviation_wide=20.0,
        min_percent_b=0.0,
        max_percent_b=1.0,
        bbw_threshold=0.0,
        bbr_wide_threshold=1.0,
        exit_bars=3,
    )
    for i in range(40):
        node.add_candle(_candle(i, 100.0 - i * 0.1))

    # Block new entries so we only observe the time exit on a forced long.
    node.min_percent_b = 2.0
    node.max_percent_b = 3.0
    node._position = 1
    node._bars_in_position = 1
    outputs = [node.add_candle(_candle(i, 100.0))[0] for i in range(40, 46)]

    assert outputs[0] == 1.0
    assert outputs[-1] == 0.0


def test_invalid_percent_b_band_raises() -> None:
    with pytest.raises(ValueError, match="min_percent_b must be"):
        VolatilePivotReversal(
            Ticker.ES,
            TimeFrame.D,
            min_percent_b=0.6,
            max_percent_b=0.4,
        )
