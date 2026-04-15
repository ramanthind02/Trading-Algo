"""Unit tests for CumulativeRSISignal (smoothed RSI + discrete RSI signal rules)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from nodes.mean_reversion.rsi.cumulative_rsi_signal import CumulativeRSISignal
from utils.core import helpers
from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle


def _candle(day: int, *, c: float = 100.0) -> Candle:
    return Candle(
        datetime=datetime(2020, 1, 1) + timedelta(days=day),
        open=c,
        high=c + 0.5,
        low=c - 0.5,
        close=c,
        volume=1.0,
        ticker=Ticker.ES,
        tf=TimeFrame.D,
    )


def test_create_fresh_bias_node_resolves() -> None:
    node = helpers.create_fresh_bias_node(
        "cumulative_rsi_signal",
        Ticker.ES,
        TimeFrame.D,
        {
            "lookback": 5,
            "avg_period": 5,
            "oversold": 30.0,
            "overbought": 70.0,
            "strategy_mode": "long",
            "exit_policy": "threshold",
            "exit_bars": 5,
        },
    )
    assert type(node).__name__ == "CumulativeRSISignal"


def test_invalid_lookback_raises() -> None:
    with pytest.raises(ValueError, match="lookback must be >= 2"):
        CumulativeRSISignal(Ticker.ES, TimeFrame.D, lookback=1, avg_period=3)


def test_outputs_are_discrete_tri_state() -> None:
    node = CumulativeRSISignal(
        Ticker.ES,
        TimeFrame.D,
        lookback=5,
        avg_period=3,
        oversold=30.0,
        overbought=70.0,
        strategy_mode="long",
        exit_policy="threshold",
        exit_bars=5,
    )
    closes = [100.0 + i * 0.1 for i in range(200)]
    for i, cl in enumerate(closes):
        out = node.add_candle(_candle(i, c=cl))[0]
        assert out in (-1.0, 0.0, 1.0)
