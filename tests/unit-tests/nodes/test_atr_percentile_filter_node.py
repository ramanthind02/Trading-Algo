"""Unit tests for AtrPercentileFilterNode (ATR / ATR% rank low-vol gate)."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np

from nodes.volatility.atr.atr_percentile_filter import (
    AtrPercentileFilterNode,
    _strict_less_rank_fraction,
)
from utils.core import helpers
from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle


def _candle(day: int, *, h: float = 101.0, low: float = 99.0, c: float = 100.0) -> Candle:
    return Candle(
        datetime=datetime(2020, 1, 1) + timedelta(days=day),
        open=c,
        high=h,
        low=low,
        close=c,
        volume=1.0,
        ticker=Ticker.ES,
        tf=TimeFrame.D,
    )


def test_strict_less_rank_fraction() -> None:
    arr = np.asarray([1.0, 2.0, 3.0, 10.0], dtype=np.float64)
    assert _strict_less_rank_fraction(arr, 1.0) == 0.0
    assert _strict_less_rank_fraction(arr, 10.0) == 1.0


def test_create_fresh_bias_node() -> None:
    node = helpers.create_fresh_bias_node(
        "atr_percentile_filter",
        Ticker.ES,
        TimeFrame.D,
        {"atr_period": 2, "lookback": 3, "max_rank_fraction": 0.5, "rank_metric": "atr"},
    )
    assert isinstance(node, AtrPercentileFilterNode)


@patch("nodes.volatility.atr.atr_percentile_filter.compute_atr_fast")
def test_low_metric_opens_gate(mock_fast: MagicMock) -> None:
    """Current ATR lowest in window → rank fraction 0 → gate open."""
    rows = [
        (10.0, 1.0, 0, 1),
        (10.0, 1.0, 0, 2),
        (10.0, 1.0, 0, 2),
        (3.0, 0.3, 0, 2),
    ]
    mock_fast.side_effect = rows

    node = AtrPercentileFilterNode(
        Ticker.ES,
        TimeFrame.D,
        atr_period=2,
        lookback=3,
        max_rank_fraction=0.3,
        rank_metric="atr",
    )
    outs = [node.add_candle(_candle(i))[0] for i in range(4)]
    assert outs[:3] == [0.0, 0.0, 0.0]
    assert outs[3] == 1.0


@patch("nodes.volatility.atr.atr_percentile_filter.compute_atr_fast")
def test_high_metric_closes_gate(mock_fast: MagicMock) -> None:
    rows = [
        (3.0, 0.3, 0, 1),
        (3.0, 0.3, 0, 2),
        (3.0, 0.3, 0, 2),
        (10.0, 1.0, 0, 2),
    ]
    mock_fast.side_effect = rows

    node = AtrPercentileFilterNode(
        Ticker.ES,
        TimeFrame.D,
        atr_period=2,
        lookback=3,
        max_rank_fraction=0.3,
        rank_metric="atr",
    )
    outs = [node.add_candle(_candle(i))[0] for i in range(4)]
    assert outs[3] == 0.0
