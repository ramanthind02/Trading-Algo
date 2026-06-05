"""Unit tests for LrsHookSignal (Algomatic-style LRS hook mean reversion)."""

from __future__ import annotations

from datetime import datetime, timedelta

from nodes.mean_reversion.price_action.lrs_hook_signal import (
    FridayEntryPolicy,
    LrsHookExitVariant,
    LrsHookSignal,
    _ols_slope,
)
from lib.core import helpers
from lib.core.enums import Ticker, TimeFrame
from lib.core.models import Candle


def _candle(
    day: int,
    *,
    o: float,
    c: float,
    h: float | None = None,
    low: float | None = None,
) -> Candle:
    hi = h if h is not None else max(o, c) + 0.25
    lo = low if low is not None else min(o, c) - 0.25
    return Candle(
        datetime=datetime(2020, 1, 6) + timedelta(days=day),
        open=o,
        high=hi,
        low=lo,
        close=c,
        volume=1.0,
        ticker=Ticker.ES,
        tf=TimeFrame.D,
    )


def test_create_fresh_bias_node_resolves() -> None:
    node = helpers.create_fresh_bias_node(
        "lrs_hook_signal",
        Ticker.ES,
        TimeFrame.D,
        {"lrs_period": 3, "ma_period": 5},
    )
    assert type(node).__name__ == "LrsHookSignal"


def test_ols_slope_matches_manual_three_point_trend() -> None:
    assert abs(_ols_slope([100.0, 99.0, 98.0]) - (-1.0)) < 1e-9
    assert abs(_ols_slope([98.0, 99.0, 100.0]) - 1.0) < 1e-9


def test_hook_and_red_candle_enters_long_after_warmup() -> None:
    """TP path [100,99.5,99,98,102] yields hook; red bar on last candle fires long."""
    node = LrsHookSignal(Ticker.ES, TimeFrame.D)
    outs: list[float] = []
    levels = [(100.0, 100.0), (99.5, 99.5), (99.0, 99.0), (98.0, 98.0)]
    for i, (px_o, px_c) in enumerate(levels):
        outs.append(node.add_candle(_candle(i, o=px_o, c=px_c))[0])
    outs.append(
        node.add_candle(_candle(4, o=103.0, c=102.0, h=103.0, low=101.0))[0]
    )
    assert outs[:4] == [0.0, 0.0, 0.0, 0.0]
    assert outs[4] == 1.0


def test_exit_ma5_clears_position() -> None:
    node = LrsHookSignal(Ticker.ES, TimeFrame.D)
    for i, (o, c) in enumerate([(100.0, 100.0), (99.5, 99.5), (99.0, 99.0), (98.0, 98.0)]):
        node.add_candle(_candle(i, o=o, c=c))
    node.add_candle(_candle(4, o=103.0, c=102.0, h=103.0, low=101.0))
    assert node._position == 1
    out = node.add_candle(_candle(5, o=103.0, c=103.0, h=103.5, low=102.5))[0]
    assert out == 0.0
    assert node._position == 0


def test_exit_typical_price_variant() -> None:
    node = LrsHookSignal(
        Ticker.ES, TimeFrame.D, exit_variant=LrsHookExitVariant.TYPICAL_PRICE
    )
    for i, (o, c) in enumerate([(100.0, 100.0), (99.5, 99.5), (99.0, 99.0), (98.0, 98.0)]):
        node.add_candle(_candle(i, o=o, c=c))
    node.add_candle(_candle(4, o=103.0, c=102.0, h=103.0, low=101.0))
    assert node._position == 1
    out = node.add_candle(_candle(5, o=99.0, c=102.0, h=102.0, low=99.0))[0]
    assert out == 0.0


def test_friday_skip_blocks_entry() -> None:
    """Fifth bar falls on 2020-01-10 (Friday); SKIP avoids new long."""
    node = LrsHookSignal(
        Ticker.ES, TimeFrame.D, friday_entry_policy=FridayEntryPolicy.SKIP
    )
    outs: list[float] = []
    for i, (o, c) in enumerate([(100.0, 100.0), (99.5, 99.5), (99.0, 99.0), (98.0, 98.0)]):
        outs.append(node.add_candle(_candle(i, o=o, c=c))[0])
    outs.append(
        node.add_candle(_candle(4, o=103.0, c=102.0, h=103.0, low=101.0))[0]
    )
    assert outs[4] == 0.0
    assert node._position == 0
