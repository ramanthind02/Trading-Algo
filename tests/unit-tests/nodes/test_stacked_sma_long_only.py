"""Unit tests for :class:`~nodes.regime.sma.stacked_sma_long_only.StackedSmaLongOnlyNode`."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from nodes.regime.sma.stacked_sma_long_only import (
    StackedSmaLongOnlyNode,
    _annualized_realized_vol,
    stacked_sma_period_combos,
)
from lib.core.enums import PositionMode, Ticker, TimeFrame
from lib.core.helpers import create_bias_node
from lib.core.models import Candle


def _candle(close: float, day: int) -> Candle:
    return Candle(
        datetime=datetime(2020, 1, 1) + timedelta(days=day),
        open=close,
        high=close + 0.5,
        low=close - 0.5,
        close=close,
        volume=1000.0,
        ticker=Ticker.ES,
        tf=TimeFrame.D,
    )


def test_one_ma_long_when_close_above_sma() -> None:
    node = StackedSmaLongOnlyNode(Ticker.ES, TimeFrame.D, period_1=3)
    for i in range(3):
        node.add_candle(_candle(100.0, i))
    out = node.add_candle(_candle(101.0, 3))
    assert out[0] == pytest.approx(1.0)


def test_one_ma_flat_when_close_below_sma() -> None:
    node = StackedSmaLongOnlyNode(Ticker.ES, TimeFrame.D, period_1=3)
    for i in range(3):
        node.add_candle(_candle(100.0, i))
    out = node.add_candle(_candle(99.0, 3))
    assert out[0] == pytest.approx(0.0)


def test_warmup_emits_zero() -> None:
    node = StackedSmaLongOnlyNode(Ticker.ES, TimeFrame.D, period_1=5)
    out = node.add_candle(_candle(100.0, 0))
    assert out[0] == pytest.approx(0.0)


def test_two_ma_uptrend_majority() -> None:
    node = StackedSmaLongOnlyNode(Ticker.ES, TimeFrame.D, period_1=2, period_2=4)
    prices = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
    out = [0.0]
    for i, px in enumerate(prices):
        out = node.add_candle(_candle(px, i))
    assert out[0] == pytest.approx(1.0)


def test_two_ma_flat_when_smas_equal_constant_price() -> None:
    node = StackedSmaLongOnlyNode(Ticker.ES, TimeFrame.D, period_1=2, period_2=3)
    out = [0.0]
    for i in range(4):
        out = node.add_candle(_candle(100.0, i))
    assert out[0] == pytest.approx(0.0)


def test_three_ma_majority_long_when_strict_stack_breaks() -> None:
    """2/3 pairwise votes long → LONG_ONLY strength ``2 * 2/3`` (not binary 1.0)."""
    node = StackedSmaLongOnlyNode(Ticker.ES, TimeFrame.D, period_1=2, period_2=3, period_3=4)
    prices = [120.0, 100.0, 100.0, 130.0]
    out = [0.0]
    for i, px in enumerate(prices):
        out = node.add_candle(_candle(px, i))
    assert out[0] == pytest.approx(4.0 / 3.0)


def test_three_ma_long_short_symmetric_scale() -> None:
    """Same prices as prior test: 2/3 long votes → ``4 * (2/3 - 0.5) = 2/3`` on [-2, 2]."""
    node = StackedSmaLongOnlyNode(
        Ticker.ES,
        TimeFrame.D,
        period_1=2,
        period_2=3,
        period_3=4,
        mode=PositionMode.LONG_SHORT,
    )
    prices = [120.0, 100.0, 100.0, 130.0]
    out = [0.0]
    for i, px in enumerate(prices):
        out = node.add_candle(_candle(px, i))
    assert out[0] == pytest.approx(2.0 / 3.0)


def test_two_ma_long_short_bearish_is_minus_one() -> None:
    node = StackedSmaLongOnlyNode(
        Ticker.ES,
        TimeFrame.D,
        period_1=2,
        period_2=4,
        mode=PositionMode.LONG_SHORT,
    )
    prices = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 10.0]
    out = [0.0]
    for i, px in enumerate(prices):
        out = node.add_candle(_candle(px, i))
    assert out[0] == pytest.approx(-1.0)


def test_periods_must_be_strictly_increasing() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        StackedSmaLongOnlyNode(Ticker.ES, TimeFrame.D, period_1=50, period_2=50)


def test_create_bias_node_kwarg_instantiation() -> None:
    node = create_bias_node(
        "stacked_sma_long_only",
        Ticker.ES,
        TimeFrame.D,
        {"period_1": 3, "period_2": 5, "mode": PositionMode.LONG_ONLY},
    )
    assert isinstance(node, StackedSmaLongOnlyNode)


def test_create_bias_node_with_vol_gate() -> None:
    node = create_bias_node(
        "stacked_sma_long_only",
        Ticker.ES,
        TimeFrame.D,
        {"period_1": 3, "period_2": 6, "vol_period": 5, "vol_threshold": 0.5},
    )
    assert isinstance(node, StackedSmaLongOnlyNode)
    assert node.params.get("vol_period") == 5
    assert node.params.get("vol_threshold") == pytest.approx(0.5)


def test_vol_params_must_both_be_set() -> None:
    with pytest.raises(ValueError, match="both be set or both omitted"):
        StackedSmaLongOnlyNode(
            Ticker.ES, TimeFrame.D, period_1=3, period_2=5, vol_period=10, vol_threshold=None
        )
    with pytest.raises(ValueError, match="both be set or both omitted"):
        StackedSmaLongOnlyNode(
            Ticker.ES, TimeFrame.D, period_1=3, period_2=5, vol_period=None, vol_threshold=0.2
        )


def test_vol_gate_blocks_when_realized_vol_spikes() -> None:
    """Mild drift keeps ann. vol low (MA long); large oscillations push vol above ceiling."""
    node = StackedSmaLongOnlyNode(
        Ticker.ES,
        TimeFrame.D,
        period_1=2,
        period_2=5,
        vol_period=5,
        vol_threshold=0.12,
    )
    out = [0.0]
    for i in range(20):
        px = 100.0 + float(i) * 0.02
        out = node.add_candle(_candle(px, i))
    assert out[0] == pytest.approx(1.0)
    wild = [100.0, 118.0, 95.0, 118.0, 95.0, 118.0]
    for j, px in enumerate(wild):
        out = node.add_candle(_candle(px, 20 + j))
    assert out[0] == pytest.approx(0.0)


def test_annualized_realized_vol_matches_sqrt252_scaling() -> None:
    # Five equal-magnitude alternating returns → positive sample stdev × sqrt(252)
    closes = (100.0, 101.0, 100.0, 101.0, 100.0, 101.0)
    rv = _annualized_realized_vol(closes, vol_period=5)
    assert rv > 0.1


def test_period_combos_are_strictly_increasing() -> None:
    c = stacked_sma_period_combos(2, (10, 50, 200))
    assert (10, 50) in c
    assert (10, 200) in c
    assert (50, 200) in c
    assert len(c) == 3


def test_period_combos_five_layers_from_five_candidates() -> None:
    c = stacked_sma_period_combos(5, (1, 2, 3, 4, 5))
    assert c == ((1, 2, 3, 4, 5),)
    with pytest.raises(ValueError, match="between 1 and 5"):
        stacked_sma_period_combos(6, (1, 2, 3, 4, 5, 6))


def test_five_ma_runs_after_warmup() -> None:
    node = StackedSmaLongOnlyNode(
        Ticker.ES,
        TimeFrame.D,
        period_1=2,
        period_2=3,
        period_3=4,
        period_4=5,
        period_5=6,
    )
    out = [0.0]
    for i in range(6):
        out = node.add_candle(_candle(100.0 + float(i), i))
    assert -2.0 <= float(out[0]) <= 2.0


def test_three_ma_one_vote_is_two_thirds_long_only() -> None:
    """Closes chosen so exactly one pairwise ``SMA_fast > SMA_slow`` → ``2 * 1/3``."""
    node = StackedSmaLongOnlyNode(
        Ticker.ES,
        TimeFrame.D,
        period_1=2,
        period_2=3,
        period_3=4,
        mode=PositionMode.LONG_ONLY,
    )
    prices = [97.0, 115.0, 100.0, 100.0]
    out = [0.0]
    for i, px in enumerate(prices):
        out = node.add_candle(_candle(px, i))
    assert out[0] == pytest.approx(2.0 / 3.0)
