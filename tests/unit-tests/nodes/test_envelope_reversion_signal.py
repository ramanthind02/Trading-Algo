"""Unit tests for EnvelopeReversionSignal (MA envelope + trend filter)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from nodes.mean_reversion.bands.envelope_reversion_signal import (
    EnvelopeReversionSignal,
    EnvelopeWidthMode,
)
from utils.core import helpers
from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle


def _candle(day: int, *, o: float, h: float, l: float, c: float) -> Candle:
    return Candle(
        datetime=datetime(2020, 1, 1) + timedelta(days=day),
        open=o,
        high=h,
        low=l,
        close=c,
        volume=1.0,
        ticker=Ticker.ES,
        tf=TimeFrame.D,
    )


def test_create_fresh_bias_node_resolves() -> None:
    node = helpers.create_fresh_bias_node(
        "envelope_reversion_signal",
        Ticker.ES,
        TimeFrame.D,
        {
            "trend_period": 30,
            "envelope_period": 10,
            "width_mode": "percent",
            "percent_width": 1.5,
            "strategy_mode": "long",
            "exit_policy": "threshold",
            "exit_bars": 5,
        },
    )
    assert type(node).__name__ == "EnvelopeReversionSignal"


def test_invalid_params_raise() -> None:
    with pytest.raises(ValueError, match="trend_period must be"):
        EnvelopeReversionSignal(Ticker.ES, TimeFrame.D, trend_period=1)
    with pytest.raises(ValueError, match="width_mode must be"):
        EnvelopeReversionSignal(
            Ticker.ES, TimeFrame.D, width_mode="invalid_mode"  # type: ignore[arg-type]
        )


def test_outputs_are_tri_state_after_warmup() -> None:
    node = EnvelopeReversionSignal(
        Ticker.ES,
        TimeFrame.D,
        trend_period=25,
        envelope_period=8,
        width_mode=EnvelopeWidthMode.PERCENT,
        percent_width=3.0,
        strategy_mode="long",
        exit_policy="threshold",
        exit_bars=5,
    )
    base = 5000.0
    for i in range(120):
        wobble = (i % 13) * 2.5
        out = node.add_candle(
            _candle(
                i,
                o=base + wobble,
                h=base + wobble + 4,
                l=base + wobble - 4,
                c=base + wobble * 0.2,
            )
        )[0]
        assert out in (-1.0, 0.0, 1.0)


def test_long_entry_requires_uptrend_and_cross_below_lower() -> None:
    """Long SMA lags so a pullback can sit below the short envelope but above trend SMA."""
    node = EnvelopeReversionSignal(
        Ticker.ES,
        TimeFrame.D,
        trend_period=15,
        envelope_period=4,
        width_mode="percent",
        percent_width=2.0,
        strategy_mode="long",
        exit_policy="threshold",
        exit_bars=5,
    )
    day = 0
    for _ in range(40):
        node.add_candle(_candle(day, o=100, h=100.5, l=99.5, c=100.0))
        day += 1
    for _ in range(6):
        node.add_candle(_candle(day, o=200, h=200.5, l=199.5, c=200.0))
        day += 1
    assert node.add_candle(_candle(day, o=190, h=200, l=189, c=190.0))[0] == 1.0
    day += 1
    assert node.add_candle(_candle(day, o=205, h=206, l=204, c=205.0))[0] == 0.0
