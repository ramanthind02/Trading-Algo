"""Unit tests for CaseyPercentCSignal (Casey Bands PercentC + discrete rules)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from nodes.mean_reversion.bands.casey_percent_c_signal import CaseyPercentCSignal
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
        "casey_percent_c_signal",
        Ticker.ES,
        TimeFrame.D,
        {
            "ema_lookback": 5,
            "atr_lookback": 5,
            "multiplier": 1.25,
            "smoothing": 1,
            "oversold": 30.0,
            "overbought": 70.0,
            "strategy_mode": "long",
            "exit_policy": "threshold",
            "exit_bars": 5,
        },
    )
    assert type(node).__name__ == "CaseyPercentCSignal"


def test_invalid_params_raise() -> None:
    with pytest.raises(ValueError, match="ema_lookback must be"):
        CaseyPercentCSignal(Ticker.ES, TimeFrame.D, ema_lookback=1)
    with pytest.raises(ValueError, match="smoothing must be"):
        CaseyPercentCSignal(Ticker.ES, TimeFrame.D, smoothing=0)


def test_outputs_are_tri_state_after_warmup() -> None:
    node = CaseyPercentCSignal(
        Ticker.ES,
        TimeFrame.D,
        ema_lookback=5,
        atr_lookback=5,
        multiplier=1.25,
        smoothing=1,
        oversold=30.0,
        overbought=70.0,
        strategy_mode="long",
        exit_policy="threshold",
        exit_bars=5,
    )
    base = 100.0
    for i in range(300):
        wobble = (i % 11) * 0.3
        out = node.add_candle(
            _candle(i, o=base, h=base + 1 + wobble, l=base - 1, c=base + wobble * 0.1)
        )[0]
        assert out in (-1.0, 0.0, 1.0)
