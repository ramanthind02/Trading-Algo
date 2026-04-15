"""Unit tests for IBSLowerBand bias node."""

from __future__ import annotations

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from nodes.ibs_lower_band import IBSLowerBand
from utils.core.enums import Ticker, TimeFrame
from utils.core.helpers import create_bias_node
from utils.core.models import Candle


def _candle(day: int, o: float, h: float, l: float, c: float) -> Candle:
    base = datetime(2020, 1, 1, tzinfo=None)
    return Candle(
        id=uuid4(),
        datetime=base + timedelta(days=day),
        open=o,
        high=h,
        low=l,
        close=c,
        volume=1,
        ticker=Ticker.ES,
        tf=TimeFrame.D,
    )


def test_warmup_outputs_zero() -> None:
    node = IBSLowerBand(Ticker.ES, TimeFrame.D)
    assert node.front_bad == 25
    for i in range(24):
        r = node.add_candle(_candle(i, 99.0, 100.0, 99.0, 99.5))
        assert r[0] == 0.0
    r25 = node.add_candle(_candle(24, 99.0, 100.0, 99.0, 99.5))
    assert r25[0] in (0.0, 1.0)


def test_entry_on_weak_ibs_below_band() -> None:
    node = IBSLowerBand(Ticker.ES, TimeFrame.D)
    for i in range(25):
        node.add_candle(_candle(i, 99.5, 100.0, 99.0, 99.5))
    sig = node.add_candle(_candle(25, 98.0, 100.0, 97.0, 97.2))
    assert sig[0] == 1.0


def test_exit_when_close_above_prior_high() -> None:
    node = IBSLowerBand(Ticker.ES, TimeFrame.D)
    for i in range(25):
        node.add_candle(_candle(i, 99.5, 100.0, 99.0, 99.5))
    node.add_candle(_candle(25, 98.0, 100.0, 97.0, 97.2))
    sig_exit = node.add_candle(_candle(26, 100.5, 102.0, 100.0, 101.0))
    assert sig_exit[0] == 0.0


def test_create_bias_node_registry() -> None:
    n = create_bias_node(
        "ibs_lower_band",
        Ticker.ES,
        TimeFrame.D,
        {
            "hl_mean_lookback": 25,
            "band_high_lookback": 10,
            "band_width_mult": 2.5,
            "ibs_entry_max": 0.3,
        },
    )
    assert isinstance(n, IBSLowerBand)


def test_max_lookback_metadata() -> None:
    node = IBSLowerBand(Ticker.ES, TimeFrame.D, hl_mean_lookback=30, band_high_lookback=12)
    assert node.max_lookback() == 30


def test_invalid_ibs_entry_max_raises() -> None:
    with pytest.raises(ValueError):
        IBSLowerBand(Ticker.ES, TimeFrame.D, ibs_entry_max=1.0)
