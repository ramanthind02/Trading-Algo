"""Unit tests for AdxFilterNode (Wilder ADX vs threshold gate)."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Sequence

from nodes.regime.adx.adx_filter import AdxFilterCompare, AdxFilterNode
from utils.core import helpers
from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle


def _candle(day: int, *, o: float = 100.0, h: float = 101.0, low: float = 99.0, c: float = 100.0) -> Candle:
    return Candle(
        datetime=datetime(2020, 1, 1) + timedelta(days=day),
        open=o,
        high=h,
        low=low,
        close=c,
        volume=1.0,
        ticker=Ticker.ES,
        tf=TimeFrame.D,
    )


def _reference_last_adx(high: Sequence[float], low: Sequence[float], close: Sequence[float], period: int) -> float | None:
    """Batch Wilder ADX matching AdxFilterNode (last bar ADX value)."""
    h = [float(x) for x in high]
    l_ = [float(x) for x in low]
    c = [float(x) for x in close]
    n = len(c)
    if n < 2 or period < 2:
        return None
    prev_h, prev_l, prev_c = h[0], l_[0], c[0]
    tr_seeded = False
    sum_tr = sum_pdm = sum_mdm = 0.0
    acc = 0
    s_tr = s_pdm = s_mdm = 0.0
    adx_smooth: float | None = None
    sum_dx = 0.0
    dx_bars = 0
    last_adx: float | None = None

    for i in range(1, n):
        hi, lo, cl = h[i], l_[i], c[i]
        tr = max(hi - lo, abs(hi - prev_c), abs(lo - prev_c))
        up = hi - prev_h
        down = prev_l - lo
        pdm = up if up > down and up > 0 else 0.0
        mdm = down if down > up and down > 0 else 0.0
        prev_h, prev_l, prev_c = hi, lo, cl

        if not tr_seeded:
            sum_tr += tr
            sum_pdm += pdm
            sum_mdm += mdm
            acc += 1
            if acc < period:
                continue
            s_tr, s_pdm, s_mdm = sum_tr, sum_pdm, sum_mdm
            tr_seeded = True
        else:
            s_tr = s_tr - s_tr / period + tr
            s_pdm = s_pdm - s_pdm / period + pdm
            s_mdm = s_mdm - s_mdm / period + mdm

        dip = 100.0 * s_pdm / s_tr if s_tr > 0 else 0.0
        dim = 100.0 * s_mdm / s_tr if s_tr > 0 else 0.0
        di_sum = dip + dim
        dx = 100.0 * abs(dip - dim) / di_sum if di_sum > 0 else 0.0

        if adx_smooth is None:
            sum_dx += dx
            dx_bars += 1
            if dx_bars < period:
                continue
            adx_smooth = sum_dx / period
        else:
            adx_smooth = (adx_smooth * (period - 1) + dx) / period
        last_adx = adx_smooth
    return last_adx


def _series_uptrend(n: int) -> tuple[list[float], list[float], list[float]]:
    high, low, close = [], [], []
    base = 100.0
    for i in range(n):
        c = base + float(i) * 0.5
        high.append(c + 0.3)
        low.append(c - 0.3)
        close.append(c)
    return high, low, close


def test_adx_filter_matches_reference_batch() -> None:
    period = 14
    n = 120
    high, low, close = _series_uptrend(n)
    candles = [
        _candle(i, h=high[i], low=low[i], c=close[i], o=close[i])
        for i in range(n)
    ]
    node = AdxFilterNode(Ticker.ES, TimeFrame.D, length=period, threshold=25.0, compare=AdxFilterCompare.BELOW)
    last_gate: List[float] = []
    for cd in candles:
        last_gate = node.add_candle(cd)

    ref_adx = _reference_last_adx(high, low, close, period)
    assert ref_adx is not None
    expected = 1.0 if ref_adx < 25.0 else 0.0
    assert last_gate == [expected]


def test_create_fresh_bias_node_string_compare() -> None:
    node = helpers.create_fresh_bias_node(
        "adx_filter",
        Ticker.ES,
        TimeFrame.D,
        {"length": 5, "threshold": 50.0, "compare": ">"},
    )
    assert isinstance(node, AdxFilterNode)
    assert node.compare is AdxFilterCompare.ABOVE


def test_warmup_emits_zero_until_ready() -> None:
    period = 3
    node = AdxFilterNode(Ticker.ES, TimeFrame.D, length=period, threshold=101.0, compare=AdxFilterCompare.BELOW)
    high, low, close = _series_uptrend(20)
    outs: list[float] = []
    for i in range(len(close)):
        outs.extend(node.add_candle(_candle(i, h=high[i], low=low[i], c=close[i], o=close[i])))
    assert all(v == 0.0 for v in outs[: 2 * period - 1])
    assert any(v != 0.0 for v in outs[2 * period - 1 :])
