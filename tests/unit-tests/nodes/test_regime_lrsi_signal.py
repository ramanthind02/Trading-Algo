"""Unit tests for RegimeLrsiSignal (CL daily regime + Laguerre RSI)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from nodes.mean_reversion.regime.regime_lrsi_signal import (
    RegimeLrsiSignal,
    _laguerre_rsi_gamma_zero,
    _percentile_rank,
)
from utils.core import helpers
from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle


def _candle(day: int, close: float, *, spread: float = 1.0) -> Candle:
    return Candle(
        datetime=datetime(2010, 1, 4) + timedelta(days=day),
        open=close,
        high=close + spread,
        low=close - spread,
        close=close,
        volume=1.0,
        ticker=Ticker.CL,
        tf=TimeFrame.D,
    )


def test_create_fresh_bias_node_resolves() -> None:
    node = helpers.create_fresh_bias_node(
        "regime_lrsi_signal",
        Ticker.CL,
        TimeFrame.D,
        {"regime_ma_period": 58, "hl_sum_period": 76, "hl_range_period": 64},
    )
    assert type(node).__name__ == "RegimeLrsiSignal"


def test_laguerre_rsi_gamma_zero_matches_three_bar_ratio() -> None:
    closes = (100.0, 101.0, 100.0, 99.0)
    lrsi = _laguerre_rsi_gamma_zero(closes)
    assert lrsi == pytest.approx(2.0 / 3.0)


def test_percentile_rank_counts_strictly_lower_history() -> None:
    rank = _percentile_rank(5.0, (1.0, 3.0, 5.0, 7.0))
    assert rank == pytest.approx(50.0)


def test_warmup_emits_zero_until_front_bad() -> None:
    node = RegimeLrsiSignal(
        Ticker.CL,
        TimeFrame.D,
        regime_ma_period=10,
        hl_sum_period=10,
        hl_range_period=10,
        mr_avg_period=5,
        mr_prank_period=5,
    )
    for day in range(node.front_bad - 1):
        assert node.add_candle(_candle(day, 100.0 + day * 0.1))[0] == 0.0


def test_output_is_discrete_signed_signal() -> None:
    node = RegimeLrsiSignal(
        Ticker.CL,
        TimeFrame.D,
        regime_ma_period=10,
        hl_sum_period=10,
        hl_range_period=10,
        mr_avg_period=5,
        mr_prank_period=5,
        exit_bars=1,
        stop_loss_ticks=0,
    )
    signals = [node.add_candle(_candle(day, 80.0 + (day % 7) * 0.5))[0] for day in range(500)]
    assert all(signal in {-1.0, 0.0, 1.0} for signal in signals[node.front_bad :])


def test_invalid_thresholds_raise() -> None:
    with pytest.raises(ValueError, match="long_threshold must be"):
        RegimeLrsiSignal(
            Ticker.CL,
            TimeFrame.D,
            long_threshold=0.2,
            short_threshold=0.8,
        )
