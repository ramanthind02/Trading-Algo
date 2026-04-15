"""Unit tests for RSISignalAtrSlopeEntry (ATR slope only on the RSI entry cross bar)."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List
from unittest.mock import MagicMock, patch

from nodes.mean_reversion.rsi.rsi_signal_atr_slope_entry import RSISignalAtrSlopeEntry
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


def _stack_filter_returns(values: List[float]) -> List[List[float]]:
    return [[v] for v in values]


def test_create_fresh_bias_node_resolves() -> None:
    node = helpers.create_fresh_bias_node(
        "rsi_signal_atr_slope_entry",
        Ticker.ES,
        TimeFrame.D,
        {"rsi_period": 2, "atr_period": 1, "oversold": 30.0},
    )
    assert type(node).__name__ == "RSISignalAtrSlopeEntry"


@patch("nodes.mean_reversion.rsi.rsi_signal_atr_slope_entry.update_rsi")
@patch("nodes.mean_reversion.rsi.rsi_signal_atr_slope_entry.compute_rsi_initial")
def test_late_atr_does_not_enter_after_cross_without_filter(
    mock_init: MagicMock,
    mock_update: MagicMock,
) -> None:
    mock_init.return_value = (1.0, 1.0)
    # update_rsi runs only after the init bar (first RSI bar uses compute_rsi_initial).
    rsi_seq = iter([40.0, 25.0, 24.0])

    def _upd(
        prev_close: float,
        curr_close: float,
        upsum: float,
        dnsum: float,
        rsi_period: int,
    ) -> tuple[float, float, float]:
        rsi = next(rsi_seq)
        return upsum, dnsum, rsi

    mock_update.side_effect = _upd

    node = RSISignalAtrSlopeEntry(
        Ticker.ES,
        TimeFrame.D,
        rsi_period=2,
        oversold=30.0,
        overbought=70.0,
        strategy_mode="long",
        exit_policy="threshold",
        atr_period=1,
    )
    mock_atr = MagicMock()
    mock_atr.add_candle.side_effect = _stack_filter_returns([0.0, 0.0, 0.0, 0.0, 1.0])
    node._atr_filter = mock_atr

    signals = [node.add_candle(_candle(i))[0] for i in range(5)]
    assert signals == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert 1.0 not in signals


@patch("nodes.mean_reversion.rsi.rsi_signal_atr_slope_entry.update_rsi")
@patch("nodes.mean_reversion.rsi.rsi_signal_atr_slope_entry.compute_rsi_initial")
def test_no_entry_if_oversold_exited_before_atr(
    mock_init: MagicMock,
    mock_update: MagicMock,
) -> None:
    mock_init.return_value = (1.0, 1.0)
    rsi_seq = iter([40.0, 25.0, 35.0])

    def _upd(
        prev_close: float,
        curr_close: float,
        upsum: float,
        dnsum: float,
        rsi_period: int,
    ) -> tuple[float, float, float]:
        rsi = next(rsi_seq)
        return upsum, dnsum, rsi

    mock_update.side_effect = _upd

    node = RSISignalAtrSlopeEntry(
        Ticker.ES,
        TimeFrame.D,
        rsi_period=2,
        oversold=30.0,
        overbought=70.0,
        strategy_mode="long",
        exit_policy="threshold",
        atr_period=1,
    )
    mock_atr = MagicMock()
    mock_atr.add_candle.side_effect = _stack_filter_returns([0.0, 0.0, 0.0, 0.0, 1.0])
    node._atr_filter = mock_atr

    signals = [node.add_candle(_candle(i))[0] for i in range(5)]
    assert signals[-1] == 0.0
    assert 1.0 not in signals


@patch("nodes.mean_reversion.rsi.rsi_signal_atr_slope_entry.update_rsi")
@patch("nodes.mean_reversion.rsi.rsi_signal_atr_slope_entry.compute_rsi_initial")
def test_same_bar_cross_and_atr_enters(
    mock_init: MagicMock,
    mock_update: MagicMock,
) -> None:
    mock_init.return_value = (1.0, 1.0)
    rsi_seq = iter([25.0])

    def _upd(
        prev_close: float,
        curr_close: float,
        upsum: float,
        dnsum: float,
        rsi_period: int,
    ) -> tuple[float, float, float]:
        rsi = next(rsi_seq)
        return upsum, dnsum, rsi

    mock_update.side_effect = _upd

    node = RSISignalAtrSlopeEntry(
        Ticker.ES,
        TimeFrame.D,
        rsi_period=2,
        oversold=30.0,
        overbought=70.0,
        strategy_mode="long",
        exit_policy="threshold",
        atr_period=1,
    )
    mock_atr = MagicMock()
    mock_atr.add_candle.side_effect = _stack_filter_returns([0.0, 0.0, 1.0])
    node._atr_filter = mock_atr

    signals = [node.add_candle(_candle(i))[0] for i in range(3)]
    assert signals == [0.0, 0.0, 1.0]
