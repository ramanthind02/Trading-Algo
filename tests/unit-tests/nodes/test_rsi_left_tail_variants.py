from datetime import datetime, timedelta
from uuid import uuid4

import numpy as np
import pytest

from nodes.lagged_rsi import LaggedRSI
from nodes.rsi_left_tail_pressure import RSILeftTailPressure
from nodes.rsi_left_tail_streak import RSILeftTailStreak
from nodes.rsi_rebound_velocity import RSIReboundVelocity
from lib.core.enums import Ticker, TimeFrame
from lib.compute.fast_nodes import compute_rsi_initial_fast, update_rsi_fast
from lib.core.helpers import _resolve_bias_node_import_path, create_bias_node
from lib.core.models import Candle


def _build_candles(count: int, start_close: float = 100.0) -> list[Candle]:
    base_time = datetime(2024, 1, 1)
    return [
        Candle(
            id=uuid4(),
            datetime=base_time + timedelta(days=i),
            open=start_close + i,
            high=start_close + i + 1.0,
            low=start_close + i - 1.0,
            close=start_close + i,
            volume=1000.0,
            ticker=Ticker.ES,
            tf=TimeFrame.D,
        )
        for i in range(count)
    ]


def _build_candles_from_closes(closes: list[float]) -> list[Candle]:
    base_time = datetime(2024, 1, 1)
    return [
        Candle(
            id=uuid4(),
            datetime=base_time + timedelta(days=i),
            open=close,
            high=close + 1.0,
            low=close - 1.0,
            close=close,
            volume=1000.0,
            ticker=Ticker.ES,
            tf=TimeFrame.D,
        )
        for i, close in enumerate(closes)
    ]


def _expected_rsi_stream(closes: list[float], period: int) -> list[float | None]:
    rsi_stream: list[float | None] = [None] * len(closes)
    if len(closes) < period:
        return rsi_stream

    init_prices = np.asarray(closes[:period], dtype=np.float64)
    upsum, dnsum = compute_rsi_initial_fast(init_prices, period)
    rsi_stream[period - 1] = 100.0 * upsum / (upsum + dnsum)

    for idx in range(period, len(closes)):
        upsum, dnsum, rsi = update_rsi_fast(closes[idx - 1], closes[idx], upsum, dnsum, period)
        rsi_stream[idx] = rsi

    return rsi_stream


def _expected_lagged_outputs(closes: list[float], period: int, lag: int) -> list[float]:
    outputs: list[float] = []
    rsi_stream = _expected_rsi_stream(closes, period)
    rsi_history: list[float] = []
    front_bad = period + lag

    for idx, rsi in enumerate(rsi_stream):
        if rsi is None:
            outputs.append(50.0)
            continue

        rsi_history.append(rsi)
        if idx + 1 < front_bad or len(rsi_history) <= lag:
            outputs.append(50.0)
        else:
            outputs.append(rsi_history[-(lag + 1)])

    return outputs


def _expected_pressure_outputs(closes: list[float], period: int, left_tail_level: float) -> list[float]:
    rsi_stream = _expected_rsi_stream(closes, period)
    return [
        0.0
        if rsi is None
        else min(1.0, max(0.0, (left_tail_level - rsi) / left_tail_level))
        for rsi in rsi_stream
    ]


def _expected_streak_outputs(closes: list[float], period: int, left_tail_level: float, max_streak: int) -> list[float]:
    outputs: list[float] = []
    streak_count = 0
    for rsi in _expected_rsi_stream(closes, period):
        if rsi is None:
            outputs.append(0.0)
            continue

        streak_count = streak_count + 1 if rsi < left_tail_level else 0
        outputs.append(min(streak_count, max_streak) / max_streak)

    return outputs


def _expected_rebound_outputs(closes: list[float], period: int, left_tail_level: float) -> list[float]:
    outputs: list[float] = []
    prev_rsi: float | None = None

    for rsi in _expected_rsi_stream(closes, period):
        if rsi is None:
            outputs.append(0.0)
            continue

        if prev_rsi is None:
            outputs.append(0.0)
        else:
            delta_rsi = rsi - prev_rsi
            if prev_rsi < left_tail_level and delta_rsi > 0.0:
                outputs.append(min(1.0, delta_rsi / 100.0))
            else:
                outputs.append(0.0)
        prev_rsi = rsi

    return outputs


def test_taxonomy_entries_resolve_to_canonical_modules() -> None:
    assert _resolve_bias_node_import_path("lagged_rsi") == "nodes.mean_reversion.rsi.lagged_rsi"
    assert _resolve_bias_node_import_path("rsi_left_tail_pressure") == "nodes.mean_reversion.rsi.rsi_left_tail_pressure"
    assert _resolve_bias_node_import_path("rsi_left_tail_streak") == "nodes.mean_reversion.rsi.rsi_left_tail_streak"
    assert _resolve_bias_node_import_path("rsi_rebound_velocity") == "nodes.mean_reversion.rsi.rsi_rebound_velocity"


def test_create_bias_node_instantiates_left_tail_rsi_variants() -> None:
    lagged = create_bias_node("lagged_rsi", Ticker.ES, TimeFrame.D, {"rsiPeriod": 3, "lagPeriod": 2})
    pressure = create_bias_node("rsi_left_tail_pressure", Ticker.ES, TimeFrame.D, {"rsiPeriod": 3, "leftTailLevel": 30.0})
    streak = create_bias_node("rsi_left_tail_streak", Ticker.ES, TimeFrame.D, {"rsiPeriod": 3, "leftTailLevel": 30.0, "maxStreak": 5})
    rebound = create_bias_node("rsi_rebound_velocity", Ticker.ES, TimeFrame.D, {"rsiPeriod": 3, "leftTailLevel": 30.0})

    assert isinstance(lagged, LaggedRSI)
    assert isinstance(pressure, RSILeftTailPressure)
    assert isinstance(streak, RSILeftTailStreak)
    assert isinstance(rebound, RSIReboundVelocity)


def test_lagged_rsi_matches_reference_path_without_double_counting_first_ready_candle() -> None:
    closes = [100.0, 103.0, 104.0, 101.0, 102.0, 100.0, 99.0]
    candles = _build_candles_from_closes(closes)
    node = LaggedRSI(Ticker.ES, TimeFrame.D, rsiPeriod=3, lagPeriod=2)

    outputs = [node.add_candle(candle)[0] for candle in candles]

    assert outputs == pytest.approx(_expected_lagged_outputs(closes, period=3, lag=2))


def test_left_tail_pressure_matches_reference_path_without_mocking_rsi_math() -> None:
    closes = [100.0, 96.0, 95.0, 98.0, 97.0, 99.0]
    candles = _build_candles_from_closes(closes)
    node = RSILeftTailPressure(Ticker.ES, TimeFrame.D, rsiPeriod=3, leftTailLevel=40.0)

    outputs = [node.add_candle(candle)[0] for candle in candles]

    assert outputs == pytest.approx(_expected_pressure_outputs(closes, period=3, left_tail_level=40.0))


def test_left_tail_streak_matches_reference_path_without_mocking_rsi_math() -> None:
    closes = [100.0, 97.0, 95.0, 93.0, 95.0, 96.0, 94.0]
    candles = _build_candles_from_closes(closes)
    node = RSILeftTailStreak(Ticker.ES, TimeFrame.D, rsiPeriod=3, leftTailLevel=35.0, maxStreak=3)

    outputs = [node.add_candle(candle)[0] for candle in candles]

    assert outputs == pytest.approx(_expected_streak_outputs(closes, period=3, left_tail_level=35.0, max_streak=3))


def test_rebound_velocity_matches_reference_path_without_mocking_rsi_math() -> None:
    closes = [100.0, 96.0, 94.0, 95.0, 97.0, 96.0, 98.0]
    candles = _build_candles_from_closes(closes)
    node = RSIReboundVelocity(Ticker.ES, TimeFrame.D, rsiPeriod=3, leftTailLevel=40.0)

    outputs = [node.add_candle(candle)[0] for candle in candles]

    assert outputs == pytest.approx(_expected_rebound_outputs(closes, period=3, left_tail_level=40.0))


@pytest.mark.parametrize("left_tail_level", [-1.0, 0.0, 100.0, 120.0])
def test_left_tail_level_validation_rejects_out_of_range_values(left_tail_level: float) -> None:
    with pytest.raises(ValueError, match="leftTailLevel must be between 0 and 100"):
        RSILeftTailPressure(Ticker.ES, TimeFrame.D, leftTailLevel=left_tail_level)

    with pytest.raises(ValueError, match="leftTailLevel must be between 0 and 100"):
        RSILeftTailStreak(Ticker.ES, TimeFrame.D, leftTailLevel=left_tail_level)

    with pytest.raises(ValueError, match="leftTailLevel must be between 0 and 100"):
        RSIReboundVelocity(Ticker.ES, TimeFrame.D, leftTailLevel=left_tail_level)


def test_new_nodes_expose_expected_warmup_and_param_metadata() -> None:
    lagged = LaggedRSI(Ticker.ES, TimeFrame.D)
    pressure = RSILeftTailPressure(Ticker.ES, TimeFrame.D)
    streak = RSILeftTailStreak(Ticker.ES, TimeFrame.D)
    rebound = RSIReboundVelocity(Ticker.ES, TimeFrame.D)

    assert lagged.params.keys() == {"rsiPeriod", "lagPeriod"}
    assert pressure.params.keys() == {"rsiPeriod", "leftTailLevel"}
    assert streak.params.keys() == {"rsiPeriod", "leftTailLevel", "maxStreak"}
    assert rebound.params.keys() == {"rsiPeriod", "leftTailLevel"}

    assert lagged.front_bad == lagged.rsi_period + lagged.lag_period
    assert pressure.front_bad == pressure.rsi_period
    assert streak.front_bad == streak.rsi_period
    assert rebound.front_bad == rebound.rsi_period
