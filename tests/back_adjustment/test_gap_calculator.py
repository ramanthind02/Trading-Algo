from __future__ import annotations

import pytest
from datetime import datetime

from data_platform.providers.norgate.backadjust.gap_calculator import (
    AdjustmentFactor,
    calculate_adjustments,
)
from data_platform.providers.norgate.backadjust.roll_detector import RollEvent


def _make_events(*gaps: float) -> list[RollEvent]:
    events: list[RollEvent] = []
    for i, gap in enumerate(gaps):
        month = i + 1
        events.append(RollEvent(
            roll_date=datetime(2023, month, 15),
            old_contract_close=4000.0 + i * 100,
            new_contract_close=4000.0 + i * 100 + gap,
            gap_points=gap,
        ))
    return events


class TestGapCalculator:

    def test_cumulative_calculation(self) -> None:
        events = _make_events(10.0, -5.0, 3.0)
        adjustments = calculate_adjustments(events)
        assert len(adjustments) == 3
        assert adjustments[0].cumulative_adjustment == pytest.approx(-2.0)
        assert adjustments[1].cumulative_adjustment == pytest.approx(3.0)
        assert adjustments[2].cumulative_adjustment == pytest.approx(0.0)

    def test_most_recent_roll_zero(self) -> None:
        for n in range(1, 5):
            events = _make_events(*[float(i + 1) for i in range(n)])
            adjustments = calculate_adjustments(events)
            assert adjustments[-1].cumulative_adjustment == pytest.approx(0.0)

    def test_future_gap_sum(self) -> None:
        events = _make_events(10.0, 20.0, 30.0, 5.0)
        adjustments = calculate_adjustments(events)
        gaps = [10.0, 20.0, 30.0, 5.0]
        for i, adj in enumerate(adjustments):
            expected = sum(gaps[i + 1:])
            assert adj.cumulative_adjustment == pytest.approx(expected)

    def test_determinism(self) -> None:
        events = _make_events(10.0, -5.0, 3.0)
        a = calculate_adjustments(events)
        b = calculate_adjustments(events)
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert x.cumulative_adjustment == y.cumulative_adjustment

    def test_empty_input(self) -> None:
        assert calculate_adjustments([]) == []

    def test_single_roll(self) -> None:
        events = _make_events(25.0)
        adjustments = calculate_adjustments(events)
        assert len(adjustments) == 1
        assert adjustments[0].cumulative_adjustment == pytest.approx(0.0)
        assert adjustments[0].gap_points == pytest.approx(25.0)

    def test_adjustment_factor_immutability(self) -> None:
        events = _make_events(10.0)
        adjustments = calculate_adjustments(events)
        with pytest.raises(AttributeError):
            adjustments[0].cumulative_adjustment = 99.0
