from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from data_platform.providers.norgate.backadjust.roll_detector import RollEvent, detect_roll_dates
from data_platform.providers.norgate.backadjust.roll_rules import RollRule
from utils.core.enums import Ticker


def _make_daily_series(
    n_days: int = 120,
    base_price: float = 4000.0,
    gap_day: int = 60,
    gap_size: float = 50.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic daily DataFrame with one gap at gap_day."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    closes: list[float] = []
    for i in range(n_days):
        price = base_price + i * 1.0
        if i >= gap_day:
            price += gap_size
        closes.append(price + rng.randn() * 0.3)
    closes_arr = np.array(closes)
    return pd.DataFrame({
        "datetime": dates.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": (dates.astype(np.int64) // 10**9).astype(np.uint32),
        "open": closes_arr - rng.rand(n_days) * 0.5,
        "high": closes_arr + rng.rand(n_days) * 1.0,
        "low": closes_arr - rng.rand(n_days) * 1.5,
        "close": closes_arr,
    })


_PERMISSIVE_RULE = RollRule(
    ticker=Ticker.ES,
    rollover_offset=-30,
    reference_point="expiration",
    description="test rule: wide window",
)


class TestRollDetector:

    def test_detect_known_roll(self) -> None:
        df = _make_daily_series(gap_day=60, gap_size=50.0)
        events = detect_roll_dates(df, _PERMISSIVE_RULE)
        assert len(events) >= 1
        assert any(abs(e.gap_points - 50.0) < 5.0 for e in events)

    def test_no_rolls_found(self) -> None:
        df = _make_daily_series(gap_day=999, gap_size=0.0)
        events = detect_roll_dates(df, _PERMISSIVE_RULE)
        assert events == []

    def test_roll_event_ordering(self) -> None:
        rng = np.random.RandomState(42)
        n_days = 200
        dates = pd.bdate_range("2023-01-02", periods=n_days)
        closes: list[float] = []
        for i in range(n_days):
            price = 4000.0 + i * 1.0
            if i >= 60:
                price += 50.0
            if i >= 130:
                price += 40.0
            closes.append(price + rng.randn() * 0.3)
        closes_arr = np.array(closes)
        df = pd.DataFrame({
            "datetime": dates.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": (dates.astype(np.int64) // 10**9).astype(np.uint32),
            "open": closes_arr,
            "high": closes_arr + 1.0,
            "low": closes_arr - 1.0,
            "close": closes_arr,
        })
        events = detect_roll_dates(df, _PERMISSIVE_RULE)
        if len(events) >= 2:
            for i in range(len(events) - 1):
                assert events[i].roll_date <= events[i + 1].roll_date

    def test_determinism(self) -> None:
        df = _make_daily_series()
        result_a = detect_roll_dates(df, _PERMISSIVE_RULE)
        result_b = detect_roll_dates(df, _PERMISSIVE_RULE)
        assert len(result_a) == len(result_b)
        for a, b in zip(result_a, result_b):
            assert a.roll_date == b.roll_date
            assert a.gap_points == b.gap_points

    def test_roll_event_fields(self) -> None:
        df = _make_daily_series(gap_day=60, gap_size=50.0)
        events = detect_roll_dates(df, _PERMISSIVE_RULE)
        assert len(events) >= 1
        e = events[0]
        assert isinstance(e.roll_date, datetime)
        assert isinstance(e.old_contract_close, float)
        assert isinstance(e.new_contract_close, float)
        assert isinstance(e.gap_points, float)
        assert e.gap_points == e.new_contract_close - e.old_contract_close

    def test_roll_event_immutability(self) -> None:
        df = _make_daily_series(gap_day=60, gap_size=50.0)
        events = detect_roll_dates(df, _PERMISSIVE_RULE)
        assert len(events) >= 1
        with pytest.raises(AttributeError):
            events[0].gap_points = 0.0
