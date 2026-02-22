from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_m1_dataframe() -> pd.DataFrame:
    """
    Synthetic 1-minute OHLCV DataFrame spanning 60 trading days.
    Includes one deliberate price gap (roll) at day 30.
    """
    rows = []
    base_price = 4000.0
    minutes_per_day = 390  # 6.5 hours
    gap_day = 30
    gap_size = 50.0  # points

    for day in range(60):
        date = datetime(2023, 1, 2) + timedelta(days=day)
        if date.weekday() >= 5:
            continue
        price = base_price + day * 2.0
        if day >= gap_day:
            price += gap_size  # simulate roll gap

        for minute in range(minutes_per_day):
            ts = date + timedelta(hours=9, minutes=30) + timedelta(minutes=minute)
            noise = np.random.RandomState(day * 1000 + minute).randn() * 0.5
            o = price + noise
            h = o + abs(noise) * 0.5
            l = o - abs(noise) * 0.5
            c = o + noise * 0.3
            rows.append({
                "datetime": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "timestamp": int(ts.timestamp()),
                "open": round(o, 2),
                "high": round(h, 2),
                "low": round(l, 2),
                "close": round(c, 2),
            })

    return pd.DataFrame(rows)


@pytest.fixture
def sample_daily_dataframe() -> pd.DataFrame:
    """Synthetic daily OHLC DataFrame with a known gap at row 30."""
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-02", periods=60)
    base = 4000.0
    gap_size = 50.0

    closes = []
    for i in range(60):
        price = base + i * 2.0
        if i >= 30:
            price += gap_size
        closes.append(price + np.random.randn() * 0.5)

    closes = np.array(closes)
    return pd.DataFrame({
        "datetime": dates.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": (dates.astype(np.int64) // 10**9).astype(np.uint32),
        "open": closes - np.random.rand(60) * 2,
        "high": closes + np.random.rand(60) * 2,
        "low": closes - np.random.rand(60) * 3,
        "close": closes,
    })
