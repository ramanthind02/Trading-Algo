from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from data_platform.providers.norgate.backadjust.back_adjuster import (
    apply_back_adjustment,
    validate_adjusted_data,
)
from data_platform.providers.norgate.backadjust.gap_calculator import AdjustmentFactor


def _make_ohlc(n: int = 100, base: float = 4000.0) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    dates = pd.bdate_range("2023-01-02", periods=n)
    closes = base + np.arange(n, dtype=float) + rng.randn(n) * 0.5
    return pd.DataFrame({
        "datetime": dates.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": (dates.astype(np.int64) // 10**9).astype(np.uint32),
        "open": closes - rng.rand(n),
        "high": closes + rng.rand(n) * 2,
        "low": closes - rng.rand(n) * 2,
        "close": closes,
    })


def _make_adjustments() -> list[AdjustmentFactor]:
    return [
        AdjustmentFactor(
            roll_date=datetime(2023, 2, 14),
            gap_points=20.0,
            cumulative_adjustment=10.0,
        ),
        AdjustmentFactor(
            roll_date=datetime(2023, 3, 22),
            gap_points=10.0,
            cumulative_adjustment=0.0,
        ),
    ]


class TestBackAdjuster:

    def test_apply_adjustment_correctness(self) -> None:
        """Pre-first-roll gets +30 (sum ALL gaps), between rolls +10, after last roll +0."""
        df = _make_ohlc()
        adjustments = _make_adjustments()
        result = apply_back_adjustment(df, adjustments)
        dt = pd.to_datetime(result["datetime"])

        pre_mask = dt < datetime(2023, 2, 14)
        if pre_mask.any():
            diff = result.loc[pre_mask, "close"].values - df.loc[pre_mask, "close"].values
            np.testing.assert_allclose(diff, 30.0, atol=1e-6)

        mid_mask = (dt >= datetime(2023, 2, 14)) & (dt < datetime(2023, 3, 22))
        if mid_mask.any():
            diff = result.loc[mid_mask, "close"].values - df.loc[mid_mask, "close"].values
            np.testing.assert_allclose(diff, 10.0, atol=1e-6)

        post_mask = dt >= datetime(2023, 3, 22)
        if post_mask.any():
            diff = result.loc[post_mask, "close"].values - df.loc[post_mask, "close"].values
            np.testing.assert_allclose(diff, 0.0, atol=1e-6)

    def test_volume_unchanged(self) -> None:
        df = _make_ohlc()
        df["volume"] = np.arange(len(df), dtype=float) * 100
        adjustments = _make_adjustments()
        result = apply_back_adjustment(df, adjustments)
        pd.testing.assert_series_equal(result["volume"], df["volume"])

    def test_negative_prices_warns(self) -> None:
        """Negative adjusted prices log a warning (expected for commodities)."""
        df = _make_ohlc(base=5.0)
        huge_negative = [
            AdjustmentFactor(
                roll_date=datetime(2023, 2, 14),
                gap_points=-1000.0,
                cumulative_adjustment=-1000.0,
            ),
        ]
        result = apply_back_adjustment(df, huge_negative)
        # Should not raise — just warns
        assert validate_adjusted_data(df, result, huge_negative) is True

    def test_immutability(self) -> None:
        df = _make_ohlc()
        original = df.copy()
        apply_back_adjustment(df, _make_adjustments())
        pd.testing.assert_frame_equal(df, original)

    def test_determinism(self) -> None:
        df = _make_ohlc()
        adjustments = _make_adjustments()
        a = apply_back_adjustment(df, adjustments)
        b = apply_back_adjustment(df, adjustments)
        pd.testing.assert_frame_equal(a, b)

    def test_empty_adjustments(self) -> None:
        df = _make_ohlc()
        result = apply_back_adjustment(df, [])
        pd.testing.assert_frame_equal(result, df)

    def test_row_count_preserved(self) -> None:
        df = _make_ohlc()
        result = apply_back_adjustment(df, _make_adjustments())
        assert len(result) == len(df)

    def test_all_ohlc_columns_adjusted(self) -> None:
        df = _make_ohlc()
        adjustments = _make_adjustments()
        result = apply_back_adjustment(df, adjustments)
        for col in ("open", "high", "low", "close"):
            diff = result[col].values - df[col].values
            np.testing.assert_allclose(
                diff,
                result["close"].values - df["close"].values,
                atol=1e-6,
            )
