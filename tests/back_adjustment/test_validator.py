from __future__ import annotations

import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from data_platform.providers.norgate.backadjust.validator import (
    compare_price_levels,
    compare_roll_dates,
    generate_comparison_report,
)
from utils.core.enums import Ticker


def _make_aligned_data(
    n: int = 100,
    offset: float = 5.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-02", periods=n)
    base = 4000.0 + np.arange(n, dtype=float) + rng.randn(n) * 0.5

    old_df = pd.DataFrame({
        "datetime": dates.strftime("%Y-%m-%d"),
        "close": base,
    })
    new_df = pd.DataFrame({
        "Date": dates,
        "Close": base + offset + rng.randn(n) * 0.1,
    })
    return old_df, new_df


class TestValidator:

    def test_price_level_comparison(self) -> None:
        old_df, new_df = _make_aligned_data(offset=5.0)
        stats = compare_price_levels(old_df, new_df)
        assert "correlation" in stats
        assert "mean_diff" in stats
        assert "max_divergence" in stats
        assert "rmse" in stats
        assert stats["correlation"] > 0.99
        assert abs(stats["mean_diff"] - 5.0) < 1.0

    def test_roll_date_comparison(self, tmp_path: Path) -> None:
        meta = {
            "ticker": "ES",
            "roll_dates": ["2023-03-15T00:00:00", "2023-06-15T00:00:00"],
            "gap_points": [10.0, 5.0],
        }
        meta_path = tmp_path / "ES.json"
        meta_path.write_text(json.dumps(meta))

        dates = pd.bdate_range("2023-01-02", periods=200)
        delivery = [202303.0] * 50 + [202306.0] * 70 + [202309.0] * 80
        new_df = pd.DataFrame({
            "Date": dates,
            "Close": np.arange(200, dtype=float) + 4000,
            "Delivery Month": delivery,
        })

        result = compare_roll_dates(meta_path, new_df)
        assert isinstance(result, pd.DataFrame)
        assert "roll_date_old" in result.columns
        assert "roll_date_new" in result.columns
        assert "delta_days" in result.columns

    def test_report_generation(self, tmp_path: Path) -> None:
        old_df, new_df = _make_aligned_data()

        meta = {
            "ticker": "ES",
            "roll_dates": ["2023-03-15T00:00:00"],
            "gap_points": [10.0],
            "num_rolls_detected": 1,
        }
        meta_path = tmp_path / "ES.json"
        meta_path.write_text(json.dumps(meta))

        old_path = tmp_path / "old_ES.parquet"
        old_df.to_parquet(old_path, index=False)

        new_path = tmp_path / "new_ES.parquet"
        new_df.to_parquet(new_path, index=False)

        report = generate_comparison_report(
            Ticker.ES, old_path, new_path, meta_path,
            output_dir=tmp_path / "reports",
        )
        assert isinstance(report, str)
        assert "ES" in report
        assert "correlation" in report.lower() or "Correlation" in report

    def test_determinism(self) -> None:
        old_df, new_df = _make_aligned_data()
        a = compare_price_levels(old_df, new_df)
        b = compare_price_levels(old_df, new_df)
        assert a == b
