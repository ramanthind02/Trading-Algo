"""Unit tests for helper functions in portfolio_research.run_portfolio_test."""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from portfolio_research.run_portfolio_test import (
    _build_daily_dates_per_ticker,
    _group_ensembles_by_timeframe,
)
from utils.core.enums import TimeFrame


class _DummyEnsemble:
    def __init__(self, base_tf):  # noqa: ANN001
        self.base_tf = base_tf


def test_group_ensembles_by_timeframe_defaults_none_to_daily() -> None:
    grouped = _group_ensembles_by_timeframe(
        [
            ("daily_default", _DummyEnsemble(None)),
            ("weekly", _DummyEnsemble(TimeFrame.W)),
            ("daily", _DummyEnsemble(TimeFrame.D)),
        ]
    )

    assert set(grouped.keys()) == {TimeFrame.D, TimeFrame.W}
    assert len(grouped[TimeFrame.D]) == 2
    assert len(grouped[TimeFrame.W]) == 1


def test_build_daily_dates_per_ticker_returns_sorted_unique_indices() -> None:
    candles = pd.DataFrame(
        {
            "ticker": ["ES", "ES", "ES", "NQ", "NQ"],
            "datetime": [
                pd.Timestamp("2024-01-03"),
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-03"),
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-01"),
            ],
        }
    )

    result = _build_daily_dates_per_ticker(candles)

    assert list(result.keys()) == ["ES", "NQ"]
    assert result["ES"].tolist() == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-03")]
    assert result["NQ"].tolist() == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
