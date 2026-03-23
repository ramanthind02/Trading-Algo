"""Unit tests for helper functions in portfolio_research.run_portfolio_test."""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from portfolio_research.pipelines.portfolio_test import (
    _build_daily_dates_per_ticker,
    _group_ensembles_by_timeframe,
    _portfolio_cache_query,
)
from utils.core.enums import Ticker, TimeFrame


class _DummyEnsemble:
    def __init__(self, base_tf):  # noqa: ANN001
        self.base_tf = base_tf


class _DummyConfig:
    def __init__(self) -> None:
        self.tickers = [Ticker.ES, "NQ"]


def test_portfolio_cache_query_builds_ticker_strings_and_timeframe_tuple() -> None:
    config = _DummyConfig()
    start = pd.Timestamp("2024-01-01").to_pydatetime()
    end = pd.Timestamp("2024-01-05").to_pydatetime()

    query = _portfolio_cache_query(config, start, end, (TimeFrame.D, TimeFrame.W))

    assert query.tickers == ("ES", "NQ")
    assert query.start == start
    assert query.end == end
    assert query.timeframes == (TimeFrame.D, TimeFrame.W)


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
