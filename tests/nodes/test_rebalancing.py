"""Unit tests for the RebalancingNode (monthly rebalancing pairs node).

Tests cover:
- Column naming conventions
- Observation period returns 0
- Cross-ticker outperforms → long primary ticker signal
- Primary ticker outperforms → flat then long at month end + carry-over
- Month transitions and state reset
- Missing cross-ticker data returns neutral
- Singleton compatibility
- Params contract includes cross_tickers
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from nodes import BiasNode
from nodes.pairs.rebalancing import RebalancingNode
from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle
from utils.data.cross_ticker_store import CrossTickerDataStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ohlcv_df(
    base_close: float = 100.0,
    n_rows: int = 50,
    start_date: datetime = datetime(2020, 1, 1),
    daily_increment: float = 0.5,
) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame indexed by datetime."""
    dates = [start_date + timedelta(days=i) for i in range(n_rows)]
    closes = [base_close + i * daily_increment for i in range(n_rows)]
    return pd.DataFrame(
        {
            "datetime": dates,
            "open": [c - 0.1 for c in closes],
            "high": [c + 1.0 for c in closes],
            "low": [c - 1.0 for c in closes],
            "close": closes,
            "volume": [1000.0] * n_rows,
        }
    ).set_index("datetime")


def _make_candle(
    close: float, dt: datetime, ticker: Ticker = Ticker.ES
) -> Candle:
    return Candle(
        datetime=dt,
        open=close - 0.1,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=1000.0,
        ticker=ticker,
        tf=TimeFrame.D,
    )


def _trading_days(year: int, month: int, n: int = 31) -> list[datetime]:
    """Return up to *n* weekday dates in the given month."""
    import calendar
    _, last_day = calendar.monthrange(year, month)
    days: list[datetime] = []
    for d in range(1, last_day + 1):
        dt = datetime(year, month, d)
        if dt.weekday() < 5:  # Mon-Fri
            days.append(dt)
        if len(days) >= n:
            break
    return days


@pytest.fixture(autouse=True)
def _reset_store():
    """Ensure a clean store for every test."""
    CrossTickerDataStore.reset()
    BiasNode._instances.clear()
    yield
    CrossTickerDataStore.reset()
    BiasNode._instances.clear()


# ---------------------------------------------------------------------------
# Column naming
# ---------------------------------------------------------------------------

class TestColumnNaming:
    def test_column_name_contains_module_and_cross_ticker(self) -> None:
        store = CrossTickerDataStore.get_instance()
        store.set_data(Ticker.TLT, TimeFrame.D, _make_ohlcv_df())

        node = RebalancingNode(Ticker.ES, TimeFrame.D, cross_tickers=["TLT"])
        cols = node.get_column_names()
        assert len(cols) == 1
        assert "rebalancing" in cols[0]
        assert "crossTickers" in cols[0]
        assert "TLT" in cols[0]


# ---------------------------------------------------------------------------
# Observation period
# ---------------------------------------------------------------------------

class TestObservationPeriod:
    def test_first_15_days_return_zero(self) -> None:
        """During observation (calendar days 1-15), signal must be 0."""
        days = _trading_days(2020, 3)  # March 2020
        obs_days = [d for d in days if d.day <= 15]

        # TLT flat, ES flat → doesn't matter, should be 0 during observation
        store = CrossTickerDataStore.get_instance()
        tlt_df = _make_ohlcv_df(
            base_close=100.0, n_rows=60, start_date=datetime(2020, 2, 25),
            daily_increment=0.0,
        )
        store.set_data(Ticker.TLT, TimeFrame.D, tlt_df)

        node = RebalancingNode(Ticker.ES, TimeFrame.D, cross_tickers=["TLT"])
        for dt in obs_days:
            result = node.add_candle(_make_candle(100.0, dt))
            assert result == [0.0], f"Expected [0.0] during observation on {dt}"


# ---------------------------------------------------------------------------
# TLT outperforms → long ES
# ---------------------------------------------------------------------------

class TestTLTWins:
    def test_long_es_after_tlt_outperforms(self) -> None:
        """When TLT gains more than ES in first 15 days, signal = 1 after day 15."""
        days = _trading_days(2020, 3)

        # TLT rises fast (1% per day), ES flat
        store = CrossTickerDataStore.get_instance()
        tlt_df = _make_ohlcv_df(
            base_close=100.0, n_rows=60, start_date=datetime(2020, 2, 25),
            daily_increment=1.0,
        )
        store.set_data(Ticker.TLT, TimeFrame.D, tlt_df)

        node = RebalancingNode(Ticker.ES, TimeFrame.D, cross_tickers=["TLT"])
        signals: dict[datetime, float] = {}
        for dt in days:
            result = node.add_candle(_make_candle(100.0, dt))  # ES flat
            signals[dt] = result[0]

        # After day 15 (decision made), should be long
        post_decision = {d: s for d, s in signals.items() if d.day > 15}
        assert all(s == 1.0 for s in post_decision.values()), (
            f"Expected all 1.0 after TLT wins, got {post_decision}"
        )


# ---------------------------------------------------------------------------
# ES outperforms → flat then long ES at month end + carry-over
# ---------------------------------------------------------------------------

class TestESWins:
    def test_flat_mid_month_then_long_at_end(self) -> None:
        """When ES outperforms TLT, signal = 0 mid-month, 1 from day 25+."""
        days = _trading_days(2020, 3)

        # ES rises fast, TLT flat
        store = CrossTickerDataStore.get_instance()
        tlt_df = _make_ohlcv_df(
            base_close=100.0, n_rows=60, start_date=datetime(2020, 2, 25),
            daily_increment=0.0,
        )
        store.set_data(Ticker.TLT, TimeFrame.D, tlt_df)

        node = RebalancingNode(Ticker.ES, TimeFrame.D, cross_tickers=["TLT"])
        signals: dict[datetime, float] = {}
        for i, dt in enumerate(days):
            es_close = 100.0 + i * 2.0  # ES rises
            result = node.add_candle(_make_candle(es_close, dt))
            signals[dt] = result[0]

        # Mid-month after decision but before day 25 should be 0
        mid_month = {d: s for d, s in signals.items() if 15 < d.day < 25}
        assert all(s == 0.0 for s in mid_month.values()), (
            f"Expected 0.0 mid-month when ES wins, got {mid_month}"
        )

        # Day 25+ should be 1
        end_month = {d: s for d, s in signals.items() if d.day >= 25}
        assert all(s == 1.0 for s in end_month.values()), (
            f"Expected 1.0 at end of month when ES wins, got {end_month}"
        )

    def test_carry_over_into_next_month(self) -> None:
        """ES-wins carry-over: long ES through 5th trading day of next month."""
        march_days = _trading_days(2020, 3)
        april_days = _trading_days(2020, 4)

        # ES rises, TLT flat → ES wins March
        store = CrossTickerDataStore.get_instance()
        tlt_df = _make_ohlcv_df(
            base_close=100.0, n_rows=120, start_date=datetime(2020, 2, 25),
            daily_increment=0.0,
        )
        store.set_data(Ticker.TLT, TimeFrame.D, tlt_df)

        node = RebalancingNode(Ticker.ES, TimeFrame.D, cross_tickers=["TLT"])

        # Stream March
        for i, dt in enumerate(march_days):
            node.add_candle(_make_candle(100.0 + i * 2.0, dt))

        # Stream first 10 April trading days — first 5 should carry over
        april_signals: dict[int, float] = {}
        for i, dt in enumerate(april_days[:10]):
            result = node.add_candle(_make_candle(100.0, dt))
            april_signals[i + 1] = result[0]  # 1-indexed trading day

        # Trading days 1-5 of April should be 1.0 (carry-over)
        for td in range(1, 6):
            assert april_signals[td] == 1.0, (
                f"Expected carry-over signal 1.0 on trading day {td}, got {april_signals[td]}"
            )

        # Trading day 6+ should NOT carry over (but may be 0 if in observation)
        for td in range(6, 11):
            if td in april_signals:
                assert april_signals[td] == 0.0, (
                    f"Expected 0.0 after carry-over ends on trading day {td}, got {april_signals[td]}"
                )


# ---------------------------------------------------------------------------
# Month transitions and state reset
# ---------------------------------------------------------------------------

class TestMonthTransitions:
    def test_state_resets_on_new_month(self) -> None:
        """Switching months should reset observation state."""
        store = CrossTickerDataStore.get_instance()
        # TLT rises so TLT outperforms ES (no carry-over into April)
        tlt_df = _make_ohlcv_df(
            base_close=100.0, n_rows=120, start_date=datetime(2020, 2, 25),
            daily_increment=1.0,
        )
        store.set_data(Ticker.TLT, TimeFrame.D, tlt_df)

        node = RebalancingNode(Ticker.ES, TimeFrame.D, cross_tickers=["TLT"])

        # Stream all of March (TLT wins — ES flat, TLT rises)
        march_days = _trading_days(2020, 3)
        for dt in march_days:
            node.add_candle(_make_candle(90.0, dt))  # ES flat, TLT rises

        # First day of April: node should start fresh observation
        april_1 = _trading_days(2020, 4)[0]
        result = node.add_candle(_make_candle(100.0, april_1))
        # First day is always 0 (recording month start)
        assert result == [0.0]


# ---------------------------------------------------------------------------
# Missing cross-ticker data
# ---------------------------------------------------------------------------

class TestMissingData:
    def test_returns_neutral_when_tlt_missing(self) -> None:
        """If TLT candle is unavailable, return [0.0]."""
        store = CrossTickerDataStore.get_instance()
        # Load TLT data for a completely different date range
        tlt_df = _make_ohlcv_df(
            base_close=100.0, n_rows=5, start_date=datetime(1999, 1, 1),
        )
        store.set_data(Ticker.TLT, TimeFrame.D, tlt_df)

        node = RebalancingNode(Ticker.ES, TimeFrame.D, cross_tickers=["TLT"])
        for dt in _trading_days(2020, 3)[:10]:
            result = node.add_candle(_make_candle(100.0, dt))
            assert result == [0.0], f"Expected [0.0] when TLT data missing on {dt}"


# ---------------------------------------------------------------------------
# Singleton compatibility
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_singleton_returns_same_instance(self) -> None:
        store = CrossTickerDataStore.get_instance()
        store.set_data(Ticker.TLT, TimeFrame.D, _make_ohlcv_df())

        a = RebalancingNode.get_instance(Ticker.ES, TimeFrame.D, cross_tickers=["TLT"])
        b = RebalancingNode.get_instance(Ticker.ES, TimeFrame.D, cross_tickers=["TLT"])
        assert a is b

    def test_different_cross_tickers_different_instance(self) -> None:
        store = CrossTickerDataStore.get_instance()
        store.set_data(Ticker.TLT, TimeFrame.D, _make_ohlcv_df())
        store.set_data(Ticker.NQ, TimeFrame.D, _make_ohlcv_df())

        a = RebalancingNode.get_instance(Ticker.ES, TimeFrame.D, cross_tickers=["TLT"])
        b = RebalancingNode.get_instance(Ticker.ES, TimeFrame.D, cross_tickers=["NQ"])
        assert a is not b


# ---------------------------------------------------------------------------
# Params contract
# ---------------------------------------------------------------------------

class TestParamsContract:
    def test_params_include_cross_tickers(self) -> None:
        store = CrossTickerDataStore.get_instance()
        store.set_data(Ticker.TLT, TimeFrame.D, _make_ohlcv_df())

        node = RebalancingNode(Ticker.ES, TimeFrame.D, cross_tickers=["TLT"])
        assert "cross_tickers" in node.params
        assert node.params["cross_tickers"] == ["TLT"]

    def test_default_cross_ticker_is_tlt(self) -> None:
        store = CrossTickerDataStore.get_instance()
        store.set_data(Ticker.TLT, TimeFrame.D, _make_ohlcv_df())

        node = RebalancingNode(Ticker.ES, TimeFrame.D)
        assert node.cross_ticker == Ticker.TLT
