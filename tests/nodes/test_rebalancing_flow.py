"""Unit tests for RebalancingFlowNode (single-ticker, long-only monthly flow node).

Covers:
- Column naming + feature-name round-trip parse.
- Observation window (flat before the decision trading day).
- REVERSAL leg: long self when the peer wins; flat when self wins.
- CONTINUATION leg: flat mid-month → long in the EOM window → carry into next month;
  flat when the peer wins.
- BOTH reproduces the slide's asymmetric long-only behavior across a month boundary.
- Configurable decision_trading_day / eom_lead_days.
- Missing cross data → [0.0]; month-change reset; params contract; singleton;
  flow coercion; default peer by ticker; peer-equals-self guard; taxonomy resolution.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

import lib.core.helpers as helpers
from nodes import BiasNode
from nodes.pairs.rebalancing_flow import (
    RebalancingDirection,
    RebalancingFlow,
    RebalancingFlowNode,
)
from lib.core.enums import Ticker, TimeFrame
from lib.core.models import Candle
from utils.data.cross_ticker_store import CrossTickerDataStore


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_ohlcv_df(
    base_close: float = 100.0,
    n_rows: int = 120,
    start_date: datetime = datetime(2020, 2, 25),
    daily_increment: float = 0.0,
) -> pd.DataFrame:
    """Synthetic peer OHLCV indexed by *consecutive calendar* days."""
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


def _make_candle(close: float, dt: datetime, ticker: Ticker = Ticker.ES) -> Candle:
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
    """Up to *n* weekday dates in the given month (Mon-Fri)."""
    import calendar as _cal

    _, last_day = _cal.monthrange(year, month)
    days: list[datetime] = []
    for d in range(1, last_day + 1):
        dt = datetime(year, month, d)
        if dt.weekday() < 5:
            days.append(dt)
        if len(days) >= n:
            break
    return days


@pytest.fixture(autouse=True)
def _reset_store():
    CrossTickerDataStore.reset()
    BiasNode._instances.clear()
    yield
    CrossTickerDataStore.reset()
    BiasNode._instances.clear()


def _set_peer(ticker: Ticker, daily_increment: float) -> None:
    store = CrossTickerDataStore.get_instance()
    store.set_data(ticker, TimeFrame.D, _make_ohlcv_df(daily_increment=daily_increment))


# 15th trading day of March 2020 is 2020-03-20; EOM trading days are Mar 30, 31.
_MARCH = _trading_days(2020, 3)
_APRIL = _trading_days(2020, 4)


# ---------------------------------------------------------------------------
# Column naming
# ---------------------------------------------------------------------------

class TestColumnNaming:
    def test_column_name_and_roundtrip(self) -> None:
        _set_peer(Ticker.TLT, 0.0)
        node = RebalancingFlowNode(
            Ticker.ES, TimeFrame.D, cross_tickers=["TLT"], flow="reversal"
        )
        cols = node.get_column_names()
        assert len(cols) == 1
        col = cols[0]
        for token in ("rebalancing_flow", "flow", "reversal", "crossTickers", "TLT"):
            assert token in col

        parsed = helpers.parse_feature_column_name(col)
        assert parsed["module"] == "rebalancing_flow"
        assert parsed["feature"] == "signal"
        assert parsed["tf"] == TimeFrame.D


# ---------------------------------------------------------------------------
# Observation window
# ---------------------------------------------------------------------------

class TestObservationWindow:
    def test_flat_before_decision_trading_day(self) -> None:
        _set_peer(Ticker.TLT, 0.0)
        node = RebalancingFlowNode(Ticker.ES, TimeFrame.D, cross_tickers=["TLT"])
        # First 14 trading days (before the 15th) must be flat regardless of decision.
        for dt in _MARCH[:14]:
            assert node.add_candle(_make_candle(100.0, dt)) == [0.0]


# ---------------------------------------------------------------------------
# Reversal leg
# ---------------------------------------------------------------------------

class TestReversal:
    def test_long_self_when_peer_wins(self) -> None:
        _set_peer(Ticker.TLT, 1.0)  # TLT rises fast, ES flat → peer wins
        node = RebalancingFlowNode(
            Ticker.ES, TimeFrame.D, cross_tickers=["TLT"], flow=RebalancingFlow.REVERSAL
        )
        sig = {dt: node.add_candle(_make_candle(100.0, dt))[0] for dt in _MARCH}
        # Decision on the 15th trading day (Mar 20) → long through EOM.
        for i, dt in enumerate(_MARCH):
            expected = 1.0 if (i + 1) >= 15 else 0.0
            assert sig[dt] == expected, f"{dt} (td {i+1}) -> {sig[dt]}"

    def test_flat_when_self_wins(self) -> None:
        _set_peer(Ticker.TLT, 0.0)  # ES rises, TLT flat → self wins
        node = RebalancingFlowNode(
            Ticker.ES, TimeFrame.D, cross_tickers=["TLT"], flow=RebalancingFlow.REVERSAL
        )
        sig = [node.add_candle(_make_candle(100.0 + i * 2.0, dt))[0]
               for i, dt in enumerate(_MARCH)]
        assert all(s == 0.0 for s in sig)


# ---------------------------------------------------------------------------
# Continuation leg
# ---------------------------------------------------------------------------

class TestContinuation:
    def test_eom_then_carry_into_next_month(self) -> None:
        _set_peer(Ticker.TLT, 0.0)  # ES rises, TLT flat → self wins
        node = RebalancingFlowNode(
            Ticker.ES, TimeFrame.D, cross_tickers=["TLT"],
            flow=RebalancingFlow.CONTINUATION,
        )
        march = {dt: node.add_candle(_make_candle(100.0 + i * 2.0, dt))[0]
                 for i, dt in enumerate(_MARCH)}
        april = {i + 1: node.add_candle(_make_candle(100.0, dt))[0]
                 for i, dt in enumerate(_APRIL[:8])}

        # Decision (Mar 20) made, but flat until the EOM window.
        for i, dt in enumerate(_MARCH):
            td = i + 1
            if td < 15 or (15 <= td and dt.day <= 28):
                assert march[dt] == 0.0, f"{dt} -> {march[dt]}"
        # EOM window (within 3 calendar days of Mar 31): Mar 30, 31 → long.
        assert march[datetime(2020, 3, 30)] == 1.0
        assert march[datetime(2020, 3, 31)] == 1.0
        # Carry-over: first 5 April trading days long, then flat.
        for td in range(1, 6):
            assert april[td] == 1.0
        for td in range(6, 9):
            assert april[td] == 0.0

    def test_flat_when_peer_wins(self) -> None:
        _set_peer(Ticker.TLT, 1.0)  # TLT rises → peer wins, no continuation
        node = RebalancingFlowNode(
            Ticker.ES, TimeFrame.D, cross_tickers=["TLT"],
            flow=RebalancingFlow.CONTINUATION,
        )
        march = [node.add_candle(_make_candle(100.0, dt))[0] for dt in _MARCH]
        april = [node.add_candle(_make_candle(100.0, dt))[0] for dt in _APRIL[:5]]
        assert all(s == 0.0 for s in march)
        assert all(s == 0.0 for s in april)


# ---------------------------------------------------------------------------
# BOTH reproduces the asymmetric long-only slide behavior
# ---------------------------------------------------------------------------

class TestBoth:
    def test_es_self_wins_reversal_off_continuation_on(self) -> None:
        """ES wins → no reversal long; continuation long at EOM + into next month."""
        _set_peer(Ticker.TLT, 0.0)  # ES rises, TLT flat → self wins
        node = RebalancingFlowNode(
            Ticker.ES, TimeFrame.D, cross_tickers=["TLT"], flow=RebalancingFlow.BOTH
        )
        march = {dt: node.add_candle(_make_candle(100.0 + i * 2.0, dt))[0]
                 for i, dt in enumerate(_MARCH)}
        april = {i + 1: node.add_candle(_make_candle(100.0, dt))[0]
                 for i, dt in enumerate(_APRIL[:6])}
        # Mid-month flat, EOM long, carry into April.
        assert march[datetime(2020, 3, 23)] == 0.0
        assert march[datetime(2020, 3, 31)] == 1.0
        for td in range(1, 6):
            assert april[td] == 1.0
        assert april[6] == 0.0

    def test_tlt_reversal_only_matches_legacy_cross(self) -> None:
        """TLT leg with REVERSAL ~ old RebalancingCrossNode: long TLT when ES wins."""
        _set_peer(Ticker.ES, 1.0)  # ES (peer) rises → peer wins from TLT's view
        node = RebalancingFlowNode(
            Ticker.TLT, TimeFrame.D, cross_tickers=["ES"], flow=RebalancingFlow.REVERSAL
        )
        sig = {dt: node.add_candle(_make_candle(100.0, dt, ticker=Ticker.TLT))[0]
               for dt in _MARCH}
        for i, dt in enumerate(_MARCH):
            assert sig[dt] == (1.0 if (i + 1) >= 15 else 0.0)


# ---------------------------------------------------------------------------
# Configurable timing
# ---------------------------------------------------------------------------

class TestConfigurableTiming:
    def test_decision_trading_day_override(self) -> None:
        _set_peer(Ticker.TLT, 1.0)  # peer wins
        node = RebalancingFlowNode(
            Ticker.ES, TimeFrame.D, cross_tickers=["TLT"],
            flow=RebalancingFlow.REVERSAL, decision_trading_day=5,
        )
        sig = [node.add_candle(_make_candle(100.0, dt))[0] for dt in _MARCH]
        for i, s in enumerate(sig):
            assert s == (1.0 if (i + 1) >= 5 else 0.0)

    def test_eom_lead_days_override(self) -> None:
        _set_peer(Ticker.TLT, 0.0)  # self wins → continuation
        node = RebalancingFlowNode(
            Ticker.ES, TimeFrame.D, cross_tickers=["TLT"],
            flow=RebalancingFlow.CONTINUATION, eom_lead_days=1,
        )
        march = {dt: node.add_candle(_make_candle(100.0 + i * 2.0, dt))[0]
                 for i, dt in enumerate(_MARCH)}
        # eom_lead_days=1 → only the last calendar day (Mar 31) is in-window.
        assert march[datetime(2020, 3, 30)] == 0.0
        assert march[datetime(2020, 3, 31)] == 1.0


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

class TestMissingData:
    def test_returns_neutral_when_peer_missing(self) -> None:
        store = CrossTickerDataStore.get_instance()
        store.set_data(Ticker.TLT, TimeFrame.D, _make_ohlcv_df(
            n_rows=5, start_date=datetime(1999, 1, 1)))
        node = RebalancingFlowNode(Ticker.ES, TimeFrame.D, cross_tickers=["TLT"])
        for dt in _MARCH[:10]:
            assert node.add_candle(_make_candle(100.0, dt)) == [0.0]


class TestMonthReset:
    def test_state_resets_on_new_month(self) -> None:
        _set_peer(Ticker.TLT, 1.0)  # peer wins, flow reversal → no carry
        node = RebalancingFlowNode(
            Ticker.ES, TimeFrame.D, cross_tickers=["TLT"], flow=RebalancingFlow.REVERSAL
        )
        for dt in _MARCH:
            node.add_candle(_make_candle(100.0, dt))
        # First April day starts fresh observation.
        assert node.add_candle(_make_candle(100.0, _APRIL[0])) == [0.0]


# ---------------------------------------------------------------------------
# Long/short direction
# ---------------------------------------------------------------------------

class TestLongShort:
    def test_reversal_longs_loser_shorts_winner(self) -> None:
        # peer (TLT) wins -> self (ES) is loser -> long +1 from TD15 to EOM
        _set_peer(Ticker.TLT, 1.0)
        node = RebalancingFlowNode(Ticker.ES, TimeFrame.D, cross_tickers=["TLT"],
                                   flow=RebalancingFlow.REVERSAL, direction="long_short")
        sig = [node.add_candle(_make_candle(100.0, dt))[0] for dt in _MARCH]
        for i, s in enumerate(sig):
            assert s == (1.0 if (i + 1) >= 15 else 0.0)

        # self (ES) wins -> short -1 from TD15 to EOM
        _set_peer(Ticker.TLT, 0.0)
        BiasNode._instances.clear()
        node = RebalancingFlowNode(Ticker.ES, TimeFrame.D, cross_tickers=["TLT"],
                                   flow=RebalancingFlow.REVERSAL, direction="long_short")
        sig = [node.add_candle(_make_candle(100.0 + i * 2.0, dt))[0] for i, dt in enumerate(_MARCH)]
        for i, s in enumerate(sig):
            assert s == (-1.0 if (i + 1) >= 15 else 0.0)

    def test_continuation_shorts_loser_into_next_month(self) -> None:
        # peer (TLT) wins -> self (ES) is loser -> continuation shorts the loser
        _set_peer(Ticker.TLT, 1.0)
        node = RebalancingFlowNode(Ticker.ES, TimeFrame.D, cross_tickers=["TLT"],
                                   flow=RebalancingFlow.CONTINUATION, direction="long_short")
        for dt in _MARCH:
            node.add_candle(_make_candle(100.0, dt))
        april = {i + 1: node.add_candle(_make_candle(100.0, dt))[0] for i, dt in enumerate(_APRIL[:6])}
        for td in range(1, 6):
            assert april[td] == -1.0, f"expected short carry on April TD{td}, got {april[td]}"
        assert april[6] == 0.0

    def test_long_only_is_default_and_never_shorts(self) -> None:
        _set_peer(Ticker.TLT, 0.0)  # self wins
        node = RebalancingFlowNode(Ticker.ES, TimeFrame.D, cross_tickers=["TLT"])
        assert node.params["direction"] == "long_only"
        sig = [node.add_candle(_make_candle(100.0 + i * 2.0, dt))[0] for i, dt in enumerate(_MARCH)]
        assert min(sig) == 0.0 and max(sig) == 1.0  # only 0/1, never -1


# ---------------------------------------------------------------------------
# Params / construction
# ---------------------------------------------------------------------------

class TestParamsContract:
    def test_params_keys_and_values(self) -> None:
        _set_peer(Ticker.TLT, 0.0)
        node = RebalancingFlowNode(Ticker.ES, TimeFrame.D, cross_tickers=["TLT"])
        assert node.params == {
            "cross_tickers": ["TLT"],
            "flow": "both",
            "direction": "long_only",
            "decision_trading_day": 15,
            "eom_lead_days": 3,
            "continuation_hold": 5,
        }

    def test_default_peer_depends_on_ticker(self) -> None:
        store = CrossTickerDataStore.get_instance()
        store.set_data(Ticker.TLT, TimeFrame.D, _make_ohlcv_df())
        store.set_data(Ticker.ES, TimeFrame.D, _make_ohlcv_df())
        assert RebalancingFlowNode(Ticker.ES, TimeFrame.D).cross_ticker == Ticker.TLT
        assert RebalancingFlowNode(Ticker.TLT, TimeFrame.D).cross_ticker == Ticker.ES

    def test_peer_equal_to_self_raises(self) -> None:
        with pytest.raises(ValueError):
            RebalancingFlowNode(Ticker.ES, TimeFrame.D, cross_tickers=["ES"])

    def test_invalid_flow_raises(self) -> None:
        with pytest.raises(ValueError):
            RebalancingFlowNode(Ticker.ES, TimeFrame.D, cross_tickers=["TLT"], flow="bogus")


class TestSingleton:
    def test_flow_distinguishes_instances(self) -> None:
        _set_peer(Ticker.TLT, 0.0)
        a = RebalancingFlowNode.get_instance(
            Ticker.ES, TimeFrame.D, cross_tickers=["TLT"], flow="both")
        b = RebalancingFlowNode.get_instance(
            Ticker.ES, TimeFrame.D, cross_tickers=["TLT"], flow="both")
        c = RebalancingFlowNode.get_instance(
            Ticker.ES, TimeFrame.D, cross_tickers=["TLT"], flow="reversal")
        assert a is b
        assert a is not c


class TestTaxonomyResolution:
    def test_create_bias_node_resolves(self) -> None:
        _set_peer(Ticker.TLT, 0.0)
        node = helpers.create_fresh_bias_node(
            "rebalancing_flow", Ticker.ES, TimeFrame.D,
            {"flow": "both", "cross_tickers": ["TLT"]},
        )
        assert isinstance(node, RebalancingFlowNode)
        assert "rebalancing_flow" in node.get_column_names()[0]
