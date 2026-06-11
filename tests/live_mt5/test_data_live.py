"""Tier 1 (read-only): live tick + historical bar parsing on REAL FTMO data,
and MT5DataClient construction against real NautilusTrader components.
No orders are placed. Asserts that depend on an open market skip cleanly
when the feed is empty (weekend / holiday).
"""
from __future__ import annotations

import asyncio
import datetime as dt

import pytest

from nautilus_trader.model.data import Bar, QuoteTick
from mt5connect.providers import MT5InstrumentProvider
from mt5connect.parsing import parse_quote_tick, parse_bar
from mt5connect.data import MT5DataClient


def _eurusd_instrument(live, conn):
    return MT5InstrumentProvider(conn).load_symbol(live.ORDER_SYMBOL)


def test_live_tick_parses_to_quote_tick(live, ftmo_conn):
    raw = live.mt5.symbol_info_tick(live.ORDER_SYMBOL)
    if raw is None or not raw.bid:
        pytest.skip("no live EURUSD tick (market closed?)")

    inst = _eurusd_instrument(live, ftmo_conn)
    tick = parse_quote_tick(raw, inst)

    assert isinstance(tick, QuoteTick)
    assert tick.instrument_id == inst.id
    assert float(tick.ask_price) >= float(tick.bid_price) > 0.0


def test_historical_bars_parse(live, ftmo_conn):
    inst = _eurusd_instrument(live, ftmo_conn)
    end = dt.datetime.now(dt.timezone.utc)
    rates = live.mt5.copy_rates_range(
        live.ORDER_SYMBOL, live.mt5.TIMEFRAME_H1, end - dt.timedelta(days=7), end
    )
    if rates is None or len(rates) == 0:
        pytest.skip("no H1 bars in last 7d (market closed?)")

    bars = [parse_bar(r, inst, live.mt5.TIMEFRAME_H1) for r in rates]
    assert all(isinstance(b, Bar) for b in bars)
    last = bars[-1]
    assert float(last.high) >= float(last.low) > 0.0
    assert float(last.high) >= float(last.close) >= float(last.low)
    assert float(last.volume) >= 0.0


def test_data_client_constructs_against_real_nt(live, ftmo_conn, nt_components):
    msgbus, cache, clock = nt_components
    provider = MT5InstrumentProvider(ftmo_conn)
    loop = asyncio.new_event_loop()
    try:
        client = MT5DataClient(
            loop=loop,
            connection=ftmo_conn,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
            instrument_provider=provider,
            config=live.make_config([live.ORDER_SYMBOL]),
        )
        assert client.id is not None
        assert client.is_polling is False
        assert client.subscribed_tick_symbols == []
    finally:
        loop.close()
