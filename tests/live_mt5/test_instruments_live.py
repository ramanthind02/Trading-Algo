"""Tier 1 (read-only): MT5InstrumentProvider parses real FTMO symbols into
NautilusTrader instruments. No orders are placed.
"""
from __future__ import annotations

import pytest

from nautilus_trader.model.instruments import Cfd, CurrencyPair
from mt5connect.providers import MT5InstrumentProvider


def test_load_fx_symbol_is_currency_pair(live, ftmo_conn):
    provider = MT5InstrumentProvider(ftmo_conn)
    inst = provider.load_symbol(live.ORDER_SYMBOL)  # EURUSD
    assert isinstance(inst, CurrencyPair)
    assert inst.price_precision == 5
    assert str(inst.id) == f"{live.ORDER_SYMBOL}.MT5"


def test_load_index_cfd_symbol(live, ftmo_conn):
    provider = MT5InstrumentProvider(ftmo_conn)
    inst = provider.load_symbol(live.INDEX_SYMBOL)  # US100.cash
    assert isinstance(inst, Cfd)
    expected_pp = live.mt5.symbol_info(live.INDEX_SYMBOL).digits
    assert inst.price_precision == expected_pp


@pytest.mark.asyncio
async def test_load_all_with_filter_loads_only_requested(live, ftmo_conn):
    provider = MT5InstrumentProvider(ftmo_conn)
    # Filter uppercases names internally, so use all-caps FX majors.
    await provider.load_all_async(filters={"symbols": ["EURUSD", "GBPUSD"]})

    assert provider.failed_symbols == []
    assert provider.get_instrument("EURUSD") is not None
    assert provider.get_instrument("GBPUSD") is not None
    # Filter must have excluded everything else on the 160+ symbol book.
    assert provider.count == 2


def test_get_instrument_none_before_load(ftmo_conn):
    provider = MT5InstrumentProvider(ftmo_conn)
    assert provider.get_instrument("EURUSD") is None
