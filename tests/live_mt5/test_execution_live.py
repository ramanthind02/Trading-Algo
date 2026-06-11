"""Tier 2 (PLACES ORDERS — gated by MT5_LIVE_ORDERS=1): a real round-trip
through the adapter's own MT5LiveExecutionClient against the FTMO demo.

Unlike the smoke driver (which calls mt5.order_send directly), this drives the
adapter's REAL methods — _submit_order → _poll_exec_once → _cancel_order — so the
production order/fill/close path is exercised end-to-end. Everything is tagged
TEST_MAGIC and flattened on teardown; the broker-side open/close is the
authoritative assertion.
"""
from __future__ import annotations

import asyncio
import time
import uuid

import pytest

from nautilus_trader.model.enums import OrderSide, OrderType, TimeInForce
from nautilus_trader.model.identifiers import ClientOrderId, InstrumentId, Symbol
from nautilus_trader.model.objects import Quantity
from unittest.mock import MagicMock

from mt5connect.constants import MT5_VENUE
from mt5connect.providers import MT5InstrumentProvider
from mt5connect.execution import MT5LiveExecutionClient


# ── Builders ───────────────────────────────────────────────────────────────────

def _build_exec_client(live, conn, nt_components):
    """Real MT5LiveExecutionClient with EURUSD preloaded + spy'd NT emitters."""
    msgbus, cache, clock = nt_components
    provider = MT5InstrumentProvider(conn)
    provider.load_symbol(live.ORDER_SYMBOL)

    loop = asyncio.new_event_loop()
    client = MT5LiveExecutionClient(
        loop=loop,
        connection=conn,
        msgbus=msgbus,
        cache=cache,
        clock=clock,
        instrument_provider=provider,
        config=live.make_config([live.ORDER_SYMBOL]),
    )
    # Spy the NT emitters (as the unit suite does) so we assert the adapter's
    # decisions without needing a fully NT-registered order/strategy.
    client.generate_order_accepted = MagicMock()
    client.generate_order_rejected = MagicMock()
    client.generate_order_filled = MagicMock()
    client.generate_account_state = MagicMock()
    return client, loop


def _market_order(live, side=OrderSide.BUY, volume=None):
    """A MagicMock order with REAL Nautilus identifiers (the adapter only reads
    attributes off the order)."""
    si = live.mt5.symbol_info(live.ORDER_SYMBOL)
    vol = float(volume if volume is not None else si.volume_min)
    step = si.volume_step or 0.01
    size_precision = len(f"{step:.10f}".rstrip("0").split(".")[1]) if "." in f"{step}" else 0

    order = MagicMock()
    order.client_order_id = ClientOrderId(f"LT-{uuid.uuid4().hex[:8]}")
    order.strategy_id = MagicMock()
    order.instrument_id = InstrumentId(Symbol(live.ORDER_SYMBOL), MT5_VENUE)
    order.order_type = OrderType.MARKET
    order.side = side
    order.quantity = Quantity(vol, size_precision)
    order.time_in_force = TimeInForce.GTC
    order.price = None
    order.trigger_price = None
    order.sl_trigger_price = None
    order.tp_price = None
    return order


def _open_test_positions(live):
    return [p for p in (live.mt5.positions_get(symbol=live.ORDER_SYMBOL) or ())
            if p.magic == live.TEST_MAGIC]


# ── Tests ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_market_order_round_trip(live, ftmo_conn, nt_components, orders_enabled):
    # A real fill needs an open market; skip cleanly on weekends/holidays.
    live.require_market_open(live.ORDER_SYMBOL)

    client, loop = _build_exec_client(live, ftmo_conn, nt_components)
    try:
        order = _market_order(live, side=OrderSide.BUY)
        cmd = MagicMock(); cmd.order = order

        # ── OPEN through the adapter ───────────────────────────────────────────
        await client._submit_order(cmd)
        client.generate_order_rejected.assert_not_called()
        client.generate_order_accepted.assert_called_once()
        coid = str(order.client_order_id)
        assert coid in client._client_order_id_to_ticket

        # Broker-side truth: the position is actually open with our magic.
        opened = []
        for _ in range(8):
            time.sleep(0.75)
            opened = _open_test_positions(live)
            if opened:
                break
        assert opened, "no TEST_MAGIC position opened on the broker"

        # ── FILL detection through the adapter's poll loop ─────────────────────
        # _emit_fill now looks the order up in the cache for strategy_id (NT 1.227
        # requires it); our MagicMock order isn't NT-registered, so stub the lookup.
        client._cache.order = MagicMock(return_value=order)
        await client._poll_exec_once()
        assert client.generate_order_filled.call_count >= 1
        # Guard the NT 1.227 signature fix: strategy_id supplied, ts_init NOT passed.
        _kw = client.generate_order_filled.call_args.kwargs
        assert "strategy_id" in _kw and "ts_init" not in _kw

        # ── CLOSE through the adapter ──────────────────────────────────────────
        close_cmd = MagicMock()
        close_cmd.client_order_id = order.client_order_id
        close_cmd.venue_order_id = None
        close_cmd.instrument_id = order.instrument_id
        await client._cancel_order(close_cmd)

        # ── FLAT ───────────────────────────────────────────────────────────────
        left = opened
        for _ in range(8):
            time.sleep(0.75)
            left = _open_test_positions(live)
            if not left:
                break
        assert not left, f"{len(left)} residual TEST_MAGIC position(s) after close"
    finally:
        live.flatten()
        loop.close()


@pytest.mark.asyncio
async def test_oversized_order_is_rejected(live, ftmo_conn, nt_components, orders_enabled):
    """An impossible volume must be rejected by the broker → the adapter emits
    OrderRejected and opens NOTHING (exercises the reject path live, no residue)."""
    client, loop = _build_exec_client(live, ftmo_conn, nt_components)
    try:
        order = _market_order(live, side=OrderSide.BUY, volume=100_000.0)  # ~10bn notional
        cmd = MagicMock(); cmd.order = order

        await client._submit_order(cmd)

        client.generate_order_accepted.assert_not_called()
        client.generate_order_rejected.assert_called_once()
        assert str(order.client_order_id) not in client._client_order_id_to_ticket
        assert _open_test_positions(live) == []
    finally:
        live.flatten()
        loop.close()
