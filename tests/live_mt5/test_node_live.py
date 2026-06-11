"""Tier 3: a real NautilusTrader ``TradingNode`` wired to FTMO via the adapter
factories (``build_mt5_node_config`` + ``MT5Live*ClientFactory``).

* test_node_connectivity   — read-only: build + run the node, assert the MT5
  clients connect, instruments load into the cache, and an account is reported.
  Runs under MT5_LIVE_TESTS (no orders).
* test_node_places_and_closes_order — full loop: a one-shot strategy submits a
  broker-min EURUSD order through the node; we assert a TEST_MAGIC position
  opens on the broker, then guarantee-flatten. Gated by MT5_LIVE_ORDERS=1 and a
  live market.

The node drives its OWN event loop (kernel.loop); we advance it with
run_until_complete and always dispose + clear_connection_registry in teardown.
"""
from __future__ import annotations

import asyncio
import time as _time
from decimal import Decimal

import pytest

from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.config import StrategyConfig, LoggingConfig

from mt5connect.constants import MT5_VENUE
from mt5connect.factories import (
    build_mt5_node_config,
    clear_connection_registry,
    MT5LiveDataClientFactory,
    MT5LiveExecClientFactory,
)


def _build_node(live, symbols, strategy=None) -> TradingNode:
    clear_connection_registry()
    cfg = build_mt5_node_config(
        live.make_config(symbols),
        logging_config=LoggingConfig(log_level="ERROR"),
    )
    node = TradingNode(config=cfg)
    node.add_data_client_factory("MT5", MT5LiveDataClientFactory)
    node.add_exec_client_factory("MT5", MT5LiveExecClientFactory)
    if strategy is not None:
        node.trader.add_strategy(strategy)
    return node


def _drive_until(node: TradingNode, predicate, timeout_s: float = 30.0):
    """Run the node's loop in small slices until predicate() or timeout."""
    loop = node.kernel.loop
    run_task = loop.create_task(node.run_async())
    deadline = _time.time() + timeout_s
    ok = False
    while _time.time() < deadline:
        loop.run_until_complete(asyncio.sleep(0.5))
        if predicate():
            ok = True
            break
    return loop, run_task, ok


def _shutdown(node: TradingNode, loop, run_task) -> None:
    try:
        loop.run_until_complete(node.stop_async())
    finally:
        run_task.cancel()
        try:
            loop.run_until_complete(run_task)
        except asyncio.CancelledError:
            pass


def test_node_connectivity(live):
    """Build + run a real TradingNode against FTMO; no orders."""
    node = _build_node(live, [live.ORDER_SYMBOL, live.INDEX_SYMBOL])
    loop = run_task = None
    try:
        node.build()
        assert node.is_built()

        loop, run_task, ok = _drive_until(
            node, lambda: node.is_running() and bool(node.cache.instruments())
        )
        assert ok, "node did not reach running + instruments loaded within timeout"

        eurusd = node.cache.instrument(InstrumentId.from_str(f"{live.ORDER_SYMBOL}.MT5"))
        assert eurusd is not None
        assert len(node.cache.accounts()) >= 1, "no account reported by exec client"

        _shutdown(node, loop, run_task)
        loop = run_task = None
    finally:
        if loop is not None and run_task is not None:
            _shutdown(node, loop, run_task)
        try:
            node.dispose()
        except Exception:
            pass
        clear_connection_registry()


# ── One-shot strategy used by the gated order test ─────────────────────────────

class _OneShotConfig(StrategyConfig, frozen=True):
    instrument_id: str
    trade_size: str


class _OneShotBuy(Strategy):
    """Submits a single broker-min BUY on start; flattens on stop."""

    def __init__(self, config: _OneShotConfig) -> None:
        super().__init__(config)
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.trade_size = Decimal(config.trade_size)

    def on_start(self) -> None:
        inst = self.cache.instrument(self.instrument_id)
        if inst is None:
            self.log.error(f"instrument {self.instrument_id} not in cache")
            return
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.BUY,
            quantity=Quantity(float(self.trade_size), inst.size_precision),
        )
        self.submit_order(order)


def test_node_places_and_closes_order(live, orders_enabled):
    """Full node loop: strategy → exec client → broker fill → cache position."""
    live.require_market_open(live.ORDER_SYMBOL)

    strat = _OneShotBuy(_OneShotConfig(
        instrument_id=f"{live.ORDER_SYMBOL}.MT5", trade_size="0.01",
    ))
    node = _build_node(live, [live.ORDER_SYMBOL], strategy=strat)
    loop = run_task = None

    def _test_magic_positions():
        return [p for p in (live.mt5.positions_get(symbol=live.ORDER_SYMBOL) or ())
                if p.magic == live.TEST_MAGIC]

    try:
        node.build()
        loop, run_task, opened = _drive_until(
            node, lambda: bool(_test_magic_positions()), timeout_s=40.0
        )
        assert opened, "node strategy did not open a TEST_MAGIC position on the broker"

        _shutdown(node, loop, run_task)
        loop = run_task = None
    finally:
        if loop is not None and run_task is not None:
            _shutdown(node, loop, run_task)
        # Guarantee flat (hedging-mode close is handled by position-ticket close).
        live.flatten()
        for _ in range(10):
            if not _test_magic_positions():
                break
            _time.sleep(0.5)
        try:
            node.dispose()
        except Exception:
            pass
        clear_connection_registry()
        assert not _test_magic_positions(), "FTMO not flat after node order test"
