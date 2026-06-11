"""Gated live smoke test for the vault sandbox runtime (deployment.live).

Builds the REAL sandbox TradingNode (MT5 live DATA client bound to the FTMO
terminal + Nautilus SandboxExecutionClient for LOCAL fills) via
``deployment.live.runtime.node_builder.build_sandbox_node`` and drives its loop
until instruments load and an account is reported. No broker orders are ever
placed — the execution client is the local sandbox, and the daily decision only
fires at the US cash close, so this is a connectivity/wiring smoke test.

Gated by the shared harness in this directory: skips unless MT5_LIVE_TESTS=1 and
a verified FTMO demo binding (see conftest.py).
"""
from __future__ import annotations

import asyncio
import time as _time

from nautilus_trader.model.identifiers import InstrumentId

from mt5connect.factories import clear_connection_registry

from deployment.live.config_mt5_sandbox import SandboxCredentials
from deployment.live.runtime.node_builder import LiveRuntimeConfig, build_sandbox_node


def _drive_until(node, predicate, timeout_s: float = 30.0):
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


def _shutdown(node, loop, run_task) -> None:
    try:
        loop.run_until_complete(node.stop_async())
    finally:
        run_task.cancel()
        try:
            loop.run_until_complete(run_task)
        except asyncio.CancelledError:
            pass


def test_vault_sandbox_node_connects_and_loads_instruments(live):
    """Sandbox node binds FTMO data, loads instruments, reports a (sandbox) account."""
    clear_connection_registry()
    runtime = LiveRuntimeConfig(
        broker="ftmo",
        tickers=("ES", "NQ"),     # -> US500.cash, US100.cash
        warmup_min_bars=0,
        poll_interval_secs=5,
        heartbeat_secs=3600,
    )
    creds = SandboxCredentials.from_env()

    node = build_sandbox_node(runtime, creds, connect=True)  # build() connects to FTMO
    loop = run_task = None
    try:
        assert node.is_built()
        loop, run_task, ok = _drive_until(
            node, lambda: node.is_running() and bool(node.cache.instruments()), timeout_s=40.0
        )
        assert ok, "sandbox node did not reach running + instruments loaded within timeout"

        us500 = node.cache.instrument(InstrumentId.from_str("US500.cash.MT5"))
        assert us500 is not None, "expected US500.cash.MT5 (ES) loaded from FTMO"
        # The SandboxExecutionClient registers a virtual account on the MT5 venue.
        assert len(node.cache.accounts()) >= 1, "no sandbox account reported"

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
