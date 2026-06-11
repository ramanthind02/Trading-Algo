"""Unit tests for the exec-tier guard + node assembly per tier."""
from __future__ import annotations

import asyncio

# Import the REAL MetaTrader5 at collection time. A sibling deployment test
# (test_forecast_server_cache) installs a SimpleNamespace MT5 mock *only if* it is
# absent; importing it here first keeps the vendored mt5connect.constants bound to
# the real ORDER_FILLING_IOC when build_node lazily imports the adapter.
import MetaTrader5  # noqa: F401

import pytest

from deployment.live.credentials import BrokerCredentials, ExecTier
from deployment.live.runtime.node_builder import LiveRuntimeConfig, _strategy_config, build_node
from deployment.live.run_vault_sandbox import _assert_armable


@pytest.fixture
def fresh_event_loop():
    """Give each node build a fresh loop.

    ``TradingNode`` resolves the ambient loop via ``asyncio.get_event_loop()``;
    a prior test that built+disposed a node leaves a CLOSED current loop, which
    breaks the next build. Production runs one node per process, so this only
    matters for test isolation.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield
    if not loop.is_closed():
        loop.close()


def _demo_creds(server: str = "FTMO-Demo") -> BrokerCredentials:
    return BrokerCredentials(broker="ftmo", server=server, login=1, password="x")


# ── guard ─────────────────────────────────────────────────────────────────
def test_guard_sandbox_always_ok() -> None:
    _assert_armable(ExecTier.SANDBOX, _demo_creds(), live_flag=False)  # no raise


def test_guard_demo_requires_live_flag() -> None:
    with pytest.raises(SystemExit, match="Pass --live"):
        _assert_armable(ExecTier.DEMO, _demo_creds(), live_flag=False)


def test_guard_demo_requires_demo_server() -> None:
    with pytest.raises(SystemExit, match="DEMO server"):
        _assert_armable(ExecTier.DEMO, _demo_creds(server="FTMO-Live"), live_flag=True)


def test_guard_demo_ok_with_live_flag_and_demo_server() -> None:
    _assert_armable(ExecTier.DEMO, _demo_creds(), live_flag=True)  # no raise


def test_guard_live_requires_live_flag() -> None:
    with pytest.raises(SystemExit, match="Pass --live"):
        _assert_armable(ExecTier.LIVE, _demo_creds(), live_flag=False)


# ── signal source of truth (shared Darwinex cache across brokers) ───────────
def test_signal_cache_shared_across_brokers() -> None:
    """Every execution broker must read the SAME (Darwinex) signal cache_root."""
    cfgs = {b: _strategy_config(LiveRuntimeConfig(broker=b)) for b in ("ftmo", "fundednext", "darwinex")}
    roots = {c.cache_root for c in cfgs.values()}
    assert len(roots) == 1, f"signal cache_root must be identical across brokers, got {roots}"
    assert "darwinex" in next(iter(roots))
    # Execution stays per-broker even though the signal is shared.
    assert {b: cfgs[b].broker for b in cfgs} == {"ftmo": "ftmo", "fundednext": "fundednext", "darwinex": "darwinex"}


def test_signal_broker_override_repoints_cache() -> None:
    cfg = _strategy_config(LiveRuntimeConfig(broker="ftmo", signal_broker="ftmo"))
    assert "ftmo" in cfg.cache_root and "darwinex" not in cfg.cache_root


# ── node assembly per tier (offline, no connect) ────────────────────────────
@pytest.mark.parametrize("tier", [ExecTier.SANDBOX, ExecTier.DEMO])
def test_build_node_assembles_each_tier(tier: ExecTier, fresh_event_loop) -> None:
    runtime = LiveRuntimeConfig(broker="ftmo", tickers=("ES", "NQ"))
    creds = BrokerCredentials(broker="ftmo", server="FTMO-Demo", login=1, password="placeholder")
    node = build_node(runtime, creds, connect=False, exec_tier=tier)
    try:
        assert node is not None
        assert not node.is_built()  # connect=False
    finally:
        node.dispose()
