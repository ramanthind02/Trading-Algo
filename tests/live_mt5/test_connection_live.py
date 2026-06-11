"""Tier 1 (read-only): the REAL MT5Connection lifecycle against FTMO demo.

This is also the live proof of the path-gap fix — MT5Connection.connect()
now binds deterministically to the FTMO terminal via config.path (no
pre-initialize hack), so it reaches CONNECTED on a multi-terminal machine.
No orders are placed.
"""
from __future__ import annotations

import pytest

from mt5connect.connection import MT5Connection, ConnectionState, AccountSnapshot


def test_connect_reaches_connected(live):
    conn = MT5Connection(live.make_config([live.ORDER_SYMBOL]))
    try:
        conn.connect()
        assert conn.state == ConnectionState.CONNECTED
        assert conn.is_connected is True
        assert conn.uptime_seconds() is not None and conn.uptime_seconds() >= 0.0
    finally:
        conn.disconnect()
    assert conn.state == ConnectionState.DISCONNECTED


def test_account_snapshot_is_the_ftmo_demo(live, ftmo_conn):
    snap = ftmo_conn.get_account_info()
    assert isinstance(snap, AccountSnapshot)
    assert snap.login == live.session.login
    assert snap.server == live.session.server
    assert snap.balance > 0.0
    assert snap.currency  # non-empty (USD on FTMO)
    assert snap.leverage > 0


def test_terminal_info_reports_connected(ftmo_conn):
    info = ftmo_conn.get_terminal_info()
    assert isinstance(info, dict) and info
    assert info.get("connected") is True
    # The bound terminal must be the FTMO install, not Darwinex.
    assert "ftmo" in str(info.get("path", "")).lower()


def test_hard_demo_guard_holds_via_adapter(live, ftmo_conn):
    """Independent re-assertion through the live mt5 handle: still DEMO + FTMO."""
    raw = live.mt5.account_info()
    assert int(raw.trade_mode) == int(live.mt5.ACCOUNT_TRADE_MODE_DEMO)
    assert int(raw.login) == live.session.login
    assert str(raw.server) == live.session.server


def test_disconnect_is_idempotent(live):
    conn = MT5Connection(live.make_config([live.ORDER_SYMBOL]))
    conn.connect()
    conn.disconnect()
    conn.disconnect()  # must not raise
    assert conn.state == ConnectionState.DISCONNECTED
