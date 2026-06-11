"""ADR-7 scraper broker guard (§3.4).

The MT5 scraper must refuse to write into the Darwinex-declared store
(data/mt5_data) when attached to the wrong broker's terminal.

All tests monkeypatch the ``mt5`` module object on the ``scraper`` module so
no real MT5 terminal is needed.
"""
from __future__ import annotations

import types
from unittest.mock import MagicMock

import pytest

from data_platform.providers.mt5 import scraper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_mt5(
    initialize_ok: bool = True,
    server: str | None = "Darwinex-Live",
) -> types.SimpleNamespace:
    """Return a fake ``mt5`` namespace suitable for monkeypatching.

    ``server=None`` makes account_info() return None (terminal not logged in).
    """
    stub = types.SimpleNamespace()
    stub.initialize = MagicMock(return_value=initialize_ok)
    stub.shutdown = MagicMock()
    stub.last_error = MagicMock(return_value=(0, "ok"))

    if server is None:
        stub.account_info = MagicMock(return_value=None)
    else:
        acct = types.SimpleNamespace(login=12345, server=server)
        stub.account_info = MagicMock(return_value=acct)

    info = types.SimpleNamespace(build=3815)
    stub.terminal_info = MagicMock(return_value=info)
    return stub


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_connect_raises_when_mt5_path_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """MT5_PATH unset → RuntimeError mentioning MT5_PATH; initialize never called."""
    monkeypatch.delenv("MT5_PATH", raising=False)
    stub = _fake_mt5(server="Darwinex-Live")
    monkeypatch.setattr(scraper, "mt5", stub)

    with pytest.raises(RuntimeError, match="MT5_PATH"):
        scraper.connect()

    stub.initialize.assert_not_called()


def test_connect_raises_for_wrong_broker(monkeypatch: pytest.MonkeyPatch) -> None:
    """account_info().server = 'FTMO-Demo' → RuntimeError naming the actual server
    and the expected broker ('darwinex'); mt5.shutdown() must have been called."""
    monkeypatch.setenv("MT5_PATH", r"C:\MT5\ftmo\terminal64.exe")
    stub = _fake_mt5(server="FTMO-Demo")
    monkeypatch.setattr(scraper, "mt5", stub)

    with pytest.raises(RuntimeError) as exc_info:
        scraper.connect()

    msg = str(exc_info.value).lower()
    assert "ftmo-demo" in msg, "error must name the actual server"
    assert "darwinex" in msg, "error must name the expected broker"
    stub.shutdown.assert_called_once()


def test_connect_succeeds_for_darwinex_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """account_info().server = 'Darwinex-Live' → connect() returns True without
    calling shutdown."""
    monkeypatch.setenv("MT5_PATH", r"C:\MT5\darwinex\terminal64.exe")
    stub = _fake_mt5(server="Darwinex-Live")
    monkeypatch.setattr(scraper, "mt5", stub)

    result = scraper.connect()

    assert result is True
    stub.initialize.assert_called_once()
    stub.shutdown.assert_not_called()
