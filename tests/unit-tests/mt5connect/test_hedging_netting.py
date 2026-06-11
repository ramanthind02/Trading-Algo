"""Unit tests for the no-hedge execution fixes in the vendored MT5 adapter.

Covers (no MT5 terminal — ``mt5`` is faked):
  * ``MT5LiveExecutionClient._submit_market_netting`` — the hedging-safe close/reduce
    logic that REDUCES opposing magic tickets (``position=ticket``) instead of opening
    an opposing short, and opens a residual only on a true reversal.
  * ``constants.normalize_symbol`` — the ``.cash`` suffix fix (US500.cash → US500).

These exercise the exact decomposition the live rollover EXIT relies on, so a
regression that re-introduces hedging would fail here without needing a broker.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from deployment.live.config_mt5_sandbox import vendored_adapter_path

sys.path.insert(0, str(vendored_adapter_path()))

from nautilus_trader.model.enums import OrderSide, TimeInForce  # noqa: E402
import mt5connect.execution as E  # noqa: E402
from mt5connect.constants import normalize_symbol  # noqa: E402

LONG = 0   # mt5.POSITION_TYPE_BUY
SHORT = 1  # mt5.POSITION_TYPE_SELL


class _Pos:
    def __init__(self, ticket, volume, type_, magic=510):
        self.ticket, self.volume, self.type, self.magic = ticket, volume, type_, magic


class _Tick:
    bid, ask = 100.0, 100.2


class _Res:
    def __init__(self, order, retcode):
        self.order, self.retcode = order, retcode


def _fake_mt5(positions, *, fail_on=None):
    """A stand-in MetaTrader5 module exposing only what the helper touches."""
    sends: list[dict] = []
    n = {"i": 0}

    def order_send(request):
        n["i"] += 1
        sends.append(dict(request))
        rc = 10018 if (fail_on is not None and n["i"] == fail_on) else 10009  # 10018=market closed
        return _Res(order=9000 + n["i"], retcode=rc)

    fake = SimpleNamespace(
        POSITION_TYPE_BUY=LONG, POSITION_TYPE_SELL=SHORT,
        ORDER_TYPE_BUY=0, ORDER_TYPE_SELL=1,
        TRADE_ACTION_DEAL=1,
        TRADE_RETCODE_DONE=10009, TRADE_RETCODE_DONE_PARTIAL=10010,
        ORDER_TIME_GTC=0, ORDER_TIME_DAY=1, ORDER_TIME_SPECIFIED=2,
        positions_get=lambda symbol=None: list(positions),
        symbol_info_tick=lambda symbol: _Tick(),
        order_send=order_send,
    )
    return fake, sends


def _client(monkeypatch, fake):
    monkeypatch.setattr(E, "mt5", fake)
    rejects: list[str] = []
    self_ = SimpleNamespace(
        _config=SimpleNamespace(magic_number=510),
        _ticket_to_client_order_id={},
        _log=SimpleNamespace(info=lambda *a, **k: None),
        _generate_order_rejected=lambda order, reason: rejects.append(reason),
    )
    return self_, rejects


def _order(side, qty):
    return SimpleNamespace(side=side, quantity=qty, client_order_id="O-1",
                           time_in_force=TimeInForce.GTC)


def _run(self_, order, symbol="XYZ"):
    return E.MT5LiveExecutionClient._submit_market_netting(self_, order, symbol)


def test_pure_open_no_opposing_positions(monkeypatch):
    fake, sends = _fake_mt5([])
    self_, rejects = _client(monkeypatch, fake)
    ticket = _run(self_, _order(OrderSide.BUY, 0.02))
    assert not rejects and ticket is not None
    assert len(sends) == 1
    assert "position" not in sends[0]            # a pure open never targets a ticket
    assert sends[0]["type"] == fake.ORDER_TYPE_BUY
    assert sends[0]["volume"] == pytest.approx(0.02)


def test_full_close_reduces_not_hedges(monkeypatch):
    fake, sends = _fake_mt5([_Pos(111, 0.02, LONG)])
    self_, rejects = _client(monkeypatch, fake)
    ticket = _run(self_, _order(OrderSide.SELL, 0.02))
    assert not rejects and ticket is not None
    assert len(sends) == 1
    assert sends[0]["position"] == 111           # closes the long ticket — NO opposing short
    assert sends[0]["type"] == fake.ORDER_TYPE_SELL
    assert sends[0]["volume"] == pytest.approx(0.02)


def test_partial_reduce(monkeypatch):
    fake, sends = _fake_mt5([_Pos(111, 0.02, LONG)])
    self_, rejects = _client(monkeypatch, fake)
    _run(self_, _order(OrderSide.SELL, 0.01))
    assert len(sends) == 1
    assert sends[0]["position"] == 111
    assert sends[0]["volume"] == pytest.approx(0.01)


def test_reversal_closes_then_opens_residual(monkeypatch):
    fake, sends = _fake_mt5([_Pos(111, 0.02, LONG)])
    self_, rejects = _client(monkeypatch, fake)
    _run(self_, _order(OrderSide.SELL, 0.03))
    assert len(sends) == 2
    assert sends[0]["position"] == 111 and sends[0]["volume"] == pytest.approx(0.02)
    assert "position" not in sends[1] and sends[1]["volume"] == pytest.approx(0.01)


def test_multi_ticket_fifo(monkeypatch):
    # Provided out of order — helper must close the lowest ticket first.
    fake, sends = _fake_mt5([_Pos(222, 0.02, LONG), _Pos(111, 0.02, LONG)])
    self_, rejects = _client(monkeypatch, fake)
    _run(self_, _order(OrderSide.SELL, 0.03))
    assert len(sends) == 2
    assert sends[0]["position"] == 111
    assert sends[1]["position"] == 222 and sends[1]["volume"] == pytest.approx(0.01)


def test_buy_reduces_existing_shorts(monkeypatch):
    fake, sends = _fake_mt5([_Pos(111, 0.02, SHORT)])
    self_, rejects = _client(monkeypatch, fake)
    _run(self_, _order(OrderSide.BUY, 0.02))
    assert len(sends) == 1
    assert sends[0]["position"] == 111           # a BUY closes the short ticket
    assert sends[0]["type"] == fake.ORDER_TYPE_BUY


def test_close_rejection_returns_none(monkeypatch):
    fake, sends = _fake_mt5([_Pos(111, 0.02, LONG)], fail_on=1)
    self_, rejects = _client(monkeypatch, fake)
    ticket = _run(self_, _order(OrderSide.SELL, 0.02))
    assert ticket is None
    assert rejects and "10018" in rejects[0]


def test_fills_attributed_to_order(monkeypatch):
    fake, sends = _fake_mt5([_Pos(111, 0.02, LONG)])
    self_, rejects = _client(monkeypatch, fake)
    ticket = _run(self_, _order(OrderSide.SELL, 0.02))
    # each sub-deal's broker order id maps back to the client_order_id for fill attribution
    assert self_._ticket_to_client_order_id.get(ticket) == "O-1"


def test_same_direction_add_opens_new_ticket(monkeypatch):
    # Holding a long, a further BUY has no opposing (short) tickets → opens a new long
    # ticket (same-direction add is NOT a hedge; net stays one-directional).
    fake, sends = _fake_mt5([_Pos(111, 0.02, LONG)])
    self_, rejects = _client(monkeypatch, fake)
    _run(self_, _order(OrderSide.BUY, 0.01))
    assert len(sends) == 1
    assert "position" not in sends[0] and sends[0]["volume"] == pytest.approx(0.01)


def test_normalize_cash_suffix():
    assert normalize_symbol("US500.cash") == "US500"
    assert normalize_symbol("US100.cash") == "US100"
    assert normalize_symbol("USOIL.cash") == "USOIL"
    assert normalize_symbol("XAUUSD") == "XAUUSD"     # unaffected
    assert normalize_symbol("EURUSDm") == "EURUSD"    # other suffixes still work
