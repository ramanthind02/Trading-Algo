"""Unit tests for the MT5-derived market facts (broker time, sessions, swap)."""
from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

from deployment.live.runtime.rollover_market import (
    SwapInfo,
    broker_now,
    fetch_swap,
    load_broker_sessions,
    swap_bps_for,
)
from deployment.live.runtime.rollover_schedule import SymbolSession


def test_broker_now_from_tick_is_wall_clock() -> None:
    # tick.time is broker wall-clock as a Unix epoch → adding to 1970 gives it back.
    wall = datetime(2026, 6, 9, 1, 5)
    epoch = int((wall - datetime(1970, 1, 1)).total_seconds())
    got = broker_now(tick_time_fn=lambda _s: epoch)
    assert got == wall


def test_broker_now_none_without_tick() -> None:
    assert broker_now(tick_time_fn=lambda _s: None) is None
    assert broker_now(tick_time_fn=lambda _s: 0) is None


def test_load_broker_sessions(tmp_path) -> None:
    root = tmp_path
    (root / "ftmo").mkdir()
    (root / "ftmo" / "_symbol_sessions.json").write_text(
        json.dumps(
            {
                "US100.cash": {"open_hour_broker": 1, "open_minute_broker": 0,
                               "close_hour_broker": 0, "close_minute_broker": 0},
                "_bad": {"note": "no open_hour_broker → skipped"},
            }
        ),
        encoding="utf-8",
    )
    sessions = load_broker_sessions("ftmo", root=root)
    assert sessions == {"US100.cash": SymbolSession(1, 0, 0, 0)}
    assert load_broker_sessions("darwinex", root=root) == {}  # absent → empty


def _si(swap_long, swap_short, *, point, contract, mode=1, triple=5):
    return SimpleNamespace(
        swap_long=swap_long, swap_short=swap_short, point=point,
        trade_contract_size=contract, swap_mode=mode, swap_rollover3days=triple,
    )


def test_fetch_swap_parses_symbol_info() -> None:
    info = fetch_swap("US100.cash", symbol_info_fn=lambda _s: _si(-619.48, -12.41, point=0.01, contract=1))
    assert info == SwapInfo("US100.cash", -619.48, -12.41, 0.01, 1.0, 1, 5)
    assert info.points_mode is True
    assert fetch_swap("X", symbol_info_fn=lambda _s: None) is None


def test_swap_bps_long_index_is_negative_and_triples() -> None:
    # FTMO US100.cash long: swap_long -619.48 pts, point 0.01, mid ~28900.
    info = fetch_swap("US100.cash", symbol_info_fn=lambda _s: _si(-619.48, -12.41, point=0.01, contract=1, triple=5))
    single = swap_bps_for(info, position=+1.0, mid=28900.0, triple=False)
    triple = swap_bps_for(info, position=+1.0, mid=28900.0, triple=True)
    assert single < 0.0                       # we PAY to hold long
    assert abs(triple - 3 * single) < 1e-9    # ×3 on triple nights
    # magnitude sanity: -619.48 * 0.01 / 28900 * 1e4 ≈ -2.14 bps
    assert -2.2 < single < -2.0


def test_swap_bps_short_index_can_be_positive_carry() -> None:
    # Darwinex NDX short: swap_short +18.79 pts → POSITIVE carry (earn) when short.
    info = fetch_swap("NDX", symbol_info_fn=lambda _s: _si(-45.51, +18.79, point=0.1, contract=10))
    bps_short = swap_bps_for(info, position=-1.0, mid=29000.0, triple=False)
    assert bps_short > 0.0                     # holding the short EARNS swap → hold to collect


def test_swap_bps_long_positive_carry_holds() -> None:
    # FTMO USOIL.cash long: swap_long +36.51 → POSITIVE carry when long.
    info = fetch_swap("USOIL.cash", symbol_info_fn=lambda _s: _si(+36.51, -168.32, point=0.001, contract=100))
    assert swap_bps_for(info, position=+1.0, mid=94.0, triple=False) > 0.0


def test_non_points_swap_mode_returns_zero_carry() -> None:
    # mode 3 (interest) — we can't price it in points → carry 0 → leg is HELD, never flattened.
    info = fetch_swap("XTIUSD", symbol_info_fn=lambda _s: _si(+42.2, -145.3, point=0.01, contract=1000, mode=3))
    assert info.points_mode is False
    assert swap_bps_for(info, position=+1.0, mid=92.0, triple=False) == 0.0
