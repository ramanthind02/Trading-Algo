"""MT5-derived market facts for the rollover overlay (the impure shell).

Three things the overlay needs from the live terminal, none of which the Nautilus
``Instrument`` carries:

* **broker wall-clock now** — from a fresh tick (``tick.time`` IS broker local
  time, mislabelled UTC). This is the schedule's clock — never the host clock.
* **per-broker session times** — loaded from
  ``data/mt5_data/{broker}/_symbol_sessions.json`` (written by
  ``build_symbol_sessions --broker``); the reopen time differs per broker/symbol.
* **live swap** — ``swap_long/short`` (points), ``point``, ``trade_contract_size``,
  ``swap_mode`` and ``swap_rollover3days`` from ``mt5.symbol_info``.

Every MT5 call is injectable (``*_fn`` params default to the real ``MetaTrader5``
functions) so the logic is unit-testable on any platform with stubs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from deployment.live.runtime.rollover_schedule import SymbolSession
from execution.rollover_overlay import swap_bps_for_position

_EPOCH = datetime(1970, 1, 1)  # naive; broker tick.time is broker wall-clock as epoch


# ── broker wall-clock ─────────────────────────────────────────────────────────

def _mt5_tick_time(symbol: str):
    import MetaTrader5 as mt5  # local import: module stays importable without MT5

    if not mt5.symbol_select(symbol, True):
        return None
    tick = mt5.symbol_info_tick(symbol)
    return None if tick is None else getattr(tick, "time", None)


def broker_now(*, ref_symbol: str = "EURUSD", tick_time_fn=None) -> datetime | None:
    """Current BROKER wall-clock (naive) from a fresh reference tick, or None.

    ``tick.time`` is the broker's local time encoded as a Unix epoch, so adding it
    to the 1970 epoch yields the broker wall-clock directly — no tz conversion
    (which would re-introduce the offset bug). ``EURUSD`` is liquid 24/5 so its
    last tick is ~now during any trading session.
    """
    t = (tick_time_fn or _mt5_tick_time)(ref_symbol)
    if t is None or int(t) <= 0:
        return None
    return _EPOCH + timedelta(seconds=int(t))


# ── per-broker sessions ───────────────────────────────────────────────────────

def _mt5_data_dir() -> Path:
    from cache.runtime.cache_paths import project_root

    return project_root() / "data" / "mt5_data"


def broker_sessions_path(broker: str, *, root: Path | None = None) -> Path:
    return (root or _mt5_data_dir()) / broker / "_symbol_sessions.json"


def load_broker_sessions(broker: str, *, root: Path | None = None) -> dict[str, SymbolSession]:
    """Load ``{native_symbol -> SymbolSession}`` for ``broker`` (empty if absent)."""
    path = broker_sessions_path(broker, root=root)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, SymbolSession] = {}
    for sym, v in raw.items():
        if not isinstance(v, dict) or "open_hour_broker" not in v:
            continue
        out[sym] = SymbolSession(
            open_h=int(v["open_hour_broker"]),
            open_m=int(v["open_minute_broker"]),
            close_h=int(v["close_hour_broker"]),
            close_m=int(v["close_minute_broker"]),
        )
    return out


# ── live swap ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SwapInfo:
    """The swap facts for one symbol, read live from ``mt5.symbol_info``."""

    symbol: str
    swap_long_points: float
    swap_short_points: float
    point: float
    contract_size: float
    swap_mode: int          # MT5 SYMBOL_SWAP_MODE_*: 1 = POINTS (handled)
    triple_weekday: int     # swap_rollover3days (MT5: Sun=0 … Sat=6)

    @property
    def points_mode(self) -> bool:
        return self.swap_mode == 1


def _mt5_symbol_info(symbol: str):
    import MetaTrader5 as mt5

    mt5.symbol_select(symbol, True)
    return mt5.symbol_info(symbol)


def fetch_swap(native_symbol: str, *, symbol_info_fn=None) -> SwapInfo | None:
    """Read the swap facts for ``native_symbol`` (None if the symbol is unknown)."""
    si = (symbol_info_fn or _mt5_symbol_info)(native_symbol)
    if si is None:
        return None
    return SwapInfo(
        symbol=native_symbol,
        swap_long_points=float(si.swap_long),
        swap_short_points=float(si.swap_short),
        point=float(si.point),
        contract_size=float(si.trade_contract_size),
        swap_mode=int(si.swap_mode),
        triple_weekday=int(si.swap_rollover3days),
    )


def swap_bps_for(info: SwapInfo, *, position: float, mid: float, triple: bool) -> float:
    """Tonight's swap (bps of notional, signed) for the held direction.

    Only POINTS-mode (``swap_mode == 1``) is computed; any other mode (e.g. the
    interest-current mode 3 seen on Darwinex WTI) returns 0.0 — i.e. carry is
    treated as ZERO so the leg is **held**, never flattened on a swap we cannot
    price. The caller logs the skip.
    """
    if not info.points_mode:
        return 0.0
    return swap_bps_for_position(
        position=position,
        mid=mid,
        point=info.point,
        contract_size=info.contract_size,
        swap_long_points=info.swap_long_points,
        swap_short_points=info.swap_short_points,
        triple=triple,
    )


__all__ = [
    "SwapInfo",
    "broker_now",
    "broker_sessions_path",
    "fetch_swap",
    "load_broker_sessions",
    "swap_bps_for",
]
