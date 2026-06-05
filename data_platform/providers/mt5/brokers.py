"""Broker-aware MT5 symbol resolution + per-broker data namespace.

We trade MULTIPLE MT5 brokers, each exposing the SAME underlying instrument
under DIFFERENT native symbol names (e.g. Nasdaq-100 is ``NDX`` on Darwinex
but may be ``US100`` / ``NAS100`` on FTMO; S&P-500 is ``SP500`` on Darwinex
but may be ``US500`` on FTMO). FTMO is a different broker from Darwinex, so it
also has a different price history — its data must not collide with Darwinex's.

This module is the single source of truth for:

  resolve(broker, canonical)         canonical ticker -> that broker's symbol
  canonical_for(broker, symbol)      that broker's symbol -> canonical ticker
  broker_data_dir(broker[, symbol])  data/mt5_data/{broker}/{SYMBOL}/ path
  discover_symbols(broker)           STUB — enumerate mt5.symbols_get() on a
                                     SEPARATE terminal to fill FTMO placeholders

Mappings are loaded from ``configs/mt5_brokers.yaml``. Darwinex is seeded from
the existing confirmed mappings; FTMO entries are placeholders (``UNKNOWN``)
until confirmed against a separate FTMO terminal.

SAFETY
------
Importing or calling resolve/canonical_for/broker_data_dir NEVER touches a
terminal. ``discover_symbols`` is a stub that documents — but does NOT perform —
the ``mt5.symbols_get()`` enumeration; it must only be run against a SEPARATE
broker terminal, never the live Darwinex terminal, and is gated behind an
explicit opt-in flag.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Final

import yaml

# Canonical sentinel for an unconfirmed broker symbol.
UNKNOWN: Final[str] = "UNKNOWN"

# The legacy (pre-broker-namespace) data layout was Darwinex-implicit:
#   data/mt5_data/{SYMBOL}/...
# so existing on-disk data is treated as belonging to this broker.
DEFAULT_BROKER: Final[str] = "darwinex"


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    return next(
        (p for p in here.parents if (p / ".git").exists() or (p / "AGENTS.md").exists()),
        here.parents[3],
    )


def _config_path() -> Path:
    return _repo_root() / "configs" / "mt5_brokers.yaml"


@functools.lru_cache(maxsize=1)
def _load_config() -> dict:
    text = _config_path().read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    brokers = data.get("brokers")
    if not isinstance(brokers, dict) or not brokers:
        raise ValueError(f"No 'brokers' mapping found in {_config_path()}")
    return data


def _broker_block(broker: str) -> dict:
    brokers = _load_config()["brokers"]
    key = broker.strip().lower()
    block = brokers.get(key)
    if block is None:
        known = ", ".join(sorted(brokers))
        raise KeyError(f"Unknown MT5 broker {broker!r}. Known brokers: {known}")
    return block


def known_brokers() -> tuple[str, ...]:
    """Return the broker keys defined in the config (sorted)."""
    return tuple(sorted(_load_config()["brokers"]))


def is_confirmed(broker: str) -> bool:
    """True iff the broker's mappings have been confirmed on a live terminal."""
    return bool(_broker_block(broker).get("confirmed", False))


def symbol_map(broker: str) -> dict[str, str]:
    """Raw ``{canonical: broker_symbol}`` map for the broker (may contain UNKNOWN)."""
    return dict(_broker_block(broker).get("symbols") or {})


def resolve(broker: str, canonical: str) -> str:
    """Map a canonical repo ticker to *broker*'s native MT5 symbol.

    Raises ``KeyError`` if the canonical ticker is not mapped for the broker,
    and ``ValueError`` if the mapping is still an unconfirmed placeholder
    (``UNKNOWN``) — we never silently trade a guessed symbol.
    """
    symbols = symbol_map(broker)
    broker_symbol = symbols.get(canonical)
    if broker_symbol is None:
        mapped = ", ".join(sorted(symbols)) or "<none>"
        raise KeyError(
            f"No {broker!r} symbol mapping for canonical ticker {canonical!r}. "
            f"Mapped canonicals: {mapped}"
        )
    if str(broker_symbol).upper() == UNKNOWN:
        raise ValueError(
            f"{broker!r} symbol for canonical {canonical!r} is UNKNOWN "
            f"(placeholder). Confirm it via discover_symbols({broker!r}) against "
            f"a separate {broker} terminal, then fill it into configs/mt5_brokers.yaml."
        )
    return str(broker_symbol)


def canonical_for(broker: str, broker_symbol: str) -> str:
    """Reverse map: *broker*'s native MT5 symbol -> canonical repo ticker.

    Raises ``KeyError`` if the broker symbol is not a confirmed mapping for the
    broker. UNKNOWN placeholders are never matched.
    """
    target = broker_symbol.strip()
    for canonical, sym in symbol_map(broker).items():
        if sym is None or str(sym).upper() == UNKNOWN:
            continue
        if str(sym) == target:
            return canonical
    raise KeyError(
        f"No canonical ticker for {broker!r} symbol {broker_symbol!r}. "
        f"(Unconfirmed placeholders are not matched.)"
    )


def candidates(broker: str, canonical: str) -> tuple[str, ...]:
    """Plausible native names to probe for *canonical* during discovery.

    Used by ``discover_symbols`` to narrow the search; empty if none listed.
    """
    block = _broker_block(broker)
    cand = (block.get("candidates") or {}).get(canonical) or []
    return tuple(str(c) for c in cand)


# ---------------------------------------------------------------------------
# Per-broker data namespace
# ---------------------------------------------------------------------------
#
# Going forward, MT5 data is broker-scoped to stop FTMO data colliding with
# Darwinex data (different broker -> different prices for the "same" symbol):
#
#     data/mt5_data/{broker}/{SYMBOL}/bars_M1/...
#     data/mt5_data/{broker}/{SYMBOL}/ticks/...
#
# Backward-compat: the existing flat layout ``data/mt5_data/{SYMBOL}/`` is
# treated as belonging to DEFAULT_BROKER (darwinex). Existing data is NOT moved
# by this module (non-destructive); see README for the optional migration.

def mt5_data_root() -> Path:
    """Root of the MT5 data store: ``data/mt5_data``."""
    return _repo_root() / "data" / "mt5_data"


def legacy_broker_data_dir(broker_symbol: str | None = None) -> Path:
    """Legacy flat (Darwinex-implicit) path: ``data/mt5_data/{SYMBOL}/``.

    Provided for reading pre-existing Darwinex data that has not been migrated
    into the broker-scoped layout. New writes should use ``broker_data_dir``.
    """
    root = mt5_data_root()
    return root / broker_symbol if broker_symbol else root


def broker_data_dir(broker: str, broker_symbol: str | None = None) -> Path:
    """Broker-scoped data dir: ``data/mt5_data/{broker}/{SYMBOL}/``.

    Pass ``broker_symbol`` for the per-symbol dir, or omit it for the broker
    root. This is the path convention all NEW broker-scoped writes use.
    """
    base = mt5_data_root() / broker.strip().lower()
    return base / broker_symbol if broker_symbol else base


# ---------------------------------------------------------------------------
# Discovery (STUB — does NOT connect to any terminal)
# ---------------------------------------------------------------------------

def discover_symbols(broker: str, *, allow_terminal_connection: bool = False) -> dict[str, str]:
    """STUB: confirm a broker's native symbol names against ITS OWN terminal.

    When implemented, this will:
      1. Attach to the broker's terminal via ``mt5.initialize(...)`` — using a
         SEPARATE terminal instance for that broker (e.g. a dedicated FTMO MT5
         install), NEVER the live Darwinex terminal.
      2. Enumerate the broker's instruments via ``mt5.symbols_get()``.
      3. For each canonical ticker with an UNKNOWN placeholder, match against
         ``candidates(broker, canonical)`` (and a fuzzy fallback) to pick the
         broker's native symbol.
      4. Return ``{canonical: discovered_symbol}`` for a human to review and
         paste into ``configs/mt5_brokers.yaml`` (we do NOT auto-write the
         config — confirmation is a manual, audited step).

    SAFETY: this stub deliberately does NOT call ``mt5`` so importing/using the
    module never touches a terminal. It raises unless explicitly opted in, and
    even then only documents the prerequisite — it performs no connection.

    Prerequisite to actually run discovery: a SEPARATE terminal for *broker*
    must be running and logged in (creds from .env, e.g. FTMO_DEMO_* for FTMO).
    The live Darwinex terminal must NOT be disturbed.
    """
    if not allow_terminal_connection:
        raise NotImplementedError(
            f"discover_symbols({broker!r}) is a stub and will NOT connect to a "
            f"terminal. To implement: run against a SEPARATE {broker} terminal "
            f"(NOT the live Darwinex terminal), enumerate mt5.symbols_get(), match "
            f"against candidates(), and review results before editing "
            f"configs/mt5_brokers.yaml. Pass allow_terminal_connection=True only "
            f"once that separate terminal is connected and this body is implemented."
        )
    raise NotImplementedError(
        "discover_symbols terminal-enumeration body is not implemented yet. "
        "Implement mt5.symbols_get() matching here once a separate broker terminal "
        "is available and authorized."
    )
