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
  asset_class(broker, sym)           fx | index | metal | energy | crypto
  session(broker, asset_class)       per-asset-class market hours (broker tz)
  is_market_open(broker, canonical)  pure schedule check at a given UTC instant
  execution_rules / risk_rules       per-broker order mechanics + prop-firm limits
  terminal_path(broker)              that broker's terminal64.exe (-> MT5Config.path)

Each broker differs (symbol names, sessions, filling mode, hedging vs netting,
and — for prop firms — daily/total-loss + profit-target rules), so all of this is
declared per broker in ``configs/mt5_brokers.yaml`` and surfaced here as typed,
immutable accessors. Session/rule values tagged ``# APPROX`` in the YAML are
best-known starting points (calibrate later against ``_symbol_sessions.json``).

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
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Final, Optional
from zoneinfo import ZoneInfo

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


# ===========================================================================
# Per-broker semantics: timezone, asset class, market hours, rules
#
# Everything below is PURE config access — no terminal contact. Enums + frozen
# dataclasses keep the public surface immutable and typed (repo convention).
# ===========================================================================


class AssetClass(str, Enum):
    """Repo-side asset-class tag for an instrument (broker-independent)."""
    FX = "fx"
    INDEX = "index"
    METAL = "metal"
    ENERGY = "energy"
    CRYPTO = "crypto"
    OTHER = "other"


class AccountMode(str, Enum):
    """MT5 position accounting model for the account."""
    HEDGING = "hedging"   # opposite-side deal opens a separate position (close via position=ticket)
    NETTING = "netting"   # opposite-side deal nets the existing position


class FillingMode(str, Enum):
    """MT5 order filling policy (maps to mt5.ORDER_FILLING_*)."""
    IOC = "IOC"
    FOK = "FOK"
    RETURN = "RETURN"


# MT5 brokers run their server clock in EET/EEST. ``Europe/Athens`` observes
# exactly that DST schedule, so it is the correct IANA zone for "EET" labels.
_TZ_LABEL_TO_IANA: Final[dict[str, str]] = {
    "EET": "Europe/Athens",
    "EEST": "Europe/Athens",
    "UTC": "UTC",
    "GMT": "UTC",
}

_MINUTES_PER_DAY: Final[int] = 24 * 60


@dataclass(frozen=True)
class Session:
    """Weekly market hours for one asset class, in the broker's server tz.

    Times are stored as minutes-from-midnight (0..1440; 1440 == "24:00" /
    end of day) so open/close/break checks are pure integer math and the
    midnight-wrapping rollover break is handled correctly.
    """
    asset_class: AssetClass
    weekdays: frozenset[int]           # Monday=0 .. Sunday=6
    open_minute: int
    close_minute: int
    break_minutes: Optional[tuple[int, int]]
    tz: str

    def contains(self, weekday: int, minute_of_day: int) -> bool:
        """True if ``minute_of_day`` on ``weekday`` is inside the session."""
        if weekday not in self.weekdays:
            return False
        if not (self.open_minute <= minute_of_day < self.close_minute):
            return False
        if self.break_minutes is not None:
            b0, b1 = self.break_minutes
            in_break = (b0 <= minute_of_day < b1) if b0 <= b1 else (
                minute_of_day >= b0 or minute_of_day < b1
            )
            if in_break:
                return False
        return True


@dataclass(frozen=True)
class ExecutionRules:
    """How orders are placed on this broker."""
    account_mode: AccountMode
    filling_mode: FillingMode
    max_deviation_points: int
    default_lot_min: float
    default_lot_step: float
    weekend_holding: bool
    magic_number: int


@dataclass(frozen=True)
class RiskRules:
    """Prop-firm evaluation limits. All optional; ``None`` field == not set."""
    max_daily_loss_pct: Optional[float]
    max_total_loss_pct: Optional[float]
    profit_target_pct: Optional[float]
    min_trading_days: Optional[int]
    max_leverage: Optional[int]
    news_trading_restricted: bool


@dataclass(frozen=True)
class BrokerRules:
    execution: ExecutionRules
    risk: Optional[RiskRules]   # None for live-retail brokers (no prop limits)


# ---------------------------------------------------------------------------
# Parsing helpers (YAML string -> typed value)
# ---------------------------------------------------------------------------

def _parse_weekdays(spec) -> frozenset[int]:
    """``"0-4"`` / ``"0-6"`` / ``"0,2,4"`` -> {ints} (Mon=0 .. Sun=6)."""
    out: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    if not out or any(d < 0 or d > 6 for d in out):
        raise ValueError(f"Invalid weekdays spec {spec!r} (expected Mon=0..Sun=6).")
    return frozenset(out)


def _hhmm_to_minute(s) -> int:
    """``"HH:MM"`` -> minutes-from-midnight (0..1440; ``"24:00"`` -> 1440)."""
    h, m = str(s).strip().split(":")
    val = int(h) * 60 + int(m)
    if not (0 <= val <= _MINUTES_PER_DAY):
        raise ValueError(f"Time {s!r} out of range 00:00..24:00.")
    return val


def _parse_break(spec) -> Optional[tuple[int, int]]:
    if not spec:
        return None
    a, b = str(spec).split("-", 1)
    return (_hhmm_to_minute(a), _hhmm_to_minute(b))


def _build_session(asset_class: str, block: dict) -> Session:
    return Session(
        asset_class=AssetClass(str(asset_class).lower()),
        weekdays=_parse_weekdays(block["weekdays"]),
        open_minute=_hhmm_to_minute(block["open"]),
        close_minute=_hhmm_to_minute(block["close"]),
        break_minutes=_parse_break(block.get("daily_break")),
        tz="",  # filled by sessions() with the broker tz label
    )


# ---------------------------------------------------------------------------
# Timezone
# ---------------------------------------------------------------------------

def broker_timezone(broker: str) -> str:
    """The broker server timezone *label* (e.g. ``"EET"``). Defaults to UTC."""
    return str(_broker_block(broker).get("timezone") or "UTC")


def broker_tzinfo(broker: str) -> ZoneInfo:
    """The broker server timezone as a ``ZoneInfo`` (EET/EEST -> Europe/Athens)."""
    label = broker_timezone(broker)
    return ZoneInfo(_TZ_LABEL_TO_IANA.get(label.upper(), label))


# ---------------------------------------------------------------------------
# Asset class
# ---------------------------------------------------------------------------

def asset_classes(broker: str) -> dict[str, AssetClass]:
    """Raw ``{canonical: AssetClass}`` map for the broker."""
    raw = _broker_block(broker).get("asset_classes") or {}
    return {k: AssetClass(str(v).lower()) for k, v in raw.items()}


def asset_class(broker: str, canonical_or_symbol: str) -> AssetClass:
    """Asset class for a canonical ticker OR that broker's native symbol.

    Returns ``AssetClass.OTHER`` when the instrument is not tagged for the broker.
    """
    table = asset_classes(broker)
    key = canonical_or_symbol.strip()
    if key in table:
        return table[key]
    # Maybe it's a native broker symbol — map back to canonical first.
    try:
        canon = canonical_for(broker, key)
    except KeyError:
        canon = None
    if canon is not None and canon in table:
        return table[canon]
    return AssetClass.OTHER


# ---------------------------------------------------------------------------
# Sessions / market hours
# ---------------------------------------------------------------------------

def sessions(broker: str) -> dict[AssetClass, Session]:
    """All declared sessions for the broker, keyed by asset class (broker tz)."""
    raw = _broker_block(broker).get("sessions") or {}
    tz = broker_timezone(broker)
    out: dict[AssetClass, Session] = {}
    for ac_str, block in raw.items():
        sess = _build_session(ac_str, dict(block))
        out[sess.asset_class] = Session(
            asset_class=sess.asset_class,
            weekdays=sess.weekdays,
            open_minute=sess.open_minute,
            close_minute=sess.close_minute,
            break_minutes=sess.break_minutes,
            tz=tz,
        )
    return out


def session(broker: str, asset_class_: AssetClass | str) -> Session:
    """The session for one asset class. Raises ``KeyError`` if not declared."""
    ac = asset_class_ if isinstance(asset_class_, AssetClass) else AssetClass(str(asset_class_).lower())
    sess = sessions(broker).get(ac)
    if sess is None:
        have = ", ".join(sorted(a.value for a in sessions(broker))) or "<none>"
        raise KeyError(f"No {broker!r} session for asset class {ac.value!r}. Declared: {have}")
    return sess


def is_market_open(broker: str, canonical: str, at: Optional[datetime] = None) -> bool:
    """Pure schedule check: is *canonical* tradeable on *broker* at instant *at*?

    ``at`` defaults to now (UTC); naive datetimes are assumed UTC. Uses the
    declarative ``sessions`` block converted to the broker server tz. This is a
    SCHEDULE check only — it does not know about ad-hoc holidays/early closes, so
    a live tick-freshness probe remains the ground truth for actual tradeability.
    Returns ``False`` for instruments with no declared session/asset class.
    """
    moment = at or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    ac = asset_class(broker, canonical)
    try:
        sess = session(broker, ac)
    except KeyError:
        return False
    local = moment.astimezone(broker_tzinfo(broker))
    return sess.contains(local.weekday(), local.hour * 60 + local.minute)


def inferred_session_hours(symbol: str) -> Optional[dict]:
    """Per-symbol empirical session hours (NY tz) from ``_symbol_sessions.json``.

    Reference/refinement only — ``is_market_open`` uses the declarative
    ``sessions`` block. Returns ``None`` when the JSON or symbol is absent.
    Produced by ``data_platform.providers.mt5.build_symbol_sessions``.
    """
    path = mt5_data_root() / "_symbol_sessions.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get(symbol)


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def _optf(v) -> Optional[float]:
    return None if v is None else float(v)


def _opti(v) -> Optional[int]:
    return None if v is None else int(v)


def execution_rules(broker: str) -> ExecutionRules:
    """Order-placement rules for the broker (with sane defaults)."""
    blk = (_broker_block(broker).get("rules") or {}).get("execution") or {}
    return ExecutionRules(
        account_mode=AccountMode(str(blk.get("account_mode", "hedging")).lower()),
        filling_mode=FillingMode(str(blk.get("filling_mode", "IOC")).upper()),
        max_deviation_points=int(blk.get("max_deviation_points", 20)),
        default_lot_min=float(blk.get("default_lot_min", 0.01)),
        default_lot_step=float(blk.get("default_lot_step", 0.01)),
        weekend_holding=bool(blk.get("weekend_holding", False)),
        magic_number=int(blk.get("magic_number", 510)),
    )


def risk_rules(broker: str) -> Optional[RiskRules]:
    """Prop-firm risk limits, or ``None`` for live-retail brokers (no limits)."""
    blk = (_broker_block(broker).get("rules") or {}).get("risk")
    if not blk:  # None or empty mapping
        return None
    return RiskRules(
        max_daily_loss_pct=_optf(blk.get("max_daily_loss_pct")),
        max_total_loss_pct=_optf(blk.get("max_total_loss_pct")),
        profit_target_pct=_optf(blk.get("profit_target_pct")),
        min_trading_days=_opti(blk.get("min_trading_days")),
        max_leverage=_opti(blk.get("max_leverage")),
        news_trading_restricted=bool(blk.get("news_trading_restricted", False)),
    )


def rules(broker: str) -> BrokerRules:
    """Combined execution + risk rules for the broker."""
    return BrokerRules(execution=execution_rules(broker), risk=risk_rules(broker))


def terminal_path(broker: str) -> Optional[str]:
    """The broker's ``terminal64.exe`` path (for ``MT5Config.path``), or None."""
    p = _broker_block(broker).get("terminal_path")
    return str(p) if p else None


# ---------------------------------------------------------------------------
# Validation + CLI summary (no terminal contact)
# ---------------------------------------------------------------------------

def validate_config() -> None:
    """Assert every broker block is well-formed. Raises on the first problem."""
    for b in known_brokers():
        broker_tzinfo(b)            # tz label resolvable
        asset_classes(b)           # every value is a valid AssetClass
        for ac, sess in sessions(b).items():
            assert isinstance(ac, AssetClass)
            assert 0 <= sess.open_minute <= _MINUTES_PER_DAY
            assert 0 <= sess.close_minute <= _MINUTES_PER_DAY
            assert sess.open_minute < sess.close_minute, f"{b}/{ac.value}: open >= close"
        execution_rules(b)         # enums + numbers parse
        risk_rules(b)


def _print_summary() -> None:
    print(f"MT5 broker config: {_config_path()}")
    for b in known_brokers():
        blk = _broker_block(b)
        er = execution_rules(b)
        rr = risk_rules(b)
        sess_keys = ", ".join(sorted(a.value for a in sessions(b))) or "<none>"
        print(f"\n[{b}] {blk.get('display_name', b)}  "
              f"confirmed={is_confirmed(b)}  tz={broker_timezone(b)}")
        print(f"  symbols={len(symbol_map(b))}  asset_classes={len(asset_classes(b))}  "
              f"sessions=[{sess_keys}]")
        print(f"  exec: mode={er.account_mode.value} filling={er.filling_mode.value} "
              f"weekend_hold={er.weekend_holding} magic={er.magic_number}")
        if rr is None:
            print("  risk: none (live retail)")
        else:
            print(f"  risk: daily={rr.max_daily_loss_pct}% total={rr.max_total_loss_pct}% "
                  f"target={rr.profit_target_pct}% min_days={rr.min_trading_days} "
                  f"max_lev={rr.max_leverage}")


if __name__ == "__main__":
    validate_config()
    _print_summary()
    print("\nvalidate_config: OK")
