"""
Typed value objects for the CFD prop-firm MT5 execution flow.

All domain objects in :mod:`execution.mt5_rebalancer`,
:mod:`execution.mt5_trade_executor`, :mod:`execution.mt5_order_safety`, and
:mod:`execution.run_mt5_execution` use the types defined here. These are
deliberately decoupled from the ``MetaTrader5`` Python package so the rest
of the codebase imports cleanly on non-Windows machines (the broker
package is Windows-only).

The MT5 connector layer (``mt5_trade_executor``) is the only module that
imports ``MetaTrader5``; it converts broker dicts/namedtuples to these
typed objects at the boundary.

Conventions
-----------
- ``signed_lots`` / ``signed_volume``: positive = long, negative = short.
- All monetary values are USD unless otherwise noted (the four supported
  CFDs — US100, US500, XAUUSD, XAGUSD — are all USD-quoted).
- Timestamps are timezone-aware UTC.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Sequence


class OrderSide(str, Enum):
    """Direction of a market order. Avoids booleans in the public API."""

    BUY = "buy"
    SELL = "sell"

    @classmethod
    def from_signed(cls, signed_value: float) -> "OrderSide":
        if signed_value > 0:
            return cls.BUY
        if signed_value < 0:
            return cls.SELL
        raise ValueError("Cannot derive OrderSide from zero signed_value")

    def signed(self) -> int:
        return +1 if self is OrderSide.BUY else -1

    def opposite(self) -> "OrderSide":
        return OrderSide.SELL if self is OrderSide.BUY else OrderSide.BUY


class TradeMode(str, Enum):
    """MT5 ``symbol_info.trade_mode`` enum, mapped to a readable subset.

    The MT5 package exposes integer constants. The executor translates them
    to this enum at the boundary so safety gates can pattern-match.
    """

    FULL = "full"           # SYMBOL_TRADE_MODE_FULL
    LONG_ONLY = "long_only"  # SYMBOL_TRADE_MODE_LONGONLY
    SHORT_ONLY = "short_only"  # SYMBOL_TRADE_MODE_SHORTONLY
    CLOSE_ONLY = "close_only"  # SYMBOL_TRADE_MODE_CLOSEONLY
    DISABLED = "disabled"   # SYMBOL_TRADE_MODE_DISABLED


@dataclass(frozen=True)
class MT5AccountInfo:
    """Snapshot of an MT5 account, as returned by ``mt5.account_info()``."""

    login: int
    server: str
    currency: str
    balance: float
    equity: float
    margin: float
    margin_free: float
    margin_level: float
    leverage: int
    trade_allowed: bool
    trade_expert: bool

    def sizing_basis_value(self, basis: str) -> float:
        """Return ``equity`` or ``balance`` depending on configured basis."""
        if basis == "equity":
            return self.equity
        if basis == "balance":
            return self.balance
        raise ValueError(f"Unknown sizing_basis: {basis!r}")


@dataclass(frozen=True)
class MT5SymbolInfo:
    """Snapshot of a tradable MT5 symbol."""

    name: str
    trade_mode: TradeMode
    point: float
    digits: int
    trade_contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    spread: int           # in points
    visible: bool


@dataclass(frozen=True)
class MT5Tick:
    """Latest top-of-book tick for a symbol."""

    symbol: str
    bid: float
    ask: float
    time_utc: datetime

    def mid(self) -> float:
        return (self.bid + self.ask) * 0.5

    def for_side(self, side: OrderSide) -> float:
        """Price the broker will fill a market order at."""
        return self.ask if side is OrderSide.BUY else self.bid


@dataclass(frozen=True)
class MT5Position:
    """A single open ticket from ``mt5.positions_get()``.

    ``signed_volume`` is volume with sign baked in (BUY → +, SELL → −) so
    callers don't need to keep an extra ``side`` field around.
    """

    ticket: int
    symbol: str
    side: OrderSide
    volume: float                 # always > 0 (broker convention)
    open_price: float
    magic: int
    open_time_utc: datetime
    comment: str = ""

    @property
    def signed_volume(self) -> float:
        return self.volume * self.side.signed()


class RebalanceActionKind(str, Enum):
    NO_OP = "no_op"
    OPEN_NEW = "open_new"
    CLOSE_TICKET = "close_ticket"
    ABORT_UNMANAGED = "abort_unmanaged"


@dataclass(frozen=True)
class RebalanceAction:
    """A single ticket-level action to bring an account toward target.

    For ``OPEN_NEW``:    ``side`` and ``volume`` are set; ``ticket`` is None.
    For ``CLOSE_TICKET``: ``ticket`` and ``volume`` are set; ``side`` is the
                          original ticket's side (the executor will send an
                          opposite-side IOC deal with ``position=<ticket>``).
    For ``ABORT_UNMANAGED``: only ``symbol`` and ``reason`` are meaningful.
    For ``NO_OP``: only ``symbol`` and ``reason`` are meaningful.
    """

    kind: RebalanceActionKind
    symbol: str
    side: Optional[OrderSide] = None
    volume: Optional[float] = None
    ticket: Optional[int] = None
    reason: str = ""


@dataclass(frozen=True)
class AccountPlan:
    """Per-account intent built before requesting batch approval."""

    label: str
    login: int
    server: str
    currency: str
    balance: float
    equity: float
    margin_free: float
    actions_by_symbol: Dict[str, List[RebalanceAction]]
    preflight_ok: bool
    preflight_messages: List[str] = field(default_factory=list)

    def has_real_actions(self) -> bool:
        for actions in self.actions_by_symbol.values():
            for a in actions:
                if a.kind in (RebalanceActionKind.OPEN_NEW, RebalanceActionKind.CLOSE_TICKET):
                    return True
        return False


@dataclass(frozen=True)
class OrderResult:
    """Outcome of a single placed action."""

    action: RebalanceAction
    success: bool
    deal_ticket: Optional[int] = None
    retcode: Optional[int] = None
    comment: str = ""


@dataclass(frozen=True)
class AccountExecutionReport:
    """Final per-account outcome surfaced in the post-execution summary."""

    label: str
    login: int
    decision: str                  # ApprovalDecision.value
    plan: Optional[AccountPlan]
    results: List[OrderResult] = field(default_factory=list)
    residual_signed_lots: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def num_successful(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def num_failed(self) -> int:
        return sum(1 for r in self.results if not r.success)


# Re-exportable helpers --------------------------------------------------

# Standard MT5 retcodes we care about (subset). The full enum is large; we
# only encode the ones our retry/abort logic branches on.
TRADE_RETCODE_DONE = 10009
TRADE_RETCODE_REQUOTE = 10004
TRADE_RETCODE_PRICE_OFF = 10021
TRADE_RETCODE_NO_PRICES = 10020
TRADE_RETCODE_TIMEOUT = 10008
TRADE_RETCODE_INVALID_VOLUME = 10014

RETRYABLE_RETCODES: FrozenSet[int] = frozenset({
    TRADE_RETCODE_REQUOTE,
    TRADE_RETCODE_PRICE_OFF,
    TRADE_RETCODE_NO_PRICES,
    TRADE_RETCODE_TIMEOUT,
})
