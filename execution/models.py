"""
Execution data models: OrderIntent, OrderResult, ExecutionConfig.

Frozen dataclasses used to pass order descriptions and results between the
pure rebalancer, the safety layer, the IB executor, and the audit log.

All quantities are ``Decimal`` so fractional-share arithmetic is exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import List, Optional


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    TIMED_OUT = "TIMED_OUT"
    ERROR = "ERROR"


@dataclass(frozen=True)
class OrderIntent:
    """A single ETF order to be placed, produced by the rebalancer."""

    etf: str
    side: OrderSide
    shares: Decimal
    est_price: float
    est_notional: float

    def __post_init__(self) -> None:
        if self.shares <= Decimal("0"):
            raise ValueError(f"OrderIntent shares must be positive; got {self.shares}")
        if self.est_price <= 0:
            raise ValueError(f"OrderIntent est_price must be positive; got {self.est_price}")


@dataclass(frozen=True)
class OrderResult:
    """Terminal outcome of a placed order."""

    intent: OrderIntent
    ib_order_id: int
    status: OrderStatus
    filled_shares: Decimal
    avg_fill_price: Optional[float]
    commission_usd: Optional[float]
    error_code: Optional[int] = None
    error_text: Optional[str] = None


@dataclass(frozen=True)
class ExecutionConfig:
    """Configuration for the execution layer, sourced from the profile JSON.

    All caps are hard ceilings; the rebalancer's dead-band filters are *floors*
    (below which orders are suppressed as noise).
    """

    ib_account_id: str
    max_orders_per_run: int
    min_rebalance_shares: Decimal
    min_rebalance_notional_usd: float
    authorized_telegram_user_ids: List[int]
    approval_timeout_seconds: int
    order_fill_timeout_seconds: int = 60
    market_open_et: str = "09:35"
    market_close_et: str = "15:55"
    allow_any_approver: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "ExecutionConfig":
        return cls(
            ib_account_id=str(data["ib_account_id"]),
            max_orders_per_run=int(data["max_orders_per_run"]),
            min_rebalance_shares=Decimal(str(data["min_rebalance_shares"])),
            min_rebalance_notional_usd=float(data["min_rebalance_notional_usd"]),
            authorized_telegram_user_ids=[int(u) for u in data["authorized_telegram_user_ids"]],
            approval_timeout_seconds=int(data["approval_timeout_seconds"]),
            order_fill_timeout_seconds=int(data.get("order_fill_timeout_seconds", 60)),
            market_open_et=str(data.get("market_open_et", "09:35")),
            market_close_et=str(data.get("market_close_et", "15:55")),
            allow_any_approver=bool(data.get("allow_any_approver", False)),
        )


@dataclass(frozen=True)
class BatchSummary:
    """Summary of a single execution run for audit and Telegram reporting."""

    run_id: str
    intents: List[OrderIntent]
    results: List[OrderResult] = field(default_factory=list)
    approved: bool = False
    aborted_reason: Optional[str] = None
