from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import pandas as pd

from prop_firms.base.enums import (
    AccountPhase,
    AccountStatus,
    BreachReason,
    EventType,
    PayoutPolicy,
    ResetPolicy,
)


@dataclass(frozen=True)
class ContractLimit:
    """Maximum allowed mini and micro contracts."""

    mini_contracts: int
    micro_contracts: int

    def __post_init__(self) -> None:
        if self.mini_contracts < 0:
            raise ValueError("mini_contracts must be >= 0")
        if self.micro_contracts < 0:
            raise ValueError("micro_contracts must be >= 0")

    @property
    def mini_equivalent_limit(self) -> float:
        return max(float(self.mini_contracts), float(self.micro_contracts) / 10.0)


@dataclass(frozen=True)
class AccountFees:
    """User-editable fees kept in config."""

    challenge_fee: float
    reset_fee: float
    activation_fee: float = 0.0

    def __post_init__(self) -> None:
        if self.challenge_fee < 0.0:
            raise ValueError("challenge_fee must be >= 0")
        if self.reset_fee < 0.0:
            raise ValueError("reset_fee must be >= 0")
        if self.activation_fee < 0.0:
            raise ValueError("activation_fee must be >= 0")


@dataclass(frozen=True)
class ConsistencyRule:
    """Largest-day profit rule used by Lucid evaluation accounts."""

    max_ratio: float
    cushion_multiplier: float = 1.04
    inclusive_limit: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.max_ratio <= 1.0:
            raise ValueError("max_ratio must be in (0, 1]")
        if self.cushion_multiplier <= 0.0:
            raise ValueError("cushion_multiplier must be > 0")

    def allowed_largest_day_profit(self, total_profit: float) -> float:
        positive_profit = max(total_profit, 0.0)
        return positive_profit * self.max_ratio * self.cushion_multiplier

    def is_satisfied(self, largest_day_profit: float, total_profit: float) -> bool:
        allowed_profit = self.allowed_largest_day_profit(total_profit)
        return (
            largest_day_profit <= allowed_profit
            if self.inclusive_limit
            else largest_day_profit < allowed_profit
        )


@dataclass(frozen=True)
class EodTrailingDrawdownRule:
    """End-of-day trailing max-loss rule."""

    loss_limit_amount: float
    initial_trail_balance: float
    locked_mll_balance: float

    def __post_init__(self) -> None:
        if self.loss_limit_amount <= 0.0:
            raise ValueError("loss_limit_amount must be > 0")
        if self.initial_trail_balance <= 0.0:
            raise ValueError("initial_trail_balance must be > 0")
        if self.locked_mll_balance <= 0.0:
            raise ValueError("locked_mll_balance must be > 0")
        if self.locked_mll_balance > self.initial_trail_balance:
            raise ValueError(
                "locked_mll_balance must be <= initial_trail_balance"
            )


@dataclass(frozen=True)
class ProfitDayRule:
    """Minimum number of profitable days in a payout cycle."""

    required_days: int
    minimum_profit_amount: float

    def __post_init__(self) -> None:
        if self.required_days < 1:
            raise ValueError("required_days must be >= 1")
        if self.minimum_profit_amount <= 0.0:
            raise ValueError("minimum_profit_amount must be > 0")


@dataclass(frozen=True)
class PayoutRule:
    """Funded-account payout constraints."""

    trader_profit_split: float
    min_request_amount: float
    max_profit_share: float
    max_payout_amount: float
    max_requests_per_account: int
    profit_day_rule: ProfitDayRule
    protected_balance: float | None = None
    consistency_rule: ConsistencyRule | None = None
    payout_cap_schedule: tuple[float, ...] = tuple()

    def __post_init__(self) -> None:
        if not 0.0 < self.trader_profit_split <= 1.0:
            raise ValueError("trader_profit_split must be in (0, 1]")
        if self.min_request_amount <= 0.0:
            raise ValueError("min_request_amount must be > 0")
        if not 0.0 < self.max_profit_share <= 1.0:
            raise ValueError("max_profit_share must be in (0, 1]")
        if self.max_payout_amount <= 0.0:
            raise ValueError("max_payout_amount must be > 0")
        if self.max_requests_per_account < 1:
            raise ValueError("max_requests_per_account must be >= 1")
        if self.protected_balance is not None and self.protected_balance <= 0.0:
            raise ValueError("protected_balance must be > 0 when set")
        if any(amount <= 0.0 for amount in self.payout_cap_schedule):
            raise ValueError("payout_cap_schedule entries must be > 0")

    def payout_cap_for_request(self, payout_number: int) -> float:
        if payout_number < 1:
            raise ValueError("payout_number must be >= 1")
        if payout_number <= len(self.payout_cap_schedule):
            return self.payout_cap_schedule[payout_number - 1]
        return self.max_payout_amount


@dataclass(frozen=True)
class ScalingTier:
    """Funded-account contract tier keyed off current simulated profits."""

    min_profit: float
    max_profit: float | None
    contract_limit: ContractLimit
    daily_loss_limit: float | None = None

    def __post_init__(self) -> None:
        if self.min_profit < 0.0:
            raise ValueError("min_profit must be >= 0")
        if self.max_profit is not None and self.max_profit < self.min_profit:
            raise ValueError("max_profit must be >= min_profit when set")
        if self.daily_loss_limit is not None and self.daily_loss_limit <= 0.0:
            raise ValueError("daily_loss_limit must be > 0 when set")

    def matches(self, profit: float) -> bool:
        above_floor = profit >= self.min_profit
        below_ceiling = self.max_profit is None or profit <= self.max_profit
        return above_floor and below_ceiling


@dataclass(frozen=True)
class EvaluationRules:
    """Rules applied during the evaluation phase."""

    profit_target_amount: float
    drawdown_rule: EodTrailingDrawdownRule
    max_contract_limit: ContractLimit
    consistency_rule: ConsistencyRule | None = None
    minimum_trading_days: int = 0
    evaluation_expiry_calendar_days: int | None = None
    daily_loss_limit: float | None = None
    max_leverage: float | None = None

    def __post_init__(self) -> None:
        if self.profit_target_amount <= 0.0:
            raise ValueError("profit_target_amount must be > 0")
        if self.minimum_trading_days < 0:
            raise ValueError("minimum_trading_days must be >= 0")
        if (
            self.evaluation_expiry_calendar_days is not None
            and self.evaluation_expiry_calendar_days < 1
        ):
            raise ValueError("evaluation_expiry_calendar_days must be >= 1 when set")
        if self.daily_loss_limit is not None and self.daily_loss_limit <= 0.0:
            raise ValueError("daily_loss_limit must be > 0 when set")
        if self.max_leverage is not None and self.max_leverage <= 0.0:
            raise ValueError("max_leverage must be > 0 when set")


@dataclass(frozen=True)
class FundedRules:
    """Rules applied after the evaluation passes."""

    drawdown_rule: EodTrailingDrawdownRule
    max_contract_limit: ContractLimit
    payout_rule: PayoutRule
    scaling_plan: tuple[ScalingTier, ...]
    daily_loss_limit: float | None = None
    max_leverage: float | None = None

    def __post_init__(self) -> None:
        if not self.scaling_plan:
            raise ValueError("scaling_plan must be non-empty")
        if self.daily_loss_limit is not None and self.daily_loss_limit <= 0.0:
            raise ValueError("daily_loss_limit must be > 0 when set")
        if self.max_leverage is not None and self.max_leverage <= 0.0:
            raise ValueError("max_leverage must be > 0 when set")


@dataclass(frozen=True)
class AccountDefinition:
    """Full provider/account configuration loaded from JSON."""

    provider_name: str
    program_name: str
    account_code: str
    account_size: float
    fees: AccountFees
    evaluation: EvaluationRules
    funded: FundedRules
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.account_size <= 0.0:
            raise ValueError("account_size must be > 0")


@dataclass(frozen=True)
class DayInput:
    """Single aligned daily input row for the simulator."""

    trading_day: pd.Timestamp
    daily_return: float
    mini_contracts: int = 0
    micro_contracts: int = 0
    notional_exposure: float | None = None


@dataclass(frozen=True)
class SimulationRequest:
    """Public request object for a prop-firm simulation."""

    account_code: str
    returns: pd.Series
    held_contracts: pd.DataFrame | None = None
    payout_policy: PayoutPolicy = PayoutPolicy.AUTO_MAX
    reset_policy: ResetPolicy = ResetPolicy.NO_RESETS
    payout_amount: float | None = None
    max_resets: int = 0
    starting_balance_override: float | None = None

    def __post_init__(self) -> None:
        if self.max_resets < 0:
            raise ValueError("max_resets must be >= 0")
        if self.payout_policy == PayoutPolicy.FIXED_AMOUNT:
            if self.payout_amount is None or self.payout_amount <= 0.0:
                raise ValueError(
                    "payout_amount must be set and > 0 for FIXED_AMOUNT policy"
                )
        if self.starting_balance_override is not None and self.starting_balance_override <= 0.0:
            raise ValueError("starting_balance_override must be > 0 when set")


@dataclass(frozen=True)
class DailySnapshot:
    """Daily timeline row emitted by the simulator."""

    trading_day: pd.Timestamp
    phase: AccountPhase
    status: AccountStatus
    balance: float
    net_profit: float
    daily_return: float
    closing_high_balance: float
    max_loss_floor: float
    profit_target_balance: float
    largest_daily_profit: float
    consistency_ratio: float | None
    consistency_cap_ratio: float | None
    payout_cycle_profit: float
    payout_cycle_largest_profit: float
    payout_profit_days: int
    payouts_taken: int
    daily_loss_limit: float | None
    daily_loss_limit_hit: int
    scale_mini_limit: int
    scale_micro_limit: int
    mini_contracts: int
    micro_contracts: int
    mini_equivalent_contracts: float
    leverage: float | None
    contract_limit_breached: int
    leverage_limit_breached: int

    def to_record(self) -> dict[str, object]:
        return {
            "trading_day": self.trading_day,
            "phase": self.phase.value,
            "status": self.status.value,
            "balance": self.balance,
            "net_profit": self.net_profit,
            "daily_return": self.daily_return,
            "closing_high_balance": self.closing_high_balance,
            "max_loss_floor": self.max_loss_floor,
            "profit_target_balance": self.profit_target_balance,
            "largest_daily_profit": self.largest_daily_profit,
            "consistency_ratio": self.consistency_ratio,
            "consistency_cap_ratio": self.consistency_cap_ratio,
            "payout_cycle_profit": self.payout_cycle_profit,
            "payout_cycle_largest_profit": self.payout_cycle_largest_profit,
            "payout_profit_days": self.payout_profit_days,
            "payouts_taken": self.payouts_taken,
            "daily_loss_limit": self.daily_loss_limit,
            "daily_loss_limit_hit": self.daily_loss_limit_hit,
            "scale_mini_limit": self.scale_mini_limit,
            "scale_micro_limit": self.scale_micro_limit,
            "mini_contracts": self.mini_contracts,
            "micro_contracts": self.micro_contracts,
            "mini_equivalent_contracts": self.mini_equivalent_contracts,
            "leverage": self.leverage,
            "contract_limit_breached": self.contract_limit_breached,
            "leverage_limit_breached": self.leverage_limit_breached,
        }


@dataclass(frozen=True)
class SimulationEvent:
    """Discrete event emitted during account simulation."""

    trading_day: pd.Timestamp
    event_type: EventType
    phase: AccountPhase
    message: str
    amount: float | None = None
    breach_reason: BreachReason | None = None

    def to_record(self) -> dict[str, object]:
        return {
            "trading_day": self.trading_day,
            "event_type": self.event_type.value,
            "phase": self.phase.value,
            "message": self.message,
            "amount": self.amount,
            "breach_reason": None if self.breach_reason is None else self.breach_reason.value,
        }


@dataclass(frozen=True)
class SimulationSummary:
    """Terminal summary of the full run."""

    provider_name: str
    account_code: str
    final_phase: AccountPhase
    final_status: AccountStatus
    final_balance: float
    total_profit: float
    challenge_fee_paid: float
    reset_fee_paid: float
    total_payout_amount: float
    payout_count: int
    resets_used: int
    evaluation_pass_day: pd.Timestamp | None
    breach_reason: BreachReason | None = None

    def to_record(self) -> dict[str, object]:
        return {
            "provider_name": self.provider_name,
            "account_code": self.account_code,
            "final_phase": self.final_phase.value,
            "final_status": self.final_status.value,
            "final_balance": self.final_balance,
            "total_profit": self.total_profit,
            "challenge_fee_paid": self.challenge_fee_paid,
            "reset_fee_paid": self.reset_fee_paid,
            "total_payout_amount": self.total_payout_amount,
            "payout_count": self.payout_count,
            "resets_used": self.resets_used,
            "evaluation_pass_day": self.evaluation_pass_day,
            "breach_reason": None if self.breach_reason is None else self.breach_reason.value,
        }


@dataclass(frozen=True)
class SimulationResult:
    """Public simulator output."""

    timeline: pd.DataFrame
    events: pd.DataFrame
    summary: SimulationSummary
    account_definition: AccountDefinition
