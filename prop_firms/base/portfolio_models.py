from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from prop_firms.base.enums import AccountPhase, BreachReason
from prop_firms.base.models import AccountDefinition


class AccountLifecycleState(str, Enum):
    """Explicit multi-account lifecycle states."""

    CHALLENGE_ACTIVE = "challenge_active"
    CHALLENGE_DISCARDED = "challenge_discarded"
    CHALLENGE_EXPIRED = "challenge_expired"
    CHALLENGE_FAILED = "challenge_failed"
    FUNDED_ACTIVE = "funded_active"
    FUNDED_CLOSED = "funded_closed"


class PortfolioEventType(str, Enum):
    """Portfolio-level event types."""

    SIMULATION_STARTED = "simulation_started"
    CHALLENGE_PURCHASED = "challenge_purchased"
    CHALLENGE_DISCARDED = "challenge_discarded"
    CHALLENGE_EXPIRED = "challenge_expired"
    CHALLENGE_FAILED = "challenge_failed"
    CHALLENGE_PASSED = "challenge_passed"
    FUNDED_STARTED = "funded_started"
    PAYOUT_REQUESTED = "payout_requested"
    FUNDED_CLOSED = "funded_closed"
    PURCHASE_SKIPPED = "purchase_skipped"
    SIMULATION_COMPLETED = "simulation_completed"


class PortfolioPayoutPolicyMode(str, Enum):
    """Supported payout strategies for funded accounts."""

    AGGRESSIVE = "aggressive"
    BUFFER = "buffer"
    FRACTIONAL = "fractional"


@dataclass(frozen=True)
class PurchasePolicyConfig:
    """Configurable challenge purchase policy."""

    funded_account_cap: int = 20
    challenge_account_cap: int = 20
    challenges_per_purchase_window: int = 1

    def __post_init__(self) -> None:
        if self.funded_account_cap < 1:
            raise ValueError("funded_account_cap must be >= 1")
        if self.challenge_account_cap < 1:
            raise ValueError("challenge_account_cap must be >= 1")
        if self.challenges_per_purchase_window < 1:
            raise ValueError("challenges_per_purchase_window must be >= 1")


@dataclass(frozen=True)
class PortfolioPayoutPolicyConfig:
    """Config for funded-account payout behavior."""

    mode: PortfolioPayoutPolicyMode = PortfolioPayoutPolicyMode.AGGRESSIVE
    buffer_amount: float = 0.0
    withdrawal_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.buffer_amount < 0.0:
            raise ValueError("buffer_amount must be >= 0")
        if not 0.0 < self.withdrawal_fraction <= 1.0:
            raise ValueError("withdrawal_fraction must be in (0, 1]")


@dataclass(frozen=True)
class ReturnEngineConfig:
    """Return-engine settings for synthetic or externally supplied replay."""

    target_annual_volatility: float | None = None
    target_sharpe: float = 1.0
    annualization_factor: float = 252.0
    start_date: str | None = None
    end_date: str | None = None
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.target_annual_volatility is not None and self.target_annual_volatility <= 0.0:
            raise ValueError("target_annual_volatility must be > 0 when set")
        if self.annualization_factor <= 0.0:
            raise ValueError("annualization_factor must be > 0")
        if self.random_seed < 0:
            raise ValueError("random_seed must be >= 0")
        if self.start_date is not None:
            pd.Timestamp(self.start_date)
        if self.end_date is not None:
            pd.Timestamp(self.end_date)
        if self.start_date is not None and self.end_date is not None:
            if pd.Timestamp(self.start_date) > pd.Timestamp(self.end_date):
                raise ValueError("start_date must be <= end_date")


@dataclass(frozen=True)
class PortfolioSimulationConfig:
    """Top-level config for the multi-account portfolio simulator."""

    account_code: str = "25000"
    challenge_vol_multiplier: float = 1.0
    funded_vol_multiplier: float = 1.0
    purchase_policy: PurchasePolicyConfig = field(default_factory=PurchasePolicyConfig)
    payout_policy: PortfolioPayoutPolicyConfig = field(
        default_factory=PortfolioPayoutPolicyConfig
    )
    return_engine: ReturnEngineConfig = field(default_factory=ReturnEngineConfig)
    max_payouts_per_funded_account: int | None = None

    def __post_init__(self) -> None:
        if self.challenge_vol_multiplier <= 0.0:
            raise ValueError("challenge_vol_multiplier must be > 0")
        if self.funded_vol_multiplier <= 0.0:
            raise ValueError("funded_vol_multiplier must be > 0")
        if self.max_payouts_per_funded_account is not None:
            if self.max_payouts_per_funded_account < 1:
                raise ValueError("max_payouts_per_funded_account must be >= 1")


@dataclass(frozen=True)
class PortfolioAccountState:
    """State carried for each simulated account."""

    account_id: str
    account_code: str
    lifecycle_state: AccountLifecycleState
    phase: AccountPhase
    start_day: pd.Timestamp
    balance: float
    closing_high_balance: float
    max_loss_floor: float
    largest_daily_profit: float
    evaluation_trading_days: int
    payout_cycle_profit: float
    payout_cycle_largest_profit: float
    payout_profit_days: int
    payout_count: int
    total_gross_payouts: float
    total_trader_payouts: float
    challenge_fee_paid: float
    activation_fee_paid: float
    reset_fee_paid: float
    scale_mini_limit: int
    scale_micro_limit: int
    pass_day: pd.Timestamp | None = None
    close_day: pd.Timestamp | None = None
    last_payout_day: pd.Timestamp | None = None
    breach_reason: BreachReason | None = None

    def to_record(self) -> dict[str, object]:
        return {
            "account_id": self.account_id,
            "account_code": self.account_code,
            "lifecycle_state": self.lifecycle_state.value,
            "phase": self.phase.value,
            "start_day": self.start_day,
            "balance": self.balance,
            "closing_high_balance": self.closing_high_balance,
            "max_loss_floor": self.max_loss_floor,
            "largest_daily_profit": self.largest_daily_profit,
            "evaluation_trading_days": self.evaluation_trading_days,
            "payout_cycle_profit": self.payout_cycle_profit,
            "payout_cycle_largest_profit": self.payout_cycle_largest_profit,
            "payout_profit_days": self.payout_profit_days,
            "payout_count": self.payout_count,
            "total_gross_payouts": self.total_gross_payouts,
            "total_trader_payouts": self.total_trader_payouts,
            "challenge_fee_paid": self.challenge_fee_paid,
            "activation_fee_paid": self.activation_fee_paid,
            "reset_fee_paid": self.reset_fee_paid,
            "scale_mini_limit": self.scale_mini_limit,
            "scale_micro_limit": self.scale_micro_limit,
            "pass_day": self.pass_day,
            "close_day": self.close_day,
            "last_payout_day": self.last_payout_day,
            "breach_reason": None if self.breach_reason is None else self.breach_reason.value,
        }


@dataclass(frozen=True)
class PortfolioDailySnapshot:
    """Aggregate daily portfolio snapshot across all accounts."""

    trading_day: pd.Timestamp
    daily_return: float
    active_challenges: int
    active_funded: int
    purchased_challenges: int
    challenge_failures: int
    challenge_passes: int
    funded_closures: int
    gross_payouts: float
    trader_payouts: float
    challenge_costs: float
    activation_costs: float
    reset_costs: float
    net_cashflow: float
    cumulative_net_cashflow: float

    def to_record(self) -> dict[str, object]:
        return {
            "trading_day": self.trading_day,
            "daily_return": self.daily_return,
            "active_challenges": self.active_challenges,
            "active_funded": self.active_funded,
            "purchased_challenges": self.purchased_challenges,
            "challenge_failures": self.challenge_failures,
            "challenge_passes": self.challenge_passes,
            "funded_closures": self.funded_closures,
            "gross_payouts": self.gross_payouts,
            "trader_payouts": self.trader_payouts,
            "challenge_costs": self.challenge_costs,
            "activation_costs": self.activation_costs,
            "reset_costs": self.reset_costs,
            "net_cashflow": self.net_cashflow,
            "cumulative_net_cashflow": self.cumulative_net_cashflow,
        }


@dataclass(frozen=True)
class PortfolioEvent:
    """Event emitted by the portfolio simulator."""

    trading_day: pd.Timestamp
    event_type: PortfolioEventType
    message: str
    account_id: str | None = None
    lifecycle_state: AccountLifecycleState | None = None
    amount: float | None = None

    def to_record(self) -> dict[str, object]:
        return {
            "trading_day": self.trading_day,
            "event_type": self.event_type.value,
            "message": self.message,
            "account_id": self.account_id,
            "lifecycle_state": (
                None if self.lifecycle_state is None else self.lifecycle_state.value
            ),
            "amount": self.amount,
        }


@dataclass(frozen=True)
class PortfolioSimulationSummary:
    """Terminal summary for one portfolio replay run."""

    account_code: str
    total_days: int
    challenges_purchased: int
    challenges_failed: int
    funded_accounts_created: int
    funded_accounts_closed: int
    total_gross_payouts: float
    total_trader_payouts: float
    total_challenge_costs: float
    total_activation_costs: float
    total_reset_costs: float
    net_cashflow: float
    first_payout_day: pd.Timestamp | None
    days_to_first_payout: int | None
    average_active_funded_accounts: float
    average_active_challenges: float
    payouts_per_funded_account: float
    negative_net_cashflow_probability: float
    funded_closed_from_drawdown: int = 0
    funded_closed_from_max_payouts: int = 0

    def to_record(self) -> dict[str, object]:
        return {
            "account_code": self.account_code,
            "total_days": self.total_days,
            "challenges_purchased": self.challenges_purchased,
            "challenges_failed": self.challenges_failed,
            "funded_accounts_created": self.funded_accounts_created,
            "funded_accounts_closed": self.funded_accounts_closed,
            "total_gross_payouts": self.total_gross_payouts,
            "total_trader_payouts": self.total_trader_payouts,
            "total_challenge_costs": self.total_challenge_costs,
            "total_activation_costs": self.total_activation_costs,
            "total_reset_costs": self.total_reset_costs,
            "net_cashflow": self.net_cashflow,
            "first_payout_day": self.first_payout_day,
            "days_to_first_payout": self.days_to_first_payout,
            "average_active_funded_accounts": self.average_active_funded_accounts,
            "average_active_challenges": self.average_active_challenges,
            "payouts_per_funded_account": self.payouts_per_funded_account,
            "negative_net_cashflow_probability": self.negative_net_cashflow_probability,
            "funded_closed_from_drawdown": self.funded_closed_from_drawdown,
            "funded_closed_from_max_payouts": self.funded_closed_from_max_payouts,
        }


@dataclass(frozen=True)
class PortfolioSimulationResult:
    """Public output for one portfolio simulation run."""

    daily_timeline: pd.DataFrame
    events: pd.DataFrame
    account_summaries: pd.DataFrame
    summary: PortfolioSimulationSummary
    realized_returns: pd.Series
    account_definition: AccountDefinition


@dataclass(frozen=True)
class PortfolioBatchStatistics:
    """Aggregate statistics across many portfolio runs."""

    n_runs: int
    expected_net_cashflow: float
    expected_total_gross_payouts: float
    expected_total_trader_payouts: float
    expected_total_challenge_costs: float
    expected_total_activation_costs: float
    expected_total_reset_costs: float
    expected_funded_accounts_created: float
    expected_average_active_funded_accounts: float
    expected_average_active_challenges: float
    expected_payouts_per_funded_account: float
    expected_days_to_first_payout: float | None
    probability_negative_net_cashflow: float

    def to_record(self) -> dict[str, object]:
        return {
            "n_runs": self.n_runs,
            "expected_net_cashflow": self.expected_net_cashflow,
            "expected_total_gross_payouts": self.expected_total_gross_payouts,
            "expected_total_trader_payouts": self.expected_total_trader_payouts,
            "expected_total_challenge_costs": self.expected_total_challenge_costs,
            "expected_total_activation_costs": self.expected_total_activation_costs,
            "expected_total_reset_costs": self.expected_total_reset_costs,
            "expected_funded_accounts_created": self.expected_funded_accounts_created,
            "expected_average_active_funded_accounts": self.expected_average_active_funded_accounts,
            "expected_average_active_challenges": self.expected_average_active_challenges,
            "expected_payouts_per_funded_account": self.expected_payouts_per_funded_account,
            "expected_days_to_first_payout": self.expected_days_to_first_payout,
            "probability_negative_net_cashflow": self.probability_negative_net_cashflow,
        }
