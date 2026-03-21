from __future__ import annotations

from enum import Enum


class AccountPhase(str, Enum):
    """Lifecycle phase for a prop-firm account."""

    EVALUATION = "evaluation"
    FUNDED = "funded"


class AccountStatus(str, Enum):
    """Current account state."""

    ACTIVE = "active"
    BREACHED = "breached"
    COMPLETED = "completed"


class BreachReason(str, Enum):
    """Typed breach reasons used in summaries and events."""

    CONSISTENCY = "consistency"
    CONTRACT_LIMIT = "contract_limit"
    DATA_VALIDATION = "data_validation"
    EVALUATION_EXPIRED = "evaluation_expired"
    LEVERAGE_LIMIT = "leverage_limit"
    MAX_LOSS_LIMIT = "max_loss_limit"
    PAYOUT_LIMIT = "payout_limit"


class EventType(str, Enum):
    """Event types emitted during simulation."""

    SIMULATION_STARTED = "simulation_started"
    EVALUATION_EXPIRED = "evaluation_expired"
    EVALUATION_PASSED = "evaluation_passed"
    FUNDED_STARTED = "funded_started"
    BREACH = "breach"
    RESET_APPLIED = "reset_applied"
    DAILY_LOSS_LIMIT_HIT = "daily_loss_limit_hit"
    PAYOUT_ELIGIBLE = "payout_eligible"
    PAYOUT_REQUESTED = "payout_requested"
    SCALE_CHANGED = "scale_changed"
    SIMULATION_COMPLETED = "simulation_completed"


class PayoutPolicy(str, Enum):
    """How payout requests should be handled during funded simulation."""

    AUTO_MAX = "auto_max"
    AUTO_MINIMUM = "auto_minimum"
    FIXED_AMOUNT = "fixed_amount"
    NO_REQUESTS = "no_requests"


class ResetPolicy(str, Enum):
    """How evaluation breaches should be handled."""

    AUTO_RESET = "auto_reset"
    NO_RESETS = "no_resets"
