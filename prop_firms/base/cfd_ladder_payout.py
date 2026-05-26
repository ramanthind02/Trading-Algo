from __future__ import annotations

from dataclasses import dataclass

from prop_firms.base.portfolio_models import PortfolioPayoutPolicyConfig


@dataclass(frozen=True)
class CfdLadderPayoutDecision:
    """Resolved CFD ladder withdrawal for the current account balance."""

    gross_amount: float
    milestones_taken: int


def count_cfd_ladder_milestones(
    *,
    account_size: float,
    account_balance: float,
    payout_policy: PortfolioPayoutPolicyConfig,
) -> int:
    """Count earned profit milestones above the configured buffer."""

    profit_pct = (account_balance - account_size) / account_size
    first_threshold = payout_policy.cfd_buffer_pct + payout_policy.cfd_profit_step_pct
    if profit_pct < first_threshold:
        return 0
    return int((profit_pct - payout_policy.cfd_buffer_pct) // payout_policy.cfd_profit_step_pct)


def resolve_cfd_ladder_payout(
    *,
    account_size: float,
    account_balance: float,
    payout_policy: PortfolioPayoutPolicyConfig,
    milestones_already_taken: int,
) -> CfdLadderPayoutDecision | None:
    """Withdraw 2%% of nominal size for each new 2%% profit step after a 3%% buffer."""

    milestones_earned = count_cfd_ladder_milestones(
        account_size=account_size,
        account_balance=account_balance,
        payout_policy=payout_policy,
    )
    new_milestones = milestones_earned - milestones_already_taken
    if new_milestones <= 0:
        return None

    payout_per_milestone = account_size * payout_policy.cfd_payout_amount_pct
    gross_amount = new_milestones * payout_per_milestone
    buffer_balance = account_size * (1.0 + payout_policy.cfd_buffer_pct)
    available = max(account_balance - buffer_balance, 0.0)
    if gross_amount > available:
        gross_amount = available
    if gross_amount <= 0.0:
        return None

    return CfdLadderPayoutDecision(
        gross_amount=gross_amount,
        milestones_taken=milestones_earned,
    )
