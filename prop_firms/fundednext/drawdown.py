from __future__ import annotations

from prop_firms.base.models import EvaluationRules, FundedRules


def compute_static_max_loss_floor(
    account_size: float,
    rules: EvaluationRules | FundedRules,
) -> float:
    """Static max-loss floor fixed at initial balance minus max loss."""

    return account_size - rules.drawdown_rule.loss_limit_amount
