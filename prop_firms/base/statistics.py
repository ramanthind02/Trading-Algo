from __future__ import annotations

from statistics import mean

from prop_firms.base.portfolio_models import (
    PortfolioBatchStatistics,
    PortfolioSimulationResult,
)


def compute_portfolio_statistics(
    results: list[PortfolioSimulationResult],
) -> PortfolioBatchStatistics:
    """Aggregate EV-style statistics across one or more portfolio runs."""

    if not results:
        raise ValueError("results must be non-empty")

    summaries = [result.summary for result in results]
    days_to_first_payout = [
        summary.days_to_first_payout
        for summary in summaries
        if summary.days_to_first_payout is not None
    ]

    return PortfolioBatchStatistics(
        n_runs=len(results),
        expected_net_cashflow=mean(summary.net_cashflow for summary in summaries),
        expected_total_gross_payouts=mean(
            summary.total_gross_payouts for summary in summaries
        ),
        expected_total_trader_payouts=mean(
            summary.total_trader_payouts for summary in summaries
        ),
        expected_total_challenge_costs=mean(
            summary.total_challenge_costs for summary in summaries
        ),
        expected_total_activation_costs=mean(
            summary.total_activation_costs for summary in summaries
        ),
        expected_total_reset_costs=mean(
            summary.total_reset_costs for summary in summaries
        ),
        expected_total_fee_refunds=mean(
            summary.total_fee_refunds for summary in summaries
        ),
        expected_funded_accounts_created=mean(
            summary.funded_accounts_created for summary in summaries
        ),
        expected_average_active_funded_accounts=mean(
            summary.average_active_funded_accounts for summary in summaries
        ),
        expected_average_active_challenges=mean(
            summary.average_active_challenges for summary in summaries
        ),
        expected_payouts_per_funded_account=mean(
            summary.payouts_per_funded_account for summary in summaries
        ),
        expected_days_to_first_payout=(
            None if not days_to_first_payout else mean(days_to_first_payout)
        ),
        probability_negative_net_cashflow=mean(
            1.0 if summary.net_cashflow < 0.0 else 0.0 for summary in summaries
        ),
    )
