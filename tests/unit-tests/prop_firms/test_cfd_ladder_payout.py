import unittest

from prop_firms.base.cfd_ladder_payout import (
    count_cfd_ladder_milestones,
    resolve_cfd_ladder_payout,
)
from prop_firms.base.portfolio_models import (
    PortfolioPayoutPolicyConfig,
    PortfolioPayoutPolicyMode,
)


class TestCfdLadderPayout(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = PortfolioPayoutPolicyConfig(
            mode=PortfolioPayoutPolicyMode.CFD_LADDER,
            cfd_buffer_pct=0.03,
            cfd_profit_step_pct=0.02,
            cfd_payout_amount_pct=0.02,
        )
        self.account_size = 50_000.0

    def test_no_payout_before_five_percent_profit(self) -> None:
        balance = self.account_size * 1.04
        self.assertEqual(
            count_cfd_ladder_milestones(
                account_size=self.account_size,
                account_balance=balance,
                payout_policy=self.policy,
            ),
            0,
        )

    def test_first_payout_at_five_percent(self) -> None:
        balance = self.account_size * 1.05
        decision = resolve_cfd_ladder_payout(
            account_size=self.account_size,
            account_balance=balance,
            payout_policy=self.policy,
            milestones_already_taken=0,
        )
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertAlmostEqual(decision.gross_amount, 1_000.0)
        self.assertEqual(decision.milestones_taken, 1)

    def test_second_payout_at_seven_percent(self) -> None:
        balance = self.account_size * 1.07
        decision = resolve_cfd_ladder_payout(
            account_size=self.account_size,
            account_balance=balance,
            payout_policy=self.policy,
            milestones_already_taken=1,
        )
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertAlmostEqual(decision.gross_amount, 1_000.0)
        self.assertEqual(decision.milestones_taken, 2)


if __name__ == "__main__":
    unittest.main()
