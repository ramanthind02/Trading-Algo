import unittest
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prop_firms import (
    PortfolioSimulationConfig,
    PurchasePolicyConfig,
    create_apex_portfolio_simulator,
)


class TestApexPortfolioSimulator(unittest.TestCase):
    def setUp(self) -> None:
        self.simulator = create_apex_portfolio_simulator()

    def test_apex_portfolio_uses_fifty_k_account_and_applies_fees(self) -> None:
        returns = pd.Series(
            [0.061],
            index=pd.bdate_range("2026-01-05", periods=1),
        )
        result = self.simulator.simulate(
            returns=returns,
            config=PortfolioSimulationConfig(account_code="50000"),
        )

        self.assertEqual(result.summary.account_code, "50000")
        self.assertEqual(result.summary.funded_accounts_created, 1)
        self.assertAlmostEqual(result.summary.total_challenge_costs, 30.0, places=6)
        self.assertAlmostEqual(result.summary.total_activation_costs, 100.0, places=6)

    def test_month_start_purchase_respects_twenty_account_caps(self) -> None:
        returns = pd.Series(
            [0.0] * 70,
            index=pd.bdate_range("2026-01-05", periods=70),
        )
        result = self.simulator.simulate(
            returns=returns,
            config=PortfolioSimulationConfig(
                account_code="50000",
                purchase_policy=PurchasePolicyConfig(
                    funded_account_cap=20,
                    challenge_account_cap=20,
                    challenges_per_purchase_window=25,
                ),
            ),
        )

        self.assertGreaterEqual(result.summary.challenges_purchased, 20)
        self.assertEqual(int(result.daily_timeline["active_challenges"].max()), 20)
        self.assertIn("purchase_skipped", set(result.events["event_type"]))

    def test_passed_challenges_transition_into_funded_slots(self) -> None:
        returns = pd.Series(
            [0.061, 0.0, 0.0, 0.0, 0.0],
            index=pd.bdate_range("2026-01-05", periods=5),
        )
        result = self.simulator.simulate(
            returns=returns,
            config=PortfolioSimulationConfig(
                account_code="50000",
                purchase_policy=PurchasePolicyConfig(
                    funded_account_cap=1,
                    challenge_account_cap=20,
                    challenges_per_purchase_window=1,
                ),
            ),
        )

        self.assertEqual(result.summary.funded_accounts_created, 1)
        self.assertIn("challenge_passed", set(result.events["event_type"]))
        self.assertIn("funded_started", set(result.events["event_type"]))
        funded_accounts = result.account_summaries[
            result.account_summaries["lifecycle_state"] == "funded_active"
        ]
        self.assertFalse(funded_accounts.empty)

    def test_challenge_expires_after_thirty_calendar_days_without_activation(self) -> None:
        returns = pd.Series(
            [0.0] * len(pd.bdate_range("2026-01-05", "2026-02-04")),
            index=pd.bdate_range("2026-01-05", "2026-02-04"),
        )
        result = self.simulator.simulate(
            returns=returns,
            config=PortfolioSimulationConfig(
                account_code="50000",
                purchase_policy=PurchasePolicyConfig(
                    funded_account_cap=1,
                    challenge_account_cap=1,
                    challenges_per_purchase_window=1,
                ),
            ),
        )

        self.assertEqual(result.summary.funded_accounts_created, 0)
        self.assertEqual(result.summary.challenges_failed, 0)
        self.assertAlmostEqual(result.summary.total_activation_costs, 0.0, places=6)
        self.assertIn("challenge_expired", set(result.events["event_type"]))
        self.assertNotIn("funded_started", set(result.events["event_type"]))
        expired_accounts = result.account_summaries[
            result.account_summaries["lifecycle_state"] == "challenge_expired"
        ]
        self.assertFalse(expired_accounts.empty)


if __name__ == "__main__":
    unittest.main()
