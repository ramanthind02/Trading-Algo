import unittest

import pandas as pd

from prop_firms import PayoutPolicy, SimulationRequest, create_fundednext_simulator
from prop_firms.fundednext import load_fundednext_account


class TestFundedNextRuleEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.simulator = create_fundednext_simulator()
        self.account_definition = load_fundednext_account("50000")

    def test_config_matches_stellar_two_step_rules(self) -> None:
        self.assertEqual(self.account_definition.fees.challenge_fee, 300.0)
        self.assertTrue(self.account_definition.fees.refundable_on_first_payout)
        self.assertEqual(self.account_definition.evaluation.profit_target_amount, 4000.0)
        self.assertEqual(self.account_definition.verification.profit_target_amount, 2500.0)
        self.assertEqual(
            self.account_definition.funded.payout_rule.payout_cycle_rule.first_cycle_calendar_days,
            21,
        )
        self.assertEqual(
            self.account_definition.funded.payout_rule.payout_cycle_rule.subsequent_cycle_calendar_days,
            14,
        )

    def test_static_drawdown_floor_stays_at_ninety_percent(self) -> None:
        trading_days = pd.date_range("2026-01-05", periods=3, freq="B")
        returns = pd.Series([0.02, 0.02, -0.01], index=trading_days)

        result = self.simulator.simulate(
            SimulationRequest(
                account_code="50000",
                returns=returns,
                payout_policy=PayoutPolicy.NO_REQUESTS,
            )
        )

        self.assertAlmostEqual(
            float(result.timeline.iloc[-1]["max_loss_floor"]),
            45000.0,
        )

    def test_two_step_pass_transitions_to_funded(self) -> None:
        phase1_days = pd.date_range("2026-02-02", periods=5, freq="B")
        phase1_returns = pd.Series([0.016] * 5, index=phase1_days)
        phase2_days = pd.date_range("2026-02-09", periods=5, freq="B")
        phase2_returns = pd.Series([0.010] * 5, index=phase2_days)
        returns = pd.concat([phase1_returns, phase2_returns])

        result = self.simulator.simulate(
            SimulationRequest(
                account_code="50000",
                returns=returns,
                payout_policy=PayoutPolicy.NO_REQUESTS,
            )
        )

        self.assertEqual(result.summary.final_phase.value, "funded")
        self.assertIn("evaluation_passed", set(result.events["event_type"]))
        self.assertIn("funded_started", set(result.events["event_type"]))

    def test_first_payout_includes_fee_refund_after_twenty_one_days(self) -> None:
        phase1_days = pd.date_range("2026-03-02", periods=5, freq="B")
        phase1_returns = pd.Series([0.016] * 5, index=phase1_days)
        phase2_days = pd.date_range("2026-03-09", periods=5, freq="B")
        phase2_returns = pd.Series([0.010] * 5, index=phase2_days)
        funded_days = pd.date_range("2026-03-16", periods=22, freq="B")
        funded_returns = pd.Series([0.005] * 22, index=funded_days)
        returns = pd.concat([phase1_returns, phase2_returns, funded_returns])

        result = self.simulator.simulate(
            SimulationRequest(account_code="50000", returns=returns)
        )

        self.assertIn("payout_requested", set(result.events["event_type"]))
        self.assertIn("fee_refunded", set(result.events["event_type"]))
        self.assertAlmostEqual(result.summary.fee_refund_amount, 300.0)


if __name__ == "__main__":
    unittest.main()
