import unittest
from pathlib import Path

import pandas as pd

from prop_firms import PayoutPolicy, SimulationRequest, create_lucid_simulator


class TestLucidRuleEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.simulator = create_lucid_simulator()

    def test_evaluation_consistency_cushion_allows_two_day_pass(self) -> None:
        trading_days = pd.date_range("2026-01-01", periods=2, freq="B")
        returns = pd.Series([0.026, 0.024], index=trading_days)

        result = self.simulator.simulate(
            SimulationRequest(
                account_code="25000",
                returns=returns,
                payout_policy=PayoutPolicy.NO_REQUESTS,
            )
        )

        self.assertEqual(result.summary.final_phase.value, "funded")
        self.assertEqual(result.timeline.iloc[-1]["phase"], "evaluation")
        self.assertGreater(result.timeline.iloc[-1]["consistency_ratio"], 0.5)
        self.assertLess(result.timeline.iloc[-1]["consistency_ratio"], 0.52)
        self.assertIn("evaluation_passed", set(result.events["event_type"]))

    def test_funded_drawdown_locks_after_payout(self) -> None:
        trading_days = pd.date_range("2026-02-02", periods=7, freq="B")
        returns = pd.Series(
            [0.026, 0.024, 0.01, 0.01, 0.01, 0.01, 0.01],
            index=trading_days,
        )

        result = self.simulator.simulate(
            SimulationRequest(account_code="25000", returns=returns)
        )

        funded_rows = result.timeline[result.timeline["phase"] == "funded"].reset_index(drop=True)
        self.assertGreater(funded_rows.iloc[0]["max_loss_floor"], 24000.0)
        self.assertAlmostEqual(funded_rows.iloc[-1]["max_loss_floor"], 25100.0)
        self.assertIn("payout_requested", set(result.events["event_type"]))

    def test_payout_request_honors_account_cap(self) -> None:
        trading_days = pd.date_range("2026-03-02", periods=7, freq="B")
        returns = pd.Series(
            [0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02],
            index=trading_days,
        )

        result = self.simulator.simulate(
            SimulationRequest(account_code="100000", returns=returns)
        )

        payout_rows = result.events[result.events["event_type"] == "payout_requested"]
        self.assertEqual(len(payout_rows), 1)
        self.assertAlmostEqual(float(payout_rows.iloc[0]["amount"]), 2500.0)

    def test_contract_limit_breach_from_held_contracts(self) -> None:
        trading_days = pd.date_range("2026-04-01", periods=1, freq="B")
        returns = pd.Series([0.001], index=trading_days)
        held_contracts = pd.DataFrame(
            {
                "trading_day": trading_days,
                "mini_contracts": [3],
                "micro_contracts": [0],
            }
        )

        result = self.simulator.simulate(
            SimulationRequest(
                account_code="25000",
                returns=returns,
                held_contracts=held_contracts,
                payout_policy=PayoutPolicy.NO_REQUESTS,
            )
        )

        self.assertEqual(result.summary.final_status.value, "breached")
        self.assertEqual(result.summary.breach_reason.value, "contract_limit")
        self.assertEqual(int(result.timeline.iloc[0]["contract_limit_breached"]), 1)


if __name__ == "__main__":
    unittest.main()
