import unittest
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prop_firms import PayoutPolicy, SimulationRequest, create_apex_simulator


def _build_return_series_from_profits(
    start_day: str,
    evaluation_returns: list[float],
    funded_daily_profits: list[float],
) -> pd.Series:
    funded_open_balances: list[float] = []
    current_balance = 50000.0
    for profit in funded_daily_profits:
        funded_open_balances.append(current_balance)
        current_balance += profit
    funded_returns = [
        profit / opening_balance
        for profit, opening_balance in zip(
            funded_daily_profits,
            funded_open_balances,
            strict=True,
        )
    ]
    return pd.Series(
        evaluation_returns + funded_returns,
        index=pd.bdate_range(start_day, periods=len(evaluation_returns) + len(funded_returns)),
    )


def _build_multi_payout_return_series(
    start_day: str,
    cycle_daily_profits: list[float],
    payout_schedule: list[float],
) -> pd.Series:
    returns = [0.061]
    current_balance = 50000.0
    for requested_amount in payout_schedule:
        for profit in cycle_daily_profits:
            returns.append(profit / current_balance)
            current_balance += profit
        current_balance -= requested_amount
    return pd.Series(returns, index=pd.bdate_range(start_day, periods=len(returns)))


class TestApexRuleEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.simulator = create_apex_simulator()

    def test_evaluation_passes_after_one_profitable_day(self) -> None:
        trading_days = pd.date_range("2026-01-05", periods=1, freq="B")
        returns = pd.Series([0.061], index=trading_days)

        result = self.simulator.simulate(
            SimulationRequest(
                account_code="50000",
                returns=returns,
                payout_policy=PayoutPolicy.NO_REQUESTS,
            )
        )

        self.assertEqual(result.summary.final_phase.value, "funded")
        self.assertEqual(result.summary.final_status.value, "active")
        self.assertEqual(result.timeline.iloc[-1]["phase"], "evaluation")
        self.assertIn("evaluation_passed", set(result.events["event_type"]))

    def test_evaluation_expires_after_thirty_calendar_days_without_passing(self) -> None:
        returns = pd.Series(
            [0.0] * len(pd.bdate_range("2026-01-05", "2026-02-04")),
            index=pd.bdate_range("2026-01-05", "2026-02-04"),
        )

        result = self.simulator.simulate(
            SimulationRequest(
                account_code="50000",
                returns=returns,
                payout_policy=PayoutPolicy.NO_REQUESTS,
            )
        )

        self.assertEqual(result.summary.final_phase.value, "evaluation")
        self.assertEqual(result.summary.final_status.value, "breached")
        self.assertEqual(result.summary.breach_reason.value, "evaluation_expired")
        self.assertEqual(
            pd.Timestamp(result.timeline.iloc[-1]["trading_day"]),
            pd.Timestamp("2026-02-04"),
        )
        self.assertAlmostEqual(float(result.timeline.iloc[-1]["balance"]), 50000.0, places=6)
        self.assertIn("evaluation_expired", set(result.events["event_type"]))

    def test_evaluation_breaches_when_balance_touches_active_eod_threshold(self) -> None:
        returns = pd.Series(
            [0.02, -0.0392156862745098],
            index=pd.bdate_range("2026-01-05", periods=2),
        )

        result = self.simulator.simulate(
            SimulationRequest(
                account_code="50000",
                returns=returns,
                payout_policy=PayoutPolicy.NO_REQUESTS,
            )
        )

        self.assertEqual(result.summary.final_status.value, "breached")
        self.assertEqual(result.summary.breach_reason.value, "max_loss_limit")
        self.assertAlmostEqual(float(result.timeline.iloc[-1]["balance"]), 49000.0, places=6)

    def test_funded_threshold_locks_at_fifty_thousand_one_hundred(self) -> None:
        returns = pd.Series(
            [0.061, 0.022, 0.02],
            index=pd.bdate_range("2026-01-05", periods=3),
        )

        result = self.simulator.simulate(
            SimulationRequest(
                account_code="50000",
                returns=returns,
                payout_policy=PayoutPolicy.NO_REQUESTS,
            )
        )

        funded_rows = result.timeline[result.timeline["phase"] == "funded"].reset_index(drop=True)
        self.assertAlmostEqual(float(funded_rows.iloc[-1]["max_loss_floor"]), 50100.0)

    def test_daily_loss_limit_stops_the_day_without_closing_pa(self) -> None:
        returns = pd.Series(
            [0.061, -0.03],
            index=pd.bdate_range("2026-01-05", periods=2),
        )

        result = self.simulator.simulate(SimulationRequest(account_code="50000", returns=returns))

        funded_row = result.timeline[result.timeline["phase"] == "funded"].iloc[-1]
        self.assertEqual(result.summary.final_phase.value, "funded")
        self.assertEqual(result.summary.final_status.value, "active")
        self.assertEqual(int(funded_row["daily_loss_limit_hit"]), 1)
        self.assertAlmostEqual(float(funded_row["balance"]), 49000.0, places=6)
        self.assertIn("daily_loss_limit_hit", set(result.events["event_type"]))

    def test_weekend_boundary_expires_on_first_trading_day_after_day_thirty(self) -> None:
        returns = pd.Series(
            [0.0] * len(pd.bdate_range("2026-01-02", "2026-02-02")),
            index=pd.bdate_range("2026-01-02", "2026-02-02"),
        )

        result = self.simulator.simulate(
            SimulationRequest(
                account_code="50000",
                returns=returns,
                payout_policy=PayoutPolicy.NO_REQUESTS,
            )
        )

        self.assertEqual(
            pd.Timestamp(result.timeline.iloc[-1]["trading_day"]),
            pd.Timestamp("2026-02-02"),
        )
        self.assertEqual(result.summary.breach_reason.value, "evaluation_expired")

    def test_funded_accounts_do_not_expire_after_evaluation_pass(self) -> None:
        returns = pd.Series(
            [0.061] + ([0.0] * 40),
            index=pd.bdate_range("2026-01-05", periods=41),
        )

        result = self.simulator.simulate(
            SimulationRequest(
                account_code="50000",
                returns=returns,
                payout_policy=PayoutPolicy.NO_REQUESTS,
            )
        )

        self.assertEqual(result.summary.final_phase.value, "funded")
        self.assertEqual(result.summary.final_status.value, "active")
        self.assertNotIn("evaluation_expired", set(result.events["event_type"]))

    def test_payout_requires_five_qualifying_days_and_strict_consistency(self) -> None:
        returns = _build_return_series_from_profits(
            start_day="2026-01-05",
            evaluation_returns=[0.061],
            funded_daily_profits=[1300.0, 325.0, 325.0, 325.0, 325.0],
        )

        result = self.simulator.simulate(SimulationRequest(account_code="50000", returns=returns))

        self.assertEqual(result.summary.payout_count, 0)
        self.assertNotIn("payout_requested", set(result.events["event_type"]))
        funded_row = result.timeline[result.timeline["phase"] == "funded"].iloc[-1]
        self.assertAlmostEqual(float(funded_row["consistency_ratio"]), 0.5, places=6)
        self.assertAlmostEqual(float(funded_row["consistency_cap_ratio"]), 0.5, places=6)

    def test_payout_caps_follow_apex_schedule(self) -> None:
        payout_schedule = [1500.0, 1500.0, 2000.0, 2500.0, 2500.0, 3000.0]
        returns = _build_multi_payout_return_series(
            start_day="2026-01-05",
            cycle_daily_profits=[1000.0] * 5,
            payout_schedule=payout_schedule,
        )

        result = self.simulator.simulate(SimulationRequest(account_code="50000", returns=returns))

        payout_rows = result.events[result.events["event_type"] == "payout_requested"].reset_index(drop=True)
        self.assertEqual(result.summary.final_status.value, "completed")
        self.assertEqual(result.summary.payout_count, 6)
        self.assertEqual(
            [float(amount) for amount in payout_rows["amount"]],
            payout_schedule,
        )


if __name__ == "__main__":
    unittest.main()
