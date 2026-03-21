import unittest
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prop_firms import (
    PortfolioPayoutPolicyConfig,
    PortfolioPayoutPolicyMode,
    PortfolioSimulationConfig,
    PurchasePolicyConfig,
    ReturnEngineConfig,
    apply_target_volatility,
    build_return_series,
    compute_portfolio_statistics,
    create_lucid_portfolio_simulator,
)


class TestLucidPortfolioSimulator(unittest.TestCase):
    def setUp(self) -> None:
        self.simulator = create_lucid_portfolio_simulator()

    def test_month_start_purchase_respects_challenge_cap(self) -> None:
        returns = pd.Series(
            [0.0] * 70,
            index=pd.bdate_range("2026-01-02", periods=70),
        )
        config = PortfolioSimulationConfig(
            purchase_policy=PurchasePolicyConfig(
                funded_account_cap=5,
                challenge_account_cap=2,
            )
        )

        result = self.simulator.simulate(returns=returns, config=config)

        self.assertEqual(result.summary.challenges_purchased, 2)
        self.assertEqual(int(result.daily_timeline["active_challenges"].max()), 2)
        self.assertIn("purchase_skipped", set(result.events["event_type"]))

    def test_apply_target_volatility_preserves_index(self) -> None:
        returns = pd.Series(
            [0.01, -0.02, 0.015, -0.01, 0.005, 0.02],
            index=pd.bdate_range("2026-01-02", periods=6),
        )

        scaled = apply_target_volatility(
            returns=returns,
            config=ReturnEngineConfig(target_annual_volatility=0.10),
        )

        self.assertTrue(scaled.index.equals(returns.index))
        realized_vol = float(scaled.std(ddof=1) * (252.0**0.5))
        self.assertAlmostEqual(realized_vol, 0.10, places=8)

    def test_build_return_series_filters_to_configured_date_window(self) -> None:
        returns = pd.Series(
            [0.01, -0.02, 0.015, -0.01, 0.005],
            index=pd.to_datetime(
                [
                    "2020-12-31",
                    "2021-01-04",
                    "2023-06-15",
                    "2025-12-31",
                    "2026-01-02",
                ]
            ),
        )

        filtered = build_return_series(
            config=ReturnEngineConfig(
                start_date="2021-01-01",
                end_date="2025-12-31",
            ),
            external_returns=returns,
        )

        self.assertEqual(filtered.index.min(), pd.Timestamp("2021-01-04"))
        self.assertEqual(filtered.index.max(), pd.Timestamp("2025-12-31"))
        self.assertEqual(len(filtered), 3)

    def test_build_return_series_generates_deterministic_sharpe_path(self) -> None:
        config = ReturnEngineConfig(
            target_annual_volatility=0.10,
            target_sharpe=1.0,
            start_date="2025-01-01",
            end_date="2025-01-31",
            random_seed=7,
        )

        first = build_return_series(config=config)
        second = build_return_series(config=config)

        self.assertTrue(first.equals(second))
        self.assertTrue(first.index.equals(pd.bdate_range("2025-01-01", "2025-01-31")))
        realized_vol = float(first.std(ddof=1) * (config.annualization_factor**0.5))
        realized_sharpe = float(
            first.mean() / first.std(ddof=1) * (config.annualization_factor**0.5)
        )
        self.assertAlmostEqual(realized_vol, 0.10, places=8)
        self.assertAlmostEqual(realized_sharpe, 1.0, places=8)

    def test_first_payout_cashflow_matches_expected_value(self) -> None:
        returns = pd.Series(
            [0.026, 0.024, 0.01, 0.01, 0.01, 0.01, 0.01],
            index=pd.bdate_range("2026-01-02", periods=7),
        )

        result = self.simulator.simulate(
            returns=returns,
            config=PortfolioSimulationConfig(),
        )

        self.assertEqual(result.summary.funded_accounts_created, 1)
        self.assertAlmostEqual(result.summary.total_challenge_costs, 70.0, places=6)
        self.assertAlmostEqual(
            result.summary.total_trader_payouts,
            573.8630631281264,
            places=6,
        )
        self.assertAlmostEqual(
            result.summary.net_cashflow,
            503.86306312812635,
            places=6,
        )
        stats = compute_portfolio_statistics([result])
        self.assertAlmostEqual(stats.expected_net_cashflow, result.summary.net_cashflow)

    def test_funded_account_closes_after_six_payouts(self) -> None:
        returns = pd.Series(
            [0.026, 0.024] + ([0.01] * 30),
            index=pd.bdate_range("2026-01-02", periods=32),
        )

        result = self.simulator.simulate(
            returns=returns,
            config=PortfolioSimulationConfig(),
        )

        closed_accounts = result.account_summaries[
            result.account_summaries["lifecycle_state"] == "funded_closed"
        ]
        self.assertFalse(closed_accounts.empty)
        self.assertIn(6, set(closed_accounts["payout_count"]))
        self.assertGreater(result.summary.funded_closed_from_max_payouts, 0)
        self.assertEqual(result.summary.funded_closed_from_drawdown, 0)

    def test_payout_policies_change_net_cashflow(self) -> None:
        returns = pd.Series(
            [0.026, 0.024] + ([0.01] * 15),
            index=pd.bdate_range("2026-01-02", periods=17),
        )
        aggressive = self.simulator.simulate(
            returns=returns,
            config=PortfolioSimulationConfig(
                payout_policy=PortfolioPayoutPolicyConfig(
                    mode=PortfolioPayoutPolicyMode.AGGRESSIVE
                )
            ),
        )
        fractional = self.simulator.simulate(
            returns=returns,
            config=PortfolioSimulationConfig(
                payout_policy=PortfolioPayoutPolicyConfig(
                    mode=PortfolioPayoutPolicyMode.FRACTIONAL,
                    withdrawal_fraction=0.5,
                )
            ),
        )

        self.assertGreater(
            aggressive.summary.total_trader_payouts,
            fractional.summary.total_trader_payouts,
        )
        self.assertGreater(
            aggressive.summary.net_cashflow,
            fractional.summary.net_cashflow,
        )

    def test_buffer_policy_with_zero_buffer_matches_aggressive_payouts(self) -> None:
        returns = pd.Series(
            [0.026, 0.024] + ([0.01] * 15),
            index=pd.bdate_range("2026-01-02", periods=17),
        )
        aggressive = self.simulator.simulate(
            returns=returns,
            config=PortfolioSimulationConfig(
                payout_policy=PortfolioPayoutPolicyConfig(
                    mode=PortfolioPayoutPolicyMode.AGGRESSIVE
                )
            ),
        )
        buffered = self.simulator.simulate(
            returns=returns,
            config=PortfolioSimulationConfig(
                payout_policy=PortfolioPayoutPolicyConfig(
                    mode=PortfolioPayoutPolicyMode.BUFFER,
                    buffer_amount=0.0,
                )
            ),
        )

        self.assertAlmostEqual(
            aggressive.summary.total_trader_payouts,
            buffered.summary.total_trader_payouts,
            places=6,
        )
        self.assertAlmostEqual(
            aggressive.summary.net_cashflow,
            buffered.summary.net_cashflow,
            places=6,
        )

    def test_buffer_policy_takes_full_lucid_25k_max_payout_when_allowed(self) -> None:
        returns = pd.Series(
            [0.026, 0.024] + ([0.02] * 25),
            index=pd.bdate_range("2026-01-02", periods=27),
        )
        buffer_amount = 1_500.0
        result = self.simulator.simulate(
            returns=returns,
            config=PortfolioSimulationConfig(
                account_code="25000",
                payout_policy=PortfolioPayoutPolicyConfig(
                    mode=PortfolioPayoutPolicyMode.BUFFER,
                    buffer_amount=buffer_amount,
                ),
            ),
        )

        payout_events = result.events[
            result.events["event_type"] == "payout_requested"
        ].reset_index(drop=True)
        self.assertFalse(payout_events.empty)
        self.assertTrue((payout_events["amount"] == 1000.0).all())

        funded_accounts = result.account_summaries[
            (result.account_summaries["phase"] == "funded")
            & (result.account_summaries["payout_count"] > 0)
        ]
        protected_plus_buffer = 25_100.0 + buffer_amount
        self.assertTrue((funded_accounts["balance"] >= protected_plus_buffer).all())

    def test_challenge_vol_multiplier_changes_challenge_progression_only(self) -> None:
        base_returns = pd.Series(
            [0.005] * 200,
            index=pd.bdate_range("2026-01-02", periods=200),
        )
        low_vol_config = PortfolioSimulationConfig(
            return_engine=ReturnEngineConfig(),
            challenge_vol_multiplier=0.5,
            funded_vol_multiplier=1.0,
        )
        high_vol_config = PortfolioSimulationConfig(
            return_engine=ReturnEngineConfig(),
            challenge_vol_multiplier=2.0,
            funded_vol_multiplier=1.0,
        )

        low = self.simulator.simulate(returns=base_returns, config=low_vol_config)
        high = self.simulator.simulate(returns=base_returns, config=high_vol_config)

        self.assertNotEqual(
            low.summary.funded_accounts_created,
            high.summary.funded_accounts_created,
        )
        self.assertGreater(
            low.summary.days_to_first_payout,
            high.summary.days_to_first_payout,
        )

    def test_funded_vol_multiplier_changes_funded_payouts_only(self) -> None:
        base_returns = pd.Series(
            [0.005] * 200,
            index=pd.bdate_range("2026-01-02", periods=200),
        )
        low_vol_config = PortfolioSimulationConfig(
            return_engine=ReturnEngineConfig(),
            challenge_vol_multiplier=1.0,
            funded_vol_multiplier=0.5,
        )
        high_vol_config = PortfolioSimulationConfig(
            return_engine=ReturnEngineConfig(),
            challenge_vol_multiplier=1.0,
            funded_vol_multiplier=2.0,
        )

        low = self.simulator.simulate(returns=base_returns, config=low_vol_config)
        high = self.simulator.simulate(returns=base_returns, config=high_vol_config)

        self.assertEqual(
            low.summary.challenges_failed,
            high.summary.challenges_failed,
        )
        self.assertNotEqual(
            low.summary.total_trader_payouts,
            high.summary.total_trader_payouts,
        )

    def test_phase_vol_multipliers_default_to_identity(self) -> None:
        returns = pd.Series(
            [0.01, -0.02, 0.015, -0.01, 0.005, 0.02],
            index=pd.bdate_range("2026-01-02", periods=6),
        )
        default_config = PortfolioSimulationConfig()
        explicit_config = PortfolioSimulationConfig(
            challenge_vol_multiplier=1.0,
            funded_vol_multiplier=1.0,
        )

        default_result = self.simulator.simulate(
            returns=returns,
            config=default_config,
        )
        explicit_result = self.simulator.simulate(
            returns=returns,
            config=explicit_config,
        )

        self.assertTrue(
            default_result.daily_timeline.equals(explicit_result.daily_timeline)
        )
        self.assertTrue(default_result.events.equals(explicit_result.events))

    def test_passed_challenges_are_discarded_when_funded_slots_are_full(self) -> None:
        returns = pd.Series(
            ([0.0] * 84) + [0.026, 0.024],
            index=pd.bdate_range("2026-01-02", periods=86),
        )
        result = self.simulator.simulate(
            returns=returns,
            config=PortfolioSimulationConfig(
                purchase_policy=PurchasePolicyConfig(
                    funded_account_cap=1,
                    challenge_account_cap=10,
                    challenges_per_purchase_window=1,
                )
            ),
        )

        self.assertLessEqual(int(result.daily_timeline["active_funded"].max()), 1)
        discarded_accounts = result.account_summaries[
            result.account_summaries["lifecycle_state"] == "challenge_discarded"
        ]
        self.assertFalse(discarded_accounts.empty)
        self.assertIn("challenge_discarded", set(result.events["event_type"]))
        self.assertEqual(result.summary.funded_accounts_created, 1)


if __name__ == "__main__":
    unittest.main()
