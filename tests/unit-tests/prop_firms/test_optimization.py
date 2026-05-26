import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import optuna
import pandas as pd

from prop_firms import (
    FloatSearchRange,
    IntSearchRange,
    LucidHyperoptConfig,
    PortfolioPayoutPolicyMode,
    PortfolioSimulationConfig,
    PortfolioSimulationSummary,
    evaluate_portfolio_config,
    generate_optimization_report,
    optimize_lucid_hyperparameters,
    sample_portfolio_config,
)


@dataclass(frozen=True)
class FakePortfolioSimulationResult:
    summary: PortfolioSimulationSummary


class SequenceSimulator:
    def __init__(self, net_cashflows: tuple[float, ...]) -> None:
        self._net_cashflows = net_cashflows
        self._cursor = 0

    def simulate(
        self,
        returns: pd.Series,
        config: PortfolioSimulationConfig,
    ) -> FakePortfolioSimulationResult:
        del returns
        del config
        net_cashflow = self._net_cashflows[self._cursor]
        self._cursor += 1
        return FakePortfolioSimulationResult(
            summary=_build_summary(net_cashflow=net_cashflow)
        )


class ConfigScoreSimulator:
    def simulate(
        self,
        returns: pd.Series,
        config: PortfolioSimulationConfig,
    ) -> FakePortfolioSimulationResult:
        del returns
        withdrawal_bonus = config.payout_policy.withdrawal_fraction * 25.0
        mode_bonus = {
            PortfolioPayoutPolicyMode.AGGRESSIVE: 1.0,
            PortfolioPayoutPolicyMode.BUFFER: -1.0,
            PortfolioPayoutPolicyMode.FRACTIONAL: 0.5,
        }[config.payout_policy.mode]
        cap_penalty = (
            0.0
            if config.max_payouts_per_funded_account is None
            else float(config.max_payouts_per_funded_account) * 2.0
        )
        net_cashflow = (
            config.challenge_vol_multiplier * 100.0
            + config.funded_vol_multiplier * 10.0
            + float(config.purchase_policy.challenges_per_purchase_window)
            + withdrawal_bonus
            + mode_bonus
            - cap_penalty
        )
        return FakePortfolioSimulationResult(
            summary=_build_summary(net_cashflow=net_cashflow)
        )


def _build_summary(net_cashflow: float) -> PortfolioSimulationSummary:
    return PortfolioSimulationSummary(
        account_code="25000",
        total_days=10,
        challenges_purchased=5,
        challenges_failed=1,
        funded_accounts_created=2,
        funded_accounts_closed=1,
        total_gross_payouts=net_cashflow + 100.0,
        total_trader_payouts=net_cashflow + 50.0,
        total_challenge_costs=50.0,
        total_activation_costs=0.0,
        total_reset_costs=0.0,
        total_fee_refunds=0.0,
        net_cashflow=net_cashflow,
        first_payout_day=pd.Timestamp("2026-01-10"),
        days_to_first_payout=5,
        average_active_funded_accounts=1.5,
        average_active_challenges=2.5,
        payouts_per_funded_account=1.0,
        negative_net_cashflow_probability=1.0 if net_cashflow < 0.0 else 0.0,
    )


class TestPropFirmOptimization(unittest.TestCase):
    def test_sample_portfolio_config_uses_conditional_buffer_param(self) -> None:
        config = LucidHyperoptConfig(
            payout_modes=(PortfolioPayoutPolicyMode.BUFFER,),
            max_payouts_per_funded_account_choices=(None, 3),
        )
        trial = optuna.trial.FixedTrial(
            {
                "payout_mode": "buffer",
                "buffer_amount": 1250.0,
                "max_payouts_per_funded_account": "3",
                "challenges_per_purchase_window": 4,
                "challenge_vol_multiplier": 1.25,
                "funded_vol_multiplier": 0.75,
            }
        )

        sampled = sample_portfolio_config(trial=trial, config=config)

        self.assertEqual(sampled.payout_policy.mode, PortfolioPayoutPolicyMode.BUFFER)
        self.assertAlmostEqual(sampled.payout_policy.buffer_amount, 1250.0)
        self.assertAlmostEqual(sampled.payout_policy.withdrawal_fraction, 1.0)
        self.assertEqual(sampled.max_payouts_per_funded_account, 3)

    def test_build_monte_carlo_evaluation_averages_seed_runs(self) -> None:
        simulator = SequenceSimulator(net_cashflows=(100.0, 200.0, 300.0))
        config = LucidHyperoptConfig(monte_carlo_runs=3)
        portfolio_config = PortfolioSimulationConfig()

        trial_result, seed_runs = evaluate_portfolio_config(
            simulator=simulator,  # type: ignore[arg-type]
            config=config,
            portfolio_config=portfolio_config,
            trial_number=7,
        )

        self.assertEqual(trial_result.trial_number, 7)
        self.assertEqual(trial_result.monte_carlo_seeds, (10_000, 10_001, 10_002))
        self.assertAlmostEqual(trial_result.objective_value, 200.0)
        self.assertEqual(tuple(run.net_cashflow for run in seed_runs), (100.0, 200.0, 300.0))

    def test_run_lucid_hyperopt_ranks_best_trial_by_objective(self) -> None:
        config = LucidHyperoptConfig(
            monte_carlo_runs=2,
            n_trials=4,
            payout_modes=(PortfolioPayoutPolicyMode.AGGRESSIVE,),
            max_payouts_per_funded_account_choices=(None,),
            challenge_vol_multiplier_range=FloatSearchRange(low=0.5, high=1.5),
            funded_vol_multiplier_range=FloatSearchRange(low=0.5, high=1.5),
            challenges_per_purchase_window_range=IntSearchRange(low=1, high=2),
        )

        result = optimize_lucid_hyperparameters(
            config=config,
            simulator=ConfigScoreSimulator(),  # type: ignore[arg-type]
        )

        best_value = float(result.trials_frame["objective_value"].max())
        self.assertAlmostEqual(result.best_trial.objective_value, best_value)
        self.assertEqual(
            result.best_trial.trial_number,
            int(result.trials_frame.iloc[0]["trial_number"]),
        )

    def test_generate_optimization_report_writes_markdown_html_and_csvs(self) -> None:
        config = LucidHyperoptConfig(
            monte_carlo_runs=2,
            n_trials=2,
            payout_modes=(PortfolioPayoutPolicyMode.AGGRESSIVE,),
            max_payouts_per_funded_account_choices=(None,),
        )
        result = optimize_lucid_hyperparameters(
            config=config,
            simulator=ConfigScoreSimulator(),  # type: ignore[arg-type]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts = generate_optimization_report(
                result=result,
                output_dir=Path(tmp_dir),
                report_stem="unit_test_hyperopt",
                save_csvs=True,
            )

            self.assertTrue(artifacts.report_markdown_path.exists())
            self.assertTrue(artifacts.report_html_path.exists())
            self.assertTrue(artifacts.trials_csv_path.exists())
            self.assertTrue(artifacts.seed_runs_csv_path.exists())

            markdown_text = artifacts.report_markdown_path.read_text(encoding="utf-8")
            self.assertIn("Lucid Hyperopt Report", markdown_text)
            self.assertIn("Top Trials", markdown_text)

            html_text = artifacts.report_html_path.read_text(encoding="utf-8")
            self.assertIn("<html", html_text)
            self.assertIn("Best Trial", html_text)
            self.assertIn("Best Trial Seed Runs", html_text)


if __name__ == "__main__":
    unittest.main()
