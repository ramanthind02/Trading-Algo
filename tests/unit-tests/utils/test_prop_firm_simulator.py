"""
Unit Tests for Prop Firm Challenge Simulator

Tests cover:
- ChallengeRules validation
- ChallengeCosts validation
- SimulationConfig validation
- Path generators
- Drawdown breach detection
- Challenge engine
- Statistics computation

Author: Trading Research Team
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime

from utils.simulation.prop_firm_simulator import (
    PropFirmChallengeSimulator,
    SimulationConfig,
    SimulationMethod,
    ChallengeRules,
    ChallengeCosts,
    ChallengePass,
    ChallengeFail,
    SimulationStatistics,
    HistoricalWalkForwardGenerator,
    MonteCarloBlockGenerator,
    compute_statistics
)
from metrics.risk.drawdown import (
    trailing_drawdown_threshold,
    daily_drawdown,
    check_drawdown_breach
)


class TestChallengeRules(unittest.TestCase):
    """Test ChallengeRules validation."""

    def test_valid_rules(self):
        """Test valid rule creation."""
        rules = ChallengeRules(
            max_drawdown_pct=0.10,
            profit_target_pct=0.08,
            min_trading_days=5
        )
        self.assertEqual(rules.max_drawdown_pct, 0.10)
        self.assertEqual(rules.profit_target_pct, 0.08)
        self.assertEqual(rules.min_trading_days, 5)

    def test_optional_rules(self):
        """Test optional rule parameters."""
        rules = ChallengeRules(
            max_drawdown_pct=0.10,
            profit_target_pct=0.08,
            min_trading_days=5,
            max_daily_drawdown_pct=0.05,
            trailing_drawdown_pct=0.05
        )
        self.assertEqual(rules.max_daily_drawdown_pct, 0.05)
        self.assertEqual(rules.trailing_drawdown_pct, 0.05)

    def test_invalid_max_drawdown(self):
        """Test that non-positive max_drawdown raises error."""
        with self.assertRaises(ValueError):
            ChallengeRules(max_drawdown_pct=0, profit_target_pct=0.08, min_trading_days=5)

        with self.assertRaises(ValueError):
            ChallengeRules(max_drawdown_pct=-0.1, profit_target_pct=0.08, min_trading_days=5)

    def test_invalid_profit_target(self):
        """Test that non-positive profit_target raises error."""
        with self.assertRaises(ValueError):
            ChallengeRules(max_drawdown_pct=0.10, profit_target_pct=0, min_trading_days=5)

    def test_invalid_min_days(self):
        """Test that min_trading_days < 1 raises error."""
        with self.assertRaises(ValueError):
            ChallengeRules(max_drawdown_pct=0.10, profit_target_pct=0.08, min_trading_days=0)


class TestChallengeCosts(unittest.TestCase):
    """Test ChallengeCosts validation."""

    def test_valid_costs_one_time(self):
        """Test valid costs with one-time fee."""
        costs = ChallengeCosts(reset_fee=50.0, one_time_fee=500.0)
        self.assertEqual(costs.reset_fee, 50.0)
        self.assertEqual(costs.one_time_fee, 500.0)
        self.assertIsNone(costs.monthly_fee)

    def test_valid_costs_monthly(self):
        """Test valid costs with monthly fee."""
        costs = ChallengeCosts(reset_fee=50.0, monthly_fee=100.0)
        self.assertEqual(costs.reset_fee, 50.0)
        self.assertEqual(costs.monthly_fee, 100.0)
        self.assertIsNone(costs.one_time_fee)

    def test_both_fees(self):
        """Test that both monthly and one-time fees are allowed."""
        costs = ChallengeCosts(reset_fee=50.0, monthly_fee=100.0, one_time_fee=500.0)
        self.assertEqual(costs.monthly_fee, 100.0)
        self.assertEqual(costs.one_time_fee, 500.0)

    def test_no_fees_raises_error(self):
        """Test that no fees raises error."""
        with self.assertRaises(ValueError):
            ChallengeCosts(reset_fee=50.0)

    def test_negative_reset_fee(self):
        """Test that negative reset_fee raises error."""
        with self.assertRaises(ValueError):
            ChallengeCosts(reset_fee=-10.0, one_time_fee=500.0)

    def test_initial_cost_one_time(self):
        """Test compute_initial_cost with one-time fee."""
        costs = ChallengeCosts(reset_fee=50.0, one_time_fee=500.0)
        self.assertEqual(costs.compute_initial_cost(), 500.0)

    def test_initial_cost_monthly(self):
        """Test compute_initial_cost with monthly fee only."""
        costs = ChallengeCosts(reset_fee=50.0, monthly_fee=100.0)
        self.assertEqual(costs.compute_initial_cost(), 100.0)


class TestSimulationConfig(unittest.TestCase):
    """Test SimulationConfig validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.rules = ChallengeRules(0.10, 0.08, 5)
        self.costs = ChallengeCosts(50.0, one_time_fee=500.0)

    def test_valid_historical_config(self):
        """Test valid historical walk-forward config."""
        config = SimulationConfig(
            method=SimulationMethod.HISTORICAL_WALKFORWARD,
            rules=self.rules,
            costs=self.costs,
            n_simulations=100
        )
        self.assertEqual(config.n_simulations, 100)
        self.assertEqual(config.max_attempts_per_sim, 10)

    def test_valid_monte_carlo_config(self):
        """Test valid Monte Carlo config."""
        config = SimulationConfig(
            method=SimulationMethod.MONTE_CARLO_BLOCK,
            rules=self.rules,
            costs=self.costs,
            n_simulations=100,
            block_length=20
        )
        self.assertEqual(config.block_length, 20)

    def test_monte_carlo_requires_block_length(self):
        """Test that Monte Carlo method requires block_length."""
        with self.assertRaises(ValueError):
            SimulationConfig(
                method=SimulationMethod.MONTE_CARLO_BLOCK,
                rules=self.rules,
                costs=self.costs,
                n_simulations=100
            )

    def test_invalid_n_simulations(self):
        """Test that n_simulations < 1 raises error."""
        with self.assertRaises(ValueError):
            SimulationConfig(
                method=SimulationMethod.HISTORICAL_WALKFORWARD,
                rules=self.rules,
                costs=self.costs,
                n_simulations=0
            )

    def test_get_path_length_default(self):
        """Test default path length calculation."""
        config = SimulationConfig(
            method=SimulationMethod.HISTORICAL_WALKFORWARD,
            rules=self.rules,
            costs=self.costs,
            n_simulations=100
        )
        # Default: max(252, min_trading_days * 5) = max(252, 25) = 252
        self.assertEqual(config.get_path_length(), 252)

    def test_get_path_length_custom(self):
        """Test custom path length."""
        config = SimulationConfig(
            method=SimulationMethod.HISTORICAL_WALKFORWARD,
            rules=self.rules,
            costs=self.costs,
            n_simulations=100,
            path_length=500
        )
        self.assertEqual(config.get_path_length(), 500)


class TestDrawdownFunctions(unittest.TestCase):
    """Test drawdown-related functions."""

    def test_trailing_drawdown_threshold(self):
        """Test trailing drawdown threshold calculation."""
        equity = pd.Series([0.0, 0.02, 0.05, 0.03, 0.08])
        threshold = trailing_drawdown_threshold(equity, 0.05)

        # Peak at each point: [0, 0.02, 0.05, 0.05, 0.08]
        # Threshold: peak - 0.05
        expected = pd.Series([0 - 0.05, 0.02 - 0.05, 0.05 - 0.05, 0.05 - 0.05, 0.08 - 0.05])
        pd.testing.assert_series_equal(threshold, expected)

    def test_daily_drawdown(self):
        """Test daily drawdown is just returns."""
        returns = pd.Series([0.01, -0.02, 0.03, -0.05])
        dd = daily_drawdown(returns)
        pd.testing.assert_series_equal(dd, returns)

    def test_check_drawdown_breach_max_dd(self):
        """Test max drawdown breach detection."""
        equity = pd.Series([0.0, 0.02, -0.05, -0.12])
        breached, idx, reason = check_drawdown_breach(equity, max_drawdown_pct=0.10)

        self.assertTrue(breached)
        self.assertEqual(idx, 3)  # -0.12 < -0.10
        self.assertEqual(reason, "max_drawdown")

    def test_check_drawdown_breach_no_breach(self):
        """Test no breach scenario."""
        equity = pd.Series([0.0, 0.02, 0.05, 0.08])
        breached, idx, reason = check_drawdown_breach(equity, max_drawdown_pct=0.10)

        self.assertFalse(breached)
        self.assertEqual(idx, -1)
        self.assertEqual(reason, "none")

    def test_check_drawdown_breach_trailing(self):
        """Test trailing drawdown breach detection."""
        equity = pd.Series([0.0, 0.10, 0.04, 0.03])  # 0.10 peak, then drops to 0.03 (-7%)
        breached, idx, reason = check_drawdown_breach(
            equity,
            max_drawdown_pct=0.20,
            trailing_drawdown_pct=0.05
        )

        self.assertTrue(breached)
        self.assertEqual(reason, "trailing_drawdown")

    def test_check_drawdown_breach_daily(self):
        """Test daily drawdown breach detection."""
        equity = pd.Series([0.0, 0.02, 0.01, -0.02])
        returns = pd.Series([0.0, 0.02, -0.01, -0.03])

        # -0.03 single day loss exceeds 0.02 daily limit
        breached, idx, reason = check_drawdown_breach(
            equity,
            max_drawdown_pct=0.10,
            returns=returns,
            max_daily_drawdown_pct=0.02
        )

        self.assertTrue(breached)
        self.assertEqual(idx, 3)
        self.assertEqual(reason, "daily_drawdown")


class TestPathGenerators(unittest.TestCase):
    """Test path generator classes."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='B')
        self.returns = pd.Series(
            np.random.normal(0.001, 0.02, 100),
            index=dates,
            name='test_returns'
        )

    def test_historical_generator_length(self):
        """Test historical generator produces correct length."""
        generator = HistoricalWalkForwardGenerator()
        path = generator.generate(self.returns, iteration=0, random_seed=42, path_length=50)

        self.assertEqual(len(path), 50)

    def test_historical_generator_reproducibility(self):
        """Test historical generator is reproducible."""
        generator = HistoricalWalkForwardGenerator()
        path1 = generator.generate(self.returns, iteration=0, random_seed=42, path_length=50)
        path2 = generator.generate(self.returns, iteration=0, random_seed=42, path_length=50)

        pd.testing.assert_series_equal(path1, path2)

    def test_historical_generator_different_iterations(self):
        """Test different iterations produce different paths."""
        generator = HistoricalWalkForwardGenerator()
        path1 = generator.generate(self.returns, iteration=0, random_seed=42, path_length=50)
        path2 = generator.generate(self.returns, iteration=1, random_seed=42, path_length=50)

        # Should be different
        self.assertFalse(np.allclose(path1.values, path2.values))

    def test_monte_carlo_generator_length(self):
        """Test Monte Carlo generator produces correct length."""
        generator = MonteCarloBlockGenerator(block_length=10)
        path = generator.generate(self.returns, iteration=0, random_seed=42, path_length=50)

        self.assertEqual(len(path), 50)

    def test_monte_carlo_generator_reproducibility(self):
        """Test Monte Carlo generator is reproducible."""
        generator = MonteCarloBlockGenerator(block_length=10)
        path1 = generator.generate(self.returns, iteration=0, random_seed=42)
        path2 = generator.generate(self.returns, iteration=0, random_seed=42)

        pd.testing.assert_series_equal(path1, path2)

    def test_monte_carlo_generator_preserves_distribution(self):
        """Test Monte Carlo preserves mean and std roughly."""
        generator = MonteCarloBlockGenerator(block_length=10)
        path = generator.generate(self.returns, iteration=0, random_seed=42)

        # Mean and std should be close (not exact due to resampling)
        self.assertAlmostEqual(path.mean(), self.returns.mean(), delta=0.01)
        self.assertAlmostEqual(path.std(), self.returns.std(), delta=0.01)


class TestPropFirmChallengeSimulator(unittest.TestCase):
    """Test PropFirmChallengeSimulator class."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=252, freq='B')
        self.returns = pd.Series(
            np.random.normal(0.001, 0.015, 252),
            index=dates,
            name='test_returns'
        )

        self.rules = ChallengeRules(
            max_drawdown_pct=0.10,
            profit_target_pct=0.08,
            min_trading_days=5
        )
        self.costs = ChallengeCosts(reset_fee=50.0, one_time_fee=500.0)

    def test_simulator_initialization(self):
        """Test simulator initializes correctly."""
        config = SimulationConfig(
            method=SimulationMethod.HISTORICAL_WALKFORWARD,
            rules=self.rules,
            costs=self.costs,
            n_simulations=10,
            random_seed=42
        )
        simulator = PropFirmChallengeSimulator(self.returns, config)

        self.assertEqual(simulator.config, config)
        pd.testing.assert_series_equal(simulator.returns, self.returns)

    def test_simulator_invalid_returns_type(self):
        """Test simulator rejects non-Series returns."""
        config = SimulationConfig(
            method=SimulationMethod.HISTORICAL_WALKFORWARD,
            rules=self.rules,
            costs=self.costs,
            n_simulations=10
        )

        with self.assertRaises(TypeError):
            PropFirmChallengeSimulator([1, 2, 3], config)

    def test_simulator_invalid_returns_index(self):
        """Test simulator rejects non-DatetimeIndex."""
        config = SimulationConfig(
            method=SimulationMethod.HISTORICAL_WALKFORWARD,
            rules=self.rules,
            costs=self.costs,
            n_simulations=10
        )
        bad_returns = pd.Series([0.01, 0.02, 0.03], index=[1, 2, 3])

        with self.assertRaises(TypeError):
            PropFirmChallengeSimulator(bad_returns, config)

    def test_simulator_run_historical(self):
        """Test running simulation with historical method."""
        config = SimulationConfig(
            method=SimulationMethod.HISTORICAL_WALKFORWARD,
            rules=self.rules,
            costs=self.costs,
            n_simulations=10,
            random_seed=42
        )
        simulator = PropFirmChallengeSimulator(self.returns, config)
        results, stats = simulator.run_simulation()

        self.assertEqual(len(results), 10)
        self.assertEqual(stats.n_simulations, 10)
        self.assertEqual(stats.n_passed + stats.n_failed, 10)

    def test_simulator_run_monte_carlo(self):
        """Test running simulation with Monte Carlo method."""
        config = SimulationConfig(
            method=SimulationMethod.MONTE_CARLO_BLOCK,
            rules=self.rules,
            costs=self.costs,
            n_simulations=10,
            block_length=20,
            random_seed=42
        )
        simulator = PropFirmChallengeSimulator(self.returns, config)
        results, stats = simulator.run_simulation()

        self.assertEqual(len(results), 10)
        self.assertEqual(stats.n_simulations, 10)

    def test_simulator_reproducibility(self):
        """Test simulation is reproducible with same seed."""
        config = SimulationConfig(
            method=SimulationMethod.HISTORICAL_WALKFORWARD,
            rules=self.rules,
            costs=self.costs,
            n_simulations=10,
            random_seed=42
        )

        simulator1 = PropFirmChallengeSimulator(self.returns, config)
        results1, stats1 = simulator1.run_simulation()

        simulator2 = PropFirmChallengeSimulator(self.returns, config)
        results2, stats2 = simulator2.run_simulation()

        self.assertEqual(stats1.pass_probability, stats2.pass_probability)
        self.assertEqual(stats1.n_passed, stats2.n_passed)

    def test_pass_result_structure(self):
        """Test ChallengePass has correct structure."""
        # Create a favorable returns series to ensure pass
        dates = pd.date_range('2024-01-01', periods=100, freq='B')
        favorable_returns = pd.Series(
            [0.02] * 10 + [0.005] * 90,  # 20% in first 10 days
            index=dates
        )

        config = SimulationConfig(
            method=SimulationMethod.HISTORICAL_WALKFORWARD,
            rules=self.rules,
            costs=self.costs,
            n_simulations=1,
            random_seed=42,
            path_length=100
        )
        simulator = PropFirmChallengeSimulator(favorable_returns, config)
        results, _ = simulator.run_simulation()

        # Should have passed
        passes = [r for r in results if r.passed]
        self.assertGreater(len(passes), 0)

        for p in passes:
            self.assertTrue(hasattr(p, 'days_to_pass'))
            self.assertTrue(hasattr(p, 'final_equity'))
            self.assertTrue(hasattr(p, 'max_equity_peak'))
            self.assertTrue(hasattr(p, 'total_cost'))
            self.assertTrue(hasattr(p, 'n_resets'))
            self.assertTrue(hasattr(p, 'equity_curve'))

    def test_fail_result_structure(self):
        """Test ChallengeFail has correct structure."""
        # Create unfavorable returns with consistent losses to ensure fail
        # Use all negative returns so any start point will fail
        dates = pd.date_range('2024-01-01', periods=100, freq='B')
        unfavorable_returns = pd.Series(
            [-0.02] * 100,  # Consistent losses throughout
            index=dates
        )

        # Use Monte Carlo to ensure we get the bad sequence
        config = SimulationConfig(
            method=SimulationMethod.MONTE_CARLO_BLOCK,
            rules=self.rules,
            costs=self.costs,
            n_simulations=5,
            block_length=10,
            random_seed=42,
            path_length=100
        )
        simulator = PropFirmChallengeSimulator(unfavorable_returns, config)
        results, _ = simulator.run_simulation()

        # Should have failed (all returns are negative)
        fails = [r for r in results if not r.passed]
        self.assertGreater(len(fails), 0)

        for f in fails:
            self.assertTrue(hasattr(f, 'days_to_failure'))
            self.assertTrue(hasattr(f, 'failure_reason'))
            self.assertTrue(hasattr(f, 'max_equity_peak'))
            self.assertTrue(hasattr(f, 'equity_at_failure'))
            self.assertTrue(hasattr(f, 'total_cost'))
            self.assertTrue(hasattr(f, 'n_resets'))
            self.assertTrue(hasattr(f, 'equity_curve'))


class TestStatisticsComputation(unittest.TestCase):
    """Test statistics computation functions."""

    def test_compute_statistics_all_pass(self):
        """Test statistics with all passes."""
        passes = [
            ChallengePass(
                days_to_pass=10,
                final_equity=0.10,
                max_equity_peak=0.12,
                total_cost=500.0,
                n_resets=0,
                equity_curve=pd.Series([0.0, 0.05, 0.10])
            )
            for _ in range(10)
        ]

        rules = ChallengeRules(0.10, 0.08, 5)
        costs = ChallengeCosts(50.0, one_time_fee=500.0)
        config = SimulationConfig(
            method=SimulationMethod.HISTORICAL_WALKFORWARD,
            rules=rules,
            costs=costs,
            n_simulations=10
        )

        stats = compute_statistics(passes, config)

        self.assertEqual(stats.pass_probability, 1.0)
        self.assertEqual(stats.fail_probability, 0.0)
        self.assertEqual(stats.n_passed, 10)
        self.assertEqual(stats.n_failed, 0)

    def test_compute_statistics_all_fail(self):
        """Test statistics with all failures."""
        fails = [
            ChallengeFail(
                days_to_failure=5,
                failure_reason="max_drawdown",
                max_equity_peak=0.02,
                equity_at_failure=-0.12,
                total_cost=550.0,
                n_resets=1,
                equity_curve=pd.Series([0.0, 0.02, -0.12])
            )
            for _ in range(10)
        ]

        rules = ChallengeRules(0.10, 0.08, 5)
        costs = ChallengeCosts(50.0, one_time_fee=500.0)
        config = SimulationConfig(
            method=SimulationMethod.HISTORICAL_WALKFORWARD,
            rules=rules,
            costs=costs,
            n_simulations=10
        )

        stats = compute_statistics(fails, config)

        self.assertEqual(stats.pass_probability, 0.0)
        self.assertEqual(stats.fail_probability, 1.0)
        self.assertEqual(stats.n_passed, 0)
        self.assertEqual(stats.n_failed, 10)
        self.assertIn("max_drawdown", stats.failure_reasons)

    def test_compute_statistics_mixed(self):
        """Test statistics with mixed results."""
        passes = [
            ChallengePass(
                days_to_pass=10 + i,
                final_equity=0.10,
                max_equity_peak=0.12,
                total_cost=500.0,
                n_resets=0,
                equity_curve=pd.Series([0.0, 0.05, 0.10])
            )
            for i in range(5)
        ]
        fails = [
            ChallengeFail(
                days_to_failure=5,
                failure_reason="max_drawdown",
                max_equity_peak=0.02,
                equity_at_failure=-0.12,
                total_cost=550.0,
                n_resets=1,
                equity_curve=pd.Series([0.0, 0.02, -0.12])
            )
            for _ in range(5)
        ]

        results = passes + fails

        rules = ChallengeRules(0.10, 0.08, 5)
        costs = ChallengeCosts(50.0, one_time_fee=500.0)
        config = SimulationConfig(
            method=SimulationMethod.HISTORICAL_WALKFORWARD,
            rules=rules,
            costs=costs,
            n_simulations=10
        )

        stats = compute_statistics(results, config)

        self.assertEqual(stats.pass_probability, 0.5)
        self.assertEqual(stats.fail_probability, 0.5)
        self.assertEqual(stats.n_passed, 5)
        self.assertEqual(stats.n_failed, 5)

    def test_compute_statistics_empty(self):
        """Test statistics with empty results."""
        rules = ChallengeRules(0.10, 0.08, 5)
        costs = ChallengeCosts(50.0, one_time_fee=500.0)
        # Use a valid config (n_simulations >= 1) but pass empty results
        config = SimulationConfig(
            method=SimulationMethod.HISTORICAL_WALKFORWARD,
            rules=rules,
            costs=costs,
            n_simulations=1
        )

        # Test compute_statistics with empty list
        stats = compute_statistics([], config)

        self.assertEqual(stats.pass_probability, 0.0)
        self.assertEqual(stats.n_simulations, 0)


if __name__ == '__main__':
    unittest.main()
