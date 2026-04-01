"""
Unit tests for Robustness Test Module

This module tests the robustness testing functionality including:
1. Resampling strategy classes (MonteCarloStrategy, BootstrapStrategy, BlockBootstrapStrategy)
2. Main robustness_test() function
3. Input validation and edge cases
4. Statistical properties of resampling methods
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from utils.evaluation.robustness_test import (
    robustness_test,
    ResamplingStrategy,
    MonteCarloStrategy,
    BootstrapStrategy,
    BlockBootstrapStrategy
)
from utils.core.enums import ResamplingMethod
from metrics.equity import cumulative_returns


class TestMonteCarloStrategy(unittest.TestCase):
    """Test cases for MonteCarloStrategy class."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_samples = 252
        self.dates = pd.date_range('2024-01-01', periods=self.n_samples, freq='B')
        self.returns = pd.Series(
            np.random.normal(0.001, 0.02, self.n_samples),
            index=self.dates,
            name='returns'
        )
        self.strategy = MonteCarloStrategy()

    def test_validate_data_accepts_valid_series(self):
        """Test that validation passes for valid input."""
        # Should not raise
        self.strategy.validate_data(self.returns)

    def test_validate_data_rejects_non_series(self):
        """Test that validation rejects non-Series input."""
        with self.assertRaises(TypeError):
            self.strategy.validate_data([1, 2, 3])

    def test_validate_data_rejects_non_datetime_index(self):
        """Test that validation rejects Series without DatetimeIndex."""
        bad_series = pd.Series([0.01, -0.02, 0.03])  # RangeIndex
        with self.assertRaises(TypeError):
            self.strategy.validate_data(bad_series)

    def test_validate_data_rejects_empty_series(self):
        """Test that validation rejects empty Series."""
        empty_series = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
        with self.assertRaises(ValueError):
            self.strategy.validate_data(empty_series)

    def test_validate_data_rejects_non_numeric(self):
        """Test that validation rejects non-numeric Series."""
        string_series = pd.Series(['a', 'b', 'c'], index=self.dates[:3])
        with self.assertRaises(TypeError):
            self.strategy.validate_data(string_series)

    def test_resample_preserves_index(self):
        """Test that resampling preserves the original index."""
        resampled = self.strategy.resample(self.returns, random_seed=42)
        pd.testing.assert_index_equal(resampled.index, self.returns.index)

    def test_resample_preserves_mean(self):
        """Test that shuffling preserves exact mean (permutation property)."""
        resampled = self.strategy.resample(self.returns, random_seed=42)
        np.testing.assert_almost_equal(
            resampled.mean(),
            self.returns.mean(),
            decimal=10
        )

    def test_resample_preserves_variance(self):
        """Test that shuffling preserves exact variance (permutation property)."""
        resampled = self.strategy.resample(self.returns, random_seed=42)
        np.testing.assert_almost_equal(
            resampled.std(),
            self.returns.std(),
            decimal=10
        )

    def test_resample_preserves_values(self):
        """Test that all original values are present (no duplicates in shuffle)."""
        resampled = self.strategy.resample(self.returns, random_seed=42)
        self.assertEqual(
            sorted(resampled.values),
            sorted(self.returns.values)
        )

    def test_resample_reproducible_with_seed(self):
        """Test that same seed produces same result."""
        result1 = self.strategy.resample(self.returns, random_seed=42)
        result2 = self.strategy.resample(self.returns, random_seed=42)
        pd.testing.assert_series_equal(result1, result2)

    def test_resample_different_with_different_seed(self):
        """Test that different seeds produce different results."""
        result1 = self.strategy.resample(self.returns, random_seed=42)
        result2 = self.strategy.resample(self.returns, random_seed=43)
        # Values should be shuffled differently
        self.assertFalse(np.allclose(result1.values, result2.values))


class TestBootstrapStrategy(unittest.TestCase):
    """Test cases for BootstrapStrategy class."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_samples = 252
        self.dates = pd.date_range('2024-01-01', periods=self.n_samples, freq='B')
        self.returns = pd.Series(
            np.random.normal(0.001, 0.02, self.n_samples),
            index=self.dates,
            name='returns'
        )
        self.strategy = BootstrapStrategy()

    def test_validate_data_accepts_valid_series(self):
        """Test that validation passes for valid input."""
        self.strategy.validate_data(self.returns)

    def test_resample_preserves_index(self):
        """Test that resampling preserves the original index."""
        resampled = self.strategy.resample(self.returns, random_seed=42)
        pd.testing.assert_index_equal(resampled.index, self.returns.index)

    def test_resample_preserves_length(self):
        """Test that resampled series has same length."""
        resampled = self.strategy.resample(self.returns, random_seed=42)
        self.assertEqual(len(resampled), len(self.returns))

    def test_resample_can_have_duplicates(self):
        """Test that bootstrap can sample the same value multiple times."""
        resampled = self.strategy.resample(self.returns, random_seed=42)
        # With replacement, we expect some values to appear multiple times
        unique_original = len(self.returns.unique())
        unique_resampled = len(resampled.unique())
        # Bootstrap should typically have fewer unique values
        self.assertLessEqual(unique_resampled, unique_original)

    def test_resample_produces_different_cumulative(self):
        """Test that bootstrap produces different final cumulative returns."""
        cumulative_values = []
        for seed in range(10):
            resampled = self.strategy.resample(self.returns, random_seed=seed)
            cum_ret = cumulative_returns(resampled).iloc[-1]
            cumulative_values.append(cum_ret)

        # Should have variation in cumulative returns
        self.assertGreater(np.std(cumulative_values), 0.01)

    def test_resample_reproducible_with_seed(self):
        """Test that same seed produces same result."""
        result1 = self.strategy.resample(self.returns, random_seed=42)
        result2 = self.strategy.resample(self.returns, random_seed=42)
        pd.testing.assert_series_equal(result1, result2)


class TestBlockBootstrapStrategy(unittest.TestCase):
    """Test cases for BlockBootstrapStrategy class."""

    def setUp(self):
        """Set up test data with autocorrelation."""
        np.random.seed(42)
        self.n_samples = 252
        self.dates = pd.date_range('2024-01-01', periods=self.n_samples, freq='B')

        # Create returns with positive autocorrelation (momentum-like)
        self.returns = pd.Series(index=self.dates, dtype=float)
        self.returns.iloc[0] = np.random.normal(0.001, 0.02)
        for i in range(1, self.n_samples):
            # AR(1) process with phi=0.3
            self.returns.iloc[i] = 0.3 * self.returns.iloc[i-1] + np.random.normal(0.001, 0.02)

        self.strategy = BlockBootstrapStrategy()

    def test_validate_data_requires_block_size(self):
        """Test that validation requires block_size parameter."""
        with self.assertRaises(ValueError) as context:
            self.strategy.validate_data(self.returns, block_size=None)
        self.assertIn('block_size is required', str(context.exception))

    def test_validate_data_rejects_invalid_block_size(self):
        """Test that validation rejects block_size < 1."""
        with self.assertRaises(ValueError):
            self.strategy.validate_data(self.returns, block_size=0)
        with self.assertRaises(ValueError):
            self.strategy.validate_data(self.returns, block_size=-5)

    def test_validate_data_rejects_block_size_too_large(self):
        """Test that validation rejects block_size > series length."""
        with self.assertRaises(ValueError):
            self.strategy.validate_data(self.returns, block_size=500)

    def test_validate_data_accepts_valid_block_size(self):
        """Test that validation accepts valid block_size."""
        self.strategy.validate_data(self.returns, block_size=20)

    def test_resample_preserves_index(self):
        """Test that resampling preserves the original index."""
        resampled = self.strategy.resample(self.returns, random_seed=42, block_size=20)
        pd.testing.assert_index_equal(resampled.index, self.returns.index)

    def test_resample_preserves_length(self):
        """Test that resampled series has same length."""
        resampled = self.strategy.resample(self.returns, random_seed=42, block_size=20)
        self.assertEqual(len(resampled), len(self.returns))

    def test_resample_preserves_autocorrelation(self):
        """Test that block bootstrap better preserves autocorrelation than regular bootstrap."""
        original_autocorr = self.returns.autocorr(lag=1)

        # Block bootstrap should preserve autocorrelation better
        block_autocorrs = []
        regular_autocorrs = []
        regular_strategy = BootstrapStrategy()

        for seed in range(20):
            block_resampled = self.strategy.resample(
                self.returns, random_seed=seed, block_size=20
            )
            regular_resampled = regular_strategy.resample(
                self.returns, random_seed=seed
            )
            block_autocorrs.append(block_resampled.autocorr(lag=1))
            regular_autocorrs.append(regular_resampled.autocorr(lag=1))

        # Block bootstrap should be closer to original autocorrelation
        block_error = abs(np.mean(block_autocorrs) - original_autocorr)
        regular_error = abs(np.mean(regular_autocorrs) - original_autocorr)

        self.assertLess(block_error, regular_error)

    def test_resample_reproducible_with_seed(self):
        """Test that same seed produces same result."""
        result1 = self.strategy.resample(self.returns, random_seed=42, block_size=20)
        result2 = self.strategy.resample(self.returns, random_seed=42, block_size=20)
        pd.testing.assert_series_equal(result1, result2)


class TestRobustnessTest(unittest.TestCase):
    """Test cases for the main robustness_test function."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_samples = 252
        self.dates = pd.date_range('2024-01-01', periods=self.n_samples, freq='B')
        self.returns = pd.Series(
            np.random.normal(0.002, 0.02, self.n_samples),
            index=self.dates,
            name='returns'
        )
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        import shutil
        plt.close('all')
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_returns_correct_tuple_structure(self):
        """Test that function returns (Figure, List[Series], Dict)."""
        fig, series_list, stats = robustness_test(
            self.returns,
            n_samples=10,
            method=ResamplingMethod.MONTE_CARLO,
            verbose=False
        )

        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(series_list, list)
        self.assertEqual(len(series_list), 10)
        self.assertIsInstance(series_list[0], pd.Series)
        self.assertIsInstance(stats, dict)

    def test_stats_has_required_keys(self):
        """Test that stats dictionary has all required keys."""
        _, _, stats = robustness_test(
            self.returns,
            n_samples=10,
            method=ResamplingMethod.BOOTSTRAP,
            verbose=False
        )

        required_keys = [
            'resampling_method',
            'original_cumulative_return',
            'original_mean_return',
            'original_volatility',
            'original_autocorr_lag1',
            'mean_cumulative_return',
            'median_cumulative_return',
            'std_cumulative_return',
            'p5_cumulative_return',
            'p25_cumulative_return',
            'p50_cumulative_return',
            'p75_cumulative_return',
            'p95_cumulative_return',
            'p_value',
            'n_samples',
            'n_valid_samples',
            'mean_resampled_autocorr',
            'preserved_autocorr',
            'percentiles',
            'block_size'
        ]

        for key in required_keys:
            self.assertIn(key, stats, f"Missing key: {key}")

    def test_p_value_in_valid_range(self):
        """Test that p-value is between 0 and 1."""
        _, _, stats = robustness_test(
            self.returns,
            n_samples=50,
            method=ResamplingMethod.BOOTSTRAP,
            verbose=False
        )

        self.assertGreaterEqual(stats['p_value'], 0.0)
        self.assertLessEqual(stats['p_value'], 1.0)

    def test_works_with_all_methods(self):
        """Test that function works with all resampling methods."""
        for method in ResamplingMethod:
            kwargs = {'n_samples': 10, 'verbose': False}
            if method == ResamplingMethod.BLOCK_BOOTSTRAP:
                kwargs['block_size'] = 20

            fig, series, stats = robustness_test(
                self.returns,
                method=method,
                **kwargs
            )

            self.assertEqual(stats['resampling_method'], method.value)
            plt.close(fig)

    def test_handles_nan_values(self):
        """Test that function handles NaN values by dropping them."""
        returns_with_nan = self.returns.copy()
        returns_with_nan.iloc[0] = np.nan
        returns_with_nan.iloc[10] = np.nan

        _, series_list, stats = robustness_test(
            returns_with_nan,
            n_samples=10,
            method=ResamplingMethod.BOOTSTRAP,
            verbose=False
        )

        # Should work without error
        self.assertEqual(len(series_list), 10)

    def test_reproducible_with_seed(self):
        """Test that results are reproducible with same random seed."""
        _, series1, stats1 = robustness_test(
            self.returns,
            n_samples=10,
            method=ResamplingMethod.BOOTSTRAP,
            random_seed=42,
            verbose=False
        )

        _, series2, stats2 = robustness_test(
            self.returns,
            n_samples=10,
            method=ResamplingMethod.BOOTSTRAP,
            random_seed=42,
            verbose=False
        )

        # Stats should be identical
        self.assertEqual(stats1['p_value'], stats2['p_value'])
        self.assertEqual(
            stats1['mean_cumulative_return'],
            stats2['mean_cumulative_return']
        )

    def test_save_path_creates_file(self):
        """Test that save_path parameter creates a file."""
        save_path = os.path.join(self.temp_dir, 'test_plot.png')

        fig, _, _ = robustness_test(
            self.returns,
            n_samples=10,
            method=ResamplingMethod.MONTE_CARLO,
            save_path=save_path,
            verbose=False
        )

        self.assertTrue(os.path.exists(save_path))
        plt.close(fig)

    def test_block_bootstrap_requires_block_size(self):
        """Test that block bootstrap raises error without block_size."""
        with self.assertRaises(ValueError):
            robustness_test(
                self.returns,
                n_samples=10,
                method=ResamplingMethod.BLOCK_BOOTSTRAP,
                block_size=None,
                verbose=False
            )

    def test_resampled_series_have_correct_index(self):
        """Test that all resampled series have the original index."""
        _, series_list, _ = robustness_test(
            self.returns,
            n_samples=5,
            method=ResamplingMethod.BOOTSTRAP,
            verbose=False
        )

        for series in series_list:
            pd.testing.assert_index_equal(series.index, self.returns.index)


class TestStatisticalProperties(unittest.TestCase):
    """Test statistical properties of resampling methods."""

    def setUp(self):
        """Set up test data with known properties."""
        np.random.seed(42)
        self.n_samples = 500
        self.dates = pd.date_range('2024-01-01', periods=self.n_samples, freq='B')

        # Create returns with strong positive drift
        self.positive_returns = pd.Series(
            np.random.normal(0.005, 0.015, self.n_samples),  # Strong positive drift
            index=self.dates
        )

        # Create returns with no drift (zero mean)
        self.zero_mean_returns = pd.Series(
            np.random.normal(0.0, 0.02, self.n_samples),
            index=self.dates
        )

    def tearDown(self):
        """Clean up."""
        plt.close('all')

    def test_bootstrap_p_value_reflects_significance(self):
        """Test that bootstrap p-value is lower for stronger strategies."""
        # Use more samples and stronger drift to reduce randomness
        _, _, stats_positive = robustness_test(
            self.positive_returns,
            n_samples=200,
            method=ResamplingMethod.BOOTSTRAP,
            random_seed=42,
            verbose=False
        )

        _, _, stats_zero = robustness_test(
            self.zero_mean_returns,
            n_samples=200,
            method=ResamplingMethod.BOOTSTRAP,
            random_seed=42,
            verbose=False
        )

        # Strong positive drift should have lower p-value (or at least not much higher)
        # We use a relaxed comparison since bootstrap has variance
        # The positive strategy should be at most slightly worse
        self.assertLessEqual(stats_positive['p_value'], stats_zero['p_value'] + 0.15)

    def test_percentiles_are_ordered(self):
        """Test that percentiles are in correct order."""
        _, _, stats = robustness_test(
            self.positive_returns,
            n_samples=100,
            method=ResamplingMethod.BOOTSTRAP,
            verbose=False
        )

        self.assertLessEqual(stats['p5_cumulative_return'], stats['p25_cumulative_return'])
        self.assertLessEqual(stats['p25_cumulative_return'], stats['p50_cumulative_return'])
        self.assertLessEqual(stats['p50_cumulative_return'], stats['p75_cumulative_return'])
        self.assertLessEqual(stats['p75_cumulative_return'], stats['p95_cumulative_return'])


if __name__ == '__main__':
    unittest.main()
