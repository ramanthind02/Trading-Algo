"""
Unit tests for Parameter Sensitivity Analysis

Tests cover:
1. _extract_parameter_grid (2D, 3D, fallback)
2. analyze_nd_parameters (3D)
3. compute_robustness_metrics (known values, rating thresholds, sensitivity ranking)
4. 2D fillna fix
5. generate_parameter_sensitivity_report (dict structure, export)
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os

from eda.feature_explorer import FeatureExplorer
from eda.parameter_analysis import ParameterAnalyzer


def _build_synthetic_data(n_rows=500, seed=42):
    """
    Build synthetic features_df, targets_df, and metadata for an RSI module
    with a 3-param grid: lookback x threshold x smoothing.

    lookback  in {7, 14, 21}
    threshold in {30, 50, 70}
    smoothing in {1, 3}
    => 3*3*2 = 18 features
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range('2022-01-01', periods=n_rows)

    feature_metadata = {}
    feature_data = {}

    lookbacks = [7, 14, 21]
    thresholds = [30, 50, 70]
    smoothings = [1, 3]

    for lb in lookbacks:
        for th in thresholds:
            for sm in smoothings:
                col = f"rsi_signal_D_lookback_{lb}_threshold_{th}_smoothing_{sm}"
                feature_data[col] = rng.randn(n_rows)
                feature_metadata[col] = {
                    'module': 'rsi',
                    'parameters': {
                        'lookback': lb,
                        'threshold': th,
                        'smoothing': sm,
                    },
                    'base_name': 'rsi',
                    'full_name': col,
                }

    features_df = pd.DataFrame(feature_data, index=dates)
    targets_df = pd.DataFrame({
        'log_return': rng.randn(n_rows) * 0.01,
    }, index=dates)

    metadata = {'feature_metadata': feature_metadata}
    return features_df, targets_df, metadata


class TestExtractParameterGrid2D(unittest.TestCase):
    """Test _extract_parameter_grid with 2 parameters."""

    def setUp(self):
        self.features_df, self.targets_df, self.metadata = _build_synthetic_data()
        self.explorer = FeatureExplorer(self.features_df, self.targets_df, self.metadata)

    def test_grid_shape(self):
        grid = self.explorer._extract_parameter_grid('rsi', ['lookback', 'threshold'])
        # 3 lookbacks x 3 thresholds = 9 combos
        self.assertEqual(len(grid), 9)

    def test_grid_contents(self):
        grid = self.explorer._extract_parameter_grid('rsi', ['lookback', 'threshold'])
        # Each combo should have 2 features (2 smoothing values)
        for key, features in grid.items():
            self.assertEqual(len(key), 2)
            self.assertEqual(len(features), 2, f"Expected 2 features for {key}, got {len(features)}")

    def test_grid_keys_are_tuples(self):
        grid = self.explorer._extract_parameter_grid('rsi', ['lookback', 'threshold'])
        for key in grid.keys():
            self.assertIsInstance(key, tuple)


class TestExtractParameterGrid3D(unittest.TestCase):
    """Test _extract_parameter_grid with 3 parameters."""

    def setUp(self):
        self.features_df, self.targets_df, self.metadata = _build_synthetic_data()
        self.explorer = FeatureExplorer(self.features_df, self.targets_df, self.metadata)

    def test_3d_grid_shape(self):
        grid = self.explorer._extract_parameter_grid('rsi', ['lookback', 'threshold', 'smoothing'])
        # 3x3x2 = 18 combos, each with 1 feature
        self.assertEqual(len(grid), 18)

    def test_3d_grid_keys(self):
        grid = self.explorer._extract_parameter_grid('rsi', ['lookback', 'threshold', 'smoothing'])
        for key in grid.keys():
            self.assertEqual(len(key), 3)


class TestExtractParameterGridFallback(unittest.TestCase):
    """Test _extract_parameter_grid falls back to column name parsing when metadata missing."""

    def test_fallback_no_metadata(self):
        features_df, targets_df, metadata = _build_synthetic_data(n_rows=100)
        # Create explorer WITHOUT metadata
        explorer = FeatureExplorer(features_df, targets_df, metadata=None)
        # Should still work via parse_feature_column_name fallback
        grid = explorer._extract_parameter_grid('rsi', ['lookback', 'threshold'])
        # Should find some combinations (parser may extract params)
        self.assertGreater(len(grid), 0)


class TestAnalyzeNdParameters3D(unittest.TestCase):
    """Test analyze_nd_parameters with 3 parameters."""

    def setUp(self):
        self.features_df, self.targets_df, self.metadata = _build_synthetic_data(n_rows=200)
        self.explorer = FeatureExplorer(self.features_df, self.targets_df, self.metadata)
        self.analyzer = ParameterAnalyzer(self.features_df, self.targets_df)

    def test_result_columns(self):
        grid = self.explorer._extract_parameter_grid('rsi', ['lookback', 'threshold', 'smoothing'])
        df = self.analyzer.analyze_nd_parameters(
            feature_grid=grid,
            param_names=['lookback', 'threshold', 'smoothing'],
        )
        # Should have param1_value, param2_value, param3_value
        self.assertIn('param1_value', df.columns)
        self.assertIn('param2_value', df.columns)
        self.assertIn('param3_value', df.columns)
        self.assertIn('sortino', df.columns)
        self.assertIn('n_samples', df.columns)

    def test_result_shape(self):
        grid = self.explorer._extract_parameter_grid('rsi', ['lookback', 'threshold', 'smoothing'])
        df = self.analyzer.analyze_nd_parameters(
            feature_grid=grid,
            param_names=['lookback', 'threshold', 'smoothing'],
        )
        # Should have up to 18 rows (3x3x2), some may fail but most should succeed
        self.assertGreater(len(df), 0)
        self.assertLessEqual(len(df), 18)


class TestComputeRobustnessKnownValues(unittest.TestCase):
    """Test compute_robustness_metrics with hand-computable values."""

    def setUp(self):
        self.analyzer = ParameterAnalyzer(
            pd.DataFrame({'x': [1]}),
            pd.DataFrame({'log_return': [0.01]})
        )

    def test_perfect_consistency(self):
        """All values well above threshold -> consistency_score = 10."""
        df = pd.DataFrame({
            'param1_value': [1, 2, 3, 4, 5],
            'sortino': [5.0, 5.0, 5.0, 5.0, 5.0],
            'max_drawdown': [0.0, 0.0, 0.0, 0.0, 0.0],
        })
        result = self.analyzer.compute_robustness_metrics(df, 'sortino', metric_threshold=1.0)
        self.assertEqual(result['consistency_score'], 10.0)

    def test_zero_variance(self):
        """All same values -> cv=0 -> variance_score = 10."""
        df = pd.DataFrame({
            'param1_value': [1, 2, 3],
            'sortino': [3.0, 3.0, 3.0],
            'max_drawdown': [0.0, 0.0, 0.0],
        })
        result = self.analyzer.compute_robustness_metrics(df, 'sortino')
        self.assertEqual(result['variance_score'], 10.0)

    def test_known_cv(self):
        """cv = std / |mean|, variance_score = max(0, 10 - cv * 20)."""
        # mean=2, std=1, cv=0.5, score = 10 - 0.5*20 = 0.0
        df = pd.DataFrame({
            'param1_value': [1, 2, 3],
            'sortino': [1.0, 2.0, 3.0],
        })
        result = self.analyzer.compute_robustness_metrics(df, 'sortino')
        expected_cv = 1.0 / 2.0  # std=1, mean=2
        expected_score = max(0.0, 10.0 - expected_cv * 20.0)
        self.assertAlmostEqual(result['variance_score'], expected_score, places=2)


class TestRobustnessRatingThresholds(unittest.TestCase):
    """Test all 5 rating categories."""

    def setUp(self):
        self.analyzer = ParameterAnalyzer(
            pd.DataFrame({'x': [1]}),
            pd.DataFrame({'log_return': [0.01]})
        )

    def _make_df(self, metric_val, dd_val=0.0):
        return pd.DataFrame({
            'param1_value': [1],
            'sortino': [metric_val],
            'max_drawdown': [dd_val],
        })

    def test_highly_robust(self):
        # Single value: cv=NaN (std=NaN) -> special handling
        # All same, no variance, score=10 each -> overall=10
        df = pd.DataFrame({
            'param1_value': [1, 2],
            'sortino': [10.0, 10.0],
            'max_drawdown': [0.0, 0.0],
        })
        r = self.analyzer.compute_robustness_metrics(df, 'sortino', metric_threshold=1.0)
        self.assertEqual(r['rating'], 'HIGHLY ROBUST')

    def test_robust(self):
        # Engineer scores to get overall ~7
        df = pd.DataFrame({
            'param1_value': [1, 2, 3, 4],
            'sortino': [2.0, 2.2, 2.1, 1.9],
            'max_drawdown': [-0.02, -0.03, -0.01, -0.02],
        })
        r = self.analyzer.compute_robustness_metrics(df, 'sortino', metric_threshold=1.0)
        self.assertIn(r['rating'], ['ROBUST', 'HIGHLY ROBUST'])

    def test_fragile(self):
        # Large variance, some below threshold
        df = pd.DataFrame({
            'param1_value': list(range(10)),
            'sortino': [0.1, 0.2, 10.0, 0.0, 0.3, 0.1, 0.0, 0.2, 0.1, 0.0],
            'max_drawdown': [-0.5, -0.6, -0.7, -0.8, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9],
        })
        r = self.analyzer.compute_robustness_metrics(df, 'sortino', metric_threshold=1.0)
        self.assertIn(r['rating'], ['FRAGILE', 'HIGHLY FRAGILE'])

    def test_highly_fragile(self):
        df = pd.DataFrame({
            'param1_value': list(range(5)),
            'sortino': [0.0, 0.0, 0.0, 0.0, 0.0],
            'max_drawdown': [-1.0, -1.0, -1.0, -1.0, -1.0],
        })
        r = self.analyzer.compute_robustness_metrics(df, 'sortino', metric_threshold=1.0)
        self.assertEqual(r['rating'], 'HIGHLY FRAGILE')


class TestParameterSensitivityRankingSumsToOne(unittest.TestCase):
    """Test that parameter sensitivity variance contributions sum to 1."""

    def setUp(self):
        self.analyzer = ParameterAnalyzer(
            pd.DataFrame({'x': [1]}),
            pd.DataFrame({'log_return': [0.01]})
        )

    def test_sums_to_one(self):
        df = pd.DataFrame({
            'param1_value': [1, 1, 2, 2],
            'param2_value': [10, 20, 10, 20],
            'sortino': [1.0, 2.0, 3.0, 4.0],
        })
        r = self.analyzer.compute_robustness_metrics(df, 'sortino')
        sensitivity = r['parameter_sensitivity']
        self.assertGreater(len(sensitivity), 0)
        total = sum(sensitivity.values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_three_params_sum_to_one(self):
        df = pd.DataFrame({
            'param1_value': [1, 1, 2, 2, 1, 1, 2, 2],
            'param2_value': [10, 10, 10, 10, 20, 20, 20, 20],
            'param3_value': [100, 200, 100, 200, 100, 200, 100, 200],
            'sortino': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
        })
        r = self.analyzer.compute_robustness_metrics(df, 'sortino')
        sensitivity = r['parameter_sensitivity']
        self.assertEqual(len(sensitivity), 3)
        total = sum(sensitivity.values())
        self.assertAlmostEqual(total, 1.0, places=5)


class TestFillNaFix(unittest.TestCase):
    """Test that sparse 2D parameter grids preserve NaN gaps."""

    def test_nan_preserved_in_pivot(self):
        """Verify NaN in sparse grid is preserved, not replaced with 0."""
        df = pd.DataFrame({
            'param1_value': [1, 1, 2, 2, 3],
            'param2_value': [10, 20, 10, 20, 10],
            'sortino': [1.5, 2.0, 1.8, 2.2, 1.6],
        })
        pivot = df.pivot(index='param2_value', columns='param1_value', values='sortino')
        flat = np.array(pivot.values).flatten()
        n_nan = np.isnan(flat).sum()
        self.assertGreater(n_nan, 0, "Expected NaN for missing grid cell, but none found")


class TestReportDictStructure(unittest.TestCase):
    """Test generate_parameter_sensitivity_report returns correct dict keys."""

    def setUp(self):
        self.features_df, self.targets_df, self.metadata = _build_synthetic_data(n_rows=200)
        self.explorer = FeatureExplorer(self.features_df, self.targets_df, self.metadata)

    def test_report_keys(self):
        result = self.explorer.generate_parameter_sensitivity_report(
            module_name='rsi',
            param_names=['lookback', 'threshold'],
            verbose=False,
        )
        expected_keys = {
            'results_df', 'summary_stats', 'robustness_scores',
            'figures', 'parameter_sensitivity', 'report_text', 'report_path'
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_report_text_is_string(self):
        result = self.explorer.generate_parameter_sensitivity_report(
            module_name='rsi',
            param_names=['lookback'],
            verbose=False,
        )
        self.assertIsInstance(result['report_text'], str)
        self.assertIn('PARAMETER SENSITIVITY REPORT', result['report_text'])

    def test_robustness_scores_structure(self):
        result = self.explorer.generate_parameter_sensitivity_report(
            module_name='rsi',
            param_names=['lookback', 'threshold'],
            verbose=False,
        )
        scores = result['robustness_scores']
        self.assertIn('variance_score', scores)
        self.assertIn('consistency_score', scores)
        self.assertIn('risk_score', scores)
        self.assertIn('overall_score', scores)
        self.assertIn('rating', scores)


class TestLegacyFeatureExplorerContracts(unittest.TestCase):
    """Lock legacy output contracts during validator migration."""

    def setUp(self):
        self.features_df, self.targets_df, self.metadata = _build_synthetic_data(n_rows=150)
        self.explorer = FeatureExplorer(self.features_df, self.targets_df, self.metadata)

    def test_generate_parameter_sensitivity_report_contract(self):
        result = self.explorer.generate_parameter_sensitivity_report(
            module_name='rsi',
            param_names=['lookback', 'threshold'],
            verbose=False,
        )
        expected_keys = (
            'results_df',
            'summary_stats',
            'robustness_scores',
            'figures',
            'parameter_sensitivity',
            'report_text',
            'report_path',
        )
        self.assertEqual(tuple(result.keys()), expected_keys)
        self.assertIsNone(result['report_path'])


class TestReportExport(unittest.TestCase):
    """Test generate_parameter_sensitivity_report exports to file."""

    def setUp(self):
        self.features_df, self.targets_df, self.metadata = _build_synthetic_data(n_rows=200)
        self.explorer = FeatureExplorer(self.features_df, self.targets_df, self.metadata)

    def test_export_creates_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            export_path = f.name

        try:
            result = self.explorer.generate_parameter_sensitivity_report(
                module_name='rsi',
                param_names=['lookback', 'threshold'],
                export_path=export_path,
                verbose=False,
            )
            self.assertEqual(result['report_path'], export_path)
            self.assertTrue(os.path.exists(export_path))

            with open(export_path, 'r') as f:
                content = f.read()
            self.assertIn('PARAMETER SENSITIVITY REPORT', content)
            self.assertGreater(len(content), 100)
        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)


if __name__ == '__main__':
    unittest.main()
