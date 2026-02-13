"""
Unit tests for WeightLayer and InverseCorrelationWeighter.

Tests cover:
- Inverse correlation weight calculation
- FDM calculation from forecast correlations
- Combining forecasts with weights and FDM
- Edge cases (single model, empty data, etc.)
"""

import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import pandas as pd
import numpy as np


_WEIGHT_LAYER_PATH = Path(__file__).resolve().parents[1] / "ensemble" / "weight_layer.py"
_WEIGHT_LAYER_SPEC = spec_from_file_location("weight_layer_module", _WEIGHT_LAYER_PATH)
if _WEIGHT_LAYER_SPEC is None or _WEIGHT_LAYER_SPEC.loader is None:
    raise RuntimeError(f"Unable to load weight_layer module from {_WEIGHT_LAYER_PATH}")
_WEIGHT_LAYER_MODULE = module_from_spec(_WEIGHT_LAYER_SPEC)
_WEIGHT_LAYER_SPEC.loader.exec_module(_WEIGHT_LAYER_MODULE)

WeightLayer = _WEIGHT_LAYER_MODULE.WeightLayer
InverseCorrelationWeighter = _WEIGHT_LAYER_MODULE.InverseCorrelationWeighter


class TestInverseCorrelationWeighter(unittest.TestCase):
    """Tests for InverseCorrelationWeighter class."""

    def test_single_model_weight(self):
        """Single model should have weight of 1.0."""
        weighter = InverseCorrelationWeighter()
        signals = pd.DataFrame({'model_a': [1, 0, 1, 0, 1]})

        weighter.fit(signals)
        weights = weighter.get_weights()

        self.assertEqual(len(weights), 1)
        self.assertAlmostEqual(weights['model_a'], 1.0)

    def test_uncorrelated_models_equal_weights(self):
        """Uncorrelated models should have approximately equal weights."""
        weighter = InverseCorrelationWeighter()
        np.random.seed(42)

        # Create uncorrelated signals
        signals = pd.DataFrame({
            'model_a': np.random.randint(0, 2, 100),
            'model_b': np.random.randint(0, 2, 100),
            'model_c': np.random.randint(0, 2, 100)
        })

        weighter.fit(signals)
        weights = weighter.get_weights()

        # Weights should sum to 1.0
        self.assertAlmostEqual(weights.sum(), 1.0, places=5)

        # Each weight should be roughly 1/3 for uncorrelated models
        for weight in weights.values:
            self.assertTrue(0.2 <= weight <= 0.5)

    def test_correlated_models_different_weights(self):
        """Correlated models should get lower weights."""
        weighter = InverseCorrelationWeighter()

        # model_a and model_b are highly correlated
        # model_c is uncorrelated
        signals = pd.DataFrame({
            'model_a': [1, 0, 1, 0, 1, 0, 1, 0] * 10,
            'model_b': [1, 0, 1, 0, 1, 0, 1, 0] * 10,  # Same as model_a
            'model_c': [0, 1, 0, 1, 1, 0, 0, 1] * 10   # Different pattern
        })

        weighter.fit(signals)
        weights = weighter.get_weights()

        # model_c should have higher weight (less correlated with others)
        self.assertTrue(weights['model_c'] > weights['model_a'])
        self.assertTrue(weights['model_c'] > weights['model_b'])

    def test_fit_raises_on_empty_signals(self):
        """Should raise ValueError on empty signals."""
        weighter = InverseCorrelationWeighter()
        signals = pd.DataFrame()

        with self.assertRaises(ValueError):
            weighter.fit(signals)

    def test_fit_raises_on_single_sample(self):
        """Should raise ValueError with insufficient data."""
        weighter = InverseCorrelationWeighter()
        signals = pd.DataFrame({'model_a': [1]})

        with self.assertRaises(ValueError):
            weighter.fit(signals)

    def test_get_weights_before_fit_raises(self):
        """Should raise ValueError if getting weights before fit."""
        weighter = InverseCorrelationWeighter()

        with self.assertRaises(ValueError):
            weighter.get_weights()


class TestWeightLayer(unittest.TestCase):
    """Tests for WeightLayer class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample forecast vectors
        self.forecast_vectors = [
            pd.DataFrame({
                'ticker': ['ES', 'ES', 'NQ', 'NQ'],
                'model_name': ['model_a', 'model_b', 'model_a', 'model_b'],
                'forecast': [1.0, 0.5, 0.8, 0.6],
                'signal': [1, 1, 1, 1]
            })
        ]

        # Create sample signals for weight calculation
        self.signals = pd.DataFrame({
            'model_a': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            'model_b': [0, 1, 1, 0, 0, 1, 0, 1, 1, 0]
        })

    def test_fit_and_combine(self):
        """Test basic fit and combine workflow."""
        layer = WeightLayer(fdm_max=2.0)
        layer.fit(self.forecast_vectors, self.signals)

        result = layer.combine(self.forecast_vectors)

        # Check output structure
        self.assertIn('ticker', result.columns)
        self.assertIn('forecast_score', result.columns)

        # Should have one row per unique ticker
        self.assertEqual(len(result), 2)  # ES and NQ

    def test_fdm_calculation_single_model(self):
        """Single model should have FDM of 1.0."""
        layer = WeightLayer()

        single_model_forecasts = [
            pd.DataFrame({
                'ticker': ['ES', 'NQ'],
                'model_name': ['model_a', 'model_a'],
                'forecast': [1.0, 0.8],
                'signal': [1, 1]
            })
        ]

        single_model_signals = pd.DataFrame({
            'model_a': [1, 0, 1, 0, 1]
        })

        layer.fit(single_model_forecasts, single_model_signals)

        self.assertEqual(set(layer.fdm_.keys()), {'ES', 'NQ'})
        self.assertTrue(all(np.isclose(fdm, 1.0) for fdm in layer.fdm_.values()))

    def test_fdm_capped_at_max(self):
        """FDM should be capped at fdm_max."""
        layer = WeightLayer(fdm_max=2.0)

        # Create uncorrelated forecasts that would give high FDM
        np.random.seed(42)
        forecasts = [
            pd.DataFrame({
                'ticker': ['ES'] * 100,
                'model_name': ['model_a'] * 50 + ['model_b'] * 50,
                'forecast': list(np.random.randn(100)),
                'signal': [1] * 100
            })
        ]

        signals = pd.DataFrame({
            'model_a': np.random.randint(0, 2, 50),
            'model_b': np.random.randint(0, 2, 50)
        })

        layer.fit(forecasts, signals)

        self.assertIn('ES', layer.fdm_)
        self.assertLessEqual(layer.fdm_['ES'], 2.0)

    def test_combine_applies_fdm(self):
        """Combined forecast should include FDM scaling."""
        layer = WeightLayer(fdm_max=2.0)
        layer.fit(self.forecast_vectors, self.signals)

        result = layer.combine(self.forecast_vectors)

        # The forecast_score should be FDM-scaled
        # Verify it's not zero and reasonable
        self.assertTrue(all(result['forecast_score'] > 0))

    def test_combine_before_fit_raises(self):
        """Should raise error if combining before fit."""
        layer = WeightLayer()

        with self.assertRaises(ValueError):
            layer.combine(self.forecast_vectors)

    def test_diagnostics(self):
        """Test get_diagnostics method."""
        layer = WeightLayer(fdm_max=2.0)
        layer.fit(self.forecast_vectors, self.signals)

        diag = layer.get_diagnostics()

        self.assertTrue(diag['is_fitted'])
        self.assertIn('tickers', diag)
        self.assertIn('summary', diag)
        self.assertIn('ES', diag['tickers'])
        self.assertIn('NQ', diag['tickers'])
        self.assertEqual(diag['tickers']['ES']['n_models'], 2)

    def test_invalid_weight_method(self):
        """Should raise on invalid weight method."""
        with self.assertRaises(ValueError):
            WeightLayer(weight_method='invalid_method')

    def test_sortino_weight_method_removed(self):
        """Should reject sortino method and only allow inverse correlation."""
        with self.assertRaises(ValueError):
            WeightLayer(weight_method='sortino_optimized')

    def test_empty_forecast_vectors(self):
        """Should handle empty forecast vectors."""
        layer = WeightLayer()

        with self.assertRaises(ValueError):
            layer.fit([], self.signals)

    def test_combine_with_empty_input(self):
        """Combining empty forecast should return empty DataFrame."""
        layer = WeightLayer()
        layer.fit(self.forecast_vectors, self.signals)

        result = layer.combine([])

        self.assertEqual(len(result), 0)
        self.assertIn('ticker', result.columns)
        self.assertIn('forecast_score', result.columns)


class TestWeightLayerIntegration(unittest.TestCase):
    """Integration tests for WeightLayer with realistic data."""

    def test_multi_ticker_multi_model(self):
        """Test with multiple tickers and models."""
        layer = WeightLayer(fdm_max=2.0)

        # Create realistic forecast data
        forecasts = [
            pd.DataFrame({
                'ticker': ['ES', 'ES', 'ES', 'NQ', 'NQ', 'NQ', 'GC', 'GC', 'GC'],
                'model_name': ['ewmac_8', 'ewmac_16', 'rsi_14'] * 3,
                'forecast': [1.0, 0.5, 0.8, 1.2, 0.6, 0.9, 0.7, 0.4, 0.5],
                'signal': [1, 1, 1, 1, 1, 0, 1, 0, 1]
            })
        ]

        signals = pd.DataFrame({
            'ewmac_8': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            'ewmac_16': [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            'rsi_14': [0, 1, 1, 0, 0, 1, 1, 0, 0, 1]
        })

        layer.fit(forecasts, signals)
        result = layer.combine(forecasts)

        # Should have one row per ticker
        self.assertEqual(len(result), 3)
        self.assertEqual(set(result['ticker']), {'ES', 'NQ', 'GC'})

        # All forecast scores should be non-negative
        self.assertTrue(all(result['forecast_score'] >= 0))


if __name__ == '__main__':
    unittest.main()
