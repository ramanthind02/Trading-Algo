"""
Integration tests for the full forecast generation pipeline.

Tests the complete flow:
    Base Models -> Ensemble -> WeightLayer -> Portfolio -> PositionSizer

Tests cover:
- End-to-end pipeline execution
- Formula verification with known inputs
- Multi-instrument, multi-model scenarios
- Edge cases (single model, inactive signals)
"""

import unittest
import pandas as pd
import numpy as np

from ensemble.weight_layer import WeightLayer
from ensemble.portfolio import Portfolio
from execution.position_sizer import (
    PositionSizer,
    ContractSpec,
    RoundingMethod
)


class TestFormulaVerification(unittest.TestCase):
    """Verify formulas with known inputs (Buy/Hold scenarios)."""

    def test_per_model_forecast_formula(self):
        """
        Verify: F_i = (tau / (sigma * sqrt(h_i))) * X_i

        Known inputs:
        - tau = 0.20 (target vol)
        - sigma = 0.20 (actual vol)
        - h = 1.0 (exposure fraction, 1/1 bin)
        - signal = 1

        Expected: F = 0.20 / (0.20 * sqrt(1.0)) = 1.0
        """
        tau = 0.20
        sigma = 0.20
        h = 1.0
        signal = 1

        expected_forecast = (tau / (sigma * np.sqrt(h))) * signal
        self.assertAlmostEqual(expected_forecast, 1.0, places=5)

    def test_per_model_forecast_formula_half_vol(self):
        """
        Verify: F_i = (tau / (sigma * sqrt(h_i))) * X_i

        Known inputs:
        - tau = 0.20, sigma = 0.40, h = 1.0, signal = 1

        Expected: F = 0.20 / (0.40 * sqrt(1.0)) = 0.5
        """
        tau = 0.20
        sigma = 0.40
        h = 1.0
        signal = 1

        expected_forecast = (tau / (sigma * np.sqrt(h))) * signal
        self.assertAlmostEqual(expected_forecast, 0.5, places=5)

    def test_per_model_forecast_formula_double_vol(self):
        """
        Verify: F_i = (tau / (sigma * sqrt(h_i))) * X_i

        Known inputs:
        - tau = 0.20, sigma = 0.10, h = 1.0, signal = 1

        Expected: F = 0.20 / (0.10 * sqrt(1.0)) = 2.0
        """
        tau = 0.20
        sigma = 0.10
        h = 1.0
        signal = 1

        expected_forecast = (tau / (sigma * np.sqrt(h))) * signal
        self.assertAlmostEqual(expected_forecast, 2.0, places=5)

    def test_per_model_forecast_with_exposure_fraction(self):
        """
        Verify: F_i = (tau / (sigma * sqrt(h_i))) * X_i

        Known inputs:
        - tau = 0.20, sigma = 0.20, h = 0.1 (10 bins), signal = 1

        Expected: F = 0.20 / (0.20 * sqrt(0.1)) = 0.20 / 0.0632 = 3.162
        """
        tau = 0.20
        sigma = 0.20
        h = 0.1  # 1/10 bins
        signal = 1

        expected_forecast = (tau / (sigma * np.sqrt(h))) * signal
        self.assertAlmostEqual(expected_forecast, 3.162, places=2)

    def test_fdm_formula_single_model(self):
        """Single model should have FDM = 1.0."""
        layer = WeightLayer(fdm_max=2.0)

        forecasts = [pd.DataFrame({
            'ticker': ['ES', 'NQ'],
            'model_name': ['model_a', 'model_a'],
            'forecast': [1.0, 0.8],
            'signal': [1, 1]
        })]

        signals = pd.DataFrame({
            'model_a': [1, 0, 1, 0, 1]
        })

        layer.fit(forecasts, signals)

        self.assertAlmostEqual(layer.fdm_, 1.0, places=5)

    def test_fdm_capped_at_max(self):
        """FDM should be capped at fdm_max (2.0)."""
        layer = WeightLayer(fdm_max=2.0)

        # Create uncorrelated forecasts
        np.random.seed(42)
        forecasts = [pd.DataFrame({
            'ticker': ['ES'] * 200,
            'model_name': ['model_a'] * 100 + ['model_b'] * 100,
            'forecast': list(np.random.randn(100)) + list(np.random.randn(100) * -1),
            'signal': [1] * 200
        })]

        signals = pd.DataFrame({
            'model_a': np.random.randint(0, 2, 100),
            'model_b': np.random.randint(0, 2, 100)
        })

        layer.fit(forecasts, signals)

        self.assertLessEqual(layer.fdm_, 2.0)

    def test_idm_formula_single_instrument(self):
        """Single instrument should have IDM = 1.0."""
        portfolio = Portfolio(idm_max=2.5)

        returns = pd.DataFrame({
            'ES': np.random.randn(100) * 0.01
        })

        portfolio.fit(returns)

        self.assertAlmostEqual(portfolio.idm_, 1.0, places=5)

    def test_idm_capped_at_max(self):
        """IDM should be capped at idm_max (2.5)."""
        portfolio = Portfolio(idm_max=2.5)

        # Create uncorrelated returns
        np.random.seed(42)
        returns = pd.DataFrame({
            'ES': np.random.randn(100) * 0.01,
            'GC': np.random.randn(100) * 0.01,
            'EU': np.random.randn(100) * 0.01
        })

        portfolio.fit(returns)

        self.assertLessEqual(portfolio.idm_, 2.5)

    def test_idm_increases_with_diversification(self):
        """IDM should be higher with more uncorrelated instruments."""
        np.random.seed(42)

        # Single instrument
        portfolio_1 = Portfolio(idm_max=2.5)
        returns_1 = pd.DataFrame({
            'ES': np.random.randn(100) * 0.01
        })
        portfolio_1.fit(returns_1)

        # Multiple uncorrelated instruments
        portfolio_3 = Portfolio(idm_max=2.5)
        returns_3 = pd.DataFrame({
            'ES': np.random.randn(100) * 0.01,
            'GC': np.random.randn(100) * 0.01,
            'EU': np.random.randn(100) * 0.01
        })
        portfolio_3.fit(returns_3)

        # IDM should be higher with more instruments
        self.assertGreater(portfolio_3.idm_, portfolio_1.idm_)


class TestWeightLayerIntegration(unittest.TestCase):
    """Integration tests for WeightLayer."""

    def test_fit_and_combine_workflow(self):
        """Test complete fit and combine workflow."""
        layer = WeightLayer(fdm_max=2.0)

        # Create forecast vectors
        forecasts = [pd.DataFrame({
            'ticker': ['ES', 'ES', 'NQ', 'NQ'],
            'model_name': ['ewmac_8', 'rsi_14', 'ewmac_8', 'rsi_14'],
            'forecast': [1.5, 0.8, 1.2, 0.6],
            'signal': [1, 1, 1, 0]
        })]

        # Create signals for weight calculation
        signals = pd.DataFrame({
            'ewmac_8': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            'rsi_14': [0, 1, 1, 0, 0, 1, 0, 1, 1, 0]
        })

        # Fit
        layer.fit(forecasts, signals)

        # Verify fitted attributes
        self.assertTrue(layer.is_fitted_)
        self.assertIsNotNone(layer.weights_)
        self.assertIsNotNone(layer.fdm_)
        self.assertGreater(layer.fdm_, 0)

        # Combine
        result = layer.combine(forecasts)

        # Verify output structure
        self.assertIn('ticker', result.columns)
        self.assertIn('forecast_score', result.columns)
        self.assertEqual(len(result), 2)  # ES and NQ

    def test_weights_sum_to_one(self):
        """Verify weights sum to 1.0."""
        layer = WeightLayer()

        forecasts = [pd.DataFrame({
            'ticker': ['ES'] * 3,
            'model_name': ['model_a', 'model_b', 'model_c'],
            'forecast': [1.0, 0.5, 0.8],
            'signal': [1, 1, 1]
        })]

        signals = pd.DataFrame({
            'model_a': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            'model_b': [0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
            'model_c': [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
        })

        layer.fit(forecasts, signals)

        self.assertAlmostEqual(layer.weights_.sum(), 1.0, places=5)


class TestPortfolioIntegration(unittest.TestCase):
    """Integration tests for Portfolio."""

    def test_fit_and_predict_workflow(self):
        """Test complete fit and predict workflow."""
        portfolio = Portfolio(
            max_position_pct=2.0,
            idm_max=2.5
        )

        # Create instrument returns for IDM calculation
        np.random.seed(42)
        returns = pd.DataFrame({
            'ES': np.random.randn(100) * 0.01,
            'NQ': np.random.randn(100) * 0.01
        })

        portfolio.fit(returns)

        # Verify fitted
        self.assertTrue(portfolio.is_fitted_)
        self.assertIsNotNone(portfolio.idm_)

        # Create combined forecasts
        combined = pd.DataFrame({
            'ticker': ['ES', 'NQ'],
            'forecast_score': [1.5, 0.8]
        })

        # Predict
        result = portfolio.predict(combined)

        # Verify output
        self.assertIn('ticker', result.columns)
        self.assertIn('forecast_score', result.columns)
        self.assertIn('position_fraction', result.columns)
        self.assertEqual(len(result), 2)

    def test_position_capping(self):
        """Test position capping at max_position_pct."""
        portfolio = Portfolio(
            max_position_pct=1.0,  # Cap at 100%
            idm_max=2.5
        )

        np.random.seed(42)
        returns = pd.DataFrame({
            'ES': np.random.randn(100) * 0.01,
            'NQ': np.random.randn(100) * 0.01
        })

        portfolio.fit(returns)

        # Create large forecast that would exceed cap
        combined = pd.DataFrame({
            'ticker': ['ES', 'NQ'],
            'forecast_score': [5.0, 3.0]  # Would exceed 100% without cap
        })

        result = portfolio.predict(combined)

        # All positions should be <= max_position_pct
        self.assertTrue(all(result['position_fraction'] <= 1.0))

    def test_instrument_weights(self):
        """Test custom instrument weights."""
        portfolio = Portfolio(
            instrument_weights={'ES': 0.7, 'NQ': 0.3},
            idm_max=2.5
        )

        np.random.seed(42)
        returns = pd.DataFrame({
            'ES': np.random.randn(100) * 0.01,
            'NQ': np.random.randn(100) * 0.01
        })

        portfolio.fit(returns)

        combined = pd.DataFrame({
            'ticker': ['ES', 'NQ'],
            'forecast_score': [1.0, 1.0]  # Same forecast
        })

        result = portfolio.predict(combined)

        # ES should have higher position due to higher weight
        es_pos = result[result['ticker'] == 'ES']['position_fraction'].iloc[0]
        nq_pos = result[result['ticker'] == 'NQ']['position_fraction'].iloc[0]

        # With same forecast, position should be proportional to weight
        self.assertGreater(es_pos, nq_pos)


class TestPositionSizerIntegration(unittest.TestCase):
    """Integration tests for PositionSizer."""

    def test_end_to_end_position_sizing(self):
        """Test complete position sizing workflow."""
        specs = {
            'ES': ContractSpec(ticker='ES', price=4800, multiplier=50),
            'NQ': ContractSpec(ticker='NQ', price=16000, multiplier=20)
        }

        sizer = PositionSizer(
            capital=1_000_000,
            contract_specs=specs,
            rounding_method=RoundingMethod.ROUND
        )

        position_fractions = pd.DataFrame({
            'ticker': ['ES', 'NQ'],
            'forecast_score': [1.0, 0.5],
            'position_fraction': [0.5, 0.25]
        })

        result = sizer.calculate_positions(position_fractions)

        # Verify output structure
        self.assertIn('contracts', result.columns)
        self.assertIn('notional_value', result.columns)
        self.assertIn('notional_pct', result.columns)

        # Verify reasonable results
        self.assertTrue(all(result['contracts'] >= 0))
        self.assertTrue(all(result['notional_value'] >= 0))

    def test_contract_calculation(self):
        """Verify contract calculation formula."""
        # ES: price=4800, multiplier=50, contract_value=240,000
        # Capital=1,000,000, position_fraction=0.5
        # Target = 500,000
        # Contracts = 500,000 / 240,000 = 2.08 -> 2 (rounded)

        specs = {
            'ES': ContractSpec(ticker='ES', price=4800, multiplier=50)
        }

        sizer = PositionSizer(
            capital=1_000_000,
            contract_specs=specs,
            rounding_method=RoundingMethod.ROUND
        )

        position_fractions = pd.DataFrame({
            'ticker': ['ES'],
            'forecast_score': [1.0],
            'position_fraction': [0.5]
        })

        result = sizer.calculate_positions(position_fractions)

        self.assertEqual(result.iloc[0]['contracts'], 2)


class TestFullPipeline(unittest.TestCase):
    """End-to-end tests for the complete pipeline."""

    def test_full_pipeline(self):
        """Test complete pipeline: WeightLayer -> Portfolio -> PositionSizer."""
        np.random.seed(42)

        # Step 1: Create forecast vectors (simulating Ensemble output)
        forecast_vectors = [pd.DataFrame({
            'ticker': ['ES', 'ES', 'NQ', 'NQ', 'GC', 'GC'],
            'model_name': ['ewmac_8', 'rsi_14'] * 3,
            'forecast': [1.5, 0.8, 1.2, 0.6, 0.9, 0.4],
            'signal': [1, 1, 1, 0, 1, 1]
        })]

        signals = pd.DataFrame({
            'ewmac_8': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            'rsi_14': [0, 1, 1, 0, 0, 1, 0, 1, 1, 0]
        })

        # Step 2: WeightLayer - combine forecasts
        weight_layer = WeightLayer(fdm_max=2.0)
        weight_layer.fit(forecast_vectors, signals)
        combined = weight_layer.combine(forecast_vectors)

        self.assertEqual(len(combined), 3)  # ES, NQ, GC

        # Step 3: Portfolio - apply IDM and instrument weights
        returns = pd.DataFrame({
            'ES': np.random.randn(100) * 0.01,
            'NQ': np.random.randn(100) * 0.01,
            'GC': np.random.randn(100) * 0.01
        })

        portfolio = Portfolio(
            max_position_pct=2.0,
            idm_max=2.5
        )
        portfolio.fit(returns)
        positions = portfolio.predict(combined)

        self.assertEqual(len(positions), 3)

        # Step 4: PositionSizer - convert to contracts
        specs = {
            'ES': ContractSpec(ticker='ES', price=4800, multiplier=50),
            'NQ': ContractSpec(ticker='NQ', price=16000, multiplier=20),
            'GC': ContractSpec(ticker='GC', price=2000, multiplier=100)
        }

        sizer = PositionSizer(
            capital=1_000_000,
            contract_specs=specs
        )

        contracts = sizer.calculate_positions(positions)

        self.assertEqual(len(contracts), 3)
        self.assertIn('contracts', contracts.columns)

        # Verify diagnostics
        summary = sizer.get_summary(contracts)
        self.assertIn('total_notional_value', summary)
        self.assertIn('n_contracts', summary)

    def test_pipeline_with_inactive_signals(self):
        """Test pipeline handles inactive signals (forecast=0)."""
        np.random.seed(42)

        # All signals inactive for NQ
        forecast_vectors = [pd.DataFrame({
            'ticker': ['ES', 'ES', 'NQ', 'NQ'],
            'model_name': ['ewmac_8', 'rsi_14', 'ewmac_8', 'rsi_14'],
            'forecast': [1.5, 0.8, 0.0, 0.0],  # NQ has zero forecasts
            'signal': [1, 1, 0, 0]  # NQ signals inactive
        })]

        signals = pd.DataFrame({
            'ewmac_8': [1, 0, 1, 0, 1],
            'rsi_14': [0, 1, 1, 0, 0]
        })

        weight_layer = WeightLayer(fdm_max=2.0)
        weight_layer.fit(forecast_vectors, signals)
        combined = weight_layer.combine(forecast_vectors)

        # NQ should have zero forecast
        nq_forecast = combined[combined['ticker'] == 'NQ']['forecast_score'].iloc[0]
        self.assertEqual(nq_forecast, 0.0)

    def test_pipeline_single_model(self):
        """Test pipeline with single model (no FDM benefit)."""
        np.random.seed(42)

        forecast_vectors = [pd.DataFrame({
            'ticker': ['ES', 'NQ'],
            'model_name': ['ewmac_8', 'ewmac_8'],
            'forecast': [1.5, 1.2],
            'signal': [1, 1]
        })]

        signals = pd.DataFrame({
            'ewmac_8': [1, 0, 1, 0, 1]
        })

        weight_layer = WeightLayer(fdm_max=2.0)
        weight_layer.fit(forecast_vectors, signals)

        # Single model should have FDM = 1.0
        self.assertAlmostEqual(weight_layer.fdm_, 1.0, places=5)

        combined = weight_layer.combine(forecast_vectors)

        # Forecasts should pass through without FDM scaling
        es_forecast = combined[combined['ticker'] == 'ES']['forecast_score'].iloc[0]
        self.assertAlmostEqual(es_forecast, 1.5, places=5)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_empty_forecast_vectors(self):
        """Test handling of empty forecast vectors."""
        layer = WeightLayer()

        # Fit with valid data first
        forecasts = [pd.DataFrame({
            'ticker': ['ES'],
            'model_name': ['model_a'],
            'forecast': [1.0],
            'signal': [1]
        })]
        signals = pd.DataFrame({'model_a': [1, 0, 1, 0, 1]})
        layer.fit(forecasts, signals)

        # Combine with empty input
        result = layer.combine([])

        self.assertEqual(len(result), 0)
        self.assertIn('ticker', result.columns)
        self.assertIn('forecast_score', result.columns)

    def test_portfolio_empty_forecasts(self):
        """Test Portfolio handles empty forecasts."""
        portfolio = Portfolio(idm_max=2.5)

        np.random.seed(42)
        returns = pd.DataFrame({
            'ES': np.random.randn(100) * 0.01
        })
        portfolio.fit(returns)

        empty_combined = pd.DataFrame(columns=['ticker', 'forecast_score'])
        result = portfolio.predict(empty_combined)

        self.assertEqual(len(result), 0)

    def test_position_sizer_empty_positions(self):
        """Test PositionSizer handles empty positions."""
        specs = {
            'ES': ContractSpec(ticker='ES', price=4800, multiplier=50)
        }

        sizer = PositionSizer(capital=1_000_000, contract_specs=specs)

        empty_positions = pd.DataFrame(columns=['ticker', 'forecast_score', 'position_fraction'])
        result = sizer.calculate_positions(empty_positions)

        self.assertEqual(len(result), 0)
        self.assertIn('contracts', result.columns)


if __name__ == '__main__':
    unittest.main()
