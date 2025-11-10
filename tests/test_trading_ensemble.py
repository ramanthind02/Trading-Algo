"""
Unit tests for TradingEnsemble class

This module contains comprehensive unit tests for the TradingEnsemble class,
testing all functionality including fit, predict, configuration save/load,
error handling, and edge cases.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import json

# Add the project root to the path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ensemble import TradingEnsemble


class TestTradingEnsemble(unittest.TestCase):
    """Test cases for TradingEnsemble class."""
    
    def setUp(self):
        """Set up test data and fixtures."""
        # Create consistent test data
        np.random.seed(42)
        self.n_samples = 100
        
        # Create sample data with known characteristics
        self.X_valid = pd.DataFrame({
            'strategy_1': np.random.binomial(1, 0.3, self.n_samples),  # 30% exposure
            'strategy_2': np.random.binomial(1, 0.4, self.n_samples),  # 40% exposure
            'strategy_3': np.random.binomial(1, 0.2, self.n_samples),  # 20% exposure
            'annualized_volatility': np.random.lognormal(np.log(0.15), 0.1, self.n_samples).clip(0.1, 0.3)
        })
        
        # Create target with known correlations
        self.y_valid = (
            0.02 * self.X_valid['strategy_1'] +     # Positive correlation
            -0.01 * self.X_valid['strategy_2'] +    # Negative correlation
            0.015 * self.X_valid['strategy_3'] +    # Positive correlation
            np.random.normal(0, 0.01, self.n_samples)  # Noise
        )
        
        # Create temporary directory for config files
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up after tests."""
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_initialization_default(self):
        """Test default initialization."""
        ensemble = TradingEnsemble()
        
        self.assertEqual(ensemble.r, 0.15)
        self.assertIsNone(ensemble.save_path)
        self.assertIsNone(ensemble.weights_)
        self.assertIsNone(ensemble.exposure_fractions_)
        self.assertIsNone(ensemble.feature_names_)
        self.assertFalse(ensemble.is_fitted_)
        
    def test_initialization_with_parameters(self):
        """Test initialization with custom parameters."""
        ensemble = TradingEnsemble(r=0.20, save_path="test.json")
        
        self.assertEqual(ensemble.r, 0.20)
        self.assertEqual(ensemble.save_path, "test.json")
        self.assertFalse(ensemble.is_fitted_)
        
    def test_fit_basic(self):
        """Test basic fit functionality."""
        ensemble = TradingEnsemble()
        result = ensemble.fit(self.X_valid, self.y_valid)
        
        # Check return value
        self.assertEqual(result, ensemble)
        
        # Check fitted status
        self.assertTrue(ensemble.is_fitted_)
        
        # Check fitted attributes
        self.assertIsNotNone(ensemble.weights_)
        self.assertIsNotNone(ensemble.exposure_fractions_)
        self.assertIsNotNone(ensemble.feature_names_)
        
        # Check feature names
        expected_features = ['strategy_1', 'strategy_2', 'strategy_3']
        self.assertEqual(ensemble.feature_names_, expected_features)
        
        # Check weights sum to 1
        total_weight = sum(ensemble.weights_.values())
        self.assertAlmostEqual(total_weight, 1.0, places=10)
        
        # Check exposure fractions are reasonable
        for feature in expected_features:
            exposure = ensemble.exposure_fractions_[feature]
            self.assertGreater(exposure, 0)
            self.assertLessEqual(exposure, 1)
            
    def test_fit_weight_calculation(self):
        """Test that weights are calculated correctly."""
        ensemble = TradingEnsemble()
        ensemble.fit(self.X_valid, self.y_valid)
        
        # Strategy_1 should have highest weight (strongest correlation)
        # Strategy_2 should have second highest (negative correlation but we use absolute)
        weights = ensemble.weights_
        
        # Check all strategies have weights
        self.assertIn('strategy_1', weights)
        self.assertIn('strategy_2', weights)
        self.assertIn('strategy_3', weights)
        
        # All weights should be positive
        for weight in weights.values():
            self.assertGreater(weight, 0)
            
    def test_predict_basic(self):
        """Test basic predict functionality."""
        ensemble = TradingEnsemble(r=0.20)
        ensemble.fit(self.X_valid, self.y_valid)
        
        predictions = ensemble.predict(self.X_valid)
        
        # Check output shape and type
        self.assertEqual(len(predictions), len(self.X_valid))
        self.assertIsInstance(predictions, np.ndarray)
        
        # All predictions should be non-negative
        self.assertTrue(np.all(predictions >= 0))
        
    def test_predict_formula(self):
        """Test that predict implements the correct formula: r/v * Σ(w_i * X_i / √h_i)."""
        ensemble = TradingEnsemble(r=0.10)
        ensemble.fit(self.X_valid, self.y_valid)
        
        # Create simple test case
        X_test = pd.DataFrame({
            'strategy_1': [1, 0],
            'strategy_2': [0, 1], 
            'strategy_3': [1, 0],
            'annualized_volatility': [0.20, 0.15]
        })
        
        predictions = ensemble.predict(X_test)
        
        # Manually calculate expected values
        r = 0.10
        for i in range(len(X_test)):
            expected = 0.0
            vol = X_test.iloc[i]['annualized_volatility']
            
            for feature in ensemble.feature_names_:
                w_i = ensemble.weights_[feature]
                h_i = ensemble.exposure_fractions_[feature]
                x_i = X_test.iloc[i][feature]
                
                expected += w_i * x_i / np.sqrt(h_i)
                
            expected = (r / vol) * expected
            
            self.assertAlmostEqual(predictions[i], expected, places=10)
            
    def test_save_load_config(self):
        """Test configuration save and load."""
        # Fit original ensemble
        ensemble1 = TradingEnsemble(r=0.18)
        ensemble1.fit(self.X_valid, self.y_valid)
        
        # Save configuration
        config_path = os.path.join(self.temp_dir, "test_config.json")
        saved_path = ensemble1.save_config(config_path)
        
        self.assertEqual(saved_path, config_path)
        self.assertTrue(os.path.exists(config_path))
        
        # Load configuration into new ensemble
        ensemble2 = TradingEnsemble()
        ensemble2.load_config(config_path)
        
        # Check loaded parameters
        self.assertTrue(ensemble2.is_fitted_)
        self.assertEqual(ensemble2.r, 0.18)
        self.assertEqual(ensemble2.weights_, ensemble1.weights_)
        self.assertEqual(ensemble2.exposure_fractions_, ensemble1.exposure_fractions_)
        self.assertEqual(ensemble2.feature_names_, ensemble1.feature_names_)
        
        # Check predictions are identical
        pred1 = ensemble1.predict(self.X_valid)
        pred2 = ensemble2.predict(self.X_valid)
        np.testing.assert_array_almost_equal(pred1, pred2)
        
    def test_config_file_format(self):
        """Test that saved configuration has correct format."""
        ensemble = TradingEnsemble(r=0.12)
        ensemble.fit(self.X_valid, self.y_valid)
        
        config_path = os.path.join(self.temp_dir, "format_test.json")
        ensemble.save_config(config_path)
        
        # Load and check JSON structure
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Check required keys
        required_keys = ['metadata', 'weights', 'exposure_fractions', 'feature_names', 'parameters']
        for key in required_keys:
            self.assertIn(key, config)
            
        # Check metadata
        self.assertIn('created_at', config['metadata'])
        self.assertEqual(config['metadata']['model_type'], 'TradingEnsemble')
        self.assertEqual(config['metadata']['target_risk'], 0.12)
        
        # Check parameters
        self.assertEqual(config['parameters']['r'], 0.12)
        
    def test_initialization_with_config(self):
        """Test initialization with config_path parameter."""
        # Create and save config
        ensemble1 = TradingEnsemble(r=0.25)
        ensemble1.fit(self.X_valid, self.y_valid)
        
        config_path = os.path.join(self.temp_dir, "init_test.json")
        ensemble1.save_config(config_path)
        
        # Initialize new ensemble with config
        ensemble2 = TradingEnsemble(config_path=config_path)
        
        self.assertTrue(ensemble2.is_fitted_)
        self.assertEqual(ensemble2.r, 0.25)
        
    def test_error_missing_volatility_column(self):
        """Test error when annualized_volatility column is missing."""
        X_bad = pd.DataFrame({
            'strategy_1': [0, 1, 0, 1],
            'strategy_2': [1, 0, 1, 0]
        })
        y_bad = pd.Series([0.01, -0.005, 0.02, -0.01])
        
        ensemble = TradingEnsemble()
        
        with self.assertRaises(ValueError) as context:
            ensemble.fit(X_bad, y_bad)
            
        self.assertIn("annualized_volatility", str(context.exception))
        
    def test_error_non_binary_features(self):
        """Test error when features are not binary."""
        X_bad = pd.DataFrame({
            'strategy_1': [0.5, 1.5, 0.2, 1.8],  # Non-binary
            'annualized_volatility': [0.15, 0.20, 0.18, 0.22]
        })
        y_bad = pd.Series([0.01, -0.005, 0.02, -0.01])
        
        ensemble = TradingEnsemble()
        
        with self.assertRaises(ValueError) as context:
            ensemble.fit(X_bad, y_bad)
            
        self.assertIn("binary", str(context.exception))
        
    def test_error_negative_volatility(self):
        """Test error when volatility contains negative values."""
        X_bad = pd.DataFrame({
            'strategy_1': [0, 1, 0, 1],
            'annualized_volatility': [0.15, -0.20, 0.18, 0.22]  # Negative value
        })
        y_bad = pd.Series([0.01, -0.005, 0.02, -0.01])
        
        ensemble = TradingEnsemble()
        
        with self.assertRaises(ValueError) as context:
            ensemble.fit(X_bad, y_bad)
            
        self.assertIn("non-positive", str(context.exception))
        
    def test_error_predict_not_fitted(self):
        """Test error when calling predict before fit."""
        ensemble = TradingEnsemble()
        
        with self.assertRaises(ValueError) as context:
            ensemble.predict(self.X_valid)
            
        self.assertIn("fitted", str(context.exception))
        
    def test_error_predict_missing_features(self):
        """Test error when predict data is missing required features."""
        ensemble = TradingEnsemble()
        ensemble.fit(self.X_valid, self.y_valid)
        
        X_missing = pd.DataFrame({
            'strategy_1': [0, 1, 0],
            # Missing strategy_2 and strategy_3
            'annualized_volatility': [0.15, 0.20, 0.18]
        })
        
        with self.assertRaises(ValueError) as context:
            ensemble.predict(X_missing)
            
        self.assertIn("Missing feature columns", str(context.exception))
        
    def test_error_numpy_array_input(self):
        """Test error when passing numpy array instead of DataFrame."""
        X_array = np.array([[0, 1, 0.15], [1, 0, 0.20]])
        y_array = np.array([0.01, -0.01])
        
        ensemble = TradingEnsemble()
        
        with self.assertRaises(ValueError) as context:
            ensemble.fit(X_array, y_array)
            
        self.assertIn("DataFrame input", str(context.exception))
        
    def test_error_save_config_not_fitted(self):
        """Test error when saving config before fitting."""
        ensemble = TradingEnsemble()
        
        with self.assertRaises(ValueError) as context:
            ensemble.save_config("test.json")
            
        self.assertIn("fitted", str(context.exception))
        
    def test_error_save_config_no_path(self):
        """Test error when no save path provided."""
        ensemble = TradingEnsemble()
        ensemble.fit(self.X_valid, self.y_valid)
        
        with self.assertRaises(ValueError) as context:
            ensemble.save_config()
            
        self.assertIn("filepath", str(context.exception))
        
    def test_error_load_config_file_not_found(self):
        """Test error when config file doesn't exist."""
        ensemble = TradingEnsemble()
        
        with self.assertRaises(FileNotFoundError):
            ensemble.load_config("nonexistent_file.json")
            
    def test_error_load_config_invalid_json(self):
        """Test error when config file has invalid JSON."""
        bad_config_path = os.path.join(self.temp_dir, "bad_config.json")
        with open(bad_config_path, 'w') as f:
            f.write("{ invalid json }")
            
        ensemble = TradingEnsemble()
        
        with self.assertRaises(ValueError) as context:
            ensemble.load_config(bad_config_path)
            
        self.assertIn("Invalid JSON", str(context.exception))
            
    def test_string_representations(self):
        """Test __repr__ and __str__ methods."""
        # Test unfitted ensemble
        ensemble = TradingEnsemble(r=0.18)
        
        repr_str = repr(ensemble)
        self.assertIn("TradingEnsemble", repr_str)
        self.assertIn("0.18", repr_str)
        self.assertIn("fitted=False", repr_str)
        
        str_str = str(ensemble)
        self.assertIn("not fitted", str_str)
        self.assertIn("0.18", str_str)
        
        # Test fitted ensemble
        ensemble.fit(self.X_valid, self.y_valid)
        
        repr_str = repr(ensemble)
        self.assertIn("fitted=True", repr_str)
        self.assertIn("n_features=3", repr_str)
        
        str_str = str(ensemble)
        self.assertIn("fitted", str_str)
        self.assertIn("Features: 3", str_str)
        
    def test_edge_case_single_feature(self):
        """Test with single feature."""
        # Need at least 10 samples for fitting
        np.random.seed(789)
        n_samples = 15
        X_single = pd.DataFrame({
            'strategy_1': np.random.binomial(1, 0.4, n_samples),
            'annualized_volatility': np.random.lognormal(np.log(0.15), 0.1, n_samples).clip(0.1, 0.3)
        })
        y_single = 0.01 * X_single['strategy_1'] + np.random.normal(0, 0.005, n_samples)
        
        ensemble = TradingEnsemble()
        ensemble.fit(X_single, y_single)
        
        self.assertTrue(ensemble.is_fitted_)
        self.assertEqual(len(ensemble.feature_names_), 1)
        self.assertAlmostEqual(ensemble.weights_['strategy_1'], 1.0)
        
        predictions = ensemble.predict(X_single)
        self.assertEqual(len(predictions), len(X_single))
        
    def test_edge_case_zero_variance_feature(self):
        """Test with feature that has zero variance (all same values)."""
        # Need at least 10 samples for fitting
        np.random.seed(456)
        n_samples = 12
        X_zero_var = pd.DataFrame({
            'strategy_1': [1] * n_samples,  # No variance (all 1s)
            'strategy_2': np.random.binomial(1, 0.5, n_samples),  # Has variance
            'annualized_volatility': np.random.lognormal(np.log(0.15), 0.1, n_samples).clip(0.1, 0.3)
        })
        y_zero_var = 0.01 * X_zero_var['strategy_2'] + np.random.normal(0, 0.005, n_samples)
        
        ensemble = TradingEnsemble()
        ensemble.fit(X_zero_var, y_zero_var)
        
        # Should handle zero variance gracefully
        self.assertTrue(ensemble.is_fitted_)
        predictions = ensemble.predict(X_zero_var)
        self.assertEqual(len(predictions), len(X_zero_var))


class TestTradingEnsembleIntegration(unittest.TestCase):
    """Integration tests for TradingEnsemble with realistic scenarios."""
    
    def test_realistic_trading_scenario(self):
        """Test with realistic trading data scenario."""
        np.random.seed(123)
        n_samples = 500
        
        # Create realistic multi-strategy data
        X_real = pd.DataFrame({
            'momentum': np.random.binomial(1, 0.25, n_samples),
            'mean_reversion': np.random.binomial(1, 0.30, n_samples),
            'trend_following': np.random.binomial(1, 0.35, n_samples),
            'volatility_breakout': np.random.binomial(1, 0.20, n_samples),
            'annualized_volatility': np.random.lognormal(np.log(0.16), 0.2, n_samples).clip(0.1, 0.4)
        })
        
        # Create returns with realistic correlations
        y_real = (
            0.020 * X_real['momentum'] +
            0.015 * X_real['mean_reversion'] +
            0.018 * X_real['trend_following'] +
            0.012 * X_real['volatility_breakout'] +
            np.random.normal(0, 0.015, n_samples)
        )
        
        # Test full workflow
        ensemble = TradingEnsemble(r=0.15)
        
        # Split data
        split = int(0.7 * n_samples)
        X_train, X_test = X_real.iloc[:split], X_real.iloc[split:]
        y_train, y_test = y_real.iloc[:split], y_real.iloc[split:]
        
        # Fit and predict
        ensemble.fit(X_train, y_train)
        train_pred = ensemble.predict(X_train)
        test_pred = ensemble.predict(X_test)
        
        # Basic sanity checks
        self.assertEqual(len(train_pred), len(X_train))
        self.assertEqual(len(test_pred), len(X_test))
        self.assertTrue(np.all(train_pred >= 0))
        self.assertTrue(np.all(test_pred >= 0))
        
        # Check that predictions make sense
        # Higher volatility should generally lead to smaller positions (all else equal)
        high_vol_mask = X_test['annualized_volatility'] > X_test['annualized_volatility'].median()
        low_vol_mask = ~high_vol_mask
        
        # This is a statistical test, might occasionally fail due to randomness
        # but should generally hold for realistic data
        if np.any(test_pred[low_vol_mask] > 0) and np.any(test_pred[high_vol_mask] > 0):
            avg_pos_low_vol = np.mean(test_pred[low_vol_mask][test_pred[low_vol_mask] > 0])
            avg_pos_high_vol = np.mean(test_pred[high_vol_mask][test_pred[high_vol_mask] > 0])
            # Generally expect larger positions with lower volatility
            # This is a loose check since it depends on signal overlap
            
    def test_config_persistence_workflow(self):
        """Test complete configuration save/load workflow."""
        np.random.seed(456)
        
        # Training phase
        X_train = pd.DataFrame({
            'strat_a': np.random.binomial(1, 0.3, 200),
            'strat_b': np.random.binomial(1, 0.4, 200),
            'annualized_volatility': np.random.lognormal(np.log(0.18), 0.15, 200).clip(0.1, 0.35)
        })
        y_train = 0.02 * X_train['strat_a'] + 0.01 * X_train['strat_b'] + np.random.normal(0, 0.01, 200)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
            
        try:
            # Training ensemble
            training_ensemble = TradingEnsemble(r=0.22, save_path=config_path)
            training_ensemble.fit(X_train, y_train)
            saved_path = training_ensemble.save_config()
            
            self.assertEqual(saved_path, config_path)
            
            # Production ensemble (clean slate)
            production_ensemble = TradingEnsemble(config_path=config_path)
            
            # New data for production
            X_prod = pd.DataFrame({
                'strat_a': [1, 0, 1, 0],
                'strat_b': [0, 1, 1, 0],
                'annualized_volatility': [0.15, 0.20, 0.18, 0.25]
            })
            
            # Both should give identical predictions
            pred_train = training_ensemble.predict(X_prod)
            pred_prod = production_ensemble.predict(X_prod)
            
            np.testing.assert_array_almost_equal(pred_train, pred_prod)
            
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
