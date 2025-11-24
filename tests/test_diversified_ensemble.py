"""
Unit tests for DiversifiedEnsemble class

This module contains comprehensive unit tests for the DiversifiedEnsemble class,
testing all functionality including fit, predict, configuration save/load,
error handling, and edge cases with the new API that includes ticker and volatility parameters.
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

from ensemble import DiversifiedEnsemble

class TestDiversifiedEnsemble(unittest.TestCase):
    """Test cases for DiversifiedEnsemble class."""
    
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
        })
        
        # Create separate ticker and volatility arrays
        tickers = ['ES', 'NQ', 'YM']
        self.ticker_valid = pd.Series(np.random.choice(tickers, self.n_samples))
        self.volatility_valid = pd.Series(np.random.lognormal(np.log(0.15), 0.1, self.n_samples).clip(0.1, 0.3))
        
        # Create target with some correlation structure (though it will be ignored)
        self.y_valid = (
            0.02 * self.X_valid['strategy_1'] +     
            -0.01 * self.X_valid['strategy_2'] +    
            0.015 * self.X_valid['strategy_3'] +    
            np.random.normal(0, 0.01, self.n_samples)
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
        ensemble = DiversifiedEnsemble()
        
        self.assertEqual(ensemble.target_volatility, 0.15)
        self.assertIsNone(ensemble.instrument_weights)
        self.assertIsNone(ensemble.save_path)
        self.assertIsNone(ensemble.weights_)
        self.assertIsNone(ensemble.exposure_fractions_)
        self.assertIsNone(ensemble.feature_names_)
        self.assertIsNone(ensemble.unique_tickers_)
        self.assertIsNone(ensemble.instrument_weights_)
        self.assertFalse(ensemble.is_fitted_)
        
    def test_initialization_with_parameters(self):
        """Test initialization with custom parameters."""
        instrument_weights = {'ES': 0.5, 'NQ': 0.3, 'YM': 0.2}
        ensemble = DiversifiedEnsemble(
            target_volatility=0.20, 
            instrument_weights=instrument_weights,
            save_path="test.json"
        )
        
        self.assertEqual(ensemble.target_volatility, 0.20)
        self.assertEqual(ensemble.instrument_weights, instrument_weights)
        self.assertEqual(ensemble.save_path, "test.json")
        self.assertFalse(ensemble.is_fitted_)
        
    def test_fit_basic(self):
        """Test basic fit functionality."""
        ensemble = DiversifiedEnsemble()
        result = ensemble.fit(self.X_valid, self.ticker_valid, self.volatility_valid, self.y_valid)
        
        # Check return value
        self.assertEqual(result, ensemble)
        
        # Check fitted status
        self.assertTrue(ensemble.is_fitted_)
        
        # Check fitted attributes
        self.assertIsNotNone(ensemble.weights_)
        self.assertIsNotNone(ensemble.exposure_fractions_)
        self.assertIsNotNone(ensemble.feature_names_)
        self.assertIsNotNone(ensemble.unique_tickers_)
        self.assertIsNotNone(ensemble.instrument_weights_)
        
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
            
        # Check ticker tracking
        expected_tickers = sorted(self.ticker_valid.unique())
        self.assertEqual(ensemble.unique_tickers_, expected_tickers)
        self.assertEqual(ensemble.n_tickers_, len(expected_tickers))
        
        # Check instrument weights sum to 1 
        total_instrument_weight = sum(ensemble.instrument_weights_.values())
        self.assertAlmostEqual(total_instrument_weight, 1.0, places=10)
            
    def test_fit_diversified_weights(self):
        """Test that weights are calculated using diversification strategy."""
        # Create data with known correlation structure
        n_samples = 200
        np.random.seed(123)
        
        # Create highly correlated features
        base_signal = np.random.binomial(1, 0.3, n_samples)
        X_corr = pd.DataFrame({
            'corr_1': base_signal,  # Identical to base
            'corr_2': base_signal,  # Identical to base  
            'indep_1': np.random.binomial(1, 0.3, n_samples),  # Independent
        })
        
        ticker = pd.Series(np.random.choice(['ES', 'NQ'], n_samples))
        volatility = pd.Series(np.random.lognormal(np.log(0.15), 0.1, n_samples).clip(0.1, 0.3))
        y = pd.Series(np.random.normal(0, 0.01, n_samples))
        
        ensemble = DiversifiedEnsemble()
        ensemble.fit(X_corr, ticker, volatility, y)
        
        # Independent feature should get higher weight than correlated ones
        weights = ensemble.weights_
        self.assertIn('corr_1', weights)
        self.assertIn('corr_2', weights)
        self.assertIn('indep_1', weights)
        
        # All weights should be positive
        for weight in weights.values():
            self.assertGreater(weight, 0)
            
        # Independent feature should have higher or equal weight
        self.assertGreaterEqual(weights['indep_1'], min(weights['corr_1'], weights['corr_2']))
        
    def test_fit_custom_instrument_weights(self):
        """Test fitting with custom instrument weights."""
        custom_weights = {'ES': 0.6, 'NQ': 0.4}  # Note: YM not included, will be normalized
        
        # Filter data to only include ES and NQ
        mask = self.ticker_valid.isin(['ES', 'NQ'])
        X_filtered = self.X_valid[mask].copy()
        ticker_filtered = self.ticker_valid[mask].copy() 
        volatility_filtered = self.volatility_valid[mask].copy()
        y_filtered = self.y_valid[mask].copy()
        
        ensemble = DiversifiedEnsemble()
        ensemble.fit(X_filtered, ticker_filtered, volatility_filtered, y_filtered, 
                    instrument_weights=custom_weights)
        
        # Check that weights were normalized
        expected_total = custom_weights['ES'] + custom_weights['NQ']
        expected_es = custom_weights['ES'] / expected_total
        expected_nq = custom_weights['NQ'] / expected_total
        
        self.assertAlmostEqual(ensemble.instrument_weights_['ES'], expected_es, places=10)
        self.assertAlmostEqual(ensemble.instrument_weights_['NQ'], expected_nq, places=10)
        
    def test_predict_basic(self):
        """Test basic predict functionality."""
        ensemble = DiversifiedEnsemble(target_volatility=0.20)
        ensemble.fit(self.X_valid, self.ticker_valid, self.volatility_valid, self.y_valid)
        
        predictions = ensemble.predict(self.X_valid, self.ticker_valid, self.volatility_valid)
        
        # Check output shape and type
        self.assertEqual(len(predictions), len(self.X_valid))
        self.assertIsInstance(predictions, np.ndarray)
        
        # All predictions should be non-negative (since we use absolute values)
        self.assertTrue(np.all(predictions >= 0))
        
    def test_predict_formula(self):
        """Test that predict implements the correct formula: Σ((τ × w_i) / (σ_i × √h_i)) × instrument_weight."""
        ensemble = DiversifiedEnsemble(target_volatility=0.10)
        ensemble.fit(self.X_valid, self.ticker_valid, self.volatility_valid, self.y_valid)
        
        # Create simple test case
        X_test = pd.DataFrame({
            'strategy_1': [1, 0],
            'strategy_2': [0, 1], 
            'strategy_3': [1, 0]
        })
        ticker_test = pd.Series(['ES', 'NQ'])
        volatility_test = pd.Series([0.20, 0.15])
        
        predictions = ensemble.predict(X_test, ticker_test, volatility_test)
        
        # Manually calculate expected values
        target_vol = 0.10
        for i in range(len(X_test)):
            expected = 0.0
            vol = volatility_test.iloc[i]
            tick = ticker_test.iloc[i]
            
            for feature in ensemble.feature_names_:
                w_i = ensemble.weights_[feature]
                h_i = ensemble.exposure_fractions_[feature]
                x_i = X_test.iloc[i][feature]
                
                if x_i == 1:  # Only count active features
                    expected += (target_vol * w_i) / (vol * np.sqrt(h_i))
                    
            # Apply instrument weight
            expected *= ensemble.instrument_weights_[tick]
            
            self.assertAlmostEqual(predictions[i], expected, places=10)
            
    def test_save_load_config(self):
        """Test configuration save and load."""
        # Fit original ensemble
        ensemble1 = DiversifiedEnsemble(target_volatility=0.18)
        ensemble1.fit(self.X_valid, self.ticker_valid, self.volatility_valid, self.y_valid)
        
        # Save configuration
        config_path = os.path.join(self.temp_dir, "test_config.json")
        saved_path = ensemble1.save_config(config_path)
        
        self.assertEqual(saved_path, config_path)
        self.assertTrue(os.path.exists(config_path))
        
        # Load configuration into new ensemble
        ensemble2 = DiversifiedEnsemble()
        ensemble2.load_config(config_path)
        
        # Check loaded parameters
        self.assertTrue(ensemble2.is_fitted_)
        self.assertEqual(ensemble2.target_volatility_, 0.18)
        self.assertEqual(ensemble2.weights_, ensemble1.weights_)
        self.assertEqual(ensemble2.exposure_fractions_, ensemble1.exposure_fractions_)
        self.assertEqual(ensemble2.feature_names_, ensemble1.feature_names_)
        self.assertEqual(ensemble2.unique_tickers_, ensemble1.unique_tickers_)
        self.assertEqual(ensemble2.instrument_weights_, ensemble1.instrument_weights_)
        
        # Check predictions are identical
        pred1 = ensemble1.predict(self.X_valid, self.ticker_valid, self.volatility_valid)
        pred2 = ensemble2.predict(self.X_valid, self.ticker_valid, self.volatility_valid)
        np.testing.assert_array_almost_equal(pred1, pred2)
        
    def test_config_file_format(self):
        """Test that saved configuration has correct format."""
        ensemble = DiversifiedEnsemble(target_volatility=0.12)
        ensemble.fit(self.X_valid, self.ticker_valid, self.volatility_valid, self.y_valid)
        
        config_path = os.path.join(self.temp_dir, "format_test.json")
        ensemble.save_config(config_path)
        
        # Load and check JSON structure
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Check required keys
        required_keys = [
            'metadata', 'weights', 'exposure_fractions', 'feature_names', 
            'target_volatility', 'unique_tickers', 'instrument_weights', 'n_tickers', 'parameters'
        ]
        for key in required_keys:
            self.assertIn(key, config)
            
        # Check metadata
        self.assertIn('created_at', config['metadata'])
        self.assertEqual(config['metadata']['model_type'], 'DiversifiedEnsemble')
        self.assertEqual(config['metadata']['target_volatility'], 0.12)
        
        # Check parameters
        self.assertEqual(config['parameters']['target_volatility'], 0.12)
        
    def test_initialization_with_config(self):
        """Test initialization with config_path parameter."""
        # Create and save config
        ensemble1 = DiversifiedEnsemble(target_volatility=0.25)
        ensemble1.fit(self.X_valid, self.ticker_valid, self.volatility_valid, self.y_valid)
        
        config_path = os.path.join(self.temp_dir, "init_test.json")
        ensemble1.save_config(config_path)
        
        # Initialize new ensemble with config
        ensemble2 = DiversifiedEnsemble(config_path=config_path)
        
        self.assertTrue(ensemble2.is_fitted_)
        self.assertEqual(ensemble2.target_volatility_, 0.25)
        
    def test_error_missing_ticker_volatility_in_fit(self):
        """Test error when ticker or volatility missing in fit."""
        ensemble = DiversifiedEnsemble()
        
        # Missing ticker
        with self.assertRaises(ValueError) as context:
            ensemble.fit(self.X_valid, None, self.volatility_valid, self.y_valid)
        self.assertIn("ticker and volatility parameters are required", str(context.exception))
        
        # Missing volatility  
        with self.assertRaises(ValueError) as context:
            ensemble.fit(self.X_valid, self.ticker_valid, None, self.y_valid)
        self.assertIn("ticker and volatility parameters are required", str(context.exception))
        
    def test_error_missing_ticker_volatility_in_predict(self):
        """Test error when ticker or volatility missing in predict."""
        ensemble = DiversifiedEnsemble()
        ensemble.fit(self.X_valid, self.ticker_valid, self.volatility_valid, self.y_valid)
        
        # Missing ticker
        with self.assertRaises(ValueError) as context:
            ensemble.predict(self.X_valid, None, self.volatility_valid)
        self.assertIn("ticker and volatility parameters are required", str(context.exception))
        
        # Missing volatility
        with self.assertRaises(ValueError) as context:
            ensemble.predict(self.X_valid, self.ticker_valid, None)
        self.assertIn("ticker and volatility parameters are required", str(context.exception))
        
    def test_error_missing_y_parameter(self):
        """Test error when y parameter missing in fit."""
        ensemble = DiversifiedEnsemble()
        
        with self.assertRaises(ValueError) as context:
            ensemble.fit(self.X_valid, self.ticker_valid, self.volatility_valid, None)
        self.assertIn("y parameter is required for fit()", str(context.exception))
        
    def test_error_non_binary_features(self):
        """Test error when features are not binary."""
        X_bad = pd.DataFrame({
            'strategy_1': [0.5, 1.5, 0.2, 1.8]  # Non-binary
        })
        ticker_bad = pd.Series(['ES', 'NQ', 'ES', 'NQ'])
        volatility_bad = pd.Series([0.15, 0.20, 0.18, 0.22])
        y_bad = pd.Series([0.01, -0.005, 0.02, -0.01])
        
        ensemble = DiversifiedEnsemble()
        
        with self.assertRaises(ValueError) as context:
            ensemble.fit(X_bad, ticker_bad, volatility_bad, y_bad)
            
        self.assertIn("binary", str(context.exception))
        
    def test_error_negative_volatility(self):
        """Test error when volatility contains negative values."""
        volatility_bad = pd.Series([0.15, -0.20, 0.18, 0.22])  # Negative value
        ticker_bad = pd.Series(['ES', 'NQ', 'ES', 'NQ'])
        X_bad = pd.DataFrame({
            'strategy_1': [0, 1, 0, 1]
        })
        y_bad = pd.Series([0.01, -0.005, 0.02, -0.01])
        
        ensemble = DiversifiedEnsemble()
        
        with self.assertRaises(ValueError) as context:
            ensemble.fit(X_bad, ticker_bad, volatility_bad, y_bad)
            
        self.assertIn("non-positive", str(context.exception))
        
    def test_error_mismatched_lengths(self):
        """Test error when parameter lengths don't match."""
        ensemble = DiversifiedEnsemble()
        
        # Ticker length mismatch
        ticker_short = self.ticker_valid[:50]
        with self.assertRaises(ValueError) as context:
            ensemble.fit(self.X_valid, ticker_short, self.volatility_valid, self.y_valid)
        self.assertIn("ticker parameter length", str(context.exception))
        
        # Volatility length mismatch
        volatility_short = self.volatility_valid[:50]
        with self.assertRaises(ValueError) as context:
            ensemble.fit(self.X_valid, self.ticker_valid, volatility_short, self.y_valid)
        self.assertIn("volatility parameter length", str(context.exception))
        
    def test_error_predict_not_fitted(self):
        """Test error when calling predict before fit."""
        ensemble = DiversifiedEnsemble()
        
        with self.assertRaises(ValueError) as context:
            ensemble.predict(self.X_valid, self.ticker_valid, self.volatility_valid)
            
        self.assertIn("fitted", str(context.exception))
        
    def test_error_predict_missing_features(self):
        """Test error when predict data is missing required features."""
        ensemble = DiversifiedEnsemble()
        ensemble.fit(self.X_valid, self.ticker_valid, self.volatility_valid, self.y_valid)
        
        X_missing = pd.DataFrame({
            'strategy_1': [0, 1, 0]
            # Missing strategy_2 and strategy_3
        })
        ticker_missing = pd.Series(['ES', 'NQ', 'YM'])
        volatility_missing = pd.Series([0.15, 0.20, 0.18])
        
        with self.assertRaises(ValueError) as context:
            ensemble.predict(X_missing, ticker_missing, volatility_missing)
            
        self.assertIn("Missing feature columns", str(context.exception))
        
    def test_error_unseen_tickers(self):
        """Test error when predict encounters unseen tickers."""
        ensemble = DiversifiedEnsemble()
        ensemble.fit(self.X_valid, self.ticker_valid, self.volatility_valid, self.y_valid)
        
        # Use ticker not seen in training
        X_test = self.X_valid.iloc[:3].copy()
        ticker_unseen = pd.Series(['UNKNOWN_TICKER'] * 3)
        volatility_test = pd.Series([0.15, 0.20, 0.18])
        
        with self.assertRaises(ValueError) as context:
            ensemble.predict(X_test, ticker_unseen, volatility_test)
            
        self.assertIn("Unseen ticker symbols", str(context.exception))
        
    def test_error_numpy_array_input(self):
        """Test error when passing numpy array instead of DataFrame."""
        X_array = np.array([[0, 1, 0], [1, 0, 1]])
        ticker_array = np.array(['ES', 'NQ'])
        volatility_array = np.array([0.15, 0.20])
        y_array = np.array([0.01, -0.01])
        
        ensemble = DiversifiedEnsemble()
        
        with self.assertRaises(ValueError) as context:
            ensemble.fit(X_array, ticker_array, volatility_array, y_array)
            
        self.assertIn("DataFrame input", str(context.exception))
        
    def test_error_save_config_not_fitted(self):
        """Test error when saving config before fitting."""
        ensemble = DiversifiedEnsemble()
        
        with self.assertRaises(ValueError) as context:
            ensemble.save_config("test.json")
            
        self.assertIn("fitted", str(context.exception))
        
    def test_error_save_config_no_path(self):
        """Test error when no save path provided."""
        ensemble = DiversifiedEnsemble()
        ensemble.fit(self.X_valid, self.ticker_valid, self.volatility_valid, self.y_valid)
        
        with self.assertRaises(ValueError) as context:
            ensemble.save_config()
            
        self.assertIn("filepath", str(context.exception))
        
    def test_error_load_config_file_not_found(self):
        """Test error when config file doesn't exist."""
        ensemble = DiversifiedEnsemble()
        
        with self.assertRaises(FileNotFoundError):
            ensemble.load_config("nonexistent_file.json")
            
    def test_error_load_config_invalid_json(self):
        """Test error when config file has invalid JSON."""
        bad_config_path = os.path.join(self.temp_dir, "bad_config.json")
        with open(bad_config_path, 'w') as f:
            f.write("{ invalid json }")
            
        ensemble = DiversifiedEnsemble()
        
        with self.assertRaises(ValueError) as context:
            ensemble.load_config(bad_config_path)
            
        self.assertIn("Invalid JSON", str(context.exception))
        
    def test_error_custom_instrument_weights_missing_tickers(self):
        """Test error when custom instrument weights missing tickers."""
        custom_weights = {'ES': 0.5}  # Missing NQ and YM
        
        ensemble = DiversifiedEnsemble()
        
        with self.assertRaises(ValueError) as context:
            ensemble.fit(self.X_valid, self.ticker_valid, self.volatility_valid, self.y_valid,
                        instrument_weights=custom_weights)
        self.assertIn("instrument_weights missing tickers", str(context.exception))
            
    def test_string_representations(self):
        """Test __repr__ and __str__ methods."""
        # Test unfitted ensemble
        ensemble = DiversifiedEnsemble(target_volatility=0.18)
        
        repr_str = repr(ensemble)
        self.assertIn("DiversifiedEnsemble", repr_str)
        self.assertIn("0.18", repr_str)
        self.assertIn("fitted=False", repr_str)
        
        str_str = str(ensemble)
        self.assertIn("not fitted", str_str)
        self.assertIn("0.18", str_str)
        
        # Test fitted ensemble
        ensemble.fit(self.X_valid, self.ticker_valid, self.volatility_valid, self.y_valid)
        
        repr_str = repr(ensemble)
        self.assertIn("fitted=True", repr_str)
        self.assertIn("n_features=3", repr_str)
        self.assertIn("n_tickers=", repr_str)
        
        str_str = str(ensemble)
        self.assertIn("fitted", str_str)
        self.assertIn("Features: 3", str_str)
        self.assertIn("Tickers:", str_str)
        
    def test_edge_case_single_feature(self):
        """Test with single feature."""
        n_samples = 15
        np.random.seed(789)
        
        X_single = pd.DataFrame({
            'strategy_1': np.random.binomial(1, 0.4, n_samples)
        })
        ticker_single = pd.Series(np.random.choice(['ES', 'NQ'], n_samples))
        volatility_single = pd.Series(np.random.lognormal(np.log(0.15), 0.1, n_samples).clip(0.1, 0.3))
        y_single = pd.Series(np.random.normal(0, 0.01, n_samples))
        
        ensemble = DiversifiedEnsemble()
        ensemble.fit(X_single, ticker_single, volatility_single, y_single)
        
        self.assertTrue(ensemble.is_fitted_)
        self.assertEqual(len(ensemble.feature_names_), 1)
        self.assertAlmostEqual(ensemble.weights_['strategy_1'], 1.0)
        
        predictions = ensemble.predict(X_single, ticker_single, volatility_single)
        self.assertEqual(len(predictions), len(X_single))
        
    def test_edge_case_zero_variance_feature(self):
        """Test with feature that has zero variance (all same values)."""
        n_samples = 12
        np.random.seed(456)
        
        X_zero_var = pd.DataFrame({
            'strategy_1': [1] * n_samples,  # No variance (all 1s)
            'strategy_2': np.random.binomial(1, 0.5, n_samples)  # Has variance
        })
        ticker_zero_var = pd.Series(np.random.choice(['ES', 'NQ'], n_samples))
        volatility_zero_var = pd.Series(np.random.lognormal(np.log(0.15), 0.1, n_samples).clip(0.1, 0.3))
        y_zero_var = pd.Series(np.random.normal(0, 0.005, n_samples))
        
        ensemble = DiversifiedEnsemble()
        ensemble.fit(X_zero_var, ticker_zero_var, volatility_zero_var, y_zero_var)
        
        # Should handle zero variance gracefully
        self.assertTrue(ensemble.is_fitted_)
        predictions = ensemble.predict(X_zero_var, ticker_zero_var, volatility_zero_var)
        self.assertEqual(len(predictions), len(X_zero_var))

class TestDiversifiedEnsembleIntegration(unittest.TestCase):
    """Integration tests for DiversifiedEnsemble with realistic scenarios."""
    
    def test_realistic_trading_scenario(self):
        """Test with realistic trading data scenario."""
        np.random.seed(123)
        n_samples = 500
        
        # Create realistic multi-strategy data
        X_real = pd.DataFrame({
            'momentum': np.random.binomial(1, 0.25, n_samples),
            'mean_reversion': np.random.binomial(1, 0.30, n_samples),
            'trend_following': np.random.binomial(1, 0.35, n_samples),
            'volatility_breakout': np.random.binomial(1, 0.20, n_samples)
        })
        
        # Create realistic ticker and volatility data
        tickers = ['ES', 'NQ', 'YM', 'RTY']
        ticker_real = pd.Series(np.random.choice(tickers, n_samples))
        volatility_real = pd.Series(np.random.lognormal(np.log(0.16), 0.2, n_samples).clip(0.1, 0.4))
        
        # Create returns (will be ignored but required)
        y_real = pd.Series(np.random.normal(0, 0.015, n_samples))
        
        # Test full workflow
        ensemble = DiversifiedEnsemble(target_volatility=0.15)
        
        # Split data
        split = int(0.7 * n_samples)
        X_train, X_test = X_real.iloc[:split].copy(), X_real.iloc[split:].copy()
        ticker_train, ticker_test = ticker_real.iloc[:split].copy(), ticker_real.iloc[split:].copy()
        volatility_train, volatility_test = volatility_real.iloc[:split].copy(), volatility_real.iloc[split:].copy()
        y_train, y_test = y_real.iloc[:split].copy(), y_real.iloc[split:].copy()
        
        # Fit and predict
        ensemble.fit(X_train, ticker_train, volatility_train, y_train)
        train_pred = ensemble.predict(X_train, ticker_train, volatility_train)
        test_pred = ensemble.predict(X_test, ticker_test, volatility_test)
        
        # Basic sanity checks
        self.assertEqual(len(train_pred), len(X_train))
        self.assertEqual(len(test_pred), len(X_test))
        self.assertTrue(np.all(train_pred >= 0))
        self.assertTrue(np.all(test_pred >= 0))
        
    def test_config_persistence_workflow(self):
        """Test complete configuration save/load workflow."""
        np.random.seed(456)
        
        # Training phase
        X_train = pd.DataFrame({
            'strat_a': np.random.binomial(1, 0.3, 200),
            'strat_b': np.random.binomial(1, 0.4, 200)
        })
        ticker_train = pd.Series(np.random.choice(['ES', 'NQ'], 200))
        volatility_train = pd.Series(np.random.lognormal(np.log(0.18), 0.15, 200).clip(0.1, 0.35))
        y_train = pd.Series(np.random.normal(0, 0.01, 200))
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
            
        try:
            # Training ensemble
            training_ensemble = DiversifiedEnsemble(target_volatility=0.22, save_path=config_path)
            training_ensemble.fit(X_train, ticker_train, volatility_train, y_train)
            saved_path = training_ensemble.save_config()
            
            self.assertEqual(saved_path, config_path)
            
            # Production ensemble (clean slate)
            production_ensemble = DiversifiedEnsemble(config_path=config_path)
            
            # New data for production
            X_prod = pd.DataFrame({
                'strat_a': [1, 0, 1, 0],
                'strat_b': [0, 1, 1, 0]
            })
            ticker_prod = pd.Series(['ES', 'NQ', 'ES', 'NQ'])
            volatility_prod = pd.Series([0.15, 0.20, 0.18, 0.25])
            
            # Both should give identical predictions
            pred_train = training_ensemble.predict(X_prod, ticker_prod, volatility_prod)
            pred_prod = production_ensemble.predict(X_prod, ticker_prod, volatility_prod)
            
            np.testing.assert_array_almost_equal(pred_train, pred_prod)
            
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)

if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
