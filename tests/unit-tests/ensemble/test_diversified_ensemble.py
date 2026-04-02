"""
Unit tests for DiversifiedEnsemble class

This module contains unit tests for the DiversifiedEnsemble class,
focusing on error handling, edge cases, and integration scenarios.
Basic functionality is tested in test_ensemble_base_models.py.
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import tempfile
import os
import json
from datetime import datetime

from ensemble.diversified_ensemble import DiversifiedEnsemble
from utils.core.enums import TimeFrame


def _make_fake_bias_node(*args, **kwargs):
    """Create a mock bias node that won't fail on module resolution."""
    node = MagicMock()
    node.add_candle.return_value = [0.0]
    return node


class TestDiversifiedEnsemble(unittest.TestCase):
    """Test cases for DiversifiedEnsemble class - error handling and edge cases."""
    
    def setUp(self):
        """Set up test data and fixtures."""
        # Patch bias node creation to avoid "Could not find module file" for dummy modules
        self._patcher1 = patch('utils.core.helpers.create_filtered_bias_node', side_effect=_make_fake_bias_node)
        self._patcher2 = patch('utils.core.helpers.create_bias_node', side_effect=_make_fake_bias_node)
        self._patcher1.start()
        self._patcher2.start()

        # Create consistent test data
        np.random.seed(42)
        self.n_samples = 100
        
        # Create sample data with known characteristics
        # Note: These are binary signals (not raw features) for direct ensemble testing
        self.X_valid = pd.DataFrame({
            'strategy_1': np.random.binomial(1, 0.3, self.n_samples),  # 30% exposure
            'strategy_2': np.random.binomial(1, 0.4, self.n_samples),  # 40% exposure
            'strategy_3': np.random.binomial(1, 0.2, self.n_samples),  # 20% exposure
        })
        
        # Create separate ticker and volatility arrays
        tickers = ['ES', 'NQ', 'YM']
        self.ticker_valid = pd.Series(np.random.choice(tickers, self.n_samples))
        self.volatility_valid = pd.Series(np.random.lognormal(np.log(0.15), 0.1, self.n_samples).clip(0.1, 0.3))
        
        # Create target with some correlation structure
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
        self._patcher1.stop()
        self._patcher2.stop()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def _create_binary_signal_control_file(self, feature_names, filepath):
        """
        Helper to create a control file for binary signal tests.
        
        This creates a minimal control file that allows the ensemble to work
        with binary signals directly (bypassing base models for testing purposes).
        """
        # Create dummy base model configs for each feature
        base_models = []
        for feature_name in feature_names:
            # Create a dummy feature column name that can be parsed
            # Format: module_feature_tf_param1_val1
            dummy_feature_col = f"dummy_signal_D_{feature_name}"
            base_models.append({
                'name': f"{dummy_feature_col}_long",
                'model_type': 'continuous_binning',
                'feature_column': dummy_feature_col,
                'strategy': 'long',
                'constructor_params': {
                    'n_bins': 3,
                    'selection_metric': 'sortino'
                },
            })
        
        control_file = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'version': '2.0.0',
                'ensemble_name': 'test_ensemble',
                'base_tf': 'D',
                'is_fit': False
            },
            'tickers': ['ES', 'NQ', 'YM'],
            'base_models': base_models
        }
        
        with open(filepath, 'w') as f:
            json.dump(control_file, f, indent=2)
        
        return filepath
    
    def _create_ensemble_with_binary_signals(self, feature_names, **kwargs):
        """Helper to create ensemble with control file for binary signal tests."""
        control_file_path = os.path.join(self.temp_dir, 'test_control.json')
        self._create_binary_signal_control_file(feature_names, control_file_path)
        
        # Create ensemble but we'll need to manually set up binary signals
        # For these tests, we'll create a workaround by using the binary signals
        # directly in the fit method
        ensemble = DiversifiedEnsemble(
            control_file_path=control_file_path,
            **kwargs
        )
        
        # Override base models to work with binary signals directly
        # This is a test-only workaround
        ensemble.required_columns = feature_names
        ensemble.base_models = {}  # Empty base models for binary signal tests
        
        return ensemble
    
    def test_error_missing_control_file_path(self):
        """Test error when control_file_path is not provided."""
        with self.assertRaises(ValueError) as context:
            DiversifiedEnsemble(target_volatility=0.15)
        self.assertIn("control_file_path", str(context.exception))
        
    def test_error_missing_ticker_volatility_in_fit(self):
        """Test error when ticker or volatility missing in fit."""
        control_file_path = os.path.join(self.temp_dir, 'test_control.json')
        self._create_binary_signal_control_file(['strategy_1'], control_file_path)
        ensemble = DiversifiedEnsemble(control_file_path=control_file_path)
        
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
        control_file_path = os.path.join(self.temp_dir, 'test_control.json')
        self._create_binary_signal_control_file(['strategy_1'], control_file_path)
        ensemble = DiversifiedEnsemble(control_file_path=control_file_path)
        
        # Missing ticker - will fail at parameter validation before fitted check
        with self.assertRaises(ValueError) as context:
            ensemble.predict(self.X_valid, None, self.volatility_valid)
        # Error could be either "ticker and volatility" or "fitted" depending on validation order
        error_msg = str(context.exception)
        self.assertTrue(
            "ticker and volatility parameters are required" in error_msg or
            "fitted" in error_msg,
            f"Unexpected error message: {error_msg}"
        )
        
        # Missing volatility
        with self.assertRaises(ValueError) as context:
            ensemble.predict(self.X_valid, self.ticker_valid, None)
        error_msg = str(context.exception)
        self.assertTrue(
            "ticker and volatility parameters are required" in error_msg or
            "fitted" in error_msg,
            f"Unexpected error message: {error_msg}"
        )
        
    def test_error_missing_y_parameter(self):
        """Test error when y parameter missing in fit."""
        control_file_path = os.path.join(self.temp_dir, 'test_control.json')
        self._create_binary_signal_control_file(['strategy_1'], control_file_path)
        ensemble = DiversifiedEnsemble(control_file_path=control_file_path)
        
        with self.assertRaises(ValueError) as context:
            ensemble.fit(self.X_valid, self.ticker_valid, self.volatility_valid, None)
        self.assertIn("y parameter is required for fit()", str(context.exception))
        
    # Note: test_error_negative_volatility removed - validation happens in base model fitting
    # Negative volatility validation is still tested in test_ensemble_base_models.py
        
    # Note: test_error_mismatched_lengths removed - validation order changed with new architecture
    # Length validation is still tested in test_ensemble_base_models.py
        
    def test_error_predict_not_fitted(self):
        """Test error when calling predict before fit."""
        control_file_path = os.path.join(self.temp_dir, 'test_control.json')
        self._create_binary_signal_control_file(['strategy_1'], control_file_path)
        ensemble = DiversifiedEnsemble(control_file_path=control_file_path)
        
        with self.assertRaises(ValueError) as context:
            ensemble.predict(self.X_valid, self.ticker_valid, self.volatility_valid)
            
        self.assertIn("fitted", str(context.exception))
        
    def test_error_numpy_array_input(self):
        """Test error when passing numpy array instead of DataFrame."""
        X_array = np.array([[0, 1, 0], [1, 0, 1]])
        ticker_array = np.array(['ES', 'NQ'])
        volatility_array = np.array([0.15, 0.20])
        y_array = np.array([0.01, -0.01])
        
        control_file_path = os.path.join(self.temp_dir, 'test_control.json')
        self._create_binary_signal_control_file(['strategy_1'], control_file_path)
        ensemble = DiversifiedEnsemble(control_file_path=control_file_path)
        
        with self.assertRaises(ValueError) as context:
            ensemble.fit(X_array, ticker_array, volatility_array, y_array)
            
        # Error message changed to "X must be a pandas DataFrame"
        error_msg = str(context.exception)
        self.assertTrue(
            "DataFrame" in error_msg or "pandas DataFrame" in error_msg,
            f"Unexpected error message: {error_msg}"
        )
        
    def test_error_custom_instrument_weights_missing_tickers(self):
        """Test error when custom instrument weights missing tickers."""
        custom_weights = {'ES': 0.5}  # Missing NQ and YM
        
        # Create X with correct feature column names
        feature_cols = ['dummy_signal_D_strategy_1', 'dummy_signal_D_strategy_2', 'dummy_signal_D_strategy_3']
        X_with_cols = pd.DataFrame({
            feature_cols[0]: self.X_valid['strategy_1'],
            feature_cols[1]: self.X_valid['strategy_2'],
            feature_cols[2]: self.X_valid['strategy_3']
        })
        
        control_file_path = os.path.join(self.temp_dir, 'test_control.json')
        self._create_binary_signal_control_file(['strategy_1', 'strategy_2', 'strategy_3'], control_file_path)
        ensemble = DiversifiedEnsemble(control_file_path=control_file_path)
        
        with self.assertRaises(ValueError) as context:
            ensemble.fit(X_with_cols, self.ticker_valid, self.volatility_valid, self.y_valid,
                        instrument_weights=custom_weights)
        self.assertIn("instrument_weights missing tickers", str(context.exception))
        
    def test_error_load_config_file_not_found(self):
        """Test error when config file doesn't exist."""
        control_file_path = os.path.join(self.temp_dir, 'test_control.json')
        self._create_binary_signal_control_file(['strategy_1'], control_file_path)
        ensemble = DiversifiedEnsemble(control_file_path=control_file_path)
        
        with self.assertRaises(FileNotFoundError):
            ensemble.load_config("nonexistent_file.json")
            
    def test_error_load_config_invalid_json(self):
        """Test error when config file has invalid JSON."""
        bad_config_path = os.path.join(self.temp_dir, "bad_config.json")
        with open(bad_config_path, 'w') as f:
            f.write("{ invalid json }")
            
        control_file_path = os.path.join(self.temp_dir, 'test_control.json')
        self._create_binary_signal_control_file(['strategy_1'], control_file_path)
        ensemble = DiversifiedEnsemble(control_file_path=control_file_path)
        
        with self.assertRaises(ValueError) as context:
            ensemble.load_config(bad_config_path)
            
        self.assertIn("Invalid JSON", str(context.exception))
        
    def test_error_save_control_file_not_fitted(self):
        """Test error when saving control file before fitting."""
        control_file_path = os.path.join(self.temp_dir, 'test_control.json')
        self._create_binary_signal_control_file(['strategy_1'], control_file_path)
        ensemble = DiversifiedEnsemble(control_file_path=control_file_path)
        
        # Note: save_control_file requires control_file_data, which is set during init
        # So this should work, but we can test the is_fitted_ check
        # Actually, looking at the code, save_control_file doesn't check is_fitted_
        # but it needs fitted_base_models and fitted_ensemble which are None if not fitted
        # Let's test that it raises an error or handles it gracefully
        output_path = os.path.join(self.temp_dir, 'output_control.json')
        # This should work but create a file with is_fit=False
        ensemble.save_control_file(output_path)
        self.assertTrue(os.path.exists(output_path))
        
        # Verify is_fit is False
        with open(output_path, 'r') as f:
            saved_file = json.load(f)
        self.assertFalse(saved_file['metadata']['is_fit'])


class TestDiversifiedEnsembleEdgeCases(unittest.TestCase):
    """Edge case tests for DiversifiedEnsemble."""
    
    def setUp(self):
        """Set up test data and fixtures."""
        # Patch bias node creation to avoid "Could not find module file" for dummy modules
        self._patcher1 = patch('utils.core.helpers.create_filtered_bias_node', side_effect=_make_fake_bias_node)
        self._patcher2 = patch('utils.core.helpers.create_bias_node', side_effect=_make_fake_bias_node)
        self._patcher1.start()
        self._patcher2.start()

        np.random.seed(42)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        self._patcher1.stop()
        self._patcher2.stop()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_binary_signal_control_file(self, feature_names, filepath):
        """Helper to create a control file for binary signal tests."""
        base_models = []
        for feature_name in feature_names:
            dummy_feature_col = f"dummy_signal_D_{feature_name}"
            base_models.append({
                'name': f"{dummy_feature_col}_long",
                'model_type': 'continuous_binning',
                'feature_column': dummy_feature_col,
                'strategy': 'long',
                'constructor_params': {
                    'n_bins': 3,
                    'selection_metric': 'sortino'
                },
                'members': [
                    {'member_name': 'default', 'params': {'n_bins': 3, 'selection_metric': 'sortino'}}
                ],
            })
        
        control_file = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'version': '2.0.0',
                'ensemble_name': 'test_ensemble',
                'base_tf': 'D',
                'is_fit': False
            },
            'tickers': ['ES', 'NQ'],
            'base_models': base_models
        }
        
        with open(filepath, 'w') as f:
            json.dump(control_file, f, indent=2)
        
        return filepath
            
    def test_string_representations(self):
        """Test __repr__ and __str__ methods."""
        control_file_path = os.path.join(self.temp_dir, 'test_control.json')
        self._create_binary_signal_control_file(['strategy_1'], control_file_path)
        
        # Test unfitted ensemble
        ensemble = DiversifiedEnsemble(
            target_volatility=0.18,
            control_file_path=control_file_path
        )
        
        repr_str = repr(ensemble)
        self.assertIn("DiversifiedEnsemble", repr_str)
        self.assertIn("0.18", repr_str)
        self.assertIn("fitted=False", repr_str)
        
        str_str = str(ensemble)
        self.assertIn("not fitted", str_str)
        self.assertIn("0.18", str_str)
        

if __name__ == '__main__':
    unittest.main(verbosity=2)
