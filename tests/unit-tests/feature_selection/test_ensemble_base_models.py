"""
Unit tests for Ensemble Base Model Ownership Workflow

This module tests the complete workflow of:
1. Saving base models to feature_list files
2. Loading ensembles from feature_list files
3. Fitting ensembles with base models
4. Saving and loading ensemble models
5. Full end-to-end workflow
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import json
from datetime import datetime

# Add the project root to the path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ensemble.diversified_ensemble import DiversifiedEnsemble
from feature_selection.base_models import ContinuousBinningModel


class TestEnsembleBaseModelsWorkflow(unittest.TestCase):
    """Test cases for ensemble base model ownership workflow."""
    
    def setUp(self):
        """Set up test data and fixtures."""
        # Create consistent test data
        np.random.seed(42)
        self.n_samples = 200
        
        # Create dummy feature data (raw feature values, not binary)
        # Use standardized column names that can be parsed
        self.feature_data = pd.DataFrame({
            'rsi_signal_D_lookback_14': np.random.uniform(0, 100, self.n_samples),
            'momentum_signal_D_lookback_20': np.random.uniform(-0.1, 0.1, self.n_samples),
            'atr_pct_D_period_252': np.random.uniform(0.01, 0.05, self.n_samples),
        }, index=pd.date_range('2020-01-01', periods=self.n_samples, freq='D'))
        
        # Create target returns
        self.y = pd.Series(
            np.random.normal(0.001, 0.02, self.n_samples),
            index=self.feature_data.index
        )
        
        # Create ticker and volatility data
        tickers = ['ES', 'NQ', 'YM']
        self.ticker = pd.Series(
            np.random.choice(tickers, self.n_samples),
            index=self.feature_data.index
        )
        self.volatility = pd.Series(
            np.random.lognormal(np.log(0.15), 0.1, self.n_samples).clip(0.1, 0.3),
            index=self.feature_data.index
        )
        
        # Create normalization data (EWSD)
        self.normalization_data = pd.DataFrame({
            'rsi_signal_D_lookback_14': np.random.uniform(0.01, 0.05, self.n_samples),
            'momentum_signal_D_lookback_20': np.random.uniform(0.01, 0.05, self.n_samples),
            'atr_pct_D_period_252': np.random.uniform(0.01, 0.05, self.n_samples),
        }, index=self.feature_data.index)
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.control_file_path = os.path.join(self.temp_dir, 'test_ensemble.json')
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_base_model_save_to_feature_list(self):
        """Test saving a base model to feature_list file."""
        # Create a base model
        model = ContinuousBinningModel(n_bins=3, strategy='long')
        
        # Fit the model first to set feature_column from Series name
        feature_series = pd.Series(self.feature_data['rsi_signal_D_lookback_14'], name='rsi_signal_D_lookback_14')
        model.fit(feature_series, self.y)
        
        # Save to control file (model_name will be auto-generated as rsi_signal_D_lookback_14_long)
        model.save_to_feature_list(
            filepath=self.control_file_path,
            tickers=['ES', 'NQ']
        )
        
        # Verify file was created
        self.assertTrue(os.path.exists(self.control_file_path))
        
        # Load and verify structure
        with open(self.control_file_path, 'r') as f:
            control_file = json.load(f)
        
        # Check structure
        self.assertIn('metadata', control_file)
        self.assertIn('tickers', control_file)
        self.assertIn('base_models', control_file)
        self.assertEqual(len(control_file['base_models']), 1)
        
        # Check is_fit flag
        self.assertIn('is_fit', control_file['metadata'])
        self.assertFalse(control_file['metadata']['is_fit'])  # Should be False for unfitted
        
        # Check base model config
        model_config = control_file['base_models'][0]
        self.assertEqual(model_config['name'], 'rsi_signal_D_lookback_14_long')  # Auto-generated
        self.assertEqual(model_config['model_type'], 'continuous_binning')
        self.assertEqual(model_config['feature_column'], 'rsi_signal_D_lookback_14')
        self.assertEqual(model_config['strategy'], 'long')
        self.assertIn('constructor_params', model_config)
        self.assertEqual(model_config['constructor_params']['n_bins'], 3)
        self.assertEqual(model_config['constructor_params']['strategy'], 'long')
        
        # Check tickers
        self.assertIn('ES', control_file['tickers'])
        self.assertIn('NQ', control_file['tickers'])
        
        # Check fitted params are not present
        self.assertNotIn('fitted_base_models', control_file)
        self.assertNotIn('fitted_ensemble', control_file)
    
    def test_base_model_save_to_feature_list_short_strategy(self):
        """Test saving a base model with short strategy."""
        # Create a base model with short strategy
        model = ContinuousBinningModel(
            n_bins=5,
            strategy='short',
        )
        
        # Fit the model first to set feature_column
        feature_series = pd.Series(self.feature_data['momentum_signal_D_lookback_20'], name='momentum_signal_D_lookback_20')
        model.fit(feature_series, self.y)
        
        # Save to control file (model_name will be auto-generated as momentum_signal_D_lookback_20_short)
        model.save_to_feature_list(
            filepath=self.control_file_path,
            tickers=['YM']
        )
        
        # Load and verify strategy
        with open(self.control_file_path, 'r') as f:
            control_file = json.load(f)
        
        model_config = control_file['base_models'][0]
        self.assertEqual(model_config['name'], 'momentum_signal_D_lookback_20_short')  # Auto-generated
        self.assertEqual(model_config['strategy'], 'short')
        self.assertEqual(model_config['model_type'], 'continuous_binning')
        self.assertFalse(control_file['metadata']['is_fit'])
    
    def test_base_model_save_to_existing_feature_list(self):
        """Test adding multiple features to feature_list."""
        # Save first model
        model1 = ContinuousBinningModel(n_bins=3, strategy='long')
        feature_series1 = pd.Series(self.feature_data['rsi_signal_D_lookback_14'], name='rsi_signal_D_lookback_14')
        model1.fit(feature_series1, self.y)
        model1.save_to_feature_list(
            filepath=self.control_file_path,
            tickers=['ES']
        )
        
        # Save second model to same file
        model2 = ContinuousBinningModel(n_bins=5, strategy='long')
        feature_series2 = pd.Series(self.feature_data['momentum_signal_D_lookback_20'], name='momentum_signal_D_lookback_20')
        model2.fit(feature_series2, self.y)
        model2.save_to_feature_list(
            filepath=self.control_file_path,
            tickers=['NQ', 'YM']
        )
        
        # Load and verify both models
        with open(self.control_file_path, 'r') as f:
            control_file = json.load(f)
        
        self.assertEqual(len(control_file['base_models']), 2)
        model_names = [m['name'] for m in control_file['base_models']]
        self.assertIn('rsi_signal_D_lookback_14_long', model_names)
        self.assertIn('momentum_signal_D_lookback_20_long', model_names)
        
        # Check tickers are merged
        self.assertIn('ES', control_file['tickers'])
        self.assertIn('NQ', control_file['tickers'])
        self.assertIn('YM', control_file['tickers'])
        self.assertFalse(control_file['metadata']['is_fit'])
    
    def test_ensemble_initialize_from_control_file(self):
        """Test initializing ensemble from control file."""
        # Create control file
        model = ContinuousBinningModel(n_bins=3, strategy='long')
        feature_series = pd.Series(self.feature_data['rsi_signal_D_lookback_14'], name='rsi_signal_D_lookback_14')
        model.fit(feature_series, self.y)
        model.save_to_feature_list(
            filepath=self.control_file_path,
            tickers=['ES']
        )
        
        # Initialize ensemble from control file
        ensemble = DiversifiedEnsemble(
            target_volatility=0.15,
            control_file_path=self.control_file_path
        )
        
        # Verify base models are initialized
        self.assertEqual(len(ensemble.base_models), 1)
        self.assertIn('rsi_signal_D_lookback_14_long', ensemble.base_models)
        self.assertIsInstance(
            ensemble.base_models['rsi_signal_D_lookback_14_long'].binning_model,
            ContinuousBinningModel
        )
        
        # Verify base models are not fitted yet
        self.assertFalse(ensemble.base_models['rsi_signal_D_lookback_14_long'].is_fitted_)
        
        # Verify required columns
        self.assertEqual(ensemble.required_columns, ['rsi_signal_D_lookback_14'])
        self.assertEqual(ensemble.get_required_columns(), ['rsi_signal_D_lookback_14'])
        
        # Verify column to model mapping
        self.assertEqual(ensemble.column_to_model['rsi_signal_D_lookback_14'], 'rsi_signal_D_lookback_14_long')
    
    def test_ensemble_initialize_from_control_file_multiple_models(self):
        """Test initializing ensemble with multiple base models."""
        # Create control file with multiple models
        model1 = ContinuousBinningModel(n_bins=3, strategy='long')
        feature_series1 = pd.Series(self.feature_data['rsi_signal_D_lookback_14'], name='rsi_signal_D_lookback_14')
        model1.fit(feature_series1, self.y)
        model1.save_to_feature_list(
            filepath=self.control_file_path,
            tickers=['ES']
        )
        
        model2 = ContinuousBinningModel(n_bins=5, strategy='long')
        feature_series2 = pd.Series(self.feature_data['momentum_signal_D_lookback_20'], name='momentum_signal_D_lookback_20')
        model2.fit(feature_series2, self.y)
        model2.save_to_feature_list(
            filepath=self.control_file_path,
            tickers=['NQ']
        )
        
        # Initialize ensemble
        ensemble = DiversifiedEnsemble(
            target_volatility=0.15,
            control_file_path=self.control_file_path
        )
        
        # Verify both models are initialized
        self.assertEqual(len(ensemble.base_models), 2)
        self.assertIn('rsi_signal_D_lookback_14_long', ensemble.base_models)
        self.assertIn('momentum_signal_D_lookback_20_long', ensemble.base_models)
        
        # Verify required columns
        self.assertEqual(set(ensemble.required_columns), {'rsi_signal_D_lookback_14', 'momentum_signal_D_lookback_20'})
    
    def test_ensemble_fit_with_base_models(self):
        """Test fitting ensemble with base models."""
        # Create control file
        model = ContinuousBinningModel(n_bins=3, strategy='long')
        feature_series = pd.Series(self.feature_data['rsi_signal_D_lookback_14'], name='rsi_signal_D_lookback_14')
        model.fit(feature_series, self.y)
        model.save_to_feature_list(
            filepath=self.control_file_path,
            tickers=['ES']
        )
        
        # Initialize and fit ensemble
        ensemble = DiversifiedEnsemble(
            target_volatility=0.15,
            control_file_path=self.control_file_path
        )
        
        # Fit ensemble
        ensemble.fit(
            X=self.feature_data[['rsi_signal_D_lookback_14']],
            ticker=self.ticker,
            volatility=self.volatility,
            y=self.y,
            normalization_data=self.normalization_data[['rsi_signal_D_lookback_14']]
        )
        
        # Verify base model is fitted
        self.assertTrue(ensemble.base_models['rsi_signal_D_lookback_14_long'].is_fitted_)
        self.assertIsNotNone(ensemble.base_models['rsi_signal_D_lookback_14_long'].bin_edges_)
        self.assertTrue(
            len(ensemble.base_models['rsi_signal_D_lookback_14_long'].active_bins_by_strategy_['long']) >= 0
        )
        self.assertEqual(ensemble.base_models['rsi_signal_D_lookback_14_long'].feature_column, 'rsi_signal_D_lookback_14')
        
        # Verify ensemble is fitted
        self.assertTrue(ensemble.is_fitted_)
        self.assertIsNotNone(ensemble.weights_)
        self.assertIsNotNone(ensemble.exposure_fractions_)
        self.assertEqual(len(ensemble.feature_names_), 1)
        self.assertEqual(ensemble.feature_names_[0], 'rsi_signal_D_lookback_14_long')
    
    def test_ensemble_predict_with_base_models(self):
        """Test predicting with ensemble that uses base models."""
        # Create control file
        model = ContinuousBinningModel(n_bins=3, strategy='long')
        feature_series = pd.Series(self.feature_data['rsi_signal_D_lookback_14'], name='rsi_signal_D_lookback_14')
        model.fit(feature_series, self.y)
        model.save_to_feature_list(
            filepath=self.control_file_path,
            tickers=['ES']
        )
        
        # Initialize, fit, and predict
        ensemble = DiversifiedEnsemble(
            target_volatility=0.15,
            control_file_path=self.control_file_path
        )
        
        ensemble.fit(
            X=self.feature_data[['rsi_signal_D_lookback_14']],
            ticker=self.ticker,
            volatility=self.volatility,
            y=self.y,
            normalization_data=self.normalization_data[['rsi_signal_D_lookback_14']]
        )
        
        # Predict
        predictions = ensemble.predict(
            X=self.feature_data[['rsi_signal_D_lookback_14']].iloc[:10],
            ticker=self.ticker.iloc[:10],
            volatility=self.volatility.iloc[:10],
            normalization_data=self.normalization_data[['rsi_signal_D_lookback_14']].iloc[:10]
        )

        # Verify predictions - new vector format returns DataFrame
        # with columns ['ticker', 'model_name', 'forecast', 'signal']
        # One row per (sample, model) combination
        self.assertIsInstance(predictions, pd.DataFrame)
        self.assertIn('ticker', predictions.columns)
        self.assertIn('model_name', predictions.columns)
        self.assertIn('forecast', predictions.columns)
        self.assertIn('signal', predictions.columns)
        # 10 samples × 1 model = 10 rows
        self.assertEqual(len(predictions), 10)
        self.assertTrue(np.all(np.isfinite(predictions['forecast'])))
    
    def test_ensemble_save_control_file(self):
        """Test saving complete control file with fitted parameters."""
        # Create control file
        model = ContinuousBinningModel(n_bins=3, strategy='long')
        feature_series = pd.Series(self.feature_data['rsi_signal_D_lookback_14'], name='rsi_signal_D_lookback_14')
        model.fit(feature_series, self.y)
        model.save_to_feature_list(
            filepath=self.control_file_path,
            tickers=['ES']
        )
        
        # Initialize and fit ensemble
        ensemble = DiversifiedEnsemble(
            target_volatility=0.15,
            control_file_path=self.control_file_path
        )
        
        ensemble.fit(
            X=self.feature_data[['rsi_signal_D_lookback_14']],
            ticker=self.ticker,
            volatility=self.volatility,
            y=self.y,
            normalization_data=self.normalization_data[['rsi_signal_D_lookback_14']]
        )
        
        # Save control file (with fitted params)
        saved_path = ensemble.save_control_file(self.control_file_path)
        
        # Verify file was created
        self.assertTrue(os.path.exists(saved_path))
        
        # Load and verify structure
        with open(saved_path, 'r') as f:
            control_file = json.load(f)
        
        # Check structure
        self.assertIn('base_models', control_file)
        self.assertIn('fitted_base_models', control_file)
        self.assertIn('fitted_ensemble', control_file)
        self.assertIn('metadata', control_file)
        
        # Check is_fit flag
        self.assertTrue(control_file['metadata']['is_fit'])
        
        # Check fitted base models
        self.assertIn('rsi_signal_D_lookback_14_long', control_file['fitted_base_models'])
        fitted_model = control_file['fitted_base_models']['rsi_signal_D_lookback_14_long']
        self.assertEqual(fitted_model['model_version'], 'binning_v2')
        self.assertIn('bin_edges', fitted_model)
        self.assertIn('bin_stats', fitted_model)
        self.assertIn('position_multipliers_by_strategy', fitted_model)
        
        # Check fitted ensemble
        fitted_ensemble = control_file['fitted_ensemble']
        self.assertIn('weights', fitted_ensemble)
        self.assertIn('exposure_fractions', fitted_ensemble)
        self.assertIn('target_volatility', fitted_ensemble)
    
    def test_ensemble_load_control_file(self):
        """Test loading control file with fitted states."""
        # Create, fit, and save ensemble
        model = ContinuousBinningModel(n_bins=3, strategy='long')
        feature_series = pd.Series(self.feature_data['rsi_signal_D_lookback_14'], name='rsi_signal_D_lookback_14')
        model.fit(feature_series, self.y)
        model.save_to_feature_list(
            filepath=self.control_file_path,
            tickers=['ES']
        )
        
        ensemble1 = DiversifiedEnsemble(
            target_volatility=0.15,
            control_file_path=self.control_file_path
        )
        
        ensemble1.fit(
            X=self.feature_data[['rsi_signal_D_lookback_14']],
            ticker=self.ticker,
            volatility=self.volatility,
            y=self.y,
            normalization_data=self.normalization_data[['rsi_signal_D_lookback_14']]
        )
        
        ensemble1.save_control_file(self.control_file_path)
        
        # Load control file
        ensemble2 = DiversifiedEnsemble(
            target_volatility=0.15,
            control_file_path=self.control_file_path
        )
        
        # Verify base model is fitted
        self.assertTrue(ensemble2.base_models['rsi_signal_D_lookback_14_long'].is_fitted_)
        self.assertIsNotNone(ensemble2.base_models['rsi_signal_D_lookback_14_long'].bin_edges_)
        
        # Verify ensemble is fitted
        self.assertTrue(ensemble2.is_fitted_)
        self.assertIsNotNone(ensemble2.weights_)
        self.assertEqual(ensemble2.target_volatility_, 0.15)
        
        # Verify we can predict without refitting
        predictions = ensemble2.predict(
            X=self.feature_data[['rsi_signal_D_lookback_14']].iloc[:10],
            ticker=self.ticker.iloc[:10],
            volatility=self.volatility.iloc[:10],
            normalization_data=self.normalization_data[['rsi_signal_D_lookback_14']].iloc[:10]
        )

        # New vector format: DataFrame with one row per (sample, model)
        self.assertIsInstance(predictions, pd.DataFrame)
        # 10 samples × 1 model = 10 rows
        self.assertEqual(len(predictions), 10)
        self.assertTrue(np.all(np.isfinite(predictions['forecast'])))
    
    def test_full_workflow(self):
        """Test complete end-to-end workflow."""
        # Step 1: Save base models to control file
        model1 = ContinuousBinningModel(n_bins=3, strategy='long')
        feature_series1 = pd.Series(self.feature_data['rsi_signal_D_lookback_14'], name='rsi_signal_D_lookback_14')
        model1.fit(feature_series1, self.y)
        model1.save_to_feature_list(
            filepath=self.control_file_path,
            tickers=['ES', 'NQ']
        )
        
        model2 = ContinuousBinningModel(n_bins=5, strategy='long')
        feature_series2 = pd.Series(self.feature_data['momentum_signal_D_lookback_20'], name='momentum_signal_D_lookback_20')
        model2.fit(feature_series2, self.y)
        model2.save_to_feature_list(
            filepath=self.control_file_path,
            tickers=['ES', 'NQ']
        )
        
        # Step 2: Initialize ensemble from control file
        ensemble = DiversifiedEnsemble(
            target_volatility=0.15,
            control_file_path=self.control_file_path
        )
        
        # Verify initialization
        self.assertEqual(len(ensemble.base_models), 2)
        self.assertEqual(set(ensemble.required_columns), {'rsi_signal_D_lookback_14', 'momentum_signal_D_lookback_20'})
        self.assertFalse(ensemble.is_fitted_)  # Should not be fitted yet
        
        # Step 3: Fit ensemble
        ensemble.fit(
            X=self.feature_data[['rsi_signal_D_lookback_14', 'momentum_signal_D_lookback_20']],
            ticker=self.ticker,
            volatility=self.volatility,
            y=self.y,
            normalization_data=self.normalization_data[['rsi_signal_D_lookback_14', 'momentum_signal_D_lookback_20']]
        )
        
        # Verify fit
        self.assertTrue(ensemble.is_fitted_)
        self.assertTrue(all(model.is_fitted_ for model in ensemble.base_models.values()))
        
        # Step 4: Save control file (with fitted params)
        ensemble.save_control_file(self.control_file_path)
        self.assertTrue(os.path.exists(self.control_file_path))
        
        # Verify is_fit flag in saved file
        with open(self.control_file_path, 'r') as f:
            control_file = json.load(f)
        self.assertTrue(control_file['metadata']['is_fit'])
        
        # Step 5: Load control file
        ensemble_loaded = DiversifiedEnsemble(
            target_volatility=0.15,
            control_file_path=self.control_file_path
        )
        
        # Verify loaded state
        self.assertTrue(ensemble_loaded.is_fitted_)
        self.assertTrue(all(model.is_fitted_ for model in ensemble_loaded.base_models.values()))
        
        # Step 6: Predict with loaded ensemble
        predictions = ensemble_loaded.predict(
            X=self.feature_data[['rsi_signal_D_lookback_14', 'momentum_signal_D_lookback_20']].iloc[:20],
            ticker=self.ticker.iloc[:20],
            volatility=self.volatility.iloc[:20],
            normalization_data=self.normalization_data[['rsi_signal_D_lookback_14', 'momentum_signal_D_lookback_20']].iloc[:20]
        )

        # New vector format: DataFrame with one row per (sample, model)
        self.assertIsInstance(predictions, pd.DataFrame)
        # 20 samples × 2 models = 40 rows
        self.assertEqual(len(predictions), 40)
        self.assertTrue(np.all(np.isfinite(predictions['forecast'])))
    
    def test_ensemble_initialization_validation(self):
        """Test that ensemble requires control_file_path."""
        # Test that None raises error
        with self.assertRaises(ValueError) as context:
            DiversifiedEnsemble(target_volatility=0.15)
        self.assertIn('control_file_path', str(context.exception))
    
    def test_ensemble_fit_missing_columns(self):
        """Test that fit raises error if required columns are missing."""
        model = ContinuousBinningModel(n_bins=3, strategy='long')
        feature_series = pd.Series(self.feature_data['rsi_signal_D_lookback_14'], name='rsi_signal_D_lookback_14')
        model.fit(feature_series, self.y)
        model.save_to_feature_list(
            filepath=self.control_file_path,
            tickers=['ES']
        )
        
        ensemble = DiversifiedEnsemble(
            target_volatility=0.15,
            control_file_path=self.control_file_path
        )
        
        # Try to fit with missing column
        with self.assertRaises(ValueError) as context:
            ensemble.fit(
                X=self.feature_data[['momentum_signal_D_lookback_20']],  # Missing rsi_signal_D_lookback_14
                ticker=self.ticker,
                volatility=self.volatility,
                y=self.y
            )
        self.assertIn('Missing required feature columns', str(context.exception))
    
    def test_base_model_fit_without_series_name(self):
        """Test that fit raises error if Series has no name."""
        model = ContinuousBinningModel(n_bins=3, strategy='long')
        # Create Series without name (explicitly set name=None)
        feature_series = pd.Series(self.feature_data['rsi_signal_D_lookback_14'].values, name=None)
        
        with self.assertRaises(ValueError) as context:
            model.fit(feature_series, self.y)
        self.assertIn('must have a name attribute', str(context.exception))
    
    def test_base_model_save_without_fit(self):
        """Test that save_to_feature_list raises error if fit() not called."""
        model = ContinuousBinningModel(n_bins=3, strategy='long')
        
        with self.assertRaises(ValueError) as context:
            model.save_to_feature_list(filepath=self.control_file_path)
        self.assertIn('feature_column not set', str(context.exception))
        self.assertIn('Call fit()', str(context.exception))
    
    def test_model_name_auto_generation(self):
        """Test that model_name is automatically generated as {feature_column}_{strategy}."""
        # Test long strategy
        model_long = ContinuousBinningModel(n_bins=3, strategy='long')
        feature_series = pd.Series(self.feature_data['rsi_signal_D_lookback_14'], name='rsi_signal_D_lookback_14')
        model_long.fit(feature_series, self.y)
        model_long.save_to_feature_list(filepath=self.control_file_path)
        
        with open(self.control_file_path, 'r') as f:
            control_file = json.load(f)
        
        self.assertEqual(control_file['base_models'][0]['name'], 'rsi_signal_D_lookback_14_long')
        
        # Test short strategy
        model_short = ContinuousBinningModel(n_bins=5, strategy='short')
        feature_series2 = pd.Series(self.feature_data['momentum_signal_D_lookback_20'], name='momentum_signal_D_lookback_20')
        model_short.fit(feature_series2, self.y)
        model_short.save_to_feature_list(filepath=self.control_file_path)
        
        with open(self.control_file_path, 'r') as f:
            control_file = json.load(f)
        
        # Should have both models now
        model_names = [m['name'] for m in control_file['base_models']]
        self.assertIn('rsi_signal_D_lookback_14_long', model_names)
        self.assertIn('momentum_signal_D_lookback_20_short', model_names)

    def test_control_file_is_fit_flag(self):
        """Test that is_fit flag is correctly set in control files."""
        # Create unfitted control file
        model = ContinuousBinningModel(n_bins=3, strategy='long')
        feature_series = pd.Series(self.feature_data['rsi_signal_D_lookback_14'], name='rsi_signal_D_lookback_14')
        model.fit(feature_series, self.y)
        model.save_to_feature_list(filepath=self.control_file_path)
        
        # Check is_fit is False
        with open(self.control_file_path, 'r') as f:
            control_file = json.load(f)
        self.assertFalse(control_file['metadata']['is_fit'])
        self.assertNotIn('fitted_base_models', control_file)
        self.assertNotIn('fitted_ensemble', control_file)
        
        # Fit and save
        ensemble = DiversifiedEnsemble(
            target_volatility=0.15,
            control_file_path=self.control_file_path
        )
        ensemble.fit(
            X=self.feature_data[['rsi_signal_D_lookback_14']],
            ticker=self.ticker,
            volatility=self.volatility,
            y=self.y,
            normalization_data=self.normalization_data[['rsi_signal_D_lookback_14']]
        )
        ensemble.save_control_file(self.control_file_path)
        
        # Check is_fit is True
        with open(self.control_file_path, 'r') as f:
            control_file = json.load(f)
        self.assertTrue(control_file['metadata']['is_fit'])
        self.assertIn('fitted_base_models', control_file)
        self.assertIn('fitted_ensemble', control_file)
    
    def test_column_name_validation(self):
        """Test that column names must follow standardized format."""
        model = ContinuousBinningModel(n_bins=3, strategy='long')
        # Use a non-standardized column name
        feature_series = pd.Series(self.feature_data['rsi_signal_D_lookback_14'], name='invalid_column_name')
        model.fit(feature_series, self.y)
        
        # Should raise error when saving
        with self.assertRaises(ValueError) as context:
            model.save_to_feature_list(filepath=self.control_file_path)
        self.assertIn('invalid timeframe', str(context.exception))
    
    def test_add_base_model_without_fitted_params(self):
        """Test adding base models to control file without fitted params."""
        from ensemble.ensemble_utils import add_feature_to_control_file
        
        # Create and save first model (fitted, using normal method)
        model1 = ContinuousBinningModel(n_bins=3, strategy='long')
        feature_series1 = pd.Series(self.feature_data['rsi_signal_D_lookback_14'], name='rsi_signal_D_lookback_14')
        model1.fit(feature_series1, self.y)
        model1.save_to_feature_list(
            filepath=self.control_file_path,
            tickers=['ES']
        )
        
        # Add second model using utility function directly (simulating adding without fitting)
        # This demonstrates that we can add base models to control files without fitted params
        feature_config2 = {
            'name': 'momentum_signal_D_lookback_20_long',
            'model_type': 'continuous_binning',
            'feature_column': 'momentum_signal_D_lookback_20',
            'strategy': 'long',
            'constructor_params': {
                'n_bins': 5,
                'bin_counts': [5],
                'strategy': 'long',
            }
        }
        add_feature_to_control_file(
            filepath=self.control_file_path,
            feature_config=feature_config2,
            tickers=['NQ']
        )
        
        # Verify both models are in control file
        with open(self.control_file_path, 'r') as f:
            control_file = json.load(f)
        
        self.assertEqual(len(control_file['base_models']), 2)
        model_names = [m['name'] for m in control_file['base_models']]
        self.assertIn('rsi_signal_D_lookback_14_long', model_names)
        self.assertIn('momentum_signal_D_lookback_20_long', model_names)
        self.assertFalse(control_file['metadata']['is_fit'])
        
        # Verify we can initialize ensemble from this control file
        ensemble = DiversifiedEnsemble(
            target_volatility=0.15,
            control_file_path=self.control_file_path
        )
        
        # Both models should be initialized but not fitted
        self.assertEqual(len(ensemble.base_models), 2)
        self.assertFalse(ensemble.base_models['rsi_signal_D_lookback_14_long'].is_fitted_)
        self.assertFalse(ensemble.base_models['momentum_signal_D_lookback_20_long'].is_fitted_)
    
    def test_load_control_file_partial_fitted_base_models(self):
        """Test loading control file with is_fit=True where some models have fitted params and some don't."""
        # Create control file with two models
        model1 = ContinuousBinningModel(n_bins=3, strategy='long')
        feature_series1 = pd.Series(self.feature_data['rsi_signal_D_lookback_14'], name='rsi_signal_D_lookback_14')
        model1.fit(feature_series1, self.y)
        model1.save_to_feature_list(
            filepath=self.control_file_path,
            tickers=['ES']
        )
        
        model2 = ContinuousBinningModel(n_bins=5, strategy='long')
        feature_series2 = pd.Series(self.feature_data['momentum_signal_D_lookback_20'], name='momentum_signal_D_lookback_20')
        model2.fit(feature_series2, self.y)
        model2.save_to_feature_list(
            filepath=self.control_file_path,
            tickers=['NQ']
        )
        
        # Initialize and fit ensemble with all features
        ensemble = DiversifiedEnsemble(
            target_volatility=0.15,
            control_file_path=self.control_file_path
        )
        
        # Fit with all features
        ensemble.fit(
            X=self.feature_data[['rsi_signal_D_lookback_14', 'momentum_signal_D_lookback_20']],
            ticker=self.ticker,
            volatility=self.volatility,
            y=self.y,
            normalization_data=self.normalization_data[['rsi_signal_D_lookback_14', 'momentum_signal_D_lookback_20']]
        )
        
        # Manually create a control file with partial fitted params
        # (model1 fitted, model2 not fitted)
        with open(self.control_file_path, 'r') as f:
            control_file = json.load(f)
        
        # Create partial fitted_base_models (only model1)
        fitted_base_models = {
            'rsi_signal_D_lookback_14_long': (
                ensemble.base_models['rsi_signal_D_lookback_14_long'].binning_model.get_fitted_params()
            )
            # Note: momentum model is NOT in fitted_base_models
        }
        
        control_file['metadata']['is_fit'] = True
        control_file['fitted_base_models'] = fitted_base_models
        control_file['fitted_ensemble'] = {
            'weights': ensemble.weights_,
            'exposure_fractions': ensemble.exposure_fractions_,
            'feature_names': ensemble.feature_names_,
            'target_volatility': ensemble.target_volatility_,
            'unique_tickers': ensemble.unique_tickers_,
            'instrument_weights': ensemble.instrument_weights_,
            'n_tickers': ensemble.n_tickers_
        }
        
        # Save modified control file
        with open(self.control_file_path, 'w') as f:
            json.dump(control_file, f, indent=2)
        
        # Load control file
        ensemble_loaded = DiversifiedEnsemble(
            target_volatility=0.15,
            control_file_path=self.control_file_path
        )
        
        # Verify model1 is fitted (has fitted params)
        self.assertTrue(ensemble_loaded.base_models['rsi_signal_D_lookback_14_long'].is_fitted_)
        self.assertIsNotNone(ensemble_loaded.base_models['rsi_signal_D_lookback_14_long'].bin_edges_)
        
        # Verify model2 is NOT fitted (no fitted params)
        self.assertFalse(ensemble_loaded.base_models['momentum_signal_D_lookback_20_long'].is_fitted_)
        self.assertIsNone(ensemble_loaded.base_models['momentum_signal_D_lookback_20_long'].bin_edges_)
        
        # Verify ensemble is fitted
        self.assertTrue(ensemble_loaded.is_fitted_)
        
        # Verify we can still use the ensemble (model2 will be fitted during predict if needed)
        # Actually, predict requires all base models to generate signals, so model2 needs to be fitted
        # But the point is that loading works correctly with partial fitted params
    
    def test_save_control_file_partial_fitted_base_models(self):
        """Test saving control file where only some base models are fitted."""
        # Create control file with two models
        model1 = ContinuousBinningModel(n_bins=3, strategy='long')
        feature_series1 = pd.Series(self.feature_data['rsi_signal_D_lookback_14'], name='rsi_signal_D_lookback_14')
        model1.fit(feature_series1, self.y)
        model1.save_to_feature_list(
            filepath=self.control_file_path,
            tickers=['ES']
        )
        
        model2 = ContinuousBinningModel(n_bins=5, strategy='long')
        feature_series2 = pd.Series(self.feature_data['momentum_signal_D_lookback_20'], name='momentum_signal_D_lookback_20')
        model2.fit(feature_series2, self.y)
        model2.save_to_feature_list(
            filepath=self.control_file_path,
            tickers=['NQ']
        )
        
        # Initialize ensemble
        ensemble = DiversifiedEnsemble(
            target_volatility=0.15,
            control_file_path=self.control_file_path
        )
        
        # Fit with all features
        ensemble.fit(
            X=self.feature_data[['rsi_signal_D_lookback_14', 'momentum_signal_D_lookback_20']],
            ticker=self.ticker,
            volatility=self.volatility,
            y=self.y,
            normalization_data=self.normalization_data[['rsi_signal_D_lookback_14', 'momentum_signal_D_lookback_20']]
        )
        
        # Manually unfit model2 to simulate partial fitting scenario
        # (e.g., model2 failed to fit or was intentionally not fitted)
        momentum_binning_model = ensemble.base_models['momentum_signal_D_lookback_20_long'].binning_model
        momentum_binning_model.is_fitted_ = False
        momentum_binning_model.bin_edges_ = None
        momentum_binning_model.bin_stats_ = {}
        momentum_binning_model.active_bins_by_strategy_ = {
            'long': [],
            'short': [],
            'long_short': [],
        }
        momentum_binning_model.position_multipliers_by_strategy_ = {
            'long': {},
            'short': {},
            'long_short': {},
        }
        
        # Save control file (should only save fitted params for model1)
        ensemble.save_control_file(self.control_file_path)
        
        # Verify saved file
        with open(self.control_file_path, 'r') as f:
            control_file = json.load(f)
        
        self.assertTrue(control_file['metadata']['is_fit'])
        self.assertIn('fitted_base_models', control_file)
        self.assertIn('fitted_ensemble', control_file)
        
        # Verify only model1 has fitted params
        self.assertIn('rsi_signal_D_lookback_14_long', control_file['fitted_base_models'])
        self.assertNotIn('momentum_signal_D_lookback_20_long', control_file['fitted_base_models'])
        
        # Verify both models are still in base_models list
        model_names = [m['name'] for m in control_file['base_models']]
        self.assertIn('rsi_signal_D_lookback_14_long', model_names)
        self.assertIn('momentum_signal_D_lookback_20_long', model_names)
        
        # Verify we can load this control file
        ensemble_loaded = DiversifiedEnsemble(
            target_volatility=0.15,
            control_file_path=self.control_file_path
        )
        
        # Verify model1 is fitted
        self.assertTrue(ensemble_loaded.base_models['rsi_signal_D_lookback_14_long'].is_fitted_)
        
        # Verify model2 is NOT fitted
        self.assertFalse(ensemble_loaded.base_models['momentum_signal_D_lookback_20_long'].is_fitted_)
    
    def test_load_control_file_empty_fitted_base_models(self):
        """Test loading control file with is_fit=True but empty fitted_base_models dict."""
        # Create control file
        model = ContinuousBinningModel(n_bins=3, strategy='long')
        feature_series = pd.Series(self.feature_data['rsi_signal_D_lookback_14'], name='rsi_signal_D_lookback_14')
        model.fit(feature_series, self.y)
        model.save_to_feature_list(
            filepath=self.control_file_path,
            tickers=['ES']
        )
        
        # Manually create control file with is_fit=True but empty fitted_base_models
        with open(self.control_file_path, 'r') as f:
            control_file = json.load(f)
        
        control_file['metadata']['is_fit'] = True
        control_file['fitted_base_models'] = {}  # Empty dict
        control_file['fitted_ensemble'] = {
            'weights': {'rsi_signal_D_lookback_14_long': 1.0},
            'exposure_fractions': {'rsi_signal_D_lookback_14_long': 0.3},
            'feature_names': ['rsi_signal_D_lookback_14_long'],
            'target_volatility': 0.15,
            'unique_tickers': ['ES', 'NQ'],
            'instrument_weights': {'ES': 0.5, 'NQ': 0.5},
            'n_tickers': 2
        }
        
        with open(self.control_file_path, 'w') as f:
            json.dump(control_file, f, indent=2)
        
        # Load control file
        ensemble = DiversifiedEnsemble(
            target_volatility=0.15,
            control_file_path=self.control_file_path
        )
        
        # Verify ensemble is fitted
        self.assertTrue(ensemble.is_fitted_)
        
        # Verify base model is NOT fitted (no fitted params)
        self.assertFalse(ensemble.base_models['rsi_signal_D_lookback_14_long'].is_fitted_)
        self.assertIsNone(ensemble.base_models['rsi_signal_D_lookback_14_long'].bin_edges_)
        
        # Verify we can still fit the base model
        ensemble.fit(
            X=self.feature_data[['rsi_signal_D_lookback_14']],
            ticker=self.ticker,
            volatility=self.volatility,
            y=self.y,
            normalization_data=self.normalization_data[['rsi_signal_D_lookback_14']]
        )
        
        # Now base model should be fitted
        self.assertTrue(ensemble.base_models['rsi_signal_D_lookback_14_long'].is_fitted_)


if __name__ == '__main__':
    unittest.main()
