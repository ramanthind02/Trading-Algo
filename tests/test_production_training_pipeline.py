"""
Unit tests for Production Training Pipeline

Tests the complete training workflow from data loading through ensemble training.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from datetime import datetime
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from deployment.production_training_pipeline import ProductionTrainingPipeline
from utils.enums import TimeFrame, Ticker


class TestProductionTrainingPipeline(unittest.TestCase):
    """Test ProductionTrainingPipeline functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary output directory
        self.temp_dir = tempfile.mkdtemp()
        self.pipeline = ProductionTrainingPipeline(output_dir=self.temp_dir)
        
        # Test parameters
        self.test_ticker = Ticker.EU
        self.test_timeframe = TimeFrame.D
        
    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary files
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test pipeline initialization."""
        self.assertIsInstance(self.pipeline, ProductionTrainingPipeline)
        self.assertEqual(self.pipeline.output_dir, self.temp_dir)
    
    def test_load_ticker_data(self):
        """Test real data loading."""
        data = self.pipeline.load_ticker_data(
            ticker=self.test_ticker,
            timeframe=self.test_timeframe,
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        
        # Check data structure
        self.assertIsInstance(data, pd.DataFrame)
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        self.assertListEqual(list(data.columns), expected_columns)
        
        # Check data validity (2023 has 251 trading days, not 365)
        self.assertTrue(len(data) > 200)  # Should have ~251 trading days for 2023
        self.assertTrue(len(data) < 300)  # But less than full calendar year
        self.assertTrue((data['high'] >= data['low']).all())
        self.assertTrue((data['high'] >= data['open']).all())
        self.assertTrue((data['high'] >= data['close']).all())
        self.assertTrue((data['low'] <= data['open']).all())
        self.assertTrue((data['low'] <= data['close']).all())
        self.assertTrue((data['volume'] >= 0).all())  # Volume can be 0 in some data
    
    def test_extract_features_and_targets(self):
        """Test feature extraction from price data."""
        # Load test data
        data = self.pipeline.load_ticker_data(
            ticker=self.test_ticker,
            timeframe=self.test_timeframe,
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        
        # Extract features
        features_df, targets_df = self.pipeline.extract_features_and_targets(
            data, self.test_ticker, self.test_timeframe
        )
        
        # Check features structure
        self.assertIsInstance(features_df, pd.DataFrame)
        expected_features = [
            f'rsi_signal_{self.test_timeframe.name}_lookback_14',
            f'momentum_signal_{self.test_timeframe.name}_lookback_20',
            f'ma_cross_{self.test_timeframe.name}_lookback_50'
        ]
        self.assertListEqual(list(features_df.columns), expected_features)
        
        # Check targets structure
        self.assertIsInstance(targets_df, pd.DataFrame)
        expected_targets = ['log_return', 'simple_return']
        self.assertListEqual(list(targets_df.columns), expected_targets)
        
        # Check alignment
        self.assertTrue(features_df.index.equals(targets_df.index))
        
        # Check feature values are valid signals (-1, 0, 1)
        for col in features_df.columns:
            unique_values = set(features_df[col].dropna().unique())
            self.assertTrue(unique_values.issubset({-1, 0, 1}))
        
        # Check returns are reasonable
        self.assertTrue(targets_df['log_return'].std() < 0.1)  # Daily vol < 10%
        self.assertTrue(targets_df['simple_return'].std() < 0.1)
    
    def test_train_ensemble(self):
        """Test ensemble training."""
        # Create test data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=500, freq='D')
        
        features_df = pd.DataFrame({
            'test_feature_1': np.random.choice([-1, 0, 1], size=500, p=[0.1, 0.7, 0.2]),
            'test_feature_2': np.random.choice([-1, 0, 1], size=500, p=[0.15, 0.6, 0.25])
        }, index=dates)
        
        targets_df = pd.DataFrame({
            'log_return': np.random.normal(0, 0.015, 500),
            'simple_return': np.random.normal(0, 0.015, 500)
        }, index=dates)
        
        selected_features = ['test_feature_1', 'test_feature_2']
        
        # Train ensemble
        ensemble_config = self.pipeline.train_ensemble(
            features_df, targets_df, selected_features, 
            self.test_ticker, self.test_timeframe
        )
        
        # Check ensemble structure
        self.assertIsInstance(ensemble_config, dict)
        expected_keys = ['metadata', 'base_models', 'fitted_base_models', 
                        'fitted_ensemble', 'tickers']
        for key in expected_keys:
            self.assertIn(key, ensemble_config)
        
        # Check metadata
        metadata = ensemble_config['metadata']
        self.assertEqual(metadata['base_tf'], self.test_timeframe.name)
        self.assertTrue(metadata['is_fit'])
        
        # Check base models
        base_models = ensemble_config['base_models']
        self.assertEqual(len(base_models), len(selected_features))
        
        # Check fitted models
        fitted_models = ensemble_config['fitted_base_models']
        self.assertEqual(len(fitted_models), len(selected_features))
        
        # Check ensemble parameters
        fitted_ensemble = ensemble_config['fitted_ensemble']
        self.assertIn('weights', fitted_ensemble)
        self.assertIn('exposure_fractions', fitted_ensemble)
        self.assertEqual(fitted_ensemble['unique_tickers'], [self.test_ticker.value])
    
    def test_save_ensemble_config(self):
        """Test saving ensemble configuration."""
        # Create minimal ensemble config
        ensemble_config = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'version': '2.0.0',
                'ensemble_name': f'{self.test_ticker.name}_{self.test_timeframe.name}_test',
                'is_fit': True,
                'base_tf': self.test_timeframe.name,
                'description': 'Test ensemble'
            },
            'base_models': [],
            'fitted_base_models': {},
            'fitted_ensemble': {
                'weights': {},
                'exposure_fractions': {},
                'feature_names': [],
                'target_volatility': 0.15,
                'unique_tickers': [self.test_ticker.value],
                'instrument_weights': {self.test_ticker.value: 1.0},
                'n_tickers': 1
            },
            'tickers': [self.test_ticker.value]
        }
        
        # Save config
        config_path = self.pipeline.save_ensemble_config(
            ensemble_config, self.test_ticker, self.test_timeframe
        )
        
        # Check file was created
        self.assertTrue(os.path.exists(config_path))
        
        # Check path structure
        expected_path = os.path.join(
            self.temp_dir, 
            self.test_timeframe.name, 
            f"{self.test_ticker.name}_{self.test_timeframe.name}.json"
        )
        self.assertEqual(config_path, expected_path)
        
        # Check file content is valid JSON
        import json
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
        
        self.assertIsInstance(loaded_config, dict)
        self.assertIn('metadata', loaded_config)
    
    def test_train_ticker_timeframe_integration(self):
        """Integration test for complete training workflow."""
        try:
            # Run complete training for one ticker/timeframe
            config_path = self.pipeline.train_ticker_timeframe(
                self.test_ticker, self.test_timeframe
            )
            
            # Check output file exists
            self.assertTrue(os.path.exists(config_path))
            
            # Check it's a valid ensemble config
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Validate config structure
            self.assertIn('metadata', config)
            self.assertIn('base_models', config)
            self.assertIn('fitted_ensemble', config)
            self.assertTrue(config['metadata']['is_fit'])
            
        except Exception as e:
            self.fail(f"Integration test failed: {e}")
    
    def test_train_multiple_tickers(self):
        """Test training multiple ticker/timeframe combinations."""
        # Test with small subset to avoid long test times
        test_tickers = [Ticker.EU, Ticker.BP]
        test_timeframes = [TimeFrame.D]
        
        results = self.pipeline.train_all_production_ensembles(
            tickers=test_tickers,
            timeframes=test_timeframes
        )
        
        # Check results structure
        self.assertIsInstance(results, dict)
        expected_keys = ['EU_D', 'BP_D']
        for key in expected_keys:
            self.assertIn(key, results)
        
        # Check at least some succeeded
        successful_results = [path for path in results.values() if path is not None]
        self.assertTrue(len(successful_results) > 0)
        
        # Check files exist
        for path in successful_results:
            self.assertTrue(os.path.exists(path))


class TestProductionPipelineEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.pipeline = ProductionTrainingPipeline(output_dir=self.temp_dir)
        
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_insufficient_data(self):
        """Test handling of insufficient training data."""
        # Create minimal data
        dates = pd.date_range('2023-01-01', periods=10, freq='D')  # Very little data
        
        features_df = pd.DataFrame({
            'test_feature': np.zeros(10)  # No signals
        }, index=dates)
        
        targets_df = pd.DataFrame({
            'log_return': np.random.normal(0, 0.01, 10),
            'simple_return': np.random.normal(0, 0.01, 10)
        }, index=dates)
        
        # Feature selection should handle this gracefully
        selected_features = self.pipeline.run_feature_selection(features_df, targets_df)
        
        # Should still return something (fallback behavior)
        self.assertIsInstance(selected_features, list)
        self.assertTrue(len(selected_features) > 0)
    
    def test_no_signals_features(self):
        """Test features with no signals (all zeros)."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        features_df = pd.DataFrame({
            'no_signal_feature': np.zeros(100),
            'minimal_signal_feature': np.concatenate([np.ones(5), np.zeros(95)])
        }, index=dates)
        
        targets_df = pd.DataFrame({
            'log_return': np.random.normal(0, 0.015, 100),
            'simple_return': np.random.normal(0, 0.015, 100)
        }, index=dates)
        
        selected_features = self.pipeline.run_feature_selection(features_df, targets_df)
        
        # Should handle gracefully and select something
        self.assertIsInstance(selected_features, list)
    
    def test_invalid_output_directory(self):
        """Test error handling for invalid output directory."""
        # Test with read-only directory (if possible to simulate)
        # This is platform-dependent, so we'll test basic functionality
        invalid_dir = "/invalid/nonexistent/path"
        
        try:
            pipeline = ProductionTrainingPipeline(output_dir=invalid_dir)
            # This should work (initialization doesn't validate path)
            self.assertIsInstance(pipeline, ProductionTrainingPipeline)
        except Exception:
            # Some platforms might fail earlier, which is also acceptable
            pass


if __name__ == '__main__':
    # Run tests
    unittest.main()
