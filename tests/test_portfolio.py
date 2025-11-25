"""
Unit tests for Portfolio Class

This module tests the Portfolio class functionality including:
1. Initialization from feature_list directory
2. Bias node spec extraction
3. Fit and predict workflows
4. Timeframe filtering
5. Multiple ensembles
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

from ensemble.portfolio import Portfolio
from ensemble.diversified_ensemble import DiversifiedEnsemble
from feature_selection.base_models import QuantileBinningModel
from utils.enums import TimeFrame, Ticker
import utils.helpers as helpers


class TestPortfolio(unittest.TestCase):
    """Test cases for Portfolio class."""
    
    def setUp(self):
        """Set up test data and fixtures."""
        # Create consistent test data
        np.random.seed(42)
        self.n_samples = 200
        
        # Create dummy feature data with timeframe in column names
        # Format: module_feature_tf_param1_val1
        self.feature_data = pd.DataFrame({
            'rsi_signal_D_lookback_14': np.random.uniform(0, 100, self.n_samples),
            'momentum_signal_D_lookback_20': np.random.uniform(-0.1, 0.1, self.n_samples),
            'rsi_signal_W_lookback_14': np.random.uniform(0, 100, self.n_samples),  # Weekly
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
        
        # Create normalization data
        self.normalization_data = pd.DataFrame({
            'rsi_signal_D_lookback_14': np.random.uniform(0.01, 0.05, self.n_samples),
            'momentum_signal_D_lookback_20': np.random.uniform(0.01, 0.05, self.n_samples),
            'rsi_signal_W_lookback_14': np.random.uniform(0.01, 0.05, self.n_samples),
            'atr_pct_D_period_252': np.random.uniform(0.01, 0.05, self.n_samples),
        }, index=self.feature_data.index)
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.control_file_dir = os.path.join(self.temp_dir, 'control_files')
        os.makedirs(self.control_file_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _create_control_file(self, filename: str, features: list, base_tf: str = 'D', is_fit: bool = False):
        """Helper to create a control file."""
        filepath = os.path.join(self.control_file_dir, filename)
        
        control_file = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'version': '2.0.0',
                'ensemble_name': os.path.splitext(filename)[0],
                'base_tf': base_tf,
                'is_fit': is_fit
            },
            'tickers': ['ES', 'NQ'],
            'base_models': []
        }
        
        for feature_col in features:
            # Create a simple base model config
            model_config = {
                'name': f"{feature_col}_long",
                'model_type': 'QuantileBinningModel',
                'feature_column': feature_col,
                'strategy': 'long',
                'constructor_params': {
                    'n_bins': 3,
                    'selection_metric': 'sortino'
                }
            }
            control_file['base_models'].append(model_config)
        
        if is_fit:
            # Add dummy fitted params
            control_file['fitted_base_models'] = {}
            control_file['fitted_ensemble'] = {
                'weights': {},
                'exposure_fractions': {},
                'feature_names': [],
                'target_volatility': 0.15,
                'unique_tickers': ['ES', 'NQ'],
                'instrument_weights': {'ES': 0.5, 'NQ': 0.5},
                'n_tickers': 2
            }
        
        with open(filepath, 'w') as f:
            json.dump(control_file, f, indent=2)
        
        return filepath
    
    def test_portfolio_initialization_from_control_file_dir(self):
        """Test Portfolio initialization from control file directory."""
        # Create control files
        self._create_control_file(
            'indices_D.json',
            ['rsi_signal_D_lookback_14', 'momentum_signal_D_lookback_20'],
            base_tf='D',
            is_fit=False
        )
        
        # Initialize portfolio (is_fit=False for unfitted ensembles)
        portfolio = Portfolio(control_file_dir=self.control_file_dir, is_fit=False)
        
        # Verify portfolio was initialized
        self.assertEqual(len(portfolio.ensembles), 1)
        self.assertIn('indices_D', portfolio.ensembles)
        
        ensemble, base_tf = portfolio.ensembles['indices_D']
        self.assertIsInstance(ensemble, DiversifiedEnsemble)
        self.assertEqual(base_tf, TimeFrame.D)
        self.assertFalse(ensemble.is_fitted_)  # Should not be fitted
    
    def test_portfolio_initialization_multiple_ensembles(self):
        """Test Portfolio initialization with multiple ensembles."""
        # Create multiple control files
        self._create_control_file(
            'indices_D.json',
            ['rsi_signal_D_lookback_14', 'momentum_signal_D_lookback_20'],
            base_tf='D',
            is_fit=False
        )
        self._create_control_file(
            'indices_W.json',
            ['rsi_signal_W_lookback_14'],
            base_tf='W',
            is_fit=False
        )
        
        # Initialize portfolio (is_fit=False for unfitted ensembles)
        portfolio = Portfolio(control_file_dir=self.control_file_dir, is_fit=False)
        
        # Verify both ensembles were loaded
        self.assertEqual(len(portfolio.ensembles), 2)
        self.assertIn('indices_D', portfolio.ensembles)
        self.assertIn('indices_W', portfolio.ensembles)
        
        # Verify timeframes
        _, base_tf_d = portfolio.ensembles['indices_D']
        _, base_tf_w = portfolio.ensembles['indices_W']
        self.assertEqual(base_tf_d, TimeFrame.D)
        self.assertEqual(base_tf_w, TimeFrame.W)
    
    def test_get_required_bias_nodes(self):
        """Test extracting required bias nodes from control files."""
        # Create control files
        self._create_control_file(
            'indices_D.json',
            ['rsi_signal_D_lookback_14', 'momentum_signal_D_lookback_20'],
            base_tf='D',
            is_fit=False
        )
        
        # Initialize portfolio (is_fit=False for unfitted ensembles)
        portfolio = Portfolio(control_file_dir=self.control_file_dir, is_fit=False)
        
        # Get required bias nodes
        bias_nodes = portfolio.get_required_bias_nodes()
        
        # Verify we got bias node specs
        self.assertIsInstance(bias_nodes, list)
        self.assertGreater(len(bias_nodes), 0)
        
        # Check structure
        for spec in bias_nodes:
            self.assertIn('module_name', spec)
            self.assertIn('timeframes', spec)
            self.assertIn('params', spec)
            self.assertIsInstance(spec['timeframes'], list)
    
    def test_portfolio_fit(self):
        """Test Portfolio fit method."""
        # Create control file
        self._create_control_file(
            'indices_D.json',
            ['rsi_signal_D_lookback_14', 'momentum_signal_D_lookback_20'],
            base_tf='D',
            is_fit=False
        )
        
        # Initialize portfolio (is_fit=False for unfitted ensembles)
        portfolio = Portfolio(control_file_dir=self.control_file_dir, is_fit=False)
        
        # Fit portfolio
        portfolio.fit(
            X=self.feature_data,
            ticker=self.ticker,
            volatility=self.volatility,
            y=self.y,
            normalization_data=self.normalization_data
        )
        
        # Verify ensembles are fitted
        for ensemble_name, (ensemble, _) in portfolio.ensembles.items():
            self.assertTrue(ensemble.is_fitted_)
    
    def test_portfolio_predict(self):
        """Test Portfolio predict method."""
        # Create control file
        self._create_control_file(
            'indices_D.json',
            ['rsi_signal_D_lookback_14', 'momentum_signal_D_lookback_20'],
            base_tf='D',
            is_fit=False
        )
        
        # Initialize and fit portfolio
        portfolio = Portfolio(control_file_dir=self.control_file_dir)
        portfolio.fit(
            X=self.feature_data,
            ticker=self.ticker,
            volatility=self.volatility,
            y=self.y,
            normalization_data=self.normalization_data
        )
        
        # Predict (MLManager feeds one timeframe at a time)
        predictions = portfolio.predict(
            X=self.feature_data[['rsi_signal_D_lookback_14', 'momentum_signal_D_lookback_20']],
            ticker=self.ticker,
            volatility=self.volatility,
            timeframe=TimeFrame.D,
            normalization_data=self.normalization_data[['rsi_signal_D_lookback_14', 'momentum_signal_D_lookback_20']]
        )
        
        # Verify output format
        self.assertIsInstance(predictions, pd.DataFrame)
        self.assertIn('ticker', predictions.columns)
        self.assertIn('%_to_risk', predictions.columns)
        self.assertEqual(len(predictions), len(self.feature_data))
        
        # Verify predictions are averaged (should be finite values)
        self.assertTrue(np.all(np.isfinite(predictions['%_to_risk'])))
    
    def test_timeframe_filtering(self):
        """Test that Portfolio filters features by timeframe correctly."""
        # Create control files for different timeframes
        self._create_control_file(
            'indices_D.json',
            ['rsi_signal_D_lookback_14', 'atr_pct_D_period_252'],
            base_tf='D',
            is_fit=False
        )
        self._create_control_file(
            'indices_W.json',
            ['rsi_signal_W_lookback_14'],
            base_tf='W',
            is_fit=False
        )
        
        # Initialize portfolio (is_fit=False for unfitted ensembles)
        portfolio = Portfolio(control_file_dir=self.control_file_dir, is_fit=False)
        
        # Fit portfolio
        portfolio.fit(
            X=self.feature_data,
            ticker=self.ticker,
            volatility=self.volatility,
            y=self.y,
            normalization_data=self.normalization_data
        )
        
        # Verify each ensemble only sees its timeframe features
        ensemble_d, _ = portfolio.ensembles['indices_D']
        ensemble_w, _ = portfolio.ensembles['indices_W']
        
        # Daily ensemble should have D features
        required_d = ensemble_d.get_required_columns()
        self.assertIn('rsi_signal_D_lookback_14', required_d)
        self.assertIn('atr_pct_D_period_252', required_d)
        self.assertNotIn('rsi_signal_W_lookback_14', required_d)
        
        # Weekly ensemble should have W features
        required_w = ensemble_w.get_required_columns()
        self.assertIn('rsi_signal_W_lookback_14', required_w)
        self.assertNotIn('rsi_signal_D_lookback_14', required_w)
    
    def test_portfolio_predict_multiple_ensembles(self):
        """Test Portfolio predict with multiple ensembles - should average predictions."""
        # Create multiple control files
        self._create_control_file(
            'indices_D.json',
            ['rsi_signal_D_lookback_14'],
            base_tf='D',
            is_fit=False
        )
        self._create_control_file(
            'indices_W.json',
            ['rsi_signal_W_lookback_14'],
            base_tf='W',
            is_fit=False
        )
        
        # Initialize and fit portfolio
        portfolio = Portfolio(control_file_dir=self.control_file_dir)
        portfolio.fit(
            X=self.feature_data,
            ticker=self.ticker,
            volatility=self.volatility,
            y=self.y,
            normalization_data=self.normalization_data
        )
        
        # Get individual ensemble predictions for comparison
        ensemble_d = portfolio.ensembles['indices_D'][0]
        ensemble_w = portfolio.ensembles['indices_W'][0]
        
        # Get individual ensemble predictions for comparison
        # D ensemble gets D features
        pred_d = ensemble_d.predict(
            X=self.feature_data[['rsi_signal_D_lookback_14']],
            ticker=self.ticker,
            volatility=self.volatility,
            normalization_data=self.normalization_data[['rsi_signal_D_lookback_14']]
        )
        # W ensemble gets W features
        pred_w = ensemble_w.predict(
            X=self.feature_data[['rsi_signal_W_lookback_14']],
            ticker=self.ticker,
            volatility=self.volatility,
            normalization_data=self.normalization_data[['rsi_signal_W_lookback_14']]
        )
        
        # Predict with portfolio for D timeframe (should only use D ensemble)
        predictions_d = portfolio.predict(
            X=self.feature_data[['rsi_signal_D_lookback_14']],
            ticker=self.ticker,
            volatility=self.volatility,
            timeframe=TimeFrame.D,
            normalization_data=self.normalization_data[['rsi_signal_D_lookback_14']]
        )
        
        # Predict with portfolio for W timeframe (should only use W ensemble)
        predictions_w = portfolio.predict(
            X=self.feature_data[['rsi_signal_W_lookback_14']],
            ticker=self.ticker,
            volatility=self.volatility,
            timeframe=TimeFrame.W,
            normalization_data=self.normalization_data[['rsi_signal_W_lookback_14']]
        )
        
        # Verify output format for D timeframe
        self.assertIsInstance(predictions_d, pd.DataFrame)
        self.assertIn('ticker', predictions_d.columns)
        self.assertIn('%_to_risk', predictions_d.columns)
        self.assertEqual(len(predictions_d), len(self.feature_data))
        
        # Verify D predictions match D ensemble (only one ensemble for D timeframe)
        np.testing.assert_array_almost_equal(
            predictions_d['%_to_risk'].values,
            pred_d,
            decimal=10,
            err_msg="Portfolio D predictions should match D ensemble predictions"
        )
        
        # Verify output format for W timeframe
        self.assertIsInstance(predictions_w, pd.DataFrame)
        self.assertIn('ticker', predictions_w.columns)
        self.assertIn('%_to_risk', predictions_w.columns)
        self.assertEqual(len(predictions_w), len(self.feature_data))
        
        # Verify W predictions match W ensemble (only one ensemble for W timeframe)
        np.testing.assert_array_almost_equal(
            predictions_w['%_to_risk'].values,
            pred_w,
            decimal=10,
            err_msg="Portfolio W predictions should match W ensemble predictions"
        )
    
    def test_save_control_files(self):
        """Test saving control files."""
        # Create control file
        self._create_control_file(
            'indices_D.json',
            ['rsi_signal_D_lookback_14'],
            base_tf='D',
            is_fit=False
        )
        
        # Initialize and fit portfolio
        portfolio = Portfolio(control_file_dir=self.control_file_dir)
        portfolio.fit(
            X=self.feature_data,
            ticker=self.ticker,
            volatility=self.volatility,
            y=self.y,
            normalization_data=self.normalization_data
        )
        
        # Save control files
        output_dir = os.path.join(self.temp_dir, 'saved_control_files')
        saved_paths = portfolio.save_control_files(output_dir)
        
        # Verify files were saved
        self.assertIn('indices_D', saved_paths)
        self.assertTrue(os.path.exists(saved_paths['indices_D']))
        
        # Verify is_fit flag in saved file
        with open(saved_paths['indices_D'], 'r') as f:
            control_file = json.load(f)
        self.assertTrue(control_file['metadata']['is_fit'])
        
        # Verify we can load from saved control files (is_fit=True for fitted ensembles)
        portfolio2 = Portfolio(control_file_dir=output_dir, is_fit=True)
        self.assertEqual(len(portfolio2.ensembles), 1)
        self.assertIn('indices_D', portfolio2.ensembles)
        self.assertTrue(portfolio2.ensembles['indices_D'][0].is_fitted_)
    
    def test_create_ml_manager_from_portfolio_bias_nodes(self):
        """Test creating MLManager using Portfolio's get_required_bias_nodes()."""
        # Create control file with RSI and momentum features
        self._create_control_file(
            'indices_D.json',
            ['rsi_signal_D_lookback_14', 'momentum_signal_D_lookback_20'],
            base_tf='D',
            is_fit=False
        )
        
        # Initialize portfolio
        portfolio = Portfolio(control_file_dir=self.control_file_dir, is_fit=False)
        
        # Get required bias nodes from portfolio
        bias_node_specs = portfolio.get_required_bias_nodes()
        
        # Verify we got specs
        self.assertIsInstance(bias_node_specs, list)
        self.assertGreater(len(bias_node_specs), 0)
        
        # Verify spec structure
        for spec in bias_node_specs:
            self.assertIn('module_name', spec)
            self.assertIn('timeframes', spec)
            self.assertIn('params', spec)
            self.assertIsInstance(spec['timeframes'], list)
        
        # Create MLManager using the bias node specs
        ml_manager = helpers.create_ml_manager(
            ticker=Ticker.ES,
            base_tf=TimeFrame.D,
            build_matrix=True,
            bias_node_specs=bias_node_specs
        )
        
        # Verify MLManager was created
        self.assertIsNotNone(ml_manager)
        self.assertEqual(ml_manager.ticker, Ticker.ES)
        self.assertEqual(ml_manager.base_tf, TimeFrame.D)
        
        # Verify MLManager has bias nodes
        self.assertGreater(len(ml_manager.bias_nodes), 0)
        
        # Verify MLManager has columns that match our feature names
        # The columns should include the features we specified
        ml_manager_columns = set(ml_manager.columns)
        
        # Check that we have columns for RSI and momentum
        # Column names follow format: module_feature_tf_param1_val1
        rsi_columns = [col for col in ml_manager_columns if 'rsi' in col.lower() and '14' in col]
        momentum_columns = [col for col in ml_manager_columns if 'momentum' in col.lower() and '20' in col]
        
        self.assertGreater(len(rsi_columns), 0, "MLManager should have RSI columns")
        self.assertGreater(len(momentum_columns), 0, "MLManager should have momentum columns")
        
        # Verify the matrix DataFrame structure
        self.assertIsNotNone(ml_manager.matrix)
        self.assertIsInstance(ml_manager.matrix, pd.DataFrame)
        self.assertEqual(len(ml_manager.matrix.columns), len(ml_manager.columns))
    
    def test_create_ml_manager_multiple_ensembles(self):
        """Test creating MLManager from Portfolio with multiple ensembles."""
        # Create control files for different timeframes
        self._create_control_file(
            'indices_D.json',
            ['rsi_signal_D_lookback_14', 'atr_pct_D_period_252'],
            base_tf='D',
            is_fit=False
        )
        self._create_control_file(
            'indices_W.json',
            ['rsi_signal_W_lookback_14'],
            base_tf='W',
            is_fit=False
        )
        
        # Initialize portfolio
        portfolio = Portfolio(control_file_dir=self.control_file_dir, is_fit=False)
        
        # Get required bias nodes (should aggregate from both ensembles)
        bias_node_specs = portfolio.get_required_bias_nodes()
        
        # Verify we got specs from both ensembles
        self.assertGreater(len(bias_node_specs), 0)
        
        # Check that we have specs for both D and W timeframes
        d_specs = [spec for spec in bias_node_specs if TimeFrame.D in spec['timeframes']]
        w_specs = [spec for spec in bias_node_specs if TimeFrame.W in spec['timeframes']]
        
        self.assertGreater(len(d_specs), 0, "Should have D timeframe specs")
        self.assertGreater(len(w_specs), 0, "Should have W timeframe specs")
        
        # Create MLManager using the aggregated bias node specs
        ml_manager = helpers.create_ml_manager(
            ticker=Ticker.ES,
            base_tf=TimeFrame.D,
            build_matrix=True,
            bias_node_specs=bias_node_specs
        )
        
        # Verify MLManager was created
        self.assertIsNotNone(ml_manager)
        
        # Verify MLManager has columns for both timeframes
        ml_manager_columns = set(ml_manager.columns)
        
        # Check for D timeframe columns (format: module_feature_D_params)
        d_columns = [col for col in ml_manager_columns if '_D' in col]
        self.assertGreater(len(d_columns), 0, "Should have D timeframe columns")
        
        # Check for W timeframe columns (format: module_feature_W_params)
        w_columns = [col for col in ml_manager_columns if '_W' in col]
        self.assertGreater(len(w_columns), 0, "Should have W timeframe columns")
        
        # Verify we have the specific features
        # RSI D: should match pattern like 'rsi_signal_D_lookback_14'
        rsi_d_columns = [col for col in ml_manager_columns if 'rsi' in col.lower() and '_D' in col and '14' in col]
        # RSI W: should match pattern like 'rsi_signal_W_lookback_14'
        rsi_w_columns = [col for col in ml_manager_columns if 'rsi' in col.lower() and '_W' in col and '14' in col]
        # ATR: should match pattern like 'atr_atr_D_period_252' or 'atr_atrPct_D_period_252'
        atr_columns = [col for col in ml_manager_columns if 'atr' in col.lower() and '252' in col]
        
        self.assertGreater(len(rsi_d_columns), 0, f"Should have RSI D columns. Found columns: {list(ml_manager_columns)}")
        self.assertGreater(len(rsi_w_columns), 0, f"Should have RSI W columns. Found columns: {list(ml_manager_columns)}")
        self.assertGreater(len(atr_columns), 0, f"Should have ATR columns. Found columns: {list(ml_manager_columns)}")
    
    def test_ml_manager_column_names_match_ensemble_features(self):
        """Test that MLManager column names match the feature columns in ensembles."""
        # Create control file with specific features
        features = ['rsi_signal_D_lookback_14', 'momentum_signal_D_lookback_20']
        self._create_control_file(
            'indices_D.json',
            features,
            base_tf='D',
            is_fit=False
        )
        
        # Initialize portfolio
        portfolio = Portfolio(control_file_dir=self.control_file_dir, is_fit=False)
        
        # Get required columns from ensemble
        ensemble = portfolio.ensembles['indices_D'][0]
        required_columns = ensemble.get_required_columns()
        
        # Get bias node specs and create MLManager
        bias_node_specs = portfolio.get_required_bias_nodes()
        ml_manager = helpers.create_ml_manager(
            ticker=Ticker.ES,
            base_tf=TimeFrame.D,
            build_matrix=True,
            bias_node_specs=bias_node_specs
        )
        
        # Verify MLManager columns include the required feature columns
        ml_manager_columns = set(ml_manager.columns)
        
        # Check that all required columns are present in MLManager
        for required_col in required_columns:
            # The exact column name might have slight variations, so check if any column matches
            matching_columns = [col for col in ml_manager_columns if required_col in col or col in required_col]
            self.assertGreater(
                len(matching_columns), 0,
                f"MLManager should have a column matching required feature: {required_col}"
            )
    
    def test_ml_manager_deduplication(self):
        """Test that Portfolio deduplicates bias node specs across ensembles."""
        # Create two control files with overlapping features
        self._create_control_file(
            'indices_D.json',
            ['rsi_signal_D_lookback_14', 'momentum_signal_D_lookback_20'],
            base_tf='D',
            is_fit=False
        )
        self._create_control_file(
            'commodities_D.json',
            ['rsi_signal_D_lookback_14', 'atr_pct_D_period_252'],  # RSI overlaps
            base_tf='D',
            is_fit=False
        )
        
        # Initialize portfolio
        portfolio = Portfolio(control_file_dir=self.control_file_dir, is_fit=False)
        
        # Get required bias nodes (should deduplicate RSI)
        bias_node_specs = portfolio.get_required_bias_nodes()
        
        # Count specs by module_name
        module_counts = {}
        for spec in bias_node_specs:
            module_name = spec['module_name']
            module_counts[module_name] = module_counts.get(module_name, 0) + 1
        
        # RSI should only appear once (deduplicated)
        self.assertEqual(module_counts.get('rsi', 0), 1, "RSI should be deduplicated")
        
        # Create MLManager
        ml_manager = helpers.create_ml_manager(
            ticker=Ticker.ES,
            base_tf=TimeFrame.D,
            build_matrix=True,
            bias_node_specs=bias_node_specs
        )
        
        # Verify MLManager was created successfully
        self.assertIsNotNone(ml_manager)
        self.assertGreater(len(ml_manager.bias_nodes), 0)


if __name__ == '__main__':
    unittest.main()

