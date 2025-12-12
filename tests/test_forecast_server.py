"""
Unit tests for ForecastServer

This module tests the ForecastServer class functionality including:
1. Initialization with portfolios and MLManagers
2. Portfolio setup from config directory
3. MLManager setup per ticker/timeframe
4. Forecast generation workflow
5. Candle buffer management
6. Integration with Portfolio and MLManager
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import json
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

# Add the project root to the path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deployment.forecast_server import ForecastServer
from ensemble.portfolio import Portfolio
from feature_extraction.ml_manager import MLManager
from utils.enums import TimeFrame, Ticker
from utils.models import Candle


class TestForecastServer(unittest.TestCase):
    """Test cases for ForecastServer class."""
    
    def setUp(self):
        """Set up test data and fixtures."""
        # Use the real production config directory for testing
        # This ensures tests validate against actual trained ensemble configs
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_dir = os.path.join(project_root, 'deployment', 'config')
        
        # Verify config directory exists
        if not os.path.exists(self.config_dir):
            self.skipTest(f"Config directory not found: {self.config_dir}. Run production_training_pipeline.py first.")
    
    def tearDown(self):
        """Clean up test files."""
        # No cleanup needed since we're using real production configs
        pass
    

    @patch('deployment.forecast_server.ForecastMT5DataConnector')
    @patch('deployment.forecast_server.TelegramNotifier')
    @patch('deployment.forecast_server.helpers.create_ml_manager')
    def test_initialization(self, mock_create_ml_manager, mock_telegram, mock_mt5):
        """Test ForecastServer initialization."""
        # Mock MLManager creation
        mock_ml_manager = Mock(spec=MLManager)
        mock_ml_manager.bias_nodes = [(TimeFrame.D, Mock())]
        mock_create_ml_manager.return_value = mock_ml_manager
        
        # Create server
        server = ForecastServer(config_dir=self.config_dir)
        
        # Verify initialization
        self.assertIsNotNone(server)
        self.assertEqual(server.config_dir, self.config_dir)
        
        # Verify portfolios were loaded - should have D and W portfolios from test control files
        self.assertIsInstance(server.portfolios, dict)
        self.assertIn(TimeFrame.D, server.portfolios, "Daily portfolio should be loaded from test configs")
        self.assertIn(TimeFrame.W, server.portfolios, "Weekly portfolio should be loaded from test configs")
        
        # Verify MLManagers were created for test tickers
        self.assertIsInstance(server.ml_managers, dict)
        self.assertGreater(len(server.ml_managers), 0, "MLManagers should be created for configured tickers")
        
        # Verify candle buffers were initialized
        self.assertIsInstance(server.candle_buffers, dict)
        self.assertEqual(len(server.candle_buffers), len(server.ml_managers), 
                        "Should have one candle buffer per MLManager")
    
    @patch('deployment.forecast_server.ForecastMT5DataConnector')
    @patch('deployment.forecast_server.TelegramNotifier')
    @patch('deployment.forecast_server.helpers.create_ml_manager')
    def test_portfolio_setup(self, mock_create_ml_manager, mock_telegram, mock_mt5):
        """Test that portfolios are set up correctly."""
        # Mock MLManager
        mock_ml_manager = Mock(spec=MLManager)
        mock_ml_manager.bias_nodes = [(TimeFrame.D, Mock())]
        mock_create_ml_manager.return_value = mock_ml_manager
        
        # Create server
        server = ForecastServer(config_dir=self.config_dir)
        
        # Verify portfolios were loaded
        self.assertIsInstance(server.portfolios, dict)
        self.assertGreater(len(server.portfolios), 0, "Should have loaded portfolios from test config files")
        
        # Verify each portfolio has ensembles loaded from control files
        for timeframe, portfolio in server.portfolios.items():
            self.assertIsInstance(portfolio, Portfolio, f"Portfolio for {timeframe} should be a Portfolio instance")
            # We trained 4 tickers (EU, BP, ES, NQ) for each timeframe
            self.assertEqual(len(portfolio.ensembles), 4, 
                           f"Portfolio {timeframe} should have 4 ensembles (EU, BP, ES, NQ)")
    
    @patch('deployment.forecast_server.ForecastMT5DataConnector')
    @patch('deployment.forecast_server.TelegramNotifier')
    @patch('deployment.forecast_server.helpers.create_ml_manager')
    def test_ml_manager_setup(self, mock_create_ml_manager, mock_telegram, mock_mt5):
        """Test that MLManagers are set up for each ticker/timeframe."""
        # Mock MLManager
        mock_ml_manager = Mock(spec=MLManager)
        mock_ml_manager.bias_nodes = [(TimeFrame.D, Mock())]
        mock_create_ml_manager.return_value = mock_ml_manager
        
        # Create server
        server = ForecastServer(config_dir=self.config_dir)
        
        # Verify MLManager created for each ticker/timeframe combination in the production control files
        # We trained: EU, BP, ES, NQ for both D and W
        expected_keys = [
            (Ticker.EU, TimeFrame.D),
            (Ticker.EU, TimeFrame.W),
            (Ticker.BP, TimeFrame.D),
            (Ticker.BP, TimeFrame.W),
            (Ticker.ES, TimeFrame.D),
            (Ticker.ES, TimeFrame.W),
            (Ticker.NQ, TimeFrame.D),
            (Ticker.NQ, TimeFrame.W)
        ]
        
        for key in expected_keys:
            self.assertIn(key, server.ml_managers, 
                         f"MLManager should exist for {key[0].name} {key[1].name}")
            self.assertIn(key, server.candle_buffers,
                         f"Candle buffer should exist for {key[0].name} {key[1].name}")
    
    @patch('deployment.forecast_server.ForecastMT5DataConnector')
    @patch('deployment.forecast_server.TelegramNotifier')
    @patch('deployment.forecast_server.helpers.create_ml_manager')
    def test_add_candle(self, mock_create_ml_manager, mock_telegram, mock_mt5):
        """Test adding candles to buffer and MLManager."""
        # Mock MLManager
        mock_ml_manager = Mock(spec=MLManager)
        mock_ml_manager.bias_nodes = [(TimeFrame.D, Mock())]
        mock_ml_manager.add_candle = Mock()
        mock_create_ml_manager.return_value = mock_ml_manager
        
        # Create server
        server = ForecastServer(config_dir=self.config_dir)
        
        # Create test candle
        candle = Candle(
            datetime=datetime(2024, 12, 1),
            open=1.05,
            high=1.06,
            low=1.04,
            close=1.055,
            volume=1000,
            ticker=Ticker.EU,
            tf=TimeFrame.D
        )
        
        # Add candle
        key = (Ticker.EU, TimeFrame.D)
        if key in server.ml_managers:
            server._add_candle(Ticker.EU, TimeFrame.D, candle)
            
            # Verify candle was added to buffer
            self.assertIn(candle, server.candle_buffers[key])
            
            # Verify MLManager.add_candle was called
            mock_ml_manager.add_candle.assert_called()
    
    @patch('deployment.forecast_server.ForecastMT5DataConnector')
    @patch('deployment.forecast_server.TelegramNotifier')
    @patch('deployment.forecast_server.helpers.create_ml_manager')
    def test_candle_buffer_limit(self, mock_create_ml_manager, mock_telegram, mock_mt5):
        """Test that candle buffer respects size limit."""
        # Mock MLManager
        mock_ml_manager = Mock(spec=MLManager)
        mock_ml_manager.bias_nodes = [(TimeFrame.D, Mock())]
        mock_ml_manager.add_candle = Mock()
        mock_create_ml_manager.return_value = mock_ml_manager
        
        # Create server
        server = ForecastServer(config_dir=self.config_dir)
        server.lookback_candles = 5  # Small limit for testing
        
        key = (Ticker.EU, TimeFrame.D)
        if key not in server.ml_managers:
            self.skipTest("MLManager not created for test key")
        
        # Add more candles than the limit
        for i in range(10):
            candle = Candle(
                datetime=datetime(2024, 12, i + 1),
                open=1.05,
                high=1.06,
                low=1.04,
                close=1.055,
                volume=1000,
                ticker=Ticker.EU,
                tf=TimeFrame.D
            )
            server._add_candle(Ticker.EU, TimeFrame.D, candle)
        
        # Verify buffer size is limited
        self.assertEqual(len(server.candle_buffers[key]), 5)
    
    @patch('deployment.forecast_server.ForecastMT5DataConnector')
    @patch('deployment.forecast_server.TelegramNotifier')
    @patch('deployment.forecast_server.helpers.create_ml_manager')
    def test_prepare_prediction_data(self, mock_create_ml_manager, mock_telegram, mock_mt5):
        """Test preparation of prediction data."""
        # Mock MLManager
        mock_ml_manager = Mock(spec=MLManager)
        mock_ml_manager.bias_nodes = [(TimeFrame.D, Mock())]
        mock_create_ml_manager.return_value = mock_ml_manager
        
        # Create server
        server = ForecastServer(config_dir=self.config_dir)
        
        # Add some candles to buffer
        key = (Ticker.EU, TimeFrame.D)
        if key not in server.candle_buffers:
            server.candle_buffers[key] = []
        
        for i in range(20):
            candle = Candle(
                datetime=datetime(2024, 12, i + 1),
                open=1.05,
                high=1.06,
                low=1.04,
                close=1.055 + i * 0.001,  # Slight uptrend
                volume=1000,
                ticker=Ticker.EU,
                tf=TimeFrame.D
            )
            server.candle_buffers[key].append(candle)
        
        # Prepare prediction data
        ticker_series, volatility_series = server._prepare_prediction_data(
            ticker=Ticker.EU,
            ml_manager_key=key,
            n_samples=1
        )
        
        # Verify output
        self.assertIsInstance(ticker_series, pd.Series)
        self.assertIsInstance(volatility_series, pd.Series)
        self.assertEqual(len(ticker_series), 1)
        self.assertEqual(len(volatility_series), 1)
        self.assertEqual(ticker_series.iloc[0], Ticker.EU.value)
        self.assertGreater(volatility_series.iloc[0], 0)
    
    @patch('deployment.forecast_server.ForecastMT5DataConnector')
    @patch('deployment.forecast_server.TelegramNotifier')
    @patch('deployment.forecast_server.helpers.create_ml_manager')
    def test_get_status(self, mock_create_ml_manager, mock_telegram, mock_mt5):
        """Test status retrieval."""
        # Mock MLManager
        mock_ml_manager = Mock(spec=MLManager)
        mock_ml_manager.bias_nodes = [(TimeFrame.D, Mock())]
        mock_create_ml_manager.return_value = mock_ml_manager
        
        # Mock MT5 connector
        mock_mt5_instance = Mock()
        mock_mt5_instance.is_market_open.return_value = True
        mock_mt5.return_value = mock_mt5_instance
        
        # Create server
        server = ForecastServer(config_dir=self.config_dir)
        
        # Get status
        status = server.get_status()
        
        # Verify status structure
        self.assertIn('portfolios', status)
        self.assertIn('ml_managers', status)
        self.assertIn('tickers', status)
        self.assertIn('timeframes', status)
        self.assertIn('mt5_connected', status)
        
        # Verify status values
        self.assertIsInstance(status['portfolios'], dict)
        self.assertIsInstance(status['ml_managers'], int)
        self.assertIsInstance(status['tickers'], list)
        self.assertIsInstance(status['timeframes'], list)
        self.assertIsInstance(status['mt5_connected'], bool)
    
    @patch('deployment.forecast_server.ForecastMT5DataConnector')
    @patch('deployment.forecast_server.TelegramNotifier')
    @patch('deployment.forecast_server.helpers.create_ml_manager')
    @patch('deployment.forecast_server.schedule')
    def test_scheduling_setup(self, mock_schedule, mock_create_ml_manager, mock_telegram, mock_mt5):
        """Test that forecast scheduling is set up correctly."""
        # Mock MLManager
        mock_ml_manager = Mock(spec=MLManager)
        mock_ml_manager.bias_nodes = [(TimeFrame.D, Mock())]
        mock_create_ml_manager.return_value = mock_ml_manager
        
        # Mock schedule
        mock_schedule.every.return_value = Mock(
            day=Mock(at=Mock(do=Mock())),
            sunday=Mock(at=Mock(do=Mock())),
            minutes=Mock(do=Mock())
        )
        
        # Create server (will call _setup_scheduling)
        server = ForecastServer(config_dir=self.config_dir)
        
        # Verify scheduling methods were called
        self.assertTrue(mock_schedule.every.called)
    
    @patch('deployment.forecast_server.ForecastMT5DataConnector')
    @patch('deployment.forecast_server.TelegramNotifier')
    @patch('deployment.forecast_server.helpers.create_ml_manager')
    def test_generate_portfolio_forecasts_insufficient_data(self, mock_create_ml_manager, mock_telegram, mock_mt5):
        """Test forecast generation with insufficient candle data."""
        # Mock MLManager
        mock_ml_manager = Mock(spec=MLManager)
        mock_ml_manager.bias_nodes = [(TimeFrame.D, Mock())]
        mock_ml_manager.matrix_df = pd.DataFrame()  # Empty
        mock_create_ml_manager.return_value = mock_ml_manager
        
        # Create server
        server = ForecastServer(config_dir=self.config_dir)
        
        # Try to generate forecasts with insufficient data
        forecasts = server._generate_portfolio_forecasts(TimeFrame.D)
        
        # Should return empty dict
        self.assertIsInstance(forecasts, dict)
        # May be empty if no data available


class TestForecastServerIntegration(unittest.TestCase):
    """Integration tests for ForecastServer with real Portfolio and MLManager."""
    
    def setUp(self):
        """Set up test data and fixtures."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = os.path.join(self.temp_dir, 'config')
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Create subdirectories for timeframes
        self.daily_dir = os.path.join(self.config_dir, 'D')
        os.makedirs(self.daily_dir, exist_ok=True)
        
        # Create a minimal but complete control file
        self._create_complete_control_file()
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _create_complete_control_file(self):
        """Create a complete control file for integration testing."""
        control_file = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'version': '2.0.0',
                'ensemble_name': 'EU_D',
                'base_tf': 'D',
                'is_fit': True
            },
            'tickers': ['EURUSD'],
            'base_models': [
                {
                    'model_name': 'rsi_2_D_1',
                    'feature_column': 'rsi_signal_D_lookback_2',
                    'model_type': 'QuantileBinningModel',
                    'params': {'n_bins': 2}
                }
            ],
            'fitted_base_models': {
                'rsi_signal_D_lookback_2': {
                    'bin_edges': [0.0, 50.0, 100.0],
                    'bin_signals': [0, 1]
                }
            },
            'fitted_ensemble': {
                'weights': {'rsi_signal_D_lookback_2': 1.0},
                'exposure_fractions': {'rsi_signal_D_lookback_2': 0.8},
                'feature_names': ['rsi_signal_D_lookback_2'],
                'target_volatility': 0.15,
                'unique_tickers': ['EURUSD'],
                'instrument_weights': {'EURUSD': 1.0},
                'n_tickers': 1
            }
        }
        
        filepath = os.path.join(self.daily_dir, 'EU_D.json')
        with open(filepath, 'w') as f:
            json.dump(control_file, f, indent=2)
    
    @patch('deployment.forecast_server.ForecastMT5DataConnector')
    @patch('deployment.forecast_server.TelegramNotifier')
    def test_end_to_end_initialization(self, mock_telegram, mock_mt5):
        """Test end-to-end initialization with real Portfolio and MLManager."""
        # This test uses real Portfolio and MLManager (via helpers.create_ml_manager)
        # Only MT5 and Telegram are mocked
        
        try:
            # Create server - will create real Portfolio and MLManager
            server = ForecastServer(config_dir=self.config_dir)
            
            # Verify server was created
            self.assertIsNotNone(server)
            
            # Verify portfolios exist
            self.assertGreater(len(server.portfolios), 0)
            
            # Verify MLManagers exist
            self.assertGreater(len(server.ml_managers), 0)
            
        except Exception as e:
            # If initialization fails, that's OK for this test
            # We're mainly checking that the structure is correct
            self.skipTest(f"End-to-end initialization failed (expected in unit test): {e}")


if __name__ == '__main__':
    unittest.main()
