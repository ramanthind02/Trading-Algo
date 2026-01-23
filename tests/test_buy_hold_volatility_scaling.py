"""
Unit Tests for Buy/Hold Volatility Scaling

Tests the buy/hold workflow to ensure:
1. Exposure fraction is correctly set to 1.0 for buy_hold models
2. Forecast scores are correctly calculated and capped at 2.0
3. Portfolio positions are correctly calculated with instrument weights
4. Portfolio volatility matches target volatility

Based on forecast_specs.md buy/hold scenarios.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
from utils.enums import TimeFrame, Ticker
from ensemble.diversified_ensemble import DiversifiedEnsemble
from ensemble.portfolio import Portfolio
from ensemble.weight_layer import WeightLayer
from feature_selection.base_models.feature_base_model import BaseModel
from feature_selection.base_models.quantile_binning import QuantileBinningModel
from utils.models import Candle


# Hardcoded buy_hold feature config for testing
BUY_HOLD_FEATURE_CONFIG = {
    "feature_name": "buy_hold_signal_D",
    "feature_column": "buy_hold_signal_D",
    "bias_node_spec": {
        "module_name": "buy_hold",
        "timeframes": ["D"],
        "params": {}
    },
    "base_models": [
        {
            "model_id": "quantile_binning_2",
            "model_name": "buy_hold_signal_D::quantile_binning_2",
            "binning_model_type": "QuantileBinningModel",
            "strategy": "long",
            "binning_model_params": {
                "n_bins": 2,
                "selection_metric": "sortino"
            },
            "is_fitted": True,
            "fitted_params": {
                "thresholds": [],
                "best_long_bin": 0,
                "best_short_bin": 0,
                "bin_stats": {
                    "0": {
                        "mean_return": 0.0004,
                        "std_return": 0.0129,
                        "downside_std": 0.0100,
                        "sortino_metric": 0.697,
                        "count": 1260,
                        "feature_min": 1.0,
                        "feature_max": 1.0
                    }
                }
            }
        }
    ]
}


def create_buy_hold_ensemble(
    target_volatility: float = 0.20,
    tickers: list = [Ticker.ES]
) -> DiversifiedEnsemble:
    """
    Create a buy_hold ensemble for testing.
    
    Parameters
    ----------
    target_volatility : float, default=0.20
        Target annual portfolio volatility
    tickers : list, default=[Ticker.ES]
        List of tickers for the ensemble
        
    Returns
    -------
    DiversifiedEnsemble
        Configured buy_hold ensemble
    """
    import tempfile
    import json
    import os
    
    # Create base model config
    base_models_config = []
    for model in BUY_HOLD_FEATURE_CONFIG['base_models']:
        base_model_config = {
            'name': model['model_name'],
            'feature_column': BUY_HOLD_FEATURE_CONFIG['feature_column'],
            'model_type': model['binning_model_type'],
            'strategy': model['strategy'],
            'constructor_params': model['binning_model_params'],
            'bias_node_spec': BUY_HOLD_FEATURE_CONFIG['bias_node_spec']
        }
        base_models_config.append(base_model_config)
    
    # Create temporary control file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        control_file = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'is_fit': False,
                'base_tf': 'D'
            },
            'base_models': base_models_config,
            'tickers': [t.name for t in tickers]
        }
        json.dump(control_file, f)
        temp_path = f.name
    
    # Create ensemble
    ensemble = DiversifiedEnsemble(
        target_volatility=target_volatility,
        control_file_path=temp_path,
        base_tf=TimeFrame.D
    )
    
    # Clean up temp file after ensemble loads it
    # (ensemble loads it in __init__, so safe to delete now)
    os.unlink(temp_path)
    
    return ensemble


def create_test_candles(
    ticker: Ticker,
    start_date: datetime,
    num_days: int,
    price: float = 100.0,
    volatility: float = 0.20
) -> pd.DataFrame:
    """
    Create test candles DataFrame for a single ticker.
    
    Parameters
    ----------
    ticker : Ticker
        Ticker enum
    start_date : datetime
        Start date for candles
    num_days : int
        Number of days of candles to generate
    price : float, default=100.0
        Starting price
    volatility : float, default=0.20
        Annual volatility (for generating realistic price movements)
        
    Returns
    -------
    pd.DataFrame
        Candles DataFrame with columns: datetime, open, high, low, close, volume, ticker, timeframe
    """
    dates = pd.date_range(start=start_date, periods=num_days, freq='D')
    
    # Generate random returns with specified volatility
    daily_vol = volatility / np.sqrt(252)  # Convert annual to daily
    returns = np.random.normal(0, daily_vol, num_days)
    
    # Generate prices from returns
    prices = [price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    candles = []
    for i, dt in enumerate(dates):
        price_val = prices[i]
        # Simple OHLC (high/low within 2% of close)
        high = price_val * (1 + abs(np.random.normal(0, 0.01)))
        low = price_val * (1 - abs(np.random.normal(0, 0.01)))
        open_price = price_val * (1 + np.random.normal(0, 0.005))
        
        candles.append({
            'datetime': dt,
            'open': open_price,
            'high': max(open_price, high, price_val),
            'low': min(open_price, low, price_val),
            'close': price_val,
            'volume': 1000000,
            'ticker': ticker.name,
            'timeframe': TimeFrame.D
        })
    
    return pd.DataFrame(candles)


class TestBuyHoldVolatilityScaling:
    """Test buy/hold volatility scaling scenarios."""
    
    def test_exposure_fraction_buy_hold(self):
        """Test that buy_hold models have exposure fraction h_i = 1.0."""
        ensemble = create_buy_hold_ensemble(target_volatility=0.20)
        
        # Create minimal test data to trigger fit()
        # We need to fit the ensemble to set model_exposure_fractions_
        candles = create_test_candles(Ticker.ES, datetime(2020, 1, 1), 100, volatility=0.20)
        target = pd.Series(
            index=pd.to_datetime(candles['datetime']),
            data=np.random.normal(0, 0.01, len(candles))
        )
        
        # Fit ensemble
        ensemble.fit_from_candles(candles, target)
        
        # Check exposure fractions
        assert ensemble.model_exposure_fractions_ is not None
        for model_name, h_i in ensemble.model_exposure_fractions_.items():
            # Buy_hold models should have h_i = 1.0
            assert h_i == 1.0, f"Expected h_i=1.0 for buy_hold model, got {h_i}"
    
    def test_perfect_buy_hold_single_ticker(self):
        """
        Test 1: Perfect Buy/Hold (Single Ticker)
        
        Setup:
        - target_vol=0.20, asset_vol=0.20, h_i=1.0, FDM=1.0, IDM=1.0, w=1.0
        Expected: forecast_score = 1.0, position_fraction = 1.0
        """
        ensemble = create_buy_hold_ensemble(target_volatility=0.20, tickers=[Ticker.ES])
        
        # Create test candles with 20% volatility
        candles = create_test_candles(Ticker.ES, datetime(2020, 1, 1), 100, volatility=0.20)
        target = pd.Series(
            index=pd.to_datetime(candles['datetime']),
            data=np.random.normal(0, 0.20/np.sqrt(252), len(candles))
        )
        
        # Fit ensemble
        ensemble.fit_from_candles(candles, target)
        
        # Verify exposure fraction
        assert ensemble.model_exposure_fractions_ is not None
        for h_i in ensemble.model_exposure_fractions_.values():
            assert h_i == 1.0, f"Expected h_i=1.0, got {h_i}"
        
        # Generate predictions
        test_candles = create_test_candles(Ticker.ES, datetime(2020, 2, 1), 10, volatility=0.20)
        predictions = ensemble.predict_from_candles(test_candles)
        
        # Check forecast scores
        # For buy_hold: F_i = 0.20 / (0.20 * sqrt(1.0)) = 1.0
        assert len(predictions) > 0
        forecast_scores = predictions['forecast_score']
        assert np.allclose(forecast_scores, 1.0, rtol=0.01), \
            f"Expected forecast_score ≈ 1.0, got {forecast_scores.values}"
    
    def test_buy_hold_four_tickers(self):
        """
        Test 2: Buy/Hold with 4 Tickers
        
        Setup:
        - target_vol=0.20, asset_vol=0.20, h_i=1.0, FDM=1.0, IDM=1.0, w=0.25
        Expected: forecast_score = 1.0, position_fraction = 0.25 per ticker
        """
        tickers = [Ticker.ES, Ticker.NQ, Ticker.YM, Ticker.RTY]
        ensemble = create_buy_hold_ensemble(target_volatility=0.20, tickers=tickers)
        
        # Create test candles for all tickers
        all_candles = []
        for ticker in tickers:
            candles = create_test_candles(ticker, datetime(2020, 1, 1), 100, volatility=0.20)
            all_candles.append(candles)
        candles_df = pd.concat(all_candles, ignore_index=True)
        
        target = pd.Series(
            index=pd.to_datetime(candles_df['datetime']),
            data=np.random.normal(0, 0.20/np.sqrt(252), len(candles_df))
        )
        
        # Fit ensemble
        ensemble.fit_from_candles(candles_df, target)
        
        # Create portfolio
        portfolio = Portfolio(
            ensembles=[ensemble],
            trading_timeframe=TimeFrame.D,
            target_volatility=0.20,
            dm=1.0,  # IDM = 1.0
            max_position_pct=2.0
        )
        
        # Fit portfolio (calculate IDM)
        returns_df = portfolio._calculate_returns_from_candles(candles_df)
        portfolio.fit(returns_df, idm_override=1.0)  # Override IDM to 1.0
        
        # Generate predictions
        test_candles_list = []
        for ticker in tickers:
            test_candles = create_test_candles(ticker, datetime(2020, 2, 1), 10, volatility=0.20)
            test_candles_list.append(test_candles)
        test_candles_df = pd.concat(test_candles_list, ignore_index=True)
        
        positions = portfolio.predict_from_candles(test_candles_df)
        
        # Check forecast scores (should be ≈ 1.0)
        forecast_scores = positions['forecast_score']
        assert np.allclose(forecast_scores, 1.0, rtol=0.05), \
            f"Expected forecast_score ≈ 1.0, got range [{forecast_scores.min():.4f}, {forecast_scores.max():.4f}]"
        
        # Check position fractions (should be ≈ 0.25 per ticker)
        # position_fraction = forecast_score * instrument_weight * IDM
        # = 1.0 * 0.25 * 1.0 = 0.25
        position_fractions = positions['position_fraction']
        assert np.allclose(position_fractions, 0.25, rtol=0.05), \
            f"Expected position_fraction ≈ 0.25, got range [{position_fractions.min():.4f}, {position_fractions.max():.4f}]"
    
    def test_high_volatility_instrument(self):
        """
        Test 3: High Volatility Instrument
        
        Setup:
        - target_vol=0.20, asset_vol=0.40, h_i=1.0
        Expected: forecast_score = 0.5, position_fraction = 0.5 (single ticker)
        """
        ensemble = create_buy_hold_ensemble(target_volatility=0.20, tickers=[Ticker.ES])
        
        # Create test candles with 40% volatility
        candles = create_test_candles(Ticker.ES, datetime(2020, 1, 1), 100, volatility=0.40)
        target = pd.Series(
            index=pd.to_datetime(candles['datetime']),
            data=np.random.normal(0, 0.40/np.sqrt(252), len(candles))
        )
        
        # Fit ensemble
        ensemble.fit_from_candles(candles, target)
        
        # Generate predictions
        test_candles = create_test_candles(Ticker.ES, datetime(2020, 2, 1), 10, volatility=0.40)
        predictions = ensemble.predict_from_candles(test_candles)
        
        # Check forecast scores
        # For buy_hold: F_i = 0.20 / (0.40 * sqrt(1.0)) = 0.5
        forecast_scores = predictions['forecast_score']
        assert np.allclose(forecast_scores, 0.5, rtol=0.05), \
            f"Expected forecast_score ≈ 0.5, got range [{forecast_scores.min():.4f}, {forecast_scores.max():.4f}]"
    
    def test_low_volatility_instrument_capped(self):
        """
        Test 4: Low Volatility Instrument (Should Cap)
        
        Setup:
        - target_vol=0.20, asset_vol=0.05, h_i=1.0
        Expected: forecast_score = 2.0 (capped), position_fraction = 2.0 (capped)
        """
        ensemble = create_buy_hold_ensemble(target_volatility=0.20, tickers=[Ticker.ES])
        
        # Create test candles with 5% volatility
        candles = create_test_candles(Ticker.ES, datetime(2020, 1, 1), 100, volatility=0.05)
        target = pd.Series(
            index=pd.to_datetime(candles['datetime']),
            data=np.random.normal(0, 0.05/np.sqrt(252), len(candles))
        )
        
        # Fit ensemble
        ensemble.fit_from_candles(candles, target)
        
        # Create portfolio
        portfolio = Portfolio(
            ensembles=[ensemble],
            trading_timeframe=TimeFrame.D,
            target_volatility=0.20,
            dm=1.0,
            max_position_pct=2.0
        )
        
        returns_df = portfolio._calculate_returns_from_candles(candles)
        portfolio.fit(returns_df, idm_override=1.0)
        
        # Generate predictions
        test_candles = create_test_candles(Ticker.ES, datetime(2020, 2, 1), 10, volatility=0.05)
        positions = portfolio.predict_from_candles(test_candles)
        
        # Check forecast scores (should be capped at 2.0)
        # Uncapped: F_i = 0.20 / (0.05 * sqrt(1.0)) = 4.0
        # Capped: F_i = 2.0
        forecast_scores = positions['forecast_score']
        assert np.allclose(forecast_scores, 2.0, rtol=0.01), \
            f"Expected forecast_score = 2.0 (capped), got range [{forecast_scores.min():.4f}, {forecast_scores.max():.4f}]"
        
        # Check position fractions (should be capped at 2.0)
        # position_fraction = 2.0 * 1.0 * 1.0 = 2.0 (capped)
        position_fractions = positions['position_fraction']
        assert np.allclose(position_fractions, 2.0, rtol=0.01), \
            f"Expected position_fraction = 2.0 (capped), got range [{position_fractions.min():.4f}, {position_fractions.max():.4f}]"
    
    def test_end_to_end_buy_hold_portfolio(self):
        """
        Test 5: End-to-End Buy/Hold Portfolio
        
        Setup: 4 tickers, buy_hold ensemble, target_vol=0.20
        Expected: portfolio volatility ≈ 0.20
        """
        from ensemble.portfolio_tester import PortfolioTester
        
        tickers = [Ticker.ES, Ticker.NQ, Ticker.YM, Ticker.RTY]
        ensemble = create_buy_hold_ensemble(target_volatility=0.20, tickers=tickers)
        
        # Create portfolio
        portfolio = Portfolio(
            ensembles=[ensemble],
            trading_timeframe=TimeFrame.D,
            target_volatility=0.20,
            dm=1.0,  # IDM = 1.0
            max_position_pct=2.0
        )
        
        # Create test data
        all_candles = []
        for ticker in tickers:
            candles = create_test_candles(ticker, datetime(2020, 1, 1), 252, volatility=0.20)
            all_candles.append(candles)
        candles_df = pd.concat(all_candles, ignore_index=True)
        
        # Create tester
        tester = PortfolioTester(portfolio, baseline_mode='equal_weight')
        
        # Fit portfolio
        tester.fit(candles_df)
        
        # Generate predictions
        positions = tester.predict(candles_df)
        
        # Calculate returns
        strategy_returns = tester.calculate_strategy_returns(candles_df)
        
        # Calculate portfolio volatility
        if len(strategy_returns) > 0:
            annual_vol = strategy_returns.std() * np.sqrt(252)
            
            # Should be close to target volatility (0.20)
            # Allow some tolerance due to random data generation
            assert 0.15 <= annual_vol <= 0.25, \
                f"Expected portfolio volatility ≈ 0.20, got {annual_vol:.4f}"
    
    def test_forecast_capping_at_ensemble(self):
        """Test that forecasts are capped at 2.0 in ensemble layer."""
        ensemble = create_buy_hold_ensemble(target_volatility=0.20, tickers=[Ticker.ES])
        
        # Create test candles with very low volatility (should produce high forecast)
        candles = create_test_candles(Ticker.ES, datetime(2020, 1, 1), 100, volatility=0.05)
        target = pd.Series(
            index=pd.to_datetime(candles['datetime']),
            data=np.random.normal(0, 0.05/np.sqrt(252), len(candles))
        )
        
        # Fit ensemble
        ensemble.fit_from_candles(candles, target)
        
        # Generate predictions
        test_candles = create_test_candles(Ticker.ES, datetime(2020, 2, 1), 10, volatility=0.05)
        predictions = ensemble.predict_from_candles(test_candles)
        
        # Check that forecast scores are capped at 2.0
        forecast_scores = predictions['forecast_score']
        assert (forecast_scores <= 2.0).all(), \
            f"Expected all forecast_scores <= 2.0, got max {forecast_scores.max():.4f}"
    
    def test_forecast_capping_at_weight_layer(self):
        """Test that forecast_score is capped at 2.0 in weight layer."""
        from ensemble.weight_layer import WeightLayer
        
        # Create weight layer
        weight_layer = WeightLayer(fdm_max=2.0)
        
        # Create mock forecast vectors with high values (should be capped)
        forecast_vectors = [
            pd.DataFrame({
                'ticker': ['ES', 'NQ'],
                'model_name': ['model1', 'model1'],
                'forecast': [3.0, 3.0],  # High forecasts
                'signal': [1, 1]
            })
        ]
        
        # Create mock signals for fitting
        signals = pd.DataFrame({
            'model1': [1, 1, 1, 1, 1]
        })
        
        # Fit weight layer
        weight_layer.fit(forecast_vectors, signals)
        
        # Combine forecasts
        combined = weight_layer.combine(forecast_vectors)
        
        # Check that forecast_score is capped at 2.0
        forecast_scores = combined['forecast_score']
        assert (forecast_scores <= 2.0).all(), \
            f"Expected all forecast_scores <= 2.0, got max {forecast_scores.max():.4f}"
        assert (forecast_scores >= -2.0).all(), \
            f"Expected all forecast_scores >= -2.0, got min {forecast_scores.min():.4f}"
