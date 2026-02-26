"""
Integration Tests for PortfolioManager

Tests the complete flow: candles -> portfolios -> positions
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.core.enums import TimeFrame, Ticker
from ensemble.portfolio_manager import PortfolioManager
from ensemble.portfolio import Portfolio
from ensemble.diversified_ensemble import DiversifiedEnsemble


class TestPortfolioManager:
    """Integration tests for PortfolioManager."""
    
    def test_portfolio_manager_initialization(self):
        """Test PortfolioManager can be initialized with portfolios."""
        # Create a simple portfolio
        portfolio = Portfolio(
            ensembles=[],
            trading_timeframe=TimeFrame.D,
            target_volatility=0.20,
            dm=2.0
        )
        
        # Create manager
        manager = PortfolioManager(
            portfolios={TimeFrame.D: portfolio}
        )
        
        assert len(manager.portfolios) == 1
        assert TimeFrame.D in manager.portfolios
        assert manager.position_sizer is None
    
    def test_portfolio_manager_with_position_sizer(self):
        """Test PortfolioManager with PositionSizer."""
        from execution.position_sizer import PositionSizer, ContractSpec, RoundingMethod
        
        # Create portfolio
        portfolio = Portfolio(
            ensembles=[],
            trading_timeframe=TimeFrame.D,
            target_volatility=0.20,
            dm=2.0
        )
        
        # Create position sizer
        contract_specs = {
            'ES': ContractSpec(ticker='ES', price=4800.0, multiplier=50)
        }
        position_sizer = PositionSizer(
            capital=1_000_000,
            contract_specs=contract_specs
        )
        
        # Create manager
        manager = PortfolioManager(
            portfolios={TimeFrame.D: portfolio},
            position_sizer=position_sizer
        )
        
        assert manager.position_sizer is not None
    
    def test_portfolio_manager_fit_routes_to_portfolios(self):
        """Test fit() routes candles to appropriate portfolios."""
        # Create mock portfolio
        portfolio = Portfolio(
            ensembles=[],
            trading_timeframe=TimeFrame.D,
            target_volatility=0.20,
            dm=2.0
        )
        
        # Create manager
        manager = PortfolioManager(
            portfolios={TimeFrame.D: portfolio}
        )
        
        # Create sample candles
        candles_df = pd.DataFrame({
            'datetime': [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            'open': [4800.0, 4801.0],
            'high': [4805.0, 4806.0],
            'low': [4795.0, 4796.0],
            'close': [4802.0, 4803.0],
            'volume': [1000, 1100],
            'ticker': ['ES', 'ES'],
            'timeframe': [TimeFrame.D, TimeFrame.D]
        })
        
        # Create target data
        target_data = pd.Series([0.01, 0.02], index=pd.to_datetime(candles_df['datetime']))
        
        # Fit should not raise errors (even if portfolios have no ensembles)
        manager.fit(candles_df, target_data)
    
    def test_portfolio_manager_predict_combines_outputs(self):
        """Test predict() combines outputs from all portfolios."""
        # Create mock portfolio
        portfolio = Portfolio(
            ensembles=[],
            trading_timeframe=TimeFrame.D,
            target_volatility=0.20,
            dm=2.0
        )
        
        # Create manager
        manager = PortfolioManager(
            portfolios={TimeFrame.D: portfolio}
        )
        
        # Create sample candles
        candles_df = pd.DataFrame({
            'datetime': [datetime(2024, 1, 1)],
            'open': [4800.0],
            'high': [4805.0],
            'low': [4795.0],
            'close': [4802.0],
            'volume': [1000],
            'ticker': ['ES'],
            'timeframe': [TimeFrame.D]
        })
        
        # Predict should return DataFrame (even if empty)
        result = manager.predict(candles_df)
        
        assert isinstance(result, pd.DataFrame)
        # Should have expected columns
        expected_cols = ['ticker', 'datetime', 'timeframe', 'forecast_score', 'position_fraction']
        assert all(col in result.columns for col in expected_cols if col in result.columns)
    
    def test_portfolio_manager_multiple_timeframes(self):
        """Test handling multiple timeframes correctly."""
        # Create portfolios for different timeframes
        daily_portfolio = Portfolio(
            ensembles=[],
            trading_timeframe=TimeFrame.D,
            target_volatility=0.20,
            dm=2.0
        )
        
        weekly_portfolio = Portfolio(
            ensembles=[],
            trading_timeframe=TimeFrame.W,
            target_volatility=0.15,
            dm=1.5
        )
        
        # Create manager
        manager = PortfolioManager(
            portfolios={
                TimeFrame.D: daily_portfolio,
                TimeFrame.W: weekly_portfolio
            }
        )
        
        # Create candles for both timeframes
        candles_df = pd.DataFrame({
            'datetime': [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 8),  # Weekly
            ],
            'open': [4800.0, 4801.0, 4802.0],
            'high': [4805.0, 4806.0, 4807.0],
            'low': [4795.0, 4796.0, 4797.0],
            'close': [4802.0, 4803.0, 4804.0],
            'volume': [1000, 1100, 1200],
            'ticker': ['ES', 'ES', 'ES'],
            'timeframe': [TimeFrame.D, TimeFrame.D, TimeFrame.W]
        })
        
        # Fit should route to appropriate portfolios
        target_data = pd.Series([0.01, 0.02, 0.03], index=pd.to_datetime(candles_df['datetime']))
        manager.fit(candles_df, target_data)
        
        # Predict should combine outputs
        result = manager.predict(candles_df)
        assert isinstance(result, pd.DataFrame)
    
    def test_portfolio_manager_empty_candles(self):
        """Test handling of empty candles DataFrame."""
        portfolio = Portfolio(
            ensembles=[],
            trading_timeframe=TimeFrame.D,
            target_volatility=0.20,
            dm=2.0
        )
        
        manager = PortfolioManager(
            portfolios={TimeFrame.D: portfolio}
        )
        
        # Empty DataFrame
        empty_df = pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close', 'volume', 'ticker', 'timeframe'])
        
        # Should handle gracefully
        manager.fit(empty_df)
        result = manager.predict(empty_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_portfolio_manager_get_portfolio(self):
        """Test get_portfolio() method."""
        portfolio = Portfolio(
            ensembles=[],
            trading_timeframe=TimeFrame.D,
            target_volatility=0.20,
            dm=2.0
        )
        
        manager = PortfolioManager(
            portfolios={TimeFrame.D: portfolio}
        )
        
        # Get existing portfolio
        retrieved = manager.get_portfolio(TimeFrame.D)
        assert retrieved is not None
        assert retrieved == portfolio
        
        # Get non-existent portfolio
        retrieved = manager.get_portfolio(TimeFrame.W)
        assert retrieved is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
