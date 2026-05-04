"""
Tests for TWS Live Forecast Pipeline

Simple, readable tests for key functions in ``enigma_live_forecast.py``
(legacy name: ``tws_live_forecast``). These tests do NOT require a live TWS connection.
"""

import pytest
import pandas as pd
from datetime import datetime

from scripts.enigma_live_forecast import (
    create_futures_contract,
    create_etf_contract,
    calculate_etf_shares,
    format_telegram_message,
)


# =============================================================================
# Contract Creation Tests
# =============================================================================

class TestContractCreation:
    """Test TWS contract creation functions."""

    def test_create_futures_contract(self):
        """Futures contract should have correct attributes."""
        contract = create_futures_contract("ES", "CME")

        assert contract.symbol == "ES"
        assert contract.secType == "CONTFUT"
        assert contract.exchange == "CME"
        assert contract.currency == "USD"

    def test_create_etf_contract(self):
        """ETF contract should have correct attributes."""
        contract = create_etf_contract("SPY")

        assert contract.symbol == "SPY"
        assert contract.secType == "STK"
        assert contract.exchange == "SMART"
        assert contract.currency == "USD"


# =============================================================================
# Position Sizing Tests
# =============================================================================

class TestPositionSizing:
    """Test ETF share calculation logic."""

    @pytest.fixture
    def sample_positions_df(self):
        """Sample portfolio positions DataFrame."""
        return pd.DataFrame({
            'ticker': ['ES', 'NQ', 'YM', 'RTY'],
            'datetime': [datetime.now()] * 4,
            'forecast_score': [1.0, 0.5, -0.5, 0.0],
            'position_fraction': [0.50, 0.25, -0.15, 0.10]
        })

    @pytest.fixture
    def sample_prices(self):
        """Sample ETF prices."""
        return {
            'SPY': 500.00,
            'QQQ': 400.00,
            'DIA': 350.00,
            'IWM': 200.00
        }

    @pytest.fixture
    def instrument_config(self):
        """Instrument to ETF mapping."""
        return {
            'ES': {'etf': 'SPY'},
            'NQ': {'etf': 'QQQ'},
            'YM': {'etf': 'DIA'},
            'RTY': {'etf': 'IWM'}
        }

    def test_calculate_etf_shares_basic(
        self, sample_positions_df, sample_prices, instrument_config
    ):
        """Calculate shares for a simple portfolio."""
        capital = 10000.0

        result = calculate_etf_shares(
            sample_positions_df,
            sample_prices,
            capital,
            instrument_config
        )

        # Should have 4 rows (one per ticker)
        assert len(result) == 4

        # Check ES allocation: 50% of $10k = $5000, at $500/share = 10 shares
        es_row = result[result['ticker'] == 'ES'].iloc[0]
        assert es_row['target_dollars'] == 5000.0
        assert es_row['shares_fractional'] == 10.0

        # Check NQ allocation: 25% of $10k = $2500, at $400/share = 6.25 shares
        nq_row = result[result['ticker'] == 'NQ'].iloc[0]
        assert nq_row['target_dollars'] == 2500.0
        assert nq_row['shares_fractional'] == 6.25

    def test_calculate_etf_shares_negative_position(
        self, sample_positions_df, sample_prices, instrument_config
    ):
        """Negative position fraction should give negative target dollars."""
        capital = 10000.0

        result = calculate_etf_shares(
            sample_positions_df,
            sample_prices,
            capital,
            instrument_config
        )

        # YM has -15% position
        ym_row = result[result['ticker'] == 'YM'].iloc[0]
        assert ym_row['target_dollars'] == -1500.0
        assert ym_row['shares_fractional'] < 0

    def test_calculate_etf_shares_small_capital(
        self, sample_positions_df, sample_prices, instrument_config
    ):
        """Small capital should result in fractional shares."""
        capital = 100.0  # Very small

        result = calculate_etf_shares(
            sample_positions_df,
            sample_prices,
            capital,
            instrument_config
        )

        # ES: 50% of $100 = $50, at $500/share = 0.1 shares
        es_row = result[result['ticker'] == 'ES'].iloc[0]
        assert es_row['shares_fractional'] == 0.1


# =============================================================================
# Telegram Message Formatting Tests
# =============================================================================

class TestTelegramFormatting:
    """Test Telegram message formatting."""

    @pytest.fixture
    def sample_shares_df(self):
        """Sample shares DataFrame for formatting."""
        return pd.DataFrame({
            'ticker': ['ES', 'NQ'],
            'etf': ['SPY', 'QQQ'],
            'forecast': [1.5, -0.3],
            'etf_price': [500.0, 400.0],
            'target_dollars': [5000.0, -1000.0],
            'shares_fractional': [10.0, -2.5],
            'shares_whole': [10, -2],
            'actual_dollars': [5000.0, -800.0],
            'actual_pct': [50.0, -8.0]
        })

    def test_telegram_message_contains_key_info(self, sample_shares_df):
        """Telegram message should contain capital and positions (personal / ETF layout)."""
        capital = 10000.0

        message = format_telegram_message(sample_shares_df, capital, profile="personal")

        assert "ENIGMA ALGOS FORECAST" in message

        # Should contain capital
        assert "$10,000.00" in message

        # Should contain ticker symbols
        assert "ES" in message
        assert "NQ" in message

        # Should contain ETF symbols
        assert "SPY" in message
        assert "QQQ" in message

    def test_telegram_message_shows_signal_strength(self, sample_shares_df):
        """Message should indicate bullish/bearish signals."""
        capital = 10000.0

        message = format_telegram_message(sample_shares_df, capital, profile="personal")

        # ES has +1.5 forecast -> Strong Bull
        assert "Strong Bull" in message

        # NQ has -0.3 forecast -> Weak Bear
        assert "Weak Bear" in message


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
