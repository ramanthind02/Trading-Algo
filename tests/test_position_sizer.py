"""
Unit tests for PositionSizer and related classes.

Tests cover:
- ContractSpec calculations
- Contract rounding methods
- Position sizing calculations
- Edge cases and error handling
"""

import unittest
import pandas as pd
from execution.position_sizer import (
    PositionSizer,
    ContractSpec,
    RoundingMethod,
)


# Test fixture: Standard contract specifications for common futures
STANDARD_CONTRACT_SPECS = {
    'NQ': ContractSpec(ticker='NQ', price=16000, multiplier=20, min_tick=0.25),
    'ES': ContractSpec(ticker='ES', price=4800, multiplier=50, min_tick=0.25),
    'GC': ContractSpec(ticker='GC', price=2000, multiplier=100, min_tick=0.10),
    'CL': ContractSpec(ticker='CL', price=75, multiplier=1000, min_tick=0.01),
    'EU': ContractSpec(ticker='EU', price=1.10, multiplier=125000, min_tick=0.00005),
}


class TestContractSpec(unittest.TestCase):
    """Tests for ContractSpec dataclass."""

    def test_contract_value_calculation(self):
        """Test contract value property."""
        spec = ContractSpec(
            ticker='NQ',
            price=16000,
            multiplier=20,
            fx_rate=1.0
        )

        # Contract value = price * multiplier * fx_rate
        expected = 16000 * 20 * 1.0
        self.assertEqual(spec.contract_value, expected)

    def test_contract_value_with_fx_rate(self):
        """Test contract value with non-USD FX rate."""
        spec = ContractSpec(
            ticker='EU',
            price=1.10,
            multiplier=125000,
            fx_rate=1.0  # EUR futures are already in USD
        )

        expected = 1.10 * 125000 * 1.0
        self.assertEqual(spec.contract_value, expected)

    def test_immutability(self):
        """ContractSpec should be frozen/immutable."""
        spec = ContractSpec(ticker='ES', price=4800, multiplier=50)

        with self.assertRaises(AttributeError):
            spec.price = 5000


class TestPositionSizer(unittest.TestCase):
    """Tests for PositionSizer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.contract_specs = {
            'ES': ContractSpec(ticker='ES', price=4800, multiplier=50),
            'NQ': ContractSpec(ticker='NQ', price=16000, multiplier=20),
            'GC': ContractSpec(ticker='GC', price=2000, multiplier=100)
        }
        self.capital = 1_000_000

    def test_basic_position_calculation(self):
        """Test basic position sizing."""
        sizer = PositionSizer(
            capital=self.capital,
            contract_specs=self.contract_specs
        )

        position_fractions = pd.DataFrame({
            'ticker': ['ES', 'NQ'],
            'forecast_score': [1.0, 0.5],
            'position_fraction': [0.5, 0.25]
        })

        result = sizer.calculate_positions(position_fractions)

        # Check output structure
        self.assertIn('contracts', result.columns)
        self.assertIn('notional_value', result.columns)
        self.assertIn('notional_pct', result.columns)

        # Check ES calculation
        es_row = result[result['ticker'] == 'ES'].iloc[0]
        es_contract_value = 4800 * 50  # 240,000
        expected_contracts = round((0.5 * 1_000_000) / es_contract_value)
        self.assertEqual(es_row['contracts'], expected_contracts)

    def test_rounding_methods(self):
        """Test different rounding methods."""
        capital = 500_000  # Chosen to give fractional contracts

        # Create a spec that gives fractional contracts
        specs = {
            'TEST': ContractSpec(ticker='TEST', price=1000, multiplier=100)
        }
        # Contract value = 100,000
        # With position_fraction=0.15, target = 75,000
        # Contracts = 0.75 (fractional)

        position_fractions = pd.DataFrame({
            'ticker': ['TEST'],
            'forecast_score': [1.0],
            'position_fraction': [0.15]
        })

        # Test ROUND
        sizer_round = PositionSizer(
            capital=capital,
            contract_specs=specs,
            rounding_method=RoundingMethod.ROUND
        )
        result_round = sizer_round.calculate_positions(position_fractions)
        self.assertEqual(result_round.iloc[0]['contracts'], 1)  # 0.75 rounds to 1

        # Test FLOOR
        sizer_floor = PositionSizer(
            capital=capital,
            contract_specs=specs,
            rounding_method=RoundingMethod.FLOOR
        )
        result_floor = sizer_floor.calculate_positions(position_fractions)
        self.assertEqual(result_floor.iloc[0]['contracts'], 0)  # 0.75 floors to 0

        # Test CEILING
        sizer_ceiling = PositionSizer(
            capital=capital,
            contract_specs=specs,
            rounding_method=RoundingMethod.CEILING
        )
        result_ceiling = sizer_ceiling.calculate_positions(position_fractions)
        self.assertEqual(result_ceiling.iloc[0]['contracts'], 1)  # 0.75 ceils to 1

    def test_notional_value_calculation(self):
        """Test notional value equals contracts * contract_value."""
        sizer = PositionSizer(
            capital=self.capital,
            contract_specs=self.contract_specs
        )

        position_fractions = pd.DataFrame({
            'ticker': ['ES'],
            'forecast_score': [1.0],
            'position_fraction': [1.0]
        })

        result = sizer.calculate_positions(position_fractions)
        row = result.iloc[0]

        expected_notional = row['contracts'] * (4800 * 50)
        self.assertEqual(row['notional_value'], expected_notional)

    def test_notional_pct_calculation(self):
        """Test notional percentage of capital."""
        sizer = PositionSizer(
            capital=self.capital,
            contract_specs=self.contract_specs
        )

        position_fractions = pd.DataFrame({
            'ticker': ['ES'],
            'forecast_score': [1.0],
            'position_fraction': [0.5]
        })

        result = sizer.calculate_positions(position_fractions)
        row = result.iloc[0]

        expected_pct = row['notional_value'] / self.capital
        self.assertAlmostEqual(row['notional_pct'], expected_pct)

    def test_missing_ticker_raises(self):
        """Should raise ValueError for unknown ticker."""
        sizer = PositionSizer(
            capital=self.capital,
            contract_specs=self.contract_specs
        )

        position_fractions = pd.DataFrame({
            'ticker': ['UNKNOWN'],
            'forecast_score': [1.0],
            'position_fraction': [0.5]
        })

        with self.assertRaises(ValueError) as context:
            sizer.calculate_positions(position_fractions)

        self.assertIn('UNKNOWN', str(context.exception))

    def test_missing_columns_raises(self):
        """Should raise ValueError for missing columns."""
        sizer = PositionSizer(
            capital=self.capital,
            contract_specs=self.contract_specs
        )

        position_fractions = pd.DataFrame({
            'ticker': ['ES'],
            'forecast_score': [1.0]
            # Missing 'position_fraction'
        })

        with self.assertRaises(ValueError):
            sizer.calculate_positions(position_fractions)

    def test_empty_input(self):
        """Empty input should return empty DataFrame with correct columns."""
        sizer = PositionSizer(
            capital=self.capital,
            contract_specs=self.contract_specs
        )

        position_fractions = pd.DataFrame(columns=['ticker', 'forecast_score', 'position_fraction'])

        result = sizer.calculate_positions(position_fractions)

        self.assertEqual(len(result), 0)
        self.assertIn('contracts', result.columns)
        self.assertIn('notional_value', result.columns)

    def test_zero_position_fraction(self):
        """Zero position fraction should result in zero contracts."""
        sizer = PositionSizer(
            capital=self.capital,
            contract_specs=self.contract_specs
        )

        position_fractions = pd.DataFrame({
            'ticker': ['ES'],
            'forecast_score': [0.0],
            'position_fraction': [0.0]
        })

        result = sizer.calculate_positions(position_fractions)
        self.assertEqual(result.iloc[0]['contracts'], 0)

    def test_update_prices(self):
        """Test price update functionality."""
        sizer = PositionSizer(
            capital=self.capital,
            contract_specs=self.contract_specs
        )

        # Update ES price
        sizer.update_prices({'ES': 5000})

        self.assertEqual(sizer.contract_specs['ES'].price, 5000)

    def test_update_capital(self):
        """Test capital update functionality."""
        sizer = PositionSizer(
            capital=self.capital,
            contract_specs=self.contract_specs
        )

        sizer.update_capital(2_000_000)
        self.assertEqual(sizer.capital, 2_000_000)

    def test_invalid_capital_raises(self):
        """Should raise ValueError for non-positive capital."""
        with self.assertRaises(ValueError):
            PositionSizer(capital=0, contract_specs=self.contract_specs)

        with self.assertRaises(ValueError):
            PositionSizer(capital=-100000, contract_specs=self.contract_specs)

    def test_empty_contract_specs_raises(self):
        """Should raise ValueError for empty contract specs."""
        with self.assertRaises(ValueError):
            PositionSizer(capital=self.capital, contract_specs={})

    def test_get_summary(self):
        """Test summary statistics calculation."""
        sizer = PositionSizer(
            capital=self.capital,
            contract_specs=self.contract_specs
        )

        position_fractions = pd.DataFrame({
            'ticker': ['ES', 'NQ', 'GC'],
            'forecast_score': [1.0, 0.5, 0.8],
            'position_fraction': [0.3, 0.2, 0.1]
        })

        positions = sizer.calculate_positions(position_fractions)
        summary = sizer.get_summary(positions)

        self.assertIn('total_target_dollars', summary)
        self.assertIn('total_notional_value', summary)
        self.assertIn('n_instruments', summary)
        self.assertIn('n_contracts', summary)

        self.assertEqual(summary['n_instruments'], 3)


class TestStandardContractSpecs(unittest.TestCase):
    """Tests for standard contract specifications."""

    def test_all_tickers_have_specs(self):
        """Verify standard specs include common futures."""
        expected_tickers = ['NQ', 'ES', 'GC', 'CL', 'EU']

        for ticker in expected_tickers:
            self.assertIn(ticker, STANDARD_CONTRACT_SPECS)

    def test_specs_have_valid_values(self):
        """All specs should have positive values."""
        for ticker, spec in STANDARD_CONTRACT_SPECS.items():
            self.assertGreater(spec.price, 0, f"{ticker} price should be > 0")
            self.assertGreater(spec.multiplier, 0, f"{ticker} multiplier should be > 0")
            self.assertGreater(spec.fx_rate, 0, f"{ticker} fx_rate should be > 0")


class TestPositionSizerIntegration(unittest.TestCase):
    """Integration tests for PositionSizer."""

    def test_realistic_portfolio(self):
        """Test with a realistic multi-asset portfolio."""
        # Use standard contract specs with updated prices
        specs = {
            'ES': ContractSpec(ticker='ES', price=4800, multiplier=50),
            'NQ': ContractSpec(ticker='NQ', price=16000, multiplier=20),
            'GC': ContractSpec(ticker='GC', price=2000, multiplier=100),
            'CL': ContractSpec(ticker='CL', price=75, multiplier=1000)
        }

        sizer = PositionSizer(
            capital=500_000,
            contract_specs=specs,
            rounding_method=RoundingMethod.ROUND
        )

        # 20% each for diversification
        position_fractions = pd.DataFrame({
            'ticker': ['ES', 'NQ', 'GC', 'CL'],
            'forecast_score': [1.0, 1.0, 0.8, 0.6],
            'position_fraction': [0.2, 0.2, 0.15, 0.1]
        })

        positions = sizer.calculate_positions(position_fractions)
        summary = sizer.get_summary(positions)

        # Check reasonable allocation
        self.assertEqual(len(positions), 4)

        # Total exposure should be reasonable (not 10x leverage)
        self.assertLess(summary['total_notional_pct'], 2.0)


if __name__ == '__main__':
    unittest.main()
