"""
Position Sizer for Contract Conversion

This module implements the Execution Layer that converts position fractions
to tradeable contract quantities based on contract specifications and capital.

The Position Sizer is the final layer in the trading system:
    Portfolio.predict() -> PositionSizer.calculate_positions() -> Contracts

Key responsibilities:
1. Calculate target dollar allocation from position fractions
2. Convert dollar amounts to contract quantities
3. Handle contract rounding (round, floor, ceiling)
4. Calculate notional values and portfolio exposure

Reference: Robert Carver's "Systematic Trading" and "Leveraged Trading"
"""

import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, Optional, List
from dataclasses import dataclass


class RoundingMethod(Enum):
    """Methods for rounding fractional contracts to integers."""
    ROUND = 'round'     # Standard rounding (0.5 rounds up)
    FLOOR = 'floor'     # Always round down (conservative)
    CEILING = 'ceiling' # Always round up (aggressive)


@dataclass(frozen=True)
class ContractSpec:
    """
    Specification for a futures/forex contract.

    Parameters
    ----------
    ticker : str
        Instrument ticker symbol (e.g., 'NQ', 'ES', 'GC')
    price : float
        Current market price of the instrument
    multiplier : float
        Contract multiplier - converts price points to dollars
        (e.g., $20/point for NQ, $50/point for ES)
    fx_rate : float, default=1.0
        FX conversion rate to base currency (USD)
        Set to 1.0 for USD-denominated contracts
    min_tick : float, default=0.25
        Minimum price increment for the contract

    Examples
    --------
    >>> nq_spec = ContractSpec(ticker='NQ', price=16000, multiplier=20)
    >>> es_spec = ContractSpec(ticker='ES', price=4800, multiplier=50)
    >>> gc_spec = ContractSpec(ticker='GC', price=2000, multiplier=100)
    """
    ticker: str
    price: float
    multiplier: float
    fx_rate: float = 1.0
    min_tick: float = 0.25

    @property
    def contract_value(self) -> float:
        """Calculate the notional value of one contract in base currency."""
        return self.price * self.multiplier * self.fx_rate


@dataclass(frozen=True)
class Position:
    """
    Calculated position for an instrument.

    This dataclass holds all the information about a position including
    the original forecast, target allocation, and actual contracts.

    Parameters
    ----------
    ticker : str
        Instrument ticker symbol
    forecast_score : float
        Original forecast score from Portfolio
    position_fraction : float
        Position as fraction of capital
    target_dollars : float
        Target dollar allocation
    contracts : int
        Number of contracts to trade (after rounding)
    notional_value : float
        Actual notional value of position
    notional_pct : float
        Notional value as percentage of capital
    """
    ticker: str
    forecast_score: float
    position_fraction: float
    target_dollars: float
    contracts: int
    notional_value: float
    notional_pct: float


class PositionSizer:
    """
    Convert position fractions to tradeable contracts.

    The PositionSizer is the final layer in the trading system that converts
    portfolio position fractions into actual contract quantities that can
    be traded.

    Parameters
    ----------
    capital : float
        Account capital in base currency (typically USD)
    contract_specs : dict[str, ContractSpec]
        Contract specifications per ticker
    rounding_method : RoundingMethod, default=RoundingMethod.ROUND
        How to round fractional contracts to integers

    Attributes
    ----------
    capital : float
        Account capital
    contract_specs : dict
        Contract specifications
    rounding_method : RoundingMethod
        Rounding method for contracts

    Examples
    --------
    >>> specs = {
    ...     'NQ': ContractSpec(ticker='NQ', price=16000, multiplier=20),
    ...     'ES': ContractSpec(ticker='ES', price=4800, multiplier=50)
    ... }
    >>> sizer = PositionSizer(capital=1_000_000, contract_specs=specs)
    >>> positions = sizer.calculate_positions(position_fractions_df)
    """

    def __init__(
        self,
        capital: float,
        contract_specs: Dict[str, ContractSpec],
        rounding_method: RoundingMethod = RoundingMethod.ROUND
    ):
        if capital <= 0:
            raise ValueError(f"capital must be positive, got {capital}")

        if not contract_specs:
            raise ValueError("contract_specs cannot be empty")

        self.capital = capital
        self.contract_specs = contract_specs
        self.rounding_method = rounding_method

    def calculate_positions(
        self,
        position_fractions: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Convert position fractions to contracts.

        Parameters
        ----------
        position_fractions : pd.DataFrame
            DataFrame with columns: ['ticker', 'forecast_score', 'position_fraction']
            - ticker: Instrument identifier
            - forecast_score: Original forecast from Portfolio
            - position_fraction: Position as fraction of capital

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - ticker: Instrument identifier
            - forecast_score: Original forecast score
            - position_fraction: Target position fraction
            - target_dollars: Target dollar allocation
            - contracts: Number of contracts to trade (integer)
            - notional_value: Actual notional value of position
            - notional_pct: Notional value as % of capital

        Raises
        ------
        ValueError
            If required columns are missing or contract specs not found
        """
        # Validate input columns
        required_cols = ['ticker', 'forecast_score', 'position_fraction']
        missing_cols = set(required_cols) - set(position_fractions.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        if position_fractions.empty:
            return pd.DataFrame(columns=[
                'ticker', 'forecast_score', 'position_fraction',
                'target_dollars', 'contracts', 'notional_value', 'notional_pct'
            ])

        results = []

        for _, row in position_fractions.iterrows():
            ticker = row['ticker']
            forecast_score = row['forecast_score']
            position_fraction = row['position_fraction']

            # Get contract spec
            spec = self.contract_specs.get(ticker)
            if spec is None:
                raise ValueError(
                    f"No contract specification found for ticker '{ticker}'. "
                    f"Available tickers: {list(self.contract_specs.keys())}"
                )

            # Calculate target dollar allocation
            target_dollars = position_fraction * self.capital

            # Calculate contract value
            contract_value = spec.contract_value

            # Calculate raw contracts
            if contract_value > 0:
                contracts_raw = target_dollars / contract_value
            else:
                contracts_raw = 0.0

            # Round contracts
            contracts = self._round_contracts(contracts_raw)

            # Calculate actual notional value
            notional_value = contracts * contract_value

            # Calculate notional as percentage of capital
            notional_pct = notional_value / self.capital if self.capital > 0 else 0.0

            results.append({
                'ticker': ticker,
                'forecast_score': forecast_score,
                'position_fraction': position_fraction,
                'target_dollars': target_dollars,
                'contracts': contracts,
                'notional_value': notional_value,
                'notional_pct': notional_pct
            })

        return pd.DataFrame(results)

    def _round_contracts(self, contracts_raw: float) -> int:
        """
        Round fractional contracts based on rounding method.

        Parameters
        ----------
        contracts_raw : float
            Raw (fractional) number of contracts

        Returns
        -------
        int
            Rounded number of contracts (non-negative)
        """
        if contracts_raw < 0:
            # Handle negative positions (short)
            if self.rounding_method == RoundingMethod.ROUND:
                return int(round(contracts_raw))
            elif self.rounding_method == RoundingMethod.FLOOR:
                return int(np.floor(contracts_raw))
            elif self.rounding_method == RoundingMethod.CEILING:
                return int(np.ceil(contracts_raw))
        else:
            # Handle positive positions (long)
            if self.rounding_method == RoundingMethod.ROUND:
                return int(round(contracts_raw))
            elif self.rounding_method == RoundingMethod.FLOOR:
                return int(np.floor(contracts_raw))
            elif self.rounding_method == RoundingMethod.CEILING:
                return int(np.ceil(contracts_raw))

        return 0

    def update_prices(self, price_updates: Dict[str, float]) -> None:
        """
        Update contract prices for position sizing.

        Parameters
        ----------
        price_updates : dict[str, float]
            Dictionary mapping ticker -> new price
        """
        for ticker, price in price_updates.items():
            if ticker in self.contract_specs:
                old_spec = self.contract_specs[ticker]
                self.contract_specs[ticker] = ContractSpec(
                    ticker=old_spec.ticker,
                    price=price,
                    multiplier=old_spec.multiplier,
                    fx_rate=old_spec.fx_rate,
                    min_tick=old_spec.min_tick
                )

    def update_capital(self, new_capital: float) -> None:
        """
        Update account capital.

        Parameters
        ----------
        new_capital : float
            New account capital value
        """
        if new_capital <= 0:
            raise ValueError(f"capital must be positive, got {new_capital}")
        self.capital = new_capital

    def get_summary(self, positions: pd.DataFrame) -> Dict:
        """
        Get summary statistics for a set of positions.

        Parameters
        ----------
        positions : pd.DataFrame
            Output from calculate_positions()

        Returns
        -------
        dict
            Summary statistics including:
            - total_target_dollars: Sum of target allocations
            - total_notional_value: Sum of actual notional values
            - total_notional_pct: Total exposure as % of capital
            - n_instruments: Number of instruments
            - n_contracts: Total number of contracts
            - allocation_error: Difference between target and actual (%)
        """
        if positions.empty:
            return {
                'total_target_dollars': 0.0,
                'total_notional_value': 0.0,
                'total_notional_pct': 0.0,
                'n_instruments': 0,
                'n_contracts': 0,
                'allocation_error_pct': 0.0
            }

        total_target = positions['target_dollars'].sum()
        total_notional = positions['notional_value'].sum()

        return {
            'total_target_dollars': total_target,
            'total_notional_value': total_notional,
            'total_notional_pct': positions['notional_pct'].sum(),
            'n_instruments': len(positions),
            'n_contracts': positions['contracts'].sum(),
            'allocation_error_pct': (
                (total_notional - total_target) / total_target * 100
                if total_target != 0 else 0.0
            )
        }
