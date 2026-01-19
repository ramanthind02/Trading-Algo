"""
Execution Layer for Position Sizing and Contract Conversion

This module provides the execution layer that converts portfolio position
fractions into tradeable contract quantities.

Components:
- PositionSizer: Main class for converting positions to contracts
- ContractSpec: Dataclass for contract specifications
- Position: Dataclass for calculated positions
- RoundingMethod: Enum for contract rounding methods

Example Usage:
    >>> from execution import PositionSizer, ContractSpec, RoundingMethod
    >>> specs = {'NQ': ContractSpec(ticker='NQ', price=16000, multiplier=20)}
    >>> sizer = PositionSizer(capital=1_000_000, contract_specs=specs)
    >>> positions = sizer.calculate_positions(position_fractions_df)
"""

from .position_sizer import (
    PositionSizer,
    ContractSpec,
    Position,
    RoundingMethod,
)

__all__ = [
    'PositionSizer',
    'ContractSpec',
    'Position',
    'RoundingMethod',
]
