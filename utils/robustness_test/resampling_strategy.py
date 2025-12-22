"""
Resampling Strategies for Robustness Testing

This module provides a unified, modular resampling framework that:
1. Abstracts the core resampling logic
2. Supports Monte Carlo, Bootstrap, and Block Bootstrap methods
3. Preserves datetime indices for equity curve computation

Architecture:
- ResamplingStrategy: Abstract interface for resampling methods
- MonteCarloStrategy: Random permutation without replacement
- BootstrapStrategy: Sampling with replacement
- BlockBootstrapStrategy: Block-wise sampling preserving autocorrelation

Author: Trading Research Team
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional


class ResamplingStrategy(ABC):
    """
    Abstract base class for resampling strategies.

    A resampling strategy defines HOW to resample the return series.
    Each strategy produces a new pd.Series with synthetic returns
    that maintains the original datetime index structure.
    """

    @abstractmethod
    def resample(
        self,
        returns: pd.Series,
        random_seed: int,
        **kwargs
    ) -> pd.Series:
        """
        Resample the return series.

        Args:
            returns: Return series with datetime index
            random_seed: Random seed for reproducibility
            **kwargs: Strategy-specific parameters

        Returns:
            Resampled return series with same index as input
        """
        pass

    @abstractmethod
    def validate_data(self, returns: pd.Series, **kwargs) -> None:
        """
        Validate that data is in the correct format for this strategy.

        Args:
            returns: Return series to validate
            **kwargs: Strategy-specific parameters

        Raises:
            TypeError: If returns is not pd.Series
            ValueError: If returns is empty or missing datetime index
        """
        pass

    def _base_validation(self, returns: pd.Series) -> None:
        """Common validation logic for all strategies."""
        if not isinstance(returns, pd.Series):
            raise TypeError("returns must be a pandas Series")

        if len(returns) == 0:
            raise ValueError("returns series is empty")

        if not isinstance(returns.index, pd.DatetimeIndex):
            raise TypeError(
                f"returns must have DatetimeIndex. "
                f"Current index type: {type(returns.index).__name__}"
            )

        if not pd.api.types.is_numeric_dtype(returns):
            raise TypeError(
                f"returns must be numeric. Current dtype: {returns.dtype}"
            )


class MonteCarloStrategy(ResamplingStrategy):
    """
    Monte Carlo permutation strategy: Random shuffle without replacement.

    This completely destroys the temporal structure and autocorrelation,
    testing if the observed returns could arise by random chance.

    Statistical Properties:
    - Mean preserved exactly
    - Variance preserved exactly
    - Autocorrelation destroyed
    - Distribution shape preserved
    """

    def validate_data(self, returns: pd.Series, **kwargs) -> None:
        """Validate input for Monte Carlo resampling."""
        self._base_validation(returns)

    def resample(
        self,
        returns: pd.Series,
        random_seed: int,
        **kwargs
    ) -> pd.Series:
        """
        Randomly permute returns without replacement.

        Args:
            returns: Return series with datetime index
            random_seed: Random seed for reproducibility

        Returns:
            Permuted return series with same index
        """
        rng = np.random.default_rng(random_seed)

        # Shuffle values, keep original index
        shuffled_values = rng.permutation(returns.values)

        return pd.Series(
            shuffled_values,
            index=returns.index,
            name=returns.name
        )


class BootstrapStrategy(ResamplingStrategy):
    """
    Bootstrap strategy: Sampling with replacement.

    Randomly draws returns with replacement, creating a synthetic
    return series of the same length. Does not preserve autocorrelation.

    Statistical Properties:
    - Mean: approximately preserved (sampling variation)
    - Variance: approximately preserved
    - Autocorrelation: destroyed
    - Can produce duplicate returns in same sample
    """

    def validate_data(self, returns: pd.Series, **kwargs) -> None:
        """Validate input for Bootstrap resampling."""
        self._base_validation(returns)

    def resample(
        self,
        returns: pd.Series,
        random_seed: int,
        **kwargs
    ) -> pd.Series:
        """
        Sample returns with replacement.

        Args:
            returns: Return series with datetime index
            random_seed: Random seed for reproducibility

        Returns:
            Bootstrapped return series with same index
        """
        rng = np.random.default_rng(random_seed)

        n = len(returns)
        # Sample indices with replacement
        bootstrap_indices = rng.integers(0, n, size=n)
        bootstrap_values = returns.values[bootstrap_indices]

        return pd.Series(
            bootstrap_values,
            index=returns.index,
            name=returns.name
        )


class BlockBootstrapStrategy(ResamplingStrategy):
    """
    Block Bootstrap strategy: Block-wise sampling with replacement.

    Samples contiguous blocks of returns, preserving short-term
    autocorrelation structure within blocks. Essential for time
    series with serial correlation.

    Statistical Properties:
    - Mean: approximately preserved
    - Variance: approximately preserved
    - Autocorrelation: partially preserved (within blocks)
    - Block size controls tradeoff: larger blocks = more autocorrelation

    Recommended block sizes:
    - Daily returns: 5-20 days (1-4 weeks)
    - Weekly returns: 4-12 weeks
    - Use sqrt(n) as rule of thumb
    """

    def validate_data(
        self,
        returns: pd.Series,
        block_size: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Validate input for Block Bootstrap resampling.

        Args:
            returns: Return series to validate
            block_size: Size of each block (required)
        """
        self._base_validation(returns)

        if block_size is None:
            raise ValueError(
                "block_size is required for BlockBootstrapStrategy. "
                "Recommended: sqrt(n) or based on autocorrelation decay."
            )

        if not isinstance(block_size, int) or block_size < 1:
            raise ValueError(
                f"block_size must be a positive integer, got: {block_size}"
            )

        if block_size > len(returns):
            raise ValueError(
                f"block_size ({block_size}) cannot exceed series length ({len(returns)})"
            )

    def resample(
        self,
        returns: pd.Series,
        random_seed: int,
        block_size: int,
        **kwargs
    ) -> pd.Series:
        """
        Sample blocks of returns with replacement.

        Uses circular block bootstrap:
        1. Randomly select starting indices
        2. Extract blocks (wrapping around if needed)
        3. Concatenate until we have n observations

        Args:
            returns: Return series with datetime index
            random_seed: Random seed for reproducibility
            block_size: Size of each contiguous block

        Returns:
            Block-bootstrapped return series with same index
        """
        rng = np.random.default_rng(random_seed)

        n = len(returns)
        values = returns.values

        # Number of blocks needed (may need one extra partial block)
        n_blocks = int(np.ceil(n / block_size))

        # Sample block starting indices
        block_starts = rng.integers(0, n, size=n_blocks)

        # Extract blocks (with circular wrapping)
        resampled_values = []
        for start in block_starts:
            for j in range(block_size):
                idx = (start + j) % n  # Circular wrap
                resampled_values.append(values[idx])
                if len(resampled_values) >= n:
                    break
            if len(resampled_values) >= n:
                break

        # Trim to exact length
        resampled_values = np.array(resampled_values[:n])

        return pd.Series(
            resampled_values,
            index=returns.index,
            name=returns.name
        )
