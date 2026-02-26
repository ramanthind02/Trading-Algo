"""
Path Generators for Prop Firm Challenge Simulator

This module provides path generation strategies for simulating
prop firm challenge outcomes.

Author: Trading Research Team
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Optional

from utils.evaluation.robustness_test.resampling_strategy import BlockBootstrapStrategy


class PathGenerator(ABC):
    """Abstract base class for path generators."""

    @abstractmethod
    def generate(
        self,
        returns: pd.Series,
        iteration: int,
        random_seed: int,
        path_length: Optional[int] = None
    ) -> pd.Series:
        """
        Generate a return path for simulation.

        Parameters
        ----------
        returns : pd.Series
            Historical returns to use as source
        iteration : int
            Current simulation iteration (used for seed derivation)
        random_seed : int
            Base random seed
        path_length : int, optional
            Desired path length (only used by Monte Carlo method)

        Returns
        -------
        pd.Series
            Generated return path
        """
        pass


class HistoricalWalkForwardGenerator(PathGenerator):
    """
    Generate paths by walking forward from random historical start points.

    This generator selects a random starting point in the historical data
    and walks forward from that point. If the end of data is reached,
    it wraps around to the beginning.
    """

    def generate(
        self,
        returns: pd.Series,
        iteration: int,
        random_seed: int,
        path_length: Optional[int] = None
    ) -> pd.Series:
        """
        Generate a return path by walking forward from a random start.

        Parameters
        ----------
        returns : pd.Series
            Historical returns to use as source
        iteration : int
            Current simulation iteration
        random_seed : int
            Base random seed
        path_length : int, optional
            Not used for historical walk-forward; uses full remaining data

        Returns
        -------
        pd.Series
            Return path from random start to end (with wraparound if needed)
        """
        rng = np.random.default_rng(random_seed + iteration)
        n = len(returns)

        # Select random starting point
        start_idx = rng.integers(0, n)

        # Walk forward from start, wrapping around if needed
        # For historical walk-forward, we use the actual remaining data length
        # but can optionally limit it
        if path_length is not None:
            effective_length = min(path_length, n)
        else:
            # Use remaining data from start point (no wraparound needed for now)
            effective_length = n - start_idx

        indices = [(start_idx + i) % n for i in range(effective_length)]
        path_values = returns.iloc[indices].values

        # Create new series with fresh datetime index
        new_index = pd.date_range(
            start=returns.index[0],
            periods=len(path_values),
            freq='B'
        )

        return pd.Series(path_values, index=new_index, name=returns.name)


class MonteCarloBlockGenerator(PathGenerator):
    """
    Generate synthetic paths using block bootstrap resampling.

    This generator reuses the BlockBootstrapStrategy from the robustness
    testing module to create synthetic return paths that preserve
    autocorrelation structure.
    """

    def __init__(self, block_length: int):
        """
        Initialize the Monte Carlo block generator.

        Parameters
        ----------
        block_length : int
            Block length for block bootstrap resampling
        """
        self._block_length = block_length
        self._strategy = BlockBootstrapStrategy()

    def generate(
        self,
        returns: pd.Series,
        iteration: int,
        random_seed: int,
        path_length: Optional[int] = None
    ) -> pd.Series:
        """
        Generate a synthetic return path using block bootstrap.

        Parameters
        ----------
        returns : pd.Series
            Historical returns to use as source
        iteration : int
            Current simulation iteration
        random_seed : int
            Base random seed
        path_length : int, optional
            Desired path length; defaults to length of returns

        Returns
        -------
        pd.Series
            Synthetic return path
        """
        seed = random_seed + iteration

        # Generate resampled path
        resampled = self._strategy.resample(
            returns,
            random_seed=seed,
            block_size=self._block_length
        )

        # If path_length specified and different from resampled length,
        # we need to generate a longer or shorter path
        if path_length is not None and path_length != len(resampled):
            # Re-generate with target length by resampling multiple times
            # and concatenating, then trimming
            if path_length > len(returns):
                # Need to generate longer path by concatenating resamples
                all_values = []
                current_seed = seed
                while len(all_values) < path_length:
                    resampled = self._strategy.resample(
                        returns,
                        random_seed=current_seed,
                        block_size=self._block_length
                    )
                    all_values.extend(resampled.values.tolist())
                    current_seed += 1000  # Increment seed for next chunk

                path_values = all_values[:path_length]
            else:
                # Trim to path_length
                path_values = resampled.iloc[:path_length].values

            # Create new series with fresh datetime index
            new_index = pd.date_range(
                start=returns.index[0],
                periods=path_length,
                freq='B'
            )
            return pd.Series(path_values, index=new_index, name=returns.name)

        return resampled

    @property
    def block_length(self) -> int:
        """Return the block length used for resampling."""
        return self._block_length
