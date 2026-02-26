"""
Data Structures for Prop Firm Challenge Simulator

This module provides dataclasses and enums for configuring and storing
results from prop firm challenge simulations.

Author: Trading Research Team
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Union
import pandas as pd


class SimulationMethod(Enum):
    """Enumeration of simulation methods for path generation."""
    HISTORICAL_WALKFORWARD = "historical_walkforward"
    MONTE_CARLO_BLOCK = "monte_carlo_block"


@dataclass(frozen=True)
class ChallengeRules:
    """
    Configuration for prop firm challenge rules.

    Attributes
    ----------
    max_drawdown_pct : float
        Maximum allowed drawdown from initial equity (e.g., 0.10 for 10%)
    profit_target_pct : float
        Required profit target to pass (e.g., 0.08 for 8%)
    min_trading_days : int
        Minimum number of trading days required to pass
    max_daily_drawdown_pct : float, optional
        Maximum allowed single-day loss (e.g., 0.05 for 5%)
    trailing_drawdown_pct : float, optional
        Maximum allowed drawdown from peak equity (e.g., 0.05 for 5%)

    Examples
    --------
    >>> rules = ChallengeRules(
    ...     max_drawdown_pct=0.10,
    ...     profit_target_pct=0.08,
    ...     min_trading_days=5,
    ...     trailing_drawdown_pct=0.05
    ... )
    """
    max_drawdown_pct: float
    profit_target_pct: float
    min_trading_days: int
    max_daily_drawdown_pct: Optional[float] = None
    trailing_drawdown_pct: Optional[float] = None

    def __post_init__(self):
        """Validate rule parameters."""
        if self.max_drawdown_pct <= 0:
            raise ValueError("max_drawdown_pct must be positive")
        if self.profit_target_pct <= 0:
            raise ValueError("profit_target_pct must be positive")
        if self.min_trading_days < 1:
            raise ValueError("min_trading_days must be at least 1")
        if self.max_daily_drawdown_pct is not None and self.max_daily_drawdown_pct <= 0:
            raise ValueError("max_daily_drawdown_pct must be positive if set")
        if self.trailing_drawdown_pct is not None and self.trailing_drawdown_pct <= 0:
            raise ValueError("trailing_drawdown_pct must be positive if set")


@dataclass(frozen=True)
class ChallengeCosts:
    """
    Configuration for prop firm challenge costs.

    At least one of monthly_fee or one_time_fee must be provided.

    Attributes
    ----------
    reset_fee : float
        Fee charged for each challenge reset
    monthly_fee : float, optional
        Monthly subscription fee
    one_time_fee : float, optional
        One-time challenge entry fee

    Examples
    --------
    >>> costs = ChallengeCosts(reset_fee=50.0, one_time_fee=500.0)
    """
    reset_fee: float
    monthly_fee: Optional[float] = None
    one_time_fee: Optional[float] = None

    def __post_init__(self):
        """Validate cost parameters."""
        if self.reset_fee < 0:
            raise ValueError("reset_fee cannot be negative")
        if self.monthly_fee is None and self.one_time_fee is None:
            raise ValueError("Either monthly_fee or one_time_fee must be provided")
        if self.monthly_fee is not None and self.monthly_fee < 0:
            raise ValueError("monthly_fee cannot be negative")
        if self.one_time_fee is not None and self.one_time_fee < 0:
            raise ValueError("one_time_fee cannot be negative")

    def compute_initial_cost(self) -> float:
        """Compute the initial cost to start a challenge."""
        if self.one_time_fee is not None:
            return self.one_time_fee
        return self.monthly_fee  # First month's fee


@dataclass(frozen=True)
class SimulationConfig:
    """
    Configuration for the simulation run.

    Attributes
    ----------
    method : SimulationMethod
        Path generation method to use
    rules : ChallengeRules
        Challenge rules configuration
    costs : ChallengeCosts
        Challenge costs configuration
    n_simulations : int
        Number of simulations to run
    max_attempts_per_sim : int
        Maximum number of resets before giving up (default: 10)
    block_length : int, optional
        Block length for Monte Carlo block bootstrap
    path_length : int, optional
        Length of synthetic paths; default: max(252, min_trading_days * 5)
    random_seed : int, optional
        Random seed for reproducibility

    Examples
    --------
    >>> config = SimulationConfig(
    ...     method=SimulationMethod.MONTE_CARLO_BLOCK,
    ...     rules=ChallengeRules(0.10, 0.08, 5),
    ...     costs=ChallengeCosts(50.0, one_time_fee=500.0),
    ...     n_simulations=1000,
    ...     block_length=20,
    ...     random_seed=42
    ... )
    """
    method: SimulationMethod
    rules: ChallengeRules
    costs: ChallengeCosts
    n_simulations: int
    max_attempts_per_sim: int = 10
    block_length: Optional[int] = None
    path_length: Optional[int] = None
    random_seed: Optional[int] = None

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.n_simulations < 1:
            raise ValueError("n_simulations must be at least 1")
        if self.max_attempts_per_sim < 1:
            raise ValueError("max_attempts_per_sim must be at least 1")
        if self.method == SimulationMethod.MONTE_CARLO_BLOCK:
            if self.block_length is None:
                raise ValueError("block_length required for MONTE_CARLO_BLOCK method")
            if self.block_length < 1:
                raise ValueError("block_length must be at least 1")
        if self.path_length is not None and self.path_length < self.rules.min_trading_days:
            raise ValueError("path_length must be at least min_trading_days")

    def get_path_length(self) -> int:
        """Get the effective path length for simulation."""
        if self.path_length is not None:
            return self.path_length
        return max(252, self.rules.min_trading_days * 5)


@dataclass
class ChallengePass:
    """
    Result for a successful challenge completion.

    Attributes
    ----------
    days_to_pass : int
        Number of trading days to pass the challenge
    final_equity : float
        Final equity at challenge completion
    max_equity_peak : float
        Maximum equity peak reached during challenge
    total_cost : float
        Total cost including fees and resets
    n_resets : int
        Number of resets before passing
    equity_curve : pd.Series
        Full equity curve from challenge

    Examples
    --------
    >>> result = ChallengePass(
    ...     days_to_pass=15,
    ...     final_equity=0.085,
    ...     max_equity_peak=0.09,
    ...     total_cost=500.0,
    ...     n_resets=0,
    ...     equity_curve=pd.Series([0.0, 0.01, ...])
    ... )
    """
    days_to_pass: int
    final_equity: float
    max_equity_peak: float
    total_cost: float
    n_resets: int
    equity_curve: pd.Series

    @property
    def passed(self) -> bool:
        """Return True indicating this is a pass result."""
        return True


@dataclass
class ChallengeFail:
    """
    Result for a failed challenge.

    Attributes
    ----------
    days_to_failure : int
        Number of trading days before failure
    failure_reason : str
        Reason for failure: "max_drawdown", "trailing_drawdown",
        "daily_drawdown", or "max_attempts"
    max_equity_peak : float
        Maximum equity peak reached before failure
    equity_at_failure : float
        Equity level at failure
    total_cost : float
        Total cost including fees and resets
    n_resets : int
        Number of resets before final failure
    equity_curve : pd.Series
        Full equity curve from challenge

    Examples
    --------
    >>> result = ChallengeFail(
    ...     days_to_failure=8,
    ...     failure_reason="max_drawdown",
    ...     max_equity_peak=0.03,
    ...     equity_at_failure=-0.11,
    ...     total_cost=550.0,
    ...     n_resets=1,
    ...     equity_curve=pd.Series([0.0, 0.02, ...])
    ... )
    """
    days_to_failure: int
    failure_reason: str
    max_equity_peak: float
    equity_at_failure: float
    total_cost: float
    n_resets: int
    equity_curve: pd.Series

    @property
    def passed(self) -> bool:
        """Return False indicating this is a fail result."""
        return False


# Type alias for challenge results
ChallengeResult = Union[ChallengePass, ChallengeFail]


@dataclass
class SimulationStatistics:
    """
    Aggregate statistics from simulation results.

    Attributes
    ----------
    pass_probability : float
        Probability of passing the challenge
    fail_probability : float
        Probability of failing the challenge
    days_to_pass_distribution : dict
        Distribution stats for days to pass (mean, median, percentiles)
    max_equity_peak_failure_distribution : dict
        Distribution of max equity peaks for failed challenges
    expected_n_resets : float
        Expected number of resets
    expected_total_cost : float
        Expected total cost
    n_simulations : int
        Total number of simulations run
    n_passed : int
        Number of simulations that passed
    n_failed : int
        Number of simulations that failed
    failure_reasons : dict
        Count of each failure reason
    """
    pass_probability: float
    fail_probability: float
    days_to_pass_distribution: Dict[str, float]
    max_equity_peak_failure_distribution: Dict[str, float]
    expected_n_resets: float
    expected_total_cost: float
    n_simulations: int
    n_passed: int
    n_failed: int
    failure_reasons: Dict[str, int]
