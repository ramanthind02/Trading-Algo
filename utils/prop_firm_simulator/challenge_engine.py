"""
Challenge Engine for Prop Firm Simulator

This module provides the main simulation engine that orchestrates
prop firm challenge simulations.

Author: Trading Research Team
"""

import pandas as pd
from typing import List, Tuple

from utils.prop_firm_simulator.data_structures import (
    SimulationConfig,
    SimulationMethod,
    ChallengePass,
    ChallengeFail,
    ChallengeResult,
    SimulationStatistics
)
from utils.prop_firm_simulator.path_generators import (
    PathGenerator,
    HistoricalWalkForwardGenerator,
    MonteCarloBlockGenerator
)
from utils.prop_firm_simulator.statistics import compute_statistics
from metrics.risk.drawdown import check_drawdown_breach


class PropFirmChallengeSimulator:
    """
    Simulator for prop firm trading challenges.

    This class runs Monte Carlo simulations of prop firm challenges
    to estimate pass probability, expected costs, and other statistics.

    Parameters
    ----------
    returns : pd.Series
        Historical strategy returns with datetime index
    config : SimulationConfig
        Simulation configuration

    Examples
    --------
    >>> from utils.prop_firm_simulator import (
    ...     PropFirmChallengeSimulator, SimulationConfig, SimulationMethod,
    ...     ChallengeRules, ChallengeCosts
    ... )
    >>> rules = ChallengeRules(0.10, 0.08, 5)
    >>> costs = ChallengeCosts(50.0, one_time_fee=500.0)
    >>> config = SimulationConfig(
    ...     method=SimulationMethod.HISTORICAL_WALKFORWARD,
    ...     rules=rules,
    ...     costs=costs,
    ...     n_simulations=1000,
    ...     random_seed=42
    ... )
    >>> simulator = PropFirmChallengeSimulator(returns, config)
    >>> results, stats = simulator.run_simulation()
    >>> print(f"Pass probability: {stats.pass_probability:.2%}")
    """

    def __init__(self, returns: pd.Series, config: SimulationConfig):
        """
        Initialize the prop firm challenge simulator.

        Parameters
        ----------
        returns : pd.Series
            Historical strategy returns
        config : SimulationConfig
            Simulation configuration
        """
        self._validate_returns(returns)
        self._returns = returns
        self._config = config
        self._generator = self._create_generator()

    def _validate_returns(self, returns: pd.Series) -> None:
        """Validate the input returns series."""
        if not isinstance(returns, pd.Series):
            raise TypeError("returns must be a pandas Series")
        if len(returns) < 2:
            raise ValueError("returns must have at least 2 data points")
        if not isinstance(returns.index, pd.DatetimeIndex):
            raise TypeError("returns must have a DatetimeIndex")
        if returns.isna().any():
            raise ValueError("returns cannot contain NaN values")

    def _create_generator(self) -> PathGenerator:
        """Create the appropriate path generator based on config."""
        if self._config.method == SimulationMethod.HISTORICAL_WALKFORWARD:
            return HistoricalWalkForwardGenerator()
        elif self._config.method == SimulationMethod.MONTE_CARLO_BLOCK:
            return MonteCarloBlockGenerator(self._config.block_length)
        else:
            raise ValueError(f"Unknown simulation method: {self._config.method}")

    def run_simulation(
        self,
        verbose: bool = False
    ) -> Tuple[List[ChallengeResult], SimulationStatistics]:
        """
        Run the full simulation.

        Parameters
        ----------
        verbose : bool, default=False
            Whether to print progress information

        Returns
        -------
        Tuple[List[ChallengeResult], SimulationStatistics]
            - List of individual challenge results
            - Aggregate statistics from all simulations
        """
        base_seed = self._config.random_seed if self._config.random_seed is not None else 42
        results: List[ChallengeResult] = []

        if verbose:
            print(f"\n{'='*70}")
            print("PROP FIRM CHALLENGE SIMULATION")
            print(f"{'='*70}")
            print(f"Method: {self._config.method.value}")
            print(f"Simulations: {self._config.n_simulations}")
            print(f"Max attempts per sim: {self._config.max_attempts_per_sim}")
            print(f"Rules: max_dd={self._config.rules.max_drawdown_pct:.1%}, "
                  f"target={self._config.rules.profit_target_pct:.1%}, "
                  f"min_days={self._config.rules.min_trading_days}")
            print(f"{'='*70}\n")

        for i in range(self._config.n_simulations):
            # Generate path for this simulation
            path_length = self._config.get_path_length()
            returns_path = self._generator.generate(
                self._returns,
                iteration=i,
                random_seed=base_seed,
                path_length=path_length
            )

            # Simulate the challenge with retry logic
            result = self._simulate_single_challenge(returns_path, i)
            results.append(result)

            if verbose and (i + 1) % 100 == 0:
                print(f"  Completed {i + 1}/{self._config.n_simulations} simulations...")

        # Compute aggregate statistics
        stats = compute_statistics(results, self._config)

        if verbose:
            print(f"\n{'='*70}")
            print("SIMULATION COMPLETE")
            print(f"{'='*70}")
            print(f"Pass rate: {stats.pass_probability:.2%}")
            print(f"Expected resets: {stats.expected_n_resets:.2f}")
            print(f"Expected cost: ${stats.expected_total_cost:.2f}")
            print(f"{'='*70}\n")

        return results, stats

    def _simulate_single_challenge(
        self,
        returns_path: pd.Series,
        simulation_idx: int
    ) -> ChallengeResult:
        """
        Simulate a single challenge with retry logic.

        Parameters
        ----------
        returns_path : pd.Series
            Return path to use for simulation
        simulation_idx : int
            Index of current simulation (for tracking)

        Returns
        -------
        ChallengeResult
            Result of the challenge (pass or fail)
        """
        rules = self._config.rules
        costs = self._config.costs

        n_resets = 0
        total_cost = costs.compute_initial_cost()
        all_equity_curves: List[pd.Series] = []

        # Track offset into the path for each attempt
        path_offset = 0
        remaining_path = returns_path

        while n_resets < self._config.max_attempts_per_sim:
            # Simulate one attempt
            result = self._simulate_attempt(remaining_path, rules)

            # Store equity curve
            all_equity_curves.append(result['equity_curve'])

            if result['passed']:
                # Challenge passed
                return ChallengePass(
                    days_to_pass=result['days'],
                    final_equity=result['final_equity'],
                    max_equity_peak=result['max_peak'],
                    total_cost=total_cost,
                    n_resets=n_resets,
                    equity_curve=self._combine_equity_curves(all_equity_curves)
                )

            # Challenge failed - check if we can retry
            n_resets += 1
            total_cost += costs.reset_fee

            # Advance path offset by the days used in this attempt
            path_offset += result['days']

            # Check if we have enough path remaining
            if path_offset >= len(returns_path) - rules.min_trading_days:
                # Not enough data to retry - final failure
                return ChallengeFail(
                    days_to_failure=result['days'],
                    failure_reason=result['failure_reason'],
                    max_equity_peak=result['max_peak'],
                    equity_at_failure=result['equity_at_failure'],
                    total_cost=total_cost,
                    n_resets=n_resets,
                    equity_curve=self._combine_equity_curves(all_equity_curves)
                )

            # Continue with remaining path
            remaining_path = returns_path.iloc[path_offset:]

        # Max attempts reached
        last_curve = all_equity_curves[-1] if all_equity_curves else pd.Series([0.0])
        return ChallengeFail(
            days_to_failure=len(last_curve),
            failure_reason="max_attempts",
            max_equity_peak=last_curve.max() if len(last_curve) > 0 else 0.0,
            equity_at_failure=last_curve.iloc[-1] if len(last_curve) > 0 else 0.0,
            total_cost=total_cost,
            n_resets=n_resets,
            equity_curve=self._combine_equity_curves(all_equity_curves)
        )

    def _simulate_attempt(
        self,
        returns_path: pd.Series,
        rules
    ) -> dict:
        """
        Simulate a single challenge attempt.

        Parameters
        ----------
        returns_path : pd.Series
            Returns to use for this attempt
        rules : ChallengeRules
            Challenge rules

        Returns
        -------
        dict
            Result dictionary with keys:
            - passed: bool
            - days: int
            - final_equity: float
            - max_peak: float
            - equity_at_failure: float (if failed)
            - failure_reason: str (if failed)
            - equity_curve: pd.Series
        """
        # Build equity curve incrementally (additive)
        equity_values = [0.0]  # Start at 0
        for ret in returns_path.values:
            equity_values.append(equity_values[-1] + ret)

        equity = pd.Series(equity_values[1:], index=returns_path.index)
        max_peak = 0.0
        days_traded = 0
        target_reached_day = -1

        for i in range(len(equity)):
            days_traded = i + 1
            current_equity = equity.iloc[i]
            max_peak = max(max_peak, current_equity)

            # Check drawdown breach
            equity_so_far = equity.iloc[:i+1]
            returns_so_far = returns_path.iloc[:i+1]

            breached, breach_idx, reason = check_drawdown_breach(
                equity=equity_so_far,
                max_drawdown_pct=rules.max_drawdown_pct,
                returns=returns_so_far,
                trailing_drawdown_pct=rules.trailing_drawdown_pct,
                max_daily_drawdown_pct=rules.max_daily_drawdown_pct
            )

            if breached:
                return {
                    'passed': False,
                    'days': days_traded,
                    'final_equity': current_equity,
                    'max_peak': max_peak,
                    'equity_at_failure': current_equity,
                    'failure_reason': reason,
                    'equity_curve': equity_so_far
                }

            # Check if target reached
            if current_equity >= rules.profit_target_pct:
                if target_reached_day < 0:
                    target_reached_day = days_traded

                # Check if min days met
                if days_traded >= rules.min_trading_days:
                    return {
                        'passed': True,
                        'days': days_traded,
                        'final_equity': current_equity,
                        'max_peak': max_peak,
                        'equity_curve': equity_so_far
                    }

        # Path exhausted without passing or failing by drawdown
        # If target was reached but min days not met, it's still a fail
        return {
            'passed': False,
            'days': days_traded,
            'final_equity': equity.iloc[-1] if len(equity) > 0 else 0.0,
            'max_peak': max_peak,
            'equity_at_failure': equity.iloc[-1] if len(equity) > 0 else 0.0,
            'failure_reason': "path_exhausted",
            'equity_curve': equity
        }

    def _combine_equity_curves(
        self,
        curves: List[pd.Series]
    ) -> pd.Series:
        """
        Combine multiple equity curves from retry attempts.

        Parameters
        ----------
        curves : List[pd.Series]
            List of equity curves from each attempt

        Returns
        -------
        pd.Series
            Combined equity curve
        """
        if not curves:
            return pd.Series([0.0])

        if len(curves) == 1:
            return curves[0]

        # Concatenate all curves, resetting to 0 at each retry
        all_values = []
        for curve in curves:
            all_values.extend(curve.values.tolist())

        # Create new datetime index
        start_date = curves[0].index[0] if len(curves[0]) > 0 else pd.Timestamp.now()
        new_index = pd.date_range(start=start_date, periods=len(all_values), freq='B')

        return pd.Series(all_values, index=new_index)

    @property
    def config(self) -> SimulationConfig:
        """Return the simulation configuration."""
        return self._config

    @property
    def returns(self) -> pd.Series:
        """Return the historical returns."""
        return self._returns
