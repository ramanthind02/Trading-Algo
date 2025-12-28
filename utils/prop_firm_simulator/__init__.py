"""
Prop Firm Challenge Simulator Package

This package provides simulation tools for testing trading strategies
against prop firm challenge rules (max drawdown, profit targets, etc.).

Simulation Methods:
- Historical Walk-Forward: Random start points in actual data
- Monte Carlo Block Bootstrap: Synthetic paths preserving autocorrelation

Example Usage:
    from utils.prop_firm_simulator import (
        PropFirmChallengeSimulator,
        SimulationConfig,
        SimulationMethod,
        ChallengeRules,
        ChallengeCosts
    )

    # Define challenge rules
    rules = ChallengeRules(
        max_drawdown_pct=0.10,
        profit_target_pct=0.08,
        min_trading_days=5,
        trailing_drawdown_pct=0.05
    )

    # Define costs
    costs = ChallengeCosts(reset_fee=50.0, one_time_fee=500.0)

    # Configure simulation
    config = SimulationConfig(
        method=SimulationMethod.MONTE_CARLO_BLOCK,
        rules=rules,
        costs=costs,
        n_simulations=1000,
        block_length=20,
        random_seed=42
    )

    # Run simulation
    simulator = PropFirmChallengeSimulator(strategy_returns, config)
    results, stats = simulator.run_simulation()

    print(f"Pass probability: {stats.pass_probability:.2%}")
    print(f"Expected cost: ${stats.expected_total_cost:.2f}")
"""

from utils.prop_firm_simulator.data_structures import (
    SimulationMethod,
    ChallengeRules,
    ChallengeCosts,
    SimulationConfig,
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

from utils.prop_firm_simulator.challenge_engine import PropFirmChallengeSimulator

from utils.prop_firm_simulator.statistics import (
    compute_statistics,
    print_statistics
)

__all__ = [
    # Main simulator
    'PropFirmChallengeSimulator',

    # Data structures
    'SimulationMethod',
    'ChallengeRules',
    'ChallengeCosts',
    'SimulationConfig',
    'ChallengePass',
    'ChallengeFail',
    'ChallengeResult',
    'SimulationStatistics',

    # Path generators
    'PathGenerator',
    'HistoricalWalkForwardGenerator',
    'MonteCarloBlockGenerator',

    # Statistics
    'compute_statistics',
    'print_statistics',
]
