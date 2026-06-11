"""Final-validation lane for promotion candidates.

Runs the REAL live :class:`~deployment.live.vault_strategy.VaultRebalanceStrategy`
in a Nautilus ``BacktestEngine`` over historical data, generating signals
on-the-fly via the real :class:`~deployment.live.forecast_engine.VaultForecastEngine`
bounded to the simulation clock — so signal AND execution share one causal clock
and the run is lookahead-free by construction. See :mod:`research.validation.validation_lane`.
"""
from research.validation.validation_lane import (
    ValidationConfig,
    ValidationResult,
    run_validation_backtest,
)

__all__ = ["ValidationConfig", "ValidationResult", "run_validation_backtest"]
