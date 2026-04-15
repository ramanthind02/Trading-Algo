"""Bridge portfolio phase outputs to prop-firm simulators (simple returns + dispatch)."""
from __future__ import annotations

from typing import Literal

import pandas as pd

from ensemble.portfolio_impl.portfolio_tester import (
    aggregate_intraday_returns_to_daily,
    calculate_strategy_returns_from_positions,
)
from portfolio_research.pipelines.portfolio_test import PhaseResult
from prop_firms.apex.provider import create_apex_simulator
from prop_firms.base.portfolio_models import ReturnEngineConfig
from prop_firms.base.return_engine import build_return_series
from prop_firms.base.enums import PayoutPolicy, ResetPolicy
from prop_firms.base.models import SimulationRequest, SimulationResult
from prop_firms.lucid.provider import create_lucid_simulator


def build_prop_firm_returns(phase: PhaseResult) -> pd.Series:
    """Build a daily simple-return series aligned with prop-firm compounding rules."""
    if phase.combined_positions.empty or phase.daily_test_candles.empty:
        raise ValueError(
            "PhaseResult must include non-empty combined_positions and daily_test_candles"
        )
    raw = calculate_strategy_returns_from_positions(
        phase.combined_positions,
        phase.daily_test_candles,
        instrument_return_kind="simple",
    )
    daily = aggregate_intraday_returns_to_daily(raw)
    daily = daily.dropna()
    if daily.empty:
        raise ValueError("Prop-firm returns series is empty after aggregation")
    idx = pd.DatetimeIndex(pd.to_datetime(daily.index))
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    daily = daily.copy()
    daily.index = idx
    daily = daily.sort_index(kind="stable").astype(float)
    return daily


def align_portfolio_returns_with_report_engine(
    raw_returns: pd.Series,
    return_engine: ReturnEngineConfig,
) -> pd.Series:
    """Apply the same date filter and optional target-vol scaling as ``build_return_series``.

    Uses ``PortfolioSimulationConfig.return_engine`` from
    ``prop_firms.report_config`` so portfolio-backed runs match the report runner
    semantics for external replay.
    """
    return build_return_series(
        config=return_engine,
        external_returns=raw_returns,
        data_path=None,
    )


def run_prop_firm_simulation(
    *,
    provider: Literal["lucid", "apex"],
    account_code: str,
    returns: pd.Series,
    held_contracts: pd.DataFrame | None = None,
    payout_policy: PayoutPolicy = PayoutPolicy.AUTO_MAX,
    reset_policy: ResetPolicy = ResetPolicy.NO_RESETS,
    payout_amount: float | None = None,
    max_resets: int = 0,
    starting_balance_override: float | None = None,
) -> SimulationResult:
    """Run Lucid or Apex rule engine on a return stream."""
    request = SimulationRequest(
        account_code=account_code,
        returns=returns,
        held_contracts=held_contracts,
        payout_policy=payout_policy,
        reset_policy=reset_policy,
        payout_amount=payout_amount,
        max_resets=max_resets,
        starting_balance_override=starting_balance_override,
    )
    if provider == "lucid":
        engine = create_lucid_simulator()
    elif provider == "apex":
        engine = create_apex_simulator()
    else:
        raise ValueError(f"provider must be 'lucid' or 'apex', got {provider!r}")
    return engine.simulate(request)
