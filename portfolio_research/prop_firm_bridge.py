"""Bridge portfolio phase outputs to QuantFoundry Core prop-firm simulation."""
from __future__ import annotations

import pandas as pd
from quantfoundry_core.prop_firm import (
    PortfolioPayoutPolicyConfig,
    PortfolioPayoutPolicyMode,
    PortfolioSimulationConfig,
    PurchasePolicyConfig,
    ReturnEngineConfig,
    build_return_series,
    create_simulator_for_firm,
)
from quantfoundry_core.prop_firm.simulator import CfdPortfolioSimulator

from ensemble.portfolio_impl.portfolio_tester import (
    aggregate_intraday_returns_to_daily,
    calculate_strategy_returns_from_positions,
)
from portfolio_research.config import PortfolioResearchConfig, PropFirmReportConfig
from portfolio_research.pipelines.portfolio_test import PhaseResult


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
    return daily.sort_index(kind="stable").astype(float)


def return_engine_for_phase(
    phase_name: str,
    report_cfg: PropFirmReportConfig,
    portfolio_cfg: PortfolioResearchConfig,
) -> ReturnEngineConfig:
    """Return-engine window aligned to the portfolio research phase being simulated."""
    match phase_name:
        case "train":
            start = portfolio_cfg.train_window.start
            end = portfolio_cfg.train_window.end
        case "validation":
            start = portfolio_cfg.validation_window.start
            end = portfolio_cfg.validation_window.end
        case "test":
            start = portfolio_cfg.test_window.start
            end = portfolio_cfg.test_window.end
        case _:
            raise ValueError(
                f"return_engine_for_phase expects train/validation/test, got {phase_name!r}"
            )
    return ReturnEngineConfig(
        target_annual_volatility=report_cfg.return_target_annual_volatility,
        target_sharpe=report_cfg.return_target_sharpe,
        annualization_factor=report_cfg.return_annualization_factor,
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
        random_seed=report_cfg.return_random_seed,
    )


def return_engine_for_research(
    report_cfg: PropFirmReportConfig,
    portfolio_cfg: PortfolioResearchConfig,
) -> ReturnEngineConfig:
    """Return-engine window aligned to portfolio research train→test span."""
    return ReturnEngineConfig(
        target_annual_volatility=report_cfg.return_target_annual_volatility,
        target_sharpe=report_cfg.return_target_sharpe,
        annualization_factor=report_cfg.return_annualization_factor,
        start_date=portfolio_cfg.train_window.start.strftime("%Y-%m-%d"),
        end_date=portfolio_cfg.test_window.end.strftime("%Y-%m-%d"),
        random_seed=report_cfg.return_random_seed,
    )


def portfolio_simulation_config(
    phase_name: str,
    report_cfg: PropFirmReportConfig,
    portfolio_cfg: PortfolioResearchConfig,
) -> PortfolioSimulationConfig:
    """Build QF portfolio simulation config from portfolio research settings."""
    return PortfolioSimulationConfig(
        account_code=report_cfg.account_code,
        challenge_vol_multiplier=report_cfg.challenge_vol_multiplier,
        funded_vol_multiplier=report_cfg.funded_vol_multiplier,
        purchase_policy=PurchasePolicyConfig(
            funded_account_cap=report_cfg.funded_account_cap,
            challenge_account_cap=report_cfg.challenge_account_cap,
            challenges_per_purchase_window=report_cfg.challenges_per_purchase_window,
        ),
        payout_policy=PortfolioPayoutPolicyConfig(
            mode=PortfolioPayoutPolicyMode.AGGRESSIVE,
            buffer_amount=report_cfg.payout_buffer_amount,
            withdrawal_fraction=report_cfg.payout_withdrawal_fraction,
        ),
        return_engine=return_engine_for_phase(phase_name, report_cfg, portfolio_cfg),
    )


def align_portfolio_returns_with_report_engine(
    raw_returns: pd.Series,
    phase_name: str,
    report_cfg: PropFirmReportConfig,
    portfolio_cfg: PortfolioResearchConfig,
) -> pd.Series:
    """Apply the same date filter and optional target-vol scaling as external replay."""
    return build_return_series(
        config=return_engine_for_phase(phase_name, report_cfg, portfolio_cfg),
        external_returns=raw_returns,
    )


def create_prop_firm_portfolio_simulator(firm_id: str) -> CfdPortfolioSimulator:
    """Load the bundled CFD portfolio simulator for a firm preset."""
    return create_simulator_for_firm(firm_id)
