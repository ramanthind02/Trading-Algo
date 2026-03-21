from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from prop_firms.optimization import (
    FloatSearchRange,
    IntSearchRange,
    LucidHyperoptConfig,
)
from prop_firms.base.portfolio_models import (
    PortfolioPayoutPolicyConfig,
    PortfolioPayoutPolicyMode,
    PortfolioSimulationConfig,
    PurchasePolicyConfig,
    ReturnEngineConfig,
)


_PROP_FIRMS_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class LucidPortfolioReportConfig:
    """Editable config for the Lucid portfolio report runner."""

    simulation: PortfolioSimulationConfig
    output_dir: Path = field(
        default_factory=lambda: _PROP_FIRMS_DIR / "results" / "lucid_portfolio"
    )
    data_path: Path | None = None
    report_stem: str = "lucid_25k_portfolio_report"
    save_csvs: bool = True


@dataclass(frozen=True)
class ApexPortfolioReportConfig:
    """Editable config for the Apex portfolio report runner."""

    simulation: PortfolioSimulationConfig
    output_dir: Path = field(
        default_factory=lambda: _PROP_FIRMS_DIR / "results" / "apex_portfolio"
    )
    data_path: Path | None = None
    report_stem: str = "apex_50k_portfolio_report"
    save_csvs: bool = True


@dataclass(frozen=True)
class LucidPortfolioHyperoptReportConfig:
    """Editable config for the Lucid hyperparameter optimizer runner."""

    optimization: LucidHyperoptConfig
    output_dir: Path = field(
        default_factory=lambda: _PROP_FIRMS_DIR / "results" / "lucid_hyperopt"
    )
    report_stem: str = "lucid_25k_hyperopt_report"
    save_csvs: bool = True


def load_report_config() -> LucidPortfolioReportConfig:
    """Single source of truth for the Lucid portfolio report runner."""

    # ==========================================================================
    # EDIT BELOW
    # ==========================================================================
    simulation = PortfolioSimulationConfig(
        account_code="25000",
        purchase_policy=PurchasePolicyConfig(
            funded_account_cap=5,
            challenge_account_cap=8,
            challenges_per_purchase_window=1,
        ),
        payout_policy=PortfolioPayoutPolicyConfig(
            mode=PortfolioPayoutPolicyMode.BUFFER,
            buffer_amount=2000.0,
            withdrawal_fraction=1,
        ),
        return_engine=ReturnEngineConfig(
            target_annual_volatility=0.10,
            target_sharpe=2,
            annualization_factor=252.0,
            start_date="2021-01-01",
            end_date="2025-12-31",
            random_seed=42 ,
        ),
        challenge_vol_multiplier=2,
        funded_vol_multiplier=0.5,
    )
    output_dir = _PROP_FIRMS_DIR / "results" / "lucid_portfolio"
    data_path = None
    report_stem = "lucid_25k_portfolio_report"
    save_csvs = True
    # ==========================================================================
    # EDIT ABOVE
    # ==========================================================================

    return LucidPortfolioReportConfig(
        simulation=simulation,
        output_dir=output_dir,
        data_path=data_path,
        report_stem=report_stem,
        save_csvs=save_csvs,
    )


def load_apex_report_config() -> ApexPortfolioReportConfig:
    """Single source of truth for the Apex portfolio report runner."""

    # ==========================================================================
    # EDIT BELOW
    # ==========================================================================
    simulation = PortfolioSimulationConfig(
        account_code="50000",
        purchase_policy=PurchasePolicyConfig(
            funded_account_cap=20,
            challenge_account_cap=20,
            challenges_per_purchase_window=3,
        ),
        payout_policy=PortfolioPayoutPolicyConfig(
            mode=PortfolioPayoutPolicyMode.FRACTIONAL,
            buffer_amount=2000.0,
            withdrawal_fraction=1,
        ),
        return_engine=ReturnEngineConfig(
            target_annual_volatility=0.10,
            target_sharpe=1,
            annualization_factor=252.0,
            start_date="2021-01-01",
            end_date="2025-12-31",
            random_seed=43,
        ),
        challenge_vol_multiplier=2.5,
        funded_vol_multiplier=0.5,
    )
    output_dir = _PROP_FIRMS_DIR / "results" / "apex_portfolio"
    data_path = None
    report_stem = "apex_50k_portfolio_report"
    save_csvs = True
    # ==========================================================================
    # EDIT ABOVE
    # ==========================================================================

    return ApexPortfolioReportConfig(
        simulation=simulation,
        output_dir=output_dir,
        data_path=data_path,
        report_stem=report_stem,
        save_csvs=save_csvs,
    )


def load_hyperopt_config() -> LucidPortfolioHyperoptReportConfig:
    """Single source of truth for the Lucid hyperopt runner."""

    # ==========================================================================
    # EDIT BELOW
    # ==========================================================================
    optimization = LucidHyperoptConfig(
        account_code="25000",
        funded_account_cap=20,
        challenge_account_cap=20,
        target_sharpe=2.0,
        monte_carlo_runs=40,
        n_trials=20,
        n_jobs=4,
        target_annual_volatility=0.10,
        annualization_factor=252.0,
        start_date="2021-01-01",
        end_date="2025-12-31",
        sampler_seed=42,
        random_seed_start=10_000,
        challenge_vol_multiplier_range=FloatSearchRange(low=0.9, high=1.5),
        funded_vol_multiplier_range=FloatSearchRange(low=0.4, high=0.9),
        payout_buffer_range=FloatSearchRange(low=500.0, high=2_500.0),
        withdrawal_fraction_range=FloatSearchRange(low=0.6, high=1.0),
        challenges_per_purchase_window_range=IntSearchRange(low=2, high=3),
        max_payouts_per_funded_account_choices=(5,),
    )
    output_dir = _PROP_FIRMS_DIR / "results" / "lucid_hyperopt"
    report_stem = "lucid_25k_hyperopt_report"
    save_csvs = True
    # ==========================================================================
    # EDIT ABOVE
    # ==========================================================================

    return LucidPortfolioHyperoptReportConfig(
        optimization=optimization,
        output_dir=output_dir,
        report_stem=report_stem,
        save_csvs=save_csvs,
    )
