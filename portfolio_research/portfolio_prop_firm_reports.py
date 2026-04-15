"""Run vault-backed portfolio phases through prop_firms multi-account simulators + reports."""
from __future__ import annotations

from typing import Literal, Sequence

from prop_firms import create_apex_portfolio_simulator, create_lucid_portfolio_simulator
from prop_firms.report_config import ApexPortfolioReportConfig, LucidPortfolioReportConfig
from prop_firms.reporting import PropFirmReportArtifacts, generate_portfolio_report

from portfolio_research.config import PortfolioResearchConfig
from portfolio_research.pipelines.portfolio_test import (
    run_portfolio_research_cache_preflight,
    run_single_phase_for_prop_firm,
)
from portfolio_research.prop_firm_bridge import (
    align_portfolio_returns_with_report_engine,
    build_prop_firm_returns,
)

PortfolioReportConfig = LucidPortfolioReportConfig | ApexPortfolioReportConfig
PhaseName = Literal["train", "validation", "test"]


def run_portfolio_prop_firm_portfolio_reports(
    portfolio_config: PortfolioResearchConfig,
    report_config: PortfolioReportConfig,
    provider: Literal["lucid", "apex"],
    phases: Sequence[PhaseName],
    *,
    emit_tearsheets: bool = False,
    report_output_subdir: str = "vault_portfolio",
) -> dict[str, PropFirmReportArtifacts]:
    """For each phase: fit/predict portfolio, align returns, run portfolio simulator, write reports.

    Uses ``report_config.simulation`` (purchase/payout/vol multipliers, ``account_code``) and
    ``report_config.simulation.return_engine`` for the same date filtering and optional
    target-volatility scaling applied when ``build_return_series`` gets external returns.

    Reports and CSVs are written under
    ``report_config.output_dir / report_output_subdir`` with stems
    ``{report_stem}_{phase}`` so synthetic runs in ``report_config.output_dir`` are not
    overwritten.
    """
    if not phases:
        raise ValueError("phases must be non-empty")

    run_portfolio_research_cache_preflight(portfolio_config)
    out_dir = report_config.output_dir / report_output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, PropFirmReportArtifacts] = {}
    for phase in phases:
        phase_result = run_single_phase_for_prop_firm(
            portfolio_config,
            phase,
            emit_tearsheets=emit_tearsheets,
            run_preflight=False,
        )
        raw_returns = build_prop_firm_returns(phase_result)
        aligned_returns = align_portfolio_returns_with_report_engine(
            raw_returns,
            report_config.simulation.return_engine,
        )
        simulator = (
            create_lucid_portfolio_simulator()
            if provider == "lucid"
            else create_apex_portfolio_simulator()
        )
        sim_result = simulator.simulate(
            returns=aligned_returns,
            config=report_config.simulation,
        )
        stem = f"{report_config.report_stem}_{phase}"
        artifacts[phase] = generate_portfolio_report(
            result=sim_result,
            output_dir=out_dir,
            report_stem=stem,
            save_csvs=report_config.save_csvs,
        )
    return artifacts
