"""Prop-firm HTML/MD reports from portfolio phase results (no duplicate phase fits)."""
from __future__ import annotations

from collections.abc import Mapping

from prop_firms.reporting import PropFirmReportArtifacts, generate_portfolio_report

from research.portfolio.config import PortfolioResearchConfig
from research.portfolio.pipelines.portfolio_test import PhaseResult
from research.portfolio.prop_firm_bridge import (
    align_portfolio_returns_with_report_engine,
    build_prop_firm_returns,
    create_prop_firm_portfolio_simulator,
    portfolio_simulation_config,
)


def run_prop_firm_reports_for_phases(
    portfolio_config: PortfolioResearchConfig,
    phase_results: Mapping[str, PhaseResult],
) -> dict[str, PropFirmReportArtifacts]:
    """Run FundedNext (or other QF preset) portfolio simulation per research phase."""
    report_cfg = portfolio_config.prop_firm_report
    if not report_cfg.enabled:
        return {}

    simulator = create_prop_firm_portfolio_simulator(report_cfg.firm_id)
    artifacts: dict[str, PropFirmReportArtifacts] = {}

    for phase_name in report_cfg.phases:
        phase_result = phase_results.get(phase_name)
        if phase_result is None:
            raise ValueError(
                f"Prop-firm report requested phase {phase_name!r} but it is missing from "
                "phase_results."
            )
        raw_returns = build_prop_firm_returns(phase_result)
        aligned_returns = align_portfolio_returns_with_report_engine(
            raw_returns,
            phase_name,
            report_cfg,
            portfolio_config,
        )
        simulation_config = portfolio_simulation_config(
            phase_name,
            report_cfg,
            portfolio_config,
        )
        sim_result = simulator.simulate(
            returns=aligned_returns,
            config=simulation_config,
        )
        rolling_result = None
        if report_cfg.rolling_enabled:
            try:
                rolling_result = simulator.simulate_rolling(
                    returns=aligned_returns,
                    config=simulation_config,
                    window_months=report_cfg.rolling_window_months,
                )
                print(
                    f"  Prop-firm rolling ({phase_name}): "
                    f"{len(rolling_result.windows)} x {report_cfg.rolling_window_months}-month windows"
                )
            except ValueError as exc:
                print(f"  Skipping prop-firm rolling for {phase_name}: {exc}")
        out_dir = (
            phase_result.output_dir
            / report_cfg.output_subdir
            / report_cfg.firm_id
        )
        stem = f"{report_cfg.report_stem}_{phase_name}"
        artifacts[phase_name] = generate_portfolio_report(
            result=sim_result,
            output_dir=out_dir,
            report_stem=stem,
            save_csvs=report_cfg.save_csvs,
            rolling=rolling_result,
        )
    return artifacts


def run_portfolio_prop_firm_portfolio_reports(
    portfolio_config: PortfolioResearchConfig,
    phase_results: Mapping[str, PhaseResult],
    *,
    emit_tearsheets: bool = False,
) -> dict[str, PropFirmReportArtifacts]:
    """Backward-compatible alias for integrated prop-firm reporting."""
    _ = emit_tearsheets
    return run_prop_firm_reports_for_phases(portfolio_config, phase_results)
