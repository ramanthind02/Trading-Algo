"""Pure helpers for the portfolio research workspace UI."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Final

from portfolio_research.config import (
    PortfolioFitMode,
    PortfolioResearchConfig,
    describe_weight_layer_policy,
    rebuild_weight_layer_kwargs,
)
from portfolio_research.shared.phase import PortfolioResearchPhase, artifact_view_phases
from portfolio_research.shared.visualization_paths import holdout_root
from portfolio_research.ui.contracts import (
    PhaseOption,
    PhaseRunPlan,
    PlanField,
    PortfolioResearchUiDefaults,
    PortfolioResearchUiRequest,
)
from portfolio_research.ui.workflow_steps import workflow_steps
from utils.core.enums import Ticker

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

_PHASE_OPTIONS: Final[tuple[PhaseOption, ...]] = (
    PhaseOption(
        PortfolioResearchPhase.FULL_PIPELINE,
        "Full pipeline (all stages)",
        "Run portfolio test, strategy holdout monitoring, and portfolio holdout in one command.",
    ),
    PhaseOption(
        PortfolioResearchPhase.PORTFOLIO_TEST,
        "Portfolio Test",
        "Train/validation/test tearsheets, FundedNext prop-firm reports, and holdout returns.",
    ),
    PhaseOption(
        PortfolioResearchPhase.STRATEGY_HOLDOUT,
        "Strategy Holdout",
        "Per-strategy holdout monitoring plots on the test window.",
    ),
    PhaseOption(
        PortfolioResearchPhase.PORTFOLIO_HOLDOUT,
        "Portfolio Holdout",
        "Full portfolio holdout report and aggregate monitoring charts.",
    ),
    PhaseOption(
        PortfolioResearchPhase.PROP_FIRM_REPORTS,
        "Prop Firm Reports",
        "FundedNext CFD portfolio simulation reports for train/validation/test phases.",
    ),
)


def build_ui_defaults(config: PortfolioResearchConfig) -> PortfolioResearchUiDefaults:
    ticker_options = tuple(sorted({ticker.name for ticker in config.tickers}))
    return PortfolioResearchUiDefaults(
        default_phase=PortfolioResearchPhase.FULL_PIPELINE,
        default_tickers=tuple(config.tickers),
        ticker_options=ticker_options,
        phase_options=_PHASE_OPTIONS,
        workflow_steps=workflow_steps(),
        ensemble_count=len(config.ensemble_dirs),
        config_source_path="portfolio_research/config.py",
        portfolio_fit_mode=config.portfolio_fit_mode,
        weight_layer_policy=describe_weight_layer_policy(rebuild_weight_layer_kwargs(config)),
        output_root=Path(config.output_root),
    )


def build_ui_request(
    *,
    phase_name: str,
    tickers_text: str | None,
    fallback_tickers: tuple[Ticker, ...],
    fit_mode_name: str | None = None,
) -> PortfolioResearchUiRequest:
    phase = PortfolioResearchPhase(phase_name)
    tickers = _parse_tickers(tickers_text, fallback_tickers)
    fit_mode = PortfolioFitMode[fit_mode_name] if fit_mode_name else None
    return PortfolioResearchUiRequest(
        phase=phase,
        tickers=tickers,
        portfolio_fit_mode=fit_mode,
    )


def apply_ui_request(
    config: PortfolioResearchConfig,
    request: PortfolioResearchUiRequest,
) -> PortfolioResearchConfig:
    updated = replace(config, tickers=list(request.tickers))
    if request.portfolio_fit_mode is not None:
        updated = replace(updated, portfolio_fit_mode=request.portfolio_fit_mode)
    return updated


def build_phase_plan(
    config: PortfolioResearchConfig,
    request: PortfolioResearchUiRequest,
) -> PhaseRunPlan:
    configured = apply_ui_request(config, request)
    output_path = (
        Path(configured.output_root)
        if request.phase
        in {
            PortfolioResearchPhase.PORTFOLIO_TEST,
            PortfolioResearchPhase.FULL_PIPELINE,
            PortfolioResearchPhase.PROP_FIRM_REPORTS,
        }
        else holdout_root(configured.output_root)
    )
    command = f"python -m portfolio_research.ui.run_phase --phase {request.phase.value} --execute"
    fields = (
        PlanField("Phase", request.phase.value),
        PlanField("Tickers", ", ".join(ticker.name for ticker in request.tickers)),
        PlanField("Ensembles", str(len(configured.ensemble_dirs))),
        PlanField("Fit mode", configured.portfolio_fit_mode.name),
        PlanField(
            "Weight layer",
            describe_weight_layer_policy(rebuild_weight_layer_kwargs(configured)),
        ),
        PlanField("Train", f"{configured.train_window.start.date()} → {configured.train_window.end.date()}"),
        PlanField("Validation", f"{configured.validation_window.start.date()} → {configured.validation_window.end.date()}"),
        PlanField("Test", f"{configured.test_window.start.date()} → {configured.test_window.end.date()}"),
        PlanField("Output", output_path.as_posix()),
    )
    will_run = _phase_steps(request.phase)
    return PhaseRunPlan(
        phase=request.phase,
        title=_plan_title(request.phase),
        description=_plan_description(request.phase),
        command=command,
        output_path=output_path,
        fields=fields,
        will_run=will_run,
        notes=("Edit portfolio_research/config.py for vault ensembles and windows.",),
    )


def defaults_to_dict(defaults: PortfolioResearchUiDefaults) -> dict[str, object]:
    return {
        "default_phase": defaults.default_phase.value,
        "default_tickers": [ticker.name for ticker in defaults.default_tickers],
        "ticker_options": list(defaults.ticker_options),
        "phase_options": [
            {"value": option.phase.value, "label": option.label, "description": option.description}
            for option in defaults.phase_options
        ],
        "workflow_steps": [
            {
                "phase": step.phase.value,
                "title": step.title,
                "summary": step.summary,
                "gate": step.gate,
                "docs_path": step.docs_path,
            }
            for step in defaults.workflow_steps
        ],
        "ensemble_count": defaults.ensemble_count,
        "config_source_path": defaults.config_source_path,
        "portfolio_fit_mode": defaults.portfolio_fit_mode.name,
        "weight_layer_policy": defaults.weight_layer_policy,
        "output_root": defaults.output_root.as_posix(),
    }


def plan_to_dict(plan: PhaseRunPlan) -> dict[str, object]:
    return {
        "phase": plan.phase.value,
        "title": plan.title,
        "description": plan.description,
        "command": plan.command,
        "output_path": plan.output_path.as_posix(),
        "fields": [{"label": field.label, "value": field.value} for field in plan.fields],
        "will_run": list(plan.will_run),
        "notes": list(plan.notes),
    }


def _parse_tickers(text: str | None, fallback: tuple[Ticker, ...]) -> tuple[Ticker, ...]:
    if not text or not text.strip():
        return fallback
    names = [part.strip().upper() for part in text.split(",") if part.strip()]
    return tuple(Ticker[name] for name in names)


def _plan_title(phase: PortfolioResearchPhase) -> str:
    if phase is PortfolioResearchPhase.FULL_PIPELINE:
        return "Full portfolio research pipeline"
    return next(step.title for step in workflow_steps() if step.phase == phase)


def _plan_description(phase: PortfolioResearchPhase) -> str:
    if phase is PortfolioResearchPhase.FULL_PIPELINE:
        return (
            "Runs stages 1–3 in order: train/validation/test tearsheets, per-strategy "
            "holdout monitoring, then portfolio-level holdout analytics."
        )
    return next(step.summary for step in workflow_steps() if step.phase == phase)


def _phase_steps(phase: PortfolioResearchPhase) -> tuple[str, ...]:
    if phase is PortfolioResearchPhase.FULL_PIPELINE:
        return (
            *_phase_steps(PortfolioResearchPhase.PORTFOLIO_TEST),
            *_phase_steps(PortfolioResearchPhase.STRATEGY_HOLDOUT),
            *_phase_steps(PortfolioResearchPhase.PORTFOLIO_HOLDOUT),
        )
    match phase:
        case PortfolioResearchPhase.PORTFOLIO_TEST:
            return (
                "Run cache preflight for vault ensembles.",
                "Fit/score train, validation, and test windows; write QuantStats tearsheets.",
                "Write composite tearsheets and holdout return CSVs for strategy monitoring.",
            )
        case PortfolioResearchPhase.STRATEGY_HOLDOUT:
            return (
                "Load stitched strategy return matrices.",
                "Run holdout robustness per vault ensemble.",
                "Write Matplotlib monitoring charts per strategy.",
            )
        case PortfolioResearchPhase.PORTFOLIO_HOLDOUT:
            return (
                "Run holdout evaluation when return CSVs are missing.",
                "Compute portfolio holdout analytics (correlation, IDM, contribution).",
                "Write portfolio holdout summary and aggregate plots.",
            )
        case PortfolioResearchPhase.PROP_FIRM_REPORTS:
            return (
                "Run portfolio test pipeline (includes FundedNext simulation).",
                "Write HTML/MD/CSV prop-firm reports under {phase}/prop_firm/fundednext/.",
            )
