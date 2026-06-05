"""Typed contracts for the portfolio research workspace UI."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from research.portfolio.config import PortfolioFitMode
from research.portfolio.shared.phase import PortfolioResearchPhase
from lib.core.enums import Ticker


class UiRunAction(str, Enum):
    PREVIEW = "preview"
    EXECUTE = "execute"


@dataclass(frozen=True)
class UiWorkflowStep:
    phase: PortfolioResearchPhase
    title: str
    summary: str
    gate: str
    docs_path: str


@dataclass(frozen=True)
class PhaseOption:
    phase: PortfolioResearchPhase
    label: str
    description: str


@dataclass(frozen=True)
class PortfolioResearchUiDefaults:
    default_phase: PortfolioResearchPhase
    default_tickers: tuple[Ticker, ...]
    ticker_options: tuple[str, ...]
    phase_options: tuple[PhaseOption, ...]
    workflow_steps: tuple[UiWorkflowStep, ...]
    ensemble_count: int
    config_source_path: str
    portfolio_fit_mode: PortfolioFitMode
    weight_layer_policy: str
    output_root: Path


@dataclass(frozen=True)
class PortfolioResearchUiRequest:
    phase: PortfolioResearchPhase
    tickers: tuple[Ticker, ...]
    portfolio_fit_mode: PortfolioFitMode | None = None


@dataclass(frozen=True)
class PlanField:
    label: str
    value: str


@dataclass(frozen=True)
class PhaseRunPlan:
    phase: PortfolioResearchPhase
    title: str
    description: str
    command: str
    output_path: Path
    fields: tuple[PlanField, ...]
    will_run: tuple[str, ...]
    notes: tuple[str, ...]
