"""Typed contracts for the lightweight feature_research UI."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from feature_research.shared import FeatureResearchPhase
from utils.core.enums import Ticker


class UiRunAction(str, Enum):
    """Supported actions for the lightweight UI runner."""

    PREVIEW = "preview"
    EXECUTE = "execute"


class UiFieldKind(str, Enum):
    """Supported lightweight form control types."""

    TEXT = "text"
    SELECT = "select"
    TEXTAREA = "textarea"


@dataclass(frozen=True)
class UiFieldOption:
    """One option for a select field."""

    value: str
    label: str


@dataclass(frozen=True)
class UiParameterField:
    """One module-specific UI field."""

    key: str
    label: str
    help_text: str
    default_value: str
    kind: UiFieldKind = UiFieldKind.TEXT
    placeholder: str = ""
    options: tuple[UiFieldOption, ...] = ()


@dataclass(frozen=True)
class UiModuleOption:
    """One selectable bias-module option in the UI."""

    value: str
    label: str
    description: str
    fields: tuple[UiParameterField, ...]


@dataclass(frozen=True)
class UiWorkflowStep:
    """One docs-driven research workflow step."""

    phase: FeatureResearchPhase
    title: str
    summary: str
    gate: str
    docs_path: str


@dataclass(frozen=True)
class PhaseWindowOverride:
    """Optional train/validation window override for validation-style phases."""

    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime

    @property
    def test_start(self) -> datetime:
        return self.val_start

    @property
    def test_end(self) -> datetime:
        return self.val_end


@dataclass(frozen=True)
class PhaseOption:
    """One user-facing phase option in the small UI."""

    phase: FeatureResearchPhase
    label: str
    description: str


@dataclass(frozen=True)
class FeatureResearchUiDefaults:
    """Initial form defaults for the UI."""

    default_phase: FeatureResearchPhase
    default_tickers: tuple[Ticker, ...]
    ticker_options: tuple[str, ...]
    phase_options: tuple[PhaseOption, ...]
    workflow_steps: tuple[UiWorkflowStep, ...]
    configured_module: str
    configured_module_description: str
    configured_search_space: str
    config_source_path: str
    research_window: PhaseWindowOverride | None


@dataclass(frozen=True)
class FeatureResearchUiRequest:
    """Normalized form input for one planned run."""

    phase: FeatureResearchPhase
    tickers: tuple[Ticker, ...]
    #: Optional override for portfolio addition gate baseline discovery (e.g. ES,NQ,GC).
    #: When unset, ``portfolio_source.tickers`` from ``feature_research/config.py`` is kept.
    portfolio_tickers: tuple[Ticker, ...] | None = None
    window_override: PhaseWindowOverride | None = None


@dataclass(frozen=True)
class ComboDiagnostics:
    """Rendered combo-count diagnostics for the selected workflow."""

    raw_combo_count: int
    loaded_combo_count: int
    effective_combo_count: float | None
    evaluation_combo_label: str
    surface_interpretation: str | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlanField:
    """One label/value pair shown in the run preview."""

    label: str
    value: str


@dataclass(frozen=True)
class PhaseRunPlan:
    """Rendered preview state for a feature_research phase run."""

    phase: FeatureResearchPhase
    title: str
    description: str
    command: str
    output_path: Path
    combo_diagnostics: ComboDiagnostics | None
    fields: tuple[PlanField, ...]
    will_run: tuple[str, ...]
    notes: tuple[str, ...]
