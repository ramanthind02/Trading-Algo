"""Legacy binning report generation removed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from matplotlib.figure import Figure

from feature_selection.validation.binning.diagnostics import BinningSuccessCriteria, RegionMetadata


LEGACY_BINNING_REMOVED_ERROR = (
    "Legacy binning report generation removed; use frozen signed-signal validation."
)


@dataclass(frozen=True)
class BinningDiagnosticsReport:
    feature_column: str
    parameter_combo: dict[str, object]
    success_verdict: bool
    failure_mode: Literal["no_regions", "isolated_spikes", "insufficient_edge", "none"] | None
    criteria: BinningSuccessCriteria
    regions: list[RegionMetadata]
    shape_summary: dict[str, int]
    coverage_breakdown: list[object]
    total_coverage_pct: float
    adjacency_analysis: object
    timestamp: str


def _raise_legacy_binning_removed() -> None:
    raise RuntimeError(LEGACY_BINNING_REMOVED_ERROR)


def generate_binning_report(*_args: object, **_kwargs: object) -> tuple[BinningDiagnosticsReport, dict[str, Figure]]:
    _raise_legacy_binning_removed()


def save_report(*_args: object, **_kwargs: object) -> str:
    _raise_legacy_binning_removed()

