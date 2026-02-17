"""Comprehensive report generation for binning diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from matplotlib.figure import Figure

from feature_selection.base_models.base_model import BinningModelBase
from feature_selection.validators.binning.diagnostics import (
    BinningSuccessCriteria,
    RegionMetadata,
)
from feature_selection.validators.binning.shape_analysis import (
    AdjacencyAnalysis,
    RegionCoverage,
)


@dataclass(frozen=True)
class BinningDiagnosticsReport:
    """Comprehensive container for all binning diagnostics."""

    feature_column: str
    parameter_combo: dict[str, object]
    success_verdict: bool
    failure_mode: Literal["no_regions", "isolated_spikes", "insufficient_edge", "none"] | None
    criteria: BinningSuccessCriteria
    regions: list[RegionMetadata]
    shape_summary: dict[str, int]
    coverage_breakdown: list[RegionCoverage]
    total_coverage_pct: float
    adjacency_analysis: AdjacencyAnalysis
    diagnostic_plots: dict[str, Figure]
    timestamp: str


def generate_binning_report(
    model: BinningModelBase,
    feature_data: pd.Series,
    criteria: BinningSuccessCriteria,
    strategy: str = "long",
) -> BinningDiagnosticsReport:
    """Orchestrate all diagnostic steps and return comprehensive report.

    Args:
        model: Fitted binning model
        feature_data: Original feature values
        criteria: Success criteria for validation
        strategy: Trading strategy ("long" or "short")

    Returns:
        Comprehensive diagnostics report
    """
    raise NotImplementedError("Agent report-builder will implement")


def detect_failure_mode(
    model: BinningModelBase, criteria: BinningSuccessCriteria
) -> Literal["no_regions", "isolated_spikes", "insufficient_edge", "none"]:
    """Classify binning failure type.

    Args:
        model: Fitted binning model
        criteria: Success criteria

    Returns:
        Failure mode classification
    """
    raise NotImplementedError("Agent report-builder will implement")


def save_report(report: BinningDiagnosticsReport, output_dir: str) -> str:
    """Save report to JSON (metadata) and PNG (plots).

    Args:
        report: Binning diagnostics report
        output_dir: Directory to save report and plots

    Returns:
        Path to saved JSON report file
    """
    raise NotImplementedError("Agent report-builder will implement")


def display_report_summary(report: BinningDiagnosticsReport) -> None:
    """Pretty-print key metrics to console.

    Args:
        report: Binning diagnostics report
    """
    raise NotImplementedError("Agent report-builder will implement")
