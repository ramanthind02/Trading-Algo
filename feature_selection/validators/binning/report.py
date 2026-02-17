"""Comprehensive report generation for binning diagnostics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
from matplotlib.figure import Figure

from feature_selection.base_models.base_model import BinningModelBase
from feature_selection.validators.binning.diagnostics import (
    BinningSuccessCriteria,
    RegionMetadata,
    extract_region_metadata,
    validate_binning_success,
    _require_fitted_model,
)
from feature_selection.validators.binning.shape_analysis import (
    AdjacencyAnalysis,
    RegionCoverage,
    analyze_multi_region_shapes,
    calculate_region_coverage_breakdown,
    detect_region_adjacency,
)
from feature_selection.validators.binning.plots import (
    create_diagnostic_panel,
    plot_bin_heatmap,
    plot_position_multiplier_curve,
    plot_region_boundaries,
)


def _build_region_metadata(model: BinningModelBase, bins: list[int]) -> RegionMetadata:
    """Build RegionMetadata from a list of bin indices.

    Args:
        model: Fitted binning model
        bins: List of bin indices to include in region

    Returns:
        RegionMetadata with aggregated statistics
    """
    sharpe_values = [float(model.bin_stats_[b]["sharpe"]) for b in bins]
    t_values = [float(model.bin_stats_[b]["t_stat"]) for b in bins]
    counts = [int(model.bin_stats_[b]["count"]) for b in bins]
    feature_mins = [float(model.bin_stats_[b]["feature_min"]) for b in bins]
    feature_maxs = [float(model.bin_stats_[b]["feature_max"]) for b in bins]

    return RegionMetadata(
        start_bin=bins[0],
        end_bin=bins[-1],
        bins=bins,
        mean_sharpe=sum(sharpe_values) / len(sharpe_values),
        mean_t_stat=sum(t_values) / len(t_values),
        sample_count=sum(counts),
        feature_range=(min(feature_mins), max(feature_maxs)),
    )


def extract_directional_regions(
    model: BinningModelBase,
    criteria: BinningSuccessCriteria,
    direction_filter: Literal["long", "short", "both"] = "both",
) -> list[RegionMetadata]:
    """Extract regions by filtering bins directionally using SIGNED t-stat.

    Filters individual bins by directional criteria, then forms contiguous regions
    from the qualified bins. This ensures each region contains only bins with
    consistent directional edge.

    Args:
        model: Fitted binning model
        criteria: Success criteria with thresholds
        direction_filter: Direction to filter ("long", "short", or "both")

    Returns:
        List of regions formed from directionally-qualified bins
    """
    _require_fitted_model(model)

    # Filter bins by directional criteria using SIGNED t-stat
    qualified_bins: list[int] = []

    for bin_idx in sorted(model.bin_stats_.keys()):
        stat = model.bin_stats_[bin_idx]
        t_stat = float(stat["t_stat"])
        sharpe = float(stat["sharpe"])
        metric_threshold = float(criteria.metric_threshold)
        t_threshold = float(criteria.t_threshold)

        # Directional filtering - uses SIGNED t-stat (not absolute!)
        if direction_filter == "long":
            # For long: positive edge with positive t-stat
            if t_stat > t_threshold and sharpe > metric_threshold:
                qualified_bins.append(bin_idx)
        elif direction_filter == "short":
            # For short: negative edge with negative t-stat
            if t_stat < -t_threshold and sharpe < -metric_threshold:
                qualified_bins.append(bin_idx)
        else:  # "both"
            # Accept bins meeting criteria in either direction
            long_ok = t_stat > t_threshold and sharpe > metric_threshold
            short_ok = t_stat < -t_threshold and sharpe < -metric_threshold
            if long_ok or short_ok:
                qualified_bins.append(bin_idx)

    if not qualified_bins:
        return []

    # Form contiguous regions from qualified bins
    regions: list[RegionMetadata] = []
    current_region_bins: list[int] = [qualified_bins[0]]

    for i in range(1, len(qualified_bins)):
        if qualified_bins[i] == current_region_bins[-1] + 1:
            # Contiguous - add to current region
            current_region_bins.append(qualified_bins[i])
        else:
            # Gap found - finalize current region if meets min_width
            if len(current_region_bins) >= criteria.min_region_width:
                regions.append(_build_region_metadata(model, current_region_bins))
            # Start new region
            current_region_bins = [qualified_bins[i]]

    # Finalize the last region
    if len(current_region_bins) >= criteria.min_region_width:
        regions.append(_build_region_metadata(model, current_region_bins))

    return regions


def select_best_regions(
    regions: list[RegionMetadata],
    max_regions: int = 1,
    direction_filter: Literal["long", "short", "both"] = "both",
) -> list[RegionMetadata]:
    """Select top N regions by SIGNED t-stat (respects direction).

    Args:
        regions: Directionally-filtered regions
        max_regions: Maximum number of regions to return
        direction_filter: Direction context ("long", "short", or "both")

    Returns:
        Top regions ranked by signed t-stat
    """
    if not regions:
        return []

    # Rank by SIGNED t-stat (not absolute)
    # For long: highest positive t-stat wins
    # For short: most negative t-stat wins
    # For both: highest absolute t-stat wins
    if direction_filter == "long":
        sorted_regions = sorted(regions, key=lambda r: r.mean_t_stat, reverse=True)
    elif direction_filter == "short":
        sorted_regions = sorted(regions, key=lambda r: r.mean_t_stat, reverse=False)
    else:  # "both"
        sorted_regions = sorted(regions, key=lambda r: abs(r.mean_t_stat), reverse=True)

    return sorted_regions[:max_regions]


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
    max_regions: int = 1,
    direction_filter: Literal["long", "short", "both"] = "both",
) -> BinningDiagnosticsReport:
    """Orchestrate all diagnostic steps and return comprehensive report.

    Args:
        model: Fitted binning model
        feature_data: Original feature values
        criteria: Success criteria for validation
        strategy: Trading strategy ("long" or "short")
        max_regions: Maximum number of regions to include (default 1)
        direction_filter: Direction filter ("long", "short", or "both")

    Returns:
        Comprehensive diagnostics report
    """
    success_verdict = validate_binning_success(model, criteria)

    # Extract directionally-filtered regions using signed t-stat
    directional_regions = extract_directional_regions(model, criteria, direction_filter)

    # Select best regions by signed t-stat
    regions = select_best_regions(directional_regions, max_regions, direction_filter)

    n_bins = int(model.n_bins)

    shape_summary = analyze_multi_region_shapes(regions, n_bins)
    coverage_breakdown = calculate_region_coverage_breakdown(regions, feature_data)
    adjacency_analysis = detect_region_adjacency(regions)

    total_coverage_pct = sum(rc.individual_coverage_pct for rc in coverage_breakdown)
    total_coverage_pct = max(0.0, min(100.0, total_coverage_pct))

    diagnostic_plots: dict[str, Figure] = {
        "heatmap": plot_bin_heatmap(model),
        "boundaries": plot_region_boundaries(model, feature_data, regions),
        "multiplier_curve": plot_position_multiplier_curve(model, strategy=strategy),
        "panel": create_diagnostic_panel(model, feature_data, regions, strategy=strategy),
    }

    failure_mode = detect_failure_mode(model, criteria)

    feature_column = model.feature_column or "unknown"
    parameter_combo = _extract_parameter_combo(model)

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    return BinningDiagnosticsReport(
        feature_column=feature_column,
        parameter_combo=parameter_combo,
        success_verdict=success_verdict,
        failure_mode=failure_mode,
        criteria=criteria,
        regions=regions,
        shape_summary=shape_summary,
        coverage_breakdown=coverage_breakdown,
        total_coverage_pct=total_coverage_pct,
        adjacency_analysis=adjacency_analysis,
        diagnostic_plots=diagnostic_plots,
        timestamp=timestamp,
    )


def _extract_parameter_combo(model: BinningModelBase) -> dict[str, object]:
    """Extract parameter combination from model fit config."""
    params = model.get_params()
    return {
        k: v
        for k, v in params.items()
        if k not in {"strategy", "normalize_by"}
    }


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
    regions = extract_region_metadata(model)

    if not regions:
        return "no_regions"

    min_width = int(criteria.min_region_width)
    wide_enough = [r for r in regions if len(r.bins) >= min_width]

    if not wide_enough:
        return "isolated_spikes"

    metric_threshold = float(criteria.metric_threshold)
    t_threshold = float(criteria.t_threshold)

    for region in wide_enough:
        all_bins_pass = True
        for bin_idx in region.bins:
            stat = model.bin_stats_[bin_idx]
            metric_long = float(stat.get("selection_metric_long", float("-inf")))
            metric_short = float(stat.get("selection_metric_short", float("-inf")))
            metric_ok = metric_long >= metric_threshold or metric_short >= metric_threshold
            t_ok = abs(float(stat["t_stat"])) >= t_threshold
            if not (metric_ok and t_ok):
                all_bins_pass = False
                break
        if all_bins_pass:
            return "none"

    return "insufficient_edge"


def save_report(report: BinningDiagnosticsReport, output_dir: str) -> str:
    """Save report to JSON (metadata) and PNG (plots).

    Args:
        report: Binning diagnostics report
        output_dir: Directory to save report and plots

    Returns:
        Path to saved JSON report file
    """
    base_dir = Path(output_dir) / report.feature_column
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_paths: dict[str, str] = {}
    for plot_name, fig in report.diagnostic_plots.items():
        plot_filename = f"{plot_name}_{report.feature_column}.png"
        plot_path = plots_dir / plot_filename
        fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
        plot_paths[plot_name] = f"plots/{plot_filename}"

    report_dict = _report_to_serializable_dict(report, plot_paths)

    param_hash = _compute_param_hash(report.parameter_combo)
    report_filename = f"binning_report_{param_hash}.json"
    report_path = base_dir / report_filename

    with open(str(report_path), "w") as f:
        json.dump(report_dict, f, indent=2)

    return str(report_path)


def _report_to_serializable_dict(
    report: BinningDiagnosticsReport,
    plot_paths: dict[str, str],
) -> dict[str, object]:
    """Convert report to JSON-serializable dictionary."""
    criteria_dict = {
        "metric_threshold": report.criteria.metric_threshold,
        "t_threshold": report.criteria.t_threshold,
        "min_region_width": report.criteria.min_region_width,
    }

    regions_list = [
        {
            "start_bin": r.start_bin,
            "end_bin": r.end_bin,
            "bins": r.bins,
            "mean_sharpe": r.mean_sharpe,
            "mean_t_stat": r.mean_t_stat,
            "sample_count": r.sample_count,
            "feature_range": list(r.feature_range),
        }
        for r in report.regions
    ]

    coverage_list = [
        {
            "region_id": rc.region_id,
            "individual_coverage_pct": rc.individual_coverage_pct,
            "cumulative_coverage_pct": rc.cumulative_coverage_pct,
        }
        for rc in report.coverage_breakdown
    ]

    adjacency_dict = {
        "gap_sizes": list(report.adjacency_analysis.gap_sizes),
        "is_connected": report.adjacency_analysis.is_connected,
        "isolation_score": report.adjacency_analysis.isolation_score,
    }

    return {
        "feature_column": report.feature_column,
        "parameter_combo": report.parameter_combo,
        "success_verdict": report.success_verdict,
        "failure_mode": report.failure_mode,
        "criteria": criteria_dict,
        "regions": regions_list,
        "shape_summary": report.shape_summary,
        "coverage_breakdown": coverage_list,
        "total_coverage_pct": report.total_coverage_pct,
        "adjacency_analysis": adjacency_dict,
        "diagnostic_plots": plot_paths,
        "timestamp": report.timestamp,
    }


def _compute_param_hash(parameter_combo: dict[str, object]) -> str:
    """Compute a short deterministic hash from parameter combo for filenames."""
    import hashlib

    param_str = json.dumps(parameter_combo, sort_keys=True, default=str)
    return hashlib.md5(param_str.encode()).hexdigest()[:8]


def display_report_summary(report: BinningDiagnosticsReport) -> None:
    """Pretty-print key metrics to console.

    Args:
        report: Binning diagnostics report
    """
    verdict_str = "PASS" if report.success_verdict else "FAIL"

    print(f"\n{'=' * 60}")
    print(f"  Binning Diagnostics Report")
    print(f"{'=' * 60}")
    print(f"  Feature:       {report.feature_column}")
    print(f"  Timestamp:     {report.timestamp}")
    print(f"  Verdict:       {verdict_str}")
    print(f"  Failure Mode:  {report.failure_mode}")
    print(f"{'─' * 60}")
    print(f"  Regions:       {len(report.regions)}")
    print(f"  Coverage:      {report.total_coverage_pct:.1f}%")
    print(f"{'─' * 60}")
    print(f"  Shape Summary:")
    for shape_type, count in report.shape_summary.items():
        print(f"    {shape_type}: {count}")
    print(f"{'─' * 60}")
    print(f"  Criteria:")
    print(f"    metric_threshold:  {report.criteria.metric_threshold}")
    print(f"    t_threshold:       {report.criteria.t_threshold}")
    print(f"    min_region_width:  {report.criteria.min_region_width}")
    print(f"{'=' * 60}\n")
