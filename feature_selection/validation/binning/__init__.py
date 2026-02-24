"""Public binning diagnostics interfaces (T005-T008)."""

from .diagnostics import (
    BinningSuccessCriteria,
    RegionMetadata,
    calculate_coverage,
    detect_region_shape,
    extract_region_metadata,
    validate_binning_success,
)
from .plots import (
    create_diagnostic_panel,
    plot_bin_heatmap,
    plot_position_multiplier_curve,
    plot_region_boundaries,
)
from .report import (
    BinningDiagnosticsReport,
    detect_failure_mode,
    display_report_summary,
    extract_directional_regions,
    generate_binning_report,
    save_report,
    select_best_regions,
)
from .shape_analysis import (
    AdjacencyAnalysis,
    RegionCoverage,
    ShapeClassification,
    analyze_multi_region_shapes,
    calculate_region_coverage_breakdown,
    classify_region_shape,
    detect_region_adjacency,
)

__all__ = [
    "AdjacencyAnalysis",
    "BinningSuccessCriteria",
    "RegionCoverage",
    "RegionMetadata",
    "ShapeClassification",
    "analyze_multi_region_shapes",
    "calculate_coverage",
    "calculate_region_coverage_breakdown",
    "classify_region_shape",
    "create_diagnostic_panel",
    "detect_region_adjacency",
    "detect_region_shape",
    "extract_region_metadata",
    "plot_bin_heatmap",
    "plot_position_multiplier_curve",
    "plot_region_boundaries",
    "validate_binning_success",
    "BinningDiagnosticsReport",
    "generate_binning_report",
    "detect_failure_mode",
    "save_report",
    "select_best_regions",
    "extract_directional_regions",
    "display_report_summary",
]
