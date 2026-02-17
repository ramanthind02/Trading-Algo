"""Feature validator public interfaces."""

from feature_selection.validators.binning import (
    AdjacencyAnalysis,
    BinningDiagnosticsReport,
    BinningSuccessCriteria,
    RegionCoverage,
    RegionMetadata,
    ShapeClassification,
    analyze_multi_region_shapes,
    calculate_coverage,
    calculate_region_coverage_breakdown,
    classify_region_shape,
    create_diagnostic_panel,
    detect_failure_mode,
    detect_region_adjacency,
    detect_region_shape,
    display_report_summary,
    extract_region_metadata,
    generate_binning_report,
    plot_bin_heatmap,
    plot_position_multiplier_curve,
    plot_region_boundaries,
    save_report,
    validate_binning_success,
)

__all__ = [
    # T005
    "BinningSuccessCriteria",
    "RegionMetadata",
    "validate_binning_success",
    "extract_region_metadata",
    "detect_region_shape",
    "calculate_coverage",
    # T006
    "ShapeClassification",
    "RegionCoverage",
    "AdjacencyAnalysis",
    "classify_region_shape",
    "analyze_multi_region_shapes",
    "calculate_region_coverage_breakdown",
    "detect_region_adjacency",
    # T007
    "plot_bin_heatmap",
    "plot_region_boundaries",
    "plot_position_multiplier_curve",
    "create_diagnostic_panel",
    # T008
    "BinningDiagnosticsReport",
    "generate_binning_report",
    "detect_failure_mode",
    "save_report",
    "display_report_summary",
]
