"""QuantStats-aligned metric tables (in-repo vendor implementation)."""

from quantfoundry_core.metrics.catalog import (
    BASIC_ROWS_STRATEGY_ONLY,
    EXTRA_FULL_ROWS,
    QUANTSTATS_METRICS_REFERENCE,
    full_row_set_with_benchmark,
)
from quantfoundry_core.metrics.engine import aligned_report_to_dataframe, compute_aligned_performance_metrics
from quantfoundry_core.metrics.models import AlignedMetricsReport, ReportMode, ReturnsCompounding

__all__ = (
    "AlignedMetricsReport",
    "BASIC_ROWS_STRATEGY_ONLY",
    "EXTRA_FULL_ROWS",
    "QUANTSTATS_METRICS_REFERENCE",
    "ReportMode",
    "ReturnsCompounding",
    "aligned_report_to_dataframe",
    "compute_aligned_performance_metrics",
    "full_row_set_with_benchmark",
)
