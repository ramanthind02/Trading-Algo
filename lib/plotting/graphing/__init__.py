"""Graphing namespace for tearsheets and performance-table helpers."""

from lib.plotting.graphing.quantstats_reports import (
    compute_performance_report,
    generate_tearsheet,
)

__all__ = [
    "generate_tearsheet",
    "compute_performance_report",
]
