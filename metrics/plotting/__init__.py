"""Tearsheet-focused plotting namespace.

Only the QuantStats tearsheet helpers remain part of the supported package
surface, alongside QuantFoundry-core backed performance-table helpers. All
other plotting helpers are being retired from the public barrel exports so
callers import the concrete modules directly when needed.
"""

from metrics.plotting.graphing.quantstats_reports import (
    compute_performance_report,
    generate_tearsheet,
)

__all__ = [
    "generate_tearsheet",
    "compute_performance_report",
]
