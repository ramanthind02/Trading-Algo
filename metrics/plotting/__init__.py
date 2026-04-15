"""Tearsheet-focused plotting namespace.

Only the QuantStats tearsheet helpers remain part of the supported package
surface. All other plotting helpers are being retired from the public barrel
exports so callers import the concrete modules directly when needed.
"""

from metrics.plotting.graphing.quantstats_reports import (
    compute_baseline_results,
    generate_tearsheet,
)

__all__ = [
    "generate_tearsheet",
    "compute_baseline_results",
]
