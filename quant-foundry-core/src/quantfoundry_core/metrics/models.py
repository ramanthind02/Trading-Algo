"""Metric report types (QuantStats-aligned programmatic table)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Mapping, TypeAlias

import pandas as pd

MetricCell: TypeAlias = float | str | int | None


class ReportMode(Enum):
    """Subset of QuantStats ``reports.metrics(..., mode=...)``."""

    BASIC = "basic"
    FULL = "full"


class ReturnsCompounding(Enum):
    """Maps to QuantStats ``compounded`` flag."""

    COMPOUNDED = auto()
    SIMPLE = auto()


@dataclass(frozen=True)
class AlignedMetricsReport:
    """Immutable snapshot of QuantStats-style metrics rows.

    ``rows`` should be built with immutable mappings (e.g. ``MappingProxyType``).
    """

    rows: Mapping[str, Mapping[str, MetricCell]]
    row_order: tuple[str, ...]
    column_order: tuple[str, ...]
    mode: ReportMode
    compounding: ReturnsCompounding
    periods_per_year: int
    reference_implementation: str

    def to_plain_dict(self) -> dict[str, dict[str, MetricCell]]:
        """Nested dict (JSON-friendly after filtering None / normalizing)."""
        return {r: {c: self.rows[r][c] for c in self.column_order} for r in self.row_order}

    def to_dataframe(self) -> pd.DataFrame:
        """Materialize as a metrics × columns DataFrame (same layout as QuantStats)."""
        from quantfoundry_core.metrics.engine import aligned_report_to_dataframe

        return aligned_report_to_dataframe(self)
