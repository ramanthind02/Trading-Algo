"""QuantStats-aligned performance metrics table (in-repo, Apache-2.0 derived)."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from quantfoundry_core.metrics.catalog import QUANTSTATS_METRICS_REFERENCE
from quantfoundry_core.metrics.models import (
    AlignedMetricsReport,
    MetricCell,
    ReportMode,
    ReturnsCompounding,
)
from quantfoundry_core.metrics.vendor_qs.reports_metrics import metrics as qs_metrics_programmatic

if TYPE_CHECKING:
    pass


def _strip_index_to_utc_naive(series: pd.Series) -> pd.Series:
    out = series.copy()
    idx = out.index
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        out.index = idx.tz_convert("UTC").tz_localize(None)
    return out


def _coerce_cell(value: object) -> MetricCell:
    if value is None:
        return None
    if isinstance(value, (np.floating, float)) and (np.isnan(value) or np.isinf(value)):
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (np.integer, np.floating)):
        return float(value) if isinstance(value, np.floating) else int(value)
    return value  # type: ignore[return-value]


def compute_aligned_performance_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    *,
    mode: ReportMode = ReportMode.FULL,
    compounding: ReturnsCompounding = ReturnsCompounding.SIMPLE,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
    match_dates: bool = True,
    prepare_returns: bool = False,
    strategy_title: str = "Strategy",
    benchmark_title: str = "Benchmark",
) -> AlignedMetricsReport:
    """Build the same metric table as QuantStats ``reports.metrics(..., display=False)``.

    Uses the vendored in-repo ``vendor_qs`` implementation (Apache-2.0; no PyPI QuantStats wheel).

    **Index contract:** inputs use a ``DatetimeIndex`` interpreted in UTC for tz-aware
    series (converted to UTC-naive before the metrics pipeline).

    **Column order:** when ``benchmark_returns`` is set, columns are ordered
    ``[Benchmark, Strategy]`` (benchmark first), matching upstream QuantStats.
    """
    if not isinstance(strategy_returns, pd.Series):
        raise TypeError("strategy_returns must be a pandas Series")
    if not isinstance(strategy_returns.index, pd.DatetimeIndex):
        raise TypeError("strategy_returns must use a DatetimeIndex")

    s = _strip_index_to_utc_naive(strategy_returns)
    bench: pd.Series | None = None
    if benchmark_returns is not None:
        if not isinstance(benchmark_returns, pd.Series):
            raise TypeError("benchmark_returns must be a pandas Series or None")
        if not isinstance(benchmark_returns.index, pd.DatetimeIndex):
            raise TypeError("benchmark_returns must use a DatetimeIndex")
        bench = _strip_index_to_utc_naive(benchmark_returns)

    compounded = compounding is ReturnsCompounding.COMPOUNDED
    raw: pd.DataFrame = qs_metrics_programmatic(
        s,
        benchmark=bench,
        rf=risk_free_rate,
        display=False,
        mode=mode.value,
        sep=False,
        compounded=compounded,
        periods_per_year=periods_per_year,
        prepare_returns=prepare_returns,
        match_dates=match_dates,
        strategy_title=strategy_title,
        benchmark_title=benchmark_title,
    )

    row_order = tuple(str(i) for i in raw.index)
    column_order = tuple(str(c) for c in raw.columns)
    frozen: dict[str, MappingProxyType[str, MetricCell]] = {
        str(idx): MappingProxyType(
            {str(col): _coerce_cell(raw.loc[idx, col]) for col in raw.columns}
        )
        for idx in raw.index
    }

    return AlignedMetricsReport(
        rows=MappingProxyType(frozen),
        row_order=row_order,
        column_order=column_order,
        mode=mode,
        compounding=compounding,
        periods_per_year=periods_per_year,
        reference_implementation=QUANTSTATS_METRICS_REFERENCE,
    )


def aligned_report_to_dataframe(report: AlignedMetricsReport) -> pd.DataFrame:
    """Recreate a QuantStats-shaped DataFrame (metrics × columns)."""
    data = {
        col: [report.rows[r][col] for r in report.row_order] for col in report.column_order
    }
    return pd.DataFrame(data, index=list(report.row_order))
