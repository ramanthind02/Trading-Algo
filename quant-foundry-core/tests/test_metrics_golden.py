"""Golden-file parity for the in-repo QuantStats-derived metrics table."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantfoundry_core.metrics import (
    ReportMode,
    ReturnsCompounding,
    compute_aligned_performance_metrics,
)

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "quantstats_metrics_golden.json"
_GOLDEN: dict[str, object] = json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _series(*, seed: int) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=220, freq="D", tz="UTC")
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0002, 0.011, len(idx)), index=idx)


def _expected_block(name: str) -> pd.DataFrame:
    block = _GOLDEN[name]
    assert isinstance(block, dict)
    rows_obj = block["rows"]
    order_obj = block["row_order"]
    cols_obj = block["col_order"]
    assert isinstance(rows_obj, dict)
    assert isinstance(order_obj, list)
    assert isinstance(cols_obj, list)
    rows: dict[str, dict[str, object]] = {str(k): v for k, v in rows_obj.items()}  # type: ignore[misc]
    order = [str(x) for x in order_obj]
    cols = [str(x) for x in cols_obj]
    data = {c: [rows[r][c] for r in order] for c in cols}
    return pd.DataFrame(data, index=order)


def test_metrics_match_golden_full_with_benchmark() -> None:
    s, b = _series(seed=0), _series(seed=1)
    aligned = compute_aligned_performance_metrics(
        s,
        benchmark_returns=b,
        mode=ReportMode.FULL,
        compounding=ReturnsCompounding.SIMPLE,
        periods_per_year=252,
        match_dates=False,
        prepare_returns=False,
    )
    pd.testing.assert_frame_equal(aligned.to_dataframe(), _expected_block("full_bench"))


def test_metrics_match_golden_basic_strategy_only() -> None:
    aligned = compute_aligned_performance_metrics(
        _series(seed=2),
        benchmark_returns=None,
        mode=ReportMode.BASIC,
        compounding=ReturnsCompounding.SIMPLE,
        match_dates=False,
        prepare_returns=False,
    )
    pd.testing.assert_frame_equal(aligned.to_dataframe(), _expected_block("basic_only"))


def test_metrics_match_golden_basic_benchmark_seeded() -> None:
    s, b = _series(seed=3), _series(seed=4)
    aligned = compute_aligned_performance_metrics(
        s,
        benchmark_returns=b,
        mode=ReportMode.BASIC,
        compounding=ReturnsCompounding.SIMPLE,
        match_dates=False,
        prepare_returns=False,
    )
    pd.testing.assert_frame_equal(aligned.to_dataframe(), _expected_block("tz_naive_ref"))
