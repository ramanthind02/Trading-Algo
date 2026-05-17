"""Thin helpers used by vendored QuantStats utils (no yfinance/network)."""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd


def safe_concat(
    objs: Iterable[Any],
    axis: int = 0,
    ignore_index: bool = False,
    sort: bool = False,
    **kwargs: Any,
) -> pd.DataFrame | pd.Series:
    """Delegate to pandas.concat (QuantStats shim)."""
    return pd.concat(objs, axis=axis, ignore_index=ignore_index, sort=sort, **kwargs)


def safe_resample(data: pd.DataFrame | pd.Series, freq: str, func_name: str | None = None, **kwargs: Any):
    agg = getattr(data.resample(freq, **kwargs), func_name or "first")
    return agg()
