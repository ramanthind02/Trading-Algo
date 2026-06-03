"""Data helpers for parameter-sensitivity CSV exports (pivot explorer payload)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
import pandas as pd

ScalarParamValue: TypeAlias = str | int | float


@dataclass(frozen=True)
class PreparedDataset:
    name: str
    title: str
    frame: pd.DataFrame
    axis_params: tuple[str, ...]
    filter_dimensions: tuple[str, ...]
    value_options: dict[str, tuple[str, ...]]


def _read_csv_if_nonempty(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    return None if frame.empty else frame


def _parse_param_value(value: object) -> ScalarParamValue:
    if value is None:
        return "NA"
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return int(numeric) if numeric.is_integer() else numeric
    text = str(value).strip()
    if text == "":
        return ""
    try:
        numeric = float(text)
    except ValueError:
        return text
    return int(numeric) if numeric.is_integer() else numeric


def _format_param_value(value: ScalarParamValue) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else f"{value:g}"
    return value


def _sort_key(value: ScalarParamValue) -> tuple[int, float | str]:
    if isinstance(value, (int, float)):
        return (0, float(value))
    return (1, value)


def _ordered_unique(values: pd.Series) -> tuple[ScalarParamValue, ...]:
    uniques = {_parse_param_value(value) for value in values.tolist()}
    return tuple(sorted(uniques, key=_sort_key))


def _param_order(long_df: pd.DataFrame) -> tuple[str, ...]:
    ordered = (
        long_df[["param_key", "param_sort_order"]]
        .drop_duplicates()
        .sort_values(["param_sort_order", "param_key"], kind="mergesort")
    )
    return tuple(ordered["param_key"].astype(str).tolist())


def _wide_param_frame(long_df: pd.DataFrame) -> pd.DataFrame:
    parsed = long_df.assign(parsed_param_value=long_df["param_value"].map(_parse_param_value))
    wide = (
        parsed.pivot_table(
            index="param_combo_label",
            columns="param_key",
            values="parsed_param_value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    return wide


def _varying_param_columns(frame: pd.DataFrame, ordered_params: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        param for param in ordered_params if param in frame.columns and frame[param].nunique(dropna=False) > 1
    )


def _value_options(frame: pd.DataFrame, dimensions: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    return {
        dimension: tuple(_format_param_value(value) for value in _ordered_unique(frame[dimension]))
        for dimension in dimensions
    }


def _merge_parameter_dimensions(metrics_df: pd.DataFrame, long_df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    ordered_params = _param_order(long_df)
    wide = _wide_param_frame(long_df)
    merged = metrics_df.merge(wide, on="param_combo_label", how="left")
    return merged, ordered_params


def _build_dataset(
    *,
    name: str,
    title: str,
    metrics_df: pd.DataFrame,
    long_df: pd.DataFrame,
    extra_filter_dimensions: tuple[str, ...] = (),
) -> PreparedDataset | None:
    merged, ordered_params = _merge_parameter_dimensions(metrics_df, long_df)
    axis_params = _varying_param_columns(merged, ordered_params)
    if len(axis_params) < 2:
        return None
    filter_dimensions = tuple(
        dimension
        for dimension in extra_filter_dimensions
        if dimension in merged.columns and merged[dimension].nunique(dropna=False) > 1
    )
    value_options = _value_options(merged, tuple((*filter_dimensions, *axis_params)))
    return PreparedDataset(
        name=name,
        title=title,
        frame=merged,
        axis_params=axis_params,
        filter_dimensions=filter_dimensions,
        value_options=value_options,
    )


def load_parameter_sensitivity_datasets(input_dir: Path) -> tuple[PreparedDataset, ...]:
    """Load merged parameter-sensitivity datasets from visualization CSV exports."""

    return _prepare_datasets(input_dir)


def _prepare_datasets(input_dir: Path) -> tuple[PreparedDataset, ...]:
    pooled_path = input_dir / "param_sensitivity.csv"
    by_ticker_path = input_dir / "param_sensitivity_by_ticker.csv"
    long_path = input_dir / "param_combo_long.csv"

    pooled_df = _read_csv_if_nonempty(pooled_path)
    long_df = _read_csv_if_nonempty(long_path)
    if pooled_df is None or long_df is None:
        return ()

    by_ticker_df = _read_csv_if_nonempty(by_ticker_path)
    pooled_dataset = _build_dataset(
        name="pooled",
        title="Pooled across tickers",
        metrics_df=pooled_df,
        long_df=long_df,
    )
    by_ticker_dataset = (
        _build_dataset(
            name="by_ticker",
            title="Per-ticker slices",
            metrics_df=by_ticker_df,
            long_df=long_df,
            extra_filter_dimensions=("ticker",),
        )
        if by_ticker_df is not None
        else None
    )
    return tuple(dataset for dataset in (pooled_dataset, by_ticker_dataset) if dataset is not None)
