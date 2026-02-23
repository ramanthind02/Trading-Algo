"""
Grid-aware neighbor averaging for parameter stability analysis.

Adds a smoothed objective column where each row's value is the mean of its own
objective and all axis-aligned 1-step neighbors that exist in the grid.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def add_smoothed_objective(
    df: pd.DataFrame,
    param_columns: List[str],
    objective_column: str,
    output_column: str = "avg_objective",
) -> pd.DataFrame:
    """
    Add a stability-smoothed objective column using grid-aware neighbor averaging.

    For each row, computes the mean of the objective at that row and at all
    axis-aligned 1-step neighbor rows (neighbors differ in exactly one
    parameter by one step). Result is written to a new column; input is unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with param columns and one objective column.
    param_columns : List[str]
        Column names that define the grid (order used for neighbor definition).
    objective_column : str
        Name of the numeric column to smooth.
    output_column : str, default "avg_objective"
        Name of the new column to add.

    Returns
    -------
    pd.DataFrame
        New DataFrame with same rows/columns as df plus output_column.
    """
    _validate_inputs(df, param_columns, objective_column)

    result = df.copy()

    if df.empty:
        result[output_column] = pd.Series(dtype=float)
        return result

    # Build ordered adjacency maps: param_col -> {value: (prev, next)}
    adjacency = _build_adjacency_maps(df, param_columns)

    # Build lookup: param tuple -> representative objective (mean over duplicates)
    cell_lookup = _build_cell_lookup(df, param_columns, objective_column)

    # Compute smoothed values for each row
    smoothed = _compute_smoothed_values(
        df, param_columns, objective_column, adjacency, cell_lookup
    )

    result[output_column] = smoothed
    return result


def _validate_inputs(
    df: pd.DataFrame,
    param_columns: List[str],
    objective_column: str,
) -> None:
    """Validate that required columns exist and objective is numeric."""
    missing_params = [c for c in param_columns if c not in df.columns]
    if missing_params:
        raise ValueError(
            f"Parameter columns not found in DataFrame: {missing_params}"
        )

    if objective_column not in df.columns:
        raise ValueError(
            f"Objective column '{objective_column}' not found in DataFrame"
        )

    if not df.empty and not pd.api.types.is_numeric_dtype(df[objective_column]):
        raise TypeError(
            f"Objective column '{objective_column}' must be numeric"
        )


def _build_adjacency_maps(
    df: pd.DataFrame,
    param_columns: List[str],
) -> Dict[str, Dict[object, Tuple[Optional[object], Optional[object]]]]:
    """
    Build adjacency index for each parameter column.

    Returns a dict: column_name -> {value: (prev_value_or_None, next_value_or_None)}.
    Numeric columns use natural sort order; others use order of first appearance.
    """
    adjacency: Dict[str, Dict[object, Tuple[Optional[object], Optional[object]]]] = {}

    for col in param_columns:
        unique_vals = df[col].dropna().unique()

        if pd.api.types.is_numeric_dtype(df[col]):
            levels = sorted(unique_vals)
        else:
            # Preserve first-appearance order per spec
            _, idx = np.unique(df[col].dropna().values, return_index=True)
            levels = list(df[col].dropna().values[np.sort(idx)])

        col_adj: Dict[object, Tuple[Optional[object], Optional[object]]] = {}
        for i, val in enumerate(levels):
            prev_val = levels[i - 1] if i > 0 else None
            next_val = levels[i + 1] if i < len(levels) - 1 else None
            col_adj[val] = (prev_val, next_val)

        adjacency[col] = col_adj

    return adjacency


def _build_cell_lookup(
    df: pd.DataFrame,
    param_columns: List[str],
    objective_column: str,
) -> Dict[tuple, float]:
    """
    Build lookup from parameter tuple to representative objective value.

    For duplicate param tuples, the representative value is the mean of their
    objective values (excluding NaN).
    """
    grouped = df.groupby(param_columns, sort=False)[objective_column].mean()
    return {
        key if isinstance(key, tuple) else (key,): val
        for key, val in grouped.items()
    }


def _compute_smoothed_values(
    df: pd.DataFrame,
    param_columns: List[str],
    objective_column: str,
    adjacency: Dict[str, Dict[object, Tuple[Optional[object], Optional[object]]]],
    cell_lookup: Dict[tuple, float],
) -> np.ndarray:
    """
    Compute smoothed objective for every row in the DataFrame.

    For each row: collect self objective + objectives of all existing
    axis-aligned 1-step neighbors, then take nanmean.
    """
    param_values = df[param_columns].values
    obj_values = df[objective_column].values

    # Pre-compute column indices for adjacency lookup
    col_indices = {col: i for i, col in enumerate(param_columns)}

    smoothed = np.empty(len(df), dtype=float)

    for row_idx in range(len(df)):
        row_params = param_values[row_idx]
        self_val = obj_values[row_idx]

        neighbor_vals = _collect_neighbor_values(
            row_params, param_columns, col_indices, adjacency, cell_lookup
        )

        all_vals = [self_val] + neighbor_vals
        smoothed[row_idx] = np.nanmean(all_vals)

    return smoothed


def _collect_neighbor_values(
    row_params: np.ndarray,
    param_columns: List[str],
    col_indices: Dict[str, int],
    adjacency: Dict[str, Dict[object, Tuple[Optional[object], Optional[object]]]],
    cell_lookup: Dict[tuple, float],
) -> List[float]:
    """
    Collect objective values for all existing axis-aligned 1-step neighbors.

    For each parameter column, checks prev and next adjacent values. If the
    resulting neighbor tuple exists in the cell lookup, its value is included.
    """
    neighbors: List[float] = []
    row_tuple = tuple(row_params)

    for col in param_columns:
        idx = col_indices[col]
        current_val = row_params[idx]
        adj = adjacency[col].get(current_val)

        if adj is None:
            continue

        for neighbor_val in adj:
            if neighbor_val is None:
                continue
            neighbor_tuple = (
                row_tuple[:idx] + (neighbor_val,) + row_tuple[idx + 1:]
            )
            cell_val = cell_lookup.get(neighbor_tuple)
            if cell_val is not None and not np.isnan(cell_val):
                neighbors.append(cell_val)

    return neighbors
