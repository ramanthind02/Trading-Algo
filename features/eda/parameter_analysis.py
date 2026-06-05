"""Grid neighbor smoothing and parameter-sensitivity report for EDA / research pipelines."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Tuple

import numpy as np
import pandas as pd

from lib.compute.grid_smoothing import add_smoothed_objective


def _param_value_columns(n_dims: int) -> list[str]:
    return [f"param{k}_value" for k in range(1, n_dims + 1)]


def identify_neighbors(
    param_values: tuple[object, ...],
    grid_structure: dict[str, list[object]],
) -> list[tuple[object, ...]]:
    """Return all 1-step axis-aligned neighbor tuples for a parameter combination."""
    param_names = list(grid_structure.keys())
    neighbors: list[tuple[object, ...]] = []

    for dim_idx, name in enumerate(param_names):
        sorted_vals = grid_structure[name]
        current_val = param_values[dim_idx]

        try:
            pos = sorted_vals.index(current_val)
        except ValueError:
            continue

        for offset in (-1, 1):
            neighbor_pos = pos + offset
            if 0 <= neighbor_pos < len(sorted_vals):
                neighbor_list = list(param_values)
                neighbor_list[dim_idx] = sorted_vals[neighbor_pos]
                neighbors.append(tuple(neighbor_list))

    return neighbors


def compute_neighbor_smoothing(
    results_df: pd.DataFrame,
    param_names: list[str],
    metric_col: str,
    self_weight: float = 2.0,
) -> pd.DataFrame:
    """
    Add smoothed objective, stability ratio, and neighbor count columns.

    Wraps ``utils.compute.grid_smoothing.add_smoothed_objective()`` and enriches the
    result with ``smoothed_{metric_col}``, ``stability_ratio``, and ``n_neighbors``.
    """
    n_params = len(param_names)
    param_cols = _param_value_columns(n_params)

    smoothed_col = f"smoothed_{metric_col}"

    smoothed_df = add_smoothed_objective(
        df=results_df,
        param_columns=param_cols,
        objective_column=metric_col,
        output_column=smoothed_col,
        self_weight=self_weight,
    )

    raw = smoothed_df[metric_col].values.astype(float)
    smoothed = smoothed_df[smoothed_col].values.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(raw != 0.0, smoothed / raw, np.nan)
    smoothed_df["stability_ratio"] = ratio

    grid_structure: dict[str, list[object]] = {}
    for col, name in zip(param_cols, param_names):
        unique_vals = sorted(smoothed_df[col].dropna().unique())
        grid_structure[name] = unique_vals

    existing_points = {tuple(row) for row in smoothed_df[param_cols].values}

    n_neighbors_arr = np.empty(len(smoothed_df), dtype=int)
    param_values_arr = smoothed_df[param_cols].values

    for i, row_params in enumerate(param_values_arr):
        all_neighbors = identify_neighbors(tuple(row_params), grid_structure)
        n_neighbors_arr[i] = sum(1 for nb in all_neighbors if nb in existing_points)

    smoothed_df["n_neighbors"] = n_neighbors_arr

    return smoothed_df


@dataclass(frozen=True)
class ParameterSensitivityReport:
    """Structured output of parameter sensitivity smoothing and ranking."""

    param_names: list[str]
    metric_name: str
    stability_threshold: float
    metric_floor: float | None

    grid_results: pd.DataFrame
    stable_regions: Tuple[()]

    mean_stability_ratio: float
    median_stability_ratio: float
    pct_stable_combinations: float

    recommended_combinations: list[tuple[object, ...]]
    top_k_combinations: list[tuple[object, ...]]

    n_parameter_combinations: int
    n_stable_regions: int
    timestamp: str


def generate_parameter_sensitivity_report(
    results_df: pd.DataFrame,
    param_names: list[str],
    metric_col: str,
    stability_threshold: float = 0.8,
    top_k: int = 3,
    plot_3d_mode: str = "heatmap_slices",
    smoothing_self_weight: float = 2.0,
    metric_floor: float | None = 2.0,
) -> ParameterSensitivityReport:
    """
    Neighbor-smooth the objective grid and rank parameter combinations.

    Recommendations are ordered by the *raw* objective (peak realised performance);
    smoothing supplies stability diagnostics on the grid.
    """
    n_dims = len(param_names)

    smoothed_df = compute_neighbor_smoothing(
        results_df, param_names, metric_col, self_weight=smoothing_self_weight
    )
    sorted_raw = smoothed_df.sort_values(metric_col, ascending=False)
    pcols = _param_value_columns(n_dims)
    recommended = [tuple(row[c] for c in pcols) for _, row in sorted_raw.iterrows()]
    top_k_combos = recommended[:top_k]
    pct_stable = len(top_k_combos) / max(len(results_df), 1)

    ratios = smoothed_df["stability_ratio"].dropna()
    mean_ratio = float(ratios.mean()) if len(ratios) > 0 else 0.0
    median_ratio = float(ratios.median()) if len(ratios) > 0 else 0.0

    if n_dims >= 3 and plot_3d_mode not in {"heatmap_slices", "surface_slices"}:
        raise ValueError(
            f"Invalid plot_3d_mode: {plot_3d_mode}. "
            "Expected one of: ['heatmap_slices', 'surface_slices']."
        )

    return ParameterSensitivityReport(
        param_names=param_names,
        metric_name=metric_col,
        stability_threshold=stability_threshold,
        metric_floor=metric_floor,
        grid_results=smoothed_df,
        stable_regions=(),
        mean_stability_ratio=mean_ratio,
        median_stability_ratio=median_ratio,
        pct_stable_combinations=float(pct_stable),
        recommended_combinations=recommended,
        top_k_combinations=top_k_combos,
        n_parameter_combinations=len(smoothed_df),
        n_stable_regions=0,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
