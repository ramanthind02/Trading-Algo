"""Marginal Peak Selection (MPS): select params by largest peak-vs-second gap in marginal tables."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping

import pandas as pd


def _canonical_param_label(params: dict[str, object]) -> str:
    """Same convention as runner and stable_region_selection."""
    return "|".join(f"{key}={params[key]}" for key in sorted(params))


def _varying_dimensions(param_grid: list[dict[str, object]]) -> list[str]:
    """Return param keys that have more than one distinct value in the grid.

    Dimensions with a single value (e.g. selected_bin when bin_index_min == bin_index_max)
    are excluded so fixed params do not affect marginal table construction.
    """
    if not param_grid:
        return []
    keys = list(param_grid[0].keys())
    return sorted(
        d for d in keys if len({p[d] for p in param_grid}) > 1
    )


def _filter_param_grid_by_ranges(
    param_grid: list[dict[str, object]],
    raw_objectives: Mapping[str, float],
    dimension_ranges: Mapping[str, tuple[int, int]] | None,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    """Restrict param_grid and raw_objectives to combos satisfying dimension_ranges.

    Each range is (min_incl, max_incl). When dimension_ranges is None, returns
    (param_grid, dict(raw_objectives)) unchanged.
    """
    if not dimension_ranges:
        return (param_grid, dict(raw_objectives))
    labels = [_canonical_param_label(p) for p in param_grid]

    def in_range(params: dict[str, object]) -> bool:
        for dim, (lo, hi) in dimension_ranges.items():
            if dim not in params:
                return False
            val = params[dim]
            if not isinstance(val, (int, float)):
                return False
            if int(val) < lo or int(val) > hi:
                return False
        return True

    filtered_grid = [p for p in param_grid if in_range(p)]
    filtered_labels = {_canonical_param_label(p) for p in filtered_grid}
    filtered_objectives = {
        lbl: raw_objectives[lbl]
        for lbl in filtered_labels
        if lbl in raw_objectives
    }
    return (filtered_grid, filtered_objectives)


@dataclass(frozen=True)
class MarginalPeakConfig:
    """Config for MPS. Typically marginal_dim is 2 (pairwise) or 3 (3D tables)."""

    k_max: int = 5
    min_gap: float = 0.10
    min_cell_size: int = 2
    fallback_k: int = 5
    marginal_dim: int = 2  # dimensions per marginal table: 2 = C(D,2) pairwise, 3 = C(D,3)
    dimension_ranges: Mapping[str, tuple[int, int]] | None = None  # e.g. {"selected_bin": (5, 8)}; only these combos used for marginals/means


@dataclass(frozen=True)
class MarginalPeakResult:
    selected_labels: list[str]
    per_param_detail: pd.DataFrame  # param_label, raw_objective, cell_id, cell_mean, in_peak_cell, selected
    selected_pair: tuple[str, ...] | None  # selected dimension names (length = marginal_dim)
    peak_cell: tuple[object, ...] | None  # peak cell values (same length)
    gap: float
    fallback_used: bool


def compute_pairwise_marginal_tables(
    param_grid: list[dict[str, object]],
    raw_objectives: Mapping[str, float],
    min_cell_size: int = 2,
    marginal_dim: int = 2,
) -> dict[tuple[str, ...], pd.DataFrame]:
    """Compute marginal tables over dimension combinations.

    For each combination of marginal_dim dimensions, groups param combos by that key,
    computes cell_mean = mean(raw_objective) per group, and returns a DataFrame per combo.
    Only cells with at least min_cell_size combos are included.
    marginal_dim=2 yields C(D,2) pairwise tables; marginal_dim=3 yields C(D,3) 3D tables (if D >= 3).

    Dimensions with a single value in the grid are excluded (use _varying_dimensions).
    Caller should pass pre-filtered param_grid/raw_objectives if dimension_ranges apply.

    Returns
    -------
    dict mapping tuple of dim names -> DataFrame with columns [dim_1, ..., "cell_mean", "cell_size"].
    """
    if not param_grid:
        return {}

    labels = [_canonical_param_label(p) for p in param_grid]
    dimensions = _varying_dimensions(param_grid)
    if len(dimensions) < marginal_dim:
        return {}

    result: dict[tuple[str, ...], pd.DataFrame] = {}
    for param_combo in combinations(dimensions, marginal_dim):
        dims = tuple(param_combo)
        group_to_labels: dict[tuple[object, ...], list[str]] = {}
        for params, label in zip(param_grid, labels):
            key = tuple(params[d] for d in dims)
            group_to_labels.setdefault(key, []).append(label)

        rows = [
            {**{d: key[i] for i, d in enumerate(dims)}, "cell_mean": sum(raw_objectives.get(lbl, float("nan")) for lbl in lbls) / len(lbls), "cell_size": len(lbls)}
            for key, lbls in group_to_labels.items()
            if len(lbls) >= min_cell_size
        ]
        if rows:
            result[dims] = pd.DataFrame(rows)
    return result


def run_marginal_peak_selection(
    raw_objectives: Mapping[str, float],
    param_grid: list[dict[str, object]],
    config: MarginalPeakConfig,
    strategy: str | None = None,
    objective_metric_name: str = "",
) -> MarginalPeakResult:
    """Run MPS: pick the marginal table with largest gap, restrict to peak cell, return top k_max by raw."""
    dimension_ranges = getattr(config, "dimension_ranges", None)
    param_grid, raw_objectives = _filter_param_grid_by_ranges(
        param_grid, raw_objectives, dimension_ranges
    )
    labels = [_canonical_param_label(p) for p in param_grid]

    tables = compute_pairwise_marginal_tables(
        param_grid,
        raw_objectives,
        min_cell_size=config.min_cell_size,
        marginal_dim=config.marginal_dim,
    )

    best_dims: tuple[str, ...] = ()
    best_gap: float = 0.0
    best_peak_cell: tuple[object, ...] | None = None
    best_table: pd.DataFrame | None = None

    for dims, df in tables.items():
        if df.empty or len(df) < 2:
            gap = 0.0
            peak_cell = tuple(df.iloc[0][list(dims)].tolist()) if len(df) == 1 else None
        else:
            means = df["cell_mean"].values
            sorted_means = sorted(means, reverse=True)
            gap = float(sorted_means[0] - sorted_means[1])
            peak_row = df.loc[df["cell_mean"].idxmax()]
            peak_cell = tuple(peak_row[d] for d in dims)
        if gap > best_gap:
            best_gap = gap
            best_dims = dims
            best_peak_cell = peak_cell
            best_table = df

    fallback_used = (
        best_gap < config.min_gap or not best_dims or best_peak_cell is None
    )

    n_dim = config.marginal_dim
    empty_cell = (None,) * n_dim

    if fallback_used:
        selected_labels = sorted(
            labels,
            key=lambda lbl: (-(raw_objectives.get(lbl, float("-inf")) or 0.0), lbl),
        )[: config.fallback_k]
        selected_set = set(selected_labels)
        peak_cell_for_detail = best_peak_cell if best_peak_cell is not None else empty_cell
        dims_for_detail = best_dims if best_dims else ()
        table_for_cell_mean = (
            best_table if best_table is not None else pd.DataFrame()
        )
    else:
        dims_for_detail = best_dims
        peak_cell_for_detail = best_peak_cell
        table_for_cell_mean = (
            best_table if best_table is not None else pd.DataFrame()
        )
        in_peak = [
            tuple(params[d] for d in dims_for_detail) == peak_cell_for_detail
            for params in param_grid
        ]
        peak_labels = [lbl for lbl, ok in zip(labels, in_peak) if ok]
        selected_labels = sorted(
            peak_labels,
            key=lambda lbl: (-(raw_objectives.get(lbl, float("-inf")) or 0.0), lbl),
        )[: config.k_max]
        selected_set = set(selected_labels)

    def cell_mean_for_label(lbl: str) -> float:
        params = next((p for p, l in zip(param_grid, labels) if l == lbl), {})
        if table_for_cell_mean.empty or not dims_for_detail:
            return float("nan")
        if not all(d in params for d in dims_for_detail):
            return float("nan")
        key = tuple(params[d] for d in dims_for_detail)
        mask = table_for_cell_mean[dims_for_detail[0]] == key[0]
        for i in range(1, len(dims_for_detail)):
            mask = mask & (table_for_cell_mean[dims_for_detail[i]] == key[i])
        row = table_for_cell_mean[mask]
        if row.empty:
            return float("nan")
        return float(row["cell_mean"].iloc[0])

    def cell_id_for_label(lbl: str) -> tuple[object, ...]:
        params = next((p for p, l in zip(param_grid, labels) if l == lbl), {})
        if dims_for_detail and all(d in params for d in dims_for_detail):
            return tuple(params[d] for d in dims_for_detail)
        return empty_cell

    detail_rows = [
        {
            "param_label": lbl,
            "raw_objective": raw_objectives.get(lbl, float("nan")),
            "cell_id": cell_id_for_label(lbl),
            "cell_mean": cell_mean_for_label(lbl),
            "in_peak_cell": (not fallback_used)
            and (cell_id_for_label(lbl) == peak_cell_for_detail),
            "selected": lbl in selected_set,
        }
        for lbl in labels
    ]
    per_param_detail = pd.DataFrame(detail_rows)

    return MarginalPeakResult(
        selected_labels=selected_labels,
        per_param_detail=per_param_detail,
        selected_pair=best_dims if not fallback_used else None,
        peak_cell=best_peak_cell if not fallback_used else None,
        gap=best_gap if not fallback_used else 0.0,
        fallback_used=fallback_used,
    )
