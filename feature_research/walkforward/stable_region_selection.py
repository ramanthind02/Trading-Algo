"""Stable Region Ensemble Selection.

Implements the landscape-aware parameter selection algorithm described in
docs/library/Feature_selection/Parameter_Sensitivity/top_k_ensemble_selection.md.

Algorithm summary
-----------------
1. Hard filters  — remove params below trade_freq_min or bin_count_min.
2. Relative floor — adaptive (sigma-based) or fixed-δ floor from surviving params.
3. Superlevel set — qualifying params are those above the floor.
4. Connected components — BFS over grid adjacency graph restricted to qualifying set.
5. Degenerate filter — discard components of size < min_region_size or all ≤ 0.
6. Select per region — top k_per_region by smoothed_objective; trim to k_max globally.

This module is self-contained: it takes pre-computed scores and frequencies and
has no dependency on the walkforward runner or config.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Mapping, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StableRegionConfig:
    """Pre-committed configuration for stable region parameter selection.

    All parameters must be fixed before any walkforward fold is evaluated.
    They are not tuned per feature or per fold.

    Attributes
    ----------
    floor_method : str
        ``"adaptive"`` (σ-based, recommended) or ``"relative"`` (fixed δ).
    delta : float
        Relative tolerance used when ``floor_method="relative"``.
        floor = best × (1 − δ).
    adaptive_sigma_multiplier : float
        Multiplier on σ when ``floor_method="adaptive"``.
        floor = best − adaptive_sigma_multiplier × σ.
    k_per_region : int
        Maximum params selected from each stable region.
    k_max : int
        Hard cap on total selected params across all regions.
    bin_count_min : int
        Hard filter for continuous features: params whose ``bin_count`` value
        is below this threshold are excluded before ranking.
        Set to 0 to disable.
    min_region_size : int
        Connected components smaller than this are treated as isolated spikes
        and discarded.

    Use ``StableRegionConfig.aggressive()`` for a stricter preset (smaller
    adaptive_sigma_multiplier, smaller k_max) when running selection experiments.
    """

    floor_method: str = "adaptive"
    delta: float = 0.20
    adaptive_sigma_multiplier: float = 1.0
    k_per_region: int = 3
    k_max: int = 6
    bin_count_min: int = 5
    min_region_size: int = 2

    def __post_init__(self) -> None:
        if self.floor_method not in ("adaptive", "relative"):
            raise ValueError(
                f"floor_method must be 'adaptive' or 'relative', got '{self.floor_method}'"
            )
        if not (0.0 < self.delta <= 1.0):
            raise ValueError("delta must be in (0, 1]")
        if self.adaptive_sigma_multiplier <= 0:
            raise ValueError("adaptive_sigma_multiplier must be > 0")
        if self.k_per_region < 1:
            raise ValueError("k_per_region must be >= 1")
        if self.k_max < 1:
            raise ValueError("k_max must be >= 1")
        if self.min_region_size < 1:
            raise ValueError("min_region_size must be >= 1")

    @classmethod
    def aggressive(cls) -> "StableRegionConfig":
        """Preset for more aggressive selection: smaller sigma, fewer params (for experiments)."""
        return cls(adaptive_sigma_multiplier=0.5, k_max=3, k_per_region=2, min_region_size=2)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StableRegionResult:
    """Output of ``run_stable_region_selection``.

    Attributes
    ----------
    selected_labels : list[str]
        Canonical param labels selected for the ensemble this fold.
        Empty when no valid stable region exists.
    per_param_detail : pd.DataFrame
        One row per param in the input grid.  Columns:

        ``param_label`` ``smoothed_objective`` ``raw_objective``
        ``trade_frequency`` ``passed_hard_filters`` ``above_floor``
        ``region_id`` (int or NaN) ``region_size`` (int or NaN) ``selected``
    """

    selected_labels: list[str]
    per_param_detail: pd.DataFrame = field(compare=False)


# ---------------------------------------------------------------------------
# Grid adjacency
# ---------------------------------------------------------------------------

def _canonical_param_label(params: dict[str, object]) -> str:
    return "|".join(f"{k}={params[k]}" for k in sorted(params))


def _build_grid_adjacency(
    param_grid: list[dict[str, object]],
) -> dict[str, list[str]]:
    """Build a 1-step axis-aligned adjacency map over the param grid.

    Two param combos are adjacent iff they differ in exactly one axis by
    exactly one unit (integer distance = 1).  Non-integer axes are ignored
    for adjacency purposes (two combos with identical non-integer values on
    all axes and differing only on a non-integer axis are *not* adjacent).

    Returns
    -------
    dict mapping canonical_label → list[canonical_label of neighbours]
    """
    labels = [_canonical_param_label(p) for p in param_grid]
    label_to_params: dict[str, dict[str, object]] = dict(zip(labels, param_grid))
    adjacency: dict[str, list[str]] = {label: [] for label in labels}

    axes = sorted({k for p in param_grid for k in p})

    for i, label_a in enumerate(labels):
        params_a = label_to_params[label_a]
        for label_b in labels[i + 1 :]:
            params_b = label_to_params[label_b]
            diffs = 0
            is_unit = False
            for ax in axes:
                val_a = params_a.get(ax)
                val_b = params_b.get(ax)
                if val_a != val_b:
                    diffs += 1
                    if diffs > 1:
                        break
                    if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                        is_unit = abs(float(val_a) - float(val_b)) == 1.0
                    else:
                        is_unit = False
            if diffs == 1 and is_unit:
                adjacency[label_a].append(label_b)
                adjacency[label_b].append(label_a)

    return adjacency


def _build_grid_adjacency_from_structure(
    param_grid: list[dict[str, object]],
    grid_structure: Mapping[str, list],
) -> dict[str, list[str]]:
    """Build 1-step axis-aligned adjacency from explicit grid structure.

    Two param combos are adjacent iff they differ in exactly one key present
    in grid_structure and the two values are consecutive in that key's
    ordered list (by index). Keys in param_grid not in grid_structure are
    ignored for adjacency.

    Returns
    -------
    dict mapping canonical_label → list[canonical_label of neighbours]
    """
    labels = [_canonical_param_label(p) for p in param_grid]
    label_to_params: dict[str, dict[str, object]] = dict(zip(labels, param_grid))
    adjacency: dict[str, list[str]] = {label: [] for label in labels}

    axes = [k for k in sorted(grid_structure.keys()) if grid_structure.get(k)]
    if not axes:
        return adjacency

    # Value -> index for each axis (for O(1) consecutive check)
    value_to_index: dict[str, dict[object, int]] = {}
    for ax in axes:
        values = list(grid_structure[ax])
        value_to_index[ax] = {v: i for i, v in enumerate(values)}

    for i, label_a in enumerate(labels):
        params_a = label_to_params[label_a]
        for label_b in labels[i + 1 :]:
            params_b = label_to_params[label_b]
            diff_axis: Optional[str] = None
            for ax in axes:
                val_a = params_a.get(ax)
                val_b = params_b.get(ax)
                if val_a != val_b:
                    if diff_axis is not None:
                        diff_axis = None
                        break
                    idx_a = value_to_index.get(ax, {}).get(val_a)
                    idx_b = value_to_index.get(ax, {}).get(val_b)
                    if idx_a is None or idx_b is None:
                        diff_axis = None
                        break
                    if abs(idx_a - idx_b) != 1:
                        diff_axis = None
                        break
                    diff_axis = ax
            if diff_axis is not None:
                adjacency[label_a].append(label_b)
                adjacency[label_b].append(label_a)

    return adjacency


# ---------------------------------------------------------------------------
# Connected components
# ---------------------------------------------------------------------------

def _find_connected_components(
    qualifying_labels: set[str],
    adjacency: dict[str, list[str]],
) -> dict[str, int]:
    """BFS connected components restricted to qualifying_labels.

    Returns
    -------
    dict mapping label → component_id (0-indexed integers)
    Params not in qualifying_labels are absent from the result.
    """
    label_to_component: dict[str, int] = {}
    component_id = 0
    remaining = set(qualifying_labels)

    while remaining:
        seed = next(iter(remaining))
        queue: deque[str] = deque([seed])
        visited: set[str] = {seed}
        while queue:
            current = queue.popleft()
            label_to_component[current] = component_id
            for neighbour in adjacency.get(current, []):
                if neighbour in remaining and neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
        remaining -= visited
        component_id += 1

    return label_to_component


# ---------------------------------------------------------------------------
# Main selection function
# ---------------------------------------------------------------------------

def run_stable_region_selection(
    smoothed_objectives: Mapping[str, float],
    raw_objectives: Mapping[str, float],
    trade_frequencies: Mapping[str, float],
    param_grid: list[dict[str, object]],
    config: StableRegionConfig,
    grid_structure: Optional[Mapping[str, list]] = None,
    strategy: Optional[str] = None,
    objective_metric_name: str = "",
) -> StableRegionResult:
    """Select param combos from stable regions of the smoothed performance landscape.

    Parameters
    ----------
    smoothed_objectives : Mapping[str, float]
        Label → neighbour-smoothed metric value (from grid-aware smoothing).
    raw_objectives : Mapping[str, float]
        Label → raw (unsmoothed) training metric value.
    trade_frequencies : Mapping[str, float]
        Label → fraction of periods where signal ≠ 0.
    param_grid : list[dict]
        All param combinations evaluated this fold.
    config : StableRegionConfig
        Pre-committed algorithm configuration.
    grid_structure : Optional[Mapping[str, list]], optional
        If provided, param name → ordered list of values per axis. Adjacency
        is then "one grid step" (consecutive index in that list). If None,
        adjacency uses value difference == 1 (unit-step) for backward compatibility.
    strategy : str, optional
        When "long" and objective_metric_name is "t_stat", only params with
        smoothed_objective > 0 are considered (long-only filter).
    objective_metric_name : str, optional
        Used with strategy for long-only t_stat filter.

    Returns
    -------
    StableRegionResult
        Selected labels and per-param diagnostic detail.
    """
    all_labels = [_canonical_param_label(p) for p in param_grid]
    if grid_structure is not None:
        adjacency = _build_grid_adjacency_from_structure(param_grid, grid_structure)
    else:
        adjacency = _build_grid_adjacency(param_grid)

    # -----------------------------------------------------------------------
    # Step 1: Hard filters
    # -----------------------------------------------------------------------
    passed_hard: dict[str, bool] = {}
    for label, params in zip(all_labels, param_grid):
        tf = trade_frequencies.get(label, 0.0)
        fails_tf = tf < 0.0  # trade_freq_min is handled by the caller (config.trade_freq_min
        #                       is not stored here since it's in WalkforwardResearchConfig)
        #                       We just record what was passed in.
        bin_count = params.get("bin_count")
        fails_bin = (
            config.bin_count_min > 0
            and bin_count is not None
            and isinstance(bin_count, (int, float))
            and int(bin_count) < config.bin_count_min
        )
        passed_hard[label] = not (fails_tf or fails_bin)

    surviving = [label for label in all_labels if passed_hard[label]]
    # Long-only + t_stat: exclude params with non-positive objective
    if strategy == "long" and objective_metric_name == "t_stat":
        surviving = [
            label for label in surviving
            if smoothed_objectives.get(label, float("-inf")) > 0
        ]

    # -----------------------------------------------------------------------
    # Step 2: Relative floor computation (from surviving only)
    # -----------------------------------------------------------------------
    if not surviving:
        return _empty_result(all_labels, smoothed_objectives, raw_objectives, trade_frequencies, passed_hard)

    surviving_scores = [smoothed_objectives.get(lbl, float("-inf")) for lbl in surviving]
    best = max(surviving_scores)

    if config.floor_method == "relative":
        floor = best * (1.0 - config.delta)
    else:
        # adaptive: floor = best − multiplier × σ
        import statistics
        if len(surviving_scores) >= 2:
            sigma = statistics.stdev(surviving_scores)
        else:
            sigma = 0.0
        floor = best - config.adaptive_sigma_multiplier * sigma

    # -----------------------------------------------------------------------
    # Step 3: Superlevel set
    # -----------------------------------------------------------------------
    above_floor: dict[str, bool] = {
        label: (smoothed_objectives.get(label, float("-inf")) > floor)
        for label in all_labels
    }
    qualifying_labels: set[str] = {
        label for label in surviving if above_floor.get(label, False)
    }

    # -----------------------------------------------------------------------
    # Step 4: Connected components
    # -----------------------------------------------------------------------
    if not qualifying_labels:
        return _empty_result(
            all_labels, smoothed_objectives, raw_objectives, trade_frequencies,
            passed_hard, above_floor=above_floor, floor=floor,
        )

    label_to_component = _find_connected_components(qualifying_labels, adjacency)
    component_sizes: dict[int, int] = {}
    for comp_id in label_to_component.values():
        component_sizes[comp_id] = component_sizes.get(comp_id, 0) + 1

    # -----------------------------------------------------------------------
    # Step 5: Filter degenerate components
    # -----------------------------------------------------------------------
    valid_components: set[int] = set()
    for comp_id, size in component_sizes.items():
        if size < config.min_region_size:
            continue
        members = [lbl for lbl, cid in label_to_component.items() if cid == comp_id]
        if any(smoothed_objectives.get(m, float("-inf")) > 0 for m in members):
            valid_components.add(comp_id)

    # -----------------------------------------------------------------------
    # Step 6: Select from each valid region
    # -----------------------------------------------------------------------
    selected_labels: list[str] = []

    for comp_id in sorted(valid_components):
        members = [
            lbl for lbl, cid in label_to_component.items()
            if cid == comp_id
        ]
        members_ranked = sorted(
            members,
            key=lambda lbl: (-smoothed_objectives.get(lbl, float("-inf")), lbl),
        )
        selected_labels.extend(members_ranked[: config.k_per_region])

    # Global cap
    if len(selected_labels) > config.k_max:
        selected_labels = sorted(
            selected_labels,
            key=lambda lbl: (-smoothed_objectives.get(lbl, float("-inf")), lbl),
        )[: config.k_max]

    selected_set: set[str] = set(selected_labels)

    # -----------------------------------------------------------------------
    # Build per-param detail DataFrame
    # -----------------------------------------------------------------------
    rows = []
    for label in all_labels:
        comp_id = label_to_component.get(label)
        rows.append(
            {
                "param_label": label,
                "smoothed_objective": smoothed_objectives.get(label, float("nan")),
                "raw_objective": raw_objectives.get(label, float("nan")),
                "trade_frequency": trade_frequencies.get(label, float("nan")),
                "passed_hard_filters": passed_hard.get(label, False),
                "above_floor": above_floor.get(label, False),
                "region_id": comp_id,
                "region_size": component_sizes.get(comp_id) if comp_id is not None else None,
                "selected": label in selected_set,
            }
        )

    return StableRegionResult(
        selected_labels=selected_labels,
        per_param_detail=pd.DataFrame(rows),
    )


# ---------------------------------------------------------------------------
# Helper: empty result
# ---------------------------------------------------------------------------

def _empty_result(
    all_labels: list[str],
    smoothed_objectives: Mapping[str, float],
    raw_objectives: Mapping[str, float],
    trade_frequencies: Mapping[str, float],
    passed_hard: dict[str, bool],
    above_floor: dict[str, bool] | None = None,
    floor: float | None = None,
) -> StableRegionResult:
    rows = [
        {
            "param_label": lbl,
            "smoothed_objective": smoothed_objectives.get(lbl, float("nan")),
            "raw_objective": raw_objectives.get(lbl, float("nan")),
            "trade_frequency": trade_frequencies.get(lbl, float("nan")),
            "passed_hard_filters": passed_hard.get(lbl, False),
            "above_floor": (above_floor or {}).get(lbl, False),
            "region_id": None,
            "region_size": None,
            "selected": False,
        }
        for lbl in all_labels
    ]
    return StableRegionResult(
        selected_labels=[],
        per_param_detail=pd.DataFrame(rows),
    )
