# T010 — Stability Ratio Computation and Stable Region Identification

## Goal
Compute stability ratios for parameter combinations and identify contiguous stable regions (stability ratio > threshold) to distinguish genuine signal (broad stable regions) from overfitting (isolated parameter peaks).

## Context / References
- `docs/library/Feature_selection/feature_validator.md` (lines 182-255: Phase 3: Parameter Sensitivity Analysis)
- `docs/library/Feature_selection/stability/grid_search_parameter_stability.md` — Grid search parameter stability theory
- `eda/parameter_analysis.py` — `ParameterAnalyzer.compute_robustness_metrics()` (existing reference)
- T009 — Grid-aware neighbor smoothing (dependency)

## Scope
In scope:
- Stability ratio threshold configuration (default: 0.8 for stable regions)
- Stable region identification: Contiguous parameter combinations exceeding stability ratio threshold
- Region metadata: parameter ranges, mean stability ratio, mean objective, sample counts
- Support for both continuous and rule-based features

Out of scope:
- Neighbor smoothing computation (covered in T009)
- Visualization of stable regions (covered in T011)
- Permutation testing integration (Phase 4)

## Interfaces (must match)
- Add: `eda/parameter_analysis.py::identify_stable_regions(smoothed_df: pd.DataFrame, metric_col: str, stability_threshold: float = 0.8) -> List[StableRegion]`
  - Input: DataFrame with `paramK_value`, `smoothed_{metric_col}`, `stability_ratio` columns
  - Output: List of `StableRegion` dataclass instances
  - Behavior: Identify contiguous parameter combinations where `stability_ratio > stability_threshold`

- Add: `eda/parameter_analysis.py::StableRegion` dataclass:
  ```python
  @dataclass(frozen=True)
  class StableRegion:
      param_ranges: Dict[str, Tuple[Any, Any]]  # param_name -> (min_val, max_val)
      param_combinations: List[Tuple]  # All param combos in region
      mean_stability_ratio: float
      mean_objective: float
      min_objective: float
      max_objective: float
      n_combinations: int
      is_boundary_region: bool  # Touches grid boundary
  ```

## Data Contracts
- Input DataFrame schema (from T009):
  - Required columns: `param1_value`, `param2_value`, ..., `paramN_value`
  - Required columns: `smoothed_{metric_col}`, `stability_ratio`, `n_neighbors`
  - All parameter values must be sortable

- Output `StableRegion` schema:
  - `param_ranges`: For each parameter, store (min, max) values in region
  - `param_combinations`: List of all (p1, p2, ..., pN) tuples in region
  - `mean_stability_ratio`: Average stability ratio across region
  - `mean_objective`: Average smoothed objective across region
  - `is_boundary_region`: True if any param combination touches grid boundary

- Contiguity definition:
  - 1D: Consecutive indices in sorted parameter list
  - 2D+: 4-connected (Manhattan distance = 1) or 8-connected (Chebyshev distance = 1)
  - Use 4-connected (axis-aligned) for consistency with neighbor definition in T009

## Dependencies
- pandas (DataFrame operations)
- numpy (aggregation)
- dataclasses (StableRegion model)
- typing (type hints)

## Invariants / Constraints
- Deterministic: same smoothed_df and threshold → same stable regions identified
- No overlapping regions: Each parameter combination belongs to at most one stable region
- Stability ratio bounds: threshold must be in (0, 2.0), default 0.8 per spec (line 211-216)
- Minimum region size: At least 2 parameter combinations (reject isolated points)
- Threshold interpretation: stability_ratio > 0.8 (high) = stable, < 0.5 (low) = isolated peak

## Acceptance tests
1. `pytest tests/integration/feature_validator/param_sens/test_stable_region_1d.py::test_rsi_stable_region` — 1D grid with RSI lookback=[2,3,4,5,6,7,8,9,10], manually set stability ratios, verify contiguous region [4,5,6,7] identified
2. `pytest tests/integration/feature_validator/param_sens/test_stable_region_2d.py::test_ewmac_stable_plateau` — 2D grid with stable plateau in center, verify region boundaries and mean metrics
3. `pytest tests/integration/feature_validator/param_sens/test_stable_region_boundary.py::test_boundary_detection` — Verify `is_boundary_region=True` when region touches grid edge
4. `pytest tests/integration/feature_validator/param_sens/test_multiple_stable_regions.py::test_two_separate_regions` — Grid with two separate stable regions, verify both identified and non-overlapping

## Definition of done
- [ ] Tests added under `tests/integration/feature_validator/param_sens/`
- [ ] `StableRegion` dataclass added to `eda/parameter_analysis.py`
- [ ] `identify_stable_regions()` implemented with full type hints and docstrings
- [ ] All acceptance tests pass: `pytest tests/integration/feature_validator/param_sens/test_stable_region*.py -v`
- [ ] Minimum region size enforced (≥ 2 combinations)

## Notes
- Contiguity algorithm (1D): Simple consecutive scan
  ```python
  # Example: stability_ratios = [0.6, 0.85, 0.9, 0.88, 0.7, 0.82, 0.79]
  # With threshold=0.8: stable region = [1,2,3] (indices), param_values=[p2,p3,p4]
  ```

- Contiguity algorithm (2D+): Use graph search (BFS/DFS) to find connected components
  - Build adjacency: params are neighbors if differ by 1 grid step in one dimension
  - Filter: only include params with stability_ratio > threshold
  - Connected components = stable regions

- Boundary detection: Check if any param value equals min/max in grid structure

- Stability ratio interpretation (from spec line 211-216):
  - High ratio (> 0.8): Parameter is in a stable region
  - Low ratio (< 0.5): Isolated peak, likely overfit

- Region size heuristic: Larger regions (more param combos) indicate robustness
