# T010 — Stability Ratio Computation and Stable Region Identification

## Goal
Compute stability ratios for parameter combinations and identify contiguous stable regions (stability ratio > threshold) to distinguish genuine signal (broad stable regions) from overfitting (isolated parameter peaks).

## Context / References
- `docs/library/Feature_selection/feature_validator.md` (lines 182-255: Phase 3: Parameter Sensitivity Analysis)
- `docs/library/Feature_selection/stability/grid_search_parameter_stability.md` — Grid search parameter stability theory
- `eda/parameter_analysis.py` — `ParameterAnalyzer.compute_robustness_metrics()` (existing reference)
- T009 — Grid-aware neighbor smoothing (dependency)
- `docs/kanban/to-do/feature_validator/INTEGRATION_TESTING_SPEC.md` — Unit vs integration test standards

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

**Unit tests:**
- `test_identify_stable_regions_1d_contiguous()` — synthetic 1D DataFrame with manually set stability_ratios=[0.6, 0.85, 0.9, 0.88, 0.7]; verify region [1,2,3] (indices) is identified with threshold=0.8
- `test_identify_stable_regions_1d_two_regions()` — synthetic 1D DataFrame with two separate stable sub-sequences; verify both regions identified and non-overlapping
- `test_identify_stable_regions_2d_plateau()` — synthetic 2D 4x4 DataFrame with stable plateau in center cells; verify BFS/DFS finds correct connected component, boundaries reported correctly
- `test_boundary_region_flag()` — 1D grid where stable region includes first or last param value; verify `is_boundary_region=True`
- `test_interior_region_flag()` — 1D grid where stable region does not touch boundary; verify `is_boundary_region=False`
- `test_minimum_region_size_enforced()` — synthetic DataFrame with one isolated stable point (n_combinations=1); verify it is excluded from results
- `test_region_metadata()` — synthetic DataFrame with known values; verify `mean_stability_ratio`, `mean_objective`, `min_objective`, `max_objective`, `n_combinations` computed correctly
- `test_no_stable_regions()` — all stability_ratios below threshold; verify empty list returned
- `test_determinism()` — identical inputs and threshold → identical list of StableRegion objects
- Location: `tests/validators/param_sens/test_stability_ratio_computation.py`

**Integration tests:**
- Covered by `tests/integration/feature_validator/test_parameter_sensitivity.py::test_parameter_sensitivity_pipeline()`
- Default config: RSI lookback grid [3, 4, 5, 10, 14, 20], ES daily, 2020-2023
- Customizable for any bias node/param grid (see INTEGRATION_TESTING_SPEC.md)
- Verifies: at least one stable region identified, region metadata fields all populated, no overlapping regions
- Cache policy: `USE_CACHE=True`; skip with message if cache missing: "Run CacheManager.populate_cache() first"
- Researcher manual verification: inspect terminal output showing stable region parameter ranges and mean stability ratios; confirm stable regions are contiguous and make intuitive sense for RSI lookback (e.g., medium lookbacks 5-14 expected to cluster)

## Definition of done
- [ ] Unit tests added under `tests/validators/param_sens/test_stability_ratio_computation.py`
- [ ] `StableRegion` dataclass added to `eda/parameter_analysis.py`
- [ ] `identify_stable_regions()` implemented with full type hints and docstrings
- [ ] All unit tests pass: `pytest tests/validators/param_sens/test_stability_ratio_computation.py -v`
- [ ] Integration coverage provided by `tests/integration/feature_validator/test_parameter_sensitivity.py::test_parameter_sensitivity_pipeline()`
- [ ] Minimum region size enforced (>= 2 combinations)

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

- T010 is a pure algorithm; unit tests are sufficient for correctness verification. Integration coverage is inherited from the full parameter sensitivity pipeline test (T012 scope).
