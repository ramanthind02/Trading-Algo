# T009 — Grid-Aware Neighbor Smoothing Algorithm

## Goal
Implement grid-aware neighbor smoothing to compute smoothed objective metrics for parameter combinations by averaging with 1-step axis-aligned neighbors in parameter grids (1D, 2D, 3D+), enabling identification of stable parameter regions versus isolated overfitted peaks.

## Context / References
- `docs/library/Feature_selection/feature_validator.md` (lines 182-255: Phase 3: Parameter Sensitivity Analysis)
- `docs/library/Feature_selection/stability/grid_search_parameter_stability.md` — Grid-aware neighbor smoothing theory
- `eda/parameter_analysis.py` — `ParameterAnalyzer` class (existing infrastructure)
- `metrics/plotting/parameter_plots.py` — Parameter visualization functions
- `docs/kanban/to-do/feature_validator/INTEGRATION_TESTING_SPEC.md` — Unit vs integration test standards

## Scope
In scope:
- Neighbor definition logic for 1D grids (single parameter: lookback=[2,3,4,5,6,7,8,9,10])
- Neighbor definition logic for 2D grids (two parameters: lookback x threshold)
- Neighbor definition logic for 3D+ grids (three or more parameters)
- Smoothed objective computation: `smoothed_objective(P) = mean([objective(P)] + [objective(N) for N in neighbors(P)])`
- Stability ratio computation: `stability_ratio = smoothed_objective / raw_objective`
- Support for both continuous features (via binning model) and rule-based features

Out of scope:
- Visualization (covered in T011)
- Report generation (covered in T012)
- Permutation testing integration

## Interfaces (must match)
- Add: `eda/parameter_analysis.py::compute_neighbor_smoothing(results_df: pd.DataFrame, param_names: List[str], metric_col: str) -> pd.DataFrame`
  - Input: DataFrame with `paramK_value` columns (K=1..N) and metric column (e.g., 'sortino')
  - Output: DataFrame with original columns plus `smoothed_{metric_col}` and `stability_ratio` columns
  - Behavior: For each row, find 1-step axis-aligned neighbors, compute mean of raw + neighbor metrics

- Add: `eda/parameter_analysis.py::identify_neighbors(param_values: Tuple, grid_structure: Dict[str, List]) -> List[Tuple]`
  - Input: Current parameter combination tuple, grid structure mapping param names to value lists
  - Output: List of neighbor parameter combination tuples (1-step axis-aligned only)
  - Behavior: For each parameter dimension, check if ±1 grid step exists; if so, add to neighbors list

## Data Contracts
- Input DataFrame schema:
  - Required columns: `param1_value`, `param2_value`, ..., `paramN_value` (1 ≤ N ≤ 4)
  - Required metric column: user-specified (e.g., 'sortino', 'sharpe', 'mean')
  - All parameter values must be sortable (numeric or string)

- Output DataFrame schema:
  - All input columns preserved
  - Added columns:
    - `smoothed_{metric_col}`: float (smoothed objective value)
    - `stability_ratio`: float (smoothed / raw, may be > 1.0 or < 1.0)
    - `n_neighbors`: int (count of neighbors found, for diagnostics)

- Neighbor definition:
  - 1D grid: For param value at index i in sorted list, neighbors are indices i-1 and i+1 (if they exist)
  - 2D grid: For (p1[i], p2[j]), neighbors are (p1[i±1], p2[j]) and (p1[i], p2[j±1]) (4 max)
  - 3D+ grid: Generalize to N dimensions (2N max neighbors)

## Dependencies
- pandas (DataFrame manipulation)
- numpy (numerical operations)
- typing (type hints for Protocol and generics)

## Invariants / Constraints
- Deterministic: same grid structure and metric values → same smoothed values
- Stability ratio bounds: Can be < 1.0 (neighbors are worse) or > 1.0 (neighbors are better)
- No lookahead: Only use neighbors defined by grid structure, not future test data
- Grid structure must be complete: All parameter combinations in results_df must be valid grid points
- Missing neighbors: If a parameter is at grid boundary, use available neighbors only (no extrapolation)

## Acceptance tests

**Unit tests:**
- `test_identify_neighbors_1d()` — synthetic 1D grid [10, 14, 20, 30]; verify lookback=14 has neighbors {10, 20}, lookback=10 (boundary) has only {14}
- `test_identify_neighbors_2d()` — synthetic 2D grid fast=[8,16,32] x slow=[32,64,128]; verify interior point (16,64) has 4 neighbors, corner point (8,32) has 2 neighbors, edge point (16,32) has 3 neighbors
- `test_identify_neighbors_3d()` — synthetic 3D grid; verify interior point has up to 6 neighbors, boundary points have fewer
- `test_compute_neighbor_smoothing_known_values()` — handcrafted 1D DataFrame with 5 points and explicit metric values; manually verify smoothed values match expected formula: `smoothed(P) = mean([raw(P)] + [raw(N) for N in neighbors(P)])`
- `test_stability_ratio_formula()` — verify stability_ratio = smoothed_metric / raw_metric for each row in synthetic DataFrame
- `test_determinism()` — identical inputs → identical smoothed values and stability ratios (no random state)
- `test_n_neighbors_column()` — verify `n_neighbors` column counts correctly: boundary=1, edge=2, interior=2 for 1D grid of size 3
- `test_single_point_grid()` — grid with exactly one parameter value; verify n_neighbors=0, smoothed=raw, stability_ratio=1.0
- Location: `tests/validators/param_sens/test_grid_neighbor_smoothing.py`

**Integration tests:**
- Covered by `tests/integration/feature_validator/test_parameter_sensitivity.py::test_parameter_sensitivity_pipeline()`
- Default config: RSI lookback grid [3, 4, 5, 10, 14, 20], ES daily, 2020-2023
- Customizable for any bias node/param grid (see INTEGRATION_TESTING_SPEC.md)
- Verifies: smoothed metrics computed for all grid points, stability ratios non-negative, `n_neighbors` correct for boundary vs interior points
- Cache policy: `USE_CACHE=True`; skip with message if cache missing: "Run CacheManager.populate_cache() first"
- Researcher manual verification: inspect terminal output showing per-param smoothed values, confirm boundary params (lookback=3, lookback=20) have fewer neighbors than interior params

## Definition of done
- [ ] Unit tests added under `tests/validators/param_sens/test_grid_neighbor_smoothing.py`
- [ ] Implementation in `eda/parameter_analysis.py` with type hints and docstrings
- [ ] All unit tests pass: `pytest tests/validators/param_sens/test_grid_neighbor_smoothing.py -v`
- [ ] Integration coverage provided by `tests/integration/feature_validator/test_parameter_sensitivity.py::test_parameter_sensitivity_pipeline()`
- [ ] Edge cases handled: grid boundaries, isolated points, missing neighbors

## Notes
- Grid structure inference: Infer from unique values in `paramK_value` columns, sorted
- Example (1D): RSI lookback=[10, 14, 20, 30] → For lookback=14, neighbors={10, 20}
- Example (2D): EWMAC fast=[8,16] slow=[32,64] → For (16,32), neighbors={(8,32), (16,64)}
- Stability ratio interpretation: High ratio (> 0.8) = stable region, low ratio (< 0.5) = isolated peak
- Reference formula from spec (line 194-196):
  ```
  smoothed_objective(P) = mean([objective(P)] + [objective(N) for N in neighbors(P)])
  stability_ratio = smoothed_objective / raw_objective
  ```
- T009 is a pure algorithm; unit tests are sufficient for correctness verification. Integration coverage is inherited from the full parameter sensitivity pipeline test (T012 scope).
