# T009 — Grid-Aware Neighbor Smoothing Algorithm

## Goal
Implement grid-aware neighbor smoothing to compute smoothed objective metrics for parameter combinations by averaging with 1-step axis-aligned neighbors in parameter grids (1D, 2D, 3D+), enabling identification of stable parameter regions versus isolated overfitted peaks.

## Context / References
- `docs/library/Feature_selection/feature_validator.md` (lines 182-255: Phase 3: Parameter Sensitivity Analysis)
- `docs/library/Feature_selection/stability/grid_search_parameter_stability.md` — Grid-aware neighbor smoothing theory
- `eda/parameter_analysis.py` — `ParameterAnalyzer` class (existing infrastructure)
- `metrics/plotting/parameter_plots.py` — Parameter visualization functions

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
1. `pytest tests/integration/feature_validator/param_sens/test_1d_neighbor_smoothing.py::test_rsi_lookback_smoothing` — 1D grid with RSI lookback=[2,3,4,5,6,7,8,9,10], verify edge cases (lookback=2 has 1 neighbor, lookback=5 has 2 neighbors)
2. `pytest tests/integration/feature_validator/param_sens/test_2d_neighbor_smoothing.py::test_ewmac_fast_slow_grid` — 2D grid with fast=[8,16,32,64] slow=[32,64,128,256], verify corner/edge/interior neighbors
3. `pytest tests/integration/feature_validator/param_sens/test_nd_neighbor_smoothing.py::test_3d_grid_stability_ratio` — 3D grid, verify stability ratio = smoothed/raw for all points
4. `pytest tests/integration/feature_validator/param_sens/test_neighbor_determinism.py` — Fixed seed, identical inputs → identical smoothed values and stability ratios

## Definition of done
- [ ] Tests added under `tests/integration/feature_validator/param_sens/`
- [ ] Implementation in `eda/parameter_analysis.py` with type hints and docstrings
- [ ] All acceptance tests pass: `pytest tests/integration/feature_validator/param_sens/test_*neighbor*.py -v`
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
