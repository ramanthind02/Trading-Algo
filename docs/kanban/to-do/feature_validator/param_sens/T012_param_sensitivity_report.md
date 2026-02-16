# T012 — ParameterSensitivityReport Dataclass and Generation

## Goal
Define the `ParameterSensitivityReport` dataclass to encapsulate per-parameter raw/smoothed objectives, stability ratios, stable region identification, interactive plots, and recommendations for which parameter combinations are in stable regions, providing a structured output for Phase 3 of the Feature Validator pipeline.

## Context / References
- `docs/library/Feature_selection/feature_validator.md` (lines 246-254: ParameterSensitivityReport output)
- T009 — Grid-aware neighbor smoothing (provides smoothed metrics)
- T010 — Stable region identification (provides regions)
- T011 — Interactive visualizations (provides plots)
- Existing: `utils/models.py` — Dataclass patterns (frozen, Pydantic models)

## Scope
In scope:
- `ParameterSensitivityReport` dataclass with all Phase 3 outputs
- Report generation function that orchestrates T009, T010, T011 outputs
- Recommendations: Which parameter combos are in stable regions (ranked by mean objective)
- Support for both continuous and rule-based features

Out of scope:
- Neighbor smoothing implementation (T009)
- Stable region identification logic (T010)
- Visualization generation (T011)
- Permutation testing integration (Phase 4)

## Interfaces (must match)
- Add: `eda/parameter_analysis.py::ParameterSensitivityReport` dataclass:
  ```python
  @dataclass(frozen=True)
  class ParameterSensitivityReport:
      # Input metadata
      param_names: List[str]
      metric_name: str
      stability_threshold: float

      # Grid analysis results
      grid_results: pd.DataFrame  # All param combos with raw, smoothed, stability_ratio
      stable_regions: List[StableRegion]  # From T010

      # Summary statistics
      mean_stability_ratio: float
      median_stability_ratio: float
      pct_stable_combinations: float  # % with stability_ratio > threshold

      # Recommendations
      recommended_combinations: List[Tuple]  # Param combos in stable regions, ranked by smoothed objective
      top_k_combinations: List[Tuple]  # Top K by smoothed objective (default K=3)

      # Visualizations
      plot_1d: Optional[go.Figure] = None  # For 1D grids
      plot_2d: Optional[go.Figure] = None  # For 2D grids
      plot_3d: Optional[go.Figure] = None  # For 3D+ grids

      # Diagnostic info
      n_parameter_combinations: int
      n_stable_regions: int
      timestamp: str  # ISO format
  ```

- Add: `eda/parameter_analysis.py::generate_parameter_sensitivity_report(results_df: pd.DataFrame, param_names: List[str], metric_col: str, stability_threshold: float = 0.8, top_k: int = 3) -> ParameterSensitivityReport`
  - Input: DataFrame with grid search results (param values + metric), parameter names, metric column
  - Output: Fully populated `ParameterSensitivityReport` instance
  - Behavior: Orchestrate T009 (smoothing), T010 (region ID), T011 (plots), generate recommendations

## Data Contracts
- Input DataFrame schema:
  - Required columns: `param1_value`, ..., `paramN_value`, `{metric_col}`
  - Optional columns: `n_samples`, `feature` (for diagnostics)

- `ParameterSensitivityReport` schema:
  - `grid_results`: DataFrame with `param1_value`, ..., `paramN_value`, `{metric}`, `smoothed_{metric}`, `stability_ratio`, `n_neighbors`
  - `stable_regions`: List of `StableRegion` objects from T010
  - `recommended_combinations`: List of tuples `(p1, p2, ..., pN)` sorted by `smoothed_{metric}` descending, filtered to only include combos in stable regions
  - `top_k_combinations`: Top K parameter combos by smoothed objective (K=3 default)
  - `plot_*`: Plotly Figure objects (None if grid dimensionality doesn't match)

- Recommendations logic:
  - Filter: Only include param combos with `stability_ratio > stability_threshold`
  - Rank: Sort by `smoothed_{metric}` descending
  - Top-K: Select first K combos for ensemble formation

## Dependencies
- pandas (DataFrame operations)
- dataclasses (report model)
- typing (type hints)
- plotly (Figure objects)
- datetime (timestamp generation)
- T009, T010, T011 (orchestrated by report generation function)

## Invariants / Constraints
- Deterministic: Same inputs → same report (including plots, recommendations)
- Frozen dataclass: Report is immutable after creation
- No lookahead: All analysis based on in-sample data only
- Recommendations must be subset of stable region combos
- Top-K size: If fewer than K stable combos exist, return all available
- Timestamp: ISO 8601 format (e.g., "2026-02-15T10:30:45Z")

## Acceptance tests
1. `pytest tests/integration/feature_validator/param_sens/test_report_generation.py::test_1d_rsi_report` — 1D grid with RSI lookback=[2,3,4,5,6,7,8,9,10], verify report contains grid_results, stable_regions, recommendations, plot_1d
2. `pytest tests/integration/feature_validator/param_sens/test_report_2d.py::test_ewmac_2d_report` — 2D grid, verify `plot_2d` is populated and `recommended_combinations` are ranked by smoothed objective
3. `pytest tests/integration/feature_validator/param_sens/test_report_top_k.py::test_top_k_selection` — Grid with 10 stable combos, verify `top_k_combinations` contains exactly 3 when K=3
4. `pytest tests/integration/feature_validator/param_sens/test_report_determinism.py::test_report_reproducibility` — Fixed seed, identical inputs → identical report fields (excluding timestamp)

## Definition of done
- [ ] `ParameterSensitivityReport` dataclass added to `eda/parameter_analysis.py`
- [ ] `generate_parameter_sensitivity_report()` implemented with full type hints and docstrings
- [ ] Tests added under `tests/integration/feature_validator/param_sens/`
- [ ] All acceptance tests pass: `pytest tests/integration/feature_validator/param_sens/test_report*.py -v`
- [ ] Report serialization tested (can save/load as JSON or pickle)

## Notes
- Report generation workflow:
  1. Call T009: `compute_neighbor_smoothing(results_df, param_names, metric_col)` → smoothed_df
  2. Call T010: `identify_stable_regions(smoothed_df, metric_col, stability_threshold)` → stable_regions
  3. Call T011: Generate appropriate plot(s) based on grid dimensionality
  4. Generate recommendations: Filter to stable combos, rank by smoothed objective
  5. Compute summary statistics: mean/median stability ratio, % stable
  6. Package into `ParameterSensitivityReport` instance

- Recommendations example (from spec line 253):
  ```
  recommended_combinations = [
      (5, 32),   # RSI lookback=5, threshold=32 (smoothed Sharpe=1.2)
      (4, 32),   # RSI lookback=4, threshold=32 (smoothed Sharpe=1.15)
      (6, 32),   # RSI lookback=6, threshold=32 (smoothed Sharpe=1.10)
  ]
  ```
  All three are in the same stable region around (5, 32)

- Top-K selection for ensemble formation:
  - Default K=3 (spec recommendation: 2-10 parameter combinations)
  - High correlation within ensemble is expected (parameter smoothing, not redundancy)

- Summary statistics:
  - `mean_stability_ratio`: Overall stability across grid
  - `pct_stable_combinations`: Percentage of grid points in stable regions
  - These inform overall feature robustness (feed into Phase 4 walkforward stability)

- Plot assignment by dimensionality:
  - 1D grid (N=1): `plot_1d` populated, `plot_2d=None`, `plot_3d=None`
  - 2D grid (N=2): `plot_2d` populated, `plot_1d=None`, `plot_3d=None`
  - 3D+ grid (N≥3): `plot_3d` populated, `plot_1d=None`, `plot_2d=None`

- Serialization considerations:
  - Plotly figures can be saved as HTML (preserve interactivity)
  - DataFrames can be saved as parquet (preserve dtypes)
  - Full report can be pickled for Python-to-Python transfer

- Integration with Feature Validator:
  - Phase 3 output: `ParameterSensitivityReport`
  - Phase 4 input: Use `recommended_combinations` for permutation testing (Stage 1, 2)
  - Researcher decision: Review `stable_regions` and `plot_*` to select final ensemble members
