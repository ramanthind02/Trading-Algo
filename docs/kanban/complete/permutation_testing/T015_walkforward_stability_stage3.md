# T015 — Stage 3: Walkforward Stability Analysis

## Goal
Implement Stage 3 temporal stability analysis that assesses whether the same parameter region is consistently selected as top-K across non-overlapping walkforward folds, distinguishing stable features (genuine signal) from unstable features (noise or regime-dependent).

## Context / References
- `docs/library/Feature_selection/feature_validator.md` — Phase 4: Permutation Testing (lines 318-352)
- `docs/library/Feature_selection/Permutation Testing/in-sample_pt.md` — §3 Walkforward Stability Analysis (lines 82-169)
- `docs/library/Feature_selection/stability/grid_search_parameter_stability.md` — Grid-aware neighbor smoothing theory
- `eda/parameter_analysis.py` — `ParameterAnalyzer` class for neighbor smoothing
- `docs/kanban/to-do/feature_validator/INTEGRATION_TESTING_SPEC.md` — Unit vs integration test standards, default config, cache policy

## Scope
In scope:
- Divide in-sample data into non-overlapping walkforward folds (e.g., 2-year chunks: 2015-2016, 2017-2018, 2019-2020, 2021-2022)
- For each fold independently:
  - Evaluate ALL parameter combinations (including those that failed Stage 1-2, needed for neighbor smoothing)
  - Continuous features: Fit binning pipeline on fold data, evaluate on same fold
  - Rule-based features: Compute rule output on fold data, evaluate (no fitting)
  - Compute smoothed neighbor metric per param: `smoothed_obj(P) = mean([obj(P)] + [obj(N) for N in neighbors(P)])`
  - Select top-K param combos by smoothed objective (default K=3)
- Compare top-K selections across folds to assess consistency
- Overlay: which top-K params also passed permutation tests (Stage 1-2)
- Generate `WalkforwardStabilityReport` with per-fold top-K, consistency metrics, stability verdict

Out of scope:
- Permutation testing (Stages 1-2) — already implemented (T013, T014)
- Researcher ensemble formation — manual decision using stability report (T016)
- Out-of-sample validation — separate pipeline after ensemble locked

## Interfaces (must match)
- Add: `feature_selection/validation/stability_analysis.py::run_walkforward_stability()`
  - Signature: `run_walkforward_stability(candles_df: pd.DataFrame, feature_spec: Dict, target: pd.Series, objective_func: Callable, param_grid: List[Dict], fold_structure: List[Tuple[pd.Timestamp, pd.Timestamp]], top_k: int = 3, permutation_passers: Optional[Set[str]] = None) -> WalkforwardStabilityReport`
  - `feature_spec`: Either bias_node_spec (continuous) or rule_spec (rule-based)
  - `param_grid`: List of parameter combinations to evaluate (all combos, not just permutation passers)
  - `fold_structure`: List of (start, end) datetime tuples defining non-overlapping folds
  - `permutation_passers`: Optional set of param combo names that passed Stage 1-2 (for overlay in report)

- Add: `feature_selection/validation/reports.py::WalkforwardStabilityReport`
  - Fields: `feature_name: str`, `feature_type: Literal['continuous', 'rule_based']`, `fold_results: List[FoldResult]`, `consistency_metrics: Dict[str, float]`, `is_stable: bool`, `stability_verdict: str`, `top_k: int`
  - Nested: `FoldResult` dataclass with `fold_id: str`, `fold_period: Tuple[pd.Timestamp, pd.Timestamp]`, `top_k_params: List[str]`, `smoothed_objectives: Dict[str, float]`, `passed_permutation_overlay: List[bool]`

- Modify: `eda/parameter_analysis.py::ParameterAnalyzer`
  - Add method: `compute_neighbor_smoothed_grid(param_results: pd.DataFrame, param_cols: List[str]) -> pd.DataFrame`
  - Returns DataFrame with additional column `smoothed_objective` computed via 1-step axis-aligned neighbor averaging
  - Neighbor definition: differ in one parameter dimension by one grid step (ordered grid)

## Data Contracts
- **Fold structure**: Non-overlapping time windows (e.g., 2-year chunks), user-defined before analysis
- **Input param_grid**: ALL parameter combinations (not just permutation passers) — needed for neighbor smoothing
- **Neighbor smoothing formula**: `smoothed_obj(P) = mean([obj(P)] + [obj(N) for N in 1-step_neighbors(P)])`
- **1-step neighbors**: Axis-aligned, differ in one parameter by one grid step (e.g., RSI lookback [2,3,4,5] → neighbors of 3 are {2, 4})
- **Consistency metrics**:
  - Overlap rate: fraction of top-K params shared between consecutive folds
  - Median parameter distance: median grid distance between top-K centers across folds
  - Stability flag: True if params cluster in consistent region (e.g., 3+ of 4 folds select params within 3-step neighborhood)

## Dependencies
- `eda/parameter_analysis.py` — `ParameterAnalyzer` for neighbor smoothing
- `feature_selection/base_models/base_model.py` — `BaseModel` for fitting per fold
- `feature_selection/base_models/quantile_binning.py` — `QuantileBinningModel`
- `nodes/` — Bias node modules for continuous features
- `numpy`, `pandas` for statistical computations

## Invariants / Constraints
- Deterministic: same fold structure and data → same top-K selections and consistency metrics
- No lookahead: each fold evaluation uses only data within that fold's time window
- Independent folds: folds are non-overlapping, no information leakage between folds
- Pre-committed fold structure: fold boundaries defined before running analysis (no data-driven fold selection)
- Consistent evaluation: same objective_func, binning config, and threshold across all folds

## Acceptance tests

**Unit tests:**
- `test_neighbor_smoothing_1d()` — synthetic 1D param grid with known objective values, assert smoothed_objective matches hand-calculated neighbor averages for each grid point
- `test_neighbor_smoothing_2d()` — synthetic 2D param grid (e.g., 3x3), assert boundary points (with fewer neighbors) are averaged correctly
- `test_neighbor_smoothing_single_point()` — grid with one param combination, assert smoothed value equals original (no neighbors)
- `test_top_k_selection()` — synthetic per-fold smoothed objectives with known rankings, assert top_k_params matches expected set
- `test_stable_feature_detection()` — synthetic FoldResult list where top-K is {A, B, C} in all 4 folds, assert `is_stable=True` and overlap_rate=1.0
- `test_unstable_feature_detection()` — synthetic FoldResult list with alternating top-K sets ({A,B,C} vs {X,Y,Z}), assert `is_stable=False`
- `test_permutation_overlay_flags()` — synthetic top-K list and permutation_passers set, assert `passed_permutation_overlay` booleans match expected per-param membership
- `test_fold_result_fields()` — assert FoldResult contains all required fields: `fold_id`, `fold_period`, `top_k_params`, `smoothed_objectives`, `passed_permutation_overlay`
- `test_walkforward_stability_report_fields()` — assert WalkforwardStabilityReport contains: `feature_name`, `feature_type`, `fold_results`, `consistency_metrics`, `is_stable`, `stability_verdict`, `top_k`

**Integration tests:**
- Covered by `tests/integration/feature_validator/test_permutation_testing.py::test_walkforward_stability_stage3()`
- Uses default config: RSI lookback 5, ES daily, 2020-2023
- Loads real candles from `data/ohlc_data/` via `CacheManager`; skips if cache is missing
- Customizable: `bias_module`, `param_value`, `ticker`, `timeframe`, and `fold_structure` are exposed as function parameters to allow researcher exploration with different feature types and periods
- Runs `run_walkforward_stability()` over RSI lookback grid [3, 5, 10, 14] with 2-year non-overlapping folds (2020-2021, 2022-2023)
- Verifies: `WalkforwardStabilityReport` returned, `fold_results` has one entry per fold, `consistency_metrics` dict is non-empty, `is_stable` is a boolean, `stability_verdict` is a non-empty string
- Researcher manual verification:
  - Inspect terminal output for per-fold top-K param selections
  - Confirm consistency metrics (overlap rate, median parameter distance) are printed
  - Review stability verdict and check if top-K params cluster in a consistent region or jump across the grid
  - Check `passed_permutation_overlay` flags if permutation_passers are provided

**Cache policy (integration):**
- Use existing cache: `USE_CACHE=True`
- If cache missing: skip with message "Run CacheManager.populate_cache() first"
- Cache spec: RSI lookback [5], ES, D, 2020-2023

## Definition of done
- [ ] Unit tests added under `tests/validators/permutation/test_walkforward_stability_unit.py`
- [ ] Integration test `test_walkforward_stability_stage3()` added to `tests/integration/feature_validator/test_permutation_testing.py`
- [ ] `WalkforwardStabilityReport` and `FoldResult` dataclasses defined in `feature_selection/validation/reports.py`
- [ ] `run_walkforward_stability()` implemented in `feature_selection/validation/stability_analysis.py`
- [ ] `ParameterAnalyzer.compute_neighbor_smoothed_grid()` method added
- [ ] `pytest tests/validators/permutation/test_walkforward_stability_unit.py -v` passes
- [ ] `pytest tests/integration/feature_validator/test_permutation_testing.py::test_walkforward_stability_stage3 -v` passes
- [ ] Docs updated: docstrings with examples, references to in-sample_pt.md spec

## Notes
- **Stability is diagnostic, not a statistical test**: No p-values, only consistency metrics and researcher-interpretable verdicts
- **Why use ALL params (not just permutation passers)**: Neighbor smoothing requires full grid; relative ranking needs complete landscape
- **Consistency criteria (pre-committed examples)**:
  - "At least 3 of 4 folds select params within 3-step neighborhood"
  - "Top-3 selections overlap by ≥2 params across consecutive folds"
  - "Best smoothed param is within 2 grid steps across all folds"
- **Stable feature (good)**: Same param region consistently top-K across folds (e.g., RSI {3,4,5} appears in 3+ folds)
- **Unstable feature (bad)**: Top-K jumps across parameter space (e.g., fold 1: {3,4,5}, fold 2: {14,20,10}, fold 3: {2,3,4}, fold 4: {10,14,20})
- **Researcher decision input**: Stability report + permutation test results → manual ensemble formation (T016)
- **Continuous vs rule-based difference**:
  - Continuous: Binning refitted per fold (dual purpose: param stability + generalization check)
  - Rule-based: No fitting, purely temporal stability of param ranking
