# T015 — Stage 3: Walkforward Stability Analysis

## Goal
Implement Stage 3 temporal stability analysis that assesses whether the same parameter region is consistently selected as top-K across non-overlapping walkforward folds, distinguishing stable features (genuine signal) from unstable features (noise or regime-dependent).

## Context / References
- `docs/library/Feature_selection/feature_validator.md` — Phase 4: Permutation Testing (lines 318-352)
- `docs/library/Feature_selection/Permutation Testing/in-sample_pt.md` — §3 Walkforward Stability Analysis (lines 82-169)
- `docs/library/Feature_selection/stability/grid_search_parameter_stability.md` — Grid-aware neighbor smoothing theory
- `eda/parameter_analysis.py` — `ParameterAnalyzer` class for neighbor smoothing

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
1. `pytest tests/integration/feature_validator/permutation_testing/test_walkforward_stability.py::test_stable_feature_example` — RSI lookbacks [2,3,4,5,10,14,20], 4 folds, verify consistent top-K around {3,4,5} produces is_stable=True
2. `pytest tests/integration/feature_validator/permutation_testing/test_walkforward_stability.py::test_unstable_feature_example` — RSI lookbacks [2,3,4,5,10,14,20], 4 folds with injected instability (top-K jumps from {3,4,5} to {14,20,10}), verify is_stable=False
3. `pytest tests/integration/feature_validator/permutation_testing/test_walkforward_stability.py::test_neighbor_smoothing` — Verify smoothed_objective computation matches hand-calculated values for 1D and 2D grids
4. `pytest tests/integration/feature_validator/permutation_testing/test_walkforward_stability.py::test_permutation_overlay` — Given permutation_passers={RSI_3, RSI_4, RSI_5}, verify overlay correctly flags which top-K params passed permutation tests
5. `pytest tests/integration/feature_validator/permutation_testing/test_walkforward_stability.py::test_continuous_vs_rule_based` — Test both continuous (RSI) and rule-based (breakout) features
6. `pytest tests/integration/feature_validator/permutation_testing/test_walkforward_stability.py::test_fold_independence` — Verify each fold evaluation is independent (refits binning model per fold for continuous features)

## Definition of done
- [ ] Tests added under `tests/integration/feature_validator/permutation_testing/test_walkforward_stability.py`
- [ ] `WalkforwardStabilityReport` and `FoldResult` dataclasses defined in `feature_selection/validation/reports.py`
- [ ] `run_walkforward_stability()` implemented in `feature_selection/validation/stability_analysis.py`
- [ ] `ParameterAnalyzer.compute_neighbor_smoothed_grid()` method added
- [ ] `pytest tests/integration/feature_validator/permutation_testing/test_walkforward_stability.py -v` passes
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
