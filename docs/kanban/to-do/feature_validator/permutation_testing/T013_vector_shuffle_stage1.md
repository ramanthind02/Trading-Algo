# T013 — Stage 1: Vector Shuffle Permutation Test

## Goal
Implement Stage 1 vector-level shuffling permutation test as a quick filter to identify parameter combinations with statistically significant feature-target relationships before proceeding to more computationally expensive pipeline tests.

## Context / References
- `docs/library/Feature_selection/feature_validator.md` — Phase 4: Permutation Testing (lines 257-289)
- `docs/library/Feature_selection/Permutation Testing/in-sample_pt.md` — §1 Shuffling Permutation Test (lines 20-41)
- `utils/permutation_test/permutation_engine.py` — `PermutationEngine`, `FeaturePermutationStrategy`

## Scope
In scope:
- Implement vector shuffle test per parameter combination for both continuous (binned output) and rule-based features
- Shuffle fitted feature vector (position multipliers or discrete signals) while preserving target order
- Compute objective metric on shuffled data to build null distribution
- Pass/fail criterion: original metric > (1-α) quantile of null distribution (α = 0.10 default)
- Generate `VectorShuffleReport` dataclass with p-value, critical value, pass/fail verdict per param combo
- Early stopping: only param combos passing Stage 1 proceed to Stage 2

Out of scope:
- Pipeline permutation (Stage 2) — separate task
- Walkforward stability analysis (Stage 3) — separate task
- Ensemble formation — researcher decision after all stages complete

## Interfaces (must match)
- Add: `feature_selection/validation/permutation_tests.py::run_vector_shuffle_test()`
  - Signature: `run_vector_shuffle_test(fitted_feature: pd.Series, target: pd.Series, objective_func: Callable, nreps: int = 1000, alpha: float = 0.10, random_seed: Optional[int] = None) -> VectorShuffleReport`
  - Uses `FeaturePermutationStrategy` from `permutation_engine.py`
  - Returns structured report with p-value and pass/fail verdict

- Add: `feature_selection/validation/reports.py::VectorShuffleReport`
  - Fields: `param_combo: str`, `original_metric: float`, `null_distribution: np.ndarray`, `critical_value: float`, `p_value: float`, `passed: bool`, `alpha: float`, `nreps: int`

- Modify: `feature_selection/base_models/base_model.py::BaseModel`
  - Add method: `get_fitted_vector() -> pd.Series` — returns binned output or rule output after fitting
  - Ensures alignment with target (same index)

## Data Contracts
- **Input DataFrame**: datetime index (timezone-aware or naive, consistent), fitted feature column, target column
- **Null hypothesis**: No relationship between feature values and target returns
- **Alternative hypothesis**: Feature values predict target returns better than chance
- **Alignment**: fitted_feature and target must have matching indices (same length, same timestamps)
- **Objective metric signature**: `objective_func(returns: pd.Series) -> float` where returns = target * feature (or position multiplier * target)

## Dependencies
- `utils/permutation_test/permutation_engine.py` — `PermutationEngine`, `FeaturePermutationStrategy`
- `feature_selection/base_models/base_model.py` — `BaseModel` for fitted vector extraction
- `numpy`, `pandas` for statistical computations
- `dataclasses` for report structures

## Invariants / Constraints
- Deterministic: same random_seed → same null distribution and p-value
- No data snooping: α (significance level) and objective metric must be pre-specified before seeing data
- Same threshold for original and all permutations (no adaptive thresholds)
- Preserves marginal distribution: shuffled vector has same value frequencies, only order changes
- No lookahead: feature vector is from fitted model on in-sample data only

## Acceptance tests
1. `pytest tests/integration/feature_validator/permutation_testing/test_vector_shuffle.py::test_deterministic_shuffle` — Fixed seed yields identical p-values across runs
2. `pytest tests/integration/feature_validator/permutation_testing/test_vector_shuffle.py::test_null_hypothesis_rsi` — Test RSI with lookbacks [2,3,4,5,6,7,8,9,10], PERMUTATION_REPS=100, verify p-values are computed correctly
3. `pytest tests/integration/feature_validator/permutation_testing/test_vector_shuffle.py::test_continuous_vs_rule_based` — Verify both continuous (binned) and rule-based feature vectors produce valid VectorShuffleReports
4. `pytest tests/integration/feature_validator/permutation_testing/test_vector_shuffle.py::test_early_stopping_filter` — Given 10 param combos, verify only those passing Stage 1 (p <= alpha) are flagged for Stage 2

## Definition of done
- [ ] Tests added under `tests/integration/feature_validator/permutation_testing/test_vector_shuffle.py`
- [ ] `VectorShuffleReport` dataclass defined in `feature_selection/validation/reports.py`
- [ ] `run_vector_shuffle_test()` implemented in `feature_selection/validation/permutation_tests.py`
- [ ] `BaseModel.get_fitted_vector()` method added
- [ ] `pytest tests/integration/feature_validator/permutation_testing/test_vector_shuffle.py -v` passes
- [ ] Docs updated: docstrings with examples, references to in-sample_pt.md spec

## Notes
- Use α = 0.10 (90th percentile) as default for lax filtering to reduce false negatives across multiple validation layers
- Stricter option: α = 0.05 for production deployment
- Computational cost is low (vector-only, no pipeline re-runs) — this is the fast filter before expensive Stage 2 tests
- For continuous features, shuffle the binned output (position multipliers), not the raw continuous values
- For rule-based features, shuffle the rule output (-1, 0, +1), preserving the frequency of each level
