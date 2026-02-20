# T013 — Stage 1: Vector Shuffle Permutation Test

## Goal
Implement Stage 1 vector-level shuffling permutation test as a quick filter to identify parameter combinations with statistically significant feature-target relationships before proceeding to more computationally expensive pipeline tests.

## Context / References
- `docs/library/Feature_selection/feature_validator.md` — Phase 4: Permutation Testing (lines 257-289)
- `docs/library/Feature_selection/Permutation Testing/in-sample_pt.md` — §1 Shuffling Permutation Test (lines 20-41)
- `utils/permutation_test/permutation_engine.py` — `PermutationEngine`, `FeaturePermutationStrategy`
- `docs/kanban/to-do/feature_validator/INTEGRATION_TESTING_SPEC.md` — Unit vs integration test standards, default config, cache policy

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

**Unit tests:**
- `test_deterministic_shuffle()` — synthetic feature/target Series with fixed seed, assert identical null distribution and p-value across two runs
- `test_shuffle_preserves_marginal_distribution()` — synthetic Series with known value frequencies, assert shuffled vector has same value counts as original
- `test_pass_fail_criterion()` — synthetic null distribution where original metric is above 90th percentile, assert `passed=True`; below 90th percentile, assert `passed=False`
- `test_early_stopping_filter_unit()` — synthetic list of 10 VectorShuffleReports with known p-values, assert filter returns only those with p <= alpha
- `test_vector_shuffle_report_fields()` — assert VectorShuffleReport contains all required fields: `param_combo`, `original_metric`, `null_distribution`, `critical_value`, `p_value`, `passed`, `alpha`, `nreps`
- `test_get_fitted_vector_continuous()` — mock BinningModelBase, assert `get_fitted_vector()` returns pd.Series aligned to target index
- `test_get_fitted_vector_rule_based()` — mock rule-based model output (-1, 0, +1), assert alignment and value set

**Integration tests:**
- Covered by `tests/integration/feature_validator/test_permutation_testing.py::test_vector_shuffle_stage1()`
- Uses default config: RSI lookback 5, ES daily, 2020-2023
- Loads real candles from `data/ohlc_data/` via `CacheManager`; skips if cache is missing
- Customizable: `bias_module`, `param_value`, `ticker`, `timeframe` are exposed as function parameters to allow researcher exploration across any bias node/parameter
- Runs `run_vector_shuffle_test()` with `nreps=100` (fast) on fitted RSI feature vector
- Verifies: `VectorShuffleReport` returned, `p_value` is in [0, 1], `null_distribution` has length == nreps, `critical_value` matches (1-alpha) quantile of null distribution
- Researcher manual verification:
  - Inspect terminal output for p-value, critical value, and pass/fail verdict
  - Confirm null distribution histogram (printed shape summary) is unimodal and centred near zero
  - Verify `passed` field is consistent with printed p-value vs alpha comparison

**Cache policy (integration):**
- Use existing cache: `USE_CACHE=True`
- If cache missing: skip with message "Run CacheManager.populate_cache() first"
- Cache spec: RSI lookback [5], ES, D, 2020-2023

## Definition of done
- [ ] Unit tests added under `tests/validators/permutation/test_vector_shuffle_unit.py`
- [ ] Integration test `test_vector_shuffle_stage1()` added to `tests/integration/feature_validator/test_permutation_testing.py`
- [ ] `VectorShuffleReport` dataclass defined in `feature_selection/validation/reports.py`
- [ ] `run_vector_shuffle_test()` implemented in `feature_selection/validation/permutation_tests.py`
- [ ] `BaseModel.get_fitted_vector()` method added
- [ ] `pytest tests/validators/permutation/test_vector_shuffle_unit.py -v` passes
- [ ] `pytest tests/integration/feature_validator/test_permutation_testing.py::test_vector_shuffle_stage1 -v` passes
- [ ] Docs updated: docstrings with examples, references to in-sample_pt.md spec

## Notes
- Use α = 0.10 (90th percentile) as default for lax filtering to reduce false negatives across multiple validation layers
- Stricter option: α = 0.05 for production deployment
- Computational cost is low (vector-only, no pipeline re-runs) — this is the fast filter before expensive Stage 2 tests
- For continuous features, shuffle the binned output (position multipliers), not the raw continuous values
- For rule-based features, shuffle the rule output (-1, 0, +1), preserving the frequency of each level
