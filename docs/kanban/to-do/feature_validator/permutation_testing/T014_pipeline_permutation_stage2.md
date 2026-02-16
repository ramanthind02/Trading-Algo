# T014 — Stage 2: Pipeline Permutation Test

## Goal
Implement Stage 2 pipeline permutation test that validates the entire feature extraction and binning pipeline under progressively stronger null hypotheses, with separate logic for continuous features (feature shuffle → candle shuffle) and rule-based features (candle shuffle only).

## Context / References
- `docs/library/Feature_selection/feature_validator.md` — Phase 4: Permutation Testing (lines 291-316)
- `docs/library/Feature_selection/Permutation Testing/in-sample_pt.md` — §2 Pipeline Permutation Test (lines 43-80)
- `utils/permutation_test/permutation_engine.py` — `PermutationEngine`, `BarPermutationStrategy`
- `docs/library/Feature_selection/features/Continuous_binning.md` — Quantile binning pipeline specification
- `docs/library/Feature_selection/features/rule_based.md` — Rule-based feature specification

## Scope
In scope:
- **Continuous features — two-stage approach**:
  - Stage 2a (quick screen): Shuffle raw continuous feature vector → run full binning pipeline → evaluate
  - Stage 2b (rigorous test): Shuffle candles → recompute bias-node feature → run full binning pipeline → evaluate
  - Null 2a: "No relationship between feature values and target"
  - Null 2b (stronger): "No temporal structure in price that feature exploits" (destroys autocorrelation, momentum runs)
- **Rule-based features — candle shuffle only**:
  - Shuffle candles → recompute rule output → evaluate
  - Rationale: Shuffling rule output creates unrealistic rapid flipping (rule outputs are serially correlated)
- Apply pre-specified metric threshold consistently (same for original and all permutations)
- If no bins meet threshold → assign 0 for all timestamps (permutation takes no trades)
- Generate `PipelinePermutationReport` with p-value, critical value, pass/fail verdict per param combo
- Early stopping: only param combos that passed Stage 1 are tested in Stage 2

Out of scope:
- Stage 1 vector shuffle — already implemented (T013)
- Stage 3 walkforward stability — separate task (T015)
- Defining custom bias nodes or binning strategies — use existing implementations

## Interfaces (must match)
- Add: `feature_selection/validation/permutation_tests.py::run_pipeline_permutation_continuous()`
  - Signature: `run_pipeline_permutation_continuous(candles_df: pd.DataFrame, bias_node_spec: Dict, binning_model: BinningModelBase, target: pd.Series, objective_func: Callable, permutation_mode: Literal['feature_shuffle', 'candle_shuffle'], metric_threshold: float, nreps: int = 1000, alpha: float = 0.10, random_seed: Optional[int] = None) -> PipelinePermutationReport`
  - Uses `FeaturePermutationStrategy` for mode='feature_shuffle', `BarPermutationStrategy` for mode='candle_shuffle'

- Add: `feature_selection/validation/permutation_tests.py::run_pipeline_permutation_rule_based()`
  - Signature: `run_pipeline_permutation_rule_based(candles_df: pd.DataFrame, rule_spec: Dict, target: pd.Series, objective_func: Callable, metric_threshold: float, nreps: int = 1000, alpha: float = 0.10, random_seed: Optional[int] = None) -> PipelinePermutationReport`
  - Uses `BarPermutationStrategy` only

- Add: `feature_selection/validation/reports.py::PipelinePermutationReport`
  - Fields: `param_combo: str`, `feature_type: Literal['continuous', 'rule_based']`, `permutation_mode: str`, `original_metric: float`, `null_distribution: np.ndarray`, `critical_value: float`, `p_value: float`, `passed: bool`, `alpha: float`, `nreps: int`, `no_trade_permutations: int` (count of permutations that produced 0 signal)

## Data Contracts
- **Continuous features**:
  - Null 2a (feature_shuffle): Shuffle raw bias-node feature (before binning) → pipeline produces binned output
  - Null 2b (candle_shuffle): Shuffle bars → recompute bias-node → pipeline produces binned output
  - Pipeline: quantile binning → per-bin stats → contiguous regions → threshold filter → position multipliers
  - If no valid bins: assign 0 (no trades for that permutation)

- **Rule-based features**:
  - Null (candle_shuffle): Shuffle bars → recompute rule output → evaluate
  - Preserves serial correlation of rule logic (no direct output shuffle)

- **Metric threshold**: Pre-specified before seeing data, same for original and all permutations
- **Alignment**: Candles, features, and target must align on datetime index

## Dependencies
- `utils/permutation_test/permutation_engine.py` — `PermutationEngine`, `FeaturePermutationStrategy`, `BarPermutationStrategy`
- `feature_selection/base_models/quantile_binning.py` — `QuantileBinningModel`
- `feature_selection/base_models/base_model.py` — `BaseModel`
- `nodes/` — Bias node modules for feature recomputation from shuffled candles
- `numpy`, `pandas` for statistical computations

## Invariants / Constraints
- Deterministic: same random_seed → same null distribution and p-value
- No data snooping: metric threshold, α, and objective metric must be pre-specified
- Fixed threshold: same metric_threshold for original and all permutation replicates
- No lookahead: features and targets aligned by timestamp, no future information leakage
- Consistent pipeline: binning pipeline configuration (n_bins, selection_metric, strategy) must be identical for original and permuted data

## Acceptance tests
1. `pytest tests/integration/feature_validator/permutation_testing/test_pipeline_permutation.py::test_continuous_feature_shuffle` — Test RSI lookback=5 with feature_shuffle mode, PERMUTATION_REPS=100, verify p-value computation
2. `pytest tests/integration/feature_validator/permutation_testing/test_pipeline_permutation.py::test_continuous_candle_shuffle` — Test RSI lookback=5 with candle_shuffle mode, verify stronger null (should have higher p-values than feature_shuffle for same feature)
3. `pytest tests/integration/feature_validator/permutation_testing/test_pipeline_permutation.py::test_rule_based_candle_shuffle` — Test rule-based feature (e.g., breakout rule) with candle_shuffle, verify rule logic is recomputed from shuffled bars
4. `pytest tests/integration/feature_validator/permutation_testing/test_pipeline_permutation.py::test_no_valid_bins_handling` — Verify that permutations with no bins meeting threshold assign 0 signal and metric=0
5. `pytest tests/integration/feature_validator/permutation_testing/test_pipeline_permutation.py::test_early_stopping_integration` — Only param combos that passed Stage 1 are tested in Stage 2
6. `pytest tests/integration/feature_validator/permutation_testing/test_pipeline_permutation.py::test_rsi_grid_stage2` — Test RSI lookbacks [2,3,4,5,6,7,8,9,10], PERMUTATION_REPS=100, both feature_shuffle and candle_shuffle modes

## Definition of done
- [ ] Tests added under `tests/integration/feature_validator/permutation_testing/test_pipeline_permutation.py`
- [ ] `PipelinePermutationReport` dataclass defined in `feature_selection/validation/reports.py`
- [ ] `run_pipeline_permutation_continuous()` implemented in `feature_selection/validation/permutation_tests.py`
- [ ] `run_pipeline_permutation_rule_based()` implemented in `feature_selection/validation/permutation_tests.py`
- [ ] `pytest tests/integration/feature_validator/permutation_testing/test_pipeline_permutation.py -v` passes
- [ ] Docs updated: docstrings with examples, references to in-sample_pt.md spec

## Notes
- **Recommended workflow for continuous features**: Run Stage 2a (feature_shuffle) as quick screen, then Stage 2b (candle_shuffle) as rigorous test
- **Candle shuffle is stronger null**: Destroys temporal structure (autocorrelation, momentum runs, mean reversion patterns)
- **Computational cost**: candle_shuffle >> feature_shuffle (must recompute bias-node from shuffled bars)
- **Edge case handling**: If permutation produces no valid bins (threshold not met), record as no_trade_permutation, assign metric=0 or NaN (handle consistently)
- **Pre-specified threshold enforcement**: Critical for avoiding data snooping — threshold must be set before running permutation tests
- **For rule-based features**: No feature_shuffle mode (would create unrealistic rapid level changes); candle_shuffle only
