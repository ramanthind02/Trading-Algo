# T016 — Early Stopping Orchestration

## Goal
Implement orchestration layer that coordinates the three-stage permutation testing funnel (Stage 1 → Stage 2 → Stage 3) with early stopping, filtering out parameter combinations that fail earlier stages to minimize computational cost while providing comprehensive validation reports.

## Context / References
- `docs/library/Feature_selection/feature_validator.md` — Phase 4: Permutation Testing (lines 257-352), Pipeline behavior with early stopping (lines 265-270)
- `docs/library/Feature_selection/Permutation Testing/in-sample_pt.md` — Summary Flow (lines 237-256), Funnel and cost (lines 261-271)
- Tasks T013 (Stage 1), T014 (Stage 2), T015 (Stage 3)
- `docs/kanban/to-do/feature_validator/INTEGRATION_TESTING_SPEC.md` — Unit vs integration test standards, default config, cache policy

## Scope
In scope:
- Orchestrate sequential execution of Stages 1, 2, 3 with early stopping:
  - Stage 1: All param combos → filter passers
  - Stage 2: Only Stage 1 passers → filter passers
  - Stage 3: ALL param combos (needed for neighbor smoothing) → stability analysis
- Track which param combos passed each stage
- Compute funnel statistics: how many params entered each stage, how many passed, computational savings from early stopping
- Aggregate reports from all stages into unified `PermutationTestSuite` report
- Provide researcher-facing summary: which param combos passed all tests AND show temporal stability (candidates for ensemble)

Out of scope:
- Individual stage implementations (T013-T015) — already implemented
- Researcher ensemble formation — manual decision using orchestration output (T017)
- Vault integration for saving validated features — separate workflow
- GUI or interactive visualization — report generation only

## Interfaces (must match)
- Add: `feature_selection/validation/orchestration.py::run_permutation_test_suite()`
  - Signature: `run_permutation_test_suite(candles_df: pd.DataFrame, feature_spec: Dict, target: pd.Series, param_grid: List[Dict], objective_func: Callable, fold_structure: List[Tuple[pd.Timestamp, pd.Timestamp]], config: PermutationTestConfig) -> PermutationTestSuite`
  - `config`: Configuration object with nreps, alpha, metric_threshold, top_k, random_seed, permutation_mode for Stage 2
  - Runs Stages 1-3 sequentially with early stopping

- Add: `feature_selection/validation/reports.py::PermutationTestSuite`
  - Fields:
    - `feature_name: str`
    - `feature_type: Literal['continuous', 'rule_based']`
    - `stage1_reports: Dict[str, VectorShuffleReport]` — key: param_combo_name
    - `stage2_reports: Dict[str, PipelinePermutationReport]` — only for Stage 1 passers
    - `stage3_report: WalkforwardStabilityReport` — uses all params for smoothing
    - `funnel_stats: FunnelStatistics`
    - `ensemble_candidates: List[str]` — param combos that passed Stage 1-2 AND appear in stable region (top-K in 3+ folds)
    - `summary: str` — researcher-readable summary

- Add: `feature_selection/validation/config.py::PermutationTestConfig`
  - Fields: `nreps: int = 1000`, `alpha: float = 0.10`, `metric_threshold: float`, `top_k: int = 3`, `random_seed: Optional[int] = None`, `permutation_mode_stage2: Literal['feature_shuffle', 'candle_shuffle'] = 'candle_shuffle'`, `min_folds_stable: int = 3` (for ensemble candidate selection)

- Add: `feature_selection/validation/reports.py::FunnelStatistics`
  - Fields: `total_params: int`, `stage1_pass: int`, `stage2_pass: int`, `stable_params: int` (top-K in ≥min_folds_stable folds), `ensemble_candidates: int`, `computational_savings_pct: float` (savings from early stopping vs running all stages on all params)

## Data Contracts
- **Funnel flow**:
  1. All param combos (N) → Stage 1 vector shuffle → N1 passers (N1 ≤ N)
  2. N1 passers → Stage 2 pipeline permutation → N2 passers (N2 ≤ N1)
  3. All param combos (N) → Stage 3 walkforward stability → identify stable region
  4. Intersection: N2 (permutation passers) ∩ stable params → ensemble_candidates

- **Early stopping savings**:
  - Without: N×nreps (Stage 1) + N×nreps (Stage 2) + N×n_folds (Stage 3) evaluations
  - With: N×nreps (Stage 1) + N1×nreps (Stage 2) + N×n_folds (Stage 3) evaluations
  - Savings = (N - N1) × nreps Stage 2 evaluations

- **Ensemble candidate criteria** (all must be satisfied):
  - Passed Stage 1 (vector shuffle, p ≤ α)
  - Passed Stage 2 (pipeline permutation, p ≤ α)
  - Appears in top-K in at least `min_folds_stable` walkforward folds (default: 3)

## Dependencies
- `feature_selection/validation/permutation_tests.py` — Stage 1 and Stage 2 implementations (T013, T014)
- `feature_selection/validation/stability_analysis.py` — Stage 3 implementation (T015)
- `feature_selection/validation/reports.py` — Report dataclasses
- `dataclasses` for configuration and statistics
- `typing` for type hints

## Invariants / Constraints
- Deterministic: same config and random_seed → same funnel results and ensemble candidates
- Sequential execution: Stage 2 only runs on Stage 1 passers (early stopping enforced)
- Stage 3 independence: uses ALL params regardless of permutation test results (needed for neighbor smoothing)
- No backtracking: param combos that fail earlier stages are excluded from later permutation tests (but still used in Stage 3 for smoothing)
- Pre-committed configuration: all thresholds, α, metric, fold structure must be set before running suite

## Acceptance tests

**Unit tests:**
- `test_early_stopping_stage2_receives_only_stage1_passers()` — mock Stage 1 returning 6 pass / 4 fail from 10 params, assert Stage 2 is called exactly 6 times
- `test_stage3_receives_all_params()` — mock Stage 1 returning partial passers, assert Stage 3 `run_walkforward_stability()` is called with the full 10-param grid, not just passers
- `test_ensemble_candidate_intersection()` — synthetic stage1_passers={A,B,C,D}, stage2_passers={A,B,C}, stable_top_k={B,C,D}, assert ensemble_candidates={B,C}
- `test_computational_savings_formula()` — N=10, N1=6, nreps=100, assert savings_pct = (N - N1) * nreps / total_without_early_stopping * 100
- `test_funnel_statistics_fields()` — assert FunnelStatistics contains: `total_params`, `stage1_pass`, `stage2_pass`, `stable_params`, `ensemble_candidates`, `computational_savings_pct`
- `test_permutation_test_suite_fields()` — assert PermutationTestSuite contains: `feature_name`, `feature_type`, `stage1_reports`, `stage2_reports`, `stage3_report`, `funnel_stats`, `ensemble_candidates`, `summary`
- `test_deterministic_suite_unit()` — mock all stages with fixed outputs, call `run_permutation_test_suite()` twice, assert identical PermutationTestSuite results
- `test_permutation_test_config_defaults()` — assert PermutationTestConfig defaults: nreps=1000, alpha=0.10, top_k=3, permutation_mode_stage2='candle_shuffle', min_folds_stable=3

**Integration tests:**
- Covered by `tests/integration/feature_validator/test_permutation_testing.py::test_early_stopping_orchestration()`
- Uses default config: RSI lookback 5, ES daily, 2020-2023
- Loads real candles from `data/ohlc_data/` via `CacheManager`; skips if cache is missing
- Customizable: `bias_module`, `param_value`, `ticker`, `timeframe`, `nreps`, and `alpha` are exposed as function parameters to allow researcher exploration with different features and strictness levels
- Runs `run_permutation_test_suite()` on RSI lookback grid [3, 5, 10, 14] with `nreps=100` (fast), 2-year folds (2020-2021, 2022-2023)
- Verifies: `PermutationTestSuite` returned, `stage1_reports` has one entry per param, `stage2_reports` has no more entries than `stage1_reports`, `stage3_report` covers all params, `funnel_stats.computational_savings_pct` >= 0, `ensemble_candidates` list is a subset of Stage 2 passers
- Researcher manual verification:
  - Inspect terminal output for the full funnel summary (params per stage, pass counts, savings percentage)
  - Confirm `ensemble_candidates` list is printed with justification (passed Stage 1-2 AND stable top-K)
  - Verify Stage 2 count is <= Stage 1 pass count (early stopping is enforced)
  - Review `summary` field output to confirm it matches the per-stage terminal data

**Cache policy (integration):**
- Use existing cache: `USE_CACHE=True`
- If cache missing: skip with message "Run CacheManager.populate_cache() first"
- Cache spec: RSI lookback [5], ES, D, 2020-2023

## Definition of done
- [ ] Unit tests added under `tests/validators/permutation/test_orchestration_unit.py`
- [ ] Integration test `test_early_stopping_orchestration()` added to `tests/integration/feature_validator/test_permutation_testing.py`
- [ ] `PermutationTestSuite`, `FunnelStatistics` dataclasses defined in `feature_selection/validation/reports.py`
- [ ] `PermutationTestConfig` dataclass defined in `feature_selection/validation/config.py`
- [ ] `run_permutation_test_suite()` implemented in `feature_selection/validation/orchestration.py`
- [ ] `pytest tests/validators/permutation/test_orchestration_unit.py -v` passes
- [ ] `pytest tests/integration/feature_validator/test_permutation_testing.py::test_early_stopping_orchestration -v` passes
- [ ] Docs updated: docstrings with examples, references to in-sample_pt.md spec

## Notes
- **Early stopping rationale**: If 50% of params fail Stage 1, Stage 2 saves ~50% of computation (most expensive stage)
- **Computational cost breakdown**:
  - Stage 1 (vector shuffle): Low cost (vector-only, no pipeline)
  - Stage 2 (pipeline permutation): High cost (full pipeline × nreps × N1 params)
  - Stage 3 (walkforward stability): Medium cost (no replicates, but evaluates N params × n_folds)
- **Stage 3 uses ALL params**: Critical for neighbor smoothing and full parameter landscape visualization
- **Ensemble candidate selection**: Automated filtering for researcher convenience, but researcher makes final ensemble decision (T017)
- **Summary format**: Researcher-readable markdown or plain text summarizing:
  - How many params entered each stage
  - How many passed each stage
  - Which params are ensemble candidates (passed all tests + stable)
  - Computational savings achieved
- **Example summary**:
  ```
  Feature: RSI_signal_D
  Parameter grid: lookback [2,3,4,5,6,7,8,9,10] (9 combos)

  Stage 1 (Vector Shuffle): 9 tested → 7 passed (78%)
  Stage 2 (Pipeline Permutation): 7 tested → 5 passed (71% of tested, 56% of total)
  Stage 3 (Walkforward Stability): 9 evaluated → {3,4,5} stable (top-K in 4/4 folds)

  Ensemble candidates: {RSI_3, RSI_4, RSI_5} (passed Stage 1-2 AND stable)
  Computational savings: 22% (2 params excluded from Stage 2)

  Recommendation: Select 2-3 members from {RSI_3, RSI_4, RSI_5} for ensemble
  ```
