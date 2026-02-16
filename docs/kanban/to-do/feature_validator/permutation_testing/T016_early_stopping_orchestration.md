# T016 — Early Stopping Orchestration

## Goal
Implement orchestration layer that coordinates the three-stage permutation testing funnel (Stage 1 → Stage 2 → Stage 3) with early stopping, filtering out parameter combinations that fail earlier stages to minimize computational cost while providing comprehensive validation reports.

## Context / References
- `docs/library/Feature_selection/feature_validator.md` — Phase 4: Permutation Testing (lines 257-352), Pipeline behavior with early stopping (lines 265-270)
- `docs/library/Feature_selection/Permutation Testing/in-sample_pt.md` — Summary Flow (lines 237-256), Funnel and cost (lines 261-271)
- Tasks T013 (Stage 1), T014 (Stage 2), T015 (Stage 3)

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
1. `pytest tests/integration/feature_validator/permutation_testing/test_orchestration.py::test_funnel_early_stopping` — Given 10 param combos, if 6 pass Stage 1, verify Stage 2 runs on exactly 6 combos (not 10)
2. `pytest tests/integration/feature_validator/permutation_testing/test_orchestration.py::test_ensemble_candidate_selection` — RSI lookbacks [2,3,4,5,6,7,8,9,10], verify ensemble_candidates are intersection of (Stage 1-2 passers) ∩ (stable top-K params)
3. `pytest tests/integration/feature_validator/permutation_testing/test_orchestration.py::test_computational_savings` — Verify funnel_stats.computational_savings_pct correctly computes savings from early stopping
4. `pytest tests/integration/feature_validator/permutation_testing/test_orchestration.py::test_continuous_vs_rule_based` — Run full suite on both continuous (RSI) and rule-based (breakout) features
5. `pytest tests/integration/feature_validator/permutation_testing/test_orchestration.py::test_deterministic_suite` — Fixed random_seed → identical suite results across runs
6. `pytest tests/integration/feature_validator/permutation_testing/test_orchestration.py::test_stage3_uses_all_params` — Verify Stage 3 walkforward stability runs on ALL params (N), not just Stage 2 passers (N2)

## Definition of done
- [ ] Tests added under `tests/integration/feature_validator/permutation_testing/test_orchestration.py`
- [ ] `PermutationTestSuite`, `FunnelStatistics` dataclasses defined in `feature_selection/validation/reports.py`
- [ ] `PermutationTestConfig` dataclass defined in `feature_selection/validation/config.py`
- [ ] `run_permutation_test_suite()` implemented in `feature_selection/validation/orchestration.py`
- [ ] `pytest tests/integration/feature_validator/permutation_testing/test_orchestration.py -v` passes
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
