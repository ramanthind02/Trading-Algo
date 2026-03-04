# T004 - Rewrite Validation/OOS Permutation For Train/Val/Test

## Goal
Replace rolling-fold permutation orchestration with explicit train/validation/test permutation behavior aligned to the new validation pipeline.

## Context / References
- `feature_research/walkforward/run_walkforward_permutation.py`
- `feature_research/oos/run_oos_permutation.py`
- `utils/evaluation/walkforward/permutation_*.py`
- `tests/feature_research/test_run_oos_permutation.py`

## Scope
In scope:
- Add `feature_research/validation/run_validation_permutation.py` as replacement for walkforward permutation entry script.
- Rewrite `feature_research/oos/run_oos_permutation.py` internals to explicit train/val/test semantics.
- Use explicit windows:
  - Validation permutation fold: `train -> validation`.
  - OOS permutation fold: `(train + validation) -> test`.
- Import engine/permutation logic from `utils.evaluation.walkforward.*`.
- Validation output naming:
  - `.../validation/permutation/validation_permutation_report.json`
  - `null_distribution.npy` retained.
- OOS output naming retained at `.../oos/permutation/oos_permutation_report.json`.

Out of scope:
- Generic in-sample permutation suite in `feature_research/pipelines/permutation.py`.
- Broader feature-selection permutation module design.

## Interfaces (must match)
- `feature_research/validation/run_validation_permutation.py` CLI should mirror existing script ergonomics (`--nreps`, `--seed`, `--n-jobs`, `--output-dir`).
- OOS permutation script remains importable as `feature_research.oos.run_oos_permutation:main`.

## Data Contracts
- Report payload must include mode, nreps, alpha, original metric, p-value, critical value, pass/fail.
- Per-ticker aggregation behavior remains deterministic for multi-ticker configs.

## Dependencies
- `feature_research/validation/*`
- `feature_research/oos/run_oos_permutation.py`
- `utils/evaluation/walkforward/permutation_*`

## Invariants / Constraints
- No rolling `num_steps/test_step` fold generation.
- Permutation metric/statistic must be computed consistently between original and null runs.
- Preserve deterministic behavior for fixed random seeds.

## Acceptance tests
1. `source venv/bin/activate && pytest -q tests/feature_research/test_run_oos_permutation.py`
2. `source venv/bin/activate && pytest -q tests/feature_research/validation/test_run_validation_permutation.py`

## Definition of done
- [ ] New validation permutation script exists and runs.
- [ ] OOS permutation is explicit-window based and imports utils engine.
- [ ] Acceptance tests pass.

## Notes
- Coordinate with T003 for shared validation output-directory resolution.
