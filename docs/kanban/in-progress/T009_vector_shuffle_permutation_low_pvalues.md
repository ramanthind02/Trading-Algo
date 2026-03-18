# T009 — Vector Shuffle Permutation Low P-Values

## Goal
Fix the vector shuffle permutation test so p-values reflect correct null behavior, and ensure permutation testing runs per-ticker before aggregating results.

## Context / References
- `feature_research/oos/run_oos_permutation.py`
- `feature_research/walkforward/run_walkforward_permutation.py`
- `feature_research/walkforward/permutation_core.py`
- `utils/evaluation/permutation_test/permutation_engine.py`

## Observed behavior
- Walkforward permutation (vector_shuffle) reports extremely low p-values and a zero critical value.
- Example output (2026-02-27):
  - Original metric: 2.0285
  - Critical: 0.0000
  - p-value: 0.0010 (nreps=1000)
- Rule-based vector_shuffle uses "return shuffle" of aggregate OOS returns, which is degenerate for
  our supported objective metrics (mean_return/sharpe/sortino/t_stat) because they are invariant
  to return order. This yields p-values of 1.0 and critical == original.

## Expected behavior
- Null distribution should be well-formed and produce sensible critical thresholds.
- P-values should be consistent with the null distribution and the observed metric.
- Permutation tests should run per ticker and aggregate those results.

## Reproduction
1. `python feature_research/walkforward/run_walkforward_permutation.py`
2. `python feature_research/oos/run_oos_permutation.py`

## Regression window
- Unknown (needs investigation).

## Scope
In scope:
- Fix vector shuffle permutation logic and aggregation.
- Update permutation reporting to reflect per-ticker testing.

Out of scope:
- Changes to other permutation methods unless required for consistency.

## Interfaces (must match)
- Modify: `feature_research/walkforward/permutation_core.py`
- Modify: `utils/evaluation/permutation_test/permutation_engine.py`
- Modify: `feature_research/oos/run_oos_permutation.py`
- Modify: `feature_research/walkforward/run_walkforward_permutation.py`

## Constraints / Risk
- Must preserve no-lookahead alignment and deterministic shuffling.
- Risk: Incorrect aggregation could mask per-ticker failures.

## Acceptance tests
1. `pytest tests/...::test_vector_shuffle_null_distribution -q` — add or update.
2. `pytest tests/...::test_per_ticker_permutation_aggregation -q` — add or update.

## Definition of done
- [ ] Tests updated/added under `tests/`
- [ ] Docs (if interfaces changed) under `docs/api/`
- [ ] `pytest tests/... -q` passes

## Notes
- Confirm critical value calculation and p-value definition (>= vs >) for one-sided tests.
