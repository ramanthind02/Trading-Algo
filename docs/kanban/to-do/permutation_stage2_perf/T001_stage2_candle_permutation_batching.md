# T001 — Stage 2 Candle Permutation Batching + Shuffler Reuse

## Goal
Reduce in-sample Stage 2 candle-shuffle permutation runtime by reusing prepared candle-shuffle state and batching Stage 2 passers so the shuffled candle stream is generated once per replication (not once per param combo).

## Context / References
- Source of truth (algorithm invariants): `docs/library/Feature_selection/Phase_1_IS/candle_permutation.md`
- Source of truth (Stage 2 semantics): `docs/library/Feature_selection/Phase_1_IS/permutation_testing.md`
- Research entrypoint: `feature_research/in_sample/run_is.py`
- Stage 2 orchestration: `feature_selection/validation/orchestration.py`
- Stage 2 test runners: `feature_selection/validation/permutation_tests.py`
- Canonical shuffler: `utils/evaluation/permutation_test/candle_shuffle.py`
- Compatibility/legacy wrapper surface: `utils/evaluation/permutation_test/permute_bars.py`

## Observed behavior
- Stage 2 runs permutation per param combo and performs `nreps` independent candle shuffles for each combo.
- This duplicates expensive work when multiple Stage 1 passers use the same candle dataset.
- `permute_bars.py` and `candle_shuffle.py` both exist, which obscures ownership even though `BarPermute` is now mainly a compatibility adapter around `CandleShuffler`.

## Expected behavior
- Stage 2 generates one shuffled candle dataframe per replication and reuses it across all Stage 1 passers.
- `CandleShuffler` keeps its public API stable while supporting a reusable prepared internal path for repeated seeded permutations.
- `permute_bars.py` remains import-compatible but explicitly documents that `candle_shuffle.py` is canonical.

## Reproduction
1. `source venv/bin/activate`
2. `python feature_research/in_sample/run_is.py`
3. Enable permutation in config and compare Stage 2 runtime with multiple Stage 1 passers.

## Regression window
- Unknown (performance bottleneck is architectural, not a recent functional regression).

## Scope
In scope:
- Prepared/reusable internals for `CandleShuffler`
- Batch Stage 2 permutation helpers for continuous and rule-based features
- Stage 2 orchestration integration
- Unit tests for wrapper-vs-batch equivalence and batching behavior
- Clarifying docs/comments for `permute_bars.py` ownership

Out of scope:
- Full redesign of `utils/evaluation/permutation_test/permutation_engine.py`
- Hard removal/deprecation of `BarPermute` / `BarPermuteWalkForward`
- OOS permutation batching

## Interfaces (must match)
- Preserve: `utils/evaluation/permutation_test/candle_shuffle.py` — `CandleShuffler.__init__`, `CandleShuffler.permute()`
- Preserve: `feature_selection/validation/permutation_tests.py` — `run_pipeline_permutation_continuous(...)`, `run_pipeline_permutation_rule_based(...)`
- Preserve exports: `utils/evaluation/permutation_test/__init__.py`
- Add internal-only helpers: prepared shuffler internals and Stage 2 batch helpers (underscore-prefixed)

## Constraints / Risk
- Must preserve candle permutation invariants (trend preservation, relative-quantity shuffling, no gap mixing, datetime ordering)
- Must preserve Stage 2 semantics (Stage 1 passers only; per-combo null distribution / p-value logic)
- Risk if incorrect: silent statistical drift in permutation nulls and false pass/fail changes

## Acceptance tests
1. `pytest tests/unit-tests/validators/permutation/test_permutation_candle_shuffle.py -q` — invariants + prepared path parity
2. `pytest tests/unit-tests/validators/permutation/test_permutation_tests_batch.py -q` — wrapper/batch equivalence + multi-combo batch shapes
3. `pytest tests/unit-tests/validators/permutation/test_orchestration_unit.py -q` — Stage 2 passers and batch call behavior
4. `pytest tests/unit-tests/validators/permutation/test_permutation_engine_bar_strategy.py -q` — compatibility with `permute_bars` symbols

## Definition of done
- [x] Tests updated/added under permutation unit test folders
- [x] Module ownership clarified in `permute_bars.py`
- [x] Stage 2 batching path wired into orchestration
- [x] `pytest` targeted permutation suites pass in shared `venv`

## Notes
- `candle_shuffle.py` is the canonical candle permutation implementation.
- `permute_bars.py` is retained as a compatibility/legacy integration surface until walk-forward-specific logic is consolidated.
