# T001 - Move Walkforward Engine To utils/evaluation/walkforward

## Goal
Relocate walkforward engine and permutation internals from `feature_research/walkforward/` into `utils/evaluation/walkforward/` for future reuse, without changing behavior.

## Context / References
- `feature_research/walkforward/*.py`
- `utils/evaluation/walkforward.py`
- `tests/feature_research/walkforward/test_runner.py`
- `tests/feature_research/walkforward/test_io.py`

## Scope
In scope:
- Move these files:
  - `runner.py`
  - `io.py`
  - `config.py`
  - `research_data.py`
  - `evaluators.py`
  - `metrics.py`
  - `visualization.py`
  - `portfolio_evaluator.py`
  - `top_k_selection.py`
  - `permutation_core.py`
  - `permutation_runtime.py`
  - `permutation_helpers.py`
- Create package form under `utils/evaluation/walkforward/` with `__init__.py`.
- Preserve legacy `utils.evaluation.walkforward` import surface for splitter utilities currently in `utils/evaluation/walkforward.py`.
- Rewrite internal imports from `feature_research.walkforward.*` to `utils.evaluation.walkforward.*`.

Out of scope:
- Algorithm changes in scoring, selection, permutation, IO, or portfolio behavior.
- Validation pipeline or config simplification.

## Interfaces (must match)
- Preserve importability and signatures for:
  - `run_walkforward_research`
  - `WalkforwardResearchConfig`
  - `write_walkforward_artifacts`
  - permutation functions consumed by OOS and validation scripts.
- Keep compatibility for code that imports splitter helpers from `utils.evaluation.walkforward`.

## Data Contracts
- No artifact schema changes in this task.
- No metric naming changes in this task.

## Dependencies
- `feature_research/*`
- `utils/evaluation/*`
- `tests/feature_research/walkforward/*`

## Invariants / Constraints
- Path relocation only; behavior must stay identical.
- No lookahead guarantees remain unchanged.
- No hidden defaults changed.

## Acceptance tests
1. `source venv/bin/activate && python -c "from utils.evaluation.walkforward.runner import run_walkforward_research; print('ok')"`
2. `source venv/bin/activate && pytest -q tests/feature_research/walkforward/test_runner.py tests/feature_research/walkforward/test_io.py`

## Definition of done
- [ ] All 12 modules exist under `utils/evaluation/walkforward/`.
- [ ] Internal engine imports reference `utils.evaluation.walkforward.*`.
- [ ] Acceptance commands pass.

## Notes
- This is prerequisite for T003 and T004.
