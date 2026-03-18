# T000 - Train/Val/Test Cutover Master Coordination

## Goal
Coordinate the full hard-cutover refactor from walkforward orchestration to train/validation/test orchestration across multiple agents with clear ownership, merge order, and global acceptance.

## Context / References
- `docs/kanban/templates/feature.md`
- `feature_research/config.py`
- `feature_research/in_sample/config.py`
- `feature_research/pipeline.py`
- `feature_research/pipelines/`
- `feature_research/walkforward/`
- `feature_research/oos/`
- `utils/evaluation/walkforward.py`
- `tests/feature_research/`

## Locked Decisions
- Hard cutover to validation naming with no walkforward compatibility API in `feature_research`.
- Public runtime entrypoints are `run_validation + run_oos`.
- Test scoring refits on `train + validation`.
- Walkforward engine code moves to `utils/evaluation/walkforward`.
- Validation artifacts use validation names and `validation/` output subdir.
- `feature_research.pipeline` exports validation names only.
- Validation/OOS permutation is rewritten around explicit train/val/test windows.

## Ownership Map
- Agent 1: engine relocation into `utils/evaluation/walkforward/*`.
- Agent 2: config contract simplification in `feature_research/config.py` and `feature_research/in_sample/config.py`.
- Agent 3: validation pipeline, validation scripts, and pipeline facade exports.
- Agent 4: validation and OOS permutation rewrite.
- Agent 5: tests/docs migration and hard-cut cleanup.

## Merge Order
1. T001
2. T002
3. T003
4. T004
5. T005

## Integration Constraints
- Do not touch unrelated dirty files already present in repo.
- Use shared venv only: `source venv/bin/activate`.
- Avoid API alias shims for removed walkforward names in `feature_research.pipeline`.
- Keep behavioral logic unchanged when moving engine modules to `utils` (path change only).

## Global Acceptance
1. `source venv/bin/activate && pytest -q tests/feature_research`
2. `source venv/bin/activate && pytest -q tests/feature_research/validation tests/feature_research/test_run_oos_permutation.py`
3. `source venv/bin/activate && python -c "import feature_research.pipeline as p; print(hasattr(p,'run_validation_pipeline'), hasattr(p,'run_walkforward_pipeline'))"` and expected output is `True False`

## Definition of Done
- [ ] T001-T005 merged in order with no unresolved interface drift.
- [ ] Global acceptance commands pass.
- [ ] Final surface is validation-first and no walkforward facade exports remain in `feature_research.pipeline`.

## Notes
- This task is orchestration-only; implementation details live in T001-T005.
