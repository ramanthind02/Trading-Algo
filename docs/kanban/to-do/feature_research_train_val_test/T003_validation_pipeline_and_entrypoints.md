# T003 - Validation Pipeline And Entrypoints

## Goal
Replace walkforward orchestration in `feature_research` with validation orchestration and expose validation-first public entrypoints.

## Context / References
- `feature_research/pipelines/walkforward.py`
- `feature_research/pipelines/oos.py`
- `feature_research/pipeline.py`
- `feature_research/oos/run_oos.py`
- `utils/evaluation/walkforward/*`

## Scope
In scope:
- Add `feature_research/pipelines/validation.py`.
- Add `feature_research/validation/__init__.py`.
- Add `feature_research/validation/run_validation.py`.
- Remove `feature_research/pipelines/walkforward.py`.
- Update `feature_research/pipeline.py` exports:
  - Keep: `run_eda_pipeline`, `run_oos_pipeline`, `run_permutation_pipeline`, `write_permutation_summary`.
  - Add: `run_validation_pipeline`.
  - Remove: all walkforward-named exports.
- Update `feature_research/oos/run_oos.py` imports and messaging to use utils engine path.
- Validation flow semantics:
  - selection/validation fold uses `validation_window`.
  - test fold uses `oos_window` with training on `train + validation`.
- Artifact naming for validation stage:
  - output subdir `validation/`
  - validation-specific figure/report naming.

Out of scope:
- permutation rewrite logic (T004).
- full test/doc migration (T005).

## Interfaces (must match)
- `feature_research.pipeline.run_validation_pipeline(config, output_dir)` is the new public phase function.
- `run_walkforward_pipeline` and type-specific walkforward exports are removed from facade.
- `run_validation.py` is the canonical script replacing prior walkforward script semantics.

## Data Contracts
- Validation artifacts are written under `.../{feature_type}/{module}/validation/`.
- Validation summary tables include selected params and validation/test metrics.

## Dependencies
- `feature_research/pipelines/*`
- `feature_research/validation/*`
- `feature_research/pipeline.py`
- `utils/evaluation/walkforward/*`

## Invariants / Constraints
- No sliding multi-step fold generation in new validation pipeline.
- Test evaluation must use refit on `train + validation`.
- Hard cutover: no walkforward alias exports in public facade.

## Acceptance tests
1. `source venv/bin/activate && python -c "from feature_research.pipeline import run_validation_pipeline, run_oos_pipeline; print('imports ok')"`
2. `source venv/bin/activate && pytest -q tests/feature_research/test_import_smoke.py tests/feature_research/test_oos_pipeline_tearsheet.py`

## Definition of done
- [ ] Validation pipeline and script exist and are wired.
- [ ] Walkforward pipeline module removed.
- [ ] Facade exports validation-only public API.
- [ ] Acceptance commands pass.

## Notes
- Coordinate with T002 to consume flat evaluation fields from `ResearchConfig`.
