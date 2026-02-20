# T009 — Continuous Binning Analysis Outputs + Integration Smoke Test

## Goal
Add a continuous-binning analysis entrypoint that mirrors the EDA pipeline output layout and refactor the binning integration test into a compile/smoke check so pipeline wiring does not regress.

## Context / References
- `docs/library/Feature_selection/feature_validator.md` (continuous feature validation flow)
- `feature_research/continuous_binning/run_eda.py`
- `feature_research/continuous_binning/pipeline.py`
- `feature_research/continuous_binning/data_loader.py`
- `tests/integration/feature_validator/binning/test_binning_full_pipeline.py`
- `docs/kanban/complete/binning/T005_binning_diagnostics_infrastructure.md`
- `docs/kanban/complete/binning/T008_binning_report_generation.md`

## Scope
In scope:
- Add a binning analysis pipeline under `feature_research/continuous_binning/` that reuses the existing config + data loader and emits per-parameter reports.
- Add a script entrypoint that runs the binning analysis pipeline for researchers.
- Refactor the binning integration test to import/compile the new analysis pipeline and execute a dry-run smoke path (no data extraction).
- Persist outputs under `feature_research/continuous_binning/results/{module_name}/{param_combo}/` alongside EDA outputs.

Out of scope:
- Rule-based pipeline implementation.
- Parameter sensitivity or permutation testing changes.
- Any changes to binning model internals or report semantics.

## Interfaces (must match)
- Add: `feature_research/continuous_binning/binning_analysis.py`
  - `run_binning_analysis_pipeline(config: ResearchConfig, output_dir: Path, dry_run: bool = False) -> dict[str, Path]`
    - If `dry_run=True`, only validates imports and creates output folders (no data extraction).
    - If `dry_run=False`, runs the full report generation per param combo and saves reports.
- Add: `feature_research/continuous_binning/run_binning_analysis.py`
  - CLI entrypoint that loads config and calls `run_binning_analysis_pipeline(...)`.
- Modify: `tests/integration/feature_validator/binning/test_binning_full_pipeline.py`
  - Replace the full data pipeline with a smoke test that imports and executes the dry-run path.
- Modify: `docs/api/data_pipeline.md`
  - Document the new `feature_research/continuous_binning` analysis entrypoint(s).

## Data Contracts
- Output layout: `feature_research/continuous_binning/results/{module_name}/{param_combo}/`.
- Each param combo directory contains the binning report JSON and plots (from `save_report(...)`) when `dry_run=False`.
- Dry-run mode creates the directory structure only (no data access, no report files).

## Dependencies
- `feature_research/continuous_binning/config.py`
- `feature_research/continuous_binning/data_loader.py`
- `feature_selection/validators/binning/report.py` (report generation and saving)
- `feature_selection/validators/binning/__init__.py` exports

## Invariants / Constraints
- Deterministic output layout for a given config and parameter grid.
- Dry-run mode must not call feature extraction or cache population.
- Use the same `param_combo_label` convention as the EDA pipeline.

## Acceptance tests
1. `python -m py_compile feature_research/continuous_binning/run_binning_analysis.py`
2. `pytest tests/integration/feature_validator/binning/test_binning_full_pipeline.py -q`

### Integration Test Data Contract (required when integration tests are in scope)
- Data source path: N/A (dry-run smoke test)
- Tickers: N/A
- Timeframe: N/A
- Date range: N/A
- Bias node spec / model config: N/A
- Cache mode: `USE_CACHE`/`POPULATE_CACHE` not used in dry-run

## Definition of done
- [ ] New binning analysis pipeline + entrypoint added under `feature_research/continuous_binning/`.
- [ ] Binning integration test refactored to compile/smoke the pipeline without data access.
- [ ] Output layout matches `{module_name}/{param_combo}` folder structure.
- [ ] `docs/api/data_pipeline.md` updated with new entrypoint(s).
- [ ] `python -m py_compile feature_research/continuous_binning/run_binning_analysis.py` passes.
- [ ] `pytest tests/integration/feature_validator/binning/test_binning_full_pipeline.py -q` passes.

## Notes
- Rule-based research pipeline will mirror this structure in a follow-on task.
