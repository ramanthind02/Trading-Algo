# Feature Validator Binning Analysis Design (Continuous)

## Summary
Add a continuous-binning analysis pipeline that mirrors the existing EDA research flow, writes per-parameter outputs under `feature_research/continuous_binning/results/{module_name}/{param_combo}/`, and refactors the binning integration test to a dry-run smoke check that only validates compilation and wiring.

## Context
- Existing continuous EDA pipeline: `feature_research/continuous_binning/pipeline.py` and `run_eda.py`.
- Binning diagnostics/report generation already implemented and exercised via `tests/integration/feature_validator/binning/test_binning_full_pipeline.py`.
- Feature validator specs: `docs/library/Feature_selection/feature_validator.md` and completed kanban items T005–T008.

## Goals
- Provide a researcher-friendly binning analysis entrypoint that reuses the same config/data loader as EDA.
- Persist binning reports per parameter combo alongside EDA outputs for easy comparison.
- Convert the integration test to a compile/smoke test to guard against accidental pipeline breakage without requiring data access.

## Non-Goals
- Implement rule-based research pipelines (separate effort).
- Change binning model internals, report semantics, or validation thresholds.
- Add parameter sensitivity or permutation testing in this phase.

## Proposed Design

### Components
1. **Binning analysis pipeline module**
   - New module: `feature_research/continuous_binning/binning_analysis.py`.
   - Public entrypoint:
     ```python
     run_binning_analysis_pipeline(
         config: ResearchConfig,
         output_dir: Path,
         dry_run: bool = False,
     ) -> dict[str, Path]
     ```
   - Uses existing helpers (`expand_bias_specs`, `load_features_for_combo`, `param_combo_label`) to iterate param grids.
   - For each combo:
     - Load feature + target series with cache-backed extraction.
     - Fit a `ContinuousBinningModel` with configurable settings.
     - Generate a `BinningDiagnosticsReport` via `generate_binning_report(...)`.
     - Save outputs with `save_report(...)` under `output_dir / param_label`.
   - `dry_run=True` skips extraction and report generation; it only validates imports and prepares output directories.

2. **Entry point script**
   - New script: `feature_research/continuous_binning/run_binning_analysis.py`.
   - Mirrors `run_eda.py`: loads config, runs pipeline, prints a short completion line.

3. **Config surface**
   - Extend `feature_research/continuous_binning/config.py` with binning-analysis settings so researchers can adjust binning hyperparameters without editing code.
   - Minimal fields: `n_bins`, `selection_metric`, `strategy`, `metric_threshold`, `t_threshold`, `min_region_width`, `max_regions`, `direction_filter`.

4. **Integration test refactor**
   - Update `tests/integration/feature_validator/binning/test_binning_full_pipeline.py` to:
     - Import the new pipeline module.
     - Run `run_binning_analysis_pipeline(..., dry_run=True)` with a dummy RSI bias spec.
     - Assert that the output directory exists.
   - This keeps the test focused on compilation and wiring rather than data availability.

### Output Layout
- Root: `feature_research/continuous_binning/results/{module_name}/`.
- Per combo: `.../{param_combo}/` (same labeling as EDA).
- Each combo directory contains the saved JSON report and a `plots/` subfolder when not in dry-run mode.

### Error Handling
- If extraction fails or returns empty data, log a one-line skip and continue to the next combo.
- Dry-run mode must not call cache population or feature extraction.

## Testing
- `python -m py_compile feature_research/continuous_binning/run_binning_analysis.py`
- `pytest tests/integration/feature_validator/binning/test_binning_full_pipeline.py -q`

## Open Questions
- None. The pipeline mirrors existing EDA flow and leverages existing binning report infrastructure.
