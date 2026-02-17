# Dynamic Feature Selection for Binning Integration Test

## Context
- The existing `tests/integration/feature_validator/binning/test_binning_full_pipeline.py` test hard codes the `rsi_signal_D_lookback_5` column when extracting features, saving outputs, and asserting report metadata. This breaks flexibility as soon as a different bias node spec is used during cache population.
- The goal is to keep the test generic so any bias node feature column extracted from `feature_extraction.feature_extractor.extract_features_for_bias_node` is accepted without additional configuration.

## Requirements
1. Automatically pick a bias node feature column produced by the current bias spec (no manual column names).
2. Keep the cache population, feature extraction, and validation logic unchanged except for the column selection step.
3. Adapt the saved output paths, plot verification, and JSON assertions to work for the dynamically selected column.
4. Keep the test deterministic by relying on the first column returned by the extractor.

## Approaches
1. **Keep the RSI column hard coded** (status quo). Pros: least work. Cons: test still fragile and requires manual updates for every feature.
2. **Select `features_df.columns[0]` dynamically** (recommended). Pros: no config changes, automatically works with any bias node while keeping the rest of the flow intact. Cons: assumes the extractor orders columns consistently so the first column is stable for the desired feature (currently true for single-feature specs).
3. **Add a CLI/env var to declare the desired column**. Pros: explicit and flexible. Cons: adds complexity and still requires config management for every test run.

## Selected Design
- Switch `_extract_rsi_features` to `_extract_features`, keeping the BIAS spec as-is, but remove the hard-coded column name from the test.
- After extraction, assert that `features_df` is not empty, then set `feature_col = features_df.columns[0]`. Use a copy of that series for modeling and display logic.
- Replace the old `output_dir` (`.../rsi_lookback_5`) with a shared binning output root, e.g., `tests/integration/outputs/binning`, so `save_report` still nests the per-feature subdirectory automatically.
- Update the `plot_dir`/plot file checks to reference the dynamic column and keep the same naming convention (`heatmap_{feature_col}.png`, etc.).
- Verify that the JSON report’s `feature_column` field matches the selected column.

## Implementation Steps
1. Update the test helper to return the extracted DataFrames and rename it to reflect general feature extraction.
2. Determine `feature_col = features_df.columns[0]` and keep the rest of the pipeline unchanged besides referencing this variable.
3. Set `output_dir` to `project_root/tests/integration/outputs/binning` so new features share a consistent directory root. Ensure directory creation and assertions reflect the new layout.
4. Update `plot_dir` references and JSON assertions to use the dynamic feature name.

## Testing
- Run the binning integration test locally with `pytest tests/integration/feature_validator/binning/test_binning_full_pipeline.py` to confirm the report saves and assertions still pass for the dynamic feature.
- Inspect `tests/integration/outputs/binning/<feature>/<feature>` after the run to ensure plots and JSON artifacts are generated as expected.
