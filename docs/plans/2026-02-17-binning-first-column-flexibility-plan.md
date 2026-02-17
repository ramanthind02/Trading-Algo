# Binning First Column Flexibility Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Let the binning integration test select the first bias-node feature column automatically while keeping plots/JSON assertions aligned with that column.

**Architecture:** Use the existing `extract_features_for_bias_node` helper and derive the feature column from `features_df.columns[0]`, leveraging `save_report`'s per-feature subdirectory rather than hard-coding `rsi_lookback_5`. Keep the WIP pipeline unchanged apart from referencing the new column variable.

**Tech Stack:** Python 3.11, Pytest-driven verification, pandas-based feature extraction helpers.

---

### Task 1: Update binning integration test to pick the first feature column

**Files:**
- Modify: `tests/integration/feature_validator/binning/test_binning_full_pipeline.py`
**Step 1: Update the test to reference the first column before other logic runs**
```python
features_df, targets_df = _extract_features()
feature_col = features_df.columns[0]
feature_series = features_df[feature_col].copy()
feature_series.name = feature_col
```
Leave output_dir, plot_dir and assertions pointing to `feature_col`. This edit will make expectations for saved artifacts refer to the dynamic column, so the test will initially fail until the helper/output path logic is updated.

**Step 2: Run the targeted test to observe the failure and capture the mismatch**
```bash
pytest tests/integration/feature_validator/binning/test_binning_full_pipeline.py::test_binning_full_pipeline_integration -vv
```
Expected: FAIL because the helper/paths still assume `rsi_signal_D_lookback_5` outputs (missing directories or mismatched JSON claims).

**Step 3: Update the helper logic and path assertions**
- Rename `_extract_rsi_features` to `_extract_features` and keep returning `(features_df, targets_df)` from `_extract_features()`.
- Set `output_dir = project_root / "tests" / "integration" / "outputs" / "binning"` so the per-feature subdirectory is created by `save_report` itself.
- When verifying plots and JSON contents, refer to `feature_col`, e.g. `plot_dir = output_dir / feature_col / "plots"` and assert `report_json["feature_column"] == feature_col`.

**Step 4: Run the same pytest command to ensure the test now passes with the dynamic column.**
Expected: PASS and `tests/integration/outputs/binning/<feature_col>` contains plots and JSON report.

**Step 5: Commit the change describing the adjustment.**
```bash
git add tests/integration/feature_validator/binning/test_binning_full_pipeline.py docs/plans/2026-02-17-binning-first-column-flexibility-design.md docs/plans/2026-02-17-binning-first-column-flexibility-plan.md
git commit -m "chore: let binning test use first feature column"
```
