# Joint n_bins Parameter Sensitivity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update the continuous-binning param sensitivity notebook so `n_bins` is optimized jointly with bias-node parameters in a single report.

**Architecture:** Use the existing extraction flow, then extend the metric-grid loop to evaluate a cartesian product of expanded bias specs and `N_BINS_GRID`. Append `n_bins` to each result row and include it in `param_names` for report generation.

**Tech Stack:** Jupyter notebook JSON (`.ipynb`), `ContinuousBinningModel`, `SortinoRatio`, `generate_parameter_sensitivity_report`.

---

### Task 1: Update configuration and metadata cells

**Files:**
- Modify: `feature_research/continuous_binning/param_sensitivity.ipynb`

1. Replace scalar `N_BINS` with `N_BINS_GRID = list(config.binning_params.bin_counts)`.
2. Validate non-empty bin grid and print grid summary.
3. Update markdown text to describe joint sensitivity with `n_bins`.

### Task 2: Update metric grid loop

**Files:**
- Modify: `feature_research/continuous_binning/param_sensitivity.ipynb`

1. Iterate through each `n_bins` value for every loaded feature combo.
2. Keep signal-quality guard (`len(selected_returns) < 5`), but include `n_bins` in skip/error messages.
3. Add `row["n_bins"] = n_bins` before appending rows.

### Task 3: Update report call

**Files:**
- Modify: `feature_research/continuous_binning/param_sensitivity.ipynb`

1. Build `report_param_names = [*varying_params, "n_bins"]`.
2. Pass `report_param_names` to `generate_parameter_sensitivity_report`.
3. Use `report_param_names` in recommendation display.

### Task 4: Verify notebook structure

**Files:**
- Modify: `feature_research/continuous_binning/param_sensitivity.ipynb`

1. Run a JSON parse check for the notebook.
2. Run a targeted script asserting expected tokens (`N_BINS_GRID`, `report_param_names`, `row["n_bins"]`) exist.
