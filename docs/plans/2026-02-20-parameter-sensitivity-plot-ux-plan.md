# Parameter Sensitivity Plot UX Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve 2D parameter sensitivity visualization usability by adding interactive base-layer and overlay toggles while preserving one-plot workflow.

**Architecture:** Extend `plot_2d_stability_heatmap` to build multiple base heatmap traces and optional overlays, then control visibility via Plotly `updatemenus`. Keep existing API behavior compatible by default and support extra metric layers through column-key resolution.

**Tech Stack:** Python, Plotly (`go.Figure`, `Heatmap`, `Contour`, `Scatter`), pytest.

---

### Task 1: Add failing visualization tests first

**Files:**
- Modify: `tests/integration/feature_validator/param_sens/test_visualizations.py`

1. Add test asserting layer selector dropdown exists and includes expected buttons (`smoothed`, `raw`, `stability_ratio`).
2. Add test asserting overlay control menu exists with 4 modes.
3. Add test asserting `default_layer="raw"` sets raw layer visible initially.
4. Add test asserting custom DataFrame metric column can be selected via `base_layers`.

### Task 2: Implement 2D plot controls and layer support

**Files:**
- Modify: `metrics/plotting/parameter_plots.py`

1. Extend `plot_2d_stability_heatmap` signature with optional layer/control arguments.
2. Implement predefined layer alias resolution (`raw`, `smoothed`, `stability_ratio`, `n_neighbors`, `delta`).
3. Allow base layer keys that directly match DataFrame columns.
4. Build one heatmap trace per selected layer and toggle visibility via dropdown.
5. Build overlay traces (stability contour and stable-region markers) and add overlay mode buttons.
6. Keep defaults backward compatible and preserve stable-region hover context.

### Task 3: Update library docs for UX/research workflow

**Files:**
- Modify: `docs/library/Feature_selection/feature_validator.md`

1. Update parameter sensitivity section to describe single-figure layer/overlay toggles.
2. Document interpretation guidance for `raw`, `smoothed`, `stability_ratio`, `delta`, and `n_neighbors`.

### Task 4: Verify

**Files:**
- Modify: none

1. Run targeted test module:
   - `pytest tests/integration/feature_validator/param_sens/test_visualizations.py -q`
2. Run related report integration tests:
   - `pytest tests/integration/feature_validator/param_sens/test_report.py -q`
