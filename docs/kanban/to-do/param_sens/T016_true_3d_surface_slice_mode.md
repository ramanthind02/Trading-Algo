# T016 — True 3D Surface Slice Mode for Parameter Sensitivity

## Goal
Add an optional true 3D surface mode for multi-parameter slice plots while keeping existing heatmap-slice behavior.

## Context / References
- `metrics/plotting/parameter_plots.py`
- `eda/parameter_analysis.py`
- `tests/integration/feature_validator/param_sens/test_visualizations.py`
- `tests/integration/feature_validator/param_sens/test_report.py`

## Scope
In scope:
- Add `plot_type` support to `plot_3d_slices` (`heatmap` default, `surface` optional).
- Add report-level routing knob `plot_3d_mode`.
- Add tests for surface mode and invalid mode handling.

Out of scope:
- Redesign of other plots.

## Acceptance tests
1. `pytest tests/integration/feature_validator/param_sens/test_visualizations.py::Test3DSlices -q`
2. `pytest tests/integration/feature_validator/param_sens/test_report.py::Test3DReportGeneration -q`
