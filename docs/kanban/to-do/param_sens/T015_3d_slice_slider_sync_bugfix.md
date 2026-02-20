# T015 — 3D Slice Slider Sync Bugfix

## Goal
Fix 3D parameter sensitivity slice controls so dropdown selection updates slider controls for the selected fixed parameter.

## Context / References
- `metrics/plotting/parameter_plots.py`
- `tests/integration/feature_validator/param_sens/test_visualizations.py`
- `feature_research/rule_based/param_sensitivity.ipynb`

## Scope
In scope:
- Fix dropdown/slider synchronization in `plot_3d_slices`.
- Add regression test covering slider update payload.

Out of scope:
- Redesign of 3D plotting style.
- Changes to report dataclass APIs.

## Interfaces (must match)
- Modify: `metrics/plotting/parameter_plots.py`
- Modify: `tests/integration/feature_validator/param_sens/test_visualizations.py`

## Acceptance tests
1. `pytest tests/integration/feature_validator/param_sens/test_visualizations.py::Test3DSlices::test_dropdown_updates_slider_for_selected_fixed_param -q`
2. `pytest tests/integration/feature_validator/param_sens/test_visualizations.py -q`

## Definition of done
- [ ] Dropdown button updates include slider config for selected fixed parameter.
- [ ] New regression test passes.
- [ ] Existing visualization tests remain green.
