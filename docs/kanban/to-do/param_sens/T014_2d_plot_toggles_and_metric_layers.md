# T014 — 2D Param Sensitivity Plot Toggles + Metric Layers

## Goal
Improve 2D parameter sensitivity research UX by keeping one plot while adding toggles for base metric layers and overlays.

## Context / References
- `metrics/plotting/parameter_plots.py`
- `eda/parameter_analysis.py`
- `docs/library/Feature_selection/feature_validator.md`
- `docs/library/Feature_selection/Parameter Sensitivity/grid_search_parameter_stability.md`

## Scope
In scope:
- Extend 2D stability heatmap with selectable base layers and overlay modes.
- Support additional DataFrame metric columns as selectable layers.
- Update integration tests and docs.

Out of scope:
- New multi-plot report API.
- 1D/3D plot redesign.

## Interfaces (must match)
- Modify: `metrics/plotting/parameter_plots.py`
  - `plot_2d_stability_heatmap(...)` keeps backward compatibility and returns `go.Figure`.
- Modify: `tests/integration/feature_validator/param_sens/test_visualizations.py`
- Modify: `docs/library/Feature_selection/feature_validator.md`

## Invariants / Constraints
- Deterministic figure construction for same input data.
- Existing callers that pass only previous arguments still work.

## Acceptance tests
1. `pytest tests/integration/feature_validator/param_sens/test_visualizations.py -q`
2. `pytest tests/integration/feature_validator/param_sens/test_report.py -q`

## Definition of done
- [ ] 2D figure supports base layer dropdown and overlay mode controls.
- [ ] Additional DataFrame metric columns can be selected as base layers.
- [ ] Integration tests pass.
- [ ] Docs updated.
