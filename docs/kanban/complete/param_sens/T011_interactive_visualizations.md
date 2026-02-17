# T011 — Interactive Visualizations for Parameter Sensitivity

## Goal
Provide interactive visualizations (line plots, heatmaps, contours) for parameter sensitivity analysis across 1D, 2D, and 3D+ grids, with hover details showing raw/smoothed objectives, stability ratios, and neighbor information to enable researcher inspection of stable regions.

## Context / References
- `docs/library/Feature_selection/feature_validator.md` (lines 222-243: Interactive Visualizations)
- `metrics/plotting/parameter_plots.py` — Existing plotting functions (pure, stateless)
- `eda/parameter_analysis.py` — `ParameterAnalyzer.plot_parameter_sensitivity()` and `plot_2d_parameter_surface()`
- T009 — Grid-aware neighbor smoothing (provides smoothed metrics)
- T010 — Stable region identification (provides region boundaries)
- `docs/kanban/to-do/feature_validator/INTEGRATION_TESTING_SPEC.md` — Unit vs integration test standards

## Scope
In scope:
- 1D parameter grids: Line plot (raw vs smoothed objective), shaded regions for stable zones (stability ratio > 0.8)
- 2D parameter grids: Heatmap (raw objective), contour plot (smoothed objective), overlay highlighting stable regions
- 3D+ parameter grids: 2D slices (fix one parameter, plot others), scatter plot (raw vs smoothed with color-coding)
- Interactive features: hover (exact values, stability ratio, neighbors), click (select param combo), zoom (inspect regions)
- Support for both continuous and rule-based features

Out of scope:
- Statistical testing (Phase 4)
- Neighbor smoothing computation (T009)
- Report generation (T012)

## Interfaces (must match)
- Extend: `eda/parameter_analysis.py::plot_parameter_sensitivity_with_stability(df: pd.DataFrame, param_name: str, metric: str, stability_threshold: float = 0.8, show_plot: bool = True) -> go.Figure`
  - Input: DataFrame with `param1_value`, `{metric}`, `smoothed_{metric}`, `stability_ratio`
  - Output: Plotly figure with dual traces (raw + smoothed) and shaded stable regions
  - Behavior: Plot raw metric as solid line, smoothed metric as dashed line, shade regions where stability_ratio > threshold

- Extend: `eda/parameter_analysis.py::plot_2d_stability_heatmap(df: pd.DataFrame, param1: str, param2: str, metric: str, stable_regions: List[StableRegion], show_plot: bool = True) -> go.Figure`
  - Input: DataFrame with `param1_value`, `param2_value`, `smoothed_{metric}`, `stability_ratio`; list of StableRegion objects
  - Output: Plotly heatmap with contour overlay and stable region boundaries highlighted

- Add: `eda/parameter_analysis.py::plot_3d_slices(df: pd.DataFrame, param_names: List[str], metric: str, fixed_param_idx: int, show_plot: bool = True) -> go.Figure`
  - Input: DataFrame with `param1_value`, `param2_value`, `param3_value`, metric columns
  - Output: Interactive figure with dropdown to select fixed parameter and slider to step through values
  - Behavior: For each fixed parameter value, show 2D heatmap of remaining two parameters

## Data Contracts
- Input DataFrame schema:
  - Required columns: `param1_value`, ..., `paramN_value` (1 ≤ N ≤ 4)
  - Required columns: `{metric}`, `smoothed_{metric}`, `stability_ratio`
  - Optional: `n_neighbors`, `n_samples` (for hover details)

- Plotly figure requirements:
  - Template: 'plotly_white' for consistency
  - Hover template: Display parameter values, raw metric, smoothed metric, stability ratio, n_neighbors
  - Interactive controls: zoom, pan, hover, reset axes
  - Colorscale: 'Viridis' for metric values, 'RdYlGn' for stability ratio

- Stable region overlay:
  - 1D: Vertical shaded regions (fill_between) for contiguous param ranges
  - 2D: Contour lines or polygon overlays around region boundaries
  - Color coding: Green for stable (ratio > 0.8), red for unstable (ratio < 0.5), yellow for marginal

## Dependencies
- plotly (interactive visualizations)
- pandas (DataFrame operations)
- numpy (array operations)
- typing (type hints)
- Existing: `metrics/plotting/parameter_plots.py` (pure plotting functions)

## Invariants / Constraints
- No side effects: Plotting functions are pure (do not modify input DataFrames)
- Deterministic: Same data → same visualization
- Responsive: Interactive controls (hover, zoom) must work smoothly for grids up to 10×10×10 (1000 points)
- Accessibility: Colorblind-friendly palettes, clear labels, legend included
- Export: Figures can be saved as HTML or PNG without loss of interactivity

## Acceptance tests

**Unit tests:**
- `test_plot_1d_returns_figure()` — synthetic 1D DataFrame with 5 points; verify `plot_parameter_sensitivity_with_stability()` returns a `go.Figure` instance without raising
- `test_plot_1d_dual_traces()` — synthetic 1D DataFrame; verify returned figure contains exactly 2 traces (raw solid, smoothed dashed)
- `test_plot_1d_stable_region_shading()` — synthetic 1D DataFrame where points [1,2,3] have stability_ratio > 0.8 and points [0,4] do not; verify figure contains shaded region covering only those points
- `test_plot_2d_heatmap_returns_figure()` — synthetic 3x3 2D DataFrame; verify `plot_2d_stability_heatmap()` returns a `go.Figure` without raising
- `test_plot_2d_stable_region_overlay()` — synthetic 2D DataFrame with a known stable region; verify figure contains overlay trace corresponding to that region
- `test_plot_3d_slices_returns_figure()` — synthetic 2x2x2 3D DataFrame; verify `plot_3d_slices()` returns a `go.Figure` with a dropdown/slider control
- `test_hover_template_fields()` — synthetic 1D DataFrame; verify hover template string in returned figure includes 'stability_ratio' and 'smoothed' fields
- `test_no_side_effects()` — verify input DataFrame is unchanged after calling any plotting function
- `test_figure_export_html()` — verify `go.Figure.to_html()` succeeds without error for 1D and 2D plots
- Location: `tests/validators/param_sens/test_interactive_visualizations.py`

**Integration tests:**
- Covered by `tests/integration/feature_validator/test_parameter_sensitivity.py::test_parameter_sensitivity_pipeline()`
- Default config: RSI lookback grid [3, 4, 5, 10, 14, 20], ES daily, 2020-2023
- Customizable for any bias node/param grid (see INTEGRATION_TESTING_SPEC.md)
- Verifies: 1D line plot generated and saved as HTML, stable regions visible in plot, hover template renders without error
- Cache policy: `USE_CACHE=True`; skip with message if cache missing: "Run CacheManager.populate_cache() first"
- Output: plots saved to `tests/integration/outputs/param_sensitivity/` when `SAVE_INTEGRATION_OUTPUTS=1` env var is set
- Researcher manual verification:
  - Open saved HTML file in browser; confirm stable regions are shaded green
  - Hover over individual parameter values; confirm tooltip shows raw metric, smoothed metric, stability ratio, and neighbor count
  - Confirm boundary parameter values (lookback=3, lookback=20) show fewer neighbors in hover details than interior values

## Definition of done
- [ ] Unit tests added under `tests/validators/param_sens/test_interactive_visualizations.py`
- [ ] Plotting functions added to `eda/parameter_analysis.py` with type hints and docstrings
- [ ] All unit tests pass: `pytest tests/validators/param_sens/test_interactive_visualizations.py -v`
- [ ] Integration coverage provided by `tests/integration/feature_validator/test_parameter_sensitivity.py::test_parameter_sensitivity_pipeline()`
- [ ] Interactive features work: hover, zoom, dropdown, slider
- [ ] Export to HTML verified (save and reload in browser)

## Notes
- 1D visualization example (from spec line 224-226):
  - Line plot: raw vs smoothed objective
  - Shaded regions: stability ratio > 0.8 (green fill)
  - Hover: show exact values, stability ratio, neighbor list

- 2D visualization example (from spec line 228-231):
  - Heatmap: raw objective surface (color intensity)
  - Contour plot: smoothed objective surface (contour lines)
  - Overlay: highlight stable regions (thick green contours or polygon fills)

- 3D+ visualization strategy (from spec line 233-235):
  - Use `plot_3d_parameter_interactive()` from `metrics/plotting/parameter_plots.py` (lines 314-520)
  - Dropdown: select which parameter to fix
  - Slider: step through values of fixed parameter
  - Show 2D heatmap/contour of remaining two parameters

- Colorscale recommendations:
  - Metric values: 'Viridis' (perceptually uniform, colorblind-friendly)
  - Stability ratio: 'RdYlGn' (red=unstable, yellow=marginal, green=stable)

- Hover template format:
  ```
  <b>Param1</b>: {value}<br>
  <b>Metric</b>: {raw:.4f}<br>
  <b>Smoothed</b>: {smoothed:.4f}<br>
  <b>Stability Ratio</b>: {ratio:.2f}<br>
  <b>Neighbors</b>: {n_neighbors}
  ```

- Reference existing implementation:
  - `plot_parameter_sensitivity()` (lines 19-131): 1D line plot with dual y-axis
  - `plot_2d_parameter_surface()` (lines 134-311): 2D surface/heatmap/contour
  - `plot_3d_parameter_interactive()` (lines 314-520): 3D with dropdown/slider
