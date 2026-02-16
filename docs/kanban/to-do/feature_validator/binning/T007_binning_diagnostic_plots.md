# T007 — Binning Diagnostic Plots

## Goal
Generate rich visualizations for binning diagnostics, including bin heatmaps color-coded by Sharpe/t-stat, region boundaries overlaid on feature distribution, and position multiplier curves showing continuous scaling functions to guide researcher validation of binning success.

## Context / References
- `docs/library/Feature_selection/feature_validator.md` (lines 168-172: diagnostic plots specification)
- `feature_selection/validators/binning/diagnostics.py` — T005 infrastructure (RegionMetadata)
- `feature_selection/validators/binning/shape_analysis.py` — T006 (ShapeClassification, RegionCoverage)
- `feature_selection/base_models/base_model.py` — position_multipliers_by_strategy_, bin_stats_

## Scope
In scope:
- Bin heatmap plot: color-coded grid showing per-bin Sharpe, t-stat, sample count
- Region boundary overlay: feature distribution histogram with detected region boundaries highlighted
- Position multiplier curve: continuous function showing position scaling across feature range
- Multi-panel diagnostic figure: combined view of all three plots for comprehensive inspection
- Customizable color schemes: researcher can specify colormap for heatmaps

Out of scope:
- Interactive plots (use static matplotlib for reproducibility)
- Report generation (T008: BinningDiagnosticsReport embeds these plots)
- Parameter sensitivity visualizations (separate phase)
- Real-time plot updates (batch generation only)

## Interfaces (must match)
- Add: `feature_selection/validators/binning/plots.py`
  - `plot_bin_heatmap(model: BinningModelBase, metric: str = "sharpe", figsize: Tuple[int, int] = (12, 4), cmap: str = "RdYlGn") -> Figure`
    - Generates horizontal bar chart heatmap color-coded by selected metric
    - metric options: "sharpe", "t_stat", "sample_count", "mean_return"
    - Returns matplotlib Figure object
  - `plot_region_boundaries(model: BinningModelBase, feature_data: pd.Series, regions: List[RegionMetadata], figsize: Tuple[int, int] = (10, 6)) -> Figure`
    - Histogram of feature_data with vertical shaded regions for tradeable zones
    - Annotates region boundaries with bin numbers and feature ranges
    - Returns matplotlib Figure object
  - `plot_position_multiplier_curve(model: BinningModelBase, strategy: str = "long", figsize: Tuple[int, int] = (10, 6)) -> Figure`
    - Continuous step function showing position_multipliers_by_strategy_
    - X-axis: feature value range, Y-axis: position multiplier
    - Highlights active bins with distinct colors
    - Returns matplotlib Figure object
  - `create_diagnostic_panel(model: BinningModelBase, feature_data: pd.Series, regions: List[RegionMetadata], strategy: str = "long", figsize: Tuple[int, int] = (18, 12)) -> Figure`
    - Combined 3-panel figure (heatmap, boundaries, multiplier curve)
    - Shared title with model metadata (feature_column, n_bins, strategy)
    - Returns matplotlib Figure object

- Modify: `feature_selection/validators/binning/__init__.py` — export plot functions

## Data Contracts
- Input: fitted ContinuousBinningModel with:
  - bin_stats_: Dict[int, Dict[str, float]]
  - position_multipliers_by_strategy_: Dict[str, Dict[int, float]]
  - bin_edges_: List[float]
  - feature_column: str (for plot titles)

- Input: RegionMetadata list from T005
- Input: feature_data pd.Series (for distribution overlay)
- Output: matplotlib Figure objects (can be saved to file or displayed)

- Plot data requirements:
  - Bin heatmap: requires bin_stats_ with selected metric values
  - Region boundaries: requires feature_data, bin_edges_, regions
  - Multiplier curve: requires position_multipliers_by_strategy_, bin_edges_

## Dependencies
- matplotlib (figure generation, colormaps)
- numpy (bin edge calculations, linspace for curves)
- pandas (feature data handling)
- feature_selection/validators/binning/diagnostics.py (RegionMetadata)
- feature_selection/base_models/base_model.py (BinningModelBase)

## Invariants / Constraints
- Deterministic: same model + feature_data => identical plots (no randomness)
- Color mappings: high Sharpe = green, low/negative = red (intuitive)
- Region overlay: shaded regions must align with bin_edges_ boundaries
- Multiplier curve: step function must match bin_edges_ and position_multipliers_
- Figure size: customizable but defaults must be readable at 100 DPI
- Axis labels: clear units (Sharpe dimensionless, feature values in native units)

## Acceptance tests
1. `pytest tests/integration/feature_validator/binning/test_plots.py::test_plot_bin_heatmap_sharpe -q` — generate heatmap for RSI model, verify color coding matches bin_stats_
2. `pytest tests/integration/feature_validator/binning/test_plots.py::test_plot_bin_heatmap_t_stat -q` — heatmap with t_stat metric, verify high |t| bins highlighted
3. `pytest tests/integration/feature_validator/binning/test_plots.py::test_plot_region_boundaries_rsi -q` — overlay regions on RSI distribution, verify boundary alignment
4. `pytest tests/integration/feature_validator/binning/test_plots.py::test_plot_multiplier_curve_long -q` — plot long strategy multiplier curve, verify step heights match position_multipliers_
5. `pytest tests/integration/feature_validator/binning/test_plots.py::test_create_diagnostic_panel -q` — generate 3-panel figure, verify all subplots present and titled
6. `pytest tests/integration/feature_validator/binning/test_plots.py::test_save_diagnostic_plots -q` — save plots to temp directory, verify files created and readable

## Definition of done
- [ ] Tests added under `tests/integration/feature_validator/binning/`
- [ ] Plot functions implemented in `feature_selection/validators/binning/plots.py`
- [ ] All plots include clear titles, axis labels, legends, and annotations
- [ ] Colormap customization tested (RdYlGn default, alternative colormaps work)
- [ ] Docs updated under `docs/api/feature_selection/validators.md`
- [ ] `pytest tests/integration/feature_validator/binning/test_plots.py -q` passes

## Notes
- Test with RSI continuous feature: lookback=[2,3,4,5,6,7,8,9,10], TimeFrame.D
- Use quantile binning (n_bins=15)
- Bin heatmap: horizontal bars allow easy reading of bin numbers (0-14)
- Region boundaries: use semi-transparent shading (alpha=0.3) for overlay visibility
- Multiplier curve: step function reflects discrete bin assignments in production
- Diagnostic panel: enables at-a-glance assessment of binning quality
- Researcher interpretation: heatmap shows statistical significance, boundaries show feature coverage, curve shows position scaling
- Save as PNG (default) or PDF (publication quality) using `fig.savefig()`
