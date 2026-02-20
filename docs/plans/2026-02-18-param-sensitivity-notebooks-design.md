# Parameter Sensitivity Notebooks — Design Doc

**Date:** 2026-02-18
**Status:** Approved

## Problem

The existing `eda/parameter_analysis.py` infrastructure (`generate_parameter_sensitivity_report`,
`compute_neighbor_smoothing`, `identify_stable_regions`, interactive Plotly plots) is fully built
but not wired into the research pipeline. Researchers currently have no notebook-based workflow
to run parameter sensitivity on real features extracted from the live pipeline.

## Solution

Two Jupyter notebooks — one per feature type — placed alongside their respective pipeline modules
in `feature_research/`. Each notebook is a thin orchestration layer over existing infrastructure:
cache-backed feature extraction → metric grid construction → `generate_parameter_sensitivity_report()` → interactive Plotly figures.

## Notebook Locations

```
feature_research/
  continuous_binning/
    param_sensitivity.ipynb    ← NEW
  rule_based/
    param_sensitivity.ipynb    ← NEW
```

## Approach: Thin Orchestration Shell (Hybrid Config)

- Notebooks import `load_config()` from their respective `config.py` as the default
- A Config Override cell uses `dataclasses.replace()` to allow inline overrides without mutating `config.py`
- Param-sensitivity-specific knobs (`STABILITY_THRESHOLD`, `TOP_K`, `METRIC_COL`, `N_BINS`) live in the override cell

## Cell Structure (both notebooks)

| Section | Content |
|---------|---------|
| 0 — Setup | `sys.path` insert, all imports |
| 1 — Config | `load_config()` + override cell with `dataclasses.replace()` |
| 2 — Feature Extraction | `populate_cache_if_needed()`, `expand_bias_specs()`, loop over combos via `load_features_for_combo()` |
| 3 — Metric Grid | Build `results_df` with `param{K}_value` + metric columns |
| 4 — Parameter Sensitivity | `generate_parameter_sensitivity_report()`, print summary stats |
| 5 — Interactive Plots | `report.plot_1d/2d/3d.show()` |
| 6 — Optional Save | Commented-out cells for HTML + parquet export |

## Data Flow

### Continuous Binning

```
expand_bias_specs(config.bias_spec) → [spec1 ... specN]

for each spec:
  load_features_for_combo(spec, config) → (feature, target, col)
  ContinuousBinningModel(n_bins=N_BINS).fit(feature, target)
  signals = model.predict(feature, strategy='long')
  selected_returns = target[signals == 1]
  metric_value = SortinoRatio(252).compute(selected_returns)
  → row: {param1_value, [param2_value, ...], metric}

results_df → generate_parameter_sensitivity_report(
    results_df, param_names, METRIC_COL, STABILITY_THRESHOLD, TOP_K
)
```

`N_BINS` defaults to `config.binning_params.bin_counts[0]`, overridable in config cell.

### Rule-Based

```
expand_bias_specs(config.bias_spec) → [spec1 ... specN]

for each spec:
  load_features_for_combo(spec, config) → (feature, target, col)
  # No model fitting — feature IS the binary signal
  selected_returns = target[feature == 1]
  metric_value = SortinoRatio(252).compute(selected_returns)
  → row: {param1_value, [param2_value, ...], metric}

results_df → generate_parameter_sensitivity_report(...)
```

`param_names` is derived from `bias_spec["params"].keys()`.

## Key Infrastructure Reused

| Component | Location | Role |
|-----------|----------|------|
| `load_config()` | `feature_research/{cb,rb}/config.py` | Default config |
| `expand_bias_specs()` | `feature_research/continuous_binning/data_loader.py` | Expand param grid |
| `load_features_for_combo()` | `feature_research/*/data_loader.py` | Cache-backed extraction |
| `populate_cache_if_needed()` | `feature_research/*/data_loader.py` | Cache management |
| `ContinuousBinningModel` | `feature_selection/base_models/continuous_binning.py` | CB model fitting |
| `SortinoRatio` | `metrics/performance.py` | Metric computation |
| `generate_parameter_sensitivity_report()` | `eda/parameter_analysis.py` | Full sensitivity report |
| `plot_1d/2d/3d` on report | `metrics/plotting/parameter_plots.py` | Interactive Plotly figs |

## Config Override Pattern

```python
config = load_config()

from dataclasses import replace
# config = replace(config, tickers=[Ticker.ES])
# config = replace(config, bias_spec={...})

STABILITY_THRESHOLD = 0.8
TOP_K = 3
METRIC_COL = "sortino"
N_BINS = config.binning_params.bin_counts[0]  # continuous only
```

## Out of Scope

- Permutation testing (separate pipeline step)
- Walkforward analysis (separate pipeline step)
- Auto-saving outputs (optional/manual via commented cells)
- Modifying any existing `.py` files
