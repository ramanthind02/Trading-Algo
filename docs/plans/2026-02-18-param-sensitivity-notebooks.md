# Parameter Sensitivity Notebooks Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create two Jupyter notebooks (`feature_research/continuous_binning/param_sensitivity.ipynb` and `feature_research/rule_based/param_sensitivity.ipynb`) that integrate the existing parameter sensitivity infrastructure with the live research pipeline.

**Architecture:** Each notebook is a thin orchestration shell: import config → override cell → loop over param combos using cache-backed extraction → build `results_df` → call `generate_parameter_sensitivity_report()` → display interactive Plotly figures. The continuous binning notebook fits `ContinuousBinningModel` per combo; the rule-based notebook computes metric directly from `feature × target` with no model fitting.

**Tech Stack:** Jupyter notebooks (`.ipynb`), `eda/parameter_analysis.py`, `metrics/plotting/parameter_plots.py`, `feature_research/{continuous_binning,rule_based}/{config,data_loader}.py`, `feature_selection/base_models/continuous_binning.py`, `metrics/performance.py`

---

## Key Infrastructure Reference

Before starting, understand these existing functions:

| Function | Location | What it does |
|----------|----------|-------------|
| `load_config()` | `feature_research/continuous_binning/config.py` | Returns frozen `ResearchConfig` dataclass |
| `expand_bias_specs(bias_spec)` | `feature_research/continuous_binning/data_loader.py` | Expands list-valued params into one spec per combo |
| `load_features_for_combo(spec, config)` | `feature_research/continuous_binning/data_loader.py` | Returns `(feature_series, target_series, feature_col)` or `None` |
| `populate_cache_if_needed(config)` | `feature_research/continuous_binning/data_loader.py` | Runs cache population if `config.populate_cache=True` |
| `param_combo_label(combo)` | `feature_research/continuous_binning/data_loader.py` | Returns human-readable label e.g. `"lookback_5"` |
| `ContinuousBinningModel(n_bins)` | `feature_selection/base_models/continuous_binning.py` | Binning model: `.fit(X, y)`, `.predict(X, strategy='long')` → bool array |
| `SortinoRatio(annualization_factor)` | `metrics/performance.py` | `.compute(returns_array)` → float |
| `generate_parameter_sensitivity_report(results_df, param_names, metric_col, stability_threshold, top_k)` | `eda/parameter_analysis.py` | Full pipeline: smooth → stable regions → plots → report |

Rule-based data loader (`feature_research/rule_based/data_loader.py`) has the **identical API** — same function names, same signatures.

---

## Critical: Deriving `param_names` from config

Only **list-valued** params in `bias_spec["params"]` are varying parameters. Fixed params (single values) should not get their own `paramK_value` column.

```python
# Correct way to get varying param names:
varying_params = [
    k for k, v in config.bias_spec["params"].items()
    if isinstance(v, list) and len(v) > 1
]
# Example for continuous: ["lookback"]  (only rsi lookback varies)
# Example for rule-based: ["rsi_period"]  (oversold, overbought etc are fixed)
```

If **all** params are lists (full 2D/3D grid), `varying_params` will capture all of them.

---

## Task 1: Continuous Binning Parameter Sensitivity Notebook

**File:** Create `feature_research/continuous_binning/param_sensitivity.ipynb`

### Step 1: Create the notebook skeleton

Use the `Write` tool to create the notebook at `feature_research/continuous_binning/param_sensitivity.ipynb` with the following JSON structure. Each cell is documented below.

**Cell 0 — markdown title:**
```markdown
# Continuous Binning — Parameter Sensitivity Analysis

Sweeps the bias node parameter grid defined in `config.py`, fits a `ContinuousBinningModel`
for each combo, and runs `generate_parameter_sensitivity_report()` to identify stable
parameter regions and produce interactive visualizations.

**Workflow:**
1. Edit `config.py` (or override inline below) to set tickers, date range, bias_spec
2. Run all cells (Kernel → Restart & Run All)
3. Interact with the Plotly figures in Section 5
```

**Cell 1 — code: Setup**
```python
import sys, os
sys.path.insert(0, os.path.abspath('../..'))

import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import replace

from feature_research.continuous_binning.config import load_config
from feature_research.continuous_binning.data_loader import (
    expand_bias_specs,
    load_features_for_combo,
    populate_cache_if_needed,
    param_combo_label,
)
from feature_selection.base_models.continuous_binning import ContinuousBinningModel
from metrics.performance import SortinoRatio
from eda.parameter_analysis import generate_parameter_sensitivity_report
from utils.enums import Ticker, TimeFrame

print("Imports OK")
```

**Cell 2 — markdown:**
```markdown
## 1. Configuration

Edit `config.py` for persistent changes, or use the override cell below for one-off runs.
`dataclasses.replace()` is used because `ResearchConfig` is a frozen dataclass.
```

**Cell 3 — code: Config + Override**
```python
# --- Load defaults from config.py ---
config = load_config()

# ============================================================
# OVERRIDE CELL — edit here to customise without touching config.py
# ============================================================

# config = replace(config, tickers=[Ticker.ES])
# config = replace(config, start=datetime(2015, 1, 1), end=datetime(2024, 12, 31))
# config = replace(config,
#     bias_spec={
#         "module_name": "rsi",
#         "timeframes": [TimeFrame.D],
#         "params": {"lookback": [5, 10, 14, 20, 30]},
#     }
# )

# --- Param sensitivity knobs ---
STABILITY_THRESHOLD = 0.8
TOP_K = 3
METRIC_COL = "sortino"
N_BINS = config.binning_params.bin_counts[0]

# ============================================================

# Derive varying param names (list-valued params only)
varying_params = [
    k for k, v in config.bias_spec["params"].items()
    if isinstance(v, list) and len(v) > 1
]
if not varying_params:
    # Fallback: use all params if none have multiple values
    varying_params = list(config.bias_spec["params"].keys())

print(f"Module   : {config.bias_spec['module_name']}")
print(f"Tickers  : {[t.name for t in config.tickers]}")
print(f"Period   : {config.start.date()} → {config.end.date()}")
print(f"Varying  : {varying_params}")
print(f"N_BINS   : {N_BINS}  |  METRIC: {METRIC_COL}  |  THRESHOLD: {STABILITY_THRESHOLD}")
```

**Cell 4 — markdown:**
```markdown
## 2. Feature Extraction

Loads feature + target data for every param combo. Uses the cache if `config.use_cache=True`.
Set `config.populate_cache=True` to auto-populate missing cache entries.
```

**Cell 5 — code: Feature Extraction**
```python
populate_cache_if_needed(config)

expanded_specs = expand_bias_specs(config.bias_spec)
print(f"Parameter grid: {len(expanded_specs)} combinations\n")

combo_data = {}
for spec in expanded_specs:
    result = load_features_for_combo(spec, config)
    if result is not None:
        feature, target, feature_col = result
        combo_data[param_combo_label(spec["params"])] = {
            "feature": feature,
            "target": target,
            "params": spec["params"],
        }
    else:
        print(f"  SKIP: {spec['params']}")

print(f"\nLoaded {len(combo_data)}/{len(expanded_specs)} combos successfully")
```

**Cell 6 — markdown:**
```markdown
## 3. Metric Grid

Fits `ContinuousBinningModel` for each combo and computes the Sortino ratio on selected
returns. Builds `results_df` with `param{K}_value` columns required by
`generate_parameter_sensitivity_report()`.
```

**Cell 7 — code: Metric Grid**
```python
metric = SortinoRatio(annualization_factor=252)
rows = []

for spec in expanded_specs:
    label = param_combo_label(spec["params"])
    if label not in combo_data:
        continue

    data = combo_data[label]
    feature = data["feature"]
    target = data["target"]

    try:
        model = ContinuousBinningModel(n_bins=N_BINS)
        model.fit(feature, target)
        signals = model.predict(feature, strategy="long")
        signals = np.asarray(signals).astype(bool).flatten()
        selected_returns = target.values[signals]

        if len(selected_returns) < 5:
            selected_returns = target.values

        metric_value = metric.compute(selected_returns)

        row = {
            f"param{k+1}_value": spec["params"][key]
            for k, key in enumerate(varying_params)
        }
        row[METRIC_COL] = metric_value
        rows.append(row)

    except Exception as e:
        print(f"  ERROR [{label}]: {e}")

results_df = pd.DataFrame(rows)
print(f"Grid results: {len(results_df)} rows")
results_df
```

**Cell 8 — markdown:**
```markdown
## 4. Parameter Sensitivity Report

Runs: neighbor smoothing → stable region detection → recommendations.
```

**Cell 9 — code: Generate Report**
```python
report = generate_parameter_sensitivity_report(
    results_df=results_df,
    param_names=varying_params,
    metric_col=METRIC_COL,
    stability_threshold=STABILITY_THRESHOLD,
    top_k=TOP_K,
)

print(f"Module              : {config.bias_spec['module_name']}")
print(f"Metric              : {report.metric_name}")
print(f"Grid size           : {report.n_parameter_combinations}")
print(f"Stable regions      : {report.n_stable_regions}")
print(f"Mean stability ratio: {report.mean_stability_ratio:.3f}")
print(f"% stable combos     : {report.pct_stable_combinations:.1%}")
print(f"\nTop-{TOP_K} recommendations:")
for i, combo in enumerate(report.top_k_combinations, 1):
    params_str = ", ".join(
        f"{name}={val}" for name, val in zip(varying_params, combo)
    )
    print(f"  {i}. {params_str}")
```

**Cell 10 — markdown:**
```markdown
## 5. Interactive Visualizations

The appropriate plot is selected automatically based on grid dimensionality:
- 1D grid → raw + smoothed line with shaded stable regions
- 2D grid → heatmap with stability contour overlay
- 3D+ grid → heatmap slices with dropdown + slider controls
```

**Cell 11 — code: Plots**
```python
if report.plot_1d is not None:
    report.plot_1d.show()

if report.plot_2d is not None:
    report.plot_2d.show()

if report.plot_3d is not None:
    report.plot_3d.show()
```

**Cell 12 — markdown:**
```markdown
## 6. Optional Save

Uncomment cells below to save plots as interactive HTML or grid results as parquet.
```

**Cell 13 — code: Optional Save**
```python
# --- OPTIONAL: Save outputs (uncomment to enable) ---

# from pathlib import Path
# save_dir = config.reports_dir / "param_sensitivity"
# save_dir.mkdir(parents=True, exist_ok=True)

# if report.plot_1d is not None:
#     report.plot_1d.write_html(save_dir / f"param_sensitivity_1d_{config.bias_spec['module_name']}.html")
#     print(f"Saved 1D plot → {save_dir}")

# if report.plot_2d is not None:
#     report.plot_2d.write_html(save_dir / f"param_sensitivity_2d_{config.bias_spec['module_name']}.html")
#     print(f"Saved 2D plot → {save_dir}")

# if report.plot_3d is not None:
#     report.plot_3d.write_html(save_dir / f"param_sensitivity_3d_{config.bias_spec['module_name']}.html")
#     print(f"Saved 3D plot → {save_dir}")

# report.grid_results.to_parquet(save_dir / "param_sensitivity_grid.parquet")
# print(f"Saved grid results → {save_dir / 'param_sensitivity_grid.parquet'}")
```

### Step 2: Verify the notebook structure

After creating the file, confirm it can be opened in Jupyter and all cells are present in order.
Run `jupyter nbconvert --to script feature_research/continuous_binning/param_sensitivity.ipynb --stdout` to validate the cell structure is intact.

### Step 3: Commit

```bash
git add feature_research/continuous_binning/param_sensitivity.ipynb
git commit -m "feat(research): add continuous binning param sensitivity notebook"
```

---

## Task 2: Rule-Based Parameter Sensitivity Notebook

**File:** Create `feature_research/rule_based/param_sensitivity.ipynb`

This notebook is structurally identical to Task 1 with two differences:
1. Imports come from `feature_research.rule_based.config` and `feature_research.rule_based.data_loader`
2. Section 3 (Metric Grid) computes metric directly from `feature × target` — no model fitting

### Step 1: Create the notebook

**Cell 0 — markdown title:**
```markdown
# Rule-Based Features — Parameter Sensitivity Analysis

Sweeps the rule-based bias node parameter grid defined in `config.py`, computes the
Sortino ratio directly from feature signals (no model fitting needed), and runs
`generate_parameter_sensitivity_report()` to identify stable parameter regions.

**Workflow:**
1. Edit `config.py` (or override inline below) to set tickers, date range, bias_spec
2. Run all cells (Kernel → Restart & Run All)
3. Interact with the Plotly figures in Section 5
```

**Cell 1 — code: Setup**
```python
import sys, os
sys.path.insert(0, os.path.abspath('../..'))

import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import replace

from feature_research.rule_based.config import load_config
from feature_research.rule_based.data_loader import (
    expand_bias_specs,
    load_features_for_combo,
    populate_cache_if_needed,
    param_combo_label,
)
from metrics.performance import SortinoRatio
from eda.parameter_analysis import generate_parameter_sensitivity_report
from utils.enums import Ticker, TimeFrame

print("Imports OK")
```

**Cell 2 — markdown:**
```markdown
## 1. Configuration
```

**Cell 3 — code: Config + Override**
```python
# --- Load defaults from config.py ---
config = load_config()

# ============================================================
# OVERRIDE CELL — edit here to customise without touching config.py
# ============================================================

# config = replace(config, tickers=[Ticker.ES])
# config = replace(config, start=datetime(2015, 1, 1), end=datetime(2024, 12, 31))
# config = replace(config,
#     bias_spec={
#         "module_name": "rsi_signal",
#         "timeframes": [TimeFrame.D],
#         "params": {
#             "rsi_period": [2, 3, 5, 7, 10, 14],
#             "oversold": 25.0,
#             "overbought": 65.0,
#             "strategy_mode": "long",
#             "exit_policy": "threshold_or_bars",
#             "exit_bars": 5,
#         },
#     }
# )

# --- Param sensitivity knobs ---
STABILITY_THRESHOLD = 0.8
TOP_K = 3
METRIC_COL = "sortino"

# ============================================================

# Derive varying param names (list-valued params only)
varying_params = [
    k for k, v in config.bias_spec["params"].items()
    if isinstance(v, list) and len(v) > 1
]
if not varying_params:
    varying_params = list(config.bias_spec["params"].keys())

print(f"Module   : {config.bias_spec['module_name']}")
print(f"Tickers  : {[t.name for t in config.tickers]}")
print(f"Period   : {config.start.date()} → {config.end.date()}")
print(f"Varying  : {varying_params}")
print(f"METRIC: {METRIC_COL}  |  THRESHOLD: {STABILITY_THRESHOLD}")
```

**Cell 4 — markdown:**
```markdown
## 2. Feature Extraction
```

**Cell 5 — code: Feature Extraction**

*(Identical to the continuous binning notebook — same data loader API)*

```python
populate_cache_if_needed(config)

expanded_specs = expand_bias_specs(config.bias_spec)
print(f"Parameter grid: {len(expanded_specs)} combinations\n")

combo_data = {}
for spec in expanded_specs:
    result = load_features_for_combo(spec, config)
    if result is not None:
        feature, target, feature_col = result
        combo_data[param_combo_label(spec["params"])] = {
            "feature": feature,
            "target": target,
            "params": spec["params"],
        }
    else:
        print(f"  SKIP: {spec['params']}")

print(f"\nLoaded {len(combo_data)}/{len(expanded_specs)} combos successfully")
```

**Cell 6 — markdown:**
```markdown
## 3. Metric Grid

Rule-based features are binary signals (1 = trade, 0 = no trade).
Metric is computed directly on returns where the signal fires — no model fitting required.
```

**Cell 7 — code: Metric Grid (rule-based — no model fitting)**
```python
metric = SortinoRatio(annualization_factor=252)
rows = []

for spec in expanded_specs:
    label = param_combo_label(spec["params"])
    if label not in combo_data:
        continue

    data = combo_data[label]
    feature = data["feature"]
    target = data["target"]

    try:
        # Rule-based: feature IS the signal — no model fitting
        signals = feature.values == 1
        selected_returns = target.values[signals]

        if len(selected_returns) < 5:
            selected_returns = target.values

        metric_value = metric.compute(selected_returns)

        row = {
            f"param{k+1}_value": spec["params"][key]
            for k, key in enumerate(varying_params)
        }
        row[METRIC_COL] = metric_value
        rows.append(row)

    except Exception as e:
        print(f"  ERROR [{label}]: {e}")

results_df = pd.DataFrame(rows)
print(f"Grid results: {len(results_df)} rows")
results_df
```

**Cells 8–13:** Identical to Task 1 (Generate Report, Plots, Optional Save cells).
Use the exact same markdown and code as Task 1 Cells 8–13, with one change in Cell 13:
replace `continuous_binning` path reference with `rule_based`:
```python
# save_dir = config.reports_dir / "param_sensitivity"
```
This already uses `config.reports_dir` which for rule-based points to `feature_research/rule_based/results/{module_name}/`.

### Step 2: Verify the notebook structure

```bash
jupyter nbconvert --to script feature_research/rule_based/param_sensitivity.ipynb --stdout
```

### Step 3: Commit

```bash
git add feature_research/rule_based/param_sensitivity.ipynb
git commit -m "feat(research): add rule-based param sensitivity notebook"
```

---

## Verification

After both notebooks are created:

1. **Smoke test continuous binning** (with cache available):
   - Open `feature_research/continuous_binning/param_sensitivity.ipynb`
   - Run Cell 1 (imports) → expect `"Imports OK"`
   - Run Cell 3 (config) → expect config summary printed
   - Run Cell 5 (extraction) → expect `"Loaded N/N combos successfully"`
   - Run Cell 7 (metric grid) → expect DataFrame with `param1_value` + `sortino` columns
   - Run Cell 9 (report) → expect stable region count + top-K printed
   - Run Cell 11 (plots) → expect interactive Plotly figure rendered inline

2. **Smoke test rule-based** (with cache available):
   - Same flow, verify Cell 7 output has `param1_value` = rsi_period values

3. **Verify 2D grid works** by temporarily overriding `bias_spec` to include 2 list-valued params.
   Expect Cell 11 to render a 2D heatmap instead of 1D line chart.
