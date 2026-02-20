# Design: Rule-Based EDA Research Pipeline

**Date:** 2026-02-17
**Branch:** setup-feature-val
**Status:** Approved

---

## Problem

Researchers need a repeatable, config-driven way to run the full rule-based feature EDA pipeline across a parameter grid (e.g. RSI signal with different `rsi_period` values) and inspect per-param-combo results. The existing rule-based EDA machinery is fully implemented; what's missing is the research harness that wires it together and outputs results to disk.

A secondary goal is to ensure integration tests exercise exactly the same codepath as the research script — so a regression in the pipeline is caught immediately.

---

## Approach: Full Mirror of `continuous_binning/` (Option A)

- `feature_research/rule_based/` is a self-contained package (rename from `rule-based/` — hyphens are not valid Python package names)
- `config.py` is the single edit point for the researcher
- `pipeline.py` exports `run_rule_based_eda_pipeline(config, output_dir)` — the reusable core
- `run_eda.py` is a thin `__main__` entry point
- `data_loader.py` copies helpers from `continuous_binning/data_loader.py` (YAGNI — 60 lines, no shared abstraction needed)
- Integration tests import and call `run_rule_based_eda_pipeline` with a minimal 1-combo config

---

## Files

### `feature_research/rule_based/__init__.py`
Re-exports `RuleBasedResearchConfig`, `load_config`, `run_rule_based_eda_pipeline`.

### `feature_research/rule_based/config.py`
- `RuleBasedResearchConfig` frozen dataclass: `tickers`, `start`, `end`, `bias_spec`, `target_col`, `strategy`, `use_cache`, `populate_cache`, `reports_dir`
- `reports_dir` property: `feature_research/rule_based/results/{module_name}/`
- `load_config() -> RuleBasedResearchConfig` returns default config:
  - Module: `rsi_signal`
  - Parameter grid: `rsi_period: [2, 3, 5, 7]`, fixed `oversold=25.0`, `overbought=65.0`, `strategy_mode="long"`, `exit_policy="threshold_or_bars"`, `exit_bars=5`
  - Tickers: ES, NQ
  - Dates: 2020–2024
  - Target: `log_return`
  - Strategy: `long`

### `feature_research/rule_based/data_loader.py`
Same four helpers as `continuous_binning/data_loader.py`:
- `expand_bias_specs(bias_spec) -> list[dict]`
- `param_combo_label(combo) -> str`
- `populate_cache_if_needed(config) -> None`
- `load_features_for_combo(single_spec, config) -> tuple[pd.Series, pd.Series, str] | None`

### `feature_research/rule_based/pipeline.py`
- `run_rule_based_eda_pipeline(config, output_dir) -> dict[str, Path]`
  1. `populate_cache_if_needed(config)`
  2. For each param combo:
     a. `load_features_for_combo` → `(feature, target, feature_col)`
     b. Build `EDAMetadata` and `EDAConfig(rolling_window=..., bootstrap_iterations=500)`
     c. `run_eda_for_rule_based_feature(feature, target, timestamps, metadata, config_obj)`
     d. `save_eda_report(report, output_dir / param_label, overwrite=True)`
     e. Print per-combo summary (rule-based specific):
        ```
        [rsi_period_2] n=1,000  viable=VIABLE  L[-1]: sharpe=-0.12  L[0]: sharpe=+0.05  L[1]: sharpe=+0.31  warnings=0
        ```
  3. Returns `{param_label: report_path}` dict

### `feature_research/rule_based/run_eda.py`
```python
if __name__ == "__main__":
    config = load_config()
    results = run_rule_based_eda_pipeline(config, config.reports_dir)
    print(f"\nEDA complete. {len(results)} param combos written to {config.reports_dir}")
```

---

## Results Directory Structure

```
feature_research/rule_based/results/
└── rsi_signal/
    ├── rsi_period_2/
    │   ├── {feature_col}/
    │   │   └── {param_hash}/
    │   │       ├── metadata.json
    │   │       ├── common_stats.json
    │   │       ├── feature_stats.json
    │   │       ├── diagnostics.json
    │   │       └── plots/
    │   │           ├── time_series_fig.png
    │   │           ├── rolling_corr_fig.png
    │   │           ├── rolling_obj_fig.png
    │   │           ├── level_plot_fig.png
    │   │           └── transition_heatmap_fig.png
    ├── rsi_period_3/
    └── ...
```

---

## Integration Test Refactor

### New file: `tests/integration/feature_validator/test_rule_based_eda_pipeline.py`

Imports `run_rule_based_eda_pipeline` and `RuleBasedResearchConfig` from `feature_research.rule_based`.

Two tests:
1. `test_rule_based_eda_pipeline_smoke()` — `rsi_period=2`, ES, 2020–2023, 1 param combo, tempdir output. Asserts no exception + 4 JSON files + 5 plots.
2. (Optional) `test_rule_based_eda_pipeline_multi_combo()` — `rsi_period=[2, 3]`, ES, 2020–2023. Asserts 2 result subdirs.

### `tests/integration/feature_validator/test_eda_pipeline.py`

Remove `test_rule_based_eda` (superseded by `test_rule_based_eda_pipeline.py`). File becomes empty and can be deleted.

---

## Data Contracts

- `RuleBasedResearchConfig` (new) — research-specific config
- `EDAMetadata`, `EDAConfig` — existing frozen dataclasses from `feature_selection/eda/eda_dataclasses.py`
- `RuleBasedEDAReport` — existing frozen dataclass
- Rule-based plots: `level_plot_fig.png`, `transition_heatmap_fig.png` (5 total including 3 common)

---

## Out of Scope (this PR)

- Permutation testing research script
- Parameter sensitivity research script
- Notebook version

---

## Success Criteria

- `python feature_research/rule_based/run_eda.py` runs end-to-end with default config
- `results/rsi_signal/rsi_period_N/` directories created with 4 JSON files + 5 plots each
- `pytest tests/integration/feature_validator/test_rule_based_eda_pipeline.py -v` passes
- Old `test_rule_based_eda` removed from `test_eda_pipeline.py`
- `test_eda_pipeline.py` deleted (now empty)
