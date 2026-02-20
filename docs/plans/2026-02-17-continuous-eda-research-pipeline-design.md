# Design: Continuous Binning EDA Research Pipeline

**Date:** 2026-02-17
**Branch:** setup-feature-val
**Status:** Approved

---

## Problem

Researchers need a repeatable, config-driven way to run the full continuous-feature EDA pipeline across a parameter grid (e.g. RSI lookbacks 2–10) and inspect per-param-combo results. The existing EDA machinery is fully implemented; what's missing is the research harness that wires it together and outputs results to disk.

A secondary goal is to ensure integration tests exercise exactly the same codepath as the research script — so a regression in the pipeline is caught immediately.

---

## Approach: Config-Driven Pipeline Function (Option B)

- `config.py` is the single edit point for the researcher
- `pipeline.py` exports `run_continuous_eda_pipeline(config, output_dir)` — the reusable core
- `run_eda.py` is a thin `__main__` entry point
- Integration tests import and call `run_continuous_eda_pipeline` with a minimal 1-combo config

---

## Files

### `feature_research/continuous_binning/config.py`
- `ResearchConfig` dataclass: `tickers`, `start`, `end`, `bias_spec`, `target_col`, `strategy`, `use_cache`, `populate_cache`
- `reports_dir` property: `feature_research/continuous_binning/results/{module_name}/`
- `load_config() -> ResearchConfig` returns default config (RSI lookbacks 2–10, ES/NQ/YM/RTY, 2000–2024, log_return, long-short)

### `feature_research/continuous_binning/data_loader.py`
- `populate_cache_if_needed(config: ResearchConfig) -> None`
  - Calls `CacheManager(candle_dir=...).populate_cache(bias_node_specs=..., tickers=..., ...)`
  - Only runs if `config.populate_cache is True`
- `load_param_combos(config: ResearchConfig, ticker_override=None) -> list[ParamComboData]`
  - Expands param grid from bias_spec
  - Calls `extract_features_for_bias_node` per combo (or once for all combos if the extractor supports it)
  - Returns list of `(param_combo: dict, feature: pd.Series, target: pd.Series, feature_col: str)`
- `expand_bias_specs(bias_spec: dict) -> list[dict]` — same helper as in integration tests

### `feature_research/continuous_binning/pipeline.py`
- `run_continuous_eda_pipeline(config: ResearchConfig, output_dir: Path) -> dict[str, Path]`
  1. Calls `populate_cache_if_needed(config)` if needed
  2. For each param combo (expanded from bias_spec):
     a. Calls `load_param_combos` for that combo across all config tickers (concatenated)
     b. Builds `EDAMetadata` and `EDAConfig`
     c. Calls `run_eda_for_continuous_feature(feature, target, timestamps, metadata, config_obj)`
     d. Calls `save_eda_report(report, output_dir / param_label, overwrite=True)`
     e. Prints per-combo summary (is_viable, pearson, kendall_tau, decile trend)
  3. Returns `{param_label: report_path}` dict
- `param_combo_label(combo: dict) -> str` — human-readable folder name, e.g. `lookback_5`

### `feature_research/continuous_binning/run_eda.py`
```python
if __name__ == "__main__":
    config = load_config()
    results = run_continuous_eda_pipeline(config, config.reports_dir)
    print(f"\nEDA complete. {len(results)} param combos written to {config.reports_dir}")
```

### `feature_research/continuous_binning/__init__.py`
- Re-exports `load_config`, `run_continuous_eda_pipeline` for use in tests

---

## Results Directory Structure

```
feature_research/continuous_binning/results/
└── rsi/
    ├── lookback_2/
    │   ├── metadata.json
    │   ├── common_stats.json
    │   ├── feature_stats.json
    │   ├── diagnostics.json
    │   └── plots/
    │       ├── time_series_fig.png
    │       ├── rolling_corr_fig.png
    │       ├── rolling_obj_fig.png
    │       ├── decile_plot_fig.png
    │       ├── histogram_fig.png
    │       ├── qq_plot_fig.png
    │       └── kde_fig.png
    ├── lookback_3/
    └── ...
```

---

## Integration Test Refactor

### New file: `tests/integration/feature_validator/test_continuous_eda_pipeline.py`

Imports `run_continuous_eda_pipeline` and `load_config` from `feature_research.continuous_binning`.

Two tests:
1. `test_continuous_eda_pipeline_smoke()` — minimal RSI-5, ES, 2020–2023, 1 param combo, tempdir output. Asserts no exception + output dir exists with expected JSON/plot files.
2. (Optional) `test_continuous_eda_pipeline_multi_combo()` — RSI lookbacks [5, 10], ES, 2020–2023. Asserts 2 result subdirs created.

### `tests/integration/feature_validator/test_eda_pipeline.py`

Remove `test_common_eda_continuous` (replaced by `test_continuous_eda_pipeline.py`). Keep `test_rule_based_eda` until a dedicated rule-based script is built.

---

## Data Contracts

All inputs/outputs flow through existing types:
- `ResearchConfig` (new) — research-specific config
- `EDAMetadata`, `EDAConfig` — existing frozen dataclasses from `feature_selection/eda/eda_dataclasses.py`
- `ContinuousEDAReport` — existing frozen dataclass

---

## Out of Scope (this PR)

- Rule-based EDA research script (`feature_research/rule-based/`)
- Permutation testing research script
- Parameter sensitivity research script
- Notebook version

---

## Success Criteria

- `python feature_research/continuous_binning/run_eda.py` runs end-to-end with default config
- `results/rsi/lookback_N/` directories created with 4 JSON files + 7 plots each
- `pytest tests/integration/feature_validator/test_continuous_eda_pipeline.py -v` passes
- Old `test_common_eda_continuous` test removed from `test_eda_pipeline.py`
