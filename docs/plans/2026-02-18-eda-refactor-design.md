# EDA Refactor Design — 2026-02-18

## Overview

Refactor the shared EDA pipeline (`feature_selection/eda/`) in two passes:

- **Pass 1**: Remove irrelevant or redundant stats/plots
- **Pass 2**: Add IC decay, feature ACF, and quintile spread (TDD)

Applies to both the `feature_research/continuous_binning` and `feature_research/rule_based` pipelines.

---

## Approach

Two-pass sequential delivery. Pass 1 ships as a clean removal diff with all tests passing before Pass 2 begins. Pass 2 follows TDD: tests written before implementation.

---

## Pass 1 — Removals

### What is removed and why

| Item | Location | Reason |
|---|---|---|
| `MonotonicityTest` dataclass | `eda_dataclasses.py` | Replaced by quintile spread; tau redundant with existing trend label |
| `kendall` field on `CorrelationAnalysis` | `eda_dataclasses.py` | Redundant alongside Spearman; user confirmed full removal |
| `rolling_objective: pd.Series` on `CommonEDAStats` | `eda_dataclasses.py` | Pre-binning rolling Sharpe on raw feature values is not interpretable |
| `rolling_obj_fig` on `CommonEDAPlots` | `eda_dataclasses.py` | Plot of the above |
| `qq_plot_fig` on `ContinuousEDAPlots` | `eda_dataclasses.py` | Normality irrelevant for trading features |
| `kde_fig` on `ContinuousEDAPlots` | `eda_dataclasses.py` | Redundant with histogram |
| `TransitionMatrix` dataclass | `eda_dataclasses.py` | Placeholder `np.eye` — not real data; deferred to later |
| `transition_heatmap_fig` on `RuleBasedEDAPlots` | `eda_dataclasses.py` | Plot of the above |
| `transition_matrix` on `RuleBasedEDAStats` | `eda_dataclasses.py` | Field for the above |
| Target returns subplot | `common_eda.py` `create_common_eda_plots` | Same target across all features for a ticker — noise |

### File-level changes

**`eda_dataclasses.py`**
- Remove `MonotonicityTest`
- Remove `kendall` from `CorrelationAnalysis`
- Remove `rolling_objective` from `CommonEDAStats`
- Remove `rolling_obj_fig` from `CommonEDAPlots`
- Remove `qq_plot_fig`, `kde_fig` from `ContinuousEDAPlots`
- Remove `MonotonicityTest` from `ContinuousEDAStats`
- Remove `TransitionMatrix`, `transition_heatmap_fig`, `transition_matrix` from rule-based dataclasses

**`common_eda.py`**
- Delete `compute_rolling_objective` function
- Remove `rolling_obj` param from `create_common_eda_plots`; remove its figure creation block
- Change `time_series_fig` from 2 subplots to 1 (feature only)

**`continuous_eda.py`**
- Delete `compute_monotonicity_test` function
- Remove QQ and KDE figure creation blocks from `create_continuous_eda_plots`
- Remove `dist_diagnostics` param from `create_continuous_eda_plots` if no longer needed

**`rule_based_eda.py`**
- Delete `compute_transition_matrix` function
- Remove `transition_heatmap_fig` generation from `create_rule_based_eda_plots`

**`eda_reporter.py`**
- Remove `compute_monotonicity_test` call; remove `monotonicity_test` from `ContinuousEDAStats` construction
- Remove `compute_rolling_objective` call; remove `rolling_objective` from `CommonEDAStats` construction
- Remove `compute_transition_matrix` call; remove `transition_matrix` from `RuleBasedEDAStats` construction
- Remove `rolling_obj_fig` from `create_common_eda_plots` call
- Remove kendall tau diagnostic flag from `compute_diagnostic_flags`
- Update `_common_stats_from_json`, `_continuous_stats_from_json`, `_rule_stats_from_json`
- Update `_common_plots_from_dir`, `_continuous_plots_from_dir`, `_rule_plots_from_dir`

**Tests**
- `test_common_eda.py` — remove assertions for `kendall`, `rolling_objective`, `rolling_obj_fig`, target subplot
- `test_continuous_eda.py` — remove assertions for `MonotonicityTest`, `qq_plot_fig`, `kde_fig`
- `test_rule_based_eda.py` — remove assertions for `TransitionMatrix`, `transition_heatmap_fig`
- `test_eda_reporter.py` — update save/load round-trip assertions to match new shapes
- Integration tests — update field presence assertions

---

## Pass 2 — Additions (TDD)

### New dataclasses (`eda_dataclasses.py`)

```python
@dataclass(frozen=True)
class ICDecay:
    horizons: list[int]               # e.g. [1, 5, 10, 21]
    ic_by_horizon: dict[int, float]   # horizon → Spearman IC

@dataclass(frozen=True)
class FeatureACF:
    lags: np.ndarray        # shape (max_acf_lag,)
    acf_values: np.ndarray  # shape (max_acf_lag,)
    pacf_values: np.ndarray # shape (max_acf_lag,)

@dataclass(frozen=True)
class QuintileSpread:       # continuous only
    quintile_means: np.ndarray  # shape (5,), Q1..Q5 mean return
    spread: float               # Q5_mean - Q1_mean
```

### Dataclass field additions

- `CommonEDAStats` gains: `ic_decay: ICDecay`, `feature_acf: FeatureACF`
- `CommonEDAPlots` gains: `ic_decay_fig: Figure`, `acf_fig: Figure`
- `ContinuousEDAStats` gains: `quintile_spread: QuintileSpread`
- `ContinuousEDAPlots` gains: `quintile_spread_fig: Figure`
- `EDAConfig` gains: `ic_horizons: tuple[int, ...] = (1, 5, 10, 21)`, `max_acf_lag: int = 20`

### New compute functions

**`common_eda.py`**

`compute_ic_decay(feature, target, horizons) -> ICDecay`
- For each horizon `h`, compute `spearman_corr(feature, target.shift(-h))` on aligned, non-NaN pairs
- Returns `ICDecay(horizons=horizons, ic_by_horizon={h: ic, ...})`

`compute_feature_acf(feature, max_lag) -> FeatureACF`
- Drop NaNs, use `statsmodels.tsa.stattools.acf(nlags=max_lag, fft=True)` and `pacf(nlags=max_lag)`
- Returns `FeatureACF(lags=np.arange(1, max_lag+1), acf_values=..., pacf_values=...)`

**`continuous_eda.py`**

`compute_quintile_spread(feature, target) -> QuintileSpread`
- 5-quantile cut (`pd.qcut(q=5)`), per-quintile mean return
- `spread = quintile_means[4] - quintile_means[0]`

### New plots

**`common_eda.py` — `create_common_eda_plots`** gains two new figures:
- `ic_decay_fig`: bar chart of IC at each horizon; zero reference line; x-axis = horizon label
- `acf_fig`: 2-subplot correlogram — top ACF, bottom PACF, with 95% confidence bands at `±1.96/√n`

**`continuous_eda.py` — `create_continuous_eda_plots`** gains:
- `quintile_spread_fig`: bar chart of quintile mean returns (Q1–Q5) with spread annotated as text

### Reporter updates (`eda_reporter.py`)

- Wire `compute_ic_decay` and `compute_feature_acf` into `_build_common_stats_and_plots`
- Wire `compute_quintile_spread` into `run_eda_for_continuous_feature`
- Update JSON serialisation/deserialisation for new dataclasses
- Update `_common_plots_from_dir` and `_continuous_plots_from_dir` load paths

### Test strategy (TDD order)

1. Write failing unit tests for `compute_ic_decay`, `compute_feature_acf`, `compute_quintile_spread` using synthetic fixtures
2. Implement functions until tests pass
3. Write plot-existence tests (assert figure is a `Figure` instance, has correct axes count)
4. Update `test_eda_reporter.py` round-trip test to assert new fields survive save/load
5. Update integration test to assert `ic_decay`, `feature_acf`, `quintile_spread` are present in reports

---

## Affected Files Summary

| File | Pass 1 | Pass 2 |
|---|---|---|
| `feature_selection/eda/eda_dataclasses.py` | Remove 8 items | Add 3 dataclasses + 6 fields |
| `feature_selection/eda/common_eda.py` | Remove 2 functions, simplify plot | Add 2 functions + 2 plot blocks |
| `feature_selection/eda/continuous_eda.py` | Remove 1 function + 2 plot blocks | Add 1 function + 1 plot block |
| `feature_selection/eda/rule_based_eda.py` | Remove 1 function + 1 plot block | No changes |
| `feature_selection/eda/eda_reporter.py` | Remove 5 call sites + JSON paths | Add 3 call sites + JSON paths |
| `tests/unit-tests/validators/eda/test_common_eda.py` | Prune removed assertions | Add IC decay + ACF tests |
| `tests/unit-tests/validators/eda/test_continuous_eda.py` | Prune removed assertions | Add quintile spread tests |
| `tests/unit-tests/validators/eda/test_rule_based_eda.py` | Prune removed assertions | No new tests |
| `tests/validators/eda/test_eda_reporter.py` | Update round-trip assertions | Update round-trip assertions |
| `tests/integration/feature_validator/test_continuous_eda_pipeline.py` | Update field assertions | Update field assertions |
| `tests/integration/feature_validator/test_rule_based_eda_pipeline.py` | Update field assertions | No new assertions |
