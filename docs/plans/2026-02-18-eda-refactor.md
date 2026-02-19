# EDA Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove 10 redundant/irrelevant EDA items and add IC decay, feature ACF, and quintile spread across both EDA pipelines.

**Architecture:** Two-pass approach. Pass 1 removes dead code top-down through the dependency chain (dataclasses → compute modules → reporter → tests). Pass 2 adds new metrics using TDD (test first, then implement). The shared EDA layer lives in `feature_selection/eda/`; both `feature_research/continuous_binning` and `feature_research/rule_based` pipelines consume it transparently.

**Tech Stack:** Python, pandas, numpy, scipy, statsmodels (for ACF/PACF), matplotlib (Agg backend)

---

## PASS 1 — Removals

---

### Task 1: Strip removed items from `eda_dataclasses.py`

**Files:**
- Modify: `feature_selection/eda/eda_dataclasses.py`

**Step 1: Delete `MonotonicityTest` dataclass (lines 89–94)**

Remove the entire block:
```python
@dataclass(frozen=True)
class MonotonicityTest:
    """Kendall's tau monotonicity test over bin means."""
    kendall_tau: float
    p_value: float
    is_monotonic: bool            # |tau| > 0.5 and p < 0.05
```

**Step 2: Remove `kendall` field from `CorrelationAnalysis` (line 47)**

Before:
```python
@dataclass(frozen=True)
class CorrelationAnalysis:
    """Feature-target correlation at various lags."""
    pearson: float
    spearman: float
    kendall: float
    lagged_correlations: dict[int, float]
```
After:
```python
@dataclass(frozen=True)
class CorrelationAnalysis:
    """Feature-target correlation at various lags."""
    pearson: float
    spearman: float
    lagged_correlations: dict[int, float]
```

**Step 3: Remove `rolling_obj_fig` from `CommonEDAPlots` and `rolling_objective` from `CommonEDAStats`**

Before (`CommonEDAPlots`):
```python
@dataclass(frozen=True)
class CommonEDAPlots:
    time_series_fig: Figure
    rolling_corr_fig: Figure
    rolling_obj_fig: Figure
```
After:
```python
@dataclass(frozen=True)
class CommonEDAPlots:
    time_series_fig: Figure
    rolling_corr_fig: Figure
```

Before (`CommonEDAStats`):
```python
@dataclass(frozen=True)
class CommonEDAStats:
    feature_stats: DescriptiveStats
    target_stats: DescriptiveStats
    temporal_stability: TemporalStability
    correlation_analysis: CorrelationAnalysis
    rolling_objective: pd.Series
```
After:
```python
@dataclass(frozen=True)
class CommonEDAStats:
    feature_stats: DescriptiveStats
    target_stats: DescriptiveStats
    temporal_stability: TemporalStability
    correlation_analysis: CorrelationAnalysis
```

**Step 4: Remove `qq_plot_fig` and `kde_fig` from `ContinuousEDAPlots`**

Before:
```python
@dataclass(frozen=True)
class ContinuousEDAPlots:
    decile_plot_fig: Figure
    histogram_fig: Figure
    qq_plot_fig: Figure
    kde_fig: Figure
```
After:
```python
@dataclass(frozen=True)
class ContinuousEDAPlots:
    decile_plot_fig: Figure
    histogram_fig: Figure
```

**Step 5: Remove `monotonicity_test` from `ContinuousEDAStats`**

Before:
```python
@dataclass(frozen=True)
class ContinuousEDAStats:
    decile_analysis: DecileAnalysis
    monotonicity_test: MonotonicityTest
    distribution_diagnostics: DistributionDiagnostics
```
After:
```python
@dataclass(frozen=True)
class ContinuousEDAStats:
    decile_analysis: DecileAnalysis
    distribution_diagnostics: DistributionDiagnostics
```

**Step 6: Delete `TransitionMatrix` dataclass (lines 161–165)**

Remove the entire block:
```python
@dataclass(frozen=True)
class TransitionMatrix:
    """Level-to-level transition counts and probabilities."""
    transition_counts: np.ndarray
    transition_probs: np.ndarray
```

**Step 7: Remove `transition_heatmap_fig` from `RuleBasedEDAPlots` and `transition_matrix` from `RuleBasedEDAStats`**

Before (`RuleBasedEDAPlots`):
```python
@dataclass(frozen=True)
class RuleBasedEDAPlots:
    level_plot_fig: Figure
    transition_heatmap_fig: Figure
```
After:
```python
@dataclass(frozen=True)
class RuleBasedEDAPlots:
    level_plot_fig: Figure
```

Before (`RuleBasedEDAStats`):
```python
@dataclass(frozen=True)
class RuleBasedEDAStats:
    per_level_stats: PerLevelStats
    bootstrap_ci_results: BootstrapCIResults
    transition_matrix: TransitionMatrix
```
After:
```python
@dataclass(frozen=True)
class RuleBasedEDAStats:
    per_level_stats: PerLevelStats
    bootstrap_ci_results: BootstrapCIResults
```

---

### Task 2: Update `common_eda.py`

**Files:**
- Modify: `feature_selection/eda/common_eda.py`

**Step 1: Delete the entire `compute_rolling_objective` function (lines 114–129)**

Remove:
```python
def compute_rolling_objective(
    signals: pd.Series,
    returns: pd.Series,
    objective_fn: Callable[[pd.Series, pd.Series], float],
    window: int = 252,
) -> pd.Series:
    ...
```

**Step 2: Remove `Callable` import if no longer used**

Check `from typing import Callable` — only used by `compute_rolling_objective`. Remove it.

**Step 3: Remove `rolling_obj` param from `create_common_eda_plots` and its figure block**

Before signature:
```python
def create_common_eda_plots(
    feature: pd.Series,
    target: pd.Series,
    timestamps: pd.DatetimeIndex,
    rolling_corr: pd.Series,
    rolling_obj: pd.Series,
) -> CommonEDAPlots:
```
After:
```python
def create_common_eda_plots(
    feature: pd.Series,
    target: pd.Series,
    timestamps: pd.DatetimeIndex,
    rolling_corr: pd.Series,
) -> CommonEDAPlots:
```

**Step 4: Simplify `time_series_fig` to single subplot (feature only)**

Before (inside `create_common_eda_plots`):
```python
fig_ts, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
ax1.plot(timestamps, feature.values, linewidth=0.8, color="steelblue")
ax1.set_title("Feature over time")
ax1.set_ylabel("Feature value")
ax2.plot(timestamps, target.values, linewidth=0.8, color="darkorange")
ax2.set_title("Target (returns) over time")
ax2.set_ylabel("Return")
fig_ts.tight_layout()
plt.close(fig_ts)
```
After:
```python
fig_ts, ax1 = plt.subplots(1, 1, figsize=(12, 4))
ax1.plot(timestamps, feature.values, linewidth=0.8, color="steelblue")
ax1.set_title("Feature over time")
ax1.set_ylabel("Feature value")
fig_ts.tight_layout()
plt.close(fig_ts)
```

**Step 5: Remove rolling_obj figure creation block and update return**

Delete this block entirely:
```python
# 3. Rolling objective plot
fig_ro, ax = plt.subplots(figsize=(12, 3))
ax.plot(rolling_obj.index, rolling_obj.values, linewidth=0.8, color="green")
ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax.set_title("Rolling objective metric")
ax.set_ylabel("Metric value")
fig_ro.tight_layout()
plt.close(fig_ro)
```

Update return:
```python
return CommonEDAPlots(
    time_series_fig=fig_ts,
    rolling_corr_fig=fig_rc,
)
```

---

### Task 3: Update `continuous_eda.py`

**Files:**
- Modify: `feature_selection/eda/continuous_eda.py`

**Step 1: Delete `compute_monotonicity_test` function (lines 99–110)**

Remove:
```python
def compute_monotonicity_test(bin_means: np.ndarray) -> MonotonicityTest:
    ...
```

**Step 2: Remove `MonotonicityTest` from imports at top of file**

Remove from the `from feature_selection.eda.eda_dataclasses import (...)` block:
```python
    MonotonicityTest,
```

**Step 3: Remove `dist_diagnostics` param from `create_continuous_eda_plots` and QQ/KDE blocks**

Before signature:
```python
def create_continuous_eda_plots(
    feature: pd.Series,
    target: pd.Series,
    decile_analysis: DecileAnalysis,
    dist_diagnostics: DistributionDiagnostics,
) -> ContinuousEDAPlots:
```
After:
```python
def create_continuous_eda_plots(
    feature: pd.Series,
    target: pd.Series,
    decile_analysis: DecileAnalysis,
) -> ContinuousEDAPlots:
```

**Step 4: Delete QQ and KDE figure blocks from `create_continuous_eda_plots`**

Delete these two blocks:
```python
# 3. Q-Q plot
fig_qq, ax = plt.subplots(figsize=(6, 6))
stats.probplot(clean, dist="norm", plot=ax)
ax.set_title("Q-Q plot vs Normal")
fig_qq.tight_layout()
plt.close(fig_qq)

# 4. KDE plot
fig_kde, ax = plt.subplots(figsize=(10, 4))
kde = stats.gaussian_kde(clean)
x_range = np.linspace(clean.min(), clean.max(), 300)
ax.plot(x_range, kde(x_range), color="purple", linewidth=1.5)
ax.set_title("Kernel density estimate")
fig_kde.tight_layout()
plt.close(fig_kde)
```

**Step 5: Update return statement**

Before:
```python
return ContinuousEDAPlots(
    decile_plot_fig=fig_d,
    histogram_fig=fig_h,
    qq_plot_fig=fig_qq,
    kde_fig=fig_kde,
)
```
After:
```python
return ContinuousEDAPlots(
    decile_plot_fig=fig_d,
    histogram_fig=fig_h,
)
```

**Step 6: Remove unused `DistributionDiagnostics` from the import block (only if not used elsewhere in the file)**

The `compute_distribution_diagnostics` function still uses it, so keep it. Only `dist_diagnostics` as a param is gone.

---

### Task 4: Update `rule_based_eda.py`

**Files:**
- Modify: `feature_selection/eda/rule_based_eda.py`

**Step 1: Delete `compute_transition_matrix` function (lines 136–162)**

Remove the entire function.

**Step 2: Remove `TransitionMatrix` from imports**

Remove from `from feature_selection.eda.eda_dataclasses import (...)`:
```python
    TransitionMatrix,
```

**Step 3: Remove transition heatmap block from `create_rule_based_eda_plots`**

Delete the entire `fig_heatmap` creation block:
```python
fig_heatmap, ax = plt.subplots(figsize=(5, 4))
placeholder = np.eye(len(levels), dtype=float)
image = ax.imshow(placeholder, cmap="Blues", vmin=0.0, vmax=1.0)
ax.set_title("Transition probabilities (placeholder)")
ax.set_xticks(range(len(levels)))
ax.set_yticks(range(len(levels)))
ax.set_xticklabels(levels)
ax.set_yticklabels(levels)
fig_heatmap.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
fig_heatmap.tight_layout()
plt.close(fig_heatmap)
```

**Step 4: Update return statement**

Before:
```python
return RuleBasedEDAPlots(
    level_plot_fig=fig_level,
    transition_heatmap_fig=fig_heatmap,
)
```
After:
```python
return RuleBasedEDAPlots(
    level_plot_fig=fig_level,
)
```

---

### Task 5: Update `eda_reporter.py`

**Files:**
- Modify: `feature_selection/eda/eda_reporter.py`

**Step 1: Remove dead imports at top of file**

From the `from feature_selection.eda.common_eda import (...)` block, remove:
```python
    compute_rolling_objective,
```

From the `from feature_selection.eda.continuous_eda import (...)` block, remove:
```python
    compute_monotonicity_test,
```

From the `from feature_selection.eda.eda_dataclasses import (...)` block, remove:
```python
    MonotonicityTest,
    TransitionMatrix,
```

From the `from feature_selection.eda.rule_based_eda import (...)` block, remove:
```python
    compute_transition_matrix,
```

**Step 2: Update `run_eda_for_continuous_feature` — remove monotonicity_test**

Before:
```python
decile_analysis = compute_decile_analysis(feature=feature, target=target, n_bins=config.n_bins)
monotonicity_test = compute_monotonicity_test(decile_analysis.bin_stats.mean_return)
distribution_diagnostics = compute_distribution_diagnostics(feature)
continuous_stats = ContinuousEDAStats(
    decile_analysis=decile_analysis,
    monotonicity_test=monotonicity_test,
    distribution_diagnostics=distribution_diagnostics,
)
continuous_plots = create_continuous_eda_plots(
    feature=feature,
    target=target,
    decile_analysis=decile_analysis,
    dist_diagnostics=distribution_diagnostics,
)
```
After:
```python
decile_analysis = compute_decile_analysis(feature=feature, target=target, n_bins=config.n_bins)
distribution_diagnostics = compute_distribution_diagnostics(feature)
continuous_stats = ContinuousEDAStats(
    decile_analysis=decile_analysis,
    distribution_diagnostics=distribution_diagnostics,
)
continuous_plots = create_continuous_eda_plots(
    feature=feature,
    target=target,
    decile_analysis=decile_analysis,
)
```

**Step 3: Update `run_eda_for_rule_based_feature` — remove transition_matrix**

Before:
```python
transition_matrix = compute_transition_matrix(feature=feature)
rule_stats = RuleBasedEDAStats(
    per_level_stats=per_level_stats,
    bootstrap_ci_results=bootstrap_ci,
    transition_matrix=transition_matrix,
)
```
After:
```python
rule_stats = RuleBasedEDAStats(
    per_level_stats=per_level_stats,
    bootstrap_ci_results=bootstrap_ci,
)
```

**Step 4: Update `_build_common_stats_and_plots` — remove rolling_objective**

Before:
```python
rolling_objective = compute_rolling_objective(
    signals=feature,
    returns=target,
    objective_fn=config.objective_fn,
    window=config.rolling_window,
)
common_stats = CommonEDAStats(
    feature_stats=feature_stats,
    target_stats=target_stats,
    temporal_stability=temporal_stability,
    correlation_analysis=correlation_analysis,
    rolling_objective=rolling_objective,
)
common_plots = create_common_eda_plots(
    feature=feature,
    target=target,
    timestamps=timestamps,
    rolling_corr=temporal_stability.rolling_correlation,
    rolling_obj=rolling_objective,
)
```
After:
```python
common_stats = CommonEDAStats(
    feature_stats=feature_stats,
    target_stats=target_stats,
    temporal_stability=temporal_stability,
    correlation_analysis=correlation_analysis,
)
common_plots = create_common_eda_plots(
    feature=feature,
    target=target,
    timestamps=timestamps,
    rolling_corr=temporal_stability.rolling_correlation,
)
```

**Step 5: Update `compute_correlation_analysis` call — remove kendall computation in `common_eda.py`**

In `common_eda.py`, update `compute_correlation_analysis`:

Before:
```python
pearson = float(f.corr(t, method="pearson"))
spearman = float(f.corr(t, method="spearman"))
kendall = float(f.corr(t, method="kendall"))

lagged: dict[int, float] = {
    lag: float(f.corr(t.shift(-lag), method="pearson"))
    for lag in range(1, max_lag + 1)
}

return CorrelationAnalysis(
    pearson=pearson,
    spearman=spearman,
    kendall=kendall,
    lagged_correlations=lagged,
)
```
After:
```python
pearson = float(f.corr(t, method="pearson"))
spearman = float(f.corr(t, method="spearman"))

lagged: dict[int, float] = {
    lag: float(f.corr(t.shift(-lag), method="pearson"))
    for lag in range(1, max_lag + 1)
}

return CorrelationAnalysis(
    pearson=pearson,
    spearman=spearman,
    lagged_correlations=lagged,
)
```

**Step 6: Remove kendall tau diagnostic flag from `compute_diagnostic_flags`**

Delete:
```python
if isinstance(feature_stats, ContinuousEDAStats) and abs(feature_stats.monotonicity_test.kendall_tau) < 0.3:
    warnings.append("Weak monotonic signal (|kendall_tau| < 0.3)")
```

**Step 7: Update `_common_stats_from_json` — remove rolling_objective load**

Before:
```python
return CommonEDAStats(
    feature_stats=_descriptive_stats_from_json(payload["feature_stats"]),
    target_stats=_descriptive_stats_from_json(payload["target_stats"]),
    temporal_stability=TemporalStability(...),
    correlation_analysis=CorrelationAnalysis(
        pearson=float(corr_payload["pearson"]),
        spearman=float(corr_payload["spearman"]),
        kendall=float(corr_payload["kendall"]),
        lagged_correlations={...},
    ),
    rolling_objective=_series_from_json(payload["rolling_objective"]),
)
```
After:
```python
return CommonEDAStats(
    feature_stats=_descriptive_stats_from_json(payload["feature_stats"]),
    target_stats=_descriptive_stats_from_json(payload["target_stats"]),
    temporal_stability=TemporalStability(...),
    correlation_analysis=CorrelationAnalysis(
        pearson=float(corr_payload["pearson"]),
        spearman=float(corr_payload["spearman"]),
        lagged_correlations={
            int(lag): float(value)
            for lag, value in corr_payload["lagged_correlations"].items()
        },
    ),
)
```

**Step 8: Update `_continuous_stats_from_json` — remove monotonicity_test**

Before:
```python
return ContinuousEDAStats(
    decile_analysis=DecileAnalysis(...),
    monotonicity_test=MonotonicityTest(
        kendall_tau=float(monotonicity_payload["kendall_tau"]),
        p_value=float(monotonicity_payload["p_value"]),
        is_monotonic=bool(monotonicity_payload["is_monotonic"]),
    ),
    distribution_diagnostics=DistributionDiagnostics(...),
)
```
After:
```python
return ContinuousEDAStats(
    decile_analysis=DecileAnalysis(...),
    distribution_diagnostics=DistributionDiagnostics(...),
)
```

**Step 9: Update `_rule_stats_from_json` — remove transition_matrix**

Before:
```python
return RuleBasedEDAStats(
    per_level_stats=PerLevelStats(...),
    bootstrap_ci_results=BootstrapCIResults(...),
    transition_matrix=TransitionMatrix(
        transition_counts=np.array(transition_payload["transition_counts"], dtype=int),
        transition_probs=np.array(transition_payload["transition_probs"], dtype=float),
    ),
)
```
After:
```python
return RuleBasedEDAStats(
    per_level_stats=PerLevelStats(...),
    bootstrap_ci_results=BootstrapCIResults(...),
)
```

**Step 10: Update `_common_plots_from_dir` — remove rolling_obj_fig**

Before:
```python
def _common_plots_from_dir(plots_dir: Path) -> CommonEDAPlots:
    return CommonEDAPlots(
        time_series_fig=_figure_from_png(plots_dir / "time_series_fig.png"),
        rolling_corr_fig=_figure_from_png(plots_dir / "rolling_corr_fig.png"),
        rolling_obj_fig=_figure_from_png(plots_dir / "rolling_obj_fig.png"),
    )
```
After:
```python
def _common_plots_from_dir(plots_dir: Path) -> CommonEDAPlots:
    return CommonEDAPlots(
        time_series_fig=_figure_from_png(plots_dir / "time_series_fig.png"),
        rolling_corr_fig=_figure_from_png(plots_dir / "rolling_corr_fig.png"),
    )
```

**Step 11: Update `_continuous_plots_from_dir` — remove qq/kde**

Before:
```python
def _continuous_plots_from_dir(plots_dir: Path) -> ContinuousEDAPlots:
    return ContinuousEDAPlots(
        decile_plot_fig=_figure_from_png(plots_dir / "decile_plot_fig.png"),
        histogram_fig=_figure_from_png(plots_dir / "histogram_fig.png"),
        qq_plot_fig=_figure_from_png(plots_dir / "qq_plot_fig.png"),
        kde_fig=_figure_from_png(plots_dir / "kde_fig.png"),
    )
```
After:
```python
def _continuous_plots_from_dir(plots_dir: Path) -> ContinuousEDAPlots:
    return ContinuousEDAPlots(
        decile_plot_fig=_figure_from_png(plots_dir / "decile_plot_fig.png"),
        histogram_fig=_figure_from_png(plots_dir / "histogram_fig.png"),
    )
```

**Step 12: Update `_rule_plots_from_dir` — remove transition_heatmap_fig**

Before:
```python
def _rule_plots_from_dir(plots_dir: Path) -> RuleBasedEDAPlots:
    return RuleBasedEDAPlots(
        level_plot_fig=_figure_from_png(plots_dir / "level_plot_fig.png"),
        transition_heatmap_fig=_figure_from_png(plots_dir / "transition_heatmap_fig.png"),
    )
```
After:
```python
def _rule_plots_from_dir(plots_dir: Path) -> RuleBasedEDAPlots:
    return RuleBasedEDAPlots(
        level_plot_fig=_figure_from_png(plots_dir / "level_plot_fig.png"),
    )
```

---

### Task 6: Update tests for Pass 1

**Files:**
- Modify: `tests/unit-tests/validators/eda/test_common_eda.py`
- Modify: `tests/unit-tests/validators/eda/test_continuous_eda.py`
- Modify: `tests/unit-tests/validators/eda/test_rule_based_eda.py`
- Modify: `tests/validators/eda/test_eda_reporter.py`
- Modify: `tests/integration/feature_validator/test_continuous_eda_pipeline.py`
- Modify: `tests/integration/feature_validator/test_rule_based_eda_pipeline.py`

**Step 1: Update `test_common_eda.py`**

- Remove `compute_rolling_objective` from imports
- Remove `test_rolling_objective_sharpe_manual` test function entirely
- Remove `rolling_obj_fig` assertion from `test_common_eda_plots_smoke`
- Update `create_common_eda_plots` call — remove `rolling_obj` arg:
  ```python
  plots = create_common_eda_plots(feature, target, idx, rolling_corr)
  ```
- Remove `test_correlation_analysis_has_all_lags` reference to `result.kendall` if any
- Remove `rolling_obj` from `CommonEDAPlots` import

**Step 2: Update `test_continuous_eda.py`**

- Remove `compute_monotonicity_test` from imports
- Remove `MonotonicityTest` from dataclass imports
- Delete `test_monotonicity_test_monotonic_increasing` test
- Delete `test_monotonicity_test_flat` test
- Update `test_continuous_eda_plots_smoke`:
  - Remove `dd` (distribution diagnostics) arg from `create_continuous_eda_plots` call:
    ```python
    plots = create_continuous_eda_plots(feature, target, da)
    ```
  - Remove `assert plots.qq_plot_fig is not None`
  - Remove `assert plots.kde_fig is not None`

**Step 3: Update `test_rule_based_eda.py`**

- Remove `compute_transition_matrix` from imports
- Remove `TransitionMatrix` from dataclass imports
- Delete `test_transition_matrix_rows_sum_to_one` test
- Delete `test_transition_matrix_known_counts` test
- Update `test_rule_based_plots_smoke`:
  - Remove `assert plots.transition_heatmap_fig is not None`

**Step 4: Update `tests/validators/eda/test_eda_reporter.py`**

- Remove `MonotonicityTest`, `TransitionMatrix` from dataclass imports
- Remove `kendall` kwarg from `_common_stats()` helper (line 63) and from `CorrelationAnalysis` construction inside it
- Remove `rolling_objective` field from `CommonEDAStats` construction inside `_common_stats()`
- Remove `monotonicity_test` field from any `ContinuousEDAStats` construction
- Remove `transition_matrix` field from any `RuleBasedEDAStats` construction
- Remove any assertions referencing `kendall_tau`, `rolling_obj_fig`, `qq_plot_fig`, `kde_fig`, `transition_heatmap_fig`
- Remove diagnostic flag assertion for `"Weak monotonic signal"`

**Step 5: Update integration tests**

In `test_continuous_eda_pipeline.py` and `test_rule_based_eda_pipeline.py`:
- Remove any `assert hasattr(report.continuous_stats, 'monotonicity_test')` style checks
- Remove assertions for `rolling_obj_fig`, `kde_fig`, `qq_plot_fig`, `transition_heatmap_fig`

**Step 6: Run the full test suite and confirm it passes**

```bash
pytest tests/unit-tests/validators/eda/ tests/validators/eda/ tests/integration/feature_validator/ -v
```
Expected: All tests PASS. If any fail, fix before proceeding.

**Step 7: Commit Pass 1**

```bash
git add feature_selection/eda/ tests/unit-tests/validators/eda/ tests/validators/eda/ tests/integration/feature_validator/
git commit -m "refactor: remove kendall tau, rolling obj, kde, qq, transition matrix from EDA"
```

---

## PASS 2 — Additions (TDD)

---

### Task 7: Add new dataclasses to `eda_dataclasses.py`

**Files:**
- Modify: `feature_selection/eda/eda_dataclasses.py`

**Step 1: Add `ICDecay`, `FeatureACF`, `QuintileSpread` dataclasses**

After the `CorrelationAnalysis` dataclass (in the T001 Common EDA section), add:

```python
@dataclass(frozen=True)
class ICDecay:
    """Spearman IC at multiple forward-return horizons."""
    horizons: list[int]               # e.g. [1, 5, 10, 21]
    ic_by_horizon: dict[int, float]   # horizon → IC value


@dataclass(frozen=True)
class FeatureACF:
    """Autocorrelation and partial autocorrelation of the feature."""
    lags: np.ndarray        # shape (max_acf_lag,), values 1..max_acf_lag
    acf_values: np.ndarray  # shape (max_acf_lag,)
    pacf_values: np.ndarray # shape (max_acf_lag,)
```

After `DecileAnalysis` (in the T002 Continuous section), add:

```python
@dataclass(frozen=True)
class QuintileSpread:
    """Mean return per quintile and Q5-Q1 spread."""
    quintile_means: np.ndarray  # shape (5,), Q1 to Q5
    spread: float               # quintile_means[4] - quintile_means[0]
```

**Step 2: Add fields to `CommonEDAStats` and `CommonEDAPlots`**

```python
@dataclass(frozen=True)
class CommonEDAStats:
    feature_stats: DescriptiveStats
    target_stats: DescriptiveStats
    temporal_stability: TemporalStability
    correlation_analysis: CorrelationAnalysis
    ic_decay: ICDecay
    feature_acf: FeatureACF


@dataclass(frozen=True)
class CommonEDAPlots:
    time_series_fig: Figure
    rolling_corr_fig: Figure
    ic_decay_fig: Figure
    acf_fig: Figure
```

**Step 3: Add fields to `ContinuousEDAStats` and `ContinuousEDAPlots`**

```python
@dataclass(frozen=True)
class ContinuousEDAStats:
    decile_analysis: DecileAnalysis
    distribution_diagnostics: DistributionDiagnostics
    quintile_spread: QuintileSpread


@dataclass(frozen=True)
class ContinuousEDAPlots:
    decile_plot_fig: Figure
    histogram_fig: Figure
    quintile_spread_fig: Figure
```

**Step 4: Add new config fields to `EDAConfig`**

```python
@dataclass(frozen=True)
class EDAConfig:
    n_bins: int = 15
    rolling_window: int = 252
    objective_fn: Callable[[pd.Series, pd.Series], float] = field(
        default=lambda signals, returns: (returns.mean() / returns.std()) if returns.std() > 0 else 0.0
    )
    max_lag: int = 5
    bootstrap_iterations: int = 1000
    random_seed: int = 42
    ic_horizons: tuple[int, ...] = (1, 5, 10, 21)
    max_acf_lag: int = 20
```

Note: `Callable` is still imported (used by `objective_fn`). The `objective_fn` field can be retained for now even though `compute_rolling_objective` is removed — it may be used by other systems. If not needed, remove it separately.

**Step 5: Commit dataclass scaffold**

```bash
git add feature_selection/eda/eda_dataclasses.py
git commit -m "feat: add ICDecay, FeatureACF, QuintileSpread dataclasses"
```

---

### Task 8: TDD — `compute_ic_decay`

**Files:**
- Modify: `tests/unit-tests/validators/eda/test_common_eda.py`
- Modify: `feature_selection/eda/common_eda.py`

**Step 1: Write failing tests**

Add to `test_common_eda.py`:

```python
from feature_selection.eda.common_eda import compute_ic_decay
from feature_selection.eda.eda_dataclasses import ICDecay


def test_ic_decay_horizons_present() -> None:
    """Result contains exactly the requested horizons."""
    n = 300
    idx = _daily_index(n)
    np.random.seed(0)
    feature = pd.Series(np.random.randn(n), index=idx)
    target = pd.Series(np.random.randn(n), index=idx)
    result = compute_ic_decay(feature, target, horizons=[1, 5, 10, 21])
    assert set(result.ic_by_horizon.keys()) == {1, 5, 10, 21}
    assert result.horizons == [1, 5, 10, 21]


def test_ic_decay_perfect_lag1_signal() -> None:
    """Feature that perfectly predicts 1-bar returns has IC(1) near 1.0."""
    n = 300
    idx = _daily_index(n)
    feature = pd.Series(np.linspace(0, 1, n), index=idx)
    # target is feature shifted forward by 1 — feature perfectly leads target
    target = feature.shift(-1).fillna(0)
    result = compute_ic_decay(feature, target, horizons=[1, 5])
    assert result.ic_by_horizon[1] > 0.9


def test_ic_decay_values_bounded() -> None:
    """All IC values must lie in [-1, 1]."""
    n = 200
    idx = _daily_index(n)
    np.random.seed(7)
    feature = pd.Series(np.random.randn(n), index=idx)
    target = pd.Series(np.random.randn(n), index=idx)
    result = compute_ic_decay(feature, target, horizons=[1, 5, 10, 21])
    for h, ic in result.ic_by_horizon.items():
        assert -1.0 <= ic <= 1.0, f"IC at horizon {h} out of bounds: {ic}"
```

**Step 2: Run to verify failure**

```bash
pytest tests/unit-tests/validators/eda/test_common_eda.py::test_ic_decay_horizons_present -v
```
Expected: `ImportError` or `AttributeError` — function does not exist yet.

**Step 3: Implement `compute_ic_decay` in `common_eda.py`**

```python
def compute_ic_decay(
    feature: pd.Series,
    target: pd.Series,
    horizons: list[int],
) -> ICDecay:
    """Compute Spearman IC between feature and forward returns at each horizon.

    For each h in horizons: IC(h) = spearman_corr(feature[t], target[t+h]).
    """
    from feature_selection.eda.eda_dataclasses import ICDecay

    ic_by_horizon: dict[int, float] = {}
    for h in horizons:
        forward_target = target.shift(-h)
        aligned = pd.DataFrame({"f": feature, "t": forward_target}).dropna()
        if len(aligned) < 10:
            ic_by_horizon[h] = float("nan")
        else:
            ic_by_horizon[h] = float(aligned["f"].corr(aligned["t"], method="spearman"))

    return ICDecay(horizons=horizons, ic_by_horizon=ic_by_horizon)
```

Add `ICDecay` to the imports at the top of `common_eda.py`:
```python
from feature_selection.eda.eda_dataclasses import (
    CorrelationAnalysis,
    CommonEDAPlots,
    DescriptiveStats,
    ICDecay,
    TemporalStability,
)
```

**Step 4: Run tests**

```bash
pytest tests/unit-tests/validators/eda/test_common_eda.py -k "ic_decay" -v
```
Expected: All 3 PASS.

---

### Task 9: TDD — `compute_feature_acf`

**Files:**
- Modify: `tests/unit-tests/validators/eda/test_common_eda.py`
- Modify: `feature_selection/eda/common_eda.py`

**Step 1: Write failing tests**

Add to `test_common_eda.py`:

```python
from feature_selection.eda.common_eda import compute_feature_acf
from feature_selection.eda.eda_dataclasses import FeatureACF


def test_feature_acf_shapes() -> None:
    """lags, acf_values, pacf_values all have length max_lag."""
    n = 200
    idx = _daily_index(n)
    feature = pd.Series(np.random.randn(n), index=idx)
    result = compute_feature_acf(feature, max_lag=20)
    assert len(result.lags) == 20
    assert len(result.acf_values) == 20
    assert len(result.pacf_values) == 20


def test_feature_acf_lags_values() -> None:
    """lags array is [1, 2, ..., max_lag]."""
    n = 100
    idx = _daily_index(n)
    feature = pd.Series(np.random.randn(n), index=idx)
    result = compute_feature_acf(feature, max_lag=5)
    assert list(result.lags) == [1, 2, 3, 4, 5]


def test_feature_acf_highly_persistent_series() -> None:
    """AR(1) series with phi=0.9 -> ACF lag-1 > 0.7."""
    n = 500
    idx = _daily_index(n)
    np.random.seed(0)
    values = np.zeros(n)
    for i in range(1, n):
        values[i] = 0.9 * values[i - 1] + np.random.randn() * 0.1
    feature = pd.Series(values, index=idx)
    result = compute_feature_acf(feature, max_lag=5)
    assert result.acf_values[0] > 0.7   # lag-1 ACF
```

**Step 2: Run to verify failure**

```bash
pytest tests/unit-tests/validators/eda/test_common_eda.py::test_feature_acf_shapes -v
```
Expected: `ImportError` — function does not exist yet.

**Step 3: Implement `compute_feature_acf` in `common_eda.py`**

```python
def compute_feature_acf(
    feature: pd.Series,
    max_lag: int = 20,
) -> FeatureACF:
    """Compute ACF and PACF of the feature series up to max_lag lags.

    Uses statsmodels FFT-based ACF and OLS-based PACF.
    Returns lags 1..max_lag (lag-0 autocorrelation of 1.0 is excluded).
    """
    from statsmodels.tsa.stattools import acf, pacf

    from feature_selection.eda.eda_dataclasses import FeatureACF

    clean = feature.dropna().to_numpy(dtype=float)
    acf_full = acf(clean, nlags=max_lag, fft=True)   # shape (max_lag+1,), index 0..max_lag
    pacf_full = pacf(clean, nlags=max_lag)             # shape (max_lag+1,)

    lags = np.arange(1, max_lag + 1)
    return FeatureACF(
        lags=lags,
        acf_values=acf_full[1:],   # drop lag-0
        pacf_values=pacf_full[1:],
    )
```

Add `FeatureACF` to the import block in `common_eda.py`.

**Step 4: Run tests**

```bash
pytest tests/unit-tests/validators/eda/test_common_eda.py -k "acf" -v
```
Expected: All 3 PASS.

---

### Task 10: TDD — `compute_quintile_spread`

**Files:**
- Modify: `tests/unit-tests/validators/eda/test_continuous_eda.py`
- Modify: `feature_selection/eda/continuous_eda.py`

**Step 1: Write failing tests**

Add to `test_continuous_eda.py`:

```python
from feature_selection.eda.continuous_eda import compute_quintile_spread
from feature_selection.eda.eda_dataclasses import QuintileSpread


def test_quintile_spread_shape() -> None:
    """quintile_means has exactly 5 values."""
    n = 500
    idx = pd.bdate_range("2020-01-01", periods=n)
    feature = pd.Series(np.linspace(0, 1, n), index=idx)
    target = pd.Series(np.random.randn(n), index=idx)
    result = compute_quintile_spread(feature, target)
    assert len(result.quintile_means) == 5


def test_quintile_spread_positive_for_trending_feature() -> None:
    """Feature perfectly correlated with target -> Q5 > Q1 -> spread > 0."""
    n = 500
    idx = pd.bdate_range("2020-01-01", periods=n)
    np.random.seed(42)
    feature = pd.Series(np.linspace(0, 1, n), index=idx)
    target = pd.Series(np.linspace(0, 1, n) + np.random.randn(n) * 0.01, index=idx)
    result = compute_quintile_spread(feature, target)
    assert result.spread > 0.0


def test_quintile_spread_formula() -> None:
    """spread == quintile_means[4] - quintile_means[0]."""
    n = 500
    idx = pd.bdate_range("2020-01-01", periods=n)
    np.random.seed(0)
    feature = pd.Series(np.random.randn(n), index=idx)
    target = pd.Series(np.random.randn(n), index=idx)
    result = compute_quintile_spread(feature, target)
    assert result.spread == pytest.approx(result.quintile_means[4] - result.quintile_means[0])
```

**Step 2: Run to verify failure**

```bash
pytest tests/unit-tests/validators/eda/test_continuous_eda.py::test_quintile_spread_shape -v
```
Expected: `ImportError` — function does not exist yet.

**Step 3: Implement `compute_quintile_spread` in `continuous_eda.py`**

```python
def compute_quintile_spread(
    feature: pd.Series,
    target: pd.Series,
) -> QuintileSpread:
    """Bin feature into 5 quantiles and compute per-quintile mean return.

    spread = mean_return(Q5) - mean_return(Q1).
    """
    from feature_selection.eda.eda_dataclasses import QuintileSpread

    aligned = pd.DataFrame({"f": feature, "t": target}).dropna()
    aligned["quintile"] = pd.qcut(aligned["f"], q=5, labels=False, duplicates="drop")
    quintile_means = (
        aligned.groupby("quintile")["t"]
        .mean()
        .reindex(range(5))
        .to_numpy(dtype=float)
    )
    spread = float(quintile_means[4] - quintile_means[0])
    return QuintileSpread(quintile_means=quintile_means, spread=spread)
```

Add `QuintileSpread` to the import block in `continuous_eda.py`.

**Step 4: Run tests**

```bash
pytest tests/unit-tests/validators/eda/test_continuous_eda.py -k "quintile" -v
```
Expected: All 3 PASS.

---

### Task 11: Add plots for IC decay, ACF, and quintile spread

**Files:**
- Modify: `feature_selection/eda/common_eda.py`
- Modify: `feature_selection/eda/continuous_eda.py`

**Step 1: Add `ic_decay_fig` and `acf_fig` to `create_common_eda_plots`**

Update signature:
```python
def create_common_eda_plots(
    feature: pd.Series,
    target: pd.Series,
    timestamps: pd.DatetimeIndex,
    rolling_corr: pd.Series,
    ic_decay: "ICDecay",
    feature_acf: "FeatureACF",
) -> CommonEDAPlots:
```

Add after the rolling_corr figure block:

```python
# 3. IC decay figure
fig_ic, ax = plt.subplots(figsize=(10, 4))
horizons = ic_decay.horizons
ic_vals = [ic_decay.ic_by_horizon[h] for h in horizons]
ax.bar([str(h) for h in horizons], ic_vals, color="steelblue")
ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax.set_title("IC decay by forward-return horizon")
ax.set_xlabel("Horizon (bars)")
ax.set_ylabel("Spearman IC")
fig_ic.tight_layout()
plt.close(fig_ic)

# 4. ACF/PACF figure
n_clean = feature.dropna().shape[0]
conf_band = 1.96 / np.sqrt(n_clean)
fig_acf, (ax_acf, ax_pacf) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

ax_acf.bar(feature_acf.lags, feature_acf.acf_values, color="steelblue", width=0.6)
ax_acf.axhline(conf_band, color="red", linestyle="--", linewidth=0.8)
ax_acf.axhline(-conf_band, color="red", linestyle="--", linewidth=0.8)
ax_acf.axhline(0, color="black", linewidth=0.5)
ax_acf.set_title("Autocorrelation Function (ACF)")
ax_acf.set_ylabel("ACF")

ax_pacf.bar(feature_acf.lags, feature_acf.pacf_values, color="darkorange", width=0.6)
ax_pacf.axhline(conf_band, color="red", linestyle="--", linewidth=0.8)
ax_pacf.axhline(-conf_band, color="red", linestyle="--", linewidth=0.8)
ax_pacf.axhline(0, color="black", linewidth=0.5)
ax_pacf.set_title("Partial Autocorrelation Function (PACF)")
ax_pacf.set_ylabel("PACF")
ax_pacf.set_xlabel("Lag")

fig_acf.tight_layout()
plt.close(fig_acf)
```

Update return:
```python
return CommonEDAPlots(
    time_series_fig=fig_ts,
    rolling_corr_fig=fig_rc,
    ic_decay_fig=fig_ic,
    acf_fig=fig_acf,
)
```

**Step 2: Add `quintile_spread_fig` to `create_continuous_eda_plots`**

Update signature:
```python
def create_continuous_eda_plots(
    feature: pd.Series,
    target: pd.Series,
    decile_analysis: DecileAnalysis,
    quintile_spread: "QuintileSpread",
) -> ContinuousEDAPlots:
```

Add after the histogram block:

```python
# 3. Quintile spread figure
fig_qs, ax = plt.subplots(figsize=(8, 4))
quintile_labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
colors = ["#d73027" if v < 0 else "#1a9850" for v in quintile_spread.quintile_means]
ax.bar(quintile_labels, np.nan_to_num(quintile_spread.quintile_means), color=colors)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_title(f"Mean return by quintile  |  spread = {quintile_spread.spread:.4f}")
ax.set_xlabel("Quintile")
ax.set_ylabel("Mean return")
fig_qs.tight_layout()
plt.close(fig_qs)
```

Update return:
```python
return ContinuousEDAPlots(
    decile_plot_fig=fig_d,
    histogram_fig=fig_h,
    quintile_spread_fig=fig_qs,
)
```

**Step 3: Update plot smoke tests**

In `test_common_eda.py`, update `test_common_eda_plots_smoke`:
```python
def test_common_eda_plots_smoke() -> None:
    n = 60
    idx = _daily_index(n)
    np.random.seed(0)
    feature = pd.Series(np.random.randn(n), index=idx)
    target = pd.Series(np.random.randn(n), index=idx)
    rolling_corr = pd.Series(np.random.randn(n), index=idx)
    ic_decay = compute_ic_decay(feature, target, horizons=[1, 5, 10, 21])
    feature_acf = compute_feature_acf(feature, max_lag=20)
    plots = create_common_eda_plots(feature, target, idx, rolling_corr, ic_decay, feature_acf)
    assert plots.time_series_fig is not None
    assert plots.rolling_corr_fig is not None
    assert plots.ic_decay_fig is not None
    assert plots.acf_fig is not None
```

In `test_continuous_eda.py`, update `test_continuous_eda_plots_smoke`:
```python
def test_continuous_eda_plots_smoke() -> None:
    feature, target = _series(500)
    da = compute_decile_analysis(feature, target, n_bins=15)
    qs = compute_quintile_spread(feature, target)
    plots = create_continuous_eda_plots(feature, target, da, qs)
    assert plots.decile_plot_fig is not None
    assert plots.histogram_fig is not None
    assert plots.quintile_spread_fig is not None
```

**Step 4: Run all unit tests**

```bash
pytest tests/unit-tests/validators/eda/ -v
```
Expected: All PASS.

---

### Task 12: Wire new functions into `eda_reporter.py`

**Files:**
- Modify: `feature_selection/eda/eda_reporter.py`

**Step 1: Add new imports**

```python
from feature_selection.eda.common_eda import (
    compute_correlation_analysis,
    compute_descriptive_stats,
    compute_temporal_stability,
    compute_ic_decay,
    compute_feature_acf,
    create_common_eda_plots,
)
from feature_selection.eda.continuous_eda import (
    compute_decile_analysis,
    compute_distribution_diagnostics,
    compute_quintile_spread,
    create_continuous_eda_plots,
)
from feature_selection.eda.eda_dataclasses import (
    ...,  # add ICDecay, FeatureACF, QuintileSpread
)
```

**Step 2: Update `_build_common_stats_and_plots`**

```python
def _build_common_stats_and_plots(
    feature: pd.Series,
    target: pd.Series,
    timestamps: pd.DatetimeIndex,
    config: EDAConfig,
) -> tuple[CommonEDAStats, CommonEDAPlots]:
    feature_stats = compute_descriptive_stats(feature)
    target_stats = compute_descriptive_stats(target)
    temporal_stability = compute_temporal_stability(
        feature=feature, target=target, timestamps=timestamps, rolling_window=config.rolling_window,
    )
    correlation_analysis = compute_correlation_analysis(
        feature=feature, target=target, max_lag=config.max_lag,
    )
    ic_decay = compute_ic_decay(
        feature=feature, target=target, horizons=list(config.ic_horizons),
    )
    feature_acf = compute_feature_acf(feature=feature, max_lag=config.max_acf_lag)

    common_stats = CommonEDAStats(
        feature_stats=feature_stats,
        target_stats=target_stats,
        temporal_stability=temporal_stability,
        correlation_analysis=correlation_analysis,
        ic_decay=ic_decay,
        feature_acf=feature_acf,
    )
    common_plots = create_common_eda_plots(
        feature=feature,
        target=target,
        timestamps=timestamps,
        rolling_corr=temporal_stability.rolling_correlation,
        ic_decay=ic_decay,
        feature_acf=feature_acf,
    )
    return common_stats, common_plots
```

**Step 3: Update `run_eda_for_continuous_feature` — add quintile_spread**

```python
decile_analysis = compute_decile_analysis(feature=feature, target=target, n_bins=config.n_bins)
distribution_diagnostics = compute_distribution_diagnostics(feature)
quintile_spread = compute_quintile_spread(feature=feature, target=target)
continuous_stats = ContinuousEDAStats(
    decile_analysis=decile_analysis,
    distribution_diagnostics=distribution_diagnostics,
    quintile_spread=quintile_spread,
)
continuous_plots = create_continuous_eda_plots(
    feature=feature,
    target=target,
    decile_analysis=decile_analysis,
    quintile_spread=quintile_spread,
)
```

**Step 4: Update JSON serialisation in `_common_stats_from_json`**

Add IC decay and feature ACF load paths:
```python
ic_decay_payload = payload["ic_decay"]
feature_acf_payload = payload["feature_acf"]

common_stats = CommonEDAStats(
    ...
    ic_decay=ICDecay(
        horizons=list(ic_decay_payload["horizons"]),
        ic_by_horizon={int(h): float(v) for h, v in ic_decay_payload["ic_by_horizon"].items()},
    ),
    feature_acf=FeatureACF(
        lags=np.array(feature_acf_payload["lags"], dtype=float),
        acf_values=np.array(feature_acf_payload["acf_values"], dtype=float),
        pacf_values=np.array(feature_acf_payload["pacf_values"], dtype=float),
    ),
)
```

**Step 5: Update `_continuous_stats_from_json`**

Add quintile spread load:
```python
qs_payload = payload["quintile_spread"]
return ContinuousEDAStats(
    decile_analysis=DecileAnalysis(...),
    distribution_diagnostics=DistributionDiagnostics(...),
    quintile_spread=QuintileSpread(
        quintile_means=np.array(qs_payload["quintile_means"], dtype=float),
        spread=float(qs_payload["spread"]),
    ),
)
```

**Step 6: Update `_common_plots_from_dir`**

```python
def _common_plots_from_dir(plots_dir: Path) -> CommonEDAPlots:
    return CommonEDAPlots(
        time_series_fig=_figure_from_png(plots_dir / "time_series_fig.png"),
        rolling_corr_fig=_figure_from_png(plots_dir / "rolling_corr_fig.png"),
        ic_decay_fig=_figure_from_png(plots_dir / "ic_decay_fig.png"),
        acf_fig=_figure_from_png(plots_dir / "acf_fig.png"),
    )
```

**Step 7: Update `_continuous_plots_from_dir`**

```python
def _continuous_plots_from_dir(plots_dir: Path) -> ContinuousEDAPlots:
    return ContinuousEDAPlots(
        decile_plot_fig=_figure_from_png(plots_dir / "decile_plot_fig.png"),
        histogram_fig=_figure_from_png(plots_dir / "histogram_fig.png"),
        quintile_spread_fig=_figure_from_png(plots_dir / "quintile_spread_fig.png"),
    )
```

---

### Task 13: Update reporter test for Pass 2 round-trip

**Files:**
- Modify: `tests/validators/eda/test_eda_reporter.py`

**Step 1: Update `_common_stats()` helper to include `ic_decay` and `feature_acf`**

```python
from feature_selection.eda.eda_dataclasses import (
    ...,
    ICDecay,
    FeatureACF,
    QuintileSpread,
)

def _common_stats(...) -> CommonEDAStats:
    ...
    return CommonEDAStats(
        feature_stats=base_desc,
        target_stats=base_desc,
        temporal_stability=TemporalStability(
            rolling_correlation=pd.Series(np.zeros(sample_size), index=idx),
            structural_breaks=structural_breaks or [],
        ),
        correlation_analysis=CorrelationAnalysis(
            pearson=0.3,
            spearman=0.3,
            lagged_correlations={1: 0.2, 2: 0.1, 3: 0.05, 4: 0.02, 5: 0.01},
        ),
        ic_decay=ICDecay(
            horizons=[1, 5, 10, 21],
            ic_by_horizon={1: 0.2, 5: 0.15, 10: 0.1, 21: 0.05},
        ),
        feature_acf=FeatureACF(
            lags=np.arange(1, 21),
            acf_values=np.linspace(0.5, 0.0, 20),
            pacf_values=np.linspace(0.5, 0.0, 20),
        ),
    )
```

**Step 2: Update `ContinuousEDAStats` construction in test helpers to include `quintile_spread`**

```python
ContinuousEDAStats(
    decile_analysis=...,
    distribution_diagnostics=...,
    quintile_spread=QuintileSpread(
        quintile_means=np.array([-0.02, -0.01, 0.0, 0.01, 0.02]),
        spread=0.04,
    ),
)
```

**Step 3: Run all reporter tests**

```bash
pytest tests/validators/eda/ -v
```
Expected: All PASS.

---

### Task 14: Run full test suite and commit Pass 2

**Step 1: Run all affected tests**

```bash
pytest tests/unit-tests/validators/eda/ tests/validators/eda/ tests/integration/feature_validator/ -v
```
Expected: All PASS.

**Step 2: Final commit**

```bash
git add feature_selection/eda/ tests/unit-tests/validators/eda/ tests/validators/eda/ tests/integration/feature_validator/
git commit -m "feat: add IC decay, feature ACF, and quintile spread to EDA pipeline"
```
