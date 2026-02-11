# Rule-Based Feature Grid Search Performance Report Specification

**Version**: 1.0.0  
**Date**: 2025-02-10  
**Status**: Specification (Design)  
**Scope**: Report content and generation API for analyzing rule-based feature performance across parameter permutations; how grid results are produced is out of scope.

---

## Table of Contents

1. [Overview and Objectives](#1-overview-and-objectives)
2. [Input: Grid Results Schema](#2-input-grid-results-schema)
3. [Metric Thresholds and Aggregations](#3-metric-thresholds-and-aggregations)
4. [Robustness Verdict and Scores](#4-robustness-verdict-and-scores)
5. [Best and Worst Permutations](#5-best-and-worst-permutations)
6. [Parameter Sensitivity](#6-parameter-sensitivity)
7. [Low-Sample and Data Quality](#7-low-sample-and-data-quality)
8. [Report Structure (Output)](#8-report-structure-output)
9. [API](#9-api)
10. [Example Report Snippet](#10-example-report-snippet)
11. [Edge Cases](#11-edge-cases)
12. [Implementation Scope and Dependencies](#12-implementation-scope-and-dependencies)
13. [Multi-Feature Batch and Recommendations](#13-multi-feature-batch-and-recommendations)

---

## 1. Overview and Objectives

### Purpose

After running a grid search over a rule-based feature’s parameters, produce a single report that summarizes whether the feature is **robust** across permutations: good performance in a high fraction of parameter combinations. The report surfaces best and worst cases, parameter sensitivity, and optional integration with a smoothed objective for stability.

### Scope

- **In scope**: Report content (what to compute and output) and the API to generate it. The spec assumes a **grid results DataFrame** is available with a defined schema.
- **Out of scope**: How grid results are produced (backtest runner, EDA pipeline, etc.). No new backtest or EDA pipeline is defined here.

### Audience

Researchers and quants reviewing rule-based features (e.g. bias nodes) for inclusion or parameter tuning.

---

## 2. Input: Grid Results Schema

### Required input

A single DataFrame with **one row per parameter permutation**:

| Component | Description |
|-----------|-------------|
| **Parameter columns** | One column per grid dimension (e.g. `lookback`, `threshold`). Names and count are configurable. |
| **Metric columns** | At least one numeric performance column; multiple are recommended. See supported metrics below. |

### Supported metric names and semantics

| Metric name | Semantics | Typical threshold (success) |
|------------|-----------|-----------------------------|
| `profit_factor` | Gross profit / gross loss; > 1 means profitable. | `> 1.0` |
| `net_return` or `total_return` | Cumulative or sum of returns over the evaluation period. | `> 0` |
| `sharpe` | Risk-adjusted return (annualized Sharpe ratio). | `> 0.3` (configurable) |
| `sortino` | Downside-risk-adjusted return (annualized Sortino ratio). | `> 1.0` (configurable) |
| `max_drawdown` | Worst peak-to-trough decline (e.g. -0.15 = -15%). | `> -0.2` (lower bound) |
| `n_trades` | Number of trades or signals in the evaluation. | Optional min (e.g. `>= 30`) |
| `n_samples` | Number of samples (e.g. bars) used to compute metrics. | Optional min (e.g. `>= 30`) |

Schema may align with existing [eda/parameter_analysis.py](../eda/parameter_analysis.py) output from `analyze_parameter` / `analyze_2d_parameters` (param columns plus metric columns such as `sortino`, `sharpe`, `max_drawdown`, `n_samples`), or from a dedicated rule-based grid runner.

### Optional column: smoothed objective

If present, a pre-computed **smoothed objective** column (e.g. from [grid_neighbor_smoothing_specs.md](grid_neighbor_smoothing_specs.md)) may be included in the report. The report can then show rankings by smoothed metric (stability-focused) alongside raw metric rankings.

---

## 3. Metric Thresholds and Aggregations

### Configurable thresholds

Each metric that participates in the report can have an optional **threshold** for "success":

- **profit_factor**: e.g. `> 1.0` (default).
- **net_return** / **total_return**: e.g. `> 0`.
- **sharpe**: e.g. `> 0.3` (configurable).
- **sortino**: e.g. `> 1.0` (configurable).
- **max_drawdown**: threshold as a **lower bound** (e.g. `> -0.2` means "less severe than -20%").
- **n_trades** / **n_samples**: optional **minimum** (e.g. `>= 30`) to flag or exclude low-sample permutations.

Comparison is **strict** unless specified otherwise: e.g. "above threshold" means `value > threshold` for ratio/return metrics, and `value >= min_value` for sample-count metrics.

### Per-metric aggregations (across all permutations)

For each metric (over rows that are not excluded by low-sample rules):

- **Percentage above threshold**: `pct_above_threshold = (count where metric passes) / (count valid) * 100`.
- **Distribution**: mean, median, min, max, std; optionally coefficient of variation (CV = std / abs(mean)).
- **Counts**: total permutations (valid), count_above_threshold, count_below_threshold, n_excluded (e.g. low sample or NaN).

### Primary metric

One metric is designated **primary** (e.g. `sortino`) for:

- Overall robustness score (when using a single-metric consistency component).
- Best/worst permutation ranking (unless overridden by a "rank_by" option such as smoothed objective).
- Parameter sensitivity (variance contribution can be computed per metric; primary is the default).

Alignment with existing [ParameterAnalyzer.compute_robustness_metrics](eda/parameter_analysis.py) (variance, consistency, risk) is recommended for the primary-metric part of the overall score.

---

## 4. Robustness Verdict and Scores

### Overall robustness verdict

A **verdict** (e.g. ROBUST / MODERATELY ROBUST / FRAGILE) is derived from:

- **Threshold pass rates**: Percentage of (valid) permutations passing **each** configured threshold. Optionally require a minimum pass rate per metric (e.g. 80% for profit_factor, 70% for net_return).
- **Optional composite score**: Combine variance stability (CV), consistency (% above threshold), and risk (e.g. worst drawdown) into a 0–10 score, then map to verdict bands (e.g. as in [param_sens.md](../complete/param_sens.md)).

Suggested verdict bands (configurable):

- **HIGHLY ROBUST**: e.g. overall_score >= 8.0 and all key pass rates above configurable cutoffs.
- **ROBUST**: e.g. overall_score in [6.0, 8.0).
- **MODERATELY ROBUST**: e.g. overall_score in [4.0, 6.0).
- **FRAGILE**: e.g. overall_score in [2.0, 4.0).
- **HIGHLY FRAGILE**: e.g. overall_score < 2.0.

### Per-metric pass rate table

The report includes a table (or equivalent structure):

- **Rows**: One per configured metric.
- **Columns**: metric name, threshold used, pct_above_threshold, count_above, count_below, mean, median (and optionally min, max, n_excluded).

---

## 5. Best and Worst Permutations

### Best permutation(s)

- **Top-k** parameter combinations by a chosen metric (default: primary metric; optional: smoothed objective column if present).
- For each: report the **param tuple** (e.g. dict or ordered list of param name -> value) and **key metric values** (at least primary metric; optionally all configured metrics).
- k is configurable (e.g. `top_k=5`).

### Worst permutation(s)

- **Bottom-k** (or all permutations below a threshold) for diagnostics.
- Same structure: param tuple + key metric values.

### Stability highlight

If a **smoothed objective** column is present, the report may optionally:

- Rank by smoothed metric and list "best by stability" separately, or
- Flag permutations that rank high on **raw** metric but low on **smoothed** (potential fragile peaks) vs high on both (stable and good).

---

## 6. Parameter Sensitivity

### Variance contribution

For each **parameter** dimension, report its contribution to the variance of the **primary** (or each) metric across permutations. Recommended approach (align with [ParameterAnalyzer.compute_robustness_metrics](eda/parameter_analysis.py)):

- Group rows by that parameter’s value; compute mean metric per group.
- Variance of those group means (or similar measure) is the "between-group" contribution.
- Normalize so that contributions across parameters sum to 1 (or 100%) for a clear ranking.

### Sensitivity ranking

- Order parameters by impact (e.g. "threshold explains 45% of variance, lookback 35%, ...").
- Enables the user to see which parameters drive performance variation and where to focus tuning.

---

## 7. Low-Sample and Data Quality

### Minimum sample size

- Configurable **min_n_samples** and/or **min_n_trades** (optional).
- Permutations with `n_samples < min_n_samples` (or `n_trades < min_n_trades`) can be:
  - **Excluded** from all aggregations (threshold pass rates, means, best/worst). The report must state: e.g. "X permutations excluded due to low sample."
  - **Included but flagged**: e.g. "Y permutations have n_samples < 30" in the warnings section; aggregations may still include them (spec to choose one policy; recommend exclude by default).

### Missing / NaN metrics

- **Behavior**: Exclude rows with NaN in a metric from that metric’s aggregations (mean, pct_above). Do **not** count NaN as "below threshold"; report the count of excluded (NaN) rows in per_metric and in warnings.
- **Example warning**: "net_return has 2 NaN values; those permutations excluded from net_return aggregations."

---

## 8. Report Structure (Output)

### Structured output (dict or dataclass)

| Field | Type | Description |
|-------|------|-------------|
| `feature_id` / `module_name` | str | Identifier of the rule-based feature. |
| `grid_shape` | dict or list | Number of permutations; list of param names and their levels (optional). |
| `thresholds_used` | dict | metric -> threshold and optional min_sample. |
| `per_metric` | dict | metric -> { pct_above_threshold, mean, median, min, max, std, cv?, count_above, count_below, n_excluded }. |
| `overall_verdict` | str | e.g. ROBUST, FRAGILE. |
| `overall_score` | float, optional | 0–10 composite score. |
| `best_permutations` | list | Each element: { param_tuple, metrics_dict }; top-k. |
| `worst_permutations` | list | Each element: { param_tuple, metrics_dict }; bottom-k or below-threshold. |
| `parameter_sensitivity` | dict | param_name -> variance_contribution (or impact score). |
| `warnings` | list[str] | e.g. "3 permutations excluded (low sample)", "net_return has 2 NaN". |

### Text / Markdown export

Optional formatted report for saving to file or printing. Suggested sections (aligned with structured output):

1. **Summary**: feature_id, grid shape, overall_verdict, overall_score.
2. **Threshold pass rates**: Table (per-metric pass rate table).
3. **Best permutations**: Top-k with params and metrics.
4. **Worst permutations**: Bottom-k with params and metrics.
5. **Parameter sensitivity**: Ranking and contributions.
6. **Warnings**: List of warning strings.

Implementation can generate this text from the same structured report object.

---

## 9. API

### Data types

- **MetricThresholdConfig**: Holds `threshold: float` and optional `min_sample: int` (for n_trades/n_samples). For ratio/return metrics only threshold is used; for n_trades/n_samples, min_sample is used for exclusion/flagging.
- **RuleBasedGridReport**: The structured report (dict or dataclass) described in §8.

### Public entry point

```python
def generate_rule_based_grid_report(
    grid_results_df: pd.DataFrame,
    param_columns: List[str],
    metric_config: Dict[str, MetricThresholdConfig],
    feature_id: str = "rule_based_feature",
    primary_metric: str = "sortino",
    rank_by: Optional[str] = None,
    top_k: int = 5,
    exclude_low_sample: bool = True,
    export_path: Optional[str] = None,
) -> RuleBasedGridReport:
    """
    Generate a performance report for a rule-based feature over its parameter grid.

    Parameters
    ----------
    grid_results_df : pd.DataFrame
        One row per parameter permutation; columns = param_columns + metric columns.
    param_columns : List[str]
        Column names that define the grid dimensions.
    metric_config : Dict[str, MetricThresholdConfig]
        Metric name -> threshold config (threshold, optional min_sample).
    feature_id : str, default "rule_based_feature"
        Identifier for the feature in the report.
    primary_metric : str, default "sortino"
        Metric used for overall score and default best/worst ranking.
    rank_by : Optional[str], default None
        If set (e.g. "avg_objective"), use this column for best/worst ranking instead of primary_metric.
    top_k : int, default 5
        Number of best and worst permutations to include.
    exclude_low_sample : bool, default True
        If True, exclude permutations below min_sample (from metric_config) from aggregations.
    export_path : Optional[str], default None
        If set, write the formatted text report to this path.

    Returns
    -------
    RuleBasedGridReport
        Structured report (dict or dataclass) with all sections in §8.
    """
```

- **Return**: Structured report. If `export_path` is set, also write the text/markdown report to file.

---

## 10. Example Report Snippet

Example for a small grid (2 params, 9 permutations). Suppose:

- **Params**: lookback ∈ {7, 14, 21}, threshold ∈ {30, 50, 70}.
- **Thresholds**: profit_factor > 1.0, sortino > 1.0.
- **Primary metric**: sortino.

**Per-metric pass rate table** (illustrative):

| Metric         | Threshold | pct_above | count_above | count_below | mean  | median |
|----------------|-----------|-----------|-------------|-------------|-------|--------|
| profit_factor  | 1.0       | 77.8%     | 7           | 2           | 1.12  | 1.05   |
| sortino        | 1.0       | 66.7%     | 6           | 3           | 1.08  | 1.02   |

**Overall**: ROBUST (overall_score 6.8). Best permutation: lookback=14, threshold=50 (sortino 1.4, profit_factor 1.35). Worst: lookback=7, threshold=70 (sortino 0.6, profit_factor 0.85).

This gives implementers a concrete target for the report format and content.

---

## 11. Edge Cases

| Case | Behavior |
|------|----------|
| **Empty grid** | Return a report with zero permutations; verdict FRAGILE or N/A; warnings = ["No permutations in grid."]. |
| **All NaN for a metric** | Exclude that metric’s aggregations; pct_above_threshold and distribution stats omitted or set to None; warnings include "metric X has all NaN." |
| **Single permutation** | Compute per_metric from that one row; best_permutations = worst_permutations = that row; parameter_sensitivity empty or N/A. |
| **No param columns** | Flat list of runs (no grid dimensions). Report still works: grid_shape = n_rows, no parameter_sensitivity; best/worst by metric only. |
| **Column missing** | If a metric in metric_config is not in the DataFrame, skip it and add a warning. |

---

## 12. Implementation Scope and Dependencies

### Scope

- **Report generation only**. No change to how grid search is run; no new backtest or EDA pipeline. Consumes an existing DataFrame.
- Implementation can live in `eda/` (e.g. alongside [parameter_analysis.py](../eda/parameter_analysis.py)) or a dedicated module (e.g. `reporting/rule_based_grid_report.py`).

### Dependencies

- **pandas** for DataFrame handling.
- **Optional**: Reuse [ParameterAnalyzer](eda/parameter_analysis.py) for parameter sensitivity (variance contribution) if desired.

### Related documentation

- [docs/bias_nodes/base_bias_node_specs.md](../bias_nodes/base_bias_node_specs.md) — rule-based vs continuous features.
- [docs/complete/param_sens.md](../complete/param_sens.md) — robustness scores and parameter sensitivity.
- [docs/to-do/grid_neighbor_smoothing_specs.md](grid_neighbor_smoothing_specs.md) — optional smoothed objective column.

---

## 13. Multi-Feature Batch and Recommendations

### Multi-feature batch (optional)

- **API**: Optionally support running the report for **multiple** rule-based features:
  - Input: either a list of (grid_results_df, param_columns, feature_id) or a single DataFrame with a **feature_id** column that groups rows by feature.
  - Output: list of `RuleBasedGridReport` or a **summary table** (e.g. one row per feature with columns: feature_id, verdict, overall_score, pct_above_threshold for each key metric).
- Not required for the initial spec but recommended as a future extension.

### Recommendations section

- Include a short **recommendation** text block in the report, e.g.:
  - "Recommendation: ROBUST – 85% of permutations have profit factor > 1 and sortino > 1. Consider using smoothed objective to pick stable params."
  - Or for FRAGILE: "Recommendation: FRAGILE – only 40% of permutations pass profit factor > 1. Consider narrowing the parameter range or excluding this feature."
- Can be generated from the structured fields (verdict, pass rates, optional smoothed column presence).
