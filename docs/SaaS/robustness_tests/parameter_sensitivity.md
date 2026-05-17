# Parameter Sensitivity Tests and Plots

## 1. Purpose

This document specifies the parameter sensitivity test suite for QuantFoundry. These tests answer the second critical in-sample question:

> **Is my chosen parameter combination sitting on a stable plateau, or is it a fragile peak that happens to look good on this dataset?**

A strategy whose performance depends sharply on its exact parameter values is almost certainly overfit — even if it passes the search-bias tests in `in_sample.md`. Genuine edges tend to be robust to small changes in implementation; noise peaks collapse as soon as you move slightly away from the optimised value.

Related documents:
- `docs/SaaS/robustness_tests/in_sample.md` — search-bias tests (Full Grid Permutation, DSR); run those first
- `docs/SaaS/data_flow.md` — IS zone lifecycle
- `docs/SaaS/metrics_library.md` — canonical metric conventions

Existing implementation references:
- `feature_selection/eda/parameter_analysis.py` — `compute_neighbor_smoothing`, `generate_parameter_sensitivity_report`, `ParameterSensitivityReport`
- `utils/compute/grid_smoothing.py` — `add_smoothed_objective` (core neighbourhood averaging)

---

## 2. The Perturbation Test

### 2.1 Core Idea

Take the researcher's chosen parameter combination. Perturb each parameter by ±10%. Run the strategy on all perturbed combinations. The **median result across the perturbation set is the realistic performance estimate** — the number the researcher should plan around, not the peak backtest result.

This is a shrinkage estimator. In practice a researcher never trades the exact optimised parameter: values drift, data shifts, implementation differs slightly. The median across the perturbation neighbourhood is a more honest expectation of live performance than the IS peak. The gap between peak and median is the **optimism bias** introduced by optimisation.

A large gap (peak 1.8, median 0.9) means the optimiser found a noise spike — the strategy is fragile. A small gap (peak 1.8, median 1.6) means the strategy sits on a plateau — robust.

### 2.2 Perturbation Method

> **Note:** The exact perturbation scheme (how the ±10% band is enumerated across parameter types, grid-rounding behaviour, and whether the Cartesian product or joint perturbation is used) is TBD and will be specified separately once finalised.

The key invariants regardless of scheme:
- All perturbed combinations stay within ±10% of each chosen parameter value
- The chosen combination itself is included in the perturbation set
- The perturbation set is the same for both the formal test and the sensitivity plots

### 2.3 Outputs

For each run of the perturbation test, report:

| Metric | Description |
|---|---|
| `peak_metric` | Metric value of the chosen combination |
| `median_metric` | Median across all perturbed combinations — the realistic performance estimate |
| `p10_metric` | 10th percentile — downside tail |
| `p90_metric` | 90th percentile — upside tail |
| `optimism_bias` | `peak_metric − median_metric` — how much the optimiser inflated the result |
| `n_perturbed` | Number of combinations in the perturbation set |
| `passed` | `median_metric ≥ metric_floor` (default: t-stat 2.0) |

The `passed` condition is deliberately simple: does the strategy still show a meaningful edge after shrinking toward the perturbation median? If the median clears the floor, the researcher has a robust signal. If it does not, the peak was a noise spike.

```python
@dataclass(frozen=True)
class PerturbationTestResult:
    chosen_combination: dict[str, Any]
    perturbation_pct: float          # default 0.10

    peak_metric: float
    median_metric: float
    p10_metric: float
    p90_metric: float
    optimism_bias: float             # peak - median
    n_perturbed: int

    metric_floor: float
    passed: bool                     # median_metric >= metric_floor
    interpretation: str              # sentence-level summary for UI
```

---

## 3. Parameter Sensitivity Plots

The plots are the primary deliverable for most researchers. They make the performance surface tangible and allow visual identification of fragility that the single median number cannot convey on its own.

### 3.1 1D Marginal Sensitivity Curves

**What they show:** How performance changes when one parameter is varied, all others held at their chosen values.

One plot per parameter. Each plot shows:

- **Raw metric curve** — actual backtest metric at each parameter value
- **Smoothed metric curve** — neighbour-averaged metric (separates signal from noise on sparse grids)
- **Chosen combination marker** — vertical line at the selected parameter value
- **Perturbation band** — vertical shaded region spanning the ±10% perturbation window around the chosen value
- **Metric floor line** — horizontal dashed line at `metric_floor`

The researcher can see at a glance whether the chosen point sits in a flat region (robust) or at a spike (fragile), and whether the perturbation band stays above the floor.

```
Metric (t-stat)
   4.0 ┤         ╔══════════╗ ← perturbation band
   3.0 ┤    ╭────╫──────────╫────╮
   2.5 ┤╭───╯    ║    ↑     ║    ╰──  ← raw metric
   2.0 ┤─────────╫──────────╫───────  ← metric floor
   1.5 ┤         ╚══════════╝
       └────────────────────────────
        5   10   [14]  20   30   40
```

### 3.2 2D Parameter Heatmaps

**What they show:** The joint performance surface for each pair of parameters, all others held at their chosen values.

For $D$ parameters there are $\binom{D}{2}$ pairwise heatmaps. For a 3-parameter model: 3 heatmaps. For 4 parameters: 6.

Each heatmap contains:

- **Metric colourmap** — performance value encoded in colour
- **Contour lines** — at 80% and 90% of the peak metric value
- **Chosen combination marker** — crosshair at the selected $(p_i, p_j)$ pair
- **Perturbation box** — rectangle marking the ±10% band around the chosen combination
- **Raw vs smoothed toggle** — switch between the raw performance grid and neighbour-smoothed grid

The perturbation box makes it immediately visible whether the high-performance region is wider than the perturbation window (robust) or if the box clips the edge of the good region (fragile).

**Sparse grid handling:** If either axis has fewer than 4 unique values, the heatmap warns that the surface estimate is unreliable and falls back to the 1D curve view for that pair.

### 3.3 3D+ Grids: Slice Projection

For models with 3 or more parameters, direct visualisation requires slicing. Two modes:

- **`heatmap_slices` (default):** Fix the third parameter at each of its grid values; render one 2D heatmap per value. The researcher pages through slices to see how the surface evolves.
- **`surface_slices`:** Same structure rendered as 3D surface plots. More visually striking, harder to read precise values from.

For 4+ parameter models the UI defaults to pairwise slices through the chosen combination, with an option to explore other slice positions.

---

## 4. Relationship to Other IS Tests

The perturbation test and search-bias tests (Full Grid Permutation, DSR) are independent diagnostics:

| Scenario | Interpretation |
|---|---|
| Passes search-bias, passes perturbation | Strong IS evidence of real edge. Proceed to walkforward. |
| Passes search-bias, fails perturbation | Search was not the problem — the chosen combination is a fragile peak within an otherwise real signal space. Select a combination from the stable neighbourhood instead of the raw peak. |
| Fails search-bias, passes perturbation | The stable region is real, but the search was large enough that finding it by chance is plausible. Reduce grid size or obtain more data. |
| Fails both | Discard. Weak signal, large search, and a fragile peak. |

**Connection to $N_\text{eff}$:** A strategy with a smooth performance surface (perturbation median close to peak) naturally has high inter-combination correlation and low $N_\text{eff}$. The 2D heatmaps give visual confirmation of the $\bar{\rho}$ estimate used in the DSR. A flat, plateau-like heatmap supports a low $N_\text{eff}$ adjustment; a jagged heatmap should make the researcher distrust it.

---

## 5. UI Surfaces

### 5.1 Sensitivity Summary Card

Shown alongside the search-bias test results card from `in_sample.md`:

```
┌─ Parameter Sensitivity (±10% perturbation) ────────────────────────┐
│  Chosen:   lookback=14, threshold=0.3, vol_window=20               │
│                                                                     │
│  Peak performance:      t-stat  3.42                               │
│  Realistic performance: t-stat  2.91  (median of perturbation set) │
│  Optimism bias:                 0.51                               │
│  Downside (p10):        t-stat  2.14  ✓  above floor               │
│                                                                     │
│  Result:  PASS  — median clears floor (2.0)                        │
│                                                                     │
│  [View 1D Curves]  [View 2D Heatmaps]                              │
└────────────────────────────────────────────────────────────────────┘
```

### 5.2 Perturbation Distribution Plot

A horizontal boxplot (or violin) showing the distribution of metric values across all perturbed combinations. The chosen combination's peak is marked as a dot to the right of the box, making the optimism bias visually obvious as the distance between the peak dot and the median line.

### 5.3 1D Curve Panel

Full-width panel with one tab per parameter. Includes the perturbation band overlay described in §3.1.

### 5.4 2D Heatmap Panel

Grid layout of all pairwise heatmaps with the perturbation box overlay described in §3.2. For a 2-parameter model: 1 heatmap. For 3-parameter: 3 in a row. For 4-parameter: 6 in a 2×3 grid. Beyond 4 parameters: dropdown selector for the parameter pair.

---

## 6. Computation Contract

### 6.1 Inputs

| Field | Type | Source | Description |
|---|---|---|---|
| `results_df` | `pd.DataFrame` | IS zone backtest output | One row per combination; param value columns and metric column |
| `param_names` | `list[str]` | Strategy spec | Names of parameters that were swept |
| `metric_col` | `str` | User config | Objective column (e.g. `t_stat`, `sharpe`) |
| `chosen_combination` | `dict[str, Any]` | Researcher selection | The combination selected as best |
| `perturbation_pct` | `float` | User config | Default 0.10 (±10%) |
| `metric_floor` | `float` | User config | Default 2.0 (t-stat units) |

### 6.2 Outputs

`PerturbationTestResult` (§2.3) plus a `ParameterSensitivityPlotBundle`:

```python
@dataclass(frozen=True)
class ParameterSensitivityPlotBundle:
    marginal_curves: dict[str, MarginalCurveData]         # one per parameter
    pairwise_heatmaps: dict[tuple[str, str], HeatmapData] # one per param pair
    smoothed_grid: pd.DataFrame                            # full grid with smoothed metric column
    plot_3d_mode: Literal["heatmap_slices", "surface_slices"]
```

Each `MarginalCurveData` and `HeatmapData` carries data arrays for client-side rendering — not pre-rendered images. The Web frontend owns rendering; the worker produces data.

### 6.3 Worker Behaviour

Parameter sensitivity computation runs synchronously on the API server after the IS backtest job completes — no separate job dispatch. Input is the already-computed `results_df`; no additional backtesting is required. Neighbour averaging and perturbation statistics are sub-second for grids up to ~10,000 combinations.

Plot data is stored as a structured artifact alongside the backtest result and returned to the frontend on demand.
