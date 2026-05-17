# Validation Set Robustness Tests

## 1. Purpose

This document specifies the robustness test suite for the strategy-level validation zone. These tests answer a different question from the IS suite:

> **Is my strategy's IS performance consistent with what I see on data it never touched — or did the IS result only look good because of the specific sample I trained on?**

The validation zone is OOS with respect to the strategy's training window, but it sits inside the pre-test research period. It is the researcher's only iterative feedback signal before committing to the portfolio and opening the project test zone. Using it well requires understanding what it can and cannot tell you.

**What validation can tell you:**
- Whether IS performance degrades gracefully or catastrophically on OOS data
- Whether the stable parameter region from IS continues to hold OOS
- Whether the signal's statistical properties have changed (strategy death detection)
- Whether the IS-OOS ranking of parameter combinations is preserved

**What validation cannot tell you:**
- Whether the strategy will perform well in the future (that requires live experience)
- Whether the IS result was genuinely overfitted or just unlucky on this particular val set (val samples are short)

Related documents:
- `docs/SaaS/robustness_tests/in_sample.md` — IS test suite; all IS tests must pass before running these
- `docs/SaaS/robustness_tests/parameter_sensitivity.md` — IS perturbation test and plots; the perturbation set is reused here
- `docs/SaaS/zone_manager.md` — zone structure; validation zones are constrained to the pre-test window

---

## 2. Why Permutation Tests Are Omitted

Permutation tests are not included in the validation suite. On a typical 2–3 year validation set (~500 daily bars), the test has very low power. A t-stat of 2.0 is borderline significant on its own — a permutation test would return p ≈ 0.05–0.10 and add almost no information beyond the t-stat itself.

More importantly, the right question on validation is not "is this result significant in isolation?" — it is "is this result consistent with the IS?" The comparison tests in §3 answer that question directly and are more informative than a standalone significance test on a short OOS sample.

---

## 3. Test Suite

### 3.1 Sharpe Comparison — IS vs Validation

**What it answers:** Has performance degraded from IS to validation, and is the degradation larger than sampling variation alone can explain?

**Outputs:**

| Metric | Description |
|---|---|
| `sr_is` | NW-adjusted IS Sharpe (from IS tests) |
| `sr_val` | NW-adjusted validation Sharpe |
| `ci_is` | 95% CI on IS Sharpe (from §3.6 of IS tests) |
| `ci_val` | 95% bootstrap CI on validation Sharpe (§3.2 below) |
| `degradation_ratio` | `sr_val / sr_is` — how much of IS performance survived OOS |
| `degradation_z` | `(sr_is - sr_val) / sqrt(se_is² + se_val²)` — standard scores of degradation |
| `ci_overlap` | Whether the IS and val 95% CIs overlap |

```python
def sharpe_comparison(
    is_returns: np.ndarray,
    val_returns: np.ndarray,
    nw_is: NWResult,       # from IS test suite
    sharpe_ci_is: SharpeCI, # from IS test suite
) -> SharpeComparisonResult:
    sr_val = annualised_sharpe(val_returns)
    se_val = sharpe_standard_error(sr_val, len(val_returns), *return_moments(val_returns))
    ci_val = (sr_val - 1.96 * se_val, sr_val + 1.96 * se_val)

    degradation_ratio = sr_val / nw_is.sr_nw if nw_is.sr_nw != 0 else np.nan
    degradation_z = (nw_is.sr_nw - sr_val) / np.sqrt(sharpe_ci_is.se**2 + se_val**2)
    ci_overlap = ci_val[0] <= sharpe_ci_is.upper and sharpe_ci_is.lower <= ci_val[1]

    return SharpeComparisonResult(
        sr_is=nw_is.sr_nw,
        sr_val=sr_val,
        ci_is=sharpe_ci_is,
        ci_val=ci_val,
        degradation_ratio=degradation_ratio,
        degradation_z=degradation_z,
        ci_overlap=ci_overlap,
    )
```

**Degradation ratio interpretation:**

| Degradation Ratio | Interpretation |
|---|---|
| ≥ 0.70 | Healthy — most IS edge survived OOS |
| 0.40 – 0.70 | Acceptable — meaningful degradation but signal is present |
| 0.10 – 0.40 | Concerning — most IS edge did not survive; likely partial overfitting |
| ≤ 0.10 | Critical — IS performance was largely noise |

Some degradation is always expected and does not indicate overfitting. IS performance is measured on the same data used for parameter selection; validation is genuinely OOS. The question is whether the degradation is consistent with sampling variation.

**Pass condition:** `ci_overlap = True` OR `degradation_z < 2.0` (degradation is within 2 standard errors of zero).

---

### 3.2 Block Bootstrap CI on Validation Sharpe

**What it answers:** What is the honest confidence interval on the validation Sharpe, given the small OOS sample?

The normal approximation for the Sharpe CI (used in the IS suite) is unreliable on short validation windows of 500–750 bars. Block bootstrap produces a CI without distributional assumptions and preserves the autocorrelation structure of the return stream.

**Procedure:**

1. Choose block length $L = \lfloor T_\text{val}^{1/3} \rfloor$ (e.g. $L \approx 8$ for a 2-year daily val window).
2. Draw $B = 1000$ bootstrap samples: for each sample, resample blocks of length $L$ with replacement until $T_\text{val}$ bars are obtained.
3. Compute annualised Sharpe on each bootstrap sample.
4. The 95% CI is the [2.5th, 97.5th] percentile of the bootstrap distribution.

```python
def block_bootstrap_sharpe_ci(
    returns: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> BootstrapCI:
    T = len(returns)
    block_len = max(1, int(T ** (1/3)))
    rng = np.random.default_rng(seed)

    bootstrap_sharpes = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        n_blocks = int(np.ceil(T / block_len))
        starts = rng.integers(0, T - block_len + 1, size=n_blocks)
        sample = np.concatenate([returns[s:s + block_len] for s in starts])[:T]
        bootstrap_sharpes[i] = annualised_sharpe(sample)

    alpha = (1 - ci_level) / 2
    return BootstrapCI(
        point_estimate=annualised_sharpe(returns),
        lower=np.percentile(bootstrap_sharpes, 100 * alpha),
        upper=np.percentile(bootstrap_sharpes, 100 * (1 - alpha)),
        n_bootstrap=n_bootstrap,
        block_length=block_len,
    )
```

This CI is used in the Sharpe comparison (§3.1) and displayed on the validation results panel.

---

### 3.3 CUSUM with IS Parameters

**What it answers:** Have the statistical properties of the strategy's returns changed in the validation period relative to what IS estimated?

This is the most principled application of the CUSUM test. Unlike the IS rolling CUSUM (§3.7 of `in_sample.md`), which uses the IS mean to test IS data — introducing slight circularity — the validation CUSUM uses IS parameters as an external prior to test a genuinely independent sample.

**Procedure:**

1. Estimate $\mu_\text{IS}$ (mean daily return) and $\sigma_\text{IS}$ (daily return std) from the IS period.
2. Standardise validation returns using IS parameters: $z_t = (r_t^\text{val} - \mu_\text{IS}) / \sigma_\text{IS}$.
3. Compute cumulative sum: $S_t = \sum_{i=1}^{t} z_i$.
4. Test statistic: $C = \max_t |S_t| / \sqrt{T_\text{val}}$.
5. Compare to KS critical value: 1.36 at 5% level.

```python
def cusum_vs_is_params(
    val_returns: np.ndarray,
    mu_is: float,
    sigma_is: float,
    alpha: float = 0.05,
) -> CUSUMResult:
    z = (val_returns - mu_is) / sigma_is
    cusum = np.cumsum(z)
    T = len(val_returns)
    stat = np.max(np.abs(cusum)) / np.sqrt(T)
    critical_value = {0.10: 1.22, 0.05: 1.36, 0.01: 1.63}[alpha]
    return CUSUMResult(
        statistic=stat,
        critical_value=critical_value,
        break_detected=stat > critical_value,
        cusum_series=cusum,
        z_series=z,
    )
```

**Interpretation:** If the CUSUM statistic exceeds the critical value, the validation returns are systematically inconsistent with the IS distribution — the strategy's return-generating process has changed. This is the signal that a strategy is dead or has entered a materially different regime.

A CUSUM trigger on validation does not necessarily mean the IS result was noise. The strategy may have been genuine but the regime has shifted. The rolling z-score plot (§3.4) shows when in the validation period the break occurred.

**Pass condition:** `break_detected = False`.

---

### 3.4 Rolling Z-Score Plot

**What it shows:** The visual companion to the CUSUM test. Plots validation returns standardised by IS parameters over time, making it easy to see when and how the strategy's behaviour diverged from its IS baseline.

**Construction:**
- Plot $z_t = (r_t^\text{val} - \mu_\text{IS}) / \sigma_\text{IS}$ as a bar chart over the validation period.
- Overlay a rolling mean of $z_t$ (window: 60 bars). Should hover near 0 if the strategy is working as expected.
- Draw horizontal dashed lines at $z = \pm 1$ (one standard deviation from IS mean).
- Overlay the CUSUM $S_t$ series on a secondary axis.

A rolling mean that persistently sits below 0 indicates the strategy is earning less than its IS average — either the IS was overfit or the regime has changed. A sudden step down in the CUSUM series indicates a discrete structural break rather than gradual decay.

This plot requires no computation beyond what CUSUM already produces and is the most intuitive single visualisation for the "is the strategy still working?" question.

---

### 3.5 Neighbourhood Performance on Validation

**What it answers:** Does the stable parameter region identified in IS continue to hold on OOS data — or did only the single chosen combination survive?

**Procedure:**

1. Take the perturbation set from the IS perturbation test (all combinations within ±10% of chosen parameters).
2. Run each combination on the validation set without refitting — use the model fitted on IS data, evaluated on validation bars.
3. Compute the validation metric (NW t-stat) for each combination.
4. Report the distribution: p10, median, p90 across the perturbation set.
5. Compute the IS-to-val shift in the median: `val_median - is_median`.

**Key outputs:**

```python
@dataclass(frozen=True)
class NeighbourhoodValResult:
    n_combinations: int

    is_peak: float           # IS metric of chosen combination
    is_median: float         # IS median across perturbation set
    val_peak: float          # val metric of chosen combination
    val_median: float        # val median across perturbation set
    val_p10: float
    val_p90: float

    median_shift: float      # val_median - is_median (negative = degradation)
    neighbourhood_passed: bool  # val_p10 >= metric_floor
```

**Heatmap comparison:** The full IS parameter grid and the same grid evaluated on validation are displayed side-by-side using the same colour scale. The researcher can visually compare whether the performance surface shape is preserved — whether the high-performance region in IS maps to a similar region on validation.

A stable surface (IS and val heatmaps look similar, high-performance region overlaps) is strong evidence the feature is genuine. A collapsed or shifted surface (val high-performance region is in a completely different part of parameter space) suggests the IS surface was noise.

**Pass condition:** `val_p10 >= metric_floor` (even the worst neighbours in the perturbation set remain above the minimum acceptable threshold on validation).

---

### 3.6 Rank Correlation of Parameter Grid

**What it answers:** Is the relative ordering of parameter combinations preserved between IS and validation? A genuine feature should produce similar rankings on both sides — IS best performers should generally also be better val performers, even if absolute values are lower.

**Procedure:**

1. Take all $K$ combinations from the IS sweep.
2. IS rank: rank each combination by NW t-stat on IS data (rank 1 = best).
3. Val rank: run all $K$ combinations on validation without refitting, rank by NW t-stat on val data.
4. Compute Spearman rank correlation $\rho$ between IS ranks and val ranks.

```python
from scipy.stats import spearmanr

def parameter_rank_correlation(
    is_metrics: np.ndarray,    # shape (K,) — IS metric per combination
    val_metrics: np.ndarray,   # shape (K,) — val metric per combination
) -> RankCorrelationResult:
    rho, p_value = spearmanr(is_metrics, val_metrics)
    return RankCorrelationResult(
        spearman_rho=rho,
        p_value=p_value,
        n_combinations=len(is_metrics),
    )
```

**Interpretation:**

| Spearman ρ | Interpretation |
|---|---|
| ≥ 0.50 | Strong order preservation — IS rankings reflect genuine feature properties |
| 0.20 – 0.50 | Moderate — some signal in the IS rankings, some noise |
| 0.00 – 0.20 | Weak — IS ordering was largely noise |
| < 0.00 | Inverse relationship — severe overfitting; IS best performers are val worst |

**Scatter plot:** Plot IS t-stat vs val t-stat for each combination (one dot per combination, chosen combination highlighted). The slope and scatter of this cloud visualise the rank correlation directly. A tight positive relationship indicates a robust signal across the parameter space; a cloud with no slope indicates the IS surface was noise.

**Pass condition:** `spearman_rho >= 0.20` (at least weak order preservation).

---

## 4. Interpretation Guide

The validation suite produces six test outcomes. They address three independent failure modes:

| Failure mode | Tests | Question |
|---|---|---|
| IS performance was overfitted | Sharpe comparison, degradation ratio | Did the IS edge survive OOS? |
| Strategy has died / regime changed | CUSUM, rolling z-score | Is the return process consistent with IS? |
| IS parameter surface was noise | Neighbourhood performance, rank correlation | Was the IS parameter ordering meaningful? |

**Decision logic:**

```
All 6 tests pass
    → Proceed to portfolio inclusion. Strong validation.

Sharpe comparison passes, CUSUM passes, neighbourhood passes,
rank correlation fails (ρ < 0.20)
    → Proceed with caution. Strategy works but IS parameter selection
      was largely arbitrary. Consider equal-weighting more combinations
      rather than relying on the IS-optimal choice.

CUSUM triggers
    → Do not include. The strategy's return process has changed.
      Investigate whether this is regime-specific or permanent before
      reconsidering.

Degradation ratio < 0.10
    → Discard. IS performance was primarily noise.

Neighbourhood val_p10 < metric_floor
    → Fragile. Even if the chosen combination passes, the parameter
      space around it has collapsed OOS. The IS perturbation test
      was misleading. Do not include.

Mixed signals (some pass, some borderline)
    → Use the rolling z-score and heatmap comparison to form a
      qualitative judgement. No single test is definitive on a short
      val sample.
```

---

## 5. UI Surfaces

### 5.1 Validation Summary Card

```
┌─ Validation Results ─────────────────────────────────────────────────┐
│  IS Sharpe (NW-adj):    1.19   95% CI  [0.90, 1.48]                 │
│  Val Sharpe (NW-adj):   0.84   95% CI  [0.41, 1.27]  (bootstrap)    │
│                                                                       │
│  Degradation ratio:     0.71   ✓  Healthy                            │
│  CI overlap:            Yes    ✓                                      │
│  CUSUM:                 PASS   ✓  No structural break detected        │
│  Neighbourhood (p10):   1.93   ✓  Above floor (2.0) — borderline     │
│  Rank correlation ρ:    0.44   ✓  Moderate order preservation        │
│                                                                       │
│  [Rolling Z-Score]  [IS vs Val Heatmaps]  [Rank Scatter]            │
└───────────────────────────────────────────────────────────────────────┘
```

### 5.2 IS vs Val Sharpe Chart

A horizontal comparison chart showing IS and val Sharpe as bars with 95% CI error bars. The degradation ratio is displayed as a percentage below the two bars. The CI overlap region is shaded to make visual overlap obvious.

### 5.3 Rolling Z-Score / CUSUM Chart

Two-panel time-series chart covering the validation period:
- Upper panel: bar chart of daily z-scores with rolling 60-bar mean overlaid. Horizontal lines at ±1.
- Lower panel: CUSUM series with the 5% critical threshold shown as a horizontal line.

### 5.4 IS vs Val Parameter Heatmaps

Side-by-side 2D heatmaps (one per parameter pair) using a shared colour scale. The chosen combination is marked on both. The perturbation box (±10% band) is overlaid on both, making it easy to see whether the stable region is in the same location on both surfaces.

### 5.5 Rank Correlation Scatter

Scatter plot: IS metric (x-axis) vs val metric (y-axis), one dot per combination. The chosen combination is highlighted. A best-fit line shows the overall trend. Spearman ρ and p-value are annotated on the plot.

---

## 6. Computation Contract

### 6.1 Inputs

| Field | Type | Source | Description |
|---|---|---|---|
| `is_returns` | `np.ndarray` | IS zone | Per-bar returns from the IS period |
| `val_returns` | `np.ndarray` | Validation zone | Per-bar returns from the validation period |
| `is_test_results` | `ISRobustnessReport` | IS test suite | Includes NW result, Sharpe CI, SR point estimate |
| `is_param_results` | `list[ParamResult]` | IS sweep | Metric and return series per combination |
| `perturbation_set` | `list[ParamResult]` | IS perturbation test | Combinations within ±10% band |
| `chosen_combination` | `dict[str, Any]` | Researcher selection | The IS-selected combination |
| `metric_floor` | `float` | User config | Default 2.0 (t-stat units) |

### 6.2 Outputs

```python
@dataclass(frozen=True)
class ValidationRobustnessReport:
    # §3.1 — Sharpe comparison
    sharpe_comparison: SharpeComparisonResult

    # §3.2 — Bootstrap CI (feeds into sharpe_comparison.ci_val)
    bootstrap_ci: BootstrapCI

    # §3.3 / §3.4 — CUSUM and rolling z-score
    cusum: CUSUMResult             # includes z_series and cusum_series for plot

    # §3.5 — Neighbourhood performance
    neighbourhood: NeighbourhoodValResult

    # §3.6 — Rank correlation
    rank_correlation: RankCorrelationResult

    # Aggregate
    all_passed: bool
    interpretation: str            # sentence-level summary for UI
```

### 6.3 Worker Behaviour

All validation tests run synchronously on the API server after the validation backtest job completes — no separate job dispatch. The only compute beyond the validation backtest itself is the block bootstrap (1,000 iterations, sub-second for typical val window sizes) and the full grid re-evaluation for rank correlation (K combinations on T_val bars, already fast from the IS sweep infrastructure).

Plot data is stored as a structured artifact alongside the validation result and returned to the frontend on demand.
