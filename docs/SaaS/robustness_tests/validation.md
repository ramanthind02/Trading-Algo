# Validation Set Robustness Tests

## 1. Purpose

This document specifies the robustness test suite for the strategy-level validation zone. These tests answer a different question from the IS suite:

> **Is my strategy's IS performance consistent with what I see on data it never touched — or did the IS result only look good because of the specific sample I trained on?**

The validation zone is OOS with respect to the strategy's training window, but it sits inside the pre-test research period. It is the researcher's only iterative feedback signal before portfolio admission and before opening the project test zone. In local `Trading-Algo` docs and code, this is the `validation` phase in the canonical `exploration -> validation -> portfolio_addition` flow.

**Implementation anchor (current code):** the validation report type is `quantfoundry_core.robustness.validation.ValidationRobustnessReport`. The repo orchestrates and persists it via `feature_research/validation/robustness_runner.py` (`run_validation_robustness_pipeline`, `run_and_write_validation_robustness`) and renders it via `feature_research/visualization/validation_reports.py`. The same report drives portfolio-research holdout monitoring (`portfolio_research/holdout/`). Field names below reflect the current `ValidationRobustnessReport`; the pseudocode blocks are illustrative of the contract, not verbatim source.

**What validation can tell you:**
- Whether IS performance degrades gracefully or catastrophically on OOS data
- Whether the signal's statistical properties have changed (strategy death detection)
- Whether the IS-OOS ranking of parameter combinations is preserved
- Whether the cumulative equity curve stays within the range IS parameters predict
- Whether risk-adjusted performance is locally consistent throughout the validation period

**What validation cannot tell you:**
- Whether the strategy will perform well in the future (that requires live experience)
- Whether the IS result was genuinely overfitted or just unlucky on this particular val set (val samples are short)

Related documents:
- `docs/SaaS/robustness_tests/in_sample.md` — IS test suite; all IS tests must pass before running these
- `docs/SaaS/robustness_tests/parameter_sensitivity.md` — IS perturbation test and plots
- `docs/SaaS/zone_manager.md` — zone structure; validation zones are constrained to the pre-test window

---

## 2. Why Permutation Tests and Bootstrap CIs Are Omitted

Permutation tests are not included in the validation suite. On a typical 2–3 year validation set (~500 daily bars), the test has very low power. A t-stat of 2.0 is borderline significant on its own — a permutation test would return p ≈ 0.05–0.10 and add almost no information beyond the t-stat itself.

The right question on validation is not "is this result significant in isolation?" — it is "is this result consistent with the IS?" The comparison tests in §3 answer that question directly.

Block bootstrap CIs on validation Sharpe are also omitted. The normal approximation SE (`SR / √(2T)`) is honest enough for daily returns on a 500–750 bar window, and the difference from bootstrap rarely changes the pass/fail decision. The bootstrap idea is preserved in §3.3, applied to the equity curve where it provides more diagnostic value.

---

## 3. Test Suite

The five tests cover four independent failure modes:

| Failure mode | Tests |
|---|---|
| IS performance was overfitted | Sharpe degradation |
| Strategy has died / regime changed | CUSUM |
| Equity curve inconsistent with IS | Equity curve confidence bands |
| Risk-adjusted performance locally unstable | Rolling Sharpe z-score |
| IS parameter surface was noise | Rank correlation |

---

### 3.1 Sharpe Degradation — IS vs Validation

**What it answers:** Has performance degraded from IS to validation, and is the degradation larger than sampling variation alone can explain?

> In the current report this leg is carried on the `sharpe_comparison` field (type `SharpeComparisonResult`); the prose below uses "degradation" for the concept it measures.

**Outputs:**

| Metric | Description |
|---|---|
| `sr_is` | IS Sharpe (from IS tests) |
| `sr_val` | Validation Sharpe |
| `degradation_ratio` | `sr_val / sr_is` — how much of IS performance survived OOS |
| `degradation_z` | `(sr_is - sr_val) / sqrt(se_is² + se_val²)` — standard scores of degradation |
| `ci_overlap` | Whether the IS and val 95% CIs overlap (using normal approximation SEs) |

```python
def sharpe_degradation(
    is_returns: np.ndarray,
    val_returns: np.ndarray,
    sr_is: float,
    sharpe_ci_is: SharpeCI,  # from IS test suite
) -> SharpeDegradationResult:
    sr_val = annualised_sharpe(val_returns)
    T_val = len(val_returns)
    se_val = sr_val / np.sqrt(2 * T_val)
    ci_val = (sr_val - 1.96 * se_val, sr_val + 1.96 * se_val)

    degradation_ratio = sr_val / sr_is if sr_is != 0 else np.nan
    degradation_z = (sr_is - sr_val) / np.sqrt(sharpe_ci_is.se**2 + se_val**2)
    ci_overlap = ci_val[0] <= sharpe_ci_is.upper and sharpe_ci_is.lower <= ci_val[1]

    return SharpeDegradationResult(
        sr_is=sr_is,
        sr_val=sr_val,
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
| ≤ 0.10 | Critical — IS performance was primarily noise |

Some degradation is always expected. IS performance is measured on the same data used for parameter selection; validation is genuinely OOS.

**Pass condition:** `ci_overlap = True` OR `degradation_z < 2.0`.

---

### 3.2 CUSUM with IS Parameters

**What it answers:** Have the statistical properties of the strategy's returns changed in the validation period relative to what IS estimated?

This is the most principled application of the CUSUM test. Unlike the IS rolling CUSUM (§3.6 of `in_sample.md`), which uses the IS mean to test IS data — introducing slight circularity — the validation CUSUM uses IS parameters as an external prior to test a genuinely independent sample.

**Procedure:**

1. Estimate $\mu_\text{IS}$ (mean daily return) and $\sigma_\text{IS}$ (daily return std) from the IS period.
2. Standardise validation returns using IS parameters: $z_t = (r_t^\text{val} - \mu_\text{IS}) / \sigma_\text{IS}$.
3. Compute cumulative sum: $S_t = \sum_{i=1}^{t} z_i$.
4. Lower envelope: $L = -c \cdot \sqrt{T_\text{val}}$ where $c$ is the KS critical value (1.36 at 5%).
5. Test statistic (lower side only): $C = \max(0, -\min_t S_t) / \sqrt{T_\text{val}}$.
6. **Pass** when $\min_t S_t \geq L$ (no breach of the lower bound). Positive excursions above $+c\sqrt{T}$ do **not** fail.

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
    critical_value = {0.10: 1.22, 0.05: 1.36, 0.01: 1.63}[alpha]
    lower_threshold = critical_value * np.sqrt(T)
    min_cusum = float(cusum.min())
    stat = max(0.0, -min_cusum) / np.sqrt(T)
    return CUSUMResult(
        statistic=stat,
        critical_value=critical_value,
        break_detected=min_cusum < -lower_threshold,
        cusum_series=cusum,
        z_series=z,
    )
```

**Interpretation:** A fail means sustained **underperformance** vs the IS mean (CUSUM broke the **lower** envelope). Outperformance that pushes CUSUM above the upper envelope is not a failure — we only cull on deterioration, not on unusually strong OOS results.

**Distinction from equity curve bands (§3.3):** CUSUM tests for a *shift in the mean* (a structural break in the return process). Equity curve bands test whether the *cumulative level* over time is consistent with IS parameters — a strategy can pass CUSUM (no discrete break) but still drift below the lower band if it earns less than its IS average throughout.

**Pass condition:** `break_detected = False` (equivalently $\min_t S_t \geq -c\sqrt{T}$).

---

### 3.3 Equity Curve Confidence Bands

**What it answers:** Does the cumulative equity curve over the validation period stay within the range that IS parameters predict? This catches gradual underperformance that has no single structural break and would not trigger CUSUM.

**Construction:**

Using IS parameters ($\mu_\text{IS}$, $\sigma_\text{IS}$), the expected cumulative return at bar $t$ and its 95% confidence bands follow from the distribution of cumulative sums of i.i.d. returns:

$$
\text{Expected path:} \quad E_t = t \cdot \mu_\text{IS}
$$
$$
\text{95\% bands:} \quad E_t \pm 1.96 \cdot \sigma_\text{IS} \cdot \sqrt{t}
$$

```python
def equity_curve_confidence_bands(
    val_returns: np.ndarray,
    mu_is: float,
    sigma_is: float,
    z_crit: float = 1.96,
) -> EquityCurveBandsResult:
    T = len(val_returns)
    t = np.arange(1, T + 1)

    expected = t * mu_is
    half_width = z_crit * sigma_is * np.sqrt(t)
    upper_band = expected + half_width
    lower_band = expected - half_width

    actual = np.cumsum(val_returns)
    below_lower = actual < lower_band
    fraction_below = below_lower.mean()

    # Flag if actual curve spends more than 20% of the period below the lower band
    outside_band = fraction_below > 0.20

    return EquityCurveBandsResult(
        actual=actual,
        expected=expected,
        upper_band=upper_band,
        lower_band=lower_band,
        fraction_below_lower=fraction_below,
        outside_band=outside_band,
    )
```

**Interpretation:** The bands widen over time (proportional to $\sqrt{t}$), reflecting genuine uncertainty in cumulative returns. A strategy that briefly dips below the lower band is unremarkable. A strategy that spends most of the validation period below it is earning materially less than its IS average — a sign of either partial overfitting or regime change.

**Pass condition:** `outside_band = False` (actual curve spends ≤ 20% of the period below the lower 95% band).

---

### 3.4 Rolling Sharpe Z-Score

**What it answers:** Is risk-adjusted performance locally consistent throughout the validation period, or does the strategy have prolonged sub-periods of poor performance that average out in the aggregate Sharpe?

The aggregate Sharpe comparison (§3.1) can pass even if the strategy performs well for one half of the validation period and poorly for the other. The rolling Sharpe z-score exposes this by tracking the local Sharpe relative to its IS distribution.

**Procedure:**

1. Compute the IS rolling Sharpe distribution using the same window (default 60 bars): derive $\mu_\text{SR,IS}$ (mean of IS rolling Sharpes) and $\sigma_\text{SR,IS}$ (std of IS rolling Sharpes).
2. Compute the rolling Sharpe over the validation period using the same window.
3. Standardise: $Z_t^\text{SR} = (\text{SR}_t^\text{val} - \mu_\text{SR,IS}) / \sigma_\text{SR,IS}$.
4. Flag if more than 30% of rolling windows have $Z_t^\text{SR} < -1.5$.

```python
def rolling_sharpe_zscore(
    is_returns: np.ndarray,
    val_returns: np.ndarray,
    window: int = 60,
    z_threshold: float = -1.5,
    fraction_limit: float = 0.30,
) -> RollingSharpezScoreResult:
    def rolling_sharpe(returns: np.ndarray) -> np.ndarray:
        s = pd.Series(returns)
        return s.rolling(window).apply(annualised_sharpe, raw=True).to_numpy()

    is_sr = rolling_sharpe(is_returns)
    val_sr = rolling_sharpe(val_returns)

    is_sr_valid = is_sr[~np.isnan(is_sr)]
    mu_sr_is = is_sr_valid.mean()
    sigma_sr_is = is_sr_valid.std()

    z = (val_sr - mu_sr_is) / sigma_sr_is
    z_valid = z[~np.isnan(z)]
    fraction_below = (z_valid < z_threshold).mean()
    unstable = fraction_below > fraction_limit

    return RollingSharpezScoreResult(
        rolling_sharpe_val=val_sr,
        z_scores=z,
        mu_sr_is=mu_sr_is,
        sigma_sr_is=sigma_sr_is,
        fraction_below_threshold=fraction_below,
        z_threshold=z_threshold,
        unstable=unstable,
    )
```

**Distinction from equity curve bands (§3.3):** Equity curve bands test whether the *cumulative return level* is consistent with IS. The rolling Sharpe z-score tests whether *risk-adjusted performance* is locally consistent — it is sensitive to periods where volatility increases even if the mean return holds up, and vice versa.

**Distinction from CUSUM (§3.2):** CUSUM tests for a single structural break in the return mean. The rolling Sharpe z-score detects diffuse instability spread across many sub-periods — a strategy that is intermittently poor rather than permanently shifted.

**Pass condition:** `unstable = False` (≤ 30% of rolling windows fall below z = −1.5).

---

### 3.5 Rolling Z-Score Plot

**What it shows:** The visual companion to the CUSUM test and equity curve bands. Plots validation returns standardised by IS parameters over time.

**Construction:**
- Plot $z_t = (r_t^\text{val} - \mu_\text{IS}) / \sigma_\text{IS}$ as a bar chart over the validation period.
- Overlay a rolling mean of $z_t$ (window: 60 bars). Should hover near 0 if the strategy is working as expected.
- Draw horizontal dashed lines at $z = \pm 1$.
- Overlay the CUSUM $S_t$ series on a secondary axis.
- Overlay the rolling Sharpe z-score $Z_t^\text{SR}$ (from §3.4) on a third panel for direct comparison.

This plot surfaces the CUSUM, return z-score, and rolling Sharpe z-score in a single view, making it easy to distinguish discrete breaks from diffuse instability.

---

### 3.6 Rank Correlation of Parameter Grid

**What it answers:** Is the relative ordering of parameter combinations preserved between IS and validation? A genuine feature should produce similar rankings on both sides — IS best performers should generally also be better val performers, even if absolute values are lower.

**Procedure:**

1. Take all $K$ combinations from the IS sweep.
2. IS rank: rank each combination by the configured metric on IS data (rank 1 = best).
3. Val rank: run all $K$ combinations on validation without refitting, rank by the same metric on validation data.
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

**Pass condition:** `spearman_rho >= 0.20`.

---

## 4. Interpretation Guide

The validation suite produces five test outcomes, each covering a distinct failure mode:

| Failure mode | Test | Question |
|---|---|---|
| IS performance was overfitted | Sharpe degradation | Did the IS edge survive OOS? |
| Strategy has died / regime changed | CUSUM | Is the return process consistent with IS? |
| Gradual OOS underperformance | Equity curve bands | Does cumulative performance stay within IS-predicted bounds? |
| Risk-adjusted performance locally unstable | Rolling Sharpe z-score | Is local risk-adjusted performance consistent throughout? |
| IS parameter surface was noise | Rank correlation | Was the IS parameter ordering meaningful? |

**Decision logic:**

```
All 5 tests pass
    → Proceed to portfolio inclusion. Strong validation.

All pass except rank correlation (ρ < 0.20)
    → Proceed with caution. Strategy works but IS parameter selection
      was largely arbitrary. Consider equal-weighting more combinations
      rather than relying on the IS-optimal choice.

CUSUM triggers
    → Do not include. The strategy's return process has changed.
      Investigate whether this is regime-specific or permanent before
      reconsidering.

Equity curve outside band (CUSUM passes)
    → Gradual underperformance with no discrete break. The strategy
      is earning less than IS average throughout. Likely partial
      overfitting or slow regime shift.

Rolling Sharpe z-score fails (equity curve bands pass)
    → Intermittent instability: the strategy alternates between good
      and poor risk-adjusted sub-periods. The aggregate Sharpe is
      acceptable but the strategy is unreliable. Investigate whether
      the weak sub-periods cluster around specific market conditions.

Degradation ratio < 0.10
    → Discard. IS performance was primarily noise.

Mixed signals (some pass, some borderline)
    → Use the rolling z-score / CUSUM chart and equity curve chart
      to form a qualitative judgement. No single test is definitive
      on a short val sample.
```

---

## 5. UI Surfaces

### 5.1 Validation Summary Card

```
┌─ Validation Results ─────────────────────────────────────────────────┐
│  IS Sharpe:             1.19   95% CI  [0.90, 1.48]                 │
│  Val Sharpe:            0.84   95% CI  [0.56, 1.12]                 │
│                                                                       │
│  Degradation ratio:     0.71   ✓  Healthy                            │
│  CI overlap:            Yes    ✓                                      │
│  CUSUM:                 PASS   ✓  No structural break detected        │
│  Equity curve bands:    PASS   ✓  8% of period below lower band      │
│  Rolling SR z-score:    PASS   ✓  12% of windows below z = -1.5     │
│  Rank correlation ρ:    0.44   ✓  Moderate order preservation        │
│                                                                       │
│  [Z-Score / CUSUM / SR]  [Equity Curve Bands]  [Rank Scatter]       │
└───────────────────────────────────────────────────────────────────────┘
```

### 5.2 IS vs Val Sharpe Chart

A horizontal comparison chart showing IS and val Sharpe as bars with 95% CI error bars. The degradation ratio is displayed as a percentage below the two bars. The CI overlap region is shaded to make visual overlap obvious.

### 5.3 Rolling Z-Score / CUSUM / SR Chart

Three-panel time-series chart covering the validation period:
- Upper panel: bar chart of daily return z-scores with rolling 60-bar mean overlaid. Horizontal lines at ±1.
- Middle panel: CUSUM series with the 5% critical threshold shown as a horizontal line.
- Lower panel: rolling Sharpe z-score $Z_t^\text{SR}$ with threshold at −1.5 shown as a dashed line and shading below it.

### 5.4 Equity Curve Confidence Bands Chart

Single-panel chart covering the validation period:
- Actual cumulative return curve (solid line).
- Expected path under IS parameters (dashed centre line).
- 95% confidence band (shaded region, widening with $\sqrt{t}$).
- Fraction of period spent below lower band annotated on the chart.

### 5.5 Rank Correlation Scatter

Scatter plot: IS metric (x-axis) vs val metric (y-axis), one dot per combination. The chosen combination is highlighted. A best-fit line shows the overall trend. Spearman ρ and p-value are annotated on the plot.

---

## 6. Computation Contract

### 6.1 Inputs

| Field | Type | Source | Description |
|---|---|---|---|
| `is_returns` | `np.ndarray` | IS zone | Per-bar returns from the IS period |
| `val_returns` | `np.ndarray` | Validation zone | Per-bar returns from the validation period |
| `is_test_results` | `InSampleRobustnessReport` | IS test suite | Includes Sharpe CI, DSR, rolling stability, and SR point estimate |
| `is_param_results` | `list[ParamResult]` | IS sweep | Metric and return series per combination |
| `chosen_combination` | `dict[str, Any]` | Researcher selection | The IS-selected combination |
| `metric_floor` | `float` | User config | Default 2.0 (t-stat units) |

### 6.2 Outputs

The current `ValidationRobustnessReport` (from `quantfoundry_core.robustness.validation`) exposes these fields (consumed by `robustness_runner.py` and `validation_reports.py`):

```python
@dataclass(frozen=True)
class ValidationRobustnessReport:
    # §3.1 — Sharpe degradation (IS vs validation)
    sharpe_comparison: SharpeComparisonResult   # field is `sharpe_comparison`, .passed gates

    # §3.2 / §3.5 — CUSUM and rolling z-score plot
    cusum: CUSUMResult                          # includes z_series and cusum_series for plot

    # §3.3 — Equity curve confidence bands
    equity_curve_bands: EquityCurveBandsResult

    # §3.4 — Rolling Sharpe z-score
    rolling_sharpe_zscore: RollingSharpezScoreResult

    # §3.6 — Rank correlation
    rank_correlation: RankCorrelationResult

    # Aggregate
    all_passed: bool
    interpretation: str                         # sentence-level summary for UI
```

Holdout monitoring (`portfolio_research/holdout/monitoring_policy.py`) counts failures across exactly four legs — `sharpe_comparison.passed`, `cusum.passed`, `rolling_sharpe_zscore.passed`, `equity_curve_bands.passed` — to derive its Green/Yellow/Red traffic light.

### 6.3 Worker Behaviour

All validation tests run synchronously after the validation backtest job completes — no separate job dispatch. The only non-trivial compute is the full grid re-evaluation for rank correlation (K combinations on T_val bars, already fast from the IS sweep infrastructure). Equity curve bands and CUSUM are both O(T_val) with no iteration. In the repo this is `feature_research/validation/robustness_runner.py::run_validation_robustness_pipeline`.

Plot data is stored as a structured artifact alongside the validation result and returned to the frontend on demand.

> _Verified against commit a07b6bf on 2026-06-04 (docs Phase A)._
