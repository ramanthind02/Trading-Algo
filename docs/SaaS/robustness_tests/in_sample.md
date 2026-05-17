# In-Sample Robustness Tests

## 1. Purpose

This document specifies the in-sample robustness test suite for QuantFoundry. These tests answer the most important question in systematic trading research:

> **What is the probability that my observed in-sample performance is the product of search rather than genuine edge?**

The tests sit at the end of the research pipeline, after a user has run a parameter sweep and selected their best combination. They do not validate the strategy out-of-sample — that is the walkforward zone's job. They validate the integrity of the in-sample selection itself.

Related documents:
- `docs/SaaS/robustness_tests/` — index of all robustness test categories
- `docs/SaaS/data_flow.md` — zone lifecycle; IS tests run inside the IS zone
- `docs/SaaS/zone_manager.md` — UTC slicing contract and zone boundaries
- `docs/SaaS/metrics_library.md` — canonical metric conventions used in test outputs

---

## 2. The Core Statistical Problem

When a researcher searches N parameter combinations and selects the best performer, they are not running one test — they are running N tests and selecting the maximum. The expected maximum of N random variables grows with N even when none have any true edge. This is the **multiple comparisons problem**, and it means the observed t-stat for the winning combination is systematically inflated relative to its true value.

The inflation grows in two dimensions:

**Dimension 1 — Number of nominal trials (N):**
The more combinations searched, the higher the expected best t-stat under the null hypothesis of no edge. A backtest Sharpe of 1.5 arising from a search of 10,000 combinations is statistically indistinguishable from luck. The same Sharpe from a single pre-specified hypothesis is genuinely impressive.

**Dimension 2 — Independence of the search space:**
Adjacent parameter combinations often produce highly correlated returns. A grid of 286 slow-MA triplets may be statistically equivalent to 30 independent trials because the performance surface is smooth. The relevant quantity is the **effective number of independent trials** $N_\text{eff}$, not the nominal count $N$:

$$N_\text{eff} \approx \frac{N}{1 + 2\sum_{k=1}^{K} \rho_k}$$

where $\rho_k$ is the average pairwise return correlation between combinations separated by lag $k$ in the parameter grid. A jagged, discontinuous performance surface (e.g. calendar masks) has $N_\text{eff} \approx N$ and overfits fast. A smooth surface (e.g. slow MA periods) has $N_\text{eff} \ll N$ and resists overfitting even at large nominal grid sizes.

The in-sample test suite exposes both dimensions to the researcher, giving them the tools to determine whether their best combination is a genuine discovery or a noise peak.

---

## 3. Test Suite

### 3.1 Full Grid Permutation Test

**What it answers:** Did my search process explain my result — independent of whether the underlying signal is real?

**Null hypothesis:** A search of $N$ combinations over return data with no temporal structure would not produce a result this good by chance.

**Procedure:**

1. Take the raw return series from the IS zone.
2. Shuffle the entire series (breaking all temporal structure).
3. Run the full parameter search on the shuffled data using the same selection criterion (e.g. t-stat, Sharpe).
4. Record the best metric value found.
5. Repeat steps 2–4 for $M$ iterations (default $M = 1000$).
6. The resulting distribution is the **null distribution of best metrics under no-edge**.
7. Compute the p-value: the fraction of null iterations where the best metric exceeded the real best metric.

```python
def full_grid_permutation_test(
    returns: np.ndarray,
    param_grid: list[dict],
    backtest_fn: Callable[[np.ndarray, dict], float],  # returns metric (e.g. t-stat)
    n_iterations: int = 1000,
    seed: int = 42,
) -> PermutationTestResult:
    rng = np.random.default_rng(seed)
    real_best = max(backtest_fn(returns, p) for p in param_grid)
    
    null_bests = np.array([
        max(backtest_fn(rng.permutation(returns), p) for p in param_grid)
        for _ in range(n_iterations)
    ])
    
    p_value = np.mean(null_bests >= real_best)
    return PermutationTestResult(
        real_best=real_best,
        null_distribution=null_bests,
        p_value=p_value,
        n_trials=len(param_grid),
        n_iterations=n_iterations,
    )
```

**Vectorized implementation note:** For performance, all $M$ permutations should be generated as a single $(M, T)$ matrix and metrics applied in a vectorized batch — not looped. This achieves ~50,000 iterations per second for simple strategies, keeping a 1,000-iteration test under a second.

**What it does not measure:** Whether any individual combination is genuinely good. A strategy can pass this test while being a poor individual choice. It can also fail this test even though the underlying signal is real — if the grid is simply too large relative to available data.

**Pass condition:** $p \leq 0.05$ (the real best metric is in the top 5% of what random search would produce).

---

### 3.2 Individual Combination Permutation Test

**What it answers:** Is this specific parameter combination capturing genuine temporal structure in prices, or is its performance consistent with a lucky draw from noise?

**Null hypothesis:** This specific combination has no edge. Its observed performance is explainable by the random ordering of returns.

**Procedure:**

1. Fix one parameter combination (the user's selected strategy).
2. Shuffle the return series.
3. Run that single backtest on the shuffled data.
4. Record the metric.
5. Repeat steps 2–4 for $M$ iterations.
6. Compute p-value: fraction of iterations where shuffled metric exceeded real metric.

This is equivalent to the Full Grid test with a grid of size 1. It treats the parameter combination as if it were the researcher's only hypothesis — as if they had pre-specified it before seeing any data.

**When to use it:**
- Validating a combination derived from economic theory before seeing the data.
- Confirming the stability of a single production combination after a manual research process.
- As a complement to the Full Grid test: passing the Full Grid test but failing the Individual test would be unusual and worth investigating.

**Important caveat:** If the user searched any number of combinations before arriving at this one, the Individual test is answering the wrong question. The Full Grid test is always the appropriate tool once a search has occurred. The Individual test is valid only for genuinely pre-specified, single-hypothesis validation.

**Pass condition:** $p \leq 0.05$.

---

### 3.3 Deflated Sharpe Ratio (DSR)

**What it answers:** What is the probability that this strategy's true Sharpe ratio is positive, after analytically accounting for $N_\text{eff}$ trials of search?

The DSR is the analytical counterpart to the Full Grid permutation test. It uses Extreme Value Theory to approximate the expected maximum Sharpe under the null without running simulations, making it nearly instantaneous. Bailey & Lopez de Prado (2014) formalized this framework.

**Inputs:**

| Input | Source | Description |
|---|---|---|
| `SR_observed` | Backtest output | Annualized Sharpe of the best combination |
| `T` | IS zone | Number of trading bars in the IS period |
| `skewness` | Backtest output | Skewness of daily strategy returns |
| `kurtosis` | Backtest output | Excess kurtosis of daily strategy returns |
| `N_eff` | Grid correlation analysis | Effective number of independent trials (§3.4) |
| `SR_benchmark` | User config | Minimum acceptable Sharpe (default: 0) |

**Formula:**

**Step 1 — Standard error of the Sharpe estimator** (accounts for non-normality):

$$\text{SE}(\hat{SR}) = \sqrt{\frac{1 + \frac{1}{2}\hat{SR}^2 - \gamma_1 \hat{SR} + \frac{\gamma_2}{4}\hat{SR}^2}{T}}$$

where $\gamma_1$ is skewness and $\gamma_2$ is excess kurtosis.

**Step 2 — Expected maximum Sharpe under $N_\text{eff}$ null trials** (EVT approximation):

$$E[\max SR \mid N_\text{eff}] \approx \frac{(1 - \gamma_E)\,\Phi^{-1}\!\left(1 - \tfrac{1}{N_\text{eff}}\right) + \gamma_E\,\Phi^{-1}\!\left(1 - \tfrac{1}{N_\text{eff} \cdot e}\right)}{\sqrt{T}}$$

where $\gamma_E \approx 0.5772$ is the Euler–Mascheroni constant and $\Phi^{-1}$ is the standard normal quantile function.

**Step 3 — DSR probability:**

$$\text{DSR} = \Phi\!\left(\frac{\hat{SR} - E[\max SR \mid N_\text{eff}]}{\text{SE}(\hat{SR})}\right)$$

DSR is the probability that the true Sharpe exceeds `SR_benchmark` after correcting for search. Values below 0.5 indicate the search itself is more likely the explanation than genuine edge.

```python
from scipy import stats
import numpy as np

def deflated_sharpe_ratio(
    sr_observed: float,
    t: int,
    skewness: float,
    excess_kurtosis: float,
    n_eff: float,
    sr_benchmark: float = 0.0,
) -> DSRResult:
    euler_gamma = 0.5772156649
    
    se = np.sqrt(
        (1 + 0.5 * sr_observed**2
         - skewness * sr_observed
         + (excess_kurtosis / 4) * sr_observed**2) / t
    )
    
    e_max_sr = (
        (1 - euler_gamma) * stats.norm.ppf(1 - 1 / n_eff)
        + euler_gamma * stats.norm.ppf(1 - 1 / (n_eff * np.e))
    ) / np.sqrt(t)
    
    dsr = stats.norm.cdf((sr_observed - sr_benchmark - e_max_sr) / se)
    
    return DSRResult(
        probability=dsr,
        sr_observed=sr_observed,
        e_max_sr_under_null=e_max_sr,
        standard_error=se,
        n_eff=n_eff,
    )
```

**Interpretation thresholds:**

| DSR Probability | Interpretation |
|---|---|
| ≥ 0.95 | Strong evidence of real edge after accounting for search |
| 0.75 – 0.95 | Moderate evidence; worth walkforward validation |
| 0.50 – 0.75 | Marginal; result may be search-driven |
| < 0.50 | Search is the more likely explanation; discard |

**Advantage over permutation test:** Near-zero computation cost. Run on every backtest result automatically.

**Limitation:** Assumes approximate return normality and that the $N_\text{eff}$ estimate is accurate. For highly non-normal strategies or when $N_\text{eff}$ is uncertain, the Full Grid Permutation Test is more reliable because it makes no distributional assumptions and uses the actual search procedure on actual return data.

---

### 3.4 Effective Trials Estimation ($N_\text{eff}$)

**Why it matters:** Both the DSR and the interpretation of the Full Grid test depend on knowing the effective number of independent trials. Using nominal $N$ without correction for inter-combination correlation will dramatically understate DSR probability for strategies with smooth parameter surfaces (slow MAs, volatility scaling factors) and overstate it for strategies with jagged surfaces (calendar masks, regime filters).

**Computation:**

Given the matrix of backtest return series across all parameter combinations $\{r^{(1)}, r^{(2)}, \ldots, r^{(N)}\}$:

$$\bar{\rho} = \frac{2}{N(N-1)} \sum_{i < j} \text{corr}(r^{(i)}, r^{(j)})$$

$$N_\text{eff} = \frac{N}{1 + (N-1)\bar{\rho}}$$

This is equivalent to the standard formula for the effective sample size in a correlated sample, applied here to the grid of parameter combinations rather than observations.

**Interpretation:**

| $\bar{\rho}$ | $N_\text{eff} / N$ | Surface type |
|---|---|---|
| 0.9 | ~0.10 | Very smooth (slow MA periods) — hard to overfit |
| 0.6 | ~0.27 | Moderate (volatility lookbacks) |
| 0.3 | ~0.44 | Mixed (indicator thresholds) |
| 0.0 | 1.00 | Independent (calendar masks, binary regime flags) — easy to overfit |

**UI exposure:** Report $\bar{\rho}$, $N_\text{eff}$, and $N$ side-by-side on any DSR or permutation test result. Include a sentence-level interpretation of the parameter surface topology.

---

### 3.5 Newey-West Autocorrelation Correction

**What it answers:** Is the reported t-stat honest, or is it inflated by autocorrelation in the return series?

The standard t-stat formula assumes returns are independently drawn each bar. Continuous signal strategies — particularly slow trend-following — produce positively autocorrelated daily returns by construction: a position held for multiple days generates serially correlated P&L. When returns are autocorrelated, the naive t-stat overstates significance because the denominator ($\sigma / \sqrt{T}$) assumes $T$ independent observations when the effective count is lower.

The Newey-West correction replaces the sample variance in the denominator with the **long-run variance (LRV)**, which accounts for autocovariance across lags:

$$\text{LRV} = \gamma_0 + 2\sum_{k=1}^{L} \left(1 - \frac{k}{L+1}\right)\gamma_k$$

where $\gamma_k = \frac{1}{T}\sum_{t=k+1}^{T}(r_t - \bar{r})(r_{t-k} - \bar{r})$ is the sample autocovariance at lag $k$, and the Bartlett weights $\left(1 - k/(L+1)\right)$ ensure the LRV estimate is positive semi-definite.

The NW-adjusted t-stat is then:

$$t_\text{NW} = \frac{\bar{r}\,\sqrt{T}}{\sqrt{\text{LRV}}} = \frac{t_\text{naive}}{\sqrt{\lambda}}, \quad \lambda = \frac{\text{LRV}}{\gamma_0}$$

$\lambda > 1$ whenever returns are positively autocorrelated, so $t_\text{NW} < t_\text{naive}$. For a slow trend-following strategy $\lambda$ is typically in the range 1.5–3.0, reducing the t-stat by 20–40%.

**Lag selection:** Use the data-driven rule $L = \lfloor 4(T/100)^{2/9} \rfloor$. For an 18-year daily IS period ($T \approx 4{,}500$), this gives $L \approx 9$.

```python
def newey_west_tstat(returns: np.ndarray, max_lag: int | None = None) -> NWResult:
    T = len(returns)
    if max_lag is None:
        max_lag = int(4 * (T / 100) ** (2 / 9))

    r_bar = returns.mean()
    demeaned = returns - r_bar
    gamma_0 = (demeaned ** 2).mean()

    lrv = gamma_0
    for k in range(1, max_lag + 1):
        gamma_k = (demeaned[k:] * demeaned[:-k]).mean()
        lrv += 2 * (1 - k / (max_lag + 1)) * gamma_k

    t_naive = r_bar * np.sqrt(T) / np.sqrt(gamma_0)
    t_nw = r_bar * np.sqrt(T) / np.sqrt(lrv)
    inflation_factor = lrv / gamma_0

    return NWResult(
        t_naive=t_naive,
        t_nw=t_nw,
        inflation_factor=inflation_factor,
        lrv=lrv,
        max_lag=max_lag,
    )
```

**Scope of impact:** The NW-adjusted t-stat is the canonical metric used as input to the Full Grid Permutation Test, Individual Permutation Test, and DSR. All t-stat thresholds and permutation test comparisons in this document operate on $t_\text{NW}$, not $t_\text{naive}$. The inflation factor $\lambda$ is reported alongside results so the researcher can see how much autocorrelation is adjusting the headline number.

---

### 3.6 Sharpe Ratio Confidence Interval

**What it answers:** How precisely estimated is the IS Sharpe? What range of true Sharpe values is consistent with the observed data?

The IS Sharpe is a point estimate from a finite sample. The precision of that estimate depends on track record length and return distribution. Showing the 95% confidence interval alongside the point estimate calibrates the researcher's confidence before they interpret any other test result.

The 95% CI uses the same standard error formula as the DSR (§3.3):

$$\text{SE}(\hat{SR}) = \sqrt{\frac{1 + \frac{1}{2}\hat{SR}^2 - \gamma_1\hat{SR} + \frac{\gamma_2}{4}\hat{SR}^2}{T}}$$

$$\hat{SR} \pm 1.96 \cdot \text{SE}(\hat{SR})$$

Note that the SE here uses $T$ in its raw-observation sense — the Newey-West adjustment affects the t-stat (§3.5) but the Sharpe CI is reported on the unadjusted Sharpe since that is the quantity the researcher optimised. Both are shown.

**Typical magnitudes for daily IS data:**

| IS Length | SR = 0.8 CI | SR = 1.5 CI |
|---|---|---|
| 2 years (~504 bars) | ±0.87 | ±0.96 |
| 5 years (~1,260 bars) | ±0.55 | ±0.61 |
| 10 years (~2,520 bars) | ±0.39 | ±0.43 |
| 18 years (~4,536 bars) | ±0.29 | ±0.32 |

The typical IS period of 5–10 years produces a CI width of roughly ±0.4 to ±0.6. A researcher who sees SR = 1.2 with a CI of [0.6, 1.8] should treat the result with substantially more humility than the point estimate suggests.

**UI:** Show as an error bar on the IS Sharpe display. Also show the NW-adjusted Sharpe and its CI side-by-side, so the researcher can see both the headline number and the autocorrelation-corrected version in one glance.

---

### 3.7 Rolling IS Performance

**What it answers:** Is the strategy's edge consistent throughout the IS period, or is the aggregate IS Sharpe driven by a single sub-period?

The aggregate IS Sharpe hides temporal structure. A strategy with a 1.5 IS Sharpe that earned entirely in 2010–2012 and was flat or negative for 15 other years is not the same thing as one that earned steadily throughout. The permutation tests and DSR both operate on the aggregate — they cannot detect this.

**Procedure:**

1. Compute the annualised Sharpe in a rolling window of $W$ bars (default $W = 504$, approximately 2 years for daily data).
2. Plot the rolling Sharpe over time with the IS zone boundaries marked.
3. Apply a CUSUM test to detect structural breaks in the return stream.

**CUSUM structural break test:**

Standardise the return series: $z_t = (r_t - \bar{r}) / \hat{\sigma}$. Compute the cumulative sum $S_t = \sum_{i=1}^{t} z_i$. Under the null of no structural break, $\max_t |S_t| / \sqrt{T}$ follows the Kolmogorov-Smirnov distribution. A structural break is flagged when this statistic exceeds the 5% critical value (~1.36).

```python
def cusum_break_test(returns: np.ndarray, alpha: float = 0.05) -> CUSUMResult:
    z = (returns - returns.mean()) / returns.std(ddof=1)
    cusum = np.cumsum(z)
    T = len(returns)
    stat = np.max(np.abs(cusum)) / np.sqrt(T)
    critical_value = {0.10: 1.22, 0.05: 1.36, 0.01: 1.63}[alpha]
    return CUSUMResult(
        statistic=stat,
        critical_value=critical_value,
        break_detected=stat > critical_value,
        cusum_series=cusum,
    )
```

**Pass conditions:**

| Condition | Threshold |
|---|---|
| Rolling Sharpe positive fraction | ≥ 60% of windows (configurable) |
| CUSUM structural break | Not detected at 5% level |

A strategy that fails the rolling test but passes the search-bias and permutation tests may still be tradeable — but the researcher should understand which sub-period drove the result and whether that environment is likely to recur.

**UI:** A time-series chart with two overlaid series — rolling Sharpe and the CUSUM statistic. The 5% CUSUM critical threshold is shown as a horizontal dashed line. A vertical marker flags the detected break point if one is found.

---

## 4. Test Relationships and Interpretation Guide

The tests are complementary, not competing. Each corrects for or reveals a different failure mode:

$$\text{Observed metric} = \underbrace{\text{true edge}}_{\text{what we want}} + \underbrace{\text{autocorrelation inflation}}_{\text{Newey-West}} + \underbrace{\text{finite-sample noise}}_{\text{Individual test, SR CI}} + \underbrace{\text{selection inflation from search}}_{\text{Full Grid, DSR}} + \underbrace{\text{temporal instability}}_{\text{Rolling IS}}$$

| Test | What it catches | Distributional assumptions | Compute cost | Run automatically? |
|---|---|---|---|---|
| Newey-West correction | Autocorrelation inflating t-stat | None | Negligible | Yes — preprocessing |
| Sharpe CI | Imprecise estimate from short IS period | Approximate normality | Negligible | Yes |
| Individual Permutation | Finite-sample noise for a single combo | None | Low–Medium | On demand |
| Full Grid Permutation | Selection inflation from search | None | High | On demand |
| Deflated Sharpe Ratio | Selection inflation (analytical) | Approximate normality | Negligible | Yes |
| Rolling IS Performance | Edge concentrated in one sub-period | None | Negligible | Yes |

**Decision tree for researchers:**

```
Run parameter search
        |
        ├─ DSR < 0.50? ──────────────────────────────────────> Discard. Search explains the result.
        |
        ├─ 0.50 ≤ DSR < 0.75? ──> Run Full Grid Permutation Test
        |                                   |
        |                          p > 0.05? ──────────────────> Discard.
        |                                   |
        |                          p ≤ 0.05 ──────────────────> Proceed to walkforward with caution.
        |
        └─ DSR ≥ 0.75? ──────────────────────────────────────> Proceed to walkforward validation.
                                                                 (Full Grid test still recommended)
```

**The critical distinction — Individual vs Full Grid:**

These tests are often confused. A concrete example shows why they cannot substitute for each other:

You search 10,000 parameter combinations. The best has a t-stat of 3.5. You run the Individual Permutation Test on it: $p = 0.001$ — highly significant. You conclude you have found a real strategy.

But the Full Grid test tells a different story. You run 1,000 shuffles; each time you search all 10,000 combinations and record the best t-stat. The null distribution has a mean of 4.2. Your real best of 3.5 falls at the 20th percentile. $p = 0.80$.

The Individual test said significant. The Full Grid test said noise. The Full Grid test is correct — the Individual test was answering the wrong question.

---

## 5. UI Surfaces

### 5.1 IS Zone Results Panel — Robustness Summary Card

After any parameter sweep completes, the results panel shows a summary card alongside the Sharpe/performance metrics:

```
┌─ Robustness Check ─────────────────────────────────────────────────┐
│  IS Sharpe:  1.42   95% CI  [1.13, 1.71]                          │
│  NW-adjusted Sharpe:  1.19  (autocorr inflation factor: λ = 1.43)  │
│                                                                     │
│  Combinations searched:   286                                       │
│  Avg pairwise corr:        0.76                                     │
│  Effective trials (Neff):  33                                       │
│                                                                     │
│  Deflated Sharpe Ratio:   0.94  ✓  Strong evidence of real edge    │
│  Rolling IS:              PASS  ✓  Edge consistent across IS period │
│                                                                     │
│  Full Grid Permutation:   [Run — est. 12s]                         │
└─────────────────────────────────────────────────────────────────────┘
```

NW correction, Sharpe CI, DSR, and rolling IS stability are all computed automatically after every sweep. The Full Grid Permutation Test is triggered manually due to compute cost.

### 5.2 Full Grid Permutation Test Panel

When triggered, renders:
- A histogram of the null distribution of best t-stats
- A vertical line marking the researcher's real best t-stat
- The p-value and pass/fail annotation
- The number of null iterations and estimated confidence in the p-value estimate

### 5.3 DSR Trend View (Multi-Complexity)

When a researcher runs sweeps at multiple complexity levels (e.g. 1-param → 2-param → 3-param combinations), the UI shows:

- DSR probability vs complexity level
- $N_\text{eff}$ vs complexity level
- Train Sharpe vs complexity level
- Overlaid on a single chart

This lets the researcher see the exact point where search outran signal — a visual answer to the overfitting question rather than a single number.

### 5.4 Individual Combination Detail

On any individual parameter combination's result row, a secondary action opens a panel running the Individual Permutation Test for that specific combination, showing its null distribution separately from the grid-level test.

---

## 6. Computation Contract

### 6.1 Inputs

| Field | Type | Source | Description |
|---|---|---|---|
| `is_returns` | `np.ndarray[float64]` | IS zone backtest output | Per-bar simple returns, all combinations |
| `param_results` | `list[ParamResult]` | IS zone backtest output | Metric and return series per combination |
| `selection_metric` | `MetricEnum` | User config | Criterion used to select best combination |
| `n_permutation_iterations` | `int` | User config | Default 1000 |
| `n_eff_override` | `float \| None` | User config | Optional manual N_eff override |

### 6.2 Outputs

```python
@dataclass(frozen=True)
class ISRobustnessReport:
    n_combinations: int 
    avg_pairwise_corr: float
    n_eff: float

    # always computed — runs synchronously after backtest
    nw: NWResult                          # §3.5 — NW-adjusted t-stat and inflation factor
    sharpe_ci: SharpeCI                   # §3.6 — 95% CI on IS Sharpe
    dsr: DSRResult                        # §3.3 — deflated Sharpe probability
    rolling_is: RollingISResult           # §3.7 — rolling Sharpe and CUSUM break test

    # on demand — requires a worker job
    full_grid_permutation: PermutationTestResult | None
    individual_permutation: PermutationTestResult | None

    best_combination: ParamResult
    interpretation: str                   # sentence-level summary for UI display
```

### 6.3 Worker Behavior

**Synchronous (API server, no job dispatch):**
- Newey-West correction (§3.5) — sub-millisecond
- Sharpe ratio CI (§3.6) — sub-millisecond
- Deflated Sharpe Ratio (§3.3) — sub-millisecond
- $N_\text{eff}$ computation — uses pre-computed pairwise correlations from the backtest run
- Rolling IS performance and CUSUM test (§3.7) — sub-second

**On demand (worker job):**
- Full Grid Permutation Test — dispatches to a worker job; streams progress events back to the UI (percentage complete, estimated time remaining)
- Individual Combination Permutation Test — same dispatch pattern, lower compute cost

---

## 7. Implementation Notes

**Vectorized null distribution:** All $M$ permutations are generated as a single $(M, T)$ matrix. Metrics are applied across the batch dimension simultaneously. No Python-level loops, no per-iteration overhead. Matches the pattern already in use in `feature_selection/validation/permutation_tests.py`.

**Return shuffling vs block bootstrap:** Simple shuffling (iid permutation) is the default. It assumes the null hypothesis is that return order carries no information. It does not preserve autocorrelation in the return series. For strategies that explicitly trade autocorrelation (mean reversion), block bootstrap may be more appropriate — but this is a later extension, not MVP scope.

**N_eff sensitivity:** The DSR result is sensitive to the N_eff estimate. Report the raw N, average correlation, and N_eff separately so the researcher can audit the calculation. For strategies where the researcher believes their grid is effectively independent (e.g. a calendar mask swept over 12 months), they should be able to override N_eff to N.

**Relationship to walkforward permutation tests:** The in-sample tests described here focus on selection bias within the IS zone. Walkforward permutation tests (return shuffle, pipeline permutation) answer a different question — whether OOS performance is temporally structured. Both suites are needed; they are not substitutes.
