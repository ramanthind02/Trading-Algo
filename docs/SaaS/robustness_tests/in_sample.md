# In-Sample Robustness Tests

## 1. Purpose

This document specifies the in-sample robustness test suite for QuantFoundry. These tests answer the most important question in systematic trading research:

> **What is the probability that my observed in-sample performance is the product of search rather than genuine edge?**

The tests sit at the end of the exploration phase, after a user has run a parameter sweep and identified the leading combinations. They do not validate the strategy on the later validation slice — that is the validation phase's job. They validate the integrity of the in-sample selection itself.

**Implementation anchor (current code):** the IS robustness primitives are provided by `quantfoundry_core.robustness` — `deflated_sharpe_ratio` / `DSRResult`, `compute_n_effective` / `NEffectiveResult`, `sharpe_confidence_interval` / `SharpeCI`, `rolling_is_performance` / `RollingISResult`, `run_grid_permutation_test` / `run_individual_combination_permutation_test` / `PermutationTestResult`. The repo orchestrates them in `research/feature/pipelines/robustness.py` (which assembles an `InSampleRobustnessReport`) and runs them inside `research/feature/exploration/orchestrate.py::execute_exploration_phase`. The signal-timing **vector shuffle** is local: `features/validation/permutation_tests.py::run_vector_shuffle_test`. The pseudocode blocks below are illustrative of the contract, not verbatim source.

Related documents:
- `docs/SaaS/robustness_tests/index.md` — index of all robustness test categories
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

**Pass condition (heterogeneous mining / audit):** $p \leq 0.05$ (real best metric in top 5% of random search).

**Gate status (structured parameter families):** treat as a **diagnostic** — report null percentile and flag below the 20th percentile for manual review. Prefer **DSR $\geq 0.95$** as the Mode 2 gate (§4.4). See §4.1 for why plain-Sharpe return-shuffle tests can read conservative on autocorrelated strategy returns.

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
| ≥ 0.95 | **Gate:** strong evidence of real edge after search correction — proceed if other gates pass |
| 0.75 – 0.95 | Moderate evidence; review $N_\text{eff}$, rolling, and vector shuffle before lock |
| 0.50 – 0.75 | Marginal; result may be search-driven |
| < 0.50 | Search is the more likely explanation; discard |

**Advantage over permutation test:** Near-zero computation cost. Run on every backtest result automatically.

**Limitation:** Assumes approximate return normality and that the $N_\text{eff}$ estimate is accurate. For highly non-normal strategies or when $N_\text{eff}$ is uncertain, run the Full Grid Permutation Test as an **empirical diagnostic** — it makes no distributional assumptions and replicates the actual search procedure. For structured indicator families with reliable $N_\text{eff}$, DSR is the preferred **gate**; full-grid discord when DSR passes warrants manual review (§4.3).

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

### 3.5 Sharpe Ratio Confidence Interval

**What it answers:** How precisely estimated is the IS Sharpe? What range of true Sharpe values is consistent with the observed data?

The IS Sharpe is a point estimate from a finite sample. The precision of that estimate depends on track record length and return distribution. Showing the 95% confidence interval alongside the point estimate calibrates the researcher's confidence before they interpret any other test result.

The 95% CI uses the same standard error formula as the DSR (§3.3):

$$\text{SE}(\hat{SR}) = \sqrt{\frac{1 + \frac{1}{2}\hat{SR}^2 - \gamma_1\hat{SR} + \frac{\gamma_2}{4}\hat{SR}^2}{T}}$$

$$\hat{SR} \pm 1.96 \cdot \text{SE}(\hat{SR})$$

**Typical magnitudes for daily IS data:**

| IS Length | SR = 0.8 CI | SR = 1.5 CI |
|---|---|---|
| 2 years (~504 bars) | ±0.87 | ±0.96 |
| 5 years (~1,260 bars) | ±0.55 | ±0.61 |
| 10 years (~2,520 bars) | ±0.39 | ±0.43 |
| 18 years (~4,536 bars) | ±0.29 | ±0.32 |

The typical IS period of 5–10 years produces a CI width of roughly ±0.4 to ±0.6. A researcher who sees SR = 1.2 with a CI of [0.6, 1.8] should treat the result with substantially more humility than the point estimate suggests.

**UI:** Show as an error bar on the IS Sharpe display so the researcher can see estimate precision at a glance.

---

### 3.6 Rolling IS Performance

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

## 4. Failure Modes and Test Selection

Before choosing a test, be precise about which overfitting failure mode you are trying to catch. In-sample research has **two distinct failure modes**:

| Failure mode | Question | Typical cause |
|---|---|---|
| **1 — Temporal overfitting** | Did this combo only work because its signal values happened to align with this particular return sequence? | Signal timing matters, but there is no durable predictive structure |
| **2 — Parameter mining** | Did grid search find a peak that a naive search over $N$ correlated trials could have hit on noise? | Genuine family structure, but the *best* combo was selected after comparing many alternatives |

**Which test targets which mode:**

| Test | Failure mode | What it permutes / corrects |
|---|---|---|
| **Vector shuffle** (local: per-combo signal timing null) | Mode 1 | Breaks signal–return temporal alignment; preserves marginal signal distribution |
| **Individual return-shuffle permutation** (SaaS §3.2) | Mode 1 (single pre-specified combo only) | Breaks return order for one fixed hypothesis |
| **Full-grid return-shuffle permutation** (§3.1) | Mode 2 | Re-scores entire grid on shuffled returns; empirical null of $\max_i \text{metric}_i$ |
| **Deflated Sharpe Ratio** (§3.3) | Mode 2 (analytical) | Expected maximum Sharpe under $N_\text{eff}$ trials via extreme-value theory |
| **Rolling IS / CUSUM** (§3.6) | Temporal instability | Edge concentrated in one sub-period or structural break |

Mode 1 and Mode 2 are **orthogonal**. A strategy can fail Mode 1 (lucky timing) while passing Mode 2 (search bias looks fine), and vice versa. Neither DSR nor full-grid permutation substitutes for a signal-timing null.

---

### 4.1 Why full-grid permutation can fail unexpectedly

For long IS windows ($T \approx 4{,}500$ daily bars), a back-of-envelope IID Sharpe noise floor suggests the null max should sit far below modest observed Sharpes. Empirical full-grid tests often report **high p-values anyway**. Three mechanisms explain the gap:

**1. Strategy-return autocorrelation collapses effective $T$.**

Systematic strategies with holding persistence (e.g. exit rules spanning several bars) produce serially correlated strategy returns. Newey–West diagnostics ($\max\_\text{lag}$, exit horizons) are useful sanity checks. The effective time sample for Sharpe inference is not $T$ — it is closer to $T / (1 + 2\sum_k w_k \rho_k)$. With persistent autocorrelation, $T_\text{eff}$ can fall to a few hundred bars even when $T > 4{,}000$.

At $T_\text{eff} \approx 400$, the per-period Sharpe standard error scales like $\sqrt{252 / 400} \approx 0.79$ annualized. With $N_\text{eff} \approx 2.5$ effective trials, the expected null maximum can land near 0.7 annualized — **above** an observed best of 0.66. The test correctly flags “not surprising under search,” even though the naive $\sqrt{252/T}$ formula suggested overwhelming power.

**2. Plain Sharpe on observed vs IID null returns is not the same experiment.**

Return-shuffle nulls are **IID by construction**. Observed strategy returns are typically **autocorrelated**. Scoring both sides with plain annualized Sharpe assumes IID in both cases, but only the null satisfies that assumption. Newey–West t-stat adapts per series — which helps single-combo inference on real data — but under return shuffle the HAC correction collapses, producing the asymmetry: NW-deflated observed scores vs near-plain null scores when NW is used for both.

Local implementation therefore splits metrics:

- **Combo selection on real data:** NW-adjusted selection metric (e.g. Newey–West t-stat).
- **Full-grid return-shuffle null:** plain metric (default: annualized Sharpe) on **both** observed and null paths so the comparison is internally consistent under return shuffle.

This removes the NW-vs-null inflation bug but does **not** restore IID-equivalent power when observed returns are autocorrelated and the metric does not encode that structure. That is expected: return-shuffle + plain Sharpe is a conservative, approximate search-bias check, not a perfect substitute for DSR on autocorrelated strategy returns.

**3. Monte Carlo noise at low iteration counts.**

Default local presets often use $M = 100$ null iterations for speed. DSR has no simulation variance and should be preferred as the **gate** for structured grids.

---

### 4.2 Do not pre-filter the grid to “vector-shuffle passers”

It is tempting to run full-grid permutation only on the subset of combos that passed individual vector shuffle. **Do not use that as a valid full-grid gate.**

The passing combos were identified using the **same** return history the full-grid test evaluates. Conditioning the grid on in-sample individual results introduces a **data-dependent selection** that the return-shuffle null does not account for. The full-grid p-value is no longer valid — the two tests are no longer independent.

The only clean two-stage design is **split-sample**: use one holdout window to decide which combos survive individual timing tests, then run full-grid (or DSR) on a **different** historical window with that pre-locked subset. That is architecturally expensive and is not the default exploration workflow.

---

### 4.3 DSR vs full-grid permutation for Mode 2

Both address parameter mining, but DSR is often the **better gate** for focused, theoretically motivated grids:

| Property | Full-grid permutation | DSR |
|---|---|---|
| Null mechanism | Empirical: shuffle returns, re-search grid | Analytical: Gumbel / EVT expected $\max SR \mid N_\text{eff}, T$ |
| $N_\text{eff}$ | Implicit in simulated cross-combo correlations | Explicit from combo return correlation matrix (§3.4) |
| Non-normality | Implicit in simulated returns | Skewness / kurtosis in $\text{SE}(\hat{SR})$ |
| Compute | $O(M \times N \times T)$; noisy at small $M$ | Sub-millisecond |
| Autocorrelated strategy returns | Plain Sharpe null assumes IID; observed side does not | Uses per-period Sharpe SE; still assumes approximate normality |
| Heterogeneous mining | Strong when combos are unrelated families and $N_\text{eff}$ is hard to estimate | Weaker when cross-combo correlation structure is unreliable |

**DSR is sufficient when:**

- The grid is a **structured family** (same indicator, parameter ranges only — e.g. RSI lookback, momentum window, exit bars).
- Strategy returns are serially correlated (typical in systematic trading).
- Effect sizes are modest (annualized SR roughly 0.5–1.0).
- Parameter ranges are theory-motivated, not exhaustive pattern mining.

**Full-grid permutation is more valuable when:**

- True **heterogeneous data mining** across unrelated indicator families with unpredictable cross-correlations.
- All grid points are a priori plausible (not padding with obviously bad params).
- Returns are approximately IID **or** the scoring metric is calibrated consistently on real and null paths.
- Very large $T$ with effect sizes well above the autocorrelation-adjusted noise floor.
- High apparent Sharpes ($> 1.5$) where empirical power is unambiguous.

**Keep full-grid as a diagnostic even when DSR gates:** if DSR $\geq 0.95$ but full-grid null percentile $< 20\%$, investigate correlation structure or metric mismatch manually. That pattern suggests something unusual in the grid geometry or return process — not automatic rejection, but worth a human read.

---

### 4.4 Recommended gates vs diagnostics (exploration)

**Required gates before parameter lock:**

| # | Check | Failure mode | Pass condition |
|---|---|---|---|
| 1 | **Vector shuffle** (per combo) | Mode 1 — temporal overfitting | Combo passes empirical null quantile at configured $\alpha$ |
| 2 | **DSR** | Mode 2 — parameter mining | DSR $\geq 0.95$ |
| 3 | **NW t-stat** (best combo) | Magnitude / HAC-adjusted significance | NW t-stat $\geq 2.0$ on real data |
| 4 | **Rolling positive fraction** | Temporal consistency | $\geq 70\%$ of rolling windows positive (configurable; SaaS default 60%) |
| 5 | **CUSUM** | Structural stability | Statistic within 5% critical bounds |

**Report-only diagnostics (do not hard-gate structured grids):**

| # | Check | Role |
|---|---|---|
| 6 | **Full-grid return-shuffle permutation** | Report null percentile; flag if $< 20$th percentile for manual review |
| 7 | **Sharpe CI lower bound** | Sanity check that interval excludes zero |

Vector shuffle is the only standard exploration test that directly addresses **signal timing**. DSR replaces full-grid as the **primary Mode 2 gate** for structured parameter families. Full-grid remains implemented and reported for audit and heterogeneous-mining scenarios.

---

## 5. Test Relationships and Interpretation Guide

The tests are complementary, not competing:

$$\text{Observed metric} = \underbrace{\text{true edge}}_{\text{what we want}} + \underbrace{\text{finite-sample noise}}_{\text{SR CI}} + \underbrace{\text{selection inflation from search}}_{\text{DSR; full-grid diagnostic}} + \underbrace{\text{temporal misalignment}}_{\text{vector shuffle}} + \underbrace{\text{temporal instability}}_{\text{Rolling IS}}$$

| Test | What it catches | Gate or diagnostic | Compute cost |
|---|---|---|---|
| Sharpe CI | Imprecise estimate from finite $T$ | Diagnostic | Negligible |
| Vector shuffle | Signal timing / Mode 1 | **Gate** | Medium |
| DSR | Search bias / Mode 2 | **Gate** | Negligible |
| NW t-stat | HAC-adjusted magnitude | **Gate** | Negligible |
| Rolling IS / CUSUM | Sub-period concentration / breaks | **Gate** | Negligible |
| Full Grid Permutation | Search bias / Mode 2 (empirical) | **Diagnostic** for structured grids | High |
| Individual return-shuffle | Single-combo temporal structure (pre-specified only) | On demand | Low–Medium |

**Decision tree for researchers (structured grids):**

```
Run parameter search
        |
        ├─ Any priority combo fails vector shuffle? ───────────> Reject or demote; timing null failed.
        |
        ├─ DSR < 0.95? ───────────────────────────────────────> Do not lock params. Search likely explains result.
        |
        ├─ NW t-stat < 2.0 OR rolling/CUSUM fail? ──────────────> Do not lock params.
        |
        └─ All gates pass ──────────────────────────────────────> Review diagnostics (full-grid percentile, Sharpe CI).
                                                                    Proceed toward parameter sensitivity + lock.
```

**The critical distinction — vector shuffle vs full grid:**

These tests answer different questions and must not be substituted for one another.

Example: 108 combos searched; best NW t-stat = 3.0. Vector shuffle on the winner: $p = 0.02$ — timing looks real. Full-grid plain-Sharpe permutation: $p = 0.75$ — max Sharpe under search is not surprising. DSR with $N_\text{eff} = 2.5$: 0.96 — analytical search correction still passes.

Interpretation: timing structure may be genuine (Mode 1), while empirical full-grid under plain Sharpe + autocorrelated returns is underpowered / miscalibrated (diagnostic discord). **Trust the gate suite (vector shuffle + DSR + NW + rolling)**; use full-grid discord as a prompt to inspect $N_\text{eff}$, autocorrelation, and metric choice — not as an automatic veto when DSR and vector shuffle agree.

---

## 6. UI Surfaces

### 6.1 IS Zone Results Panel — Robustness Summary Card

After any parameter sweep completes, the results panel shows a summary card alongside the Sharpe/performance metrics:

```
┌─ Robustness Check ─────────────────────────────────────────────────┐
│  IS Sharpe:  1.42   95% CI  [1.13, 1.71]                          │
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

Sharpe CI, DSR, and rolling IS stability are all computed automatically after every sweep. The Full Grid Permutation Test is triggered manually due to compute cost.

### 6.2 Full Grid Permutation Test Panel

When triggered, renders:
- A histogram of the null distribution of best t-stats
- A vertical line marking the researcher's real best t-stat
- The p-value and pass/fail annotation
- The number of null iterations and estimated confidence in the p-value estimate

### 6.3 DSR Trend View (Multi-Complexity)

When a researcher runs sweeps at multiple complexity levels (e.g. 1-param → 2-param → 3-param combinations), the UI shows:

- DSR probability vs complexity level
- $N_\text{eff}$ vs complexity level
- Train Sharpe vs complexity level
- Overlaid on a single chart

This lets the researcher see the exact point where search outran signal — a visual answer to the overfitting question rather than a single number.

### 6.4 Individual Combination Detail

On any individual parameter combination's result row, a secondary action opens a panel running the Individual Permutation Test for that specific combination, showing its null distribution separately from the grid-level test.

---

## 7. Computation Contract

### 7.1 Inputs

| Field | Type | Source | Description |
|---|---|---|---|
| `is_returns` | `np.ndarray[float64]` | IS zone backtest output | Per-bar simple returns, all combinations |
| `param_results` | `list[ParamResult]` | IS zone backtest output | Metric and return series per combination |
| `selection_metric` | `MetricEnum` | User config | Criterion used to select best combination |
| `n_permutation_iterations` | `int` | User config | Default 1000 |
| `n_eff_override` | `float \| None` | User config | Optional manual N_eff override |

### 7.2 Outputs

The local report class is `InSampleRobustnessReport` (in `research/feature/pipelines/robustness.py`). Its current shape (abridged):

```python
@dataclass(frozen=True)
class InSampleRobustnessReport:
    feature_name: str
    feature_type: str
    selection_metric: str
    periods_per_year: int
    n_combinations: int
    n_effective: NEffectiveResult         # §3.4 — N_eff and avg pairwise corr
    nw: NeweyWestResult                   # §4.4 — Newey–West HAC t-stat

    # always computed — runs synchronously after the sweep
    sharpe_ci: SharpeCI                   # §3.5 — 95% CI on IS Sharpe
    dsr: DSRResult                        # §3.3 — deflated Sharpe probability
    rolling_is: RollingISResult           # §3.6 — rolling Sharpe and CUSUM break test
    stability_chart: RollingISResult

    # on demand — heavier path
    full_grid_permutation: PermutationTestResult | None
    individual_permutation: PermutationTestResult | None

    best_combination: PermutationCombination
    best_param_combo: str
    best_param_combo_label: str
    best_selection_score: float
    raw_sharpe_annualized: float
    nw_adjusted_sharpe_annualized: float
    skewness: float
    excess_kurtosis: float
    # … plus interpretation/diagnostic fields; persisted via to_json_dict()
```

(The result types `NEffectiveResult`, `NeweyWestResult`, `SharpeCI`, `DSRResult`, `RollingISResult`, `PermutationTestResult`, `PermutationCombination` come from `quantfoundry_core.robustness`.)

### 7.3 Worker Behavior

**Synchronous (API server, no job dispatch):**
- Sharpe ratio CI (§3.5) — sub-millisecond
- Deflated Sharpe Ratio (§3.3) — sub-millisecond
- $N_\text{eff}$ computation — uses pre-computed pairwise correlations from the backtest run
- Rolling IS performance and CUSUM test (§3.6) — sub-second

**On demand (worker job):**
- Full Grid Permutation Test — dispatches to a worker job; streams progress events back to the UI (percentage complete, estimated time remaining)
- Individual Combination Permutation Test — same dispatch pattern, lower compute cost

---

## 8. Implementation Notes

**Vectorized null distribution:** All $M$ permutations are generated as a single $(M, T)$ matrix. Metrics are applied across the batch dimension simultaneously. No Python-level loops, no per-iteration overhead. This matches the batched pattern in `features/validation/permutation_tests.py` (e.g. `run_vector_shuffle_target_perm_batch`); the grid/individual permutation nulls run through `quantfoundry_core.robustness.run_grid_permutation_test` / `run_individual_combination_permutation_test`.

**Return shuffling vs block bootstrap:** Simple shuffling (iid permutation) is the default. It assumes the null hypothesis is that return order carries no information. For strategies where preserving local dependence structure matters, block bootstrap may be more appropriate — but this is a later extension, not MVP scope.

**N_eff sensitivity:** The DSR result is sensitive to the N_eff estimate. Report the raw N, average correlation, and N_eff separately so the researcher can audit the calculation. For strategies where the researcher believes their grid is effectively independent (e.g. a calendar mask swept over 12 months), they should be able to override N_eff to N.

**Relationship to downstream permutation tests:** The in-sample tests described here focus on selection bias within the IS zone. Later validation or portfolio-addition permutation checks answer a different question — whether downstream performance is temporally structured. Both suites are needed; they are not substitutes.

> _Verified against commit a07b6bf->197221e on 2026-06-04 (docs Phase A; WP-8 restructure repoint)._
