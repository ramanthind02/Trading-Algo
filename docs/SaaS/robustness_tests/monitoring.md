# Strategy and Portfolio Monitoring

## 1. Purpose

This document specifies the live monitoring suite for deployed strategies and portfolios. These tests answer a fundamentally different — and harder — question than the IS and validation suites:

> **Is this strategy still generating the edge I researched, or has it stopped working?**

Related documents:
- `docs/SaaS/robustness_tests/in_sample.md` — IS test suite
- `docs/SaaS/robustness_tests/validation.md` — validation test suite
- `docs/SaaS/zone_manager.md` — project test zone and contamination rules

---

## 2. The Monitoring Problem — Be Honest About What Detection Can Do

Before specifying any tests, the monitoring framework must be grounded in an honest accounting of statistical power in live trading.

**The signal-to-noise problem:**

For a strategy with a Sharpe of 0.5 and 20% annualised volatility:

- Daily expected return: ≈ 0.04%
- Daily return volatility: ≈ 1.26%
- Signal-to-noise ratio: 1:30

Even if the strategy's true edge disappears entirely — expected return drops to zero permanently — a Welch t-test will fail to detect this at the 5% level for months or years. The noise distribution and the "dead strategy" distribution look nearly identical at short horizons. By the time any significance threshold is reliably crossed, the strategy has been dead for a long time.

This is not a flaw in the monitoring tests. It is the mathematics of low signal-to-noise financial data. The monitoring suite should be understood as:

- **Early warning signals** of potential problems, not definitive verdicts
- **Calibration tools** that tell researchers what normal variance looks like, so they do not mistake it for edge decay
- **Pre-specified triggers** for graduated position size responses, not binary on/off decisions

A researcher who expects monitoring tests to reliably identify the moment an edge disappears will be perpetually either overreacting to noise or waiting too long for certainty that never arrives. The right mental model is probabilistic, not diagnostic.

---

## 3. Pre-Commitment Doctrine

**All monitoring thresholds and response rules must be specified before the strategy goes live.**

This is the monitoring equivalent of the test set contamination rule (§8 of `zone_manager.md`). A researcher who defines their removal criteria after observing a drawdown — even if the criteria they choose are statistically reasonable — has introduced the same bias that contaminates a test set. The narrative of "this drawdown is different, this time I should respond" is exactly the trap Longmore describes.

The platform stores monitoring configuration as immutable metadata on each deployed strategy, timestamped at deployment. The required fields are:

```python
@dataclass(frozen=True)
class MonitoringConfig:
    # SPRT boundaries (§4.3)
    sprt_alpha: float = 0.05       # false positive rate (flag alive as dead)
    sprt_beta: float = 0.05        # false negative rate (flag dead as alive)

    # Rolling Sharpe (§4.2)
    rolling_window_bars: int = 252         # 1 year
    rolling_sharpe_floor: float            # alert threshold — typically IS Sharpe - 1.0

    # CUSUM (§4.1)
    cusum_alpha: float = 0.05

    # Drawdown cone (§5.1)
    cone_alert_percentile: float = 0.95    # alert when drawdown exceeds 95th pct

    # Signal monitors (§6)
    turnover_alert_factor: float = 2.0     # alert if turnover > 2x IS baseline
    autocorr_alert_drop: float = 0.30      # alert if autocorr drops > 0.30 below IS

    # Sizing response (§7)
    sizing_schedule: SizingSchedule        # maps monitor states to position multipliers
```

These parameters cannot be edited once the strategy is deployed. If a researcher wants to change them, they must deploy a new strategy version — creating an audit trail that the criteria changed.

---

## 3.5 Portfolio research holdout (retrospective)

When the same four-test suite runs in **portfolio research holdout** (`portfolio_research/holdout/strategy_monitoring.py`), windows differ from a naive “full holdout block” comparison:

- **Reference μ:** validation period only (default 2018–2022) — uncontaminated OOS drift.
- **Reference σ:** train + validation pooled for precision; if period volatilities differ by more than 30%, σ is weighted 70% toward validation (configurable).
- **Evaluation:** trailing 12 months ending at the holdout end date, re-evaluated at each month-end for `monitoring_history.csv`.
- **Aggregation:** Green / Yellow / Red from how many of the four tests fail (0–1 / 2 / 3–4); advisory weights 1.0 / 0.5 / 0.0. Researcher overrides are optional and display-only in Phase 1.

See `portfolio_holdout.md` §2.0 for artifact paths and UI rollup cards.

---

## 4. Performance-Based Monitors

These tests operate on the live return stream and test whether it is consistent with the IS distribution. They are lagging by construction — they require accumulated return history before producing meaningful signals.

### 4.1 CUSUM with IS Parameters

The CUSUM test from the validation suite (§3.3 of `validation.md`) extends directly to live monitoring. IS parameters μ_IS and σ_IS are fixed at deployment and never updated.

At each bar $t$ in the live period:

$$z_t = \frac{r_t - \mu_\text{IS}}{\sigma_\text{IS}}, \quad S_t = \sum_{i=1}^{t} z_i$$

$$C = \frac{\max(0, -\min_t S_t)}{\sqrt{T_\text{live}}}$$

A break is flagged when $\min_t S_t < -1.36\sqrt{T_\text{live}}$ (5% lower envelope only). Positive CUSUM excursions above the upper envelope do **not** fail monitoring. The CUSUM series is reset after each flagged break — detection of one structural change should not permanently elevate the statistic.

CUSUM detects *cumulative underperformance* vs the IS mean. It is sensitive to persistent negative drift — a strategy earning consistently below its IS mean will accumulate a large negative CUSUM even if no single period is unusual. This makes it well-suited for detecting gradual edge decay as well as abrupt breaks.

### 4.2 Rolling Sharpe Ratio

Compute the annualised Sharpe over a trailing window of $W$ bars (default $W = 252$):

$$\widehat{SR}_t = \frac{\bar{r}_{t-W:t}}{\hat{\sigma}_{t-W:t}} \times \sqrt{252}$$

Plot as a time series overlaid with:
- The IS Sharpe (horizontal dashed line)
- The IS Sharpe 95% CI lower bound (alert threshold)
- A ±1 SE band centred on the IS Sharpe

**Critical caveat:** The rolling Sharpe is a noisy estimator on 252-bar windows. Its 95% CI for a true Sharpe of 1.0 is approximately ±0.76 — meaning a rolling Sharpe of 0.24 is statistically consistent with a live strategy of true Sharpe 1.0. Show this CI band prominently so researchers do not over-interpret short-run fluctuations. The rolling Sharpe is a visualisation tool, not a trigger.

### 4.3 Sequential Probability Ratio Test (SPRT)

The SPRT is the statistically optimal test for sequential monitoring under a fixed pair of error rates. Unlike repeated t-tests — which inflate false positive rates each time they are applied — the SPRT maintains a single running statistic with provably controlled error rates (Wald-Wolfowitz theorem).

**Setup:**

- H₀: strategy expected return = 0 (dead)
- H₁: strategy expected return = μ_IS (alive at IS rate)
- α: maximum false positive rate (flag dead as alive) — default 0.05
- β: maximum false negative rate (flag alive as dead) — default 0.05

**Update rule** (Gaussian returns, known variance σ²_IS):

$$\Lambda_t = \Lambda_{t-1} + \frac{\mu_\text{IS} \cdot r_t}{\sigma_\text{IS}^2} - \frac{\mu_\text{IS}^2}{2\sigma_\text{IS}^2}$$

with $\Lambda_0 = 0$.

**Decision boundaries:**

$$A = \log\!\left(\frac{1-\beta}{\alpha}\right), \quad B = \log\!\left(\frac{\beta}{1-\alpha}\right)$$

For $\alpha = \beta = 0.05$: $A \approx +2.94$, $B \approx -2.94$.

**At each bar:**

| Condition | Decision |
|---|---|
| $\Lambda_t > A$ | Strong evidence strategy is alive — no action |
| $B \leq \Lambda_t \leq A$ | Continue monitoring — insufficient evidence either way |
| $\Lambda_t < B$ | Trigger removal protocol (§7) |

```python
def sprt_update(
    lambda_prev: float,
    r_t: float,
    mu_is: float,
    sigma2_is: float,
) -> float:
    return lambda_prev + (mu_is * r_t / sigma2_is) - (mu_is**2 / (2 * sigma2_is))

def sprt_boundaries(alpha: float, beta: float) -> tuple[float, float]:
    A = np.log((1 - beta) / alpha)
    B = np.log(beta / (1 - alpha))
    return A, B
```

**Implementation note:** σ²_IS should be the IS return variance, not a long-run variance estimate used for serial-dependence adjustments. The SPRT likelihood function uses the per-observation variance. For strategies with significant return autocorrelation, the SPRT will have lower effective power than the i.i.d. calculation suggests — this is expected and honest.

**Why SPRT over repeated t-tests:** A t-test applied monthly over 24 months has an effective false positive rate far above the nominal 5% due to multiple comparisons. The SPRT controls error rates over the entire monitoring horizon, not just at each test application.

---

## 5. Calibration Tools

These tools do not produce pass/fail signals. They calibrate the researcher's expectations by showing what normal variance looks like for a strategy with given IS characteristics. Their primary function is preventing the Longmore mistake — abandoning a good strategy during a statistically normal drawdown because it looks alarming in isolation.

### 5.1 Drawdown Probability Cone

**What it shows:** The distribution of drawdowns you should *expect* for a strategy with these IS characteristics, before interpreting any specific drawdown as evidence of edge decay.

**Construction:**

1. From IS parameters: annualised Sharpe SR_IS, annualised vol σ_IS.
2. Simulate $N = 10{,}000$ daily return paths of length $T$ (e.g. 5 years forward):
   $$r_t^{(n)} \sim \mathcal{N}\!\left(\frac{SR_\text{IS} \cdot \sigma_\text{IS}}{\sqrt{252}},\ \frac{\sigma_\text{IS}}{\sqrt{252}}\right)$$
3. For each path, compute the drawdown series $D_t^{(n)} = W_t^{(n)} / \max_{s \leq t} W_s^{(n)} - 1$.
4. At each $t$, compute percentiles across paths: 5th, 25th, 50th, 75th, 95th.
5. Plot these percentile bands as the cone. Overlay the actual equity curve.

**Interpretation:** A live drawdown sitting at the 40th percentile of the cone is a normal event for this strategy. A drawdown at the 97th percentile is genuinely unusual — but even then it is not definitive evidence of edge decay, merely evidence that warrants a response (§7).

The cone is recomputed at deployment using IS parameters and is not updated with live results. It represents the *prior* expected experience, not a posterior updated on live data.

### 5.2 Expected Return Envelope

A simpler version of the cone: plot the ±1σ and ±2σ expected P&L envelope starting from deployment, based on IS mean and vol. The actual cumulative P&L is overlaid.

Cumulative P&L after $T$ bars, under IS parameters:

$$E[P_T] = \mu_\text{IS} \cdot T, \quad \text{Std}[P_T] = \sigma_\text{IS} \cdot \sqrt{T}$$

The ±2σ band contains approximately 95% of expected outcomes. Persistently tracking below the -2σ line without CUSUM or SPRT triggering is still within the realm of unlucky-but-not-dead. It is context, not a trigger.

---

## 6. Signal-Level Monitors (Leading Indicators)

Performance-based monitors are lagging — they need months of return history. Signal-level monitors operate on the strategy's position and forecast data, which updates daily, and can flag structural changes before they accumulate in P&L.

### 6.1 Turnover Rate

**What it measures:** Whether the strategy is changing positions at the expected rate.

Rolling 63-bar (3 month) average daily turnover, compared to the IS baseline:

$$\text{turnover\_ratio}_t = \frac{\overline{\text{turnover}}_{t-63:t}}{\overline{\text{turnover}}_\text{IS}}$$

Alert conditions:
- `turnover_ratio > 2.0`: strategy has become erratic — signal is flickering, potentially noisy data or signal breakdown
- `turnover_ratio < 0.3`: strategy has gone nearly flat — signal may have died before P&L shows it

Both conditions are suspicious for different reasons. A trend-following strategy that stops turning over has stopped detecting trends; one that turns over constantly has lost its directional bias.

### 6.2 Signal Autocorrelation

**What it measures:** Whether the temporal structure of the signal is preserved.

For trend-following strategies, daily position changes should be positively autocorrelated — positions persist rather than flip randomly. Collapse in autocorrelation indicates the signal has lost its directional structure.

Rolling 252-bar first-order autocorrelation of daily position:

$$\hat{\rho}_t = \text{corr}(p_{t-252:t-1},\ p_{t-251:t})$$

Alert when $\hat{\rho}_t < \rho_\text{IS} - 0.30$ (autocorrelation has dropped more than 0.30 below IS baseline).

### 6.3 Long/Short/Flat Distribution

**What it measures:** Whether the strategy is spending expected proportions of time in each position state.

Rolling 252-bar fraction of time long, short, and flat, compared to IS baseline fractions. Alert when any fraction deviates more than 20 percentage points from its IS baseline for three consecutive months.

A mean-reversion strategy that should be 40% long, 40% short, 20% flat but has been 80% flat for a quarter has either run out of signals or its entry conditions are no longer being met. This may be a data issue, a market structure change, or a sign the regime is no longer suitable.

---

## 7. Response Framework

### 7.1 Probabilistic Sizing — Not Binary On/Off

The correct response to evidence of weakening is to reduce position size gradually, not to switch the strategy off. This is the operationalisation of the Bayesian posture: as evidence accumulates that the expected return may be lower than IS estimated, reduce risk proportionally. Abrupt removal based on weak evidence is the Longmore mistake — it abandons a potentially sound strategy during normal variance.

The SPRT log-likelihood ratio $\Lambda_t$ serves naturally as a confidence proxy:
- $\Lambda_t$ near $A$ (upper boundary): high confidence strategy is alive → full size
- $\Lambda_t$ near zero: evidence is neutral → moderate size reduction
- $\Lambda_t$ approaching $B$ (lower boundary): accumulating evidence of death → significant size reduction

A default sizing schedule (researcher must pre-specify before deployment):

| Condition | Position Multiplier |
|---|---|
| All monitors green, $\Lambda_t > 0$ | 1.00 (full size) |
| Rolling Sharpe < IS floor OR $\Lambda_t < 0$ | 0.75 |
| CUSUM triggered OR $\Lambda_t < B/2$ | 0.50 |
| CUSUM triggered AND $\Lambda_t < B/2$ | 0.25 |
| SPRT crosses removal boundary ($\Lambda_t < B$) | 0.00 — removal protocol |

Sizing reductions are applied to new position calculations. Existing positions are not force-liquidated; they are allowed to unwind naturally as the model moves to smaller targets.

### 7.2 Removal Protocol

Removal is triggered only when the SPRT crosses the lower boundary $B$. The researcher receives a notification with:

- The SPRT series showing the accumulation of evidence
- The current drawdown vs the probability cone (is the drawdown also unusual?)
- Signal-level monitor states (turnover, autocorrelation)
- A confirmation requirement before the strategy is deactivated

Confirmation is required because SPRT removal is a one-way decision — restarting a removed strategy restarts the SPRT at $\Lambda_0 = 0$ and does not recover the previous evidence. The researcher should review all monitors before confirming.

**The bar is intentionally high.** For a Sharpe 0.5 strategy the SPRT lower boundary at $\alpha = \beta = 0.05$ requires approximately 18 months of zero-return data to trigger — this reflects the honest statistical power available, not excessive conservatism. A strategy should not be removed for experiencing a drawdown that the probability cone shows is at the 70th percentile of expected experience.

### 7.3 Conditions That Warrant Removal Outside SPRT

Two situations justify removal outside the SPRT framework, because they represent structural rather than statistical problems:

- **Data or execution failure**: the strategy's data feed has failed, fills are not occurring, or position limits are preventing the strategy from executing. These are operational problems, not edge decay.
- **Market structure no longer supports the instrument**: the futures contract has been delisted, liquidity has collapsed, or the instrument is otherwise untradeable. The strategy cannot generate returns regardless of its edge.

Both conditions should be detectable from signal-level monitors (turnover dropping to zero, position stuck) before they require a performance-based decision.

---

## 8. Portfolio-Level Monitors

Beyond individual strategy monitoring, two portfolio-level diagnostics matter for a multi-strategy deployment.

### 8.1 Correlation Regime Monitor

The portfolio's IDM (Instrument Diversification Multiplier) was calibrated on IS return correlations. If live strategy correlations increase materially — strategies that were uncorrelated in IS become correlated in a drawdown — the portfolio is carrying more risk than the IDM assumes.

Rolling 252-bar pairwise correlation matrix vs IS correlation matrix. Alert when the average pairwise correlation across all strategy pairs exceeds the IS average by more than 0.20 for two consecutive quarters.

This is a portfolio-level risk management signal, not a strategy removal trigger. The appropriate response is to reduce overall portfolio exposure rather than remove any individual strategy.

### 8.2 Portfolio CUSUM

A CUSUM test on portfolio-level returns (not individual strategy returns) using IS portfolio μ and σ. This catches cases where multiple strategies are simultaneously underperforming in ways that do not individually trigger their own CUSUM tests but collectively represent a meaningful portfolio-level structural break.

---

## 9. What the Platform Does vs What the Researcher Decides

| Responsibility | Platform | Researcher |
|---|---|---|
| Compute and display all monitors daily | ✓ | |
| Store monitoring config as immutable deployment metadata | ✓ | |
| Alert when thresholds are crossed | ✓ | |
| Apply sizing schedule automatically | ✓ | |
| Confirm SPRT-triggered removal | | ✓ — required |
| Define sizing schedule before deployment | | ✓ — required |
| Define removal criteria before deployment | | ✓ — required |
| Interpret whether a drawdown is signal or noise | | ✓ — judgement |
| Decide whether to override a triggered removal | | ✓ — with audit trail |

The platform enforces pre-commitment and makes evidence visible. It does not make removal decisions autonomously. A researcher who overrides a triggered removal must confirm, and the override is logged permanently in the strategy audit trail.

---

## 10. Computation Contract

### 10.1 Inputs (from deployment metadata)

| Field | Source | Description |
|---|---|---|
| `mu_is` | IS test suite | IS mean daily return |
| `sigma_is` | IS test suite | IS daily return std |
| `sr_is` | IS test suite | IS annualised Sharpe |
| `monitoring_config` | Researcher, set at deployment | Thresholds and sizing schedule |

### 10.2 Per-Bar Outputs

```python
@dataclass
class MonitoringSnapshot:
    as_of: datetime

    # Performance monitors
    cusum_statistic: float
    cusum_triggered: bool
    sprt_lambda: float
    sprt_state: Literal["alive", "monitoring", "remove"]
    rolling_sharpe: float           # trailing window

    # Calibration
    drawdown_current: float
    drawdown_percentile: float      # position in probability cone

    # Signal monitors
    turnover_ratio: float           # vs IS baseline
    signal_autocorr: float          # rolling vs IS baseline
    long_fraction: float
    short_fraction: float
    flat_fraction: float

    # Response
    recommended_size_multiplier: float   # from pre-specified sizing schedule
    alerts: list[str]                    # human-readable alert messages
```

### 10.3 Update Frequency

All monitors update daily after the trading session closes. The SPRT and CUSUM are incremental — they carry state from the previous bar and update in O(1) per bar. The drawdown cone is computed once at deployment and looked up by current drawdown level. Signal monitors require a rolling window; all are sub-second on typical strategy histories.
