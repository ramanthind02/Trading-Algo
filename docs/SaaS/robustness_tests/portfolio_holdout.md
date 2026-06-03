# Portfolio Holdout Validation

## 1. Purpose

This document specifies the robustness tests for the project-level holdout set (the portfolio test zone defined in `zone_manager.md` §5). These tests serve two distinct purposes:

**Purpose 1 — Monitoring validation.** Apply the individual strategy monitoring framework (CUSUM, rolling Sharpe, drawdown cone) retrospectively across the holdout window to identify any strategies that died during that period. Strategies flagged by the monitoring tests are culled. This is the only portfolio modification the holdout permits.

**Purpose 2 — Portfolio construction validation.** Test whether the portfolio-level assumptions built during research — correlation structure, IDM calibration, diversification — held in genuinely OOS data.

The portfolio holdout cannot be used to improve portfolio performance. Observing that correlations were higher than expected, or that one strategy dominated returns, does not justify reweighting, removing, or adding strategies beyond what the monitoring tests trigger. Those observations are inputs to future portfolio construction decisions on new research — not adjustments to the current portfolio. The contamination rules from `zone_manager.md` §8 apply in full.

Related documents:
- `docs/SaaS/robustness_tests/monitoring.md` — individual strategy monitoring tests applied here retrospectively
- `docs/SaaS/robustness_tests/validation.md` — strategy-level validation; all strategies pass this before reaching the portfolio holdout
- `docs/SaaS/zone_manager.md` — holdout zone structure and contamination doctrine

---

## 2. Portfolio-Level Monitoring

### 2.0 Implemented windows and traffic light (portfolio research)

In `portfolio_research`, strategy and portfolio monitoring use **two distinct windows**:

| Window | Span | Role |
|--------|------|------|
| **Reference μ** | `validation_window` (default 2018–2022) | Clean OOS drift baseline for CUSUM and equity bands — do not roll forward |
| **Reference σ** | `train_window` + `validation_window` (pooled), or weighted toward validation if train/val vol differ by >30% | Stable volatility for CUSUM allowance, rolling Sharpe z-scores, and band width |
| **Evaluation** | Trailing 12 calendar months ending at `test_window.end` | Answers “is the strategy alive **now**?” |

The full holdout block (e.g. 2023–2026) is still used to **generate** returns; only the **evaluation slice** feeds the four robustness tests. At each calendar month-end inside the holdout, the same tests re-run on the trailing 12 months; results are stored in `monitoring_history.csv` (no per-month plots).

**Traffic light** (vote count on failed tests among Sharpe CI, CUSUM, rolling Sharpe z-score, equity bands):

| State | Failed tests | Advisory weight |
|-------|----------------|-----------------|
| Green | 0–1 | 1.0 (full) |
| Yellow | 2 | 0.5 (reduce, monitor) |
| Red | 3–4 | 0.0 (halt / researcher review) |

Monitoring is **advisory** in Phase 1: optional `monitoring_weight_overrides` in config override advisory weights for display only; the pipeline does not auto-cull strategies.

**Artifacts** (under `results/holdout/`):

- `monitoring_rollup.csv` — current status per strategy plus prior two month-end traffic lights
- `strategies/<name>/holdout_robustness_report.json` — current evaluation + `monitoring` block
- `strategies/<name>/monitoring_history.csv` — month-end time series
- `strategies/<name>/matplotlib/` — plots for the **current** trailing window only
- `strategies/<name>/<strategy>_full_period_tearsheet.html` — QuantStats tearsheet over **train + validation + full holdout** (DD and metrics for the entire timeline)

The individual strategy monitoring tests (CUSUM, rolling Sharpe, equity bands, Sharpe CI) run per-strategy on the **trailing evaluation window** as specified above. Researchers use the traffic light and history to decide whether to reduce or zero weight — not an automated kill switch.

Beyond individual strategy monitoring, the same tests are applied to the **combined portfolio return stream**. This catches a specific failure mode that per-strategy monitoring cannot: multiple strategies each underperforming by a small, individually-insignificant amount, which collectively represents a portfolio-level structural break.

### 2.1 Portfolio CUSUM

Using IS portfolio parameters $\mu_P$ (mean daily portfolio return) and $\sigma_P$ (daily portfolio return std):

$$z_t = \frac{r_t^P - \mu_P}{\sigma_P}, \quad S_t = \sum_{i=1}^{t} z_i, \quad C = \frac{\max(0, -\min_t S_t)}{\sqrt{T_\text{holdout}}}$$

Same lower envelope as individual strategy CUSUM: $\min_t S_t < -1.36\sqrt{T_\text{holdout}}$ at the 5% level flags portfolio-level underperformance. Upper-envelope breaches do not fail.

A portfolio CUSUM trigger without any individual strategy CUSUM triggers indicates a systemic problem — the market environment is unfavourable to all strategies simultaneously, or the correlation structure has changed so that strategies that diversified in IS are no longer doing so. Neither justifies portfolio modification, but both are important diagnostic information.

### 2.2 Portfolio Rolling Sharpe

Trailing 252-bar annualised Sharpe on the combined portfolio return stream. Overlaid with:
- IS portfolio Sharpe (horizontal dashed line)
- IS portfolio Sharpe ± 1 SE band
- Strategy-level culling events marked as vertical lines (showing when the monitoring framework removed individual strategies)

The culling event markers allow the researcher to see whether the portfolio's rolling Sharpe improved after dead strategies were removed — a retrospective validation that the monitoring framework is working correctly.

### 2.3 Portfolio Drawdown Cone

Monte Carlo simulation of 10,000 portfolio equity paths from IS portfolio Sharpe and vol, identical to the individual strategy drawdown cone in `monitoring.md` §5.1, applied to the portfolio as a whole.

The holdout portfolio equity curve is overlaid on the cone. This answers: is the portfolio's holdout drawdown experience within the range that IS parameters would predict, or is it genuinely anomalous?

---

## 3. Correlation Realisation Test

**What it answers:** Did the diversification assumption hold in the holdout? Were the pairwise strategy correlations consistent with what IS estimated?

The portfolio's IDM and position sizing are built on IS correlation estimates. If holdout correlations were materially higher — particularly during drawdowns — the portfolio carried more risk than intended and the IDM was mis-calibrated.

### 3.1 Pairwise Correlation Comparison

Compute the pairwise correlation matrix of strategy daily returns over the holdout window $\mathbf{C}_\text{holdout}$ and compare to the IS correlation matrix $\mathbf{C}_\text{IS}$:

**Frobenius distance:**
$$\Delta_F = \|\mathbf{C}_\text{holdout} - \mathbf{C}_\text{IS}\|_F = \sqrt{\sum_{i,j} (C^\text{holdout}_{ij} - C^\text{IS}_{ij})^2}$$

**Average pairwise correlation shift:**
$$\Delta\bar{\rho} = \bar{\rho}_\text{holdout} - \bar{\rho}_\text{IS}$$

where $\bar{\rho}$ is the mean of all off-diagonal elements.

Alert threshold: $\Delta\bar{\rho} > 0.20$ — correlations increased by more than 20 percentage points on average.

### 3.2 Effective Rank Comparison

A correlation matrix with high effective rank indicates a genuinely diversified strategy set. A matrix with low effective rank indicates strategies that co-move.

$$\text{eff\_rank}(\mathbf{C}) = \exp\!\left(-\sum_i \tilde{\lambda}_i \log \tilde{\lambda}_i\right), \quad \tilde{\lambda}_i = \frac{\lambda_i}{\sum_j \lambda_j}$$

where $\lambda_i$ are the eigenvalues of $\mathbf{C}$. Compare effective rank in IS vs holdout. A large drop (e.g. from 4.2 to 1.8 for a 6-strategy portfolio) indicates that strategies which diversified in IS became correlated in the holdout — the portfolio effectively traded as fewer independent bets.

```python
def effective_rank(corr_matrix: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    eigenvalues = np.maximum(eigenvalues, 0)
    normalized = eigenvalues / eigenvalues.sum()
    normalized = normalized[normalized > 0]
    return float(np.exp(-np.sum(normalized * np.log(normalized))))

def correlation_realisation_test(
    is_returns: pd.DataFrame,    # shape (T_is, N) — one column per strategy
    holdout_returns: pd.DataFrame,  # shape (T_holdout, N)
) -> CorrelationRealisationResult:
    C_is = is_returns.corr().values
    C_holdout = holdout_returns.corr().values

    frobenius = np.linalg.norm(C_holdout - C_is, "fro")
    avg_corr_is = (C_is.sum() - np.trace(C_is)) / (len(C_is)**2 - len(C_is))
    avg_corr_holdout = (C_holdout.sum() - np.trace(C_holdout)) / (len(C_holdout)**2 - len(C_holdout))

    return CorrelationRealisationResult(
        C_is=C_is,
        C_holdout=C_holdout,
        frobenius_distance=frobenius,
        avg_corr_is=avg_corr_is,
        avg_corr_holdout=avg_corr_holdout,
        avg_corr_shift=avg_corr_holdout - avg_corr_is,
        eff_rank_is=effective_rank(C_is),
        eff_rank_holdout=effective_rank(C_holdout),
        alert=avg_corr_holdout - avg_corr_is > 0.20,
    )
```

### 3.3 Drawdown Correlation Analysis

**What it answers:** Did strategies that appeared uncorrelated in normal conditions fail together during drawdown periods? Unconditional correlation can mask crisis correlation — ρ = 0.10 across all periods can become ρ = 0.70 when either strategy is in drawdown. This is the most dangerous form of diversification failure: the portfolio appears diversified until it needs to be.

**Conditional drawdown correlation.** Restrict the pairwise correlation to periods where either strategy's drawdown from peak exceeds −5%:

$$\rho_\text{DD}(A, B) = \text{Corr}(r_A, r_B \mid DD_A < -0.05 \text{ or } DD_B < -0.05)$$

Compute $\rho_\text{DD}$ for each pair over the IS window, then again over the holdout. An increase indicates strategies that diversified during normal IS conditions clustered under holdout stress.

**Drawdown overlap ratio.** For each strategy pair, the fraction of each strategy's drawdown days that coincide with the other's:

$$\text{overlap}(A, B) = \frac{|\{t : DD_A(t) < -0.05\} \cap \{t : DD_B(t) < -0.05\}|}{\min(|\{t : DD_A(t) < -0.05\}|,\ |\{t : DD_B(t) < -0.05\}|)}$$

An overlap above 0.60 means the strategies were simultaneously in drawdown more than 60% of the time either was down — they were not covering each other's weak periods.

**Joint drawdown depth.** When all strategies are simultaneously in drawdown, how deep is the combined portfolio drawdown relative to the average individual strategy drawdown?

$$\text{joint\_depth} = \mathbb{E}\!\left[DD_P(t) \mid \forall i: DD_i(t) < -0.05\right]$$

A joint depth more than 1.5× the average individual depth indicates strategies are amplifying each other's losses during stress, not absorbing them.

| Metric | Alert threshold | Interpretation |
|---|---|---|
| $\rho_\text{DD}^\text{holdout} - \rho_\text{DD}^\text{IS}$ (any pair) | > 0.25 | Crisis correlation rose significantly vs IS estimate |
| Drawdown overlap (any pair) | > 0.60 | Strategies failing together most of the time |
| Joint drawdown depth / avg individual DD | > 1.5× | Strategies amplifying each other's losses |

```python
def drawdown_from_peak(returns: np.ndarray) -> np.ndarray:
    equity = np.cumprod(1 + returns)
    rolling_max = np.maximum.accumulate(equity)
    return (equity - rolling_max) / rolling_max


def drawdown_correlation_realisation(
    is_returns: pd.DataFrame,       # shape (T_is, N)
    holdout_returns: pd.DataFrame,  # shape (T_holdout, N)
    drawdown_threshold: float = -0.05,
) -> DrawdownCorrelationResult:
    n = is_returns.shape[1]
    dd_is = is_returns.apply(lambda col: drawdown_from_peak(col.values))
    dd_holdout = holdout_returns.apply(lambda col: drawdown_from_peak(col.values))
    in_dd_is = dd_is < drawdown_threshold
    in_dd_holdout = dd_holdout < drawdown_threshold

    rho_dd_is = np.full((n, n), 1.0)
    rho_dd_holdout = np.full((n, n), 1.0)
    overlap = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            stress_is = in_dd_is.iloc[:, i] | in_dd_is.iloc[:, j]
            rho_is = (is_returns.loc[stress_is].iloc[:, [i, j]].corr().iloc[0, 1]
                      if stress_is.sum() >= 30
                      else is_returns.iloc[:, [i, j]].corr().iloc[0, 1])
            stress_h = in_dd_holdout.iloc[:, i] | in_dd_holdout.iloc[:, j]
            rho_h = (holdout_returns.loc[stress_h].iloc[:, [i, j]].corr().iloc[0, 1]
                     if stress_h.sum() >= 30
                     else holdout_returns.iloc[:, [i, j]].corr().iloc[0, 1])
            joint = (in_dd_holdout.iloc[:, i] & in_dd_holdout.iloc[:, j]).sum()
            min_dd = min(in_dd_holdout.iloc[:, i].sum(), in_dd_holdout.iloc[:, j].sum())
            rho_dd_is[i, j] = rho_dd_is[j, i] = rho_is
            rho_dd_holdout[i, j] = rho_dd_holdout[j, i] = rho_h
            overlap[i, j] = overlap[j, i] = joint / min_dd if min_dd > 0 else 0.0

    max_dd_corr_shift = float(np.nanmax(rho_dd_holdout - rho_dd_is))
    max_overlap = float(np.nanmax(np.triu(overlap, k=1)))
    return DrawdownCorrelationResult(
        rho_dd_is=rho_dd_is,
        rho_dd_holdout=rho_dd_holdout,
        overlap_matrix=overlap,
        max_dd_corr_shift=max_dd_corr_shift,
        max_overlap=max_overlap,
        alert=max_dd_corr_shift > 0.25 or max_overlap > 0.60,
    )
```

All drawdown correlation results are observational — no portfolio modification is permitted in response to them.

---

## 4. IDM Accuracy and Volatility Calibration

**What it answers:** Did the portfolio size itself correctly? Was the IDM-implied expected volatility consistent with what actually materialised?

The IDM maps IS correlations to a leverage multiplier. Given each strategy's IS volatility and the IDM, the model implies an expected portfolio volatility:

$$\sigma_P^\text{expected} = \sqrt{\mathbf{w}^\top \boldsymbol{\Sigma}_\text{IS} \mathbf{w}} \times \text{IDM}$$

where $\mathbf{w}$ is the vector of strategy weights and $\boldsymbol{\Sigma}_\text{IS}$ is the IS covariance matrix of strategy returns.

Compare to the realised holdout portfolio volatility:

$$\sigma_P^\text{realised} = \text{std}(r^P_\text{holdout}) \times \sqrt{252}$$

**Volatility ratio:**

$$\text{vol\_ratio} = \frac{\sigma_P^\text{realised}}{\sigma_P^\text{expected}}$$

| Vol Ratio | Interpretation |
|---|---|
| 0.80 – 1.20 | IDM well-calibrated — portfolio vol matched expectations |
| 1.20 – 1.50 | Moderate overcrowding — correlations higher than IS, portfolio ran hotter than intended |
| > 1.50 | Material IDM mis-calibration — portfolio was significantly more leveraged than intended |
| < 0.80 | IDM conservative — portfolio ran cooler than intended, potential underutilisation of risk budget |

**Bootstrap CI on realised vol:** Block bootstrap (same procedure as `validation.md` §3.2) to produce a 95% CI on $\sigma_P^\text{realised}$. If the IS expected vol falls within this CI, the deviation is consistent with sampling variation. If it falls outside, the model was genuinely mis-calibrated.

**Note:** A vol ratio above 1.0 does not mean the strategies failed — it means the correlation assumptions were optimistic. This is extremely common during market stress periods where correlations spike across all instruments. It is information for future IDM calibration, not grounds for portfolio modification.

---

## 5. Contribution Concentration

**What it answers:** Did all strategies contribute meaningfully to holdout returns, or did one strategy dominate? Did the portfolio actually behave as a diversified set of bets?

### 5.1 Strategy Contribution Attribution

For each strategy $i$, compute its holdout P&L contribution as a fraction of total portfolio P&L:

$$s_i = \frac{\text{PnL}_i^\text{holdout}}{\sum_j |\text{PnL}_j^\text{holdout}|}$$

Compare to the expected contribution implied by IS weights and IS Sharpe ratios:

$$s_i^\text{expected} = \frac{w_i \cdot SR_i^\text{IS}}{\sum_j w_j \cdot SR_j^\text{IS}}$$

A bar chart of actual vs expected contributions per strategy makes concentration immediately visible.

### 5.2 Herfindahl-Hirschman Index (HHI)

$$\text{HHI} = \sum_{i=1}^{N} s_i^2$$

| HHI | Interpretation |
|---|---|
| ≈ 1/N | Perfectly equal contributions — full diversification realised |
| 0.25 – 0.50 | Moderate concentration — one or two strategies dominating |
| > 0.50 | High concentration — portfolio behaving as a single-strategy fund |

Alert when HHI > 0.40. This does not trigger portfolio modification — a single strategy that genuinely outperformed all others in the holdout is not evidence of a problem. But an HHI of 0.8 from a 6-strategy portfolio suggests the other five strategies contributed nothing, which is useful context for interpreting the holdout Sharpe.

### 5.3 Negative Contributor Count

How many strategies had negative P&L in the holdout? For a diversified portfolio during a healthy period, zero or one negative contributor is expected. If three of six strategies were negative, the portfolio's positive holdout result was driven by the remaining three carrying the others — diversification was not functioning.

---

## 6. Portfolio Holdout Sharpe Assessment

**What it answers:** How does the portfolio's holdout Sharpe compare to what IS parameters predicted?

**Expected portfolio Sharpe** (from IS estimates and portfolio construction):

$$SR_P^\text{expected} = \frac{\mu_P^\text{IS}}{\sigma_P^\text{expected}}$$

**Realised holdout Sharpe** with block bootstrap 95% CI (same procedure as `validation.md` §3.2, applied to portfolio returns):

$$SR_P^\text{realised} \pm \text{bootstrap CI}$$

**Degradation ratio:** $SR_P^\text{realised} / SR_P^\text{expected}$. Interpretation thresholds are the same as the strategy-level degradation ratio in `validation.md` §3.1.

The key difference from the strategy-level comparison: at the portfolio level, a degradation ratio below 0.5 may reflect correlation blowout (§3) rather than individual strategy failure. Cross-reference with the correlation realisation and IDM accuracy results to separate these effects.

---

## 7. Interpretation and Permitted Actions

### 7.1 Decision Matrix

| Test | Outcome | Permitted action |
|---|---|---|
| Individual strategy CUSUM / monitoring | Triggered | Cull the flagged strategy |
| Portfolio CUSUM | Triggered, no individual triggers | No action — systemic environment, not strategy death |
| Correlation realisation | Alert ($\Delta\bar{\rho} > 0.20$) | No action — inform future IDM calibration only |
| Drawdown correlation | Alert (shift > 0.25 or overlap > 0.60) | No action — inform future IDM and correlation modelling |
| IDM accuracy | Vol ratio > 1.50 | No action — inform future position sizing only |
| Contribution concentration | HHI > 0.40 | No action — observational |
| Portfolio holdout Sharpe | Degradation ratio < 0.10 | No action — if monitoring didn't trigger, do not remove |

The only action the holdout permits is culling strategies flagged by the pre-specified monitoring rules. Everything else is diagnostic information. Poor holdout performance, high concentration, and correlation blowout are all observations that inform future research — they do not justify modifying the current portfolio.

### 7.2 What the Holdout Cannot Tell You

- **Whether to add strategies.** A poor holdout result is not evidence that new strategies are needed. Adding strategies motivated by poor holdout performance is contamination (§8.3 of `zone_manager.md`).
- **Whether to reweight.** Observing that strategy A contributed 80% of holdout returns while strategy B contributed nothing does not justify increasing A's weight or decreasing B's. The holdout is not an optimisation target.
- **Whether the strategies are genuinely dead.** A poor holdout Sharpe with no CUSUM trigger is normal variance, not evidence of edge decay. The monitoring framework is the arbiter of strategy death, not the holdout Sharpe headline.

### 7.3 Retrospective Monitoring Validation

One specific use of the holdout results is validating that the monitoring framework itself is appropriately calibrated. If the monitoring tests culled a strategy mid-holdout, you can observe whether culling improved the portfolio's subsequent returns within the holdout window. This is retrospective diagnostic information — it can inform future monitoring threshold choices in new projects, but it does not justify changing the current project's monitoring config.

---

## 8. UI Surfaces

### 8.1 Portfolio Holdout Summary Card

```
┌─ Portfolio Holdout Results ─────────────────────────────────────────┐
│  Holdout period:        2021-01-01 → 2024-12-31  (4 years)         │
│  Strategies at open:    6    Culled by monitoring:  1               │
│                                                                      │
│  Portfolio Sharpe:      0.81   expected  1.12   ratio  0.72  ✓     │
│  Realised vol:          14.2%  expected  12.8%  ratio  1.11  ✓     │
│                                                                      │
│  Avg correlation — IS:  0.09   holdout:  0.24   shift  +0.15  ✓   │
│  Effective rank  — IS:  4.8    holdout:  3.2                        │
│  DD correlation  — IS:  0.18   holdout:  0.31   shift  +0.13  ✓   │
│  Max DD overlap:        0.42   ✓                                    │
│                                                                      │
│  Contribution HHI:      0.31  ✓  Moderate concentration            │
│  Negative contributors: 1 of 5                                      │
│                                                                      │
│  Portfolio CUSUM:       PASS  ✓  No portfolio-level break detected  │
│                                                                      │
│  [Correlation Heatmaps]  [Contribution Chart]  [Drawdown Cone]     │
└──────────────────────────────────────────────────────────────────────┘
```

### 8.2 Correlation Heatmap Panel

Two heatmap rows on the same colour scale:
- **Row 1 — Unconditional:** IS correlation matrix vs holdout correlation matrix. Frobenius distance and average correlation shift displayed below.
- **Row 2 — Drawdown conditional:** IS drawdown correlation matrix vs holdout drawdown correlation matrix. Maximum pairwise shift and maximum overlap ratio displayed below.

Side-by-side layout makes correlation blowout and crisis clustering immediately visible. The gap between unconditional and drawdown-conditional rows reveals how much diversification benefit degrades during stress.

### 8.3 Contribution Waterfall

Horizontal bar chart showing each strategy's holdout P&L contribution alongside its expected contribution. Strategies sorted by expected contribution descending. The HHI is displayed as a single concentration score beneath the chart.

### 8.4 Portfolio Equity Curve with Culling Events

Portfolio equity curve across the full holdout window. Vertical dashed lines mark the dates when individual strategies were culled by the monitoring framework. The post-culling equity trajectory shows whether the removals improved portfolio behaviour.

---

## 9. Computation Contract

### 9.1 Inputs

| Field | Type | Source | Description |
|---|---|---|---|
| `portfolio_holdout_returns` | `pd.Series` | Portfolio engine | Daily combined portfolio returns over holdout window |
| `strategy_holdout_returns` | `pd.DataFrame` | Portfolio engine | Daily returns per strategy over holdout window |
| `is_portfolio_params` | `PortfolioISParams` | IS research | IS μ_P, σ_P, SR_P, correlation matrix, IDM |
| `strategy_weights` | `dict[str, float]` | Portfolio config | Strategy weights at holdout open |
| `monitoring_snapshots` | `list[MonitoringSnapshot]` | Monitoring | Per-strategy monitoring state across holdout |
| `culling_events` | `list[CullingEvent]` | Monitoring | Strategies removed and dates |

### 9.2 Outputs

```python
@dataclass(frozen=True)
class PortfolioHoldoutReport:
    # §2 — Portfolio-level monitoring
    portfolio_cusum: CUSUMResult
    portfolio_rolling_sharpe: np.ndarray
    portfolio_drawdown_cone: DrawdownConeResult

    # §3 — Correlation realisation
    correlation_realisation: CorrelationRealisationResult
    drawdown_correlation: DrawdownCorrelationResult

    # §4 — IDM accuracy
    vol_ratio: float
    vol_ratio_bootstrap_ci: BootstrapCI
    idm_alert: bool

    # §5 — Contribution concentration
    strategy_contributions: dict[str, float]       # realised
    expected_contributions: dict[str, float]       # from IS weights and SRs
    hhi: float
    n_negative_contributors: int

    # §6 — Portfolio Sharpe
    sr_realised: float
    sr_expected: float
    sr_bootstrap_ci: BootstrapCI
    degradation_ratio: float

    # Aggregate
    culling_events: list[CullingEvent]
    interpretation: str
```

### 9.3 Worker Behaviour

All portfolio holdout tests run as a single synchronous job triggered when the researcher opens the portfolio holdout view for the first time. Results are cached as a portfolio artifact. Monitoring snapshots are computed incrementally (one per bar) during the holdout window as part of the monitoring pipeline; the holdout report aggregates them rather than recomputing from scratch.
