# Portfolio Addition Gate

## 1. Purpose

This document specifies the portfolio addition gate — a dedicated evaluation phase that runs after a strategy has passed all individual robustness tests (IS, perturbation, validation) and before it is committed to the portfolio. The gate answers one question:

> **Does adding this strategy improve the portfolio, using only data it has already seen?**

The reasoning is conservative by design: if a strategy cannot improve the portfolio on IS + validation data — the most favourable possible evaluation — it has no credible path to improving it out-of-sample. The gate uses combined IS + validation data and therefore introduces no contamination of the project test zone.

**What the gate is not:** It is not a mechanism to tune or improve a strategy. It is a binary pass/fail. A strategy that fails cannot be adjusted in response to this result and re-evaluated — that would convert the validation data into a fitness function. Failure means the strategy is discarded and a new development cycle begins from scratch.

Related documents:
- `docs/SaaS/robustness_tests/validation.md` — individual strategy validation; must pass before this gate
- `docs/SaaS/weight_layer.md` — weight layer method; must be selected before this gate runs
- `docs/SaaS/zone_manager.md` §8 — contamination doctrine
- `docs/SaaS/research_flow.md` — where this gate sits in the end-to-end sequence

---

## 2. When This Gate Runs

The gate runs after:
1. The strategy has passed all IS robustness tests
2. The strategy has passed all validation robustness tests
3. The weight layer method has been selected (IS walk-forward CV)

The gate requires at least one strategy already committed to the portfolio. For the first strategy, the gate is skipped — there is no existing portfolio to compare against.

**Data used:** Combined IS + validation return series. The exact date range is the union of all IS and validation zone bars the strategy was evaluated on. No data from the project test zone is used at any point.

---

## 3. The Analytical Hurdle

**What it answers:** Can this strategy theoretically improve the portfolio, given its Sharpe and its correlation with the existing portfolio?

From portfolio theory, adding a new strategy to an existing portfolio improves the portfolio's Sharpe ratio if and only if:

$$SR_\text{new} > \rho_{\text{new}, P} \times SR_P$$

where:
- $SR_\text{new}$ = NW-adjusted Sharpe of the new strategy on combined IS + val data
- $\rho_{\text{new}, P}$ = **effective correlation** of the new strategy with the existing portfolio (see §3.1)
- $SR_P$ = NW-adjusted Sharpe of the existing portfolio on the same data

The right-hand side is the **correlation hurdle** — the minimum Sharpe the new strategy must demonstrate to be worth adding at any positive weight.

**Interpretation by region:**

| Scenario | Condition | Meaning |
|---|---|---|
| Zero correlation | $\rho \approx 0$ | Any positive Sharpe clears the hurdle — pure diversification benefit |
| Moderate correlation | $\rho = 0.5$, $SR_P = 1.0$ | New strategy needs $SR > 0.50$ |
| High correlation | $\rho = 0.8$, $SR_P = 1.0$ | New strategy needs $SR > 0.80$ — strong standalone required |
| Very high correlation | $\rho \geq 0.90$ | Strategy is nearly redundant with the portfolio; only the strongest edge clears |

```python
@dataclass(frozen=True)
class AnalyticalHurdleResult:
    sr_new: float
    sr_portfolio: float
    corr_unconditional: float        # ρ across all periods
    corr_drawdown_conditional: float # ρ restricted to drawdown stress periods
    corr_effective: float            # max(unconditional, drawdown_conditional) — used in hurdle
    hurdle: float                    # corr_effective × SR_P
    margin: float                    # sr_new - hurdle (positive = clears)
    passed: bool                     # sr_new > hurdle
    drawdown_overlap: float          # fraction of drawdown days that coincide
    joint_drawdown_depth: float      # avg combined portfolio DD when both in drawdown
```

**Role in the gate:** Soft gate. Failing is a strong signal to discard, but the empirical comparison (§4) is the primary gate.

---

### 3.1 Drawdown Correlation Analysis

Unconditional return correlation is measured across all market conditions and includes quiet periods where both strategies are doing little. The correlations that matter most for portfolio risk are those during drawdown periods — when both strategies are losing simultaneously, the diversification assumption is most consequential.

Two strategies can show ρ = 0.1 unconditionally but ρ = 0.7 during market stress. This is well-documented empirically: crisis episodes (2008, March 2020, 2022) produce correlation convergence across systematic strategies. The **effective correlation** used in the analytical hurdle is the more conservative of unconditional and drawdown-conditional:

$$\rho_\text{effective} = \max(\rho_\text{unconditional},\ \rho_\text{drawdown})$$

**Conditional correlation during stress periods:**

A "stress period" is any day where either the new strategy or the existing portfolio is in a drawdown exceeding a threshold from its rolling 252-bar peak:

```python
def drawdown_from_peak(returns: np.ndarray, window: int = 252) -> np.ndarray:
    equity = np.cumprod(1 + returns)
    rolling_max = np.maximum.accumulate(equity)  # simplified; use rolling window in practice
    return (equity - rolling_max) / rolling_max  # negative values = in drawdown

def conditional_drawdown_correlation(
    new_returns: np.ndarray,
    portfolio_returns: np.ndarray,
    drawdown_threshold: float = -0.05,  # 5% drawdown
) -> float:
    dd_new = drawdown_from_peak(new_returns)
    dd_port = drawdown_from_peak(portfolio_returns)
    stress_mask = (dd_new < drawdown_threshold) | (dd_port < drawdown_threshold)
    if stress_mask.sum() < 30:  # insufficient stress periods — return unconditional
        return float(np.corrcoef(new_returns, portfolio_returns)[0, 1])
    return float(np.corrcoef(new_returns[stress_mask], portfolio_returns[stress_mask])[0, 1])
```

**Drawdown overlap ratio:**

Measures the fraction of one strategy's drawdown days that coincide with the other's — independent of how large each drawdown is:

$$\text{overlap} = \frac{|\{t : \text{DD}_\text{new}(t) < 0\} \cap \{t : \text{DD}_P(t) < 0\}|}{\min\!\left(|\{t : \text{DD}_\text{new}(t) < 0\}|,\ |\{t : \text{DD}_P(t) < 0\}|\right)}$$

A value near 1.0 means whenever the new strategy is in drawdown, the portfolio is also in drawdown — a direct measure of "they fail together." This is often more informative than correlation for assessing regime coverage.

**Joint drawdown depth:**

On days when both the new strategy and the existing portfolio are simultaneously in drawdown, what is the average combined portfolio drawdown? Compared to what independent drawdowns would produce:

$$\text{joint\_DD\_depth} = \mathbb{E}\!\left[\frac{W_\text{with}(t) - \max_{s \leq t} W_\text{with}(s)}{\max_{s \leq t} W_\text{with}(s)} \,\middle|\, \text{DD}_\text{new}(t) < 0 \text{ and } \text{DD}_P(t) < 0\right]$$

A deep joint drawdown relative to either strategy's standalone drawdown indicates the strategies amplify each other's losses rather than cushioning them.

**Alert thresholds:**

| Measure | Alert threshold | Meaning |
|---|---|---|
| $\rho_\text{drawdown} - \rho_\text{unconditional}$ | > 0.25 | Significant crisis correlation uplift — strategies appear more diversified than they are |
| Drawdown overlap ratio | > 0.60 | Strategies spend the majority of their drawdown time failing simultaneously |
| Joint drawdown depth | > 1.5× average individual DD | Combined drawdowns are materially worse than either strategy alone |

These are observational alerts, not additional gates. They inform the researcher's qualitative judgement about whether the strategy provides genuine regime diversification. A strategy that clears the hurdle but shows high drawdown overlap should be noted as providing less protection than the unconditional correlation suggests.

---

## 4. Empirical Portfolio Sharpe Comparison (Primary Gate)

**What it answers:** Does the portfolio's Sharpe actually improve when this strategy is added at the weight the algorithm assigns?

The analytical hurdle is derived from two-asset theory and assumes the new strategy receives a meaningful, theoretically optimal weight. In practice the weight layer may assign the strategy a small weight due to its correlation structure with the full set of existing strategies — making the empirical improvement negligible even when the analytical hurdle is cleared.

The empirical test cuts through this: it uses the actual weight layer output.

**Procedure:**

1. Compute the combined IS + validation portfolio return stream **without** the new strategy, using the weight layer's current fitted weights.
2. Re-fit the weight layer **with** the new strategy included, using the same IS data and the same weight layer method.
3. Compute the combined IS + validation portfolio return stream **with** the new strategy.
4. NW-adjust both Sharpe ratios.
5. Compute $\Delta SR = SR_\text{with} - SR_\text{without}$.
6. Block bootstrap CI on $\Delta SR$ (block length $L = T^{1/3}$, 1000 iterations).

```python
@dataclass(frozen=True)
class EmpiricalComparisonResult:
    sr_without: float               # portfolio Sharpe without new strategy
    sr_with: float                  # portfolio Sharpe with new strategy added
    delta_sr: float                 # sr_with - sr_without
    delta_sr_ci: BootstrapCI        # 95% bootstrap CI on delta_sr
    weight_assigned: float          # weight the weight layer gave the new strategy
    passed: bool                    # delta_sr > 0
```

**Pass condition:** $\Delta SR > 0$.

This is deliberately permissive. The user's principle applies: if the strategy cannot produce a positive $\Delta SR$ even on the data it was trained on, it cannot do so out-of-sample. A $\Delta SR$ that is positive but within the bootstrap CI of zero is still a pass — the point estimate must simply be positive.

**Interpreting the bootstrap CI:** If the CI on $\Delta SR$ is entirely above zero, the improvement is statistically distinguishable from noise. If the CI straddles zero, the improvement is real but small — the researcher should weigh this against the added portfolio complexity. Neither case changes the pass/fail outcome — only $\Delta SR > 0$ determines it.

---

## 5. Weight Assessment

**What it answers:** Does the weight layer assign the strategy a meaningful weight, or is it effectively zero-weighted?

A strategy that clears both the analytical hurdle and the empirical comparison but receives a 1% weight contributes almost nothing to portfolio performance. The complexity cost of maintaining, monitoring, and refitting an additional strategy is not justified by a 1% allocation.

```python
@dataclass(frozen=True)
class WeightAssessmentResult:
    weight_assigned: float          # fraction of portfolio budget assigned by weight layer
    weight_floor: float             # configurable threshold (default 0.03)
    meaningful: bool                # weight_assigned >= weight_floor
```

**Alert threshold:** weight < 3%. Below this, the strategy is flagged as negligibly weighted. This is not a hard gate — the researcher may accept a small weight if they expect the strategy's role to grow as the portfolio evolves, or if the 1% diversification benefit is nonetheless the right decision. But the platform surfaces it explicitly so the decision is deliberate.

---

## 6. IDM Improvement

**What it answers:** Does adding this strategy increase the portfolio's Instrument Diversification Multiplier — the direct measure of portfolio-level diversification?

The IDM is computed from the cross-instrument correlation matrix of the portfolio. Adding a low-correlation strategy increases the IDM, which increases the total risk budget available to the portfolio. A strategy that raises the IDM is a net positive for portfolio leverage even if its standalone Sharpe is modest.

$$\Delta \text{IDM} = \text{IDM}_\text{with} - \text{IDM}_\text{without}$$

```python
@dataclass(frozen=True)
class IDMImprovementResult:
    idm_without: float              # IDM before adding new strategy
    idm_with: float                 # IDM after adding new strategy
    delta_idm: float                # idm_with - idm_without
```

**Interpretation:**

| $\Delta$IDM | Meaning |
|---|---|
| > 0.10 | Material diversification benefit — low correlation with existing set |
| 0.02 – 0.10 | Modest benefit — some diversification captured |
| < 0.02 | Negligible diversification — strategy is highly correlated with the existing set |
| < 0 | IDM decreases — strategy is more correlated than the average existing pair (unusual) |

$\Delta$IDM is observational — it does not gate the decision. It explains *why* the analytical hurdle result came out as it did. A negative $\Delta$IDM alongside a failed analytical hurdle tells the researcher the strategy is redundant with the existing set. A positive $\Delta$IDM alongside a cleared analytical hurdle tells them the portfolio genuinely benefits from the diversification.

---

## 7. Interpretation and Decision Logic

The four tests address distinct concerns. Reading them together tells the full story:

```
Analytical hurdle passed?
    NO → Strategy's Sharpe is too low given its correlation with the portfolio.
         Check ΔIDM: if close to zero, the strategy is largely redundant. Discard.
         If empirical ΔSR > 0 despite hurdle failure, weight is negligible (see Test 3).
    YES ↓

Empirical ΔSR > 0?
    NO → Strategy does not improve the portfolio at the weight the algorithm assigns.
         Even if analytically it should help, the weight layer sees no room for it
         in the current portfolio's correlation structure. Discard.
    YES ↓

Weight assigned ≥ floor?
    NO → Strategy adds value in principle but is effectively zero-weighted.
         Adding it produces negligible improvement at the cost of monitoring complexity.
         Researcher decides — flag is advisory, not a hard gate.
    YES ↓

PASS — Strategy approved for portfolio inclusion.
```

**Special case — hurdle failed, ΔSR > 0, weight < floor:**
The strategy has a very low weight, which is why the empirical test just barely passes despite the analytical hurdle failing. The weight assessment flag (Test 3) captures this. The researcher should be aware they are adding a strategy that the weight layer considers nearly negligible.

**Special case — hurdle cleared, ΔSR > 0, weight < floor:**
The strategy is genuinely good but the portfolio is already well-diversified in the direction this strategy covers. The marginal benefit is real but small. A reasonable decision is to proceed but note the limited impact.

---

## 8. Contamination Discipline

**Binary gate only.** The result of the portfolio addition gate is pass or fail. The following actions are explicitly prohibited after viewing results:

| Action | Contaminated? | Reason |
|---|---|---|
| Adding the strategy unchanged after passing | No | Gate result used for its intended purpose |
| Discarding the strategy after failing | No | Gate result used for its intended purpose |
| Adjusting the strategy's parameters to improve ΔSR and re-running | **Yes** | Validation + IS data becomes a fitness function |
| Changing the strategy's target instruments to reduce correlation | **Yes** | Strategy redesigned in response to gate outcome |
| Re-running the gate with a different weight layer method to find one where ΔSR > 0 | **Yes** | Weight layer selection becomes a fitness function |

The weight layer method is locked before this gate runs (see `weight_layer.md` §5). It cannot be changed to make a failing strategy pass.

If the strategy fails, the researcher may develop a **new, independent** strategy that serves the same portfolio role — but must start from scratch (new IS research, new validation) without carrying over any design decisions informed by the gate's output.

---

## 9. UI Surface

### 9.1 Portfolio Addition Gate Summary Card

```
┌─ Portfolio Addition Gate ────────────────────────────────────────────┐
│  New strategy:     ES_MEAN_REV_14                                     │
│  Evaluation data:  IS + Validation   2018-01-01 → 2021-12-31         │
│                                                                       │
│  Existing portfolio SR:    0.94                                       │
│  New strategy SR:          0.61   correlation to portfolio:  0.31    │
│  Correlation hurdle:       0.29   margin:  +0.32  ✓                  │
│                                                                       │
│  Portfolio SR without:     0.94                                       │
│  Portfolio SR with:        1.08   ΔSR:  +0.14   CI: [+0.02, +0.26]  │
│                                                                       │
│  Weight assigned:          0.14   ✓  above floor (0.03)              │
│  IDM without:              1.42   IDM with:  1.51   ΔIDM:  +0.09     │
│                                                                       │
│  Result:   PASS  ✓                                                    │
│  [Add to Portfolio]    [View Detail]                                  │
└───────────────────────────────────────────────────────────────────────┘
```

### 9.2 Failure Card with Diagnostic

```
┌─ Portfolio Addition Gate ────────────────────────────────────────────┐
│  New strategy:     ES_BREAKOUT_50                                     │
│  Evaluation data:  IS + Validation   2018-01-01 → 2021-12-31         │
│                                                                       │
│  Existing portfolio SR:    0.94                                       │
│  New strategy SR:          0.52   correlation to portfolio:  0.74    │
│  Correlation hurdle:       0.70   margin:  -0.18  ✗                  │
│                                                                       │
│  Portfolio SR without:     0.94                                       │
│  Portfolio SR with:        0.91   ΔSR:  -0.03   CI: [-0.12, +0.06]  │
│                                                                       │
│  Weight assigned:          0.06                                       │
│  IDM without:              1.42   IDM with:  1.40   ΔIDM:  -0.02     │
│                                                                       │
│  Result:   FAIL  ✗                                                    │
│  Diagnostic: strategy is too correlated with the existing portfolio   │
│  to improve diversification. Its Sharpe does not clear the hurdle    │
│  set by that correlation. The portfolio does not benefit from its     │
│  addition on IS + validation data.                                    │
│                                                                       │
│  [Discard Strategy]                                                   │
│                                                                       │
│  ⚠ To develop a replacement: start a new strategy from scratch.      │
│    Do not modify this strategy's parameters in response to this       │
│    result — that would contaminate your validation data.              │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 10. Computation Contract

### 10.1 Inputs

| Field | Type | Source | Description |
|---|---|---|---|
| `new_strategy_returns` | `pd.Series` | Backtest engine | Daily returns of new strategy on IS + val data |
| `existing_portfolio_returns` | `pd.Series` | Portfolio engine | Daily combined portfolio returns on IS + val data |
| `strategy_is_returns` | `pd.DataFrame` | Backtest engine | Per-strategy IS returns (for weight layer refit) |
| `weight_layer_config` | `WeightLayerConfig` | Portfolio config | Locked weight layer method and config |
| `weight_floor` | `float` | Platform config | Default 0.03 |
| `n_bootstrap` | `int` | Platform config | Default 1000 |

### 10.2 Outputs

```python
@dataclass(frozen=True)
class PortfolioAdditionReport:
    # §3 — Analytical hurdle
    analytical_hurdle: AnalyticalHurdleResult

    # §4 — Empirical comparison
    empirical_comparison: EmpiricalComparisonResult

    # §5 — Weight assessment
    weight_assessment: WeightAssessmentResult

    # §6 — IDM improvement
    idm_improvement: IDMImprovementResult

    # Aggregate
    passed: bool                    # delta_sr > 0 (primary gate)
    weight_warning: bool            # weight_assigned < weight_floor
    interpretation: str             # sentence-level diagnostic for UI
```

### 10.3 Worker Behaviour

All portfolio addition gate tests run as a single synchronous job triggered when the researcher clicks "Check Portfolio Fit" from the strategy validation results view. The weight layer refit (with new strategy included) is the most expensive step — it runs on the IS data only, using the same CV procedure as the original weight layer selection. Results are cached against the strategy version ID and the current portfolio version ID; they are invalidated if either changes.
