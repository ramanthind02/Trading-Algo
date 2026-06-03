# Portfolio Addition Gate

## 1. Purpose

This document specifies the portfolio addition gate — a dedicated evaluation phase that runs after a strategy has passed all individual robustness tests (IS, perturbation, validation) and before it is committed to the portfolio. The gate answers one question:

> **Does adding this strategy improve the portfolio, using only data it has already seen?**

The reasoning is conservative by design: if a strategy cannot improve the portfolio on IS + validation data — the most favourable possible evaluation — it has no credible path to improving it out-of-sample. The gate uses combined IS + validation data and therefore introduces no contamination of the project test zone.

**What the gate is not:** It is not a mechanism to tune or improve a strategy. It is a binary pass/fail. A strategy that fails cannot be adjusted in response to this result and re-evaluated — that would convert the validation data into a fitness function. Failure means the strategy is discarded and a new development cycle begins from scratch.

For local `Trading-Algo` docs and code, this is the canonical `portfolio_addition` phase. Some compatibility surfaces still use the older `oos` name for commands or artifact folders, but the workflow meaning is portfolio admission, not a separate fourth strategy phase.

Related documents:
- `docs/SaaS/robustness_tests/validation.md` — individual strategy validation; must pass before this gate
- `docs/SaaS/weight_layer.md` — weight layer method; must be selected before this gate runs
- `docs/SaaS/weight_layer_spec.md` — asset-first `hierarchy_equal` tree used by the portfolio addition gate (default in feature-research `PortfolioSourceConfig`)
- `docs/SaaS/zone_manager.md` §8 — contamination doctrine
- `docs/SaaS/research_flow.md` — where this gate sits in the end-to-end sequence

---

## 2. When This Gate Runs

The gate runs after:
1. The strategy has passed all IS robustness tests
2. The strategy has passed all validation robustness tests
3. The weight layer method has been selected (IS cross-validation)

The gate requires at least one strategy already committed to the portfolio. For the first strategy, the gate is skipped — there is no existing portfolio to compare against, so there is nothing to compare against.

**Data used:** Combined IS + validation return series. The exact date range is the union of all IS and validation zone bars the strategy was evaluated on. No data from the project test zone is used at any point.

**Sequence of checks:** §3 pairwise redundancy (warning) → §4 analytical hurdle (soft gate) → §5 empirical ΔSR (primary gate) → §6 weight assessment (advisory) → §7 IDM (context only).

---

## 3. Pairwise Redundancy Check

**What it answers:** Is the new strategy too similar to any single strategy already in the portfolio?

The analytical hurdle in §4 measures correlation against the *portfolio as a whole*. That can look deceptively low: if your portfolio has eight strategies and the new one overlaps heavily with just one of them, the portfolio-level number gets diluted by all the others. The weight layer will recognise the overlap and assign a tiny weight — but by then you've already run the full gate. This check catches the problem up front.

**Procedure:** Compute the return correlation between the new strategy and each existing strategy individually, over the combined IS + validation data. Flag if any single pairwise correlation exceeds 0.75.

```python
@dataclass(frozen=True)
class PairwiseRedundancyResult:
    max_pairwise_corr: float        # highest correlation with any single existing strategy
    most_similar_strategy: str      # name of the most correlated existing strategy
    flagged: bool                   # max_pairwise_corr > 0.75
```

**This is a warning, not a hard gate.** A flag means: "this strategy largely does what an existing strategy already does — be deliberate about why you're adding it." A flagged strategy can still pass the gate if the empirical comparison shows a genuine improvement. The flag just ensures the researcher notices the overlap rather than discovering it after the fact from a near-zero weight.

| Max pairwise correlation | Interpretation |
|---|---|
| < 0.50 | No meaningful overlap with any individual strategy |
| 0.50 – 0.75 | Partial overlap — worth noting but not alarming |
| > 0.75 | High overlap with at least one existing strategy — flag raised |
| > 0.90 | Near-duplicate — almost certainly zero-weighted by the weight layer |

---

## 4. The Analytical Hurdle

**What it answers:** Does the new strategy's performance justify adding it, given how correlated it is with the existing portfolio?

The key insight is that correlation and Sharpe trade off against each other. A strategy that is nearly uncorrelated with the rest of the portfolio adds diversification value even with a modest Sharpe. A strategy that moves in lockstep with the portfolio needs to be materially better than the portfolio to earn its place.

From portfolio theory, adding a new strategy improves the portfolio's Sharpe ratio if and only if:

$$SR_\text{new} > \rho_{\text{new}, P} \times SR_P$$

where:
- $SR_\text{new}$ = Sharpe of the new strategy on combined IS + val data
- $\rho_{\text{new}, P}$ = **effective correlation** of the new strategy with the existing portfolio (see §3.1)
- $SR_P$ = Sharpe of the existing portfolio on the same data

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

**Role in the gate:** Soft gate. Failing is a strong signal to discard, but the empirical comparison (§5) is the primary gate.

---

### 4.1 Drawdown Correlation Analysis

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

## 5. Empirical Sharpe Comparison (Primary Gate — Sleeve Scope)

**What it answers:** Does the **sleeve's** combined Sharpe improve — by a meaningful amount — when this strategy is added at the weight the algorithm assigns **within that sleeve**?

In feature-research, a *sleeve* is the asset-first weight-layer bucket `asset_class / style_group` (for example `equity_indices / momentum`), derived from the candidate's `weight_hierarchy_group` and tickers. The gate runs portfolio phases on **only the ensembles that contribute streams to that sleeve**, then refits the weight layer on that subset. Pass/fail (`passed` on `portfolio_addition_report.json`) follows the **sleeve composite gate** (Sharpe + risk legs below).

**Full-portfolio metrics** (same checks on the global combined portfolio) are computed in parallel and stored under `portfolio` in the report JSON for context. A strategy can pass its sleeve but fail globally (or the reverse); the UI shows both.

**What the legacy full-portfolio-only description measured:** Does the portfolio's Sharpe actually improve — by a meaningful amount — when this strategy is added at the weight the algorithm assigns?

The analytical hurdle is based on simplified two-asset theory. It doesn't know what weight the strategy will actually receive once it enters a larger portfolio. A strategy can clear the hurdle but still receive a tiny weight from the weight layer (because it overlaps with several existing strategies simultaneously), making the real-world improvement negligible.

The empirical test cuts through this: it uses the actual weight layer output and requires the improvement to meet a minimum bar.

**Procedure:**

1. Compute the combined IS + validation portfolio return stream **without** the new strategy, using the weight layer's current fitted weights.
2. Re-fit the weight layer **with** the new strategy included, using the same IS data and the same weight layer method.
3. Compute the combined IS + validation portfolio return stream **with** the new strategy.
4. Compute both Sharpe ratios on the same combined sample.
5. Compute $\Delta SR = SR_\text{with} - SR_\text{without}$.
6. Block bootstrap CI on $\Delta SR$ (block length $L = T^{1/3}$, 1000 iterations).

```python
@dataclass(frozen=True)
class EmpiricalComparisonResult:
    sr_without: float               # portfolio Sharpe without new strategy
    sr_with: float                  # portfolio Sharpe with new strategy added
    delta_sr: float                 # sr_with - sr_without
    delta_sr_threshold: float       # minimum required improvement (default 0.02)
    delta_sr_ci: BootstrapCI        # 95% bootstrap CI on delta_sr
    weight_assigned: float          # weight the weight layer gave the new strategy
    passed: bool                    # delta_sr >= delta_sr_threshold
```

**Pass condition:** ΔSR ≥ 0.02.

The threshold exists because a ΔSR of +0.001 is indistinguishable from rounding noise — it adds a strategy to the portfolio for no practical benefit. The 0.02 floor is small enough that any genuinely useful strategy clears it easily, but large enough to filter out cases where the weight layer assigned a near-zero allocation and the improvement is purely cosmetic.

The core principle still holds: if the strategy cannot improve the portfolio on data it was trained on, it cannot do so out-of-sample. The threshold just ensures "improvement" means something measurable.

**Interpreting the bootstrap CI:** If the CI on ΔSR is entirely above zero, the improvement is robust across resampled time blocks. If the CI straddles zero, the improvement is present on average but sensitive to the specific period — the researcher should note this as a fragility signal. The CI is context, not a separate gate leg.

---

## 5.2 Portfolio Risk Impact (Composite Gate — Sleeve Scope)

**What it answers:** When the strategy is added at its fitted weight, does the **sleeve portfolio** get meaningfully worse on downside risk — not just better on Sharpe?

The composite primary gate requires **all** of the following on the same IS + validation sample:

| Leg | Pass condition (defaults) |
|-----|---------------------------|
| Sharpe (§5) | ΔSR ≥ 0.02 |
| Max drawdown | ΔmaxDD ≥ −0.01 (≤ 1 pp deeper MDD allowed) |
| Ulcer index | Δulcer ≤ +0.05 |
| Stress max drawdown | Δstress_maxDD ≥ −0.01 on stress bars |

**Stress bars** reuse §4.1: either the candidate or the sleeve portfolio (without the new strategy) is in drawdown below −5% (rolling 252-bar peak). If fewer than 30 stress bars exist, the stress leg is skipped (`stress_metrics_reliable: false`) and does not fail the gate; the UI warns.

Sortino and Calmar are reported for context only — they do not gate admission.

```python
@dataclass(frozen=True)
class PortfolioRiskImpactResult:
    max_dd_without: float
    max_dd_with: float
    delta_max_dd: float
    ulcer_without: float
    ulcer_with: float
    delta_ulcer: float
    stress_max_dd_without: float
    stress_max_dd_with: float
    delta_stress_max_dd: float
    stress_metrics_reliable: bool
    passed: bool  # AND of max_dd, ulcer, stress legs
```

`portfolio_addition_report.json` includes `gate_criteria` with per-leg booleans and `portfolio_risk_impact.csv` for offline review.

---

## 6. Weight Assessment

**What it answers:** Does the weight layer assign the strategy a meaningful weight, or is it effectively zero-weighted?

A strategy that clears both the analytical hurdle and the empirical comparison but receives a 1% weight contributes almost nothing to portfolio performance. The complexity cost of maintaining, monitoring, and refitting an additional strategy is not justified by a 1% allocation.

```python
@dataclass(frozen=True)
class WeightAssessmentResult:
    weight_assigned: float          # fraction of portfolio budget assigned by weight layer
    weight_floor: float             # configurable threshold (default 0.03)
    meaningful: bool                # weight_assigned >= weight_floor
```

**Alert threshold:** weight < 3%. Below this, the strategy is flagged as negligibly weighted. This is not a hard gate — the researcher may accept a small weight if they expect the strategy's role to grow as the portfolio evolves, or if the small diversification benefit is nonetheless the right decision. But the platform surfaces it explicitly so the decision is deliberate.

---

## 7. IDM Improvement

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

ΔIDM is observational — it does not gate the decision. It explains *why* the analytical hurdle result came out as it did. A negative ΔIDM alongside a failed analytical hurdle tells the researcher the strategy is redundant with the existing set. A positive ΔIDM alongside a cleared analytical hurdle confirms the portfolio genuinely benefits from the diversification.

---

## 8. Interpretation and Decision Logic

The checks address distinct concerns. Reading them together tells the full story:

```
Pairwise redundancy flagged? (max single-strategy correlation > 0.75)
    YES → The new strategy is very similar to an existing one. Proceed with caution —
          the weight layer will likely assign it a small allocation. Not a hard gate,
          but note it before continuing.
    ↓

Analytical hurdle passed?
    NO → The strategy's Sharpe is too low given how correlated it is with the portfolio.
         Check ΔIDM: if near zero, the strategy is largely redundant. Discard.
    YES ↓

Empirical ΔSR ≥ 0.02?
    NO → Sharpe leg failed. Discard.
    YES ↓

Portfolio risk impact passed?
    (max DD, ulcer, stress max DD within tolerances)
    NO → Sharpe improved but sleeve downside risk worsened materially. Discard.
    YES ↓

Weight assigned ≥ floor (3%)?
    NO → The strategy adds value in principle but the weight layer considers it nearly
         negligible. Adding it produces near-zero improvement at the cost of monitoring
         complexity. Researcher decides — this flag is advisory, not a hard gate.
    YES ↓

PASS — Strategy approved for portfolio inclusion.
```

**Special case — redundancy flagged, hurdle cleared, ΔSR ≥ 0.02, weight < floor:**
The pairwise flag predicted this outcome. The strategy overlaps heavily with one existing strategy, so the weight layer gave it almost nothing. The improvement is technically above the threshold but marginal. The researcher should decide whether that small allocation is worth the added maintenance.

**Special case — hurdle cleared, ΔSR ≥ 0.02, weight < floor:**
The strategy is genuinely useful but the portfolio is already well-diversified in the direction it covers. The marginal benefit is real but small. A reasonable decision is to proceed but note the limited impact.

**Special case — hurdle failed, ΔSR ≥ 0.02, weight < floor:**
The very low weight is why the empirical test just barely clears the threshold despite failing the hurdle. The researcher should be aware they are adding a strategy the weight layer considers nearly negligible.

---

## 9. Contamination Discipline

**Binary gate only.** The result of the portfolio addition gate is pass or fail. The following actions are explicitly prohibited after viewing results:

| Action | Contaminated? | Reason |
|---|---|---|
| Adding the strategy unchanged after passing | No | Gate result used for its intended purpose |
| Discarding the strategy after failing | No | Gate result used for its intended purpose |
| Adjusting the strategy's parameters to improve ΔSR and re-running | **Yes** | Validation + IS data becomes a fitness function |
| Changing the strategy's target instruments to reduce correlation | **Yes** | Strategy redesigned in response to gate outcome |
| Re-running the gate with a different weight layer method to find one where ΔSR ≥ 0.02 | **Yes** | Weight layer selection becomes a fitness function |

The weight layer method is locked before this gate runs (see `weight_layer.md` §5). It cannot be changed to make a failing strategy pass.

If the strategy fails, the researcher may develop a **new, independent** strategy that serves the same portfolio role — but must start from scratch (new IS research, new validation) without carrying over any design decisions informed by the gate's output.

---

## 10. UI Surface

### 10.1 Portfolio Addition Gate Summary Card

```
┌─ Portfolio Addition Gate ────────────────────────────────────────────┐
│  New strategy:     ES_MEAN_REV_14                                     │
│  Evaluation data:  IS + Validation   2018-01-01 → 2021-12-31         │
│                                                                       │
│  Pairwise check:   max corr 0.28 vs NQ_MEAN_REV_14   ✓  no overlap  │
│                                                                       │
│  Existing portfolio SR:    0.94                                       │
│  New strategy SR:          0.61   correlation to portfolio:  0.31    │
│  Correlation hurdle:       0.29   margin:  +0.32  ✓                  │
│                                                                       │
│  Portfolio SR without:     0.94                                       │
│  Portfolio SR with:        1.08   ΔSR:  +0.14   CI: [+0.02, +0.26]  │
│  Required ΔSR:             0.02   ✓                                  │
│                                                                       │
│  Weight assigned:          0.14   ✓  above floor (0.03)              │
│  IDM without:              1.42   IDM with:  1.51   ΔIDM:  +0.09     │
│                                                                       │
│  Result:   PASS  ✓                                                    │
│  [Add to Portfolio]    [View Detail]                                  │
└───────────────────────────────────────────────────────────────────────┘
```

### 10.2 Failure Card with Diagnostic

```
┌─ Portfolio Addition Gate ────────────────────────────────────────────┐
│  New strategy:     ES_BREAKOUT_50                                     │
│  Evaluation data:  IS + Validation   2018-01-01 → 2021-12-31         │
│                                                                       │
│  Pairwise check:   max corr 0.81 vs ES_BREAKOUT_30   ⚠  high overlap │
│                                                                       │
│  Existing portfolio SR:    0.94                                       │
│  New strategy SR:          0.52   correlation to portfolio:  0.74    │
│  Correlation hurdle:       0.70   margin:  -0.18  ✗                  │
│                                                                       │
│  Portfolio SR without:     0.94                                       │
│  Portfolio SR with:        0.91   ΔSR:  -0.03   CI: [-0.12, +0.06]  │
│  Required ΔSR:             0.02   ✗                                  │
│                                                                       │
│  Weight assigned:          0.06                                       │
│  IDM without:              1.42   IDM with:  1.40   ΔIDM:  -0.02     │
│                                                                       │
│  Result:   FAIL  ✗                                                    │
│  Diagnostic: this strategy is highly similar to ES_BREAKOUT_30       │
│  already in the portfolio. Its Sharpe isn't high enough to justify    │
│  adding something so correlated, and the portfolio Sharpe falls       │
│  when it is included.                                                 │
│                                                                       │
│  [Discard Strategy]                                                   │
│                                                                       │
│  ⚠ To develop a replacement: start a new strategy from scratch.      │
│    Do not modify this strategy's parameters in response to this       │
│    result — that would contaminate your validation data.              │
└───────────────────────────────────────────────────────────────────────┘
```

### 10.3 Sleeve portfolio tearsheets (validation phase)

When the gate runs as part of **Validation**, QuantStats HTML tearsheets are written for the **sleeve** scope (the same ensembles used for the primary composite gate):

- **With peers in sleeve:** six files — sleeve portfolio without vs with candidate × train, validation, and concatenated train+validation.
- **First in sleeve:** three files — with-candidate only (train, validation, train+validation).

Files live under `…/validation/sleeve_tearsheets_<asset>_<style>/` (for example `sleeve_tearsheets_equity_indices_mean_reversion_indices/`). Toggle with `portfolio_addition_gate.emit_sleeve_tearsheets` (default on). No extra portfolio refits beyond those already executed for the gate.

---

## 11. Computation Contract

### 11.1 Inputs

| Field | Type | Source | Description |
|---|---|---|---|
| `new_strategy_returns` | `pd.Series` | Backtest engine | Daily returns of new strategy on IS + val data |
| `existing_portfolio_returns` | `pd.Series` | Portfolio engine | Daily combined portfolio returns on IS + val data |
| `individual_strategy_returns` | `pd.DataFrame` | Portfolio engine | Per-strategy return series (for pairwise redundancy check) |
| `strategy_is_returns` | `pd.DataFrame` | Backtest engine | Per-strategy IS returns (for weight layer refit) |
| `weight_layer_config` | `WeightLayerConfig` | Portfolio config | Locked weight layer method and config |
| `pairwise_corr_threshold` | `float` | Platform config | Default 0.75 |
| `delta_sr_threshold` | `float` | Platform config | Default 0.02 |
| `weight_floor` | `float` | Platform config | Default 0.03 |
| `n_bootstrap` | `int` | Platform config | Default 1000 |

### 11.2 Outputs

```python
@dataclass(frozen=True)
class PortfolioAdditionReport:
    # §3 — Pairwise redundancy check
    pairwise_redundancy: PairwiseRedundancyResult

    # §4 — Analytical hurdle
    analytical_hurdle: AnalyticalHurdleResult

    # §5 — Empirical comparison
    empirical_comparison: EmpiricalComparisonResult

    # §6 — Weight assessment
    weight_assessment: WeightAssessmentResult

    # §7 — IDM improvement
    idm_improvement: IDMImprovementResult

    # Aggregate
    passed: bool                    # delta_sr >= delta_sr_threshold (primary gate)
    redundancy_warning: bool        # max pairwise corr > pairwise_corr_threshold
    weight_warning: bool            # weight_assigned < weight_floor
    interpretation: str             # plain-English diagnostic for UI
```

### 11.3 Worker Behaviour

All portfolio addition gate tests run as a single synchronous job triggered when the researcher clicks "Check Portfolio Fit" from the strategy validation results view. The weight layer refit (with new strategy included) is the most expensive step — it runs on the IS data only, using the same CV procedure as the original weight layer selection. Results are cached against the strategy version ID and the current portfolio version ID; they are invalidated if either changes.
