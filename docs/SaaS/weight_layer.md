# Weight Layer

## 1. Purpose

This document describes the philosophy and configuration of the weight layer in QuantFoundry. The weight layer determines how a portfolio combines the forecast signals from multiple strategies into a single set of position sizes. It is distinct from individual strategy fitting and operates above the strategy level.

The weight layer is configured once before the portfolio holdout is opened. It is not a robustness test — it is a portfolio construction decision with contamination implications identical to parameter selection.

Related documents:
- `docs/SaaS/zone_manager.md` — holdout zone structure and contamination doctrine
- `docs/SaaS/robustness_tests/portfolio_holdout.md` — portfolio-level evaluation after the weight layer config is locked

---

## 2. Signal Granularity

The most important conceptual decision in weight layer design is the unit of input: what counts as one signal?

### 2.1 Per-Instrument Signal

A strategy applied to a single instrument produces a **per-instrument signal**. Examples:
- RSI mean reversion on ES
- EWMAC trend-following on GC
- Bollinger breakout on CL

Each of these is a separate signal and enters the weight layer pool as an independent input. The weight layer sees N signals where N = (number of strategies) × (number of instruments each strategy trades).

Per-instrument signals are the common case. Pooling them together allows the weight layer to capture cross-instrument diversification — for example, reducing exposure when ES and GC signals become highly correlated.

### 2.2 Per-Portfolio Signal

A strategy that trades multiple instruments simultaneously as a single bet produces a **per-portfolio signal**. Examples:
- A calendar spread (long ES front month, short ES back month)
- A relative value strategy (long GC, short SI at a fixed ratio)
- A cross-asset momentum strategy whose P&L is computed on the combined position

Per-portfolio signals enter the weight layer as a single unit regardless of how many underlying instruments they touch. Splitting them into per-instrument components would destroy the signal's meaning — the trade is defined by the relationship between instruments, not the individual legs.

### 2.3 Pooling Principle

The weight layer pools all per-instrument signals from all strategies into a single correlation matrix and applies a single weighting method to the full pool. It does not maintain separate per-strategy or per-instrument subgroups unless sector constraints are specified (§4).

The rationale: diversification is a portfolio property. Estimating correlations and weights within artificial subgroups (e.g., "ES strategies only") leaves cross-group diversification uncaptured.

---

## 3. Weight Method Selection

The choice of weighting method — equal weight, correlation-adjusted, shrinkage-based, or others — is a research decision made on IS data before the portfolio holdout is opened.

### 3.1 Walk-Forward Cross-Validation on IS Data

Method selection uses walk-forward cross-validation across the IS window:

1. Divide the IS period into folds, each with a fit window and an OOS window, both entirely within IS.
2. For each candidate method, fit the weight layer on the fit window and evaluate on the OOS window.
3. Aggregate OOS performance across folds to produce a per-method leaderboard.
4. The researcher selects the method with the best CV performance (or overrides manually with a documented reason).

This selection happens once before any portfolio holdout results are viewed. The selected method is locked on the portfolio object with a timestamp. Changing the weight method after opening the holdout is contamination under the doctrine in `zone_manager.md` §8.

### 3.2 Default Method

The empirically validated default is `ledoit_wolf_min_corr` — a Ledoit-Wolf shrinkage estimator applied to the minimum correlation representation of the weight problem. This method was selected based on live portfolio performance across multiple strategy sets and consistently outperformed equal weighting and simpler correlation adjustments.

The default is a starting point. Walk-forward CV may recommend a different method for a given portfolio, and the researcher can override it with explicit justification.

### 3.3 Extensibility

The platform provides a `BaseWeightLayer` abstract class. Researchers can implement custom weighting methods against this interface. Custom implementations are treated as first-class methods in the CV leaderboard.

Custom method serialization (persisting the fitted state to reproduce results) is the researcher's responsibility. The platform does not guarantee round-trip serialization for user-defined implementations — only for the built-in methods. Custom implementations are supported but deferred from the MVP due to this complexity.

---

## 4. Allocation Modes

The weight layer supports two allocation modes that determine how much freedom the algorithm has.

### 4.1 Algorithm Allocation

The algorithm receives all pooled signals and determines the full weight vector without constraint. The weight method (§3) is the only input.

This mode is appropriate when the researcher does not have strong prior views on sector or instrument exposure, and trusts the correlation structure to distribute risk across the full signal pool.

### 4.2 Sector-Constrained Allocation

The researcher specifies explicit **sector weights** — the fraction of total risk budget to allocate to each named sector. The algorithm is then responsible only for within-sector allocation.

Example configuration:
```
Equities:       40%
Commodities:    35%
Fixed Income:   25%
```

Within each sector, the algorithm runs the selected weighting method (§3) on only the signals belonging to that sector. The researcher's sector weights are applied as a hard constraint on the output.

**When to use sector constraints:**
- The researcher has an economic prior that one sector should not dominate (e.g., does not want a single commodity crash to drive the entire portfolio)
- The signal pool is unbalanced — many equity signals but few fixed income signals — and equal treatment would over-represent equities by count
- Regulatory or mandate constraints require explicit sector exposure limits

**Contamination note:** Sector weights are a portfolio construction decision. They must be specified before the holdout is opened. Adjusting sector weights after viewing holdout results is contamination.

---

## 5. Configuration and Locking

The weight layer configuration is stored as an immutable snapshot on the portfolio object:

```python
@dataclass(frozen=True)
class WeightLayerConfig:
    method: str                              # e.g. "ledoit_wolf_min_corr"
    allocation_mode: Literal["algorithm", "sector_constrained"]
    sector_weights: dict[str, float] | None  # None if algorithm_allocation
    cv_leaderboard: list[CVMethodResult]     # IS CV results for all candidates
    selection_reason: str                    # "cv_best" or documented manual override
    locked_at: datetime                      # timestamp; must precede holdout_view_at
```

The `locked_at` timestamp is compared against the timestamp when holdout results are first viewed. If the weight layer configuration was modified after holdout results were viewed, the audit trail flags the portfolio as potentially contaminated.

---

## 6. What the Weight Layer Is Not

**Not a signal generator.** The weight layer does not produce forecasts. It combines signals that were each fitted and validated independently. A weight of zero on a signal is not the same as removing the strategy — the strategy remains in the portfolio; the weight layer is simply assigning it zero allocation for the current period.

**Not a strategy selection mechanism.** The weight layer cannot be used to select which strategies to include in the portfolio based on holdout performance. Strategy inclusion is determined by the monitoring framework (`monitoring.md`). The weight layer determines how included strategies are combined.

**Not an optimisation target for the holdout.** The walk-forward CV (§3.1) runs entirely on IS data. The holdout result is not used to tune, validate, or select weight methods. If the holdout reveals poor portfolio Sharpe, the weight method is not adjusted — it is evaluated by `portfolio_holdout.md` as diagnostic information for future research.
