# Weight Layer

> **Status:** This is a forward-looking product/design spec for the planned
> QuantFoundry SaaS platform. The conceptual model below (signal granularity,
> contamination doctrine, locking) is design intent. The **"How this maps to the
> current implementation"** callouts describe what `Trading-Algo` actually does
> today via `ensemble/weight_layer.py`. Where the two differ, the code is the
> source of truth.

## 1. Purpose

This document describes the philosophy and configuration of the weight layer. The weight layer determines how a portfolio combines the forecast signals from multiple strategies into a single set of position sizes. It is distinct from individual strategy fitting and operates above the strategy level.

In the SaaS product model, the weight layer is configured once before the portfolio holdout is opened. It is not a robustness test — it is a portfolio construction decision with contamination implications identical to parameter selection.

Related documents:
- `docs/SaaS/zone_manager.md` — holdout zone structure and contamination doctrine
- `docs/SaaS/weight_layer_spec.md` — the asset-first hierarchy implementation (current code)
- `docs/SaaS/robustness_tests/portfolio_holdout.md` — portfolio-level evaluation after the weight layer config is locked
- `docs/library/Ensemble/weight_layer.md` — implementation reference for `ensemble/weight_layer.py`

> **How this maps to the current implementation.** In `Trading-Algo` the weight
> layer is the cross-timeframe combiner owned by `GlobalPortfolio`. Strategies
> across all tickers, timeframes, and models are encoded into a single flat pool
> on a synthetic `__GLOBAL__` ticker and combined by `ClusteredWeightLayer`
> (constructed via the `WeightLayer(...)` factory in `ensemble/weight_layer.py`).
> See `weight_layer_spec.md` for the encoding path and hierarchy details.

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

> **Current implementation.** Each pooled input is a *stream* keyed
> `{ticker}::{timeframe}::{model_name}` (e.g. `ES::D::sma_regime`). This is the
> per-instrument signal of this section.

### 2.2 Per-Portfolio Signal

A strategy that trades multiple instruments simultaneously as a single bet produces a **per-portfolio signal**. Examples:
- A calendar spread (long ES front month, short ES back month)
- A relative value strategy (long GC, short SI at a fixed ratio)
- A cross-asset momentum strategy whose P&L is computed on the combined position

Per-portfolio signals enter the weight layer as a single unit regardless of how many underlying instruments they touch. Splitting them into per-instrument components would destroy the signal's meaning — the trade is defined by the relationship between instruments, not the individual legs.

> **Current implementation.** Per-portfolio signals are a design concept; the
> current encoder produces one stream per `(ticker, timeframe, model)`. The
> `es_tlt` group is the closest analogue: it trades ES only (TLT is a peer used
> as a rebalancing reference), so it is a per-instrument stream on ES, not a
> two-leg per-portfolio signal.

### 2.3 Pooling Principle

The weight layer pools all per-instrument signals into a single correlation matrix and weights the full pool. In the SaaS design it does not maintain artificial per-strategy or per-instrument subgroups unless sector constraints are specified (§4).

> **Current implementation.** The shipped engine pools by *ticker first*: for
> each ticker, the correlated streams on that ticker are combined into one
> per-ticker forecast (the `fit()` loop in `ClusteredWeightLayer` iterates over
> unique tickers). Cross-ticker / cross-asset budgeting is expressed through the
> manual `hierarchy_equal` tree, not a single flat correlation matrix over every
> stream. See `weight_layer_spec.md` for the asset → style → stream hierarchy.

---

## 3. Weight Method Selection

The choice of weighting method is a research decision made on in-sample (IS) data before the portfolio holdout is opened.

### 3.1 Walk-Forward Cross-Validation on IS Data

Method selection uses walk-forward cross-validation across the IS window:

1. Divide the IS period into folds, each with a fit window and an OOS window, both entirely within IS.
2. For each candidate method, fit the weight layer on the fit window and evaluate on the OOS window.
3. Aggregate OOS performance across folds to produce a per-method leaderboard.
4. The researcher selects the method with the best CV performance (or overrides manually with a documented reason).

This selection happens once before any portfolio holdout results are viewed. Changing the weight method after opening the holdout is contamination under the doctrine in `zone_manager.md` §8.

> **Current implementation.** Walk-forward method selection exists in
> `research/portfolio/weight_layer_cv.py` (`run_weight_layer_cv`,
> `build_cv_arms`, `evaluate_arm_on_fold`). It produces a per-arm CV leaderboard
> over the IS window; there is no timestamp-enforced "locking" against a holdout
> view — that locking/contamination tracking is SaaS design, not yet enforced in
> code.

### 3.2 Available Methods and Defaults

The valid weighting methods are defined by `_WEIGHT_METHODS` in `ensemble/weight_layer.py`:

| `weighting_method` | Behaviour |
|---|---|
| `equal_signal` | Equal weight across streams on a ticker. **Library default** (the `WeightLayer(...)` factory default). |
| `inverse_avg_pairwise_corr` | Weight ∝ `1 / (1 + avg positive-clipped peer correlation)`. |
| `hierarchy_equal` | Equal split at every level of a manual nested hierarchy (`hierarchy_spec` / `hierarchy_path`). Optionally tilts by Sharpe via `sr_adjustment`. **Used by the production prop portfolio config.** |
| `inverse_corr_hierarchy` | Equal hierarchy split, then inverse-correlation within each group. |
| `ledoit_wolf_min_corr` | Weight ∝ `1 / Σ(positive-clipped correlation row)`. |
| `risk_parity_corr` | Weight ∝ `1 / √Σ(positive-clipped correlation row)`. |
| `hierarchy_theme_inv_corr`, `hierarchy_theme_ledoit`, `ledoit_wolf_hierarchy_within` | Theme-level correlation tilts over a manual hierarchy. |

The two defaults differ by surface:
- `WeightLayer(...)` factory / `WeightLayerConfig`: `equal_signal`.
- The prop portfolio research config (`research/portfolio/config.py`): `hierarchy_equal` with `sr_adjustment=True` (see `weight_layer_spec.md` §SR adjustment).

> **Removed (do not configure).** The legacy HRP and Sortino methods —
> `hrp_cluster_equal`, `hrp_classic`, `optimize_sortino_capped` — were removed.
> They are listed in `_LEGACY_REMOVED_METHODS`; deserializing a saved portfolio
> that names one of them raises a `ValueError` telling you to re-fit with
> `equal_signal`, `inverse_avg_pairwise_corr`, or `hierarchy_equal`.

### 3.3 Forecast Diversification Multiplier (FDM)

After combining a ticker's streams, the layer applies an FDM computed from the
positive-clipped signal correlation matrix:

```
FDM = min( sqrt( 1 / (mean_corr + 0.01) ), fdm_max )      # default fdm_max = 2.0
```

`mean_corr` is the mean off-diagonal positive-clipped correlation of the combined
signals (`_compute_fdm_from_corr_matrix` in `ensemble/weight_layer.py`). FDM is
capped at `fdm_max` (Carver's recommendation, default 2.0) to bound leverage when
signals are weakly correlated.

### 3.4 Extensibility

The platform provides a `BaseWeightLayer` abstract base class; the concrete
shipped implementation is `ClusteredWeightLayer`. Adding a new method today means
extending `_WEIGHT_METHODS` and the `fit()` dispatch in
`ensemble/weight_layer.py`. A general user-pluggable custom-method interface with
guaranteed round-trip serialization (`serialize_weight_layer_state` /
`deserialize_weight_layer_state` currently only support `ClusteredWeightLayer`)
is SaaS-deferred.

---

## 4. Allocation Modes

The weight layer supports two conceptual allocation modes that determine how much freedom the algorithm has.

### 4.1 Algorithm Allocation

The algorithm receives all pooled signals and determines the full weight vector without sector constraint. The weight method (§3) is the only input. This is the mode used by the flat methods (`equal_signal`, `inverse_avg_pairwise_corr`, `ledoit_wolf_min_corr`, `risk_parity_corr`).

### 4.2 Hierarchy-Constrained Allocation

The researcher specifies an explicit nested budget tree — the share of total risk allocated to each named group, and groups-within-groups, down to individual streams. The algorithm splits each group's mass among its children.

> **Current implementation.** This is the `hierarchy_equal` family. The hierarchy
> is a manual JSON spec (`hierarchy_spec`) parsed by
> `ensemble/weight_hierarchy.py` (`parse_hierarchy_spec`,
> `compute_equal_split_weights`). Equal split applies at every level; an optional
> Carver mini-bootstrap **SR tilt** (`ensemble/sr_adjustment.py`) then tilts
> sibling budgets at each level while preserving parent-group mass. The current
> production tree is asset-first (`asset_class → strategy_group → stream`) — see
> `weight_layer_spec.md`. A budget-by-named-sector convenience UI (e.g.
> "Equities 40%, Commodities 35%") is SaaS design; today the budgets come from
> the equal-split tree (optionally SR-tilted).

**Contamination note:** Hierarchy/sector budgets are a portfolio construction decision. In the SaaS doctrine they must be specified before the holdout is opened; adjusting them after viewing holdout results is contamination.

---

## 5. Configuration and Locking

The current weight layer configuration is the `WeightLayerConfig` frozen
dataclass in `ensemble/weight_layer.py`:

```python
@dataclass(frozen=True)
class WeightLayerConfig:
    weighting_method: str = "equal_signal"
    fdm_max: float = 2.0
    hierarchy_spec: Optional[dict[str, object]] = None
    hierarchy_path: Optional[str] = None
    sr_adjustment: bool = False
    sr_avg: float = 0.5
    sr_p_step: float = 0.01
    sr_std: float = 0.15
    sr_min_years: float = 5.0
    sr_tilt_max_depth: Optional[int] = None
    within_group_method: str = "equal"   # or "inverse_avg_pairwise_corr"
```

A fitted layer is persisted by value via `serialize_weight_layer_state` (config
dict + fitted weights/FDM/cluster state) and restored with
`deserialize_weight_layer_state`.

> **SaaS design (not yet enforced in code).** The product model adds a
> `locked_at` timestamp compared against the time holdout results are first
> viewed, and an audit trail flag when the config changes after that point. The
> current repo persists config and fitted state but does not implement
> view-timestamp contamination tracking.

---

## 6. What the Weight Layer Is Not

**Not a signal generator.** The weight layer does not produce forecasts. It combines signals that were each fitted and validated independently. A weight of zero on a signal is not the same as removing the strategy — the strategy remains in the portfolio; the weight layer is simply assigning it zero allocation for the current period.

**Not a strategy selection mechanism.** The weight layer cannot be used to select which strategies to include in the portfolio based on holdout performance. Strategy inclusion is determined by the monitoring framework (`robustness_tests/monitoring.md`). The weight layer determines how included strategies are combined.

**Not an optimisation target for the holdout.** Walk-forward CV (§3.1) runs entirely on IS data. The holdout result is not used to tune, validate, or select weight methods. If the holdout reveals poor portfolio Sharpe, the weight method is not adjusted — it is evaluated by `robustness_tests/portfolio_holdout.md` as diagnostic information for future research.

> _Verified against commit a07b6bf->197221e on 2026-06-04 (docs Phase A; WP-8 restructure repoint)._
