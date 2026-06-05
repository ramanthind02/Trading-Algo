# Portfolio Deployment

> ⚠️ Slated for rewrite under the NautilusTrader migration (WP-4 live execution). See docs/refactor/nautilus/.

## 1. Purpose

This document covers the steps required to move a portfolio from research into live operation: the pre-deployment checklist, final fit configuration, portfolio snapshot schema, and the manual refit and strategy removal workflows used during live operation.

Related documents:
- `docs/SaaS/zone_manager.md` — holdout zone structure and contamination doctrine
- `docs/SaaS/weight_layer.md` — weight layer philosophy and method selection
- `docs/SaaS/robustness_tests/portfolio_holdout.md` — portfolio-level holdout evaluation
- `docs/SaaS/robustness_tests/monitoring.md` — live strategy monitoring framework

---

## 2. Pre-Deployment Checklist

Before the portfolio can be deployed, the following must be complete:

| Requirement | Where it happens |
|---|---|
| All strategies passed IS robustness tests | `robustness_tests/in_sample.md` |
| All strategies passed validation robustness tests | `robustness_tests/validation.md` |
| Weight layer method selected and locked | `weight_layer.md` §3 |
| Portfolio holdout evaluated and researcher satisfied | `robustness_tests/portfolio_holdout.md` |
| Monitoring config registered (CUSUM thresholds, rolling Sharpe floor) | `robustness_tests/monitoring.md` |
| Final fit window configured (see §3) | This document |

The monitoring config must be registered before the researcher views holdout results, per the pre-commitment doctrine in `monitoring.md`. The deployment step simply confirms it is in place.

---

## 3. Final Fit Configuration

### 3.1 Using Holdout Data in the Final Fit

Once the researcher is satisfied with the holdout evaluation, the holdout period is no longer a protected zone. It becomes ordinary history that the final production fit can include. The platform lifts the holdout data restriction at the point the researcher initiates deployment.

This is the correct behaviour: the holdout evaluated the portfolio under genuinely OOS conditions. That evaluation is done. Withholding that data from the production fit would reduce the quality of the fitted state for no methodological reason.

### 3.2 Window Type

The researcher selects one of two window types for the final fit:

**Expanding window:** Fit on all available data from `data_start` through the current date. Produces the most statistically stable estimates and is the right choice when the researcher believes the strategy and correlation environment is broadly stationary over the full history.

**Rolling window:** Fit on a fixed-length window ending at the current date (e.g. the most recent 3 years). Discards older history. Appropriate when correlations and volatility regimes shift over time and recent data is more representative of the current environment.

Rolling windows of 750–1250 bars (3–5 years of daily data) are typical for correlation-sensitive components. Shorter windows produce noisier estimates; longer windows approach the expanding window in practice.

### 3.3 What Gets Refitted

Not all portfolio components have the same sensitivity to new data. The table below describes what the final fit recomputes:

| Component | Refitted? | Notes |
|---|---|---|
| Strategy parameters | No | Frozen at IS selection. Never change post-deployment. |
| Correlation matrix | Yes | Uses the chosen window (expanding or rolling). Drives IDM, FDM, and weight layer. |
| IDM | Yes | Recomputed from the new correlation matrix. |
| FDM per signal | Yes | Recomputed from signal-level correlation matrix over the same window. |
| Weight layer state | Yes | Refitted on the chosen window using the locked method. |
| Target volatility estimate | Yes | Rolling 252-bar annualised vol on the portfolio return stream. Independent of the main window. |

Strategy parameters are explicitly excluded from the refit. They were selected on IS data and are fixed. The final fit only updates portfolio-level aggregation — correlation structure, diversification multipliers, and signal weights.

---

## 4. Portfolio Snapshot

A portfolio snapshot is the complete, self-contained record of the portfolio's fitted state at a point in time. Given a snapshot and a market data feed, the platform can reproduce the exact position for any bar after the snapshot's `committed_at` timestamp without any other database records.

```python
@dataclass(frozen=True)
class PortfolioSnapshot:
    snapshot_id:              uuid
    committed_at:             datetime
    committed_by:             str

    # Strategy set — frozen at IS selection
    strategies:               list[StrategySpec]

    # Fit configuration
    fit_window_type:          Literal["expanding", "rolling"]
    fit_window_length_bars:   int | None       # None if expanding
    fit_data_range:           tuple[datetime, datetime]  # actual bars used

    # Portfolio-level fitted state
    correlation_matrix:       np.ndarray       # cross-strategy signal correlations
    idm:                      float
    fdm_per_signal:           dict[str, float]
    strategy_weights:         dict[str, float] # zero for culled strategies
    weight_layer_config:      WeightLayerConfig
    weight_layer_state:       bytes            # serialised fitted weights

    # Position sizing
    target_vol:               float
    capital:                  float
    instrument_configs:       dict[str, InstrumentConfig]  # multiplier, FX rate

    # Monitoring config — locked before holdout was opened
    monitoring_config:        MonitoringConfig

    # Audit
    holdout_report_id:        uuid
    previous_snapshot_id:     uuid | None
    refit_reason:             Literal["initial_deployment", "manual_refit", "strategy_removal"]
```

Snapshots are append-only. A new snapshot is created for every deployment, every refit, and every strategy removal. Old snapshots are never modified or deleted. The version history is a linear chain of snapshots linked by `previous_snapshot_id`.

---

## 5. Deployment Commit

After the final fit completes, the platform displays a deployment summary:

```
┌─ Deployment Summary ──────────────────────────────────────────────────┐
│  Fit window:       rolling   3 years   2022-01-01 → 2024-12-31        │
│  Strategies:       6   (all active)                                    │
│                                                                        │
│  IDM:              1.38                                                │
│  Weight layer:     ledoit_wolf_min_corr                                │
│                                                                        │
│  Strategy weights:                                                     │
│    ES_EWMAC_20     0.21                                                │
│    ES_RSI_14       0.18                                                │
│    GC_TREND        0.19                                                │
│    CL_BREAKOUT     0.15                                                │
│    NQ_MEAN_REV     0.14                                                │
│    TLT_CARRY       0.13                                                │
│                                                                        │
│  Target vol:       15.0%     Capital:  $500,000                        │
│                                                                        │
│  [Deploy Portfolio]                                                    │
└────────────────────────────────────────────────────────────────────────┘
```

Clicking **Deploy Portfolio** creates the initial snapshot with `refit_reason: "initial_deployment"` and marks the portfolio as live. The monitoring framework activates immediately after deployment.

---

## 6. Manual Live Refits

### 6.1 When to Refit

The portfolio's fitted state becomes stale as new market data arrives. The correlation structure, IDM, and weight layer were fit on historical data; as the live period extends, the portfolio progressively relies on older estimates.

The researcher decides when to refit. Common triggers:
- Significant market regime shift (correlation structure visibly changed on the monitoring dashboard)
- A strategy was removed and the researcher wants to recompute IDM and weights without it
- A fixed time interval has elapsed (e.g. quarterly refit as a standing practice)

There is no automatic refit schedule in the MVP. Refitting is a deliberate researcher action.

### 6.2 Refit Behaviour

The researcher triggers a refit from the portfolio dashboard. The system runs the fit using the same window config as the most recent snapshot. The result automatically commits as a new snapshot with `refit_reason: "manual_refit"`.

There is no preview-and-discard step. Initiating a refit commits the result. This prevents the researcher from running multiple refits and selectively keeping the one with the best recent performance — a form of contamination analogous to running a parameter sweep and cherry-picking the best result.

The researcher can see the new snapshot's statistics in the version history immediately after commit.

### 6.3 Changing the Window Config

If the researcher wants to change the window type or window length for a refit (e.g. switch from 3-year rolling to 5-year rolling), they configure it before triggering the refit. The new config is stored on the new snapshot. It does not retroactively change previous snapshots.

---

## 7. Strategy Removal

### 7.1 Monitoring Alerts

The monitoring framework runs continuously against the live portfolio (see `monitoring.md`). When a strategy breaches a pre-specified threshold — CUSUM structural break, rolling Sharpe floor, or drawdown cone exceedance — the platform sends an alert to the researcher.

The alert contains:
- Which strategy triggered, and which test(s) fired
- The current monitoring state (CUSUM statistic, rolling Sharpe value, drawdown relative to cone)
- The monitoring config thresholds the strategy was measured against

### 7.2 Researcher Decision

The researcher reviews the monitoring evidence and decides whether to remove the strategy. Removal is not automatic. The monitoring alert is a signal, not a command.

The researcher may decide to keep a strategy despite an alert — for example, if the CUSUM trigger coincides with a known market dislocation event that the researcher judges to be temporary. This is a legitimate research judgement. The decision and its timestamp are recorded in the audit trail.

If the researcher decides to remove, they confirm the removal on the portfolio dashboard.

### 7.3 Weight Zeroing

On removal, the strategy's weight is set to zero in a new snapshot (`refit_reason: "strategy_removal"`). The correlation matrix, IDM, and FDM are **not** recomputed in this step. The zeroed weight is applied immediately to live position sizing.

This is a deliberate simplification. An immediate full refit triggered by a removal would change the weights of all remaining strategies simultaneously — a larger portfolio modification than warranted. Zeroing the weight and leaving everything else unchanged is the minimal intervention.

### 7.4 Optional Refit After Removal

After a strategy is removed, the researcher can trigger a full manual refit (§6) to recompute the correlation matrix and weights without the removed strategy. This is optional. The refit will typically increase the weights of the remaining strategies slightly and may adjust the IDM.

---

## 8. Version History

The platform maintains the full chain of portfolio snapshots since initial deployment. Each entry in the version history shows:

- Snapshot ID, commit timestamp, and `refit_reason`
- IDM and target vol at that snapshot
- Strategy weights (with zero-weighted removed strategies visible)
- Summary statistics at the time of commit: rolling portfolio Sharpe, realised vol, monitoring state per strategy

The version history is read-only. The researcher cannot roll back to a previous snapshot based on performance — that would convert the version history into an optimisation target. The audit trail exists for observability and reproducibility, not for selective rollback.

---

## 9. Deferred (Post-MVP)

The following features are explicitly deferred to keep the MVP scope manageable:

**Scheduled auto-refits.** Time-triggered portfolio refits (e.g. monthly) that run and commit without researcher intervention. Requires deciding what to do when a refit fails (data gap, computation error) and how to surface that to the researcher.

**Automated sanity checks.** Pre-commit validation: null values in signal history, missing data for any strategy, correlation matrix not positive semi-definite, IDM outside plausible bounds. If a check fails, the refit is rejected and the previous snapshot remains live.

**Walk-forward strategy refitting.** Strategies that evolve over time (fitted on a rolling window, or ensemble over multiple parameter variants) require a separate per-strategy refit cadence decoupled from the portfolio-level refit. The portfolio treats each strategy as a black box producing signals; strategy refits run independently and do not trigger portfolio refits.

> _Verified against commit a07b6bf on 2026-06-04 (docs Phase A)._
