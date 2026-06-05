# Robustness Tests — Index

This document is the entry point for the robustness test suite. It shows which tests apply at each research stage, what question each answers, and what happens on a pass or fail.

The statistical primitives (DSR, $N_\text{eff}$, Sharpe CI, rolling-IS/CUSUM, perturbation neighbours, the portfolio-addition gate, validation report, and holdout report) are implemented in the external **`quantfoundry_core`** package (`quantfoundry_core.robustness` and `quantfoundry_core.portfolio_gate`). The repo's `research/feature/` and `research/portfolio/` packages orchestrate them through these three top-level phases:

- `exploration` — `research/feature/exploration/orchestrate.py::execute_exploration_phase`
- `validation` — `research/feature/validation/robustness_runner.py`
- `portfolio_addition` — `research/feature/portfolio_addition/gate_runner.py`

Older local names such as `in_sample` and `oos` survive as compatibility aliases (commands, artifact folders), not the preferred mental model.

---

## 1. Research Stage Map

```
Strategy research (exploration phase)
  └── In-sample robustness tests          → in_sample.md
        └── Parameter sensitivity test    → parameter_sensitivity.md
              └── Parameter selection     → parameter_selection.md

Strategy evaluation (validation phase)
  └── Validation robustness tests         → validation.md

Portfolio admission (portfolio_addition phase)
  └── Pairwise redundancy check           → portfolio_addition.md §3  (advisory)
  └── Portfolio addition gate             → portfolio_addition.md  (primary gate)

Portfolio construction (pre-holdout)
  └── Weight layer method selection       → weight_layer.md §3
  └── Monitoring config pre-registration  → monitoring.md §2

Portfolio evaluation (Project Test zone)
  └── Portfolio holdout validation        → portfolio_holdout.md

Live operation (post-deployment)
  └── Strategy and portfolio monitoring   → monitoring.md
```

---

## 2. In-Sample Tests

**Document:** `in_sample.md`  
**When:** After a parameter sweep on the IS zone, before selecting a combination.  
**Data used:** IS zone only.

See `in_sample.md` §4 for the full failure-mode framework (temporal overfitting vs parameter mining).

### Required gates

| Test | Failure mode | Auto / On-demand | Pass condition |
|---|---|---|---|
| Vector shuffle (per combo) | Mode 1 — temporal overfitting | On-demand (exploration) | Pass at configured $\alpha$ |
| Deflated Sharpe Ratio | Mode 2 — parameter mining | Auto | **DSR $\geq 0.95$** |
| NW t-stat (best combo) | HAC-adjusted magnitude | Auto | **$\geq 2.0$** |
| Rolling IS | Temporal consistency | Auto | Positive fraction **$\geq 70\%$** (SaaS default 60%) |
| CUSUM | Structural stability | Auto | Not triggered at 5% |

### Diagnostics (report; do not hard-gate structured grids)

| Test | Role | Pass / flag |
|---|---|---|
| Sharpe CI | Estimate precision | Report; lower bound $> 0$ is sanity check |
| Full Grid Permutation | Empirical search-bias check | Report null percentile; **flag if $< 20$th percentile** |
| Individual return-shuffle permutation | Pre-specified single hypothesis only | On-demand; $p \leq 0.05$ when combo was not grid-selected |

**Gate summary:** vector shuffle + DSR + NW t-stat + rolling + CUSUM must pass before parameter lock on a structured indicator family. Full-grid permutation is retained for audit and heterogeneous mining — not as the primary Mode 2 gate when $N_\text{eff}$ is trustworthy.

**Do not** restrict full-grid to vector-shuffle passers on the same IS window; that invalidates the null (see `in_sample.md` §4.2).

---

## 3. Parameter Sensitivity Test

**Document:** `parameter_sensitivity.md`
**When:** After IS tests, on the chosen parameter combination (or before formal selection).
**Data used:** IS zone only — re-runs the strategy on ±10% parameter perturbations.

| Test | Question | Pass condition |
|---|---|---|
| Perturbation test | Is the peak IS result a stable optimum or an isolated noise spike? | Median metric ≥ metric_floor |

**Key output:** optimism bias = peak − median. A large gap means the chosen combination sits on a sharp peak; the median is the more realistic expectation of live performance.

---

## 4. Parameter Selection

**Document:** `parameter_selection.md`
**When:** After IS tests and perturbation test.
**Method:** Manual (researcher picks) or Best-by-Metric (rank 1 by the configured selection metric).

This is not a test — it is a decision. The selected combination is locked as strategy metadata and does not change after this point.

---

## 5. Validation Tests

**Document:** `validation.md`
**When:** After parameter selection, on a genuinely OOS zone within the pre-test window.
**Data used:** Validation zone (strategy-level, within pre-test window).

| Test | Question | Pass condition |
|---|---|---|
| Sharpe comparison | Did IS edge survive OOS? | CI overlap OR degradation z < 2.0 |
| Block bootstrap CI | What is the honest CI on the short OOS Sharpe? | Reported (no hard gate) |
| CUSUM with IS params | Has the return distribution structurally changed? | break_detected = False |
| Rolling z-score | When and how did returns diverge from IS baseline? | Visual (no hard gate) |
| Neighbourhood val performance | Does the stable IS region also hold OOS? | val_p10 ≥ metric_floor |
| Rank correlation | Is the IS parameter ranking preserved OOS? | Spearman ρ ≥ 0.20 |

**Gate:** CUSUM trigger or degradation ratio < 0.10 are strong reasons not to proceed. Other failures are informational — the researcher decides.

---

## 6. Portfolio Addition Gate

**Document:** `portfolio_addition.md`
**When:** After the strategy passes all individual robustness tests (IS + validation), before committing to the portfolio. Skipped for the first strategy (no existing portfolio to compare against).
**Data used:** Combined IS + validation data. No project test zone data used.

| Test | Question | Gate type | Pass condition |
|---|---|---|---|
| Analytical hurdle | Does SR_new exceed ρ × SR_P (the theoretical condition for improvement)? | Soft | SR_new > ρ_new,P × SR_P |
| Empirical Sharpe comparison | Does the portfolio Sharpe improve when the strategy is added at the weight layer's assigned weight? | **Primary gate** | ΔSR ≥ 0.02 |
| Weight assessment | Does the strategy receive a meaningful allocation (≥ 3%)? | Advisory | weight_assigned ≥ floor |
| IDM improvement | Does adding the strategy increase the portfolio IDM? | Observational | ΔIDM reported |

**Gate:** ΔSR ≥ 0.02 on combined IS + validation data. Failing means the strategy is discarded — parameters cannot be adjusted in response to this result.

**Contamination rule:** The result is binary. Using a failure to tune the strategy's parameters or to change the weight layer method converts IS + validation data into a fitness function.

---

## 7. Portfolio Holdout Tests

**Document:** `portfolio_holdout.md`
**When:** After the portfolio is fully composed and weight layer config is locked — opened once.
**Data used:** Project test zone (project-level holdout, fixed at project creation).

In `research/portfolio/holdout/`, strategy-level monitoring runs the **four** validation-suite tests on a trailing evaluation window — Sharpe comparison, CUSUM, rolling Sharpe z-score, and equity curve bands — and rolls the fail count into a Green/Yellow/Red traffic light (advisory only; the pipeline does not auto-cull in Phase 1). The drawdown cone is a portfolio-level calibration tool (§2.3 of `portfolio_holdout.md`).

| Test | Question | Alert threshold | Permitted action |
|---|---|---|---|
| Strategy-level monitoring (Sharpe comparison, CUSUM, rolling Sharpe z-score, equity bands) | Did any strategy die during the holdout? | Pre-specified monitoring thresholds → traffic light | Cull / reduce-weight flagged strategies (researcher decision) |
| Portfolio CUSUM | Did the portfolio as a whole experience a structural break? | C > 1.36 | No action — diagnostic only |
| Portfolio rolling Sharpe | How did the combined Sharpe evolve? | N/A | No action |
| Portfolio drawdown cone | Was holdout drawdown within IS-predicted range? | N/A | No action |
| Correlation realisation | Did cross-strategy correlations hold OOS? | Δρ̄ > 0.20 | No action — informs future IDM calibration |
| IDM accuracy | Did the portfolio vol match IDM-implied expectations? | vol_ratio > 1.50 | No action — informs future sizing |
| Contribution concentration (HHI) | Was portfolio return concentrated in one strategy? | HHI > 0.40 | No action — observational |
| Portfolio Sharpe degradation | How did combined Sharpe compare to IS? | degradation < 0.10 | No action — monitoring is the arbiter |

**The only permitted action from the holdout: cull strategies triggered by pre-specified monitoring rules. Everything else is diagnostic information for future research.**

---

## 8. Live Monitoring

**Document:** `monitoring.md`
**When:** Post-deployment, continuously.
**Data used:** Live bar-by-bar returns.

| Monitor | Question | Trigger | Action |
|---|---|---|---|
| CUSUM | Has the strategy's return distribution structurally changed? | C > 1.36 | Alert researcher |
| Rolling Sharpe | Is the trailing Sharpe below the pre-committed floor? | Below floor for N consecutive periods | Alert researcher; reduce sizing per schedule |
| Drawdown cone | Is the current drawdown anomalous relative to IS expectations? | Below 5th percentile of simulated paths | Alert researcher |
| SPRT | Has the strategy crossed the sequential decision boundary? | Λ_t < B (reject null of positive edge) | Alert researcher |
| Signal monitors | Is the signal distribution drifting (turnover, autocorrelation, long/short ratio)? | Beyond pre-specified bands | Alert researcher |

**Monitoring alerts do not automatically remove strategies.** The researcher reviews evidence and decides. Position sizing is reduced automatically per the pre-committed sizing schedule as monitors trigger.

---

## 9. Decision Summary

```
Vector shuffle pass (Mode 1)
  AND DSR ≥ 0.95 (Mode 2)
  AND NW t-stat ≥ 2.0
  AND rolling + CUSUM pass
  → Run perturbation test
      → Median ≥ floor
          → Select parameters
              → Run validation
                  → CUSUM clear, degradation ratio ≥ 0.10, ρ ≥ 0.20
                      → Check pairwise redundancy (advisory, §3 of portfolio_addition.md)
                      → Run portfolio addition gate
                          → ΔSR ≥ 0.02 on IS + val data (plus risk-impact legs)
                              → Commit to portfolio
                                  → Register monitoring config (before holdout)
                                  → Select weight layer (before holdout)
                                      → Open holdout
                                          → Monitor triggers → cull
                                          → Satisfied with holdout → Deploy

Diagnostics throughout: full-grid null percentile, Sharpe CI (flag discord with DSR)
```

> _Verified against commit a07b6bf->197221e on 2026-06-04 (docs Phase A; WP-8 restructure repoint)._
