# Robustness Tests — Index

This document is the entry point for the QuantFoundry robustness test suite. It shows which tests apply at each research stage, what question each answers, and what happens on a pass or fail.

---

## 1. Research Stage Map

```
Strategy development (IS zone)
  └── In-sample robustness tests          → in_sample.md
        └── Parameter sensitivity test    → parameter_sensitivity.md
              └── Parameter selection     → parameter_selection.md

Strategy evaluation (Validation zone)
  └── Validation robustness tests         → validation.md

Portfolio fit check (IS + Validation data)
  └── Portfolio correlation check         → UI-UX/research_workspace/portfolio_correlation.md  (advisory)
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

| Test | Question | Auto / On-demand | Pass condition |
|---|---|---|---|
| Newey-West correction | Is the t-stat inflated by autocorrelation? | Auto | λ reported (no hard gate; feeds other tests) |
| Sharpe CI | How wide is the uncertainty on the IS Sharpe? | Auto | CI reported (no hard gate; calibrates expectations) |
| Deflated Sharpe Ratio | What is the probability this result is real after correcting for search? | Auto | DSR ≥ 0.50 to proceed; ≥ 0.75 preferred |
| Rolling IS / CUSUM | Is the edge consistent across the IS period, or concentrated in one sub-period? | Auto | CUSUM not triggered; rolling positive fraction ≥ 60% |
| Full Grid Permutation | Did the search process explain the result? | On-demand | p ≤ 0.05 |
| Individual Permutation | Is this specific combination capturing temporal structure? | On-demand | p ≤ 0.05 |

**Gate:** DSR ≥ 0.50 is a soft minimum before proceeding to parameter sensitivity. A researcher may proceed with DSR < 0.50 but should document the reason.

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
**Method:** Manual (researcher picks) or Best-by-Metric (rank 1 by NW-adjusted t-stat).

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
| Empirical Sharpe comparison | Does the portfolio Sharpe improve when the strategy is added at the weight layer's assigned weight? | **Primary gate** | ΔSR > 0 |
| Weight assessment | Does the strategy receive a meaningful allocation (≥ 3%)? | Advisory | weight_assigned ≥ floor |
| IDM improvement | Does adding the strategy increase the portfolio IDM? | Observational | ΔIDM reported |

**Gate:** ΔSR > 0 on combined IS + validation data. Failing means the strategy is discarded — parameters cannot be adjusted in response to this result.

**Contamination rule:** The result is binary. Using a failure to tune the strategy's parameters or to change the weight layer method converts IS + validation data into a fitness function.

---

## 7. Portfolio Holdout Tests

**Document:** `portfolio_holdout.md`
**When:** After the portfolio is fully composed and weight layer config is locked — opened once.
**Data used:** Project test zone (project-level holdout, fixed at project creation).

| Test | Question | Alert threshold | Permitted action |
|---|---|---|---|
| Strategy-level monitoring (CUSUM, rolling Sharpe, drawdown cone) | Did any strategy die during the holdout? | Pre-specified monitoring thresholds | Cull flagged strategies |
| Portfolio CUSUM | Did the portfolio as a whole experience a structural break? | C > 1.36 | No action — diagnostic only |
| Portfolio rolling Sharpe | How did the combined Sharpe evolve? | N/A | No action |
| Portfolio drawdown cone | Was holdout drawdown within IS-predicted range? | N/A | No action |
| Correlation realisation | Did cross-strategy correlations hold OOS? | Δρ̄ > 0.20 | No action — informs future IDM calibration |
| IDM accuracy | Did the portfolio vol match IDM-implied expectations? | vol_ratio > 1.50 | No action — informs future sizing |
| Contribution concentration (HHI) | Was portfolio return concentrated in one strategy? | HHI > 0.40 | No action — observational |
| Portfolio Sharpe degradation | How did combined Sharpe compare to IS? | degradation < 0.10 | No action — monitoring is the arbiter |

**The only permitted action from the holdout: cull strategies triggered by pre-specified monitoring rules. Everything else is diagnostic information for future research.**

---

## 7. Live Monitoring

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

## 8. Decision Summary

```
IS tests pass (DSR ≥ 0.50, rolling consistent)
  → Run perturbation test
      → Median ≥ floor
          → Select parameters
              → Run validation
                  → CUSUM clear, degradation ratio ≥ 0.10, ρ ≥ 0.20
                      → Check portfolio correlation (advisory)
                      → Run portfolio addition gate
                          → ΔSR > 0 on IS + val data
                              → Commit to portfolio
                                  → Register monitoring config (before holdout)
                                  → Select weight layer (before holdout)
                                      → Open holdout
                                          → Monitor triggers → cull
                                          → Satisfied with holdout → Deploy
```
