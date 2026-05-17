# Research Flow — End-to-End Guide

## 1. Purpose

This document describes the ordered sequence of steps from project creation to live deployment. It is the practical companion to the detailed specifications in the other docs — it tells you what to do and in what order, with links to where each step is specified.

---

## 2. Phase 1 — Project Setup

### Step 1: Create a project and set the project test zone

On project creation:
- Name the project and select the instrument universe (tickers + timeframe)
- Configure the **project test zone** — the fixed holdout window for portfolio evaluation
- The platform suggests the final 20% of available data as the default; adjust if needed
- **This boundary is locked once the first strategy training job is submitted**

Zone model specification: `zone_manager.md`

### Step 2: Configure strategy-level zones

Within the pre-test window, define:
- One or more **Train** zones — data used for strategy development and fitting
- One **Validation** zone — a genuinely OOS slice for individual strategy evaluation

The simplest layout (and the default) is one Train zone followed by one Validation zone, both within the pre-test window.

---

## 3. Phase 2 — Strategy Research

The following steps repeat for each strategy the researcher develops. All work happens within the pre-test window.

### Step 3: Write the strategy

In the Strategy Editor:
- Implement the `compute(candles, params)` function
- Declare metadata: `name`, `version`, `tickers`, `max_lookback`, `params_schema`, `warmup_mode`
- The platform validates the signature and metadata on save

Strategy contract: `strategy_spec.md`

### Step 4: Configure cost assumptions

For each instrument the strategy trades, set:
- Slippage (ticks per side)
- Commission (dollars per contract per side)

The platform applies platform defaults if not overridden. All backtest results display net returns (after costs). Gross returns are available for comparison.

Transaction cost specification: `transaction_costs.md`

### Step 5: Run the parameter sweep (IS zone)

In the Parameter Sweep tool:
- Configure the parameter grid
- Submit the sweep — all combinations run in parallel on the IS (Train) zone
- Results display automatically: NW-adjusted t-stats, DSR, rolling IS stability, N_eff, Sharpe CI
- Inspect the heatmap to understand the parameter surface topology

IS robustness tests: `robustness_tests/in_sample.md`

**Gate:** If DSR < 0.50 for the best combination, the search process likely explains the result. Narrow the parameter grid, obtain more IS data, or discard the strategy.

### Step 6: Run the perturbation test

On the best combination (or any combination of interest):
- Trigger the perturbation test from the Parameter Sweep view
- The platform re-runs the strategy on all ±10% perturbations of the chosen parameters
- Key outputs: peak IS metric, median IS metric, optimism bias (peak − median)

The median is the more realistic expectation of live performance. A large peak-median gap indicates a sharp, fragile optimum.

Parameter sensitivity: `robustness_tests/parameter_sensitivity.md`

**Gate:** Median metric must be ≥ metric_floor (default: NW t-stat 2.0). A combination that fails even the median check is too fragile to proceed.

### Step 7: Select parameters

Choose one parameter combination:
- **Manual selection**: click a combination in the Parameter Sweep table
- **Best by metric**: the platform auto-selects the NW t-stat rank 1 result

The selection is recorded as immutable metadata on the strategy. It does not change after this point.

Parameter selection: `robustness_tests/parameter_selection.md`

### Step 8: Run validation

With the selected combination locked:
- Run a backtest on the **Validation** zone (no fitting permitted here)
- The platform computes all validation robustness tests automatically:
  - Sharpe comparison (IS vs Validation)
  - Block bootstrap CI on the Validation Sharpe
  - CUSUM with IS parameters
  - Rolling z-score
  - Neighbourhood performance (perturbation set re-evaluated on Validation)
  - Rank correlation of the IS parameter grid vs the Validation ranking

Validation robustness tests: `robustness_tests/validation.md`

**Gate (hard):** CUSUM trigger or degradation ratio < 0.10. Either is a strong reason not to proceed.

**Gate (soft):** Neighbourhood val p10 < metric_floor, or rank correlation ρ < 0.20. Researcher decides, but poor results here should be documented.

### Step 9: Check portfolio correlation (advisory)

Before running the portfolio addition gate:
- Open the **Portfolio Correlation** tool from the backtest result or Strategy Library
- The platform computes pairwise IS correlations between the new strategy and all already-committed strategies
- A warning shows if mean pairwise correlation exceeds 0.70 — the new strategy would be highly redundant

This is an advisory tool, not a gate. Its purpose is to give the researcher early signal before the more formal portfolio addition gate runs.

Portfolio correlation UI: `ui_ux.md` §3.5

### Step 10: Portfolio addition gate

After passing all individual robustness tests, the strategy must clear the portfolio addition gate before it can be committed.

The gate runs four tests on combined IS + validation data:
1. **Analytical hurdle** — does the strategy's Sharpe exceed $\rho_{\text{new},P} \times SR_P$? (soft gate)
2. **Empirical Sharpe comparison** — does the portfolio's Sharpe improve when this strategy is added at the weight the weight layer assigns? (primary gate: ΔSR > 0)
3. **Weight assessment** — does the strategy receive a meaningful weight (≥ 3%)?
4. **IDM improvement** — does adding the strategy increase the portfolio IDM?

**Pass condition:** ΔSR > 0 (empirical comparison).

**Contamination rule:** A failed gate means the strategy is discarded. The researcher cannot adjust the strategy's parameters or change the weight layer method in response to the result and re-run. A new strategy developed from scratch for the same portfolio role is permitted — but must go through the full research cycle independently.

Portfolio addition gate specification: `robustness_tests/portfolio_addition.md`

### Step 11: Commit the strategy

When the portfolio addition gate passes:
- Commit the strategy from the Strategy Library
- This creates an immutable version snapshot: source code, parameters, zone boundaries, methodology
- The strategy is now available for portfolio inclusion

---

## 4. Phase 3 — Portfolio Construction

Repeat Phase 2 for each strategy in the portfolio. When enough strategies are committed:

### Step 12: Compose the portfolio

In the Portfolio Builder:
- Select which committed strategies to include
- Assign instrument weights (equal-weight default or manual override)
- The portfolio is not yet deployed — this is the composition step only

Data flow: `data_flow.md` §5

### Step 13: Register the monitoring config (CRITICAL — must precede holdout view)

Before opening the project test zone, register the monitoring configuration:
- CUSUM critical threshold (default: 1.36 at 5% level)
- Rolling Sharpe floor and lookback window
- Drawdown cone percentile threshold
- Sizing schedule (how position sizes reduce as monitors trigger)

This config must be locked before any test zone results are viewed. Registering it after viewing test results — even with a statistically reasonable choice — is contamination, because the motivation to choose those thresholds came from the observation.

Monitoring pre-commitment: `robustness_tests/monitoring.md` §2

### Step 14: Select the weight layer method (CRITICAL — must precede holdout view)

- Run IS walk-forward cross-validation to rank weight layer methods
- The CV leaderboard is computed entirely within the IS window
- Select the method (or override manually with a documented reason)
- Lock the weight layer config on the portfolio

The weight layer config cannot be changed after the holdout is opened.

Weight layer: `weight_layer.md` §3

---

## 5. Phase 4 — Portfolio Evaluation

### Step 15: Open the portfolio holdout

This is a one-time, irreversible action. Once the holdout is opened:
- The platform runs the full portfolio holdout evaluation on the project test zone
- Results cover: portfolio monitoring, correlation realisation, IDM accuracy, contribution concentration, portfolio Sharpe degradation

**The only action the holdout permits is culling strategies flagged by pre-specified monitoring rules. Everything else is observational.**

Portfolio holdout: `robustness_tests/portfolio_holdout.md`

### Step 16: Act on monitoring triggers (if any)

Review each monitoring alert:
- If CUSUM or rolling Sharpe rules trigger on a strategy, the researcher decides whether to remove it
- If removed: the strategy's weight is set to zero; IDM/FDM/weights are not immediately recomputed
- The researcher can optionally trigger a manual refit after removals

---

## 6. Phase 5 — Deployment

### Step 17: Configure the final fit

- Choose fit window type: expanding (all data) or rolling (fixed lookback, e.g. 3 years)
- The final fit runs on all available data including the holdout period, which is now freely usable
- What gets refitted: correlation matrix, IDM, FDM, weight layer weights, target vol estimate
- What stays frozen: strategy parameters (locked at IS selection)

Portfolio deployment: `portfolio_deployment.md` §3

### Step 18: Deploy

- Review the deployment summary: IDM, strategy weights, fit data range, target vol
- Click **Deploy Portfolio**
- This creates the initial portfolio snapshot and activates the monitoring framework

Position sizing pipeline: `position_sizing.md`

### Step 19: Live operation

Post-deployment, the researcher monitors via the monitoring dashboard:
- CUSUM, rolling Sharpe, drawdown cone run continuously on live bar-by-bar returns
- Alerts surface when thresholds are breached; the sizing schedule reduces position sizes automatically
- The researcher decides whether to remove flagged strategies and whether to trigger manual refits

Manual refits incorporate new market data using the same window config as deployment. The researcher triggers them on demand — there is no automatic schedule in MVP.

Live monitoring: `robustness_tests/monitoring.md`

---

## 7. Summary Table

| Step | Phase | Tool / Location | Gate? |
|---|---|---|---|
| 1. Create project + set project test zone | Setup | Zone Manager | Project test zone locked after first training job |
| 2. Define strategy zones | Setup | Zone Manager | Zones must not overlap; must precede project_test_start |
| 3. Write strategy | Research | Strategy Editor | Contract validation on save |
| 4. Configure costs | Research | Strategy Editor / backtest config | None — defaults applied |
| 5. Parameter sweep | Research | Parameter Sweep | DSR ≥ 0.50 to proceed |
| 6. Perturbation test | Research | Parameter Sweep | Median metric ≥ floor |
| 7. Parameter selection | Research | Parameter Sweep | Required before validation |
| 8. Validation | Research | Backtest Runner | CUSUM clear; degradation ratio ≥ 0.10 |
| 9. Portfolio correlation check | Research | Portfolio Correlation | None — advisory |
| 10. Portfolio addition gate | Research | Portfolio Builder | ΔSharpe > 0 on IS + val data |
| 11. Commit strategy | Research | Strategy Library | Portfolio addition gate passed |
| 12. Compose portfolio | Portfolio | Portfolio Builder | Strategies must be committed |
| 13. Register monitoring config | Portfolio | Portfolio Builder | Must precede holdout view |
| 14. Select weight layer | Portfolio | Portfolio Builder | Must precede holdout view |
| 15. Open portfolio holdout | Evaluation | Portfolio Builder | Monitoring + weight layer config registered |
| 16. Act on monitoring triggers | Evaluation | Portfolio Builder | Only pre-specified rules trigger action |
| 17. Final fit configuration | Deployment | Portfolio Deployment | — |
| 18. Deploy | Deployment | Deployment | — |
| 19. Live monitoring | Live | Monitoring Dashboard | — |
