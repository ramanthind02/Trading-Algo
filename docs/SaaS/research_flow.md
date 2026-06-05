# Research Flow — End-to-End Guide

> **Status:** This is the planned **QuantFoundry SaaS** research workflow
> (project → strategy research → portfolio → holdout → deploy). It is design
> intent for the hosted product. The tools named below ("Strategy Editor",
> "Parameter Sweep", "Portfolio Builder", "Deployment") are SaaS surfaces and are
> **not** the current local entrypoints. For the workflow that exists in this
> repo today, see §8 ("Current local research entrypoints") and the robustness
> docs under `docs/SaaS/robustness_tests/`.

## 1. Purpose

This document describes the ordered sequence of steps from project creation to live deployment in the planned product. It is the practical companion to the detailed specifications in the other docs — it tells you what to do and in what order, with links to where each step is specified.

For local `Trading-Algo` work, treat the robustness documents under `docs/SaaS/robustness_tests/` as the reference for the strategy-research methodology, and §8 below as the map to the actual `feature_research/` and `portfolio_research/` entrypoints. The local `feature_research/` package follows a three-phase model (`in_sample` / `oos` / `validation`) feeding a vault, which `portfolio_research/` then assembles and tests.

---

## 2. Phase 1 — Project Setup

### Step 1: Create a project and set the project test zone

On project creation:
- Name the project and select the instrument universe (tickers + timeframe)
- Configure the **project test zone** — the fixed holdout window for portfolio evaluation
- The platform suggests the final 20% of available data as the default; adjust if needed
- **This boundary is locked once the first strategy training job is submitted**

Zone model specification: `zone_manager.md`  
UI: a dedicated research-workspace UI spec is planned but not yet present in `docs/`. The current local UI is the Flask app at `frontend/app.py` backed by `feature_research/ui` and `portfolio_research/ui` — see `ui_ux.md`.

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
- Results display automatically: the configured selection metric, DSR, rolling IS stability, N_eff, and Sharpe CI
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

**Gate:** Median metric must be ≥ `metric_floor` for the configured selection metric. A combination that fails even the median check is too fragile to proceed.

### Step 7: Select parameters

Choose one parameter combination:
- **Manual selection**: click a combination in the Parameter Sweep table
- **Best by metric**: the platform auto-selects the rank 1 result for the configured selection metric

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

Feature/vault correlation (current local tool): `portfolio_research/run_feature_vault_correlation.py` emits a feature-vs-vault correlation CSV (see the repo `CLAUDE.md` Commands section).

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

---

## 8. Current local research entrypoints (what exists today)

The SaaS tools above are aspirational. The workflow that runs in this repo today
is a set of Python module entrypoints plus a local Flask UI. Run them with the
shared-venv interpreter (see `CLAUDE.md`).

| Phase | Module / script | Notes |
|---|---|---|
| In-sample feature research | `python -m feature_research.in_sample.run_is` | IS phase over the feature universe. |
| OOS feature research | `feature_research/oos/run_oos.py`, `feature_research/oos/run_oos_permutation.py` | OOS + permutation testing. |
| Validation | `feature_research/validation/run_validation.py`, `run_validation_permutation.py` | Validation phase. |
| Inclusion / portfolio-addition gates | `feature_research/run_inclusion_gates.py` (`feature_research/inclusion_gates.py`) | Gate logic before vault commit. |
| Binning phase | `feature_research/binning/run_phase.py` | Base-model binning. |
| Portfolio test | `python -m portfolio_research.run_portfolio_test` | Assembles the vault portfolio and runs `portfolio_research/futures_sim.py`. |
| Prop-firm portfolio | `portfolio_research/run_portfolio_prop_firm.py` | Prop-firm-mode portfolio run. |
| Weight-layer CV | `portfolio_research/weight_layer_cv.py` (`run_weight_layer_cv`) | IS walk-forward method leaderboard. |
| Feature/vault correlation | `python -m portfolio_research.run_feature_vault_correlation` | Correlation CSV. |
| Local UI | `frontend/app.py` (Flask) + `feature_research/ui`, `portfolio_research/ui` | Phase planning, job management, artifact previews, vault commit. |

Validated features are written to the vault (`vault/` prop tree, `vault_personal/`
for personal; see `docs/library/Vault/vault.md`) under
`<vault_root>/<TF>/<weight_hierarchy_group>/<ensemble>/`. `portfolio_research/`
reads that vault to build the `GlobalPortfolio` and its weight layer.

> _Verified against commit a07b6bf on 2026-06-04 (docs Phase A)._
