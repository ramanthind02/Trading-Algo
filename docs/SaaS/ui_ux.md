# UI/UX Structure

## 1. Top-Level Navigation

The platform is organized into five primary areas accessible via a persistent left sidebar.

```
├── Dashboard
├── Research Workspace  ← primary product, has subcategories
├── Strategy Library
├── Portfolio Builder
└── Deployment
```

Functional specifications for Portfolio Builder and Deployment (lifecycle, versioning, zone snapshots) are defined in `data_flow.md`. Zone types, time boundaries, Core/API split, and persistence: `zone_manager.md`. This document focuses on navigation structure and the Research Workspace.

---

## 2. Dashboard

The landing page after login. Surfaces a high-level health summary across all deployed portfolios and quick links to the user's active research projects. Read-only — it routes the user to the right area, it is not an action surface.

---

## 3. Research Workspace

The core product. All strategy development happens here. The workspace is scoped to one strategy project at a time.

A persistent **zone timeline bar** runs across the top of every page within the workspace, showing **all** project zones (train / validation / test) with distinct styling by `zone_type`. If there are multiple test zones, each appears as its own segment. Any page that runs or displays data clearly labels which zone (id, name, and type) that data belongs to.

### MVP Subcategories

```
Research Workspace
  ├── Zone Manager              ← define data splits
  ├── Strategy Editor           ← write and version strategy code
  ├── Parameter Sweep           ← sweep parameters, view heatmap, run IS robustness tests
  ├── Backtest Runner           ← run single configurations and compare results
  └── Portfolio Correlation     ← check new strategy correlation against committed set
```

### 3.1 Zone Manager

The entry point for configuring a research project’s data split. Default layout: **one train**, **one validation**, **one test** zone. Users may **add zones**, including **multiple test zones** (e.g. regime-specific holdouts), subject to non-overlap. See `zone_manager.md` for UTC and inclusive-boundary behavior.

**Contents:**
- Date range (or datetime) picker per zone; **non-overlapping** ranges enforced; guided layout may suggest chronological train → validation → test but ordering is not a backend constraint
- Visual timeline: proportions, absolute UTC coverage, per-zone bar counts
- Instrument selector (from platform-supported list)
- Timeframe selection (daily for MVP)
- Zone summary: bar count per zone, resolved UTC range, gaps / coverage warnings when surfaced by the API or Core diagnostics

### 3.2 Strategy Editor

The code editor where the researcher writes the strategy `compute()` function.

**Contents:**
- Code editor with Python syntax highlighting
- Metadata panel: strategy name, max_lookback declaration, lookback_params, parameter schema
- Live schema validation against the strategy contract
- Version history: list of saved versions with timestamps

### 3.3 Backtest Runner

Runs the strategy across a selected zone and displays performance results.

**Contents:**
- Zone selector: choose **any** project zone by name; **test** zones may show a confirmation on first use ("You are about to evaluate a test / holdout window. This should be done sparingly.")
- Parameter input panel rendered from the strategy's parameter schema
- Results: equity curve, drawdown, Sharpe, Calmar, hit rate, average win/loss
- Per-instrument breakdown for multi-ticker strategies
- Side-by-side comparison of up to two previous runs (basic version overlay)

### 3.4 Parameter Sweep

The primary IS research tool. The researcher configures a parameter grid, the platform runs all combinations on the IS zone, and the results are displayed with robustness statistics computed automatically.

**Contents:**
- Grid configuration: select which parameters to sweep and their ranges
- Heatmap view: 2D performance surface for any two parameters, colour-coded by NW-adjusted t-stat
- Robustness summary card: IS Sharpe with CI, NW correction, DSR, N_eff, rolling IS stability — all computed automatically after the sweep completes
- Full Grid Permutation Test: triggered on demand from the summary card
- Sensitivity plots: 1D marginal curves per parameter
- Perturbation test: run on a selected combination, shows peak vs median performance
- Parameter selection: click any combination to select it; "Best by metric" auto-selects the NW t-stat rank 1 result

See `docs/SaaS/robustness_tests/in_sample.md` and `docs/SaaS/robustness_tests/parameter_sensitivity.md` for the statistical specifications.

### 3.5 Portfolio Correlation Check

A research tool for assessing how a new strategy relates to strategies already committed to the portfolio. Runs entirely on IS data — it is a research aid, not an OOS evaluation.

**Contents:**

**Panel A — Unconditional Correlation**
- Pairwise correlation matrix: the new strategy's IS returns vs all committed strategy IS returns, computed on the shared IS date range
- Correlation heatmap with the new strategy's row/column highlighted
- Summary metric: mean pairwise correlation of the new strategy with the existing set
- Warning banner if mean pairwise correlation exceeds 0.70 — the strategy would be highly redundant

**Panel B — Drawdown Conditional Correlation**
- Pairwise correlation restricted to periods where either strategy's drawdown from peak exceeds −5%: ρ_DD(new, existing_i) for each committed strategy
- Displayed as a second heatmap row alongside Panel A, same colour scale — the gap between the two rows reveals how much correlation rises during stress
- Drawdown overlap ratio for each pair: fraction of the new strategy's drawdown days that coincide with each committed strategy's drawdown days
- Warning banner if any pairwise drawdown correlation exceeds 0.60 or any overlap ratio exceeds 0.60

**Interpretation guidance displayed in UI:**
- A strategy with low unconditional correlation but high drawdown correlation offers weak diversification — it appears uncorrelated in normal markets but fails at the same time as existing strategies when it matters most.
- A strategy with moderately elevated unconditional correlation but low drawdown correlation may still add meaningful diversification — it correlates in normal periods but recovers independently during stress.
- Neither pattern is automatically a reason to reject the strategy. The researcher weighs both views alongside the strategy's standalone merit before running the portfolio addition gate.

This view is available from the Backtest Runner and Strategy Library. It does not gate the commit workflow — the researcher decides.

### Future Subcategories

| Subcategory | Description |
|-------------|-------------|
| Data Explorer | Browse raw candle data, summary statistics per zone, data quality checks |
| Strategy Comparison | Side-by-side comparison of multiple strategies with return correlation matrix |
| Notebook | Embedded Jupyter environment with platform SDK pre-configured |

---

## 4. Strategy Library

A browsable catalog of all committed strategy versions. Used to select strategies when building a portfolio.

**MVP Contents:**
- Strategy cards: name, instrument universe, last committed date, indicator if any **test** zone has been evaluated (detail may list which)
- Strategy detail: version history, validation metrics, results from **test** (and validation) runs as applicable, source code viewer
- "Add to Portfolio" button

---

## 5. Portfolio Builder

Assemble committed strategies into a portfolio, assign weights, and deploy. Functional details are in `data_flow.md`.

**MVP Contents:**
- Strategy selector from the Strategy Library
- Weight assignment (equal-weight default, manual override)
- Portfolio-level evaluation on a selected **test** zone before committing a version (when the product requires it)
- Deploy button

---

## 6. Deployment

Creates a live portfolio deployment and exposes Signal API access. Monitoring features are deferred to a future release.

**MVP Contents:**
- Active deployment status (running / stopped)
- Signal API key management
- Output mode selector: position fraction and/or contract quantities
- Link to Signal API documentation

**Future:**
- Live performance charts per portfolio version
- Per-strategy contribution breakdown
- Signal health monitoring (distribution drift detection)
- Drawdown alerts
- Multi-portfolio aggregate view

---

## 7. MVP Scope

| Area | MVP | Future |
|------|-----|--------|
| Dashboard | Basic summary + quick links | Advanced alerts, regime overlays |
| Zone Manager | ✅ (train / validation / test; multiple test zones) | Advanced timeline editing, visual drag handles |
| Strategy Editor | ✅ | Collaborative editing |
| Backtest Runner | ✅ + basic run comparison | Intraday timeframes, multi-TF |
| Data Explorer | | ✅ |
| Parameter Sweep | ✅ | |
| Portfolio Correlation Check | ✅ | |
| Strategy Comparison (full) | | ✅ |
| Notebook | | ✅ |
| Strategy Library | ✅ | Tagging, team sharing |
| Portfolio Builder | ✅ | Automated weight optimization |
| Deployment (signal API access) | ✅ | |
| Deployment monitoring | | ✅ |
| Multiple portfolios | ✅ | Aggregate cross-portfolio view |
