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

Functional specifications for Portfolio Builder and Deployment (lifecycle, versioning, zone snapshots) are defined in `data_flow.md`. This document focuses on navigation structure and the Research Workspace.

---

## 2. Dashboard

The landing page after login. Surfaces a high-level health summary across all deployed portfolios and quick links to the user's active research projects. Read-only — it routes the user to the right area, it is not an action surface.

---

## 3. Research Workspace

The core product. All strategy development happens here. The workspace is scoped to one strategy project at a time.

A persistent **zone timeline bar** runs across the top of every page within the workspace, showing the Zone 1 / Zone 2 / Zone 3 date boundaries for the current project. Any page that runs or displays data clearly labels which zone that data belongs to.

### MVP Subcategories

```
Research Workspace
  ├── Zone Manager        ← define data splits
  ├── Backtest Runner     ← run and compare results
  └── Strategy Editor     ← write and version strategy code
```

### 3.1 Zone Manager

The entry point for a new research project. The user defines the three zone date boundaries and selects the instrument universe.

**Contents:**
- Date range picker per zone with strict ordering enforced
- Visual timeline showing zone proportions and absolute date coverage
- Instrument selector (from platform-supported list)
- Timeframe selection (daily for MVP)
- Zone summary: bar count per zone, date range, gaps detected

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
- Zone selector (Zone 1 / Zone 2 / Zone 3) — Zone 3 shows a confirmation dialog on first use ("You are about to evaluate your holdout set. This should be done sparingly.")
- Parameter input panel rendered from the strategy's parameter schema
- Results: equity curve, drawdown, Sharpe, Calmar, hit rate, average win/loss
- Per-instrument breakdown for multi-ticker strategies
- Side-by-side comparison of up to two previous runs (basic version overlay)

### Future Subcategories

The following areas are deferred to future releases:

| Subcategory | Description |
|-------------|-------------|
| Data Explorer | Browse raw candle data, summary statistics per zone, data quality checks |
| Parameter Sweep | Sweep one or two parameters, render performance heatmap, Research Cache powered |
| Strategy Comparison | Side-by-side comparison of multiple strategies with return correlation matrix |
| Notebook | Embedded Jupyter environment with platform SDK pre-configured |

---

## 4. Strategy Library

A browsable catalog of all committed strategy versions. Used to select strategies when building a portfolio.

**MVP Contents:**
- Strategy cards: name, instrument universe, last committed date, Zone 3 evaluated flag
- Strategy detail: version history, validation metrics, Zone 3 results, source code viewer
- "Add to Portfolio" button

---

## 5. Portfolio Builder

Assemble committed strategies into a portfolio, assign weights, and deploy. Functional details are in `data_flow.md`.

**MVP Contents:**
- Strategy selector from the Strategy Library
- Weight assignment (equal-weight default, manual override)
- Zone 3 portfolio-level evaluation before committing a version
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
| Zone Manager | ✅ | Multiple zones per type |
| Strategy Editor | ✅ | Collaborative editing |
| Backtest Runner | ✅ + basic run comparison | Intraday timeframes, multi-TF |
| Data Explorer | | ✅ |
| Parameter Sweep | | ✅ |
| Strategy Comparison (full) | | ✅ |
| Notebook | | ✅ |
| Strategy Library | ✅ | Tagging, team sharing |
| Portfolio Builder | ✅ | Automated weight optimization |
| Deployment (signal API access) | ✅ | |
| Deployment monitoring | | ✅ |
| Multiple portfolios | ✅ | Aggregate cross-portfolio view |
