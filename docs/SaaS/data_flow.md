# Data Flow & Research Architecture

## 1. Overview

The platform organises research and deployment into four sequential phases: **Strategy Research**, **Portfolio Construction**, **Portfolio Evaluation**, and **Deployment**. The zone model underpinning this flow uses two distinct tiers — a single project-level test zone fixed at project creation, and researcher-defined strategy-level zones within the pre-test window.

```
Project creation → Strategy Research → Portfolio Construction → Portfolio Evaluation → Deployment → Live
```

Full zone model specification: `docs/SaaS/zone_manager.md`. Robustness test specifications: `docs/SaaS/robustness_tests/`. End-to-end research sequence: `docs/SaaS/research_flow.md`.

---

## 2. Data Zones

### 2.1 Two-Tier Zone Model

Zones operate at two distinct tiers.

**Tier 1 — Project Test Zone (one per project, fixed at creation)**

A single project-wide holdout window. Fixed when the project is created. No strategy in the project may use this data for fitting, parameter selection, or robustness testing. The project test zone is the exclusive window for portfolio-level evaluation and is opened exactly once — after all strategy research and portfolio construction is complete.

```
Zone:
  project_test_start:   utc_datetime  # inclusive lower bound, fixed at project creation
  project_test_end:     utc_datetime  # inclusive upper bound, fixed at project creation
```

**Tier 2 — Strategy-Level Zones (per strategy, constrained to the pre-test window)**

Each strategy has researcher-defined zones within the window `[data_start, project_test_start)`. Two zone types are supported:

| `zone_type` | Role |
|---|---|
| **Train** | Strategy development, parameter sweeps, IS robustness tests, parameter sensitivity, and parameter selection. Fitting is permitted. |
| **Validation** | OOS evaluation of the individual strategy — checks whether IS performance generalises before the strategy is committed to the portfolio. No fitting permitted. Must fall entirely before `project_test_start`. |

```
Zone:
  id:            uuid
  name:          str             # user-defined label (display only)
  zone_type:     Train | Validation
  start_at_utc:  utc_datetime   # inclusive; must be < project_test_start
  end_at_utc:    utc_datetime   # inclusive; must be < project_test_start
```

**Important:** Strategy-level Validation is OOS relative to the strategy's own Train zone, but it is inside the pre-test research period. It evaluates individual strategy generalisation. It does not evaluate portfolio performance — that is exclusively the project test zone's job.

### 2.2 Zone Rules

- Strategy zone ranges must not overlap pairwise within the same strategy.
- All strategy zones must have `end_at_utc < project_test_start`.
- The project test zone is defined once at project creation and cannot be moved after any strategy research begins.
- Boundaries are UTC-inclusive. The API accepts calendar dates from the user and normalises to UTC before persistence.

### 2.3 Why the Two-Tier Model

The single project test zone ensures all portfolio evaluation happens on data that no strategy in the project has ever seen — directly or indirectly. With per-strategy test zones, a researcher could inadvertently let individual strategy validation overlap with other strategies' training windows, introducing correlation between IS and portfolio evaluation performance.

The two-tier model cleanly separates the concerns: strategy-level zones are the researcher's workspace; the project test zone is the portfolio's unbiased scorecard. See `docs/SaaS/zone_manager.md` §8 for the full contamination doctrine.

### 2.4 Default Zone Suggestion

On project creation the platform suggests:
- Train zone: first 60% of available data
- Validation zone: next 20% of available data  
- Project test zone: final 20% of available data

The researcher adjusts before starting any research. Once the first strategy training job is submitted, the project test zone boundary is locked.

---

## 3. Training Methodology

### 3.1 Static (Fixed) Split

The researcher uses their Train zone(s) as the training window. This is the MVP default. A strategy developed with a static split has parameters that are selected once and never change — the strategy is fit on IS data, validated on the Validation zone, and committed to the portfolio.

A static strategy is tagged `methodology: static`. Reoptimisation is manual — the researcher monitors live performance and decides when to develop an updated version.

### 3.2 Walk-Forward (Future)

The researcher selects zones to use as the walk-forward window. The platform partitions the selected range into sequential folds. Each fold produces an OOS window. All OOS windows are stitched into a single aggregated validation set.

Walk-forward supports expanding window (train start fixed) and rolling window (fixed-size train slides forward) modes. A strategy developed with walk-forward is tagged `methodology: walk_forward` and carries a reoptimisation schedule.

Walk-forward strategies require periodic refitting as new data arrives, with a cadence decoupled from the portfolio-level refit cadence. See `docs/SaaS/portfolio_deployment.md` §9 for the decoupling architecture.

```
TrainingConfig:
  window_type:           Fixed | Expanding | Rolling   # MVP: Fixed only
  fold_count:            int                           # MVP: 1
  reoptimize_schedule:   None | Cron                  # MVP: None
```

### 3.3 Purge Gap at Split Boundaries (Future)

When a strategy has a significant `max_lookback`, the first N bars of any OOS window use features overlapping with the preceding IS window — a subtle data leakage at the boundary. A configurable purge gap excludes those N bars from performance evaluation while still using them for feature warmup. Not included in MVP; primarily relevant for ML-based strategies.

---

## 4. Strategy Lifecycle

```
[Research] → [Validate] → [Commit] → [Add to Portfolio]
```

### 4.1 Research Phase (Train Zones)

The researcher develops the strategy using their Train zones. Parameter sweeps, IS robustness tests, parameter sensitivity tests, and parameter selection all happen here. There are no restrictions on how many times Train zone data is accessed. The output of this phase is a selected parameter combination with passed IS robustness tests.

See `docs/SaaS/robustness_tests/index.md` for the full test sequence.

### 4.2 Validation Phase (Validation Zones)

The researcher evaluates the selected parameter combination on their Validation zone to check OOS generalisation. No fitting occurs on the Validation zone. Each evaluation on the Validation zone carries a selection cost — the researcher should be aware that repeatedly evaluating and adjusting based on validation performance accumulates bias.

The output of this phase is a strategy that has passed validation robustness tests and is ready for portfolio consideration.

### 4.3 Portfolio Correlation Check

Before committing a strategy to the portfolio, the researcher can inspect how its IS returns correlate with strategies already committed. This is a research aid — it does not gate the commit. A highly correlated new strategy adds little diversification; the researcher weighs this against its standalone merit.

See `docs/SaaS/ui_ux.md` §3.5 for the UI specification.

### 4.4 Commit Phase

The researcher commits the strategy when satisfied with IS and validation results. The commit creates an immutable strategy version snapshot.

```
StrategyVersion:
  id:               uuid
  strategy_id:      uuid
  version_number:   int
  source_code_hash: str
  params:           dict
  training_config:  TrainingConfig
  zone_snapshot:    ZoneConfig     # zone boundaries at commit time
  methodology:      Static | WalkForward
  committed_at:     timestamp
```

---

## 5. Portfolio Construction

### 5.1 Assembling the Portfolio

A portfolio is a collection of committed strategy versions. The researcher selects which committed strategies to include, assigns instrument weights, and configures the weight layer.

### 5.2 Weight Layer Method Selection

Before opening the project test zone, the researcher selects the weight layer method using IS walk-forward cross-validation. The selected method is locked on the portfolio as an immutable config. See `docs/SaaS/weight_layer.md` for the full specification.

### 5.3 Pre-Commitment Registration

Before the project test zone is opened, the researcher must register:
- Weight layer config (locked)
- Monitoring config — CUSUM thresholds, rolling Sharpe floor, drawdown cone parameters

These must be registered before any test zone results are viewed. Registering them after viewing results is contamination.

### 5.4 Portfolio Versioning

Every portfolio deployment creates a new immutable portfolio version. The platform retains all previous versions for audit and reproducibility.

```
Portfolio:
  id:       uuid
  name:     str
  versions: list[PortfolioVersion]

PortfolioVersion:
  id:                uuid
  version_number:    int
  created_at:        timestamp
  strategies:        list[(strategy_id, strategy_version_id, weight)]
  zone_snapshot:     ZoneConfig
  weight_layer_config: WeightLayerConfig
  monitoring_config:   MonitoringConfig
  deployed_at:       timestamp | None
  retired_at:        timestamp | None
```

---

## 6. Portfolio Evaluation (Project Test Zone)

When the researcher has assembled the portfolio and is satisfied with portfolio construction, they open the project test zone. This is a one-time action.

The platform evaluates the combined portfolio on `[project_test_start, project_test_end]`. Tests cover portfolio-level monitoring (CUSUM, rolling Sharpe, drawdown cone), correlation realisation, IDM accuracy, contribution concentration, and portfolio Sharpe degradation.

The only action the project test zone permits is culling strategies triggered by pre-specified monitoring rules. All other results are diagnostic information. See `docs/SaaS/robustness_tests/portfolio_holdout.md` for the full specification.

---

## 7. Deployment & Live Performance

### 7.1 Deploying a Portfolio Version

After the researcher is satisfied with portfolio holdout results, they run the final fit (incorporating holdout data) and deploy. Deploying creates a live instance and begins daily signal generation. See `docs/SaaS/portfolio_deployment.md` for the full deployment workflow.

### 7.2 Live Performance Tracking

Live performance is tracked per portfolio version. When the portfolio is updated and a new version deployed, the previous version's live performance record closes and a new one begins.

```
LivePerformanceRecord:
  portfolio_version_id:  uuid
  start_date:            date
  end_date:              date | None   # None if currently active
  daily_returns:         timeseries
  drawdowns:             timeseries
  position_log:          timeseries
```

Live performance is the platform's primary long-term scorecard. It is the only truly unbiased evaluation — project test zones are estimates; live data is ground truth.

### 7.3 Manual Strategy Review (Static Methodology)

For static strategies, the researcher monitors live performance via the monitoring dashboard. The platform surfaces CUSUM, rolling Sharpe, and drawdown cone alerts. The researcher decides when to remove a strategy or start a new research cycle.

### 7.4 Walk-Forward Auto-Reoptimisation (Future)

For walk-forward strategies, the platform runs reoptimisation automatically on the configured schedule. The researcher approves or rejects each new version before it is deployed — no automatic deployment without explicit approval.

---

## 8. Jupyter Integration

The Jupyter environment provides full SDK access for custom analysis.

**Capabilities:**
- Load any committed strategy version, inspect source, params, and training config
- Load any portfolio version, inspect composition and weights
- Query candle data by UTC range with zone-aware labels
- Pull live performance records for any portfolio version
- Run custom analysis: factor analysis, regime analysis, correlation studies

**Zone awareness:** SDK data queries surface which zone a date range intersects. The platform does not block access to any zone from a notebook — the researcher sees a clear label and is responsible for not using project test zone data during active strategy development.

---

## 9. MVP Scope

| Feature | MVP | Future |
|---|---|---|
| Two-tier zone model (project test + strategy train/validation) | ✅ | |
| Project test zone fixed at creation | ✅ | |
| Strategy zone types: Train / Validation | ✅ | |
| Robustness tests scoped by zone type | ✅ | |
| Strategy research, validate, commit flow | ✅ | |
| Static (fixed split) training methodology | ✅ | |
| Strategy versioning (immutable snapshots) | ✅ | |
| Portfolio composition and weight layer selection | ✅ | |
| Portfolio holdout evaluation | ✅ | |
| Deployment and live performance tracking | ✅ | |
| Walk-forward methodology | | ✅ |
| Walk-forward auto-reoptimisation with approval gate | | ✅ |
| Purge gap at split boundaries | | ✅ |
| Jupyter SDK with zone-aware data access | | ✅ |
