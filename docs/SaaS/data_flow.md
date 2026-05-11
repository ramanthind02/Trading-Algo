# Data Flow & Research Architecture

## 1. Overview

The platform organizes research and deployment into three sequential phases: **Research**, **Portfolio**, and **Deployment**. The researcher defines their own data zones and uses them as they see fit — the platform makes no assumptions about how many zones exist or how they are ordered.

```
[User-defined Zones] → Strategy Research → Portfolio Composition → Deployment → Live Performance
```

By default, the platform suggests three zones (Train / Validation / Test) as a starting point. The researcher can rename, add, or remove zones freely.

---

## 2. Data Zones

### 2.1 Zone Definitions

A zone is a named date range with a single semantic tag: **Train** or **OutOfSample**. This tag is the only distinction the platform makes — it determines which robustness tests are available for that zone. Everything else (number of zones, names, ordering, purpose) is entirely up to the researcher.

```
Zone:
  id:          uuid
  name:        str     # user-defined label, e.g. "train", "validation", "test_2022"
  zone_type:   enum    # Train | OutOfSample
  start_date:  date
  end_date:    date
```

The platform defaults to suggesting three zones on project creation:

| Default Name | Default Type | Suggested Purpose |
|---|---|---|
| Train | Train | Model development, parameter selection |
| Validation | OutOfSample | Performance estimation during research |
| Test | OutOfSample | Final holdout before portfolio commit |

The researcher can rename, reorder, add, or remove zones freely. There is no enforced count or ordering.

### 2.2 Zone Rules

The only rules the platform enforces:

- Zones must have valid, non-empty date ranges.
- Zones within the same project must not overlap.

No ordering is enforced between zones. No minimum or maximum zone count is enforced. The researcher is responsible for structuring zones in a way that is statistically sound for their research goals.

### 2.3 Why the Train / OutOfSample Distinction Matters

The platform provides different robustness tests depending on zone type:

- **Train zones**: fitting-aware tests. The model was fit on this data, so in-sample performance is expected to look inflated. Tests here assess whether the in-sample result is meaningful given that fitting occurred (e.g., permutation tests that account for model complexity).
- **OutOfSample zones**: standard OOS tests. No fitting occurred here. Tests assess whether OOS performance is statistically significant above chance (e.g., return shuffle, Monte Carlo).

This is the only assumption the platform makes about a zone's role. All other interpretation is left to the researcher.

### 2.4 Responsibility for Zone Integrity

The platform surfaces zone date ranges and types clearly throughout the UI. The researcher is responsible for defining zones that span varied market regimes and for exercising discipline around how frequently holdout zones are evaluated. The platform does not police zone access.

---

## 3. Training Methodology

### 3.1 Static (Fixed) Split

The researcher uses all of Zone 1 as a single training window. This is the MVP default and the simplest approach. Zone 2 is used as a fixed validation set. No rolling or expanding windows are involved.

A strategy developed with a static split is tagged `**methodology: static**`.

For static strategies, reoptimization is **manual**. The researcher monitors live performance over time (months to years) and decides independently when to evaluate whether the strategy needs updating or removal.

### 3.2 Walk-Forward (Future)

The researcher selects one or more zones to use as the walk-forward window. The platform partitions the selected date range into sequential folds. The first fold defines the initial train window. Each subsequent fold produces an OOS window. All OOS windows are stitched together into a single aggregated validation set for performance evaluation. This aggregated OOS performance is a research-quality estimate — it is not a substitute for a dedicated holdout zone.

Walk-forward supports two window modes:

- **Expanding window**: Train start is always fixed at the beginning; each fold adds more history. Better for strategies that benefit from maximum data (mean reversion, cross-sectional signals).
- **Rolling window**: Fixed-size train window slides forward; old data is dropped. Better for regime-sensitive strategies where distant history is noise.

A strategy developed with walk-forward is tagged `**methodology: walk_forward`** and carries a reoptimization schedule.

The training methodology is represented as a parameterized config from day one, so walk-forward support is an extension rather than a redesign:

```
TrainingConfig:
  window_type:           Fixed | Expanding | Rolling   # MVP: Fixed only
  fold_count:            int                           # MVP: 1
  reoptimize_schedule:   None | Cron                  # MVP: None
```

### 3.3 Purge Gap at Split Boundaries (Future)

When a strategy has a significant `max_lookback`, the first N bars of any OOS window use features that overlap with the preceding IS window, creating a subtle data leakage at the boundary. The fix is a configurable purge gap — those N bars are excluded from performance evaluation (but still used for feature warmup).

This is primarily a concern for higher-complexity ML-based strategies that can overfit to boundary patterns. Simple rule-based or low-complexity strategies are not materially affected. The purge gap will be an **optional, toggleable setting per strategy**, defaulting to off. It is not included in MVP.

---

## 4. Strategy Lifecycle

```
[Research] → [Validate] → [Commit] → [Add to Portfolio]
```

### 4.1 Research Phase (Train Zones)

The researcher develops the strategy using their defined train zones. There are no restrictions on how many times train zone data is accessed. Parameter sweeps and robustness tests run here. Walk-forward folds are also defined within train zone date ranges.

### 4.2 Validation Phase (OOS Zones)

The researcher evaluates the strategy on OOS zones to estimate out-of-sample performance. Which zones are used for validation vs final holdout is entirely the researcher's decision — the platform makes no distinction between OOS zones beyond the robustness tests it makes available.

Each evaluation on an OOS zone is an implicit selection decision. The researcher should be aware that repeatedly evaluating the same OOS zone accumulates selection bias over time.

### 4.3 Commit Phase

The researcher decides when to commit a strategy for portfolio consideration. This typically follows satisfactory evaluation on at least one OOS zone designated as a holdout. Timing is entirely user-controlled — some researchers commit after each strategy individually, others batch several strategies and evaluate a combined portfolio view before committing.

The platform does not enforce a commit gate. Discipline around holdout zone usage is the researcher's responsibility.

### 4.4 Strategy Versioning

A committed strategy is an **immutable snapshot**: source code, parameters, training config, and the zone boundaries active at commit time. If the researcher wants to change the strategy, they create a new version and restart from the Research Phase.

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

## 5. Portfolio Composition

### 5.1 Assembling the Portfolio

A portfolio is a collection of committed strategy versions with associated weights. The researcher selects which committed strategies to include and assigns weights (equal-weight or manual).

### 5.2 Portfolio Versioning

Every portfolio deployment creates a new immutable portfolio version. The platform permanently retains all previous versions so the researcher can review historical compositions and, if needed, redeploy an older version.

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
  zone_snapshot:     ZoneConfig     # snapshot of zone boundaries at version creation
  deployed_at:       timestamp | None
  retired_at:        timestamp | None
```

**Rollback** is re-deploying a previous `PortfolioVersion`. This creates a new deployment record pointing to the old strategy set — it does not modify the historical version record.

> **MVP scope note**: Portfolio versioning with rollback is noted as a future feature. MVP may track the current active portfolio only. The data model above is designed to support versioning from day one so that adding the UI and rollback logic later requires no schema changes.

### 5.3 Zone 3 at the Portfolio Level

When the researcher evaluates Zone 3 at the portfolio level (multiple strategies combined), the platform runs all strategies across Zone 3 data and aggregates the results into a combined portfolio performance view. This is the closest available estimate to live performance before deployment.

---

## 6. Deployment & Live Performance

### 6.1 Deploying a Portfolio Version

Deploying creates a live instance of a specific `PortfolioVersion`. The platform begins routing live market data through all included strategies and producing live position signals.

### 6.2 Live Performance Tracking

Live performance is tracked **per portfolio version**. When the researcher updates the portfolio and deploys a new version, the previous version's live performance record is closed and a new one begins. This allows direct comparison of performance across versions.

```
LivePerformanceRecord:
  portfolio_version_id:  uuid
  start_date:            date
  end_date:              date | None   # None if currently active
  daily_returns:         timeseries
  drawdowns:             timeseries
  position_log:          timeseries
```

Live performance is the platform's primary scorecard. It is displayed prominently and is the only truly unbiased evaluation of a strategy — Zone 3 is an estimate; live data is the ground truth.

### 6.3 Manual Strategy Review (Static Methodology)

For strategies tagged `methodology: static`, reoptimization is entirely manual. The platform surfaces live performance metrics to help the researcher evaluate whether a strategy is degrading. The researcher independently decides when to:

- Remove the strategy from the portfolio
- Start a new research cycle and develop an updated version
- Replace the strategy in the portfolio with the new version (new portfolio version + deployment)

### 6.4 Walk-Forward Auto-Reoptimization (Future)

For strategies tagged `methodology: walk_forward`, the platform runs reoptimization automatically on the configured schedule:

1. Zone 1 expands to include new data up to the schedule boundary.
2. The strategy is retrained using the walk-forward config.
3. The result is validated against Zone 2 (also expanded if applicable).
4. The researcher is notified and presented with the new version's performance vs. the current deployed version.
5. The researcher **approves or rejects** the new version before it is deployed. No automatic deployment without explicit approval.

If the researcher approves, a new strategy version is committed and a new portfolio version is created and deployed.

---

## 7. Jupyter Integration

The Jupyter environment is a first-class research tool with full SDK access to the platform ecosystem.

### 7.1 Capabilities

- **Strategy access**: load any committed strategy version, inspect source code, params, and training config
- **Portfolio access**: load any portfolio version, inspect composition and weights
- **Cache access**: retrieve cached strategy outputs without re-running compute
- **Data access**: query Zone 1/2/3 candle data directly via SDK (zone-aware — SDK labels which zone a date range falls into)
- **Live performance**: pull live performance records for any portfolio version
- **Custom analysis**: run arbitrary research — factor analysis, regime analysis, correlation studies — using platform data

### 7.2 Zone Awareness

The SDK data query methods surface which zone a requested date range falls into. This is informational only in MVP — the platform does not block access to any zone from a notebook. The researcher sees a clear label and is responsible for not contaminating holdout data during active strategy development.

```python
# Example SDK usage (illustrative, not final API)
from platform_sdk import Strategy, Portfolio, DataClient

client = DataClient(project_id="...")
candles = client.get_candles(ticker="ES", start="2018-01-01", end="2020-12-31")
# SDK response includes: zone_labels={"2018-01-01..2019-12-31": "train", "2020-01-01..2020-12-31": "validation"}
```

---

## 8. MVP Scope


| Feature | MVP | Future |
|---|---|---|
| User-defined zones (name, type, date range) | ✅ | |
| Train / OutOfSample zone type tag | ✅ | |
| Default 3-zone suggestion (Train / Validation / Test) | ✅ | |
| No enforced zone count or ordering | ✅ | |
| Robustness tests scoped by zone type | ✅ | |
| Strategy research, validate, commit flow | ✅ | |
| Static (fixed split) training methodology | ✅ | |
| Strategy versioning (immutable snapshots) | ✅ | |
| Portfolio composition | ✅ | |
| Deployment and live performance tracking | ✅ | |
| Portfolio versioning and rollback | | ✅ |
| Walk-forward methodology (user selects zone, expanding + rolling) | | ✅ |
| Walk-forward auto-reoptimization with approval gate | | ✅ |
| Purge gap at split boundaries (optional, toggleable) | | ✅ |
| Jupyter SDK with zone-aware data access | | ✅ |


