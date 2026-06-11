# Strategy Decay Monitoring

> [!summary] What Is the Monitoring Store?
> A standalone Parquet store of raw `(signal, target)` vectors per base model, co-located in
> the vault under each ensemble's `monitoring/` folder. It is designed to support rolling
> Sharpe, CUSUM, and Bayesian changepoint detection — all computed on demand from the raw
> vectors. The store is a thin set of read/write helpers; you call them explicitly.

> [!warning] Manual store — not auto-seeded
> The current code has **no automatic integration** between the monitoring store and the
> base-model save path. `BaseModel.save_to_vault()` only writes the feature control file
> (via `add_feature_to_ensemble`); it does **not** call `initialize_monitoring`. The legacy
> binning models (`BinningModelBase`, `ContinuousBinningModel`, `RuleBasedModel`) have been
> removed — they now raise `RuntimeError` on instantiation — and the old hooks they used
> (`_training_target_data`, `get_fitted_vector()`) no longer exist. To use the monitoring
> store you call `initialize_monitoring(...)` / `append_monitoring_data(...)` yourself with
> signal and target series you have on hand. `initialize_monitoring` and
> `append_monitoring_data` currently have no callers in the repository.

---

## Why Monitor Decay?

Strategies decay from crowding, regime change, and overfitting erosion. The challenge:
drawdowns and structural decay are indistinguishable by eye in the early stages. By the time
the difference is obvious, you've already lost months of returns.

The monitoring store solves this by accumulating raw `signal × target` observations over time.
All decay metrics are derived from this single time series on demand.

---

## Directory Layout

Example under the prop tree (`vault/`); the same structure applies under `vault_personal/` or any `<vault_root>`.

```
vault/D/mean_reversion_indices/mr_indices_long/
├── ensemble_config.json
├── features/
│   └── turnaroundtuesday_signal_D_mode_tue_wed.json
└── monitoring/
    └── turnaroundtuesday_signal_D_mode_tue_wed__rule_based_3.parquet
```

- One Parquet file per `(feature_name, model_id)` pair
- Double-underscore separator: `{feature_name}__{model_id}.parquet`
- The `monitoring/` directory is created lazily on first write
- The file path is resolved by `_monitoring_path(...)`, which uses `_resolve_ensemble_path(...)` from `ensemble/vault/manager.py`, so it understands both the nested and legacy-flat ensemble layouts

---

## Data Schema

Each Parquet file stores a `DatetimeIndex` DataFrame (index name `date`) built by `_build_df(...)`:

| Column   | Type    | Description |
|----------|---------|-------------|
| `signal` | float64 | Position multiplier for the bar |
| `target` | float64 | Bar return aligned to the signal |
| `period` | str     | `"IS"` / `"OOS"` / `"LIVE"` |

Rows are the union of the signal and target indices, with any row missing either value dropped.

The `period` tag is the key design choice:
- **`IS`** — in-sample seed data (the default for `initialize_monitoring`)
- **`OOS`** — out-of-sample walkforward data
- **`LIVE`** — post-deployment live observations, appended over time

The IS slice gives **μ₀** (baseline expected alpha) for CUSUM calibration without contamination from live data.

---

## API

**Module**: `ensemble/vault/monitoring_store.py`

### Initialize (IS seed)
```python
from ensemble.vault.monitoring_store import initialize_monitoring

path = initialize_monitoring(
    ensemble_dir='vault/D/mean_reversion_indices/mr_indices_long',
    feature_name='rsi_signal_D',
    model_id='signed_signal_<model_id>',
    signals=signal_series,   # pd.Series with DatetimeIndex
    targets=target_series,   # pd.Series with DatetimeIndex
    period='IS',             # default
)
```
Raises `FileExistsError` if a monitoring file already exists for this `(feature_name, model_id)` pair (use `append_monitoring_data` to add to it).

---

### Append Live / OOS Data
```python
from ensemble.vault.monitoring_store import append_monitoring_data

append_monitoring_data(
    ensemble_dir='vault/D/mean_reversion_indices/mr_indices_long',
    feature_name='rsi_signal_D',
    model_id='signed_signal_<model_id>',
    signals=live_signals,
    targets=live_targets,
    period='LIVE',           # default
)
```
Creates the file if it does not exist. Idempotent: when the file exists it concatenates,
deduplicates on the date index (last write wins for any repeated date), and re-sorts.

---

### Load for Analysis
```python
from ensemble.vault.monitoring_store import load_monitoring_data

# All periods
df = load_monitoring_data(ensemble_dir, feature_name, model_id)

# IS only (for μ₀ / baseline calibration)
is_df = load_monitoring_data(ensemble_dir, feature_name, model_id, period='IS')

# Compute per-bar P&L proxy
pnl = df['signal'] * df['target']
```
Raises `FileNotFoundError` if no monitoring file exists.

---

## Producing the signal / target series

Because there is no automatic hook, you build the inputs yourself. A `BaseModel`
(`features/models/feature_base_model.py`) is a node-backed adapter whose
`predict(candles_df, strategy=...)` returns the signed-signal series; pair that with the
aligned bar returns you trained/evaluated against:

```python
signal_series = base_model.predict(candles_df)          # position multiplier
target_series = bar_returns                              # aligned bar returns
initialize_monitoring(ensemble_dir, feature_name, model_id, signal_series, target_series)
```

---

## Computing Decay Metrics (On Demand)

All metrics are computed from `pnl = signal * target`. No pre-computation is stored.

### Rolling Sharpe
```python
pnl = df['signal'] * df['target']
window = 63  # ~3 months of daily bars
rolling_sharpe = pnl.rolling(window).mean() / pnl.rolling(window).std() * (252 ** 0.5)
```

### CUSUM (Detecting a Negative Shift in Alpha)
```python
import numpy as np

is_df = load_monitoring_data(ensemble_dir, feature_name, model_id, period='IS')
is_pnl = is_df['signal'] * is_df['target']
mu0 = is_pnl.mean()    # baseline expected return
sigma = is_pnl.std()

k = 0.5 * sigma    # allowance (slack) — standard setting
h = 4.0 * sigma    # alarm threshold

live_df = load_monitoring_data(ensemble_dir, feature_name, model_id, period='LIVE')
pnl = (live_df['signal'] * live_df['target']).to_numpy()

cusum = np.zeros(len(pnl))
for i, x in enumerate(pnl):
    prev = cusum[i - 1] if i > 0 else 0.0
    cusum[i] = max(0.0, prev + (mu0 - x) - k)

alarm_days = np.where(cusum > h)[0]
```

Key CUSUM parameters:
| Parameter | Value | Effect |
|-----------|-------|--------|
| `k = 0.5σ` | allowance | standard setting, recommended start |
| `h = 4σ`   | threshold | ~0.5 false alarms/year |
| `h = 5σ`   | threshold | very few false alarms, slower detection |

---

## Implementation Notes

### File Format

Parquet with the `fastparquet` engine — consistent with the project's other small Parquet
stores. Files are small (<5k rows for years of daily data) but read frequently.

### Relationship to portfolio materialization

The monitoring store is per-`(feature, model_id)` and tags rows with `period`
(`IS`/`OOS`/`LIVE`). It is separate from the portfolio prediction materialization store,
which lives under the central cache and tags rows with `world` (`train`/`val`/`test`/`live`).
Do not conflate `period` and `world`. See [[Vault/portfolio_snapshots_and_predictions]].

---

## Recommended Workflow

```
Daily:   append_monitoring_data(period="LIVE") after close
Weekly:  compute rolling_sharpe, check CUSUM statistic vs threshold
Monthly: compute CUSUM with fresh μ₀ from IS slice, review trend
```

Alert tiers:

| Level  | Trigger | Action |
|--------|---------|--------|
| YELLOW | Rolling Sharpe declining trend OR CUSUM > 70% of threshold | Reduce size 50% |
| ORANGE | CUSUM alarm fired | Halt new trades, review |
| RED    | Multiple metrics alarming | Full stop, paper trade until edge reconfirmed |

---

## See Also

- [[Vault/vault]] — Vault directory structure and control file schema
- [[Ensemble/base_model]] — `BaseModel` (signed-signal node adapter)
- [[Vault/portfolio_snapshots_and_predictions]] — `world`-tagged portfolio prediction store

> _Verified against current code via CodeGraph on 2026-06-07._
