# Strategy Decay Monitoring

> [!summary] What Is the Monitoring Store?
> Persistent `(signal, target)` vector store per base model, co-located in the vault.
> Initialized automatically when a fitted model is saved. Designed to support rolling Sharpe,
> CUSUM, and Bayesian changepoint detection — computed on demand from the raw vectors.

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
- `monitoring/` directory is created lazily on first save

---

## Data Schema

Each Parquet file stores a `DatetimeIndex` DataFrame:

| Column   | Type    | Description |
|----------|---------|-------------|
| `signal` | float64 | Position multiplier from `predict()` |
| `target` | float64 | Bar return used during training |
| `period` | str     | `"IS"` / `"OOS"` / `"LIVE"` |

The `period` tag is the key design choice:
- **`IS`** — in-sample training data, auto-written at `save_to_vault()` time
- **`OOS`** — out-of-sample walkforward data, written manually
- **`LIVE`** — post-deployment live observations, appended over time

The IS slice gives **μ₀** (baseline expected alpha) for CUSUM calibration without contamination from live data.

---

## Auto-Initialization

When `BaseModel.save_to_vault()` is called on a **fitted** model, the monitoring file is
created automatically with `period="IS"`:

```python
base_model.fit(candles_df, target)
model_id = base_model.save_to_vault(ensemble_dir)
# vault/D/.../monitoring/{feature}__{model_id}.parquet now exists
```

Unfitted models (e.g. rule-based) skip initialization silently — `is_fitted_` is False.
Re-saving an already-vaulted model is safe — `FileExistsError` is silenced.

---

## API

**Module**: `ensemble/vault/monitoring_store.py`

### Initialize (IS seed — called automatically by `save_to_vault`)
```python
from ensemble.vault.monitoring_store import initialize_monitoring

path = initialize_monitoring(
    ensemble_dir='vault/D/mean_reversion_indices/mr_indices_long',
    feature_name='rsi_signal_D',
    model_id='signed_signal_<model_id>',
    signals=signal_series,   # pd.Series with DatetimeIndex
    targets=target_series,
    period='IS',             # default
)
```
Raises `FileExistsError` if the file already exists.

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
Idempotent: deduplicates on date index (last write wins for any repeated date).

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

is_pnl = load_monitoring_data(ensemble_dir, feature_name, model_id, period='IS')['signal'] * ...
mu0 = (is_pnl['signal'] * is_pnl['target']).mean()   # baseline expected return
sigma = (is_pnl['signal'] * is_pnl['target']).std()

k = 0.5 * sigma    # allowance (slack) — standard setting
h = 4.0 * sigma    # alarm threshold

live_pnl = load_monitoring_data(..., period='LIVE')
pnl = live_pnl['signal'] * live_pnl['target']

cusum = np.zeros(len(pnl))
for i, x in enumerate(pnl):
    prev = cusum[i - 1] if i > 0 else 0.0
    cusum[i] = max(0.0, prev + (mu0 - x) - k)

alarm_days = np.where(cusum > h)[0]
```

Key CUSUM parameters (from blog):
| Parameter | Value | Effect |
|-----------|-------|--------|
| `k = 0.5σ` | allowance | standard setting, recommended start |
| `h = 4σ`   | threshold | ~0.5 false alarms/year |
| `h = 5σ`   | threshold | very few false alarms, slower detection |

---

## Implementation Notes

### Where `_training_target_data` Lives

Both `BinningModelBase.fit()` and `ContinuousBinningModel.fit()` (which overrides `fit()`
independently) store the target series:

```python
self._training_feature_data = feature_data.copy()   # pre-existing
self._training_target_data = target_data.copy()      # added for monitoring
```

The signal is recovered via the already-existing `get_fitted_vector()`:

```python
signal_vector = bm.get_fitted_vector(strategy=bm.strategy)
```

### File Format

Parquet with `fastparquet` engine — consistent with OHLC data storage across the project.
Files are small (<5k rows for years of daily data) but read frequently.

---

## Recommended Workflow

```
Daily:   append_monitoring_data(period="LIVE") after close
Weekly:  compute rolling_sharpe, check CUSUM statistic vs threshold
Monthly: compute CUSUM with fresh μ₀ from IS slice, review trend
```

Alert tiers (from the blog):

| Level  | Trigger | Action |
|--------|---------|--------|
| YELLOW | Rolling Sharpe declining trend OR CUSUM > 70% of threshold | Reduce size 50% |
| ORANGE | CUSUM alarm fired | Halt new trades, review |
| RED    | Multiple metrics alarming | Full stop, paper trade until edge reconfirmed |

---

## See Also

- [[vault]] — Vault directory structure and control file schema
- [[base_model]] — `BinningModelBase`, `get_fitted_vector()`
