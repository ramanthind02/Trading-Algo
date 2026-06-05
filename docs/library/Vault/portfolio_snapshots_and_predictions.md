# Portfolio Prediction Materialization

> [!summary] Implemented contract
> `materialize_global_portfolio_predictions(...)` writes portfolio-level and base-model prediction parquet outputs under the central cache (`.cache/trading_algo/central_cache/materialized/<scope>/`), keyed by a caller-supplied `portfolio_id`. `prune_inactive_base_model_materializations(...)` removes base-model parquet files whose identities are no longer active in the working vault.
>
> Related: [[Vault/architecture]], [[Vault/user_guide]], [[Vault/vault]], [[Vault/portfolio_snapshot_usage]], [[Cache/architecture]], [[Ensemble/portfolio]], [[Vault/monitoring]].

> [!warning] No immutable vault snapshot layer today
> There is **no** `GlobalPortfolio.save_to_vault(...)` and **no** `load_global_portfolio_snapshot(...)` in the current code, and the vault has **no** `portfolio_snapshots/` directory. The implemented persistence is the *materialized prediction store* described below. The `portfolio_id` is an identifier you choose and pass in; it is the stable join key for the materialized parquet rows. A frozen on-disk portfolio definition reload path may return later (tracked under the Nautilus refactor), but it is not present now.

## Where the entrypoints live

Both functions are implemented in `utils/cache/runtime/portfolio_materialization.py` and re-exported from `ensemble/portfolio.py` (thin delegating wrappers also exist in `ensemble/portfolio_impl/global_portfolio_impl.py`):

```python
from ensemble.portfolio import (
    materialize_global_portfolio_predictions,
    prune_inactive_base_model_materializations,
)
# equivalently: from utils.cache import (...)
```

## Materialized predictions

```python
materialize_global_portfolio_predictions(
    portfolio,                 # a fitted GlobalPortfolio
    query,                     # PortfolioCacheQuery
    portfolio_id,              # caller-chosen stable id (str)
    world,                     # PortfolioWorld or its string value
    research_run_id=None,      # optional grouping tag
    scope=ArtifactScope.LIVE,  # LIVE or RESEARCH
    vault_root="vault",        # working vault used to scope active base models
    cache_root=None,           # override the central-cache root (tests)
    ensemble_dirs=None,        # restrict the active base-model scan
)
```

It calls `portfolio.predict_from_cache(query)` for the portfolio frame, then collects per-base-model frames and writes parquet outputs under the central cache root:

```text
.cache/trading_algo/central_cache/materialized/<scope>/portfolio/<portfolio_id>.parquet
.cache/trading_algo/central_cache/materialized/<scope>/base_models/<timeframe>/<ensemble_name>/<sha256_identity_hash>.parquet
```

The base-model filename is a **SHA-256 hash of the identity** (`timeframe`, `ensemble_name`, `feature_name`, `model_id`), not a `feature__model` string — this keeps paths inside the Windows `MAX_PATH` limit. Pre-hash `feature__model.parquet` files are still parsed (and removed) for backward compatibility.

`<scope>` is the `ArtifactScope` value (`live` or `research`).

Materialized rows keep `world` as a row column and a nullable `research_run_id`. Valid `world` values come from `PortfolioWorld` (`train`, `val`, `test`, `live`); the materializer accepts a `PortfolioWorld` or its string value. The portfolio research pipeline maps its `Validation` phase to `val` before materialization.

Portfolio-level rows include:

- `portfolio_id`
- `world`
- `research_run_id`
- `ticker`
- `datetime`
- `forecast_score`
- `position_fraction`

Base-model rows include the identity columns plus the portfolio fields:

- `timeframe`
- `ensemble_name`
- `feature_name`
- `model_id`
- `portfolio_id`
- `world`
- `research_run_id`
- `ticker`
- `datetime`
- `forecast_score`
- `position_fraction`

Writes are upserted by the materializer's row key (`_upsert_frame` reads the existing parquet, concatenates, deduplicates on the key with last-write-wins, and rewrites), so rerunning with the same identity updates existing rows instead of duplicating them. Only base-model frames whose identity is **active in the working vault** are written; inactive frames are counted as skipped.

## Base-model cleanup

```python
prune_inactive_base_model_materializations(
    vault_root="vault",        # prop dirname by default; pass vault_personal or an absolute path
    scope=ArtifactScope.LIVE,
    ensemble_dirs=None,        # restrict the active scan; otherwise scans D/W/M
    cache_root=None,
)
```

It scans the working vault for active base-model identities (`_scan_active_live_base_model_identities`, which understands both the nested `<vault_root>/<TF>/<group>/<ensemble>/features/*.json` layout and the legacy flat `<vault_root>/<TF>/<ensemble>/features/*.json` layout), then deletes base-model parquet files whose identity hash is not in the active set.

Only stale base-model parquet files are removed. Portfolio-level materializations keyed by historical `portfolio_id` are retained. The function returns a `CleanupSummary` (active identity count, deleted/kept file counts, deleted paths).

## Live deployment handoff

`portfolio_id` is also the deployment handoff point for automatic live refresh.

Typical flow:

1. Fit or select the `GlobalPortfolio` you want to deploy and choose a stable `portfolio_id`.
2. Register that `portfolio_id` in `deployment/config/live_cache_refresh.json`.
3. Keep the LIVE candle cache current.
4. Let the runtime rematerialize live portfolio and base-model outputs automatically from candle writes.

Important distinction:

- the working vault determines the active base-model set and is used to rebuild live bias artifacts
- the deployed portfolio is assembled from the working vault ensemble directories (via `build_global_portfolio_from_ensemble_dirs` / `discover_ensemble_dirs_in_vault`), not from an immutable snapshot file

This is why automatic live refresh can prune stale working-vault base-model parquet files without deleting historical portfolio-level outputs.

## Monitoring relation

`Vault/monitoring` stores raw `(signal, target)` decay vectors per `(feature, model_id)` with a `period` column (`IS`/`OOS`/`LIVE`). The portfolio materialization store is separate and uses `world` (`train`/`val`/`test`/`live`) for research/live context.

Do not conflate `period` and `world` in application code.

> _Verified against commit a07b6bf on 2026-06-04 (docs Phase A)._
