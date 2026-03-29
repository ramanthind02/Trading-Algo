# Portfolio Snapshots and Prediction Materialization

> [!summary] Implemented contract
> `GlobalPortfolio.save_to_vault(...)` writes an immutable snapshot under `vault/portfolio_snapshots/<portfolio_id>/`. `load_global_portfolio_snapshot(...)` reloads that snapshot from the frozen files, and `materialize_global_portfolio_predictions(...)` writes portfolio and base-model parquet outputs under the central cache.
>
> Related: [[Vault/architecture]], [[Vault/user_guide]], [[Vault/vault]], [[Vault/portfolio_snapshot_usage]], [[Cache/architecture]], [[Ensemble/portfolio]], [[Vault/monitoring]].

## Snapshot storage

`GlobalPortfolio.save_to_vault(fit_start, fit_end, vault_root="vault") -> str` returns the snapshot id used as `portfolio_id`.

Each snapshot is stored at:

```text
vault/portfolio_snapshots/<portfolio_id>/snapshot.json
vault/portfolio_snapshots/<portfolio_id>/ensembles/<TF>/<ensemble_name>/ensemble_config.json
vault/portfolio_snapshots/<portfolio_id>/ensembles/<TF>/<ensemble_name>/features/*.json
```

The snapshot hash is derived from semantic payload only. It includes the fit window, portfolio config, fitted portfolio state, fitted weight-layer state, TF portfolio state, global IDM state, member identities, and hashes of the copied ensemble files. Wall-clock write time is not part of the id.

The copied ensemble files are frozen snapshot-local copies. They are used when loading the snapshot back with `load_global_portfolio_snapshot(portfolio_id, vault_root="vault")`.

## Materialized predictions

`materialize_global_portfolio_predictions(portfolio, query, portfolio_id, world, research_run_id=None, scope=ArtifactScope.LIVE)` writes parquet outputs under the central cache root:

```text
.cache/trading_algo/central_cache/materialized/<scope>/portfolio/<portfolio_id>.parquet
.cache/trading_algo/central_cache/materialized/<scope>/base_models/<timeframe>/<ensemble_name>/<feature_name>__<model_id>.parquet
```

Materialized rows keep `world` as a row column and support a nullable `research_run_id`. Valid `world` values are `train`, `val`, `test`, and `live`.
The portfolio research pipeline maps its `Validation` phase to `val` before materialization.

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

Writes are upserted by the materializer's row key, so rerunning with the same identity updates existing rows instead of duplicating them.

## Base-model cleanup

`prune_inactive_base_model_materializations(vault_root="vault", scope=ArtifactScope.LIVE, ensemble_dirs=None)` scans the current working vault for active base-model members.

Only stale base-model parquet files are removed. Portfolio-level materializations keyed by historical `portfolio_id` are retained.

This keeps the active base-model cache aligned with the working vault while preserving historical portfolio outputs.

## Live deployment handoff

Saved `portfolio_id` values are also the deployment handoff point for automatic live refresh.

Typical flow:

1. Fit or select the `GlobalPortfolio` you want to deploy.
2. Save it once with `GlobalPortfolio.save_to_vault(...)`.
3. Put that `portfolio_id` into `deployment/config/live_cache_refresh.json`.
4. Keep the LIVE candle cache current.
5. Let the runtime rematerialize live portfolio and base-model outputs automatically from candle writes.

Important distinction:

- the working vault is still used to determine the active base-model set and to rebuild live bias artifacts
- the deployed portfolio itself is loaded from the immutable snapshot referenced by `portfolio_id`

This is why automatic live refresh can prune stale working-vault base-model parquet files without deleting historical portfolio-level outputs.

## Monitoring relation

`Vault/monitoring` stores decay diagnostics per `(feature, model_id)` with a `period` column. The portfolio materialization store is separate and uses `world` for research/live context.

Do not conflate `period` and `world` in application code.
