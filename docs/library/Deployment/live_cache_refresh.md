# Live Cache Refresh

> [!summary]
> When `ArtifactScope.LIVE` candles are written into the central cache, the runtime can automatically refresh stale live bias artifacts and rematerialize live base-model and portfolio predictions. Base-model **vault** JSON is not refit here; the portfolio step rebuilds a `GlobalPortfolio` from the working vault and runs `fit_from_cache` / `predict_from_cache` on the configured replay window (no separate frozen snapshot tree).

Related: [[Cache/architecture]], [[Cache/user_guide]], [[Deployment/live_multi_timeframe]], [[Vault/vault]].

## What it does

With a valid `deployment/config/live_cache_refresh.json` in place:

- `CentralCacheStore.set_candles(...)` and `upsert_candles(...)` notify the live refresh orchestrator after the candle write succeeds
- only `ArtifactScope.LIVE` writes participate
- dirty `(ticker, timeframe)` updates are coalesced and debounced in-process
- the refresh cycle rebuilds stale live bias artifacts through `ensure_vault_cache_coverage(...)`
- the cycle then builds each affected `GlobalPortfolio` from `active_ensemble_dirs` (or per-portfolio `ensemble_dirs`), runs `fit_from_cache` on the replay-window query, and materializes live predictions
- live predictions are materialized into `.cache/trading_algo/central_cache/materialized/live/...`
- stale base-model materialization parquet files that no longer belong to the working vault are pruned

What it does **not** do:

- no base-model refits inside vault feature JSON (bias refresh only rebuilds missing/stale **cache** artifacts)
- no research-scope orchestration

## Manifest contract

`vault_root` is a string passed to `lib.core.vault_paths.resolve_vault_root` (repo-relative dirnames such as `vault` or `vault_personal`, or absolute paths). `active_ensemble_dirs` entries must be repo-relative and use the matching top-level folder.

Path:

```text
deployment/config/live_cache_refresh.json
```

Example:

```json
{
  "version": "1",
  "enabled": true,
  "vault_root": "vault",
  "debounce_seconds": 2,
  "active_ensemble_dirs": [
    "vault/M/buy_hold/buy_hold_long"
  ],
  "active_portfolios": [
    {
      "name": "buy_hold_live",
      "portfolio_id": "prop_live_portfolio",
      "tickers": ["ES", "NQ", "ZN"],
      "timeframes": ["M"],
      "volatility_timeframe": "D",
      "replay_window_days": 62,
      "research_run_id": "live_2026_03"
    }
  ]
}
```

Field meanings:

- `version`: manifest schema version. Current value is `"1"`.
- `enabled`: global on/off switch for automatic live refresh.
- `vault_root`: working vault root used for bias refresh, portfolio assembly, and stale base-model cleanup.
- `debounce_seconds`: coalescing window for repeated candle writes. Default deployment value is `2`.
- `active_ensemble_dirs`: default working-vault ensemble dirs used for live bias refresh and (unless overridden) for assembling each `GlobalPortfolio`.
- `active_portfolios`: deployed live portfolio definitions to materialize.

Per active portfolio:

- `name`: operator-facing label used in status output.
- `portfolio_id`: stable string key used in materialized parquet rows and filenames (choose one per deployed portfolio).
- `tickers`: live traded instrument set.
- `timeframes`: trading timeframes for the portfolio.
- `volatility_timeframe`: required volatility lineage timeframe, usually `D`.
- `replay_window_days`: bounded lookback window for live re-materialization. Use `62` in examples unless the deployment needs something else.
- `research_run_id`: optional row tag written into materialized parquet outputs.
- `ensemble_dirs` (optional): list of repo-relative ensemble directories for this portfolio; defaults to manifest `active_ensemble_dirs` when omitted.
- `target_volatility`, `max_position_pct`, `idm_max` (optional): portfolio construction parameters (defaults `0.20`, `2.5`, `2.5`).

Operational meaning:

- `active_ensemble_dirs` point at the **working vault** ensembles used for bias refresh and (by default) portfolio assembly
- `portfolio_id` is a **stable operator-chosen id** for cache outputs, not a content hash
- the manifest is the explicit source of truth for the active live set in v1

## Minimal operator workflow

Use this as the deployment checklist for the automatic path:

1. Save the fitted `GlobalPortfolio` and keep the returned `portfolio_id`.
2. Put that `portfolio_id` plus the active working-vault ensemble dirs into `deployment/config/live_cache_refresh.json`.
3. Make sure the LIVE candle cache is updated for the tracked `tickers`, `timeframes`, and `volatility_timeframe`.
4. Let the orchestrator refresh downstream live caches automatically.
5. Check `.cache/trading_algo/central_cache/live_refresh/last_run.json` when you need status or failure details.

In steady state, operators only need to worry about the candle cache being current.

## Refresh cycle

For each coalesced live refresh run:

1. Load the manifest and keep only portfolios whose tracked `(ticker, timeframe)` set intersects the dirty keys.
2. Compute `end` as the latest common candle timestamp available across each affected portfolio's `tickers`, `timeframes`, and `volatility_timeframe`.
3. Compute `start = end - replay_window_days`.
4. Run `ensure_vault_cache_coverage(...)` once over the aggregate min/max replay window for the configured `active_ensemble_dirs`.
5. Load each affected `GlobalPortfolio` from `portfolio_id`.
6. Call `materialize_global_portfolio_predictions(...)` with `world=live`.
7. Prune inactive base-model materialization parquet files that no longer belong to the working vault.

This is why live operations only need to keep the candle cache current: everything downstream is refreshed from that write boundary.

## Materialized outputs

Portfolio rows:

```text
.cache/trading_algo/central_cache/materialized/live/portfolio/<portfolio_id>.parquet
```

Base-model rows:

```text
.cache/trading_algo/central_cache/materialized/live/base_models/<timeframe>/<ensemble_name>/<feature_name>__<model_id>.parquet
```

Important clarification:

- "base-model cache" here means the **materialized base-model prediction parquet store**
- it does **not** mean fitted model state

## Failure semantics

The live refresh path is best-effort and async:

- candle writes still succeed if downstream refresh fails
- failed runs keep their dirty keys queued for the next retry-triggering write
- status is written to:

```text
.cache/trading_algo/central_cache/live_refresh/last_run.json
```

The status file records:

- `status`
- `dirty_keys`
- `affected_portfolios`
- `bias_refresh_summary`
- `materialized_portfolios`
- `started_at` / `finished_at`
- `error`

## Manual recovery and testing

Use the explicit runtime helper when you want to force one refresh cycle:

```python
from cache import run_live_cache_refresh_now

summary = run_live_cache_refresh_now(
    manifest_path="deployment/config/live_cache_refresh.json",
)
```

You can also pass `dirty_keys=[("ES", "D")]` when testing a narrower replay scope.

## Common mistakes

- Do not expect live candle writes to refit anything. The path is inference only.
- Do not point `portfolio_id` at the working vault. It must reference a saved portfolio snapshot.
- Do not confuse base-model materialization parquet files with fitted base-model state.
- Do not assume every candle write triggers a full refresh. Only dirty keys tracked by the manifest participate.
- Do not delete historical portfolio parquet files during cleanup. Cleanup only targets stale base-model materializations.

> _Verified against current code via CodeGraph on 2026-06-07._
