# Portfolio Cache Query API

## Module
- [`ensemble/portfolio.py`](../../../ensemble/portfolio.py)

## Query model
- `PortfolioCacheQuery`
  - `tickers: tuple[str, ...]`
  - `start: datetime`
  - `end: datetime`
  - `timeframes: tuple[TimeFrame, ...]`
  - `volatility_timeframe: TimeFrame = TimeFrame.D`
  - `scope: ArtifactScope = ArtifactScope.LIVE`
  - `grid: tuple[datetime, ...] = ()`

## Cache-native entrypoints
- `TFPortfolio.fit_from_cache(query, target_data=None)`
- `TFPortfolio.predict_from_cache(query, ...)`
- `TFPortfolio.predict_base_model_vectors_from_cache(query, ...)`
- `GlobalPortfolio.fit_from_cache(query, instrument_returns)`
- `GlobalPortfolio.predict_from_cache(query, ...)`
- `GlobalPortfolio.save_to_vault(fit_start, fit_end, vault_root="vault")`
- `load_global_portfolio_snapshot(portfolio_id, vault_root="vault")`
- `materialize_global_portfolio_predictions(portfolio, query, portfolio_id, world, research_run_id=None, scope=ArtifactScope.LIVE)`
- `prune_inactive_base_model_materializations(vault_root="vault", scope=ArtifactScope.LIVE, ensemble_dirs=None)`
- `PortfolioManager.fit_from_cache(query, target_data=None)`
- `PortfolioManager.predict_from_cache(query)`
- `PortfolioTester.fit_from_cache(query)`
- `PortfolioTester.predict_from_cache(query, ...)`

## Volatility contract
- Cache-native portfolio methods resolve EWSD volatility via central cache artifacts.
- Callers do not provide `daily_volatility_df` on cache-native methods.

## Cache contract
- These methods assume the runtime candle cache has already been bootstrapped and kept current.
- Portfolio refresh paths do not auto-ingest from `data/ohlc_data`; missing candle coverage is a setup error that must be resolved before calling the cache-native APIs.

## Compatibility
- Legacy candle-frame methods (`fit`, `predict`, `fit_from_candles`, `predict_from_candles`) remain available as migration shims.

## Snapshot and materialization contract
- `GlobalPortfolio.save_to_vault(...)` writes frozen snapshot files under `vault/portfolio_snapshots/<portfolio_id>/`.
- `load_global_portfolio_snapshot(...)` reloads the snapshot from those frozen copies instead of the mutable working vault.
- `materialize_global_portfolio_predictions(...)` writes portfolio-level rows and active base-model rows into `.cache/trading_algo/central_cache/materialized/<scope>/`.
- `PortfolioWorld` values are `train`, `val`, `test`, and `live`.
- The portfolio research pipeline maps its `Validation` phase to `val` before materialization.
- `prune_inactive_base_model_materializations(...)` scans the working vault for active members, deletes stale base-model parquet files, and keeps historical portfolio materializations.
