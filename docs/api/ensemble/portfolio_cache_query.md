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
- `discover_ensemble_dirs_in_vault(vault_root, timeframes)`
- `build_global_portfolio_from_ensemble_dirs(ensemble_dirs, active_timeframes=..., target_volatility=..., max_position_pct=..., idm_max=...)`
- `materialize_global_portfolio_predictions(portfolio, query, portfolio_id, world, research_run_id=None, scope=ArtifactScope.LIVE, vault_root="vault", cache_root=None)`
- `prune_inactive_base_model_materializations(vault_root="vault", scope=ArtifactScope.LIVE, ensemble_dirs=None, cache_root=None)`
- `PortfolioTester.fit_from_cache(query)`
- `PortfolioTester.predict_from_cache(query, ...)`

## Deprecated compatibility entrypoints
- `PortfolioManager.fit_from_cache(query, target_data=None)` and `PortfolioManager.predict_from_cache(query)` remain available through `ensemble.portfolio_manager` as compatibility shims, but they are no longer re-exported from `ensemble`.

## Volatility contract
- Cache-native portfolio methods resolve EWSD volatility via central cache artifacts.
- Callers do not provide `daily_volatility_df` on cache-native methods.

## Cache contract
- These methods assume the runtime candle cache has already been bootstrapped and kept current.
- Portfolio refresh paths do not auto-ingest from `data/ohlc_data`; missing candle coverage is a setup error that must be resolved before calling the cache-native APIs.

## Compatibility
- Legacy candle-frame methods (`fit`, `predict`, `fit_from_candles`, `predict_from_candles`) remain available as migration shims.

## Materialization contract
- `materialize_global_portfolio_predictions(...)` writes portfolio-level rows and active base-model rows into `.cache/trading_algo/central_cache/materialized/<scope>/`.
- `PortfolioWorld` values are `train`, `val`, `test`, and `live`.
- The portfolio research pipeline maps its `Validation` phase to `val` before materialization.
- `prune_inactive_base_model_materializations(...)` scans the working vault for active members, deletes stale base-model parquet files, and keeps historical portfolio materializations.
