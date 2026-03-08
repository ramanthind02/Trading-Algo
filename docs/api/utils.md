# utils

> **Path:** `utils/`  
> **Status:** Draft  
> **Last updated:** 2026-02-13

## Purpose
`utils/` is the shared infrastructure layer used across nodes, feature extraction/selection, ensembles, and deployment. It provides:
- domain enums and typed candle models
- data loading/alignment and bias-node construction helpers
- logging setup helpers
- persistent bias-node cache APIs
- optional fast math/path implementations with Python fallback

## Public API policy
This document includes public API from these modules:
- `utils.core.enums`
- `utils.core.models`
- `utils.core.helpers`
- `utils.core.logger`
- `utils.data.candle_fetcher`
- `utils.cache.bias_node_cache`
- `utils.cache.cache_manager`
- `utils.compute.fast_nodes`, `utils.compute.fast_stats`, `utils.compute.fast_volatility`, `utils.compute.fast_candle`
- `utils.evaluation.walkforward.runner`
- `utils.evaluation.walkforward.portfolio_evaluator`

Included symbols are:
- public names (not prefixed with `_`)
- symbols imported/used outside `utils/` (cross-package dependency surface)
- CLI entrypoint in `utils.cache.cache_manager`

Skipped unless required:
- `_`-prefixed helpers and fallback internals
- test-only details

## Quickstart
```python
from datetime import datetime
from utils.core.enums import Ticker, TimeFrame
from utils.core.helpers import load_data, build_feature_column_name

candles = load_data(Ticker.ES, TimeFrame.D, start=datetime(2020, 1, 1))
col = build_feature_column_name("rsi", "signal", TimeFrame.D, {"lookback": 14})
print(col)  # rsi_signal_D_lookback_14
print(candles.columns.tolist())
```

```python
import numpy as np
from utils.compute.fast_volatility import compute_ewsd_annualized_from_closes

sigma = compute_ewsd_annualized_from_closes(np.array([100.0, 101.0, 99.5, 102.0]))
print(round(sigma, 4))
```

## Data contracts
- **Candle model (`utils.core.models.Candle`)**
  - fields: `datetime, open, high, low, close, volume, ticker, tf`
  - computed fields: `range, body_high, body_low, end_time`
- **Candle DataFrame contract (loaders/alignment/cache manager)**
  - expected columns: `datetime, open, high, low, close, volume`
  - optional/enriched: `ticker, timeframe`
  - ordering: ascending by `datetime`
- **Feature column naming contract (`helpers.build_feature_column_name`)**
  - format: `{module}_{feature}_{tf}_{param}_{value}...`
  - module/feature snake_case; param keys camelCase; params alphabetically ordered
- **Bias-node cache parquet contract**
  - path format: `cache/{module_name}/{ticker}_{tf}_{params_suffix}.parquet`
  - index: datetime index named `datetime`
  - main column: `value` (or first column as primary for multi-output nodes)
- **Time alignment / no-lookahead**
  - `align_candles_with_features(...)` does index intersection/reindex only; it does not forward-fill future candles
  - `CacheManager` computes node outputs by streaming candles in time order
  - fast volatility/stat APIs are array-based; caller must pass history available at decision time only

## Public API reference

### `utils.core.enums`

`TimeFrame`  
Type: enum  
Signature: `class TimeFrame(Enum): H1, H4, D, W, M` with `bars_per_year` property  
Behavior: canonical timeframe enum with ordering, `higher_timeframes(current_tf)`, and timeframe-aware annualization via `bars_per_year`.

`Ticker`  
Type: enum  
Signature: `class Ticker(Enum): ...`  
Behavior: canonical instrument universe enum used in nodes, ensembles, and deployment.

`Bias` / `PositionMode` / `ResamplingMethod`  
Type: enum  
Signature: `class Bias(Enum)`, `class PositionMode(Enum)`, `class ResamplingMethod(Enum)`  
Behavior: direction/bias, long_short mode constraints, and robustness resampling mode constants.

`Direction`  
Type: enum  
Signature: `class Direction(Enum); from_string(direction_str: str) -> Direction`  
Behavior: long/short routing enum with case-insensitive parser; raises `ValueError` on unknown string.

### `utils.core.models`

`Candle`  
Type: class (Pydantic model)  
Signature: `class Candle(BaseModel)`  
Behavior: typed candle object used by node APIs and feature extraction.

`Candle.from_row`  
Type: classmethod  
Signature: `from_row(row: pd.Series) -> Candle`  
Behavior: converts DataFrame row into `Candle`; requires ticker/timeframe.

`Candle.from_row_fast`  
Type: classmethod  
Signature: `from_row_fast(row) -> Candle`  
Behavior: lower-overhead conversion from `itertuples`-style rows.

`Candle.convert_for_mongo_db`  
Type: method  
Signature: `convert_for_mongo_db() -> dict`  
Behavior: emits UTC timestamped dict with ticker/timeframe names.

### `utils.core.helpers` (cross-package surface)

`load_data`  
Type: function  
Signature: `load_data(ticker: Ticker, timeframe: TimeFrame, start: datetime, end: datetime) -> pd.DataFrame`  
Behavior: reads parquet candles for a single ticker/timeframe, filters date range, sorts by timestamp index.

`load_data_multi_ticker`  
Type: function  
Signature: `load_data_multi_ticker(tickers: list[Ticker], timeframe: TimeFrame, start: datetime, end: datetime, use_millisecond_offset: bool = True) -> pd.DataFrame`  
Behavior: concatenates multiple ticker candle streams; optional millisecond offset avoids duplicate datetimes.

`load_numpy_data`  
Type: function  
Signature: `load_numpy_data(ticker: Ticker, timeframe: TimeFrame, start: datetime, end: datetime) -> np.ndarray`  
Behavior: returns structured array (`open, close, high, low, datetime`) for fast paths.

`create_bias_node`  
Type: function  
Signature: `create_bias_node(module_name: str, ticker: Ticker, tf: TimeFrame, params: dict) -> Any`  
Behavior: resolves node module/class dynamically and returns instantiated node.

`build_feature_column_name`  
Type: function  
Signature: `build_feature_column_name(module: str, feature: str, tf: TimeFrame, params: dict[str, Any]) -> str`  
Behavior: canonical feature-column naming for feature extraction and ensemble parsing.

`parse_feature_column_name`  
Type: function  
Signature: `parse_feature_column_name(name: str) -> dict[str, Any]`  
Behavior: inverse parser returning `{module, feature, tf, params}`.

`align_candles_with_features`  
Type: function  
Signature: `align_candles_with_features(candles_df: pd.DataFrame, features_df: pd.DataFrame, datetime_col: str = "datetime") -> pd.DataFrame`  
Behavior: timezone-normalizes to UTC and aligns candles to feature index by exact datetime match.

`convert_ftmo_time_to_ny_time` / `convert_ny_time_to_ftmo_time`  
Type: function  
Signature: `(timestamp: int) -> datetime`, `(dt_ny: datetime) -> int`  
Behavior: FTMO server <-> New York conversions used by deployment connectors.

### `utils.core.logger`

`setup_logger`  
Type: function  
Signature: `setup_logger(log_level=logging.INFO, log_to_file=True)`  
Behavior: configures root logging handlers/filters and warning suppression.

`get_logger`  
Type: function  
Signature: `get_logger(name)`  
Behavior: returns named logger with `FeatureExtractorFilter` applied.

### `utils.data.candle_fetcher`

`get_candle`  
Type: function (Numba-jitted)  
Signature: `get_candle(data: np.ndarray, start_ts: int, method: str)`  
Behavior: returns candle tuple for exact/closest lookup from structured array.

`CandleFetcher`  
Type: class  
Signature: `CandleFetcher(ticker: Ticker, tfs: list[TimeFrame])`  
Behavior: preloads per-timeframe numpy candle arrays; `get_candle(...)` does fast timestamp lookup.

### `utils.cache.bias_node_cache`

`CacheMissError`  
Type: exception class  
Signature: `CacheMissError(module_name, params, ticker, tf, date_range=None, cache_path=None, reason=...)`  
Behavior: raised when required cache is missing/incomplete, with remediation details.

`BiasNodeCache`  
Type: class  
Signature: `BiasNodeCache(module_name: str, params: dict, ticker: Ticker, tf: TimeFrame, cache_dir: str | None = None)`  
Behavior: persistent parquet cache interface for node outputs.

`BiasNodeCache` key methods  
Type: methods  
Signature: `exists()`, `load()`, `save(data)`, `get_values(start=None, end=None, require_cache=True)`, `get_dataframe(...)`, `invalidate()`, `get_metadata()`  
Behavior: load/save/query cache slices with optional strict cache requirement.

### `utils.cache.cache_manager`

`CacheManager`  
Type: class  
Signature: `CacheManager(cache_dir: str | None = None, candle_dir: str | None = None)`  
Behavior: orchestrates multi-node, multi-ticker cache population.

`CacheManager.populate_cache`  
Type: method  
Signature: `populate_cache(bias_node_specs, tickers, start_date, end_date, max_workers=4, overwrite_existing=True, show_progress=True, timeframe=TimeFrame.D) -> dict`  
Behavior: computes and persists cache for requested matrix; auto-adds required auxiliary `atr`/`ewsd` specs scaled by `timeframe`.

`get_auxiliary_specs_for_timeframe`  
Type: function  
Signature: `get_auxiliary_specs_for_timeframe(tf: TimeFrame) -> list[dict]`  
Behavior: returns required ATR/EWSD auxiliary specs for volatility-scaled targets using timeframe-aware windows.

`CacheManager.populate_cache_for_vault`  
Type: method  
Signature: `populate_cache_for_vault(vault_ensemble_dir, start_date, end_date, max_workers=4, overwrite_existing=True, show_progress=True) -> dict`  
Behavior: parses ensemble control file and populates all required caches.

`CacheManager.list_caches` / `clear_caches`  
Type: methods  
Signature: `list_caches(module_name=None) -> list[dict]`, `clear_caches(module_name=None, ticker=None, confirm=False) -> int`  
Behavior: cache introspection and deletion utilities.

`main`  
Type: CLI entrypoint  
Signature: `main()`  
Behavior: `python -m utils.cache.cache_manager ...` command interface.

### fast modules (`utils.fast_*`)

`utils.compute.fast_nodes`  
Type: module constants/functions  
Signature: `CYTHON_NODES_AVAILABLE` and functions including `compute_atr_fast`, `compute_ema_fast`, `compute_high_low_channel_fast`, `compute_momentum_fast`, `compute_roc_fast`, `compute_rsi_initial_fast`, `update_rsi_fast`, `compute_stddev_sample_fast`, `compute_return_fast`, `batch_compute_log_returns`  
Behavior: Cython-first numeric kernels with Python fallbacks for node math.

`utils.compute.fast_stats`  
Type: module functions  
Signature: `rank_with_tie_correction`, `spearman_rho`, `optimize_threshold_fast`, `compute_ma_diff_fast`  
Behavior: optimized statistics and threshold/MA-diff helpers with fallback behavior.

`utils.compute.fast_volatility.compute_ewsd_annualized_from_closes`  
Type: function  
Signature: `compute_ewsd_annualized_from_closes(closes: np.ndarray, lambda_short=..., long_run_window=..., blend_short_weight=..., blend_long_weight=...) -> float`  
Behavior: Carver-style blended annualized volatility estimate from close series.

`utils.compute.fast_candle.FastCandle` / `create_fast_candle_from_numpy`  
Type: dataclass/function  
Signature: `FastCandle.from_numpy(...) -> FastCandle`, `create_fast_candle_from_numpy(...) -> FastCandle`  
Behavior: lightweight candle representation for performance-critical loops.

### `utils.evaluation.walkforward.runner` / `portfolio_evaluator`

`run_portfolio_simulation`  
Type: function  
Signature: `run_portfolio_simulation(candles_df, target, fold_rows, selection_summary_df, research_config, feature_data_by_combo=None, tearsheets_dir=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | None]`  
Behavior: runs fold-level portfolio evaluation and optionally writes walkforward tearsheets.  
Ticker tearsheet behavior:
- reads `research_config.generate_ticker_tearsheets` (default `False`)
- when enabled, writes per-fold ticker tearsheets at `tearsheets/fold_{fold_id}/fold_{fold_id}_{ticker}_tearsheet.html`
- when enabled, writes aggregate OOS ticker tearsheets at `tearsheets/walkforward_{ticker}_tearsheet.html`
- existing ensemble/per-signal tearsheets are unchanged

`FoldPortfolioResult`  
Type: dataclass  
Signature: includes `oos_portfolio_returns`, optional `per_signal_oos_returns`, optional `per_ticker_oos_returns`  
Behavior: carries fold OOS return series used by runner tearsheet generation for ensemble, signal, and ticker-level reports.

## Internal but required
- `helpers._get_functime_function(...)` is private but required when `create_bias_node` receives string transformations for `ts_feature`.
- `bias_node_cache` filename internals (`_build_params_suffix`, `_hash_params`) are private but operationally important for deterministic cache lookup and collision avoidance.
- `cache_manager.REQUIRED_AUXILIARY_SPECS` is a backward-compatible alias for the daily auxiliary specs; `populate_cache(...)` now resolves active auxiliary specs via `get_auxiliary_specs_for_timeframe(...)`.

## Errors & logging
- Common exceptions:
  - `FileNotFoundError`: missing candle/cache parquet paths
  - `CacheMissError`: strict cache required but unavailable
  - `ValueError`: invalid enum parsing, malformed inputs, unsupported lookup methods
  - `ImportError`/`RuntimeError`: dynamic node import/instantiation failures in `create_bias_node`
- Logging:
  - `utils.core.logger` applies `FeatureExtractorFilter` and warning suppression globally
  - cache modules emit `info/warning/error` events for save/load/partial coverage/failures
  - `CacheManager` also prints progress to stdout when `show_progress=True`

## Open questions
Q1: `helpers.get_ticker_list()` references `Ticker.NG`, but `Ticker` currently has no `NG` member in `utils.core.enums`. Is this a stale ticker or missing enum value?

Q2: `logger.setup_logger(log_to_file=True)` writes to a hard-coded Windows path (`C:/Users/Administrator/Desktop/logs`). Should this be repository-relative or environment-configurable for non-Windows runs?

Q3: `candle_fetcher.get_candle` annotation says `Optional[Dict]`, but runtime returns a tuple-like value used positionally. Should type hints/docs be updated to reflect actual return type?
