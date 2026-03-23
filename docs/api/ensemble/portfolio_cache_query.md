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
