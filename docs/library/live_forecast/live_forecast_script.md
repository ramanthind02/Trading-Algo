## How the Live Trading Script Works                                                                                                          
                                                                                                                                             
  ### Testing (no Telegram sent, prints what it would send)   
  ```python
  python scripts/tws_live_forecast.py --dry-run --port 7497
  ```

  ### Production (sends Telegram)
  ```python
  python scripts/tws_live_forecast.py --port 7497
  ```

  ### Override capital
  ```python
  python scripts/tws_live_forecast.py --dry-run --port 7497 --capital 5000
  ```

  - --port 7497 = paper trading, --port 7496 = live trading
  - --dry-run = everything runs identically except the Telegram message is printed to console instead of sent
  - --capital = base value of account to trade
  ---

### Step-by-step flow

  **Step 1 -- Load the models from vault**

```
  vault/
    D/  (5 daily ensembles)
      mr_indices_long/          → trades ES, NQ
      rebalancing_es_tlt_long/  → trades ES (uses TLT as cross-ticker)
      rebalancing_tlt_es_long/  → trades TLT (uses ES as cross-ticker)
      seasonal_bonds_long_short/ → trades TLT
      seasonal_indices_long/    → trades ES
    M/  (1 monthly ensemble)
      buy_hold_long/            → trades ES, GC, NQ, RTY, TLT, YM
```

  The script reads every `ensemble_config.json` + `features/*.json` file in the vault. Each feature file defines one bias node (technical        
  indicator) with a rule-based model that converts the indicator into a binary signal. The vault stores the configuration -- model type,     
  parameters, strategy direction -- not fitted weights. These are rule-based models (not ML), so there are no trained weights to store.      

  The vault ensembles are loaded into a `GlobalPortfolio` containing two `TFPortfolio` instances (one for daily, one for monthly).

  **Step 2 -- Discover what tickers we need**

  The script scans all ensembles to find which tickers they trade and which cross-tickers their bias nodes reference. Result: `['ES', 'GC',   
  'NQ', 'RTY', 'TLT', 'YM']`.

  **Step 3 -- Ensure the cache has historical data (bootstrap)**

  The central cache (`.cache/trading_algo/central_cache/`) is the runtime data store. On first run, it's empty. The script checks if each      
  required ticker has daily candles in the cache:

  - If empty: Bootstraps from `data/ohlc_data/` -- the Norgate parquet snapshots committed in the repo. This loads ~7000 daily bars per ticker 
  (back to 1997 for ES). This is the only time disk parquets are read.
  - If already populated: Skips bootstrap (cache has data from previous runs).

  `data/ohlc_data/` is never written to by this script. It's an immutable baseline.

  **Step 4 -- Fetch live data from TWS API**

  Connects to Interactive Brokers TWS and fetches daily OHLCV bars. The fetch uses smart lookback -- it checks each ticker's last cached date
   and only fetches the gap:

  - First run after bootstrap: cache ends at 2026-03-27 (Norgate data), today is 2026-03-28 → fetches ~10 days (minimum)
  - After running daily for a week: cache is 1 day behind → fetches ~10 days (minimum with buffer)
  - If cache is wiped: fetches 365 days (max fallback)

  Each ticker is fetched as either CONTFUT (futures: ES, NQ, YM, RTY, GC) or STK (ETF: TLT), based on sec_type in the config.

  **Step 5 -- Persist TWS data into the central cache**

  `upsert_tws_candles()` does two things:

  1. Daily candles: Calls `CentralCacheStore.upsert_candles()` per ticker -- this merges new bars with existing cached bars (deduplicates by   
  date, keeps latest). The merged result is written to `.cache/.../candles/{ticker}/D.parquet`.
  1. Monthly candles: Reads the full cached daily series back from the cache, resamples to monthly (OHLCV aggregation), and upserts the      
  monthly candles. This ensures monthly bars are always derived from the complete history, not just the TWS fetch window.

  When candles are upserted, the cache automatically marks all dependent bias node artifacts as STALE (via `mark_dependents_stale()`).

  **Step 6 -- Refresh bias node caches**

  `refresh_bias_caches()` calls `CacheManager.ensure_vault_cache_coverage()` with `refresh_mode="missing_stale_only"`. This:

  1. Scans all vault ensembles to determine which bias nodes need artifacts
  2. For each (`bias_node`, `ticker`, `timeframe`) combination, checks the lifecycle state
  3. If FRESH → skips (already up to date)
  4. If STALE or MISSING → recomputes by streaming cached candles through the bias node, then writes the result to
  `.cache/.../artifacts/live/{module_name}/{ticker}_{tf}_{params}.parquet`

  This also includes EWSD volatility as an auxiliary -- the exponentially weighted standard deviation used for forecast scaling.

  What gets cached:
  - `candles/{ticker}/D.parquet` -- daily OHLCV bars
  - `candles/{ticker}/M.parquet` -- monthly OHLCV bars
  - `artifacts/live/turnaroundtuesday/ES_D_mode_tue_wed.parquet` -- turnaround tuesday indicator values for ES
  - `artifacts/live/rebalancing/ES_D_cf3ab63d.parquet` -- rebalancing signal for ES
  - `artifacts/live/ewsd/ES_D_long_run_window_2520.parquet` -- daily volatility for ES
  - ...etc for each (bias_node, ticker, timeframe) combination

  These cached artifacts are what enable fast vectorized backtests later -- the bias nodes don't need to be recomputed from raw candles every time.

  **Step 7 -- Fit the portfolio**

  `portfolio.fit_from_cache(query, instrument_returns)` does:

  1. Reads candles from cache for each timeframe (D and M)
  2. Reads EWSD volatility from cache
  3. Fits each ensemble by streaming candles through each base model's bias node → model
  4. Computes ensemble weights and exposure fractions
  5. Computes global IDM (Instrument Diversification Multiplier) from return correlations

  "Fitting" here doesn't mean ML training -- it means initializing model state from historical candles (e.g., RSI needs prior bars to compute
   its value) and calculating diversification parameters (IDM, FDM).

  **Step 8 -- Generate forecasts**

  `portfolio.predict_from_cache(query)` does:

  1. Each bias node produces a feature value (e.g., turnaround tuesday signal = 0 or 1)
  2. Each rule-based model converts the feature to a binary signal
  3. Each signal is volatility-scaled: forecast = (target_vol / (sigma * sqrt(h))) * signal
  4. Signals within each ensemble are combined with equal weights + FDM (Forecast Diversification Multiplier)
  5. Daily and monthly ensemble forecasts are combined by GlobalWeightLayer
  6. Global instrument weights + global IDM are applied
  7. Result is clipped to [-2.5, +2.5]

  Output: DataFrame with `['ticker', 'datetime', 'forecast_score', 'position_fraction']`

  **Steps 9-11 -- Position sizing + output**

  - Fetches current ETF prices from TWS (SPY, QQQ, GLD, IWM, TLT)
  - Converts position fractions to dollar amounts and fractional shares: shares = (position_fraction * capital) / etf_price
  - Formats console output with signal interpretation (Strong Bullish → Strong Bearish)
  - --dry-run: prints the Telegram message to console
  - Without --dry-run: sends via Telegram bot API

  ---

**Summary diagram**

```
  data/ohlc_data/  ─── bootstrap (first run) ──→  .cache/central_cache/candles/
                                                           ↑
  TWS API  ─── fetch (daily, smart lookback) ──── upsert ──┘
                                                           ↓ marks stale
                                                .cache/central_cache/artifacts/
                                                           ↑ rebuild stale
                                                ensure_vault_cache_coverage()
                                                           ↓
                                                GlobalPortfolio.fit_from_cache()
                                                GlobalPortfolio.predict_from_cache()
                                                           ↓
                                                position sizing → Telegram
```
