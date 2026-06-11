# Production Testing

> [!summary] Purpose
> Verify the `ForecastServer` (`deployment/forecast_server.py`) works correctly before deployment, without waiting for the live schedule (daily 00:00 EST / weekly Sunday 18:00 EST). `ForecastServer` connects to MT5 via `deployment/mt5_data_connector.py` and uses `TelegramNotifier` from `lib/core/notify.py`.
> Bias nodes need historical candle buffers — `ForecastServer.load_historical_data()` handles this on startup.

---

## Manual forecast test

Use `scripts/run_manual_forecast.py` (wraps `deployment.forecast_server.ForecastServer`) before deployment. Unit coverage lives under `tests/unit-tests/deployment/`.

---

## Quick Start Script

```bash
# Test all timeframes
python scripts/run_manual_forecast.py

# Test specific timeframe
python scripts/run_manual_forecast.py --timeframe D

# Save results to file
python scripts/run_manual_forecast.py --output results.json
```

---

## Pre-Deployment Checklist

1. Run `python scripts/run_manual_forecast.py --output pre_prod_test.json`
2. Verify forecasts for all 8 ticker/timeframe combinations (`count` = 4 per timeframe)
3. Confirm `errors` list is empty
4. Confirm predictions are reasonable values (not NaN, not all zeros)
5. Save the deployed `GlobalPortfolio` snapshot and record its `portfolio_id`
6. Update `deployment/config/live_cache_refresh.json` with the active working-vault ensemble dirs and deployed `portfolio_id`
7. Confirm the central LIVE candle cache is being updated for the tracked ticker/timeframe set

## Live Cache Refresh Ops

For cache-native live deployment, the operational contract is:

1. Keep `CentralCacheStore` LIVE candles current.
2. Let the runtime refresh stale live bias artifacts automatically.
3. Let the runtime rematerialize live base-model and portfolio outputs automatically.
4. Inspect `.cache/trading_algo/central_cache/live_refresh/last_run.json` if forecasts stop updating.

This path is inference only. It does not refit models and it does not create new portfolio snapshots during live operation.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `No historical data for EU D` | MT5 terminal not running | Start MT5 and log in |
| `forecasts['D']['count'] == 0` | Buffers not loaded | Check `len(server.candle_buffers)` == 8, `len(server.ml_managers)` == 8 |
| `RSI requires 14 candles, only 5 available` | Lookback too short | Set `server.lookback_candles = 150` then `server.load_historical_data()` |
| Live materialized forecasts stopped updating | Live refresh cycle failed after candle ingest | Inspect `data/broker_cache/{broker}/central_cache/live_refresh/last_run.json` for the Nautilus path or `.cache/trading_algo/central_cache/live_refresh/last_run.json` for the legacy path, fix the underlying issue, then run `run_live_cache_refresh_now(...)` |

---

## Production Monitoring

- Startup: watch logs for `Loaded X candles` per instrument
- Scheduled runs: look for `Running forecasts...` at trigger times
- Errors: any line with an error indicator signals a failed forecast

---

## See Also

- [[portfolio]] — ensemble and position logic called by ForecastServer
- [[vault]] — how models are loaded at startup
- [[Deployment/live_cache_refresh]] — manifest contract and automatic refresh behavior after LIVE candle writes
- `deployment/live/README.md` — Nautilus vault live runtime (MT5/CFD path: `run_vault_sandbox.py`, exec tiers sandbox/demo/live)

> _Verified against current code via CodeGraph on 2026-06-07._
