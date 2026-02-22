# Production Testing

> [!summary] Purpose
> Verify the `ForecastServer` works correctly before deployment, without waiting for the live schedule (daily 00:00 EST / weekly Sunday 18:00 EST).
> Bias nodes need historical candle buffers — `TestForecastServer` handles this explicitly.

---

## TestForecastServer

`TestForecastServer` extends `ForecastServer` with manual test methods.

```python
from deployment.test_forecast_server import TestForecastServer
from utils.enums import TimeFrame

server = TestForecastServer()
server.load_historical_data()

# Test all timeframes
results = server.run_test_forecast()

# Test a single timeframe
results = server.run_test_forecast(timeframe=TimeFrame.D)

server.stop()
```

### Output Format

```python
{
    'timestamp': '2024-01-15T14:30:00-05:00',
    'forecasts': {
        'D': {'count': 4, 'tickers': ['EU', 'BP', 'ES', 'NQ'], 'predictions': [...]},
        'W': {'count': 4, 'tickers': ['EU', 'BP', 'ES', 'NQ'], 'predictions': [...]}
    },
    'errors': []
}
```

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

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `No historical data for EU D` | MT5 terminal not running | Start MT5 and log in |
| `forecasts['D']['count'] == 0` | Buffers not loaded | Check `len(server.candle_buffers)` == 8, `len(server.ml_managers)` == 8 |
| `RSI requires 14 candles, only 5 available` | Lookback too short | Set `server.lookback_candles = 150` then `server.load_historical_data()` |

---

## Production Monitoring

- Startup: watch logs for `Loaded X candles` per instrument
- Scheduled runs: look for `Running forecasts...` at trigger times
- Errors: any line with an error indicator signals a failed forecast

---

## See Also

- [[portfolio]] — ensemble and position logic called by ForecastServer
- [[vault]] — how models are loaded at startup
