# Production Testing Guide

Quick guide for testing ForecastServer in production environment before deployment.

### Why It's Needed

Bias nodes require historical candles for lookback calculations.


## Testing Strategy

### Problem
- Daily forecasts run at 00:00 EST
- Weekly forecasts run Sunday 18:00 EST
- Hard to verify system works without waiting

### Solution: Manual Test Forecasts

Use `TestForecastServer` (extends `ForecastServer` with test methods):

```python
from deployment.test_forecast_server import TestForecastServer
from utils.enums import TimeFrame

server = TestForecastServer()
server.load_historical_data()

# Test all timeframes
results = server.run_test_forecast()

# Test specific timeframe
results = server.run_test_forecast(timeframe=TimeFrame.D)

server.stop()
```

**Output format:**
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

## Quick Start Using Script

```bash
# Test all timeframes
python scripts/run_manual_forecast.py

# Test specific timeframe
python scripts/run_manual_forecast.py --timeframe D

# Save results
python scripts/run_manual_forecast.py --output results.json
```

---

## Troubleshooting

### No historical data
```
❌ No historical data for EU D
```
**Fix**: Check MT5 terminal is running and logged in.

### No forecasts generated
```python
results['forecasts']['D']['count']  # Shows 0
```
**Fix**: Verify historical data loaded:
```python
print(f"Buffers: {len(server.candle_buffers)}")  # Should be 8
print(f"MLManagers: {len(server.ml_managers)}")  # Should be 8
```

### Node lookback insufficient
```
❌ RSI requires 14 candles, only 5 available
```
**Fix**: Increase candles loaded:
```python
server.lookback_candles = 150
server.load_historical_data()
```

---

## Production Checklist

**Before deployment:**
1. Run `python scripts/run_manual_forecast.py --output pre_prod_test.json`
2. Verify forecasts for all 8 ticker/timeframe combinations
3. Check no errors in results
4. Confirm predictions are reasonable values

**During production:**
1. Monitor logs for "✅ Loaded X candles" at startup
2. Watch for "🔮 Running forecasts..." at scheduled times
3. Check for "❌" error indicators
