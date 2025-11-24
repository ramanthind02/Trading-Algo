# DiversifiedEnsemble

A diversified ensemble class for combining binary trading strategies with risk-adjusted position sizing and instrument allocation.

## Quick Start

```python
from ensemble import DiversifiedEnsemble
import pandas as pd
import numpy as np

# Prepare data
X = pd.DataFrame({
    'momentum': [0, 1, 0, 1, 1],
    'mean_reversion': [1, 0, 1, 0, 1],
    'breakout': [0, 0, 1, 1, 0],
})
ticker = pd.Series(['ES', 'NQ', 'ES', 'YM', 'ES'])
volatility = pd.Series([0.15, 0.18, 0.12, 0.20, 0.16])
y = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])  # Required but ignored

# Fit and predict
ensemble = DiversifiedEnsemble(target_volatility=0.15)
ensemble.fit(X, ticker, volatility, y)
positions = ensemble.predict(X, ticker, volatility)
```

## API

### Constructor
```python
DiversifiedEnsemble(
    target_volatility=0.15,          # Target portfolio volatility 
    instrument_weights=None,         # Optional: {'ES': 0.4, 'NQ': 0.6}
    config_path=None,               # Load from saved config
    save_path=None                  # Default save path
)
```

### Methods
```python
# Training
ensemble.fit(X, ticker, volatility, y, instrument_weights=None)

# Prediction  
positions = ensemble.predict(X, ticker, volatility)

# Utilities
weights = ensemble.get_feature_importance()
stats = ensemble.get_exposure_stats()
ensemble.save_config("config.json")
ensemble.load_config("config.json")
```

## Input Requirements

- **X**: DataFrame with binary (0/1) strategy signals only
- **ticker**: Series of ticker symbols for each sample
- **volatility**: Series of positive volatility values for each sample  
- **y**: Series of target values (required but ignored in this implementation)

All parameters must have the same length.

## How It Works

1. **Diversification Weighting**: Features with lower correlation to others get higher weights
2. **Exposure Adjustment**: Accounts for time-in-market using `√h_i` where `h_i` is fraction of time active
3. **Risk Scaling**: Scales by target volatility and inverse actual volatility
4. **Instrument Allocation**: Applies equal or custom weights across ticker symbols

**Formula**: `Σ((target_volatility × w_i) / (volatility × √h_i)) × instrument_weight`

## Examples

### Custom Instrument Weights
```python
# Sector allocation
weights = {'ES': 0.4, 'CL': 0.3, 'GC': 0.3}
ensemble = DiversifiedEnsemble(target_volatility=0.12, instrument_weights=weights)
```

### Save/Load Configuration
```python
# Training
ensemble.fit(X_train, ticker_train, vol_train, y_train)
ensemble.save_config("model.json")

# Production
prod_ensemble = DiversifiedEnsemble(config_path="model.json")
positions = prod_ensemble.predict(X_new, ticker_new, vol_new)
```

### Error Handling
- Raises `ValueError` for mismatched input lengths
- Raises `ValueError` for unseen tickers in predict()
- Raises `ValueError` if volatility column found in X (pass separately)
- Raises `ValueError` for non-positive volatility values

## Notes

- Target `y` is required for API consistency but ignored in weight calculation
- Uses intra-feature correlations instead of target correlations for diversification
- Equal instrument weighting by default across unique tickers
- Minimum 10 samples required for fitting
