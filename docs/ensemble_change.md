Plan to align TradingEnsemble with the specification:

## Implementation Plan: TradingEnsemble Specification Compliance

### 1. API signature changes

#### 1.1 `__init__()` method
- Add `target_volatility` parameter (τ in the formula)
- Add optional `instrument_weights` parameter (dict: ticker -> weight)
- Keep `r` (target risk) or clarify if it's replaced by `target_volatility`

#### 1.2 `fit()` method
- make `y` parameter requirement optional
- Add `ticker` parameter (vector/Series of ticker symbols)
- Add `volatility` parameter (vector/Series of instrument volatilities)
- Add `target_volatility` parameter (if not in `__init__`)
- Add optional `instrument_weights` parameter (if not in `__init__`)
- New signature: `fit(X, ticker, volatility, target_volatility, instrument_weights=None)`

#### 1.3 `predict()` method
- Add `ticker` parameter (required)
- Add `volatility` parameter (required)
- Remove dependency on `'annualized_volatility'` column in X
- New signature: `predict(X, ticker, volatility)`

### 2. Internal data structure changes

#### 2.1 New attributes to add
- `target_volatility_`: float (τ from formula)
- `unique_tickers_`: list/set of unique tickers from training
- `instrument_weights_`: dict mapping ticker -> weight (defaults to equal weights)
- `n_tickers_`: int (count of unique tickers)

#### 2.2 Attributes to modify
- `exposure_fractions_`: keep (already correct)
- `weights_`: keep (correlation coefficients, already normalized to sum to 1)
- `feature_names_`: keep

### 3. Formula implementation changes

#### 3.1 Current formula (incorrect)
```
prediction = (r / volatility) * Σ(w_i * X_i / √h_i)
```

#### 3.2 Required formula (from specification)
For each feature i where X_i == 1:
```
percent_forecast_i = (τ × w_i) / (σ_i × √h_i)
```
Then sum all active features:
```
final_forecast = Σ(percent_forecast_i) for all i where X_i == 1
```

#### 3.3 Implementation logic
- For each row in X:
  - For each feature i where X_i == 1:
    - Calculate `percent_forecast_i = (target_volatility × w_i) / (volatility × √h_i)`
  - Sum all `percent_forecast_i` values
  - Apply instrument weight if provided

### 4. fit() method implementation

#### 4.1 Remove correlation calculation
- Remove calculation of correlations with target `y`
- Keep correlation calculation but use a different approach (or remove if not needed)

Wait — re-reading the spec: "we then use our correlation coefficient we calculated during the fit method to scale each feature by its coefficient"

This suggests correlations are still needed, but `y` is not passed. Options:
- Calculate correlations from X features themselves
- Use a different weighting scheme
- Clarify with user

For now, assume:
- If correlations are needed without `y`, use feature-to-feature correlations or equal weights
- Or require `y` but make it optional with a fallback

#### 4.2 Add ticker tracking
```python
self.unique_tickers_ = sorted(ticker.unique())
self.n_tickers_ = len(self.unique_tickers_)
```

#### 4.3 Add instrument weighting
```python
if instrument_weights is None:
    # Equal weighting
    self.instrument_weights_ = {t: 1.0/len(self.unique_tickers_) 
                                 for t in self.unique_tickers_}
else:
    # Validate and normalize custom weights
    self.instrument_weights_ = instrument_weights
```

#### 4.4 Keep exposure fraction calculation
- Already correct: `h_i = mean(X_i)` (fraction of time feature is 1)

### 5. predict() method implementation

#### 5.1 Validation
- Require `ticker` parameter (raise error if missing)
- Require `volatility` parameter (raise error if missing)
- Validate `ticker` and `volatility` are same length as X
- Validate all tickers in `ticker` are in `self.unique_tickers_` (or allow new tickers with default weight)

#### 5.2 Formula application
```python
predictions = np.zeros(len(X))
for row_idx in range(len(X)):
    row_forecast = 0.0
    vol = volatility[row_idx]
    tick = ticker[row_idx]
    instrument_w = self.instrument_weights_.get(tick, default_weight)
    
    for feature in self.feature_names_:
        if X.iloc[row_idx][feature] == 1:  # Feature is active
            w_i = self.weights_[feature]
            h_i = self.exposure_fractions_[feature]
            percent_forecast_i = (self.target_volatility_ * w_i) / (vol * np.sqrt(h_i))
            row_forecast += percent_forecast_i
    
    predictions[row_idx] = row_forecast * instrument_w  # Apply instrument weight
```

### 6. Input validation updates

#### 6.1 Remove volatility column requirement
- Remove check for `'annualized_volatility'` in X columns
- X should only contain feature columns (binary 0/1)

#### 6.2 Add ticker/volatility validation
- Validate `ticker` is Series/array with correct length
- Validate `volatility` is numeric Series/array with positive values
- Validate `ticker` and `volatility` align with X index

### 7. Configuration save/load updates

#### 7.1 Update `save_config()`
- Add `target_volatility_` to saved config
- Add `unique_tickers_` to saved config
- Add `instrument_weights_` to saved config
- Add `n_tickers_` to saved config

#### 7.2 Update `load_config()`
- Load new attributes
- Validate structure

### 8. Documentation updates

#### 8.1 Update docstrings
- Update all method docstrings with new signatures
- Update class docstring with new parameters
- Add examples showing new usage

#### 8.2 Update README.md
- Rewrite examples with new API
- Update formula documentation
- Add ticker and volatility parameter documentation

### 9. Testing updates

#### 9.1 Update existing tests
- Modify all test cases to use new API
- Remove tests that rely on `y` parameter
- Add tests for ticker/volatility parameters

#### 9.2 Add new tests
- Test instrument weighting (equal and custom)
- Test ticker tracking
- Test formula correctness (match example output)
- Test error handling for missing ticker/volatility
- Test validation of ticker/volatility inputs

### 10. Example output verification

Based on the provided example:
- Features: f1, f2, f3, f4
- Risk-adjusted weights: [0.18237082, 0.18237082, 0.2887538, 0.34650456]
- These should match `self.weights_` after normalization

The final forecast calculation should match the example output values.

---

## Summary of major changes

1. Make `y` parameter optional in `fit()`
2. Add `ticker` and `volatility` as separate parameters (not in X)
3. Add `target_volatility` parameter
4. Implement correct formula: `(τ × w_i) / (σ_i × √h_i)` per active feature
5. Add instrument weighting support
6. Track unique tickers during fit
7. Require ticker/volatility in `predict()`
8. Update all validation logic
9. Update configuration save/load
10. Update all documentation and tests

This plan addresses the specification requirements. Should I proceed with implementation, or clarify any points first?