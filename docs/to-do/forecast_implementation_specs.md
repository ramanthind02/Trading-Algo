# Implementation Specifications

> **🔨 This Document**: Concrete build tasks, method signatures, unit tests, and step-by-step instructions  
> **📖 For Context**: See [forecast_specs.md](./forecast_specs.md) for design rationale, formulas, and examples

**Version**: 1.0.0  
**Date**: 2025-01-08  
**Status**: Ready for Implementation

**Purpose**: This document contains the concrete, actionable implementation tasks for building the forecast generation and position sizing system.

---

## Table of Contents

1. [Phase 1: Refactor Ensemble](#phase-1-refactor-ensemble)
2. [Phase 2: Enhance Portfolio](#phase-2-enhance-portfolio)
3. [Phase 3: Create Execution Layer](#phase-3-create-execution-layer)
4. [Phase 4: Create Volatility Utilities](#phase-4-create-volatility-utilities)
5. [Phase 5: Integration & Testing](#phase-5-integration--testing)
6. [Phase 6: DM Fitting Utility](#phase-6-dm-fitting-utility)

---

## Phase 1: Refactor Ensemble

**Goal**: Simplify ensemble to only combine signals using inverse correlation weights. Remove all volatility scaling, diversification, and position sizing logic.

**Estimated Effort**: 2-3 hours

**⚠️ BREAKING CHANGE**: This phase changes the `predict()` return type from `np.ndarray` to `pd.DataFrame`. All calling code (including Portfolio) must be updated to handle DataFrame output.

### Files to Modify

- `ensemble/diversified_ensemble.py`
- `tests/test_diversified_ensemble.py`

### Current State Analysis

**What to Remove**:
- ❌ `volatility` parameter from `predict()`
- ❌ Division by instrument volatility
- ❌ Multiplication by instrument weights
- ❌ `target_volatility` usage
- ❌ `forecast_scalar` parameter
- ❌ Diversification multiplier (FDM) calculation and application
- ❌ Forecast capping logic

**What to Keep**:
- ✅ Inverse correlation weight calculation during `fit()`
- ✅ Binary signal combination using weights
- ✅ Exposure fraction tracking

**What to Add**:
- ✅ Return `exposure_fraction` column in output DataFrame

### Method Signature Changes

#### Before:
```python
def predict(
    self,
    X: pd.DataFrame,
    ticker: pd.Series,
    volatility: pd.Series,
    normalization_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Returns: DataFrame(['ticker', 'combined_forecast'])"""
```

#### After:
```python
def predict(
    self,
    X: pd.DataFrame,
    ticker: pd.Series,
    normalization_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Returns: DataFrame(['ticker', 'forecast_score', 'exposure_fraction'])
    
    Where:
    - forecast_score: Raw combined signal in [0, 1] range
    - exposure_fraction: Average h across active models
    
    Note: This is a BREAKING CHANGE from current implementation which returns np.ndarray.
    All calling code must be updated to handle DataFrame output.
    """
```

### Implementation Steps

1. **Remove volatility parameter**:
   - Remove `volatility` from `predict()` signature
   - Remove any volatility-related calculations

2. **Simplify forecast calculation**:
   ```python
   # Calculate weighted sum of binary signals for each row
   forecast_raw = []
   for idx in range(len(binary_df)):
       row_forecast = 0.0
       for model_name in binary_df.columns:
           signal = binary_df.iloc[idx][model_name]
           if signal == 1:  # Only add weight if signal is active
               row_forecast += self.weights_[model_name]
       forecast_raw.append(row_forecast)
   
   forecast_raw = np.array(forecast_raw)
   ```
   
   **Remove these steps** (they move to Portfolio layer):
   - ❌ Division by volatility
   - ❌ Multiplication by forecast scalar
   - ❌ Multiplication by instrument weights
   - ❌ Application of FDM
   - ❌ Forecast capping

3. **Add exposure fraction calculation**:
   ```python
   # For each row, calculate average exposure of active models
   exposure_fractions = []
   for idx in range(len(binary_df)):
       # Get active models for this row
       active_model_names = [
           model_name for model_name in binary_df.columns 
           if binary_df.iloc[idx][model_name] == 1
       ]
       
       # Calculate average exposure if any models are active
       if active_model_names:
           avg_exposure = np.mean([
               self.model_exposure_fractions_[name] 
               for name in active_model_names
           ])
       else:
           avg_exposure = 0.0
       
       exposure_fractions.append(avg_exposure)
   
   # Convert to numpy array
   exposure_fraction = np.array(exposure_fractions)
   ```

4. **Update output DataFrame**:
   ```python
   # Return DataFrame instead of numpy array (BREAKING CHANGE)
   result = pd.DataFrame({
       'ticker': ticker.values if isinstance(ticker, pd.Series) else ticker,
       'forecast_score': forecast_raw,  # [0, 1] range
       'exposure_fraction': exposure_fraction  # [0, 1] range
   })
   
   return result
   ```

5. **Store model exposure fractions during fit**:
   ```python
   # In fit(), store exposure fraction for each base model
   # Note: base_models is a dict[str, BaseModel] in current implementation
   self.model_exposure_fractions_ = {
       model_name: 1.0 / base_model.n_bins 
       for model_name, base_model in self.base_models.items()
   }
   ```

### Unit Tests

Create tests in `tests/test_diversified_ensemble.py`:

```python
def test_raw_forecast_output_range():
    """Test that raw forecast is in [0, 1] range"""
    ensemble = DiversifiedEnsemble(models=[...])
    ensemble.fit(X_train, y_train, ticker_train)
    
    predictions = ensemble.predict(X_test, ticker_test)
    
    assert 'forecast_score' in predictions.columns
    assert 'exposure_fraction' in predictions.columns
    assert predictions['forecast_score'].min() >= 0.0
    assert predictions['forecast_score'].max() <= 1.0

def test_weights_sum_to_one():
    """Test that inverse correlation weights sum to 1.0"""
    ensemble = DiversifiedEnsemble(models=[...])
    ensemble.fit(X_train, y_train, ticker_train)
    
    weights = ensemble.diversification_weights_
    assert np.isclose(weights.sum(), 1.0)

def test_all_models_active():
    """Test forecast when all models are active"""
    # If all models return 1 and weights sum to 1
    # Then forecast should be 1.0
    ensemble = DiversifiedEnsemble(models=[...])
    ensemble.fit(X_train, y_train, ticker_train)
    
    # Create test data where all models will be active
    predictions = ensemble.predict(X_all_active, ticker_test)
    
    assert predictions['forecast_score'].iloc[0] == 1.0

def test_no_models_active():
    """Test forecast when no models are active"""
    # If no models return 1, forecast should be 0.0
    ensemble = DiversifiedEnsemble(models=[...])
    ensemble.fit(X_train, y_train, ticker_train)
    
    predictions = ensemble.predict(X_no_active, ticker_test)
    
    assert predictions['forecast_score'].iloc[0] == 0.0

def test_output_structure():
    """Test that output is DataFrame with correct structure"""
    ensemble = DiversifiedEnsemble(models=[...])
    ensemble.fit(X_train, y_train, ticker_train)
    
    predictions = ensemble.predict(X_test, ticker_test)
    
    # Check it's a DataFrame
    assert isinstance(predictions, pd.DataFrame)
    
    # Check required columns
    assert 'ticker' in predictions.columns
    assert 'forecast_score' in predictions.columns
    assert 'exposure_fraction' in predictions.columns
    
    # Check data types
    assert predictions['forecast_score'].dtype in [np.float64, np.float32]
    assert predictions['exposure_fraction'].dtype in [np.float64, np.float32]
```

### Migration Notes

**Breaking Changes**:
1. `predict()` now returns `pd.DataFrame` instead of `np.ndarray`
2. Output structure changed from single array to DataFrame with 3 columns
3. `volatility` parameter removed from `predict()`

**Required Updates to Calling Code**:
```python
# OLD (before Phase 1):
predictions = ensemble.predict(X, ticker, volatility)
# predictions is np.ndarray

# NEW (after Phase 1):
predictions = ensemble.predict(X, ticker)
# predictions is pd.DataFrame with columns: ['ticker', 'forecast_score', 'exposure_fraction']

# Access values:
forecast_scores = predictions['forecast_score'].values
exposure_fractions = predictions['exposure_fraction'].values
tickers = predictions['ticker'].values
```

---

## Phase 2: Enhance Portfolio

**Goal**: Make Portfolio the central risk management hub with volatility scaling, DM application, and position capping.

**Estimated Effort**: 6-8 hours

**Required Imports**:
```python
from typing import Union, Dict, Optional, List
import pandas as pd
import numpy as np
```

### Files to Modify

- `ensemble/portfolio.py`
- `tests/test_portfolio.py`

### New Parameters

Add to `__init__`:

```python
def __init__(
    self,
    ensembles: List[BaseEnsemble],  # Changed from dict[TimeFrame, list[BaseEnsemble]]
    trading_timeframe: TimeFrame,  # NEW: Explicit trading timeframe
    target_volatility: float = 0.20,  # NEW
    dm: float = 1.0,  # NEW
    max_position_pct: float = 2.0,  # NEW
    instrument_weights: Optional[dict[str, float]] = None  # NEW
):
    """
    Create a Portfolio for a SINGLE trading timeframe.
    
    Parameters
    ----------
    ensembles : List[BaseEnsemble]
        List of ensemble models (no longer organized by timeframe)
    trading_timeframe : TimeFrame
        The timeframe this portfolio trades on (e.g., TimeFrame.D for daily trading)
    target_volatility : float, default=0.20
        Target annual portfolio volatility (e.g., 0.20 = 20%)
    dm : float, default=1.0
        Diversification Multiplier. Start at 1.0, fit from backtests.
        Typical range: 1.0-3.0
    max_position_pct : float, default=2.0
        Maximum position size per instrument (as decimal, e.g., 2.0 = 200%)
    instrument_weights : dict[str, float], optional
        Weight for each instrument. If None, equal weight.
        Weights should sum to 1.0
    """
```

### Method Signature Changes

#### Before:
```python
def predict(
    self,
    X: pd.DataFrame,
    ticker: pd.Series,
    timeframe: TimeFrame,
    normalization_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Returns: DataFrame(['ticker', 'timeframe', 'forecast'])"""
```

#### After:
```python
def predict(
    self,
    X: pd.DataFrame,
    ticker: pd.Series,
    volatility: Union[pd.Series, Dict[str, float]],  # NEW: Blended volatility per instrument
    normalization_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Returns: DataFrame(['ticker', 'forecast_score', 'position_fraction'])
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    ticker : pd.Series
        Instrument ticker symbols
    volatility : pd.Series or dict
        Blended volatility per instrument (annualized).
        If Series: Must have ticker symbols as index (will be converted to dict).
        If dict: Maps ticker symbol (str) -> volatility value (float).
        Formula: 0.70 * EWMA-32 + 0.30 * 10-year average
    normalization_data : pd.DataFrame, optional
        For ensemble base models
    
    Returns
    -------
    pd.DataFrame with columns:
        - ticker: Instrument identifier
        - forecast_score: Average raw forecast across ensembles
        - position_fraction: Fraction of capital to allocate
    
    Notes
    -----
    timeframe parameter removed - Portfolio now knows its trading timeframe
    """
```

### Implementation Steps

#### Step 1: Average Ensemble Forecasts

```python
def _average_ensemble_forecasts(
    self,
    ensemble_predictions: List[pd.DataFrame]
) -> pd.DataFrame:
    """
    Average raw forecasts across ensembles for each instrument.
    
    Parameters
    ----------
    ensemble_predictions : list[pd.DataFrame]
        List of predictions from each ensemble.
        Each DataFrame has columns: ['ticker', 'forecast_score', 'exposure_fraction']
        (Note: Ensembles now return DataFrames, not numpy arrays)
    
    Returns
    -------
    pd.DataFrame with columns: ['ticker', 'forecast_avg', 'exposure_avg']
    """
    if not ensemble_predictions:
        return pd.DataFrame(columns=['ticker', 'forecast_avg', 'exposure_avg'])
    
    # Concatenate all ensemble predictions
    all_forecasts = pd.concat(ensemble_predictions, ignore_index=True)
    
    # Group by ticker and average
    averaged = all_forecasts.groupby('ticker', as_index=False).agg({
        'forecast_score': 'mean',
        'exposure_fraction': 'mean'
    })
    
    averaged.rename(columns={
        'forecast_score': 'forecast_avg',
        'exposure_fraction': 'exposure_avg'
    }, inplace=True)
    
    return averaged
```

#### Step 2: Calculate Position Fractions

```python
from typing import Union, Dict

def _calculate_position_fractions(
    self,
    forecasts: pd.DataFrame,
    volatility: Union[pd.Series, Dict[str, float]]
) -> pd.DataFrame:
    """
    Convert forecasts to position fractions using volatility scaling.
    
    Parameters
    ----------
    forecasts : pd.DataFrame
        Columns: ['ticker', 'forecast_avg', 'exposure_avg']
    volatility : pd.Series or dict
        Blended volatility per ticker (annualized).
        If Series, must have ticker symbols as index.
        If dict, maps ticker -> volatility value.
    
    Returns
    -------
    pd.DataFrame with columns: ['ticker', 'forecast_avg', 'position_fraction']
    """
    # Merge volatility data
    df = forecasts.copy()
    
    # Convert volatility to dict for mapping if needed
    vol_dict = volatility if isinstance(volatility, dict) else volatility.to_dict()
    df['volatility'] = df['ticker'].map(vol_dict)
    
    # Step 1: Use raw forecast directly (no scaling)
    df['f_norm'] = df['forecast_avg']
    
    # Step 2: Calculate volatility-adjusted position
    # position_vol = f_norm * (tau / (sigma_blended * sqrt(h)))
    df['position_vol'] = (
        df['f_norm'] * 
        (self.target_volatility / (df['volatility'] * np.sqrt(df['exposure_avg'])))
    )
    
    # Step 3: Apply diversification multiplier
    df['position_dm'] = df['position_vol'] * self.dm
    
    # Step 4: Apply instrument weights
    if self.instrument_weights is None:
        # Equal weight per unique instrument
        unique_instruments = df['ticker'].nunique()
        instrument_weight = 1.0 / unique_instruments if unique_instruments > 0 else 0.0
        df['instrument_weight'] = instrument_weight
        df['position_weighted'] = df['position_dm'] * df['instrument_weight']
    else:
        df['instrument_weight'] = df['ticker'].map(self.instrument_weights)
        df['position_weighted'] = df['position_dm'] * df['instrument_weight']
    
    # Step 5: Cap position
    df['position_fraction'] = df['position_weighted'].clip(upper=self.max_position_pct)
    
    # Return only necessary columns
    return df[['ticker', 'forecast_avg', 'position_fraction']]
```

#### Step 3: Update predict() method

```python
def predict(
    self,
    X: pd.DataFrame,
    ticker: pd.Series,
    volatility: Union[pd.Series, Dict[str, float]],
    normalization_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Main prediction method"""
    
    # Validate and convert volatility to consistent format
    if isinstance(volatility, pd.Series):
        # Series must have ticker symbols as index
        vol_dict = volatility.to_dict()
    elif isinstance(volatility, dict):
        vol_dict = volatility
    else:
        raise ValueError(
            f"volatility must be pd.Series or dict, got {type(volatility)}"
        )
    
    # Validate all tickers have volatility values
    missing_vols = set(ticker.unique()) - set(vol_dict.keys())
    if missing_vols:
        raise ValueError(
            f"Missing volatility values for tickers: {missing_vols}"
        )
    
    # Get predictions from each ensemble (now a simple list, not dict)
    # Each ensemble.predict() now returns DataFrame with columns:
    # ['ticker', 'forecast_score', 'exposure_fraction']
    ensemble_predictions = []
    for ensemble in self.ensembles:
        pred = ensemble.predict(X, ticker, normalization_data)
        # Validate output structure
        if not isinstance(pred, pd.DataFrame):
            raise ValueError(f"Ensemble returned {type(pred)}, expected pd.DataFrame")
        if not all(col in pred.columns for col in ['ticker', 'forecast_score', 'exposure_fraction']):
            raise ValueError(f"Ensemble output missing required columns")
        ensemble_predictions.append(pred)
    
    # Average forecasts across ensembles
    averaged_forecasts = self._average_ensemble_forecasts(ensemble_predictions)
    
    # Calculate position fractions (pass vol_dict for consistent format)
    positions = self._calculate_position_fractions(averaged_forecasts, vol_dict)
    
    # Rename for output
    positions.rename(columns={'forecast_avg': 'forecast_score'}, inplace=True)
    
    return positions[['ticker', 'forecast_score', 'position_fraction']]
```

### Unit Tests (Buy/Hold Scenarios)

Create tests in `tests/test_portfolio.py`:

```python
def test_buy_hold_baseline():
    """Test perfect buy/hold: target vol = instrument vol, h=1, DM=1"""
    portfolio = Portfolio(
        ensembles=[mock_ensemble],
        trading_timeframe=TimeFrame.DAILY,
        target_volatility=0.20,
        dm=1.0,
        max_position_pct=2.0
    )
    
    # Mock ensemble returns forecast=1.0, exposure=1.0
    mock_ensemble.predict.return_value = pd.DataFrame({
        'ticker': ['TEST'],
        'forecast_score': [1.0],
        'exposure_fraction': [1.0]
    })
    
    # Volatility matches target
    volatility = pd.Series({'TEST': 0.20})
    
    result = portfolio.predict(X_test, ticker_test, volatility)
    
    # Expected: 1.0 * (0.20 / (0.20 * sqrt(1.0))) * 1.0 * 1.0 = 1.0
    assert np.isclose(result['position_fraction'].iloc[0], 1.0)

def test_buy_hold_high_volatility():
    """Test buy/hold with 2x volatility instrument"""
    portfolio = Portfolio(
        ensembles=[mock_ensemble],
        trading_timeframe=TimeFrame.DAILY,
        target_volatility=0.20,
        dm=1.0,
        max_position_pct=2.0
    )
    
    mock_ensemble.predict.return_value = pd.DataFrame({
        'ticker': ['TEST'],
        'forecast_score': [1.0],
        'exposure_fraction': [1.0]
    })
    
    # Volatility is 2x target
    volatility = pd.Series({'TEST': 0.40})
    
    result = portfolio.predict(X_test, ticker_test, volatility)
    
    # Expected: 1.0 * (0.20 / (0.40 * sqrt(1.0))) * 1.0 * 1.0 = 0.5
    assert np.isclose(result['position_fraction'].iloc[0], 0.5)

def test_buy_hold_low_volatility():
    """Test buy/hold with 0.5x volatility instrument (leverage)"""
    portfolio = Portfolio(
        ensembles=[mock_ensemble],
        trading_timeframe=TimeFrame.DAILY,
        target_volatility=0.20,
        dm=1.0,
        max_position_pct=2.0
    )
    
    mock_ensemble.predict.return_value = pd.DataFrame({
        'ticker': ['TEST'],
        'forecast_score': [1.0],
        'exposure_fraction': [1.0]
    })
    
    # Volatility is 0.5x target
    volatility = pd.Series({'TEST': 0.10})
    
    result = portfolio.predict(X_test, ticker_test, volatility)
    
    # Expected: 1.0 * (0.20 / (0.10 * sqrt(1.0))) * 1.0 * 1.0 = 2.0
    assert np.isclose(result['position_fraction'].iloc[0], 2.0)

def test_diversification_multiplier():
    """Test DM scaling effect"""
    portfolio = Portfolio(
        ensembles=[mock_ensemble],
        trading_timeframe=TimeFrame.DAILY,
        target_volatility=0.20,
        dm=2.0,  # 2x scaling
        max_position_pct=2.0
    )
    
    mock_ensemble.predict.return_value = pd.DataFrame({
        'ticker': ['TEST'],
        'forecast_score': [1.0],
        'exposure_fraction': [1.0]
    })
    
    volatility = pd.Series({'TEST': 0.20})
    
    result = portfolio.predict(X_test, ticker_test, volatility)
    
    # Expected: 1.0 * (0.20 / (0.20 * sqrt(1.0))) * 2.0 * 1.0 = 2.0
    assert np.isclose(result['position_fraction'].iloc[0], 2.0)

def test_sparse_signals():
    """Test sparse signals with low exposure fraction"""
    portfolio = Portfolio(
        ensembles=[mock_ensemble],
        trading_timeframe=TimeFrame.DAILY,
        target_volatility=0.20,
        dm=1.0,
        max_position_pct=3.5
    )
    
    mock_ensemble.predict.return_value = pd.DataFrame({
        'ticker': ['TEST'],
        'forecast_score': [1.0],
        'exposure_fraction': [0.1]  # Only in market 10% of time
    })
    
    volatility = pd.Series({'TEST': 0.20})
    
    result = portfolio.predict(X_test, ticker_test, volatility)
    
    # Expected: 1.0 * (0.20 / (0.20 * sqrt(0.1))) * 1.0 * 1.0
    # = 1.0 * (0.20 / 0.0632) = 3.16
    expected = 1.0 * (0.20 / (0.20 * np.sqrt(0.1)))
    assert np.isclose(result['position_fraction'].iloc[0], expected, rtol=0.01)

def test_position_capping():
    """Test that position is capped at max_position_pct"""
    portfolio = Portfolio(
        ensembles=[mock_ensemble],
        trading_timeframe=TimeFrame.DAILY,
        target_volatility=0.20,
        dm=1.0,
        max_position_pct=2.0  # Cap at 2.0
    )
    
    mock_ensemble.predict.return_value = pd.DataFrame({
        'ticker': ['TEST'],
        'forecast_score': [1.0],
        'exposure_fraction': [1.0]
    })
    
    # Very low volatility would give 4.0 uncapped
    volatility = pd.Series({'TEST': 0.05})
    
    result = portfolio.predict(X_test, ticker_test, volatility)
    
    # Should be capped at 2.0
    assert np.isclose(result['position_fraction'].iloc[0], 2.0)

def test_multiple_instruments_equal_weight():
    """Test equal weighting across multiple instruments"""
    portfolio = Portfolio(
        ensembles=[mock_ensemble],
        trading_timeframe=TimeFrame.DAILY,
        target_volatility=0.20,
        dm=1.0,
        max_position_pct=2.0
    )
    
    # Two instruments with same forecast/exposure
    mock_ensemble.predict.return_value = pd.DataFrame({
        'ticker': ['TEST1', 'TEST2'],
        'forecast_score': [1.0, 1.0],
        'exposure_fraction': [1.0, 1.0]
    })
    
    volatility = pd.Series({'TEST1': 0.20, 'TEST2': 0.20})
    
    result = portfolio.predict(X_test, ticker_test, volatility)
    
    # Each should get 0.5 weight (1/2 instruments)
    # Position = 1.0 * (0.20 / 0.20) * 1.0 * 0.5 = 0.5
    assert len(result) == 2
    assert np.allclose(result['position_fraction'].values, [0.5, 0.5])
```

---

## Phase 3: Create Execution Layer

**Goal**: Create a new execution layer that converts position fractions to tradeable contracts.

**Estimated Effort**: 3-4 hours

### Files to Create

- `execution/__init__.py`
- `execution/position_sizer.py`
- `tests/test_position_sizer.py`

### Data Classes

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class ContractSpec:
    """Specification for a futures contract"""
    ticker: str
    price: float          # Current market price
    multiplier: float     # Points to dollars (e.g., $20 for NQ)
    fx_rate: float = 1.0  # FX conversion (default USD/USD = 1.0)
    min_tick: float = 0.25  # Minimum price increment

@dataclass(frozen=True)
class Position:
    """Calculated position for an instrument"""
    ticker: str
    forecast_score: float
    position_fraction: float
    target_dollars: float
    contracts: int
    notional_value: float
    notional_pct: float
```

### PositionSizer Class

```python
from enum import Enum
from typing import Dict
import pandas as pd
import numpy as np

class RoundingMethod(Enum):
    ROUND = 'round'
    FLOOR = 'floor'
    CEILING = 'ceiling'

class PositionSizer:
    """Convert position fractions to tradeable contracts"""
    
    def __init__(
        self,
        capital: float,
        contract_specs: Dict[str, ContractSpec],
        rounding_method: RoundingMethod = RoundingMethod.ROUND
    ):
        """
        Parameters
        ----------
        capital : float
            Account capital in base currency (USD)
        contract_specs : dict[str, ContractSpec]
            Contract specifications per ticker
        rounding_method : RoundingMethod
            How to round fractional contracts
        """
        self.capital = capital
        self.contract_specs = contract_specs
        self.rounding_method = rounding_method
    
    def calculate_positions(
        self,
        position_fractions: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Convert position fractions to contracts.
        
        Parameters
        ----------
        position_fractions : pd.DataFrame
            Columns: ['ticker', 'forecast_score', 'position_fraction']
        
        Returns
        -------
        pd.DataFrame with columns:
            - ticker
            - forecast_score
            - position_fraction
            - target_dollars
            - contracts
            - notional_value
            - notional_pct
        """
        results = []
        
        for _, row in position_fractions.iterrows():
            ticker = row['ticker']
            forecast_score = row['forecast_score']
            position_fraction = row['position_fraction']
            
            # Get contract spec
            spec = self.contract_specs.get(ticker)
            if spec is None:
                raise ValueError(f"No contract spec for {ticker}")
            
            # Calculate target dollar allocation
            target_dollars = position_fraction * self.capital
            
            # Calculate contract value
            contract_value = spec.price * spec.multiplier * spec.fx_rate
            
            # Calculate raw contracts
            contracts_raw = target_dollars / contract_value
            
            # Round contracts
            contracts = self._round_contracts(contracts_raw)
            
            # Calculate actual notional
            notional_value = contracts * contract_value
            notional_pct = notional_value / self.capital if self.capital > 0 else 0.0
            
            results.append({
                'ticker': ticker,
                'forecast_score': forecast_score,
                'position_fraction': position_fraction,
                'target_dollars': target_dollars,
                'contracts': contracts,
                'notional_value': notional_value,
                'notional_pct': notional_pct
            })
        
        return pd.DataFrame(results)
    
    def _round_contracts(self, contracts_raw: float) -> int:
        """Round fractional contracts based on rounding method"""
        if self.rounding_method == RoundingMethod.ROUND:
            return int(round(contracts_raw))
        elif self.rounding_method == RoundingMethod.FLOOR:
            return int(np.floor(contracts_raw))
        elif self.rounding_method == RoundingMethod.CEILING:
            return int(np.ceil(contracts_raw))
        else:
            raise ValueError(f"Unknown rounding method: {self.rounding_method}")
```

### Unit Tests

```python
def test_contract_calculation():
    """Test basic contract calculation"""
    specs = {
        'NQ': ContractSpec(ticker='NQ', price=16000, multiplier=20, fx_rate=1.0)
    }
    sizer = PositionSizer(capital=1_000_000, contract_specs=specs)
    
    positions = pd.DataFrame({
        'ticker': ['NQ'],
        'forecast_score': [0.5],
        'position_fraction': [0.6]
    })
    
    result = sizer.calculate_positions(positions)
    
    # Target: $600,000
    # Contract value: 16,000 * 20 = $320,000
    # Raw contracts: 600,000 / 320,000 = 1.875
    # Rounded: 2
    # Notional: 2 * 320,000 = $640,000
    assert result['contracts'].iloc[0] == 2
    assert result['notional_value'].iloc[0] == 640_000
    assert np.isclose(result['notional_pct'].iloc[0], 0.64)

def test_floor_rounding():
    """Test floor rounding method"""
    specs = {'TEST': ContractSpec(ticker='TEST', price=100, multiplier=100)}
    sizer = PositionSizer(
        capital=100_000,
        contract_specs=specs,
        rounding_method=RoundingMethod.FLOOR
    )
    
    positions = pd.DataFrame({
        'ticker': ['TEST'],
        'forecast_score': [1.0],
        'position_fraction': [0.28]  # Would give 2.8 contracts
    })
    
    result = sizer.calculate_positions(positions)
    
    # Floor(2.8) = 2
    assert result['contracts'].iloc[0] == 2

def test_ceiling_rounding():
    """Test ceiling rounding method"""
    specs = {'TEST': ContractSpec(ticker='TEST', price=100, multiplier=100)}
    sizer = PositionSizer(
        capital=100_000,
        contract_specs=specs,
        rounding_method=RoundingMethod.CEILING
    )
    
    positions = pd.DataFrame({
        'ticker': ['TEST'],
        'forecast_score': [1.0],
        'position_fraction': [0.22]  # Would give 2.2 contracts
    })
    
    result = sizer.calculate_positions(positions)
    
    # Ceiling(2.2) = 3
    assert result['contracts'].iloc[0] == 3

def test_zero_contracts():
    """Test handling of zero contracts (position too small)"""
    specs = {'TEST': ContractSpec(ticker='TEST', price=10000, multiplier=100)}
    sizer = PositionSizer(capital=100_000, contract_specs=specs)
    
    positions = pd.DataFrame({
        'ticker': ['TEST'],
        'forecast_score': [0.1],
        'position_fraction': [0.001]  # Very small position
    })
    
    result = sizer.calculate_positions(positions)
    
    # Should round to 0
    assert result['contracts'].iloc[0] == 0
    assert result['notional_value'].iloc[0] == 0.0
```

---

## Phase 4: Create Volatility Utilities

**Goal**: Create utilities for calculating blended volatility (70% EWMA-32 + 30% 10-year average).

**Estimated Effort**: 3-4 hours

### Files to Create

- `utils/volatility.py`
- `tests/test_volatility.py`

### BlendedVolatility Class

```python
import pandas as pd
import numpy as np
from typing import Optional

class BlendedVolatility:
    """
    Calculate Carver's blended volatility estimate.
    
    Formula: 0.70 * EWMA-32 + 0.30 * 10-year average
    """
    
    def __init__(
        self,
        short_span: int = 32,
        long_window: int = 2520,  # ~10 years of daily data
        short_weight: float = 0.70,
        annualization_factor: float = 252  # Trading days per year
    ):
        """
        Parameters
        ----------
        short_span : int, default=32
            Span for EWMA (short-run estimate)
        long_window : int, default=2520
            Window for rolling average (long-run estimate)
            Default is ~10 years of daily data
        short_weight : float, default=0.70
            Weight for short-run estimate (Carver uses 0.70)
        annualization_factor : float, default=252
            Factor to annualize daily volatility
        """
        self.short_span = short_span
        self.long_window = long_window
        self.short_weight = short_weight
        self.long_weight = 1.0 - short_weight
        self.annualization_factor = annualization_factor
    
    def calculate(
        self,
        returns: pd.Series,
        min_periods_short: Optional[int] = None,
        min_periods_long: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate blended volatility from returns.
        
        Parameters
        ----------
        returns : pd.Series
            Daily returns (not annualized)
        min_periods_short : int, optional
            Minimum periods for EWMA. Defaults to short_span
        min_periods_long : int, optional
            Minimum periods for rolling average. Defaults to long_window
        
        Returns
        -------
        pd.Series
            Blended annualized volatility
        """
        if min_periods_short is None:
            min_periods_short = self.short_span
        if min_periods_long is None:
            min_periods_long = self.long_window
        
        # Calculate short-run estimate (EWMA)
        short_vol = returns.ewm(
            span=self.short_span,
            min_periods=min_periods_short
        ).std()
        
        # Calculate long-run estimate (rolling average)
        long_vol = returns.rolling(
            window=self.long_window,
            min_periods=min_periods_long
        ).std()
        
        # Blend
        blended_vol = (
            self.short_weight * short_vol +
            self.long_weight * long_vol
        )
        
        # Annualize
        annualized_vol = blended_vol * np.sqrt(self.annualization_factor)
        
        return annualized_vol
    
    def calculate_by_ticker(
        self,
        returns_df: pd.DataFrame,
        ticker_col: str = 'ticker',
        return_col: str = 'returns',
        date_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate blended volatility for multiple tickers.
        
        Parameters
        ----------
        returns_df : pd.DataFrame
            DataFrame with returns data
        ticker_col : str, default='ticker'
            Column name for ticker symbols
        return_col : str, default='returns'
            Column name for returns
        date_col : str, optional
            Column name for dates (for sorting)
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: [ticker_col, date_col (if provided), 'blended_volatility']
        """
        if date_col:
            returns_df = returns_df.sort_values([ticker_col, date_col])
        
        results = []
        
        for ticker, group in returns_df.groupby(ticker_col):
            returns_series = group[return_col]
            blended_vol = self.calculate(returns_series)
            
            result_df = pd.DataFrame({
                ticker_col: ticker,
                'blended_volatility': blended_vol
            })
            
            if date_col:
                result_df[date_col] = group[date_col].values
            
            results.append(result_df)
        
        return pd.concat(results, ignore_index=True)
```

### Unit Tests

```python
def test_blended_volatility_basic():
    """Test basic blended volatility calculation"""
    # Create synthetic returns with known volatility
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 0.01, 1000))
    
    calculator = BlendedVolatility()
    blended_vol = calculator.calculate(returns)
    
    # Should be annualized and non-null after sufficient periods
    assert not blended_vol.iloc[-1] == np.nan
    assert blended_vol.iloc[-1] > 0
    # Annual vol should be roughly 0.01 * sqrt(252) ≈ 0.159
    assert 0.10 < blended_vol.iloc[-1] < 0.25

def test_blended_volatility_weights():
    """Test that weights are applied correctly"""
    # Create returns where short-run vol differs from long-run vol
    returns = pd.Series(np.random.normal(0, 0.01, 3000))
    # Add recent spike in volatility
    returns.iloc[-100:] = np.random.normal(0, 0.03, 100)
    
    calculator = BlendedVolatility()
    blended_vol = calculator.calculate(returns)
    
    # Short-run should react faster to spike
    # Long-run should be more stable
    # Blend should be in between
    assert blended_vol.iloc[-1] > blended_vol.iloc[-200]

def test_blended_volatility_by_ticker():
    """Test multi-ticker calculation"""
    # Create returns for multiple tickers
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    data = []
    
    for ticker in ['NQ', 'ES', 'GC']:
        returns = np.random.normal(0, 0.01, 500)
        for date, ret in zip(dates, returns):
            data.append({
                'date': date,
                'ticker': ticker,
                'returns': ret
            })
    
    df = pd.DataFrame(data)
    
    calculator = BlendedVolatility()
    result = calculator.calculate_by_ticker(
        df,
        ticker_col='ticker',
        return_col='returns',
        date_col='date'
    )
    
    # Should have volatility for all tickers
    assert set(result['ticker'].unique()) == {'NQ', 'ES', 'GC'}
    assert 'blended_volatility' in result.columns
    
    # Check one ticker's last volatility
    nq_vol = result[result['ticker'] == 'NQ']['blended_volatility'].iloc[-1]
    assert 0.05 < nq_vol < 0.30
```

---

## Phase 5: Integration & Testing

**Goal**: Wire everything together and create end-to-end tests.

**Estimated Effort**: 4-6 hours

### Files to Create

- `scripts/demo_position_sizing.py`
- `tests/test_e2e_position_sizing.py`

### Demo Script

```python
"""
Demo script showing end-to-end position sizing workflow.
"""
import pandas as pd
import numpy as np
from ensemble.portfolio import Portfolio
from ensemble.diversified_ensemble import DiversifiedEnsemble
from execution.position_sizer import PositionSizer, ContractSpec
from utils.volatility import BlendedVolatility
from feature_selection.base_models.base_model import BaseModel
from data_processing.time_frame import TimeFrame

# Step 1: Load data
print("Loading data...")
# (Assume data loading code here)

# Step 2: Calculate blended volatility
print("Calculating blended volatility...")
vol_calculator = BlendedVolatility()
volatility_df = vol_calculator.calculate_by_ticker(
    returns_df,
    ticker_col='ticker',
    return_col='returns',
    date_col='date'
)

# Get latest volatility for each ticker
latest_volatility = (
    volatility_df
    .groupby('ticker')
    .last()
    ['blended_volatility']
)

# Step 3: Create ensembles
print("Creating ensembles...")
# Mean reversion ensemble
mr_models = [
    BaseModel(n_bins=10, strategy='long') for _ in range(10)
]
mr_ensemble = DiversifiedEnsemble(models=mr_models)
mr_ensemble.fit(X_train_mr, y_train_mr, ticker_train_mr)

# Momentum ensemble
mom_models = [
    BaseModel(n_bins=10, strategy='long') for _ in range(10)
]
mom_ensemble = DiversifiedEnsemble(models=mom_models)
mom_ensemble.fit(X_train_mom, y_train_mom, ticker_train_mom)

# Step 4: Create portfolio
print("Creating portfolio...")
portfolio = Portfolio(
    ensembles=[mr_ensemble, mom_ensemble],
    trading_timeframe=TimeFrame.DAILY,
    target_volatility=0.20,  # 20% annual target
    dm=1.0,  # Start conservative, fit later
    max_position_pct=2.0
)

# Step 5: Generate position fractions
print("Generating position fractions...")
position_fractions = portfolio.predict(
    X_test,
    ticker_test,
    latest_volatility,
    TimeFrame.DAILY
)

print("\nPosition Fractions:")
print(position_fractions)

# Step 6: Convert to contracts
print("\nConverting to contracts...")
contract_specs = {
    'NQ': ContractSpec(ticker='NQ', price=16000, multiplier=20),
    'ES': ContractSpec(ticker='ES', price=4800, multiplier=50),
    'YM': ContractSpec(ticker='YM', price=38000, multiplier=5),
    'GC': ContractSpec(ticker='GC', price=2000, multiplier=100),
    'CL': ContractSpec(ticker='CL', price=75, multiplier=1000)
}

sizer = PositionSizer(
    capital=1_000_000,
    contract_specs=contract_specs
)

positions = sizer.calculate_positions(position_fractions)

print("\nFinal Positions:")
print(positions)

# Step 7: Summary statistics
print("\n=== Portfolio Summary ===")
print(f"Total Target Allocation: ${positions['target_dollars'].sum():,.0f}")
print(f"Total Notional Value: ${positions['notional_value'].sum():,.0f}")
print(f"Total Notional %: {positions['notional_pct'].sum():.1%}")
print(f"Number of Instruments: {len(positions)}")
print(f"Number of Contracts: {positions['contracts'].sum()}")
```

### End-to-End Tests

```python
def test_e2e_position_sizing():
    """Test complete workflow from signals to contracts"""
    # Setup
    models = [BaseModel(n_bins=10) for _ in range(5)]
    ensemble = DiversifiedEnsemble(models=models)
    ensemble.fit(X_train, y_train, ticker_train)
    
    portfolio = Portfolio(
        ensembles=[ensemble],
        trading_timeframe=TimeFrame.DAILY,
        target_volatility=0.20,
        dm=1.0,
        max_position_pct=2.0
    )
    
    volatility = pd.Series({'TEST': 0.20})
    
    # Get position fractions
    positions = portfolio.predict(X_test, ticker_test, volatility)
    
    # Convert to contracts
    specs = {'TEST': ContractSpec(ticker='TEST', price=100, multiplier=100)}
    sizer = PositionSizer(capital=100_000, contract_specs=specs)
    contracts = sizer.calculate_positions(positions)
    
    # Verify complete pipeline
    assert 'forecast_score' in positions.columns
    assert 'position_fraction' in positions.columns
    assert 'contracts' in contracts.columns
    assert 'notional_value' in contracts.columns
    
    # Position fraction should be in reasonable range
    assert 0 <= positions['position_fraction'].iloc[0] <= 2.0
    
    # Contracts should be non-negative integer
    assert contracts['contracts'].iloc[0] >= 0
    assert isinstance(contracts['contracts'].iloc[0], (int, np.integer))

def test_e2e_multiple_ensembles():
    """Test workflow with multiple ensembles"""
    # Create two ensembles
    ensemble1 = DiversifiedEnsemble(models=[BaseModel(n_bins=10) for _ in range(5)])
    ensemble2 = DiversifiedEnsemble(models=[BaseModel(n_bins=10) for _ in range(5)])
    
    ensemble1.fit(X_train, y_train, ticker_train)
    ensemble2.fit(X_train, y_train, ticker_train)
    
    # Portfolio with both ensembles
    portfolio = Portfolio(
        ensembles=[ensemble1, ensemble2],
        trading_timeframe=TimeFrame.DAILY,
        target_volatility=0.20,
        dm=1.0
    )
    
    volatility = pd.Series({'TEST': 0.20})
    positions = portfolio.predict(X_test, ticker_test, volatility)
    
    # Forecast should be average of two ensembles
    # (exact value depends on model predictions, just check it's computed)
    assert not positions['forecast_score'].isnull().any()
```

---

## Phase 6: DM Fitting Utility

**Goal**: Create utility to fit DM from backtest results.

**Estimated Effort**: 2-3 hours

### Files to Create

- `utils/diversification.py`
- `tests/test_diversification.py`

### DMFitter Class

```python
import pandas as pd
import numpy as np
from typing import Optional

class DMFitter:
    """
    Fit Diversification Multiplier from backtest results.
    
    Simple fitting: DM = target_volatility / realized_volatility
    """
    
    def __init__(self, target_volatility: float = 0.20):
        """
        Parameters
        ----------
        target_volatility : float, default=0.20
            Target annual portfolio volatility
        """
        self.target_volatility = target_volatility
    
    def fit(
        self,
        returns: pd.Series,
        annualization_factor: float = 252
    ) -> float:
        """
        Calculate optimal DM from realized returns.
        
        Parameters
        ----------
        returns : pd.Series
            Daily portfolio returns
        annualization_factor : float, default=252
            Factor to annualize volatility
        
        Returns
        -------
        float
            Optimal DM value
        """
        # Calculate realized volatility
        realized_vol = returns.std() * np.sqrt(annualization_factor)
        
        # Calculate optimal DM
        if realized_vol > 0:
            dm_optimal = self.target_volatility / realized_vol
        else:
            dm_optimal = 1.0
        
        return dm_optimal
    
    def fit_with_validation(
        self,
        returns: pd.Series,
        annualization_factor: float = 252,
        min_dm: float = 0.5,
        max_dm: float = 5.0
    ) -> dict:
        """
        Fit DM with validation and bounds checking.
        
        Parameters
        ----------
        returns : pd.Series
            Daily portfolio returns
        annualization_factor : float
            Factor to annualize volatility
        min_dm : float, default=0.5
            Minimum allowed DM
        max_dm : float, default=5.0
            Maximum allowed DM
        
        Returns
        -------
        dict with keys:
            - dm_optimal: Fitted DM value
            - dm_bounded: DM after applying bounds
            - realized_vol: Realized volatility
            - target_vol: Target volatility
            - ratio: target / realized
        """
        # Calculate realized volatility
        realized_vol = returns.std() * np.sqrt(annualization_factor)
        
        # Calculate optimal DM
        dm_optimal = self.target_volatility / realized_vol if realized_vol > 0 else 1.0
        
        # Apply bounds
        dm_bounded = np.clip(dm_optimal, min_dm, max_dm)
        
        return {
            'dm_optimal': dm_optimal,
            'dm_bounded': dm_bounded,
            'realized_vol': realized_vol,
            'target_vol': self.target_volatility,
            'ratio': dm_optimal
        }
```

### Unit Tests

```python
def test_dm_fitting_basic():
    """Test basic DM fitting"""
    # Create returns with 10% annual volatility
    np.random.seed(42)
    daily_vol = 0.10 / np.sqrt(252)
    returns = pd.Series(np.random.normal(0, daily_vol, 1000))
    
    # Fit with 20% target
    fitter = DMFitter(target_volatility=0.20)
    dm = fitter.fit(returns)
    
    # Should be approximately 0.20 / 0.10 = 2.0
    assert 1.5 < dm < 2.5

def test_dm_fitting_with_validation():
    """Test DM fitting with bounds"""
    # Create very low volatility returns (would give very high DM)
    np.random.seed(42)
    daily_vol = 0.02 / np.sqrt(252)
    returns = pd.Series(np.random.normal(0, daily_vol, 1000))
    
    fitter = DMFitter(target_volatility=0.20)
    result = fitter.fit_with_validation(returns, max_dm=3.0)
    
    # Optimal might be >5, but should be bounded at 3.0
    assert result['dm_bounded'] <= 3.0
    assert result['realized_vol'] < 0.05
    assert result['target_vol'] == 0.20

def test_dm_fitting_target_match():
    """Test DM when realized matches target"""
    # Create returns with 20% annual volatility (matches target)
    np.random.seed(42)
    daily_vol = 0.20 / np.sqrt(252)
    returns = pd.Series(np.random.normal(0, daily_vol, 1000))
    
    fitter = DMFitter(target_volatility=0.20)
    dm = fitter.fit(returns)
    
    # Should be approximately 1.0
    assert 0.8 < dm < 1.2
```

---

## Summary Checklist

### Phase 1: Ensemble ⚠️
- [ ] Remove volatility parameter from predict()
- [ ] Remove scaling/multiplier logic
- [ ] Add exposure_fraction to output
- [ ] Update tests with buy/hold scenarios
- [ ] Verify forecast range [0, 1]

### Phase 2: Portfolio ⚠️
- [ ] Add target_volatility, dm, max_position_pct parameters
- [ ] Add volatility parameter to predict()
- [ ] Implement position fraction calculation
- [ ] Add unit tests (6 buy/hold scenarios)
- [ ] Verify position sizing formula

### Phase 3: Execution 🆕
- [ ] Create ContractSpec dataclass
- [ ] Create Position dataclass
- [ ] Create PositionSizer class
- [ ] Implement contract rounding
- [ ] Add unit tests

### Phase 4: Volatility 🆕
- [ ] Create BlendedVolatility class
- [ ] Implement EWMA-32 calculation
- [ ] Implement 10-year rolling average
- [ ] Add multi-ticker support
- [ ] Add unit tests

### Phase 5: Integration ✅
- [ ] Create demo script
- [ ] Create end-to-end tests
- [ ] Validate complete pipeline
- [ ] Test multiple ensembles
- [ ] Document usage examples

### Phase 6: DM Fitting 🔧
- [ ] Create DMFitter class
- [ ] Implement fitting from returns
- [ ] Add bounds checking
- [ ] Add validation
- [ ] Add unit tests

---

## Total Estimated Effort

- **Phase 1**: 2-3 hours
- **Phase 2**: 6-8 hours
- **Phase 3**: 3-4 hours
- **Phase 4**: 3-4 hours
- **Phase 5**: 4-6 hours
- **Phase 6**: 2-3 hours

**Total**: 20-28 hours (~3-4 days of focused work)

---

## Corrections Applied (2025-01-09)

This section documents corrections made to align implementation specs with forecast specs and current codebase structure.

### Issue 1: Model Reference Inconsistency (Fixed)
**Problem**: Spec used `self.models` (list) but codebase uses `self.base_models` (dict).  
**Solution**: Updated to use dict-based structure:
```python
self.model_exposure_fractions_ = {
    model_name: 1.0 / base_model.n_bins 
    for model_name, base_model in self.base_models.items()
}
```

### Issue 2: Exposure Calculation Array Assumptions (Fixed)
**Problem**: Code assumed numpy array operations on signals, but actual structure is DataFrame with model names as columns.  
**Solution**: Updated to iterate over DataFrame rows and columns explicitly.

### Issue 3: Volatility Parameter Type Clarity (Fixed)
**Problem**: Ambiguous whether volatility parameter was indexed by sample or ticker.  
**Solution**: 
- Clarified documentation: `Union[pd.Series, Dict[str, float]]`
- If Series: must have ticker symbols as index
- If dict: maps ticker → volatility value
- Added validation and conversion logic in Portfolio.predict()

### Issue 4: Instrument Weight Calculation Bug (Fixed)
**Problem**: Used `len(df)` (number of rows) instead of number of unique instruments.  
**Solution**: Changed to `df['ticker'].nunique()` for correct equal weighting.

### Issue 5: Breaking Change - Return Type (Documented)
**Problem**: Ensemble.predict() currently returns `np.ndarray`, spec changes to `pd.DataFrame`.  
**Solution**: 
- Documented as BREAKING CHANGE in Phase 1
- Added migration notes
- Added output structure validation tests
- Updated Portfolio to expect and validate DataFrame input

### Issue 6: Missing Type Imports (Fixed)
**Problem**: Union type used without import statement.  
**Solution**: Added required imports to Phase 2 specification.

### Additional Improvements
1. **Validation**: Added input validation for volatility parameter in Portfolio
2. **Error Messages**: Added descriptive error messages for ensemble output validation
3. **Edge Cases**: Added handling for empty ensemble list in averaging
4. **Documentation**: Clarified that ensembles now return DataFrames, not arrays
5. **Consistency**: Used `as_index=False` in groupby for cleaner code

### Migration Checklist
- [ ] Phase 1: Change ensemble output to DataFrame (BREAKING)
- [ ] Phase 1: Remove volatility parameter from ensemble.predict()
- [ ] Phase 1: Update all tests expecting numpy array to expect DataFrame
- [ ] Phase 2: Add volatility parameter to portfolio.predict()
- [ ] Phase 2: Update portfolio to handle DataFrame from ensembles
- [ ] Phase 2: Add Union type imports
- [ ] Integration: Update all calling code to use new signatures

---

**End of Implementation Specifications**

