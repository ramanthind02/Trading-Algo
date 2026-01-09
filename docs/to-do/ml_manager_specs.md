# MLManager Portfolio Integration Specifications

**Version**: 2.0.0  
**Date**: 2025-01-09  
**Status**: Design Specification  

**Major Changes in v2.0.0**:
- **Single-Timeframe Portfolios**: Each Portfolio now manages ONE trading timeframe only
- **Multiple Portfolios in MLManager**: MLManager can manage multiple Portfolio instances (one per timeframe)
- **Cleaner Separation**: Features can be multi-timeframe, but trading is single-timeframe
- **Removed `base_tf`**: No longer needed - each Portfolio knows its trading timeframe

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Key Components](#key-components)
4. [Implementation Phases](#implementation-phases)
5. [API Specifications](#api-specifications)
6. [Configuration](#configuration)
7. [Error Handling](#error-handling)
8. [Testing Strategy](#testing-strategy)
9. [Usage Examples](#usage-examples)

---

## Overview

Transform MLManager into a complete trading system that ingests streaming market data and outputs actionable positions. The system will auto-discover required bias nodes from Portfolio ensembles, calculate features and volatility, generate forecasts, and convert to tradeable contracts.

### Goals

- **Flexible Integration**: MLManager works standalone OR with Portfolio(s)
- **Auto-Discovery**: Automatically configure bias nodes from ensemble control files
- **Multi-Ticker Support**: Process multiple tickers sequentially with isolated state
- **Multi-Timeframe Trading**: Support separate portfolios for different trading timeframes (D, W, M)
- **Clean State Management**: Single `TickerState` dataclass per ticker
- **Streaming Mode**: Real-world emulation with candle-by-candle processing
- **Optional Contract Conversion**: PositionSizer integration for executable trades
- **Robust Error Handling**: Graceful degradation, comprehensive logging

### Key Design Decisions

**1. Single-Timeframe Portfolios** ⭐ NEW
- **Each Portfolio manages ONE trading timeframe only**
  - Daily Portfolio: Generates positions when daily candles arrive
  - Weekly Portfolio: Generates positions when weekly candles arrive
- **Clear separation**: Features can be multi-timeframe, but trading is single-timeframe
- **Real-world alignment**: Separate trading accounts = separate portfolios
- **Simplified reasoning**: No ambiguity about when a portfolio trades

**2. Multiple Portfolios in MLManager** ⭐ NEW
- MLManager can manage **multiple Portfolio instances** (one per timeframe)
- When a candle arrives for timeframe T, route to Portfolio for timeframe T
- Each Portfolio operates independently with its own risk parameters
- Example: `portfolios={TimeFrame.D: daily_portfolio, TimeFrame.W: weekly_portfolio}`

**3. Portfolio-Optional Architecture**
- **Standalone mode** (no portfolios): Feature extraction for research/backtesting
- **Production mode** (with portfolios): Full pipeline from features to positions
- Maintains backward compatibility with existing research code

**4. Flexible Bias Node Configuration**
- **Auto-discovery**: Read from portfolio ensembles (production)
- **Explicit specs**: Provide bias_node_specs directly (research)
- **Hybrid**: Explicit specs override portfolio (testing new features)

**5. Cleaner Multi-Ticker State**
- Single `TickerState` dataclass encapsulates all per-ticker state
- Replaces multiple parallel dictionaries (ticker_bias_nodes, ticker_bias_values, etc.)
- Easier to reason about, test, and extend

**6. Auto-Predict Control**
- **auto_predict=True**: Automatic position generation when candles arrive (live trading)
- **auto_predict=False**: Manual prediction via `predict_positions()` (batch processing)
- **Removed `base_tf` concept**: Each Portfolio knows its own trading timeframe

---

## Architecture

### System Overview

```mermaid
graph TB
    subgraph input [Market Data Input]
        candle[Candle Data]
        ticker[Ticker Symbol]
        tf[TimeFrame]
    end
    
    subgraph mlmanager [MLManager Streaming Pipeline]
        addCandle[add_candle ticker tf]
        updateNodes[Update Bias Nodes per Ticker]
        routePortfolio{Route to Portfolio?}
    end
    
    subgraph portfolios [Multiple Portfolios by Timeframe]
        portfolioD[Daily Portfolio TF=D]
        portfolioW[Weekly Portfolio TF=W]
        portfolioM[Monthly Portfolio TF=M]
    end
    
    subgraph portfolio_flow [Portfolio Processing]
        extractFeatures[Extract Features]
        calcVol[Calculate Volatility]
        ensembles[Ensembles]
        baseModels[Base Models]
        positionFractions[Position Fractions]
    end
    
    subgraph execution [Execution Layer]
        sizer{PositionSizer?}
        contracts[Convert to Contracts]
        fractions[Return Fractions]
    end
    
    subgraph output [Output]
        positions[Positions Dict]
        telegram[Telegram Export]
    end
    
    candle --> addCandle
    ticker --> addCandle
    tf --> addCandle
    
    addCandle --> updateNodes
    updateNodes --> routePortfolio
    
    routePortfolio -->|TF=D| portfolioD
    routePortfolio -->|TF=W| portfolioW
    routePortfolio -->|TF=M| portfolioM
    
    portfolioD --> extractFeatures
    portfolioW --> extractFeatures
    portfolioM --> extractFeatures
    
    extractFeatures --> calcVol
    calcVol --> ensembles
    ensembles --> baseModels
    baseModels --> positionFractions
    
    positionFractions --> sizer
    sizer -->|Enabled| contracts
    sizer -->|Disabled| fractions
    
    contracts --> positions
    fractions --> positions
    positions --> telegram
```

### Data Flow

```
Market Data (Candle + Ticker + TF)
  ↓
MLManager.add_candle(candle, ticker, tf)
  ↓ Update bias nodes for this ticker/tf
  ↓ Route to appropriate Portfolio based on TF:
    ↓ If tf=D and portfolios[TimeFrame.D] exists → Daily Portfolio
    ↓ If tf=W and portfolios[TimeFrame.W] exists → Weekly Portfolio
    ↓ If tf=M and portfolios[TimeFrame.M] exists → Monthly Portfolio
    ↓
Portfolio[tf].predict(features, volatility, ticker)
  ↓ → Ensemble.predict(features, ticker)
    ↓ → BaseModel.predict(feature_values)
    ↓ ← Binary signals {0, 1}
  ↓ ← Forecast scores [0, 1]
  ↓ Apply volatility scaling, DM, instrument weights
  ↓ ← Position fractions (% of capital)
  ↓
PositionSizer.calculate_positions() [optional]
  ↓ Convert fractions to contracts
  ↓ ← Number of contracts, notional values
  ↓
Return Positions Dict
  ↓
Telegram Export
```

### Multi-Timeframe Portfolio Management

```mermaid
graph TB
    subgraph mlmanager [MLManager]
        dailyPort[Daily Portfolio]
        weeklyPort[Weekly Portfolio]
        monthlyPort[Monthly Portfolio]
    end
    
    subgraph dailyPort [Daily Portfolio TF=D]
        dailyEns[Daily Ensembles]
        dailyRisk[Risk Params DM=2.0 Target Vol=0.20]
    end
    
    subgraph weeklyPort [Weekly Portfolio TF=W]
        weeklyEns[Weekly Ensembles]
        weeklyRisk[Risk Params DM=1.5 Target Vol=0.15]
    end
    
    subgraph monthlyPort [Monthly Portfolio TF=M]
        monthlyEns[Monthly Ensembles]
        monthlyRisk[Risk Params DM=1.2 Target Vol=0.12]
    end
    
    mlmanager --> dailyPort
    mlmanager --> weeklyPort
    mlmanager --> monthlyPort
```

### Multi-Ticker State Management

```mermaid
graph LR
    subgraph ticker_state [Per-Ticker State Isolation]
        ES[ES State]
        NQ[NQ State]
        YM[YM State]
    end
    
    subgraph es_state [ES Ticker]
        esBias[Bias Nodes]
        esValues[Bias Values]
        esMatrix[Feature Matrix]
    end
    
    subgraph nq_state [NQ Ticker]
        nqBias[Bias Nodes]
        nqValues[Bias Values]
        nqMatrix[Feature Matrix]
    end
    
    ES --> es_state
    NQ --> nq_state
    
    es_state --> portfolios[Multiple Portfolio Instances]
    nq_state --> portfolios
```

---

## Key Components

### 1. Ensemble Bias Node Discovery API

**Purpose**: Allow ensembles to expose required bias nodes for auto-configuration

**Location**: `ensemble/diversified_ensemble.py`

**Key Method**:
```python
def get_required_bias_node_specs(self) -> List[Dict[str, Any]]:
    """
    Return list of bias node specs required by all base models.
    
    Reads feature control files from vault, extracts bias_node_spec
    from each feature, and deduplicates.
    
    Returns
    -------
    List[Dict[str, Any]]
        Each spec: {
            'module_name': str,
            'timeframes': [TimeFrame],
            'params': dict
        }
    """
```

**Algorithm**:
1. Get ensemble_dir from ensemble configuration
2. List all feature control files in `{ensemble_dir}/features/`
3. For each feature control file:
   - Load JSON
   - Extract `bias_node_spec`
   - Add to specs list
4. Deduplicate specs (same module_name + params + timeframes)
5. Return unique specs

### 2. Portfolio Bias Node Aggregation

**Purpose**: Aggregate bias nodes from all ensembles (single-timeframe version)

**Location**: `ensemble/portfolio.py`

**Key Method**:
```python
def get_required_bias_node_specs(self) -> List[Dict[str, Any]]:
    """
    Aggregate all bias node specs from all ensembles.
    Always includes ATR-252 and EWSD-252 for volatility calculation.
    
    Returns
    -------
    List[Dict[str, Any]]
        Deduplicated list of all required bias nodes
    """
```

**Algorithm**:
1. Initialize empty specs list
2. For each ensemble in `self.ensembles`:  # Now a list, not dict
   - Call `ensemble.get_required_bias_node_specs()`
   - Add specs to list
3. Deduplicate across all ensembles
4. Append mandatory specs:
   - ATR-252: `{'module_name': 'atr', 'timeframes': [TimeFrame.D], 'params': {'lookback': 252}}`
   - EWSD-252: `{'module_name': 'ewsd', 'timeframes': [TimeFrame.D], 'params': {'lookback': 252}}`
5. Return aggregated list

### 3. MLManager Refactoring

**Purpose**: Enable multi-ticker streaming with optional Portfolio integration (supports multiple portfolios)

**Location**: `feature_extraction/ml_manager.py`

**Key Changes**:
- Remove `ticker` from constructor (becomes per-candle parameter)
- ⭐ **NEW**: Add `portfolios: Dict[TimeFrame, Portfolio]` to constructor (supports multiple portfolios)
- Remove `base_tf` concept (no longer needed - portfolios know their trading timeframes)
- Add `bias_node_specs: Optional[List[Dict]]` to constructor (for standalone use)
- Add `position_sizer: Optional[PositionSizer]` to constructor
- Use `TickerState` dataclass for cleaner per-ticker state management
- Support both standalone (feature extraction) and portfolio (position generation) modes

**New Constructor**:
```python
def __init__(
    self,
    portfolios: Optional[Dict[TimeFrame, Portfolio]] = None,
    bias_node_specs: Optional[List[Dict[str, Any]]] = None,
    position_sizer: Optional[PositionSizer] = None,
    build_matrix: bool = False,
    auto_predict: bool = True
):
    """
    MLManager for streaming market data → features → positions.
    
    Two modes of operation:
    1. **Standalone Mode** (portfolios=None): Feature extraction only
    2. **Production Mode** (portfolios provided): Features + Positions
    
    Bias nodes can be configured in two ways:
    - Auto-discovery: Provide `portfolios` (reads from ensembles)
    - Explicit: Provide `bias_node_specs` (manual configuration)
    
    Parameters
    ----------
    portfolios : Dict[TimeFrame, Portfolio], optional
        Portfolio instances for generating positions, keyed by trading timeframe.
        Each Portfolio manages ONE trading timeframe.
        Example: {
            TimeFrame.D: daily_portfolio,
            TimeFrame.W: weekly_portfolio
        }
        When a candle arrives for a timeframe, MLManager routes it to the
        corresponding Portfolio. If None, runs in standalone mode.
    bias_node_specs : List[Dict[str, Any]], optional
        Explicit bias node specifications. If provided, these take precedence
        over portfolio auto-discovery. Each spec: {
            'module_name': str,
            'timeframes': [TimeFrame],
            'params': dict
        }
    position_sizer : PositionSizer, optional
        If provided, converts position fractions to contracts
    build_matrix : bool, default=False
        Whether to build feature matrices for analysis
    auto_predict : bool, default=True
        If True, automatically trigger predictions when candles arrive.
        If False, user must call predict_positions() manually.
        
    Raises
    ------
    ValueError
        If both portfolios and bias_node_specs are None (no config source)
        If portfolios provided but have no bias node specs
        
    Examples
    --------
    >>> # Standalone mode (research/feature extraction)
    >>> ml_manager = MLManager(
    ...     bias_node_specs=[
    ...         {'module_name': 'rsi', 'timeframes': [TimeFrame.D], 'params': {'lookback': 14}},
    ...         {'module_name': 'atr', 'timeframes': [TimeFrame.D], 'params': {'lookback': 252}}
    ...     ]
    ... )
    >>> 
    >>> # Production mode with single portfolio (daily trading)
    >>> ml_manager = MLManager(
    ...     portfolios={TimeFrame.D: daily_portfolio},
    ...     position_sizer=my_sizer,
    ...     auto_predict=True
    ... )
    >>> 
    >>> # Production mode with multiple portfolios (multi-timeframe trading)
    >>> ml_manager = MLManager(
    ...     portfolios={
    ...         TimeFrame.D: daily_portfolio,   # Trades on daily candles
    ...         TimeFrame.W: weekly_portfolio   # Trades on weekly candles
    ...     },
    ...     position_sizer=my_sizer,
    ...     auto_predict=True
    ... )
    >>> 
    >>> # Hybrid mode (portfolios + explicit bias nodes)
    >>> ml_manager = MLManager(
    ...     portfolios={TimeFrame.D: daily_portfolio},
    ...     bias_node_specs=[  # Additional nodes beyond portfolio
    ...         {'module_name': 'custom_indicator', 'timeframes': [TimeFrame.D], 'params': {...}}
    ...     ]
    ... )
    """
```

**Cleaner Multi-Ticker State Management**:

Use a single dataclass to encapsulate all per-ticker state:

```python
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

@dataclass
class TickerState:
    """
    Encapsulates all state for a single ticker.
    
    Attributes
    ----------
    ticker : Ticker
        Ticker symbol
    bias_nodes : List[Tuple[TimeFrame, BiasNode]]
        List of (timeframe, bias_node) tuples
    bias_values : List[float]
        Current values for all bias nodes (parallel to columns)
    columns : List[str]
        Feature column names
    tf_columns : Dict[TimeFrame, List[str]]
        Column names organized by timeframe
    tf_indices : Dict[TimeFrame, List[int]]
        Global indices for each timeframe's columns
    matrix : Optional[pd.DataFrame]
        Feature matrix (if build_matrix=True)
    matrix_buffer : List[Tuple[datetime, List[float]]]
        Buffer for efficient DataFrame construction
    """
    ticker: Ticker
    bias_nodes: List[Tuple[TimeFrame, Any]] = field(default_factory=list)
    bias_values: List[float] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    tf_columns: Dict[TimeFrame, List[str]] = field(default_factory=dict)
    tf_indices: Dict[TimeFrame, List[int]] = field(default_factory=dict)
    matrix: Optional[pd.DataFrame] = None
    matrix_buffer: List[Tuple[Any, List[float]]] = field(default_factory=list)

# In MLManager:
self.ticker_states: Dict[Ticker, TickerState] = {}
```

**How Prediction Triggering Works** (Removed `base_tf` Concept):

When `auto_predict=True`:
- When a candle arrives via `add_candle(candle, ticker, tf)`, MLManager checks if `portfolios[tf]` exists
- If yes: Extract features, calculate volatility, call `portfolios[tf].predict()`
- If no: Just update bias nodes (no prediction)
- Example:
  - Daily candle arrives → trigger `portfolios[TimeFrame.D]` (if exists)
  - Weekly candle arrives → trigger `portfolios[TimeFrame.W]` (if exists)
- Each Portfolio independently decides whether to generate a position

When `auto_predict=False`:
- Candles update bias nodes but don't trigger predictions
- User must manually call `predict_positions(ticker, timeframe)` when ready
- Useful for batch processing or custom prediction timing

### 4. PositionSizer Integration

**Purpose**: Optional conversion from position fractions to tradeable contracts

**Location**: `execution/position_sizer.py`

**Enhancement**: Add configuration loading

**New Class Method**:
```python
@classmethod
def from_config(
    cls,
    config_path: str,
    capital: float,
    ticker_variants: Dict[str, str],
    prices: Dict[str, float],
    rounding_method: RoundingMethod = RoundingMethod.ROUND
) -> 'PositionSizer':
    """
    Load contract specs from config file.
    
    Parameters
    ----------
    config_path : str
        Path to contract_specs.json
    capital : float
        Account capital in USD
    ticker_variants : Dict[str, str]
        Maps ticker name to variant (e.g., {'ES': 'micro', 'NQ': 'standard'})
    prices : Dict[str, float]
        Current market prices per ticker
    rounding_method : RoundingMethod
        How to round fractional contracts
        
    Returns
    -------
    PositionSizer
        Configured instance ready for use
    """
```

### 5. Contract Specs Configuration

**Purpose**: Define contract specifications per ticker with multiple variants

**Location**: `deployment/config/contract_specs.json`

**Structure**:
```json
{
  "ES": {
    "standard": {
      "multiplier": 50,
      "tick_size": 0.25,
      "tick_value": 12.50,
      "currency": "USD",
      "description": "E-mini S&P 500"
    },
    "micro": {
      "multiplier": 5,
      "tick_size": 0.25,
      "tick_value": 1.25,
      "currency": "USD",
      "description": "Micro E-mini S&P 500"
    }
  },
  "NQ": {
    "standard": {
      "multiplier": 20,
      "tick_size": 0.25,
      "tick_value": 5.00,
      "currency": "USD",
      "description": "E-mini NASDAQ-100"
    },
    "micro": {
      "multiplier": 2,
      "tick_size": 0.25,
      "tick_value": 0.50,
      "currency": "USD",
      "description": "Micro E-mini NASDAQ-100"
    }
  }
}
```

---

## Implementation Phases

### Phase 1: Ensemble Bias Node Discovery API

**Estimated Effort**: 2-3 hours

**File**: `ensemble/diversified_ensemble.py`

**Tasks**:
1. Add `ensemble_dir` attribute (path to vault ensemble directory)
2. Implement `get_required_bias_node_specs()` method
3. Add deduplication logic
4. Add validation (ensure bias_node_spec exists in control files)

**Implementation**:
```python
def get_required_bias_node_specs(self) -> List[Dict[str, Any]]:
    """Return list of bias node specs required by all base models."""
    if not hasattr(self, 'ensemble_dir') or self.ensemble_dir is None:
        raise ValueError(
            "ensemble_dir not set. Cannot auto-discover bias nodes."
        )
    
    # Read all feature control files
    features_dir = os.path.join(self.ensemble_dir, 'features')
    if not os.path.exists(features_dir):
        return []
    
    specs = []
    for filename in os.listdir(features_dir):
        if not filename.endswith('.json'):
            continue
        
        filepath = os.path.join(features_dir, filename)
        with open(filepath, 'r') as f:
            feature_config = json.load(f)
        
        bias_node_spec = feature_config.get('bias_node_spec')
        if bias_node_spec:
            specs.append(bias_node_spec)
    
    # Deduplicate
    unique_specs = []
    seen = set()
    
    for spec in specs:
        # Create hashable key
        key = (
            spec['module_name'],
            tuple(spec['timeframes']),
            tuple(sorted(spec['params'].items()))
        )
        
        if key not in seen:
            seen.add(key)
            unique_specs.append(spec)
    
    return unique_specs
```

**Tests**:
```python
def test_ensemble_bias_node_discovery_single_feature():
    """Test discovery with single feature"""
    
def test_ensemble_bias_node_discovery_deduplication():
    """Test that duplicate specs are removed"""
    
def test_ensemble_bias_node_discovery_empty():
    """Test empty ensemble returns empty list"""
```

### Phase 2: Portfolio Bias Node Discovery

**Estimated Effort**: 1 hour

**File**: `ensemble/portfolio.py`

**Tasks**:
1. Implement `get_required_bias_node_specs()` method
2. Aggregate from all ensembles
3. Add ATR-252 and EWSD-252
4. Deduplicate across ensembles

**Implementation**:
```python
def get_required_bias_node_specs(self) -> List[Dict[str, Any]]:
    """Aggregate all bias node specs from all ensembles."""
    all_specs = []
    
    # Collect from all ensembles (now a simple list)
    for ensemble in self.ensembles:
        specs = ensemble.get_required_bias_node_specs()
        all_specs.extend(specs)
    
    # Deduplicate
    unique_specs = []
    seen = set()
    
    for spec in all_specs:
        key = (
            spec['module_name'],
            tuple(spec.get('timeframes', [])),
            tuple(sorted(spec.get('params', {}).items()))
        )
        
        if key not in seen:
            seen.add(key)
            unique_specs.append(spec)
    
    # Always include ATR-252 and EWSD-252 for volatility
    mandatory_specs = [
        {
            'module_name': 'atr',
            'timeframes': [TimeFrame.D],
            'params': {'lookback': 252}
        },
        {
            'module_name': 'ewsd',
            'timeframes': [TimeFrame.D],
            'params': {'lookback': 252}
        }
    ]
    
    # Add mandatory specs if not already present
    for mandatory_spec in mandatory_specs:
        key = (
            mandatory_spec['module_name'],
            tuple(mandatory_spec['timeframes']),
            tuple(sorted(mandatory_spec['params'].items()))
        )
        
        if key not in seen:
            unique_specs.append(mandatory_spec)
    
    return unique_specs
```

**Tests**:
```python
def test_portfolio_aggregates_from_multiple_ensembles():
    """Test aggregation across ensembles (single-timeframe)"""
    
def test_portfolio_includes_atr_ewsd():
    """Test ATR-252 and EWSD-252 always included"""
    
def test_portfolio_deduplicates_across_ensembles():
    """Test deduplication works across ensembles"""

def test_portfolio_has_single_trading_timeframe():
    """Test that Portfolio enforces single trading timeframe"""
```

### Phase 3: Contract Specs Configuration

**Estimated Effort**: 2 hours

**Files**: 
- `deployment/config/contract_specs.json` (new)
- `execution/position_sizer.py` (enhance)

**Tasks**:
1. Create contract_specs.json with major futures
2. Add `from_config()` class method to PositionSizer
3. Add validation logic
4. Add tests

**Implementation**:
```python
@classmethod
def from_config(
    cls,
    config_path: str,
    capital: float,
    ticker_variants: Dict[str, str],
    prices: Dict[str, float],
    rounding_method: RoundingMethod = RoundingMethod.ROUND
) -> 'PositionSizer':
    """Load contract specs from config file."""
    import json
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Build ContractSpec dict
    contract_specs = {}
    
    for ticker_name, variant_name in ticker_variants.items():
        # Validate ticker exists
        if ticker_name not in config:
            raise ValueError(
                f"Ticker '{ticker_name}' not found in config. "
                f"Available tickers: {list(config.keys())}"
            )
        
        ticker_config = config[ticker_name]
        
        # Validate variant exists
        if variant_name not in ticker_config:
            raise ValueError(
                f"Variant '{variant_name}' not found for ticker '{ticker_name}'. "
                f"Available variants: {list(ticker_config.keys())}"
            )
        
        variant_config = ticker_config[variant_name]
        
        # Get price
        if ticker_name not in prices:
            raise ValueError(
                f"Price not provided for ticker '{ticker_name}'"
            )
        
        price = prices[ticker_name]
        
        # Create ContractSpec
        contract_specs[ticker_name] = ContractSpec(
            ticker=ticker_name,
            price=price,
            multiplier=variant_config['multiplier'],
            fx_rate=1.0,  # Assume USD for now
            min_tick=variant_config['tick_size']
        )
    
    return cls(
        capital=capital,
        contract_specs=contract_specs,
        rounding_method=rounding_method
    )
```

**Tests**:
```python
def test_position_sizer_from_config_valid():
    """Test loading valid config"""
    
def test_position_sizer_from_config_invalid_ticker():
    """Test invalid ticker raises error"""
    
def test_position_sizer_from_config_invalid_variant():
    """Test invalid variant raises error"""
    
def test_position_sizer_from_config_missing_price():
    """Test missing price raises error"""
```

### Phase 4: MLManager Multi-Ticker State

**Estimated Effort**: 4-5 hours

**File**: `feature_extraction/ml_manager.py`

**Major Refactoring**:

1. **New Constructor**:
```python
def __init__(
    self,
    portfolios: Optional[Dict[TimeFrame, Portfolio]] = None,
    bias_node_specs: Optional[List[Dict[str, Any]]] = None,
    position_sizer: Optional[PositionSizer] = None,
    build_matrix: bool = False,
    auto_predict: bool = True
):
    """Initialize MLManager with optional Portfolios."""
    self.portfolios = portfolios
    self.position_sizer = position_sizer
    self.build_matrix = build_matrix
    self.auto_predict = auto_predict
    
    # Cleaner multi-ticker state using dataclass
    self.ticker_states: Dict[Ticker, TickerState] = {}
    
    # Determine bias node specs (explicit or auto-discovery)
    if bias_node_specs is not None:
        # Explicit configuration (takes precedence)
        self.bias_node_specs = bias_node_specs
        logger.info(
            f"MLManager initialized with {len(bias_node_specs)} "
            f"explicitly provided bias node specs"
        )
    elif portfolios is not None:
        # Auto-discovery from all portfolios
        all_specs = []
        for tf, portfolio in portfolios.items():
            specs = portfolio.get_required_bias_node_specs()
            all_specs.extend(specs)
        
        # Deduplicate
        unique_specs = []
        seen = set()
        for spec in all_specs:
            key = (
                spec['module_name'],
                tuple(spec.get('timeframes', [])),
                tuple(sorted(spec.get('params', {}).items()))
            )
            if key not in seen:
                seen.add(key)
                unique_specs.append(spec)
        
        self.bias_node_specs = unique_specs
        logger.info(
            f"MLManager initialized with {len(self.bias_node_specs)} "
            f"bias node specs auto-discovered from {len(portfolios)} portfolios"
        )
    else:
        raise ValueError(
            "Must provide either 'portfolios' or 'bias_node_specs'. "
            "Cannot initialize MLManager without bias node configuration."
        )
    
    # Log mode
    if portfolios is None:
        logger.info("MLManager running in STANDALONE mode (feature extraction only)")
    else:
        timeframes = ", ".join([str(tf) for tf in portfolios.keys()])
        logger.info(
            f"MLManager running in PRODUCTION mode with portfolios for: {timeframes}"
        )
```

2. **Initialize Bias Nodes Per Ticker** (using TickerState dataclass):
```python
def _initialize_ticker(self, ticker: Ticker) -> None:
    """
    Initialize bias nodes and state for a new ticker.
    
    Uses TickerState dataclass for clean state encapsulation.
    """
    if ticker in self.ticker_states:
        return  # Already initialized
    
    # Create new ticker state
    state = TickerState(ticker=ticker)
    
    # Create bias nodes from specs
    for spec in self.bias_node_specs:
        module_name = spec['module_name']
        timeframes = spec['timeframes']
        params = spec['params']
        
        for tf in timeframes:
            # Initialize tracking for this timeframe
            if tf not in state.tf_columns:
                state.tf_columns[tf] = []
                state.tf_indices[tf] = []
            
            # Create bias node
            try:
                bias_node = helpers.create_bias_node(
                    module_name, ticker, tf, params
                )
            except Exception as e:
                logger.error(
                    f"Failed to create bias node {module_name} for {ticker.name}: {e}. "
                    f"Skipping this node."
                )
                continue
            
            state.bias_nodes.append((tf, bias_node))
            
            # Get column names
            node_columns = bias_node.get_column_names()
            
            for col_name in node_columns:
                state.columns.append(col_name)
                state.tf_columns[tf].append(col_name)
                state.tf_indices[tf].append(len(state.bias_values))
                state.bias_values.append(0.0)  # Initialize
    
    # Initialize matrix if needed
    if self.build_matrix:
        state.matrix = pd.DataFrame(columns=state.columns)
        state.matrix_buffer = []
    
    # Store state
    self.ticker_states[ticker] = state
    
    logger.info(
        f"Initialized {len(state.bias_nodes)} bias nodes for {ticker.name} "
        f"with {len(state.columns)} columns"
    )
```

3. **Updated add_candle()** (with cleaner state access and multi-portfolio routing):
```python
def add_candle(
    self, 
    candle: Candle, 
    ticker: Ticker, 
    tf: TimeFrame
) -> Optional[Dict[str, Any]]:
    """
    Add candle for specific ticker and timeframe.
    
    Updates bias nodes for this ticker/timeframe. If auto_predict=True and
    a Portfolio exists for this timeframe, automatically triggers position prediction.
    
    Parameters
    ----------
    candle : Candle
        Market candle data (OHLCV + datetime)
    ticker : Ticker
        Ticker symbol enum
    tf : TimeFrame
        Timeframe of the candle
        
    Returns
    -------
    Optional[Dict[str, Any]]
        If auto_predict=True and portfolios[tf] exists:
        Returns positions dict:
        {
            'ticker': str,
            'datetime': datetime,
            'timeframe': TimeFrame,
            'forecast_score': float,
            'position_fraction': float,
            'contracts': int (if PositionSizer enabled),
            'notional_value': float (if PositionSizer enabled),
            'notional_pct': float (if PositionSizer enabled)
        }
        
        Otherwise returns None.
        
    Notes
    -----
    - First call for a ticker initializes bias nodes
    - State is isolated per ticker (NQ doesn't affect ES)
    - Errors are logged but don't crash (returns None)
    - In standalone mode (no portfolios), always returns None
    - Each timeframe routes to its corresponding Portfolio
    """
    try:
        # Initialize ticker state if needed
        if ticker not in self.ticker_states:
            try:
                self._initialize_ticker(ticker)
            except Exception as e:
                logger.error(
                    f"Failed to initialize ticker {ticker.name}: {e}"
                )
                return None
        
        state = self.ticker_states[ticker]
        
        # Add to matrix buffer if enabled (for any timeframe we're tracking)
        if self.build_matrix:
            state.matrix_buffer.append(
                (candle.datetime, state.bias_values.copy())
            )
            
            # Flush buffer periodically
            if len(state.matrix_buffer) >= 100:
                self._flush_matrix_buffer(ticker)
        
        # Update bias nodes for this ticker/timeframe
        tf_column_idx = 0
        
        for node_tf, bias_node in state.bias_nodes:
            if tf != node_tf:
                continue
            
            try:
                vals = bias_node.add_candle(candle)
                num_vals = len(vals)
                
                # Update bias_values
                for j in range(num_vals):
                    val = vals[j].value if isinstance(vals[j], Bias) else vals[j]
                    
                    # Get global index for this timeframe's column
                    global_idx = state.tf_indices[node_tf][tf_column_idx]
                    state.bias_values[global_idx] = val
                    tf_column_idx += 1
                    
            except Exception as e:
                logger.error(
                    f"Error updating bias node for {ticker.name}: {e}"
                )
                continue
        
        # Auto-predict if a Portfolio exists for this timeframe
        if self.auto_predict and self.portfolios is not None and tf in self.portfolios:
            return self._predict_positions(ticker, tf, candle.datetime)
        
        return None
        
    except Exception as e:
        logger.error(
            f"Unexpected error in add_candle for {ticker.name}: {e}",
            exc_info=True
        )
        return None
```

4. **Predict Positions** (cleaner state access, multi-portfolio support):
```python
def predict_positions(
    self, 
    ticker: Ticker, 
    timeframe: TimeFrame
) -> Optional[Dict[str, Any]]:
    """
    Manually trigger position prediction for a ticker at a specific timeframe.
    
    Useful when auto_predict=False or for custom prediction timing.
    
    Parameters
    ----------
    ticker : Ticker
        Ticker to generate positions for
    timeframe : TimeFrame
        Trading timeframe to use (must have corresponding Portfolio)
        
    Returns
    -------
    Optional[Dict[str, Any]]
        Positions dict if successful, None otherwise
        
    Raises
    ------
    ValueError
        If ticker not initialized or no portfolio configured for timeframe
    """
    if ticker not in self.ticker_states:
        raise ValueError(
            f"Ticker {ticker.name} not initialized. "
            f"Call add_candle() first to initialize bias nodes."
        )
    
    if self.portfolios is None or timeframe not in self.portfolios:
        raise ValueError(
            f"No portfolio configured for timeframe {timeframe}. "
            f"Cannot generate positions without portfolio."
        )
    
    state = self.ticker_states[ticker]
    
    # Use latest datetime from matrix if available
    if self.build_matrix and state.matrix is not None and len(state.matrix) > 0:
        latest_datetime = state.matrix.index[-1]
    else:
        latest_datetime = datetime.now()
    
    return self._predict_positions(ticker, timeframe, latest_datetime)

def _predict_positions(
    self, 
    ticker: Ticker,
    timeframe: TimeFrame,
    datetime: dt
) -> Optional[Dict[str, Any]]:
    """Internal method to generate positions for a ticker at a specific timeframe."""
    try:
        state = self.ticker_states[ticker]
        
        # Get the portfolio for this timeframe
        portfolio = self.portfolios.get(timeframe)
        if portfolio is None:
            logger.warning(
                f"No portfolio for {timeframe}, skipping prediction"
            )
            return None
        
        # Flush buffer to get latest features
        if self.build_matrix:
            self._flush_matrix_buffer(ticker)
        
        # Get features (last row)
        if not self.build_matrix or state.matrix is None or len(state.matrix) == 0:
            logger.warning(
                f"No feature matrix available for {ticker.name}. "
                f"Skipping position prediction."
            )
            return None
        
        features_df = state.matrix.tail(1)
        
        # Calculate volatility
        volatility_dict = self._calculate_blended_volatility(
            ticker, features_df
        )
        
        # Create ticker series
        ticker_series = pd.Series(
            [ticker.name], 
            index=features_df.index
        )
        
        # Call the portfolio for this timeframe
        positions_df = portfolio.predict(
            X=features_df,
            ticker=ticker_series,
            volatility=volatility_dict,
            normalization_data=None
        )
        
        # Convert to dict
        position_dict = positions_df.to_dict('records')[0]
        position_dict['datetime'] = datetime
        position_dict['timeframe'] = timeframe
        
        # Optionally convert to contracts
        if self.position_sizer:
            try:
                contracts_df = self.position_sizer.calculate_positions(
                    positions_df
                )
                contract_dict = contracts_df.to_dict('records')[0]
                position_dict.update(contract_dict)
            except Exception as e:
                logger.error(
                    f"PositionSizer failed for {ticker.name}: {e}. "
                    f"Returning position fractions only."
                )
        
        return position_dict
        
    except Exception as e:
        logger.error(
            f"Error predicting positions for {ticker.name} at {timeframe}: {e}",
            exc_info=True
        )
        return None
```

5. **Calculate Volatility**:
```python
def _calculate_blended_volatility(
    self, 
    ticker: Ticker, 
    features_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate blended volatility from ATR and EWSD columns.
    
    For now: Use EWSD as proxy (simplification).
    Future: 70% EWMA-32 + 30% 10-year average.
    """
    # Find EWSD column
    ewsd_cols = [
        col for col in features_df.columns 
        if 'ewsd' in col.lower() and '252' in col
    ]
    
    if not ewsd_cols:
        logger.warning(
            f"No EWSD-252 column found for {ticker.name}. "
            f"Using default volatility 0.20"
        )
        return {ticker.name: 0.20}
    
    ewsd_col = ewsd_cols[0]
    ewsd_value = features_df[ewsd_col].iloc[-1]
    
    # EWSD is in percentage, convert to decimal
    volatility = ewsd_value / 100.0
    
    return {ticker.name: volatility}
```

**Tests**:
```python
def test_mlmanager_single_ticker_streaming():
    """Test single ticker streaming mode"""
    
def test_mlmanager_multi_ticker_streaming():
    """Test ES + NQ streaming simultaneously"""
    
def test_mlmanager_bias_node_isolation():
    """Test NQ candle doesn't affect ES state"""
    
def test_mlmanager_portfolio_prediction_trigger():
    """Test prediction triggered on base_tf"""
    
def test_mlmanager_volatility_extraction():
    """Test volatility calculation"""
    
def test_mlmanager_position_output_format():
    """Test output dict structure"""
```

### Phase 5: Portfolio Integration in MLManager

**Estimated Effort**: 3-4 hours

**File**: `feature_extraction/ml_manager.py`

This is covered in Phase 4 (integrated approach).

### Phase 6: Error Handling and Logging

**Estimated Effort**: 2-3 hours

**Files**: All modified files

**Error Handling Strategy**:

1. **Initialization Validation**:
```python
def __init__(self, portfolio: Portfolio, ...):
    # Validate portfolio
    if not isinstance(portfolio, Portfolio):
        raise TypeError("portfolio must be Portfolio instance")
    
    # Validate position_sizer
    if position_sizer and not isinstance(position_sizer, PositionSizer):
        raise TypeError("position_sizer must be PositionSizer instance")
    
    # Try to get bias node specs
    try:
        self.bias_node_specs = portfolio.get_required_bias_node_specs()
    except Exception as e:
        logger.error(f"Failed to get bias node specs: {e}")
        raise
    
    if not self.bias_node_specs:
        logger.warning("No bias node specs found. Portfolio may be empty.")
```

2. **Graceful Degradation in add_candle()**:
```python
def add_candle(self, candle: Candle, ticker: Ticker, tf: TimeFrame):
    try:
        # Initialization
        if ticker not in self.ticker_bias_nodes:
            try:
                self._initialize_ticker_bias_nodes(ticker)
            except Exception as e:
                logger.error(
                    f"Failed to initialize bias nodes for {ticker.name}: {e}"
                )
                return None
        
        # Update bias nodes (wrapped in try/except per node)
        # ... (see above)
        
        # Prediction
        if tf == self.base_tf:
            try:
                return self._predict_positions(ticker, candle.datetime)
            except Exception as e:
                logger.error(
                    f"Failed to predict positions for {ticker.name}: {e}",
                    exc_info=True
                )
                return None
        
        return None
        
    except Exception as e:
        logger.error(
            f"Unexpected error in add_candle for {ticker.name}: {e}",
            exc_info=True
        )
        return None
```

3. **Missing Volatility Handling**:
```python
def _calculate_blended_volatility(self, ticker, features_df):
    try:
        # Find EWSD column
        ewsd_cols = [col for col in features_df.columns if 'ewsd' in col.lower()]
        
        if not ewsd_cols:
            logger.warning(
                f"No EWSD column for {ticker.name}. Using default 0.20"
            )
            return {ticker.name: 0.20}
        
        # ... extract value
        
    except Exception as e:
        logger.error(
            f"Error calculating volatility for {ticker.name}: {e}. "
            f"Using default 0.20"
        )
        return {ticker.name: 0.20}
```

**Tests**:
```python
def test_mlmanager_missing_volatility_columns():
    """Test graceful handling of missing volatility"""
    
def test_mlmanager_portfolio_prediction_failure():
    """Test continues after portfolio error"""
    
def test_mlmanager_position_sizer_failure():
    """Test falls back to fractions on sizer error"""
    
def test_mlmanager_invalid_ticker():
    """Test handles invalid ticker gracefully"""
```

### Phase 7: Integration Tests

**Estimated Effort**: 3-4 hours

**File**: `tests/test_mlmanager_portfolio_integration.py` (new)

**Test Suite**:

```python
import pytest
import pandas as pd
from datetime import datetime
from utils.enums import Ticker, TimeFrame
from feature_extraction.ml_manager import MLManager
from ensemble.portfolio import Portfolio
from execution.position_sizer import PositionSizer

class TestMLManagerPortfolioIntegration:
    """End-to-end integration tests for MLManager + Portfolio."""
    
    def test_single_ticker_streaming(self):
        """Test full pipeline: candles → positions (single ticker)"""
        # Setup portfolio with mock ensembles
        # Create MLManager
        # Stream candles
        # Assert positions generated
        pass
    
    def test_multi_ticker_streaming(self):
        """Test ES + NQ streaming simultaneously"""
        # Setup multi-ticker portfolio
        # Create MLManager
        # Interleave ES and NQ candles
        # Assert separate positions for each
        pass
    
    def test_position_fractions_only(self):
        """Test without PositionSizer"""
        # Create MLManager without position_sizer
        # Stream candles
        # Assert output has position_fraction, no contracts
        pass
    
    def test_with_position_sizer(self):
        """Test with contract conversion"""
        # Create MLManager with position_sizer
        # Stream candles
        # Assert output has contracts, notional_value
        pass
    
    def test_bias_node_auto_discovery(self):
        """Test MLManager discovers correct bias nodes from portfolio"""
        # Create portfolio with known features
        # Create MLManager
        # Assert bias_node_specs match expected
        pass
    
    def test_volatility_calculation(self):
        """Test volatility extraction from features"""
        # Mock features with known EWSD values
        # Test _calculate_blended_volatility
        # Assert correct extraction
        pass
    
    def test_graceful_error_handling(self):
        """Test system continues after errors"""
        # Inject error in portfolio.predict()
        # Stream candles
        # Assert returns None but doesn't crash
        # Assert subsequent candles still work
        pass
    
    def test_multi_timeframe_processing(self):
        """Test W and M timeframe bias nodes update correctly"""
        # Create portfolio with W and M timeframe features
        # Stream D, W, M candles
        # Assert each timeframe updates independently
        pass
```

### Phase 8: Documentation

**Estimated Effort**: 2 hours

**File**: `docs/to-do/ml_manager_specs.md` (this file)

Complete sections:
- ✅ Overview
- ✅ Architecture
- ✅ Key Components
- ✅ Implementation Phases
- [ ] API Specifications (detailed method signatures)
- [ ] Configuration (complete examples)
- [ ] Error Handling (comprehensive guide)
- [ ] Testing Strategy (test pyramid)
- [ ] Usage Examples (real-world scenarios)

---

## API Specifications

### MLManager

#### Constructor

```python
def __init__(
    self,
    base_tf: TimeFrame,
    portfolio: Optional[Portfolio] = None,
    bias_node_specs: Optional[List[Dict[str, Any]]] = None,
    position_sizer: Optional[PositionSizer] = None,
    build_matrix: bool = False,
    auto_predict: bool = True
):
    """
    Initialize MLManager for streaming market data processing.
    
    Supports two modes:
    - Standalone: Feature extraction only (no portfolio)
    - Production: Features + positions (portfolio required)
    
    Bias nodes can be configured via:
    - Auto-discovery: Provide portfolio (reads from ensembles)
    - Explicit: Provide bias_node_specs (manual configuration)
    
    Parameters
    ----------
    base_tf : TimeFrame
        Base timeframe for processing. Predictions trigger when candles
        for this timeframe arrive (if auto_predict=True).
    portfolio : Portfolio, optional
        Portfolio instance for position generation. If None, runs in
        standalone mode (feature extraction only).
    bias_node_specs : List[Dict[str, Any]], optional
        Explicit bias node specifications. Takes precedence over
        portfolio auto-discovery. Each spec: {
            'module_name': str,
            'timeframes': [TimeFrame],
            'params': dict
        }
    position_sizer : PositionSizer, optional
        Converts position fractions to contracts
    build_matrix : bool, default=False
        Whether to build feature matrices for analysis
    auto_predict : bool, default=True
        If True, automatically trigger predictions when base_tf candle
        arrives. If False, call predict_positions() manually.
        
    Raises
    ------
    ValueError
        - If both portfolio and bias_node_specs are None
        - If portfolio provided but has no bias node specs
    TypeError
        - If portfolio or position_sizer are wrong type
        
    Examples
    --------
    >>> # Standalone mode (research)
    >>> ml_manager = MLManager(
    ...     base_tf=TimeFrame.D,
    ...     bias_node_specs=[{'module_name': 'rsi', ...}]
    ... )
    >>> 
    >>> # Production mode (auto-discovery)
    >>> ml_manager = MLManager(
    ...     base_tf=TimeFrame.D,
    ...     portfolio=my_portfolio
    ... )
    """
```

#### add_candle()

```python
def add_candle(
    self, 
    candle: Candle, 
    ticker: Ticker, 
    tf: TimeFrame
) -> Optional[Dict[str, Any]]:
    """
    Add candle for specific ticker and timeframe.
    
    Updates bias nodes for this ticker/timeframe. If base_tf completes,
    triggers portfolio prediction and returns positions.
    
    Parameters
    ----------
    candle : Candle
        Market candle data (OHLCV + datetime)
    ticker : Ticker
        Ticker symbol enum
    tf : TimeFrame
        Timeframe of the candle
        
    Returns
    -------
    Optional[Dict[str, Any]]
        If base_tf complete, returns positions dict:
        {
            'ticker': str,
            'datetime': datetime,
            'forecast_score': float,
            'position_fraction': float,
            'contracts': int (if PositionSizer enabled),
            'notional_value': float (if PositionSizer enabled),
            'notional_pct': float (if PositionSizer enabled)
        }
        
        If not base_tf or error occurs, returns None
        
    Notes
    -----
    - First call for a ticker initializes bias nodes
    - Bias node state is isolated per ticker
    - Errors are logged but don't crash (returns None)
    """
```

#### predict_positions()

```python
def predict_positions(self, ticker: Ticker) -> Optional[Dict[str, Any]]:
    """
    Manually trigger position prediction for a ticker.
    
    Useful when auto_predict=False or for custom prediction timing.
    
    Parameters
    ----------
    ticker : Ticker
        Ticker to generate positions for
        
    Returns
    -------
    Optional[Dict[str, Any]]
        Positions dict if successful:
        {
            'ticker': str,
            'datetime': datetime,
            'forecast_score': float,
            'position_fraction': float,
            'contracts': int (if PositionSizer configured),
            'notional_value': float (if PositionSizer configured),
            'notional_pct': float (if PositionSizer configured)
        }
        
        Returns None if error occurs.
        
    Raises
    ------
    ValueError
        - If ticker not initialized (call add_candle() first)
        - If no portfolio configured (standalone mode)
        
    Examples
    --------
    >>> # Stream candles without auto-prediction
    >>> ml_manager = MLManager(
    ...     base_tf=TimeFrame.D,
    ...     portfolio=my_portfolio,
    ...     auto_predict=False
    ... )
    >>> 
    >>> # Add candles
    >>> for candle in candles:
    ...     ml_manager.add_candle(candle, Ticker.ES, TimeFrame.D)
    >>> 
    >>> # Manually trigger prediction
    >>> result = ml_manager.predict_positions(Ticker.ES)
    """

#### get_feature_matrix()

```python
def get_feature_matrix(self, ticker: Ticker) -> pd.DataFrame:
    """
    Get feature matrix for a ticker.
    
    Parameters
    ----------
    ticker : Ticker
        Ticker symbol
        
    Returns
    -------
    pd.DataFrame
        Feature matrix with datetime index and feature columns
        
    Raises
    ------
    ValueError
        - If build_matrix=False
        - If ticker not initialized
        
    Examples
    --------
    >>> ml_manager = MLManager(
    ...     base_tf=TimeFrame.D,
    ...     bias_node_specs=[...],
    ...     build_matrix=True
    ... )
    >>> 
    >>> # Stream candles
    >>> for candle in candles:
    ...     ml_manager.add_candle(candle, Ticker.ES, TimeFrame.D)
    >>> 
    >>> # Get feature matrix
    >>> features = ml_manager.get_feature_matrix(Ticker.ES)
    """
```

### Portfolio

#### get_required_bias_node_specs()

```python
def get_required_bias_node_specs(self) -> List[Dict[str, Any]]:
    """
    Aggregate all bias node specs from all ensembles.
    
    Always includes ATR-252 and EWSD-252 for volatility calculation.
    
    Returns
    -------
    List[Dict[str, Any]]
        Deduplicated list of bias node specs.
        Each spec: {
            'module_name': str,
            'timeframes': [TimeFrame],
            'params': dict
        }
        
    Notes
    -----
    - Aggregates from all ensembles across all timeframes
    - Deduplicates based on (module_name, timeframes, params)
    - Always appends ATR-252 and EWSD-252 if not present
    """
```

### DiversifiedEnsemble

#### get_required_bias_node_specs()

```python
def get_required_bias_node_specs(self) -> List[Dict[str, Any]]:
    """
    Return list of bias node specs required by all base models.
    
    Reads feature control files from vault ensemble directory,
    extracts bias_node_spec from each feature, and deduplicates.
    
    Returns
    -------
    List[Dict[str, Any]]
        Unique bias node specs.
        Each spec: {
            'module_name': str,
            'timeframes': [TimeFrame],
            'params': dict
        }
        
    Raises
    ------
    ValueError
        If ensemble_dir not set
        
    Notes
    -----
    - Reads from {ensemble_dir}/features/*.json
    - Deduplicates based on (module_name, timeframes, params)
    - Returns empty list if features directory doesn't exist
    """
```

### PositionSizer

#### from_config()

```python
@classmethod
def from_config(
    cls,
    config_path: str,
    capital: float,
    ticker_variants: Dict[str, str],
    prices: Dict[str, float],
    rounding_method: RoundingMethod = RoundingMethod.ROUND
) -> 'PositionSizer':
    """
    Load contract specs from config file.
    
    Parameters
    ----------
    config_path : str
        Path to contract_specs.json
    capital : float
        Account capital in USD
    ticker_variants : Dict[str, str]
        Maps ticker name to variant.
        Example: {'ES': 'micro', 'NQ': 'standard'}
    prices : Dict[str, float]
        Current market prices per ticker.
        Example: {'ES': 4800.0, 'NQ': 16000.0}
    rounding_method : RoundingMethod, default=ROUND
        How to round fractional contracts
        
    Returns
    -------
    PositionSizer
        Configured instance ready for use
        
    Raises
    ------
    FileNotFoundError
        If config_path doesn't exist
    ValueError
        If ticker not in config, variant not found, or price missing
        
    Examples
    --------
    >>> sizer = PositionSizer.from_config(
    ...     config_path='deployment/config/contract_specs.json',
    ...     capital=1_000_000,
    ...     ticker_variants={'ES': 'micro', 'NQ': 'standard'},
    ...     prices={'ES': 4800.0, 'NQ': 16000.0}
    ... )
    """
```

---

## Configuration

### Contract Specs Configuration

**File**: `deployment/config/contract_specs.json`

**Complete Example**:
```json
{
  "ES": {
    "standard": {
      "multiplier": 50,
      "tick_size": 0.25,
      "tick_value": 12.50,
      "currency": "USD",
      "description": "E-mini S&P 500",
      "exchange": "CME"
    },
    "micro": {
      "multiplier": 5,
      "tick_size": 0.25,
      "tick_value": 1.25,
      "currency": "USD",
      "description": "Micro E-mini S&P 500",
      "exchange": "CME"
    }
  },
  "NQ": {
    "standard": {
      "multiplier": 20,
      "tick_size": 0.25,
      "tick_value": 5.00,
      "currency": "USD",
      "description": "E-mini NASDAQ-100",
      "exchange": "CME"
    },
    "micro": {
      "multiplier": 2,
      "tick_size": 0.25,
      "tick_value": 0.50,
      "currency": "USD",
      "description": "Micro E-mini NASDAQ-100",
      "exchange": "CME"
    }
  },
  "YM": {
    "standard": {
      "multiplier": 5,
      "tick_size": 1.0,
      "tick_value": 5.00,
      "currency": "USD",
      "description": "E-mini Dow ($5)",
      "exchange": "CBOT"
    },
    "micro": {
      "multiplier": 0.5,
      "tick_size": 1.0,
      "tick_value": 0.50,
      "currency": "USD",
      "description": "Micro E-mini Dow",
      "exchange": "CBOT"
    }
  },
  "GC": {
    "standard": {
      "multiplier": 100,
      "tick_size": 0.10,
      "tick_value": 10.00,
      "currency": "USD",
      "description": "Gold Futures (100 oz)",
      "exchange": "COMEX"
    },
    "micro": {
      "multiplier": 10,
      "tick_size": 0.10,
      "tick_value": 1.00,
      "currency": "USD",
      "description": "Micro Gold (10 oz)",
      "exchange": "COMEX"
    }
  },
  "CL": {
    "standard": {
      "multiplier": 1000,
      "tick_size": 0.01,
      "tick_value": 10.00,
      "currency": "USD",
      "description": "Crude Oil (1000 barrels)",
      "exchange": "NYMEX"
    }
  }
}
```

---

## Error Handling

### Error Handling Philosophy

1. **Fail Fast at Initialization**: Validate all configuration at `__init__`
2. **Graceful Degradation at Runtime**: Log errors, return None, continue processing
3. **Comprehensive Logging**: Log all errors with context (ticker, datetime, traceback)
4. **Default Fallbacks**: Use sensible defaults when possible (e.g., 0.20 volatility)

### Error Scenarios

#### 1. Missing Volatility Columns

**Scenario**: EWSD-252 column not found in features

**Handling**:
```python
def _calculate_blended_volatility(self, ticker, features_df):
    ewsd_cols = [col for col in features_df.columns if 'ewsd' in col.lower()]
    
    if not ewsd_cols:
        logger.warning(
            f"No EWSD column for {ticker.name}. Using default 0.20"
        )
        return {ticker.name: 0.20}  # Default 20% annual volatility
```

**Impact**: Uses default volatility, continues processing

#### 2. Portfolio Prediction Failure

**Scenario**: `portfolio.predict()` raises exception

**Handling**:
```python
try:
    positions_df = self.portfolio.predict(...)
except Exception as e:
    logger.error(
        f"Portfolio prediction failed for {ticker.name}: {e}",
        exc_info=True
    )
    return None  # Skip this prediction, wait for next candle
```

**Impact**: Returns None for this candle, continues for next candles

#### 3. PositionSizer Failure

**Scenario**: `position_sizer.calculate_positions()` raises exception

**Handling**:
```python
try:
    if self.position_sizer:
        contracts_df = self.position_sizer.calculate_positions(positions_df)
        position_dict.update(contracts_df.to_dict('records')[0])
except Exception as e:
    logger.error(
        f"PositionSizer failed for {ticker.name}: {e}. "
        f"Returning position fractions only."
    )
    # Continue with just position_fraction (no contracts)
```

**Impact**: Returns position fractions without contracts

#### 4. Invalid Ticker

**Scenario**: Ticker not recognized or not in portfolio

**Handling**:
```python
try:
    self._initialize_ticker_bias_nodes(ticker)
except Exception as e:
    logger.error(
        f"Failed to initialize {ticker.name}: {e}. Skipping ticker."
    )
    return None
```

**Impact**: Skips this ticker entirely

#### 5. Bias Node Initialization Failure

**Scenario**: `helpers.create_bias_node()` fails

**Handling**:
```python
try:
    bias_node = helpers.create_bias_node(module_name, ticker, tf, params)
except Exception as e:
    logger.error(
        f"Failed to create bias node {module_name} for {ticker.name}: {e}. "
        f"Skipping this node."
    )
    continue  # Skip this node, continue with others
```

**Impact**: Skips faulty node, continues with others

### Logging Levels

- **ERROR**: Critical failures that prevent position generation
- **WARNING**: Non-critical issues with fallback behavior
- **INFO**: Normal operations (initialization, predictions)
- **DEBUG**: Detailed diagnostic information

---

## Testing Strategy

### Test Pyramid

```
        /\
       /  \        E2E Tests (10%)
      /    \       - Full pipeline tests
     /------\      - Multi-ticker scenarios
    /        \     
   /          \    Integration Tests (30%)
  /            \   - MLManager + Portfolio
 /              \  - Portfolio + Ensemble
/--------------  \ 
|                | Unit Tests (60%)
|                | - Individual methods
|                | - Edge cases
|                | - Error handling
```

### Unit Tests

**Coverage**: 60% of tests

**Scope**: Individual methods, pure functions, edge cases

**Examples**:
- `test_ensemble_get_required_bias_node_specs_single_feature()`
- `test_portfolio_includes_atr_ewsd()`
- `test_mlmanager_calculate_blended_volatility()`
- `test_position_sizer_from_config_invalid_ticker()`

### Integration Tests

**Coverage**: 30% of tests

**Scope**: Component interactions, data flow between layers

**Examples**:
- `test_mlmanager_portfolio_integration_single_ticker()`
- `test_portfolio_ensemble_bias_node_discovery()`
- `test_mlmanager_position_sizer_integration()`

### End-to-End Tests

**Coverage**: 10% of tests

**Scope**: Complete pipeline, real-world scenarios

**Examples**:
- `test_full_pipeline_candles_to_contracts()`
- `test_multi_ticker_simultaneous_streaming()`
- `test_graceful_degradation_under_errors()`

---

## Operating Modes

MLManager supports two distinct modes of operation:

### Mode 1: Standalone (Feature Extraction Only)

**Use Case**: Research, feature engineering, backtesting

**Configuration**: No portfolio required, explicit bias_node_specs

**Behavior**:
- Bias nodes update on every candle
- Features are computed and stored (if build_matrix=True)
- No position generation
- `add_candle()` always returns None

**When to use**:
- Exploring new features
- Building training datasets
- Running offline backtests
- Feature validation

**Example**:
```python
ml_manager = MLManager(
    base_tf=TimeFrame.D,
    bias_node_specs=[
        {'module_name': 'rsi', 'timeframes': [TimeFrame.D], 'params': {'lookback': 14}},
        {'module_name': 'atr', 'timeframes': [TimeFrame.D], 'params': {'lookback': 252}}
    ],
    build_matrix=True  # Store features for later analysis
)

# Stream candles, extract features
for candle in candles:
    ml_manager.add_candle(candle, Ticker.ES, TimeFrame.D)

# Get feature matrix
features = ml_manager.get_feature_matrix(Ticker.ES)
```

### Mode 2: Production (Features + Positions)

**Use Case**: Live trading, paper trading, production backtests

**Configuration**: Portfolios required (one or more by timeframe), optional position_sizer

**Behavior**:
- Bias nodes auto-discovered from portfolio ensembles
- Features computed and passed to appropriate portfolio based on timeframe
- Position fractions generated
- Optionally converted to contracts
- `add_candle()` returns positions when auto_predict=True and portfolios[tf] exists

**When to use**:
- Live trading systems
- Production deployment
- Real-time position generation
- Integration with brokers/exchanges
- Multi-timeframe trading strategies

**Example**:
```python
ml_manager = MLManager(
    portfolios={
        TimeFrame.D: daily_portfolio,   # Auto-discovers bias nodes
        TimeFrame.W: weekly_portfolio   # Separate portfolio for weekly
    },
    position_sizer=my_sizer,
    auto_predict=True  # Automatic prediction when candles arrive
)

# Stream candles, get positions (routes to appropriate portfolio)
for candle in candles:
    result = ml_manager.add_candle(candle, Ticker.ES, TimeFrame.D)
    if result:
        print(f"Position: {result['contracts']} contracts")
```

### Mode 3: Hybrid (Custom Bias Nodes + Portfolio(s))

**Use Case**: Production with additional experimental features

**Configuration**: Both portfolios and explicit bias_node_specs

**Behavior**:
- Uses explicit bias_node_specs (ignores portfolio's specs)
- Still generates positions via portfolios
- Allows testing new features in production

**Example**:
```python
ml_manager = MLManager(
    portfolios={TimeFrame.D: my_portfolio},
    bias_node_specs=[
        # Include portfolio's features manually
        {'module_name': 'rsi', 'timeframes': [TimeFrame.D], 'params': {'lookback': 2}},
        # Plus experimental features
        {'module_name': 'custom_indicator', 'timeframes': [TimeFrame.D], 'params': {...}}
    ],
    position_sizer=my_sizer
)
```

---

## Usage Examples

### Example 1: Standalone Mode (Feature Extraction for Research)

```python
from utils.enums import Ticker, TimeFrame
from feature_extraction.ml_manager import MLManager
from utils.models import Candle
import helpers

# 1. Create MLManager in standalone mode (no portfolios)
ml_manager = MLManager(
    bias_node_specs=[
        {'module_name': 'rsi', 'timeframes': [TimeFrame.D], 'params': {'lookback': 14}},
        {'module_name': 'atr', 'timeframes': [TimeFrame.D], 'params': {'lookback': 252}},
        {'module_name': 'ewsd', 'timeframes': [TimeFrame.D], 'params': {'lookback': 252}},
        {'module_name': 'momentum', 'timeframes': [TimeFrame.D], 'params': {'lookback': 20}}
    ],
    build_matrix=True  # Store features for analysis
)

# 2. Stream candles to extract features
data = helpers.load_numpy_data(Ticker.ES, TimeFrame.D)

for row in data:
    candle = Candle(
        open=row['open'],
        high=row['high'],
        low=row['low'],
        close=row['close'],
        volume=row['volume'],
        datetime=row['datetime']
    )
    
    # No positions returned (standalone mode)
    result = ml_manager.add_candle(candle, Ticker.ES, TimeFrame.D)
    assert result is None

# 3. Get extracted features
features_df = ml_manager.get_feature_matrix(Ticker.ES)
print(f"Extracted {len(features_df)} rows with {len(features_df.columns)} features")

# 4. Use for training, analysis, etc.
# ... your research code here ...
```

### Example 2: Production Mode - Single Portfolio (Daily Trading)

```python
from utils.enums import Ticker, TimeFrame
from ensemble.portfolio import Portfolio
from ensemble.diversified_ensemble import DiversifiedEnsemble
from feature_extraction.ml_manager import MLManager
from utils.models import Candle
import helpers

# 1. Load ensemble from vault
ensemble = DiversifiedEnsemble(
    ensemble_dir='vault/STOCK_INDICES_D_LONG'
)

# 2. Create daily trading portfolio (single timeframe)
daily_portfolio = Portfolio(
    ensembles=[ensemble],  # List of ensembles, not dict
    trading_timeframe=TimeFrame.D,  # Trades on daily candles
    target_volatility=0.20,
    dm=1.5,
    max_position_pct=2.0
)

# 3. Create MLManager with daily portfolio
ml_manager = MLManager(
    portfolios={TimeFrame.D: daily_portfolio},  # Dict of portfolios by timeframe
    auto_predict=True,  # Automatic predictions when daily candles arrive
    build_matrix=False
)

# 4. Stream candles
data = helpers.load_numpy_data(Ticker.ES, TimeFrame.D)

for row in data:
    candle = Candle(
        open=row['open'],
        high=row['high'],
        low=row['low'],
        close=row['close'],
        volume=row['volume'],
        datetime=row['datetime']
    )
    
    # Add candle (auto-predicts positions because TimeFrame.D has a portfolio)
    result = ml_manager.add_candle(candle, Ticker.ES, TimeFrame.D)
    
    if result:
        print(f"Position: {result['ticker']} - "
              f"TF: {result['timeframe']} - "
              f"Fraction: {result['position_fraction']:.2%} - "
              f"Forecast: {result['forecast_score']:.2f}")
```

### Example 3: Production Mode - Multiple Portfolios (Multi-Timeframe Trading)

```python
from utils.enums import Ticker, TimeFrame
from ensemble.portfolio import Portfolio
from ensemble.diversified_ensemble import DiversifiedEnsemble
from feature_extraction.ml_manager import MLManager

# 1. Create separate ensembles for each trading timeframe
daily_ensemble = DiversifiedEnsemble(
    ensemble_dir='vault/STOCK_INDICES_D_LONG'
)

weekly_ensemble = DiversifiedEnsemble(
    ensemble_dir='vault/STOCK_INDICES_W_LONG'
)

# 2. Create separate portfolios for each trading timeframe
daily_portfolio = Portfolio(
    ensembles=[daily_ensemble],
    trading_timeframe=TimeFrame.D,  # Trades daily
    target_volatility=0.20,
    dm=2.0
)

weekly_portfolio = Portfolio(
    ensembles=[weekly_ensemble],
    trading_timeframe=TimeFrame.W,  # Trades weekly
    target_volatility=0.15,  # Lower target for longer timeframe
    dm=1.5
)

# 3. Create MLManager with multiple portfolios
ml_manager = MLManager(
    portfolios={
        TimeFrame.D: daily_portfolio,   # Routes daily candles here
        TimeFrame.W: weekly_portfolio   # Routes weekly candles here
    },
    auto_predict=True
)

# 4. Stream candles (both timeframes)
daily_data = helpers.load_numpy_data(Ticker.ES, TimeFrame.D)
weekly_data = helpers.load_numpy_data(Ticker.ES, TimeFrame.W)

# Process daily candles
for row in daily_data:
    candle = Candle(...)
    result = ml_manager.add_candle(candle, Ticker.ES, TimeFrame.D)
    
    if result:
        print(f"Daily position: {result['position_fraction']:.2%}")

# Process weekly candles
for row in weekly_data:
    candle = Candle(...)
    result = ml_manager.add_candle(candle, Ticker.ES, TimeFrame.W)
    
    if result:
        print(f"Weekly position: {result['position_fraction']:.2%}")
```

### Example 4: Production Mode with Manual Prediction

```python
# Setup with auto_predict=False
ml_manager = MLManager(
    portfolios={TimeFrame.D: daily_portfolio},
    auto_predict=False  # Manual control over prediction timing
)

# Stream candles without triggering predictions
for candle in candles:
    ml_manager.add_candle(candle, Ticker.ES, TimeFrame.D)

# Manually trigger prediction when ready (must specify timeframe)
result = ml_manager.predict_positions(Ticker.ES, TimeFrame.D)
print(f"Position: {result}")
```

### Example 5: Multi-Ticker with PositionSizer

```python
from execution.position_sizer import PositionSizer, RoundingMethod

# 1. Setup portfolio with multiple ensembles (single timeframe)
portfolio = Portfolio(
    ensembles=[
        DiversifiedEnsemble(ensemble_dir='vault/STOCK_INDICES_D_LONG'),
        DiversifiedEnsemble(ensemble_dir='vault/COMMODITIES_D_LONG')
    ],
    trading_timeframe=TimeFrame.D,
    target_volatility=0.20,
    dm=2.0
)

# 2. Create PositionSizer from config
position_sizer = PositionSizer.from_config(
    config_path='deployment/config/contract_specs.json',
    capital=1_000_000,
    ticker_variants={
        'ES': 'micro',
        'NQ': 'micro',
        'GC': 'micro'
    },
    prices={
        'ES': 4800.0,
        'NQ': 16000.0,
        'GC': 2000.0
    },
    rounding_method=RoundingMethod.ROUND
)

# 3. Create MLManager with PositionSizer
ml_manager = MLManager(
    portfolios={TimeFrame.D: portfolio},
    position_sizer=position_sizer,
    auto_predict=True
)

# 4. Stream multi-ticker data
tickers = [Ticker.ES, Ticker.NQ, Ticker.GC]

for ticker in tickers:
    data = helpers.load_numpy_data(ticker, TimeFrame.D)
    
    for row in data:
        candle = Candle.from_numpy_row(row)
        result = ml_manager.add_candle(candle, ticker, TimeFrame.D)
        
        if result:
            print(f"{result['datetime']}: {result['ticker']} - "
                  f"{result['contracts']} contracts @ "
                  f"${result['notional_value']:,.0f}")
```

### Example 5: Multi-Timeframe Processing

```python
from utils.candle_fetcher import CandleFetcher

# 1. Setup portfolio with W and M features
portfolio = Portfolio(
    ensembles={
        TimeFrame.D: [ensemble_d],
        TimeFrame.W: [ensemble_w],
        TimeFrame.M: [ensemble_m]
    },
    target_volatility=0.20
)

# 2. Create MLManager
ml_manager = MLManager(
    base_tf=TimeFrame.D,
    portfolio=portfolio,
    auto_predict=True
)

# 3. Create candle fetcher for W and M data
candle_fetcher = CandleFetcher(
    ticker=Ticker.ES,
    tfs=[TimeFrame.W, TimeFrame.M]
)

# 4. Stream D candles, fetch W/M as needed
data = helpers.load_numpy_data(Ticker.ES, TimeFrame.D)

for row in data:
    candle = Candle.from_numpy_row(row)
    
    # Check if W or M candle needed (before D candle to update state)
    w_candle = candle_fetcher.get_candle(TimeFrame.W, candle.datetime)
    if w_candle:
        ml_manager.add_candle(w_candle, Ticker.ES, TimeFrame.W)
    
    m_candle = candle_fetcher.get_candle(TimeFrame.M, candle.datetime)
    if m_candle:
        ml_manager.add_candle(m_candle, Ticker.ES, TimeFrame.M)
    
    # Add D candle (triggers prediction since base_tf=D)
    result = ml_manager.add_candle(candle, Ticker.ES, TimeFrame.D)
    
    if result:
        print(f"Position: {result}")
```

### Example 6: With Telegram Export

```python
from deployment.telegram_notifier import TelegramNotifier

# 1. Setup components
portfolio = Portfolio(...)
ml_manager = MLManager(
    base_tf=TimeFrame.D,
    portfolio=portfolio,
    position_sizer=my_sizer,
    auto_predict=True
)
telegram = TelegramNotifier(token='...', chat_id='...')

# 2. Stream and export
for row in data:
    candle = Candle.from_numpy_row(row)
    result = ml_manager.add_candle(candle, Ticker.ES, TimeFrame.D)
    
    if result and result.get('contracts', 0) != 0:
        # Send to Telegram
        message = (
            f"🔔 New Position\n"
            f"Ticker: {result['ticker']}\n"
            f"Contracts: {result['contracts']}\n"
            f"Notional: ${result['notional_value']:,.0f}\n"
            f"Forecast: {result['forecast_score']:.2f}"
        )
        telegram.send_message(message)
```

---

## Key Design Principles Applied

1. **Functional Core, Imperative Shell**: 
   - Portfolio/PositionSizer are pure functions
   - MLManager handles I/O and state management

2. **Separation of Concerns**: 
   - MLManager: Data ingestion and bias node management
   - Portfolio: Risk management and position sizing
   - PositionSizer: Contract conversion
   - Ensemble: Signal combination
   - BaseModel: Binary signal generation

3. **Composition Over Inheritance**: 
   - MLManager owns Portfolio (not inherits)
   - Portfolio owns Ensembles (not inherits)

4. **Type Safety**: 
   - Full type hints on all methods
   - Use enums (Ticker, TimeFrame, Direction)
   - Frozen dataclasses for immutable data

5. **Graceful Degradation**: 
   - Log errors, don't crash
   - Return None on failures
   - Use sensible defaults

6. **DRY (Don't Repeat Yourself)**: 
   - Auto-discovery eliminates manual configuration
   - Centralized error handling patterns
   - Reusable components across layers

7. **Single Responsibility Principle**:
   - Each class has one clear purpose
   - Methods do one thing well
   - Clear boundaries between components

---

## Total Estimated Effort

| Phase | Description | Hours |
|-------|-------------|-------|
| 1 | Ensemble Bias Node Discovery API | 2-3 |
| 2 | Portfolio Bias Node Discovery | 1 |
| 3 | Contract Specs Configuration | 2 |
| 4 | MLManager Multi-Ticker State | 4-5 |
| 5 | Portfolio Integration (covered in Phase 4) | - |
| 6 | Error Handling and Logging | 2-3 |
| 7 | Integration Tests | 3-4 |
| 8 | Documentation | 2 |
| **Total** | | **20-24 hours** |

**Timeline**: ~3-4 days of focused work

---

## Success Criteria

- ✅ MLManager can process multi-ticker streaming data
- ✅ Auto-discovers bias nodes from Portfolio ensembles
- ✅ Generates position fractions via Portfolio
- ✅ Optionally converts to contracts via PositionSizer
- ✅ Comprehensive error handling and logging
- ✅ Full unit test coverage (>90%)
- ✅ Integration tests pass
- ✅ End-to-end pipeline works in real-time streaming mode
- ✅ Documentation complete with examples
- ✅ Ready for production deployment

---

**End of Specification Document**
