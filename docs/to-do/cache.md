---
name: Bias Node Caching
overview: Implement a persistent, vectorized caching system for bias node outputs using parquet files. The user explicitly chooses which bias nodes to cache (via CacheManager). Cached bias nodes load on instantiation and expose vectorized access via get_cached_values(); streaming uses add_candle (one candle at a time). Naming: vectorized_fit / vectorized_predict for backtesting (cached path); add_candle, stream_fit, stream_predict for live/one-candle-at-a-time. Vectorized methods warn and raise CacheMissError if cache is missing. use_cache selects which path fit/predict and fit_from_candles/predict_from_candles dispatch to.
todos:
  - id: cache_storage
    content: Create BiasNodeCache class for parquet I/O
    status: pending
  - id: cache_manager
    content: Create CacheManager for concurrent population (overwrite if same params)
    status: pending
  - id: bias_node_cache
    content: Add cache loading and get_cached_values() to BiasNode
    status: pending
  - id: base_model_cache
    content: Add vectorized_fit/vectorized_predict and stream_fit/stream_predict to BaseModel
    status: pending
  - id: ensemble_cache
    content: Add vectorized_fit_from_candles/stream_fit_from_candles and vectorized_predict_from_candles/stream_predict_from_candles; use_cache propagation
    status: pending
  - id: cache_tests
    content: Create comprehensive test suite for caching system
    status: pending
isProject: false
---

# Bias Node Caching System

## Architecture Overview

The new caching system will store deterministic bias node outputs in parquet files organized by `cache/{module_name}/{ticker}_{tf}_{params_hash}.parquet`. **The user is responsible for choosing which bias nodes to cache** (via CacheManager or CLI); only explicitly populated caches exist, avoiding an explosion of entries during EDA. When bias nodes are instantiated, they load their cache if it exists. **Vectorized path** (backtesting / speed): `get_cached_values(start, end)` at BiasNode; `vectorized_fit` / `vectorized_predict` (BaseModel); `vectorized_fit_from_candles` / `vectorized_predict_from_candles` (Portfolio/Ensemble). **Streaming path** (one candle at a time): `add_candle(candle)` at BiasNode; `stream_fit` / `stream_predict` (BaseModel); `stream_fit_from_candles` / `stream_predict_from_candles` (Portfolio/Ensemble). `fit`/`predict` and `fit_from_candles`/`predict_from_candles` dispatch based on `use_cache`. **On cache miss**, vectorized methods **log a warning and raise CacheMissError** (no silent fallback).

```
Pipeline: Candles → [Cache or BiasNode Stream] → Features → BaseModel → Ensemble → Portfolio
                           ↑
                    CacheManager (user runs to populate chosen bias nodes; overwrites if same params)
```

## Key Components

### 1. Cache Storage (`utils/bias_node_cache.py`)

**New module** providing cache I/O operations:

```python
class BiasNodeCache:
    """
    Handles loading/saving bias node outputs to/from parquet files.
    
    Cache key: (module_name, params, ticker, tf)
    File location: cache/{module_name}/{ticker}_{tf}_{params_hash}.parquet
    Schema: datetime (datetime64[ns]), value (float64)
    """
    
    def __init__(self, module_name: str, params: Dict, ticker: Ticker, tf: TimeFrame):
        self.cache_path = self._build_cache_path(module_name, params, ticker, tf)
        self._in_memory_cache: Optional[pd.Series] = None
        
    def load(self) -> pd.Series:
        """Load cache from parquet into memory, indexed by datetime."""
        
    def save(self, data: pd.Series | pd.DataFrame) -> None:
        """Save to parquet (datetime index). Series for single-output; DataFrame for multi-output (e.g. signal, signalBool)."""
        
    def exists(self) -> bool:
        """Check if cache file exists."""
        
    def get_values(self, start: datetime, end: datetime) -> pd.Series:
        """Get cached values for datetime range (vectorized)."""
        
    @staticmethod
    def _hash_params(params: Dict) -> str:
        """Generate deterministic hash for params dict."""
```

**Cache file structure**:

```
cache/
├── rsi/
│   ├── ES_D_lookback_14.parquet
│   ├── ES_D_lookback_2.parquet
│   └── NQ_D_lookback_14.parquet
├── ewmac/
│   ├── ES_D_spanFast_16_spanSlow_64.parquet
│   └── NQ_W_spanFast_8_spanSlow_32.parquet
└── buy_hold/
    ├── ES_D.parquet
    └── NQ_D.parquet
```

---

### 2. Cache Manager (`utils/cache_manager.py`)

**New module** for orchestrating cache population:

```python
class CacheManager:
    """
    Cache manager for populating bias node caches. User explicitly passes which
    bias nodes to cache (vault dir or specs); no automatic discovery of "all"
    nodes, to avoid cache explosion during EDA.
    
    Workflow:
    1. User provides bias node specs (from vault or explicit list)
    2. For each (module_name, params, ticker, tf): if cache exists with same
       params, overwrite (user is assumed to have updated the bias node);
       otherwise create or update.
    3. Populate caches concurrently (one candle stream per ticker/tf/module)
    """
    
    def populate_cache(
        self,
        bias_node_specs: List[Dict[str, Any]],
        tickers: List[Ticker],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_workers: int = 4,
        overwrite_existing: bool = True
    ) -> Dict[str, Any]:
        """
        Populate cache for the given bias node specs (user-chosen only).
        
        If overwrite_existing=True (default): re-running with the same params
        overwrites the existing cache entry (assumes user updated the bias node).
        
        Returns dict with:
        - populated: List of cache paths created or overwritten
        - already_cached: List of cache paths that existed and were skipped (if overwrite_existing=False)
        - errors: Dict of spec -> error message
        """
        
    def populate_cache_for_vault(
        self,
        vault_ensemble_dir: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Populate cache for all bias nodes in a vault ensemble."""
        
    def _populate_single_cache(
        self,
        module_name: str,
        params: Dict,
        ticker: Ticker,
        tf: TimeFrame,
        start_date: datetime,
        end_date: datetime
    ) -> str:
        """Populate a single bias node cache by streaming candles."""
```

**Concurrent population strategy**:

- Use `concurrent.futures.ThreadPoolExecutor` or `ProcessPoolExecutor`
- Each worker processes one (ticker, tf, module_name, params) combination
- Load candles from `data/ohlc_data/{ticker}/{tf}_{ticker}.parquet` (same convention as `utils.helpers`). If file missing, record in `errors` and skip that spec.
- Stream candles through bias node, collect outputs (include warmup).
- Save to cache parquet file.

---

### 3. BiasNode Modifications (`[nodes/__init__.py](nodes/__init__.py)`)

**Naming**: **Vectorized** (backtesting) = `get_cached_values(start, end)` at BiasNode; `vectorized_fit` / `vectorized_predict` at BaseModel; `vectorized_fit_from_candles` / `vectorized_predict_from_candles` at Portfolio/Ensemble. **Streaming** (one candle at a time) = `add_candle(candle)` at BiasNode; `stream_fit` / `stream_predict` at BaseModel; `stream_fit_from_candles` / `stream_predict_from_candles` at Portfolio/Ensemble. When vectorized path is used and cache is missing, **warn and raise CacheMissError** (no fallback). Add cache support to base BiasNode class:

```python
class BiasNode(ABC):
    def __init__(self, ticker: Ticker, tf: TimeFrame):
        # Existing code...
        
        # Cache support (actual load happens when subclass calls _init_cache_after_params)
        self._bias_node_cache: Optional[BiasNodeCache] = None
        self._cache_loaded: bool = False
    
    def _init_cache_after_params(self) -> None:
        """
        Initialize cache after params are set by subclass. Subclasses MUST call
        this at the end of their __init__ after setting module_name and params.
        Base class does not call it (params are not set yet in base __init__).
        """
        if hasattr(self, 'module_name') and hasattr(self, 'params'):
            from utils.bias_node_cache import BiasNodeCache
            self._bias_node_cache = BiasNodeCache(
                self.module_name, self.params, self.ticker, self.tf
            )
            # Try to load cache
            if self._bias_node_cache.exists():
                self._bias_node_cache.load()
                self._cache_loaded = True
    
    def get_cached_values(
        self,
        start: datetime,
        end: datetime,
        require_cache: bool = True
    ) -> Optional[pd.Series]:
        """
        Get cached values for datetime range (vectorized).
        
        When called from the vectorized path (vectorized_fit / vectorized_fit_from_candles),
        require_cache is True: on cache miss, log a warning and raise CacheMissError
        (no silent fallback). This forces the user to either cache first or use streaming.
        
        Parameters
        ----------
        start, end : datetime
            Datetime range
        require_cache : bool
            If True (default for vectorized path): warn and raise CacheMissError if cache missing/incomplete.
            
        Returns
        -------
        pd.Series or None
            Cached values indexed by datetime, or None only if require_cache=False and cache miss.
            
        Raises
        ------
        CacheMissError
            If require_cache=True and cache is missing or incomplete for range.
        """
```

**Exception class**:

```python
class CacheMissError(Exception):
    """Raised when cache is required but missing."""
    def __init__(self, module_name, params, ticker, tf, date_range):
        self.module_name = module_name
        self.params = params
        self.ticker = ticker
        self.tf = tf
        self.date_range = date_range
```

---

### 4. BaseModel Modifications (`[feature_selection/base_models/feature_base_model.py](feature_selection/base_models/feature_base_model.py)`)

**Vectorized path** = `vectorized_fit` / `vectorized_predict` (backtesting, uses cache). **Streaming path** = `stream_fit` / `stream_predict` (one candle at a time, uses add_candle). `fit` / `predict` dispatch based on `use_cache`:

```python
class BaseModel:
    def __init__(self, ..., use_cache: bool = True):
        # Existing code...
        self.use_cache = use_cache

    def fit(
        self,
        candles_df: pd.DataFrame,
        target_data: pd.Series,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> 'BaseModel':
        """Dispatch to vectorized_fit or stream_fit based on self.use_cache."""
        if self.use_cache:
            return self.vectorized_fit(candles_df, target_data, start_date, end_date)
        return self.stream_fit(candles_df, target_data)

    def vectorized_fit(
        self,
        candles_df: pd.DataFrame,
        target_data: pd.Series,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> 'BaseModel':
        """
        Fit using bias node cache (vectorized, for backtesting). Uses get_cached_values(require_cache=True) per node.
        On cache miss: log warning and raise CacheMissError (no fallback). User must cache first or use stream_fit.
        """
        feature_data = self._get_features_from_cache(candles_df, start_date, end_date)
        # ... fit on feature_data

    def stream_fit(self, candles_df: pd.DataFrame, target_data: pd.Series) -> 'BaseModel':
        """Fit by streaming candles through add_candle() (one candle at a time). For live trading."""
        feature_data = self._get_features_from_streaming(candles_df)
        # ... fit on feature_data

    def predict(...) -> pd.Series:
        """Dispatch to vectorized_predict or stream_predict based on self.use_cache."""

    def vectorized_predict(self, candles_df, start_date=None, end_date=None) -> pd.Series:
        """Predict using cached bias node outputs (vectorized, for backtesting)."""

    def stream_predict(self, candles_df: pd.DataFrame) -> pd.Series:
        """Predict by streaming candles (one candle at a time, for live trading)."""

    def _get_features_from_cache(
        self,
        candles_df: pd.DataFrame,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> pd.Series:
        """Extract features from bias node cache (vectorized). Raises CacheMissError if cache missing."""

    def _get_features_from_streaming(self, candles_df: pd.DataFrame) -> pd.Series:
        """Extract features by streaming candles (add_candle, one at a time)."""
```

---

### 5. Portfolio/Ensemble Modifications

**Naming**: **vectorized_fit_from_candles** / **vectorized_predict_from_candles** (backtesting, cached). **stream_fit_from_candles** / **stream_predict_from_candles** (one candle at a time). **On cache miss**, vectorized methods warn and raise CacheMissError (no fallback).

`[ensemble/portfolio.py](ensemble/portfolio.py)`:

```python
class Portfolio:
    def __init__(self, ..., use_cache: bool = True):
        self.use_cache = use_cache

    def fit_from_candles(self, candles_df, target_data, start_date=None, end_date=None):
        """Dispatch to vectorized_fit_from_candles or stream_fit_from_candles based on use_cache."""
        if self.use_cache:
            return self.vectorized_fit_from_candles(candles_df, target_data, start_date, end_date)
        return self.stream_fit_from_candles(candles_df, target_data)

    def vectorized_fit_from_candles(self, candles_df, target_data, start_date=None, end_date=None):
        """Vectorized fit using cached bias node outputs (backtesting). On cache miss: warn and raise CacheMissError."""
        for ensemble in self.ensembles:
            ensemble.vectorized_fit_from_candles(
                candles_df, target_data, start_date=start_date, end_date=end_date
            )

    def stream_fit_from_candles(self, candles_df, target_data):
        """Stream fit (one candle at a time, live trading). Propagates to ensembles."""
        for ensemble in self.ensembles:
            ensemble.stream_fit_from_candles(candles_df, target_data)

    def predict_from_candles(self, ...):  # dispatch
    def vectorized_predict_from_candles(self, candles_df, start_date=None, end_date=None): ...
    def stream_predict_from_candles(self, candles_df): ...
```

`[ensemble/diversified_ensemble.py](ensemble/diversified_ensemble.py)`:

```python
class DiversifiedEnsemble:
    def fit_from_candles(self, candles_df, target_data, use_cache=True, start_date=None, end_date=None):
        if use_cache:
            return self.vectorized_fit_from_candles(candles_df, target_data, start_date, end_date)
        return self.stream_fit_from_candles(candles_df, target_data)

    def vectorized_fit_from_candles(self, candles_df, target_data, start_date=None, end_date=None):
        for base_model in self.base_models.values():
            base_model.vectorized_fit(candles_df, target_data, start_date, end_date)

    def stream_fit_from_candles(self, candles_df, target_data):
        for base_model in self.base_models.values():
            base_model.stream_fit(candles_df, target_data)

    # Same pattern: vectorized_predict_from_candles / stream_predict_from_candles
```

---

### 6. Cache Manager CLI (`utils/cache_manager.py`)

**Standalone script for cache population**:

```python
def main():
    """
    CLI for populating bias node cache.
    
    Usage:
        # Populate cache for specific vault ensemble
        python -m utils.cache_manager --vault vault/D/buy_hold_long
        
        # Populate cache for specific bias nodes
        python -m utils.cache_manager --specs bias_specs.json --tickers ES,NQ --start 2015-01-01
        
        # Populate all caches for a ticker/timeframe
        python -m utils.cache_manager --ticker ES --tf D
    """
```

---

### 7. Error Handling Flow (Warn and Crash)

**Vectorized methods never silently fall back.** When cache is missing or incomplete:

1. `BiasNode.get_cached_values(require_cache=True)` (default when called from vectorized_fit / vectorized_fit_from_candles) detects cache miss.
2. **Log a warning** (e.g. which bias nodes are missing cache for the requested range).
3. **Raise CacheMissError** with details (module, params, ticker, tf, date_range).
4. Error bubbles up through BaseModel → Ensemble → Portfolio.
5. Portfolio/Ensemble can catch and re-raise with context: "Cache miss for 3 bias nodes. Either: (1) Run CacheManager to populate cache, or (2) use stream_fit_from_candles / stream_predict_from_candles."
6. User either runs CacheManager and re-runs with vectorized path, or switches to streaming methods.

---

## Implementation Tasks

### Phase 1: Core Cache Infrastructure

1. Create `utils/bias_node_cache.py` with `BiasNodeCache` class and `CacheMissError` exception (define in this module; others import from here).
2. Create `utils/cache_manager.py` with `CacheManager` class and CLI
3. Create `cache/` directory structure (or make it configurable; see Spec Review)

### Phase 2: BiasNode Integration

1. Modify `[nodes/__init__.py](nodes/__init__.py)` - add cache attributes to `BiasNode`; add `_init_cache_after_params()` and `get_cached_values()`.
2. Update concrete bias nodes (RSI, EWMAC, etc.) to call `_init_cache_after_params()` at the end of their `__init__` after setting `module_name` and `params`.

### Phase 3: BaseModel Integration

1. Modify `[feature_selection/base_models/feature_base_model.py](feature_selection/base_models/feature_base_model.py)`
  - Add `use_cache` parameter
  - Add `vectorized_fit` / `vectorized_predict` and `stream_fit` / `stream_predict`; refactor existing logic into stream methods
  - Add `_get_features_from_cache()` and `_get_features_from_streaming()`
  - Make `fit()` and `predict()` dispatch based on `use_cache`

### Phase 4: Ensemble/Portfolio Integration

1. Add `vectorized_fit_from_candles` / `vectorized_predict_from_candles` and `stream_fit_from_candles` / `stream_predict_from_candles` to DiversifiedEnsemble and Portfolio
2. Add `use_cache` to Portfolio/Ensemble; make `fit_from_candles` / `predict_from_candles` dispatch to the cached or streaming pair
3. Add optional `start_date`/`end_date` to vectorized entry points only (needed for cache range and to avoid lookahead). Streaming path continues to derive range from `candles_df` when needed.

### Phase 5: Testing

1. Create `tests/test_bias_node_cache.py` - unit tests for cache I/O, hash generation, overwrite-on-same-params
2. Create `tests/test_cache_manager.py` - test concurrent population, user-chosen specs only
3. Create `tests/test_cache_integration.py` - integration test with Portfolio (lookahead bias, multi-ticker aggregation, cached vs streaming equivalence, cache overwrite)
4. Add regression tests comparing cached vs streaming outputs (should be identical)

---

## File Organization

**New files**:

- `utils/bias_node_cache.py` - BiasNodeCache class (~200 lines)
- `utils/cache_manager.py` - CacheManager + CLI (~400 lines)
- `tests/test_bias_node_cache.py` - Unit tests (~200 lines)
- `tests/test_cache_manager.py` - Manager tests (~150 lines)
- `tests/test_cache_integration.py` - Integration tests (~300 lines); no separate performance test suite

**Modified files**:

- `[nodes/__init__.py](nodes/__init__.py)` - Add cache support to BiasNode (get_cached_values; streaming via add_candle)
- `[feature_selection/base_models/feature_base_model.py](feature_selection/base_models/feature_base_model.py)` - Add vectorized_fit/vectorized_predict, stream_fit/stream_predict, use_cache dispatch
- `[ensemble/diversified_ensemble.py](ensemble/diversified_ensemble.py)` - Add cached/streaming method pairs, use_cache propagation
- `[ensemble/portfolio.py](ensemble/portfolio.py)` - Add use_cache, vectorized_fit_from_candles / stream_fit_from_candles (and predict pair)

**Cache directory** (auto-created):

```
cache/
├── rsi/
├── ewmac/
├── buy_hold/
├── momentum/
└── ...
```

---

## Design Decisions

### Cache Key Components

- **module_name**: Bias node module (e.g., 'rsi', 'ewmac')
- **params**: Dict of hyperparameters (e.g., `{'lookback': 14}`)
- **ticker**: Ticker enum (e.g., `Ticker.ES`)
- **tf**: TimeFrame enum (e.g., `TimeFrame.D`)
- **params_hash**: Deterministic hash of params dict (sorted keys, handles nested dicts/lists)

### Vectorized Access Pattern

Instead of:

```python
for candle in candles:
    output = bias_node.add_candle(candle)  # One at a time
```

With cache:

```python
cached_values = bias_node.get_cached_values(start_date, end_date)  # Bulk
if cached_values is None:
    raise CacheMissError(...)
```

### Naming Convention (vectorized vs stream)

- **Vectorized** (backtesting / speed): Uses cache, bulk access. **Stream** (live / one candle at a time): Uses `add_candle`, incremental.
- **BiasNode**: `get_cached_values(start, end)` (vectorized); `add_candle(candle)` (stream, one candle at a time).
- **BaseModel**: `fit` / `predict` (dispatch). **vectorized_fit** / **vectorized_predict** (backtesting). **stream_fit** / **stream_predict** (one candle at a time).
- **Portfolio / Ensemble**: **fit_from_candles** / **predict_from_candles** (dispatch). **vectorized_fit_from_candles** / **vectorized_predict_from_candles** (backtesting). **stream_fit_from_candles** / **stream_predict_from_candles** (one candle at a time).

So: *vectorized_* prefix = cached path for backtesting; *stream_* prefix + **add_candle** = one-candle-at-a-time for live trading.

### Cache Miss Handling (Warn and Crash)

Vectorized methods **warn then raise**; no silent fallback. User-friendly error:

```
CacheMissError: Cache missing for 3 bias nodes:
  1. rsi (lookback=14, ticker=ES, tf=D)
  2. ewmac (spanFast=16, spanSlow=64, ticker=ES, tf=D)
  3. buy_hold (ticker=NQ, tf=D)

Either:
  (1) Populate cache: python -m utils.cache_manager --vault vault/D/buy_hold_long --start 2015-01-01
  (2) Use streaming: stream_fit_from_candles() / stream_predict_from_candles()
```

### Cache Invalidation (Overwrite on Re-run)

- **Re-run with same params = overwrite**: If the user re-runs CacheManager with the exact same (module_name, params, ticker, tf) for an existing cache entry, treat it as an intentional update (e.g. bias node code changed). **Overwrite the existing cache** rather than skipping. No timestamp or version check required for this case.

### User Chooses What to Cache

- **Explicit selection only**: The user is responsible for choosing which bias nodes get cached (via vault dir, specs file, or API). CacheManager does not auto-discover or auto-populate "all" bias nodes. This avoids an explosion of cached entries during EDA over large param sets.

### Single-Ticker Caches, Aggregate On the Fly

- **Store single-ticker caches only**: Each cache file is per (module_name, params, ticker, tf). For multi-ticker ensembles, **aggregate on the fly** from these single-ticker caches (no pre-computed cross-ticker cache). More flexible and keeps cache keys simple.

### Cache Includes Warmup Outputs

- **Include warmup in cache**: Cache stores **all** outputs from the bias node, including the warmup period (e.g. NaN or default values before the node has enough history). Downstream binning/models handle invalid values; no special trimming of warmup in the cache layer.

### Parquet Schema

Each cache file stores:

- **Index**: `datetime` (datetime64[ns], timezone-aware UTC)
- **Columns**: 
  - For single-output bias nodes (RSI, BuyHold): `value` (float64)
  - For multi-output bias nodes (EWMAC): `signal` (float64), `signalBool` (int8)
- **Warmup**: All rows from first candle onward (including warmup period with NaN/defaults).

### Thread Safety

- Cache files are immutable after creation (no concurrent writes)
- CacheManager is the only writer; re-run with same params overwrites the file
- BiasNodes are read-only consumers
- No locking needed for read-only access

---

## Testing Strategy

### Unit Tests (`tests/test_bias_node_cache.py`)

1. Test cache path generation (params hashing, file naming)
2. Test parquet save/load roundtrip
3. Test datetime range filtering
4. Test handling of missing cache files

### Integration Tests (`tests/test_cache_integration.py`)

1. **Lookahead bias test**:
  - Populate cache with full history
  - Run backtest with cache (vectorized_fit_from_candles / vectorized_predict_from_candles), limiting datetime range
  - Verify no future data leaked into past predictions
  - Compare results with streaming mode (should match exactly)
2. **Multi-ticker aggregation test**:
  - Test BaseModel with multiple tickers using cached features (single-ticker caches, aggregate on the fly)
  - Verify aggregation matches streaming mode
3. **End-to-end test**:
  - Populate cache for vault ensemble (user-chosen specs)
  - Run Portfolio.fit_from_candles() with use_cache=True (dispatches to vectorized_fit_from_candles)
  - Compare performance metrics with streaming mode
  - Verify all predictions identical
4. **Cache overwrite test**: Re-run CacheManager with same params; verify existing cache is overwritten.

---

## Migration Path

### Step 1: Populate Initial Caches

```python
from utils.cache_manager import CacheManager

manager = CacheManager()

# Populate cache for all vault ensembles
for ensemble_dir in ['vault/D/buy_hold_long', 'vault/D/rsi_5_long', ...]:
    manager.populate_cache_for_vault(ensemble_dir, start_date='2015-01-01')
```

### Step 2: Enable Cache in Code

```python
# fit_from_candles dispatches to vectorized or stream based on use_cache
portfolio = Portfolio(ensembles=[...], use_cache=True)
portfolio.fit_from_candles(candles_df, target_data, start_date=..., end_date=...)  # uses vectorized_fit_from_candles

# Explicit stream (one candle at a time, live trading)
portfolio = Portfolio(ensembles=[...], use_cache=False)
portfolio.fit_from_candles(candles_df, target_data)  # uses stream_fit_from_candles

# Or call method pairs directly:
portfolio.vectorized_fit_from_candles(candles_df, target_data, start_date=..., end_date=...)
portfolio.stream_fit_from_candles(candles_df, target_data)
```

### Step 3: Verify Equivalence

Run all existing tests with `use_cache=True` and `use_cache=False`, compare outputs.

---

## Example Usage

### Populating Cache

```python
from utils.cache_manager import CacheManager
from utils.enums import Ticker, TimeFrame

manager = CacheManager()

# Option 1: Populate from vault ensemble
result = manager.populate_cache_for_vault(
    vault_ensemble_dir='vault/D/buy_hold_long',
    start_date='2015-01-01',
    end_date='2025-01-01'
)
print(f"Populated: {len(result['populated'])}, Already cached: {len(result['already_cached'])}")

# Option 2: Populate specific bias nodes
bias_specs = [
    {'module_name': 'rsi', 'timeframes': [TimeFrame.D], 'params': {'lookback': 14}},
    {'module_name': 'ewmac', 'timeframes': [TimeFrame.D], 'params': {'spanFast': 16, 'spanSlow': 64}}
]
result = manager.populate_cache(
    bias_node_specs=bias_specs,
    tickers=[Ticker.ES, Ticker.NQ],
    start_date='2015-01-01'
)
```

### Using Cache in Portfolio

```python
from ensemble import Portfolio, DiversifiedEnsemble

# Create portfolio with cache enabled (default)
portfolio = Portfolio(ensembles=[...], use_cache=True)

try:
    # Fit with cache (dispatches to vectorized_fit_from_candles)
    portfolio.fit_from_candles(candles_df, target_data, start_date='2020-01-01', end_date='2023-12-31')
    # Predict with cache (dispatches to vectorized_predict_from_candles)
    positions = portfolio.predict_from_candles(test_candles, start_date='2024-01-01', end_date='2024-12-31')
except CacheMissError as e:
    print(f"Cache miss: {e}")
    print(f"Run: python -m utils.cache_manager --vault {ensemble_dir}")
```

### Disabling Cache (Live Trading)

```python
# Live trading - use stream (one candle at a time)
portfolio = Portfolio(ensembles=[...], use_cache=False)
portfolio.fit_from_candles(candles_df, target_data)  # uses stream_fit_from_candles / add_candle
```

---

## Performance Expectations

### Speedup Estimates

Current bottleneck (from docs): "211k Python function calls (BaseModel → BiasNode.add_candle → _compute_candle)"

With cache:

- **Cache hit**: ~100x faster (single parquet read + datetime filter vs 211k Python calls)
- **Cache miss**: First run same speed (must populate), subsequent runs 100x faster
- **Memory**: ~1-10 MB per bias node cache (depends on history length)

### Example: Portfolio with 8 Base Models

- **Streaming mode**: 211k candle → bias node calls
- **Cache mode**: 8 parquet reads + datetime filtering (vectorized pandas)
- **Expected speedup**: 50-100x for fit/predict operations

---

## Spec Review: Concerns & Improvements

### Resolved / Clarified in spec

- **Cache path**: Matches existing layout: `data/ohlc_data/{ticker}/{tf}_{ticker}.parquet` (see `utils/helpers.py`). CacheManager should use the same convention or a shared helper.
- **start_date / end_date**: Currently `fit_from_candles` / `predict_from_candles` do not take date range; spec adds them for the **vectorized** path only (needed for cache range and to avoid lookahead). Streaming path can ignore or derive range from `candles_df`; document that.
- **CacheMissError location**: Spec says "Add CacheMissError" but not where. Recommendation: define in `utils/bias_node_cache.py` (or a small `utils/cache_errors.py`) and re-export from `utils.bias_node_cache` so BaseModel/Portfolio can import from one place.

### Concerns to watch during implementation

1. **BiasNode.__init__ and _init_cache_after_params**: Base `BiasNode.__init__` runs before subclass sets `module_name`/`params`, so `_init_cache_after_params()` in base `__init__` will typically see no cache. Spec says "Called by subclasses in their __init__ after setting params"—so subclasses must call `_init_cache_after_params()` again after setting params. Make this explicit: base should not call it, or base calls it and subclasses must call it again after setting params. Recommendation: only subclasses call `_init_cache_after_params()` after setting `module_name` and `params`; document in spec.
2. **BiasNodeCache.save signature**: Spec shows `save(self, data: pd.Series)`. Multi-output nodes (EWMAC) need DataFrame (e.g. `signal`, `signalBool`). Either support `pd.DataFrame | pd.Series` or always use DataFrame with a single column for single-output. Recommendation: support both; or standardize on DataFrame (one column = single-output).
3. **Params hash stability**: `_hash_params` must be deterministic across runs and processes (same params → same hash). Use sorted keys, canonical JSON or similar; avoid Python dict repr. Spec says "sorted keys, handles nested dicts/lists"—implementation must ensure no dependence on iteration order or float repr.
4. **Concurrent overwrite**: With `overwrite_existing=True`, two processes populating the same cache key can race. Acceptable if "last write wins"; if not, consider a file lock or atomic write (write to temp then rename).
5. **Missing candle file**: `_populate_single_cache` loads from `data/ohlc_data/...`. If the parquet is missing, fail clearly (FileNotFoundError or result in `errors` dict); spec should mention that CacheManager returns errors for missing candle data.
6. **use_cache propagation**: Portfolio has `use_cache`; when it calls `ensemble.fit_from_candles(..., use_cache=self.use_cache)`, ensembles must accept and use it. DiversifiedEnsemble is shown with `use_cache=True` default—ensure Portfolio actually passes `use_cache` so behavior is consistent.

### Suggested spec additions

- **Phase 1**: State where `CacheMissError` lives (e.g. `utils/bias_node_cache.py`).
- **BiasNode**: State explicitly that subclasses must call `_init_cache_after_params()` at the end of their `__init__` after setting `module_name` and `params` (and that the base class should not call it, or that calling it in base is a no-op until params exist).
- **BiasNodeCache**: Clarify `save()` accepts `pd.Series | pd.DataFrame` for single vs multi-output; parquet schema for multi-column.
- **CacheManager**: Document behavior when candle file is missing (return in `errors`, do not overwrite existing cache with partial data).
- **Integration test**: Add a test that vectorized and stream outputs are identical for the same candle input (same dates, same order).

### Optional improvements

- **Cache directory config**: Allow override of cache root (e.g. env var or config) so CI/tests can use a temp directory.
- **CLI --ticker ES --tf D**: "Populate all caches for a ticker/timeframe" is ambiguous—all modules? Spec says user chooses what to cache; clarify that this still requires a vault or specs (e.g. all bias nodes for that ticker/tf from a given vault).

---

## Open Questions for User

1. **Multi-output bias nodes**: EWMAC outputs `[signal, signalBool]`. Should cache store both or just the primary output?
  - Recommendation: Store all outputs in separate columns for flexibility.

