# nodes

> **Path:** `nodes/`  
> **Status:** Stable  
> **Last updated:** 2026-02-13

## Purpose
`nodes` defines stateful bias-node primitives that transform streamed OHLCV candles into feature values used by feature extraction, base models, and ensemble components.

## Public API policy (what we document)
This document covers public API that is either:
- defined as public symbols in `nodes/__init__.py` (`BiasNode`), or
- used across package boundaries via `module_name` + dynamic construction (`utils.helpers.create_bias_node(...)`), or
- imported directly outside `nodes` (notably `EWSDNode` and `TimeSeriesFeatureNode`).

Private helpers and archived node implementations under `nodes/archive/` are intentionally excluded.

## Quickstart (minimal)

```python
from datetime import datetime

from nodes.rsi import RSI
from utils.enums import Ticker, TimeFrame
from utils.models import Candle

node = RSI(ticker=Ticker.ES, tf=TimeFrame.D, lookback=14)

out = node.add_candle(
    Candle(
        datetime=datetime(2024, 1, 2),
        open=4800.0,
        high=4820.0,
        low=4790.0,
        close=4810.0,
        volume=1000.0,
        ticker=Ticker.ES,
        tf=TimeFrame.D,
    )
)

print(out)  # list[float], e.g. [50.0] during warmup
print(node.get_column_names())  # standardized feature column names
```

## Data contracts
Input(s):
- `Candle` (`utils.models.Candle`) with required fields: `datetime`, `open`, `high`, `low`, `close`, `volume`, `ticker`, `tf`.
- Candles should be streamed in ascending timestamp order per `(ticker, timeframe)` for deterministic output.

Output(s):
- `BiasNode.add_candle(...) -> List`: ordered feature values matching `node.get_column_names()`/`node.columns`.
- Cache APIs return pandas objects indexed by datetime:
  - `get_cached_values(...) -> pd.Series | None`
  - `get_cached_dataframe(...) -> pd.DataFrame | None`

State contract:
- Nodes are stateful and incremental; calling `add_candle` mutates internal rolling state.
- Reprocessing the same candle key `(candle.datetime, candle.ticker)` returns memoized output and does not recompute.

## Public API reference

### `BiasNode`
Type: class

Signature:
```python
class BiasNode(ABC):
    def __init__(self, ticker: Ticker, tf: TimeFrame)
```

Description:
- Abstract base for all bias nodes.
- Provides singleton-style instance factory, per-candle memoization wrapper, standardized column naming, and optional cache access methods.

Parameters:
- `ticker (Ticker)`: instrument identifier.
- `tf (TimeFrame)`: timeframe for the node state.

Returns:
- Instance with mutable rolling state.

Raises:
- No direct constructor validation at base class level.

Notes / Constraints:
- Subclasses implement `_compute_candle(self, candle: Candle) -> List`.
- Public call path is `add_candle`; do not call `_compute_candle` directly.
- No-lookahead expectation: subclass calculations should only use current/past candles.

Examples:
```python
node = SomeBiasNodeSubclass.get_instance(Ticker.ES, TimeFrame.D, lookback=14)
values = node.add_candle(candle)
```

### `BiasNode.get_instance`
Type: classmethod

Signature:
```python
@classmethod
def get_instance(cls: Type[T], *args, **kwargs) -> T
```

Description:
- Returns a cached singleton per `(class, args, kwargs)` after hashable normalization.

Parameters:
- `*args`, `**kwargs`: constructor args for subclass.

Returns:
- Existing or newly-created subclass instance.

Notes / Constraints:
- Function objects in params are keyed by `(module, function_name)`.
- Unhashable values are stringified for keying.

### `BiasNode.add_candle`
Type: function

Signature:
```python
def add_candle(self, candle: Candle) -> List
```

Description:
- Wrapper that memoizes the most recent candle key and delegates to subclass `_compute_candle` only when key changes.

Parameters:
- `candle (Candle)`: current input candle.

Returns:
- `List`: node output for the candle.

Notes / Constraints:
- Cache key is `(candle.datetime, candle.ticker)`.
- If same key is passed twice, previously computed list is returned.

### `BiasNode` cache/vectorized access APIs
Type: functions

Signatures:
```python
def get_cached_values(
    self,
    start: datetime | None = None,
    end: datetime | None = None,
    require_cache: bool = True,
) -> pd.Series | None

def get_cached_dataframe(
    self,
    start: datetime | None = None,
    end: datetime | None = None,
    require_cache: bool = True,
) -> pd.DataFrame | None

def is_cache_loaded(self) -> bool
def cache_exists(self) -> bool
def get_cache_path(self) -> str | None
```

Description:
- Read precomputed node outputs from `BiasNodeCache` without candle-by-candle replay.

Parameters:
- `start/end`: inclusive datetime window.
- `require_cache`: strict mode for missing cache behavior.

Returns:
- Series/DataFrame slice for requested range, or `None` in permissive mode.

Raises:
- `CacheMissError` when `require_cache=True` and cache is not initialized/missing/incomplete.

Notes / Constraints:
- Subclasses must call `_init_cache_after_params()` after setting `module_name` and `params`.
- Cache interface is optional; nodes that do not initialize cache return `None`/raise per mode.

## Major bias-node entrypoints used outside package

### `nodes.rsi.RSI`
Type: class

Signature:
```python
class RSI(BiasNode):
    def __init__(self, ticker: Ticker, tf: TimeFrame, lookback: int = 14)
```

Observable behavior:
- Outputs one value (`signal`) in `[0, 100]` scale.
- Warmup output is `50.0` until lookback is reached.
- Uses Cython fast kernels (`compute_rsi_initial_fast`, `update_rsi_fast`) via `utils.fast_nodes`.

### `nodes.rsi_signal.RSISignal`
Type: class

Signature:
```python
class RSISignal(BiasNode):
    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        strategy_mode: str = "long",
        exit_policy: str = "threshold_or_bars",
        exit_bars: int = 5,
    )
```

Observable behavior:
- Outputs `-1`, `0`, or `1` based on RSI threshold crosses.
- `strategy_mode` controls long-only, short-only, or long-short behavior.
- `exit_policy="threshold_or_bars"` exits after `exit_bars` or threshold cross.
- Warmup outputs `0` until `rsi_period` candles.

### `nodes.lagged_rsi.LaggedRSI`
Type: class

Signature:
```python
class LaggedRSI(BiasNode):
    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        rsiPeriod: int = 14,
        lagPeriod: int = 3,
    )
```

Observable behavior:
- Outputs the RSI value from `lagPeriod` bars ago as a continuous feature.
- Uses Cython-backed RSI stream helpers (`compute_rsi_initial_fast`, `update_rsi_fast`).
- Warmup outputs `50.0` until both RSI and lag history are available.

### `nodes.rsi_left_tail_pressure.RSILeftTailPressure`
Type: class

Signature:
```python
class RSILeftTailPressure(BiasNode):
    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        rsiPeriod: int = 14,
        leftTailLevel: float = 30.0,
    )
```

Observable behavior:
- Outputs normalized left-tail depth `max(0, (leftTailLevel - rsi) / leftTailLevel)`.
- Range is clamped to `[0, 1]` (continuous intensity feature).
- Warmup outputs `0.0` until RSI is initialized.

### `nodes.rsi_left_tail_streak.RSILeftTailStreak`
Type: class

Signature:
```python
class RSILeftTailStreak(BiasNode):
    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        rsiPeriod: int = 14,
        leftTailLevel: float = 30.0,
        maxStreak: int = 5,
    )
```

Observable behavior:
- Tracks consecutive bars with RSI below `leftTailLevel`.
- Outputs normalized streak intensity `min(streak, maxStreak) / maxStreak` in `[0, 1]`.
- Warmup outputs `0.0` until RSI is initialized.

### `nodes.rsi_rebound_velocity.RSIReboundVelocity`
Type: class

Signature:
```python
class RSIReboundVelocity(BiasNode):
    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        rsiPeriod: int = 14,
        leftTailLevel: float = 30.0,
    )
```

Observable behavior:
- Measures one-bar RSI rebound speed after left-tail conditions.
- Emits `max(0, rsi[t] - rsi[t-1]) / 100` when `rsi[t-1] < leftTailLevel`, else `0`.
- Warmup outputs `0.0` until RSI and previous RSI context exist.

### `nodes.atr.ATRNode`
Type: class

Signature:
```python
class ATRNode(BiasNode):
    def __init__(self, ticker: Ticker, tf: TimeFrame, period: int = 252)
```

Observable behavior:
- Outputs two values: `[atr, atrPct]` where `atrPct = atr / close * 100`.
- Direction bias is always neutral (`Bias.NEUTRAL`).
- Uses Cython path when available; falls back to Python/deque implementation.

### `nodes.ewsd.EWSDNode`
Type: class

Signature:
```python
class EWSDNode(BiasNode):
    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        lambda_short: float = 0.06061,
        long_run_window: int = 2520,
        blend_short_weight: float = 0.7,
        blend_long_weight: float = 0.3,
    )
```

Observable behavior:
- Outputs `[ewsd_daily_pct, ewsd_annual_pct]` as percentages.
- First candle uses initial daily volatility estimate (`1%`).
- Long-run volatility is computed with an expanding window (sample stdev after two returns; first return seeds with `abs(return)`).
- Blends short EWMA volatility and long-run sample stdev (Carver-style 70/30).
- Annualization uses factor `16` (`sqrt(256)`).

### `nodes.ewmac.EWMACNode`
Type: class

Signature:
```python
class EWMACNode(BiasNode):
    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        spanFast: int = 16,
        spanSlow: int = 64,
        span_ewsd: int = 32,
        forecast_scalar: float = 4.10,
    )
```

Observable behavior:
- Outputs `[signal, signalBool]`.
- Signal is risk-normalized and capped to `[-20, 20]`.
- Warmup during EWSD initialization returns `[0.0, False]`.

### `nodes.momentum.Momentum`
Type: class

Signature:
```python
class Momentum(BiasNode):
    def __init__(self, ticker: Ticker, tf: TimeFrame, lookback: int = 10)
```

Observable behavior:
- Outputs one raw difference: `close[t] - close[t-lookback]`.
- Warmup returns `0.0` until enough history exists.

### `nodes.buy_hold.BuyHold`
Type: class

Signature:
```python
class BuyHold(BiasNode):
    def __init__(self, ticker: Ticker, tf: TimeFrame)
```

Observable behavior:
- Always outputs `[1.0]`.
- No effective warmup beyond first observation.

### `nodes.ts_feature.TimeSeriesFeatureNode`
Type: class

Signature:
```python
class TimeSeriesFeatureNode(BiasNode):
    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        wrapped_node: BiasNode,
        lookback: int,
        transformation: Callable,
        transformation_name: str,
        transformation_args: dict[str, Any] | None = None,
    )
```

Observable behavior:
- Wraps another bias node and stores rolling outputs.
- `add_candle` returns placeholders/forwarded values; final transformed series is computed lazily via `compute_features_from_stored_data()`.
- Transformation failures are logged at debug level and converted to `NaN` rather than raising.

### `nodes.turtle.TurtleTrading`
Type: class

Signature:
```python
class TurtleTrading(BiasNode):
    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        entry_lookback: int,
        stop_lookback: int,
        direction: str = "long",
    )
```

Observable behavior:
- Outputs signed position signal (`1.0`, `0.0`, `-1.0`) according to `direction`.
- Entry/stop channels are computed from prior candles only (explicitly excludes current candle), enforcing no-lookahead channel logic.

## Internal but required
`nodes` is typically instantiated through dynamic factory logic outside this package:

### `utils.helpers.create_bias_node`
Type: function (external dependency required to consume `nodes` at scale)

Signature:
```python
def create_bias_node(module_name: str, ticker: Ticker, tf: TimeFrame, params: Dict) -> Any
```

Description:
- Resolves `module_name` to `nodes/<module_name>.py`, imports module, finds the primary class, and instantiates via `get_instance` when available.
- Special-cases `ts_feature`/`tsFeature` to recursively create wrapped node and transformation function.

Raises:
- `ValueError` for missing module file or missing required `ts_feature` params.
- `ImportError` for import failures.
- `RuntimeError` when class instantiation fails.

Minimal example:
```python
from utils.enums import Ticker, TimeFrame
from utils.helpers import create_bias_node

node = create_bias_node("rsi", Ticker.ES, TimeFrame.D, {"lookback": 14})
```

## Errors & logging
- `BiasNode._init_cache_after_params()` logs warning when cache initialization fails, but does not break node construction.
- `BiasNode.get_cached_values()` logs warning before raising `CacheMissError` when cache is required but unavailable.
- `TimeSeriesFeatureNode._compute_transformation()` logs debug on transform exceptions and emits `NaN` output.
- Validation `ValueError` examples in node constructors:
  - `TurtleTrading`: invalid lookbacks or invalid `direction`.
  - Other nodes may validate positive periods depending on implementation.

## No-lookahead and time-alignment constraints
- Stream candles in strict chronological order per `(ticker, timeframe)`; out-of-order input breaks causal state assumptions.
- Feature at timestamp `t` must be treated as computed from data available at/through candle `t`, never future candles.
- Duplicate candle events with same `(datetime, ticker)` are memoized by `add_candle` and return prior output.
- Strategy nodes with channels/windows (e.g., Turtle) compute breakouts from trailing history excluding current candle, preserving causal execution semantics.

## Open questions
- `EWSDNode` currently defines `self.columns` directly (not standardized metadata + cache init pattern used by many newer nodes); behavior is stable but interface style differs.
