# feature_selection.base_models

> **Path:** `feature_selection/base_models/`  
> **Status:** Stable (with one legacy contract caveat noted below)  
> **Last updated:** 2026-02-13

## Purpose
`feature_selection.base_models` defines the base-model layer used by feature validation and ensemble training:
- `BaseModel` orchestrates bias-node feature extraction plus binning-model fit/predict.
- `BinningModelBase` defines the shared fit/predict/serialization contract for binning-style models.
- `ContinuousBinningModel` and `RuleBasedModel` are current concrete public implementations.

## Public API policy (what we document)
This document covers symbols exposed from `feature_selection/base_models/__init__.py` and symbols used as cross-module contracts by `ensemble/*` and research helpers.

Documented public symbols:
- `BaseModel`
- `BinningModelBase`
- `ContinuousBinningModel`
- `RuleBasedModel`
- `TwoBinBinningModel` external contract (`model_type='two_bin_binning'`) referenced by callers

## Quickstart (minimal)
```python
import pandas as pd
from feature_selection.base_models import BaseModel, ContinuousBinningModel
from utils.core.enums import Ticker, TimeFrame

feature_config = {
    "bias_node_spec": {
        "module_name": "ewmac",
        "timeframes": [TimeFrame.M15],
        "params": {"spanFast": 16, "spanSlow": 64},
    },
    "model_type": "continuous_binning",
    "constructor_params": {"n_bins": 15, "selection_metric": "sharpe"},
}

model = BaseModel(
    feature_config=feature_config,
    tickers=[Ticker.ES],
    binning_model=ContinuousBinningModel(n_bins=15, selection_metric="sharpe"),
    use_cache=False,
)

# candles_df: columns include datetime/open/high/low/close/volume/ticker/timeframe
# target_series: return series indexed by datetime
model.fit(candles_df, target_series)
signals = model.predict(candles_df, strategy="long")
```

## Data contracts

Input(s):
- `feature_config` (`dict`)
  - Required key: `bias_node_spec`
    - `module_name: str`
    - `timeframes: list[TimeFrame]`
    - `params: dict[str, Any]` (single value per parameter in `BaseModel`; lists with length > 1 raise)
  - Optional: `model_type`, `constructor_params` (used when `binning_model` not supplied)
- `candles_df` (`pd.DataFrame`)
  - Required columns for fit/predict flows: `datetime, open, high, low, close, volume, ticker, timeframe`
- `target_data` (`pd.Series` or single-target `pd.DataFrame`)
  - Datetime index required for robust alignment with extracted features
- `normalization_data` (`pd.Series`, optional)
  - Required when `normalize_by` is set and scaled prediction is requested

Output(s):
- Fitted binning state on the model (`bin_edges_`, `bin_stats_`, active bins, multipliers)
- Prediction series (`pd.Series`) indexed by candle datetimes
- Vault/control-file serializable fitted params via `get_fitted_params()` and `BaseModel.update_fitted_params_in_vault()`

## Public API reference

### BaseModel
Type: class

Signature:
```python
class BaseModel:
    def __init__(
        self,
        feature_config: Dict[str, Any],
        tickers: Union[Ticker, List[Ticker]],
        binning_model: Optional[BinningModelBase] = None,
        use_cache: bool = True,
    ) -> None
```

Description: Orchestrates end-to-end base-model flow: bias-node creation, candle ingestion, feature extraction, target alignment, binning fit, and signal prediction.

Key public methods (signatures):
```python
def add_candle(self, candle: Candle, tf: TimeFrame, ticker: Optional[Ticker] = None) -> None
def get_feature(self) -> pd.Series
def fit(self, candles_df: pd.DataFrame, target_data: pd.Series,
        start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> "BaseModel"
def stream_fit(self, candles_df: pd.DataFrame, target_data: pd.Series) -> "BaseModel"
def vectorized_fit(self, candles_df: pd.DataFrame, target_data: pd.Series,
                   start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> "BaseModel"
def predict(self, candles_df: pd.DataFrame, strategy: str = "long",
            start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> pd.Series
def stream_predict(self, candles_df: pd.DataFrame, strategy: str = "long") -> pd.Series
def vectorized_predict(self, candles_df: pd.DataFrame, strategy: str = "long",
                       start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> pd.Series
def predict_members_from_candles(self, candles_df: pd.DataFrame, strategy: str = "long") -> pd.DataFrame
def get_fitted_params_for_members(self) -> Dict[str, Dict[str, Any]]
def save_to_vault(self, ensemble_dir: Optional[str] = None,
                  tickers: Optional[List[Ticker]] = None) -> str
def update_fitted_params_in_vault(self, ensemble_dir: str, model_id: str,
                                  train_start: str, train_end: str) -> None
```

Notes / Constraints:
- `feature_config` must include `bias_node_spec`, or itself be a backward-compatible bias-node-spec dict.
- For parameter values in `bias_node_spec['params']`, list values must be length 1; multi-value grids are intentionally rejected.
- Tickers in `candles_df` must be a subset of model tickers.
- Cached mode (`use_cache=True`) requires complete cache coverage or raises cache miss errors.
- `predict()` preserves model state in streaming mode (no automatic reset).
- `fit()` also fits attached members on the same aligned feature/target data;
  pre-fitted members with `requires_fit=False` are skipped.

No-lookahead / time alignment:
- Streaming fit/predict sorts candles by `datetime` before per-candle processing to preserve causal feature state.
- Feature/target alignment is by exact datetime first; fallback alignment strips microseconds to base timestamps.
- If alignment remains poor/empty, hard errors are raised; no forward-looking joins are introduced.
- Minimum aligned sample guard (`<20`) raises to prevent fitting on unstable alignment.

Raises:
- `ValueError` for malformed config, ticker mismatch, empty features, failed alignment, insufficient aligned samples, unfitted vault updates.
- `CacheMissError` (from `utils.cache.bias_node_cache`) in vectorized fit/predict when required cache is missing.

Errors & logging:
- `logger.error(...)` on cache misses with model/ticker/timeframe context.
- `logger.warning(...)` on poor fallback alignment and NaN-heavy prediction inputs.
- `logger.debug(...)` for per-datetime missing feature values during streaming prediction.

Example:
```python
base_model = BaseModel(feature_config=feature_config, tickers=[Ticker.ES], use_cache=True)
base_model.fit(candles_df, target_series)
long_short = base_model.predict(candles_df, strategy="long_short")
```

### BinningModelBase
Type: class (abstract)

Signature:
```python
class BinningModelBase(ABC):
    def __init__(
        self,
        n_bins: int = 3,
        selection_metric: str = "sharpe",
        normalize_by: Optional[str] = "ewsd",
        strategy: str = "long",
        metric_threshold: float = 0.0,
        t_threshold: float = 2.0,
        min_region_width: int = 2,
        shrinkage_k: float = 20.0,
        long_clip_min: float = 0.5,
        long_clip_max: float = 2.0,
        short_clip_min: float = 0.5,
        short_clip_max: float = 2.0,
    ) -> None
```

Description: Shared estimator contract for binning-based models. Handles fit-state lifecycle, bin statistics, active-bin selection, and strategy-specific position multipliers.

Public methods (signatures):
```python
def fit(self, feature_data: pd.Series, target_data: pd.Series,
        normalization_data: Optional[pd.Series] = None) -> "BinningModelBase"
def predict(self, feature_data: pd.Series, strategy: str = "long",
            normalization_data: Optional[pd.Series] = None, scaled: bool = False) -> pd.Series
def get_bin_stats(self) -> Dict[int, Dict[str, float]]
def get_fitted_params(self) -> Dict[str, object]
def compute_objective_metric(self, feature_data: pd.Series, target_data: pd.Series,
                             objective_metric: "ObjectiveMetric", strategy: str = "long",
                             normalization_data: Optional[pd.Series] = None) -> float
def save_to_feature_list(self, filepath: str, tickers: Optional[List[str]] = None) -> None
def get_params(self, deep: bool = True) -> Dict[str, object]
def set_params(self, **params: object) -> "BinningModelBase"
def score(self, X: pd.Series, y: pd.Series, strategy: str = "long") -> float
```

Constraints:
- `feature_data` passed to `fit()` must be a named series (`feature_data.name` required).
- Strategy must be one of `long`, `short`, or `long_short`.
- `selection_metric` must be one of `sharpe`, `mean`, `t_stat`, `sortino`.
- `fit()` requires sufficient data (`>= max(10, n_bins*5)` after dropping NaNs).
- For scaled prediction (`scaled=True`) with `normalize_by` set, `normalization_data` is required.

Raises:
- `ValueError` for unknown strategy/metric, unfitted accessors, insufficient data, missing normalization input.
- `ImportError` in `save_to_feature_list()` if ensemble utilities are unavailable.

No-lookahead / time alignment:
- Fit is performed on paired rows after joining feature/target by index and dropping NaNs; there is no forward-fill or future-target peeking in this class.
- Any time semantics come from caller-supplied index alignment (`BaseModel` is the main time-alignment owner).

Example:
```python
bm = ContinuousBinningModel(n_bins=15, selection_metric="sortino")
bm.fit(feature_series, target_series)
signal = bm.predict(feature_series, strategy="long")
```

### ContinuousBinningModel
Type: class

Signature:
```python
class ContinuousBinningModel(BinningModelBase):
    def __init__(
        self,
        n_bins: int = 15,
        selection_metric: str = "sharpe",
        strategy: str = "long",
        normalize_by: str | None = "ewsd",
        metric_threshold: float = 0.0,
        t_threshold: float = 2.0,
        min_region_width: int = 2,
        shrinkage_k: float = 20.0,
        long_clip_min: float = 0.5,
        long_clip_max: float = 2.0,
        short_clip_min: float = 0.5,
        short_clip_max: float = 2.0,
    ) -> None
```

Description: Quantile-first continuous feature binning model (`model_type='continuous_binning'`) with robust fallback bin creation when quantile cuts degenerate.

Behavior notes:
- Attempts `pd.qcut` first, then progressively falls back to unique-value mapping and `pd.cut` variants.
- Enables contiguous significant-region filtering using `t_threshold`/`min_region_width` from the base class.

Errors/logging:
- Bin-creation fallbacks catch `ValueError`/`TypeError` internally and continue; user-facing fit errors come from base-class validation when bins remain unusable.

Example:
```python
model = ContinuousBinningModel(n_bins=10, selection_metric="t_stat", metric_threshold=1.0)
model.fit(feature_series, target_series)
```

### RuleBasedModel
Type: class

Signature:
```python
class RuleBasedModel(BinningModelBase):
    def __init__(
        self,
        selection_metric: str = "sharpe",
        strategy: str = "long",
        normalize_by: str | None = None,
        metric_threshold: float = 0.0,
        shrinkage_k: float = 20.0,
        long_clip_min: float = 0.5,
        long_clip_max: float = 2.0,
        short_clip_min: float = 0.5,
        short_clip_max: float = 2.0,
    ) -> None
```

Description: Discrete rule model for ternary features (`-1/0/+1`) using fixed bin mapping:
- `-1 -> 0`
- `0 -> 1`
- `+1 -> 2`

Constraints:
- Feature values outside `{-1, 0, 1}` raise `ValueError` in both fit-time and predict-time bin assignment.
- Middle level (`0`, mapped bin 1) is always forced flat in active-bin construction.

No-lookahead/time alignment:
- Same alignment semantics as `BinningModelBase`; this model assumes caller has provided causally aligned feature/target indices.

Example:
```python
rb = RuleBasedModel(selection_metric="mean", metric_threshold=0.0)
rb.fit(discrete_signal, target_series)
pos = rb.predict(discrete_signal, strategy="long_short")
```

### TwoBinBinningModel (external contract)
Type: externally referenced class contract

Signature:
```python
class TwoBinBinningModel(BinningModelBase):
    ...
```

Behavior/contract used externally:
- Control/vault/model configs reference `model_type='two_bin_binning'`.
- Callers (`ensemble/ensemble_utils.py`, `ensemble/vault_manager.py`, `research/bias_node_helpers.py`) assume:
  - constructor does not take `n_bins` (hardcoded two-bin behavior)
  - `normalize_by` may be removed by callers before construction

Caveat:
- This symbol is referenced by external modules but is not exported from current `feature_selection/base_models/__init__.py` in this branch.
- Treat as a legacy/expected integration contract; validate runtime availability before relying on it in new code paths.

## Internal but required
- `BaseModel.__getattr__` delegates selected fitted attributes (`is_fitted_`, `thresholds_`-style compatibility fields) to `binning_model`; external callers may depend on attribute passthrough behavior.
- Binning fitted schema consumed by ensemble/vault flows expects `model_version='binning_v2'` and keys:
  - `bin_edges`, `bin_stats`, `significant_regions`, `active_bins_by_strategy`, `position_multipliers_by_strategy`, `fit_config`.

## Errors & logging
- Common exceptions:
  - `ValueError`: invalid config/inputs/strategy/alignment/unfitted access
  - `CacheMissError`: missing required cached bias-node data in vectorized flows
  - `ImportError`: control-file save helper import failures
- Common logs:
  - Error logs on cache miss with traceback context
  - Warning logs on weak datetime alignment and NaN-filled predict inputs
  - Debug logs for missing per-candle feature extraction cases

## Open questions
Q1:
- `TwoBinBinningModel` is referenced as public integration surface but not exported here; confirm whether this is intentional deprecation or an incomplete migration.

Q2:
- `BaseModel.__getattr__` delegates `thresholds_` and best-bin fields that are not native in current `BinningModelBase` (`bin_edges_`, active-bin maps are primary); confirm whether compatibility field names should be formalized or retired.
