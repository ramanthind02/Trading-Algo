# ensemble

> **Path:** `ensemble/`  
> **Status:** Stable (with active evolution in diagnostics and fallback paths)  
> **Last updated:** 2026-03-04

## Purpose
`ensemble` is the portfolio-construction layer between base models and execution:
- `DiversifiedEnsemble` turns base-model signals into per-model forecasts with volatility and exposure scaling.
- `WeightLayer` combines model forecasts with per-ticker diversification weights and FDM.
- `Portfolio` applies instrument weights, IDM, and position caps.
- `PortfolioManager` orchestrates multiple timeframe portfolios and optional execution sizing.
- `vault_manager` + `ensemble_utils` provide the control-file and vault interfaces these classes depend on.

## Public API policy (what we document)
This document covers:
- Symbols exported by `ensemble/__init__.py` (`DiversifiedEnsemble`, `Portfolio`, `PortfolioManager`, `WeightLayer`, `BaseWeightLayer`, `InverseCorrelationWeightLayer`, `InverseCorrelationWeighter`).
- Cross-module public surfaces used by those symbols:
  - Control-file helpers in `ensemble/ensemble_utils.py`
  - Vault lifecycle helpers in `ensemble/vault_manager.py`
  - Execution handoff contract to `execution.position_sizer.PositionSizer`

Private helpers prefixed with `_` are omitted unless required to understand the public flow.

## Quickstart (minimal)
```python
from ensemble import DiversifiedEnsemble, Portfolio, PortfolioManager, WeightLayer
from execution import PositionSizer, ContractSpec
from utils.core.enums import TimeFrame

# 1) Load/construct ensemble from a unified control file
ensemble = DiversifiedEnsemble(
    control_file_path="vault/D/my_ensemble_long/control.json",
    target_volatility=0.20,
)

# 2) Fit portfolio stack
portfolio = Portfolio(
    ensembles=[ensemble],
    trading_timeframe=TimeFrame.D,
    weight_layer=WeightLayer(weight_method="inverse_correlation", fdm_max=2.0),
    max_position_pct=2.0,
)
portfolio.fit_from_candles(candles_df, target_data=returns_series)

# 3) Optional execution conversion
sizer = PositionSizer(
    capital=1_000_000,
    contract_specs={"ES": ContractSpec(ticker="ES", price=5000, multiplier=50)},
)
manager = PortfolioManager(portfolios={TimeFrame.D: portfolio}, position_sizer=sizer)

positions_or_contracts = manager.predict(candles_df)
```

## Serialization

- **Canonical:** Use control files (`save_control_file` / load via `control_file_path` or `load_control_file`) for full ensemble state (base models + fitted params).
- **Legacy:** `save_config` / `load_config` persist only fitted weights and metadata; prefer control files for new code.

## Data contracts

Input(s):
- `candles_df` (`pd.DataFrame`) used across `fit_from_candles` / `predict_from_candles`
  - Required columns: `datetime, open, high, low, close, volume, ticker, timeframe`
- Feature-frame path (`DiversifiedEnsemble.fit/predict`)
  - `X`: required model feature columns from control file
  - `ticker`: `pd.Series`/`np.ndarray` matching `len(X)`
  - `volatility`: per-sample series/array or `dict[ticker, float]`
- Control file shape (`ensemble_utils.parse_control_file`):
  - Required top-level keys: `metadata`, `base_models`
  - `members` in each base model is optional (if present, supports legacy and new member schemas)
  - `metadata.is_fit: bool`
  - When `is_fit=True`, required keys: `fitted_base_models`, `fitted_ensemble`

Output(s):
- `DiversifiedEnsemble.predict_from_candles(..., return_base_model_predictions=False)`: `pd.DataFrame` with `ticker, datetime, forecast_score`. With `return_base_model_predictions=True`: `dict` with keys `'ensemble'` (DataFrame) and `'base_models'` (dict of DataFrames).
- `DiversifiedEnsemble.predict(...)`: per-model forecast rows with `ticker, model_name, forecast, signal`
- `WeightLayer.combine(...)`: ticker(+optional datetime)-level `forecast_score`
- `Portfolio.predict(...)`: `ticker, forecast_score, position_fraction`
- `Portfolio.predict_from_candles(...)`: `ticker, datetime, forecast_score, position_fraction`
- `PortfolioManager.predict(...)`:
  - Fractions only when no position sizer
  - Contract-sized output (adds `target_dollars, contracts, notional_value, notional_pct`) when sizer provided
- `portfolio_tester` helpers:
  - `resample_positions_to_daily(...)`: ticker/datetime/position_fraction daily-aligned frame for combined tearsheet paths
  - `aggregate_intraday_returns_to_daily(...)`: daily-summed log-return series when multiple observations exist per calendar day

## Public API reference

### DiversifiedEnsemble
Type: class

Signature:
```python
class DiversifiedEnsemble:
    def __init__(
        self,
        target_volatility: float = 0.15,
        instrument_weights: Optional[Dict[str, float]] = None,
        config_path: Optional[str] = None,
        save_path: Optional[str] = None,
        control_file_path: Optional[str] = None,
        base_tf: Optional[TimeFrame] = None,
        use_cache: bool = True,
    ) -> None
```

Key public methods:
```python
def get_required_columns(self) -> List[str]
def get_required_bias_nodes(self) -> List[Dict[str, Any]]
def fit(self, X: pd.DataFrame, ticker, volatility, y, instrument_weights=None, normalization_data=None) -> "DiversifiedEnsemble"
def fit_from_candles(self, candles_df: pd.DataFrame, target_data: pd.Series, start_date=None, end_date=None) -> "DiversifiedEnsemble"
def predict(self, X: pd.DataFrame, ticker, volatility, normalization_data=None) -> pd.DataFrame
def predict_from_candles(self, candles_df: pd.DataFrame, volatility=None, return_base_model_predictions=False, start_date=None, end_date=None) -> pd.DataFrame | Dict[str, Any]
def save_config(self, filepath: Optional[str] = None) -> str
def load_config(self, filepath: str) -> "DiversifiedEnsemble"
def save_control_file(self, filepath: str) -> str
def load_control_file(self, filepath: str) -> "DiversifiedEnsemble"
```

Behavior:
- Loads base-model configs from unified control files and instantiates base models via `ensemble_utils.create_base_model_from_config`.
- Fits base models, computes diversified model weights, exposure fractions, and ticker-level allocation context.
- Produces per-model volatility-adjusted forecasts for downstream weight/portfolio layers.

Forecast scaling contract:
- Per-model forecast magnitude uses `target_volatility / (volatility * sqrt(model_exposure_fraction))`.
- Forecasts are capped at `2.0` before aggregation.

Raises:
- `ValueError` for missing control file, required feature columns, unseen tickers, invalid volatility, insufficient aligned samples, or unfitted access.
- `FileNotFoundError` / `ValueError` for config-file load failures.

Logging:
- `logger.debug` for ticker filtering, alignment counts, and model fit details.
- `logger.warning` for skipped/unsupported tickers, fallback exposure defaults, and signal-generation fallbacks.
- `logger.error(..., exc_info=True)` for per-model fit/predict failures.

Time alignment / no-lookahead:
- In candle-based fit, returns are computed per ticker and shifted forward one step (`shift(-1)`) so features at `T` pair with returns from `T -> T+1`.
- Prediction outputs are second-normalized (`floor('s')`) and right-aligned to candle `(ticker, datetime)` keys, filling missing forecasts with `0.0`.
- No forward-fill of future labels is introduced in public fit paths.

Example:
```python
ensemble.fit_from_candles(train_candles, target_data=train_returns)
forecast_vector = ensemble.predict(feature_df, ticker=ticker_series, volatility=vol_series)
```

### WeightLayer / BaseWeightLayer / InverseCorrelationWeightLayer / InverseCorrelationWeighter
Type: factory + classes

Factory signature:
```python
def WeightLayer(weight_method: str = "inverse_correlation", fdm_max: float = 2.5, **kwargs) -> BaseWeightLayer
```

Key class methods:
```python
class BaseWeightLayer(ABC):
    def fit(self, forecast_vectors: List[pd.DataFrame], signals: pd.DataFrame) -> "BaseWeightLayer"
    def combine(self, forecast_vectors: List[pd.DataFrame]) -> pd.DataFrame
    def get_diagnostics(self) -> Dict

class InverseCorrelationWeighter:
    def fit(self, signals: pd.DataFrame) -> "InverseCorrelationWeighter"
    def get_weights(self) -> pd.Series
```

Behavior:
- Fits per-ticker model weights.
- Calculates per-ticker FDM from forecast correlations and applies it during `combine`.
- For sparse/insufficient signal cases, falls back to equal weights and `FDM=1.0`.

Forecast vector contract:
- Each DataFrame in `forecast_vectors` must contain `ticker, model_name, forecast, signal`.
- `datetime` is optional but strongly preferred; if present, combination is per `(ticker, datetime)`.

Raises:
- `ValueError` for empty fit inputs, unfitted `combine`, unknown `weight_method`, or invalid weighter access before fit.

Logging:
- `logger.info` for per-ticker fit progress and summary diagnostics.
- `logger.warning` for insufficient data, unseen ticker/model fallbacks, and constant-column correlation edge cases.
- `logger.debug` for detailed correlation/FDM diagnostics.

### Portfolio
Type: class

Signature:
```python
class Portfolio:
    def __init__(
        self,
        ensembles: Optional[List] = None,
        ensemble_names: Optional[List[str]] = None,
        vault_root: str = "vault",
        trading_timeframe: TimeFrame = TimeFrame.D,
        target_volatility: Optional[float] = None,
        max_position_pct: float = 2.0,
        weight_layer: Optional[BaseWeightLayer] = None,
        instrument_weights: Optional[Dict[str, float]] = None,
        idm_max: float = 2.5,
        use_cache: bool = True,
        sector_allocation_config_path: Optional[str] = None,
    ) -> None
```

Key public methods:
```python
def fit(self, instrument_returns: pd.DataFrame, idm_override: Optional[float] = None) -> "Portfolio"
def predict(self, combined_forecasts: pd.DataFrame) -> pd.DataFrame
def fit_from_candles(self, candles_df: pd.DataFrame, target_data: Optional[pd.Series] = None, start_date=None, end_date=None) -> "Portfolio"
def predict_from_candles(self, candles_df: pd.DataFrame, return_ensemble_predictions=False, return_base_model_predictions=False, start_date=None, end_date=None) -> pd.DataFrame | Dict[str, Any]
def get_diagnostics(self) -> Dict
def print_diagnostics(self) -> None
```

Behavior:
- Combines ensemble outputs using `WeightLayer` (or averaging fallback if not fitted).
- Auto-loads ensembles from `vault/{D,W,M}/*` when `ensembles=None`; pass
  `ensemble_names=[...]` to filter by full directory names.
- Applies instrument weights, IDM, and optional cap to produce `position_fraction`.
- Supports hierarchical sector allocation configs that resolve to ticker-level instrument weights. See [Sector allocation (methodology)](../methodology/sector_allocation.md) for JSON schema, validation rules, and examples.

Risk stack order:
1. Forecast combination (already volatility-adjusted upstream)
2. Instrument weighting
3. IDM scaling
4. Position cap (`max_position_pct`)

Raises:
- `ValueError` for empty/missing required columns, insufficient data for IDM fit, invalid sector config schema, or unfitted predict access.

Logging:
- Extensive `logger.debug/info/warning/error` around returns overlap, IDM/FDM fitting paths, and fallback behavior.

Time alignment / no-lookahead:
- Returns for IDM are built from same-ticker `pct_change` and date-normalized for cross-ticker overlap checks.
- Forecast alignment to candles uses `(ticker, datetime)` merge with missing forecast fill `0.0`; no future label joins are added.

### PortfolioManager
Type: class

Signature:
```python
class PortfolioManager:
    def __init__(self, portfolios: Dict[TimeFrame, Portfolio], position_sizer: Optional[PositionSizer] = None)
    def fit(self, candles_df: pd.DataFrame, target_data: Optional[pd.Series] = None) -> None
    def predict(self, candles_df: pd.DataFrame) -> pd.DataFrame
    def get_portfolio(self, timeframe: TimeFrame) -> Optional[Portfolio]
```

Behavior:
- Routes candles to portfolios by `timeframe`.
- Aggregates per-timeframe portfolio outputs.
- Optionally invokes `PositionSizer.calculate_positions` on aggregated fractions.

Raises / logging:
- Raises `ValueError` for missing required candle columns.
- Logs warnings/errors for per-portfolio fit/predict failures and execution conversion fallbacks.

### portfolio_tester utilities
Type: module-level functions (`ensemble.portfolio_tester`)

Signatures:
```python
def resample_positions_to_daily(
    positions_df: pd.DataFrame,
    daily_dates_per_ticker: dict[object, pd.DatetimeIndex],
) -> pd.DataFrame

def aggregate_intraday_returns_to_daily(returns: pd.Series) -> pd.Series
```

Behavior:
- `resample_positions_to_daily` forward-fills sparse timeframe positions (e.g., weekly/monthly) across ticker-specific daily calendars and fills pre-signal periods with `0.0`.
- `aggregate_intraday_returns_to_daily` detects intraday return density via duplicate normalized dates and sums per day (log-return additive contract).

Notes / constraints:
- QuantStats tearsheets expect daily return inputs; use these helpers before report generation when combining mixed-frequency or intraday strategy streams.

### Vault and control interfaces (cross-module surfaces)
Type: module-level functions used by ensemble public flow

`ensemble_utils` key public contracts:
```python
def parse_control_file(filepath: str) -> Dict[str, Any]
def validate_control_file(control_file: Dict[str, Any]) -> None
def save_control_file(filepath: str, base_models: List[Dict[str, Any]], metadata: Dict[str, Any], fitted_base_models=None, fitted_ensemble=None, tickers=None) -> str
def create_base_model_from_config(config: Dict[str, Any], ticker: Optional[Ticker] = None, fitted_params: Optional[Dict[str, Any]] = None, use_cache: bool = True) -> Any
def add_feature_to_control_file(filepath: str, feature_config: Dict[str, Any], tickers: Optional[List[str]] = None) -> None
def extract_bias_node_specs_from_control_file(filepath: str) -> List[Dict[str, Any]]
def aggregate_bias_node_specs_from_directory(directory: str) -> List[Dict[str, Any]]
def filter_dataframe_by_timeframe(df: pd.DataFrame, base_tf: TimeFrame) -> pd.DataFrame
```

`vault_manager` key public contracts:
```python
def create_ensemble_directory(timeframe: TimeFrame, ensemble_name: str, direction: Direction, tickers: Optional[List[Ticker]] = None) -> str
def add_feature_to_ensemble(feature_column: str, bias_node_spec: Dict[str, Any], base_model: BaseModel, ensemble_dir: Optional[str] = None, tickers: Optional[List[Ticker]] = None) -> str
def load_feature_base_models(feature_column: str, ensemble_dir: Optional[str] = None, fitted_only: bool = False, tickers: Optional[List[Ticker]] = None) -> Dict[Tuple[Ticker, str], BaseModel]
def update_base_model_fitted_params(ensemble_dir: str, feature_column: str, model_id: str, fitted_params: Dict[str, Any], train_start: str, train_end: str) -> None
def remove_base_model_variant(ensemble_dir: str, feature_column: str, model_id: str) -> None
def list_features(ensemble_dir: Optional[str] = None) -> pd.DataFrame
def list_ensembles(vault_root: str) -> pd.DataFrame
def validate_ensemble_directory(ensemble_dir: str) -> None
def load_ensemble_from_vault(ensemble_dir: str, refit: bool = False, target_volatility: float = 0.20) -> DiversifiedEnsemble
```

Behavior notes:
- These functions define the persisted schema and assembly lifecycle used by `DiversifiedEnsemble` initialization and save/load flows.
- They enforce `is_fit` consistency, model schema/version checks (`binning_v2`), and ticker/direction consistency inside vault directories.

## Internal but required
- `DiversifiedEnsemble` public construction depends on internal control-file initialization (`_initialize_from_control_file`) and model creation hooks through `ensemble_utils.create_base_model_from_config`.
- `Portfolio.fit_from_candles`/`predict_from_candles` depend on internal conversion of ensemble outputs into WeightLayer forecast vectors (`ticker, datetime, model_name, forecast, signal`).

## Errors & logging
- Common exceptions:
  - `ValueError`: invalid DataFrame schema, missing required fields, unknown strategy/method, unfitted usage, inconsistent control/vault schema.
  - `FileNotFoundError`: missing config/control files.
- Logging patterns:
  - `debug`: alignment counts, per-model/ticker internals.
  - `info`: fitting progress and diagnostics summaries.
  - `warning`: fallback paths (equal weights, default vol/IDM/FDM).
  - `error`: model/portfolio failures with traceback context.

## Interaction with execution sizing
- Canonical handoff contract from portfolio layer into execution layer:
  - Required columns: `ticker, forecast_score, position_fraction`
  - Consumed by: `execution.position_sizer.PositionSizer.calculate_positions(...)`
- `PortfolioManager.predict(...)` performs this handoff automatically when a `PositionSizer` is configured.

Minimal execution example:
```python
fractions = portfolio.predict(combined_forecasts)
contracts = position_sizer.calculate_positions(fractions)
```

## Open questions
Q1:
- `Portfolio.predict_from_candles` has both fitted WeightLayer and averaging fallback behavior; should the fallback remain public behavior or become an explicit error after `fit_from_candles` is expected?

Q2:
- `DiversifiedEnsemble.predict` docs and some comments mention binary `{0,1}`, but implementation accepts signed `{-1,0,1}` in several paths; confirm the official public signal contract.
