# deployment

> **Path:** `deployment/`  
> **Status:** Stable (production-facing with script-oriented wrappers)  
> **Last updated:** 2026-02-13

## Purpose
`deployment` contains runtime and operational entrypoints for production forecast serving, market-data connectivity, notifications, and model-config training/export.

It is the boundary between research/training components and operational scripts:
- Live forecast loop (`ForecastServer`)
- Broker connector abstraction used by the server (`ForecastMT5DataConnector`)
- Notification transport (`TelegramNotifier`)
- Control-file training/export pipeline (`ProductionTrainingPipeline`)
- Script-facing test harness (`TestForecastServer`)

## Public API policy (what we document)
This document covers public symbols in `deployment/*.py` and entrypoints used by scripts.

Public items include:
- Classes and methods not prefixed with `_`
- Top-level factory helpers (`create_mt5_connector`, `create_telegram_notifier`)
- CLI entrypoints (`main()` functions and `python deployment/...py` execution)
- Script-facing surfaces imported by `scripts/` (`TestForecastServer`, `TelegramNotifier`)

Not public:
- Private helpers prefixed with `_` inside classes
- Internal scheduling callback wiring (`_run_forecasts`, `_setup_scheduling`, etc.)

## Quickstart (minimal)
```python
from deployment.test_forecast_server import TestForecastServer
from utils.core.enums import TimeFrame

server = TestForecastServer(config_dir="deployment/config")
server.load_historical_data(days_back=90)  # warm up node lookbacks
results = server.run_test_forecast(timeframe=TimeFrame.D)
print(results["forecasts"].get("D", {}))
server.stop()
```

## Data contracts

Input(s):
- Candle payloads consumed by deployment services are `utils.core.models.Candle` objects with:
  - `datetime`, `open`, `high`, `low`, `close`, `volume`, `ticker`, `tf`
- Forecast map passed to notifier:
  - `Dict[str, float]` (`ticker -> forecast_value`)
- Training pipeline price data:
  - `pd.DataFrame` indexed by datetime with `open, high, low, close, volume`
- Control files written by training pipeline:
  - JSON with `metadata`, `base_models`, `fitted_base_models`, `fitted_ensemble`, `tickers`

Output(s):
- Forecast-server forecast batches: `Dict[str, float]` where values are normalized to `(0, 1)` via logistic scaling
- Test-server run output:
  - `{"timestamp": str, "forecasts": Dict[str, Dict], "errors": List[str]}`
- Connector candle fetch:
  - `list[Candle]` (history) or `Optional[Candle]` (latest)
- Training pipeline:
  - Saved control-file path(s) as `str` or `Dict[str, Optional[str]]`

## Public API reference

### ForecastServer
Type: class

Signature:
```python
class ForecastServer:
    def __init__(self, config_dir: str = "deployment/config")
```

Description: Long-running production scheduler that loads per-ticker/per-timeframe ensembles, maintains ML feature state from incoming candles, generates forecasts, and publishes status/forecast/error notifications.

Key public methods (signatures):
```python
def load_historical_data(self, days_back: int = 60) -> None
def start(self) -> None
def stop(self) -> None
def get_status(self) -> Dict
```

Parameters:
- `config_dir`: directory containing control files by timeframe (`deployment/config/D/*.json`, `deployment/config/W/*.json`)
- `days_back`: warm-start history depth used before live operation

Returns:
- `get_status() -> Dict`:
  - `ensembles` by timeframe
  - `ml_managers` count
  - configured `tickers` and `timeframes`
  - MT5 connectivity/market-open probe result

Raises:
- `start()` re-raises unexpected runtime exceptions after sending error notification
- Most per-ticker/per-timeframe failures are logged and skipped (best-effort batch behavior)

Notes / Constraints:
- Requires control-file layout grouped by timeframe directory and named `<TICKER>_<TF>.json`.
- Historical warm-up must run before meaningful first forecast to satisfy bias-node lookbacks.
- Scheduler is continuous (`while True` + `schedule.run_pending()`), blocking the caller thread.

No-lookahead / time alignment:
- MT5 data source fetches 2 bars and uses the most recent *closed* bar for prediction updates.
- Forecast generation uses only already-buffered candles and latest computed feature row.
- No forward fill of future candles is introduced in deployment layer.

Example:
```python
from deployment.forecast_server import ForecastServer

server = ForecastServer(config_dir="deployment/config")
server.start()  # blocking loop
```

### ForecastMT5DataConnector
Type: class

Signature:
```python
class ForecastMT5DataConnector:
    def __init__(self)
```

Description: MT5 adapter used by `ForecastServer` to authenticate, fetch historical/latest candles, check market-open heuristics, and close terminal connections.

Public methods (signatures):
```python
def authenticate(self, first_try: bool = True) -> bool
def get_historical_candles(self, ticker: str, timeframe: TimeFrame, count: int = 100) -> list[Candle]
def get_latest_candle(self, ticker: str, timeframe: TimeFrame) -> Optional[Candle]
def is_market_open(self, ticker: str = "EURUSD") -> bool
def get_symbol_info(self, ticker: str) -> Optional[dict]
def shutdown(self) -> None
```

Factory:
```python
def create_mt5_connector() -> ForecastMT5DataConnector
```

Parameters:
- Reads environment variables if present: `MT5_USERNAME`, `MT5_PASSWORD`, `MT5_SERVER`
- Supports `Ticker` enum or string ticker aliases through internal symbol map

Returns:
- `get_historical_candles`: newest-last `list[Candle]` or empty list on failure
- `get_latest_candle`: most recent closed `Candle` or `None`
- `get_symbol_info`: dict with `symbol`, `bid`, `ask`, `spread`, `digits`, `point`

Raises:
- Does not intentionally propagate most runtime errors; logs and returns empty/`None`/`False`

Notes / Constraints:
- Unknown ticker/timeframe mappings are treated as soft failures with error logs.
- `is_market_open` is heuristic: true when latest daily candle timestamp is within 1 day.

No-lookahead / time alignment:
- `get_latest_candle` uses second-to-last bar when available to avoid using a forming candle.
- MT5 timestamps are converted from broker/FTMO time to New York time before `Candle` creation.

Example:
```python
from deployment.mt5_data_connector import create_mt5_connector
from utils.core.enums import TimeFrame

conn = create_mt5_connector()
candle = conn.get_latest_candle("EURUSD", TimeFrame.D)
conn.shutdown()
```

### TelegramNotifier
Type: class

Signature:
```python
class TelegramNotifier:
    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None)
```

Description: Telegram transport for forecast updates, status updates, errors, and connection tests.

Public methods (signatures):
```python
def send_forecast_update(self, forecasts: Dict[str, float], timeframe: TimeFrame,
                         timestamp: Optional[datetime] = None,
                         market_status: Optional[str] = None) -> bool
def send_message(self, text: str) -> bool
def send_error_notification(self, error_message: str, component: str = "ForecastServer") -> bool
def send_status_update(self, status: str, details: Optional[str] = None) -> bool
def test_connection(self) -> bool
```

Factory:
```python
def create_telegram_notifier() -> TelegramNotifier
```

Parameters:
- Credentials can be passed directly or loaded from env vars:
  - `TELEGRAM_BOT_TOKEN`
  - `TELEGRAM_CHAT_ID`
- `send_message` uses Telegram Markdown parse mode.

Returns:
- All send/test methods return `bool` success flags.

Raises:
- Network and API exceptions are caught internally; failures are logged and surfaced as `False`.

Notes / Constraints:
- If token/chat is not configured, notifier logs intent and does not send.
- Message formatting for forecast updates is ticker-sorted for stable output ordering.

Example (script-facing usage):
```python
from deployment.telegram_notifier import TelegramNotifier

notifier = TelegramNotifier()
ok = notifier.send_status_update("Forecast job complete", details="4/4 tickers updated")
```

### TestForecastServer
Type: class

Signature:
```python
class TestForecastServer(ForecastServer):
    def run_test_forecast(self, timeframe: Optional[TimeFrame] = None) -> Dict
```

Description: Script-facing test harness extending `ForecastServer`; executes immediate forecast runs without waiting for scheduler windows.

Used by:
- `scripts/run_manual_forecast.py`

Returns:
- `Dict` contract:
  - `timestamp: str` (ISO8601, NY timezone)
  - `forecasts: Dict[str, {count: int, tickers: List[str], predictions: Dict[str, float]}]`
  - `errors: List[str]`

Notes / Constraints:
- Only timeframes with loaded ensembles are tested.
- Uses inherited forecast-generation internals; ensure `load_historical_data()` is called first.

Example:
```python
from deployment.test_forecast_server import TestForecastServer

server = TestForecastServer()
server.load_historical_data()
results = server.run_test_forecast()
server.stop()
```

### ProductionTrainingPipeline
Type: class

Signature:
```python
class ProductionTrainingPipeline:
    def __init__(self, output_dir: str = "deployment/config")
```

Description: End-to-end training/export flow for generating deployment control files from historical parquet data.

Public methods (signatures):
```python
def load_ticker_data(self, ticker: Ticker, timeframe: TimeFrame,
                     start_date: str = "2020-01-01", end_date: Optional[str] = None) -> pd.DataFrame
def extract_features_and_targets(self, data: pd.DataFrame, ticker: Ticker,
                                 timeframe: TimeFrame) -> Tuple[pd.DataFrame, pd.DataFrame]
def run_feature_selection(self, features_df: pd.DataFrame, targets_df: pd.DataFrame) -> List[str]
def train_ensemble(self, features_df: pd.DataFrame, targets_df: pd.DataFrame,
                   selected_features: List[str], ticker: Ticker, timeframe: TimeFrame) -> Dict
def save_ensemble_config(self, ensemble_config: Dict, ticker: Ticker, timeframe: TimeFrame) -> str
def train_ticker_timeframe(self, ticker: Ticker, timeframe: TimeFrame) -> str
def train_all_production_ensembles(self, tickers: Optional[List[Ticker]] = None,
                                   timeframes: Optional[List[TimeFrame]] = None) -> Dict[str, Optional[str]]
```

Entrypoint:
```python
def main() -> None
```

Behavior:
- Loads ticker parquet data from `data/parquet_data/*.parquet`
- Aggregates to weekly/monthly when requested
- Generates node-based features and next-period return targets
- Runs walkforward feature selection with fallback behavior
- Fits binning base models and writes control files under `deployment/config/<TF>/<TICKER>_<TF>.json`

Raises:
- `ValueError` for unknown ticker mappings and empty model training outcomes
- `FileNotFoundError` when required parquet file is missing
- Other exceptions propagate for unrecoverable pipeline failures

Notes / Constraints:
- Designed for offline batch training, not online incremental updates.
- On feature-selection failure, falls back to all features (logged warning/error).

No-lookahead / time alignment:
- Targets are explicit next-period returns (`pct_change(1).shift(-1)` and log-return equivalent).
- Feature/target rows are intersected on valid non-null timestamps before training.
- Weekly/monthly aggregation is done before target construction to preserve timeframe-consistent causality.

Example:
```python
from deployment.production_training_pipeline import ProductionTrainingPipeline
from utils.core.enums import Ticker, TimeFrame

pipeline = ProductionTrainingPipeline(output_dir="deployment/config")
path = pipeline.train_ticker_timeframe(Ticker.ES, TimeFrame.D)
print(path)
```

## Internal but required
- `TestForecastServer.run_test_forecast()` relies on inherited private method `ForecastServer._generate_portfolio_forecasts`; this is an intentional script-facing dependency even though the underlying method is private.
- `ForecastServer.start()` operationally depends on private scheduling/setup methods and private candle/feature buffering (`_setup_*`, `_add_candle`); custom subclasses should preserve these invariants.

## Errors & logging
- Common exception families:
  - Configuration/data errors (`ValueError`, `FileNotFoundError`)
  - Runtime connector/network errors (MT5, Telegram HTTP)
- Runtime posture is mostly best-effort:
  - Per-instrument failures are logged and skipped
  - Batch continues whenever possible
- Logging style:
  - `info` for lifecycle milestones (startup, load counts, completion)
  - `warning` for partial data/coverage issues
  - `error` (and occasional traceback) for failed operations

## Operational and no-lookahead constraints
- Always warm-start with enough historical candles before first forecast.
- Treat only closed bars as eligible inference inputs.
- Do not mix timezones without conversion; deployment layer normalizes MT5 times to New York.
- Keep control-file schema stable (`metadata/base_models/fitted_base_models/fitted_ensemble/tickers`) for compatibility with server loading.

## Open questions
Q1:
- `ForecastServer._prepare_prediction_data()` emits verbose info-level debug payloads each forecast cycle; confirm whether this is intended for production or should be reduced to debug level.

Q2:
- `TelegramNotifier` and MT5 connector include fallback default credentials in code paths; verify whether this is intentionally supported for non-production only.
