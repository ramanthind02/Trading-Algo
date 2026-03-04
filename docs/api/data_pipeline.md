# data_pipeline

> **Path:** `data_cleaning/`, `feature_extraction/`, `eda/`, `research/`  
> **Status:** Draft  
> **Last updated:** 2026-02-13

## Purpose
This document describes the cross-module data-pipeline API surface used to move from raw OHLCV files to aligned feature/target frames, exploratory analysis outputs, and research reports.
Train/validation/test cutover plan references: `docs/kanban/to-do/feature_research_train_val_test/`.

## Public API policy (what we document)
This document covers public API only.

Public items include:
- Top-level functions/classes not prefixed with `_`.
- Symbols used across module boundaries in this repo (notebook/test usage included where relevant).
- Entry points that orchestrate multi-step extraction, EDA, and research reporting.

Not public in this doc:
- Private helpers (for example `_expand_param_grid`, `_extract_features_single_ticker`).
- Plotting internals in `metrics.plotting.*`.
- Internal cache plumbing unless required by public API behavior.

## Quickstart (minimal)
```python
from datetime import datetime
from pathlib import Path

from utils.core.enums import TimeFrame, Ticker
from feature_extraction.feature_extractor import extract_features_for_bias_node
from research.bias_node_helpers import build_feature_metadata
from eda.feature_explorer import FeatureExplorer
from feature_selection.base_models import QuantileBinningModel

bias_spec = {
    "module_name": "rsi",
    "timeframes": [TimeFrame.D],
    "params": {"lookback": [5, 14, 28]},
}

features_df, targets_df = extract_features_for_bias_node(
    bias_spec=bias_spec,
    ticker=[Ticker.ES, Ticker.NQ],
    start=datetime(2015, 1, 1),
    end=datetime(2024, 12, 31),
    target_col="log_return_atr",
)

explorer = FeatureExplorer(
    features_df=features_df,
    targets_df=targets_df,
    metadata=build_feature_metadata(features_df),
)

report = explorer.generate_summary_report(
    binning_model=QuantileBinningModel(n_bins=10),
    target_col="log_return_atr",
    strategy="long",
    show_plots=False,
)
```

## Data contracts
Input(s):
- Candles/price data frames must include `datetime`, `open`, `close`; multi-ticker flows also require `ticker`.
- Forward-return scaling expects ATR/EWSD feature columns when using `compute_forward_returns(..., features_df=...)`.
- ATR/EWSD auxiliary extraction is timeframe-aware in feature-research flows (`ATR period = bars_per_year`, `EWSD long_run_window = 10 * bars_per_year`).
- Feature extraction APIs accept single `Ticker` or `List[Ticker]`; multi-ticker alignment may use millisecond index offsets.
- EDA APIs require `features_df.index.equals(targets_df.index)`.

Output(s):
- Feature extraction returns `(features_df, targets_df)` where both are row-aligned and include optional `ticker`.
- Targets contain `raw_return`, `log_return`, `log_return_atr`, `log_return_ewsd` (when available/required).
- EDA/reporting entry points return dict/DataFrame bundles (figures, metrics, permutation tables, summary stats).

## Public API reference

### `txt_to_parquet`
Type: function  
Module: `data_cleaning/data_cleaning.py`

Signature:
```python
txt_to_parquet(
    folder_path: str,
    all_tickers: bool = False,
    ticker: Optional[Ticker] = None,
) -> None
```
Description: converts per-ticker `.txt` OHLCV files into parquet files under `data/parquet_data`.

Parameters:
- `folder_path` (`str`): source directory containing `<TICKER>.txt` files.
- `all_tickers` (`bool`): process all `Ticker` enum members when `True`.
- `ticker` (`Optional[Ticker]`): required when `all_tickers=False`.

Returns:
- `None`.

Raises:
- Does not re-raise processing exceptions; prints warning/error and continues.

Notes / Constraints:
- Input schema assumed: `date,open,high,low,close,volume`.
- Writes `timestamp` and string `datetime`, drops original `date`.

### `aggregate_parquet_data`
Type: function  
Module: `data_cleaning/data_cleaning.py`

Signature:
```python
aggregate_parquet_data(
    folder_path: str,
    ticker: Optional[Ticker] = None,
    all_tickers: bool = False,
) -> None
```
Description: resamples daily parquet OHLCV into `D/W/M` parquet outputs under `data/ohlc_data/<ticker>/`.

Notes / Constraints:
- Requires existing daily parquet input and either `datetime` or `timestamp` column.
- Uses `first/max/min/last/sum` OHLCV aggregation semantics.
- Prints progress/errors; continues on missing files.

### `compute_forward_returns`
Type: function  
Module: `feature_extraction/feature_extractor.py`

Signature:
```python
compute_forward_returns(
    candles_df: pd.DataFrame,
    features_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame
```
Description: computes shifted forward returns so `Feature[t]` predicts `Return[t+1]`, including ATR/EWSD normalization.

Parameters:
- `candles_df`: must contain `datetime`, `open`, `close`, `ticker`.
- `features_df`: required in current implementation; must include ticker-aligned ATR and EWSD columns.
  ATR detection is keyword-based (not tied to a `252` suffix), so non-daily ATR columns are supported.

Returns:
- DataFrame indexed by datetime with `raw_return`, `log_return`, `log_return_atr`, `log_return_ewsd`, `ticker`.

Raises:
- `ValueError` for missing `features_df`, missing ATR/EWSD columns, empty ticker features, all-NaN scaling series, or alignment failures.

Notes / Constraints:
- Last row per ticker is dropped after `shift(-1)` (no forward target available).
- Normalization uses ATR/EWSD from `t+1` to match shifted return horizon.

### `extract_features`
Type: function  
Module: `feature_extraction/feature_extractor.py`

Signature:
```python
extract_features(
    module_name: str,
    params: Dict[str, Any],
    ticker: Union[Ticker, List[Ticker]],
    start: datetime = None,
    end: datetime = None,
    timeframes: List[TimeFrame] = None,
    use_millisecond_offset: bool = True,
    use_cache: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]
```
Description: extracts bias-node features (including parameter-grid expansion) and aligned forward targets.

Notes / Constraints:
- Single ticker and multi-ticker paths return unified shape contracts.
- Multi-ticker mode can apply millisecond offsets to keep index uniqueness.
- `use_cache=True` depends on populated bias-node cache and may raise cache-miss errors.

### `extract_features_with_forward_returns`
Type: function  
Module: `feature_extraction/feature_extractor.py`

Signature:
```python
extract_features_with_forward_returns(
    module_name: str,
    params: Dict[str, Any],
    ticker: Union[Ticker, List[Ticker]],
    start: datetime = None,
    end: datetime = None,
    timeframes: List[TimeFrame] = None,
    use_millisecond_offset: bool = True,
    target_col: str = "log_return",
    use_cache: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]
```
Description: convenience orchestration API that extracts main features plus mandatory ATR/EWSD and then computes aligned forward returns.

Raises:
- `ValueError` for invalid `target_col`, no extracted features, no computed targets, or feature/target merge alignment failures.

Notes / Constraints:
- In timeframe-aware feature research, auxiliary extraction uses active timeframe scaling:
  - `atr(period=timeframes[0].bars_per_year)`
  - `ewsd(long_run_window=10 * timeframes[0].bars_per_year)`

### `extract_features_for_bias_node`
Type: function  
Module: `feature_extraction/feature_extractor.py`

Signature:
```python
extract_features_for_bias_node(
    bias_spec: Dict[str, Any],
    ticker: Union[Ticker, List[Ticker]],
    start: datetime = None,
    end: datetime = None,
    use_millisecond_offset: bool = True,
    target_col: str = "log_return",
    use_cache: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]
```
Description: high-level API used by research workflows; accepts `bias_spec` and delegates to `extract_features_with_forward_returns`.

Cross-module usage:
- Consumed by `research/bias_node_helpers.py:test_bias_node`.

### `prepare_candles_and_targets_for_basemodel`
Type: function  
Module: `feature_extraction/feature_extractor.py`

Signature:
```python
prepare_candles_and_targets_for_basemodel(
    candles_df: pd.DataFrame,
    target_col: str = "log_return",
) -> Tuple[pd.DataFrame, pd.Series]
```
Description: helper for BaseModel workflows; filters candles to rows with valid forward targets and returns model-ready candle frame plus target series.

### `FeatureExplorer`
Type: class  
Module: `eda/feature_explorer.py`

Signature:
```python
FeatureExplorer(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    metadata: Optional[Dict[str, Any]] = None,
)
```
Description: EDA facade over extracted features/targets (plots, correlations, parameter sensitivity, summary report generation).

Common cross-module methods:
- `generate_summary_report(...) -> Dict[str, Any]`
- `plot_nd_parameter_analysis(...) -> Tuple[pd.DataFrame, Any]`
- `generate_parameter_sensitivity_report(...) -> Dict[str, Any]`
- `get_summary() -> pd.DataFrame`

Raises:
- `TypeError` for non-DataFrame inputs.
- `ValueError` for feature/target index mismatch or invalid analysis selections.

### `ParameterAnalyzer`
Type: class  
Module: `eda/parameter_analysis.py`

Signature:
```python
ParameterAnalyzer(features_df: pd.DataFrame, targets_df: pd.DataFrame)
```
Description: computes 1D/2D/ND parameter sensitivity metrics and robustness scores used by `FeatureExplorer`.

Common methods:
- `analyze_parameter(...) -> pd.DataFrame`
- `analyze_2d_parameters(...) -> pd.DataFrame`
- `analyze_nd_parameters(...) -> pd.DataFrame`
- `compute_robustness_metrics(...) -> Dict[str, Any]`

Raises:
- `ValueError` when no valid data points remain after alignment/cleaning.

### EDA/Research entrypoints
- `feature_research/in_sample/continuous_binning/run_binning_analysis.py`: CLI entrypoint that orchestrates continuous binning analysis runs and emits research artifacts.

### `build_feature_metadata`
Type: function  
Module: `research/bias_node_helpers.py`

Signature:
```python
build_feature_metadata(features_df: pd.DataFrame) -> Dict[str, Any]
```
Description: builds `metadata["feature_metadata"]` mapping expected by `FeatureExplorer` from feature column names.

### `get_binning_model`
Type: function  
Module: `research/bias_node_helpers.py`

Signature:
```python
get_binning_model(
    is_continuous: bool,
    n_bins: int = 10,
    selection_metric: str = "sortino",
    strategy: str = "long",
) -> Union[QuantileBinningModel, TwoBinBinningModel]
```
Description: returns the canonical research binning model based on feature type.

### `get_best_feature_from_permutation`
Type: function  
Module: `research/bias_node_helpers.py`

Signature:
```python
get_best_feature_from_permutation(perm_df: Optional[pd.DataFrame]) -> Optional[str]
```
Description: selects the feature with minimum permutation p-value (`pval` or `p_value`).

### `test_bias_node`
Type: function  
Module: `research/bias_node_helpers.py`

Signature:
```python
test_bias_node(
    bias_spec: Dict[str, Any],
    node_name: str,
    is_continuous: bool,
    tickers: Union[Ticker, List[Ticker]],
    start_date: datetime,
    end_date: datetime,
    reports_dir: Path,
    permutation_reps: int = 100,
    show_plots: bool = False,
    export: bool = True,
    use_cache: bool = False,
    target_col: str = "log_return",
    binning_model: Optional[BinningModelBase] = None,
    strategy: str = "long",
) -> Optional[Dict[str, Any]]
```
Description: research orchestration entrypoint that extracts features, constructs metadata/explorer, generates summary report, and returns enriched result bundle.

Raises / behavior:
- Raises `ValueError` for invalid strategy.
- Otherwise handles many failures internally, prints traceback, and returns `None`.

### `generate_node_tearsheet`
Type: function  
Module: `research/bias_node_helpers.py`

Signature:
```python
generate_node_tearsheet(
    node_name: str,
    is_continuous: bool,
    start_date: datetime,
    end_date: datetime,
    reports_dir: Path,
    perm_df: Optional[pd.DataFrame] = None,
    features_df: Optional[pd.DataFrame] = None,
    targets_df: Optional[pd.DataFrame] = None,
    bias_spec: Optional[Dict[str, Any]] = None,
    strategy: str = "long-short",
    target_col: str = "log_return",
    tickers: Optional[Union[Ticker, List[Ticker]]] = None,
    binning_model: Optional[BinningModelBase] = None,
) -> Optional[str]
```
Description: generates QuantStats HTML tearsheet from extracted features, fitted model signals, and baseline returns.

Notes / Constraints:
- Requires a fitted-or-fit-able `binning_model`; for rule-based nodes, caller should pass the exact model used in testing.
- Uses equal-weight baseline for multi-ticker comparisons.

### `run_rule_based_eda_pipeline`
Type: function  
Module: `feature_research/in_sample/rule_based/pipeline.py`

Signature:
```python
run_rule_based_eda_pipeline(
    config: RuleBasedResearchConfig,
    output_dir: Path,
) -> dict[str, Path]
```
Description: runs rule-based EDA per parameter combo. Validation and OOS runs are separate entrypoints (`run_validation_pipeline`, `run_oos_pipeline`) that reuse the shared walkforward engine in `utils.evaluation.walkforward.*`.

Validation output contract (`feature_type="rule_based"`):
- Root directory: `feature_research/shared_results/rule_based/{module_name}/validation/` (or `config.output_root / "rule_based" / module_name / "validation"`).
- Files: `folds.csv`, `fold_scores.csv`, `selection_summary.csv`, `report.json`, `walkforward_stability.png`, `fold_timeline.png`.
- `selection_summary.csv` includes `selected_feature` for each fold.

### `run_continuous_eda_pipeline`
Type: function  
Module: `feature_research/in_sample/continuous_binning/pipeline.py`

Signature:
```python
run_continuous_eda_pipeline(
    config: ResearchConfig,
    output_dir: Path,
) -> dict[str, Path]
```
Description: runs continuous-feature EDA per parameter combo. Validation and OOS runs are separate entrypoints that call the shared engine in `utils.evaluation.walkforward.*`.

Validation/OOS execution details:
- Stage 1 refits `ContinuousBinningModel` per fold using train-only rows for parameter scoring (no future-data leakage).
- Stage 2 (when full `ResearchConfig` is passed through) evaluates selected top-k params via production `Portfolio`/`DiversifiedEnsemble` and records per-fold portfolio Sharpe.

Validation output contract (`feature_type="continuous"`):
- Root directory: `feature_research/shared_results/continuous/{module_name}/validation/` (or `config.output_root / "continuous" / module_name / "validation"`).
- Files: `folds.csv`, `fold_scores.csv`, `selection_summary.csv`, `oos_metrics.csv`, `selected_params_detailed.csv`, `report.json`, `walkforward_stability.png`, `fold_timeline.png`, `summary.md`, `summary.html`.
- `selection_summary.csv` includes `selected_feature` for each fold.

## Examples
```python
from eda.parameter_analysis import ParameterAnalyzer

analyzer = ParameterAnalyzer(features_df, targets_df)
grid = {
    (5,): ["rsi_signal_D_lookback_5"],
    (14,): ["rsi_signal_D_lookback_14"],
}
results = analyzer.analyze_nd_parameters(
    feature_grid=grid,
    param_names=["lookback"],
    target_col="log_return_atr",
)
```

## Internal but required
- `eda.parameter_analysis._get_metric_name_from_object(...)` is private but imported by `FeatureExplorer` for metric labeling.
- `utils.core.helpers.parse_feature_column_name(...)` is required by `build_feature_metadata` and FeatureExplorer parameter grouping.
- Bias-node cache classes (`utils.cache.bias_node_cache.*`) are required when calling extraction APIs with `use_cache=True`.

## Errors & logging
- Exception-heavy APIs (feature extraction/analysis) primarily raise `ValueError` and `TypeError` with explicit alignment/contract messages.
- `data_cleaning` and `research.bias_node_helpers` are print-driven (status, warning, traceback) rather than structured logging.
- `ParameterAnalyzer`/`FeatureExplorer` include warning/debug prints for per-feature failures and continue where possible; hard-fail when no valid analysis rows remain.

## Target alignment and no-lookahead constraints
- Core rule: features at timestamp `t` are trained/evaluated against returns for `t+1` (via `shift(-1)` on intraday returns).
- Last sample per ticker is dropped because forward return is unavailable.
- ATR/EWSD scaling must use the same forward horizon (`t+1`) as the shifted return, not contemporaneous `t` values.
- Multi-ticker alignment is performed with datetime+`ticker` joins; optional millisecond offsets are a collision-avoidance mechanism, not a temporal signal.
- Any custom consumer must preserve index equality between `features_df` and `targets_df` before EDA/modeling.

## Open questions
Q1: `data_cleaning/data_cleaning.py` currently behaves as script-style utilities (print/continue on errors). Should this surface be formalized with raised exceptions and logger-based events for production use?

Q2: `compute_forward_returns` type hints mark `features_df` optional, but current behavior requires it (for ATR/EWSD normalization). Should the annotation/docs be tightened to required?
