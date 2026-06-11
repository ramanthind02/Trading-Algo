"""Simplified Feature Extraction Module

Direct feature extraction by iterating over candle dataframes.
Extracts one bias node at a time with parameter grid exploration.

**SINGLE BIAS NODE EXTRACTION**
-------------------------------
Extract features for a single bias node with multiple parameter combinations.
Columns = parameter combinations, Rows = timestamps.

Supports single or multiple tickers:
- Single ticker: ticker=Ticker.SPY
- Multiple tickers: ticker=[Ticker.ES, Ticker.NQ]

Examples:
    # Single parameter set, single ticker
    features_df, targets_df = extract_features(
        module_name='rsi',
        params={'lookback': 14},
        ticker=Ticker.SPY
    )
    
    # Multiple parameter sets (grid search)
    features_df, targets_df = extract_features(
        module_name='rsi',
        params={'lookback': [14, 21, 28]},
        ticker=Ticker.SPY
    )
    
    # Multiple tickers
    features_df, targets_df = extract_features(
        module_name='rsi',
        params={'lookback': 14},
        ticker=[Ticker.ES, Ticker.NQ]
    )
    
    # Multiple parameters with grid search, multiple tickers
    features_df, targets_df = extract_features(
        module_name='cmma',
        params={'lookback': [20, 50], 'atr_length': [252]},
        ticker=[Ticker.ES, Ticker.NQ, Ticker.YM]
    )
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import lib.core.helpers as helpers
from cache.runtime.central_cache import CentralCacheStore
from cache.runtime.central_cache_errors import (
    ArtifactMissingError,
    CacheCoverageError,
)
from cache.runtime.central_cache_models import (
    ArtifactDescriptor,
    ArtifactScope,
    CacheRequest,
)
from cache.runtime.feature_pipeline_support import (
    assign_cached_feature_values as _assign_cached_feature_values,
    build_bias_node_descriptor as _build_bias_node_descriptor,
    ensure_utc_datetime_index as _ensure_utc_datetime_index,
    expand_param_grid as _expand_param_grid,
    normalize_ticker_series as _normalize_ticker_series,
    normalize_ticker_str as _normalize_ticker_str,
    prepare_candles_override as _prepare_candles_override,
    preload_cross_ticker_data as _preload_cross_ticker_data,
    preload_cross_ticker_override_data as _preload_cross_ticker_override_data,
    read_aligned_feature_artifact as _read_aligned_feature_artifact,
    write_feature_artifact as _write_feature_artifact,
)
from lib.core.enums import TimeFrame, Ticker
from lib.core.models import Candle

# Metadata keys that may appear in persisted bias_spec JSON but are not bias-node params.
_RESERVED_BIAS_PARAM_KEYS: frozenset[str] = frozenset({"filter_specs"})


def _strip_reserved_bias_params(params: Dict[str, Any]) -> Dict[str, Any]:
    if not params or not _RESERVED_BIAS_PARAM_KEYS.intersection(params.keys()):
        return params
    return {k: v for k, v in params.items() if k not in _RESERVED_BIAS_PARAM_KEYS}


def _safe_log_return(close: pd.Series, open_: pd.Series) -> pd.Series:
    """Log return where close/open > 0 and finite; otherwise NaN. Avoids RuntimeWarning from log(0)."""
    ratio = close / open_
    valid = (ratio > 0) & np.isfinite(ratio)
    out = pd.Series(np.nan, index=ratio.index, dtype=float)
    out.loc[valid] = np.log(ratio.loc[valid])
    return out


def _find_feature_column(columns: List[str], keyword: str) -> Optional[str]:
    """Find a feature column by keyword while skipping non-feature columns."""
    candidates = [
        col for col in columns
        if col != "ticker" and keyword in col.lower()
    ]
    if not candidates:
        return None
    # Prefer canonical patterns like atr_* or *_atr_* over incidental substring matches.
    prioritized = sorted(
        candidates,
        key=lambda col: (
            0 if col.lower().startswith(f"{keyword}_") or f"_{keyword}_" in col.lower() else 1,
            len(col),
            col,
        ),
    )
    return prioritized[0]


def _normalize_timeframes(
    timeframes: Optional[List[Union[TimeFrame, str]]],
) -> List[TimeFrame]:
    """Normalize timeframe inputs to TimeFrame enums."""
    raw_timeframes: List[Union[TimeFrame, str]] = timeframes or [TimeFrame.D]
    return [
        TimeFrame[tf] if isinstance(tf, str) else tf
        for tf in raw_timeframes
    ]


def _compute_targets(price_df: pd.DataFrame, ewsd_col: Optional[str] = None) -> pd.DataFrame:
    """Compute target columns from price data."""
    raw_return = (price_df['close'] / price_df['open']) - 1
    log_return = _safe_log_return(price_df['close'], price_df['open'])

    log_return_ewsd = log_return.copy()
    if ewsd_col and ewsd_col in price_df.columns:
        ewsd_decimal = price_df[ewsd_col] / 100.0
        log_return_ewsd = log_return / np.maximum(ewsd_decimal, 0.0001)

    return pd.DataFrame({
        'raw_return': raw_return,
        'log_return': log_return,
        'log_return_ewsd': log_return_ewsd
    }, index=price_df.index)


def compute_forward_returns(
    candles_df: pd.DataFrame,
    features_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Compute intraday returns from candles DataFrame and shift forward by 1 period.
    
    Intraday returns are calculated as: return[t] = close[t] / open[t] - 1
    Returns are then shifted forward by 1 period so Feature[t] predicts Return[t+1],
    where Return[t+1] = (close[t+1]/open[t+1] - 1) represents the return from open[t+1] to close[t+1].
    
    This avoids lookahead bias: Feature[t] (computed at end of day t using data up to close[t])
    predicts Return[t+1] (the return for day t+1, stored at index t after shifting).
    
    Alignment:
    - Feature[t] (at index t) predicts Return[t+1] (at index t)
    - Return[t+1] = (close[t+1]/open[t+1] - 1) is the return for day t+1
    - Last row is dropped (no forward return available)
    
    Supports EWSD volatility scaling using the provided
    features_df. This is important for multi-ticker scenarios where different
    tickers have different volatility levels (e.g., NQ is more volatile than ES).

    If EWSD features are provided in features_df, returns include
    log_return_ewsd normalized by EWSD[t] (causal — known before Return[t+1]).
    If not provided, log_return_ewsd falls back to unscaled log_return.
    
    Parameters
    ----------
    candles_df : pd.DataFrame
        DataFrame with candles. Must have columns: datetime, open, close, ticker
        Should be sorted by ticker and datetime
    features_df : pd.DataFrame, optional
        Features dataframe containing an EWSD column for normalization.
        If provided and contains an EWSD feature (identified by containing
        'ewsd' in the name), log_return_ewsd is computed using EWSD[t+1].
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - raw_return: (close[t+1]/open[t+1]) - 1, shifted forward by 1 period
        - log_return: log(close[t+1]/open[t+1]), shifted forward by 1 period
        - log_return_ewsd: log_return normalized by EWSD[t] (causal) when EWSD is available;
          otherwise equal to log_return
        - ticker: ticker identifier
        Indexed by datetime (aligned with features)
        Last row per ticker is dropped (no forward return available)
        
    Examples
    --------
    >>> candles_df = helpers.load_data_multi_ticker(
    ...     tickers=[Ticker.ES, Ticker.NQ],
    ...     timeframe=TimeFrame.D,
    ...     start=datetime(2000, 1, 1),
    ...     end=datetime(2024, 12, 31)
    ... )
    >>> targets_df = compute_forward_returns(candles_df)
    >>> print(f"Targets shape: {targets_df.shape}")
    >>> print(f"Targets columns: {list(targets_df.columns)}")
    """
    targets_list: List[pd.DataFrame] = []
    
    for ticker in candles_df['ticker'].unique():
        ticker_candles = candles_df[candles_df['ticker'] == ticker].copy().sort_values('datetime')
        
        # Intraday return: log(close[t] / open[t])
        ticker_candles['log_return'] = _safe_log_return(ticker_candles['close'], ticker_candles['open'])
        ticker_candles['raw_return'] = (ticker_candles['close'] / ticker_candles['open']) - 1
        # Overnight return: log(open[t] / close[t-1])
        ticker_candles['overnight_log_return'] = _safe_log_return(
            ticker_candles['open'], ticker_candles['close'].shift(1)
        )

        # Shift all returns forward by 1 so Feature[t] predicts Return[t+1]
        ticker_candles['log_return'] = ticker_candles['log_return'].shift(-1)
        ticker_candles['raw_return'] = ticker_candles['raw_return'].shift(-1)
        ticker_candles['overnight_log_return'] = ticker_candles['overnight_log_return'].shift(-1)

        # Drop last row (no forward return available)
        ticker_candles = ticker_candles.dropna(subset=['log_return'])
        
        # Normalize ticker to string name to match extract_features format
        # load_data_multi_ticker sets ticker column to enum objects, we need string names
        ticker_name = _normalize_ticker_str(ticker)
        
        ewsd_series: Optional[pd.Series] = None
        if features_df is not None:
            # Filter features_df to this ticker if ticker column exists
            if 'ticker' in features_df.columns:
                ticker_features = features_df[features_df['ticker'] == ticker_name]
            else:
                ticker_features = features_df

            ticker_features_indexed = ticker_features
            if not isinstance(ticker_features.index, pd.DatetimeIndex):
                ticker_features_indexed = ticker_features.set_index(ticker_features.index)

            original_datetime_index = _ensure_utc_datetime_index(ticker_candles['datetime'].values)
            ewsd_col = _find_feature_column(list(ticker_features_indexed.columns), "ewsd")
            if ewsd_col and len(ticker_features_indexed) > 0:
                ticker_features_aligned = ticker_features_indexed.reindex(
                    original_datetime_index, method='ffill'
                )
                # Use EWSD[t] (causal: known before Return[t+1] is realised).
                # Using EWSD[t+1] would introduce lookahead: the EWMA at t+1 already
                # incorporates the next return, so crisis-day losses appear smaller
                # than they would under production position sizing (which uses EWSD[t]).
                ewsd_series = ticker_features_aligned[ewsd_col]
        
        # Use datetime column as index for targets (after dropping last row due to shift)
        # After shifting returns forward with shift(-1) and dropping last row:
        # - ticker_candles has indices [0, 1, 2, ..., N-1] with Return[1], Return[2], ..., Return[N]
        # - We want to align with features at indices [0, 1, 2, ..., N-1] with Feature[0], Feature[1], ..., Feature[N-1]
        # - Feature[t] at index t predicts Return[t+1] at index t
        original_target_index = _ensure_utc_datetime_index(ticker_candles['datetime'].values)
        
        # After dropping last row, ticker_candles has len(targets_df) rows
        # The indices are already aligned: [0, 1, 2, ..., len-1]
        # Feature[0] at index 0 predicts Return[1] at index 0
        # Feature[1] at index 1 predicts Return[2] at index 1
        # etc.
        target_index = original_target_index[:len(ticker_candles)]
        
        # Compute EWSD-normalized returns when EWSD is available.
        if ewsd_series is not None:
            ewsd_aligned = ewsd_series.reindex(target_index)
            ewsd_values = ewsd_aligned.values
            if len(ewsd_values) > 0 and not np.isnan(ewsd_values).all():
                ewsd_decimal = ewsd_values / 100.0
                log_return_ewsd = ticker_candles['log_return'].values / np.maximum(ewsd_decimal, 0.0001)
                overnight_log_return_ewsd = (
                    ticker_candles['overnight_log_return'].values / np.maximum(ewsd_decimal, 0.0001)
                )
            else:
                log_return_ewsd = ticker_candles['log_return'].values
                overnight_log_return_ewsd = ticker_candles['overnight_log_return'].values
        else:
            log_return_ewsd = ticker_candles['log_return'].values
            overnight_log_return_ewsd = ticker_candles['overnight_log_return'].values

        # Initialize target dict with returns
        target_dict = {
            'raw_return': ticker_candles['raw_return'].values,
            'log_return': ticker_candles['log_return'].values,
            'log_return_ewsd': log_return_ewsd,
            'overnight_log_return': ticker_candles['overnight_log_return'].values,
            'overnight_log_return_ewsd': overnight_log_return_ewsd,
            'ticker': ticker_name
        }
        
        ticker_targets = pd.DataFrame(target_dict, index=target_index)
        targets_list.append(ticker_targets)
    
    # Concatenate all tickers' targets (keyed by (datetime, ticker))
    targets_df = pd.concat(targets_list, axis=0).sort_index()
    
    return targets_df

def _extract_features_single_ticker(
    module_name: str,
    params: Dict[str, Any],
    ticker: Ticker,
    start: datetime,
    end: datetime,
    timeframes: List[TimeFrame],
    use_cache: bool = False,
    populate_on_miss: bool = False,
    cache_scope: ArtifactScope = ArtifactScope.LIVE,
    price_df_override: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract features for a single ticker.

    Parameters
    ----------
    use_cache : bool, default=False
        If True, load features from BiasNodeCache instead of streaming candles.
    populate_on_miss : bool, default=False
        If True, missing cache entries are materialized by streaming and then
        written back through the central cache facade.
    cache_scope : ArtifactScope, default=ArtifactScope.LIVE
        Storage namespace used for central-cache reads and writes.
    price_df_override : pd.DataFrame | None, default=None
        When provided, use this as price data instead of load_data and do not use
        cache (used for permutation so features are computed from shuffled candles).
    """
    if price_df_override is not None:
        price_df = price_df_override.copy()
        if price_df.index.name != 'datetime' and 'datetime' in price_df.columns:
            price_df = price_df.set_index('datetime')
        use_cache = False  # Must stream from override so features reflect shuffled data
        populate_on_miss = False
    else:
        price_df = helpers.load_data(ticker, timeframes[0], start=start, end=end)
        price_df.set_index('datetime', inplace=True)
    # Ensure timezone is UTC (may already be timezone-aware)
    if price_df.index.tz is None:
        price_df.index = price_df.index.tz_localize('UTC')
    else:
        price_df.index = price_df.index.tz_convert('UTC')
    
    # Expand parameter grid
    param_combos = _expand_param_grid(params)
    
    # Pre-load cross-ticker data if any param combo references a secondary ticker
    _preload_cross_ticker_data(param_combos, timeframes, start, end)
    
    # Create bias nodes for each parameter combination
    # Store tuples of (bias_node, param_combo, tf) to preserve original params for cache lookup
    bias_nodes = []
    bias_node_info = []  # List of (bias_node, param_combo, tf) for cache lookup fallback
    column_names = []
    
    for param_combo in param_combos:
        for tf in timeframes:
            bias_node = helpers.create_fresh_bias_node(
                module_name, ticker, tf, param_combo,
            )
            bias_nodes.append(bias_node)
            bias_node_info.append((bias_node, param_combo, tf))
            
            # Get column names for this node
            node_cols = bias_node.get_column_names() if hasattr(bias_node, 'get_column_names') else getattr(bias_node, 'columns', [])
            if not node_cols:
                node_cols = [f"{module_name}_{tf.name}"]
            column_names.extend(node_cols)
    
    # Initialize feature storage
    n_rows = len(price_df)
    n_cols = len(column_names)
    feature_data = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    
    # Map nodes to column ranges
    node_to_cols = {}
    col_idx = 0
    for bias_node in bias_nodes:
        node_cols = bias_node.get_column_names() if hasattr(bias_node, 'get_column_names') else getattr(bias_node, 'columns', [])
        if not node_cols:
            node_cols = [f"{bias_node.module_name}_{bias_node.tf.name}"]
        n_node_cols = len(node_cols)
        node_to_cols[bias_node] = (col_idx, col_idx + n_node_cols)
        col_idx += n_node_cols

    cache_store = CentralCacheStore.get_instance()

    streaming_required = not use_cache
    if use_cache:
        cache_requests: list[tuple[Any, ArtifactDescriptor, CacheRequest]] = []
        for bias_node, orig_params, orig_tf in bias_node_info:
            node_module = getattr(bias_node, "module_name", module_name) or module_name
            node_params = getattr(bias_node, "params", orig_params)
            node_tf = getattr(bias_node, "tf", orig_tf)
            descriptor = _build_bias_node_descriptor(
                module_name=node_module,
                params=node_params,
                ticker=ticker,
                tf=node_tf,
                scope=cache_scope,
            )
            cache_requests.append(
                (
                    bias_node,
                    descriptor,
                    CacheRequest(start=start, end=end),
                )
            )

        missing_requests: list[tuple[Any, ArtifactDescriptor, CacheRequest]] = []
        for bias_node, descriptor, request in cache_requests:
            try:
                cached_aligned = _read_aligned_feature_artifact(
                    cache_store,
                    descriptor,
                    request,
                    price_df.index,
                )
                start_col, end_col = node_to_cols[bias_node]
                _assign_cached_feature_values(
                    feature_data,
                    cached_aligned,
                    start_col,
                    end_col,
                )
            except (ArtifactMissingError, CacheCoverageError):
                if not populate_on_miss:
                    raise
                missing_requests.append((bias_node, descriptor, request))

        if missing_requests:
            if not populate_on_miss:
                raise ArtifactMissingError(
                    module_name=module_name,
                    ticker=ticker,
                    timeframe=timeframes[0] if timeframes else None,
                    requested_at=start,
                    reason="Feature cache miss and populate_on_miss is disabled.",
                )
            streaming_required = True
        else:
            streaming_required = False

    # STREAMING PATH: Iterate over candles (slow)
    if streaming_required:
        active_timeframe = timeframes[0]
        source_dependencies = [(ticker, active_timeframe)]
        for idx, (dt, row) in enumerate(price_df.iterrows()):
            candle = Candle(
                datetime=dt,
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row.get('volume', 0)),
                ticker=ticker,
                tf=active_timeframe,
            )

            for bias_node in bias_nodes:
                values = bias_node.add_candle(candle)
                start_col, end_col = node_to_cols[bias_node]

                for i in range(min(len(values), end_col - start_col)):
                    val = values[i]
                    if hasattr(val, 'value'):
                        val = val.value
                    feature_data[idx, start_col + i] = float(val) if val is not None else np.nan

        if populate_on_miss:
            cache_request = CacheRequest(start=start, end=end)
            for bias_node, orig_params, orig_tf in bias_node_info:
                node_module = getattr(bias_node, "module_name", module_name) or module_name
                node_params = getattr(bias_node, "params", orig_params)
                node_tf = getattr(bias_node, "tf", orig_tf)
                descriptor = _build_bias_node_descriptor(
                    module_name=node_module,
                    params=node_params,
                    ticker=ticker,
                    tf=node_tf,
                    scope=cache_scope,
                )
                start_col, end_col = node_to_cols[bias_node]
                cached_slice = pd.DataFrame(
                    feature_data[:, start_col:end_col],
                    index=price_df.index,
                    columns=column_names[start_col:end_col],
                )
                try:
                    _write_feature_artifact(
                        cache_store,
                        descriptor,
                        cached_slice,
                        source_dependencies,
                    )
                except Exception:
                    pass

    # Create features DataFrame
    features_df = pd.DataFrame(feature_data, index=price_df.index, columns=column_names)
    
    # Compute targets (EWSD if available)
    # CRITICAL: Avoid lookahead bias by ensuring Feature[t] predicts Return[t+1]
    # Feature[t] is computed at end of day t using data up to close[t].
    # Execution model: enter at open[t+1], exit at close[t+1] (intraday, no overnight hold).
    # Return[t→t+1] = log(close[t+1] / open[t+1]) — avoids overnight swap.
    # The validation path must use the same intraday definition; see
    # calculate_strategy_returns_from_positions(instrument_return_kind='log_intraday').
    ewsd_col = _find_feature_column(list(features_df.columns), "ewsd")

    # Intraday log return: Return[t] = log(close[t] / open[t])
    intraday_log_return = _safe_log_return(price_df['close'], price_df['open'])
    intraday_raw_return = (price_df['close'] / price_df['open']) - 1

    # Overnight log return: Return[t] = log(open[t] / close[t-1])
    # Execution model: enter at close[t], exit at open[t+1]  (pure gap capture).
    overnight_log_return = _safe_log_return(
        price_df['open'], price_df['close'].shift(1)
    )

    # Shift forward so Return[t+1] sits at index t: Feature[t] predicts Return[t+1]
    shifted_log_return      = intraday_log_return.shift(-1)
    shifted_raw_return      = intraday_raw_return.shift(-1)
    shifted_overnight_return = overnight_log_return.shift(-1)

    # Apply EWSD normalization using EWSD[t] (causal: known before Return[t+1] is realised).
    # Using EWSD[t+1] would introduce lookahead — the EWMA update on day t+1 already
    # incorporates the crash/spike return, making crisis-day losses appear smaller than
    # they actually are under production position sizing (which must use EWSD[t]).
    log_return_ewsd = shifted_log_return.copy()
    overnight_log_return_ewsd = shifted_overnight_return.copy()
    if ewsd_col and ewsd_col in features_df.columns:
        price_df[ewsd_col] = features_df[ewsd_col]
        ewsd_decimal = price_df[ewsd_col] / 100.0  # EWSD[t]: vol known at bar t
        log_return_ewsd = shifted_log_return / np.maximum(ewsd_decimal, 0.0001)
        overnight_log_return_ewsd = shifted_overnight_return / np.maximum(ewsd_decimal, 0.0001)

    # Create targets DataFrame
    targets_df = pd.DataFrame({
        'raw_return': shifted_raw_return,
        'log_return': shifted_log_return,
        'log_return_ewsd': log_return_ewsd,
        'overnight_log_return': shifted_overnight_return,
        'overnight_log_return_ewsd': overnight_log_return_ewsd,
    }, index=price_df.index)
    
    # Drop last row (no forward return available - return was shifted forward)
    targets_df = targets_df.dropna(subset=['log_return'])
    
    # Align features to match targets
    # After shifting forward and dropping last row:
    # - targets_df has indices [0, 1, 2, ..., N-1] with Return[1], Return[2], ..., Return[N]
    # - features_df has indices [0, 1, 2, ..., N] with Feature[0], Feature[1], ..., Feature[N]
    # - We want Feature[t] to predict Return[t+1]
    # - So Feature[0] predicts Return[1], Feature[1] predicts Return[2], etc.
    # Solution: Keep first len(targets_df) features (drop last feature row)
    if len(targets_df) > 0:
        # Keep first len(targets_df) features (drop last feature row - no Return[N+1] available)
        feature_indices = features_df.index[:len(targets_df)]
        features_df = features_df.loc[feature_indices]
        
        # Targets already have correct indices [0, 1, 2, ..., len-1]
        # These correspond to Return[1], Return[2], ..., Return[len]
        # Features at indices [0, 1, 2, ..., len-1] correspond to Feature[0], Feature[1], ..., Feature[len-1]
        # So Feature[t] at index t predicts Return[t+1] at index t
        # No need to change targets_df.index - it's already aligned correctly
    
    return features_df, targets_df


def extract_features(
    module_name: str,
    params: Dict[str, Any],
    ticker: Union[Ticker, List[Ticker]],
    start: datetime = None,
    end: datetime = None,
    timeframes: List[TimeFrame] = None,
    use_millisecond_offset: bool = True,
    use_cache: bool = False,
    populate_on_miss: bool = False,
    cache_scope: ArtifactScope = ArtifactScope.LIVE,
    candles_override: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract features for a single bias node with parameter grid exploration.
    Supports single or multiple tickers.
    
    Parameters
    ----------
    module_name : str
        Name of the bias node module (e.g., 'rsi', 'atr', 'cmma')
    params : Dict[str, Any]
        Parameters for the bias node. Supports grid search:
        - Single value: {'lookback': 14}
        - List of values: {'lookback': [14, 21, 28]} -> creates columns for each
        - Multiple params: {'lookback': [14, 21], 'period': 252} -> all combinations
    ticker : Ticker or List[Ticker]
        Single ticker or list of tickers to extract features for
    start : datetime, optional
        Start date. Defaults to datetime(1990, 1, 1)
    end : datetime, optional
        End date. Defaults to datetime.now()
    timeframes : List[TimeFrame], optional
        Timeframes to use. Defaults to [TimeFrame.D]
    use_millisecond_offset : bool, default=True
        Deprecated. Ignored. Primary key is (datetime, ticker).
    populate_on_miss : bool, default=False
        If True, missing cache entries are materialized by streaming and saved.
    cache_scope : ArtifactScope, default=ArtifactScope.LIVE
        Artifact scope for cache reads and writes.
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (features_df, targets_df)
        - features_df: Columns = parameter combinations, Rows = timestamps
          Includes 'ticker' column for identification
        - targets_df: Target columns (raw_return, log_return, etc.)
          Includes 'ticker' column for identification
        
    Examples
    --------
    >>> # Single parameter, single ticker
    >>> features_df, targets_df = extract_features(
    ...     module_name='rsi',
    ...     params={'lookback': 14},
    ...     ticker=Ticker.SPY
    ... )
    >>> 
    >>> # Parameter grid, single ticker
    >>> features_df, targets_df = extract_features(
    ...     module_name='rsi',
    ...     params={'lookback': [14, 21, 28]},
    ...     ticker=Ticker.SPY
    ... )
    >>> 
    >>> # Multiple tickers
    >>> features_df, targets_df = extract_features(
    ...     module_name='rsi',
    ...     params={'lookback': 14},
    ...     ticker=[Ticker.ES, Ticker.NQ]
    ... )
    >>> 
    >>> # Multiple tickers with parameter grid
    >>> features_df, targets_df = extract_features(
    ...     module_name='cmma',
    ...     params={'lookback': [20, 50], 'atr_length': [252]},
    ...     ticker=[Ticker.ES, Ticker.NQ, Ticker.YM]
    ... )
    """
    if start is None:
        start = datetime(1990, 1, 1)
    if end is None:
        end = datetime.now()
    timeframes = _normalize_timeframes(timeframes)
    params = _strip_reserved_bias_params(params)

    # Normalize ticker to list
    if isinstance(ticker, Ticker):
        tickers = [ticker]
    else:
        tickers = ticker

    # Build per-ticker override when provided (for permutation: features from shuffled candles)
    override_by_ticker: Dict[Ticker, pd.DataFrame] = {}
    if candles_override is not None and not candles_override.empty and 'ticker' in candles_override.columns:
        _preload_cross_ticker_override_data(candles_override, timeframes)
        for t in tickers:
            ticker_str = getattr(t, 'name', str(t))
            mask = candles_override['ticker'].astype(str) == ticker_str
            if mask.any():
                slice_df = candles_override.loc[mask].drop(columns=['ticker'], errors='ignore')
                if 'datetime' in slice_df.columns:
                    slice_df = slice_df.set_index('datetime')
                override_by_ticker[t] = slice_df

    # Single ticker path
    if len(tickers) == 1:
        features_df, targets_df = _extract_features_single_ticker(
            module_name=module_name,
            params=params,
            ticker=tickers[0],
            start=start,
            end=end,
            timeframes=timeframes,
            use_cache=use_cache if not override_by_ticker else False,
            populate_on_miss=populate_on_miss,
            cache_scope=cache_scope,
            price_df_override=override_by_ticker.get(tickers[0]),
        )
        
        # Add ticker column for identification
        features_df['ticker'] = tickers[0].name
        targets_df['ticker'] = tickers[0].name
        
        return features_df, targets_df
    
    # Multi-ticker path: extract for each ticker and concatenate
    all_features_dfs = []
    all_targets_dfs = []
    
    for single_ticker in tickers:
        ticker_features_df, ticker_targets_df = _extract_features_single_ticker(
            module_name=module_name,
            params=params,
            ticker=single_ticker,
            start=start,
            end=end,
            timeframes=timeframes,
            use_cache=use_cache if single_ticker not in override_by_ticker else False,
            populate_on_miss=populate_on_miss,
            cache_scope=cache_scope,
            price_df_override=override_by_ticker.get(single_ticker),
        )
        
        # Primary key is (datetime, ticker); no millisecond offset

        # Add ticker column for identification
        ticker_features_df['ticker'] = single_ticker.name
        ticker_targets_df['ticker'] = single_ticker.name
        
        all_features_dfs.append(ticker_features_df)
        all_targets_dfs.append(ticker_targets_df)
    
    # Concatenate all tickers' data
    features_df = pd.concat(all_features_dfs, axis=0).sort_index()
    targets_df = pd.concat(all_targets_dfs, axis=0).sort_index()
    
    return features_df, targets_df


def extract_features_with_forward_returns(
    module_name: str,
    params: Dict[str, Any],
    ticker: Union[Ticker, List[Ticker]],
    start: datetime = None,
    end: datetime = None,
    timeframes: List[TimeFrame] = None,
    use_millisecond_offset: bool = True,
    target_col: str = 'log_return',
    use_cache: bool = False,
    populate_on_miss: bool = False,
    cache_scope: ArtifactScope = ArtifactScope.LIVE,
    candles_override: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract features and compute intraday returns (shifted forward) automatically.
    
    This is a convenience wrapper around extract_features() that:
    1. Extracts features using extract_features() (main module + mandatory EWSD)
    2. Loads candles to compute intraday returns (close/open) and shifts them forward by 1 period
    3. Filters features to only those with shifted returns available
    4. Returns aligned features_df and targets_df with shifted intraday returns
    
    EWSD features are ALWAYS extracted automatically to enable volatility scaling.
    
    Return calculation:
    - Intraday return: Return[t] = (close[t]/open[t] - 1)
    - Shifted forward: shifted_return[t] = Return[t+1] = (close[t+1]/open[t+1] - 1)
    - Feature[t] (at index t) predicts Return[t+1] (return from open[t+1] to close[t+1])
    - This avoids lookahead bias: Feature[t] uses only data up to close[t], predicts next day's return
    
    Parameters
    ----------
    module_name : str
        Name of the bias node module (e.g., 'rsi', 'atr', 'cmma')
    params : Dict[str, Any]
        Parameters for the bias node. Supports grid search:
        - Single value: {'lookback': 14}
        - List of values: {'lookback': [14, 21, 28]} -> creates columns for each
        - Multiple params: {'lookback': [14, 21], 'period': 252} -> all combinations
    ticker : Ticker or List[Ticker]
        Single ticker or list of tickers to extract features for
    start : datetime, optional
        Start date. Defaults to datetime(1990, 1, 1)
    end : datetime, optional
        End date. Defaults to datetime.now()
    timeframes : List[TimeFrame], optional
        Timeframes to use. Defaults to [TimeFrame.D]
    use_millisecond_offset : bool, default=True
        Deprecated. Ignored. Primary key is (datetime, ticker).
    target_col : str, default='log_return'
        Target column to use. Options:
        - 'raw_return': (close[t+1]/open[t+1]) - 1, shifted forward by 1 period
        - 'log_return': log(close[t+1]/open[t+1]), shifted forward by 1 period
        - 'log_return_ewsd': log_return normalized by EWSD (recommended for multi-ticker)
    populate_on_miss : bool, default=False
        If True, materialize missing cache entries by streaming and store them.
    cache_scope : ArtifactScope, default=ArtifactScope.LIVE
        Artifact scope for cache reads and writes.
    candles_override : pd.DataFrame | None, default=None
        Optional candle dataframe override. When provided, this dataframe is used
        for forward-return computation instead of loading candles internally.
        Required columns: ``datetime``, ``open``, ``high``, ``low``, ``close``, ``ticker``.
        Primary key is (datetime, ticker); datetime is bar time (no offset).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (features_df, targets_df)
        - features_df: Columns = parameter combinations, Rows = timestamps
          Includes 'ticker' column for identification
          Filtered to only rows with forward returns available (last row dropped)
        - targets_df: Target columns with shifted intraday returns (raw_return, log_return, log_return_ewsd)
          Includes 'ticker' column for identification
          All target types are computed, but target_col indicates which one to use
          
    Examples
    --------
    >>> # Single parameter, single ticker
    >>> features_df, targets_df = extract_features_with_forward_returns(
    ...     module_name='rsi',
    ...     params={'lookback': 14},
    ...     ticker=Ticker.SPY
    ... )
    >>> 
    >>> # Parameter grid, multiple tickers
    >>> features_df, targets_df = extract_features_with_forward_returns(
    ...     module_name='rsi',
    ...     params={'lookback': [14, 21, 28]},
    ...     ticker=[Ticker.ES, Ticker.NQ, Ticker.YM]
    ... )
    """
    if start is None:
        start = datetime(1990, 1, 1)
    if end is None:
        end = datetime.now()
    timeframes = _normalize_timeframes(timeframes)
    params = _strip_reserved_bias_params(params)

    # Normalize ticker to list
    if isinstance(ticker, Ticker):
        tickers = [ticker]
    else:
        tickers = ticker

    # Load candles to compute forward returns
    have_candles_override = candles_override is not None
    candles_df = candles_override
    if candles_df is None:
        candles_df = helpers.load_data_multi_ticker(
            tickers=tickers,
            timeframe=timeframes[0],  # Use first timeframe
            start=start,
            end=end,
            use_millisecond_offset=use_millisecond_offset
        )
    else:
        candles_df = _prepare_candles_override(
            candles_override=candles_df,
            tickers=tickers,
            use_millisecond_offset=use_millisecond_offset,
        )

    # When using shuffled candles (permutation), features must come from override too (no cache)
    extract_kw: Dict[str, Any] = {
        "ticker": ticker,
        "start": start,
        "end": end,
        "timeframes": timeframes,
        "use_millisecond_offset": use_millisecond_offset,
        "use_cache": use_cache and not have_candles_override,
        "populate_on_miss": populate_on_miss,
        "cache_scope": cache_scope,
        "candles_override": candles_df if have_candles_override else None,
    }

    # STEP 1: Extract features - ALWAYS include EWSD for volatility scaling
    # Extract main module features (pass explicit kwargs so stray keys on extract_kw
    # cannot reach extract_features — e.g. legacy filter_specs on a shared dict).
    main_features_df, _ = extract_features(
        module_name=module_name,
        params=params,
        ticker=extract_kw["ticker"],
        start=extract_kw["start"],
        end=extract_kw["end"],
        timeframes=extract_kw["timeframes"],
        use_millisecond_offset=extract_kw["use_millisecond_offset"],
        use_cache=extract_kw["use_cache"],
        populate_on_miss=extract_kw["populate_on_miss"],
        cache_scope=extract_kw["cache_scope"],
        candles_override=extract_kw["candles_override"],
    )

    # Extract EWSD features (mandatory for volatility scaling)
    bars_per_year = timeframes[0].bars_per_year
    ewsd_features_df, _ = extract_features(
        module_name='ewsd',
        params={'long_run_window': 10 * bars_per_year},
        ticker=extract_kw["ticker"],
        start=extract_kw["start"],
        end=extract_kw["end"],
        timeframes=extract_kw["timeframes"],
        use_millisecond_offset=extract_kw["use_millisecond_offset"],
        use_cache=extract_kw["use_cache"],
        populate_on_miss=extract_kw["populate_on_miss"],
        cache_scope=extract_kw["cache_scope"],
        candles_override=extract_kw["candles_override"],
    )

    # Combine all features: main + EWSD
    # Since all dataframes are extracted with the same parameters, they share the same index.
    # Normalize ticker columns to strings for consistency
    main_features_df = main_features_df.copy()
    ewsd_features_df = ewsd_features_df.copy()

    for df in [main_features_df, ewsd_features_df]:
        if 'ticker' in df.columns:
            df['ticker'] = _normalize_ticker_series(df['ticker'])

    # Select columns to merge (exclude ticker from EWSD to avoid duplication)
    ewsd_cols = [col for col in ewsd_features_df.columns if col != 'ticker']

    # Combine: main features + EWSD columns (pd.concat aligns on index).
    features_to_concat = [main_features_df]
    if ewsd_cols:
        features_to_concat.append(ewsd_features_df[ewsd_cols])
    
    features_df = pd.concat(features_to_concat, axis=1)
    
    # Ensure ticker column is present (from main_features_df)
    if 'ticker' in main_features_df.columns and 'ticker' not in features_df.columns:
        features_df['ticker'] = main_features_df['ticker']
    
    # Validate that features were extracted
    if len(features_df) == 0:
        raise ValueError(
            f"No features extracted. Check that data exists for tickers {tickers} "
            f"in date range {start} to {end}"
        )
    
    # STEP 2: Compute forward returns with EWSD volatility scaling.
    # Pass features_df to enable EWSD normalization for multi-ticker scenarios.
    # Note: features_df may have more rows than candles_df (before forward return filtering)
    # We'll align them properly in STEP 3
    candles_for_targets = candles_df.copy()
    if "ticker" in candles_for_targets.columns:
        requested_ticker_names = {_normalize_ticker_str(t) for t in tickers}
        candles_for_targets["ticker"] = _normalize_ticker_series(candles_for_targets["ticker"])
        candles_for_targets = candles_for_targets[
            candles_for_targets["ticker"].isin(requested_ticker_names)
        ]
    targets_df = compute_forward_returns(candles_for_targets, features_df=features_df)
    
    if len(targets_df) == 0:
        raise ValueError(
            f"No forward returns computed. Check that data exists for tickers {tickers} "
            f"in date range {start} to {end}"
        )
    
    # Validate target_col is available
    valid_targets = ['raw_return', 'log_return', 'log_return_ewsd',
                     'overnight_log_return', 'overnight_log_return_ewsd']
    if target_col not in valid_targets:
        raise ValueError(
            f"target_col must be one of {valid_targets}, got '{target_col}'"
        )
    
    # Check if requested target column exists
    if target_col not in targets_df.columns:
        if target_col == 'log_return_ewsd':
            raise ValueError(
                f"Requested target_col '{target_col}' not available. "
                f"EWSD columns not found in features. "
                f"Available targets: {list(targets_df.columns)}"
            )
        else:
            raise ValueError(
                f"Requested target_col '{target_col}' not found in targets_df. "
                f"Available columns: {list(targets_df.columns)}"
            )
    
    # STEP 3: Align features to targets using simple merge
    # This is foolproof: only keep rows that exist in both dataframes
    # Targets are the source of truth (they already exclude rows without forward returns)
    
    # Normalize ticker columns to strings for consistent comparison
    features_df = features_df.copy()
    targets_df = targets_df.copy()
    
    # Ensure ticker columns are strings (vectorized to avoid millions of apply callbacks)
    if 'ticker' in features_df.columns:
        features_df['ticker'] = _normalize_ticker_series(features_df['ticker'])
    if 'ticker' in targets_df.columns:
        targets_df['ticker'] = _normalize_ticker_series(targets_df['ticker'])
    
    # Create a merge key: use index + ticker (if present) for reliable alignment
    # Ensure index has a name for consistent reset_index behavior
    features_df = features_df.copy()
    targets_df = targets_df.copy()
    
    if features_df.index.name is None:
        features_df.index.name = 'datetime'
    if targets_df.index.name is None:
        targets_df.index.name = 'datetime'
    
    # Reset index temporarily to use as merge key
    features_for_merge = features_df.reset_index()
    targets_for_merge = targets_df.reset_index()
    
    # Get the datetime column name (should be 'datetime' after we set index.name)
    datetime_col = 'datetime'
    
    # Verify the column exists after reset_index
    if datetime_col not in features_for_merge.columns:
        # If reset_index didn't create 'datetime', find the index column
        # (it might be unnamed or have a different name)
        index_cols = [col for col in features_for_merge.columns if col not in features_df.columns]
        if index_cols:
            datetime_col = index_cols[0]
        else:
            raise ValueError(
                f"Could not find datetime column after reset_index. "
                f"Features columns: {features_for_merge.columns.tolist()}, "
                f"Targets columns: {targets_for_merge.columns.tolist()}"
            )
    
    if 'ticker' in features_df.columns and 'ticker' in targets_df.columns:
        # Multi-ticker: merge on composite key (datetime, ticker)
        merged = pd.merge(
            features_for_merge,
            targets_for_merge[[datetime_col, 'ticker']],
            on=[datetime_col, 'ticker'],
            how='inner',
            suffixes=('', '_target')
        )
        if datetime_col in merged.columns:
            merged = merged.set_index(datetime_col)
        merged = merged.drop(columns=[col for col in merged.columns if col.endswith('_target')])

        # Get aligned features (all columns except target columns)
        target_cols = ['raw_return', 'log_return', 'log_return_ewsd',
                       'overnight_log_return', 'overnight_log_return_ewsd']
        feature_cols = [col for col in merged.columns if col not in target_cols]
        features_df_aligned = merged[feature_cols].copy()
        
        # Get aligned targets (merge back to get target values)
        targets_aligned = pd.merge(
            features_for_merge[[datetime_col, 'ticker']],
            targets_for_merge,
            on=[datetime_col, 'ticker'],
            how='inner'
        )
        if datetime_col in targets_aligned.columns:
            targets_aligned = targets_aligned.set_index(datetime_col)
        
    else:
        # Single ticker case: merge on index only
        # Merge on datetime index (inner join = only matching rows)
        merged = pd.merge(
            features_for_merge,
            targets_for_merge[[datetime_col]],  # Only merge key from targets
            on=datetime_col,
            how='inner',  # Only keep rows that exist in both
        )
        
        # Set datetime back as index
        if datetime_col in merged.columns:
            merged = merged.set_index(datetime_col)
        
        # Get aligned features (all columns except target columns)
        target_cols = ['raw_return', 'log_return', 'log_return_ewsd',
                       'overnight_log_return', 'overnight_log_return_ewsd']
        feature_cols = [col for col in merged.columns if col not in target_cols]
        features_df_aligned = merged[feature_cols].copy()
        
        # Get aligned targets
        targets_aligned = pd.merge(
            features_for_merge[[datetime_col]],
            targets_for_merge,
            on=datetime_col,
            how='inner'
        )
        if datetime_col in targets_aligned.columns:
            targets_aligned = targets_aligned.set_index(datetime_col)
    
    # Validate alignment
    if len(features_df_aligned) == 0:
        raise ValueError(
            "No matching rows found between features and targets after alignment. "
            "Ensure (datetime, ticker) keys align; check ticker column values and date ranges. "
            f"Features shape: {features_df.shape}, Targets shape: {targets_df.shape}"
        )
    
    if len(features_df_aligned) != len(targets_aligned):
        raise ValueError(
            f"Alignment failed: features and targets have different lengths after merge. "
            f"Features: {len(features_df_aligned)}, Targets: {len(targets_aligned)}"
        )

    if 'ticker' in features_df.columns and 'ticker' in features_df_aligned.columns:
        expected_tickers = set(features_df['ticker'].unique())
        aligned_tickers = set(features_df_aligned['ticker'].unique())
        missing_aligned_tickers = sorted(expected_tickers.difference(aligned_tickers))
        if missing_aligned_tickers:
            raise ValueError(
                "Tickers dropped during feature-target alignment. "
                f"Missing tickers: {missing_aligned_tickers}. "
                f"Expected tickers: {sorted(expected_tickers)}, "
                f"aligned tickers: {sorted(aligned_tickers)}"
            )
    
    # Ensure indices match exactly
    if not features_df_aligned.index.equals(targets_aligned.index):
        # Reindex to ensure exact match
        common_index = features_df_aligned.index.intersection(targets_aligned.index)
        features_df_aligned = features_df_aligned.loc[common_index]
        targets_aligned = targets_aligned.loc[common_index]
    
    return features_df_aligned, targets_aligned


def extract_features_for_bias_node(
    bias_spec: Dict[str, Any],
    ticker: Union[Ticker, List[Ticker]],
    start: datetime = None,
    end: datetime = None,
    use_millisecond_offset: bool = True,
    target_col: str = 'log_return',
    use_cache: bool = False,
    populate_on_miss: bool = False,
    cache_scope: ArtifactScope = ArtifactScope.LIVE,
    candles_override: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    High-level convenience function for extracting features from a bias node spec.
    
    This function simplifies the workflow by accepting a bias_spec dictionary
    (as used in BaseModel) and automatically handling parameter grid expansion
    and intraday returns computation (shifted forward) with volatility scaling.
    
    Return calculation:
    - Intraday return: Return[t] = (close[t]/open[t] - 1)
    - Shifted forward: shifted_return[t] = Return[t+1] = (close[t+1]/open[t+1] - 1)
    - Feature[t] (at index t) predicts Return[t+1] (return from open[t+1] to close[t+1])
    - This avoids lookahead bias: Feature[t] uses only data up to close[t], predicts next day's return
    
    Parameters
    ----------
    bias_spec : Dict[str, Any]
        Bias node specification with keys:
        - 'module_name': str (e.g., 'rsi', 'cmma')
        - 'timeframes': List[TimeFrame] (e.g., [TimeFrame.D])
        - 'params': Dict[str, Any] (e.g., {'lookback': [2, 5, 10, 14]})
          Supports grid search with lists
    ticker : Ticker or List[Ticker]
        Single ticker or list of tickers to extract features for
    start : datetime, optional
        Start date. Defaults to datetime(1990, 1, 1)
    end : datetime, optional
        End date. Defaults to datetime.now()
    use_millisecond_offset : bool, default=True
        Deprecated. Ignored. Primary key is (datetime, ticker).
    target_col : str, default='log_return'
        Target column to use. Options:
        - 'raw_return': (close[t+1]/open[t+1]) - 1, shifted forward by 1 period
        - 'log_return': log(close[t+1]/open[t+1]), shifted forward by 1 period
        - 'log_return_ewsd': log_return normalized by EWSD (recommended for multi-ticker)
    populate_on_miss : bool, default=False
        If True, materialize missing cache entries by streaming and store them.
    cache_scope : ArtifactScope, default=ArtifactScope.LIVE
        Artifact scope for cache reads and writes.
    candles_override : pd.DataFrame | None, default=None
        Optional candle dataframe override used by
        ``extract_features_with_forward_returns``.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (features_df, targets_df)
        - features_df: Columns = parameter combinations, Rows = timestamps
          Includes 'ticker' column for identification
          Last row dropped (no forward return available)
        - targets_df: Target columns with shifted intraday returns (raw_return, log_return, log_return_ewsd)
          Includes 'ticker' column for identification
          All target types are computed, but target_col indicates which one to use
          
    Examples
    --------
    >>> bias_spec = {
    ...     'module_name': 'rsi',
    ...     'timeframes': [TimeFrame.D],
    ...     'params': {'lookback': [2, 5, 10, 14]}
    ... }
    >>> 
    >>> features_df, targets_df = extract_features_for_bias_node(
    ...     bias_spec=bias_spec,
    ...     ticker=[Ticker.ES, Ticker.NQ, Ticker.YM],
    ...     start=datetime(2000, 1, 1),
    ...     end=datetime(2024, 12, 31)
    ... )
    """
    if start is None:
        start = datetime(1990, 1, 1)
    if end is None:
        end = datetime.now()
    
    # Extract bias spec components
    module_name = bias_spec.get('module_name')
    if module_name is None:
        raise ValueError("bias_spec must include 'module_name'")
    
    timeframes = bias_spec.get('timeframes', [TimeFrame.D])
    if not isinstance(timeframes, list):
        timeframes = [timeframes]
    
    params = _strip_reserved_bias_params(bias_spec.get('params', {}))
    
    # Use extract_features_with_forward_returns
    return extract_features_with_forward_returns(
        module_name=module_name,
        params=params,
        ticker=ticker,
        start=start,
        end=end,
        timeframes=timeframes,
        use_millisecond_offset=use_millisecond_offset,
        target_col=target_col,
        use_cache=use_cache,
        populate_on_miss=populate_on_miss,
        cache_scope=cache_scope,
        candles_override=candles_override,
    )
