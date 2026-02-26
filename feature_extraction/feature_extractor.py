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
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import utils.core.helpers as helpers
from utils.core.enums import TimeFrame, Ticker
from utils.core.models import Candle


def _expand_param_grid(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand parameter grid to list of parameter dicts."""
    if not isinstance(params, dict):
        return [{}]
    
    # Check if any values are lists (grid search)
    has_lists = any(isinstance(v, list) for v in params.values())
    
    if not has_lists:
        return [params]
    
    # Grid search: expand all combinations
    keys = list(params.keys())
    values = [v if isinstance(v, list) else [v] for v in params.values()]
    
    return [dict(zip(keys, combo)) for combo in product(*values)]


def _normalize_ticker_str(ticker: object) -> str:
    """
    Normalize ticker objects (enums, strings, etc.) to a string representation.

    This keeps ticker handling consistent across single- and multi-ticker
    feature/target pipelines without changing external behavior.
    """
    if hasattr(ticker, "name"):
        return str(getattr(ticker, "name"))
    if isinstance(ticker, str):
        return ticker
    return str(ticker)


def _normalize_ticker_series(series: pd.Series) -> pd.Series:
    """
    Normalize a ticker column to strings without per-row apply().

    One type check then a single pass; avoids the apply/map_array overhead
    that dominated Stage 2 profile (millions of callbacks).
    """
    if series.empty:
        return series
    first = series.iloc[0]
    if hasattr(first, "name"):
        return pd.Series(
            [x.name for x in series],
            index=series.index,
            dtype=str,
        )
    if isinstance(first, str):
        return series
    return series.astype(str)


def _ensure_utc_datetime_index(values: object) -> pd.DatetimeIndex:
    """
    Convert arbitrary datetime-like values to a UTC-normalized DatetimeIndex.

    Mirrors the existing pattern:
    - Use pd.to_datetime to construct a DatetimeIndex
    - Localize to UTC when tz-naive
    - Convert to UTC when timezone-aware
    """
    datetime_values = pd.to_datetime(values)
    if isinstance(datetime_values, pd.Series):
        datetime_index = pd.DatetimeIndex(datetime_values.to_numpy())
    else:
        datetime_index = pd.DatetimeIndex(datetime_values)
    if datetime_index.tz is None:
        return datetime_index.tz_localize("UTC")
    return datetime_index.tz_convert("UTC")


def _prepare_candles_override(
    candles_override: pd.DataFrame,
    tickers: List[Ticker],
    use_millisecond_offset: bool,
) -> pd.DataFrame:
    """Validate and normalize override candles for forward-return computation.

    Primary key is (datetime, ticker). use_millisecond_offset is ignored (no offset).
    """
    required_columns = {"datetime", "open", "high", "low", "close", "ticker"}
    missing_columns = sorted(required_columns.difference(candles_override.columns))
    if missing_columns:
        raise ValueError(
            "candles_override missing required columns: "
            f"{missing_columns}. Expected columns include {sorted(required_columns)}"
        )

    normalized = candles_override.copy()
    normalized["datetime"] = _ensure_utc_datetime_index(normalized["datetime"])
    normalized["ticker"] = _normalize_ticker_series(normalized["ticker"])

    requested_tickers = [_normalize_ticker_str(ticker) for ticker in tickers]
    requested_set = set(requested_tickers)
    override_ticker_set = set(normalized["ticker"].unique())

    missing_tickers = sorted(requested_set.difference(override_ticker_set))
    if missing_tickers:
        raise ValueError(
            "candles_override missing ticker data for requested tickers: "
            f"{missing_tickers}. Available tickers: {sorted(override_ticker_set)}"
        )

    unexpected_tickers = sorted(override_ticker_set.difference(requested_set))
    if unexpected_tickers:
        raise ValueError(
            "candles_override contains unexpected tickers not requested: "
            f"{unexpected_tickers}. Requested tickers: {sorted(requested_set)}"
        )

    # Primary key is (datetime, ticker); no millisecond offset applied
    return normalized


def _safe_log_return(close: pd.Series, open_: pd.Series) -> pd.Series:
    """Log return where close/open > 0 and finite; otherwise NaN. Avoids RuntimeWarning from log(0)."""
    ratio = close / open_
    valid = (ratio > 0) & np.isfinite(ratio)
    out = pd.Series(np.nan, index=ratio.index, dtype=float)
    out.loc[valid] = np.log(ratio.loc[valid])
    return out


def _compute_targets(price_df: pd.DataFrame, atr_col: Optional[str] = None, ewsd_col: Optional[str] = None) -> pd.DataFrame:
    """Compute target columns from price data."""
    raw_return = (price_df['close'] / price_df['open']) - 1
    log_return = _safe_log_return(price_df['close'], price_df['open'])
    
    log_return_atr = log_return.copy()
    if atr_col and atr_col in price_df.columns:
        log_return_atr = log_return / np.maximum(price_df[atr_col], 0.0001)
    
    log_return_ewsd = log_return.copy()
    if ewsd_col and ewsd_col in price_df.columns:
        ewsd_decimal = price_df[ewsd_col] / 100.0
        log_return_ewsd = log_return / np.maximum(ewsd_decimal, 0.0001)
    
    return pd.DataFrame({
        'raw_return': raw_return,
        'log_return': log_return,
        'log_return_atr': log_return_atr,
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
    
    Supports volatility scaling (ATR/EWSD normalization) using the provided
    features_df. This is important for multi-ticker scenarios where different
    tickers have different volatility levels (e.g., NQ is more volatile than ES).
    
    ATR and EWSD features are MANDATORY: features_df must be provided and contain
    appropriate ATR/EWSD columns, otherwise this function raises an error.
    
    Parameters
    ----------
    candles_df : pd.DataFrame
        DataFrame with candles. Must have columns: datetime, open, close, ticker
        Should be sorted by ticker and datetime
    features_df : pd.DataFrame, optional
        Features dataframe containing ATR and EWSD columns for normalization.
        If provided, MUST contain:
        - An ATR column (identified by containing 'atr' and '252' in the name)
        - An EWSD column (identified by containing 'ewsd' in the name)
        Will compute log_return_atr and log_return_ewsd using these columns.
        If not provided, only raw_return and log_return will be computed.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - raw_return: (close[t+1]/open[t+1]) - 1, shifted forward by 1 period
        - log_return: log(close[t+1]/open[t+1]), shifted forward by 1 period
        - log_return_atr: log_return normalized by ATR[t+1] (mandatory if features_df provided)
        - log_return_ewsd: log_return normalized by EWSD[t+1] (mandatory if features_df provided)
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
        
        # Calculate intraday returns: return[t] = close[t] / open[t] - 1
        # This represents the return from open[t] to close[t] (during day t)
        ticker_candles['log_return'] = _safe_log_return(ticker_candles['close'], ticker_candles['open'])
        ticker_candles['raw_return'] = (ticker_candles['close'] / ticker_candles['open']) - 1
        
        # Shift returns forward by 1 period so Feature[t] predicts Return[t+1]
        # Return[t+1] = (close[t+1]/open[t+1] - 1) is the return for day t+1
        # After shift(-1): shifted_return[t] = Return[t+1]
        # This means Feature[t] (at index t) predicts Return[t+1] (the return for the next day)
        ticker_candles['log_return'] = ticker_candles['log_return'].shift(-1)
        ticker_candles['raw_return'] = ticker_candles['raw_return'].shift(-1)
        
        # Drop last row (no forward return available - return was shifted forward)
        ticker_candles = ticker_candles.dropna(subset=['log_return'])
        
        # Normalize ticker to string name to match extract_features format
        # load_data_multi_ticker sets ticker column to enum objects, we need string names
        ticker_name = _normalize_ticker_str(ticker)
        
        # ATR and EWSD are mandatory for volatility scaling
        # Find ATR and EWSD columns for this ticker
        if features_df is None:
            raise ValueError(
                "features_df is required for ATR/EWSD normalization. "
                "ATR and EWSD features must be extracted before computing returns."
            )
        
        # Filter features_df to this ticker if ticker column exists
        if 'ticker' in features_df.columns:
            ticker_features = features_df[features_df['ticker'] == ticker_name]
        else:
            ticker_features = features_df
        
        # Align features to candles by index (use original index before dropping)
        ticker_features_indexed = ticker_features
        if not isinstance(ticker_features.index, pd.DatetimeIndex):
            ticker_features_indexed = ticker_features.set_index(ticker_features.index)
        
        # Get original datetime index (before dropping first row)
        original_datetime_index = _ensure_utc_datetime_index(ticker_candles['datetime'].values)
        
        # Find ATR column (contains 'atr' and '252' in name)
        atr_col = next(
            (col for col in ticker_features_indexed.columns 
             if col != 'ticker' and 'atr' in col.lower() and '252' in col),
            None
        )
        
        # Find EWSD column (contains 'ewsd' in name)
        ewsd_col = next(
            (col for col in ticker_features_indexed.columns 
             if col != 'ticker' and 'ewsd' in col.lower()),
            None
        )
        
        # Validate that ATR and EWSD columns are found
        if atr_col is None:
            raise ValueError(
                f"ATR column not found in features_df. "
                f"Expected a column containing 'atr' and '252' in the name. "
                f"Available columns: {list(ticker_features_indexed.columns)}"
            )
        
        if ewsd_col is None:
            raise ValueError(
                f"EWSD column not found in features_df. "
                f"Expected a column containing 'ewsd' in the name. "
                f"Available columns: {list(ticker_features_indexed.columns)}"
            )
        
        # Align features to original datetime index (before dropping first row)
        if len(ticker_features_indexed) > 0:
            ticker_features_aligned = ticker_features_indexed.reindex(original_datetime_index, method='ffill')
        else:
            raise ValueError(
                f"No features found for ticker {ticker_name}. "
                f"Cannot compute volatility-scaled returns."
            )
        
        # Get ATR and EWSD values for normalizing Return[t+1]
        # Return[t+1] = (close[t+1]/open[t+1] - 1) should be normalized by ATR[t+1] and EWSD[t+1]
        # After shift(-1) on returns, shifted_return[t] = Return[t+1]
        # So we need ATR[t+1] and EWSD[t+1] to normalize it
        # We shift ATR/EWSD forward: ATR_shifted[t] = ATR[t+1]
        atr_series = ticker_features_aligned[atr_col].shift(-1)
        ewsd_series = ticker_features_aligned[ewsd_col].shift(-1)
        
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
        
        # Reindex ATR/EWSD to target_index (aligned with features)
        atr_aligned = atr_series.reindex(target_index)
        ewsd_aligned = ewsd_series.reindex(target_index)
        
        # Compute ATR-normalized return (mandatory)
        atr_values = atr_aligned.values
        if len(atr_values) == 0 or np.isnan(atr_values).all():
            raise ValueError(
                f"ATR values are all NaN for ticker {ticker_name}. "
                f"Cannot compute log_return_atr."
            )
        # ATR is typically in percentage, convert to decimal if needed
        atr_max = np.nanmax(atr_values)
        if atr_max > 1:
            atr_decimal = atr_values / 100.0
        else:
            atr_decimal = atr_values
        log_return_atr = ticker_candles['log_return'].values / np.maximum(atr_decimal, 0.0001)
        
        # Compute EWSD-normalized return (mandatory)
        ewsd_values = ewsd_aligned.values
        if len(ewsd_values) == 0 or np.isnan(ewsd_values).all():
            raise ValueError(
                f"EWSD values are all NaN for ticker {ticker_name}. "
                f"Cannot compute log_return_ewsd."
            )
        # EWSD is typically in percentage, convert to decimal
        ewsd_decimal = ewsd_values / 100.0
        log_return_ewsd = ticker_candles['log_return'].values / np.maximum(ewsd_decimal, 0.0001)
        
        # Initialize target dict with returns
        target_dict = {
            'raw_return': ticker_candles['raw_return'].values,
            'log_return': ticker_candles['log_return'].values,
            'log_return_atr': log_return_atr,
            'log_return_ewsd': log_return_ewsd,
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
    price_df_override: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract features for a single ticker.

    Parameters
    ----------
    use_cache : bool, default=False
        If True, load features from BiasNodeCache instead of streaming candles.
    price_df_override : pd.DataFrame | None, default=None
        When provided, use this as price data instead of load_data and do not use
        cache (used for permutation so features are computed from shuffled candles).
    """
    if price_df_override is not None:
        price_df = price_df_override.copy()
        if price_df.index.name != 'datetime' and 'datetime' in price_df.columns:
            price_df = price_df.set_index('datetime')
        use_cache = False  # Must stream from override so features reflect shuffled data
    else:
        price_df = helpers.load_data(ticker, TimeFrame.D, start=start, end=end)
        price_df.set_index('datetime', inplace=True)
    # Ensure timezone is UTC (may already be timezone-aware)
    if price_df.index.tz is None:
        price_df.index = price_df.index.tz_localize('UTC')
    else:
        price_df.index = price_df.index.tz_convert('UTC')
    
    # Expand parameter grid
    param_combos = _expand_param_grid(params)
    
    # Create bias nodes for each parameter combination
    # Store tuples of (bias_node, param_combo, tf) to preserve original params for cache lookup
    bias_nodes = []
    bias_node_info = []  # List of (bias_node, param_combo, tf) for cache lookup fallback
    column_names = []
    
    for param_combo in param_combos:
        for tf in timeframes:
            bias_node = helpers.create_bias_node(module_name, ticker, tf, param_combo)
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

    # CACHED PATH: Load from BiasNodeCache (fast)
    if use_cache:
        from utils.cache.bias_node_cache import BiasNodeCache, CacheMissError

        for bias_node, orig_params, orig_tf in bias_node_info:
            # Get params from bias node, falling back to original params if node doesn't have them
            # This ensures compatibility with nodes that don't set module_name/params attributes
            node_module = getattr(bias_node, 'module_name', module_name)
            node_params = getattr(bias_node, 'params', orig_params)
            node_tf = getattr(bias_node, 'tf', orig_tf)

            # Create cache instance
            cache = BiasNodeCache(
                module_name=node_module,
                params=node_params,
                ticker=ticker,
                tf=node_tf
            )

            if not cache.exists():
                raise CacheMissError(
                    module_name=node_module,
                    params=node_params,
                    ticker=ticker,
                    tf=node_tf,
                    date_range=(start, end),
                    cache_path=cache.cache_path,
                    reason="Cache file does not exist. Run CacheManager.populate_cache() first."
                )

            # Load cached values
            cached_df = cache.get_dataframe(start=start, end=end, require_cache=True)

            # Align cached data to price_df index
            # Remove timezone from cached index if needed for alignment
            if cached_df.index.tz is not None:
                cached_aligned = cached_df.reindex(price_df.index.tz_convert(cached_df.index.tz))
            else:
                cached_aligned = cached_df.reindex(price_df.index.tz_localize(None))

            # If alignment failed, try without timezone
            if cached_aligned.isna().all().all():
                price_index_naive = price_df.index.tz_localize(None) if price_df.index.tz else price_df.index
                cached_index_naive = cached_df.index.tz_localize(None) if cached_df.index.tz else cached_df.index
                cached_df_naive = cached_df.copy()
                cached_df_naive.index = cached_index_naive
                cached_aligned = cached_df_naive.reindex(price_index_naive)

            # Fill feature_data from cache
            start_col, end_col = node_to_cols[bias_node]

            if 'value' in cached_aligned.columns:
                feature_data[:, start_col] = cached_aligned['value'].values
            else:
                # Multi-column cache
                for i, col in enumerate(cached_aligned.columns):
                    if start_col + i < end_col:
                        feature_data[:, start_col + i] = cached_aligned[col].values

    # STREAMING PATH: Iterate over candles (slow)
    else:
        for idx, (dt, row) in enumerate(price_df.iterrows()):
            candle = Candle(
                datetime=dt,
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row.get('volume', 0)),
                ticker=ticker,
                tf=TimeFrame.D
            )

            for bias_node in bias_nodes:
                if bias_node.tf == TimeFrame.D:
                    values = bias_node.add_candle(candle)
                    start_col, end_col = node_to_cols[bias_node]

                    for i in range(min(len(values), end_col - start_col)):
                        val = values[i]
                        if hasattr(val, 'value'):
                            val = val.value
                        feature_data[idx, start_col + i] = float(val) if val is not None else np.nan

    # Create features DataFrame
    features_df = pd.DataFrame(feature_data, index=price_df.index, columns=column_names)
    
    # Compute targets (need ATR/EWSD if available)
    # CRITICAL: Avoid lookahead bias by ensuring Feature[t] predicts Return[t+1]
    # Feature[t] is computed at end of day t using data up to close[t]
    # Return[t+1] = (close[t+1]/open[t+1] - 1) is the return from open[t+1] to close[t+1]
    # This ensures no lookahead: Feature[t] uses only data available at end of day t
    # and predicts the return for the NEXT day (t+1)
    atr_col = next((col for col in features_df.columns if 'atr' in col.lower() and '252' in col), None)
    ewsd_col = next((col for col in features_df.columns if 'ewsd' in col.lower()), None)
    
    # Compute intraday returns: Return[t] = (close[t]/open[t] - 1)
    # This is the return DURING day t (from open to close)
    intraday_log_return = _safe_log_return(price_df['close'], price_df['open'])
    intraday_raw_return = (price_df['close'] / price_df['open']) - 1
    
    # Shift returns forward by 1 period so Return[t+1] aligns with Feature[t]
    # After shift: shifted_return[t] = Return[t+1] = (close[t+1]/open[t+1] - 1)
    # This means Feature[t] (at index t) predicts Return[t+1] (the return for day t+1)
    shifted_log_return = intraday_log_return.shift(-1)
    shifted_raw_return = intraday_raw_return.shift(-1)
    
    # Apply ATR/EWSD normalization if available
    # Use ATR/EWSD from the same period as the return (t+1)
    log_return_atr = shifted_log_return.copy()
    if atr_col and atr_col in features_df.columns:
        price_df[atr_col] = features_df[atr_col]
        # ATR[t+1] for normalizing Return[t+1]
        atr_shifted = price_df[atr_col].shift(-1)
        log_return_atr = shifted_log_return / np.maximum(atr_shifted, 0.0001)
    
    log_return_ewsd = shifted_log_return.copy()
    if ewsd_col and ewsd_col in features_df.columns:
        price_df[ewsd_col] = features_df[ewsd_col]
        # EWSD[t+1] for normalizing Return[t+1]
        ewsd_shifted = price_df[ewsd_col].shift(-1)
        ewsd_decimal = ewsd_shifted / 100.0
        log_return_ewsd = shifted_log_return / np.maximum(ewsd_decimal, 0.0001)
    
    # Create targets DataFrame
    targets_df = pd.DataFrame({
        'raw_return': shifted_raw_return,
        'log_return': shifted_log_return,
        'log_return_atr': log_return_atr,
        'log_return_ewsd': log_return_ewsd
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
    if timeframes is None:
        timeframes = [TimeFrame.D]
    
    # Normalize ticker to list
    if isinstance(ticker, Ticker):
        tickers = [ticker]
    else:
        tickers = ticker
    
    # Build per-ticker override when provided (for permutation: features from shuffled candles)
    override_by_ticker: Dict[Ticker, pd.DataFrame] = {}
    if candles_override is not None and not candles_override.empty and 'ticker' in candles_override.columns:
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
    candles_override: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract features and compute intraday returns (shifted forward) automatically.
    
    This is a convenience wrapper around extract_features() that:
    1. Extracts features using extract_features() (main module + mandatory ATR + EWSD)
    2. Loads candles to compute intraday returns (close/open) and shifts them forward by 1 period
    3. Filters features to only those with shifted returns available
    4. Returns aligned features_df and targets_df with shifted intraday returns
    
    ATR and EWSD features are ALWAYS extracted automatically to enable volatility scaling.
    
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
        - 'log_return_atr': log_return normalized by ATR (recommended for multi-ticker)
        - 'log_return_ewsd': log_return normalized by EWSD (recommended for multi-ticker)
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
        - targets_df: Target columns with shifted intraday returns (raw_return, log_return, log_return_atr, log_return_ewsd)
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
    if timeframes is None:
        timeframes = [TimeFrame.D]
    
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
        "candles_override": candles_df if have_candles_override else None,
    }

    # STEP 1: Extract features - ALWAYS include ATR and EWSD for volatility scaling
    # Extract main module features
    main_features_df, _ = extract_features(
        module_name=module_name,
        params=params,
        **extract_kw,
    )

    # Extract ATR features (mandatory for volatility scaling)
    atr_features_df, _ = extract_features(
        module_name='atr',
        params={'period': 252},
        **extract_kw,
    )

    # Extract EWSD features (mandatory for volatility scaling)
    ewsd_features_df, _ = extract_features(
        module_name='ewsd',
        params={},  # Use default parameters
        **extract_kw,
    )
    
    # Combine all features: main + ATR + EWSD
    # Since all dataframes are extracted with the same parameters, they share the same index
    # Use pd.concat to combine columns, which automatically aligns on index
    
    # Normalize ticker columns to strings for consistency
    main_features_df = main_features_df.copy()
    atr_features_df = atr_features_df.copy()
    ewsd_features_df = ewsd_features_df.copy()
    
    for df in [main_features_df, atr_features_df, ewsd_features_df]:
        if 'ticker' in df.columns:
            df['ticker'] = _normalize_ticker_series(df['ticker'])
    
    # Select columns to merge (exclude ticker from ATR/EWSD to avoid duplication)
    atr_cols = [col for col in atr_features_df.columns if col != 'ticker']
    ewsd_cols = [col for col in ewsd_features_df.columns if col != 'ticker']
    
    # Combine: main features + ATR columns + EWSD columns
    # pd.concat with axis=1 automatically aligns on index
    features_to_concat = [main_features_df]
    if atr_cols:
        features_to_concat.append(atr_features_df[atr_cols])
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
    
    # STEP 2: Compute forward returns with volatility scaling (ATR/EWSD are mandatory)
    # Pass features_df to enable ATR/EWSD normalization for multi-ticker scenarios
    # Note: features_df may have more rows than candles_df (before forward return filtering)
    # We'll align them properly in STEP 3
    targets_df = compute_forward_returns(candles_df, features_df=features_df)
    
    if len(targets_df) == 0:
        raise ValueError(
            f"No forward returns computed. Check that data exists for tickers {tickers} "
            f"in date range {start} to {end}"
        )
    
    # Validate target_col is available
    valid_targets = ['raw_return', 'log_return', 'log_return_atr', 'log_return_ewsd']
    if target_col not in valid_targets:
        raise ValueError(
            f"target_col must be one of {valid_targets}, got '{target_col}'"
        )
    
    # Check if requested target column exists
    if target_col not in targets_df.columns:
        if target_col in ['log_return_atr', 'log_return_ewsd']:
            raise ValueError(
                f"Requested target_col '{target_col}' not available. "
                f"ATR/EWSD columns not found in features. "
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
        target_cols = ['raw_return', 'log_return', 'log_return_atr', 'log_return_ewsd']
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
        target_cols = ['raw_return', 'log_return', 'log_return_atr', 'log_return_ewsd']
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


def prepare_candles_and_targets_for_basemodel(
    candles_df: pd.DataFrame,
    target_col: str = 'log_return'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare candles and targets for BaseModel fitting.
    
    This helper function:
    1. Computes forward returns from candles
    2. Filters candles to only those with forward returns available
    3. Resets candles index to integer index (as expected by BaseModel.fit)
    4. Creates target_series with proper datetime index
    
    Parameters
    ----------
    candles_df : pd.DataFrame
        DataFrame with candles. Must have columns: datetime, close, ticker
        Should be sorted by ticker and datetime
    target_col : str, default='log_return'
        Target column to extract from computed forward returns
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        (candles_df_fit, target_series)
        - candles_df_fit: Filtered candles with integer index (ready for BaseModel.fit)
        - target_series: Target values as Series with datetime index
        
    Examples
    --------
    >>> candles_df = helpers.load_data_multi_ticker(
    ...     tickers=[Ticker.ES, Ticker.NQ],
    ...     timeframe=TimeFrame.D,
    ...     start=datetime(2000, 1, 1),
    ...     end=datetime(2024, 12, 31)
    ... )
    >>> 
    >>> candles_fit, target_series = prepare_candles_and_targets_for_basemodel(candles_df)
    >>> 
    >>> # Now ready to fit BaseModel
    >>> base_model.fit(candles_fit, target_series)
    """
    # Compute forward returns
    targets_df = compute_forward_returns(candles_df)
    
    # Filter candles to match targets (only those with forward returns)
    # Use simple merge approach for reliable alignment
    candles_for_merge = candles_df.copy()
    targets_for_merge = targets_df.reset_index()
    
    # Merge on datetime (and ticker if present) to get only candles with forward returns
    if 'ticker' in candles_for_merge.columns and 'ticker' in targets_for_merge.columns:
        candles_for_merge['ticker'] = _normalize_ticker_series(candles_for_merge['ticker'])
        targets_for_merge['ticker'] = _normalize_ticker_series(targets_for_merge['ticker'])
        
        candles_filtered = pd.merge(
            candles_for_merge,
            targets_for_merge[['datetime', 'ticker']],
            on=['datetime', 'ticker'],
            how='inner'
        )
    else:
        candles_filtered = pd.merge(
            candles_for_merge,
            targets_for_merge[['datetime']],
            on='datetime',
            how='inner'
        )
    
    # Reset index to integer index (BaseModel.fit expects this)
    candles_fit = candles_filtered.reset_index(drop=True)
    
    # Create target_series with proper datetime index
    # Align targets to match filtered candles
    if 'ticker' in candles_filtered.columns and 'ticker' in targets_df.columns:
        # Multi-ticker: merge to align
        candles_for_target_merge = candles_filtered[['datetime', 'ticker']].copy()
        targets_for_target_merge = targets_df.reset_index()
        targets_aligned = pd.merge(
            candles_for_target_merge,
            targets_for_target_merge,
            on=['datetime', 'ticker'],
            how='inner'
        )
        target_series = pd.Series(
            targets_aligned[target_col].values,
            index=pd.DatetimeIndex(targets_aligned['datetime']),
            name=target_col
        )
    else:
        # Single ticker: align by datetime
        candles_for_target_merge = candles_filtered[['datetime']].copy()
        targets_for_target_merge = targets_df.reset_index()
        targets_aligned = pd.merge(
            candles_for_target_merge,
            targets_for_target_merge,
            on='datetime',
            how='inner'
        )
        target_series = pd.Series(
            targets_aligned[target_col].values,
            index=pd.DatetimeIndex(targets_aligned['datetime']),
            name=target_col
        )
    
    return candles_fit, target_series


def extract_features_for_bias_node(
    bias_spec: Dict[str, Any],
    ticker: Union[Ticker, List[Ticker]],
    start: datetime = None,
    end: datetime = None,
    use_millisecond_offset: bool = True,
    target_col: str = 'log_return',
    use_cache: bool = False,
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
        - 'log_return_atr': log_return normalized by ATR (recommended for multi-ticker)
        - 'log_return_ewsd': log_return normalized by EWSD (recommended for multi-ticker)
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
        - targets_df: Target columns with shifted intraday returns (raw_return, log_return, log_return_atr, log_return_ewsd)
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
    
    params = bias_spec.get('params', {})
    
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
        candles_override=candles_override,
    )
