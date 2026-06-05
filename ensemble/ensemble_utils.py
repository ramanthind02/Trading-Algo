"""
Ensemble Utility Functions

This module provides utility functions for parsing unified control files,
creating base model instances, and managing feature configurations.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

import lib.core.helpers as helpers
from lib.core.ticker_key import normalize_ticker_key
from ensemble.vault.feature_files import validate_signed_signal_bias_node_spec
from features.models import BaseModel
from lib.core.enums import Direction, Ticker, TimeFrame, coerce_direction


def _reject_legacy_feature_artifact(message: str) -> None:
    raise ValueError(message)


def normalize_candles_datetime_column(candles_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure 'datetime' exists only as a column so sort_values('datetime') is unambiguous."""
    df = candles_df.copy()
    index_has_datetime = (
        getattr(df.index, "name", None) == "datetime"
        or (
            isinstance(df.index, pd.MultiIndex)
            and "datetime" in df.index.names
        )
    )
    if not index_has_datetime:
        return df
    if "datetime" in df.columns:
        df = df.reset_index(drop=True)
    else:
        df = df.reset_index()
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep="first")]
    return df


def parse_control_file(filepath: str) -> Dict[str, Any]:
    """
    Parse and validate unified control file.
    
    Parameters
    ----------
    filepath : str
        Path to control file JSON
        
    Returns
    -------
    Dict[str, Any]
        Parsed control file dictionary
        
    Raises
    ------
    FileNotFoundError
        If file not found
    ValueError
        If file format is invalid
    """
    try:
        with open(filepath, 'r') as f:
            control_file = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Control file not found: {filepath}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in control file: {e}")
    
    # Validate structure
    validate_control_file(control_file)
    
    return control_file


def _validate_signed_signal_model_config(
    config: Dict[str, Any],
    *,
    index: Optional[int] = None,
) -> None:
    prefix = f"Base model at index {index}: " if index is not None else "Base model: "

    required_keys = ["name", "model_type", "feature_column", "strategy", "bias_node_spec"]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"{prefix}Missing required keys: {missing_keys}")

    if config["model_type"] != "signed_signal":
        _reject_legacy_feature_artifact(
            f"{prefix}model_type must be 'signed_signal' (got {config['model_type']!r})."
        )

    try:
        strategy = coerce_direction(config["strategy"], field_name=f"{prefix}strategy")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{prefix}Invalid strategy: {config['strategy']}. "
            f"Must be one of: {[d.value for d in Direction]}"
        ) from exc
    config["strategy"] = strategy.value

    validate_signed_signal_bias_node_spec(config["bias_node_spec"], prefix=prefix)


def validate_control_file(control_file: Dict[str, Any]) -> None:
    """
    Validate control file structure.
    
    Parameters
    ----------
    control_file : Dict[str, Any]
        Control file dictionary to validate
        
    Raises
    ------
    ValueError
        If structure is invalid
    """
    # Check required top-level keys
    required_keys = ['metadata', 'base_models']
    missing_keys = [key for key in required_keys if key not in control_file]
    if missing_keys:
        raise ValueError(f"Control file missing required keys: {missing_keys}")

    # Validate metadata
    metadata = control_file.get('metadata', {})
    if 'is_fit' not in metadata:
        raise ValueError("Control file metadata must contain 'is_fit' flag")
    
    if not isinstance(metadata['is_fit'], bool):
        raise ValueError("Control file metadata 'is_fit' must be a boolean")
    
    is_fit = metadata['is_fit']

    # Validate base_models
    if not isinstance(control_file['base_models'], list):
        raise ValueError("base_models must be a list")
    
    for i, model_config in enumerate(control_file['base_models']):
        _validate_signed_signal_model_config(model_config, index=i)
    
    # If is_fit=True, validate fitted ensemble is present
    if is_fit:
        if 'fitted_ensemble' not in control_file:
            raise ValueError("Control file with is_fit=True must contain 'fitted_ensemble'")

        # Validate fitted_ensemble structure
        fitted_ensemble = control_file['fitted_ensemble']
        required_ensemble_keys = [
            'weights', 'exposure_fractions', 'feature_names', 'target_volatility',
            'unique_tickers', 'instrument_weights', 'n_tickers'
        ]
        missing_ensemble_keys = [key for key in required_ensemble_keys if key not in fitted_ensemble]
        if missing_ensemble_keys:
            raise ValueError(f"fitted_ensemble missing required keys: {missing_ensemble_keys}")
    
    # If is_fit=False, fitted params must be absent
    if not is_fit:
        if 'fitted_ensemble' in control_file:
            raise ValueError("Control file with is_fit=False should not contain 'fitted_ensemble'")


def save_control_file(
    filepath: str,
    base_models: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    fitted_ensemble: Optional[Dict[str, Any]] = None,
    tickers: Optional[List[str]] = None
) -> str:
    """
    Save unified control file.
    
    Parameters
    ----------
    filepath : str
        Path to save control file
    base_models : List[Dict[str, Any]]
        List of base model configurations
    metadata : Dict[str, Any]
        Metadata dictionary (must include is_fit flag)
    fitted_ensemble : Dict[str, Any], optional
        Fitted ensemble parameters (required if is_fit=True)
    tickers : List[str], optional
        List of ticker symbols
        
    Returns
    -------
    str
        Path where file was saved
        
    Raises
    ------
    ValueError
        If parameters are inconsistent with is_fit flag
    """
    # Validate metadata has is_fit
    if 'is_fit' not in metadata:
        raise ValueError("metadata must contain 'is_fit' flag")
    
    is_fit = metadata['is_fit']

    # Validate consistency
    if is_fit:
        if fitted_ensemble is None:
            raise ValueError("fitted_ensemble required when is_fit=True")
    elif fitted_ensemble is not None:
        raise ValueError("fitted_ensemble should not be provided when is_fit=False")
    
    # Build control file structure
    control_file = {
        'metadata': metadata,
        'base_models': base_models,
        'tickers': tickers or []
    }

    if is_fit:
        control_file['fitted_ensemble'] = fitted_ensemble
    
    # Validate before saving
    validate_control_file(control_file)
    
    # Create directory if needed
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(control_file, f, indent=2, default=str)
    
    return filepath


def validate_base_model_config(config: Dict[str, Any], index: Optional[int] = None) -> None:
    """
    Validate base model configuration.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Base model configuration dictionary
    index : int, optional
        Index of model in list (for error messages)
        
    Raises
    ------
    ValueError
        If configuration is invalid
    """
    _validate_signed_signal_model_config(config, index=index)


def create_base_model_from_config(
    config: Dict[str, Any],
    ticker: Optional[Ticker] = None,
    fitted_params: Optional[Dict[str, Any]] = None,
    use_cache: bool = True
) -> Any:
    """
    Factory function to create base model instances from configuration.
    
    Creates thin BaseModel instances backed by native signed-signal node specs.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Base model configuration with 'model_type' and native 'bias_node_spec'
    ticker : Ticker, optional
        Ticker symbol for the base model. If None, will try to extract from feature_column or use default.
    fitted_params : Dict[str, Any], optional
        Unsupported for signed-signal base models.
    use_cache : bool, default=True
        If True, BaseModel will use vectorized cached data when available.
        If False, uses streaming candle-by-candle processing.

    Returns
    -------
        BaseModel
        Instantiated BaseModel (node-backed, no fitted feature geometry)
        
    Raises
    ------
    ValueError
        If model_type is not supported or bias_node_spec cannot be determined
    """
    if fitted_params is not None:
        _reject_legacy_feature_artifact(
            "create_base_model_from_config no longer accepts fitted_params."
        )

    _validate_signed_signal_model_config(config)

    bias_node_spec = dict(config["bias_node_spec"])
    feature_config = {
        "model_type": "signed_signal",
        "feature_column": config["feature_column"],
        "bias_node_spec": bias_node_spec,
        "strategy": config["strategy"],
    }

    if 'tickers' in config:
        tickers_from_config = config['tickers']
        if isinstance(tickers_from_config, list):
            if tickers_from_config and isinstance(tickers_from_config[0], str):
                tickers_list = [Ticker[t] for t in tickers_from_config]
            else:
                tickers_list = [ticker if isinstance(ticker, Ticker) else Ticker[str(ticker)] for ticker in tickers_from_config]
        else:
            tickers_list = [Ticker[tickers_from_config]] if isinstance(tickers_from_config, str) else [tickers_from_config]
    elif ticker is not None:
        tickers_list = [ticker]
    else:
        tickers_list = [Ticker.ES]

    try:
        return BaseModel(
            feature_config=feature_config,
            tickers=tickers_list,
            use_cache=use_cache,
        )
    except ValueError as exc:
        raise ValueError(f"Unable to create signed_signal BaseModel: {exc}") from exc


def extract_bias_node_specs_from_control_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Extract bias node specifications from a control file.
    
    Parses feature column names to reconstruct the bias node specifications
    needed to recreate those features in MLManager.
    
    Parameters
    ----------
    filepath : str
        Path to control file JSON
        
    Returns
    -------
    List[Dict[str, Any]]
        List of bias node specifications in format:
        [{'module_name': str, 'timeframes': [TimeFrame], 'params': dict}, ...]
        
    Raises
    ------
    FileNotFoundError
        If file not found
    ValueError
        If file format is invalid
    """
    control_file = parse_control_file(filepath)
    bias_node_specs: List[Dict[str, Any]] = []
    seen_specs: set[tuple[str, tuple[str, ...], str]] = set()

    for i, model_config in enumerate(control_file["base_models"]):
        _validate_signed_signal_model_config(model_config, index=i)
        bias_node_spec = dict(model_config["bias_node_spec"])
        spec_key = (
            bias_node_spec["module_name"],
            tuple(
                tf.name if isinstance(tf, TimeFrame) else str(tf)
                for tf in bias_node_spec.get("timeframes", [])
            ),
            json.dumps(bias_node_spec.get("params", {}), sort_keys=True, default=str),
        )
        if spec_key in seen_specs:
            continue
        seen_specs.add(spec_key)
        bias_node_specs.append(bias_node_spec)

    return bias_node_specs


def aggregate_bias_node_specs_from_directory(directory: str) -> List[Dict[str, Any]]:
    """
    Aggregate bias node specifications from all control files in a directory.
    
    Scans directory for control file JSON files, extracts bias node specs from each,
    and returns a deduplicated list of all required bias nodes.
    
    Parameters
    ----------
    directory : str
        Path to directory containing control file JSON files
        
    Returns
    -------
    List[Dict[str, Any]]
        Aggregated list of unique bias node specifications
        
    Raises
    ------
    FileNotFoundError
        If directory not found
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    all_specs = []
    seen_specs = set()
    
    # Find all JSON files in directory
    for filename in os.listdir(directory):
        if not filename.endswith('.json'):
            continue
        
        filepath = os.path.join(directory, filename)
        try:
            # Try to parse as control file
            specs = extract_bias_node_specs_from_control_file(filepath)
            
            # Deduplicate across files
            for spec in specs:
                module_name = spec['module_name']
                tf = spec['timeframes'][0]
                params = spec['params']
                spec_key = (module_name, tf.name if hasattr(tf, 'name') else str(tf), tuple(sorted(params.items())))
                
                if spec_key not in seen_specs:
                    seen_specs.add(spec_key)
                    all_specs.append(spec)
        except (ValueError, FileNotFoundError, json.JSONDecodeError):
            # Skip files that aren't valid control files
            continue
    
    return all_specs


def filter_dataframe_by_timeframe(df: pd.DataFrame, base_tf: TimeFrame) -> pd.DataFrame:
    """
    Filter DataFrame columns to only include those matching the specified timeframe.
    
    Parses feature column names to extract timeframe information and filters
    columns where the parsed timeframe matches base_tf.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with feature columns (column names follow standardized format)
    base_tf : TimeFrame
        Target timeframe to filter for
        
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with only columns matching base_tf
        
    Notes
    -----
    Columns that cannot be parsed (non-feature columns) are kept in the result.
    This allows for columns like 'ticker', 'volatility', etc. to pass through.
    """
    if df.empty:
        return df
    
    matching_columns = []
    
    for col in df.columns:
        # Try to parse column name
        parsed = helpers.parse_feature_column_name(col)
        tf = parsed.get('tf')
        
        # If we can't parse it or it's not a feature column, keep it
        # (allows non-feature columns like 'ticker', 'volatility' to pass through)
        if tf is None:
            matching_columns.append(col)
            continue
        
        # Convert tf to TimeFrame enum if it's a string
        if isinstance(tf, str):
            try:
                tf = TimeFrame[tf]
            except (KeyError, AttributeError):
                # If we can't convert, keep the column (might be non-feature)
                matching_columns.append(col)
                continue
        
        # Check if timeframe matches
        if tf == base_tf:
            matching_columns.append(col)
    
    return df[matching_columns]
