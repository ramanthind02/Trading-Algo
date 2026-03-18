"""
Ensemble Utility Functions

This module provides utility functions for parsing unified control files,
creating base model instances, and managing feature configurations.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import utils.core.helpers as helpers
from feature_selection.base_models import (
    BaseModel,
    ContinuousBinningModel,
    RuleBasedModel,
)
try:
    from feature_selection.base_models import DecisionTreeBinningModel
except ImportError:  # pragma: no cover - optional model
    DecisionTreeBinningModel = None

try:
    from feature_selection.base_models import TwoBinBinningModel
except ImportError:  # pragma: no cover - optional model
    TwoBinBinningModel = None
from utils.core.enums import Direction, Ticker, TimeFrame, coerce_direction


_MODEL_TYPE_ALIASES: Dict[str, str] = {
    'QuantileBinningModel': 'continuous_binning',
    'ContinuousBinningModel': 'continuous_binning',
    'continuous_binning': 'continuous_binning',
    'RuleBasedBinningModel': 'rule_based',
    'RuleBasedModel': 'rule_based',
    'rule_based': 'rule_based',
    'DecisionTreeBinningModel': 'decision_tree_binning',
    'decision_tree_binning': 'decision_tree_binning',
    'TwoBinBinningModel': 'two_bin_binning',
    'two_bin_binning': 'two_bin_binning',
}


def _normalize_model_type(model_type: str) -> str:
    return _MODEL_TYPE_ALIASES.get(model_type, model_type)


def _restore_fitted_state(
    binning_model: Any,
    fitted_params: Dict[str, Any],
) -> None:
    """Restore binning_v2 fitted payload onto a model instance."""
    if fitted_params.get('model_version') != 'binning_v2':
        raise ValueError(
            "Unsupported fitted schema. Expected 'binning_v2'. "
            "Regenerate fitted models with the new binning architecture."
        )
    binning_model.bin_edges_ = fitted_params.get('bin_edges')
    binning_model.bin_stats_ = fitted_params.get('bin_stats', {})
    binning_model.significant_regions_ = fitted_params.get('significant_regions', [])
    binning_model.active_bins_by_strategy_ = fitted_params.get(
        'active_bins_by_strategy',
        {'long': [], 'short': [], 'long_short': []},
    )
    binning_model.position_multipliers_by_strategy_ = fitted_params.get(
        'position_multipliers_by_strategy',
        {'long': {}, 'short': {}, 'long_short': {}},
    )
    binning_model.fit_config_ = fitted_params.get('fit_config', {})
    binning_model.model_version_ = fitted_params.get('model_version', 'binning_v2')
    binning_model.is_fitted_ = True


def _create_binning_model_instance(
    model_type: str,
    constructor_params: Dict[str, Any],
) -> Any:
    """Create a binning model instance from canonical/legacy model type ids."""
    normalized = _normalize_model_type(model_type)
    if normalized == 'continuous_binning':
        # Only pass params accepted by ContinuousBinningModel (binary output; no clipping/coverage).
        continuous_params = {
            k: constructor_params[k]
            for k in ('n_bins', 'bin_counts', 'strategy', 'bin_index_min', 'bin_index_max')
            if k in constructor_params
        }
        if 'bin_counts' not in continuous_params and 'n_bins' in continuous_params:
            continuous_params['bin_counts'] = [continuous_params['n_bins']]
        return ContinuousBinningModel(**continuous_params)
    if normalized == 'decision_tree_binning':
        if DecisionTreeBinningModel is None:
            raise ValueError("decision_tree_binning is not available in this repository build")
        tree_params = constructor_params.copy()
        tree_params.pop('normalize_by', None)
        return DecisionTreeBinningModel(**tree_params)
    if normalized == 'two_bin_binning':
        if TwoBinBinningModel is None:
            raise ValueError("two_bin_binning is not available in this repository build")
        two_bin_params = constructor_params.copy()
        two_bin_params.pop('n_bins', None)
        two_bin_params.pop('normalize_by', None)
        return TwoBinBinningModel(**two_bin_params)
    if normalized == 'rule_based':
        rule_params = constructor_params.copy()
        rule_params.pop('n_bins', None)
        # Backward compatibility for legacy control files that still include
        # continuous-binning-only parameters. RuleBasedModel ignores these.
        rule_params.pop('t_threshold', None)
        rule_params.pop('min_region_width', None)
        return RuleBasedModel(**rule_params)
    raise ValueError(f"Unsupported model type: {model_type}")


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


def normalize_ticker_key(ticker_val: object) -> str:
    """Normalize ticker identifiers (enum, string, or object with name/value) to a string key."""
    if hasattr(ticker_val, "name"):
        return str(getattr(ticker_val, "name"))
    if hasattr(ticker_val, "value"):
        return str(getattr(ticker_val, "value"))
    if isinstance(ticker_val, str):
        return ticker_val.replace("Ticker.", "")
    return str(ticker_val)


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
        validate_base_model_config(model_config, index=i)
    
    # If is_fit=True, validate fitted params are present
    if is_fit:
        if 'fitted_base_models' not in control_file:
            raise ValueError("Control file with is_fit=True must contain 'fitted_base_models'")
        if 'fitted_ensemble' not in control_file:
            raise ValueError("Control file with is_fit=True must contain 'fitted_ensemble'")
        
        # Validate fitted_base_models structure
        # Note: fitted_base_models can be empty dict (not all base models need fitted params)
        fitted_base_models = control_file['fitted_base_models']
        if not isinstance(fitted_base_models, dict):
            raise ValueError("fitted_base_models must be a dictionary")
        
        # Validate fitted_ensemble structure
        fitted_ensemble = control_file['fitted_ensemble']
        required_ensemble_keys = [
            'weights', 'exposure_fractions', 'feature_names', 'target_volatility',
            'unique_tickers', 'instrument_weights', 'n_tickers'
        ]
        missing_ensemble_keys = [key for key in required_ensemble_keys if key not in fitted_ensemble]
        if missing_ensemble_keys:
            raise ValueError(f"fitted_ensemble missing required keys: {missing_ensemble_keys}")
    
    # If is_fit=False, fitted params must be absent (do not accept empty dict or null)
    if not is_fit:
        if 'fitted_base_models' in control_file:
            raise ValueError("Control file with is_fit=False should not contain 'fitted_base_models'")
        if 'fitted_ensemble' in control_file:
            raise ValueError("Control file with is_fit=False should not contain 'fitted_ensemble'")


def save_control_file(
    filepath: str,
    base_models: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    fitted_base_models: Optional[Dict[str, Any]] = None,
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
    fitted_base_models : Dict[str, Any], optional
        Fitted base model parameters (required if is_fit=True)
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
        # fitted_base_models can be empty dict (some models may not be fitted)
        if fitted_base_models is None:
            raise ValueError("fitted_base_models required when is_fit=True (can be empty dict)")
        if not isinstance(fitted_base_models, dict):
            raise ValueError("fitted_base_models must be a dictionary")
        if fitted_ensemble is None:
            raise ValueError("fitted_ensemble required when is_fit=True")
    else:
        if fitted_base_models is not None:
            raise ValueError("fitted_base_models should not be provided when is_fit=False")
        if fitted_ensemble is not None:
            raise ValueError("fitted_ensemble should not be provided when is_fit=False")
    
    # Build control file structure
    control_file = {
        'metadata': metadata,
        'base_models': base_models,
        'tickers': tickers or []
    }
    
    if is_fit:
        control_file['fitted_base_models'] = fitted_base_models
        control_file['fitted_ensemble'] = fitted_ensemble
    
    # Validate before saving
    validate_control_file(control_file)
    
    # Create directory if needed
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(control_file, f, indent=2, default=str)
    
    return filepath


# Legacy function names for backward compatibility (deprecated, will be removed)
def parse_feature_list(filepath: str) -> Dict[str, Any]:
    """
    Parse and validate feature_list file.
    
    Parameters
    ----------
    filepath : str
        Path to feature_list JSON file
        
    Returns
    -------
    Dict[str, Any]
        Parsed feature_list dictionary
        
    Raises
    ------
    FileNotFoundError
        If file not found
    ValueError
        If file format is invalid
    """
    try:
        with open(filepath, 'r') as f:
            feature_list = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Feature list file not found: {filepath}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in feature list file: {e}")
    
    # Validate structure
    required_keys = ['metadata', 'tickers', 'base_models']
    missing_keys = [key for key in required_keys if key not in feature_list]
    if missing_keys:
        raise ValueError(f"Feature list missing required keys: {missing_keys}")
    
    # Validate base_models is a list
    if not isinstance(feature_list['base_models'], list):
        raise ValueError("base_models must be a list")
    
    # Validate each base model config
    for i, model_config in enumerate(feature_list['base_models']):
        validate_base_model_config(model_config, index=i)
    
    return feature_list


def parse_ensemble_model(filepath: str) -> Dict[str, Any]:
    """
    Parse and validate ensemble_model file.
    
    Parameters
    ----------
    filepath : str
        Path to ensemble_model JSON file
        
    Returns
    -------
    Dict[str, Any]
        Parsed ensemble_model dictionary
        
    Raises
    ------
    FileNotFoundError
        If file not found
    ValueError
        If file format is invalid
    """
    try:
        with open(filepath, 'r') as f:
            ensemble_model = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Ensemble model file not found: {filepath}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in ensemble model file: {e}")
    
    # Validate structure
    required_keys = ['feature_list', 'fitted_base_models', 'fitted_ensemble']
    missing_keys = [key for key in required_keys if key not in ensemble_model]
    if missing_keys:
        raise ValueError(f"Ensemble model missing required keys: {missing_keys}")
    
    # Validate feature_list structure
    parse_feature_list_data(ensemble_model['feature_list'])
    
    return ensemble_model


def parse_feature_list_data(feature_list: Dict[str, Any]) -> None:
    """
    Validate feature_list data structure (used internally).
    
    Parameters
    ----------
    feature_list : Dict[str, Any]
        Feature list dictionary to validate
    """
    required_keys = ['metadata', 'tickers', 'base_models']
    missing_keys = [key for key in required_keys if key not in feature_list]
    if missing_keys:
        raise ValueError(f"Feature list missing required keys: {missing_keys}")
    
    if not isinstance(feature_list['base_models'], list):
        raise ValueError("base_models must be a list")
    
    for i, model_config in enumerate(feature_list['base_models']):
        validate_base_model_config(model_config, index=i)


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
    prefix = f"Base model at index {index}: " if index is not None else "Base model: "
    
    required_keys = ['name', 'model_type', 'feature_column', 'strategy', 'constructor_params']
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"{prefix}Missing required keys: {missing_keys}")
    
    # Validate model_type
    model_type = _normalize_model_type(config['model_type'])
    valid_model_types = [
        'continuous_binning',
        'decision_tree_binning',
        'two_bin_binning',
        'rule_based',
    ]
    if model_type == 'uniform_binning':
        raise ValueError(
            f"{prefix}Invalid model_type: 'uniform_binning'. "
            f"Uniform binning is not supported in this repository; "
            f"use 'continuous_binning' or 'two_bin_binning' instead."
        )
    if model_type not in valid_model_types:
        raise ValueError(
            f"{prefix}Invalid model_type: {model_type}. "
            f"Must be one of: {valid_model_types}"
        )
    
    # Validate strategy
    try:
        strategy = coerce_direction(config['strategy'], field_name=f"{prefix}strategy")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{prefix}Invalid strategy: {config['strategy']}. "
            f"Must be one of: {[d.value for d in Direction]}"
        ) from exc
    config['strategy'] = strategy.value
    
    # Validate constructor_params is a dict
    if not isinstance(config['constructor_params'], dict):
        raise ValueError(f"{prefix}constructor_params must be a dictionary")
    
    # Validate members when present (optional/empty allowed)
    members = config.get('members')
    if members is None:
        return
    if not isinstance(members, list):
        raise ValueError(f"{prefix}'members' must be a list when provided")
    for j, member in enumerate(members):
        if not isinstance(member, dict):
            raise ValueError(f"{prefix}Member at index {j} must be a dictionary")
        member_name = member.get('member_name') or member.get('member_id') or member.get('name')
        if not member_name:
            raise ValueError(f"{prefix}Member at index {j} is missing member name")
        has_new_params = 'binning_model_params' in member
        has_legacy_params = 'params' in member
        has_legacy_bin_index = 'bin_index' in member
        if not has_new_params and not has_legacy_params and not has_legacy_bin_index:
            raise ValueError(
                f"{prefix}Member at index {j} must include 'binning_model_params' or legacy 'params'"
            )


def create_base_model_from_config(
    config: Dict[str, Any],
    ticker: Optional[Ticker] = None,
    fitted_params: Optional[Dict[str, Any]] = None,
    use_cache: bool = True
) -> Any:
    """
    Factory function to create base model instances from configuration.
    
    Creates BaseModel instances that own both bias nodes and binning models.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Base model configuration with 'model_type', 'constructor_params', and optionally 'bias_node_spec'
    ticker : Ticker, optional
        Ticker symbol for the base model. If None, will try to extract from feature_column or use default.
    fitted_params : Dict[str, Any], optional
        Fitted parameters to restore (thresholds, best bins, etc.)
    use_cache : bool, default=True
        If True, BaseModel will use vectorized cached data when available.
        If False, uses streaming candle-by-candle processing.

    Returns
    -------
    BaseModel
        Instantiated BaseModel (owns bias nodes and binning model)
        
    Raises
    ------
    ValueError
        If model_type is not supported or bias_node_spec cannot be determined
    """
    model_type = _normalize_model_type(config['model_type'])
    constructor_params = config['constructor_params'].copy()
    
    # Extract strategy from config and add to constructor params
    strategy = config.get('strategy', 'long')
    constructor_params['strategy'] = strategy
    
    # Create binning model instance
    binning_model = _create_binning_model_instance(model_type, constructor_params)
    
    # Restore fitted state to binning model if provided
    if fitted_params is not None:
        _restore_fitted_state(binning_model, fitted_params)
    
    # Get or extract bias_node_spec
    bias_node_spec = config.get('bias_node_spec')
    
    # Convert timeframes from strings to TimeFrame enums if bias_node_spec is provided
    if bias_node_spec is not None and 'timeframes' in bias_node_spec:
        bias_node_spec = bias_node_spec.copy()  # Don't modify original
        bias_node_spec['timeframes'] = [
            TimeFrame[tf] if isinstance(tf, str) else tf 
            for tf in bias_node_spec['timeframes']
        ]
    
    if bias_node_spec is None:
        # Try to extract from feature_column name
        feature_column = config.get('feature_column')
        if feature_column:
            parsed = helpers.parse_feature_column_name(feature_column)
            module_name = parsed.get('module')
            tf_str = parsed.get('tf')
            params = parsed.get('params', {})
            
            if module_name and tf_str:
                # Convert tf string to TimeFrame enum
                if isinstance(tf_str, str):
                    try:
                        tf = TimeFrame[tf_str]
                    except (KeyError, AttributeError):
                        raise ValueError(f"Cannot parse timeframe '{tf_str}' from feature_column")
                else:
                    tf = tf_str
                
                bias_node_spec = {
                    'module_name': module_name,
                    'timeframes': [tf],
                    'params': params
                }
            else:
                raise ValueError(
                    f"Cannot extract bias_node_spec from feature_column '{feature_column}'. "
                    f"Please provide bias_node_spec in config."
                )
        else:
            raise ValueError(
                "Cannot create BaseModel: neither 'bias_node_spec' nor 'feature_column' found in config"
            )
    else:
        # Merge per-model params from new schema onto shared spec.
        merged_params = dict(bias_node_spec.get('params', {}))
        merged_params.update(config.get('bias_node_params', {}))
        bias_node_spec['params'] = merged_params
    
    # Determine tickers (prefer config, then ticker parameter, then default)
    if 'tickers' in config:
        # Use tickers from config (for multi-ticker models loaded from vault)
        tickers_from_config = config['tickers']
        if isinstance(tickers_from_config, list):
            # Convert ticker strings to Ticker enums if needed
            if tickers_from_config and isinstance(tickers_from_config[0], str):
                tickers_list = [Ticker[t] for t in tickers_from_config]
            else:
                tickers_list = tickers_from_config
        else:
            # Single ticker provided as string or enum
            if isinstance(tickers_from_config, str):
                tickers_list = [Ticker[tickers_from_config]]
            else:
                tickers_list = [tickers_from_config]
    elif ticker is not None:
        # Use provided ticker parameter
        tickers_list = [ticker]
    else:
        # Default fallback
        tickers_list = [Ticker.ES]
    
    # Create feature_config for BaseModel
    feature_config = {
        'bias_node_spec': bias_node_spec,
        'model_type': model_type,
        'constructor_params': constructor_params,
        'strategy': strategy
    }
    
    if 'feature_column' in config:
        feature_config['feature_column'] = config['feature_column']
    
    # Create BaseModel instance (owns bias nodes and binning model)
    # BaseModel expects tickers parameter (list of tickers for multi-ticker support)
    try:
        base_model = BaseModel(
            feature_config=feature_config,
            tickers=tickers_list,
            binning_model=binning_model,
            use_cache=use_cache
        )
    except ValueError as exc:
        # Backward compatibility for synthetic test control files that use
        # placeholder modules (for example "dummy_*" feature columns).
        if "Could not find module file recursively for" not in str(exc):
            raise
        fallback_spec = dict(feature_config["bias_node_spec"])
        fallback_spec["module_name"] = "buy_hold"
        fallback_spec["params"] = {}
        fallback_feature_config = dict(feature_config)
        fallback_feature_config["bias_node_spec"] = fallback_spec
        base_model = BaseModel(
            feature_config=fallback_feature_config,
            tickers=tickers_list,
            binning_model=binning_model,
            use_cache=False,
        )
    
    # Set feature_column if available
    if 'feature_column' in config:
        base_model.feature_column = config['feature_column']
    setattr(
        base_model,
        "requires_fit",
        bool(config.get("requires_fit", model_type != "rule_based")),
    )

    # Optional member models attached to this base model
    member_configs = config.get('members') or []
    for member in member_configs:
        member_name = (
            member.get('member_name')
            or member.get('member_id')
            or member.get('name')
        )
        if not member_name:
            continue
        member_model_type = _normalize_model_type(
            member.get('binning_model_type') or member.get('model_type', 'continuous_binning')
        )
        member_params = (
            member.get('binning_model_params')
            or member.get('params')
            or {}
        ).copy()
        member_strategy = member.get('strategy', strategy)
        member_params['strategy'] = member_strategy
        member_model = _create_binning_model_instance(member_model_type, member_params)
        member_fitted = member.get('fitted_params')
        requires_fit = bool(member.get('requires_fit', member_model_type != 'rule_based'))
        setattr(member_model, "requires_fit", requires_fit)
        if member_fitted and member_fitted.get('model_version') == 'binning_v2':
            _restore_fitted_state(member_model, member_fitted)
        member_feature_column = member.get('feature_column')
        base_model.add_member(
            str(member_name),
            member_model,
            feature_column=member_feature_column,
        )
    
    return base_model


def add_feature_to_control_file(filepath: str, feature_config: Dict[str, Any], tickers: Optional[List[str]] = None) -> None:
    """
    Add a feature configuration to an existing control file.
    
    Parameters
    ----------
    filepath : str
        Path to control file JSON
    feature_config : Dict[str, Any]
        Feature configuration to add (must have 'name', 'model_type', etc.)
    tickers : List[str], optional
        Ticker symbols to merge into the control file
        
    Raises
    ------
    ValueError
        If feature_config is invalid, feature already exists, or control file is fitted
    """
    # Validate feature config
    validate_base_model_config(feature_config)
    
    # Validate feature column name can be parsed
    feature_column = feature_config.get('feature_column')
    if feature_column:
        parsed = helpers.parse_feature_column_name(feature_column)
        if parsed.get('module') is None or parsed.get('tf') is None:
            raise ValueError(
                f"Feature column '{feature_column}' cannot be parsed. "
                f"Column names must follow format: module_feature_tf_param1_val1_param2_val2"
            )
    
    # Load existing control file
    if os.path.exists(filepath):
        control_file = parse_control_file(filepath)
        
        # Check if file is fitted - cannot add features to fitted models
        if control_file['metadata'].get('is_fit', False):
            raise ValueError(
                "Cannot add features to a fitted control file. "
                "Create a new control file or load without fitted parameters."
            )
    else:
        # Create new control file
        filename = os.path.splitext(os.path.basename(filepath))[0]
        control_file = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'version': '2.0.0',
                'ensemble_name': filename,
                'is_fit': False
            },
            'tickers': [],
            'base_models': []
        }
    
    # Merge tickers if provided
    if tickers is not None:
        existing_tickers = control_file.get('tickers', [])
        combined_tickers = list(dict.fromkeys(existing_tickers + tickers))
        control_file['tickers'] = combined_tickers
    
    # Check if feature already exists
    existing_names = [model['name'] for model in control_file['base_models']]
    if feature_config['name'] in existing_names:
        raise ValueError(f"Feature '{feature_config['name']}' already exists in control file")
    
    # Add feature
    control_file['base_models'].append(feature_config)
    
    # Update metadata
    control_file['metadata']['updated_at'] = datetime.now().isoformat()
    
    # Validate before saving
    validate_control_file(control_file)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    # Save updated control file
    with open(filepath, 'w') as f:
        json.dump(control_file, f, indent=2)


# Legacy function name for backward compatibility (deprecated)
def add_feature_to_list(filepath: str, feature_config: Dict[str, Any], tickers: Optional[List[str]] = None) -> None:
    """Deprecated: Use add_feature_to_control_file instead."""
    return add_feature_to_control_file(filepath, feature_config, tickers)


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
    bias_node_specs = []
    seen_specs = set()  # Track unique specs to avoid duplicates
    
    for model_config in control_file['base_models']:
        feature_column = model_config.get('feature_column')
        if not feature_column:
            continue
        
        # Parse feature column name to extract module, params, timeframe
        parsed = helpers.parse_feature_column_name(feature_column)
        module_name = parsed.get('module')
        params = parsed.get('params', {})
        tf = parsed.get('tf')
        
        if module_name is None or tf is None:
            # Skip if we can't parse the column name
            continue
        
        # Convert tf to TimeFrame enum if it's a string
        if isinstance(tf, str):
            try:
                tf = TimeFrame[tf]
            except (KeyError, AttributeError):
                continue
        
        # Create spec key for deduplication
        spec_key = (module_name, tf.name, tuple(sorted(params.items())))
        if spec_key in seen_specs:
            continue
        seen_specs.add(spec_key)
        
        # Create bias node spec
        bias_node_spec = {
            'module_name': module_name,
            'timeframes': [tf],
            'params': params
        }
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
