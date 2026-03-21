"""
Vault Manager - Base Model Feature Management

This module provides functions for managing the vault system, which stores
validated trading features and their associated base models.

The vault organizes features by timeframe and ensemble, with each feature
having its own control file containing all base model variants.

Author: Trading Research Team
Date: 2025-01-07
"""

import json
import os
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

import utils.core.helpers as helpers
from feature_selection.base_models import ContinuousBinningModel, RuleBasedModel
from feature_selection.base_models.feature_base_model import BaseModel

try:
    from feature_selection.base_models import DecisionTreeBinningModel
except ImportError:
    DecisionTreeBinningModel = None

try:
    from feature_selection.base_models import TwoBinBinningModel
except ImportError:
    TwoBinBinningModel = None

from utils.core.enums import Direction, DirectionInput, TimeFrame, Ticker, coerce_direction
from utils.data.cross_ticker_store import SCALAR_LIST_PARAM_KEYS

# Type hint for forward reference
if TYPE_CHECKING:
    from ensemble.diversified_ensemble import DiversifiedEnsemble


# ============================================================================
# Module-Level Configuration
# ============================================================================

# Hardcoded vault root - always at project root
VAULT_ROOT: str = 'vault'

# Default ensemble directory/tickers (set by create_ensemble_directory). Use only for
# CLI/interactive defaults; in library code prefer passing ensemble_dir explicitly.
_DEFAULT_ENSEMBLE_DIR: Optional[str] = None
_DEFAULT_ENSEMBLE_TICKERS: Optional[List[Ticker]] = None


def get_default_ensemble_dir() -> Optional[str]:
    """
    Get the current default ensemble directory.
    
    Returns
    -------
    Optional[str]
        Current default ensemble directory, or None if not set
    """
    return _DEFAULT_ENSEMBLE_DIR


def get_default_ensemble_tickers() -> Optional[List[Ticker]]:
    """
    Get the current default ensemble tickers.
    
    Returns
    -------
    Optional[List[Ticker]]
        Current default ensemble tickers, or None if not set
    """
    return _DEFAULT_ENSEMBLE_TICKERS


def get_ensemble_tickers(ensemble_dir: Optional[str] = None) -> List[Ticker]:
    """
    Get tickers for an ensemble directory.
    
    Reads from ensemble_config.json if it exists, otherwise returns default.
    
    Parameters
    ----------
    ensemble_dir : str, optional
        Path to ensemble directory. If None, uses default ensemble directory.
        
    Returns
    -------
    List[Ticker]
        List of tickers for this ensemble
        
    Raises
    ------
    ValueError
        If ensemble_dir is None and no default is set, or if config file doesn't exist
    """
    global _DEFAULT_ENSEMBLE_DIR
    
    if ensemble_dir is None:
        ensemble_dir = _DEFAULT_ENSEMBLE_DIR
        if ensemble_dir is None:
            raise ValueError(
                "ensemble_dir must be provided or call create_ensemble_directory() first"
            )
    
    ensemble_path = _resolve_ensemble_path(ensemble_dir)
    ensemble_config_file = ensemble_path / 'ensemble_config.json'
    
    if not ensemble_config_file.exists():
        # Backward compatibility: return default
        return [Ticker.ES]
    
    with open(ensemble_config_file, 'r') as f:
        config = json.load(f)
    
    ticker_names = config.get('tickers', ['ES'])
    return [Ticker[ticker_name] if isinstance(ticker_name, str) else ticker_name for ticker_name in ticker_names]


def _resolve_ensemble_path(ensemble_dir: str) -> Path:
    """
    Resolve ensemble directory path, handling relative paths from any working directory.
    
    Tries multiple strategies to find the vault directory:
    1. Use path as-is if absolute
    2. Check current directory
    3. Check parent directory (for notebooks in research/)
    4. Walk up directory tree to find project root (where vault/ exists)
    
    Parameters
    ----------
    ensemble_dir : str
        Ensemble directory path (relative or absolute)
        
    Returns
    -------
    Path
        Resolved Path object pointing to ensemble directory
    """
    ensemble_path = Path(ensemble_dir)
    
    # If absolute path, use as-is
    if ensemble_path.is_absolute():
        return ensemble_path
    
    # Try current directory first
    if ensemble_path.exists():
        return ensemble_path
    
    # Try resolving from project root (where vault/ should be)
    cwd = Path.cwd()
    
    # Check current directory
    if (cwd / 'vault').exists():
        resolved = cwd / ensemble_dir
        if resolved.exists():
            return resolved
    
    # Check parent directory (common for notebooks in research/)
    if (cwd.parent / 'vault').exists():
        resolved = cwd.parent / ensemble_dir
        if resolved.exists():
            return resolved
    
    # Walk up directory tree to find project root
    current = cwd
    while current != current.parent:
        if (current / 'vault').exists():
            resolved = current / ensemble_dir
            if resolved.exists():
                return resolved
        current = current.parent
    
    # Try relative to where vault_manager.py is located
    vault_manager_dir = Path(__file__).parent.parent
    if (vault_manager_dir / 'vault').exists():
        resolved = vault_manager_dir / ensemble_dir
        if resolved.exists():
            return resolved
    
    # If still not found, return the original path (will fail later with better error)
    return ensemble_path


# ============================================================================
# Internal Helpers
# ============================================================================

_LEGACY_MODEL_TYPE_ALIASES: Dict[str, str] = {
    "QuantileBinningModel": "continuous_binning",
    "ContinuousBinningModel": "continuous_binning",
    "continuous_binning": "continuous_binning",
    "RuleBasedBinningModel": "rule_based",
    "RuleBasedModel": "rule_based",
    "rule_based": "rule_based",
    "DecisionTreeBinningModel": "decision_tree_binning",
    "decision_tree_binning": "decision_tree_binning",
    "TwoBinBinningModel": "two_bin_binning",
    "two_bin_binning": "two_bin_binning",
}


def _normalize_model_type(model_type: str) -> str:
    """Normalize old/new model type identifiers to canonical snake_case values."""
    if model_type in _LEGACY_MODEL_TYPE_ALIASES:
        return _LEGACY_MODEL_TYPE_ALIASES[model_type]
    if "_" in model_type:
        return model_type
    # Fallback CamelCase -> snake_case normalization
    name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", model_type)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()
    return name.replace("_model", "")


def _requires_fit(model_type: str) -> bool:
    """Rule-based models are treated as pre-defined and do not require refit."""
    return _normalize_model_type(model_type) != "rule_based"


def _normalize_bias_node_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize per-model bias params; unwrap singleton lists from legacy payloads.

    List-valued params that are semantically multi-valued (e.g. ``cross_tickers``)
    are preserved as-is even when they contain a single element.
    """
    if not params:
        return {}
    normalized: Dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, list) and len(value) == 1 and key not in SCALAR_LIST_PARAM_KEYS:
            normalized[key] = value[0]
        else:
            normalized[key] = value
    return normalized


def _model_id_token(value: Any) -> str:
    """Create a stable token safe for model-id suffixes."""
    if isinstance(value, list):
        # Join list elements for a clean token (e.g. ["TLT"] -> "tlt")
        value = "_".join(str(v) for v in value)
    token = re.sub(r"[^A-Za-z0-9]+", "-", str(value)).strip("-").lower()
    return token or "na"


def _extract_feature_name(feature_config: Dict[str, Any], fallback_stem: str) -> str:
    """Resolve feature name from new/legacy payload keys."""
    return (
        feature_config.get("feature_name")
        or feature_config.get("feature_column")
        or fallback_stem
    )


def _candidate_feature_names(raw_name: str) -> List[str]:
    """Generate likely feature-file stems for canonical and legacy names."""
    candidates = [raw_name]
    parsed = helpers.parse_feature_column_name(raw_name)
    tf = parsed.get("tf")
    tf_name = tf.name if hasattr(tf, "name") else str(tf) if tf else None
    if parsed.get("module") and parsed.get("feature") and tf_name:
        canonical = f"{parsed['module']}_{parsed['feature']}_{tf_name}"
        if canonical not in candidates:
            candidates.append(canonical)
    return candidates


def _resolve_feature_file_path(features_dir: Path, raw_name: str) -> Optional[Path]:
    """Find feature JSON path by filename or payload-level feature keys."""
    for candidate in _candidate_feature_names(raw_name):
        path = features_dir / f"{candidate}.json"
        if path.exists():
            return path
    for path in sorted(features_dir.glob("*.json")):
        try:
            with open(path, "r") as handle:
                payload = json.load(handle)
        except Exception:
            continue
        if payload.get("feature_name") == raw_name or payload.get("feature_column") == raw_name:
            return path
    return None


def _canonical_feature_name_from_tokens(
    raw_feature_name: str, bias_node_spec: Dict[str, Any]
) -> str:
    """
    Build canonical feature file name key: {module}_{feature}_{timeframe}.

    Falls back to the raw name if parsing fails.
    """
    parsed = helpers.parse_feature_column_name(raw_feature_name)
    module = parsed.get("module") or bias_node_spec.get("module_name")
    feature = parsed.get("feature")
    tf = parsed.get("tf")
    if hasattr(tf, "name"):
        tf_name = tf.name
    elif isinstance(tf, str):
        tf_name = tf
    else:
        spec_tfs = bias_node_spec.get("timeframes") or []
        first_tf = spec_tfs[0] if spec_tfs else None
        tf_name = first_tf.name if hasattr(first_tf, "name") else str(first_tf) if first_tf else None
    if module and feature and tf_name:
        return f"{module}_{feature}_{tf_name}"
    return raw_feature_name


def _serialize_bias_node_spec_for_storage(bias_node_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Store only module_name + timeframes in new schema."""
    timeframes = bias_node_spec.get("timeframes", [])
    serialized_tfs = [
        tf.name if isinstance(tf, TimeFrame) else str(tf)
        for tf in timeframes
    ]
    return {
        "module_name": bias_node_spec["module_name"],
        "timeframes": serialized_tfs,
    }


def _extract_bias_node_params_for_model(
    model_entry: Dict[str, Any],
    feature_config: Dict[str, Any],
    feature_name: str,
) -> Dict[str, Any]:
    """Read per-model bias params with legacy fallbacks."""
    if "bias_node_params" in model_entry:
        return _normalize_bias_node_params(model_entry.get("bias_node_params"))

    legacy_spec = feature_config.get("bias_node_spec", {})
    if "params" in legacy_spec:
        return _normalize_bias_node_params(legacy_spec.get("params"))

    parsed = helpers.parse_feature_column_name(feature_name)
    return _normalize_bias_node_params(parsed.get("params", {}))


# ============================================================================
# Model ID Generation
# ============================================================================

def generate_model_id(
    binning_model_type: str,
    binning_model_params: Dict[str, Any],
    bias_node_params: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Auto-generate model ID from binning model type and hyperparameters.
    
    Pattern: {binning_model_type_snake_case}_{key_hyperparam_values}
    
    Parameters
    ----------
    binning_model_type : str
        Binning model class name (e.g., 'continuous_binning')
    binning_model_params : Dict[str, Any]
        Constructor parameters for the binning model
        
    Returns
    -------
    str
        Auto-generated model ID
        
    Examples
    --------
    >>> generate_model_id('continuous_binning', {'n_bins': 3, 'selection_metric': 'sortino'})
    'continuous_binning_3'
    >>> generate_model_id('decision_tree_binning', {'n_bins': 5, 'min_samples_leaf_pct': 0.10})
    'decision_tree_binning_5'
    """
    name = _normalize_model_type(binning_model_type)
    
    # Append key hyperparameters (n_bins is always included)
    if 'n_bins' in binning_model_params:
        name += f"_{binning_model_params['n_bins']}"
    
    # Only include other hyperparameters that significantly change behavior
    # For DecisionTreeBinningModel, min_samples_leaf_pct is usually constant, so we skip it
    # Add more if needed for other model types
    
    # Append per-model bias-node params to avoid collisions between variants
    normalized_bias_params = _normalize_bias_node_params(bias_node_params)
    for key in sorted(normalized_bias_params):
        name += f"_{key}_{_model_id_token(normalized_bias_params[key])}"

    return name


# ============================================================================
# Ensemble Directory Management
# ============================================================================

def create_ensemble_directory(
    timeframe: Any,
    ensemble_name: Any,
    direction: DirectionInput,
    tickers: Optional[List[Ticker]] = None,
    vault_root: Optional[str] = None,
) -> str:
    """
    Create a new ensemble directory in the vault.
    
    The vault root is hardcoded to 'vault' at the project root.
    After creation, this ensemble directory becomes the default for all vault operations.
    
    Parameters
    ----------
    timeframe : TimeFrame
        Trading timeframe enum (D, W, M)
    ensemble_name : str
        Descriptive name for the ensemble (e.g., 'commodity_breakout', 'universal_momentum')
    direction : Direction
        Trading direction enum (LONG or SHORT)
    tickers : List[Ticker], optional
        List of tickers this ensemble will use. If None, defaults to [Ticker.ES].
        This is stored in ensemble_config.json and used to validate all features
        added to this ensemble have matching tickers.
        
    Returns
    -------
    str
        Path to the created ensemble directory (e.g., 'vault/D/commodity_breakout_long')
        
    Raises
    ------
    ValueError
        If ensemble directory already exists with different tickers
        
    Examples
    --------
    >>> ensemble_dir = create_ensemble_directory(
    ...     timeframe=TimeFrame.D,
    ...     ensemble_name='commodity_breakout',
    ...     direction=Direction.LONG,
    ...     tickers=[Ticker.ES, Ticker.NQ, Ticker.YM]
    ... )
    >>> print(ensemble_dir)  # 'vault/D/commodity_breakout_long'
    >>> # Now all vault functions use this as default
    >>> models = load_feature_base_models(feature_column='rsi_signal_D')
    """
    global _DEFAULT_ENSEMBLE_DIR, _DEFAULT_ENSEMBLE_TICKERS

    # Backward compatibility: legacy positional signature
    # create_ensemble_directory(vault_root, timeframe, ensemble_name, direction)
    if isinstance(timeframe, str) and isinstance(ensemble_name, TimeFrame):
        legacy_vault_root = timeframe
        legacy_timeframe = ensemble_name
        legacy_ensemble_name = str(direction)
        legacy_direction = tickers
        timeframe = legacy_timeframe
        ensemble_name = legacy_ensemble_name
        direction = legacy_direction
        tickers = None
        if vault_root is None:
            vault_root = legacy_vault_root

    if isinstance(timeframe, str):
        timeframe = TimeFrame[timeframe]
    direction = coerce_direction(direction, field_name="direction")

    # Default tickers if not provided
    if tickers is None:
        tickers = [Ticker.ES]
    
    # Normalize tickers to list
    if isinstance(tickers, Ticker):
        tickers = [tickers]
    
    # Convert to sorted list of names for consistency
    ticker_names = sorted([ticker.name if isinstance(ticker, Ticker) else ticker for ticker in tickers])
    
    # Build ensemble directory name
    ensemble_dir_name = f"{ensemble_name}_{direction.value}"
    
    # Build full path (hardcoded default root unless explicit override provided).
    if vault_root is not None:
        vault_path = Path(vault_root)
    else:
        cwd = Path.cwd()
        if (cwd / VAULT_ROOT).exists():
            vault_path = cwd / VAULT_ROOT
        elif (cwd.parent / VAULT_ROOT).exists():
            vault_path = cwd.parent / VAULT_ROOT
        else:
            # Walk up to find project root
            current = cwd
            while current != current.parent:
                if (current / VAULT_ROOT).exists():
                    vault_path = current / VAULT_ROOT
                    break
                current = current.parent
            else:
                vault_path = Path(VAULT_ROOT)  # Fallback to relative
    
    ensemble_path = vault_path / timeframe.name / ensemble_dir_name
    ensemble_dir_str = str(ensemble_path)
    
    # Ensemble config file path
    ensemble_config_file = ensemble_path / 'ensemble_config.json'
    
    # Check if already exists
    if ensemble_path.exists():
        # Directory exists - validate tickers match if config exists
        if ensemble_config_file.exists():
            with open(ensemble_config_file, 'r') as f:
                existing_config = json.load(f)
            existing_tickers = sorted(existing_config.get('tickers', []))
            
            if existing_tickers != ticker_names:
                raise ValueError(
                    f"Ensemble directory already exists with different tickers. "
                    f"Existing: {existing_tickers}, Provided: {ticker_names}. "
                    f"All features in an ensemble must use the same tickers."
                )
        else:
            # Create config file for existing directory
            config = {
                'timeframe': timeframe.name,
                'ensemble_name': ensemble_name,
                'direction': direction.value,
                'tickers': ticker_names,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
            with open(ensemble_config_file, 'w') as f:
                json.dump(config, f, indent=2)
        
        # Set as default
        _DEFAULT_ENSEMBLE_DIR = ensemble_dir_str
        _DEFAULT_ENSEMBLE_TICKERS = tickers
        return ensemble_dir_str
    
    # Create directory structure
    features_dir = ensemble_path / 'features'
    features_dir.mkdir(parents=True, exist_ok=False)
    
    # Create ensemble config file
    config = {
        'timeframe': timeframe.name,
        'ensemble_name': ensemble_name,
        'direction': direction.value,
        'tickers': ticker_names,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'updated_at': datetime.now(timezone.utc).isoformat()
    }
    with open(ensemble_config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Automatically set as default
    _DEFAULT_ENSEMBLE_DIR = ensemble_dir_str
    _DEFAULT_ENSEMBLE_TICKERS = tickers
    
    return ensemble_dir_str


def get_ensemble_path(
    timeframe: Any,
    ensemble_name: Any,
    direction: DirectionInput,
    vault_root: Optional[str] = None,
) -> str:
    """
    Get the path to an ensemble directory.
    
    The vault root is hardcoded to 'vault' at the project root.
    
    Parameters
    ----------
    timeframe : TimeFrame
        Trading timeframe enum
    ensemble_name : str
        Ensemble name
    direction : Direction
        Trading direction enum
        
    Returns
    -------
    str
        Path to the ensemble directory (e.g., 'vault/D/commodity_breakout_long')
    """
    # Backward compatibility: get_ensemble_path(vault_root, timeframe, ensemble_name, direction)
    if isinstance(timeframe, str) and isinstance(ensemble_name, TimeFrame):
        legacy_vault_root = timeframe
        legacy_timeframe = ensemble_name
        legacy_ensemble_name = str(direction)
        legacy_direction = vault_root
        timeframe = legacy_timeframe
        ensemble_name = legacy_ensemble_name
        direction = legacy_direction
        vault_root = legacy_vault_root

    if isinstance(timeframe, str):
        timeframe = TimeFrame[timeframe]
    direction = coerce_direction(direction, field_name="direction")

    ensemble_dir_name = f"{ensemble_name}_{direction.value}"
    root = vault_root or VAULT_ROOT
    ensemble_path = Path(root) / timeframe.name / ensemble_dir_name
    return str(ensemble_path)


def list_ensembles(vault_root: str) -> pd.DataFrame:
    """
    List all ensembles in the vault.
    
    Parameters
    ----------
    vault_root : str
        Root directory of the vault
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - ensemble_name: str
        - timeframe: str
        - direction: str
        - n_features: int
        - path: str (full path to ensemble directory)
    """
    vault_path = Path(vault_root)
    if not vault_path.exists():
        return pd.DataFrame(columns=['ensemble_name', 'timeframe', 'direction', 'n_features', 'path'])
    
    ensembles = []
    
    # Iterate through timeframe directories
    for tf_dir in vault_path.iterdir():
        if not tf_dir.is_dir() or tf_dir.name not in ['D', 'W', 'M']:
            continue
        
        timeframe = tf_dir.name
        
        # Iterate through ensemble directories
        for ensemble_dir in tf_dir.iterdir():
            if not ensemble_dir.is_dir():
                continue
            
            # Parse ensemble name and direction from directory name
            dir_name = ensemble_dir.name
            if dir_name.endswith('_long_short'):
                ensemble_name = dir_name[:-12]  # Remove '_long_short'
                direction = 'long_short'
            elif dir_name.endswith('_long'):
                ensemble_name = dir_name[:-5]  # Remove '_long'
                direction = 'long'
            elif dir_name.endswith('_short'):
                ensemble_name = dir_name[:-6]  # Remove '_short'
                direction = 'short'
            else:
                continue
            
            # Count features
            features_dir = ensemble_dir / 'features'
            n_features = len(list(features_dir.glob('*.json'))) if features_dir.exists() else 0
            
            ensembles.append({
                'ensemble_name': ensemble_name,
                'timeframe': timeframe,
                'direction': direction,
                'n_features': n_features,
                'path': str(ensemble_dir)
            })
    
    return pd.DataFrame(ensembles)


# ============================================================================
# Feature Management
# ============================================================================

def add_feature_to_ensemble(
    feature_name: Optional[str] = None,
    bias_node_spec: Optional[Dict[str, Any]] = None,
    bias_node_params: Optional[Dict[str, Any]] = None,
    base_model: Optional[BaseModel] = None,
    ensemble_dir: Optional[str] = None,
    tickers: Optional[List[Ticker]] = None,
    **legacy_kwargs: Any,
) -> str:
    """
    Add a base model variant to a feature control file.
    
    If the feature control file doesn't exist, creates it.
    If it exists, adds the new base model variant.
    Model ID is auto-generated based on binning model type and hyperparameters.
    
    Parameters
    ----------
    feature_column : str
        Feature column name (e.g., 'rsi_signal_D_lookback_2')
    bias_node_spec : Dict[str, Any]
        Bias node specification for feature reconstruction
        Format: {'module_name': str, 'timeframes': [TimeFrame], 'params': dict}
    base_model : BaseModel
        Fitted or unfitted base model instance (contains binning_model)
    ensemble_dir : str, optional
        Path to ensemble directory (e.g., 'vault/D/commodity_breakout_long').
        If None, uses default ensemble directory set via set_default_ensemble_dir().
    tickers : List[Ticker], optional
        List of tickers this ensemble was trained on. If None, infers from base_model.ticker.
        This is stored in the feature spec so base models can be created separately for each ticker.
        
    Returns
    -------
    str
        The auto-generated model_id
        
    Raises
    ------
    ValueError
        If model_id already exists for this feature, or if base model strategy
        doesn't match ensemble direction, or if ensemble_dir is None and no default is set
    """
    # Backward compatibility: accept feature_column kwarg
    if feature_name is None and "feature_column" in legacy_kwargs:
        feature_name = legacy_kwargs["feature_column"]

    # Backward compatibility: old positional signature had base_model as 3rd arg.
    if base_model is None and isinstance(bias_node_params, BaseModel):
        base_model = bias_node_params
        bias_node_params = None

    if feature_name is None:
        raise ValueError("feature_name must be provided")
    if bias_node_spec is None:
        raise ValueError("bias_node_spec must be provided")
    if base_model is None:
        raise ValueError("base_model must be provided")

    # Legacy support: bias params may still be nested in spec.params.
    if bias_node_params is None:
        bias_node_params = _normalize_bias_node_params(bias_node_spec.get("params", {}))
    else:
        bias_node_params = _normalize_bias_node_params(bias_node_params)

    # Use default ensemble_dir if not provided
    if ensemble_dir is None:
        ensemble_dir = _DEFAULT_ENSEMBLE_DIR
        if ensemble_dir is None:
            raise ValueError(
                "ensemble_dir must be provided or call create_ensemble_directory() first. "
                "Example: create_ensemble_directory(TimeFrame.D, 'buy_hold', Direction.LONG)"
            )
    
    ensemble_path = _resolve_ensemble_path(ensemble_dir)
    features_dir = ensemble_path / 'features'
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # Load ensemble config to get expected tickers
    ensemble_config_file = ensemble_path / 'ensemble_config.json'
    if ensemble_config_file.exists():
        with open(ensemble_config_file, 'r') as f:
            ensemble_config = json.load(f)
        expected_ticker_names = sorted(ensemble_config.get('tickers', []))
        expected_direction = coerce_direction(
            ensemble_config.get('direction', Direction.LONG.value),
            field_name="ensemble_config.direction",
        )
    else:
        # Backward compatibility: infer from directory name
        ensemble_dir_name = ensemble_path.name
        if ensemble_dir_name.endswith('_long_short'):
            expected_direction = Direction.LONG_SHORT
        elif ensemble_dir_name.endswith('_long'):
            expected_direction = Direction.LONG
        elif ensemble_dir_name.endswith('_short'):
            expected_direction = Direction.SHORT
        else:
            raise ValueError(
                f"Cannot determine ensemble direction from directory name: {ensemble_dir_name}. "
                f"Expected suffix _long, _short, or _long_short"
            )
        expected_ticker_names = None  # No ticker validation for old ensembles

    # Validate ensemble direction matches base model strategy
    model_strategy = coerce_direction(
        base_model.binning_model.strategy,
        field_name="base_model.binning_model.strategy",
    )
    if model_strategy != expected_direction:
        raise ValueError(
            f"Base model strategy '{model_strategy.value}' does not match "
            f"ensemble direction '{expected_direction.value}'"
        )
    
    # Validate tickers match ensemble tickers (if ensemble config exists)
    if expected_ticker_names is not None:
        if tickers is None:
            # Infer from base_model.ticker if available
            if hasattr(base_model, 'ticker') and base_model.ticker is not None:
                provided_ticker_names = sorted([base_model.ticker.name])
            else:
                provided_ticker_names = ['ES']  # Default
        else:
            provided_ticker_names = sorted([ticker.name if isinstance(ticker, Ticker) else ticker for ticker in tickers])
        
        if provided_ticker_names != expected_ticker_names:
            raise ValueError(
                f"Tickers do not match ensemble tickers. "
                f"Ensemble tickers: {expected_ticker_names}, Provided: {provided_ticker_names}. "
                f"All features in an ensemble must use the same tickers."
            )
    
    # Get binning model type and params
    binning_model = base_model.binning_model
    binning_model_type = _normalize_model_type(
        getattr(binning_model, "model_type", binning_model.__class__.__name__)
    )
    binning_model_params = dict(binning_model.get_params())
    binning_model_params.pop("strategy", None)
    
    # Generate model ID
    model_id = generate_model_id(
        binning_model_type,
        binning_model_params,
        bias_node_params=bias_node_params,
    )
    
    # Feature control file path
    feature_file = features_dir / f"{feature_name}.json"
    serializable_bias_spec = _serialize_bias_node_spec_for_storage(bias_node_spec)
    
    # Determine tickers to store
    # Priority: 1) Provided tickers, 2) Ensemble config tickers, 3) base_model.ticker, 4) Default
    if tickers is not None:
        # Use provided tickers (already validated above if ensemble config exists)
        pass
    elif expected_ticker_names is not None:
        # Use ensemble config tickers (convert strings back to Ticker enums)
        tickers = [Ticker[ticker_name] for ticker_name in expected_ticker_names]
    elif hasattr(base_model, 'ticker') and base_model.ticker is not None:
        # Fallback to base_model.ticker
        tickers = [base_model.ticker]
    else:
        # Default to ES if cannot infer
        tickers = [Ticker.ES]
    
    # Normalize tickers to list
    if isinstance(tickers, Ticker):
        tickers = [tickers]
    
    # Convert tickers to strings for JSON serialization (sorted for consistency)
    ticker_names = sorted([ticker.name if isinstance(ticker, Ticker) else ticker for ticker in tickers])
    
    # Load existing feature config or create new
    if feature_file.exists():
        with open(feature_file, 'r') as f:
            feature_config = json.load(f)
        existing_models = feature_config.get("base_models", [])
        if len(existing_models) > 1:
            raise ValueError(
                f"Legacy multi-model feature file is not supported: {feature_file}"
            )
        if existing_models and "members" in existing_models[0]:
            raise ValueError(
                f"Legacy member schema is not supported: {feature_file}"
            )
        
        # Validate existing tickers match ensemble tickers (if ensemble config exists)
        existing_ticker_names = sorted(feature_config.get('tickers', []))
        if expected_ticker_names is not None and existing_ticker_names != expected_ticker_names:
            raise ValueError(
                f"Feature '{feature_name}' already exists with tickers {existing_ticker_names} "
                f"but ensemble expects {expected_ticker_names}. "
                f"All features in an ensemble must use the same tickers."
            )
        
        # Use ensemble tickers (already validated above)
        feature_config['tickers'] = ticker_names
    else:
        # Create new feature config
        feature_config = {
            'feature_name': feature_name,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'updated_at': datetime.now(timezone.utc).isoformat(),
            'bias_node_spec': serializable_bias_spec,
            'tickers': ticker_names,
            'base_models': []
        }

    # Ensure top-level schema is canonical for new writes
    feature_config["feature_name"] = feature_name
    feature_config["bias_node_spec"] = serializable_bias_spec
    feature_config.pop("feature_column", None)
    
    if len(feature_config.get("base_models", [])) > 0:
        raise ValueError(
            f"Feature '{feature_name}' already has a base-model entry. "
            "Single-feature schema allows exactly one base model per feature file."
        )

    # Create base model entry
    model_entry = {
        'model_id': model_id,
        'model_name': f"{feature_name}::{model_id}",
        'bias_node_params': bias_node_params,
        'binning_model_type': binning_model_type,
        'strategy': model_strategy.value,
        'binning_model_params': binning_model_params,
        'requires_fit': _requires_fit(binning_model_type),
        'is_fitted': binning_model.is_fitted_,
        'fitted_params': None,
    }
    
    # Add fitted params if model is fitted
    if binning_model.is_fitted_:
        model_entry['fitted_at'] = datetime.now(timezone.utc).isoformat()
        fitted_payload = binning_model.get_fitted_params()
        if fitted_payload.get("model_version") == "binning_v2":
            model_entry['fitted_params'] = fitted_payload
        else:
            model_entry['is_fitted'] = False
            model_entry['fitted_params'] = None
    
    # Add model entry
    feature_config['base_models'].append(model_entry)
    feature_config['updated_at'] = datetime.now(timezone.utc).isoformat()
    
    # Save feature config
    with open(feature_file, 'w') as f:
        json.dump(feature_config, f, indent=2)
    
    return model_id


def load_feature_base_models(
    feature_name: Optional[str] = None,
    ensemble_dir: Optional[str] = None,
    fitted_only: bool = False,
    tickers: Optional[List[Ticker]] = None,
    **legacy_kwargs: Any,
) -> Dict[Tuple[Ticker, str], BaseModel]:
    """
    Load all base model variants for a feature, creating separate instances for each ticker.
    
    Parameters
    ----------
    feature_name : str
        Canonical feature name ({module}_{feature}_{tf}).
    ensemble_dir : str, optional
        Path to ensemble directory. If None, uses default ensemble directory set via
        create_ensemble_directory() or auto-detects by searching for the feature file.
    fitted_only : bool, default=False
        If True, only return fitted models
    tickers : List[Ticker], optional
        List of tickers to load models for. If None, uses tickers stored in feature spec.
        If feature spec doesn't have tickers, defaults to [Ticker.ES] for backward compatibility.
        
    Returns
    -------
    Dict[Tuple[Ticker, str], BaseModel]
        Dictionary mapping (ticker, model_id) tuple to BaseModel instance.
        Keys are tuples: (Ticker enum, model_id string)
        Values are reconstructed BaseModel instances with ticker-specific bias nodes.
        
    Raises
    ------
    ValueError
        If ensemble_dir is None, no default is set, and feature file cannot be auto-detected
        
    Examples
    --------
    >>> # Using explicit ensemble_dir
    >>> models = load_feature_base_models('rsi_signal_D', ensemble_dir='vault/D/ensemble_long')
    >>> es_model = models[(Ticker.ES, 'continuous_binning_3')]
    >>> 
    >>> # Using default ensemble_dir (set by create_ensemble_directory)
    >>> create_ensemble_directory(TimeFrame.D, 'buy_hold', Direction.LONG)
    >>> models = load_feature_base_models('buy_hold_signal_D')
    >>> 
    >>> # Auto-detection (searches vault for feature file)
    >>> models = load_feature_base_models('buy_hold_signal_D')  # Finds vault/D/buy_hold_long automatically
    """
    from ensemble.ensemble_utils import create_base_model_from_config

    global _DEFAULT_ENSEMBLE_DIR

    # Backward compatibility: accept feature_column kwarg.
    if feature_name is None and "feature_column" in legacy_kwargs:
        feature_name = legacy_kwargs["feature_column"]
    if feature_name is None:
        raise ValueError("feature_name must be provided")

    # Backward compatibility for historical positional usage:
    # load_feature_base_models(ensemble_dir, feature_name, ...)
    if ensemble_dir is not None and (
        "/" in str(feature_name) or str(feature_name).startswith("vault")
    ):
        if "/" not in str(ensemble_dir):
            feature_name, ensemble_dir = str(ensemble_dir), str(feature_name)

    # Use default ensemble_dir if not provided
    if ensemble_dir is None:
        ensemble_dir = _DEFAULT_ENSEMBLE_DIR
        if ensemble_dir is None:
            # Try to auto-detect: look for feature file in common locations
            cwd = Path.cwd()
            possible_paths = []
            
            # Check current directory and parent for vault
            for base in [cwd, cwd.parent]:
                if (base / 'vault').exists():
                    # Try to find ensemble directories with this feature
                    vault_path = base / 'vault'
                    for tf_dir in ['D', 'W', 'M']:
                        tf_path = vault_path / tf_dir
                        if tf_path.exists():
                            for ensemble_dir_path in tf_path.iterdir():
                                if ensemble_dir_path.is_dir():
                                    features_dir = ensemble_dir_path / "features"
                                    if _resolve_feature_file_path(features_dir, feature_name):
                                        possible_paths.append(str(ensemble_dir_path))
            
            # Also check relative to vault_manager.py location
            vault_manager_dir = Path(__file__).parent.parent
            if (vault_manager_dir / 'vault').exists():
                vault_path = vault_manager_dir / 'vault'
                for tf_dir in ['D', 'W', 'M']:
                    tf_path = vault_path / tf_dir
                    if tf_path.exists():
                        for ensemble_dir_path in tf_path.iterdir():
                            if ensemble_dir_path.is_dir():
                                features_dir = ensemble_dir_path / "features"
                                if _resolve_feature_file_path(features_dir, feature_name):
                                    possible_paths.append(str(ensemble_dir_path))
            
            if possible_paths:
                # Use first match and set as default for future calls
                ensemble_dir = possible_paths[0]
                _DEFAULT_ENSEMBLE_DIR = ensemble_dir
            else:
                raise ValueError(
                    f"ensemble_dir must be provided or call create_ensemble_directory() first. "
                    f"Could not auto-detect ensemble directory for feature '{feature_name}'. "
                    f"Example: create_ensemble_directory(TimeFrame.D, 'buy_hold', Direction.LONG)"
                )
    
    ensemble_path = _resolve_ensemble_path(ensemble_dir)
    features_dir = ensemble_path / "features"
    feature_file = _resolve_feature_file_path(features_dir, feature_name)
    
    if feature_file is None or not feature_file.exists():
        cwd = Path.cwd()
        raise FileNotFoundError(
            f"Feature file not found for feature '{feature_name}'\n"
            f"  Ensemble dir: {ensemble_dir}\n"
            f"  Resolved path: {ensemble_path}\n"
            f"  CWD: {cwd}\n"
            f"  Searched in: {features_dir}"
        )
    
    with open(feature_file, 'r') as f:
        feature_config = json.load(f)
    
    # Shared spec is now module+timeframes only; params live per base model.
    bias_node_spec = feature_config["bias_node_spec"].copy()
    if "timeframes" in bias_node_spec:
        bias_node_spec["timeframes"] = [
            TimeFrame[tf] if isinstance(tf, str) else tf
            for tf in bias_node_spec["timeframes"]
        ]

    resolved_feature_name = _extract_feature_name(feature_config, feature_file.stem)
    
    # Get tickers from feature spec or use provided/default
    if tickers is None:
        ticker_names = feature_config.get('tickers', [])
        if ticker_names:
            # Convert ticker name strings to Ticker enums
            tickers = [
                Ticker[ticker_name] if isinstance(ticker_name, str) else ticker_name
                for ticker_name in ticker_names
            ]
        else:
            # Backward compatibility: default to ES if no tickers stored
            tickers = [Ticker.ES]
    
    models: Dict[Tuple[Ticker, str], BaseModel] = {}
    feature_models = feature_config.get("base_models", [])
    if len(feature_models) != 1:
        raise ValueError(
            f"Feature '{resolved_feature_name}' must contain exactly one base model; "
            f"found {len(feature_models)} in {feature_file}"
        )
    model_entry = feature_models[0]
    if "members" in model_entry:
        raise ValueError(
            f"Legacy member schema is not supported in {feature_file}"
        )
    model_id = model_entry["model_id"]
    if fitted_only and not model_entry.get("is_fitted", False):
        return models

    model_type = _normalize_model_type(
        model_entry.get("binning_model_type", model_entry.get("model_type", "continuous_binning"))
    )
    model_strategy = model_entry.get("strategy", "long")
    model_params = dict(
        model_entry.get("binning_model_params", model_entry.get("constructor_params", {}))
    )
    model_params.pop("strategy", None)

    merged_bias_spec = bias_node_spec.copy()
    merged_bias_spec["params"] = _extract_bias_node_params_for_model(
        model_entry,
        feature_config,
        resolved_feature_name,
    )

    model_name = model_entry.get("model_name", f"{resolved_feature_name}::{model_id}")
    model_fitted_params = None
    if model_entry.get("is_fitted", False):
        fitted_payload = model_entry.get("fitted_params")
        if fitted_payload and fitted_payload.get("model_version") == "binning_v2":
            model_fitted_params = fitted_payload

    base_model_config = {
        "name": model_name,
        "feature_column": resolved_feature_name,
        "model_type": model_type,
        "strategy": model_strategy,
        "constructor_params": model_params,
        "bias_node_spec": merged_bias_spec,
        "bias_node_params": merged_bias_spec["params"],
    }

    for ticker in tickers:
        model_config_for_ticker = base_model_config.copy()
        model_config_for_ticker["tickers"] = [ticker]
        base_model = create_base_model_from_config(
            model_config_for_ticker,
            fitted_params=model_fitted_params,
        )
        base_model.feature_column = resolved_feature_name
        models[(ticker, model_id)] = base_model
    
    return models


def update_base_model_fitted_params(
    ensemble_dir: str,
    feature_name: Optional[str],
    model_id: str,
    fitted_params: Dict[str, Any],
    train_start: str,
    train_end: str,
    **legacy_kwargs: Any,
) -> None:
    """
    Update fitted parameters for a specific base model variant.
    
    Called after BaseModel.fit() to save the fitted state.
    
    Parameters
    ----------
    ensemble_dir : str
        Path to ensemble directory
    feature_name : str
        Feature name (canonical file stem). Legacy `feature_column` is accepted.
    model_id : str
        Model ID to update
    fitted_params : Dict[str, Any]
        Fitted parameters from the base model
    train_start : str
        Training start date (YYYY-MM-DD)
    train_end : str
        Training end date (YYYY-MM-DD)
    """
    if feature_name is None:
        feature_name = legacy_kwargs.get("feature_column")
    if feature_name is None:
        raise ValueError("feature_name must be provided")

    ensemble_path = _resolve_ensemble_path(ensemble_dir)
    features_dir = ensemble_path / "features"
    feature_file = _resolve_feature_file_path(features_dir, feature_name)
    if feature_file is None:
        raise ValueError(
            f"Feature file not found for feature '{feature_name}' in {features_dir}"
        )

    with open(feature_file, "r") as handle:
        feature_config = json.load(handle)

    feature_models = feature_config.get("base_models", [])
    if len(feature_models) != 1:
        raise ValueError(
            f"Feature '{feature_name}' must contain exactly one base model; found {len(feature_models)}"
        )
    if "members" in feature_models[0]:
        raise ValueError(
            f"Legacy member schema is not supported in {feature_file}"
        )

    model_found = False
    for model_entry in feature_models:
        if model_entry.get("model_id") != model_id:
            continue
        model_found = True
        is_valid_v2 = bool(
            fitted_params and fitted_params.get("model_version") == "binning_v2"
        )
        model_entry["is_fitted"] = is_valid_v2
        model_entry["fitted_at"] = datetime.now(timezone.utc).isoformat()
        model_entry["train_start"] = train_start
        model_entry["train_end"] = train_end
        model_entry["fitted_params"] = fitted_params if is_valid_v2 else None
        if not is_valid_v2:
            warnings.warn(
                f"Skipping non-binning_v2 fitted params for model '{model_id}'",
                RuntimeWarning,
            )
        break

    if not model_found:
        raise ValueError(f"Model ID '{model_id}' not found in feature '{feature_name}'")

    feature_config["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(feature_file, "w") as handle:
        json.dump(feature_config, handle, indent=2)


def consolidate_feature_files(ensemble_dir: str) -> List[str]:
    """
    Consolidate fragmented legacy feature files into canonical feature files.

    Returns list of consolidated file paths.
    """
    ensemble_path = _resolve_ensemble_path(ensemble_dir)
    features_dir = ensemble_path / "features"
    if not features_dir.exists():
        return []

    grouped: Dict[str, List[Tuple[Path, Dict[str, Any]]]] = {}
    for feature_file in sorted(features_dir.glob("*.json")):
        try:
            with open(feature_file, "r") as handle:
                feature_config = json.load(handle)
        except Exception:
            continue
        raw_feature_name = _extract_feature_name(feature_config, feature_file.stem)
        bias_spec = feature_config.get("bias_node_spec", {})
        canonical_name = _canonical_feature_name_from_tokens(raw_feature_name, bias_spec)
        grouped.setdefault(canonical_name, []).append((feature_file, feature_config))

    consolidated_paths: List[str] = []
    for canonical_name, payloads in grouped.items():
        first_file, first_config = payloads[0]
        first_spec = first_config.get("bias_node_spec", {})
        parsed = helpers.parse_feature_column_name(canonical_name)
        module_name = first_spec.get("module_name") or parsed.get("module")

        raw_tfs = first_spec.get("timeframes", [])
        if raw_tfs:
            first_tf = raw_tfs[0]
            timeframe_name = first_tf.name if hasattr(first_tf, "name") else str(first_tf)
        else:
            tf = parsed.get("tf")
            timeframe_name = tf.name if hasattr(tf, "name") else str(tf) if tf else "D"

        consolidated_bias_spec = {
            "module_name": module_name,
            "timeframes": [timeframe_name],
        }
        created_at = first_config.get("created_at", datetime.now(timezone.utc).isoformat())
        updated_at = datetime.now(timezone.utc).isoformat()

        all_tickers: set[str] = set()
        consolidated_models: List[Dict[str, Any]] = []
        used_model_ids: set[str] = set()

        for _src_file, feature_config in payloads:
            all_tickers.update(feature_config.get("tickers", []))
            source_feature_name = _extract_feature_name(feature_config, _src_file.stem)
            for model_entry in feature_config.get("base_models", []):
                model_type = _normalize_model_type(
                    model_entry.get(
                        "binning_model_type",
                        model_entry.get("model_type", "continuous_binning"),
                    )
                )
                strategy = model_entry.get("strategy", "long")
                model_params = dict(
                    model_entry.get(
                        "binning_model_params",
                        model_entry.get("constructor_params", {}),
                    )
                )
                model_params.pop("strategy", None)
                model_bias_params = _extract_bias_node_params_for_model(
                    model_entry, feature_config, source_feature_name
                )

                candidate_model_id = (
                    model_entry.get("model_id")
                    or generate_model_id(
                        model_type,
                        model_params,
                        bias_node_params=model_bias_params,
                    )
                )
                model_id = candidate_model_id
                suffix = 2
                while model_id in used_model_ids:
                    model_id = f"{candidate_model_id}_{suffix}"
                    suffix += 1
                used_model_ids.add(model_id)

                model_is_fitted = bool(model_entry.get("is_fitted", False))
                fitted_payload = model_entry.get("fitted_params")
                if not (fitted_payload and fitted_payload.get("model_version") == "binning_v2"):
                    model_is_fitted = False
                    fitted_payload = None

                consolidated_model = {
                    "model_id": model_id,
                    "model_name": f"{canonical_name}::{model_id}",
                    "bias_node_params": model_bias_params,
                    "binning_model_type": model_type,
                    "strategy": strategy,
                    "binning_model_params": model_params,
                    "requires_fit": _requires_fit(model_type),
                    "is_fitted": model_is_fitted,
                    "fitted_params": fitted_payload,
                }
                if model_is_fitted:
                    consolidated_model["fitted_at"] = model_entry.get(
                        "fitted_at",
                        datetime.now(timezone.utc).isoformat(),
                    )
                consolidated_models.append(consolidated_model)

        consolidated_config = {
            "feature_name": canonical_name,
            "created_at": created_at,
            "updated_at": updated_at,
            "bias_node_spec": consolidated_bias_spec,
            "tickers": sorted(all_tickers),
            "base_models": consolidated_models,
        }

        target_file = features_dir / f"{canonical_name}.json"
        with open(target_file, "w") as handle:
            json.dump(consolidated_config, handle, indent=2)
        consolidated_paths.append(str(target_file))

        # Remove legacy source files once consolidation target has been written.
        for source_file, _ in payloads:
            if source_file.resolve() == target_file.resolve():
                continue
            source_file.unlink(missing_ok=True)

    return consolidated_paths



def remove_base_model_variant(
    ensemble_dir: str,
    feature_name: str,
    model_id: str
) -> None:
    """
    Remove a base model variant from a feature.
    
    Useful if a model variant fails validation or is deprecated.
    
    Parameters
    ----------
    ensemble_dir : str
        Path to ensemble directory
    feature_name : str
        Feature name (canonical or legacy feature_column).
    model_id : str
        Model ID to remove
    """
    ensemble_path = _resolve_ensemble_path(ensemble_dir)
    features_dir = ensemble_path / "features"
    feature_file = _resolve_feature_file_path(features_dir, feature_name)
    if feature_file is None:
        raise ValueError(
            f"Feature file not found for feature '{feature_name}' in {features_dir}"
        )
    
    with open(feature_file, 'r') as f:
        feature_config = json.load(f)
    
    # Remove model entry
    original_count = len(feature_config['base_models'])
    feature_config['base_models'] = [
        bm for bm in feature_config['base_models'] if bm['model_id'] != model_id
    ]
    
    if len(feature_config['base_models']) == original_count:
        raise ValueError(f"Model ID '{model_id}' not found in feature '{feature_name}'")
    
    feature_config['updated_at'] = datetime.now(timezone.utc).isoformat()
    
    # Save updated config
    with open(feature_file, 'w') as f:
        json.dump(feature_config, f, indent=2)


def list_features(ensemble_dir: Optional[str] = None) -> pd.DataFrame:
    """
    List all features in an ensemble.
    
    Parameters
    ----------
    ensemble_dir : str, optional
        Path to ensemble directory. If None, uses default ensemble directory set via
        create_ensemble_directory() or auto-detects by finding any ensemble with features.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - feature_name: str
        - feature_column: str (legacy alias, equals feature_name for new files)
        - n_base_models: int
        - n_fitted: int
        - created_at: str
        - updated_at: str
    """
    global _DEFAULT_ENSEMBLE_DIR
    
    # Use default ensemble_dir if not provided
    if ensemble_dir is None:
        ensemble_dir = _DEFAULT_ENSEMBLE_DIR
        if ensemble_dir is None:
            # Auto-detect: find first ensemble directory with features
            cwd = Path.cwd()
            possible_paths = []
            
            # Check current directory and parent for vault
            for base in [cwd, cwd.parent]:
                if (base / 'vault').exists():
                    vault_path = base / 'vault'
                    for tf_dir in ['D', 'W', 'M']:
                        tf_path = vault_path / tf_dir
                        if tf_path.exists():
                            for ensemble_dir_path in tf_path.iterdir():
                                if ensemble_dir_path.is_dir():
                                    features_dir = ensemble_dir_path / 'features'
                                    if features_dir.exists() and list(features_dir.glob('*.json')):
                                        possible_paths.append(str(ensemble_dir_path))
            
            # Also check relative to vault_manager.py location
            vault_manager_dir = Path(__file__).parent.parent
            if (vault_manager_dir / 'vault').exists():
                vault_path = vault_manager_dir / 'vault'
                for tf_dir in ['D', 'W', 'M']:
                    tf_path = vault_path / tf_dir
                    if tf_path.exists():
                        for ensemble_dir_path in tf_path.iterdir():
                            if ensemble_dir_path.is_dir():
                                features_dir = ensemble_dir_path / 'features'
                                if features_dir.exists() and list(features_dir.glob('*.json')):
                                    possible_paths.append(str(ensemble_dir_path))
            
            if possible_paths:
                # Use first match and set as default
                ensemble_dir = possible_paths[0]
                _DEFAULT_ENSEMBLE_DIR = ensemble_dir
            else:
                raise ValueError(
                    "ensemble_dir must be provided or call create_ensemble_directory() first. "
                    "Could not auto-detect any ensemble directory with features. "
                    "Example: create_ensemble_directory(TimeFrame.D, 'buy_hold', Direction.LONG)"
                )
    
    ensemble_path = _resolve_ensemble_path(ensemble_dir)
    features_dir = ensemble_path / 'features'
    
    if not features_dir.exists():
        return pd.DataFrame(
            columns=[
                'feature_name',
                'feature_column',
                'n_base_models',
                'n_fitted',
                'created_at',
                'updated_at',
            ]
        )
    
    features = []
    
    for feature_file in sorted(features_dir.glob('*.json')):
        with open(feature_file, 'r') as f:
            feature_config = json.load(f)
        
        resolved_feature_name = _extract_feature_name(feature_config, feature_file.stem)
        n_base_models = len(feature_config.get('base_models', []))
        n_fitted = sum(1 for bm in feature_config.get('base_models', []) if bm.get('is_fitted', False))
        
        features.append({
            'feature_name': resolved_feature_name,
            'feature_column': feature_config.get('feature_column', resolved_feature_name),
            'n_base_models': n_base_models,
            'n_fitted': n_fitted,
            'created_at': feature_config.get('created_at', ''),
            'updated_at': feature_config.get('updated_at', '')
        })
    
    return pd.DataFrame(features)


def get_bias_node_specs(ensemble_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get all bias node specifications from an ensemble.
    
    This is useful for production deployment - read all bias node specs
    and use to reconstruct base models.
    
    Parameters
    ----------
    ensemble_dir : str, optional
        Path to ensemble directory. If None, uses default ensemble directory set via
        set_default_ensemble_dir(). Raises ValueError if neither is provided.
        
    Returns
    -------
    List[Dict[str, Any]]
        List of bias node specifications, one per base model. Each spec is
        top-level bias_node_spec merged with per-model bias_node_params.
    """
    # Use default ensemble_dir if not provided
    if ensemble_dir is None:
        ensemble_dir = _DEFAULT_ENSEMBLE_DIR
        if ensemble_dir is None:
            raise ValueError(
                "ensemble_dir must be provided or call create_ensemble_directory() first. "
                "Example: create_ensemble_directory(TimeFrame.D, 'buy_hold', Direction.LONG)"
            )
    
    ensemble_path = _resolve_ensemble_path(ensemble_dir)
    features_dir = ensemble_path / 'features'
    
    if not features_dir.exists():
        return []
    
    bias_specs = []
    
    for feature_file in sorted(features_dir.glob('*.json')):
        with open(feature_file, 'r') as f:
            feature_config = json.load(f)
        base_spec = dict(feature_config.get('bias_node_spec', {}))
        if not base_spec:
            continue
        raw_timeframes = base_spec.get('timeframes', [])
        base_spec['timeframes'] = [
            TimeFrame[tf] if isinstance(tf, str) else tf
            for tf in raw_timeframes
        ]
        feature_name = _extract_feature_name(feature_config, feature_file.stem)
        base_models = feature_config.get('base_models', [])
        if not base_models:
            spec_copy = dict(base_spec)
            spec_copy['params'] = _normalize_bias_node_params(base_spec.get('params', {}))
            bias_specs.append(spec_copy)
            continue
        for model_entry in base_models:
            model_spec = dict(base_spec)
            model_spec['params'] = _extract_bias_node_params_for_model(
                model_entry,
                feature_config,
                feature_name,
            )
            bias_specs.append(model_spec)
    
    return bias_specs


def get_all_base_model_names(ensemble_dir: Optional[str] = None) -> List[str]:
    """
    Get all base model names (feature::model_id) in an ensemble.
    
    These are the keys used in ensemble weights.
    
    Parameters
    ----------
    ensemble_dir : str, optional
        Path to ensemble directory. If None, uses default ensemble directory set via
        set_default_ensemble_dir(). Raises ValueError if neither is provided.
        
    Returns
    -------
    List[str]
        List of model names in format 'feature_column::model_id'
    """
    # Use default ensemble_dir if not provided
    if ensemble_dir is None:
        ensemble_dir = _DEFAULT_ENSEMBLE_DIR
        if ensemble_dir is None:
            raise ValueError(
                "ensemble_dir must be provided or call create_ensemble_directory() first. "
                "Example: create_ensemble_directory(TimeFrame.D, 'buy_hold', Direction.LONG)"
            )
    
    ensemble_path = _resolve_ensemble_path(ensemble_dir)
    features_dir = ensemble_path / 'features'
    
    if not features_dir.exists():
        return []
    
    model_names = []
    
    for feature_file in sorted(features_dir.glob('*.json')):
        with open(feature_file, 'r') as f:
            feature_config = json.load(f)
        feature_name = _extract_feature_name(feature_config, feature_file.stem)
        
        for model_config in feature_config.get('base_models', []):
            model_id = model_config['model_id']
            model_name = model_config.get('model_name', f"{feature_name}::{model_id}")
            model_names.append(model_name)
    
    return model_names


# ============================================================================
# Vault-Level Operations
# ============================================================================

def initialize_vault(vault_root: Optional[str] = None) -> None:
    """
    Initialize a new vault directory structure.
    
    Creates the root directory and README.md.
    The vault root is hardcoded to 'vault' at the project root.
    """
    vault_path = Path(vault_root or VAULT_ROOT)
    vault_path.mkdir(parents=True, exist_ok=True)
    
    # Create README.md
    readme_path = vault_path / 'README.md'
    if not readme_path.exists():
        readme_content = """# Vault - Base Model Feature Storage

This vault stores validated trading features and their associated base models.

## Directory Structure

```
vault/
├── D/                    # Daily timeframe ensembles
│   └── {ensemble}_{direction}/
│       └── features/
│           └── {feature_column}.json
├── W/                    # Weekly timeframe ensembles
└── M/                    # Monthly timeframe ensembles
```

## Feature Control Files

Each feature has its own control file (`features/{feature_column}.json`) containing:
- Bias node specification (for feature reconstruction)
- All base model variants for that feature
- Fitted and unfitted model configurations

## Usage

See `docs/to-do/vault_specs.md` for complete documentation.
"""
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)


def validate_ensemble_directory(ensemble_dir: str) -> None:
    """
    Validate an ensemble directory structure and all feature control files.
    
    Checks:
    - Directory exists and has features/ subdirectory
    - All feature control files are valid JSON
    - All base model strategies match ensemble direction
    - All bias node specs are valid and parseable
    - All model IDs are unique within each feature
    
    Parameters
    ----------
    ensemble_dir : str
        Path to ensemble directory
        
    Raises
    ------
    ValueError
        If validation fails
    """
    ensemble_path = _resolve_ensemble_path(ensemble_dir)
    
    if not ensemble_path.exists():
        raise ValueError(f"Ensemble directory does not exist: {ensemble_dir} (resolved to: {ensemble_path})")
    
    if not ensemble_path.is_dir():
        raise ValueError(f"Path is not a directory: {ensemble_dir} (resolved to: {ensemble_path})")
    
    features_dir = ensemble_path / 'features'
    if not features_dir.exists():
        raise ValueError(f"Features directory does not exist: {features_dir}")
    
    # Load ensemble config if it exists
    ensemble_config_file = ensemble_path / 'ensemble_config.json'
    if ensemble_config_file.exists():
        with open(ensemble_config_file, 'r') as f:
            ensemble_config = json.load(f)
        expected_direction = coerce_direction(
            ensemble_config.get('direction', Direction.LONG.value),
            field_name="ensemble_config.direction",
        )
        expected_ticker_names = sorted(ensemble_config.get('tickers', []))
    else:
        # Backward compatibility: infer from directory name
        ensemble_dir_name = ensemble_path.name
        if ensemble_dir_name.endswith('_long_short'):
            expected_direction = Direction.LONG_SHORT
        elif ensemble_dir_name.endswith('_long'):
            expected_direction = Direction.LONG
        elif ensemble_dir_name.endswith('_short'):
            expected_direction = Direction.SHORT
        else:
            raise ValueError(
                f"Cannot determine ensemble direction from directory name: {ensemble_dir_name}. "
                f"Expected suffix _long, _short, or _long_short"
            )
        expected_ticker_names = None  # No ticker validation for old ensembles

    # Validate all feature control files
    for feature_file in sorted(features_dir.glob('*.json')):
        try:
            with open(feature_file, 'r') as f:
                feature_config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in feature file {feature_file}: {e}")
        
        # Validate structure (feature_name-first, feature_column legacy fallback)
        required_keys = ['bias_node_spec', 'base_models']
        missing_keys = [key for key in required_keys if key not in feature_config]
        if missing_keys:
            raise ValueError(f"Feature file {feature_file} missing required keys: {missing_keys}")
        if (
            'feature_name' not in feature_config
            and 'feature_column' not in feature_config
        ):
            raise ValueError(
                f"Feature file {feature_file} missing both 'feature_name' and legacy 'feature_column'"
            )
        
        # Validate tickers field (optional for backward compatibility, but recommended)
        if 'tickers' not in feature_config:
            warnings.warn(
                f"Feature file {feature_file} missing 'tickers' field. "
                f"This is required for proper multi-ticker support. "
                f"Defaulting to [Ticker.ES] when loading.",
                DeprecationWarning
            )
        elif not isinstance(feature_config['tickers'], list):
            raise ValueError(
                f"Feature file {feature_file} has invalid 'tickers' field: "
                f"expected list, got {type(feature_config['tickers'])}"
            )
        
        # Validate bias node spec
        bias_node_spec = feature_config['bias_node_spec']
        if 'module_name' not in bias_node_spec or 'timeframes' not in bias_node_spec:
            raise ValueError(f"Invalid bias_node_spec in {feature_file}")
        if not isinstance(bias_node_spec.get('timeframes'), list):
            raise ValueError(f"Invalid bias_node_spec.timeframes in {feature_file}: expected list")

        feature_name = _extract_feature_name(feature_config, feature_file.stem)
        feature_models = feature_config['base_models']
        if len(feature_models) != 1:
            raise ValueError(
                f"Feature '{feature_name}' must contain exactly one base model, found {len(feature_models)}"
            )

        # Validate base model
        model_ids = []
        for model_config in feature_models:
            # Check required keys
            required_model_keys = ['model_id', 'model_name', 'binning_model_type', 'strategy', 'binning_model_params']
            missing_model_keys = [key for key in required_model_keys if key not in model_config]
            if missing_model_keys:
                raise ValueError(f"Model config in {feature_file} missing keys: {missing_model_keys}")
            
            # Check model ID uniqueness
            model_id = model_config['model_id']
            if model_id in model_ids:
                raise ValueError(f"Duplicate model_id '{model_id}' in feature {feature_file}")
            model_ids.append(model_id)
            
            # Check strategy matches ensemble direction
            model_strategy = coerce_direction(
                model_config['strategy'],
                field_name=f"{feature_file.name} strategy",
            )
            if model_strategy != expected_direction:
                raise ValueError(
                    f"Model strategy '{model_strategy.value}' does not match "
                    f"ensemble direction '{expected_direction.value}' in {feature_file}"
                )
            
            # Validate model_name format
            expected_model_name = f"{feature_name}::{model_id}"
            if model_config['model_name'] != expected_model_name:
                raise ValueError(
                    f"Model name '{model_config['model_name']}' does not match expected format "
                    f"'{expected_model_name}' in {feature_file}"
                )

            # Validate requires_fit if present
            if 'requires_fit' in model_config:
                expected_requires_fit = _requires_fit(
                    model_config.get('binning_model_type', model_config.get('model_type', 'continuous_binning'))
                )
                if bool(model_config['requires_fit']) != expected_requires_fit:
                    raise ValueError(
                        f"Model '{model_id}' has inconsistent requires_fit="
                        f"{model_config['requires_fit']} (expected {expected_requires_fit})"
                    )

            if "members" in model_config:
                raise ValueError(
                    f"Model '{model_id}' in {feature_file} uses legacy members schema, which is unsupported"
                )
        
        # Validate tickers match ensemble tickers (if ensemble config exists)
        if expected_ticker_names is not None:
            feature_ticker_names = sorted(feature_config.get('tickers', []))
            if feature_ticker_names != expected_ticker_names:
                raise ValueError(
                    f"Feature '{feature_file.stem}' has tickers {feature_ticker_names} "
                    f"but ensemble expects {expected_ticker_names}. "
                    f"All features in an ensemble must use the same tickers."
                )


def load_ensemble_from_vault(
    ensemble_dir: str,
    refit: bool = False,
    target_volatility: float = 0.20
) -> 'DiversifiedEnsemble':  # type: ignore
    """
    Load a DiversifiedEnsemble from vault directory.
    
    Creates a temporary control file from vault feature configs and loads the ensemble.
    
    Parameters
    ----------
    ensemble_dir : str
        Path to ensemble directory (e.g., 'vault/D/buy_hold_long')
    refit : bool, default=False
        If True, ensemble will be refitted from scratch (is_fit=False).
        If False, uses fitted params from vault (is_fit=True).
    target_volatility : float, default=0.20
        Target volatility for the ensemble
        
    Returns
    -------
    DiversifiedEnsemble
        Loaded ensemble instance
        
    Raises
    ------
    ValueError
        If ensemble directory doesn't exist or has no features
    """
    from ensemble.diversified_ensemble import DiversifiedEnsemble
    from ensemble.ensemble_utils import save_control_file
    import tempfile
    
    ensemble_path = _resolve_ensemble_path(ensemble_dir)
    
    if not ensemble_path.exists():
        raise ValueError(f"Ensemble directory does not exist: {ensemble_dir} (resolved to: {ensemble_path})")
    
    features_dir = ensemble_path / 'features'
    if not features_dir.exists():
        raise ValueError(f"Features directory does not exist: {features_dir}")
    
    feature_files = list(features_dir.glob('*.json'))
    if not feature_files:
        raise ValueError(f"No feature files found in {features_dir}")
    
    # Load ensemble config if available
    ensemble_config_file = ensemble_path / 'ensemble_config.json'
    ensemble_config = {}
    if ensemble_config_file.exists():
        with open(ensemble_config_file, 'r') as f:
            ensemble_config = json.load(f)
        timeframe_str = ensemble_config.get('timeframe', 'D')
    else:
        # Infer from directory structure
        timeframe_str = ensemble_path.parent.name  # e.g., 'D' from 'vault/D/...'
    
    # Load all feature configs and convert to control file format
    base_models_config = []
    all_tickers = set()
    
    for feature_file in feature_files:
        with open(feature_file, 'r') as f:
            feature_config = json.load(f)
        
        # Collect tickers
        feature_tickers = feature_config.get('tickers', ensemble_config.get('tickers', []))
        all_tickers.update(feature_tickers)
        
        # Convert base_models to control file format
        # Include tickers in each model config so models know which tickers they support
        feature_tickers = feature_config.get('tickers', ensemble_config.get('tickers', []))
        feature_name = _extract_feature_name(feature_config, feature_file.stem)
        feature_bias_spec = dict(feature_config.get('bias_node_spec', {}))
        raw_timeframes = feature_bias_spec.get('timeframes', [])
        feature_bias_spec['timeframes'] = [
            tf.name if isinstance(tf, TimeFrame) else str(tf)
            for tf in raw_timeframes
        ]
        feature_models = feature_config.get("base_models", [])
        if len(feature_models) != 1:
            raise ValueError(
                f"Feature '{feature_name}' must contain exactly one base model; found {len(feature_models)}"
            )
        model = feature_models[0]
        if "members" in model:
            raise ValueError(
                f"Legacy member schema is not supported in {feature_file}"
            )
        bias_node_params = _extract_bias_node_params_for_model(
            model,
            feature_config,
            feature_name,
        )
        merged_bias_spec = dict(feature_bias_spec)
        merged_bias_spec['params'] = bias_node_params

        model_type = _normalize_model_type(
            model.get('binning_model_type', model.get('model_type', 'continuous_binning'))
        )
        constructor_params = dict(
            model.get('binning_model_params', model.get('constructor_params', {}))
        )
        constructor_params.pop('strategy', None)

        base_model_config = {
            'name': model.get('model_name', f"{feature_name}::{model['model_id']}"),
            'feature_column': feature_name,
            'model_type': model_type,
            'strategy': model['strategy'],
            'constructor_params': constructor_params,
            'bias_node_spec': merged_bias_spec,
            'bias_node_params': bias_node_params,
            'tickers': feature_tickers,  # Include tickers so model knows which tickers it supports
        }
        base_models_config.append(base_model_config)
    
    if not base_models_config:
        raise ValueError(f"No base models found in ensemble directory: {ensemble_dir}")
    
    # Get tickers (prefer ensemble config, then feature configs)
    tickers = sorted(ensemble_config.get('tickers', list(all_tickers)))
    if not tickers:
        raise ValueError(f"Could not determine tickers for ensemble: {ensemble_dir}")
    
    # Create temporary control file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        # Get created_at from first feature or use current time
        first_feature_file = feature_files[0]
        with open(first_feature_file, 'r') as feat_f:
            first_feature_config = json.load(feat_f)
        
        metadata = {
            'created_at': first_feature_config.get('created_at', datetime.now(timezone.utc).isoformat()),
            'updated_at': datetime.now(timezone.utc).isoformat(),
            'is_fit': not refit,
            'base_tf': timeframe_str,
        }
        if not refit:
            metadata['selection_method'] = ensemble_config.get('selection_method', 'manual')

        fitted_base_models = None
        fitted_ensemble = None
        if not refit:
            fitted_base_models = {}
            for feature_file in feature_files:
                with open(feature_file, 'r') as feat_f:
                    feature_config = json.load(feat_f)
                feature_name = _extract_feature_name(feature_config, feature_file.stem)
                for model in feature_config.get('base_models', []):
                    model_name = model.get('model_name', f"{feature_name}::{model['model_id']}")
                    payload = model.get('fitted_params')
                    if model.get('is_fitted', False) and payload and payload.get('model_version') == 'binning_v2':
                        fitted_base_models[model_name] = payload

            model_names = [bm['name'] for bm in base_models_config]
            if model_names:
                fitted_ensemble = {
                    'weights': {name: 1.0 / len(model_names) for name in model_names},
                    'exposure_fractions': {name: 0.5 for name in model_names},
                    'model_exposure_fractions': {name: 0.5 for name in model_names},
                    'feature_names': model_names,
                    'target_volatility': target_volatility,
                    'unique_tickers': tickers,
                    'instrument_weights': {t: 1.0 / len(tickers) for t in tickers},
                    'n_tickers': len(tickers),
                }
            else:
                fitted_ensemble = {
                    'weights': {},
                    'exposure_fractions': {},
                    'model_exposure_fractions': {},
                    'feature_names': [],
                    'target_volatility': target_volatility,
                    'unique_tickers': tickers,
                    'instrument_weights': {t: 1.0 / len(tickers) for t in tickers},
                    'n_tickers': len(tickers),
                }

        temp_path = save_control_file(
            filepath=f.name,
            base_models=base_models_config,
            metadata=metadata,
            fitted_base_models=fitted_base_models,
            fitted_ensemble=fitted_ensemble,
            tickers=tickers,
        )
    
    # Create ensemble from control file
    ensemble = DiversifiedEnsemble(
        control_file_path=temp_path,
        target_volatility=target_volatility,
        base_tf=TimeFrame[timeframe_str]
    )
    
    # Clean up temp file
    os.unlink(temp_path)
    
    return ensemble
