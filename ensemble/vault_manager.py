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
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

from utils.enums import TimeFrame, Direction, Ticker
from feature_selection.base_models.feature_base_model import BaseModel
from feature_selection.base_models import QuantileBinningModel, DecisionTreeBinningModel
import utils.helpers as helpers

# Type hint for forward reference
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ensemble.diversified_ensemble import DiversifiedEnsemble


# ============================================================================
# Module-Level Configuration
# ============================================================================

# Hardcoded vault root - always at project root
VAULT_ROOT: str = 'vault'

# Default ensemble directory (set automatically when create_ensemble_directory is called)
_DEFAULT_ENSEMBLE_DIR: Optional[str] = None

# Default ensemble tickers (set automatically when create_ensemble_directory is called)
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
# Model ID Generation
# ============================================================================

def generate_model_id(binning_model_type: str, binning_model_params: Dict[str, Any]) -> str:
    """
    Auto-generate model ID from binning model type and hyperparameters.
    
    Pattern: {binning_model_type_snake_case}_{key_hyperparam_values}
    
    Parameters
    ----------
    binning_model_type : str
        Binning model class name (e.g., 'QuantileBinningModel')
    binning_model_params : Dict[str, Any]
        Constructor parameters for the binning model
        
    Returns
    -------
    str
        Auto-generated model ID
        
    Examples
    --------
    >>> generate_model_id('QuantileBinningModel', {'n_bins': 3, 'selection_metric': 'sortino'})
    'quantile_binning_3'
    >>> generate_model_id('DecisionTreeBinningModel', {'n_bins': 5, 'min_samples_leaf_pct': 0.10})
    'decision_tree_binning_5'
    """
    # Convert CamelCase to snake_case and remove 'Model' suffix
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', binning_model_type)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
    name = name.replace('_model', '')
    
    # Append key hyperparameters (n_bins is always included)
    if 'n_bins' in binning_model_params:
        name += f"_{binning_model_params['n_bins']}"
    
    # Only include other hyperparameters that significantly change behavior
    # For DecisionTreeBinningModel, min_samples_leaf_pct is usually constant, so we skip it
    # Add more if needed for other model types
    
    return name


# ============================================================================
# Ensemble Directory Management
# ============================================================================

def create_ensemble_directory(
    timeframe: TimeFrame,
    ensemble_name: str,
    direction: Direction,
    tickers: Optional[List[Ticker]] = None
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
    
    # Build full path (hardcoded vault root)
    # Resolve path to handle running from different directories
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
    timeframe: TimeFrame,
    ensemble_name: str,
    direction: Direction
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
    ensemble_dir_name = f"{ensemble_name}_{direction.value}"
    ensemble_path = Path(VAULT_ROOT) / timeframe.name / ensemble_dir_name
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
            if not dir_name.endswith('_long') and not dir_name.endswith('_short'):
                continue
            
            if dir_name.endswith('_long'):
                ensemble_name = dir_name[:-5]  # Remove '_long'
                direction = 'long'
            else:
                ensemble_name = dir_name[:-6]  # Remove '_short'
                direction = 'short'
            
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
    feature_column: str,
    bias_node_spec: Dict[str, Any],
    base_model: BaseModel,
    ensemble_dir: Optional[str] = None,
    tickers: Optional[List[Ticker]] = None
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
        expected_direction = ensemble_config.get('direction', 'long')
    else:
        # Backward compatibility: infer from directory name
        ensemble_dir_name = ensemble_path.name
        if ensemble_dir_name.endswith('_long'):
            expected_direction = 'long'
        elif ensemble_dir_name.endswith('_short'):
            expected_direction = 'short'
        else:
            raise ValueError(f"Cannot determine ensemble direction from directory name: {ensemble_dir_name}")
        expected_ticker_names = None  # No ticker validation for old ensembles
    
    # Validate ensemble direction matches base model strategy
    if base_model.binning_model.strategy != expected_direction:
        raise ValueError(
            f"Base model strategy '{base_model.binning_model.strategy}' does not match "
            f"ensemble direction '{expected_direction}'"
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
    binning_model_type = binning_model.__class__.__name__
    binning_model_params = binning_model.get_params()
    
    # Generate model ID
    model_id = generate_model_id(binning_model_type, binning_model_params)
    
    # Feature control file path
    feature_file = features_dir / f"{feature_column}.json"
    
    # Convert TimeFrame enums to strings for JSON serialization
    serializable_bias_spec = {
        'module_name': bias_node_spec['module_name'],
        'timeframes': [tf.name if isinstance(tf, TimeFrame) else tf for tf in bias_node_spec['timeframes']],
        'params': bias_node_spec['params']
    }
    
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
        
        # Validate existing tickers match ensemble tickers (if ensemble config exists)
        existing_ticker_names = sorted(feature_config.get('tickers', []))
        if expected_ticker_names is not None and existing_ticker_names != expected_ticker_names:
            raise ValueError(
                f"Feature '{feature_column}' already exists with tickers {existing_ticker_names} "
                f"but ensemble expects {expected_ticker_names}. "
                f"All features in an ensemble must use the same tickers."
            )
        
        # Use ensemble tickers (already validated above)
        feature_config['tickers'] = ticker_names
    else:
        # Create new feature config
        feature_config = {
            'feature_name': feature_column,
            'feature_column': feature_column,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'updated_at': datetime.now(timezone.utc).isoformat(),
            'bias_node_spec': serializable_bias_spec,
            'tickers': ticker_names,
            'base_models': []
        }
    
    # Check if model_id already exists
    existing_model_ids = [bm['model_id'] for bm in feature_config['base_models']]
    if model_id in existing_model_ids:
        raise ValueError(
            f"Model ID '{model_id}' already exists for feature '{feature_column}'. "
            f"Existing model IDs: {existing_model_ids}"
        )
    
    # Create base model entry
    model_entry = {
        'model_id': model_id,
        'model_name': f"{feature_column}::{model_id}",
        'binning_model_type': binning_model_type,
        'strategy': base_model.binning_model.strategy,
        'binning_model_params': binning_model_params,
        'is_fitted': binning_model.is_fitted_,
        'fitted_params': None
    }
    
    # Add fitted params if model is fitted
    if binning_model.is_fitted_:
        model_entry['fitted_at'] = datetime.now(timezone.utc).isoformat()
        model_entry['fitted_params'] = {
            'thresholds': binning_model.thresholds_.tolist() if binning_model.thresholds_ is not None else None,
            'best_long_bin': binning_model.best_long_bin_,
            'best_short_bin': binning_model.best_short_bin_,
            'bin_stats': binning_model.bin_stats_
        }
    
    # Add model entry
    feature_config['base_models'].append(model_entry)
    feature_config['updated_at'] = datetime.now(timezone.utc).isoformat()
    
    # Save feature config
    with open(feature_file, 'w') as f:
        json.dump(feature_config, f, indent=2)
    
    return model_id


def load_feature_base_models(
    feature_column: str,
    ensemble_dir: Optional[str] = None,
    fitted_only: bool = False,
    tickers: Optional[List[Ticker]] = None
) -> Dict[Tuple[Ticker, str], BaseModel]:
    """
    Load all base model variants for a feature, creating separate instances for each ticker.
    
    Parameters
    ----------
    feature_column : str
        Feature column name
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
    >>> es_model = models[(Ticker.ES, 'quantile_binning_3')]
    >>> 
    >>> # Using default ensemble_dir (set by create_ensemble_directory)
    >>> create_ensemble_directory(TimeFrame.D, 'buy_hold', Direction.LONG)
    >>> models = load_feature_base_models('buy_hold_signal_D')
    >>> 
    >>> # Auto-detection (searches vault for feature file)
    >>> models = load_feature_base_models('buy_hold_signal_D')  # Finds vault/D/buy_hold_long automatically
    """
    global _DEFAULT_ENSEMBLE_DIR
    
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
                                    feature_file = ensemble_dir_path / 'features' / f"{feature_column}.json"
                                    if feature_file.exists():
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
                                feature_file = ensemble_dir_path / 'features' / f"{feature_column}.json"
                                if feature_file.exists():
                                    possible_paths.append(str(ensemble_dir_path))
            
            if possible_paths:
                # Use first match and set as default for future calls
                ensemble_dir = possible_paths[0]
                _DEFAULT_ENSEMBLE_DIR = ensemble_dir
            else:
                raise ValueError(
                    f"ensemble_dir must be provided or call create_ensemble_directory() first. "
                    f"Could not auto-detect ensemble directory for feature '{feature_column}'. "
                    f"Example: create_ensemble_directory(TimeFrame.D, 'buy_hold', Direction.LONG)"
                )
    
    ensemble_path = _resolve_ensemble_path(ensemble_dir)
    feature_file = ensemble_path / 'features' / f"{feature_column}.json"
    
    if not feature_file.exists():
        # Provide helpful error message with debugging info
        import warnings
        cwd = Path.cwd()
        tried_paths = [
            Path(ensemble_dir),
            cwd / ensemble_dir,
            cwd.parent / ensemble_dir,
            Path(__file__).parent.parent / ensemble_dir
        ]
        warnings.warn(
            f"Feature file not found: {feature_file}\n"
            f"  Feature column: {feature_column}\n"
            f"  Ensemble dir provided: {ensemble_dir}\n"
            f"  Resolved ensemble path: {ensemble_path}\n"
            f"  Current working directory: {cwd}\n"
            f"  Tried paths: {[str(p) for p in tried_paths]}\n"
            f"  File should be at: {ensemble_path / 'features' / f'{feature_column}.json'}",
            UserWarning
        )
        return {}
    
    with open(feature_file, 'r') as f:
        feature_config = json.load(f)
    
    # Convert TimeFrame strings back to TimeFrame enums in bias_node_spec
    bias_node_spec = feature_config['bias_node_spec'].copy()
    if 'timeframes' in bias_node_spec:
        bias_node_spec['timeframes'] = [
            TimeFrame[tf] if isinstance(tf, str) else tf 
            for tf in bias_node_spec['timeframes']
        ]
    
    # Update feature_config with converted bias_node_spec
    feature_config_copy = feature_config.copy()
    feature_config_copy['bias_node_spec'] = bias_node_spec
    
    # Get tickers from feature spec or use provided/default
    if tickers is None:
        ticker_names = feature_config.get('tickers', [])
        if ticker_names:
            # Convert ticker name strings to Ticker enums
            tickers = [Ticker[ticker_name] if isinstance(ticker_name, str) else ticker_name 
                      for ticker_name in ticker_names]
        else:
            # Backward compatibility: default to ES if no tickers stored
            tickers = [Ticker.ES]
    
    models = {}
    
    for model_config in feature_config['base_models']:
        model_id = model_config['model_id']
        
        # Skip unfitted if fitted_only=True
        if fitted_only and not model_config.get('is_fitted', False):
            continue
        
        # Reconstruct binning model
        binning_model_type = model_config['binning_model_type']
        binning_model_params = model_config['binning_model_params'].copy()
        
        # Remove 'strategy' and 'normalize_by' from params (handled separately or not used in constructor)
        strategy = binning_model_params.pop('strategy', 'long')
        binning_model_params.pop('normalize_by', None)  # Not a constructor param
        
        if binning_model_type == 'QuantileBinningModel':
            binning_model = QuantileBinningModel(**binning_model_params, strategy=strategy)
        elif binning_model_type == 'DecisionTreeBinningModel':
            binning_model = DecisionTreeBinningModel(**binning_model_params, strategy=strategy)
        else:
            raise ValueError(f"Unknown binning model type: {binning_model_type}")
        
        # Load fitted params if available
        if model_config.get('is_fitted', False) and model_config.get('fitted_params'):
            fitted_params = model_config['fitted_params']
            # Handle thresholds: empty list/None for constant features, otherwise array
            thresholds_data = fitted_params.get('thresholds')
            if thresholds_data is None:
                binning_model.thresholds_ = np.array([])  # Constant feature
            elif len(thresholds_data) == 0:
                binning_model.thresholds_ = np.array([])  # Constant feature (empty list)
            else:
                binning_model.thresholds_ = np.array(thresholds_data)
            binning_model.best_long_bin_ = fitted_params.get('best_long_bin')
            binning_model.best_short_bin_ = fitted_params.get('best_short_bin')
            binning_model.bin_stats_ = fitted_params.get('bin_stats')
            binning_model.is_fitted_ = True
        
        # Create BaseModel instance for each ticker (bias nodes are ticker-specific)
        for ticker in tickers:
            base_model = BaseModel(
                feature_config=feature_config_copy,
                ticker=ticker,
                binning_model=binning_model
            )
            base_model.feature_column = feature_column
            
            # Use (ticker, model_id) as key
            models[(ticker, model_id)] = base_model
    
    return models


def update_base_model_fitted_params(
    ensemble_dir: str,
    feature_column: str,
    model_id: str,
    fitted_params: Dict[str, Any],
    train_start: str,
    train_end: str
) -> None:
    """
    Update fitted parameters for a specific base model variant.
    
    Called after BaseModel.fit() to save the fitted state.
    
    Parameters
    ----------
    ensemble_dir : str
        Path to ensemble directory
    feature_column : str
        Feature column name
    model_id : str
        Model ID to update
    fitted_params : Dict[str, Any]
        Fitted parameters from the base model
    train_start : str
        Training start date (YYYY-MM-DD)
    train_end : str
        Training end date (YYYY-MM-DD)
    """
    ensemble_path = _resolve_ensemble_path(ensemble_dir)
    feature_file = ensemble_path / 'features' / f"{feature_column}.json"
    
    if not feature_file.exists():
        raise ValueError(f"Feature file not found: {feature_file}")
    
    with open(feature_file, 'r') as f:
        feature_config = json.load(f)
    
    # Find model entry
    model_found = False
    for model_entry in feature_config['base_models']:
        if model_entry['model_id'] == model_id:
            model_entry['is_fitted'] = True
            model_entry['fitted_at'] = datetime.now(timezone.utc).isoformat()
            model_entry['train_start'] = train_start
            model_entry['train_end'] = train_end
            model_entry['fitted_params'] = fitted_params
            model_found = True
            break
    
    if not model_found:
        raise ValueError(f"Model ID '{model_id}' not found in feature '{feature_column}'")
    
    feature_config['updated_at'] = datetime.now(timezone.utc).isoformat()
    
    # Save updated config
    with open(feature_file, 'w') as f:
        json.dump(feature_config, f, indent=2)


def remove_base_model_variant(
    ensemble_dir: str,
    feature_column: str,
    model_id: str
) -> None:
    """
    Remove a base model variant from a feature.
    
    Useful if a model variant fails validation or is deprecated.
    
    Parameters
    ----------
    ensemble_dir : str
        Path to ensemble directory
    feature_column : str
        Feature column name
    model_id : str
        Model ID to remove
    """
    ensemble_path = _resolve_ensemble_path(ensemble_dir)
    feature_file = ensemble_path / 'features' / f"{feature_column}.json"
    
    if not feature_file.exists():
        raise ValueError(f"Feature file not found: {feature_file}")
    
    with open(feature_file, 'r') as f:
        feature_config = json.load(f)
    
    # Remove model entry
    original_count = len(feature_config['base_models'])
    feature_config['base_models'] = [
        bm for bm in feature_config['base_models'] if bm['model_id'] != model_id
    ]
    
    if len(feature_config['base_models']) == original_count:
        raise ValueError(f"Model ID '{model_id}' not found in feature '{feature_column}'")
    
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
        - feature_column: str
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
        return pd.DataFrame(columns=['feature_name', 'feature_column', 'n_base_models', 'n_fitted', 'created_at', 'updated_at'])
    
    features = []
    
    for feature_file in features_dir.glob('*.json'):
        with open(feature_file, 'r') as f:
            feature_config = json.load(f)
        
        n_base_models = len(feature_config.get('base_models', []))
        n_fitted = sum(1 for bm in feature_config.get('base_models', []) if bm.get('is_fitted', False))
        
        features.append({
            'feature_name': feature_config.get('feature_name', feature_file.stem),
            'feature_column': feature_config.get('feature_column', feature_file.stem),
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
        List of bias node specifications, one per feature
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
    
    for feature_file in features_dir.glob('*.json'):
        with open(feature_file, 'r') as f:
            feature_config = json.load(f)
        
        bias_spec = feature_config.get('bias_node_spec')
        if bias_spec:
            bias_specs.append(bias_spec)
    
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
    
    for feature_file in features_dir.glob('*.json'):
        with open(feature_file, 'r') as f:
            feature_config = json.load(f)
        
        feature_column = feature_config.get('feature_column', feature_file.stem)
        
        for model_config in feature_config.get('base_models', []):
            model_id = model_config['model_id']
            model_name = f"{feature_column}::{model_id}"
            model_names.append(model_name)
    
    return model_names


# ============================================================================
# Vault-Level Operations
# ============================================================================

def initialize_vault() -> None:
    """
    Initialize a new vault directory structure.
    
    Creates the root directory and README.md.
    The vault root is hardcoded to 'vault' at the project root.
    """
    vault_path = Path(VAULT_ROOT)
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
        with open(readme_path, 'w') as f:
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
        expected_direction = ensemble_config.get('direction', 'long')
        expected_ticker_names = sorted(ensemble_config.get('tickers', []))
    else:
        # Backward compatibility: infer from directory name
        ensemble_dir_name = ensemble_path.name
        if ensemble_dir_name.endswith('_long'):
            expected_direction = 'long'
        elif ensemble_dir_name.endswith('_short'):
            expected_direction = 'short'
        else:
            raise ValueError(f"Cannot determine ensemble direction from directory name: {ensemble_dir_name}")
        expected_ticker_names = None  # No ticker validation for old ensembles
    
    # Validate all feature control files
    for feature_file in features_dir.glob('*.json'):
        try:
            with open(feature_file, 'r') as f:
                feature_config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in feature file {feature_file}: {e}")
        
        # Validate structure
        required_keys = ['feature_name', 'feature_column', 'bias_node_spec', 'base_models']
        missing_keys = [key for key in required_keys if key not in feature_config]
        if missing_keys:
            raise ValueError(f"Feature file {feature_file} missing required keys: {missing_keys}")
        
        # Validate tickers field (optional for backward compatibility, but recommended)
        if 'tickers' not in feature_config:
            import warnings
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
        if 'module_name' not in bias_node_spec or 'timeframes' not in bias_node_spec or 'params' not in bias_node_spec:
            raise ValueError(f"Invalid bias_node_spec in {feature_file}")
        
        # Validate base models
        model_ids = []
        for model_config in feature_config['base_models']:
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
            if model_config['strategy'] != expected_direction:
                raise ValueError(
                    f"Model strategy '{model_config['strategy']}' does not match "
                    f"ensemble direction '{expected_direction}' in {feature_file}"
                )
            
            # Validate model_name format
            expected_model_name = f"{feature_config['feature_column']}::{model_id}"
            if model_config['model_name'] != expected_model_name:
                raise ValueError(
                    f"Model name '{model_config['model_name']}' does not match expected format "
                    f"'{expected_model_name}' in {feature_file}"
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
        for model in feature_config.get('base_models', []):
            base_model_config = {
                'name': model['model_name'],
                'feature_column': feature_config['feature_column'],
                'model_type': model['binning_model_type'],
                'strategy': model['strategy'],
                'constructor_params': model['binning_model_params'],
                'bias_node_spec': feature_config['bias_node_spec']
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
        
        control_file = {
            'metadata': {
                'created_at': first_feature_config.get('created_at', datetime.now(timezone.utc).isoformat()),
                'updated_at': datetime.now(timezone.utc).isoformat(),
                'is_fit': not refit,
                'base_tf': timeframe_str
            },
            'base_models': base_models_config,
            'tickers': tickers
        }
        
        # Add fitted params if not refitting
        if not refit:
            fitted_base_models = {}
            for feature_file in feature_files:
                with open(feature_file, 'r') as feat_f:
                    feature_config = json.load(feat_f)
                for model in feature_config.get('base_models', []):
                    if model.get('is_fitted', False):
                        fitted_base_models[model['model_name']] = model.get('fitted_params', {})
            
            # Create fitted_ensemble with default weights (equal weight)
            if base_models_config:
                model_names = [bm['name'] for bm in base_models_config]
                control_file['fitted_base_models'] = fitted_base_models
                control_file['fitted_ensemble'] = {
                    'weights': {name: 1.0 / len(model_names) for name in model_names},
                    'exposure_fractions': {name: 0.5 for name in model_names},
                    'model_exposure_fractions': {name: 0.5 for name in model_names},
                    'feature_names': model_names,
                    'target_volatility': target_volatility,
                    'unique_tickers': tickers,
                    'instrument_weights': {t: 1.0 / len(tickers) for t in tickers},
                    'n_tickers': len(tickers)
                }
        
        json.dump(control_file, f, indent=2)
        temp_path = f.name
    
    # Create ensemble from control file
    ensemble = DiversifiedEnsemble(
        control_file_path=temp_path,
        target_volatility=target_volatility,
        base_tf=TimeFrame[timeframe_str]
    )
    
    # Clean up temp file
    os.unlink(temp_path)
    
    return ensemble
