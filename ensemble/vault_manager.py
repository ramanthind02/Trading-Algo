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
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from utils.enums import TimeFrame, Direction, Ticker
from feature_selection.base_models.feature_base_model import BaseModel
from feature_selection.base_models import QuantileBinningModel, DecisionTreeBinningModel
import utils.helpers as helpers


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
    vault_root: str,
    timeframe: TimeFrame,
    ensemble_name: str,
    direction: Direction
) -> str:
    """
    Create a new ensemble directory in the vault.
    
    Parameters
    ----------
    vault_root : str
        Root directory of the vault
    timeframe : TimeFrame
        Trading timeframe enum (D, W, M)
    ensemble_name : str
        Descriptive name for the ensemble (e.g., 'commodity_breakout', 'universal_momentum')
    direction : Direction
        Trading direction enum (LONG or SHORT)
        
    Returns
    -------
    str
        Path to the created ensemble directory
        
    Raises
    ------
    ValueError
        If ensemble directory already exists
        
    Examples
    --------
    >>> ensemble_dir = create_ensemble_directory(
    ...     vault_root='vault',
    ...     timeframe=TimeFrame.D,
    ...     ensemble_name='commodity_breakout',
    ...     direction=Direction.LONG
    ... )
    >>> print(ensemble_dir)  # 'vault/D/commodity_breakout_long'
    """
    # Build ensemble directory name
    ensemble_dir_name = f"{ensemble_name}_{direction.value}"
    
    # Build full path
    ensemble_path = Path(vault_root) / timeframe.name / ensemble_dir_name
    
    # Check if already exists
    if ensemble_path.exists():
        raise ValueError(f"Ensemble directory already exists: {ensemble_path}")
    
    # Create directory structure
    features_dir = ensemble_path / 'features'
    features_dir.mkdir(parents=True, exist_ok=False)
    
    return str(ensemble_path)


def get_ensemble_path(
    vault_root: str,
    timeframe: TimeFrame,
    ensemble_name: str,
    direction: Direction
) -> str:
    """
    Get the path to an ensemble directory.
    
    Parameters
    ----------
    vault_root : str
        Root directory of the vault
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
    ensemble_path = Path(vault_root) / timeframe.name / ensemble_dir_name
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
    ensemble_dir: str,
    feature_column: str,
    bias_node_spec: Dict[str, Any],
    base_model: BaseModel
) -> str:
    """
    Add a base model variant to a feature control file.
    
    If the feature control file doesn't exist, creates it.
    If it exists, adds the new base model variant.
    Model ID is auto-generated based on binning model type and hyperparameters.
    
    Parameters
    ----------
    ensemble_dir : str
        Path to ensemble directory (e.g., 'vault/D/commodity_breakout_long')
    feature_column : str
        Feature column name (e.g., 'rsi_signal_D_lookback_2')
    bias_node_spec : Dict[str, Any]
        Bias node specification for feature reconstruction
        Format: {'module_name': str, 'timeframes': [TimeFrame], 'params': dict}
    base_model : BaseModel
        Fitted or unfitted base model instance (contains binning_model)
        
    Returns
    -------
    str
        The auto-generated model_id
        
    Raises
    ------
    ValueError
        If model_id already exists for this feature, or if base model strategy
        doesn't match ensemble direction
    """
    ensemble_path = Path(ensemble_dir)
    features_dir = ensemble_path / 'features'
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate ensemble direction matches base model strategy
    ensemble_dir_name = ensemble_path.name
    if ensemble_dir_name.endswith('_long'):
        expected_direction = 'long'
    elif ensemble_dir_name.endswith('_short'):
        expected_direction = 'short'
    else:
        raise ValueError(f"Cannot determine ensemble direction from directory name: {ensemble_dir_name}")
    
    if base_model.binning_model.strategy != expected_direction:
        raise ValueError(
            f"Base model strategy '{base_model.binning_model.strategy}' does not match "
            f"ensemble direction '{expected_direction}'"
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
    
    # Load existing feature config or create new
    if feature_file.exists():
        with open(feature_file, 'r') as f:
            feature_config = json.load(f)
    else:
        # Create new feature config
        feature_config = {
            'feature_name': feature_column,
            'feature_column': feature_column,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'updated_at': datetime.now(timezone.utc).isoformat(),
            'bias_node_spec': serializable_bias_spec,
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
    ensemble_dir: str,
    feature_column: str,
    fitted_only: bool = False
) -> Dict[str, BaseModel]:
    """
    Load all base model variants for a feature.
    
    Parameters
    ----------
    ensemble_dir : str
        Path to ensemble directory
    feature_column : str
        Feature column name
    fitted_only : bool, default=False
        If True, only return fitted models
        
    Returns
    -------
    Dict[str, BaseModel]
        Dictionary mapping model_id to BaseModel instance
        Keys are model IDs (e.g., 'quantile_binning_3')
        Values are reconstructed BaseModel instances
    """
    ensemble_path = Path(ensemble_dir)
    feature_file = ensemble_path / 'features' / f"{feature_column}.json"
    
    if not feature_file.exists():
        return {}
    
    with open(feature_file, 'r') as f:
        feature_config = json.load(f)
    
    bias_node_spec = feature_config['bias_node_spec'].copy()
    
    # Convert TimeFrame strings back to TimeFrame enums
    if 'timeframes' in bias_node_spec:
        bias_node_spec['timeframes'] = [
            TimeFrame[tf] if isinstance(tf, str) else tf 
            for tf in bias_node_spec['timeframes']
        ]
    
    # Extract ticker from feature column name if possible
    # Feature columns don't contain ticker info, so we need to infer or use default
    # In practice, ensembles are typically ticker-specific, so this is a limitation
    # For now, use a default - users should ensure ticker matches when using loaded models
    ticker = Ticker.ES  # Default - should match the ticker used when models were created
    
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
        
        # Reconstruct BaseModel
        base_model = BaseModel(
            bias_node_spec=bias_node_spec,
            binning_model=binning_model,
            ticker=ticker
        )
        base_model.feature_column = feature_column
        
        models[model_id] = base_model
    
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
    ensemble_path = Path(ensemble_dir)
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
    ensemble_path = Path(ensemble_dir)
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


def list_features(ensemble_dir: str) -> pd.DataFrame:
    """
    List all features in an ensemble.
    
    Parameters
    ----------
    ensemble_dir : str
        Path to ensemble directory
        
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
    ensemble_path = Path(ensemble_dir)
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


def get_bias_node_specs(ensemble_dir: str) -> List[Dict[str, Any]]:
    """
    Get all bias node specifications from an ensemble.
    
    This is useful for production deployment - read all bias node specs
    and use to reconstruct base models.
    
    Parameters
    ----------
    ensemble_dir : str
        Path to ensemble directory
        
    Returns
    -------
    List[Dict[str, Any]]
        List of bias node specifications, one per feature
    """
    ensemble_path = Path(ensemble_dir)
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


def get_all_base_model_names(ensemble_dir: str) -> List[str]:
    """
    Get all base model names (feature::model_id) in an ensemble.
    
    These are the keys used in ensemble weights.
    
    Parameters
    ----------
    ensemble_dir : str
        Path to ensemble directory
        
    Returns
    -------
    List[str]
        List of model names in format 'feature_column::model_id'
    """
    ensemble_path = Path(ensemble_dir)
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

def initialize_vault(vault_root: str) -> None:
    """
    Initialize a new vault directory structure.
    
    Creates the root directory and README.md.
    
    Parameters
    ----------
    vault_root : str
        Path to vault root directory
    """
    vault_path = Path(vault_root)
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
    ensemble_path = Path(ensemble_dir)
    
    if not ensemble_path.exists():
        raise ValueError(f"Ensemble directory does not exist: {ensemble_dir}")
    
    if not ensemble_path.is_dir():
        raise ValueError(f"Path is not a directory: {ensemble_dir}")
    
    features_dir = ensemble_path / 'features'
    if not features_dir.exists():
        raise ValueError(f"Features directory does not exist: {features_dir}")
    
    # Determine ensemble direction from directory name
    ensemble_dir_name = ensemble_path.name
    if ensemble_dir_name.endswith('_long'):
        expected_direction = 'long'
    elif ensemble_dir_name.endswith('_short'):
        expected_direction = 'short'
    else:
        raise ValueError(f"Cannot determine ensemble direction from directory name: {ensemble_dir_name}")
    
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
