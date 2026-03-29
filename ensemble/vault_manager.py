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
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

import utils.core.helpers as helpers
from feature_selection.base_models.feature_base_model import BaseModel
from feature_selection.domain_discrete import build_domain_discrete_bias_node_spec, load_domain_discrete_spec, raise_legacy_feature_artifact

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


def _validate_domain_discrete_bias_node_spec(
    bias_node_spec: Dict[str, Any],
    *,
    prefix: str,
) -> None:
    if not isinstance(bias_node_spec, dict):
        raise ValueError(f"{prefix}bias_node_spec must be a dictionary")
    if bias_node_spec.get("module_name") != "domain_discrete":
        raise_legacy_feature_artifact(
            f"{prefix}bias_node_spec.module_name must be 'domain_discrete'."
        )
    params = bias_node_spec.get("params")
    if not isinstance(params, dict):
        raise ValueError(f"{prefix}bias_node_spec.params must be a dictionary")
    domain_spec = load_domain_discrete_spec(params)
    raw_timeframes = bias_node_spec.get("timeframes", [])
    if raw_timeframes:
        stored_timeframes = [
            tf if isinstance(tf, TimeFrame) else TimeFrame[str(tf)]
            for tf in raw_timeframes
        ]
        if stored_timeframes != list(domain_spec.source_bias_node_spec.timeframes):
            raise ValueError(
                f"{prefix}bias_node_spec.timeframes must match source_bias_node_spec.timeframes"
            )


def _validate_domain_discrete_feature_config(
    feature_config: Dict[str, Any],
    *,
    feature_file: Path,
) -> None:
    required_keys = ["feature_name", "bias_node_spec", "base_models"]
    missing_keys = [key for key in required_keys if key not in feature_config]
    if missing_keys:
        raise ValueError(f"Feature file {feature_file} missing required keys: {missing_keys}")
    if not isinstance(feature_config["base_models"], list):
        raise ValueError(f"Feature file {feature_file} base_models must be a list")
    if len(feature_config["base_models"]) != 1:
        raise ValueError(
            f"Feature file {feature_file} must contain exactly one base model; "
            f"found {len(feature_config['base_models'])}"
        )
    if "feature_column" in feature_config:
        raise_legacy_feature_artifact(
            f"Feature file {feature_file} still stores legacy top-level feature_column."
        )

    _validate_domain_discrete_bias_node_spec(
        feature_config["bias_node_spec"],
        prefix=f"{feature_file}: ",
    )

    model_entry = feature_config["base_models"][0]
    legacy_keys = {
        key
        for key in (
            "binning_model_type",
            "binning_model_params",
            "requires_fit",
            "is_fitted",
            "fitted_params",
        )
        if key in model_entry
    }
    if legacy_keys:
        raise_legacy_feature_artifact(
            f"Feature file {feature_file} contains legacy model keys: {sorted(legacy_keys)}"
        )

    expected_feature_name = _extract_feature_name(feature_config, feature_file.stem)
    if model_entry.get("feature_column") not in {expected_feature_name, feature_file.stem}:
        raise ValueError(
            f"Feature file {feature_file} base model feature_column must match feature name"
        )
    if model_entry.get("model_type") != "domain_discrete":
        raise_legacy_feature_artifact(
            f"Feature file {feature_file} base model model_type must be 'domain_discrete'."
        )
    if model_entry.get("bias_node_spec") != feature_config["bias_node_spec"]:
        raise ValueError(
            f"Feature file {feature_file} base model bias_node_spec must match top-level bias_node_spec"
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


def _canonical_bias_spec_key(spec: Dict[str, Any]) -> tuple[str, tuple[str, ...], str]:
    """Build a stable key for deduplicating bias-node specs."""
    module_name = str(spec.get("module_name", ""))
    timeframes = tuple(
        sorted(
            tf.name if hasattr(tf, "name") else str(tf)
            for tf in spec.get("timeframes", [])
        )
    )
    params = json.dumps(spec.get("params", {}), sort_keys=True, default=str)
    return module_name, timeframes, params


# ============================================================================
# Model ID Generation
# ============================================================================

def generate_model_id(
    binning_model_type: str,
    binning_model_params: Dict[str, Any],
    bias_node_params: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Auto-generate model ID from the frozen domain-discrete spec.
    
    Parameters
    ----------
    binning_model_type : str
        Canonical model type. Must be ``domain_discrete``.
    binning_model_params : Dict[str, Any]
        Unused for domain-discrete features.
        
    Returns
    -------
    str
        Auto-generated model ID
        
    Examples
    --------
    >>> generate_model_id('domain_discrete', {}, {'params': {'spec_version': 'v1'}})
    'domain_discrete_v1'
    """
    if binning_model_type != "domain_discrete":
        _reject_legacy_feature_artifact(
            f"generate_model_id no longer supports legacy model_type={binning_model_type!r}."
        )

    spec_payload: Dict[str, Any] = {}
    if bias_node_params:
        spec_payload = dict(bias_node_params.get("params", bias_node_params))
    else:
        spec_payload = dict(binning_model_params)

    domain_spec = load_domain_discrete_spec(spec_payload)
    scope_name = domain_spec.ticker_scope.scope_name
    scope_token = scope_name or "_".join(ticker.name for ticker in domain_spec.ticker_scope.tickers)
    return "_".join([
        "domain_discrete",
        _model_id_token(domain_spec.source_bias_node_spec.module_name),
        _model_id_token(domain_spec.direction.value),
        _model_id_token(scope_token),
        _model_id_token(domain_spec.spec_version),
    ])


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
    if feature_name is None and "feature_column" in legacy_kwargs:
        feature_name = legacy_kwargs["feature_column"]
    if feature_name is None:
        raise ValueError("feature_name must be provided")
    if base_model is None:
        raise ValueError("base_model must be provided")

    domain_spec = getattr(base_model, "domain_discrete_spec", None)
    if domain_spec is None:
        _reject_legacy_feature_artifact(
            "add_feature_to_ensemble requires a domain_discrete BaseModel."
        )

    if bias_node_spec is not None and bias_node_spec != base_model.bias_node_spec:
        _reject_legacy_feature_artifact(
            "Provided bias_node_spec does not match the BaseModel's frozen domain-discrete spec."
        )

    if bias_node_params is not None:
        _reject_legacy_feature_artifact(
            "add_feature_to_ensemble no longer accepts bias_node_params."
        )

    if ensemble_dir is None:
        ensemble_dir = _DEFAULT_ENSEMBLE_DIR
        if ensemble_dir is None:
            raise ValueError(
                "ensemble_dir must be provided or call create_ensemble_directory() first. "
                "Example: create_ensemble_directory(TimeFrame.D, 'buy_hold', Direction.LONG)"
            )

    ensemble_path = _resolve_ensemble_path(ensemble_dir)
    features_dir = ensemble_path / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    ensemble_config_file = ensemble_path / "ensemble_config.json"
    if not ensemble_config_file.exists():
        raise ValueError(
            f"Ensemble config file does not exist: {ensemble_config_file}. "
            "Legacy directory-name inference is no longer supported."
        )

    with open(ensemble_config_file, "r") as f:
        ensemble_config = json.load(f)
    expected_ticker_names = sorted(ensemble_config.get("tickers", []))
    expected_direction = coerce_direction(
        ensemble_config.get("direction", Direction.LONG.value),
        field_name="ensemble_config.direction",
    )

    if domain_spec.direction != expected_direction:
        raise ValueError(
            f"Base model direction '{domain_spec.direction.value}' does not match "
            f"ensemble direction '{expected_direction.value}'"
        )

    ticker_names = sorted(
        ticker.name if isinstance(ticker, Ticker) else str(ticker)
        for ticker in (tickers or list(domain_spec.ticker_scope.tickers))
    )
    if expected_ticker_names is not None and ticker_names != expected_ticker_names:
        raise ValueError(
            f"Tickers do not match ensemble tickers. Ensemble tickers: {expected_ticker_names}, "
            f"Provided: {ticker_names}."
        )
    if sorted(ticker.name for ticker in domain_spec.ticker_scope.tickers) != ticker_names:
        raise ValueError(
            "Base model tickers must match the frozen domain-discrete ticker_scope."
        )

    model_id = generate_model_id("domain_discrete", {}, bias_node_params={"params": domain_spec.to_mapping()})
    feature_file = features_dir / f"{feature_name}.json"
    serializable_bias_spec = build_domain_discrete_bias_node_spec(domain_spec)

    existing_config: Optional[Dict[str, Any]] = None
    if feature_file.exists():
        with open(feature_file, "r") as f:
            existing_config = json.load(f)
        _validate_domain_discrete_feature_config(existing_config, feature_file=feature_file)
        existing_model_id = existing_config["base_models"][0]["model_id"]
        if existing_model_id != model_id:
            raise ValueError(
                f"Feature '{feature_name}' already exists with model_id '{existing_model_id}' "
                f"which does not match the frozen spec-derived id '{model_id}'."
            )
        if existing_config["bias_node_spec"] != serializable_bias_spec:
            raise ValueError(
                f"Feature '{feature_name}' already exists with a different frozen spec."
            )
        return model_id

    feature_config = {
        "feature_name": feature_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "bias_node_spec": serializable_bias_spec,
        "tickers": ticker_names,
        "base_models": [
            {
                "model_id": model_id,
                "model_name": f"{feature_name}::{model_id}",
                "model_type": "domain_discrete",
                "feature_column": feature_name,
                "strategy": domain_spec.direction.value,
                "bias_node_spec": serializable_bias_spec,
            }
        ],
    }
    _validate_domain_discrete_feature_config(feature_config, feature_file=feature_file)

    with open(feature_file, "w") as f:
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
    Load canonical domain-discrete base models for a feature.
    """
    from ensemble.ensemble_utils import create_base_model_from_config

    global _DEFAULT_ENSEMBLE_DIR

    if feature_name is None and "feature_column" in legacy_kwargs:
        feature_name = legacy_kwargs["feature_column"]
    if feature_name is None:
        raise ValueError("feature_name must be provided")

    if ensemble_dir is None:
        ensemble_dir = _DEFAULT_ENSEMBLE_DIR
        if ensemble_dir is None:
            raise ValueError(
                "ensemble_dir must be provided or call create_ensemble_directory() first."
            )

    ensemble_path = _resolve_ensemble_path(ensemble_dir)
    features_dir = ensemble_path / "features"
    feature_file = _resolve_feature_file_path(features_dir, feature_name)
    if feature_file is None or not feature_file.exists():
        raise FileNotFoundError(
            f"Feature file not found for feature '{feature_name}' in {features_dir}"
        )

    with open(feature_file, "r") as f:
        feature_config = json.load(f)

    _validate_domain_discrete_feature_config(feature_config, feature_file=feature_file)
    resolved_feature_name = _extract_feature_name(feature_config, feature_file.stem)
    feature_bias_spec = feature_config["bias_node_spec"]
    domain_spec = load_domain_discrete_spec(feature_bias_spec["params"])

    if tickers is None:
        tickers = [
            Ticker[ticker_name] if isinstance(ticker_name, str) else ticker_name
            for ticker_name in feature_config.get("tickers", [ticker.name for ticker in domain_spec.ticker_scope.tickers])
        ]

    model_entry = feature_config["base_models"][0]
    model_id = model_entry["model_id"]
    models: Dict[Tuple[Ticker, str], BaseModel] = {}

    for ticker in tickers:
        ticker_enum = ticker if isinstance(ticker, Ticker) else Ticker[str(ticker)]
        base_model_config = {
            "name": model_entry["model_name"],
            "feature_column": resolved_feature_name,
            "model_type": "domain_discrete",
            "strategy": model_entry["strategy"],
            "bias_node_spec": feature_bias_spec,
            "tickers": [ticker_enum],
        }
        base_model = create_base_model_from_config(
            base_model_config,
            ticker=ticker_enum,
            use_cache=not fitted_only,
        )
        base_model.feature_column = resolved_feature_name
        models[(ticker_enum, model_id)] = base_model

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
    raise_legacy_feature_artifact(
        "update_base_model_fitted_params is no longer supported; domain_discrete features are frozen."
    )


def consolidate_feature_files(ensemble_dir: str) -> List[str]:
    ensemble_path = _resolve_ensemble_path(ensemble_dir)
    features_dir = ensemble_path / "features"
    if not features_dir.exists():
        return []

    validated_paths: List[str] = []
    for feature_file in sorted(features_dir.glob("*.json")):
        with open(feature_file, "r") as handle:
            feature_config = json.load(handle)
        _validate_domain_discrete_feature_config(feature_config, feature_file=feature_file)
        validated_paths.append(str(feature_file))

    return validated_paths


def migrate_legacy_feature_members_schema(ensemble_dir: str) -> List[str]:
    ensemble_path = _resolve_ensemble_path(ensemble_dir)
    features_dir = ensemble_path / "features"
    if not features_dir.exists():
        return []

    updated_paths: List[str] = []
    for feature_file in sorted(features_dir.glob("*.json")):
        with open(feature_file, "r", encoding="utf-8") as handle:
            feature_config = json.load(handle)

        base_models = feature_config.get("base_models", [])
        if not isinstance(base_models, list) or not base_models:
            continue
        model_entry = base_models[0]
        if "members" not in model_entry:
            continue
        if model_entry["members"]:
            raise_legacy_feature_artifact(
                f"Feature file {feature_file} still contains non-empty legacy members schema payloads."
            )

        model_entry.pop("members", None)
        feature_config["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(feature_file, "w", encoding="utf-8") as handle:
            json.dump(feature_config, handle, indent=2)
            handle.write("\n")
        updated_paths.append(str(feature_file))

    return updated_paths



def remove_base_model_variant(
    ensemble_dir: str,
    feature_name: str,
    model_id: str
) -> None:
    raise_legacy_feature_artifact(
        "remove_base_model_variant is not supported after the domain_discrete cutover."
    )


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
    
    for feature_file in sorted(features_dir.glob('*.json')):
        with open(feature_file, 'r') as f:
            feature_config = json.load(f)
        
        _validate_domain_discrete_feature_config(feature_config, feature_file=feature_file)
        resolved_feature_name = _extract_feature_name(feature_config, feature_file.stem)
        features.append({
            'feature_name': resolved_feature_name,
            'feature_column': resolved_feature_name,
            'n_base_models': 1,
            'n_fitted': 0,
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
        Frozen domain-discrete bias node specifications, one per feature.
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
        _validate_domain_discrete_feature_config(feature_config, feature_file=feature_file)
        bias_specs.append(dict(feature_config['bias_node_spec']))
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
        List of model names in format 'feature_name::model_id'
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
        _validate_domain_discrete_feature_config(feature_config, feature_file=feature_file)
        model_names.append(feature_config['base_models'][0]['model_name'])
    return model_names


def ensure_vault_cache_coverage(
    vault_ensemble_dirs: Sequence[str],
    start_date: datetime,
    end_date: datetime,
    refresh_mode: str = "missing_stale_only",
) -> Dict[str, Any]:
    """Ensure vault ensemble bias-node caches cover the requested window.

    This wrapper stays intentionally thin: it resolves ensemble paths, migrates
    legacy feature files, and delegates the refresh to ``CacheManager``.
    """
    from utils.cache.cache_manager import CacheManager

    resolved_dirs = list(
        dict.fromkeys(
            str(_resolve_ensemble_path(ensemble_dir))
            for ensemble_dir in vault_ensemble_dirs
        )
    )
    if not resolved_dirs:
        raise ValueError("vault_ensemble_dirs must be non-empty")

    for ensemble_dir in resolved_dirs:
        migrate_legacy_feature_members_schema(ensemble_dir)

    manager = CacheManager()
    summary = manager.ensure_vault_cache_coverage(
        vault_ensemble_dirs=resolved_dirs,
        start_date=start_date,
        end_date=end_date,
        refresh_mode=refresh_mode,
    )
    return {
        **summary,
        "vault_ensemble_dirs": resolved_dirs,
    }


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
    - All feature control files are canonical domain_discrete payloads
    - All base model strategies match ensemble direction
    - All bias node specs are valid and parseable
    
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
    if not ensemble_config_file.exists():
        raise ValueError(
            f"Ensemble config file does not exist: {ensemble_config_file}. "
            "Legacy directory-name inference is no longer supported."
        )

    with open(ensemble_config_file, 'r') as f:
        ensemble_config = json.load(f)
    expected_direction = coerce_direction(
        ensemble_config.get('direction', Direction.LONG.value),
        field_name="ensemble_config.direction",
    )
    expected_ticker_names = sorted(ensemble_config.get('tickers', []))

    # Validate all feature control files
    for feature_file in sorted(features_dir.glob('*.json')):
        try:
            with open(feature_file, 'r') as f:
                feature_config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in feature file {feature_file}: {e}")
        _validate_domain_discrete_feature_config(feature_config, feature_file=feature_file)

        feature_ticker_names = sorted(feature_config.get("tickers", []))
        if expected_ticker_names is not None and feature_ticker_names != expected_ticker_names:
            raise ValueError(
                f"Feature '{feature_file.stem}' has tickers {feature_ticker_names} "
                f"but ensemble expects {expected_ticker_names}."
            )
        if sorted(feature_config["base_models"][0]["bias_node_spec"]["params"]["ticker_scope"]["tickers"]) != feature_ticker_names:
            raise ValueError(
                f"Feature '{feature_file.stem}' ticker_scope must match stored tickers"
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
        If True, the temporary control file is marked unfitted.
        If False, the ensemble-level fitted payload is synthesized from the frozen specs.
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
    if not ensemble_config_file.exists():
        raise ValueError(
            f"Ensemble config file does not exist: {ensemble_config_file}. "
            "Legacy directory-name inference is no longer supported."
        )

    with open(ensemble_config_file, 'r') as f:
        ensemble_config = json.load(f)
    timeframe_str = ensemble_config.get('timeframe', 'D')
    
    # Load all feature configs and convert to control file format
    base_models_config = []
    model_identity_by_name: Dict[str, Dict[str, str]] = {}
    all_tickers = set()
    
    for feature_file in feature_files:
        with open(feature_file, 'r') as f:
            feature_config = json.load(f)
        _validate_domain_discrete_feature_config(feature_config, feature_file=feature_file)
        feature_name = _extract_feature_name(feature_config, feature_file.stem)
        feature_models = feature_config["base_models"]
        model = feature_models[0]
        feature_tickers = feature_config.get("tickers", []) or [
            ticker.name for ticker in load_domain_discrete_spec(feature_config["bias_node_spec"]["params"]).ticker_scope.tickers
        ]
        all_tickers.update(feature_tickers)

        base_model_config = {
            "name": model["model_name"],
            "feature_column": feature_name,
            "model_type": "domain_discrete",
            "strategy": model["strategy"],
            "bias_node_spec": feature_config["bias_node_spec"],
            "tickers": feature_tickers,
        }
        base_models_config.append(base_model_config)
        model_identity_by_name[str(base_model_config['name'])] = {
            "feature_name": feature_name,
            "model_id": str(model["model_id"]),
        }
    
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
        model_names = [bm['name'] for bm in base_models_config]
        fitted_ensemble = None
        if not refit:
            fitted_ensemble = {
                'weights': {name: 1.0 / len(model_names) for name in model_names} if model_names else {},
                'exposure_fractions': {name: 0.5 for name in model_names},
                'model_exposure_fractions': {name: 0.5 for name in model_names},
                'feature_names': model_names,
                'target_volatility': target_volatility,
                'unique_tickers': tickers,
                'instrument_weights': {t: 1.0 / len(tickers) for t in tickers},
                'n_tickers': len(tickers),
            }

        temp_path = save_control_file(
            filepath=f.name,
            base_models=base_models_config,
            metadata=metadata,
            fitted_ensemble=fitted_ensemble,
            tickers=tickers,
        )
    
    # Create ensemble from control file
    ensemble = DiversifiedEnsemble(
        control_file_path=temp_path,
        target_volatility=target_volatility,
        base_tf=TimeFrame[timeframe_str]
    )

    ensemble.vault_ensemble_dir = str(ensemble_path)
    ensemble.vault_ensemble_name = ensemble_path.name
    ensemble.vault_timeframe = timeframe_str
    ensemble.vault_base_model_identities = model_identity_by_name

    # Clean up temp file
    os.unlink(temp_path)
    
    return ensemble
