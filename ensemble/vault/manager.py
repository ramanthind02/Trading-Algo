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
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import pandas as pd

from ensemble.vault.feature_files import (
    build_feature_listing,
    collect_base_model_names,
    collect_bias_node_specs,
    consolidate_feature_files_in_dir,
    extract_feature_name,
    feature_file_exists,
    iter_validated_feature_configs,
    load_validated_feature_config,
    migrate_legacy_feature_members_schema_in_dir,
    resolve_feature_file_path,
    validate_feature_configs_for_ensemble,
    validate_signed_signal_feature_config,
)
from features.models.feature_base_model import BaseModel

from lib.cache.runtime.cache_paths import project_root, win32_extended_path
from lib.core.enums import Direction, DirectionInput, TimeFrame, Ticker, coerce_direction
from lib.core.vault_paths import resolve_vault_personal, resolve_vault_prop, resolve_vault_root

from ensemble.vault.constants import VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES

# Type hint for forward reference
if TYPE_CHECKING:
    from ensemble.diversified_ensemble import DiversifiedEnsemble


# ============================================================================
# Module-Level Configuration
# ============================================================================

# Default vault root dirname. Resolution is handled through utils.vault_paths.
VAULT_ROOT: str = 'vault'

# Default ensemble directory/tickers (set by create_ensemble_directory). Use only for
# CLI/interactive defaults; in library code prefer passing ensemble_dir explicitly.
_DEFAULT_ENSEMBLE_DIR: Optional[str] = None
_DEFAULT_ENSEMBLE_TICKERS: Optional[List[Ticker]] = None

_REQUIRE_ENSEMBLE_DIR_MESSAGE = (
    "ensemble_dir must be provided or call create_ensemble_directory() first. "
    "Example: create_ensemble_directory(TimeFrame.D, 'buy_hold', Direction.LONG)"
)

_AUTO_DETECT_ENSEMBLE_DIR_MESSAGE = (
    "ensemble_dir must be provided or call create_ensemble_directory() first. "
    "Could not auto-detect any ensemble directory with features. "
    "Example: create_ensemble_directory(TimeFrame.D, 'buy_hold', Direction.LONG)"
)


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


def _iter_vault_root_candidates() -> List[Path]:
    """Prop vault first, then personal (deduped). Used for auto-detect and path fallbacks."""
    out: List[Path] = []
    seen: set[str] = set()
    for p in (resolve_vault_prop(), resolve_vault_personal()):
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _iter_vault_timeframe_dirs(vault_path: Path) -> List[Path]:
    if not vault_path.exists():
        return []
    return [
        timeframe_path
        for timeframe_path in sorted(vault_path.iterdir())
        if timeframe_path.is_dir() and timeframe_path.name in {"D", "W", "M"}
    ]


def _parse_ensemble_dir_identity(dir_name: str) -> Optional[tuple[str, str]]:
    suffixes = (
        ("_long_short", "long_short"),
        ("_long", "long"),
        ("_short", "short"),
    )
    for suffix, direction in suffixes:
        if dir_name.endswith(suffix):
            return dir_name[: -len(suffix)], direction
    return None


def _is_vault_weight_group_directory(path: Path) -> bool:
    return path.is_dir() and path.name in VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES


def _ensemble_leaf_directory(path: Path) -> bool:
    """True if ``path`` looks like an ensemble directory (config and/or feature JSON)."""
    if not path.is_dir():
        return False
    if (path / "ensemble_config.json").exists():
        return True
    return _has_feature_files(path)


def _iter_vault_ensemble_dirs(vault_path: Path) -> List[tuple[str, Path]]:
    """Yield ``(timeframe_name, ensemble_path)`` for each ensemble under the vault.

    Supports:

    - **Nested (preferred):** ``vault/<D|W|M>/<weight_group>/<ensemble_name>/``
    - **Legacy flat:** ``vault/<D|W|M>/<ensemble_name>/``
    """
    ensemble_dirs: List[tuple[str, Path]] = []
    for timeframe_path in _iter_vault_timeframe_dirs(vault_path):
        tf_name = timeframe_path.name
        for child in sorted(timeframe_path.iterdir()):
            if not child.is_dir():
                continue
            if _is_vault_weight_group_directory(child):
                for ensemble_path in sorted(child.iterdir()):
                    if _ensemble_leaf_directory(ensemble_path):
                        ensemble_dirs.append((tf_name, ensemble_path))
            elif _ensemble_leaf_directory(child):
                ensemble_dirs.append((tf_name, child))
    return ensemble_dirs


def _find_ensemble_path_by_leaf_name(vault_root: Path, ensemble_dir_name: str) -> Optional[Path]:
    """Locate ``vault/<TF>/<group?>/<ensemble_dir_name>`` by leaf folder name."""
    for _tf, ensemble_path in _iter_vault_ensemble_dirs(vault_root):
        if ensemble_path.name == ensemble_dir_name:
            return ensemble_path
    return None


def _has_feature_files(ensemble_path: Path) -> bool:
    features_dir = _get_features_dir(ensemble_path)
    return features_dir.exists() and next(features_dir.glob("*.json"), None) is not None


def _resolve_vault_root_path(vault_root: Optional[str] = None) -> Path:
    return resolve_vault_root(vault_root)


def _autodetect_ensemble_dir_with_features() -> Optional[str]:
    for vault_path in _iter_vault_root_candidates():
        if not vault_path.exists():
            continue
        for _, ensemble_dir_path in _iter_vault_ensemble_dirs(vault_path):
            if _has_feature_files(ensemble_dir_path):
                return str(ensemble_dir_path)
    return None


def _require_ensemble_dir(
    ensemble_dir: Optional[str],
    *,
    auto_detect: bool = False,
) -> str:
    global _DEFAULT_ENSEMBLE_DIR

    if ensemble_dir is not None:
        return ensemble_dir
    if _DEFAULT_ENSEMBLE_DIR is not None:
        return _DEFAULT_ENSEMBLE_DIR
    if auto_detect:
        detected = _autodetect_ensemble_dir_with_features()
        if detected is not None:
            _DEFAULT_ENSEMBLE_DIR = detected
            return detected
        raise ValueError(_AUTO_DETECT_ENSEMBLE_DIR_MESSAGE)
    raise ValueError(_REQUIRE_ENSEMBLE_DIR_MESSAGE)


def _load_ensemble_config(ensemble_path: Path) -> Dict[str, Any]:
    ensemble_config_file = ensemble_path / "ensemble_config.json"
    if not ensemble_config_file.exists():
        raise ValueError(
            f"Ensemble config file does not exist: {ensemble_config_file}. "
            "Legacy directory-name inference is no longer supported."
        )
    with open(ensemble_config_file, "r") as handle:
        return json.load(handle)


def _get_features_dir(ensemble_path: Path, *, require_exists: bool = False) -> Path:
    features_dir = ensemble_path / "features"
    if require_exists and not features_dir.exists():
        raise ValueError(f"Features directory does not exist: {features_dir}")
    return features_dir


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
    ensemble_dir = _require_ensemble_dir(ensemble_dir)
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

    repo_relative = project_root() / ensemble_dir
    if repo_relative.exists():
        return repo_relative

    for vault_root_path in _iter_vault_root_candidates():
        nested_resolved = vault_root_path / ensemble_dir
        if nested_resolved.exists():
            return nested_resolved
        parts = Path(ensemble_dir).parts
        if parts:
            leaf = parts[-1]
            found = _find_ensemble_path_by_leaf_name(vault_root_path, leaf)
            if found is not None:
                return found

    # If still not found, return the original path (will fail later with better error)
    return ensemble_path


def _model_id_token(value: Any) -> str:
    """Create a stable token safe for model-id suffixes."""
    if isinstance(value, list):
        # Join list elements for a clean token (e.g. ["TLT"] -> "tlt")
        value = "_".join(str(v) for v in value)
    token = re.sub(r"[^A-Za-z0-9]+", "-", str(value)).strip("-").lower()
    return token or "na"


def _timeframe_names_for_spec(raw_timeframes: object) -> list[str]:
    """Normalize ``bias_node_spec.timeframes`` to JSON-safe names (e.g. ``\"D\"``)."""
    if not isinstance(raw_timeframes, list):
        return []
    names: list[str] = []
    for tf in raw_timeframes:
        if isinstance(tf, TimeFrame):
            names.append(tf.name)
        else:
            names.append(str(tf))
    return names


def _normalize_bias_node_spec_for_storage(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Copy spec with string timeframes for JSON persistence and stable comparisons."""
    out = dict(spec)
    out["timeframes"] = _timeframe_names_for_spec(spec.get("timeframes", []))
    return out


def _normalize_ensemble_tickers(
    tickers: Optional[List[Ticker] | Ticker],
) -> tuple[List[Ticker], List[str]]:
    normalized_tickers: List[Ticker]
    if tickers is None:
        normalized_tickers = [Ticker.ES]
    elif isinstance(tickers, Ticker):
        normalized_tickers = [tickers]
    else:
        normalized_tickers = list(tickers)

    ticker_names = sorted(
        ticker.name if isinstance(ticker, Ticker) else str(ticker)
        for ticker in normalized_tickers
    )
    return normalized_tickers, ticker_names


def _build_ensemble_config(
    timeframe: TimeFrame,
    ensemble_name: str,
    direction: Direction,
    ticker_names: List[str],
) -> Dict[str, Any]:
    timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "timeframe": timeframe.name,
        "ensemble_name": ensemble_name,
        "direction": direction.value,
        "tickers": ticker_names,
        "created_at": timestamp,
        "updated_at": timestamp,
    }


def _write_ensemble_config(ensemble_path: Path, config: Dict[str, Any]) -> None:
    ensemble_config_file = ensemble_path / "ensemble_config.json"
    with open(ensemble_config_file, "w") as handle:
        json.dump(config, handle, indent=2)


def _ensure_ensemble_directory_state(
    ensemble_path: Path,
    *,
    config: Dict[str, Any],
    ticker_names: List[str],
) -> None:
    if ensemble_path.exists():
        ensemble_config_file = ensemble_path / "ensemble_config.json"
        if ensemble_config_file.exists():
            existing_tickers = sorted(_load_ensemble_config(ensemble_path).get("tickers", []))
            if existing_tickers != ticker_names:
                raise ValueError(
                    f"Ensemble directory already exists with different tickers. "
                    f"Existing: {existing_tickers}, Provided: {ticker_names}. "
                    f"All features in an ensemble must use the same tickers."
                )
        else:
            _write_ensemble_config(ensemble_path, config)
        return

    _get_features_dir(ensemble_path).mkdir(parents=True, exist_ok=False)
    _write_ensemble_config(ensemble_path, config)


def _set_default_ensemble_context(
    ensemble_dir: str,
    tickers: List[Ticker],
) -> None:
    global _DEFAULT_ENSEMBLE_DIR, _DEFAULT_ENSEMBLE_TICKERS

    _DEFAULT_ENSEMBLE_DIR = ensemble_dir
    _DEFAULT_ENSEMBLE_TICKERS = tickers


# ============================================================================
# Model ID Generation
# ============================================================================

def generate_model_id(
    model_type: str,
    model_params: Dict[str, Any],
    bias_node_params: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Auto-generate model ID from the native bias-node spec.
    
    Parameters
    ----------
    model_type : str
        Canonical model type. Must be ``signed_signal``.
    model_params : Dict[str, Any]
        Reserved for future metadata.
        
    Returns
    -------
    str
        Auto-generated model ID
        
    Examples
    --------
    >>> generate_model_id('signed_signal', {}, {'module_name': 'rsi', 'timeframes': ['D'], 'params': {'lookback': 14}})
    'signed_signal_rsi_d_lookback_14'
    """
    if model_type != "signed_signal":
        raise ValueError(f"generate_model_id only supports model_type='signed_signal', got {model_type!r}.")

    spec_payload = dict(bias_node_params or model_params)
    module_name = _model_id_token(str(spec_payload.get("module_name", "signal")))
    raw_timeframes = spec_payload.get("timeframes", [])
    tf_parts = _timeframe_names_for_spec(raw_timeframes)
    tf_token = _model_id_token("_".join(tf_parts) or "na")
    params = spec_payload.get("params", {})
    param_tokens = [
        "_".join((_model_id_token(str(key)), _model_id_token(str(value))))
        for key, value in sorted(dict(params).items())
    ]
    parts = ["signed_signal", module_name, tf_token, *param_tokens]
    return "_".join(part for part in parts if part)


# ============================================================================
# Ensemble Directory Management
# ============================================================================


def _build_ensemble_dir_name(ensemble_name: Any, direction: Direction) -> str:
    return f"{str(ensemble_name).strip()}_{direction.value}"


def _ensemble_path_for_create(
    *,
    vault_path: Path,
    timeframe_name: str,
    ensemble_dir_name: str,
    weight_hierarchy_group: Optional[str],
) -> Path:
    if weight_hierarchy_group is None:
        return vault_path / timeframe_name / ensemble_dir_name
    group = str(weight_hierarchy_group).strip()
    if not group:
        raise ValueError("weight_hierarchy_group must be a non-empty string when provided")
    if group not in VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES:
        raise ValueError(
            f"weight_hierarchy_group must be one of {sorted(VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES)}, "
            f"got {group!r}"
        )
    return vault_path / timeframe_name / group / ensemble_dir_name


def create_ensemble_directory(
    timeframe: Any,
    ensemble_name: Any,
    direction: DirectionInput,
    tickers: Optional[List[Ticker]] = None,
    vault_root: Optional[str] = None,
    *,
    weight_hierarchy_group: Optional[str] = None,
) -> str:
    """
    Create a new ensemble directory in the vault.
    
    The vault root comes from ``vault_root=`` or :func:`utils.vault_paths.resolve_vault_root`.
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
    weight_hierarchy_group : str, optional
        When set, the ensemble is created under
        ``vault/<TF>/<weight_hierarchy_group>/<ensemble_name_direction>/`` (manual
        weight-layer grouping). Must be a key in
        ``VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES``. When omitted, the legacy flat
        layout ``vault/<TF>/<ensemble_name_direction>/`` is used.

    Returns
    -------
    str
        Path to the created ensemble directory (e.g. ``vault/D/momentum/foo_long``)
        
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
    tickers, ticker_names = _normalize_ensemble_tickers(tickers)

    ensemble_dir_name = _build_ensemble_dir_name(ensemble_name, direction)
    vault_path = _resolve_vault_root_path(vault_root)
    ensemble_path = _ensemble_path_for_create(
        vault_path=vault_path,
        timeframe_name=timeframe.name,
        ensemble_dir_name=ensemble_dir_name,
        weight_hierarchy_group=weight_hierarchy_group,
    )
    ensemble_dir_str = str(ensemble_path)

    config = _build_ensemble_config(
        timeframe=timeframe,
        ensemble_name=str(ensemble_name),
        direction=direction,
        ticker_names=ticker_names,
    )
    _ensure_ensemble_directory_state(
        ensemble_path,
        config=config,
        ticker_names=ticker_names,
    )
    _set_default_ensemble_context(ensemble_dir_str, tickers)
    
    return ensemble_dir_str


def get_ensemble_path(
    timeframe: Any,
    ensemble_name: Any,
    direction: DirectionInput,
    vault_root: Optional[str] = None,
    *,
    weight_hierarchy_group: Optional[str] = None,
) -> str:
    """
    Get the path to an ensemble directory.

    The vault root comes from ``vault_root=`` or :func:`utils.vault_paths.resolve_vault_root`.

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

    ensemble_dir_name = _build_ensemble_dir_name(ensemble_name, direction)
    root = _resolve_vault_root_path(vault_root)
    if weight_hierarchy_group is not None:
        return str(
            _ensemble_path_for_create(
                vault_path=root,
                timeframe_name=timeframe.name,
                ensemble_dir_name=ensemble_dir_name,
                weight_hierarchy_group=weight_hierarchy_group,
            )
        )
    nested = _find_ensemble_path_by_leaf_name(root, ensemble_dir_name)
    if nested is not None:
        return str(nested)
    return str(root / timeframe.name / ensemble_dir_name)


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
    vault_path = resolve_vault_root(vault_root)
    if not vault_path.exists():
        return pd.DataFrame(
            columns=[
                'ensemble_name',
                'timeframe',
                'direction',
                'weight_hierarchy_group',
                'n_features',
                'path',
            ]
        )
    
    ensembles = []

    for timeframe, ensemble_dir in _iter_vault_ensemble_dirs(vault_path):
        parsed_identity = _parse_ensemble_dir_identity(ensemble_dir.name)
        if parsed_identity is None:
            continue
        ensemble_name, direction = parsed_identity
        features_dir = _get_features_dir(ensemble_dir)
        n_features = len(list(features_dir.glob("*.json"))) if features_dir.exists() else 0
        parent_name = ensemble_dir.parent.name
        weight_group = (
            parent_name if parent_name in VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES else None
        )
        ensembles.append({
            'ensemble_name': ensemble_name,
            'timeframe': timeframe,
            'direction': direction,
            'weight_hierarchy_group': weight_group,
            'n_features': n_features,
            'path': str(ensemble_dir)
        })
    
    return pd.DataFrame(ensembles)


def _feature_tickers_from_config(feature_config: Dict[str, Any]) -> List[str]:
    configured_tickers = feature_config.get("tickers", [])
    if configured_tickers:
        return list(configured_tickers)
    return []


def _base_model_config_from_feature(
    feature_file: Path,
    feature_config: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, str], List[str]]:
    feature_name = extract_feature_name(feature_config, feature_file.stem)
    model = feature_config["base_models"][0]
    feature_tickers = _feature_tickers_from_config(feature_config)
    base_model_config = {
        "name": model["model_name"],
        "feature_column": feature_name,
        "model_type": "signed_signal",
        "strategy": model["strategy"],
        "bias_node_spec": feature_config["bias_node_spec"],
        "tickers": feature_tickers,
    }
    model_identity = {
        "feature_name": feature_name,
        "model_id": str(model["model_id"]),
    }
    return base_model_config, model_identity, feature_tickers


def _build_ensemble_loader_payload(
    validated_feature_configs: List[tuple[Path, Dict[str, Any]]],
) -> tuple[List[Dict[str, Any]], Dict[str, Dict[str, str]], List[str]]:
    base_models_config: List[Dict[str, Any]] = []
    model_identity_by_name: Dict[str, Dict[str, str]] = {}
    all_tickers: set[str] = set()

    for feature_file, feature_config in validated_feature_configs:
        base_model_config, model_identity, feature_tickers = _base_model_config_from_feature(
            feature_file,
            feature_config,
        )
        base_models_config.append(base_model_config)
        model_identity_by_name[str(base_model_config["name"])] = model_identity
        all_tickers.update(feature_tickers)

    return base_models_config, model_identity_by_name, sorted(all_tickers)


def _build_vault_loader_metadata(
    *,
    first_feature_config: Dict[str, Any],
    timeframe_str: str,
    refit: bool,
    selection_method: str,
) -> Dict[str, Any]:
    metadata = {
        "created_at": first_feature_config.get("created_at", datetime.now(timezone.utc).isoformat()),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "is_fit": not refit,
        "base_tf": timeframe_str,
    }
    if not refit:
        metadata["selection_method"] = selection_method
    return metadata


def _build_vault_loader_fitted_ensemble(
    *,
    model_names: List[str],
    tickers: List[str],
    target_volatility: float,
) -> Dict[str, Any]:
    return {
        "weights": {name: 1.0 / len(model_names) for name in model_names} if model_names else {},
        "exposure_fractions": {name: 0.5 for name in model_names},
        "model_exposure_fractions": {name: 0.5 for name in model_names},
        "feature_names": model_names,
        "target_volatility": target_volatility,
        "unique_tickers": tickers,
        "instrument_weights": {ticker: 1.0 / len(tickers) for ticker in tickers},
        "n_tickers": len(tickers),
    }


def _write_temp_vault_control_file(
    *,
    base_models_config: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    fitted_ensemble: Optional[Dict[str, Any]],
    tickers: List[str],
) -> str:
    from ensemble.ensemble_utils import save_control_file
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
        return save_control_file(
            filepath=handle.name,
            base_models=base_models_config,
            metadata=metadata,
            fitted_ensemble=fitted_ensemble,
            tickers=tickers,
        )


def _validate_feature_add_request(
    *,
    base_model: BaseModel,
    bias_node_spec: Optional[Dict[str, Any]],
    bias_node_params: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    resolved_bias_node_spec = dict(base_model.bias_node_spec)
    if bias_node_spec is not None and bias_node_spec != base_model.bias_node_spec:
        raise ValueError(
            "Provided bias_node_spec does not match the BaseModel's native signed-signal spec."
        )
    if bias_node_params is not None:
        raise ValueError("add_feature_to_ensemble no longer accepts bias_node_params.")
    return resolved_bias_node_spec


def _resolve_feature_ticker_names(
    bias_node_spec: Dict[str, Any],
    tickers: Optional[List[Ticker]],
) -> List[str]:
    del bias_node_spec
    return sorted(
        ticker.name if isinstance(ticker, Ticker) else str(ticker)
        for ticker in (tickers or [])
    )


def _validate_feature_alignment(
    *,
    feature_name: str,
    bias_node_spec: Dict[str, Any],
    ensemble_config: Dict[str, Any],
    ticker_names: List[str],
) -> None:
    expected_ticker_names = sorted(ensemble_config.get("tickers", []))
    expected_direction = coerce_direction(
        ensemble_config.get("direction", Direction.LONG.value),
        field_name="ensemble_config.direction",
    )

    feature_direction = coerce_direction(
        feature_config_direction := bias_node_spec.get("params", {}).get("direction", expected_direction.value),
        field_name=f"{feature_name}.bias_node_spec.params.direction",
    ) if isinstance(bias_node_spec.get("params", {}), dict) and "direction" in bias_node_spec.get("params", {}) else expected_direction

    if feature_direction != expected_direction:
        raise ValueError(
            f"Base model direction '{feature_direction.value}' does not match "
            f"ensemble direction '{expected_direction.value}'"
        )
    if expected_ticker_names is not None and ticker_names != expected_ticker_names:
        raise ValueError(
            f"Tickers do not match ensemble tickers. Ensemble tickers: {expected_ticker_names}, "
            f"Provided: {ticker_names}."
        )


def _build_feature_control_payload(
    *,
    feature_name: str,
    model_id: str,
    ticker_names: List[str],
    serializable_bias_spec: Dict[str, Any],
    strategy: str,
) -> Dict[str, Any]:
    timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "feature_name": feature_name,
        "created_at": timestamp,
        "updated_at": timestamp,
        "bias_node_spec": serializable_bias_spec,
        "tickers": ticker_names,
        "base_models": [
            {
                "model_id": model_id,
                "model_name": f"{feature_name}::{model_id}",
                "model_type": "signed_signal",
                "feature_column": feature_name,
                "strategy": strategy,
                "bias_node_spec": serializable_bias_spec,
            }
        ],
    }


def _validate_existing_feature_file(
    *,
    feature_file: Path,
    model_id: str,
    serializable_bias_spec: Dict[str, Any],
) -> Optional[str]:
    if not feature_file_exists(feature_file):
        return None

    existing_config = load_validated_feature_config(feature_file)
    existing_model_id = existing_config["base_models"][0]["model_id"]
    if existing_model_id != model_id:
        raise ValueError(
            f"Feature '{feature_file.stem}' already exists with model_id '{existing_model_id}' "
            f"which does not match the frozen spec-derived id '{model_id}'."
        )
    if existing_config["bias_node_spec"] != serializable_bias_spec:
        raise ValueError(
            f"Feature '{feature_file.stem}' already exists with a different bias-node spec."
        )
    return existing_model_id


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
    resolved_bias_node_spec = _validate_feature_add_request(
        base_model=base_model,
        bias_node_spec=bias_node_spec,
        bias_node_params=bias_node_params,
    )

    ensemble_dir = _require_ensemble_dir(ensemble_dir)

    ensemble_path = _resolve_ensemble_path(ensemble_dir)
    features_dir = _get_features_dir(ensemble_path)
    features_dir.mkdir(parents=True, exist_ok=True)

    ensemble_config = _load_ensemble_config(ensemble_path)
    ticker_names = _resolve_feature_ticker_names(resolved_bias_node_spec, tickers)
    _validate_feature_alignment(
        feature_name=feature_name,
        bias_node_spec=resolved_bias_node_spec,
        ensemble_config=ensemble_config,
        ticker_names=ticker_names,
    )

    model_id = generate_model_id("signed_signal", {}, bias_node_params=resolved_bias_node_spec)
    from ensemble.vault.feature_files import feature_json_stem

    feature_file = features_dir / f"{feature_json_stem(feature_name)}.json"
    serializable_bias_spec = _normalize_bias_node_spec_for_storage(resolved_bias_node_spec)
    if _validate_existing_feature_file(
        feature_file=feature_file,
        model_id=model_id,
        serializable_bias_spec=serializable_bias_spec,
    ) is not None:
        return model_id

    feature_config = _build_feature_control_payload(
        feature_name=feature_name,
        model_id=model_id,
        ticker_names=ticker_names,
        serializable_bias_spec=serializable_bias_spec,
        strategy=base_model.strategy.value,
    )
    validate_signed_signal_feature_config(feature_config, feature_file=feature_file)

    with open(win32_extended_path(feature_file), "w", encoding="utf-8") as f:
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
    Load canonical signed-signal base models for a feature.
    """
    from ensemble.ensemble_utils import create_base_model_from_config

    if feature_name is None and "feature_column" in legacy_kwargs:
        feature_name = legacy_kwargs["feature_column"]
    if feature_name is None:
        raise ValueError("feature_name must be provided")

    ensemble_dir = _require_ensemble_dir(ensemble_dir)

    ensemble_path = _resolve_ensemble_path(ensemble_dir)
    features_dir = _get_features_dir(ensemble_path)
    feature_file = resolve_feature_file_path(features_dir, feature_name)
    if feature_file is None or not feature_file_exists(feature_file):
        raise FileNotFoundError(
            f"Feature file not found for feature '{feature_name}' in {features_dir}"
        )

    feature_config = load_validated_feature_config(feature_file)
    resolved_feature_name = extract_feature_name(feature_config, feature_file.stem)
    feature_bias_spec = feature_config["bias_node_spec"]
    if tickers is None:
        tickers = [
            Ticker[ticker_name] if isinstance(ticker_name, str) else ticker_name
            for ticker_name in feature_config.get("tickers", [])
        ]

    model_entry = feature_config["base_models"][0]
    model_id = model_entry["model_id"]
    models: Dict[Tuple[Ticker, str], BaseModel] = {}

    for ticker in tickers:
        ticker_enum = ticker if isinstance(ticker, Ticker) else Ticker[str(ticker)]
        base_model_config = {
            "name": model_entry["model_name"],
            "feature_column": resolved_feature_name,
            "model_type": "signed_signal",
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
    raise ValueError("update_base_model_fitted_params is no longer supported for signed-signal features.")


def consolidate_feature_files(ensemble_dir: str) -> List[str]:
    ensemble_path = _resolve_ensemble_path(ensemble_dir)
    features_dir = _get_features_dir(ensemble_path)
    return consolidate_feature_files_in_dir(features_dir)


def migrate_legacy_feature_members_schema(ensemble_dir: str) -> List[str]:
    ensemble_path = _resolve_ensemble_path(ensemble_dir)
    features_dir = _get_features_dir(ensemble_path)
    return migrate_legacy_feature_members_schema_in_dir(features_dir)



def remove_base_model_variant(
    ensemble_dir: str,
    feature_name: str,
    model_id: str
) -> None:
    raise ValueError("remove_base_model_variant is not supported for signed-signal features.")


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
    ensemble_dir = _require_ensemble_dir(ensemble_dir, auto_detect=True)
    ensemble_path = _resolve_ensemble_path(ensemble_dir)
    features_dir = _get_features_dir(ensemble_path)
    
    if not features_dir.exists():
        return pd.DataFrame(columns=['feature_name', 'feature_column', 'n_base_models', 'n_fitted', 'created_at', 'updated_at'])

    return pd.DataFrame(build_feature_listing(features_dir))


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
    ensemble_dir = _require_ensemble_dir(ensemble_dir)
    ensemble_path = _resolve_ensemble_path(ensemble_dir)
    features_dir = _get_features_dir(ensemble_path)
    
    if not features_dir.exists():
        return []

    return collect_bias_node_specs(features_dir)


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
    ensemble_dir = _require_ensemble_dir(ensemble_dir)
    ensemble_path = _resolve_ensemble_path(ensemble_dir)
    features_dir = _get_features_dir(ensemble_path)
    
    if not features_dir.exists():
        return []

    return collect_base_model_names(features_dir)


def ensure_vault_cache_coverage(
    vault_ensemble_dirs: Sequence[str],
    start_date: Optional[datetime],
    end_date: datetime,
    refresh_mode: str = "missing_stale_only",
) -> Dict[str, Any]:
    """Ensure vault ensemble bias-node caches cover the requested window.

    This wrapper stays intentionally thin: it resolves ensemble paths, migrates
    legacy feature files, and delegates the refresh to ``CacheManager``.
    """
    from lib.cache.runtime.cache_manager import CacheManager

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
    The root is ``resolve_vault_root(vault_root)``.
    """
    vault_path = resolve_vault_root(vault_root)
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


def _ensemble_dir_posix_key_relative_to_repo(ensemble_path: Path) -> str:
    """Stable repo-relative key for ``exclude_feature_stems_by_ensemble`` (posix, no backslashes)."""
    repo_root = project_root().resolve()
    try:
        return ensemble_path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return ensemble_path.resolve().as_posix()


def _exclude_stems_for_ensemble_path(
    ensemble_path: Path,
    exclude_feature_stems_by_ensemble: Mapping[str, frozenset[str]] | None,
) -> frozenset[str]:
    if not exclude_feature_stems_by_ensemble:
        return frozenset()
    key = _ensemble_dir_posix_key_relative_to_repo(ensemble_path)
    for map_key, stems in exclude_feature_stems_by_ensemble.items():
        if Path(map_key).as_posix() == Path(key).as_posix():
            return stems
        if Path(map_key).as_posix() == Path(str(ensemble_path)).as_posix():
            return stems
    return frozenset()


def validate_ensemble_directory(ensemble_dir: str) -> None:
    """
    Validate an ensemble directory structure and all feature control files.
    
    Checks:
    - Directory exists and has features/ subdirectory
    - All feature control files use native signed-signal bias-node specs
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
    
    features_dir = _get_features_dir(ensemble_path, require_exists=True)
    
    # Load ensemble config if it exists
    ensemble_config = _load_ensemble_config(ensemble_path)
    expected_ticker_names = sorted(ensemble_config.get('tickers', []))
    validate_feature_configs_for_ensemble(
        features_dir,
        expected_ticker_names=expected_ticker_names,
    )


def load_ensemble_from_vault(
    ensemble_dir: str,
    refit: bool = False,
    target_volatility: float = 0.20,
    *,
    exclude_feature_stems_by_ensemble: Mapping[str, frozenset[str]] | None = None,
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
    exclude_feature_stems_by_ensemble : Mapping[str, frozenset[str]] | None
        Optional map keyed by repo-relative ensemble directory (posix path, e.g.
        ``vault/D/group/leaf``) to feature JSON **stems** (without ``.json``) to omit.
        Used for leave-one-feature-out ablations; must match
        ``collect_streams_by_group_for_ensemble_dirs`` exclusions.
        
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
    
    ensemble_path = _resolve_ensemble_path(ensemble_dir)
    
    if not ensemble_path.exists():
        raise ValueError(f"Ensemble directory does not exist: {ensemble_dir} (resolved to: {ensemble_path})")
    
    features_dir = _get_features_dir(ensemble_path, require_exists=True)

    exclude_stems = _exclude_stems_for_ensemble_path(
        ensemble_path, exclude_feature_stems_by_ensemble
    )
    validated_feature_configs = [
        (p, cfg)
        for p, cfg in iter_validated_feature_configs(features_dir)
        if p.stem not in exclude_stems
    ]
    if not validated_feature_configs:
        raise ValueError(f"No feature files found in {features_dir}")
    
    # Load ensemble config if available
    ensemble_config = _load_ensemble_config(ensemble_path)
    timeframe_str = ensemble_config.get('timeframe', 'D')
    
    base_models_config, model_identity_by_name, fallback_tickers = _build_ensemble_loader_payload(
        validated_feature_configs
    )
    
    if not base_models_config:
        raise ValueError(f"No base models found in ensemble directory: {ensemble_dir}")
    
    # Get tickers (prefer ensemble config, then feature configs)
    tickers = sorted(ensemble_config.get('tickers', fallback_tickers))
    if not tickers:
        raise ValueError(f"Could not determine tickers for ensemble: {ensemble_dir}")

    _, first_feature_config = validated_feature_configs[0]
    metadata = _build_vault_loader_metadata(
        first_feature_config=first_feature_config,
        timeframe_str=timeframe_str,
        refit=refit,
        selection_method=str(ensemble_config.get("selection_method", "manual")),
    )
    model_names = [str(base_model["name"]) for base_model in base_models_config]
    fitted_ensemble = None
    if not refit:
        fitted_ensemble = _build_vault_loader_fitted_ensemble(
            model_names=model_names,
            tickers=tickers,
            target_volatility=target_volatility,
        )

    temp_path = _write_temp_vault_control_file(
        base_models_config=base_models_config,
        metadata=metadata,
        fitted_ensemble=fitted_ensemble,
        tickers=tickers,
    )
    try:
        ensemble = DiversifiedEnsemble(
            control_file_path=temp_path,
            target_volatility=target_volatility,
            base_tf=TimeFrame[timeframe_str]
        )
    finally:
        os.unlink(temp_path)

    ensemble.vault_ensemble_dir = str(ensemble_path)
    ensemble.vault_ensemble_name = ensemble_path.name
    ensemble.vault_timeframe = timeframe_str
    ensemble.vault_base_model_identities = model_identity_by_name
    return ensemble
