from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from ensemble.portfolio import GlobalPortfolio, TFPortfolio
from ensemble.vault_manager import load_ensemble_from_vault
from ensemble.weight_layer import deserialize_weight_layer_state, serialize_weight_layer_state
from utils.cache.runtime.cache_paths import win32_extended_path
from utils.core.enums import TimeFrame
from utils.vault_paths import resolve_vault_root

_SNAPSHOT_SCHEMA_VERSION = "1.0"


def _resolve_vault_root(vault_root: str) -> Path:
    return resolve_vault_root(vault_root)


def _snapshot_root(vault_root: str) -> Path:
    return _resolve_vault_root(vault_root) / "portfolio_snapshots"


def _snapshot_dir(vault_root: str, portfolio_id: str) -> Path:
    return _snapshot_root(vault_root) / portfolio_id


def _snapshot_file(vault_root: str, portfolio_id: str) -> Path:
    return _snapshot_dir(vault_root, portfolio_id) / "snapshot.json"


def _json_default(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_default(v) for k, v in value.items()}
    if isinstance(value, set):
        return sorted(_json_default(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [_json_default(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _canonical_json(payload: Dict[str, Any]) -> str:
    return json.dumps(_json_default(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _portfolio_id_for_payload(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _iso_datetime(value: datetime) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.isoformat()


def _sha256_file(path: Path) -> str:
    # Windows: long vault feature filenames can exceed MAX_PATH unless opened via \\?\ prefix.
    if os.name == "nt":
        with open(win32_extended_path(path), "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _serialize_diversified_ensemble_state(ensemble: Any) -> Dict[str, Any]:
    return {
        "target_volatility": float(ensemble.target_volatility),
        "target_volatility_fitted": (
            None if ensemble.target_volatility_ is None else float(ensemble.target_volatility_)
        ),
        "weights": None if ensemble.weights_ is None else {str(k): float(v) for k, v in ensemble.weights_.items()},
        "exposure_fractions": (
            None
            if ensemble.exposure_fractions_ is None
            else {str(k): float(v) for k, v in ensemble.exposure_fractions_.items()}
        ),
        "model_exposure_fractions": (
            None
            if ensemble.model_exposure_fractions_ is None
            else {str(k): float(v) for k, v in ensemble.model_exposure_fractions_.items()}
        ),
        "feature_names": (
            None
            if ensemble.feature_names_ is None
            else [str(feature_name) for feature_name in ensemble.feature_names_]
        ),
        "unique_tickers": (
            None
            if ensemble.unique_tickers_ is None
            else [str(ticker) for ticker in ensemble.unique_tickers_]
        ),
        "instrument_weights": (
            None
            if ensemble.instrument_weights_ is None
            else {str(k): float(v) for k, v in ensemble.instrument_weights_.items()}
        ),
        "n_tickers": None if ensemble.n_tickers_ is None else int(ensemble.n_tickers_),
        "is_fitted": bool(ensemble.is_fitted_),
        "use_cache": bool(getattr(ensemble, "use_cache", True)),
        "retry_on_cache_miss": bool(getattr(ensemble, "retry_on_cache_miss", True)),
        "base_tf": getattr(getattr(ensemble, "base_tf", None), "name", None),
    }


def _restore_diversified_ensemble_state(ensemble: Any, payload: Dict[str, Any]) -> None:
    ensemble.target_volatility = float(payload.get("target_volatility", ensemble.target_volatility))
    ensemble.target_volatility_ = payload.get("target_volatility_fitted")
    ensemble.weights_ = payload.get("weights")
    ensemble.exposure_fractions_ = payload.get("exposure_fractions")
    ensemble.model_exposure_fractions_ = payload.get("model_exposure_fractions")
    ensemble.feature_names_ = payload.get("feature_names")
    ensemble.unique_tickers_ = payload.get("unique_tickers")
    ensemble.instrument_weights_ = payload.get("instrument_weights")
    ensemble.n_tickers_ = payload.get("n_tickers")
    ensemble.is_fitted_ = bool(payload.get("is_fitted", False))
    ensemble.use_cache = bool(payload.get("use_cache", True))
    ensemble.retry_on_cache_miss = bool(payload.get("retry_on_cache_miss", True))
    base_tf_name = payload.get("base_tf")
    if base_tf_name is not None:
        ensemble.base_tf = TimeFrame[str(base_tf_name)]


def _serialize_tf_portfolio(tf_portfolio: TFPortfolio) -> Dict[str, Any]:
    ensembles_payload: list[Dict[str, Any]] = []
    for ensemble in tf_portfolio.ensembles:
        source_ensemble_dir = getattr(ensemble, "vault_ensemble_dir", None)
        if source_ensemble_dir is None:
            raise ValueError(
                "GlobalPortfolio.save_to_vault() requires vault-loaded ensembles with "
                "'vault_ensemble_dir' metadata."
            )
        source_path = Path(str(source_ensemble_dir))
        timeframe_name = source_path.parent.name
        ensemble_name = source_path.name
        features_dir = source_path / "features"
        copied_file_hashes = {
            "ensemble_config.json": _sha256_file(source_path / "ensemble_config.json"),
            **{
                f"features/{feature_path.name}": _sha256_file(feature_path)
                for feature_path in sorted(features_dir.glob("*.json"))
            },
        }
        ensembles_payload.append(
            {
                "timeframe": timeframe_name,
                "ensemble_name": ensemble_name,
                "source_ensemble_dir": str(source_path),
                "snapshot_ensemble_relpath": f"ensembles/{timeframe_name}/{ensemble_name}",
                "copied_file_hashes": copied_file_hashes,
                "fitted_state": _serialize_diversified_ensemble_state(ensemble),
            }
        )

    return {
        "config": {
            "trading_timeframe": tf_portfolio.trading_timeframe.name,
            "target_volatility": tf_portfolio.target_volatility,
            "max_position_pct": tf_portfolio.max_position_pct,
            "instrument_weights": tf_portfolio.instrument_weights,
            "idm_max": tf_portfolio.idm_max,
            "use_cache": tf_portfolio.use_cache,
            "sector_allocation_config_path": tf_portfolio.sector_allocation_config_path,
        },
        "fitted_state": {
            "idm": tf_portfolio.idm_,
            "mean_return_correlation": tf_portfolio.mean_return_correlation_,
            "instruments": tf_portfolio.instruments_,
            "is_fitted": tf_portfolio.is_fitted_,
        },
        "ensembles": ensembles_payload,
    }


def _serialize_global_portfolio(portfolio: GlobalPortfolio) -> Dict[str, Any]:
    return {
        "config": {
            "idm_max": portfolio.idm_max,
            "max_position_pct": portfolio.max_position_pct,
            "instrument_weights": portfolio.instrument_weights,
        },
        "fitted_state": {
            "global_idm": portfolio.global_idm_,
            "mean_instrument_return_correlation": portfolio.mean_instrument_return_correlation_,
            "instruments": portfolio.instruments_,
            "global_tf_weights_compat": portfolio.global_tf_weights_compat_,
            "weight_layer_diagnostics": portfolio.weight_layer_diagnostics_,
            "global_adapter_diagnostics": portfolio.global_adapter_diagnostics_,
            "global_eligible_models_by_ticker": {
                str(ticker): sorted(str(model_name) for model_name in model_names)
                for ticker, model_names in portfolio.global_eligible_models_by_ticker_.items()
            },
            "global_eligibility_diagnostics": portfolio.global_eligibility_diagnostics_,
            "is_fitted": portfolio.is_fitted_,
        },
        "weight_layer": serialize_weight_layer_state(portfolio.weight_layer),
        "tf_portfolios": [_serialize_tf_portfolio(tf_portfolio) for tf_portfolio in portfolio.tf_portfolios],
    }


def _strip_source_paths(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = json.loads(_canonical_json(payload))
    for tf_payload in normalized.get("global_portfolio", {}).get("tf_portfolios", []):
        for ensemble_payload in tf_payload.get("ensembles", []):
            ensemble_payload.pop("source_ensemble_dir", None)
    return normalized


def _copy_ensemble_files(snapshot_root: Path, tf_payload: Dict[str, Any]) -> None:
    for ensemble_payload in tf_payload["ensembles"]:
        source_dir = Path(str(ensemble_payload["source_ensemble_dir"]))
        destination_dir = snapshot_root / str(ensemble_payload["snapshot_ensemble_relpath"])
        destination_dir.mkdir(parents=True, exist_ok=True)
        source_config = source_dir / "ensemble_config.json"
        if source_config.exists():
            dest_cfg = destination_dir / "ensemble_config.json"
            if os.name == "nt":
                shutil.copy2(win32_extended_path(source_config), win32_extended_path(dest_cfg))
            else:
                shutil.copy2(source_config, dest_cfg)
        destination_features_dir = destination_dir / "features"
        destination_features_dir.mkdir(parents=True, exist_ok=True)
        for feature_path in sorted((source_dir / "features").glob("*.json")):
            dest_feature = destination_features_dir / feature_path.name
            if os.name == "nt":
                shutil.copy2(win32_extended_path(feature_path), win32_extended_path(dest_feature))
            else:
                shutil.copy2(feature_path, dest_feature)


def save_global_portfolio_snapshot(
    portfolio: GlobalPortfolio,
    fit_start: datetime,
    fit_end: datetime,
    vault_root: str = "vault",
) -> str:
    if not portfolio.is_fitted_:
        raise ValueError("GlobalPortfolio must be fitted before save_to_vault()")

    snapshot_payload = {
        "schema_version": _SNAPSHOT_SCHEMA_VERSION,
        "fit_window": {
            "start": _iso_datetime(fit_start),
            "end": _iso_datetime(fit_end),
        },
        "global_portfolio": _serialize_global_portfolio(portfolio),
    }
    semantic_payload = _strip_source_paths(snapshot_payload)
    portfolio_id = _portfolio_id_for_payload(semantic_payload)
    snapshot_root = _snapshot_dir(vault_root, portfolio_id)
    snapshot_file = snapshot_root / "snapshot.json"

    if snapshot_file.exists():
        portfolio.portfolio_id_ = portfolio_id
        return portfolio_id

    snapshot_root.mkdir(parents=True, exist_ok=True)
    for tf_payload in snapshot_payload["global_portfolio"]["tf_portfolios"]:
        _copy_ensemble_files(snapshot_root, tf_payload)

    snapshot_document = {
        **snapshot_payload,
        "portfolio_id": portfolio_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    snapshot_file.write_text(
        json.dumps(_json_default(snapshot_document), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    portfolio.portfolio_id_ = portfolio_id
    return portfolio_id


def _restore_tf_portfolio(tf_payload: Dict[str, Any], snapshot_root: Path) -> TFPortfolio:
    config = dict(tf_payload["config"])
    restored_ensembles = []
    for ensemble_payload in tf_payload["ensembles"]:
        ensemble_dir = snapshot_root / str(ensemble_payload["snapshot_ensemble_relpath"])
        fitted_state = dict(ensemble_payload["fitted_state"])
        restored = load_ensemble_from_vault(
            str(ensemble_dir),
            refit=False,
            target_volatility=float(fitted_state.get("target_volatility", 0.15) or 0.15),
        )
        _restore_diversified_ensemble_state(restored, fitted_state)
        restored_ensembles.append(restored)

    tf_portfolio = TFPortfolio(
        ensembles=restored_ensembles,
        trading_timeframe=TimeFrame[str(config["trading_timeframe"])],
        target_volatility=config.get("target_volatility"),
        max_position_pct=float(config["max_position_pct"]),
        instrument_weights=config.get("instrument_weights"),
        idm_max=float(config["idm_max"]),
        use_cache=bool(config.get("use_cache", True)),
    )
    fitted_state = dict(tf_payload["fitted_state"])
    tf_portfolio.idm_ = fitted_state.get("idm")
    tf_portfolio.mean_return_correlation_ = fitted_state.get("mean_return_correlation")
    tf_portfolio.instruments_ = fitted_state.get("instruments")
    tf_portfolio.is_fitted_ = bool(fitted_state.get("is_fitted", False))
    tf_portfolio.sector_allocation_config_path = config.get("sector_allocation_config_path")
    return tf_portfolio


def load_global_portfolio_snapshot(
    portfolio_id: str,
    vault_root: str = "vault",
) -> GlobalPortfolio:
    snapshot_path = _snapshot_file(vault_root, portfolio_id)
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Portfolio snapshot not found: {snapshot_path}")

    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != _SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported portfolio snapshot schema: {payload.get('schema_version')}"
        )

    snapshot_root = snapshot_path.parent
    global_payload = dict(payload["global_portfolio"])
    weight_layer = deserialize_weight_layer_state(global_payload["weight_layer"])
    tf_portfolios = [
        _restore_tf_portfolio(tf_payload, snapshot_root=snapshot_root)
        for tf_payload in global_payload["tf_portfolios"]
    ]
    config = dict(global_payload["config"])
    portfolio = GlobalPortfolio(
        tf_portfolios=tf_portfolios,
        weight_layer=weight_layer,
        instrument_weights=config.get("instrument_weights"),
        idm_max=float(config["idm_max"]),
        max_position_pct=float(config["max_position_pct"]),
    )
    fitted_state = dict(global_payload["fitted_state"])
    portfolio.global_idm_ = fitted_state.get("global_idm")
    portfolio.mean_instrument_return_correlation_ = fitted_state.get("mean_instrument_return_correlation")
    portfolio.instruments_ = fitted_state.get("instruments")
    portfolio.global_tf_weights_compat_ = dict(fitted_state.get("global_tf_weights_compat", {}))
    portfolio.weight_layer_diagnostics_ = dict(fitted_state.get("weight_layer_diagnostics", {}))
    portfolio.global_adapter_diagnostics_ = dict(fitted_state.get("global_adapter_diagnostics", {}))
    portfolio.global_eligible_models_by_ticker_ = {
        str(ticker): set(models)
        for ticker, models in dict(fitted_state.get("global_eligible_models_by_ticker", {})).items()
    }
    portfolio.global_eligibility_diagnostics_ = dict(
        fitted_state.get("global_eligibility_diagnostics", {})
    )
    portfolio.is_fitted_ = bool(fitted_state.get("is_fitted", False))
    portfolio.portfolio_id_ = portfolio_id
    return portfolio


__all__ = [
    "load_global_portfolio_snapshot",
    "save_global_portfolio_snapshot",
]
