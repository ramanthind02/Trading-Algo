from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional, Union

import pandas as pd

from .cache_paths import (
    default_live_materialized_cache_dir,
    default_research_materialized_cache_dir,
    resolve_relative_path,
    win32_extended_path,
)
from .central_cache_models import ArtifactScope
from utils.core.enums import TimeFrame

logger = logging.getLogger(__name__)


def _read_utf8_text(path: Path) -> str:
    """Read file as UTF-8. On Windows, long vault paths need ``\\\\?\\`` to open reliably."""
    if os.name == "nt":
        with open(win32_extended_path(path), encoding="utf-8") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8")


if TYPE_CHECKING:
    from ensemble.portfolio import GlobalPortfolio, PortfolioCacheQuery, PortfolioWorld


@dataclass(frozen=True)
class BaseModelMaterializationIdentity:
    timeframe: str
    ensemble_name: str
    feature_name: str
    model_id: str

    def as_dict(self) -> dict[str, str]:
        return {
            "timeframe": self.timeframe,
            "ensemble_name": self.ensemble_name,
            "feature_name": self.feature_name,
            "model_id": self.model_id,
        }


@dataclass(frozen=True)
class CleanupSummary:
    scope: ArtifactScope
    active_identities: int
    deleted_files: int
    kept_files: int
    deleted_paths: tuple[str, ...]


@dataclass(frozen=True)
class MaterializationSummary:
    portfolio_id: str
    world: str
    scope: ArtifactScope
    portfolio_rows_written: int
    portfolio_path: str
    base_model_files_written: int
    base_model_rows_written: int
    base_model_rows_skipped_inactive: int
    cleanup: CleanupSummary


_PORTFOLIO_ROW_COLUMNS = [
    "portfolio_id",
    "world",
    "research_run_id",
    "ticker",
    "datetime",
    "forecast_score",
    "position_fraction",
]
_BASE_MODEL_ROW_COLUMNS = [
    "timeframe",
    "ensemble_name",
    "feature_name",
    "model_id",
    "portfolio_id",
    "world",
    "research_run_id",
    "ticker",
    "datetime",
    "forecast_score",
    "position_fraction",
]


def _materialized_root(
    scope: ArtifactScope,
    cache_root: Optional[str] = None,
) -> Path:
    if cache_root is not None:
        root = Path(cache_root) / "materialized" / scope.value
        root.mkdir(parents=True, exist_ok=True)
        return root
    if scope is ArtifactScope.RESEARCH:
        root = default_research_materialized_cache_dir()
    else:
        root = default_live_materialized_cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _portfolio_materialization_path(
    portfolio_id: str,
    scope: ArtifactScope,
    cache_root: Optional[str] = None,
) -> Path:
    return _materialized_root(scope, cache_root=cache_root) / "portfolio" / f"{portfolio_id}.parquet"


def _identity_hash(identity: BaseModelMaterializationIdentity) -> str:
    """Stable short id for on-disk filenames (avoids Windows path-length limits)."""
    payload = (
        f"{identity.timeframe}\0{identity.ensemble_name}\0"
        f"{identity.feature_name}\0{identity.model_id}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_materialization_hash_stem(stem: str) -> bool:
    return len(stem) == 64 and all(c in "0123456789abcdef" for c in stem.lower())


def _base_model_materialization_path(
    identity: BaseModelMaterializationIdentity,
    scope: ArtifactScope,
    cache_root: Optional[str] = None,
) -> Path:
    return (
        _materialized_root(scope, cache_root=cache_root)
        / "base_models"
        / identity.timeframe
        / identity.ensemble_name
        / f"{_identity_hash(identity)}.parquet"
    )


def _unlink_legacy_duplicate_for_identity(
    identity: BaseModelMaterializationIdentity,
    scope: ArtifactScope,
    cache_root: Optional[str] = None,
) -> None:
    """Remove pre-hash ``feature__model.parquet`` files after migrating to hashed names."""
    base_models_root = _materialized_root(scope, cache_root=cache_root) / "base_models"
    ensemble_dir = base_models_root / identity.timeframe / identity.ensemble_name
    if not ensemble_dir.is_dir():
        return
    for path in ensemble_dir.glob("*.parquet"):
        stem = path.stem
        if _is_materialization_hash_stem(stem):
            continue
        if "__" not in stem:
            continue
        parsed = _parse_base_model_materialization_path(path, base_models_root)
        if parsed == identity:
            _unlink_materialized_file(path)


def _normalize_datetime_column(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["datetime"] = pd.to_datetime(normalized["datetime"])
    datetime_accessor = normalized["datetime"].dt
    if getattr(datetime_accessor, "tz", None) is not None:
        normalized["datetime"] = datetime_accessor.tz_convert(None)
    return normalized


def _normalize_portfolio_frame(
    portfolio_id: str,
    world: str,
    research_run_id: Optional[str],
    frame: pd.DataFrame,
) -> pd.DataFrame:
    normalized = frame.copy()
    expected_columns = {"ticker", "datetime", "forecast_score", "position_fraction"}
    missing = expected_columns - set(normalized.columns)
    if missing:
        raise ValueError(
            "Portfolio materialization requires columns "
            f"{sorted(expected_columns)}; missing {sorted(missing)}"
        )
    normalized = normalized.loc[:, ["ticker", "datetime", "forecast_score", "position_fraction"]]
    normalized["portfolio_id"] = portfolio_id
    normalized["world"] = world
    normalized["research_run_id"] = research_run_id
    normalized = _normalize_datetime_column(normalized)
    return normalized.loc[:, _PORTFOLIO_ROW_COLUMNS]


def _normalize_base_model_frame(
    identity: BaseModelMaterializationIdentity,
    portfolio_id: str,
    world: str,
    research_run_id: Optional[str],
    frame: pd.DataFrame,
) -> pd.DataFrame:
    normalized = frame.copy()
    expected_columns = {"ticker", "datetime", "forecast_score", "position_fraction"}
    missing = expected_columns - set(normalized.columns)
    if missing:
        raise ValueError(
            "Base-model materialization requires columns "
            f"{sorted(expected_columns)}; missing {sorted(missing)}"
        )
    normalized = normalized.loc[:, ["ticker", "datetime", "forecast_score", "position_fraction"]]
    normalized["timeframe"] = identity.timeframe
    normalized["ensemble_name"] = identity.ensemble_name
    normalized["feature_name"] = identity.feature_name
    normalized["model_id"] = identity.model_id
    normalized["portfolio_id"] = portfolio_id
    normalized["world"] = world
    normalized["research_run_id"] = research_run_id
    normalized = _normalize_datetime_column(normalized)
    return normalized.loc[:, _BASE_MODEL_ROW_COLUMNS]


def _deduplicate_nullable_rows(
    frame: pd.DataFrame,
    subset: list[str],
) -> pd.DataFrame:
    working = frame.copy()
    if "research_run_id" in subset:
        working["_research_run_id_dedup"] = working["research_run_id"].astype("string").fillna("__NULL__")
        subset = [
            "_research_run_id_dedup" if column == "research_run_id" else column
            for column in subset
        ]
    deduped = working.drop_duplicates(subset=subset, keep="last")
    if "_research_run_id_dedup" in deduped.columns:
        deduped = deduped.drop(columns="_research_run_id_dedup")
    return deduped


def _parquet_path_for_io(path: Path) -> Union[Path, str]:
    """Windows: use ``\\\\?\\`` paths so materialized filenames can exceed ``MAX_PATH``."""
    return win32_extended_path(path) if os.name == "nt" else path


def _upsert_frame(
    path: Path,
    frame: pd.DataFrame,
    dedup_subset: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    io_path = _parquet_path_for_io(path)
    combined = frame.copy()
    if Path(io_path).exists():
        existing = pd.read_parquet(io_path)
        combined = pd.concat([existing, frame], ignore_index=True)
    combined = _deduplicate_nullable_rows(combined, dedup_subset)
    combined = combined.sort_values("datetime").reset_index(drop=True)
    combined.to_parquet(io_path, index=False)


def _unlink_materialized_file(path: Path) -> None:
    if os.name == "nt":
        os.remove(win32_extended_path(path))
    else:
        path.unlink()


def _remove_empty_parent_dirs(path: Path, stop_at: Path) -> None:
    current = path.parent
    while current != stop_at and current.exists():
        if any(current.iterdir()):
            break
        current.rmdir()
        current = current.parent


def _feature_name_from_payload(feature_path: Path, payload: dict[str, Any]) -> str:
    return str(
        payload.get("feature_name")
        or payload.get("feature_column")
        or feature_path.stem
    )


def _scan_active_live_base_model_identities(
    vault_root: str,
    ensemble_dirs: Optional[Iterable[str]] = None,
) -> set[BaseModelMaterializationIdentity]:
    from ensemble.vault_manager import _resolve_ensemble_path

    resolved_vault_root = resolve_relative_path(
        vault_root,
        prefer_existing_candidate=True,
        include_project_root=True,
        project_root_fallback="candidate",
    )
    target_ensemble_dirs: list[Path]
    if ensemble_dirs is not None:
        target_ensemble_dirs = [
            _resolve_ensemble_path(str(ensemble_dir))
            for ensemble_dir in dict.fromkeys(str(item) for item in ensemble_dirs)
        ]
    else:
        target_ensemble_dirs = [
            ensemble_dir
            for tf_name in ("D", "W", "M")
            for ensemble_dir in sorted((resolved_vault_root / tf_name).glob("*"))
            if ensemble_dir.is_dir()
        ]

    identities: set[BaseModelMaterializationIdentity] = set()
    for ensemble_dir in target_ensemble_dirs:
        if not ensemble_dir.exists():
            continue
        features_dir = ensemble_dir / "features"
        if not features_dir.exists():
            continue
        timeframe = ensemble_dir.parent.name
        ensemble_name = ensemble_dir.name
        for feature_path in sorted(features_dir.glob("*.json")):
            payload = json.loads(_read_utf8_text(feature_path))
            feature_name = _feature_name_from_payload(feature_path, payload)
            for model_payload in payload.get("base_models", []):
                model_id = str(model_payload.get("model_id"))
                identities.add(
                    BaseModelMaterializationIdentity(
                        timeframe=timeframe,
                        ensemble_name=ensemble_name,
                        feature_name=feature_name,
                        model_id=model_id,
                    )
                )
    return identities


def _parse_base_model_materialization_path(
    path: Path,
    root: Path,
) -> Optional[BaseModelMaterializationIdentity]:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    if len(relative.parts) != 3:
        return None
    timeframe, ensemble_name, filename = relative.parts
    if not filename.endswith(".parquet") or "__" not in filename:
        return None
    feature_name, model_id = filename[:-8].rsplit("__", 1)
    return BaseModelMaterializationIdentity(
        timeframe=timeframe,
        ensemble_name=ensemble_name,
        feature_name=feature_name,
        model_id=model_id,
    )


def _base_model_identity_from_key(
    portfolio: "GlobalPortfolio",
    tf_portfolio_index: int,
    tf_portfolio: Any,
    key: str,
) -> BaseModelMaterializationIdentity:
    if "::" not in key:
        raise ValueError(f"Invalid TF base-model key '{key}'")
    ensemble_token, model_name = key.split("::", 1)
    if not ensemble_token.startswith("ensemble_"):
        raise ValueError(f"Invalid TF base-model key '{key}'")
    ensemble_idx = int(ensemble_token.removeprefix("ensemble_"))
    if ensemble_idx >= len(tf_portfolio.ensembles):
        raise ValueError(
            f"Base-model key '{key}' references ensemble index {ensemble_idx}, "
            f"but TF portfolio {tf_portfolio_index} only has {len(tf_portfolio.ensembles)} ensembles"
        )
    ensemble = tf_portfolio.ensembles[ensemble_idx]
    timeframe = (
        getattr(ensemble, "vault_timeframe", None)
        or getattr(getattr(ensemble, "base_tf", None), "name", None)
        or getattr(tf_portfolio.trading_timeframe, "name", None)
    )
    ensemble_name = getattr(ensemble, "vault_ensemble_name", None)
    if ensemble_name is None:
        ensemble_dir = getattr(ensemble, "vault_ensemble_dir", None)
        if ensemble_dir is not None:
            ensemble_name = Path(str(ensemble_dir)).name
    if ensemble_name is None or timeframe is None:
        raise ValueError(
            "Base-model materialization requires vault-loaded ensembles with "
            "'vault_ensemble_name' and 'vault_timeframe' metadata."
        )

    identity_map = getattr(ensemble, "vault_base_model_identities", {})
    identity_payload = identity_map.get(model_name)
    if isinstance(identity_payload, dict):
        feature_name = str(identity_payload["feature_name"])
        model_id = str(identity_payload["model_id"])
    elif "::" in model_name:
        feature_name, model_id = model_name.rsplit("::", 1)
    else:
        raise ValueError(
            f"Unable to infer feature/model identity from '{model_name}' "
            f"for ensemble '{ensemble_name}'"
        )

    return BaseModelMaterializationIdentity(
        timeframe=str(timeframe),
        ensemble_name=str(ensemble_name),
        feature_name=feature_name,
        model_id=model_id,
    )


def _collect_global_base_model_frames(
    portfolio: "GlobalPortfolio",
    query: "PortfolioCacheQuery",
) -> list[tuple[BaseModelMaterializationIdentity, pd.DataFrame]]:
    frames: list[tuple[BaseModelMaterializationIdentity, pd.DataFrame]] = []
    for tf_index, tf_portfolio in enumerate(portfolio.tf_portfolios):
        result = tf_portfolio.predict_from_cache(
            query.for_timeframe(tf_portfolio.trading_timeframe),
            return_base_model_predictions=True,
        )
        if not isinstance(result, dict):
            continue
        base_models = result.get("base_models", {})
        for key, frame in base_models.items():
            identity = _base_model_identity_from_key(
                portfolio=portfolio,
                tf_portfolio_index=tf_index,
                tf_portfolio=tf_portfolio,
                key=str(key),
            )
            if isinstance(frame, pd.DataFrame):
                frames.append((identity, frame.copy()))
    return frames


def prune_inactive_base_model_materializations(
    vault_root: str = "vault",
    scope: ArtifactScope = ArtifactScope.LIVE,
    ensemble_dirs: Optional[Iterable[str]] = None,
    cache_root: Optional[str] = None,
) -> CleanupSummary:
    active_identities = _scan_active_live_base_model_identities(
        vault_root=vault_root,
        ensemble_dirs=ensemble_dirs,
    )
    base_models_root = _materialized_root(scope, cache_root=cache_root) / "base_models"
    if not base_models_root.exists():
        return CleanupSummary(
            scope=scope,
            active_identities=len(active_identities),
            deleted_files=0,
            kept_files=0,
            deleted_paths=(),
        )

    deleted_paths: list[str] = []
    kept_files = 0
    active_hashes = {_identity_hash(i) for i in active_identities}
    for path in sorted(base_models_root.rglob("*.parquet")):
        stem = path.stem
        if _is_materialization_hash_stem(stem):
            if stem in active_hashes:
                kept_files += 1
            else:
                _unlink_materialized_file(path)
                deleted_paths.append(str(path))
                _remove_empty_parent_dirs(path, stop_at=base_models_root)
            continue
        identity = _parse_base_model_materialization_path(path, base_models_root)
        if identity is None:
            kept_files += 1
            continue
        if identity in active_identities:
            kept_files += 1
            continue
        _unlink_materialized_file(path)
        deleted_paths.append(str(path))
        _remove_empty_parent_dirs(path, stop_at=base_models_root)

    return CleanupSummary(
        scope=scope,
        active_identities=len(active_identities),
        deleted_files=len(deleted_paths),
        kept_files=kept_files,
        deleted_paths=tuple(deleted_paths),
    )


def materialize_global_portfolio_predictions(
    portfolio: "GlobalPortfolio",
    query: "PortfolioCacheQuery",
    portfolio_id: str,
    world: "PortfolioWorld | str",
    research_run_id: Optional[str] = None,
    scope: ArtifactScope = ArtifactScope.LIVE,
    vault_root: str = "vault",
    cache_root: Optional[str] = None,
) -> MaterializationSummary:
    from ensemble.portfolio import PortfolioWorld

    resolved_world = world if isinstance(world, PortfolioWorld) else PortfolioWorld(str(world))
    portfolio_frame = portfolio.predict_from_cache(query)
    if not isinstance(portfolio_frame, pd.DataFrame):
        raise ValueError("Global portfolio materialization expects DataFrame portfolio output")

    normalized_portfolio_frame = _normalize_portfolio_frame(
        portfolio_id=portfolio_id,
        world=resolved_world.value,
        research_run_id=research_run_id,
        frame=portfolio_frame,
    )
    portfolio_path = _portfolio_materialization_path(
        portfolio_id=portfolio_id,
        scope=scope,
        cache_root=cache_root,
    )
    _upsert_frame(
        portfolio_path,
        normalized_portfolio_frame,
        dedup_subset=["portfolio_id", "world", "research_run_id", "ticker", "datetime"],
    )

    active_identities = _scan_active_live_base_model_identities(vault_root=vault_root)
    base_model_files_written = 0
    base_model_rows_written = 0
    base_model_rows_skipped_inactive = 0
    for identity, frame in _collect_global_base_model_frames(portfolio, query):
        if identity not in active_identities:
            base_model_rows_skipped_inactive += len(frame)
            continue
        normalized_base_model_frame = _normalize_base_model_frame(
            identity=identity,
            portfolio_id=portfolio_id,
            world=resolved_world.value,
            research_run_id=research_run_id,
            frame=frame,
        )
        base_model_path = _base_model_materialization_path(
            identity=identity,
            scope=scope,
            cache_root=cache_root,
        )
        _upsert_frame(
            base_model_path,
            normalized_base_model_frame,
            dedup_subset=[
                "timeframe",
                "ensemble_name",
                "feature_name",
                "model_id",
                "portfolio_id",
                "world",
                "research_run_id",
                "ticker",
                "datetime",
            ],
        )
        _unlink_legacy_duplicate_for_identity(identity, scope=scope, cache_root=cache_root)
        base_model_files_written += 1
        base_model_rows_written += len(normalized_base_model_frame)

    cleanup = prune_inactive_base_model_materializations(
        vault_root=vault_root,
        scope=scope,
        cache_root=cache_root,
    )
    return MaterializationSummary(
        portfolio_id=portfolio_id,
        world=resolved_world.value,
        scope=scope,
        portfolio_rows_written=len(normalized_portfolio_frame),
        portfolio_path=str(portfolio_path),
        base_model_files_written=base_model_files_written,
        base_model_rows_written=base_model_rows_written,
        base_model_rows_skipped_inactive=base_model_rows_skipped_inactive,
        cleanup=cleanup,
    )


__all__ = [
    "BaseModelMaterializationIdentity",
    "CleanupSummary",
    "MaterializationSummary",
    "materialize_global_portfolio_predictions",
    "prune_inactive_base_model_materializations",
]
