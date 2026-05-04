"""Build ``hierarchy_equal`` JSON specs from vault feature tags and model names."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Dict, FrozenSet, List, Tuple

from ensemble.portfolio_impl.global_weight_layer_adapter import (
    build_global_model_name,
    build_global_stream_id,
)
from ensemble.vault.constants import VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES
from ensemble.vault.discovery import iter_vault_feature_members
from ensemble.vault.feature_files import load_validated_feature_config
from utils.core.enums import TimeFrame
from utils.vault_paths import vault_root_from_repo_relative_ensemble

logger = logging.getLogger(__name__)

_VAULT_TF_NAMES = frozenset({"D", "W", "M"})


def _exclude_stems_for_ensemble_rel(
    root: Path,
    ensemble_path_rel: str,
    exclude_feature_stems_by_ensemble: Mapping[str, frozenset[str]] | None,
) -> frozenset[str]:
    """Feature JSON stems to skip for this ensemble (aligned with ``load_ensemble_from_vault``)."""
    if not exclude_feature_stems_by_ensemble:
        return frozenset()
    abs_ens = (root / ensemble_path_rel).resolve()
    try:
        key = abs_ens.relative_to(root.resolve()).as_posix()
    except ValueError:
        key = abs_ens.as_posix()
    for map_key, stems in exclude_feature_stems_by_ensemble.items():
        if Path(map_key).as_posix() == key:
            return stems
    return frozenset()


def _read_ensemble_trading_timeframe(ensemble_path: Path) -> TimeFrame:
    """Match ``load_ensemble_from_vault`` / ``DiversifiedEnsemble.base_tf``."""
    cfg_path = ensemble_path / "ensemble_config.json"
    if not cfg_path.is_file():
        raise ValueError(f"Missing ensemble_config.json under {ensemble_path}")
    with cfg_path.open(encoding="utf-8") as handle:
        cfg = json.load(handle)
    raw = cfg.get("timeframe", "D")
    tf_str = raw.name if hasattr(raw, "name") else str(raw).strip()
    return TimeFrame[tf_str]


def _assign_ensemble_indices_for_repo_relative_paths(
    ordered_repo_relative_paths: Iterable[str],
    repo_root: Path,
) -> dict[str, tuple[TimeFrame, int]]:
    """``ensemble_dirs`` values are repo-relative (e.g. ``vault/M/buy_hold/buy_hold_long``)."""
    next_idx: dict[TimeFrame, int] = {}
    out: dict[str, tuple[TimeFrame, int]] = {}
    for rel in ordered_repo_relative_paths:
        ens = (repo_root / rel).resolve()
        tf = _read_ensemble_trading_timeframe(ens)
        idx = next_idx.get(tf, 0)
        next_idx[tf] = idx + 1
        out[rel] = (tf, idx)
    return out


def _assign_ensemble_indices_for_vault_relative_paths(
    ordered_vault_relative_paths: Iterable[str],
    vault_root: Path,
) -> dict[str, tuple[TimeFrame, int]]:
    """Keys are relative to the vault directory only (e.g. ``D/mean_reversion_indices/mr_long``)."""
    next_idx: dict[TimeFrame, int] = {}
    out: dict[str, tuple[TimeFrame, int]] = {}
    for rel in ordered_vault_relative_paths:
        ens = (vault_root / rel).resolve()
        tf = _read_ensemble_trading_timeframe(ens)
        idx = next_idx.get(tf, 0)
        next_idx[tf] = idx + 1
        out[rel] = (tf, idx)
    return out


def global_stream_ids_for_signed_signal_feature(
    feature_config: Mapping[str, object],
    *,
    trading_timeframe: TimeFrame,
    ensemble_idx: int,
    portfolio_ticker_names: frozenset[str] | None = None,
) -> List[str]:
    """
    Global stream ids as produced by ``TFPortfolio.predict_base_model_vectors_from_candles`` →
    ``encode_forecast_vectors_for_global_weight_layer``.

    Uses ``build_global_model_name(trading_timeframe, ensemble_idx, vault_model_name)`` for the
    third segment of ``ticker::timeframe::...``, matching runtime.

    When ``portfolio_ticker_names`` is set (uppercase symbols), only those instruments are
    included. Use this so the hierarchy matches ``PortfolioCacheQuery`` tickers—vault JSON may
    still list a wider universe (e.g. RTY) after you shrink ``config.tickers``.
    """
    tickers = feature_config.get("tickers")
    if not isinstance(tickers, list) or not tickers:
        raise ValueError("feature_config.tickers must be a non-empty list")

    if portfolio_ticker_names is not None:
        tickers = [
            t
            for t in tickers
            if str(t).strip().upper() in portfolio_ticker_names
        ]
    if not tickers:
        return []

    models = feature_config.get("base_models")
    if not isinstance(models, list) or len(models) != 1:
        raise ValueError("feature_config.base_models must contain exactly one model")
    model_entry = models[0]
    if not isinstance(model_entry, Mapping):
        raise ValueError("base_models[0] must be a mapping")
    model_name = model_entry.get("model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("base_models[0].model_name must be a non-empty string")

    inner = build_global_model_name(
        trading_timeframe,
        ensemble_idx,
        str(model_name).strip(),
    )
    tf_label = trading_timeframe.name
    return [
        build_global_stream_id(str(ticker).strip(), tf_label, inner) for ticker in tickers
    ]


def infer_weight_hierarchy_group_from_feature_path(
    feature_path: Path,
    vault_root: Path,
) -> str | None:
    """
    If the feature file lives under ``vault/<TF>/<group>/...``, return ``group`` when it is
    a known bucket name; otherwise return ``None``.
    """
    try:
        rel = feature_path.resolve().relative_to(vault_root.resolve())
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 4:
        return None
    tf, maybe_group = parts[0], parts[1]
    if tf not in _VAULT_TF_NAMES:
        return None
    if maybe_group in VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES:
        return maybe_group
    return None


def _ingest_feature_for_buckets(
    *,
    feature_path: Path,
    feature_config: Mapping[str, object],
    vault_root: Path,
    trading_timeframe: TimeFrame,
    ensemble_idx: int,
    strict_group: bool,
    buckets: dict[str, set[str]],
    owner: dict[str, str],
    portfolio_ticker_names: frozenset[str] | None,
) -> None:
    raw_group = feature_config.get("weight_hierarchy_group")
    group: str | None
    if isinstance(raw_group, str) and raw_group.strip():
        group = raw_group.strip()
        if group not in VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES:
            raise ValueError(
                f"{feature_path}: weight_hierarchy_group {group!r} not in "
                f"{sorted(VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES)}"
            )
    else:
        group = infer_weight_hierarchy_group_from_feature_path(feature_path, vault_root)
        if group is None:
            msg = f"{feature_path}: missing weight_hierarchy_group and could not infer from path"
            if strict_group:
                raise ValueError(msg)
            logger.warning("%s — skipped", msg)
            return

    try:
        sids = global_stream_ids_for_signed_signal_feature(
            feature_config,
            trading_timeframe=trading_timeframe,
            ensemble_idx=ensemble_idx,
            portfolio_ticker_names=portfolio_ticker_names,
        )
    except ValueError as exc:
        raise ValueError(f"{feature_path}: {exc}") from exc
    for sid in sids:
        prev = owner.get(sid)
        if prev is not None and prev != group:
            raise ValueError(
                f"stream_id {sid!r} mapped to both {prev!r} and {group!r} "
                f"(see {feature_path})"
            )
        owner[sid] = group
        buckets[group].add(sid)


def collect_streams_by_group_from_vault(
    vault_root: str | Path,
    *,
    strict_group: bool = False,
    portfolio_ticker_names: frozenset[str] | None = None,
) -> Dict[str, FrozenSet[str]]:
    """
    Walk validated vault feature JSONs and bucket global stream ids by ``weight_hierarchy_group``.

    Ensemble index per timeframe follows **sorted** ensemble directory paths (deterministic).
    For portfolio runs, prefer ``collect_streams_by_group_for_ensemble_dirs`` with the same
    ``ensemble_dirs`` order as ``load_ensemble_from_vault``.
    """
    root = Path(vault_root).resolve()
    grouped: dict[str, list[tuple[Path, Mapping[str, object]]]] = defaultdict(list)
    for _member_id, feature_path, feature_config in iter_vault_feature_members(root):
        ens = feature_path.parent.parent
        key = str(ens.relative_to(root))
        grouped[key].append((feature_path, feature_config))

    assignments = _assign_ensemble_indices_for_vault_relative_paths(sorted(grouped.keys()), root)
    buckets: Dict[str, set[str]] = {g: set() for g in sorted(VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES)}
    owner: Dict[str, str] = {}

    for key in sorted(grouped.keys()):
        trading_tf, ensemble_idx = assignments[key]
        for feature_path, feature_config in grouped[key]:
            _ingest_feature_for_buckets(
                feature_path=feature_path,
                feature_config=feature_config,
                vault_root=root,
                trading_timeframe=trading_tf,
                ensemble_idx=ensemble_idx,
                strict_group=strict_group,
                buckets=buckets,
                owner=owner,
                portfolio_ticker_names=portfolio_ticker_names,
            )

    return {g: frozenset(buckets[g]) for g in buckets}


def collect_streams_by_group_for_ensemble_dirs(
    repo_root: str | Path,
    ensemble_dirs: Mapping[str, str],
    *,
    strict_group: bool = False,
    portfolio_ticker_names: frozenset[str] | None = None,
    exclude_feature_stems_by_ensemble: Mapping[str, frozenset[str]] | None = None,
) -> Dict[str, FrozenSet[str]]:
    """
    Like ``collect_streams_by_group_from_vault``, but only features under the given
    ensemble directory paths (repository-relative, e.g. ``vault/M/buy_hold/buy_hold_long``).

    ``ensemble_dirs`` iteration order must match portfolio loading order (``dict`` insertion
    order from ``config.ensemble_dirs``).

    Pass ``portfolio_ticker_names`` (uppercase symbols) to align with ``PortfolioResearchConfig.tickers``.

    ``exclude_feature_stems_by_ensemble`` matches ``load_ensemble_from_vault``: repo-relative
    ensemble path keys to sets of feature JSON stems to omit.
    """
    root = Path(repo_root).resolve()
    buckets: Dict[str, set[str]] = {g: set() for g in sorted(VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES)}
    owner: Dict[str, str] = {}

    assignments = _assign_ensemble_indices_for_repo_relative_paths(ensemble_dirs.values(), root)

    for ensemble_path_rel in ensemble_dirs.values():
        vault_root = vault_root_from_repo_relative_ensemble(root, ensemble_path_rel)
        features_dir = (root / ensemble_path_rel / "features").resolve()
        if not features_dir.is_dir():
            continue
        trading_tf, ensemble_idx = assignments[ensemble_path_rel]
        skip_stems = _exclude_stems_for_ensemble_rel(
            root, ensemble_path_rel, exclude_feature_stems_by_ensemble
        )
        for feature_path in sorted(features_dir.glob("*.json")):
            if feature_path.stem in skip_stems:
                continue
            try:
                feature_config = load_validated_feature_config(feature_path)
            except (ValueError, OSError):
                continue
            _ingest_feature_for_buckets(
                feature_path=feature_path,
                feature_config=feature_config,
                vault_root=vault_root,
                trading_timeframe=trading_tf,
                ensemble_idx=ensemble_idx,
                strict_group=strict_group,
                buckets=buckets,
                owner=owner,
                portfolio_ticker_names=portfolio_ticker_names,
            )

    return {g: frozenset(buckets[g]) for g in buckets}


def build_hierarchy_equal_spec(
    streams_by_group: Mapping[str, Collection[str]],
    *,
    root_id: str = "root",
    bucket_order: Sequence[str] | None = None,
) -> dict[str, object]:
    """
    Build a JSON-serializable root group for ``WeightLayer`` ``hierarchy_equal``.

    Each known bucket becomes a child ``type: group`` with ``id`` equal to the bucket name.
    Leaves use ``type: leaf`` and ``stream_id`` (global id strings). Empty buckets are omitted.
    """
    order = (
        tuple(bucket_order)
        if bucket_order is not None
        else tuple(sorted(VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES))
    )
    unknown = frozenset(streams_by_group.keys()) - VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES
    if unknown:
        raise ValueError(f"unknown bucket keys: {sorted(unknown)}")

    children: List[dict[str, object]] = []
    for bucket in order:
        sids = sorted({str(s).strip() for s in streams_by_group.get(bucket, ()) if str(s).strip()})
        if not sids:
            continue
        children.append(
            {
                "type": "group",
                "id": bucket,
                "children": [{"type": "leaf", "stream_id": sid} for sid in sids],
            }
        )
    if not children:
        raise ValueError("hierarchy would be empty: no streams in any bucket")
    return {"type": "group", "id": root_id, "children": children}


def build_hierarchy_spec_from_vault(
    vault_root: str | Path,
    *,
    strict_group: bool = False,
    root_id: str = "root",
    bucket_order: Sequence[str] | None = None,
    portfolio_ticker_names: frozenset[str] | None = None,
) -> Tuple[dict[str, object], Dict[str, FrozenSet[str]]]:
    """
    Collect streams from the vault and return ``(hierarchy_spec, streams_by_group)``.
    """
    by_group = collect_streams_by_group_from_vault(
        vault_root,
        strict_group=strict_group,
        portfolio_ticker_names=portfolio_ticker_names,
    )
    spec = build_hierarchy_equal_spec(
        {k: sorted(v) for k, v in by_group.items() if v},
        root_id=root_id,
        bucket_order=bucket_order,
    )
    return spec, by_group


def build_hierarchy_spec_for_ensemble_dirs(
    repo_root: str | Path,
    ensemble_dirs: Mapping[str, str],
    *,
    strict_group: bool = False,
    root_id: str = "root",
    bucket_order: Sequence[str] | None = None,
    portfolio_ticker_names: frozenset[str] | None = None,
    exclude_feature_stems_by_ensemble: Mapping[str, frozenset[str]] | None = None,
) -> dict[str, object]:
    """Build a ``hierarchy_equal`` root group from only the given ensemble paths."""
    by_group = collect_streams_by_group_for_ensemble_dirs(
        repo_root,
        ensemble_dirs,
        strict_group=strict_group,
        portfolio_ticker_names=portfolio_ticker_names,
        exclude_feature_stems_by_ensemble=exclude_feature_stems_by_ensemble,
    )
    return build_hierarchy_equal_spec(
        {k: sorted(v) for k, v in by_group.items() if v},
        root_id=root_id,
        bucket_order=bucket_order,
    )


def global_stream_ids_for_ensemble_leaf(
    repo_root: str | Path,
    ensemble_dirs: Mapping[str, str],
    leaf_name: str,
    *,
    portfolio_ticker_names: frozenset[str] | None = None,
) -> FrozenSet[str]:
    """Global stream ids for all signed-signal features under one ensemble leaf.

    Ensemble index assignment follows ``collect_streams_by_group_for_ensemble_dirs`` /
    ``load_ensemble_from_vault`` ordering (``ensemble_dirs`` insertion order).
    """
    if leaf_name not in ensemble_dirs:
        raise KeyError(f"Unknown ensemble leaf {leaf_name!r}; keys: {sorted(ensemble_dirs)!r}")
    root = Path(repo_root).resolve()
    ensemble_path_rel = ensemble_dirs[leaf_name]
    assignments = _assign_ensemble_indices_for_repo_relative_paths(ensemble_dirs.values(), root)
    trading_tf, ensemble_idx = assignments[ensemble_path_rel]
    features_dir = (root / ensemble_path_rel / "features").resolve()
    out: set[str] = set()
    for feature_path in sorted(features_dir.glob("*.json")):
        try:
            feature_config = load_validated_feature_config(feature_path)
        except (ValueError, OSError):
            continue
        for sid in global_stream_ids_for_signed_signal_feature(
            feature_config,
            trading_timeframe=trading_tf,
            ensemble_idx=ensemble_idx,
            portfolio_ticker_names=portfolio_ticker_names,
        ):
            out.add(sid)
    return frozenset(out)


def global_stream_ids_for_vault_feature_member(
    repo_root: str | Path,
    ensemble_dirs: Mapping[str, str],
    vault_member_id: str,
    *,
    portfolio_ticker_names: frozenset[str] | None = None,
) -> FrozenSet[str]:
    """Global stream ids for a single vault feature (vault-root-relative member id, no ``.json``)."""
    root = Path(repo_root).resolve()
    parts = Path(vault_member_id).parts
    if "features" not in parts:
        raise ValueError(f"Expected 'features' in vault member id, got {vault_member_id!r}")
    fi = parts.index("features")
    inside_vault_ensemble = Path(*parts[:fi]) if parts[:fi] else Path()
    ensemble_key: str | None = None
    for rel in ensemble_dirs.values():
        rel_parts = Path(rel).parts
        if len(rel_parts) < 2:
            continue
        if Path(*rel_parts[1:]) == inside_vault_ensemble:
            ensemble_key = rel
            break
    if ensemble_key is None:
        raise ValueError(
            f"No ensemble_dirs entry matches inside-vault path {inside_vault_ensemble.as_posix()!r} "
            f"for vault_member_id {vault_member_id!r}"
        )
    vault_root = vault_root_from_repo_relative_ensemble(root, ensemble_key)
    assignments = _assign_ensemble_indices_for_repo_relative_paths(ensemble_dirs.values(), root)
    trading_tf, ensemble_idx = assignments[ensemble_key]
    feature_path = vault_root.joinpath(*parts).with_suffix(".json")
    feature_config = load_validated_feature_config(feature_path)
    return frozenset(
        global_stream_ids_for_signed_signal_feature(
            feature_config,
            trading_timeframe=trading_tf,
            ensemble_idx=ensemble_idx,
            portfolio_ticker_names=portfolio_ticker_names,
        )
    )


__all__ = [
    "build_hierarchy_equal_spec",
    "build_hierarchy_spec_for_ensemble_dirs",
    "build_hierarchy_spec_from_vault",
    "collect_streams_by_group_for_ensemble_dirs",
    "collect_streams_by_group_from_vault",
    "global_stream_ids_for_ensemble_leaf",
    "global_stream_ids_for_signed_signal_feature",
    "global_stream_ids_for_vault_feature_member",
    "infer_weight_hierarchy_group_from_feature_path",
]
