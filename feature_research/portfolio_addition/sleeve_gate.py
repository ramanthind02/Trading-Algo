"""Sleeve-scoped portfolio addition gate helpers.

A *sleeve* is the asset-first weight-layer bucket ``asset_class / style_group``
(for example ``equity_indices / momentum``). The primary admission test asks
whether adding the candidate improves that sleeve's combined return stream, not
only the global portfolio.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from ensemble.vault.constants import (
    STRATEGY_GROUP_ASSET_OVERRIDE,
    TICKER_ASSET_CLASS,
)
from ensemble.vault.hierarchy_spec import (
    _asset_class_for_stream,
    collect_streams_by_group_for_ensemble_dirs,
    global_stream_ids_for_signed_signal_feature,
)
from ensemble.vault.feature_files import load_validated_feature_config
from utils.core.enums import TimeFrame


def candidate_sleeve_identity(
    *,
    weight_hierarchy_group: str,
    candidate_ticker_names: frozenset[str],
) -> tuple[str, str]:
    """Return ``(asset_class, style_group)`` for a materialized candidate ensemble."""
    style_group = weight_hierarchy_group.strip()
    if style_group in STRATEGY_GROUP_ASSET_OVERRIDE:
        return STRATEGY_GROUP_ASSET_OVERRIDE[style_group], style_group
    if not candidate_ticker_names:
        return "diversified", style_group
    asset_classes = {
        TICKER_ASSET_CLASS.get(t.upper(), "diversified") for t in candidate_ticker_names
    }
    if len(asset_classes) == 1:
        return next(iter(asset_classes)), style_group
    return "diversified", style_group


def sleeve_label(asset_class: str, style_group: str) -> str:
    return f"{asset_class}/{style_group}"


def _assign_ensemble_indices_for_repo_relative_paths(
    ordered_repo_relative_paths: Sequence[str],
    repo_root: Path,
) -> dict[str, tuple[TimeFrame, int]]:
    from ensemble.vault.hierarchy_spec import (
        _assign_ensemble_indices_for_repo_relative_paths as _assign,
    )

    return _assign(ordered_repo_relative_paths, repo_root)


def ensemble_dirs_in_sleeve(
    repo_root: str | Path,
    ensemble_dirs: Mapping[str, str],
    *,
    asset_class: str,
    style_group: str,
    portfolio_ticker_names: frozenset[str] | None,
) -> dict[str, str]:
    """Keep ensembles that contribute at least one stream to the target sleeve."""
    root = Path(repo_root).resolve()
    target_asset = asset_class.strip()
    target_style = style_group.strip()
    out: dict[str, str] = {}
    assignments = _assign_ensemble_indices_for_repo_relative_paths(ensemble_dirs.values(), root)

    for name, ensemble_path_rel in ensemble_dirs.items():
        features_dir = (root / ensemble_path_rel / "features").resolve()
        if not features_dir.is_dir():
            continue
        trading_tf, ensemble_idx = assignments[ensemble_path_rel]
        for feature_path in sorted(features_dir.glob("*.json")):
            try:
                feature_config = load_validated_feature_config(feature_path)
            except (ValueError, OSError):
                continue
            raw_group = feature_config.get("weight_hierarchy_group")
            group = (
                str(raw_group).strip()
                if isinstance(raw_group, str) and raw_group.strip()
                else target_style
            )
            try:
                stream_ids = global_stream_ids_for_signed_signal_feature(
                    feature_config,
                    trading_timeframe=trading_tf,
                    ensemble_idx=ensemble_idx,
                    portfolio_ticker_names=portfolio_ticker_names,
                )
            except ValueError:
                continue
            if any(
                _asset_class_for_stream(sid, group) == target_asset
                and group == target_style
                for sid in stream_ids
            ):
                out[name] = ensemble_path_rel
                break
    return out


def sleeve_peer_keys(
    baseline_ensemble_dirs: Mapping[str, str],
    sleeve_baseline_dirs: Mapping[str, str],
) -> frozenset[str]:
    return frozenset(sleeve_baseline_dirs.keys())


def candidate_ticker_names_from_ensemble(
    repo_root: str | Path,
    ensemble_path_rel: str,
    *,
    portfolio_ticker_names: frozenset[str] | None = None,
) -> frozenset[str]:
    """Union of feature ``tickers`` on one ensemble, optionally filtered to the portfolio."""
    root = Path(repo_root).resolve()
    features_dir = (root / ensemble_path_rel / "features").resolve()
    if not features_dir.is_dir():
        return frozenset()
    names: set[str] = set()
    for feature_path in sorted(features_dir.glob("*.json")):
        try:
            feature_config = load_validated_feature_config(feature_path)
        except (ValueError, OSError):
            continue
        raw = feature_config.get("tickers")
        if not isinstance(raw, list):
            continue
        for ticker in raw:
            name = str(ticker).strip().upper()
            if not name:
                continue
            if portfolio_ticker_names is not None and name not in portfolio_ticker_names:
                continue
            names.add(name)
    return frozenset(names)


def _resolve_ensemble_dir_path(repo_root: Path, ensemble_path: str) -> Path:
    path = Path(ensemble_path)
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def candidate_stream_ids_for_ensemble(
    repo_root: str | Path,
    ensemble_path_rel: str,
    *,
    portfolio_ticker_names: frozenset[str] | None,
    portfolio_ensemble_dirs: Mapping[str, str] | None = None,
) -> frozenset[str]:
    """Global stream ids for one ensemble directory.

    When ``portfolio_ensemble_dirs`` is provided (baseline + candidate, insertion order),
    the ensemble index matches the portfolio weight-layer refit. Without it, the candidate
    is indexed as the only ensemble (``ensemble_0``), which breaks sleeve weight metrics.
    """
    root = Path(repo_root).resolve()
    candidate_abs = _resolve_ensemble_dir_path(root, ensemble_path_rel)
    if portfolio_ensemble_dirs is not None:
        ordered_paths = list(portfolio_ensemble_dirs.values())
        assignments = _assign_ensemble_indices_for_repo_relative_paths(ordered_paths, root)
        trading_tf: TimeFrame | None = None
        ensemble_idx: int | None = None
        for rel_path, assignment in assignments.items():
            if _resolve_ensemble_dir_path(root, rel_path) == candidate_abs:
                trading_tf, ensemble_idx = assignment
                break
        if trading_tf is None or ensemble_idx is None:
            raise ValueError(
                f"Candidate ensemble {ensemble_path_rel!r} is not present in "
                "portfolio_ensemble_dirs; cannot align stream ids with the refit."
            )
    else:
        assignments = _assign_ensemble_indices_for_repo_relative_paths([ensemble_path_rel], root)
        trading_tf, ensemble_idx = assignments[ensemble_path_rel]
    features_dir = candidate_abs / "features"
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


def streams_by_sleeve_from_ensemble_dirs(
    repo_root: str | Path,
    ensemble_dirs: Mapping[str, str],
    *,
    portfolio_ticker_names: frozenset[str] | None,
) -> dict[str, dict[str, frozenset[str]]]:
    """Map ``asset_class -> style_group -> stream ids`` for the given ensembles."""
    by_group = collect_streams_by_group_for_ensemble_dirs(
        repo_root,
        ensemble_dirs,
        strict_group=True,
        portfolio_ticker_names=portfolio_ticker_names,
    )
    nested: dict[str, dict[str, set[str]]] = {}
    for group, stream_ids in by_group.items():
        for sid in stream_ids:
            asset = _asset_class_for_stream(str(sid), group)
            nested.setdefault(asset, {}).setdefault(group, set()).add(str(sid))
    return {
        asset: {style: frozenset(sids) for style, sids in styles.items()}
        for asset, styles in nested.items()
    }
