"""Discover validated vault feature members (per-json) for tooling and exports."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ensemble.vault.constants import VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES
from ensemble.vault.feature_files import load_validated_feature_config

_VAULT_TIMEFRAME_TOP_LEVEL = frozenset({"D", "W", "M"})


def iter_vault_feature_members(
    vault_root: Path,
) -> Iterator[tuple[str, Path, dict[str, Any]]]:
    """Yield each validated feature under ensemble ``features`` folders.

    Layouts:

    - ``vault/<D|W|M>/<weight_group>/<ensemble>/features/*.json`` (preferred)
    - ``vault/<D|W|M>/<ensemble>/features/*.json`` (legacy flat)

    Skips ``portfolio_snapshots`` and any top-level directory that is not D, W, or M.
    Invalid JSON or failing validation is skipped (file is ignored).

    Yields
    ------
    vault_member_id
        Repository-relative id without ``.json``, e.g. ``D/my_ensemble/features/my_feat``.
    feature_file
        Absolute or resolved path to the JSON file.
    feature_config
        Validated feature payload.
    """
    root = vault_root.resolve()
    if not root.is_dir():
        return
    for tf_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if tf_dir.name not in _VAULT_TIMEFRAME_TOP_LEVEL:
            continue
        for child in sorted(d for d in tf_dir.iterdir() if d.is_dir()):
            ensemble_dirs: list[Path] = []
            if child.name in VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES:
                ensemble_dirs = sorted(d for d in child.iterdir() if d.is_dir())
            else:
                ensemble_dirs = [child]
            for ensemble_dir in ensemble_dirs:
                features_dir = ensemble_dir / "features"
                if not features_dir.is_dir():
                    continue
                for feature_file in sorted(features_dir.glob("*.json")):
                    try:
                        feature_config = load_validated_feature_config(feature_file)
                    except (ValueError, OSError):
                        continue
                    rel = feature_file.resolve().relative_to(root)
                    vault_member_id = str(rel.with_suffix(""))
                    yield vault_member_id, feature_file, feature_config
