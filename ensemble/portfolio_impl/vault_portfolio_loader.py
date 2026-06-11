"""Build :class:`GlobalPortfolio` from working vault ensemble directories (no snapshots)."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from ensemble.vault.constants import VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES
from ensemble.vault_manager import load_ensemble_from_vault
from cache.runtime.cache_paths import project_root
from lib.core.enums import TimeFrame
from lib.core.vault_paths import resolve_vault_root

from .global_portfolio_impl import GlobalPortfolio
from .tf_portfolio import TFPortfolio


def _infer_timeframe_from_ensemble_dir(ensemble_dir: str) -> TimeFrame:
    for part in Path(ensemble_dir.replace("\\", "/")).parts:
        if part in TimeFrame.__members__:
            return TimeFrame[part]
    raise ValueError(f"Could not infer timeframe from ensemble_dir={ensemble_dir!r}")


def discover_ensemble_dirs_in_vault(
    vault_root: str,
    timeframes: tuple[TimeFrame, ...],
) -> tuple[str, ...]:
    """Return repo-relative ensemble directory strings under ``vault_root`` for each timeframe."""
    root = resolve_vault_root(vault_root)
    discovered: list[str] = []
    repo_root = project_root()
    for tf in timeframes:
        tf_dir = root / tf.name
        if not tf_dir.is_dir():
            continue
        for child in sorted(tf_dir.iterdir()):
            if not child.is_dir():
                continue
            candidates = (
                sorted(d for d in child.iterdir() if d.is_dir())
                if child.name in VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES
                else [child]
            )
            for ens_dir in candidates:
                try:
                    discovered.append(ens_dir.relative_to(repo_root).as_posix())
                except ValueError:
                    discovered.append(str(ens_dir))
    return tuple(dict.fromkeys(discovered))


def build_global_portfolio_from_ensemble_dirs(
    ensemble_dirs: Sequence[str],
    *,
    active_timeframes: tuple[TimeFrame, ...],
    target_volatility: float,
    max_position_pct: float,
    idm_max: float,
) -> GlobalPortfolio:
    """Load ensembles from explicit paths and assemble TF / global portfolios."""
    by_tf: dict[TimeFrame, list[object]] = defaultdict(list)
    for ensemble_dir in ensemble_dirs:
        tf = _infer_timeframe_from_ensemble_dir(str(ensemble_dir))
        if tf not in active_timeframes:
            continue
        loaded = load_ensemble_from_vault(
            str(ensemble_dir),
            target_volatility=float(target_volatility),
        )
        by_tf[tf].append(loaded)

    tf_portfolios = [
        TFPortfolio(
            ensembles=by_tf[tf],
            trading_timeframe=tf,
            target_volatility=target_volatility,
            max_position_pct=max_position_pct,
            idm_max=idm_max,
        )
        for tf in active_timeframes
        if by_tf.get(tf)
    ]
    if not tf_portfolios:
        raise ValueError(
            "No ensembles loaded for requested timeframes "
            f"{[t.name for t in active_timeframes]} from dirs {tuple(ensemble_dirs)!r}"
        )
    return GlobalPortfolio(
        tf_portfolios=tf_portfolios,
        max_position_pct=max_position_pct,
        idm_max=idm_max,
    )


__all__ = [
    "build_global_portfolio_from_ensemble_dirs",
    "discover_ensemble_dirs_in_vault",
]
