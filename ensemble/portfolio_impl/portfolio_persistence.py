from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional

from utils.cache.runtime.central_cache_models import ArtifactScope


def save_global_portfolio_snapshot(
    portfolio: "GlobalPortfolio",
    *,
    fit_start: datetime,
    fit_end: datetime,
    vault_root: str = "vault",
) -> str:
    """Persist an immutable global-portfolio snapshot."""
    from ensemble.portfolio_impl.portfolio_vault import (
        save_global_portfolio_snapshot as _save_snapshot,
    )

    return _save_snapshot(
        portfolio,
        fit_start=fit_start,
        fit_end=fit_end,
        vault_root=vault_root,
    )


def load_global_portfolio_snapshot(
    portfolio_id: str,
    vault_root: str = "vault",
) -> "GlobalPortfolio":
    """Load a previously snapshotted GlobalPortfolio by ``portfolio_id``."""
    from ensemble.portfolio_impl.portfolio_vault import (
        load_global_portfolio_snapshot as _load_snapshot,
    )

    return _load_snapshot(portfolio_id=portfolio_id, vault_root=vault_root)


def materialize_global_portfolio_predictions(
    *,
    portfolio: "GlobalPortfolio",
    query: "PortfolioCacheQuery",
    portfolio_id: str,
    world: "PortfolioWorld | str",
    research_run_id: Optional[str] = None,
    scope: ArtifactScope = ArtifactScope.LIVE,
):
    """Materialize portfolio/base-model predictions into the dedicated cache tree."""
    from utils.cache.runtime.portfolio_materialization import (
        materialize_global_portfolio_predictions as _materialize_predictions,
    )

    return _materialize_predictions(
        portfolio=portfolio,
        query=query,
        portfolio_id=portfolio_id,
        world=world,
        research_run_id=research_run_id,
        scope=scope,
    )


def prune_inactive_base_model_materializations(
    *,
    vault_root: str = "vault",
    scope: ArtifactScope = ArtifactScope.LIVE,
    ensemble_dirs: Optional[Iterable[str]] = None,
):
    """Delete base-model materializations whose identities are no longer active."""
    from utils.cache.runtime.portfolio_materialization import (
        prune_inactive_base_model_materializations as _prune_materializations,
    )

    return _prune_materializations(
        vault_root=vault_root,
        scope=scope,
        ensemble_dirs=ensemble_dirs,
    )


if False:  # pragma: no cover
    from ensemble.portfolio import GlobalPortfolio, PortfolioCacheQuery, PortfolioWorld
