from __future__ import annotations

from typing import Iterable, Optional

from lib.cache.runtime.central_cache_models import ArtifactScope


def materialize_global_portfolio_predictions(
    *,
    portfolio: "GlobalPortfolio",
    query: "PortfolioCacheQuery",
    portfolio_id: str,
    world: "PortfolioWorld | str",
    research_run_id: Optional[str] = None,
    scope: ArtifactScope = ArtifactScope.LIVE,
    vault_root: str = "vault",
    cache_root: Optional[str] = None,
    ensemble_dirs: Optional[Iterable[str]] = None,
):
    """Materialize portfolio/base-model predictions into the dedicated cache tree."""
    from lib.cache.runtime.portfolio_materialization import (
        materialize_global_portfolio_predictions as _materialize_predictions,
    )

    return _materialize_predictions(
        portfolio=portfolio,
        query=query,
        portfolio_id=portfolio_id,
        world=world,
        research_run_id=research_run_id,
        scope=scope,
        vault_root=vault_root,
        cache_root=cache_root,
        ensemble_dirs=ensemble_dirs,
    )


def prune_inactive_base_model_materializations(
    *,
    vault_root: str = "vault",
    scope: ArtifactScope = ArtifactScope.LIVE,
    ensemble_dirs: Optional[Iterable[str]] = None,
    cache_root: Optional[str] = None,
):
    """Delete base-model materializations whose identities are no longer active."""
    from lib.cache.runtime.portfolio_materialization import (
        prune_inactive_base_model_materializations as _prune_materializations,
    )

    return _prune_materializations(
        vault_root=vault_root,
        scope=scope,
        ensemble_dirs=ensemble_dirs,
        cache_root=cache_root,
    )


if False:  # pragma: no cover
    from ensemble.portfolio import GlobalPortfolio, PortfolioCacheQuery, PortfolioWorld
