"""Repository-backed integration coverage for the cache-first portfolio cutover."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from ensemble.portfolio import Portfolio, PortfolioCacheQuery
from ensemble.vault_manager import (
    ensure_vault_cache_coverage,
    get_ensemble_tickers,
    load_ensemble_from_vault,
)
from utils.cache.central_cache import CentralCacheStore
from utils.cache.central_cache_models import ArtifactDescriptor, ArtifactScope
from utils.cache.ingest_source_candles import ingest_source_candles
from utils.core.enums import TimeFrame


@pytest.fixture
def isolated_central_cache(tmp_path: Path) -> None:
    CentralCacheStore.reset()
    CentralCacheStore._instance = CentralCacheStore(cache_dir=tmp_path / "central_cache")  # type: ignore[attr-defined]
    yield
    CentralCacheStore.reset()


def test_monthly_buy_hold_portfolio_runs_from_central_cache(
    isolated_central_cache: None,
) -> None:
    ensemble_dir = "vault/M/buy_hold_long"
    tickers = get_ensemble_tickers(ensemble_dir)
    start = datetime(2022, 1, 31)
    end = datetime(2022, 6, 30)

    ingest_summary = ingest_source_candles(
        tickers=tickers,
        timeframes=[TimeFrame.D, TimeFrame.M],
        start_date=start,
        end_date=end,
        reset_existing=True,
    )
    assert ingest_summary["failed"] == 0

    refresh_summary = ensure_vault_cache_coverage(
        [ensemble_dir],
        start_date=start,
        end_date=end,
    )
    assert refresh_summary["failed"] == 0
    assert refresh_summary["rebuilt"] >= len(tickers) * 2

    store = CentralCacheStore.get_instance()
    ewsd_descriptor = ArtifactDescriptor(
        family="bias",
        ticker=tickers[0],
        timeframe=TimeFrame.D,
        module_name="ewsd",
        params={"long_run_window": 2520},
        scope=ArtifactScope.LIVE,
        artifact_name="ewsd",
    )
    assert store.describe_artifact(ewsd_descriptor) is not None

    ensemble = load_ensemble_from_vault(
        ensemble_dir,
        refit=True,
        target_volatility=0.15,
    )
    ensemble.use_cache = True
    ensemble.retry_on_cache_miss = False
    for model in ensemble.base_models.values():
        model.use_cache = True

    portfolio = Portfolio(
        ensembles=[ensemble],
        trading_timeframe=TimeFrame.M,
        target_volatility=0.15,
        max_position_pct=3.5,
        use_cache=True,
    )
    query = PortfolioCacheQuery(
        tickers=tuple(ticker.name for ticker in tickers),
        start=start,
        end=end,
        timeframes=(TimeFrame.M,),
    )

    portfolio.fit_from_cache(query)
    positions = portfolio.predict_from_cache(query)

    assert not positions.empty
    assert list(positions.columns) == [
        "ticker",
        "datetime",
        "forecast_score",
        "position_fraction",
    ]
    assert sorted(positions["ticker"].unique().tolist()) == sorted(
        ticker.name for ticker in tickers
    )
    assert len(positions) == len(tickers) * 6
