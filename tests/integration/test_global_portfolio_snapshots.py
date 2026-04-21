"""Repository-backed integration coverage for GlobalPortfolio snapshots and materialization."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from ensemble.portfolio import (  # noqa: E402
    GlobalPortfolio,
    PortfolioCacheQuery,
    PortfolioWorld,
    TFPortfolio,
    load_global_portfolio_snapshot,
)
from ensemble.vault_manager import (  # noqa: E402
    ensure_vault_cache_coverage,
    get_ensemble_tickers,
    load_ensemble_from_vault,
)
from tests.integration._portfolio_cache_helpers import (  # noqa: E402
    instrument_returns_from_cache,
    source_data_available,
)
from utils.cache.runtime.bootstrap_source_candles import bootstrap_source_candles  # noqa: E402
from utils.cache.runtime.central_cache import CentralCacheStore  # noqa: E402
from utils.cache.runtime.central_cache_models import ArtifactScope  # noqa: E402
from utils.cache.runtime.portfolio_materialization import (  # noqa: E402
    materialize_global_portfolio_predictions,
)
from utils.core.enums import TimeFrame  # noqa: E402


def test_global_portfolio_snapshot_roundtrip_and_materialization(
    isolated_central_cache: None,
    tmp_path: Path,
) -> None:
    ensemble_dir = "vault/M/buy_hold/buy_hold_long"
    if not source_data_available(ensemble_dir):
        pytest.skip("Repository-backed OHLC dataset is missing required buy_hold_long files")

    tickers = get_ensemble_tickers(ensemble_dir)
    train_start = datetime(2022, 1, 31)
    train_end = datetime(2022, 8, 31)
    test_start = datetime(2022, 9, 1)
    test_end = datetime(2022, 11, 30)

    bootstrap_summary = bootstrap_source_candles(
        tickers=tickers,
        timeframes=[TimeFrame.D, TimeFrame.M],
        start_date=train_start,
        end_date=test_end,
        reset_existing=True,
    )
    assert bootstrap_summary["failed"] == 0

    refresh_summary = ensure_vault_cache_coverage(
        [ensemble_dir],
        start_date=train_start,
        end_date=test_end,
    )
    assert refresh_summary["failed"] == 0
    assert refresh_summary["rebuilt"] >= len(tickers) * 2

    ensemble = load_ensemble_from_vault(
        ensemble_dir,
        refit=True,
        target_volatility=0.15,
    )
    ensemble.use_cache = True
    ensemble.retry_on_cache_miss = False
    for model in ensemble.base_models.values():
        model.use_cache = True

    tf_portfolio = TFPortfolio(
        ensembles=[ensemble],
        trading_timeframe=TimeFrame.M,
        target_volatility=0.15,
        max_position_pct=3.5,
        use_cache=True,
    )
    global_portfolio = GlobalPortfolio(
        tf_portfolios=[tf_portfolio],
        max_position_pct=3.5,
    )

    instrument_returns = instrument_returns_from_cache(
        store=CentralCacheStore.get_instance(),
        tickers=tickers,
        start=train_start,
        end=test_end,
    )
    train_returns = instrument_returns.loc[
        (instrument_returns.index >= pd.Timestamp(train_start))
        & (instrument_returns.index <= pd.Timestamp(train_end))
    ]
    train_query = PortfolioCacheQuery(
        tickers=tuple(ticker.name for ticker in tickers),
        start=train_start,
        end=train_end,
        timeframes=(TimeFrame.M,),
    )
    test_query = PortfolioCacheQuery(
        tickers=tuple(ticker.name for ticker in tickers),
        start=test_start,
        end=test_end,
        timeframes=(TimeFrame.M,),
    )

    global_portfolio.fit_from_cache(train_query, train_returns)
    train_positions = global_portfolio.predict_from_cache(train_query)
    test_positions = global_portfolio.predict_from_cache(test_query)
    assert global_portfolio.is_fitted_ is True
    assert not train_positions.empty
    assert not test_positions.empty

    snapshot_vault_root = tmp_path / "snapshot_vault"
    portfolio_id = global_portfolio.save_to_vault(
        fit_start=train_start,
        fit_end=train_end,
        vault_root=str(snapshot_vault_root),
    )
    assert global_portfolio.portfolio_id_ == portfolio_id

    snapshot_dir = snapshot_vault_root / "portfolio_snapshots" / portfolio_id
    assert (snapshot_dir / "snapshot.json").exists()
    assert (snapshot_dir / "ensembles" / "M" / "buy_hold_long" / "ensemble_config.json").exists()
    assert any((snapshot_dir / "ensembles" / "M" / "buy_hold_long" / "features").glob("*.json"))

    cache_root = str(CentralCacheStore.get_instance().cache_dir)
    train_summary = materialize_global_portfolio_predictions(
        portfolio=global_portfolio,
        query=train_query,
        portfolio_id=portfolio_id,
        world=PortfolioWorld.TRAIN,
        scope=ArtifactScope.LIVE,
        vault_root="vault",
        cache_root=cache_root,
    )
    test_summary = materialize_global_portfolio_predictions(
        portfolio=global_portfolio,
        query=test_query,
        portfolio_id=portfolio_id,
        world=PortfolioWorld.TEST,
        scope=ArtifactScope.LIVE,
        vault_root="vault",
        cache_root=cache_root,
    )

    assert train_summary.portfolio_rows_written == len(train_positions)
    assert train_summary.base_model_rows_written > 0
    assert test_summary.portfolio_rows_written == len(test_positions)
    assert test_summary.base_model_rows_written > 0

    portfolio_path = (
        Path(cache_root)
        / "materialized"
        / "live"
        / "portfolio"
        / f"{portfolio_id}.parquet"
    )
    base_models_dir = (
        Path(cache_root)
        / "materialized"
        / "live"
        / "base_models"
        / str(ensemble.vault_timeframe)
        / str(ensemble.vault_ensemble_name)
    )
    base_model_parquets = sorted(base_models_dir.glob("*.parquet"))
    assert len(base_model_parquets) == 1
    base_model_path = base_model_parquets[0]

    portfolio_rows = pd.read_parquet(portfolio_path)
    assert {
        PortfolioWorld.TEST.value,
        PortfolioWorld.TRAIN.value,
    }.issubset(set(portfolio_rows["world"].unique().tolist()))
    assert portfolio_rows["portfolio_id"].unique().tolist() == [portfolio_id]

    base_model_rows = pd.read_parquet(base_model_path)
    assert {
        PortfolioWorld.TEST.value,
        PortfolioWorld.TRAIN.value,
    }.issubset(set(base_model_rows["world"].unique().tolist()))
    assert base_model_rows["portfolio_id"].unique().tolist() == [portfolio_id]

    restored = load_global_portfolio_snapshot(
        portfolio_id=portfolio_id,
        vault_root=str(snapshot_vault_root),
    )
    replay = restored.predict_from_cache(test_query)

    assert restored.portfolio_id_ == portfolio_id
    assert restored.is_fitted_ is True
    assert_frame_equal(
        test_positions.reset_index(drop=True),
        replay.reset_index(drop=True),
    )
