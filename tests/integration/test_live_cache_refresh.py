"""Repository-backed integration coverage for automatic live refresh after candle updates."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

import utils.cache.runtime.live_cache_refresh as live_refresh  # noqa: E402
from ensemble.portfolio import GlobalPortfolio, PortfolioCacheQuery, TFPortfolio  # noqa: E402
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
from utils.cache.runtime.central_cache_models import (  # noqa: E402
    ArtifactLifecycleState,
    ArtifactScope,
)
from utils.core.enums import TimeFrame  # noqa: E402


def _materialized_portfolio_row(portfolio_id: str, dt: datetime, ticker_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "portfolio_id": [portfolio_id],
            "world": ["live"],
            "research_run_id": [None],
            "ticker": [ticker_name],
            "datetime": [pd.Timestamp(dt)],
            "forecast_score": [0.0],
            "position_fraction": [0.0],
        }
    )


def test_live_candle_updates_trigger_automatic_refresh(
    isolated_central_cache: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ensemble_dir = "vault/M/buy_hold/buy_hold_long"
    if not source_data_available(ensemble_dir):
        pytest.skip("Repository-backed OHLC dataset is missing required buy_hold_long files")

    tickers = get_ensemble_tickers(ensemble_dir)
    train_start = datetime(2022, 1, 31)
    train_end = datetime(2022, 8, 31)
    live_end = datetime(2022, 11, 30)

    bootstrap_summary = bootstrap_source_candles(
        tickers=tickers,
        timeframes=[TimeFrame.D, TimeFrame.M],
        start_date=train_start,
        end_date=live_end,
        reset_existing=True,
    )
    assert bootstrap_summary["failed"] == 0

    refresh_summary = ensure_vault_cache_coverage(
        [ensemble_dir],
        start_date=train_start,
        end_date=live_end,
    )
    assert refresh_summary["failed"] == 0
    store = CentralCacheStore.get_instance()

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
        store=store,
        tickers=tickers,
        start=train_start,
        end=live_end,
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
    global_portfolio.fit_from_cache(train_query, train_returns)

    snapshot_vault_root = tmp_path / "snapshot_vault"
    portfolio_id = global_portfolio.save_to_vault(
        fit_start=train_start,
        fit_end=train_end,
        vault_root=str(snapshot_vault_root),
    )

    manifest_path = tmp_path / "deployment" / "config" / "live_cache_refresh.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "version": "1",
                "enabled": True,
                "vault_root": "vault",
                "snapshot_vault_root": str(snapshot_vault_root),
                "debounce_seconds": 0.05,
                "active_ensemble_dirs": [ensemble_dir],
                "active_portfolios": [
                    {
                        "name": "buy_hold_live",
                        "portfolio_id": portfolio_id,
                        "tickers": [ticker.name for ticker in tickers],
                        "timeframes": ["M"],
                        "volatility_timeframe": "D",
                        "replay_window_days": 62,
                        "research_run_id": "live_refresh_test",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        live_refresh,
        "default_live_cache_refresh_manifest_path",
        lambda: manifest_path,
    )

    bias_records = [
        record
        for record in store.list_artifacts(scope=ArtifactScope.LIVE)
        if record.descriptor.family == "bias"
        and record.descriptor.ticker == tickers[0]
        and record.descriptor.timeframe == TimeFrame.D
    ]
    assert bias_records
    watched_descriptor = bias_records[0].descriptor
    watched_record = store.describe_artifact(watched_descriptor)
    assert watched_record is not None
    assert watched_record.lifecycle_state is ArtifactLifecycleState.FRESH

    materialized_root = Path(store.cache_dir) / "materialized" / "live"
    historical_portfolio_path = materialized_root / "portfolio" / "historical_portfolio.parquet"
    historical_portfolio_path.parent.mkdir(parents=True, exist_ok=True)
    _materialized_portfolio_row("historical_portfolio", live_end, tickers[0].name).to_parquet(
        historical_portfolio_path,
        index=False,
    )

    stale_base_model_path = (
        materialized_root
        / "base_models"
        / "M"
        / "inactive_ensemble"
        / "fake_feature__dead_model.parquet"
    )
    stale_base_model_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timeframe": ["M"],
            "ensemble_name": ["inactive_ensemble"],
            "feature_name": ["fake_feature"],
            "model_id": ["dead_model"],
            "portfolio_id": ["historical_portfolio"],
            "world": ["live"],
            "research_run_id": [None],
            "ticker": [tickers[0].name],
            "datetime": [pd.Timestamp(live_end)],
            "forecast_score": [0.0],
            "position_fraction": [0.0],
        }
    ).to_parquet(stale_base_model_path, index=False)

    last_daily_row = (
        store.query_candles(tickers[0], TimeFrame.D, start=live_end, end=live_end)
        .reset_index()
        .iloc[[-1]]
        .copy()
    )
    last_daily_row["open"] = last_daily_row["open"] + 5.0
    last_daily_row["close"] = last_daily_row["close"] + 5.0
    last_daily_row["high"] = last_daily_row[["open", "close"]].max(axis=1) + 1.0
    last_daily_row["low"] = last_daily_row[["open", "close"]].min(axis=1) - 1.0

    store.upsert_candles(tickers[0], TimeFrame.D, last_daily_row)

    stale_record = store.describe_artifact(watched_descriptor)
    assert stale_record is not None
    assert stale_record.lifecycle_state is ArtifactLifecycleState.STALE

    orchestrator = live_refresh.get_live_cache_refresh_orchestrator()
    assert orchestrator.wait_for_idle(timeout=120.0)
    assert orchestrator.pending_dirty_keys() == ()

    refreshed_record = store.describe_artifact(watched_descriptor)
    assert refreshed_record is not None
    assert refreshed_record.lifecycle_state is ArtifactLifecycleState.FRESH

    portfolio_path = materialized_root / "portfolio" / f"{portfolio_id}.parquet"
    assert portfolio_path.exists()
    portfolio_rows = pd.read_parquet(portfolio_path)
    live_portfolio_rows = portfolio_rows[portfolio_rows["world"] == "live"]
    assert not live_portfolio_rows.empty
    assert live_portfolio_rows["portfolio_id"].unique().tolist() == [portfolio_id]
    assert live_portfolio_rows["research_run_id"].dropna().unique().tolist() == ["live_refresh_test"]

    active_base_model_paths = sorted(
        path
        for path in (materialized_root / "base_models").rglob("*.parquet")
        if path != stale_base_model_path
    )
    assert active_base_model_paths
    base_model_rows = pd.read_parquet(active_base_model_paths[0])
    live_base_model_rows = base_model_rows[base_model_rows["world"] == "live"]
    assert not live_base_model_rows.empty
    assert live_base_model_rows["portfolio_id"].unique().tolist() == [portfolio_id]
    assert live_base_model_rows["research_run_id"].dropna().unique().tolist() == ["live_refresh_test"]

    assert not stale_base_model_path.exists()
    assert historical_portfolio_path.exists()

    status_payload = json.loads(
        (Path(store.cache_dir) / "live_refresh" / "last_run.json").read_text(encoding="utf-8")
    )
    assert status_payload["status"] == "completed"
    assert status_payload["affected_portfolios"] == ["buy_hold_live"]
    assert status_payload["error"] is None
