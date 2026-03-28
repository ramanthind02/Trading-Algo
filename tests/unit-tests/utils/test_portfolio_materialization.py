from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ensemble.portfolio import GlobalPortfolio, PortfolioCacheQuery, PortfolioWorld, TFPortfolio
from ensemble.vault_manager import load_ensemble_from_vault
from utils.cache.central_cache import CentralCacheStore
from utils.cache.central_cache_models import ArtifactDescriptor, ArtifactScope
from utils.cache.portfolio_materialization import (
    materialize_global_portfolio_predictions,
    prune_inactive_base_model_materializations,
)
from utils.core.enums import Ticker, TimeFrame


def _write_vault_ensemble(vault_root: Path) -> str:
    ensemble_dir = vault_root / "D" / "sample_long"
    features_dir = ensemble_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    (ensemble_dir / "ensemble_config.json").write_text(
        json.dumps(
            {
                "timeframe": "D",
                "ensemble_name": "sample",
                "direction": "long",
                "tickers": ["ES"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (features_dir / "rsi_signal_D.json").write_text(
        json.dumps(
            {
                "feature_name": "rsi_signal_D",
                "created_at": "2026-03-22T00:00:00+00:00",
                "updated_at": "2026-03-22T00:00:00+00:00",
                "bias_node_spec": {
                    "module_name": "rsi",
                    "timeframes": ["D"],
                },
                "tickers": ["ES"],
                "base_models": [
                    {
                        "model_id": "rule_based_3",
                        "model_name": "rsi_signal_D::rule_based_3",
                        "bias_node_params": {"lookback": 2},
                        "binning_model_type": "rule_based",
                        "strategy": "long",
                        "binning_model_params": {"n_bins": 3},
                        "requires_fit": False,
                        "is_fitted": False,
                        "fitted_params": None,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return str(ensemble_dir)


def _seed_cache(store: CentralCacheStore, dates: pd.DatetimeIndex) -> None:
    candles = pd.DataFrame(
        {
            "datetime": dates,
            "open": [100.0 + idx for idx in range(len(dates))],
            "high": [101.0 + idx for idx in range(len(dates))],
            "low": [99.0 + idx for idx in range(len(dates))],
            "close": [100.5 + idx for idx in range(len(dates))],
            "volume": [1_000.0 + idx for idx in range(len(dates))],
            "ticker": ["ES"] * len(dates),
            "timeframe": [TimeFrame.D] * len(dates),
        }
    )
    store.set_candles(Ticker.ES, TimeFrame.D, candles)
    store.write_artifact(
        ArtifactDescriptor(
            family="bias",
            ticker=Ticker.ES,
            timeframe=TimeFrame.D,
            module_name="ewsd",
            params={"long_run_window": 2520},
            scope=ArtifactScope.LIVE,
            artifact_name="ewsd",
        ),
        pd.DataFrame(
            {
                "datetime": dates,
                "ewsd_annual_vol": [0.2] * len(dates),
            }
        ).set_index("datetime"),
        depends_on=((Ticker.ES, TimeFrame.D),),
    )


def _build_portfolio(tmp_path: Path) -> tuple[GlobalPortfolio, PortfolioCacheQuery, Path]:
    vault_root = tmp_path / "vault"
    ensemble_dir = _write_vault_ensemble(vault_root)
    ensemble = load_ensemble_from_vault(ensemble_dir, refit=False, target_volatility=0.15)
    tf_portfolio = TFPortfolio(
        ensembles=[ensemble],
        trading_timeframe=TimeFrame.D,
        target_volatility=0.15,
        max_position_pct=2.0,
        use_cache=True,
    )
    portfolio = GlobalPortfolio(tf_portfolios=[tf_portfolio], max_position_pct=2.0)
    portfolio.is_fitted_ = True

    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    query = PortfolioCacheQuery(
        tickers=("ES",),
        start=dates.min().to_pydatetime(),
        end=dates.max().to_pydatetime(),
        timeframes=(TimeFrame.D,),
    )
    _seed_cache(CentralCacheStore.get_instance(), dates)
    return portfolio, query, vault_root


def _portfolio_frame(values: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.DataFrame(
        {
            "ticker": ["ES"] * len(values),
            "datetime": dates,
            "forecast_score": values,
            "position_fraction": values,
        }
    )


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path: Path) -> None:
    CentralCacheStore.reset()
    CentralCacheStore._instance = CentralCacheStore(cache_dir=tmp_path / "central_cache")  # type: ignore[attr-defined]
    yield
    CentralCacheStore.reset()


def test_materialization_writes_expected_schema_and_upserts_latest_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    portfolio, query, vault_root = _build_portfolio(tmp_path)

    monkeypatch.setattr(
        GlobalPortfolio,
        "predict_from_cache",
        lambda self, _query, *args, **kwargs: _portfolio_frame([0.1, 0.2, 0.3]),
    )

    def _tf_predict(self, _query, return_ensemble_predictions=False, return_base_model_predictions=False):  # noqa: ANN001
        frame = _portfolio_frame([0.4, 0.5, 0.6])
        if return_ensemble_predictions or return_base_model_predictions:
            result = {"portfolio": frame}
            if return_ensemble_predictions:
                result["ensembles"] = {}
            if return_base_model_predictions:
                result["base_models"] = {"ensemble_0::rsi_signal_D::rule_based_3": frame}
            return result
        return frame

    monkeypatch.setattr(TFPortfolio, "predict_from_cache", _tf_predict)

    cache_root = str(CentralCacheStore.get_instance().cache_dir)
    summary = materialize_global_portfolio_predictions(
        portfolio=portfolio,
        query=query,
        portfolio_id="portfolio_a",
        world=PortfolioWorld.TRAIN,
        scope=ArtifactScope.LIVE,
        vault_root=str(vault_root),
        cache_root=cache_root,
    )
    assert summary.portfolio_rows_written == 3
    assert summary.base_model_files_written == 1
    assert summary.base_model_rows_written == 3
    assert summary.base_model_rows_skipped_inactive == 0

    portfolio_path = Path(cache_root) / "materialized" / "live" / "portfolio" / "portfolio_a.parquet"
    base_model_path = (
        Path(cache_root)
        / "materialized"
        / "live"
        / "base_models"
        / "D"
        / "sample_long"
        / "rsi_signal_D__rule_based_3.parquet"
    )
    assert portfolio_path.exists()
    assert base_model_path.exists()

    portfolio_rows = pd.read_parquet(portfolio_path)
    assert list(portfolio_rows.columns) == [
        "portfolio_id",
        "world",
        "research_run_id",
        "ticker",
        "datetime",
        "forecast_score",
        "position_fraction",
    ]

    base_model_rows = pd.read_parquet(base_model_path)
    assert list(base_model_rows.columns) == [
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

    monkeypatch.setattr(
        GlobalPortfolio,
        "predict_from_cache",
        lambda self, _query, *args, **kwargs: _portfolio_frame([0.9, 0.8, 0.7]),
    )

    def _tf_predict_updated(self, _query, return_ensemble_predictions=False, return_base_model_predictions=False):  # noqa: ANN001
        frame = _portfolio_frame([0.7, 0.8, 0.9])
        if return_ensemble_predictions or return_base_model_predictions:
            result = {"portfolio": frame}
            if return_ensemble_predictions:
                result["ensembles"] = {}
            if return_base_model_predictions:
                result["base_models"] = {"ensemble_0::rsi_signal_D::rule_based_3": frame}
            return result
        return frame

    monkeypatch.setattr(TFPortfolio, "predict_from_cache", _tf_predict_updated)

    materialize_global_portfolio_predictions(
        portfolio=portfolio,
        query=query,
        portfolio_id="portfolio_a",
        world=PortfolioWorld.TRAIN,
        scope=ArtifactScope.LIVE,
        vault_root=str(vault_root),
        cache_root=cache_root,
    )

    updated_portfolio_rows = pd.read_parquet(portfolio_path)
    assert len(updated_portfolio_rows) == 3
    assert updated_portfolio_rows["forecast_score"].tolist() == [0.9, 0.8, 0.7]

    updated_base_model_rows = pd.read_parquet(base_model_path)
    assert len(updated_base_model_rows) == 3
    assert updated_base_model_rows["forecast_score"].tolist() == [0.7, 0.8, 0.9]


def test_prune_inactive_base_model_materializations_keeps_active_and_portfolio_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    portfolio, query, vault_root = _build_portfolio(tmp_path)
    monkeypatch.setattr(
        GlobalPortfolio,
        "predict_from_cache",
        lambda self, _query, *args, **kwargs: _portfolio_frame([0.1, 0.2, 0.3]),
    )

    def _tf_predict(self, _query, return_ensemble_predictions=False, return_base_model_predictions=False):  # noqa: ANN001
        frame = _portfolio_frame([0.4, 0.5, 0.6])
        if return_ensemble_predictions or return_base_model_predictions:
            result = {"portfolio": frame}
            if return_ensemble_predictions:
                result["ensembles"] = {}
            if return_base_model_predictions:
                result["base_models"] = {"ensemble_0::rsi_signal_D::rule_based_3": frame}
            return result
        return frame

    monkeypatch.setattr(TFPortfolio, "predict_from_cache", _tf_predict)

    cache_root = str(CentralCacheStore.get_instance().cache_dir)
    materialize_global_portfolio_predictions(
        portfolio=portfolio,
        query=query,
        portfolio_id="portfolio_b",
        world=PortfolioWorld.TRAIN,
        scope=ArtifactScope.LIVE,
        vault_root=str(vault_root),
        cache_root=cache_root,
    )

    active_base_model_path = (
        Path(cache_root)
        / "materialized"
        / "live"
        / "base_models"
        / "D"
        / "sample_long"
        / "rsi_signal_D__rule_based_3.parquet"
    )
    inactive_base_model_path = (
        Path(cache_root)
        / "materialized"
        / "live"
        / "base_models"
        / "D"
        / "obsolete_long"
        / "obsolete_signal_D__rule_based_3.parquet"
    )
    inactive_base_model_path.parent.mkdir(parents=True, exist_ok=True)
    _portfolio_frame([0.9, 0.9, 0.9]).to_parquet(inactive_base_model_path, index=False)

    portfolio_path = Path(cache_root) / "materialized" / "live" / "portfolio" / "portfolio_b.parquet"
    assert portfolio_path.exists()

    cleanup = prune_inactive_base_model_materializations(
        vault_root=str(vault_root),
        scope=ArtifactScope.LIVE,
        cache_root=cache_root,
    )

    assert cleanup.deleted_files == 1
    assert active_base_model_path.exists()
    assert not inactive_base_model_path.exists()
    assert portfolio_path.exists()
