from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ensemble.portfolio import (
    GlobalPortfolio,
    PortfolioCacheQuery,
    TFPortfolio,
    load_global_portfolio_snapshot,
)
from ensemble.vault_manager import load_ensemble_from_vault
from utils.cache.central_cache import CentralCacheStore
from utils.cache.central_cache_models import ArtifactDescriptor, ArtifactScope
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


def _build_snapshot_ready_portfolio(tmp_path: Path) -> tuple[GlobalPortfolio, PortfolioCacheQuery, Path]:
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
    tf_portfolio.idm_ = 1.0
    tf_portfolio.mean_return_correlation_ = 1.0
    tf_portfolio.instruments_ = ["ES"]
    tf_portfolio.is_fitted_ = True

    portfolio = GlobalPortfolio(tf_portfolios=[tf_portfolio], max_position_pct=2.0)
    portfolio.weight_layer.weights_ = {
        "__GLOBAL__": pd.Series({"ES::D::rsi_signal_D::rule_based_3": 1.0}, dtype=float)
    }
    portfolio.weight_layer.fdm_ = {"__GLOBAL__": 1.0}
    portfolio.weight_layer.model_names_ = {"__GLOBAL__": ["ES::D::rsi_signal_D::rule_based_3"]}
    portfolio.weight_layer.cluster_assignments_ = {
        "__GLOBAL__": {"ES::D::rsi_signal_D::rule_based_3": "cluster_1"}
    }
    portfolio.weight_layer.cluster_weights_ = {"__GLOBAL__": {"cluster_1": 1.0}}
    portfolio.weight_layer.cluster_metrics_ = {
        "__GLOBAL__": {
            "cluster_1": {
                "avg_positive_corr": 0.0,
                "ulcer_index": None,
                "score": None,
                "member_count": 1,
            }
        }
    }
    portfolio.weight_layer.mean_signal_correlation_ = {"__GLOBAL__": 1.0}
    portfolio.weight_layer.mean_cluster_correlation_ = {"__GLOBAL__": 1.0}
    portfolio.weight_layer.is_fitted_ = True
    portfolio.weight_layer_diagnostics_ = portfolio.weight_layer.get_diagnostics()
    portfolio.global_idm_ = 1.0
    portfolio.mean_instrument_return_correlation_ = 1.0
    portfolio.instruments_ = ["ES"]
    portfolio.global_tf_weights_compat_ = {"D": 1.0}
    portfolio.global_eligible_models_by_ticker_ = {"ES": {"rsi_signal_D::rule_based_3"}}
    portfolio.is_fitted_ = True

    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    query = PortfolioCacheQuery(
        tickers=("ES",),
        start=dates.min().to_pydatetime(),
        end=dates.max().to_pydatetime(),
        timeframes=(TimeFrame.D,),
    )
    _seed_cache(CentralCacheStore.get_instance(), dates)
    return portfolio, query, vault_root


def _patched_predict_vectors(self, candles_df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:  # noqa: ANN001
    tf_candles = candles_df[candles_df["timeframe"] == self.trading_timeframe].copy()
    return pd.DataFrame(
        {
            "ticker": tf_candles["ticker"].tolist(),
            "datetime": pd.to_datetime(tf_candles["datetime"]).tolist(),
            "model_name": ["rsi_signal_D::rule_based_3"] * len(tf_candles),
            "forecast": [0.25] * len(tf_candles),
            "signal": [1.0] * len(tf_candles),
            "timeframe": [self.trading_timeframe.name] * len(tf_candles),
        }
    )


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path: Path) -> None:
    CentralCacheStore.reset()
    CentralCacheStore._instance = CentralCacheStore(cache_dir=tmp_path / "central_cache")  # type: ignore[attr-defined]
    yield
    CentralCacheStore.reset()


def test_save_to_vault_hash_stable_for_same_semantic_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(TFPortfolio, "predict_base_model_vectors_from_candles", _patched_predict_vectors)

    portfolio_a, query_a, vault_root_a = _build_snapshot_ready_portfolio(tmp_path / "case_a")
    portfolio_b, query_b, vault_root_b = _build_snapshot_ready_portfolio(tmp_path / "case_b")

    portfolio_id_a = portfolio_a.save_to_vault(query_a.start, query_a.end, vault_root=str(vault_root_a))
    portfolio_id_b = portfolio_b.save_to_vault(query_b.start, query_b.end, vault_root=str(vault_root_b))

    assert portfolio_id_a == portfolio_id_b


def test_save_to_vault_hash_changes_when_fitted_state_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(TFPortfolio, "predict_base_model_vectors_from_candles", _patched_predict_vectors)

    baseline, query, vault_root = _build_snapshot_ready_portfolio(tmp_path / "baseline")
    changed, changed_query, changed_vault_root = _build_snapshot_ready_portfolio(tmp_path / "changed")
    changed.global_idm_ = 1.5

    baseline_id = baseline.save_to_vault(query.start, query.end, vault_root=str(vault_root))
    changed_id = changed.save_to_vault(
        changed_query.start,
        changed_query.end,
        vault_root=str(changed_vault_root),
    )

    assert baseline_id != changed_id


def test_load_global_portfolio_snapshot_restores_predictive_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(TFPortfolio, "predict_base_model_vectors_from_candles", _patched_predict_vectors)

    portfolio, query, vault_root = _build_snapshot_ready_portfolio(tmp_path / "roundtrip")
    original = portfolio.predict_from_cache(query)

    portfolio_id = portfolio.save_to_vault(query.start, query.end, vault_root=str(vault_root))
    restored = load_global_portfolio_snapshot(portfolio_id, vault_root=str(vault_root))
    replay = restored.predict_from_cache(query)

    pd.testing.assert_frame_equal(original.reset_index(drop=True), replay.reset_index(drop=True))
    assert restored.is_fitted_ is True
    assert restored.global_idm_ == pytest.approx(1.0)
    assert restored.portfolio_id_ == portfolio_id
