from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from nodes import BiasNode
from nodes.mean_reversion.rsi.rsi import RSI
from utils.cache.cache_manager import CacheManager
from utils.cache.central_cache import CentralCacheStore
from utils.cache.central_cache_models import ArtifactLifecycleState
from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle


def _source_frame(
    ticker: Ticker,
    timeframe: TimeFrame,
    start: str = "2024-01-01",
    periods: int = 12,
) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="D")
    return pd.DataFrame(
        {
            "datetime": dates,
            "open": [100.0 + idx for idx in range(periods)],
            "high": [101.0 + idx for idx in range(periods)],
            "low": [99.0 + idx for idx in range(periods)],
            "close": [100.5 + idx for idx in range(periods)],
            "volume": [1000.0 + idx for idx in range(periods)],
            "ticker": [ticker.name] * periods,
            "timeframe": [timeframe.name] * periods,
        }
    )


def _write_feature_ensemble(
    vault_root: Path,
    *,
    ensemble_name: str,
    tickers: list[str],
    module_name: str,
    params: dict[str, object],
    timeframes: list[str],
) -> str:
    ensemble_dir = vault_root / timeframes[0] / ensemble_name
    features_dir = ensemble_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    (ensemble_dir / "ensemble_config.json").write_text(
        json.dumps(
            {
                "timeframe": timeframes[0],
                "ensemble_name": ensemble_name.replace("_long", ""),
                "direction": "long",
                "tickers": tickers,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (features_dir / f"{module_name}_signal_{timeframes[0]}.json").write_text(
        json.dumps(
            {
                "feature_name": f"{module_name}_signal_{timeframes[0]}",
                "created_at": "2026-03-22T00:00:00+00:00",
                "updated_at": "2026-03-22T00:00:00+00:00",
                "bias_node_spec": {
                    "module_name": module_name,
                    "timeframes": timeframes,
                },
                "tickers": tickers,
                "base_models": [
                    {
                        "model_id": "rule_based_3",
                        "model_name": f"{module_name}_signal_{timeframes[0]}::rule_based_3",
                        "bias_node_params": params,
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


@pytest.fixture
def isolated_cache(tmp_path: Path) -> None:
    CentralCacheStore.reset()
    CentralCacheStore._instance = CentralCacheStore(cache_dir=tmp_path / "central_cache")  # type: ignore[attr-defined]
    yield
    CentralCacheStore.reset()


@pytest.fixture
def source_candle_dir(tmp_path: Path) -> Path:
    source_dir = tmp_path / "ohlc_data"
    for ticker in [Ticker.ES, Ticker.NQ]:
        ticker_dir = source_dir / ticker.name
        ticker_dir.mkdir(parents=True, exist_ok=True)
        _source_frame(ticker, TimeFrame.D).to_parquet(ticker_dir / f"D_{ticker.name}.parquet")
    return source_dir


def _isolated_manager(source_candle_dir: Path) -> CacheManager:
    live_cache_dir = CentralCacheStore.get_instance().live_artifact_cache_dir
    return CacheManager(
        cache_dir=str(live_cache_dir),
        candle_dir=str(source_candle_dir),
    )


def _expected_rsi_frame(
    frame: pd.DataFrame,
    *,
    lookback: int,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    BiasNode._instances.clear()
    node = RSI(Ticker.ES, TimeFrame.D, lookback=lookback)
    outputs: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        candle = Candle.from_row(row)
        outputs.append(
            {
                "datetime": candle.datetime,
                "value": node.add_candle(candle)[0],
            }
        )

    result = pd.DataFrame(outputs).set_index("datetime")
    return result.loc[
        (result.index >= pd.Timestamp(start)) & (result.index <= pd.Timestamp(end))
    ].copy()


def test_ensure_vault_cache_coverage_builds_bias_and_ewsd_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    isolated_cache: None,
    source_candle_dir: Path,
    tmp_path: Path,
) -> None:
    vault_root = tmp_path / "vault"
    ensemble_dir = _write_feature_ensemble(
        vault_root,
        ensemble_name="sample_long",
        tickers=["ES"],
        module_name="rsi",
        params={"lookback": 2},
        timeframes=["D"],
    )
    monkeypatch.chdir(tmp_path)

    manager = _isolated_manager(source_candle_dir)
    manager.bootstrap_source_candles(
        tickers=[Ticker.ES],
        timeframes=[TimeFrame.D],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 8),
    )
    summary = manager.ensure_vault_cache_coverage(
        [ensemble_dir],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 8),
    )

    assert summary["failed"] == 0
    assert summary["rebuilt"] == 2
    assert summary["portfolio_tickers"] == ["ES"]

    store = CentralCacheStore.get_instance()
    bias_record = store.describe_artifact(
        manager._artifact_descriptor("rsi", {"lookback": 2}, Ticker.ES, TimeFrame.D)
    )
    ewsd_record = store.describe_artifact(
        manager._artifact_descriptor("ewsd", {"long_run_window": 2520}, Ticker.ES, TimeFrame.D)
    )
    assert bias_record is not None
    assert ewsd_record is not None


def test_ensure_vault_cache_coverage_refreshes_stale_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    isolated_cache: None,
    source_candle_dir: Path,
    tmp_path: Path,
) -> None:
    vault_root = tmp_path / "vault"
    ensemble_dir = _write_feature_ensemble(
        vault_root,
        ensemble_name="sample_long",
        tickers=["ES"],
        module_name="rsi",
        params={"lookback": 2},
        timeframes=["D"],
    )
    monkeypatch.chdir(tmp_path)

    manager = _isolated_manager(source_candle_dir)
    manager.bootstrap_source_candles(
        tickers=[Ticker.ES],
        timeframes=[TimeFrame.D],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 8),
    )
    manager.ensure_vault_cache_coverage(
        [ensemble_dir],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 8),
    )

    store = CentralCacheStore.get_instance()
    store.upsert_candles(
        Ticker.ES,
        TimeFrame.D,
        _source_frame(Ticker.ES, TimeFrame.D, periods=14),
    )
    stale_record = store.describe_artifact(
        manager._artifact_descriptor("rsi", {"lookback": 2}, Ticker.ES, TimeFrame.D)
    )
    assert stale_record is not None
    assert stale_record.lifecycle_state is ArtifactLifecycleState.STALE

    refreshed = manager.ensure_vault_cache_coverage(
        [ensemble_dir],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 10),
    )

    assert refreshed["failed"] == 0
    assert any(detail["status"] == "rebuilt" for detail in refreshed["details"])
    fresh_record = store.describe_artifact(
        manager._artifact_descriptor("rsi", {"lookback": 2}, Ticker.ES, TimeFrame.D)
    )
    assert fresh_record is not None
    assert fresh_record.lifecycle_state is ArtifactLifecycleState.FRESH


def test_ensure_vault_cache_coverage_cold_rebuilds_with_warmup_history(
    monkeypatch: pytest.MonkeyPatch,
    isolated_cache: None,
    source_candle_dir: Path,
    tmp_path: Path,
) -> None:
    vault_root = tmp_path / "vault"
    ensemble_dir = _write_feature_ensemble(
        vault_root,
        ensemble_name="sample_long",
        tickers=["ES"],
        module_name="rsi",
        params={"lookback": 2},
        timeframes=["D"],
    )
    monkeypatch.chdir(tmp_path)

    manager = _isolated_manager(source_candle_dir)
    manager.bootstrap_source_candles(
        tickers=[Ticker.ES],
        timeframes=[TimeFrame.D],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 12),
    )

    summary = manager.ensure_vault_cache_coverage(
        [ensemble_dir],
        start_date=datetime(2024, 1, 5),
        end_date=datetime(2024, 1, 8),
    )

    rebuild_detail = next(
        detail
        for detail in summary["details"]
        if detail["module_name"] == "rsi" and detail["status"] == "rebuilt"
    )
    assert rebuild_detail["cold_rebuild_start"] < rebuild_detail["effective_start"]

    store = CentralCacheStore.get_instance()
    artifact_df = store.read_artifact(
        manager._artifact_descriptor("rsi", {"lookback": 2}, Ticker.ES, TimeFrame.D)
    )
    expected = _expected_rsi_frame(
        _source_frame(Ticker.ES, TimeFrame.D, periods=12),
        lookback=2,
        start=datetime(2024, 1, 5),
        end=datetime(2024, 1, 8),
    )

    pd.testing.assert_frame_equal(artifact_df, expected)


def test_ensure_vault_cache_coverage_fails_without_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    isolated_cache: None,
    source_candle_dir: Path,
    tmp_path: Path,
) -> None:
    vault_root = tmp_path / "vault"
    ensemble_dir = _write_feature_ensemble(
        vault_root,
        ensemble_name="sample_long",
        tickers=["ES"],
        module_name="rsi",
        params={"lookback": 2},
        timeframes=["D"],
    )
    monkeypatch.chdir(tmp_path)

    manager = _isolated_manager(source_candle_dir)
    summary = manager.ensure_vault_cache_coverage(
        [ensemble_dir],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 8),
    )

    assert summary["failed"] == 2
    assert all(detail["status"] == "failed" for detail in summary["details"])


def test_ensure_vault_cache_coverage_requires_exact_candle_coverage(
    monkeypatch: pytest.MonkeyPatch,
    isolated_cache: None,
    source_candle_dir: Path,
    tmp_path: Path,
) -> None:
    vault_root = tmp_path / "vault"
    ensemble_dir = _write_feature_ensemble(
        vault_root,
        ensemble_name="sample_long",
        tickers=["ES"],
        module_name="rsi",
        params={"lookback": 2},
        timeframes=["D"],
    )
    monkeypatch.chdir(tmp_path)

    manager = _isolated_manager(source_candle_dir)
    manager.bootstrap_source_candles(
        tickers=[Ticker.ES],
        timeframes=[TimeFrame.D],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 12),
    )
    summary = manager.ensure_vault_cache_coverage(
        [ensemble_dir],
        start_date=datetime(2023, 12, 25),
        end_date=datetime(2024, 1, 20),
    )

    assert summary["failed"] == 2
    assert all(detail["status"] == "failed" for detail in summary["details"])


def test_ensure_vault_cache_coverage_allows_non_trading_day_boundary_gaps(
    monkeypatch: pytest.MonkeyPatch,
    isolated_cache: None,
    tmp_path: Path,
) -> None:
    vault_root = tmp_path / "vault"
    ensemble_dir = _write_feature_ensemble(
        vault_root,
        ensemble_name="sample_long",
        tickers=["ES"],
        module_name="rsi",
        params={"lookback": 2},
        timeframes=["D"],
    )
    monkeypatch.chdir(tmp_path)

    source_dir = tmp_path / "ohlc_data"
    ticker_dir = source_dir / Ticker.ES.name
    ticker_dir.mkdir(parents=True, exist_ok=True)
    _source_frame(Ticker.ES, TimeFrame.D, start="2024-01-03", periods=8).to_parquet(
        ticker_dir / f"D_{Ticker.ES.name}.parquet"
    )

    manager = _isolated_manager(source_dir)
    manager.bootstrap_source_candles(
        tickers=[Ticker.ES],
        timeframes=[TimeFrame.D],
        start_date=datetime(2024, 1, 3),
        end_date=datetime(2024, 1, 10),
    )
    summary = manager.ensure_vault_cache_coverage(
        [ensemble_dir],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 10),
    )

    assert summary["failed"] == 0
    assert all(
        detail["effective_start"].startswith("2024-01-03")
        for detail in summary["details"]
        if detail["status"] in {"rebuilt", "fresh"}
    )


def test_ensure_bias_cache_coverage_refreshes_stale_artifacts(
    isolated_cache: None,
    source_candle_dir: Path,
) -> None:
    manager = _isolated_manager(source_candle_dir)
    manager.bootstrap_source_candles(
        tickers=[Ticker.ES],
        timeframes=[TimeFrame.D],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 8),
    )
    manager.ensure_bias_cache_coverage(
        bias_node_specs=[
            {
                "module_name": "rsi",
                "params": {"lookback": 2},
                "timeframes": [TimeFrame.D],
            }
        ],
        tickers=[Ticker.ES],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 8),
    )

    store = CentralCacheStore.get_instance()
    store.upsert_candles(
        Ticker.ES,
        TimeFrame.D,
        _source_frame(Ticker.ES, TimeFrame.D, periods=14),
    )
    stale_record = store.describe_artifact(
        manager._artifact_descriptor("rsi", {"lookback": 2}, Ticker.ES, TimeFrame.D)
    )
    assert stale_record is not None
    assert stale_record.lifecycle_state is ArtifactLifecycleState.STALE

    refreshed = manager.ensure_bias_cache_coverage(
        bias_node_specs=[
            {
                "module_name": "rsi",
                "params": {"lookback": 2},
                "timeframes": [TimeFrame.D],
            }
        ],
        tickers=[Ticker.ES],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 10),
    )

    assert refreshed["failed"] == 0
    assert any(detail["status"] == "rebuilt" for detail in refreshed["details"])
    fresh_record = store.describe_artifact(
        manager._artifact_descriptor("rsi", {"lookback": 2}, Ticker.ES, TimeFrame.D)
    )
    assert fresh_record is not None
    assert fresh_record.lifecycle_state is ArtifactLifecycleState.FRESH


def test_ensure_bias_cache_coverage_fails_without_bootstrap(
    isolated_cache: None,
    source_candle_dir: Path,
) -> None:
    manager = _isolated_manager(source_candle_dir)
    summary = manager.ensure_bias_cache_coverage(
        bias_node_specs=[
            {
                "module_name": "rsi",
                "params": {"lookback": 2},
                "timeframes": [TimeFrame.D],
            }
        ],
        tickers=[Ticker.ES],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 8),
    )

    assert summary["failed"] == 2
    assert all(detail["status"] == "failed" for detail in summary["details"])


def test_ensure_bias_cache_coverage_requires_exact_candle_coverage(
    isolated_cache: None,
    source_candle_dir: Path,
) -> None:
    manager = CacheManager(candle_dir=str(source_candle_dir))
    manager.bootstrap_source_candles(
        tickers=[Ticker.ES],
        timeframes=[TimeFrame.D],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 12),
    )
    summary = manager.ensure_bias_cache_coverage(
        bias_node_specs=[
            {
                "module_name": "rsi",
                "params": {"lookback": 2},
                "timeframes": [TimeFrame.D],
            }
        ],
        tickers=[Ticker.ES],
        start_date=datetime(2023, 12, 25),
        end_date=datetime(2024, 1, 20),
    )

    assert summary["failed"] == 2
    assert all(detail["status"] == "failed" for detail in summary["details"])


def test_ensure_bias_cache_coverage_allows_monthly_boundary_gaps(
    isolated_cache: None,
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "ohlc_data"
    ticker_dir = source_dir / Ticker.ES.name
    ticker_dir.mkdir(parents=True, exist_ok=True)
    monthly_dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"])
    pd.DataFrame(
        {
            "datetime": monthly_dates,
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000.0, 1001.0, 1002.0],
            "ticker": [Ticker.ES.name] * len(monthly_dates),
            "timeframe": [TimeFrame.M.name] * len(monthly_dates),
        }
    ).to_parquet(ticker_dir / f"{TimeFrame.M.name}_{Ticker.ES.name}.parquet")

    manager = _isolated_manager(source_dir)
    manager.bootstrap_source_candles(
        tickers=[Ticker.ES],
        timeframes=[TimeFrame.M],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 3, 15),
    )
    summary = manager.ensure_bias_cache_coverage(
        bias_node_specs=[
            {
                "module_name": "rsi",
                "params": {"lookback": 2},
                "timeframes": [TimeFrame.M],
            }
        ],
        tickers=[Ticker.ES],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 3, 15),
        include_daily_ewsd=False,
    )

    assert summary["failed"] == 0
    rebuilt = [
        detail for detail in summary["details"] if detail["status"] in {"rebuilt", "fresh"}
    ]
    assert rebuilt
    assert all(detail["effective_start"].startswith("2024-01-31") for detail in rebuilt)
    assert all(detail["effective_end"].startswith("2024-02-29") for detail in rebuilt)


def test_ensure_bias_cache_coverage_preserves_monotonic_source_revision(
    source_candle_dir: Path,
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "central_cache"
    live_cache_dir = cache_root / "artifacts" / "live"
    CentralCacheStore.reset()
    manager = CacheManager(
        cache_dir=str(live_cache_dir),
        candle_dir=str(source_candle_dir),
    )
    spec = {
        "module_name": "rsi",
        "params": {"lookback": 2},
        "timeframes": [TimeFrame.D],
    }

    manager.bootstrap_source_candles(
        tickers=[Ticker.ES],
        timeframes=[TimeFrame.D],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 10),
    )
    initial = manager.ensure_bias_cache_coverage(
        bias_node_specs=[spec],
        tickers=[Ticker.ES],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 8),
    )
    assert initial["failed"] == 0

    store = manager._central_cache_store()
    descriptor = manager._artifact_descriptor("rsi", {"lookback": 2}, Ticker.ES, TimeFrame.D)
    artifact_path = store._artifact_path(descriptor)
    assert artifact_path is not None
    metadata_path = artifact_path.with_suffix(f"{artifact_path.suffix}.meta.json")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["source_revision"] = int(payload.get("source_revision", 0)) + 5
    payload["lifecycle_state"] = ArtifactLifecycleState.STALE.value
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    CentralCacheStore.reset()
    manager = CacheManager(
        cache_dir=str(live_cache_dir),
        candle_dir=str(source_candle_dir),
    )
    refreshed = manager.ensure_bias_cache_coverage(
        bias_node_specs=[spec],
        tickers=[Ticker.ES],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 10),
    )

    assert refreshed["failed"] == 0
    assert any(detail["status"] == "rebuilt" for detail in refreshed["details"])
    refreshed_record = manager._central_cache_store().describe_artifact(descriptor)
    assert refreshed_record is not None
    assert refreshed_record.lifecycle_state is ArtifactLifecycleState.FRESH
    assert refreshed_record.source_revision == payload["source_revision"]
