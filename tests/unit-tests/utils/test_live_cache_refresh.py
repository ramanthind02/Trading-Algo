from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

import lib.cache.runtime.live_cache_refresh as live_refresh
from lib.cache.runtime.bootstrap_source_candles import bootstrap_source_candles
from lib.cache.runtime.central_cache import CentralCacheStore
from lib.cache.runtime.central_cache_models import ArtifactScope
from lib.cache.runtime.cache_manager import CacheManager
from lib.core.enums import Ticker, TimeFrame


def _frame(start: datetime, periods: int, freq: timedelta, base: float = 100.0) -> pd.DataFrame:
    index = pd.DatetimeIndex([start + i * freq for i in range(periods)])
    return pd.DataFrame(
        {
            "open": [base + i for i in range(periods)],
            "high": [base + i + 1 for i in range(periods)],
            "low": [base + i - 1 for i in range(periods)],
            "close": [base + i + 0.5 for i in range(periods)],
            "volume": [1_000 + i for i in range(periods)],
        },
        index=index,
    )


def _source_frame(
    ticker: Ticker,
    timeframe: TimeFrame,
    start: str = "2024-01-01",
    periods: int = 5,
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


def _status_path() -> Path:
    return Path(CentralCacheStore.get_instance().cache_dir) / "live_refresh" / "last_run.json"


def _write_manifest(
    path: Path,
    *,
    enabled: bool = True,
    active_portfolios: list[dict[str, object]] | None = None,
    active_ensemble_dirs: list[str] | None = None,
    debounce_seconds: float = 0.05,
) -> Path:
    payload = {
        "version": "1",
        "enabled": enabled,
        "vault_root": "vault",
        "debounce_seconds": debounce_seconds,
        "active_ensemble_dirs": active_ensemble_dirs or ["vault/M/buy_hold/buy_hold_long"],
        "active_portfolios": active_portfolios
        or [
            {
                "name": "live_es",
                "portfolio_id": "portfolio_live",
                "tickers": ["ES"],
                "timeframes": ["D"],
                "volatility_timeframe": "D",
                "replay_window_days": 62,
                "research_run_id": "live_test",
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _completed_summary(
    manifest_path: Path,
    dirty_keys: tuple[str, ...],
) -> live_refresh.LiveCacheRefreshRunSummary:
    now = datetime.now(UTC).isoformat()
    return live_refresh.LiveCacheRefreshRunSummary(
        status="completed",
        manifest_path=str(manifest_path),
        dirty_keys=dirty_keys,
        affected_portfolios=("live_es",),
        refreshed_ensemble_dirs=("vault/M/buy_hold/buy_hold_long",),
        bias_refresh_summary={"failed": 0},
        materialized_portfolios=(),
        started_at=now,
        finished_at=now,
    )


@pytest.fixture(autouse=True)
def _isolated_runtime(tmp_path: Path) -> None:
    CentralCacheStore.reset()
    CentralCacheStore._instance = CentralCacheStore(cache_dir=tmp_path / "central_cache")  # type: ignore[attr-defined]
    live_refresh.reset_live_cache_refresh_orchestrator()
    yield
    CentralCacheStore.reset()


def test_run_live_cache_refresh_now_skips_when_manifest_is_missing(tmp_path: Path) -> None:
    summary = live_refresh.run_live_cache_refresh_now(manifest_path=str(tmp_path / "missing.json"))

    assert summary.status == "skipped"
    assert summary.error == "manifest_missing"
    assert json.loads(_status_path().read_text(encoding="utf-8"))["error"] == "manifest_missing"


def test_run_live_cache_refresh_now_skips_when_manifest_is_disabled(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path / "live_cache_refresh.json", enabled=False)

    summary = live_refresh.run_live_cache_refresh_now(manifest_path=str(manifest_path))

    assert summary.status == "skipped"
    assert summary.error == "manifest_disabled"
    assert json.loads(_status_path().read_text(encoding="utf-8"))["error"] == "manifest_disabled"


def test_select_affected_portfolios_and_latest_common_end(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path / "live_cache_refresh.json",
        active_portfolios=[
            {
                "name": "portfolio_es",
                "portfolio_id": "portfolio_es",
                "tickers": ["ES"],
                "timeframes": ["D"],
                "volatility_timeframe": "D",
                "replay_window_days": 62,
                "research_run_id": "run_es",
            },
            {
                "name": "portfolio_multi",
                "portfolio_id": "portfolio_multi",
                "tickers": ["ES", "NQ"],
                "timeframes": ["D", "W"],
                "volatility_timeframe": "D",
                "replay_window_days": 62,
                "research_run_id": "run_multi",
            },
        ],
    )
    manifest = live_refresh.load_live_cache_refresh_manifest(str(manifest_path))
    assert manifest is not None

    affected = live_refresh._select_affected_portfolios(  # type: ignore[attr-defined]
        manifest,
        {(Ticker.NQ, TimeFrame.W)},
    )
    assert [portfolio.name for portfolio in affected] == ["portfolio_multi"]

    store = CentralCacheStore.get_instance()
    store.set_candles(Ticker.ES, TimeFrame.D, _frame(datetime(2024, 1, 1), 5, timedelta(days=1)))
    store.set_candles(Ticker.ES, TimeFrame.W, _frame(datetime(2024, 1, 1), 4, timedelta(days=7)))
    store.set_candles(Ticker.NQ, TimeFrame.D, _frame(datetime(2024, 1, 1), 4, timedelta(days=1)))
    store.set_candles(Ticker.NQ, TimeFrame.W, _frame(datetime(2024, 1, 1), 3, timedelta(days=7)))

    common_end = live_refresh._latest_common_end(store, manifest.active_portfolios[1])  # type: ignore[attr-defined]
    assert common_end == datetime(2024, 1, 4)


def test_live_refresh_orchestrator_coalesces_multiple_live_writes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path / "live_cache_refresh.json")
    monkeypatch.setattr(
        live_refresh,
        "default_live_cache_refresh_manifest_path",
        lambda: manifest_path,
    )

    calls: list[tuple[str, ...]] = []

    def _fake_cycle(
        incoming_manifest_path: str | None = None,
        dirty_keys: object = None,
    ) -> live_refresh.LiveCacheRefreshRunSummary:
        del incoming_manifest_path
        dirty = live_refresh._dirty_key_strings(live_refresh._normalize_dirty_keys(dirty_keys))  # type: ignore[attr-defined]
        calls.append(dirty)
        return _completed_summary(manifest_path, dirty)

    monkeypatch.setattr(live_refresh, "_run_live_cache_refresh_cycle", _fake_cycle)

    store = CentralCacheStore.get_instance()
    store.set_candles(Ticker.ES, TimeFrame.D, _frame(datetime(2024, 1, 1), 3, timedelta(days=1)))
    store.upsert_candles(Ticker.ES, TimeFrame.D, _frame(datetime(2024, 1, 3), 1, timedelta(days=1), base=250.0))

    orchestrator = live_refresh.get_live_cache_refresh_orchestrator()
    assert orchestrator.wait_for_idle(timeout=3.0)
    assert calls == [("ES:D",)]


def test_bootstrap_source_candles_emits_one_coalesced_refresh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(
        tmp_path / "live_cache_refresh.json",
        active_portfolios=[
            {
                "name": "portfolio_multi",
                "portfolio_id": "portfolio_multi",
                "tickers": ["ES", "NQ"],
                "timeframes": ["D"],
                "volatility_timeframe": "D",
                "replay_window_days": 62,
                "research_run_id": "run_multi",
            }
        ],
    )
    monkeypatch.setattr(
        live_refresh,
        "default_live_cache_refresh_manifest_path",
        lambda: manifest_path,
    )

    source_dir = tmp_path / "ohlc"
    for ticker in (Ticker.ES, Ticker.NQ):
        ticker_dir = source_dir / ticker.name
        ticker_dir.mkdir(parents=True, exist_ok=True)
        _source_frame(ticker, TimeFrame.D).to_parquet(ticker_dir / f"{TimeFrame.D.name}_{ticker.name}.parquet")

    import lib.cache.runtime.bootstrap_source_candles as bootstrap_module

    monkeypatch.setattr(
        bootstrap_module,
        "CacheManager",
        lambda: CacheManager(
            cache_dir=str(CentralCacheStore.get_instance().live_artifact_cache_dir),
            candle_dir=str(source_dir),
        ),
    )

    calls: list[tuple[str, ...]] = []

    def _fake_cycle(
        incoming_manifest_path: str | None = None,
        dirty_keys: object = None,
    ) -> live_refresh.LiveCacheRefreshRunSummary:
        del incoming_manifest_path
        dirty = live_refresh._dirty_key_strings(live_refresh._normalize_dirty_keys(dirty_keys))  # type: ignore[attr-defined]
        calls.append(dirty)
        return _completed_summary(manifest_path, dirty)

    monkeypatch.setattr(live_refresh, "_run_live_cache_refresh_cycle", _fake_cycle)

    summary = bootstrap_source_candles(
        tickers=[Ticker.ES, Ticker.NQ],
        timeframes=[TimeFrame.D],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 5),
    )

    assert summary["failed"] == 0
    assert summary["success"] == 2
    orchestrator = live_refresh.get_live_cache_refresh_orchestrator()
    assert orchestrator.wait_for_idle(timeout=3.0)
    assert calls == [("ES:D", "NQ:D")]


def test_failed_live_refresh_cycle_keeps_dirty_keys_and_records_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path / "live_cache_refresh.json")
    monkeypatch.setattr(
        live_refresh,
        "default_live_cache_refresh_manifest_path",
        lambda: manifest_path,
    )

    def _failed_cycle(
        incoming_manifest_path: str | None = None,
        dirty_keys: object = None,
    ) -> live_refresh.LiveCacheRefreshRunSummary:
        del incoming_manifest_path
        dirty = live_refresh._dirty_key_strings(live_refresh._normalize_dirty_keys(dirty_keys))  # type: ignore[attr-defined]
        now = datetime.now(UTC).isoformat()
        summary = live_refresh.LiveCacheRefreshRunSummary(
            status="failed",
            manifest_path=str(manifest_path),
            dirty_keys=dirty,
            affected_portfolios=(),
            refreshed_ensemble_dirs=("vault/M/buy_hold/buy_hold_long",),
            bias_refresh_summary=None,
            materialized_portfolios=(),
            started_at=now,
            finished_at=now,
            error="boom",
        )
        live_refresh._write_status_file(summary)  # type: ignore[attr-defined]
        return summary

    monkeypatch.setattr(live_refresh, "_run_live_cache_refresh_cycle", _failed_cycle)

    store = CentralCacheStore.get_instance()
    store.set_candles(Ticker.ES, TimeFrame.D, _frame(datetime(2024, 1, 1), 3, timedelta(days=1)))

    orchestrator = live_refresh.get_live_cache_refresh_orchestrator()
    assert orchestrator.wait_for_idle(timeout=3.0)
    assert orchestrator.pending_dirty_keys() == ("ES:D",)

    status_payload = json.loads(_status_path().read_text(encoding="utf-8"))
    assert status_payload["status"] == "failed"
    assert status_payload["error"] == "boom"
    assert status_payload["dirty_keys"] == ["ES:D"]


def test_candle_write_does_not_schedule_refresh_for_research_scope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path / "live_cache_refresh.json")
    monkeypatch.setattr(
        live_refresh,
        "default_live_cache_refresh_manifest_path",
        lambda: manifest_path,
    )

    calls: list[tuple[str, ...]] = []

    def _fake_cycle(
        incoming_manifest_path: str | None = None,
        dirty_keys: object = None,
    ) -> live_refresh.LiveCacheRefreshRunSummary:
        del incoming_manifest_path
        dirty = live_refresh._dirty_key_strings(live_refresh._normalize_dirty_keys(dirty_keys))  # type: ignore[attr-defined]
        calls.append(dirty)
        return _completed_summary(manifest_path, dirty)

    monkeypatch.setattr(live_refresh, "_run_live_cache_refresh_cycle", _fake_cycle)

    store = CentralCacheStore.get_instance()
    store.set_candles(
        Ticker.ES,
        TimeFrame.D,
        _frame(datetime(2024, 1, 1), 3, timedelta(days=1)),
        scope=ArtifactScope.RESEARCH,
    )

    assert live_refresh.get_live_cache_refresh_orchestrator().wait_for_idle(timeout=0.2)
    assert calls == []
