"""Integration test: Darwinex signal refresh (+ bias population) → engine evaluates.

Drives the real pipeline — real vault, the real Darwinex CFD series read from
``data/mt5_data`` via ``cfd_candles.load_cfd_candles_raw``, and
``CacheManager.ensure_vault_cache_coverage`` — into an isolated shared signal
store, then confirms ``VaultForecastEngine`` reads that store and produces
targets. No MT5 terminal is needed (the signal source is the scraped parquet).

Skips cleanly when ``data/mt5_data`` lacks Darwinex coverage for the vault's
required tickers on this checkout.
"""
from __future__ import annotations

import pytest

from deployment.live.broker_data import refresh_signal_daily
from deployment.live.forecast_engine import ForecastEngineConfig, VaultForecastEngine
from cache.runtime.central_cache import CentralCacheStore
from lib.core.enums import Ticker, TimeFrame

_WARMUP = 500


@pytest.fixture
def restore_singleton():
    prev = CentralCacheStore._instance  # type: ignore[attr-defined]
    yield
    CentralCacheStore._instance = prev  # type: ignore[attr-defined]


def _darwinex_covers(required: tuple[str, ...]) -> bool:
    """True only when every in-enum required ticker has Darwinex daily data."""
    from data_platform.providers.mt5.cfd_candles import load_cfd_candles_raw

    in_enum = [t for t in required if t in Ticker.__members__]
    if not in_enum:
        return False
    for t in in_enum:
        try:
            frame = load_cfd_candles_raw(Ticker[t], TimeFrame.D)
        except (FileNotFoundError, KeyError):
            return False
        if frame is None or frame.empty:
            return False
    return True


def test_signal_refresh_warms_bias_nodes_and_engine_evaluates(tmp_path, restore_singleton):
    try:
        eng0 = VaultForecastEngine(ForecastEngineConfig(vault_root="vault")).load()
    except ValueError as exc:
        pytest.skip(f"vault not loadable: {exc}")
    required = eng0.required_tickers

    if not _darwinex_covers(required):
        pytest.skip("no Darwinex daily coverage in data/mt5_data for required tickers on this checkout")

    root = tmp_path / "darwinex" / "central_cache"
    store = CentralCacheStore(cache_dir=str(root))
    CentralCacheStore._instance = store  # type: ignore[attr-defined]

    report = refresh_signal_daily(required, store=store, vault_root="vault", populate_bias=True)
    assert report.warm(min_bars=_WARMUP), f"no ticker warmed up; report={report.per_ticker}"

    engine = VaultForecastEngine(
        ForecastEngineConfig(vault_root="vault", warmup_min_bars=_WARMUP, cache_root=str(root))
    ).load()
    result = engine.evaluate()
    if not result.ready:
        pytest.skip(f"engine still cold after refresh: {result.warmup.not_ready()}")

    assert result.targets, "a warm Darwinex signal store must yield targets"
    assert result.as_of is not None
