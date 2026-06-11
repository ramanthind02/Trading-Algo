"""Single Darwinex signal feed, shared by all execution brokers.

The live signal is generated from ONE source of truth — the Darwinex CFD series
— regardless of which broker executes the orders. Each broker still runs in its
OWN process (MT5 is one-terminal-per-process), but every process binds the
central-cache singleton to the SAME shared namespace
(``data/broker_cache/darwinex/central_cache``) and populates it from the scraped
Darwinex daily/monthly bars via :func:`cfd_candles.load_cfd_candles_raw` — the
identical loader research and validation use. The bias artifacts (incl. EWSD vol)
are then computed on that Darwinex series so the forecast engine reads one
identical signal for every execution venue.

This reads the parquet files the daily Darwinex scraper writes under
``data/mt5_data`` — it needs NO MT5 connection (the signal is always the
do-not-disturb Darwinex series). Cross-broker price differences therefore affect
execution sizing/fills only, never the signal.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

# (ticker, timeframe) -> raw OHLCV frame [datetime, open, high, low, close, volume]
LoadCandles = Callable[[object, "TimeFrame"], pd.DataFrame]

from data_platform.providers.mt5 import brokers
from cache.runtime.cache_paths import project_root
from cache.runtime.central_cache import CentralCacheStore
from lib.core.enums import Ticker, TimeFrame

logger = logging.getLogger(__name__)

# The single source of truth for live signals across all CFD brokers.
SIGNAL_BROKER = "darwinex"


@dataclass(frozen=True)
class TickerRefresh:
    ticker: str
    total_bars: int
    source: str  # darwinex | missing


@dataclass(frozen=True)
class SignalRefreshReport:
    per_ticker: tuple[TickerRefresh, ...]

    def warm(self, min_bars: int) -> tuple[str, ...]:
        return tuple(t.ticker for t in self.per_ticker if t.total_bars >= min_bars)

    def cold(self, min_bars: int) -> tuple[str, ...]:
        return tuple(t.ticker for t in self.per_ticker if 0 < t.total_bars < min_bars)


def broker_cache_root(broker: str) -> Path:
    """Per-broker persisted cache root (under the gitignored ``data/`` tree)."""
    return project_root() / "data" / "broker_cache" / broker / "central_cache"


def signal_cache_root() -> Path:
    """The single shared signal-cache root (Darwinex), read by every broker."""
    return broker_cache_root(SIGNAL_BROKER)


def bind_broker_cache(broker: str) -> CentralCacheStore:
    """Point the ``CentralCacheStore`` singleton at ``broker``'s namespace.

    Mirrors ``CacheManager._central_cache_store`` — sets ``_instance`` so all
    downstream cache reads/writes (engine, CacheManager) hit that store.
    """
    root = broker_cache_root(broker)
    store = CentralCacheStore(cache_dir=str(root))
    CentralCacheStore._instance = store  # type: ignore[attr-defined]
    logger.info("Bound central cache to %s namespace at %s", broker, root)
    return store


def bind_signal_cache() -> CentralCacheStore:
    """Bind the singleton to the shared Darwinex signal namespace."""
    return bind_broker_cache(SIGNAL_BROKER)


def _decorate(frame: pd.DataFrame, ticker: str, timeframe: TimeFrame) -> pd.DataFrame:
    """Add the ``ticker``/``timeframe`` columns the candle store expects."""
    out = frame.copy()
    out["ticker"] = ticker
    out["timeframe"] = timeframe
    return out


def refresh_signal_daily(
    canonicals: tuple[str, ...],
    *,
    store: CentralCacheStore,
    vault_root: str = "vault",
    active_timeframes: tuple[TimeFrame, ...] = (TimeFrame.D, TimeFrame.M),
    populate_bias: bool = True,
    load_fn: LoadCandles | None = None,
) -> SignalRefreshReport:
    """Populate the shared signal cache from the Darwinex CFD series.

    ``store`` must already be the bound signal singleton (see
    :func:`bind_signal_cache`). For each required canonical ticker, load the
    Darwinex daily (and monthly) bars via ``load_fn`` and persist them, then
    (optionally) populate bias artifacts over the full coverage window. Tickers
    with no Darwinex data are skipped (reported as ``missing``) rather than
    aborting — the engine's warmup gate keeps them flat.

    ``load_fn`` defaults to
    ``data_platform.providers.mt5.cfd_candles.load_cfd_candles_raw``; inject a
    stub in tests. It always resolves to Darwinex symbols (``DEFAULT_BROKER``),
    so this is broker-independent and reads only the scraped ``data/mt5_data``
    parquet files (no MT5 connection).
    """
    import json as _json
    from datetime import datetime, timezone

    if load_fn is None:
        from data_platform.providers.mt5.cfd_candles import load_cfd_candles_raw as load_fn

    # ── Registry: start job_run (resilience-wrapped) ──────────────────────────
    _reg_conn = None
    _job_run_id = None
    try:
        from data_platform.registry import db as _reg_db, writer as _reg_writer
        _reg_conn = _reg_db.connect()
        with _reg_db.transaction(_reg_conn):
            _job_run_id = _reg_writer.record_job_run(
                _reg_conn,
                _reg_writer.JobRun(
                    job_name="signal_refresh",
                    started_at=datetime.now(timezone.utc).isoformat(),
                    args_json=_json.dumps({
                        "canonicals": list(canonicals),
                        "vault_root": vault_root,
                        "populate_bias": populate_bias,
                    }),
                ),
            )
    except Exception as _exc:  # noqa: BLE001
        logger.warning("registry: could not start signal_refresh job_run: %s", _exc)

    _exit_code = 0
    try:
        reports: list[TickerRefresh] = []
        starts: list[pd.Timestamp] = []
        ends: list[pd.Timestamp] = []

        for canonical in canonicals:
            if canonical not in Ticker.__members__:
                continue
            ticker = Ticker[canonical]

            try:
                daily = load_fn(ticker, TimeFrame.D)
            except (FileNotFoundError, KeyError) as exc:
                logger.warning("No Darwinex signal data for %s: %s", canonical, exc)
                reports.append(TickerRefresh(canonical, 0, "missing"))
                continue
            if daily is None or daily.empty:
                reports.append(TickerRefresh(canonical, 0, "missing"))
                continue

            store.set_candles(ticker, TimeFrame.D, _decorate(daily, canonical, TimeFrame.D))
            for tf in active_timeframes:
                if tf is TimeFrame.D:
                    continue
                try:
                    frame = load_fn(ticker, tf)
                except (FileNotFoundError, KeyError, ValueError) as exc:
                    logger.warning("No Darwinex %s data for %s: %s", tf.name, canonical, exc)
                    continue
                if frame is not None and not frame.empty:
                    store.set_candles(ticker, tf, _decorate(frame, canonical, tf))

            reports.append(TickerRefresh(canonical, len(daily), "darwinex"))
            starts.append(pd.to_datetime(daily["datetime"]).min())
            ends.append(pd.to_datetime(daily["datetime"]).max())

        if populate_bias and starts:
            _populate_bias(vault_root, min(starts).to_pydatetime(), max(ends).to_pydatetime(), store)

        report = SignalRefreshReport(per_ticker=tuple(reports))
        logger.info(
            "Darwinex signal refresh: %s",
            ", ".join(f"{t.ticker}={t.total_bars}({t.source})" for t in report.per_ticker),
        )
    except Exception:
        _exit_code = 1
        raise
    finally:
        # ── Registry: finish job_run (resilience-wrapped) ─────────────────────
        if _reg_conn is not None and _job_run_id is not None:
            try:
                from data_platform.registry import db as _reg_db2, writer as _reg_writer2
                _warm = [t.ticker for t in reports if t.total_bars > 0]
                _miss = [t.ticker for t in reports if t.total_bars == 0]
                _cov = _json.dumps({"warm": _warm, "missing": _miss,
                                    "total_bars": sum(t.total_bars for t in reports)})
                with _reg_db2.transaction(_reg_conn):
                    _reg_writer2.finish_job_run(
                        _reg_conn, _job_run_id,
                        exit_code=_exit_code,
                        rows_written=sum(t.total_bars for t in reports),
                        coverage_json=_cov,
                        error_text=None,
                    )
            except Exception as _exc2:  # noqa: BLE001
                logger.warning("registry: could not finish signal_refresh job_run: %s", _exc2)
            try:
                _reg_conn.close()
            except Exception:  # noqa: BLE001
                pass

    return report


def _populate_bias(
    vault_root: str, start: datetime, end: datetime, store: CentralCacheStore
) -> None:
    """Populate bias artifacts (incl. EWSD vol) into the signal store.

    ``CacheManager`` resolves its own ``CentralCacheStore`` from ``cache_dir``, so
    it is pointed at THIS store's live-artifact dir (whose parent is the shared
    signal-cache root) — otherwise it would write to the shared default cache. The
    rebuild reads candles via ``store.query_candles`` (the Darwinex bars we just
    wrote), so artifacts are computed from the Darwinex series.
    """
    from ensemble.portfolio_impl.vault_portfolio_loader import discover_ensemble_dirs_in_vault
    from cache.runtime.cache_manager import CacheManager
    from lib.core.vault_paths import resolve_vault_root

    root = str(resolve_vault_root(vault_root))
    dirs = list(discover_ensemble_dirs_in_vault(root, (TimeFrame.D, TimeFrame.M)))
    if not dirs:
        logger.warning("No ensemble dirs under %s; skipping bias population", root)
        return
    CacheManager(cache_dir=str(store.live_artifact_cache_dir)).ensure_vault_cache_coverage(
        vault_ensemble_dirs=dirs,
        start_date=start,
        end_date=end,
        refresh_mode="missing_stale_only",
    )


__all__ = [
    "SIGNAL_BROKER",
    "SignalRefreshReport",
    "TickerRefresh",
    "bind_broker_cache",
    "bind_signal_cache",
    "broker_cache_root",
    "refresh_signal_daily",
    "signal_cache_root",
]
