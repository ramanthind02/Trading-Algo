from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Iterable, Optional, Sequence

import pandas as pd

from .cache_paths import project_root, resolve_relative_path
from .central_cache_models import ArtifactScope
from lib.core.enums import Ticker, TimeFrame

logger = logging.getLogger(__name__)

_DEFAULT_DEBOUNCE_SECONDS = 2.0
_MANIFEST_VERSION = "1"


@dataclass(frozen=True)
class ActiveLivePortfolioConfig:
    name: str
    portfolio_id: str
    tickers: tuple[str, ...]
    timeframes: tuple[TimeFrame, ...]
    volatility_timeframe: TimeFrame
    replay_window_days: int
    research_run_id: str | None = None
    ensemble_dirs: tuple[str, ...] | None = None
    target_volatility: float = 0.20
    max_position_pct: float = 2.5
    idm_max: float = 2.5


@dataclass(frozen=True)
class LiveCacheRefreshManifest:
    version: str
    enabled: bool
    vault_root: str
    debounce_seconds: float
    active_ensemble_dirs: tuple[str, ...]
    active_portfolios: tuple[ActiveLivePortfolioConfig, ...]
    manifest_path: str


@dataclass(frozen=True)
class LiveCacheRefreshRunSummary:
    status: str
    manifest_path: str
    dirty_keys: tuple[str, ...]
    affected_portfolios: tuple[str, ...]
    refreshed_ensemble_dirs: tuple[str, ...]
    bias_refresh_summary: dict[str, Any] | None
    materialized_portfolios: tuple[dict[str, Any], ...]
    started_at: str
    finished_at: str
    error: str | None = None


def default_live_cache_refresh_manifest_path() -> Path:
    return project_root() / "deployment" / "config" / "live_cache_refresh.json"


def _status_path(cache_root: Path) -> Path:
    return cache_root / "live_refresh" / "last_run.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_bool(payload: dict[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"Live cache refresh manifest field '{key}' must be a bool")
    return value


def _require_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Live cache refresh manifest field '{key}' must be a non-empty string")
    return value


def _require_string_list(payload: dict[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key, [])
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"Live cache refresh manifest field '{key}' must be a list of strings")
    return tuple(dict.fromkeys(str(item) for item in value))


def _coerce_timeframe(value: str, *, field_name: str) -> TimeFrame:
    try:
        return TimeFrame[str(value)]
    except KeyError as exc:
        raise ValueError(f"Unknown timeframe '{value}' in {field_name}") from exc


def _validate_tickers(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    if not values:
        raise ValueError(f"{field_name} must contain at least one ticker")
    unknown = [ticker for ticker in values if ticker not in Ticker.__members__]
    if unknown:
        raise ValueError(f"Unknown ticker(s) in {field_name}: {sorted(unknown)}")
    return tuple(dict.fromkeys(str(ticker) for ticker in values))


def _parse_active_portfolio(payload: dict[str, Any]) -> ActiveLivePortfolioConfig:
    name = _require_string(payload, "name")
    portfolio_id = _require_string(payload, "portfolio_id")
    tickers = payload.get("tickers")
    if not isinstance(tickers, list) or any(not isinstance(item, str) for item in tickers):
        raise ValueError(f"Active live portfolio '{name}' must provide tickers as a list of strings")
    raw_timeframes = payload.get("timeframes")
    if not isinstance(raw_timeframes, list) or any(not isinstance(item, str) for item in raw_timeframes):
        raise ValueError(f"Active live portfolio '{name}' must provide timeframes as a list of strings")
    if not raw_timeframes:
        raise ValueError(f"Active live portfolio '{name}' must provide at least one timeframe")
    replay_window_days = payload.get("replay_window_days")
    if not isinstance(replay_window_days, int) or replay_window_days <= 0:
        raise ValueError(
            f"Active live portfolio '{name}' must provide replay_window_days as a positive integer"
        )
    research_run_id = payload.get("research_run_id")
    if research_run_id is not None and not isinstance(research_run_id, str):
        raise ValueError(f"Active live portfolio '{name}' research_run_id must be a string or null")

    raw_ensemble_dirs = payload.get("ensemble_dirs")
    ensemble_dirs: tuple[str, ...] | None
    if raw_ensemble_dirs is None:
        ensemble_dirs = None
    else:
        if not isinstance(raw_ensemble_dirs, list) or any(
            not isinstance(item, str) or not str(item).strip() for item in raw_ensemble_dirs
        ):
            raise ValueError(
                f"Active live portfolio '{name}' ensemble_dirs must be a list of non-empty strings when set"
            )
        ensemble_dirs = tuple(dict.fromkeys(str(item) for item in raw_ensemble_dirs))

    target_volatility = payload.get("target_volatility", 0.20)
    if not isinstance(target_volatility, (int, float)) or float(target_volatility) <= 0.0:
        raise ValueError(f"Active live portfolio '{name}' target_volatility must be a positive number")

    max_position_pct = payload.get("max_position_pct", 2.5)
    if not isinstance(max_position_pct, (int, float)) or float(max_position_pct) <= 0.0:
        raise ValueError(f"Active live portfolio '{name}' max_position_pct must be a positive number")

    idm_max = payload.get("idm_max", 2.5)
    if not isinstance(idm_max, (int, float)) or float(idm_max) <= 0.0:
        raise ValueError(f"Active live portfolio '{name}' idm_max must be a positive number")

    return ActiveLivePortfolioConfig(
        name=name,
        portfolio_id=portfolio_id,
        tickers=_validate_tickers(tickers, field_name=f"{name}.tickers"),
        timeframes=tuple(
            dict.fromkeys(
                _coerce_timeframe(value, field_name=f"{name}.timeframes")
                for value in raw_timeframes
            )
        ),
        volatility_timeframe=_coerce_timeframe(
            _require_string(payload, "volatility_timeframe"),
            field_name=f"{name}.volatility_timeframe",
        ),
        replay_window_days=replay_window_days,
        research_run_id=research_run_id,
        ensemble_dirs=ensemble_dirs,
        target_volatility=float(target_volatility),
        max_position_pct=float(max_position_pct),
        idm_max=float(idm_max),
    )


def load_live_cache_refresh_manifest(
    manifest_path: str | None = None,
) -> LiveCacheRefreshManifest | None:
    resolved_path = resolve_relative_path(
        manifest_path or default_live_cache_refresh_manifest_path(),
        project_root_fallback="project_candidate",
    )
    if not resolved_path.exists():
        return None

    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Live cache refresh manifest root must be a JSON object")

    version = _require_string(payload, "version")
    if version != _MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported live cache refresh manifest version '{version}', expected '{_MANIFEST_VERSION}'"
        )

    enabled = _require_bool(payload, "enabled")
    vault_root = _require_string(payload, "vault_root")
    debounce_seconds = payload.get("debounce_seconds", _DEFAULT_DEBOUNCE_SECONDS)
    if not isinstance(debounce_seconds, (int, float)) or float(debounce_seconds) <= 0.0:
        raise ValueError("Live cache refresh manifest field 'debounce_seconds' must be > 0")

    active_ensemble_dirs = _require_string_list(payload, "active_ensemble_dirs")
    raw_portfolios = payload.get("active_portfolios", [])
    if not isinstance(raw_portfolios, list):
        raise ValueError("Live cache refresh manifest field 'active_portfolios' must be a list")

    active_portfolios = tuple(_parse_active_portfolio(item) for item in raw_portfolios)
    if active_portfolios and not active_ensemble_dirs:
        raise ValueError(
            "Live cache refresh manifest must provide non-empty active_ensemble_dirs when active_portfolios are configured"
        )

    portfolio_names = [portfolio.name for portfolio in active_portfolios]
    if len(portfolio_names) != len(set(portfolio_names)):
        raise ValueError("Live cache refresh manifest active portfolio names must be unique")

    portfolio_ids = [portfolio.portfolio_id for portfolio in active_portfolios]
    if len(portfolio_ids) != len(set(portfolio_ids)):
        raise ValueError("Live cache refresh manifest active portfolio_ids must be unique")

    return LiveCacheRefreshManifest(
        version=version,
        enabled=enabled,
        vault_root=vault_root,
        debounce_seconds=float(debounce_seconds),
        active_ensemble_dirs=active_ensemble_dirs,
        active_portfolios=active_portfolios,
        manifest_path=str(resolved_path),
    )


def _coerce_dirty_key(
    raw_key: tuple[Ticker | str, TimeFrame | str],
) -> tuple[Ticker, TimeFrame]:
    ticker_value, timeframe_value = raw_key
    ticker = ticker_value if isinstance(ticker_value, Ticker) else Ticker[str(ticker_value)]
    timeframe = (
        timeframe_value
        if isinstance(timeframe_value, TimeFrame)
        else TimeFrame[str(timeframe_value)]
    )
    return ticker, timeframe


def _normalize_dirty_keys(
    dirty_keys: Iterable[tuple[Ticker | str, TimeFrame | str]] | None,
) -> set[tuple[Ticker, TimeFrame]] | None:
    if dirty_keys is None:
        return None
    return {_coerce_dirty_key(raw_key) for raw_key in dirty_keys}


def _tracked_dirty_keys(manifest: LiveCacheRefreshManifest) -> set[tuple[Ticker, TimeFrame]]:
    tracked: set[tuple[Ticker, TimeFrame]] = set()
    for portfolio in manifest.active_portfolios:
        for ticker_name in portfolio.tickers:
            ticker = Ticker[ticker_name]
            tracked.add((ticker, portfolio.volatility_timeframe))
            for timeframe in portfolio.timeframes:
                tracked.add((ticker, timeframe))
    return tracked


def _dirty_key_strings(dirty_keys: set[tuple[Ticker, TimeFrame]] | None) -> tuple[str, ...]:
    if dirty_keys is None:
        return ()
    return tuple(sorted(f"{ticker.name}:{timeframe.name}" for ticker, timeframe in dirty_keys))


def _select_affected_portfolios(
    manifest: LiveCacheRefreshManifest,
    dirty_keys: set[tuple[Ticker, TimeFrame]] | None,
) -> tuple[ActiveLivePortfolioConfig, ...]:
    if dirty_keys is None:
        return manifest.active_portfolios

    affected: list[ActiveLivePortfolioConfig] = []
    for portfolio in manifest.active_portfolios:
        portfolio_keys = {
            (Ticker[ticker_name], timeframe)
            for ticker_name in portfolio.tickers
            for timeframe in portfolio.timeframes
        }
        portfolio_keys.update(
            (Ticker[ticker_name], portfolio.volatility_timeframe)
            for ticker_name in portfolio.tickers
        )
        if portfolio_keys.intersection(dirty_keys):
            affected.append(portfolio)
    return tuple(affected)


def _latest_common_end(
    store: Any,
    portfolio: ActiveLivePortfolioConfig,
) -> datetime:
    coverage_ends: list[pd.Timestamp] = []
    required_timeframes = tuple(dict.fromkeys(portfolio.timeframes + (portfolio.volatility_timeframe,)))
    for ticker_name in portfolio.tickers:
        ticker = Ticker[ticker_name]
        for timeframe in required_timeframes:
            record = store.describe_candle(ticker, timeframe)
            if record is None or record.coverage.end is None:
                raise ValueError(
                    f"Missing live candle coverage for {ticker.name}/{timeframe.name} required by '{portfolio.name}'"
                )
            coverage_ends.append(pd.Timestamp(record.coverage.end))
    if not coverage_ends:
        raise ValueError(f"No live candle coverage available for '{portfolio.name}'")
    return min(coverage_ends).to_pydatetime()


def _instrument_returns_from_central_store(
    store: Any,
    tickers: tuple[str, ...],
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    daily_frames = [
        store.query_candles(Ticker[ticker], TimeFrame.D, start=start, end=end)
        .reset_index()[["datetime", "ticker", "close"]]
        for ticker in tickers
    ]
    if not daily_frames:
        return pd.DataFrame()
    all_daily = pd.concat(daily_frames, ignore_index=True)
    all_daily["datetime"] = pd.to_datetime(all_daily["datetime"]).dt.normalize()
    returns = (
        all_daily.sort_values(["ticker", "datetime"])
        .assign(ret=lambda frame: frame.groupby("ticker")["close"].pct_change())
        .pivot(index="datetime", columns="ticker", values="ret")
        .fillna(0.0)
    )
    returns.columns = [str(col) for col in returns.columns]
    return returns


def _summary_dict(summary: Any) -> dict[str, Any]:
    if hasattr(summary, "__dataclass_fields__"):
        payload = asdict(summary)
    elif isinstance(summary, dict):
        payload = dict(summary)
    else:
        payload = {"value": summary}
    return payload


def _write_status_file(summary: LiveCacheRefreshRunSummary) -> None:
    from .central_cache import CentralCacheStore

    store = CentralCacheStore.get_instance()
    path = _status_path(Path(store.cache_dir))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _run_live_cache_refresh_cycle(
    manifest_path: str | None = None,
    dirty_keys: Iterable[tuple[Ticker | str, TimeFrame | str]] | None = None,
) -> LiveCacheRefreshRunSummary:
    from .cache_manager import CacheManager
    from .central_cache import CentralCacheStore
    from ensemble.portfolio import (
        PortfolioCacheQuery,
        PortfolioWorld,
        build_global_portfolio_from_ensemble_dirs,
        materialize_global_portfolio_predictions,
    )

    started_at = _utc_now_iso()
    resolved_manifest_path = str(
        resolve_relative_path(
            manifest_path or default_live_cache_refresh_manifest_path(),
            project_root_fallback="project_candidate",
        )
    )
    normalized_dirty_keys = _normalize_dirty_keys(dirty_keys)
    manifest: LiveCacheRefreshManifest | None = None

    try:
        manifest = load_live_cache_refresh_manifest(resolved_manifest_path)
        if manifest is None:
            summary = LiveCacheRefreshRunSummary(
                status="skipped",
                manifest_path=resolved_manifest_path,
                dirty_keys=_dirty_key_strings(normalized_dirty_keys),
                affected_portfolios=(),
                refreshed_ensemble_dirs=(),
                bias_refresh_summary=None,
                materialized_portfolios=(),
                started_at=started_at,
                finished_at=_utc_now_iso(),
                error="manifest_missing",
            )
            _write_status_file(summary)
            return summary

        if not manifest.enabled:
            summary = LiveCacheRefreshRunSummary(
                status="skipped",
                manifest_path=manifest.manifest_path,
                dirty_keys=_dirty_key_strings(normalized_dirty_keys),
                affected_portfolios=(),
                refreshed_ensemble_dirs=(),
                bias_refresh_summary=None,
                materialized_portfolios=(),
                started_at=started_at,
                finished_at=_utc_now_iso(),
                error="manifest_disabled",
            )
            _write_status_file(summary)
            return summary

        if not manifest.active_portfolios:
            summary = LiveCacheRefreshRunSummary(
                status="skipped",
                manifest_path=manifest.manifest_path,
                dirty_keys=_dirty_key_strings(normalized_dirty_keys),
                affected_portfolios=(),
                refreshed_ensemble_dirs=(),
                bias_refresh_summary=None,
                materialized_portfolios=(),
                started_at=started_at,
                finished_at=_utc_now_iso(),
                error="no_active_portfolios",
            )
            _write_status_file(summary)
            return summary

        affected_portfolios = _select_affected_portfolios(manifest, normalized_dirty_keys)
        if not affected_portfolios:
            summary = LiveCacheRefreshRunSummary(
                status="skipped",
                manifest_path=manifest.manifest_path,
                dirty_keys=_dirty_key_strings(normalized_dirty_keys),
                affected_portfolios=(),
                refreshed_ensemble_dirs=(),
                bias_refresh_summary=None,
                materialized_portfolios=(),
                started_at=started_at,
                finished_at=_utc_now_iso(),
                error="no_matching_dirty_keys",
            )
            _write_status_file(summary)
            return summary

        store = CentralCacheStore.get_instance()
        queries: dict[str, PortfolioCacheQuery] = {}
        for portfolio in affected_portfolios:
            common_end = _latest_common_end(store, portfolio)
            common_start = (
                pd.Timestamp(common_end) - pd.Timedelta(days=portfolio.replay_window_days)
            ).to_pydatetime()
            queries[portfolio.name] = PortfolioCacheQuery(
                tickers=portfolio.tickers,
                start=common_start,
                end=common_end,
                timeframes=portfolio.timeframes,
                volatility_timeframe=portfolio.volatility_timeframe,
                scope=ArtifactScope.LIVE,
            )

        aggregate_start = min(query.start for query in queries.values())
        aggregate_end = max(query.end for query in queries.values())
        manager = CacheManager(cache_dir=str(store.live_artifact_cache_dir))
        bias_summary = manager.ensure_vault_cache_coverage(
            vault_ensemble_dirs=manifest.active_ensemble_dirs,
            start_date=aggregate_start,
            end_date=aggregate_end,
            refresh_mode="missing_stale_only",
        )
        if int(bias_summary.get("failed", 0)) > 0:
            raise RuntimeError(
                "Automatic live cache refresh failed during bias artifact refresh: "
                f"{bias_summary['failed']} task(s) failed"
            )

        materialized_portfolios: list[dict[str, Any]] = []
        for portfolio in affected_portfolios:
            query = queries[portfolio.name]
            ensemble_dirs = portfolio.ensemble_dirs or manifest.active_ensemble_dirs
            global_portfolio = build_global_portfolio_from_ensemble_dirs(
                ensemble_dirs,
                active_timeframes=tuple(portfolio.timeframes),
                target_volatility=portfolio.target_volatility,
                max_position_pct=portfolio.max_position_pct,
                idm_max=portfolio.idm_max,
            )
            instrument_returns = _instrument_returns_from_central_store(
                store,
                portfolio.tickers,
                query.start,
                query.end,
            )
            global_portfolio.fit_from_cache(query, instrument_returns)
            materialization = materialize_global_portfolio_predictions(
                portfolio=global_portfolio,
                query=query,
                portfolio_id=portfolio.portfolio_id,
                world=PortfolioWorld.LIVE,
                research_run_id=portfolio.research_run_id,
                scope=ArtifactScope.LIVE,
                vault_root=manifest.vault_root,
                cache_root=str(store.cache_dir),
                ensemble_dirs=ensemble_dirs,
            )
            materialized_portfolios.append(
                {
                    "name": portfolio.name,
                    "portfolio_id": portfolio.portfolio_id,
                    "query_start": query.start.isoformat(),
                    "query_end": query.end.isoformat(),
                    "summary": _summary_dict(materialization),
                }
            )

        summary = LiveCacheRefreshRunSummary(
            status="completed",
            manifest_path=manifest.manifest_path,
            dirty_keys=_dirty_key_strings(normalized_dirty_keys),
            affected_portfolios=tuple(portfolio.name for portfolio in affected_portfolios),
            refreshed_ensemble_dirs=manifest.active_ensemble_dirs,
            bias_refresh_summary=_summary_dict(bias_summary),
            materialized_portfolios=tuple(materialized_portfolios),
            started_at=started_at,
            finished_at=_utc_now_iso(),
        )
    except Exception as exc:
        logger.error("Automatic live cache refresh failed: %s", exc, exc_info=True)
        summary = LiveCacheRefreshRunSummary(
            status="failed",
            manifest_path=manifest.manifest_path if manifest is not None else resolved_manifest_path,
            dirty_keys=_dirty_key_strings(normalized_dirty_keys),
            affected_portfolios=(),
            refreshed_ensemble_dirs=manifest.active_ensemble_dirs if manifest is not None else (),
            bias_refresh_summary=None,
            materialized_portfolios=(),
            started_at=started_at,
            finished_at=_utc_now_iso(),
            error=str(exc),
        )

    _write_status_file(summary)
    return summary


def run_live_cache_refresh_now(
    manifest_path: str | None = None,
    dirty_keys: Iterable[tuple[Ticker | str, TimeFrame | str]] | None = None,
) -> LiveCacheRefreshRunSummary:
    """Run one live inference refresh cycle immediately."""
    return _run_live_cache_refresh_cycle(
        manifest_path=manifest_path,
        dirty_keys=dirty_keys,
    )


class LiveCacheRefreshOrchestrator:
    _instance: ClassVar[Optional["LiveCacheRefreshOrchestrator"]] = None

    def __init__(self) -> None:
        self._dirty_keys: set[tuple[Ticker, TimeFrame]] = set()
        self._batch_depth = 0
        self._event = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._lock = threading.Lock()
        self._stop_requested = False
        self._worker: threading.Thread | None = None

    @classmethod
    def get_instance(cls) -> "LiveCacheRefreshOrchestrator":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        if cls._instance is not None:
            cls._instance.stop()
        cls._instance = None

    def stop(self) -> None:
        with self._lock:
            self._stop_requested = True
            self._event.set()
            self._idle.set()
            worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)

    def wait_for_idle(self, timeout: float = 5.0) -> bool:
        return self._idle.wait(timeout=timeout)

    def pending_dirty_keys(self) -> tuple[str, ...]:
        with self._lock:
            return _dirty_key_strings(set(self._dirty_keys))

    def begin_batch(self) -> None:
        with self._lock:
            self._batch_depth += 1

    def end_batch(self) -> None:
        with self._lock:
            if self._batch_depth == 0:
                return
            self._batch_depth -= 1
            should_schedule = self._batch_depth == 0 and bool(self._dirty_keys)
        if should_schedule:
            self._schedule_run()

    def note_candle_update(
        self,
        ticker: Ticker,
        timeframe: TimeFrame,
        scope: ArtifactScope = ArtifactScope.LIVE,
    ) -> None:
        if scope is not ArtifactScope.LIVE:
            return

        manifest = load_live_cache_refresh_manifest()
        if manifest is None or not manifest.enabled or not manifest.active_portfolios:
            return

        dirty_key = (ticker, timeframe)
        if dirty_key not in _tracked_dirty_keys(manifest):
            return

        with self._lock:
            self._dirty_keys.add(dirty_key)
            if self._batch_depth > 0:
                self._idle.clear()
                return
        self._schedule_run()

    def _schedule_run(self) -> None:
        with self._lock:
            if self._stop_requested:
                return
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._worker_loop,
                    name="live-cache-refresh",
                    daemon=True,
                )
                self._worker.start()
            self._idle.clear()
            self._event.set()

    def _worker_loop(self) -> None:
        while True:
            self._event.wait()
            with self._lock:
                if self._stop_requested:
                    return
            while True:
                manifest = load_live_cache_refresh_manifest()
                debounce_seconds = (
                    manifest.debounce_seconds
                    if manifest is not None
                    else _DEFAULT_DEBOUNCE_SECONDS
                )
                self._event.clear()
                if self._event.wait(timeout=debounce_seconds):
                    with self._lock:
                        if self._stop_requested:
                            return
                    continue

                with self._lock:
                    if self._stop_requested:
                        return
                    if self._batch_depth > 0:
                        self._idle.clear()
                        break
                    dirty_keys = set(self._dirty_keys)
                    self._dirty_keys.clear()

                if not dirty_keys:
                    self._idle.set()
                    break

                summary = _run_live_cache_refresh_cycle(dirty_keys=dirty_keys)

                with self._lock:
                    if summary.status == "failed":
                        self._dirty_keys.update(dirty_keys)
                        self._idle.set()
                        break
                    if not self._dirty_keys:
                        self._idle.set()
                        break


def get_live_cache_refresh_orchestrator() -> LiveCacheRefreshOrchestrator:
    return LiveCacheRefreshOrchestrator.get_instance()


def reset_live_cache_refresh_orchestrator() -> None:
    LiveCacheRefreshOrchestrator.reset()


__all__ = [
    "ActiveLivePortfolioConfig",
    "LiveCacheRefreshManifest",
    "LiveCacheRefreshOrchestrator",
    "LiveCacheRefreshRunSummary",
    "default_live_cache_refresh_manifest_path",
    "get_live_cache_refresh_orchestrator",
    "load_live_cache_refresh_manifest",
    "reset_live_cache_refresh_orchestrator",
    "run_live_cache_refresh_now",
]
