"""Broker-agnostic vault forecast engine for the live runtime.

This is the **signal layer**: it loads the vault :class:`GlobalPortfolio` and
produces the latest daily target ``position_fraction`` per instrument from the
central cache — the *same* cache for every broker, so all execution venues trade
one identical signal (the cache is populated by the Darwinex MT5 scraper; see
[[reference_mt5_timestamp_timezone]] / [[project_mt5_scraper]]).

It deliberately re-implements the small cache-query + required-ticker logic from
``scripts/enigma_live_forecast`` (``build_cache_query`` / ``discover_required_tickers``)
rather than importing that 1.6k-line script, which pulls in Interactive-Brokers
dependencies the live MT5 runtime must not require. The heavy lifting
(``build_global_portfolio_from_ensemble_dirs``, ``fit_from_cache`` /
``predict_from_cache``) is reused unchanged.

Warmup: bias nodes need a long daily history before their features are valid, so
``warmup_status`` reports, per required ticker, how many daily bars the cache
holds versus ``warmup_min_bars``; the strategy stays flat until every ticker is
ready.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from ensemble.portfolio_impl.vault_portfolio_loader import (
    build_global_portfolio_from_ensemble_dirs,
    discover_ensemble_dirs_in_vault,
)
from lib.core.enums import Ticker, TimeFrame
from lib.core.vault_paths import resolve_vault_root

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ForecastEngineConfig:
    """Knobs for the vault forecast engine.

    ``vault_root`` defaults to ``vault`` — the fitted prop vault that exists on
    disk (the ``vault_cfd_prop`` profile resolves to a directory that is not
    materialised in this repo). Override via the runtime config when a dedicated
    CFD vault is present.
    """

    vault_root: str = "vault"
    active_timeframes: tuple[TimeFrame, ...] = (TimeFrame.D, TimeFrame.M)
    target_volatility: float = 0.15
    max_position_pct: float = 2.5
    idm_max: float = 2.5
    warmup_min_bars: int = 500
    # Strictly > warmup_min_bars: equal values trigger a 1-bar off-by-one where a
    # ticker whose calendar yields one fewer session in the capped window fails the
    # >= warmup gate (matches the validation lane's 750/500 headroom).
    prediction_daily_max_bars: int = 750
    cache_root: str | None = None
    """If set, bind the central-cache singleton to this namespace on ``load()``.
    Live callers point every execution broker at the SHARED Darwinex signal
    namespace (``data/broker_cache/darwinex/central_cache``) so all venues read
    one identical signal. ``None`` keeps the default/shared cache."""


@dataclass(frozen=True)
class TickerWarmup:
    ticker: str
    bars: int
    ready: bool


@dataclass(frozen=True)
class WarmupStatus:
    """Per-ticker warmup readiness for the required cache tickers."""

    min_bars: int
    per_ticker: tuple[TickerWarmup, ...]

    @property
    def ready(self) -> bool:
        """True only when every required ticker has enough daily history."""
        return bool(self.per_ticker) and all(t.ready for t in self.per_ticker)

    def not_ready(self) -> tuple[str, ...]:
        return tuple(t.ticker for t in self.per_ticker if not t.ready)


@dataclass(frozen=True)
class ForecastResult:
    """Latest daily targets from one evaluation of the vault portfolio."""

    targets: Mapping[str, float]          # canonical ticker -> position_fraction
    forecast_scores: Mapping[str, float]  # canonical ticker -> forecast_score
    as_of: datetime | None
    warmup: WarmupStatus
    ready: bool = field(default=False)


def _ticker_str(value: object) -> str:
    """Normalise a ticker cell (``Ticker`` enum or str) to its canonical name."""
    return value.name if isinstance(value, Ticker) else str(value)


class VaultForecastEngine:
    """Loads the vault portfolio and emits the latest daily targets from cache."""

    def __init__(self, config: ForecastEngineConfig | None = None) -> None:
        self.config = config or ForecastEngineConfig()
        self._portfolio = None
        self._required_tickers: tuple[str, ...] = ()

    # ── lifecycle ────────────────────────────────────────────────────────
    def load(self) -> "VaultForecastEngine":
        """Build the :class:`GlobalPortfolio` and discover required tickers.

        Cheap relative to fit/predict (reads ensemble JSONs only; no candle
        reads). Raises if the vault has no ensembles for the active timeframes.
        """
        if self.config.cache_root:
            from cache.runtime.central_cache import CentralCacheStore

            CentralCacheStore._instance = CentralCacheStore(  # type: ignore[attr-defined]
                cache_dir=self.config.cache_root
            )
            logger.info("VaultForecastEngine bound to per-broker cache: %s", self.config.cache_root)

        vault_root = str(resolve_vault_root(self.config.vault_root))
        dirs = discover_ensemble_dirs_in_vault(vault_root, self.config.active_timeframes)
        if not dirs:
            raise ValueError(f"No ensembles found in vault root: {vault_root}")
        self._portfolio = build_global_portfolio_from_ensemble_dirs(
            dirs,
            active_timeframes=self.config.active_timeframes,
            target_volatility=self.config.target_volatility,
            max_position_pct=self.config.max_position_pct,
            idm_max=self.config.idm_max,
        )
        self._required_tickers = self._discover_required_tickers()
        logger.info(
            "VaultForecastEngine loaded: %d ensemble dirs, required tickers=%s",
            len(dirs),
            ", ".join(self._required_tickers),
        )
        return self

    @property
    def portfolio(self):
        if self._portfolio is None:
            raise RuntimeError("VaultForecastEngine.load() must be called first")
        return self._portfolio

    @property
    def required_tickers(self) -> tuple[str, ...]:
        return self._required_tickers

    def _discover_required_tickers(self) -> tuple[str, ...]:
        """All tickers the portfolio needs (primary + cross-reference tickers)."""
        from cache.runtime.cross_ticker_store import extract_cross_ticker_names

        required: set[str] = set()
        for tf_p in self.portfolio.tf_portfolios:
            for ens in tf_p.ensembles:
                for t in getattr(ens, "unique_tickers_", None) or ():
                    required.add(_ticker_str(t))
                for spec in ens.get_required_bias_nodes():
                    required.update(extract_cross_ticker_names(spec.get("params", {})))
        return tuple(sorted(required))

    # ── cache query construction (mirrors enigma_live_forecast.build_cache_query) ──
    def _coverage_window(self) -> tuple[datetime, datetime]:
        from cache.runtime.central_cache import CentralCacheStore

        store = CentralCacheStore.get_instance()
        starts: list[pd.Timestamp] = []
        ends: list[pd.Timestamp] = []
        for ticker_str in self._required_tickers:
            try:
                ticker_enum = Ticker[ticker_str]
            except KeyError:
                continue
            record = store.describe_candle(ticker_enum, TimeFrame.D)
            if record and record.coverage.start and record.coverage.end:
                starts.append(pd.Timestamp(record.coverage.start))
                ends.append(pd.Timestamp(record.coverage.end))
        if not starts:
            raise ValueError("No daily candle coverage in cache for required tickers")
        return max(starts).to_pydatetime(), min(ends).to_pydatetime()

    def _build_query(self, as_of: datetime | None = None):
        from cache.runtime.central_cache_models import ArtifactScope
        from ensemble.portfolio_impl.portfolio_cache import PortfolioCacheQuery

        start, end = self._coverage_window()
        if as_of is not None:
            # Causality bound: never look at candles dated after ``as_of``. In LIVE
            # this is a no-op (the cache only holds data up to now, so the wall
            # clock is >= the cache coverage end). In a BACKTEST, ``as_of`` is the
            # simulation clock, so the engine fits/predicts only on data <= the
            # current sim time — making on-the-fly signal generation lookahead-free
            # by construction (the final-validation lane).
            asof_ts = pd.Timestamp(as_of)
            if asof_ts.tzinfo is not None:
                asof_ts = asof_ts.tz_convert("UTC").tz_localize(None)
            end = min(pd.Timestamp(end), asof_ts).to_pydatetime()
        return PortfolioCacheQuery(
            tickers=self._required_tickers,
            start=start,
            end=end,
            timeframes=self.config.active_timeframes,
            scope=ArtifactScope.LIVE,
            prediction_daily_max_bars=self.config.prediction_daily_max_bars,
        )

    def _daily_frame(self, query) -> pd.DataFrame:
        from ensemble.portfolio_impl.portfolio_cache import _query_candles_from_cache

        return _query_candles_from_cache(query, TimeFrame.D)

    @staticmethod
    def _instrument_returns(daily: pd.DataFrame, tickers: tuple[str, ...]) -> pd.DataFrame:
        frames: list[pd.Series] = []
        for ticker_str in tickers:
            sub = daily.loc[daily["ticker"].map(_ticker_str) == ticker_str].sort_values("datetime")
            if not sub.empty:
                frames.append(sub.set_index("datetime")["close"].rename(ticker_str))
        if not frames:
            raise ValueError("No daily candle data in cache for returns computation")
        prices = pd.concat(frames, axis=1).sort_index()
        return prices.pct_change(fill_method=None).dropna(how="all")

    # ── warmup ───────────────────────────────────────────────────────────
    def warmup_status(self) -> WarmupStatus:
        """Per-ticker daily bar count vs ``warmup_min_bars`` (cache-only read)."""
        query = self._build_query()
        daily = self._daily_frame(query)
        counts = (
            daily.assign(_t=daily["ticker"].map(_ticker_str))
            .groupby("_t")["datetime"]
            .nunique()
            .to_dict()
        )
        min_bars = self.config.warmup_min_bars
        per_ticker = tuple(
            TickerWarmup(ticker=t, bars=int(counts.get(t, 0)), ready=int(counts.get(t, 0)) >= min_bars)
            for t in self._required_tickers
            if t in Ticker.__members__  # only tickers we actually source candles for
        )
        return WarmupStatus(min_bars=min_bars, per_ticker=per_ticker)

    # ── evaluation ───────────────────────────────────────────────────────
    def evaluate(self, as_of: datetime | None = None) -> ForecastResult:
        """Run the full daily pipeline and return the latest target per ticker.

        Gates on warmup: if any required ticker lacks enough history, returns an
        empty-target, ``ready=False`` result so the caller stays flat.

        ``as_of`` caps the candle window so no data dated after it is used. Live
        callers pass the wall clock (a no-op vs. the cache end); the backtest
        validation lane passes the simulation clock so signal generation is
        causal — see :func:`_build_query`.
        """
        query = self._build_query(as_of)
        daily = self._daily_frame(query)

        # Warmup gate (reuse the already-loaded daily frame for counts).
        counts = (
            daily.assign(_t=daily["ticker"].map(_ticker_str))
            .groupby("_t")["datetime"]
            .nunique()
            .to_dict()
        )
        min_bars = self.config.warmup_min_bars
        warmup = WarmupStatus(
            min_bars=min_bars,
            per_ticker=tuple(
                TickerWarmup(t, int(counts.get(t, 0)), int(counts.get(t, 0)) >= min_bars)
                for t in self._required_tickers
                if t in Ticker.__members__
            ),
        )
        if not warmup.ready:
            logger.warning("Warmup not ready; staying flat. Pending: %s", warmup.not_ready())
            return ForecastResult(targets={}, forecast_scores={}, as_of=None, warmup=warmup, ready=False)

        instrument_returns = self._instrument_returns(daily, self._required_tickers)
        self.portfolio.fit_from_cache(query, instrument_returns)
        positions_df = self.portfolio.predict_from_cache(query)
        if positions_df is None or positions_df.empty:
            logger.error("Portfolio produced no forecasts")
            return ForecastResult(targets={}, forecast_scores={}, as_of=None, warmup=warmup, ready=False)

        latest = positions_df.sort_values("datetime").groupby("ticker").last().reset_index()
        targets: dict[str, float] = {}
        scores: dict[str, float] = {}
        for _, row in latest.iterrows():
            t = _ticker_str(row["ticker"])
            targets[t] = float(row["position_fraction"])
            scores[t] = float(row.get("forecast_score", 0.0))
        as_of = pd.Timestamp(positions_df["datetime"].max()).to_pydatetime()
        logger.info("Forecast as_of=%s targets=%s", as_of, targets)
        return ForecastResult(
            targets=targets, forecast_scores=scores, as_of=as_of, warmup=warmup, ready=True
        )


__all__ = [
    "ForecastEngineConfig",
    "ForecastResult",
    "TickerWarmup",
    "VaultForecastEngine",
    "WarmupStatus",
]
