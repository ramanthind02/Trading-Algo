"""
MT5 historical-data fetch for the cfd_prop forecast pipeline.

The cfd_prop profile uses MetaTrader 5 for BOTH data and execution. This
module mirrors :func:`scripts.enigma_live_forecast.sync_ib_fetched_dailies_into_central_cache`
but pulls daily OHLC bars from MT5 instead of IB TWS, so the FTMO VPS does
not need TWS installed.

Design notes
------------
* The ``MetaTrader5`` Python package is Windows-only and lazily imported via
  :func:`_require_mt5` so unit tests on macOS/Linux can still import this
  module (they mock the package via ``sys.modules`` like ``test_mt5_trade_executor``).
* MT5's ``copy_rates_from_pos(symbol, TIMEFRAME_D1, 0, count)`` includes the
  current forming D1 bar. We always strip rows dated "today" in the broker
  timezone (resolved via :func:`mt5.symbol_info_tick`) before upserting to the
  cache; otherwise an intraday run would permanently pollute the cache with
  partial OHLC values.
* MT5 data is written DIRECTLY to ``CentralCacheStore.upsert_candles`` —
  we deliberately bypass :func:`upsert_tws_candles`'s IB-specific junction-ratio
  alignment + append-only filter (both inappropriate for CFD prices, which do
  not have contract rollover gaps).
* The shared central cache is global; running MT5 data on a machine that also
  runs the futures_prop pipeline would corrupt cross-profile signals. We hard-
  guard this: ``data.source == "mt5"`` is only allowed when ``profile == "cfd_prop"``
  (enforced by the caller in ``enigma_live_forecast.main``).
* If any required ticker fails to return MT5 data (or returns fewer than
  ``min_bars`` bars), :func:`sync_mt5_dailies_into_central_cache` raises a
  ``RuntimeError`` rather than silently letting that ticker fall back to stale
  bootstrap parquets.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from utils.core.enums import TimeFrame, Ticker


logger = logging.getLogger(__name__)


# Lazy import so this module is importable on macOS/Linux for unit tests.
try:  # pragma: no cover - environment-dependent
    import MetaTrader5 as _mt5  # type: ignore
    _MT5_AVAILABLE = True
except Exception:  # pragma: no cover
    _mt5 = None  # type: ignore
    _MT5_AVAILABLE = False


def _require_mt5() -> Any:
    """Return the MT5 module or raise a clear ImportError."""
    if not _MT5_AVAILABLE or _mt5 is None:  # pragma: no cover - exercised on non-Windows
        raise ImportError(
            "The MetaTrader5 Python package is not installed. "
            "Install it on a Windows machine where the MT5 terminal is also installed: "
            "  pip install MetaTrader5"
        )
    return _mt5


# ---------------------------------------------------------------------------
# Terminal path resolution
# ---------------------------------------------------------------------------

def resolve_mt5_terminal_path(config: Dict[str, Any]) -> Optional[str]:
    """Resolve the MT5 terminal path used for data fetch.

    Preference order:
      1. ``data.mt5_terminal_path`` (explicit override)
      2. First enabled account's ``terminal_path``
      3. ``None`` (let MT5 auto-detect / use whatever is in PATH)

    Returns ``None`` if nothing is configured so :func:`mt5.initialize` can
    fall back to its default search behavior.
    """
    data_cfg = config.get("data") or {}
    explicit = data_cfg.get("mt5_terminal_path")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    for acct in config.get("accounts") or []:
        if not acct.get("enabled", False):
            continue
        path = acct.get("terminal_path")
        if isinstance(path, str) and path.strip():
            return path.strip()
    return None


# ---------------------------------------------------------------------------
# Fetch + transform a single symbol
# ---------------------------------------------------------------------------

def _broker_today_date(mt5: Any, mt5_symbol: str) -> pd.Timestamp:
    """Broker's current calendar date, used to strip incomplete D1 rows.

    Uses ``mt5.symbol_info_tick(symbol).time`` (unix seconds in BROKER tz —
    MT5 returns the timestamp as a unix epoch but it's recorded in the broker's
    server-time zone, not real UTC). We treat it as a naive timestamp and
    normalise to midnight.
    """
    tick = mt5.symbol_info_tick(mt5_symbol)
    if tick is None or not getattr(tick, "time", 0):
        # Fall back to UTC today — best-effort.
        return pd.Timestamp(datetime.now(tz=timezone.utc).replace(tzinfo=None).date())
    return pd.Timestamp(datetime.utcfromtimestamp(int(tick.time)).date())


def fetch_mt5_daily_candles(
    mt5: Any,
    ticker_str: str,
    mt5_symbol: str,
    lookback_days: int,
) -> pd.DataFrame:
    """Fetch ``lookback_days`` daily bars for ``mt5_symbol`` from MT5.

    Returns a DataFrame with the standard candle columns:
    ``[datetime, open, high, low, close, volume, ticker, timeframe]``.

    Strips weekend / zero-volume rows AND the current-day forming bar so
    only completed daily bars reach the cache.

    Empty DataFrame if MT5 cannot select the symbol or returns no data.
    """
    if not mt5_symbol:
        return pd.DataFrame()

    if not mt5.symbol_select(mt5_symbol, True):
        logger.warning("MT5 could not select symbol %s (ticker=%s)", mt5_symbol, ticker_str)
        return pd.DataFrame()

    # +5 buffer so we still get the requested count after dropping current bar / weekends.
    rates = mt5.copy_rates_from_pos(mt5_symbol, mt5.TIMEFRAME_D1, 0, lookback_days + 5)
    if rates is None or len(rates) == 0:
        logger.warning("MT5 returned no rates for %s (ticker=%s)", mt5_symbol, ticker_str)
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    # MT5 structured array: time, open, high, low, close, tick_volume, spread, real_volume
    df["datetime"] = pd.to_datetime(df["time"], unit="s").dt.normalize()
    df["volume"] = df["tick_volume"].astype("int64")
    df["ticker"] = ticker_str
    df["timeframe"] = TimeFrame.D

    # Drop weekend / no-activity bars.
    df = df.loc[df["volume"] > 0].copy()

    # Drop the in-progress current-day bar (per rubber-duck critique).
    today = _broker_today_date(mt5, mt5_symbol)
    df = df.loc[df["datetime"] < today].copy()

    if df.empty:
        return df

    df = df[["datetime", "open", "high", "low", "close", "volume", "ticker", "timeframe"]]
    return df.sort_values("datetime").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def _sanity_check_price_scale(
    new_candles: pd.DataFrame,
    ticker_str: str,
    *,
    min_ratio: float = 0.5,
    max_ratio: float = 2.0,
) -> None:
    """Warn if MT5 close prices look wildly different from the cache's recent close.

    Catches the case where mt5_symbol points at the wrong instrument (e.g.
    pulling GOLD-cent-account instead of XAUUSD) or a wrong suffix returns
    a microlot variant. Compares median close from new MT5 data to median
    close from the existing cache (last 30 days). Does NOT raise — just logs
    a loud warning so the operator notices.
    """
    if new_candles.empty:
        return
    try:
        from utils.cache.runtime.central_cache import CentralCacheStore
        from utils.cache.runtime.central_cache_errors import ArtifactMissingError
    except ImportError:
        return

    try:
        ticker_enum = Ticker[ticker_str]
    except KeyError:
        return

    store = CentralCacheStore.get_instance()
    try:
        existing = store.query_candles(ticker_enum, TimeFrame.D)
    except (ArtifactMissingError, Exception):
        return

    if existing is None or existing.empty:
        return

    try:
        existing = existing.reset_index() if "datetime" not in existing.columns else existing
        existing_recent = existing.sort_values("datetime").tail(30)
        if existing_recent.empty:
            return
        median_existing = float(existing_recent["close"].median())
        median_new = float(new_candles["close"].median())
        if median_existing == 0 or median_new == 0:
            return
        ratio = median_new / median_existing
        if ratio < min_ratio or ratio > max_ratio:
            logger.warning(
                "PRICE-SCALE WARNING for %s: MT5 median close %.4f vs cache median close %.4f "
                "(ratio=%.4fx). Possible wrong symbol mapping — verify mt5_symbol in instruments.",
                ticker_str, median_new, median_existing, ratio,
            )
            print(
                f"  [!] PRICE-SCALE WARNING {ticker_str}: MT5={median_new:.4f} "
                f"vs cache={median_existing:.4f} (ratio={ratio:.4f}x). "
                "Verify the mt5_symbol mapping is correct."
            )
    except Exception as exc:  # noqa: BLE001 - sanity check, do not crash
        logger.debug("Price-scale sanity check failed for %s: %s", ticker_str, exc)


# ---------------------------------------------------------------------------
# Upsert into central cache (bypassing IB-specific filter)
# ---------------------------------------------------------------------------

def upsert_mt5_candles(
    daily_candles: pd.DataFrame,
    required_tickers: Set[str],
) -> None:
    """Write MT5 dailies directly to ``CentralCacheStore`` and rebuild monthly aggregates.

    Unlike :func:`scripts.enigma_live_forecast.upsert_tws_candles`, this does
    NOT apply IB junction-ratio alignment or append-only filtering — both are
    IB-specific (continuous-futures rollover handling) and inappropriate for
    CFDs which have no contract rollover.
    """
    from utils.cache.runtime.central_cache import CentralCacheStore

    store = CentralCacheStore.get_instance()

    for ticker_str in sorted(required_tickers):
        ticker_mask = daily_candles["ticker"].astype(str) == ticker_str
        ticker_candles = daily_candles.loc[ticker_mask].copy()
        if ticker_candles.empty:
            continue
        try:
            ticker_enum = Ticker[ticker_str]
        except KeyError:
            continue

        _sanity_check_price_scale(ticker_candles, ticker_str)
        store.upsert_candles(ticker_enum, TimeFrame.D, ticker_candles)
        print(f"    {ticker_str} D: upserted {len(ticker_candles)} bars (MT5)")

    # Rebuild monthly aggregates from full daily history (mirrors IB path).
    try:
        from scripts.enigma_live_forecast import resample_daily_to_monthly
    except ImportError:
        # Test environments may stub these out; OK to skip monthly rebuild.
        return

    for ticker_str in sorted(required_tickers):
        try:
            ticker_enum = Ticker[ticker_str]
        except KeyError:
            continue
        record = store.describe_candle(ticker_enum, TimeFrame.D)
        if record is None:
            continue
        full_daily = store.query_candles(
            ticker_enum, TimeFrame.D,
            start=record.coverage.start,
            end=record.coverage.end,
        ).reset_index()
        if full_daily.empty:
            continue
        full_daily["ticker"] = ticker_str
        monthly = resample_daily_to_monthly(full_daily)
        if not monthly.empty:
            store.upsert_candles(ticker_enum, TimeFrame.M, monthly)


# ---------------------------------------------------------------------------
# Top-level sync used by enigma_live_forecast
# ---------------------------------------------------------------------------

def sync_mt5_dailies_into_central_cache(
    *,
    config: Dict[str, Any],
    required_tickers: Set[str],
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Fetch MT5 daily history for required instruments and merge into central cache.

    Mirrors :func:`scripts.enigma_live_forecast.sync_ib_fetched_dailies_into_central_cache`
    but uses MT5 instead of IB. Returns ``(runtime_daily, None)`` — cfd_prop
    has no session-only partial overlay (today's forming bar is stripped).

    Raises
    ------
    RuntimeError
        - If ``mt5.initialize()`` fails.
        - If any required ticker with a configured ``mt5_symbol`` returns no
          data (fail-fast — see rubber-duck critique).
    """
    from utils.data.cross_ticker_store import CrossTickerDataStore

    mt5 = _require_mt5()

    instruments = config.get("instruments") or {}
    data_cfg = config.get("data") or {}
    max_lb = int(data_cfg.get("max_lookback_days", 500))
    min_bars = int(data_cfg.get("mt5_min_bars_per_ticker", 100))
    terminal_path = resolve_mt5_terminal_path(config)

    print("\n4. Fetching historical data from MT5...")
    if terminal_path:
        print(f"   Terminal: {terminal_path}")
        init_ok = mt5.initialize(terminal_path)
    else:
        print("   Terminal: <auto-detected>")
        init_ok = mt5.initialize()

    if not init_ok:
        last_err = getattr(mt5, "last_error", lambda: "<unknown>")()
        raise RuntimeError(
            f"mt5.initialize() failed: {last_err}. "
            "Ensure the MT5 terminal is installed and the path is correct."
        )

    try:
        # Sanity: surface which terminal account is active so the operator
        # knows whose data history is being fetched.
        try:
            acct = mt5.account_info()
            if acct is not None:
                print(
                    f"   Active terminal account: login={acct.login} "
                    f"server={acct.server} currency={acct.currency}"
                )
        except Exception:  # noqa: BLE001
            pass

        all_candles: List[pd.DataFrame] = []
        missing: List[str] = []
        short: List[Tuple[str, str, int]] = []  # (ticker, mt5_symbol, bar_count)

        for ticker_str in sorted(required_tickers):
            inst = instruments.get(ticker_str)
            if not isinstance(inst, dict):
                missing.append(f"{ticker_str} (no entry in config.instruments)")
                continue
            mt5_symbol = inst.get("mt5_symbol")
            if not isinstance(mt5_symbol, str) or not mt5_symbol.strip():
                missing.append(f"{ticker_str} (no mt5_symbol in config.instruments)")
                continue

            print(f"  {ticker_str} -> {mt5_symbol}: fetching {max_lb} days")
            candles = fetch_mt5_daily_candles(mt5, ticker_str, mt5_symbol, max_lb)
            if candles.empty:
                missing.append(f"{ticker_str} ({mt5_symbol})")
                continue
            if len(candles) < min_bars:
                short.append((ticker_str, mt5_symbol, len(candles)))
                continue
            print(f"    {ticker_str}: {len(candles)} bars "
                  f"({candles['datetime'].min().date()} -> {candles['datetime'].max().date()})")
            all_candles.append(candles)

        if missing or short:
            details: List[str] = []
            if missing:
                details.append(f"no data: {', '.join(missing)}")
            if short:
                details.append(
                    "insufficient bars (< "
                    f"{min_bars}): " + ", ".join(f"{t} {sym}={n}" for t, sym, n in short)
                )
            raise RuntimeError(
                "MT5 data fetch failed for required tickers — " + "; ".join(details)
                + f". Either fix mt5_symbol mappings, raise broker history, "
                + "or lower data.mt5_min_bars_per_ticker."
            )

        if not all_candles:
            raise RuntimeError("MT5 returned no historical data for any instrument.")

        daily_candles = pd.concat(all_candles, ignore_index=True)
        print(
            f"Total daily candles fetched: {len(daily_candles)} across "
            f"{daily_candles['ticker'].nunique()} instruments"
        )

        print("\n5. Upserting MT5 daily candles into cache...")
        upsert_mt5_candles(daily_candles, required_tickers)

        print("\n   Populating cross-ticker data store...")
        ct_store = CrossTickerDataStore.get_instance()
        for ticker_name in sorted(daily_candles["ticker"].unique()):
            ticker_s = str(ticker_name)
            try:
                ct_ticker = Ticker[ticker_s]
            except KeyError:
                continue
            if ct_store.is_loaded(ct_ticker, TimeFrame.D):
                continue
            mask = daily_candles["ticker"] == ticker_name
            ct_store.set_data(ct_ticker, TimeFrame.D, daily_candles.loc[mask].copy())
            print(f"   Cross-ticker {ticker_s}: loaded from runtime daily frame")

        # No partial overlay for cfd_prop — today's incomplete bar is stripped.
        return daily_candles, None

    finally:
        # Always release the MT5 session so the executor can re-init cleanly.
        try:
            mt5.shutdown()
        except Exception:  # noqa: BLE001
            pass
