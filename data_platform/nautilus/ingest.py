"""Ingest MT5 intraday parquet into the Nautilus catalog.

Reads the provider parquet written by ``data_platform/providers/mt5`` and
wrangles it into Nautilus ``Bar`` (1-minute) and ``QuoteTick`` (bid/ask)
objects, then writes them to a ``ParquetDataCatalog``.

Layout consumed (per symbol ``{SYM}``)::

    data/mt5_data/{SYM}/bars_M1/year=*/part.parquet   # M1 OHLCV bars
    data/mt5_data/{SYM}/ticks/year=*/part.parquet      # bid/ask ticks

**Timestamp convention.** MetaTrader 5 stamps each bar with its *open* time.
Nautilus wants ``ts_init`` at the bar *close* (so a bar is only "known" once
complete — this avoids look-ahead in WP-3/WP-4 execution). We therefore set
both ``ts_event`` and ``ts_init`` to ``time + 1 minute`` (the bar close) for
the 1-minute bars. Ticks carry their own ``time`` (already an event time) so
``ts_event == ts_init == time``.

**Timezone convention (IMPORTANT).** MT5 ``time``/``time_msc`` — for both bars and
ticks — is **broker server time (EET/EEST), mislabelled UTC** (see
``docs/library/Data/mt5_timezones.md``). This module preserves that convention:
bars and quotes are ingested on the same broker clock, so they align with *each
other* (the matching engine fills correctly), but a Nautilus timestamp here is
broker time, not real UTC. Any wall-clock window (e.g. the financing rollover)
must therefore be expressed in **broker time** — the daily rollover is at
**00:00 broker** (= 17:00 New York), NOT a New-York or real-UTC hour.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from nautilus_trader.model.data import Bar, BarType, QuoteTick
from nautilus_trader.model.identifiers import InstrumentId as NTInstrumentId
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog

from data_platform.core.catalog import load_catalog
from data_platform.core.instruments import Instrument

_ONE_MINUTE_NS = 60 * 1_000_000_000
_SIZE_PRECISION = 2   # must match CFD instrument size_precision (0.01 lot steps)
_QUOTE_SIZE_PRECISION = 2


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    return next(
        (p for p in here.parents if (p / ".git").exists() or (p / "AGENTS.md").exists()),
        here.parents[2],
    )


def mt5_data_root() -> Path:
    return _repo_root() / "data" / "mt5_data"


@dataclass(frozen=True)
class IngestResult:
    """Counts of objects written for one symbol."""

    symbol: str
    instrument_id: str
    bars_written: int
    quotes_written: int


def _resolve_instrument(symbol: str) -> Instrument:
    """Find the catalog row whose InstrumentId symbol matches *symbol*."""
    catalog = load_catalog()
    for inst in catalog.all():
        if inst.symbol == symbol or inst.raw_symbol == symbol:
            return inst
    raise ValueError(f"No catalog instrument found for MT5 symbol {symbol!r}")


def _ns_from_ts(ts: pd.Timestamp) -> int:
    # pandas Timestamp.value is nanoseconds since epoch (UTC-aware -> UTC nanos).
    return int(ts.value)


def _read_partitions(base: Path) -> pd.DataFrame:
    parts = sorted(base.glob("year=*/part.parquet"))
    if not parts:
        return pd.DataFrame()
    frames = [pd.read_parquet(p) for p in parts]
    return pd.concat(frames, ignore_index=True)


def _build_bars(
    df: pd.DataFrame, bar_type: BarType, price_precision: int
) -> list[Bar]:
    bars: list[Bar] = []
    for row in df.itertuples(index=False):
        # MT5 `time` is the bar-open; close = open + 1 minute.
        open_ns = _ns_from_ts(pd.Timestamp(row.time))
        close_ns = open_ns + _ONE_MINUTE_NS
        bars.append(
            Bar(
                bar_type=bar_type,
                open=Price(float(row.open), price_precision),
                high=Price(float(row.high), price_precision),
                low=Price(float(row.low), price_precision),
                close=Price(float(row.close), price_precision),
                volume=Quantity(int(row.tick_volume), _SIZE_PRECISION),
                ts_event=close_ns,
                ts_init=close_ns,
            )
        )
    return bars


def _build_quotes(
    df: pd.DataFrame, instrument_id: NTInstrumentId, price_precision: int
) -> list[QuoteTick]:
    quotes: list[QuoteTick] = []
    zero_size = Quantity(0, _QUOTE_SIZE_PRECISION)
    for row in df.itertuples(index=False):
        ts = _ns_from_ts(pd.Timestamp(row.time))
        quotes.append(
            QuoteTick(
                instrument_id=instrument_id,
                bid_price=Price(float(row.bid), price_precision),
                ask_price=Price(float(row.ask), price_precision),
                bid_size=zero_size,
                ask_size=zero_size,
                ts_event=ts,
                ts_init=ts,
            )
        )
    return quotes


def _filter_by_time(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """Filter a frame with a ``time`` column to [start, end] (tz-aware UTC compare)."""
    if df.empty or (start is None and end is None) or "time" not in df.columns:
        return df
    t = pd.to_datetime(df["time"], utc=True)
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= t >= pd.Timestamp(start, tz="UTC")
    if end is not None:
        mask &= t <= pd.Timestamp(end, tz="UTC")
    return df[mask.values]


# Public names — callers may use either the plain or leading-underscore form.
# The private names are kept as aliases so existing imports remain valid.
read_partitions = _read_partitions
filter_by_time = _filter_by_time
resolve_instrument = _resolve_instrument


def ingest_mt5_intraday(
    symbol: str,
    catalog: ParquetDataCatalog,
    *,
    max_ticks: int | None = None,
    start=None,
    end=None,
) -> IngestResult:
    """Ingest M1 bars + bid/ask ticks for *symbol* into *catalog*.

    Returns an :class:`IngestResult` with the counts written. Bars use
    ``BarType`` ``{INSTRUMENT}-1-MINUTE-LAST-EXTERNAL`` with ``ts_init`` at the
    bar close; quotes carry bid/ask at the tick event time.

    ``max_ticks`` caps the number of quote ticks ingested (the raw MT5 tick
    history can be tens of millions of rows); ``None`` ingests every tick,
    ``0`` ingests none (bars only). ``start``/``end`` (tz-aware/naive datetimes)
    window both bars and ticks so a backtest ingests only the span it replays.
    """
    inst = _resolve_instrument(symbol)
    instrument_id = NTInstrumentId.from_str(str(inst.id))
    price_precision = inst.price_precision
    bar_type = BarType.from_str(f"{inst.id}-1-MINUTE-LAST-EXTERNAL")

    sym_root = mt5_data_root() / symbol
    bars_df = _filter_by_time(_read_partitions(sym_root / "bars_M1"), start, end)
    # ``max_ticks == 0`` means bars only — do NOT read the tick store at all. The raw
    # tick history can be tens of millions of rows (e.g. NDX ~46M); reading it just to
    # discard it OOMs. The realism lane synthesizes quotes from the M1 ``spread`` field
    # instead, so it always passes max_ticks=0 here.
    if max_ticks == 0:
        ticks_df = pd.DataFrame()
    else:
        ticks_df = _filter_by_time(_read_partitions(sym_root / "ticks"), start, end)
        if max_ticks is not None and not ticks_df.empty:
            ticks_df = ticks_df.head(max_ticks)

    bars = _build_bars(bars_df, bar_type, price_precision) if not bars_df.empty else []
    quotes = (
        _build_quotes(ticks_df, instrument_id, price_precision)
        if not ticks_df.empty
        else []
    )

    if bars:
        catalog.write_data(bars)
    if quotes:
        catalog.write_data(quotes)

    return IngestResult(
        symbol=symbol,
        instrument_id=str(inst.id),
        bars_written=len(bars),
        quotes_written=len(quotes),
    )


def ingest_mt5_synth_quotes_from_bars(
    symbol: str,
    catalog: ParquetDataCatalog,
    *,
    point: float,
    max_bars: int | None = None,
    start=None,
    end=None,
) -> IngestResult:
    """Synthesize ``QuoteTick`` bid/ask from the M1 bars' ``spread`` field.

    For Darwinex symbols whose full tick store has not been scraped (only NDX has
    ticks), the M1 bars still carry a per-minute integer ``spread`` (in the
    symbol's points). We reconstruct a quote at each bar close as
    ``bid = close − spread·point/2`` and ``ask = close + spread·point/2`` so the
    matching engine charges the *real* per-minute half-spread on MARKET fills
    without a bulk tick scrape. ``point`` is the symbol's true price increment
    (probed from the live terminal — e.g. NDX/SP500 = 0.1, XAUUSD = 0.01), NOT the
    catalog ``price_increment`` (which can differ). The rollover study already
    trusts this M1 ``spread`` field.

    Quotes carry ``ts_event == ts_init == bar_close`` (open + 1 minute), matching
    the bar timestamp convention in :func:`ingest_mt5_intraday`.
    """
    inst = _resolve_instrument(symbol)
    instrument_id = NTInstrumentId.from_str(str(inst.id))
    price_precision = inst.price_precision

    bars_df = _filter_by_time(
        _read_partitions(mt5_data_root() / symbol / "bars_M1"), start, end
    )
    if "spread" not in bars_df.columns or bars_df.empty:
        return IngestResult(symbol, str(inst.id), 0, 0)
    if max_bars is not None:
        bars_df = bars_df.head(max_bars)

    half = bars_df["spread"].to_numpy().astype("float64") * float(point) / 2.0
    close = bars_df["close"].to_numpy().astype("float64")
    times = bars_df["time"].to_numpy()
    zero = Quantity(0, _QUOTE_SIZE_PRECISION)
    quotes: list[QuoteTick] = []
    for i in range(len(bars_df)):
        close_ns = _ns_from_ts(pd.Timestamp(times[i])) + _ONE_MINUTE_NS
        quotes.append(
            QuoteTick(
                instrument_id=instrument_id,
                bid_price=Price(float(close[i] - half[i]), price_precision),
                ask_price=Price(float(close[i] + half[i]), price_precision),
                bid_size=zero,
                ask_size=zero,
                ts_event=close_ns,
                ts_init=close_ns,
            )
        )
    if quotes:
        catalog.write_data(quotes)
    return IngestResult(symbol, str(inst.id), 0, len(quotes))


def _daily_window_mask(
    times_utc: pd.Series, center: _dt.time, half_width_minutes: int, tz: str
) -> pd.Series:
    """Boolean mask: tick times within ±``half_width_minutes`` of ``center`` daily.

    ``center`` is a wall-clock time-of-day in timezone ``tz``. **For MT5 data the
    stored times are broker time mislabelled UTC**, so the financing rollover is
    ``center=datetime.time(0, 0)`` with ``tz="UTC"`` (00:00 broker) — NOT
    ``time(17, 0)`` / ``America/New_York`` (that would tz-convert the broker
    timestamps and land 7h off; see docs/library/Data/mt5_timezones.md).
    ``times_utc`` is a tz-aware (broker-as-UTC) datetime Series. The comparison is
    on minute-of-day with a signed modular distance so a window straddling
    midnight (as the 00:00 rollover does) still works.
    """
    local = times_utc.dt.tz_convert(tz)
    minute_of_day = local.dt.hour * 60 + local.dt.minute
    center_min = center.hour * 60 + center.minute
    # signed distance from center in [-720, 720) minutes (handles midnight wrap)
    signed = (minute_of_day - center_min + 720) % 1440 - 720
    return signed.abs() <= half_width_minutes


def ingest_mt5_intraday_windowed(
    symbol: str,
    catalog: ParquetDataCatalog,
    *,
    rollover: _dt.time,
    half_width_minutes: int,
    tz: str = "UTC",
) -> IngestResult:
    """Hybrid ingest: FULL M1 bars + quote ticks ONLY in a daily rollover window.

    This is the data-curation half of the "tick only around entries/exits"
    backtest (see ``docs/refactor/nautilus/backtest_speed_benchmark.md``). Nautilus
    merges bars + quotes into one time-ordered stream, so loading 1-minute bars
    across the whole holding period but quote ticks only within
    ``[rollover − Δ, rollover + Δ]`` gives coarse (bar) marking outside the window
    and full bid/ask fill realism inside it — at a fraction of the tick volume.

    Parameters
    ----------
    rollover : datetime.time
        Wall-clock time-of-day of the financing rollover, in ``tz``. **For MT5
        data pass ``datetime.time(0, 0)`` with ``tz="UTC"``** — the rollover is at
        00:00 broker time (= 17:00 NY); the stored timestamps are broker time
        mislabelled UTC (docs/library/Data/mt5_timezones.md).
    half_width_minutes : int
        Δ — keep ticks within this many minutes either side of ``rollover``.
    tz : str
        Timezone of ``rollover`` (default ``"UTC"`` = the broker clock of the
        stored timestamps). Do NOT pass ``"America/New_York"`` for MT5 data.
    """
    inst = _resolve_instrument(symbol)
    instrument_id = NTInstrumentId.from_str(str(inst.id))
    price_precision = inst.price_precision
    bar_type = BarType.from_str(f"{inst.id}-1-MINUTE-LAST-EXTERNAL")

    sym_root = mt5_data_root() / symbol
    bars_df = _read_partitions(sym_root / "bars_M1")
    ticks_df = _read_partitions(sym_root / "ticks")

    if not ticks_df.empty:
        times = pd.to_datetime(ticks_df["time"], utc=True)
        ticks_df = ticks_df[_daily_window_mask(times, rollover, half_width_minutes, tz).values]

    bars = _build_bars(bars_df, bar_type, price_precision) if not bars_df.empty else []
    quotes = (
        _build_quotes(ticks_df, instrument_id, price_precision)
        if not ticks_df.empty
        else []
    )

    if bars:
        catalog.write_data(bars)
    if quotes:
        catalog.write_data(quotes)

    return IngestResult(
        symbol=symbol,
        instrument_id=str(inst.id),
        bars_written=len(bars),
        quotes_written=len(quotes),
    )


def ingest_mt5_quotes_window(
    symbol: str,
    catalog: ParquetDataCatalog,
    start,
    end,
) -> IngestResult:
    """Demand-driven quote ingest: fetch bid/ask ticks for [start, end) via the
    on-demand cache and write them as ``QuoteTick`` records to *catalog*.

    This is the on-demand counterpart to :func:`ingest_mt5_intraday_windowed`.
    Where the latter filters a *pre-scraped* full-tick store down to a window,
    this one fetches *only* the requested window from MT5 (caching it), so no
    bulk tick scrape is needed: the backtest ingests exactly the execution
    windows its orders touch, and re-runs read the warm cache.

    No bars are written here (use :func:`ingest_mt5_intraday` for the M1 spine);
    this only lands the quote stream for realistic in-window fills.
    """
    # Lazy import: keeps this module importable without MetaTrader5 for the
    # bar-only path (e.g. CI / non-Windows). Only quote ingest needs MT5.
    from data_platform.providers.mt5.tick_cache import ensure_ticks

    inst = _resolve_instrument(symbol)
    instrument_id = NTInstrumentId.from_str(str(inst.id))
    price_precision = inst.price_precision

    ticks_df = ensure_ticks(symbol, start, end)
    quotes = (
        _build_quotes(ticks_df, instrument_id, price_precision)
        if not ticks_df.empty
        else []
    )
    if quotes:
        catalog.write_data(quotes)

    return IngestResult(
        symbol=symbol,
        instrument_id=str(inst.id),
        bars_written=0,
        quotes_written=len(quotes),
    )
