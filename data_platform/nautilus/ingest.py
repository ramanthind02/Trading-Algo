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
"""
from __future__ import annotations

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
_SIZE_PRECISION = 0
_QUOTE_SIZE_PRECISION = 0


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


def ingest_mt5_intraday(
    symbol: str,
    catalog: ParquetDataCatalog,
    *,
    max_ticks: int | None = None,
) -> IngestResult:
    """Ingest M1 bars + bid/ask ticks for *symbol* into *catalog*.

    Returns an :class:`IngestResult` with the counts written. Bars use
    ``BarType`` ``{INSTRUMENT}-1-MINUTE-LAST-EXTERNAL`` with ``ts_init`` at the
    bar close; quotes carry bid/ask at the tick event time.

    ``max_ticks`` caps the number of quote ticks ingested (the raw MT5 tick
    history can be tens of millions of rows); ``None`` ingests every tick.
    """
    inst = _resolve_instrument(symbol)
    instrument_id = NTInstrumentId.from_str(str(inst.id))
    price_precision = inst.price_precision
    bar_type = BarType.from_str(f"{inst.id}-1-MINUTE-LAST-EXTERNAL")

    sym_root = mt5_data_root() / symbol
    bars_df = _read_partitions(sym_root / "bars_M1")
    ticks_df = _read_partitions(sym_root / "ticks")
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
