"""
Demand-driven MT5 tick cache — the keystone of the hybrid backtest.

`ensure_ticks(symbol, start, end)` returns bid/ask ticks for a time window,
fetching from MT5 `copy_ticks_range` *only* for sub-ranges not already on disk
and caching them. The second call for the same window is a pure parquet read —
no MT5 round-trip, no network.

Why this exists
---------------
A backtest only needs tick-level bid/ask *while an order is live* (placement →
fill/cancel). Everywhere else, M1 bars suffice. So instead of bulk-scraping the
whole tick history of the whole universe, the backtest calls `ensure_ticks` for
each order's active window; the cache fills lazily and is warm on every re-run.
This optimises all three axes at once:

  * speed   — cold MT5 fetch happens once per window, ever; re-runs are instant.
  * realism — real bid/ask at the exact fill, not an assumed constant spread.
  * ease    — no upfront "what do I scrape" decision; it fetches what it touches.

Coverage tracking
-----------------
A per-symbol JSON manifest records which UTC ranges have been *requested from
MT5* (whether or not ticks came back — a closed-market window is legitimately
empty and must not be re-fetched every run):

    data/mt5_data/{SYMBOL}/_ticks_coverage.json   # merged [start,end] intervals

Ticks themselves land in the same year-partitioned store every other consumer
already reads, so `ingest_mt5_intraday*` and `build_sliced_catalog` see them
with zero changes:

    data/mt5_data/{SYMBOL}/ticks/year=YYYY/part.parquet

Offline behaviour
-----------------
If every requested sub-range is already covered, `ensure_ticks` needs no MT5
connection at all (pure cache read) — so a warm backtest runs offline. Only a
cache *miss* requires the MT5 terminal to be running.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path

import json
import pandas as pd
import pyarrow as pa

# Reuse the scraper's connection, schema and cap. The on-demand cache uses its
# OWN append-only chunk store (see below) rather than the scraper's
# read-merge-rewrite `_write_ticks` — appending a small window must never touch
# the multi-million-row bulk `ticks/` file.
from data_platform.providers.mt5 import scraper as _scr
from data_platform.providers.mt5.scraper import MT5_DATA_DIR, TICK_CAP, _TICKS_SCHEMA
from data_platform.storage import write_mt5_ticks

import MetaTrader5 as mt5
from lib.core.logger import get_logger

logger = get_logger(__name__)

UTC = timezone.utc

# Columns persisted per cache chunk (the MT5 tick fields the catalog bridge needs).
_TICK_COLS = ["time_msc", "bid", "ask", "last", "volume", "time", "flags"]

# Largest gap window fetched in a single copy_ticks_range call. Execution
# windows are ~1-2h so this is rarely hit; the daily chunk guards the rare
# case of a multi-day gap on a high-volume symbol (200k tick cap).
_MAX_CHUNK = timedelta(days=1)

# Module-level MT5 connection latch — initialise once per process, lazily, only
# when a cache miss actually needs the terminal.
_mt5_ready = False


# ---------------------------------------------------------------------------
# Coverage manifest (pure helpers — no MT5, unit-testable)
# ---------------------------------------------------------------------------

def _coverage_path(symbol: str) -> Path:
    return MT5_DATA_DIR / symbol / "_ticks_coverage.json"


def _to_utc(ts) -> pd.Timestamp:
    """Coerce any datetime-like to a tz-aware UTC pandas Timestamp."""
    t = pd.Timestamp(ts)
    return t.tz_localize(UTC) if t.tzinfo is None else t.tz_convert(UTC)


def merge_intervals(
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Merge overlapping/adjacent [start, end) intervals; return sorted, disjoint."""
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda iv: iv[0])
    out = [ordered[0]]
    for s, e in ordered[1:]:
        ls, le = out[-1]
        if s <= le:                       # overlap or touch → extend
            out[-1] = (ls, max(le, e))
        else:
            out.append((s, e))
    return out


def missing_ranges(
    start: pd.Timestamp,
    end: pd.Timestamp,
    covered: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Sub-ranges of [start, end) NOT in *covered* (a merged, sorted list)."""
    if start >= end:
        return []
    gaps: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    for s, e in covered:
        if e <= cursor:
            continue
        if s >= end:
            break
        if s > cursor:
            gaps.append((cursor, min(s, end)))
        cursor = max(cursor, e)
        if cursor >= end:
            break
    if cursor < end:
        gaps.append((cursor, end))
    return gaps


def _load_coverage(symbol: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    path = _coverage_path(symbol)
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return merge_intervals([(_to_utc(s), _to_utc(e)) for s, e in raw])
    except Exception as exc:
        logger.warning("could not read tick coverage for %s: %s", symbol, exc)
        return []


def _save_coverage(symbol: str, intervals: list[tuple[pd.Timestamp, pd.Timestamp]]) -> None:
    path = _coverage_path(symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [[s.isoformat(), e.isoformat()] for s, e in merge_intervals(intervals)]
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# MT5 fetch (cache miss only)
# ---------------------------------------------------------------------------

def _ensure_connected() -> bool:
    global _mt5_ready
    if _mt5_ready:
        return True
    if _scr.connect():
        _mt5_ready = True
    return _mt5_ready


# ---------------------------------------------------------------------------
# Append-only chunk store
# ---------------------------------------------------------------------------
#
# Each fetched gap is written to its own immutable parquet file named by its UTC
# nanosecond range:  data/mt5_data/{SYM}/ticks_cache/{start_ns}-{end_ns}.parquet
#
# Because coverage tracking guarantees fetched ranges are disjoint, chunks never
# overlap — so writes are O(chunk) (never read-merge a big file) and reads only
# open the chunk files whose name-range intersects the request.

def _cache_dir(symbol: str) -> Path:
    return MT5_DATA_DIR / symbol / "ticks_cache"


def _chunk_path(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> Path:
    return _cache_dir(symbol) / f"{start.value}-{end.value}.parquet"


def _write_chunk(symbol: str, raw, g_start: pd.Timestamp, g_end: pd.Timestamp) -> int:
    """Write one fetched gap's ticks to its own immutable chunk file."""
    df = pd.DataFrame(raw)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    tbl = pa.Table.from_pandas(df[_TICK_COLS], schema=_TICKS_SCHEMA, preserve_index=False)
    path = _chunk_path(symbol, g_start, g_end)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_mt5_ticks(tbl, path)
    return len(df)


def _fetch_gap(symbol: str, g_start: pd.Timestamp, g_end: pd.Timestamp) -> int:
    """Fetch one [g_start, g_end) gap from MT5 in <=1-day chunks; cache it. Rows written.

    Raises RuntimeError if the symbol cannot be selected (terminal down / unknown
    symbol) so the caller does NOT mark the range covered — an empty result from a
    *failed* fetch must not poison the cache. A genuinely closed-market window
    returns 0 ticks with ``symbol_select`` succeeding, and IS marked covered.
    """
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(
            f"symbol_select({symbol!r}) failed — MT5 terminal down or symbol "
            f"unavailable; cannot fetch ticks for {g_start}..{g_end}."
        )
    total = 0
    cursor = g_start
    while cursor < g_end:
        chunk_end = min(cursor + _MAX_CHUNK, g_end)
        raw = mt5.copy_ticks_range(
            symbol, cursor.to_pydatetime(), chunk_end.to_pydatetime(), mt5.COPY_TICKS_ALL
        )
        n = len(raw) if raw is not None else 0
        if n >= TICK_CAP:
            logger.warning("%s %s..%s hit 200k cap — window may be truncated",
                           symbol, cursor, chunk_end)
        if n > 0:
            total += _write_chunk(symbol, raw, cursor, chunk_end)
        cursor = chunk_end
    return total


# ---------------------------------------------------------------------------
# Cache read
# ---------------------------------------------------------------------------

def _read_window(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Read cached ticks in [start, end) — only opens chunk files that overlap."""
    base = _cache_dir(symbol)
    if not base.exists():
        return pd.DataFrame()
    start_ns, end_ns = start.value, end.value
    frames = []
    for path in base.glob("*.parquet"):
        try:
            a_str, b_str = path.stem.split("-")
            a, b = int(a_str), int(b_str)
        except ValueError:
            continue
        if a < end_ns and b > start_ns:           # chunk range overlaps request
            frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    t = pd.to_datetime(df["time"], utc=True)
    out = df[(t >= start) & (t < end)].copy()
    return out.drop_duplicates("time_msc").sort_values("time_msc").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public keystone
# ---------------------------------------------------------------------------

def ensure_ticks(symbol: str, start, end) -> pd.DataFrame:
    """
    Return bid/ask ticks for *symbol* in [start, end) (tz-aware UTC).

    Fetches from MT5 only the sub-ranges not already cached, records them as
    covered, and returns the window from the local parquet store. A fully-cached
    window needs no MT5 connection.

    Raises RuntimeError if a sub-range is uncovered and MT5 is unavailable.
    """
    start_u, end_u = _to_utc(start), _to_utc(end)
    if start_u >= end_u:
        return pd.DataFrame()

    covered = _load_coverage(symbol)
    gaps = missing_ranges(start_u, end_u, covered)

    if gaps:
        if not _ensure_connected():
            raise RuntimeError(
                f"ensure_ticks: {len(gaps)} uncovered range(s) for {symbol!r} "
                f"and the MT5 terminal is not available to fetch them. "
                f"Start the Darwinex MT5 terminal or pre-warm the cache."
            )
        fetched = 0
        # Mark each gap covered only AFTER it is successfully fetched, so a
        # mid-batch terminal failure neither loses prior progress nor poisons
        # the failed range (which stays uncovered and is retried next call).
        for g0, g1 in gaps:
            fetched += _fetch_gap(symbol, g0, g1)
            covered = merge_intervals(covered + [(g0, g1)])
            _save_coverage(symbol, covered)
        logger.info("ensure_ticks %s [%s..%s]: filled %d gap(s), %d ticks fetched",
                    symbol, start_u.date(), end_u.date(), len(gaps), fetched)

    return _read_window(symbol, start_u, end_u)


def is_cached(symbol: str, start, end) -> bool:
    """True if [start, end) is fully covered locally (no MT5 fetch would occur)."""
    start_u, end_u = _to_utc(start), _to_utc(end)
    return not missing_ranges(start_u, end_u, _load_coverage(symbol))
