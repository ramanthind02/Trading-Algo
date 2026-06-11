"""MT5 scraper write paths: single-writer gate wired through write_mt5_bars / write_mt5_ticks.

Tests verify:
- every write path reaches write_mt5_bars / write_mt5_ticks (not a bare pq.write_table)
- schema fields match the contracts after a round-trip read
- the ADR-2 timezone metadata stamp (broker_eet_as_utc) is present in the file
- duplicate rows are deduplicated and output is sorted on the key column
- overlapping appends merge correctly (initial write + overlapping second write)
- invalid data (low > high) raises SchemaContractError and leaves the existing file intact
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from data_platform.providers.mt5 import scraper
from data_platform.storage.contracts import (
    BROKER_EET_AS_UTC,
    MT5_BARS_SCHEMA,
    MT5_TICKS_SCHEMA,
    SchemaContractError,
    TIMEZONE_METADATA_KEY,
    fields_equal,
)


# ---------------------------------------------------------------------------
# Raw-array helpers (mirror the numpy dtype that copy_rates_range /
# copy_ticks_range return, which _write_bars / _write_ticks consume via
# pd.DataFrame(raw)).
# ---------------------------------------------------------------------------

def _bars_raw(*rows: tuple) -> np.ndarray:
    """Structured array matching copy_rates_range output."""
    dt = np.dtype([
        ("time",        np.int64),   # epoch seconds
        ("open",        np.float64),
        ("high",        np.float64),
        ("low",         np.float64),
        ("close",       np.float64),
        ("tick_volume", np.int64),
        ("spread",      np.int64),
        ("real_volume", np.int64),
    ])
    return np.array(list(rows), dtype=dt)


def _ticks_raw(*rows: tuple) -> np.ndarray:
    """Structured array matching copy_ticks_range output.

    Field order: time (epoch s), bid, ask, last, volume, time_msc (epoch ms), flags.
    _write_ticks overwrites 'time' via pd.to_datetime and keys on 'time_msc'.
    """
    dt = np.dtype([
        ("time",     np.int64),   # epoch seconds → overwritten by pd.to_datetime
        ("bid",      np.float64),
        ("ask",      np.float64),
        ("last",     np.float64),
        ("volume",   np.int64),
        ("time_msc", np.int64),   # epoch milliseconds — the sort/dedup key
        ("flags",    np.int64),
    ])
    return np.array(list(rows), dtype=dt)


# Base timestamp kept entirely within 2023 so all rows land in one year=2023 partition.
_T  = 1_700_000_000        # 2023-11-14 22:13:20 UTC (seconds)
_TM = _T * 1_000           # same instant in milliseconds


# ---------------------------------------------------------------------------
# _write_bars
# ---------------------------------------------------------------------------

def test_write_bars_initial_and_overlapping_append(tmp_path: Path, monkeypatch) -> None:
    """Initial write then overlapping append → deduped, sorted, schema/metadata intact."""
    target = tmp_path / "part.parquet"
    monkeypatch.setattr(scraper, "_bars_path", lambda sym, yr: target)

    # ── initial write: 3 distinct bars ─────────────────────────────────────
    raw1 = _bars_raw(
        (_T,        1.0, 1.5, 0.5, 1.2, 10, 2, 0),
        (_T + 60,   2.0, 2.5, 1.5, 2.2, 20, 3, 0),
        (_T + 120,  3.0, 3.5, 2.5, 3.2, 30, 4, 0),
    )
    scraper._write_bars("EURUSD", raw1)

    assert target.exists(), "parquet file must exist after initial write"

    # Parquet stores timestamp[s] as timestamp[ms] on disk; cast back to contract
    # types for field-equality checking (see test_storage_writers.py for rationale).
    tbl1_raw = pq.read_table(target)
    tbl1 = tbl1_raw.cast(MT5_BARS_SCHEMA.remove_metadata())
    assert fields_equal(tbl1.schema, MT5_BARS_SCHEMA), "schema fields must match contract"
    assert tbl1_raw.schema.metadata.get(TIMEZONE_METADATA_KEY) == BROKER_EET_AS_UTC, (
        "ADR-2 timezone metadata must be stamped"
    )
    assert tbl1.num_rows == 3

    # ── overlapping append: T+60 and T+120 duplicated; T+180 is new ────────
    raw2 = _bars_raw(
        (_T + 60,   2.0, 2.5, 1.5, 2.2, 20, 3, 0),   # duplicate
        (_T + 120,  3.0, 3.5, 2.5, 3.2, 30, 4, 0),   # duplicate
        (_T + 180,  4.0, 4.5, 3.5, 4.2, 40, 5, 0),   # new
    )
    scraper._write_bars("EURUSD", raw2)

    tbl2_raw = pq.read_table(target)
    tbl2 = tbl2_raw.cast(MT5_BARS_SCHEMA.remove_metadata())
    assert tbl2.num_rows == 4, "deduped merge must yield 4 unique rows"
    assert fields_equal(tbl2.schema, MT5_BARS_SCHEMA)
    assert tbl2_raw.schema.metadata.get(TIMEZONE_METADATA_KEY) == BROKER_EET_AS_UTC

    times = tbl2.column("time").to_pylist()
    assert times == sorted(times), "output must be sorted by time"


def test_write_bars_bad_ohlc_raises_and_leaves_file_intact(tmp_path: Path, monkeypatch) -> None:
    """low > high must raise SchemaContractError; existing file is left byte-for-byte unchanged."""
    target = tmp_path / "part.parquet"
    monkeypatch.setattr(scraper, "_bars_path", lambda sym, yr: target)

    # Write a valid file first so there is something to protect.
    raw_good = _bars_raw(
        (_T,      1.0, 1.5, 0.5, 1.2, 10, 2, 0),
        (_T + 60, 2.0, 2.5, 1.5, 2.2, 20, 3, 0),
    )
    scraper._write_bars("EURUSD", raw_good)
    assert target.exists()
    before = target.read_bytes()

    # Row with high=1.0 < low=2.0 — OHLC sanity violation.
    raw_bad = _bars_raw(
        (_T + 120, 1.5, 1.0, 2.0, 1.2, 10, 2, 0),   # high < low
    )
    with pytest.raises(SchemaContractError):
        scraper._write_bars("EURUSD", raw_bad)

    assert target.read_bytes() == before, "file must be byte-identical after a rejected write"


# ---------------------------------------------------------------------------
# _write_ticks
# ---------------------------------------------------------------------------

def test_write_ticks_initial_and_overlapping_append(tmp_path: Path, monkeypatch) -> None:
    """Initial write then overlapping append → deduped, sorted, schema/metadata intact."""
    target = tmp_path / "ticks.parquet"
    monkeypatch.setattr(scraper, "_ticks_path", lambda sym, yr: target)

    # ── initial write: 3 ticks ──────────────────────────────────────────────
    raw1 = _ticks_raw(
        (_T,     1.1000, 1.1001, 0.0, 0, _TM,         6),
        (_T,     1.1001, 1.1002, 0.0, 0, _TM + 500,   6),
        (_T + 1, 1.1002, 1.1003, 0.0, 0, _TM + 1_000, 6),
    )
    scraper._write_ticks("EURUSD", raw1)

    assert target.exists(), "parquet file must exist after initial write"

    tbl1_raw = pq.read_table(target)
    tbl1 = tbl1_raw.cast(MT5_TICKS_SCHEMA.remove_metadata())
    assert fields_equal(tbl1.schema, MT5_TICKS_SCHEMA), "schema fields must match contract"
    assert tbl1_raw.schema.metadata.get(TIMEZONE_METADATA_KEY) == BROKER_EET_AS_UTC, (
        "ADR-2 timezone metadata must be stamped"
    )
    assert tbl1.num_rows == 3

    # ── overlapping append: TM+500 and TM+1000 duplicated; TM+1500 is new ──
    raw2 = _ticks_raw(
        (_T,     1.1001, 1.1002, 0.0, 0, _TM + 500,   6),   # duplicate
        (_T + 1, 1.1002, 1.1003, 0.0, 0, _TM + 1_000, 6),   # duplicate
        (_T + 1, 1.1003, 1.1004, 0.0, 0, _TM + 1_500, 6),   # new
    )
    scraper._write_ticks("EURUSD", raw2)

    tbl2_raw = pq.read_table(target)
    tbl2 = tbl2_raw.cast(MT5_TICKS_SCHEMA.remove_metadata())
    assert tbl2.num_rows == 4, "deduped merge must yield 4 unique rows"
    assert fields_equal(tbl2.schema, MT5_TICKS_SCHEMA)
    assert tbl2_raw.schema.metadata.get(TIMEZONE_METADATA_KEY) == BROKER_EET_AS_UTC

    mscs = tbl2.column("time_msc").to_pylist()
    assert mscs == sorted(mscs), "output must be sorted by time_msc"
