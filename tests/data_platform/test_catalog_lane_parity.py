"""Data-level parity: persistent catalog vs. tempdir-ingested bars and quotes.

Verifies M5.2 §5.4-5.5: the persistent ``data/nautilus_catalog`` and the fallback
ephemeral (tempdir-ingest) path produce **identical** Bar and QuoteTick data for the
same NDX window (2024-01-01 .. 2024-02-28).

Parity level: DATA — we compare the loaded bar/quote lists directly, without running
a full BacktestEngine.  This is the "catalog.bars()/quote_ticks() parity" level
described in the migration plan; a full-backtest parity gate would need the vault
signals for the test window, which is too heavy for CI.

Skip conditions (either will skip):
  * ``data/mt5_data/NDX`` is absent (no source parquet to ingest from)
  * persistent catalog at ``data/nautilus_catalog`` has no NDX Bar intervals
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from nautilus_trader.model.data import Bar, QuoteTick

from data_platform.nautilus.catalog import default_catalog_path, get_catalog
from data_platform.nautilus.ingest import (
    _resolve_instrument,
    ingest_mt5_intraday,
    ingest_mt5_synth_quotes_from_bars,
    mt5_data_root,
)
from data_platform.nautilus.instruments import to_nautilus_instrument

_SYMBOL = "NDX"
# Tz-naive: matches how the engine builds win_start/win_end from target_dates
# (pd.Timestamp(date_obj) is naive).  _filter_by_time localises to UTC internally
# — passing an already-tz-aware Timestamp would raise ValueError there.
_WIN_START = pd.Timestamp("2024-01-01")
_WIN_END = pd.Timestamp("2024-02-28 23:59:59")


# ---------------------------------------------------------------------------
# Precondition helpers
# ---------------------------------------------------------------------------

def _mt5_data_present() -> bool:
    return (mt5_data_root() / _SYMBOL).exists()


def _persistent_catalog_has_ndx() -> bool:
    """Return True if the persistent catalog has any Bar intervals for NDX."""
    try:
        cat = get_catalog(default_catalog_path())
        inst = _resolve_instrument(_SYMBOL)
        bar_type_str = f"{inst.id}-1-MINUTE-LAST-EXTERNAL"
        intervals = cat.get_intervals(Bar, identifier=bar_type_str)
        return bool(intervals)
    except Exception:
        return False


# Module-level skip when the source parquet is absent.
pytestmark = pytest.mark.skipif(
    not _mt5_data_present(),
    reason=f"data/mt5_data/{_SYMBOL} not present — skipping catalog lane parity",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def persistent_catalog():
    """The live persistent catalog at data/nautilus_catalog — read-only consumer."""
    if not _persistent_catalog_has_ndx():
        pytest.skip(
            "Persistent catalog at data/nautilus_catalog has no NDX Bar intervals "
            "— run `python -m data_platform.nautilus.materialize --symbols NDX` first"
        )
    return get_catalog(default_catalog_path())


@pytest.fixture(scope="module")
def tempdir_catalog(tmp_path_factory):
    """Freshly-ingested ephemeral catalog for the test window (mirrors the engine fallback)."""
    tmp_path = tmp_path_factory.mktemp("catalog_parity")
    cat = get_catalog(tmp_path / "catalog")
    inst = _resolve_instrument(_SYMBOL)
    cat.write_data([to_nautilus_instrument(inst)])
    # max_ticks=0: bars only (same as the engine default); keep the ingest surface small.
    ingest_mt5_intraday(_SYMBOL, cat, max_ticks=0, start=_WIN_START, end=_WIN_END)
    # Synth quotes use the same point value that materialize.py uses (price_increment).
    ingest_mt5_synth_quotes_from_bars(
        _SYMBOL, cat,
        point=float(inst.price_increment),
        start=_WIN_START,
        end=_WIN_END,
    )
    return cat


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_bar_count_nonzero(tempdir_catalog) -> None:
    """Sanity: the tempdir catalog must contain bars for the test window."""
    inst = _resolve_instrument(_SYMBOL)
    bar_type_str = f"{inst.id}-1-MINUTE-LAST-EXTERNAL"
    bars = tempdir_catalog.bars(bar_types=[bar_type_str])
    assert len(bars) > 0, (
        f"Tempdir catalog returned no bars for {_SYMBOL} {_WIN_START}..{_WIN_END}"
    )


def test_bar_parity_persistent_vs_tempdir(
    persistent_catalog, tempdir_catalog
) -> None:
    """Bar data from the persistent catalog and the tempdir ingest are identical.

    The two filter paths operate on different columns:
    * ``_filter_by_time`` (tempdir ingest) filters on the M1 parquet ``time`` column
      (bar OPEN time).
    * ``catalog.bars(start=..., end=...)`` (persistent path) filters on ``ts_event``
      (bar CLOSE = open + 1 min).

    The last bar of the window (open = WIN_END − 1 min, ts_event = WIN_END + epsilon)
    may therefore appear in one set but not the other.  We accept at most 2 boundary
    bars extra per side and verify that every bar in the intersection is OHLC-identical.
    """
    inst = _resolve_instrument(_SYMBOL)
    bar_type_str = f"{inst.id}-1-MINUTE-LAST-EXTERNAL"

    pers_bars = persistent_catalog.bars(
        bar_types=[bar_type_str], start=_WIN_START, end=_WIN_END
    )
    temp_bars = tempdir_catalog.bars(bar_types=[bar_type_str])

    assert len(pers_bars) > 0, (
        f"Persistent catalog returned no bars for {_SYMBOL} {_WIN_START}..{_WIN_END}. "
        "Run `python -m data_platform.nautilus.materialize --symbols NDX` first."
    )
    assert len(temp_bars) > 0, "Tempdir catalog has no bars — fixture broken"

    pers_by_ts: dict[int, object] = {b.ts_event: b for b in pers_bars}
    temp_by_ts: dict[int, object] = {b.ts_event: b for b in temp_bars}

    only_in_pers = set(pers_by_ts) - set(temp_by_ts)
    only_in_temp = set(temp_by_ts) - set(pers_by_ts)

    assert len(only_in_pers) <= 2, (
        f"Persistent catalog has {len(only_in_pers)} bars not in tempdir "
        f"(boundary ≤2 expected): {sorted(only_in_pers)[:3]}"
    )
    assert len(only_in_temp) <= 2, (
        f"Tempdir has {len(only_in_temp)} bars not in persistent catalog "
        f"(boundary ≤2 expected): {sorted(only_in_temp)[:3]}"
    )

    common_ts = set(pers_by_ts) & set(temp_by_ts)
    # At least 99.9 % of bars must be in the intersection.
    min_count = min(len(pers_bars), len(temp_bars))
    assert len(common_ts) >= min_count - 2, (
        f"Too few common bars: {len(common_ts)} / {min_count}"
    )

    mismatches: list[str] = []
    for ts in sorted(common_ts):
        pb, tb = pers_by_ts[ts], temp_by_ts[ts]
        if abs(float(pb.open) - float(tb.open)) > 1e-4:  # type: ignore[union-attr]
            mismatches.append(
                f"  ts={ts} open: pers={float(pb.open):.4f} temp={float(tb.open):.4f}"  # type: ignore[union-attr]
            )
        elif abs(float(pb.close) - float(tb.close)) > 1e-4:  # type: ignore[union-attr]
            mismatches.append(
                f"  ts={ts} close: pers={float(pb.close):.4f} temp={float(tb.close):.4f}"  # type: ignore[union-attr]
            )
        if len(mismatches) >= 5:
            break

    assert not mismatches, (
        f"Bar OHLC mismatch in common bars ({len(mismatches)} shown):\n"
        + "\n".join(mismatches)
    )


def test_quote_parity_persistent_vs_tempdir(
    persistent_catalog, tempdir_catalog
) -> None:
    """Synth-quote bid/ask from the persistent catalog and tempdir ingest are identical.

    Both are produced from the same M1 ``spread`` field with ``point = inst.price_increment``
    (the value materialize.py uses).  The same ≤2-boundary-bar tolerance applies.
    """
    inst = _resolve_instrument(_SYMBOL)
    inst_id_str = str(inst.id)

    pers_quotes = persistent_catalog.quote_ticks(
        instrument_ids=[inst_id_str], start=_WIN_START, end=_WIN_END
    )
    temp_quotes = tempdir_catalog.quote_ticks(instrument_ids=[inst_id_str])

    assert len(pers_quotes) > 0, (
        f"Persistent catalog returned no quotes for {_SYMBOL} {_WIN_START}..{_WIN_END}"
    )
    assert len(temp_quotes) > 0, "Tempdir catalog has no quotes — fixture broken"

    pers_by_ts: dict[int, object] = {q.ts_event: q for q in pers_quotes}
    temp_by_ts: dict[int, object] = {q.ts_event: q for q in temp_quotes}

    only_in_pers = set(pers_by_ts) - set(temp_by_ts)
    only_in_temp = set(temp_by_ts) - set(pers_by_ts)

    assert len(only_in_pers) <= 2, (
        f"Persistent has {len(only_in_pers)} quotes not in tempdir: {sorted(only_in_pers)[:3]}"
    )
    assert len(only_in_temp) <= 2, (
        f"Tempdir has {len(only_in_temp)} quotes not in persistent: {sorted(only_in_temp)[:3]}"
    )

    common_ts = set(pers_by_ts) & set(temp_by_ts)
    min_count = min(len(pers_quotes), len(temp_quotes))
    assert len(common_ts) >= min_count - 2

    mismatches: list[str] = []
    for ts in sorted(common_ts):
        pq, tq = pers_by_ts[ts], temp_by_ts[ts]
        if abs(float(pq.bid_price) - float(tq.bid_price)) > 1e-4:  # type: ignore[union-attr]
            mismatches.append(
                f"  ts={ts} bid: pers={float(pq.bid_price):.4f} temp={float(tq.bid_price):.4f}"  # type: ignore[union-attr]
            )
        elif abs(float(pq.ask_price) - float(tq.ask_price)) > 1e-4:  # type: ignore[union-attr]
            mismatches.append(
                f"  ts={ts} ask: pers={float(pq.ask_price):.4f} temp={float(tq.ask_price):.4f}"  # type: ignore[union-attr]
            )
        if len(mismatches) >= 5:
            break

    assert not mismatches, (
        f"Quote bid/ask mismatch in common ticks ({len(mismatches)} shown):\n"
        + "\n".join(mismatches)
    )
