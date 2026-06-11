"""Materialize and incrementally append the persistent Nautilus ParquetDataCatalog.

Reads provider parquet from ``data/mt5_data/`` and ingests M1 bars + synthesised
QuoteTicks into the persistent catalog at ``data/nautilus_catalog/``.

# NOTE: ``@customdataclass`` types (ResearchCandle, forecast/spread series) are
# incompatible with the PyO3 catalog.  This module is pinned to the PyArrow-backed
# ``nautilus_trader.persistence.catalog.ParquetDataCatalog`` exclusively.

Usage
-----
Full initial materialization (runs the 2-symbol pilot by default)::

    python -m data_platform.nautilus.materialize
    python -m data_platform.nautilus.materialize --symbols NDX XAUUSD
    python -m data_platform.nautilus.materialize --symbols NDX XAUUSD --from 2018

Append mode (daily chain — adds only bars newer than current catalog coverage)::

    python -m data_platform.nautilus.materialize --append [--symbols NDX XAUUSD]

Idempotency (initial mode)
--------------------------
Before ingesting any symbol the module queries ``catalog.get_intervals(Bar,
identifier=bar_type_str)``.  If the catalog already has data overlapping the
requested span the symbol is **skipped** — ``skip_disjoint_check`` is never
passed as True.

Append mode reads ``catalog.query_last_timestamp(Bar, identifier=bar_type_str)``
to find the coverage end and ingests only bars with ``ts > coverage_end``.
After each symbol's append ``catalog.consolidate_data_by_period(1 day)`` is
called as anti-fragmentation maintenance.

Repair primitive (documented here for ops runbooks)
---------------------------------------------------
When provider parquet has been rewritten or backfilled over a span that is
already in the catalog, remove then re-ingest::

    catalog.delete_data_range(Bar, identifier=bar_type_str, start=start, end=end)
    # then re-run: python -m data_platform.nautilus.materialize --symbols SYM --from YYYY

Memory
------
Bars are ingested one calendar-year partition at a time so the peak RSS is
bounded to one year's bars (~350 K rows for NDX ≈ 200 MB) rather than the
full multi-year dataset.
"""
from __future__ import annotations

import argparse
import json
import sys
import time as _time_module
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nautilus_trader.model.data import Bar, QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog

import nautilus_trader

from data_platform.core.catalog import load_catalog
from data_platform.core.instruments import Instrument
from data_platform.nautilus.catalog import default_catalog_path, get_catalog
from data_platform.nautilus.ingest import (
    _build_bars,
    _build_quotes,
    _ONE_MINUTE_NS,
    _ns_from_ts,
    _resolve_instrument,
    mt5_data_root,
)
from data_platform.nautilus.instruments import (
    build_all_nautilus_instruments,
    write_instruments_to_catalog,
)

_HERE = Path(__file__).resolve()
_REPO_ROOT = next(
    (p for p in _HERE.parents if (p / ".git").exists() or (p / "AGENTS.md").exists()),
    _HERE.parents[2],
)


# ── MT5 symbol discovery ──────────────────────────────────────────────────────

def mt5_symbols() -> list[str]:
    """Return the raw symbol names of all instruments whose data_source is 'mt5'.

    These are the ~16 symbols for which a provider parquet store under
    ``data/mt5_data/{SYM}/bars_M1/`` exists in the canonical Darwinex feed.
    """
    cat = load_catalog()
    return [
        inst.raw_symbol
        for inst in cat.all()
        if inst.data_source == "mt5"
    ]


# ── Coverage helpers ──────────────────────────────────────────────────────────

def _bar_type_str(symbol: str) -> str:
    """Build the Nautilus bar-type identifier string for an MT5 symbol."""
    inst = _resolve_instrument(symbol)
    return f"{inst.id}-1-MINUTE-LAST-EXTERNAL"


def _has_catalog_data(catalog: ParquetDataCatalog, bar_type: str) -> bool:
    """Return True if the catalog contains any Bar rows for *bar_type*."""
    try:
        intervals = catalog.get_intervals(Bar, identifier=bar_type)
        return len(intervals) > 0
    except Exception:
        return False


def _catalog_coverage_end(catalog: ParquetDataCatalog, bar_type: str) -> pd.Timestamp | None:
    """Return the latest Bar timestamp stored for *bar_type*, or None."""
    try:
        ts = catalog.query_last_timestamp(Bar, identifier=bar_type)
        return ts
    except Exception:
        return None


# ── Year-partition reader ─────────────────────────────────────────────────────

def _read_year_partition(base: Path, year: int) -> pd.DataFrame:
    """Read the single-year parquet partition ``{base}/year={year}/part.parquet``."""
    part = base / f"year={year}" / "part.parquet"
    if not part.exists():
        return pd.DataFrame()
    return pd.read_parquet(part)


# ── Per-symbol ingest helpers ─────────────────────────────────────────────────

def _available_years(symbol: str) -> list[int]:
    """Return sorted calendar years available in ``bars_M1`` for *symbol*."""
    bars_root = mt5_data_root() / symbol / "bars_M1"
    if not bars_root.exists():
        return []
    years = []
    for yr_dir in bars_root.iterdir():
        name = yr_dir.name
        if name.startswith("year=") and yr_dir.is_dir():
            try:
                years.append(int(name[5:]))
            except ValueError:
                pass
    return sorted(years)


def _ingest_year_bars(
    symbol: str,
    year: int,
    catalog: ParquetDataCatalog,
    inst: Instrument,
    after_ns: int | None = None,
) -> int:
    """Ingest one year's M1 bars for *symbol* into *catalog*.

    Parameters
    ----------
    after_ns:
        When set, only bars whose ``ts_init`` (bar close, nanos) is strictly
        greater than this value are written.  Pass the catalog's coverage_end_ns
        to implement strictly-forward append semantics.

    Returns the number of Bar objects written.
    """
    from nautilus_trader.model.data import BarType
    from nautilus_trader.model.identifiers import InstrumentId as NTInstrumentId

    bars_dir = mt5_data_root() / symbol / "bars_M1"
    df = _read_year_partition(bars_dir, year)
    if df.empty:
        return 0

    instrument_id = NTInstrumentId.from_str(str(inst.id))
    bar_type = BarType.from_str(f"{inst.id}-1-MINUTE-LAST-EXTERNAL")

    bars = _build_bars(df, bar_type, inst.price_precision)

    if after_ns is not None:
        bars = [b for b in bars if b.ts_init > after_ns]

    if bars:
        catalog.write_data(bars)
    return len(bars)


def _ingest_year_quotes(
    symbol: str,
    year: int,
    catalog: ParquetDataCatalog,
    inst: Instrument,
    after_ns: int | None = None,
) -> int:
    """Synthesise QuoteTicks from one year's bars spread field and write to catalog.

    Uses ``inst.price_increment`` as the POINT value — correct after the Step-1
    instrument fix (NDX/SP500 precision 1 / increment 0.1; the catalog value now
    matches the live-probed POINT).
    """
    from nautilus_trader.model.identifiers import InstrumentId as NTInstrumentId
    from nautilus_trader.model.objects import Price, Quantity

    bars_dir = mt5_data_root() / symbol / "bars_M1"
    df = _read_year_partition(bars_dir, year)
    if df.empty or "spread" not in df.columns:
        return 0

    instrument_id = NTInstrumentId.from_str(str(inst.id))
    point = float(inst.price_increment)

    half = df["spread"].to_numpy().astype("float64") * point / 2.0
    close_vals = df["close"].to_numpy().astype("float64")
    times = df["time"].to_numpy()
    zero = Quantity(0, 2)

    quotes: list[QuoteTick] = []
    for i in range(len(df)):
        close_ns = _ns_from_ts(pd.Timestamp(times[i])) + _ONE_MINUTE_NS
        if after_ns is not None and close_ns <= after_ns:
            continue
        quotes.append(
            QuoteTick(
                instrument_id=instrument_id,
                bid_price=Price(float(close_vals[i] - half[i]), inst.price_precision),
                ask_price=Price(float(close_vals[i] + half[i]), inst.price_precision),
                bid_size=zero,
                ask_size=zero,
                ts_event=close_ns,
                ts_init=close_ns,
            )
        )

    if quotes:
        catalog.write_data(quotes)
    return len(quotes)


# ── Symbol-level materialization ──────────────────────────────────────────────

def materialize_symbol(
    symbol: str,
    catalog: ParquetDataCatalog,
    from_year: int = 2018,
) -> dict:
    """Fully materialize one symbol into *catalog* (bars + synth quotes).

    Idempotent: if the catalog already has any Bar rows for this symbol's
    bar_type, the symbol is skipped (returns ``skipped=True``).

    Parameters
    ----------
    symbol: MT5 raw symbol, e.g. ``"NDX"``.
    catalog: destination ``ParquetDataCatalog``.
    from_year: skip years before this (default 2018 = real-M1 era).

    Returns a result dict with keys:
        symbol, skipped, bars_written, quotes_written, years_processed.
    """
    try:
        inst = _resolve_instrument(symbol)
    except ValueError as exc:
        return {"symbol": symbol, "skipped": True, "reason": str(exc),
                "bars_written": 0, "quotes_written": 0, "years_processed": []}

    bar_type = f"{inst.id}-1-MINUTE-LAST-EXTERNAL"

    if _has_catalog_data(catalog, bar_type):
        return {
            "symbol": symbol,
            "skipped": True,
            "reason": "already in catalog",
            "bars_written": 0,
            "quotes_written": 0,
            "years_processed": [],
        }

    years = [y for y in _available_years(symbol) if y >= from_year]
    if not years:
        return {
            "symbol": symbol,
            "skipped": True,
            "reason": f"no years >= {from_year} in provider parquet",
            "bars_written": 0,
            "quotes_written": 0,
            "years_processed": [],
        }

    total_bars = 0
    total_quotes = 0
    for year in years:
        b = _ingest_year_bars(symbol, year, catalog, inst)
        q = _ingest_year_quotes(symbol, year, catalog, inst)
        total_bars += b
        total_quotes += q

    return {
        "symbol": symbol,
        "skipped": False,
        "bars_written": total_bars,
        "quotes_written": total_quotes,
        "years_processed": years,
    }


# ── Symbol-level append ───────────────────────────────────────────────────────

def append_symbol(
    symbol: str,
    catalog: ParquetDataCatalog,
) -> dict:
    """Append only bars newer than the catalog's current coverage end for *symbol*.

    Strictly-forward semantics: only bars whose ts_init > coverage_end_ns are
    written.  ``consolidate_data_by_period(1 day)`` is called after any write to
    prevent file fragmentation.

    Returns a result dict with keys:
        symbol, skipped, bars_written, quotes_written, coverage_was, coverage_end.
    """
    try:
        inst = _resolve_instrument(symbol)
    except ValueError as exc:
        return {"symbol": symbol, "skipped": True, "reason": str(exc),
                "bars_written": 0, "quotes_written": 0}

    bar_type = f"{inst.id}-1-MINUTE-LAST-EXTERNAL"

    if not _has_catalog_data(catalog, bar_type):
        return {
            "symbol": symbol,
            "skipped": True,
            "reason": "not yet materialized — run full materialize first",
            "bars_written": 0,
            "quotes_written": 0,
        }

    coverage_ts = _catalog_coverage_end(catalog, bar_type)
    if coverage_ts is None:
        return {
            "symbol": symbol,
            "skipped": True,
            "reason": "could not determine coverage end",
            "bars_written": 0,
            "quotes_written": 0,
        }

    after_ns = int(coverage_ts.value)  # nanoseconds
    coverage_end_year = coverage_ts.year

    # Only look at years >= coverage_end_year (the last year might have new bars)
    years = [y for y in _available_years(symbol) if y >= coverage_end_year]

    total_bars = 0
    total_quotes = 0
    for year in years:
        b = _ingest_year_bars(symbol, year, catalog, inst, after_ns=after_ns)
        q = _ingest_year_quotes(symbol, year, catalog, inst, after_ns=after_ns)
        total_bars += b
        total_quotes += q

    if total_bars > 0:
        # Scope consolidation to the coverage day and forward only — do not
        # touch earlier historical files that were already laid down by the
        # initial materialize.  ensure_contiguous_files=False: the freshly
        # appended file and the existing same-day file are adjacent but were
        # written in separate catalog.write_data() calls.
        day_start = pd.Timestamp(coverage_ts.date(), tz="UTC")
        catalog.consolidate_data_by_period(
            Bar,
            identifier=bar_type,
            period=pd.Timedelta("1D"),
            start=day_start,
            ensure_contiguous_files=False,
        )

    return {
        "symbol": symbol,
        "skipped": False,
        "bars_written": total_bars,
        "quotes_written": total_quotes,
        "coverage_was": str(coverage_ts),
    }


# ── Registry job_run recording ────────────────────────────────────────────────

def _record_job_start(job_name: str, args_dict: dict) -> tuple[object, int | None]:
    try:
        from data_platform.registry import db as _db, writer as _w

        conn = _db.connect()
        with _db.transaction(conn):
            run_id = _w.record_job_run(
                conn,
                _w.JobRun(
                    job_name=job_name,
                    started_at=datetime.now(timezone.utc).isoformat(),
                    args_json=json.dumps(args_dict),
                ),
            )
        return conn, run_id
    except Exception:
        return None, None


def _record_job_finish(
    conn: object,
    run_id: int | None,
    exit_code: int,
    coverage_json: str | None,
    error_text: str | None,
    rows_written: int,
) -> None:
    if conn is None or run_id is None:
        return
    try:
        from data_platform.registry import db as _db, writer as _w

        with _db.transaction(conn):
            _w.finish_job_run(
                conn,
                run_id,
                exit_code=exit_code,
                rows_written=rows_written,
                coverage_json=coverage_json,
                error_text=error_text,
            )
        conn.close()
    except Exception:
        pass


# ── CLI entry points ──────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Materialize (or append to) the persistent Nautilus ParquetDataCatalog "
            "from MT5 provider parquet."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m data_platform.nautilus.materialize\n"
            "  python -m data_platform.nautilus.materialize --symbols NDX XAUUSD\n"
            "  python -m data_platform.nautilus.materialize --symbols NDX --from 2020\n"
            "  python -m data_platform.nautilus.materialize --append\n"
            "  python -m data_platform.nautilus.materialize --append --symbols NDX\n"
        ),
    )
    p.add_argument(
        "--symbols",
        nargs="+",
        metavar="SYM",
        default=None,
        help=(
            "MT5 raw symbols to process (default: all 16 mt5-sourced instruments). "
            "Specify 2 symbols for the recommended pilot run."
        ),
    )
    p.add_argument(
        "--from",
        dest="from_year",
        type=int,
        default=2018,
        help="Skip years before this (default: 2018, the real-M1 era).",
    )
    p.add_argument(
        "--append",
        action="store_true",
        default=False,
        help=(
            "Append-only mode: for each already-materialized symbol, ingest only "
            "bars newer than the catalog's current coverage end. "
            "Used by the daily chain's catalog_append step."
        ),
    )
    p.add_argument(
        "--catalog",
        metavar="PATH",
        default=None,
        help="Override the catalog root path (default: data/nautilus_catalog).",
    )
    return p


def main() -> int:
    """Entry point: ``python -m data_platform.nautilus.materialize``."""
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    # Bootstrap env / logging (tolerates missing .env).
    try:
        from lib.core.runtime_bootstrap import bootstrap_runtime
        bootstrap_runtime(_REPO_ROOT)
    except Exception:
        pass

    args = _build_parser().parse_args()
    catalog = get_catalog(args.catalog)

    symbols = args.symbols or mt5_symbols()

    if args.append:
        return _run_append(catalog, symbols)
    return _run_full(catalog, symbols, from_year=args.from_year)


def _run_full(
    catalog: ParquetDataCatalog,
    symbols: list[str],
    from_year: int = 2018,
) -> int:
    """Full initial materialization for *symbols*."""
    t0 = _time_module.monotonic()

    conn, run_id = _record_job_start(
        "catalog_materialize",
        {"symbols": symbols, "from_year": from_year, "mode": "full"},
    )

    # 1. Write instruments first (must precede any bar/quote ingest).
    nt_instruments = write_instruments_to_catalog(catalog)
    print(f"[materialize] wrote {len(nt_instruments)} instruments to catalog")

    # 2. Ingest bars + synth quotes per symbol per year (memory-bounded).
    results: list[dict] = []
    for sym in symbols:
        sym_t0 = _time_module.monotonic()
        r = materialize_symbol(sym, catalog, from_year=from_year)
        r["elapsed_s"] = round(_time_module.monotonic() - sym_t0, 2)
        results.append(r)
        status = "SKIP" if r.get("skipped") else "OK"
        print(
            f"  [{status}] {sym:12s}  bars={r.get('bars_written', 0):>8,}  "
            f"quotes={r.get('quotes_written', 0):>8,}  "
            f"{r.get('elapsed_s', 0):.1f}s"
            + (f"  ({r.get('reason', '')})" if r.get("skipped") else "")
        )

    elapsed = _time_module.monotonic() - t0
    total_bars = sum(r.get("bars_written", 0) for r in results)
    total_quotes = sum(r.get("quotes_written", 0) for r in results)

    coverage_json = json.dumps({
        "mode": "full",
        "symbols": symbols,
        "from_year": from_year,
        "total_bars": total_bars,
        "total_quotes": total_quotes,
        "elapsed_s": round(elapsed, 1),
        "nautilus_version": nautilus_trader.__version__,
    })
    print(
        f"[materialize] done  total_bars={total_bars:,}  "
        f"total_quotes={total_quotes:,}  elapsed={elapsed:.1f}s"
    )

    _record_job_finish(conn, run_id, 0, coverage_json, None, total_bars + total_quotes)
    return 0


def _run_append(
    catalog: ParquetDataCatalog,
    symbols: list[str],
) -> int:
    """Append-only mode: ingest only bars newer than catalog coverage end."""
    t0 = _time_module.monotonic()

    conn, run_id = _record_job_start(
        "catalog_append",
        {"symbols": symbols, "mode": "append"},
    )

    results: list[dict] = []
    for sym in symbols:
        sym_t0 = _time_module.monotonic()
        r = append_symbol(sym, catalog)
        r["elapsed_s"] = round(_time_module.monotonic() - sym_t0, 2)
        results.append(r)
        status = "SKIP" if r.get("skipped") else "OK"
        print(
            f"  [{status}] {sym:12s}  bars={r.get('bars_written', 0):>8,}  "
            f"quotes={r.get('quotes_written', 0):>8,}  "
            f"{r.get('elapsed_s', 0):.1f}s"
            + (f"  ({r.get('reason', '')})" if r.get("skipped") else "")
        )

    elapsed = _time_module.monotonic() - t0
    total_bars = sum(r.get("bars_written", 0) for r in results)
    total_quotes = sum(r.get("quotes_written", 0) for r in results)

    coverage_json = json.dumps({
        "mode": "append",
        "symbols": symbols,
        "total_bars": total_bars,
        "total_quotes": total_quotes,
        "elapsed_s": round(elapsed, 1),
        "nautilus_version": nautilus_trader.__version__,
    })
    print(
        f"[catalog_append] done  total_bars={total_bars:,}  "
        f"total_quotes={total_quotes:,}  elapsed={elapsed:.1f}s"
    )

    _record_job_finish(conn, run_id, 0, coverage_json, None, total_bars + total_quotes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
