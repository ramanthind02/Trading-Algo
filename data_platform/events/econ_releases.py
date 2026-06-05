"""Scrape the Norgate Economic database (CPI, NFP, GDP, yields, etc.) to parquet.

The Economic DB holds ~148 macro series (release values, not market prices). Each
series is stored as a single parquet under data/events/econ/, plus a manifest JSON
mapping the Norgate symbol -> human-readable name + coverage.

Storage layout
--------------
data/events/econ/
  {SAFE_SYMBOL}.parquet         # date32 index + value(float32) [+ open/high/low if present]
  _manifest.json                # [{symbol, name, safe_symbol, rows, start, end}]

Schema (per series)
  date    date32 (index)
  close   float32   # the release value (Norgate Economic series carry OHLC; close = value)

Run directly:
    python -m data_platform.events.econ_releases            # all ~148 series
    python -m data_platform.events.econ_releases --limit 5  # sample
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 3
ECON_DATABASE = "Economic"


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    return next(
        (p for p in here.parents if (p / ".git").exists() or (p / "AGENTS.md").exists()),
        here.parents[2],
    )


def econ_dir() -> Path:
    return _repo_root() / "data" / "events" / "econ"


def _safe_symbol(symbol: str) -> str:
    """Map a Norgate econ symbol (e.g. '#CPISA') to a filesystem-safe name."""
    out = symbol.lstrip("#")
    for ch in ". -/\\":
        out = out.replace(ch, "_")
    return out or "_"


@dataclass(frozen=True)
class EconSeriesResult:
    symbol: str
    name: str | None
    safe_symbol: str
    rows: int
    start: str | None
    end: str | None


def ensure_norgate_running() -> None:
    import norgatedata
    if not norgatedata.status():
        raise RuntimeError("Norgate Data Updater is not running — start NDU and retry.")


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """date32 index + float32 close (the release value). Keep OHL if present."""
    out = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"}).copy()
    out.index = pd.to_datetime(out.index).normalize().tz_localize(None)
    out.index.name = "date"
    keep = [c for c in ("open", "high", "low", "close") if c in out.columns]
    out = out[keep].sort_index().loc[~out.index.duplicated(keep="last")]
    for col in keep:
        out[col] = out[col].astype("float32")
    return out


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=True),
        path,
        compression=PARQUET_COMPRESSION,
        compression_level=PARQUET_COMPRESSION_LEVEL,
    )


def fetch_series(symbol: str) -> EconSeriesResult | None:
    import norgatedata
    try:
        df = norgatedata.price_timeseries(symbol, interval="D", timeseriesformat="pandas-dataframe")
    except Exception as exc:
        print(f"  ERR {symbol}: {exc}")
        return None
    if df is None or df.empty:
        return None

    name = None
    try:
        name = norgatedata.security_name(symbol)
    except Exception:
        pass

    normalized = _normalize(df)
    safe = _safe_symbol(symbol)
    path = econ_dir() / f"{safe}.parquet"
    table = pa.Table.from_pandas(normalized, preserve_index=True)
    # Stamp the raw symbol + name into file metadata for lossless reverse mapping.
    meta = {b"norgate_symbol": symbol.encode(), b"name": (name or "").encode()}
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), **meta})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression=PARQUET_COMPRESSION,
                   compression_level=PARQUET_COMPRESSION_LEVEL)

    return EconSeriesResult(
        symbol=symbol, name=name, safe_symbol=safe, rows=len(normalized),
        start=str(normalized.index.min().date()), end=str(normalized.index.max().date()),
    )


def scrape(limit: int | None = None) -> list[EconSeriesResult]:
    import norgatedata
    ensure_norgate_running()
    symbols = sorted(norgatedata.database_symbols(ECON_DATABASE))
    if limit is not None:
        symbols = symbols[:limit]

    print(f"=== Norgate Economic scrape -> {econ_dir()} ({len(symbols)} series) ===")
    results: list[EconSeriesResult] = []
    for sym in symbols:
        r = fetch_series(sym)
        if r is not None:
            results.append(r)
            print(f"  OK  {sym:<10} {r.rows:>6} rows  {r.start}->{r.end}  {r.name}")

    # Manifest
    manifest_path = econ_dir() / "_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"  -> {len(results)}/{len(symbols)} series written; manifest: {manifest_path}")
    return results


def load_econ_series(safe_symbol: str) -> pd.DataFrame:
    """Read one econ series by its safe symbol (e.g. 'CPISA')."""
    path = econ_dir() / f"{safe_symbol}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No econ series at {path}")
    return pd.read_parquet(path)


def main() -> None:
    p = argparse.ArgumentParser(description="Scrape Norgate Economic database to parquet.")
    p.add_argument("--limit", type=int, default=None, help="Only the first N series (sampling).")
    args = p.parse_args()
    scrape(limit=args.limit)


if __name__ == "__main__":
    main()
