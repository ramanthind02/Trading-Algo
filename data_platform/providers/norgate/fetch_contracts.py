"""Fetch and archive individual futures contracts and the adjusted continuous series.

Two permanent archives are maintained under data/norgate/archive/:

  contracts/{TICKER}/{NORGATE_SYMBOL}.parquet
      One file per contract expiry (e.g. ES-2024H).  Incremental — skips
      any file that already exists so re-runs are fast and safe.

  continuous/{TICKER}.parquet
      Full-history additive back-adjusted (_CCB) continuous series.
      Always overwritten on each run to reflect the latest Norgate data.

These directories are never purged by rebuild.py.  They represent the
permanent Norgate archive that remains valid after the subscription ends.

Run directly:
    python -m data_platform.providers.norgate.fetch_contracts
    python -m data_platform.providers.norgate.fetch_contracts --continuous-only
    python -m data_platform.providers.norgate.fetch_contracts --contracts-only
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import norgatedata
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ._constants import (
    PARQUET_COMPRESSION,
    PARQUET_COMPRESSION_LEVEL,
    TICKER_TO_CCB,
    TICKER_TO_CONTRACT_PREFIX,
)
from ._paths import archive_continuous_dir, archive_contracts_ticker_dir
from .fetch_continuous import ensure_norgate_running


@dataclass(frozen=True)
class ContractResult:
    symbol: str
    rows: int


# ── shared helpers ────────────────────────────────────────────────────────

def _fetch_raw(symbol: str) -> pd.DataFrame | None:
    try:
        df = norgatedata.price_timeseries(
            symbol,
            interval="D",
            timeseriesformat="pandas-dataframe",
            padding_setting=norgatedata.PaddingType.NONE,
        )
        return df if (df is not None and not df.empty) else None
    except Exception as exc:
        print(f"    ERR {symbol}: {exc}")
        return None


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical schema: date32 index, float32 OHLC, int32 volume."""
    out = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                              "Close": "close", "Volume": "volume"}).copy()
    out.index = pd.to_datetime(out.index).normalize().tz_localize(None)
    out.index.name = "date"
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in out.columns]
    out = out[keep].sort_index()
    for col in ("open", "high", "low", "close"):
        if col in out.columns:
            out[col] = out[col].astype("float32")
    if "volume" in out.columns:
        out["volume"] = out["volume"].fillna(0).astype("int32")
    return out


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=True),
        path,
        compression=PARQUET_COMPRESSION,
        compression_level=PARQUET_COMPRESSION_LEVEL,
    )


# ── continuous archive ────────────────────────────────────────────────────

def archive_continuous() -> None:
    """Write (or refresh) the permanent adjusted continuous archive."""
    out_dir = archive_continuous_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Adjusted continuous archive -> {out_dir} ===")
    total_rows = 0
    for ticker, symbol in TICKER_TO_CCB.items():
        df = _fetch_raw(symbol)
        if df is None:
            print(f"  WARN  {ticker} ({symbol}): no data returned")
            continue
        normalized = _normalize(df)
        _write(normalized, out_dir / f"{ticker}.parquet")
        total_rows += len(normalized)
        print(f"  OK  {ticker:<5} {symbol:<12}  rows={len(normalized):>6}")
    print(f"  -> {len(TICKER_TO_CCB)} tickers written, {total_rows:,} rows total")


def _root_from_continuous_symbol(symbol: str) -> str:
    """``&6E_CCB`` -> ``6E``, ``&BRN`` -> ``BRN`` (strip leading & and _CCB)."""
    root = symbol.lstrip("&")
    return root[:-4] if root.endswith("_CCB") else root


def archive_all_continuous() -> None:
    """Archive every continuous market in the Norgate Continuous Futures DB.

    Stores both the additive back-adjusted (_CCB) and unadjusted continuous for
    all ~112 markets (not just the 23 repo trading tickers), keyed by the Norgate
    root symbol::

        archive/continuous_full/{ROOT}_CCB.parquet     additive back-adjusted
        archive/continuous_full/{ROOT}_unadj.parquet   unadjusted continuous

    Separate dir from ``archive/continuous/`` (which is repo-ticker-keyed and
    feeds the live pipeline) so the two never collide. This is the permanent
    full-subscription reference for markets we do not currently trade.
    """
    import norgatedata

    out_dir = archive_continuous_dir().parent / "continuous_full"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Full continuous archive -> {out_dir} ===")

    all_cont = sorted(norgatedata.database_symbols("Continuous Futures"))
    written = skipped = errored = 0
    total_rows = 0
    for symbol in all_cont:
        root = _root_from_continuous_symbol(symbol)
        suffix = "CCB" if symbol.endswith("_CCB") else "unadj"
        out_path = out_dir / f"{root}_{suffix}.parquet"
        if out_path.exists():
            skipped += 1
            continue
        df = _fetch_raw(symbol)
        if df is None:
            errored += 1
            continue
        normalized = _normalize(df)
        _write(normalized, out_path)
        written += 1
        total_rows += len(normalized)
        if written % 25 == 0:
            print(f"  ... {written} written ({total_rows:,} rows)")
    print(f"  -> written={written} skipped={skipped} errored={errored} "
          f"({total_rows:,} rows) across {len(all_cont)} symbols")


# ── individual contracts archive ──────────────────────────────────────────

def _all_contract_symbols(prefix: str, all_futures: list[str]) -> list[str]:
    return sorted(s for s in all_futures if s.startswith(prefix + "-"))


def archive_contracts_ticker(
    ticker: str,
    prefix: str,
    all_futures: list[str],
    *,
    max_retries: int = 3,
    retry_sleep_s: float = 1.0,
) -> list[ContractResult]:
    contracts = _all_contract_symbols(prefix, all_futures)
    if not contracts:
        print(f"  WARN  {ticker}: no contracts found (prefix={prefix})")
        return []

    out_dir = archive_contracts_ticker_dir(ticker)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[ContractResult] = []
    skipped = 0

    for sym in contracts:
        out_path = out_dir / f"{sym}.parquet"
        if out_path.exists():
            skipped += 1
            continue

        df: pd.DataFrame | None = None
        for attempt in range(1, max_retries + 1):
            df = _fetch_raw(sym)
            if df is not None:
                break
            if attempt < max_retries:
                time.sleep(retry_sleep_s * attempt)

        if df is None:
            continue

        normalized = _normalize(df)
        _write(normalized, out_path)
        results.append(ContractResult(symbol=sym, rows=len(normalized)))

    total_rows = sum(r.rows for r in results)
    print(
        f"  {ticker:<5} {prefix:<5}  "
        f"{len(results):>4} new / {skipped:>4} cached / {len(contracts):>4} total"
        f"  ({total_rows:>8} rows)"
    )
    return results


def archive_contracts(*, full_universe: bool = True) -> None:
    """Incrementally archive individual contract expiries.

    Parameters
    ----------
    full_universe
        When True (default) archive every prefix found in the Norgate Futures
        database — the complete provider library, not just our 23 trading
        tickers.  Use False to restrict to TICKER_TO_CONTRACT_PREFIX only.
    """
    print(f"\n=== Individual contracts archive -> {archive_contracts_ticker_dir('_').parent} ===")
    all_futures = norgatedata.database_symbols("Futures")

    if full_universe:
        # Build prefix -> ticker-label map: our known tickers get their repo
        # name; everything else uses the Norgate prefix as the folder name.
        known_prefix_to_ticker = {v: k for k, v in TICKER_TO_CONTRACT_PREFIX.items()}
        all_prefixes = sorted(set(s.split("-")[0] for s in all_futures))
        prefix_map = {
            prefix: known_prefix_to_ticker.get(prefix, prefix)
            for prefix in all_prefixes
        }
    else:
        prefix_map = {v: k for k, v in TICKER_TO_CONTRACT_PREFIX.items()}

    grand_new = grand_rows = 0
    for prefix, label in prefix_map.items():
        results = archive_contracts_ticker(label, prefix, all_futures)
        grand_new += len(results)
        grand_rows += sum(r.rows for r in results)
    print(f"  -> {grand_new} new contracts written, {grand_rows:,} rows total")


# ── CLI ───────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Archive Norgate adjusted continuous series and individual contracts."
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--continuous-only", action="store_true",
                   help="Archive only the adjusted continuous series.")
    g.add_argument("--contracts-only", action="store_true",
                   help="Archive only individual contract expiries.")
    g.add_argument("--all-continuous", action="store_true",
                   help="Archive every continuous market in the Norgate DB "
                        "(~112 markets x CCB+unadj) to archive/continuous_full/.")
    p.add_argument("--trading-tickers-only", action="store_true",
                   help="Restrict individual contracts to the 23 repo trading tickers "
                        "(default: archive the full Norgate universe).")
    return p


def main(
    *,
    continuous: bool = True,
    contracts: bool = True,
    full_universe: bool = True,
) -> None:
    ensure_norgate_running()
    if continuous:
        archive_continuous()
    if contracts:
        archive_contracts(full_universe=full_universe)
    print("\nDone.")


if __name__ == "__main__":
    args = _build_parser().parse_args()
    if args.all_continuous:
        ensure_norgate_running()
        archive_all_continuous()
        print("\nDone.")
    else:
        main(
            continuous=not args.contracts_only,
            contracts=not args.continuous_only,
            full_universe=not args.trading_tickers_only,
        )
