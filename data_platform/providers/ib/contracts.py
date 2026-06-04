"""Archive IB individual futures contracts to parquet.

Mirrors the Norgate individual-contract archive but sourced from IB. Enumerates
each futures root's expiries via reqContractDetails, fetches daily bars per
expiry, and stores to:

    data/ib/contracts/{TICKER}/{LOCAL_SYMBOL}.parquet   e.g. ES/ESM4.parquet

Schema (matches the Norgate archive): date32 index + float32 OHLC + int32 volume,
zstd level 3. Each parquet carries IB metadata (local_symbol, last_trade, con_id)
in file-level metadata. Incremental: skips contracts whose file already exists.

IB only retains ~2 years of expired futures, so this is the *forward* archive
that extends the Norgate archive (which holds the deep history). Together they
give a continuous individual-contract record across the Norgate→IB handover.

The canonical InstrumentId for the continuous root (e.g. ES.XCME) and the IB
root/exchange come from the InstrumentCatalog `source_symbols`.

Run directly (requires TWS/Gateway on 127.0.0.1:7497):
    python -m data_platform.providers.ib.contracts --tickers ES NQ GC
    python -m data_platform.providers.ib.contracts            # all catalog futures
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data_platform.core import load_catalog
from data_platform.core.enums import InstrumentClass

from ._client import IbDataClient, IbExpiry, ib_available

PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 3
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7497
DEFAULT_CLIENT_ID = 88


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    return next(
        (p for p in here.parents if (p / ".git").exists() or (p / "AGENTS.md").exists()),
        here.parents[3],
    )


def ib_contracts_dir() -> Path:
    return _repo_root() / "data" / "ib" / "contracts"


def ib_ticker_dir(ticker: str) -> Path:
    return ib_contracts_dir() / ticker


@dataclass(frozen=True)
class ContractArchiveResult:
    ticker: str
    new: int
    cached: int
    total: int
    rows: int


def _ib_roots_from_catalog(tickers: list[str] | None) -> dict[str, tuple[str, str]]:
    """Return {repo_ticker: (ib_root, ib_exchange)} for futures in the catalog."""
    catalog = load_catalog()
    out: dict[str, tuple[str, str]] = {}
    for inst in catalog.filter(instrument_class=InstrumentClass.FUTURE):
        ss = inst.info.get("source_symbols", {})
        ib_root = ss.get("ib_contfut")
        ib_exch = ss.get("ib_exchange")
        repo_ticker = str(inst.id.symbol)
        if ib_root and ib_exch and (tickers is None or repo_ticker in tickers):
            out[repo_ticker] = (ib_root, ib_exch)
    return out


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index.name = "date"
    for col in ("open", "high", "low", "close"):
        if col in out.columns:
            out[col] = out[col].astype("float32")
    if "volume" in out.columns:
        out["volume"] = out["volume"].astype("int32")
    return out[[c for c in ("open", "high", "low", "close", "volume") if c in out.columns]]


def _write(df: pd.DataFrame, path: Path, expiry: IbExpiry) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=True)
    meta = {
        b"ib_local_symbol": expiry.local_symbol.encode(),
        b"ib_last_trade": expiry.last_trade.encode(),
        b"ib_con_id": str(expiry.con_id).encode(),
        b"ib_exchange": expiry.exchange.encode(),
        b"ib_multiplier": expiry.multiplier.encode(),
    }
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), **meta})
    pq.write_table(table, path, compression=PARQUET_COMPRESSION,
                   compression_level=PARQUET_COMPRESSION_LEVEL)


def archive_ticker(
    client: IbDataClient, ticker: str, ib_root: str, ib_exchange: str,
    *, pace_s: float = 0.3,
) -> ContractArchiveResult:
    expiries = client.list_futures_expiries(ib_root, ib_exchange)
    out_dir = ib_ticker_dir(ticker)
    out_dir.mkdir(parents=True, exist_ok=True)

    new = cached = rows = 0
    for exp in sorted(expiries, key=lambda e: e.last_trade):
        path = out_dir / f"{exp.local_symbol}.parquet"
        if path.exists():
            cached += 1
            continue
        df = client.fetch_contract_daily(exp)
        if df.empty:
            continue
        normalized = _normalize(df)
        _write(normalized, path, exp)
        new += 1
        rows += len(normalized)
        time.sleep(pace_s)  # respect IB pacing limits

    result = ContractArchiveResult(ticker, new, cached, len(expiries), rows)
    print(f"  {ticker:<5} {ib_root:<5}@{ib_exchange:<6} "
          f"{new:>4} new / {cached:>4} cached / {len(expiries):>4} expiries ({rows:>7} rows)")
    return result


def archive(
    tickers: list[str] | None = None,
    *, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, client_id: int = DEFAULT_CLIENT_ID,
) -> list[ContractArchiveResult]:
    roots = _ib_roots_from_catalog(tickers)
    if not roots:
        print("No matching futures in catalog (run to_catalog first).")
        return []

    client = IbDataClient()
    if not client.connect_and_start(host, port, client_id):
        raise RuntimeError(f"Could not connect to IB at {host}:{port} — start TWS/Gateway with API enabled.")

    print(f"=== IB individual-contract archive -> {ib_contracts_dir()} ===")
    try:
        results = [archive_ticker(client, tk, root, exch) for tk, (root, exch) in sorted(roots.items())]
    finally:
        client.disconnect()
    grand_new = sum(r.new for r in results)
    grand_rows = sum(r.rows for r in results)
    print(f"  -> {grand_new} new contracts, {grand_rows:,} rows across {len(results)} tickers")
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Archive IB individual futures contracts.")
    p.add_argument("--tickers", nargs="*", default=None, help="Repo tickers (default: all catalog futures).")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--client-id", type=int, default=DEFAULT_CLIENT_ID)
    args = p.parse_args()
    if not ib_available(args.host, args.port):
        print(f"IB not reachable on {args.host}:{args.port} (or ibapi not installed).")
        return
    archive(args.tickers, host=args.host, port=args.port, client_id=args.client_id)


if __name__ == "__main__":
    main()
