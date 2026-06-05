"""Fetch and persist contract specification metadata from Norgate.

Writes two files to data/norgate/archive/:

  contract_specs.json
      One entry per repo ticker.  Fields that are stable over time:
      point_value, tick_size, tick_value, currency, exchange, asset_class,
      market_name, history_start.  These are the constants needed for
      position sizing (ContractSpec.multiplier = point_value) and P&L
      simulation (daily_pnl = contracts * delta_points * point_value).

  contract_specs.parquet
      Same data in columnar form for easy DataFrame access.

The margin field from Norgate is a single snapshot (current exchange initial
margin), not a time series.  It is included for reference but should not be
used for historical backtesting — use the vol target as the binding constraint.

Run directly:
    python -m data_platform.providers.norgate.fetch_specs
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ._constants import PARQUET_COMPRESSION, PARQUET_COMPRESSION_LEVEL, TICKER_TO_CCB
from ._paths import archive_contracts_dir
from .fetch_continuous import ensure_norgate_running


def _archive_dir() -> Path:
    return archive_contracts_dir().parent  # data/norgate/archive/


def fetch_specs() -> list[dict]:
    import norgatedata  # local import — only needed on Norgate host

    rows: list[dict] = []
    for ticker, sym in sorted(TICKER_TO_CCB.items()):
        row: dict = {"ticker": ticker, "norgate_symbol": sym}

        for field in (
            "point_value",
            "tick_size",
            "lowest_ever_tick_size",
            "margin",
            "currency",
            "futures_market_name",
            "exchange_name",
            "exchange_name_full",
            "subtype1",
        ):
            try:
                row[field] = getattr(norgatedata, field)(sym)
            except Exception:
                row[field] = None

        # Derived
        pv = row.get("point_value")
        ts = row.get("tick_size")
        row["tick_value"] = round(pv * ts, 8) if pv and ts else None

        try:
            row["history_start"] = str(
                norgatedata.first_quoted_date(sym, datetimeformat="iso")
            )
        except Exception:
            row["history_start"] = None

        rows.append(row)
        print(
            f"  {ticker:<5} {row['futures_market_name']:<35}"
            f"  pv={str(pv):<10} tick={str(ts):<12}"
            f"  tv=${str(row['tick_value']):<8}  {row['exchange_name']}"
        )

    return rows


def save(rows: list[dict]) -> None:
    out_dir = _archive_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON — human-readable, version-controlled friendly
    json_path = out_dir / "contract_specs.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"  -> {json_path}")

    # Parquet — fast DataFrame access
    df = pd.DataFrame(rows)
    parquet_path = out_dir / "contract_specs.parquet"
    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=False),
        parquet_path,
        compression=PARQUET_COMPRESSION,
        compression_level=PARQUET_COMPRESSION_LEVEL,
    )
    print(f"  -> {parquet_path}")


def load() -> pd.DataFrame:
    """Load the persisted specs without requiring Norgate to be running."""
    path = _archive_dir() / "contract_specs.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Contract specs not found at {path}. "
            "Run: python -m data_platform.providers.norgate.fetch_specs"
        )
    return pd.read_parquet(path)


def main() -> None:
    ensure_norgate_running()
    print("=== Fetching contract specs ===")
    rows = fetch_specs()
    print("\nSaving...")
    save(rows)
    print(f"\nDone. {len(rows)} tickers written.")


if __name__ == "__main__":
    main()
