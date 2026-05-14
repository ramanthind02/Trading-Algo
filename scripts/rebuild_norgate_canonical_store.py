#!/usr/bin/env python3
"""Rebuild canonical candle data and runtime cache from Norgate end-to-end."""
from __future__ import annotations

import shutil
from pathlib import Path

try:
    from scripts._bootstrap import ensure_project_root_on_path
except ImportError:
    from _bootstrap import ensure_project_root_on_path

PROJECT_ROOT = ensure_project_root_on_path()

from scripts.fetch_norgate_data import main as fetch_norgate_data_main
from scripts.migrate_norgate_to_ohlc import main as rebuild_ohlc_main
from utils.cache.runtime.bootstrap_source_candles import bootstrap_source_candles
from utils.cache.runtime.cache_paths import default_runtime_root
from utils.core.enums import Ticker, TimeFrame


def _purge_path(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
        print(f"  Removed: {path}")


def main() -> None:
    print("=" * 60)
    print("Norgate Canonical Rebuild")
    print("=" * 60)

    runtime_root = default_runtime_root()
    norgate_root = PROJECT_ROOT / "data" / "norgate"
    ohlc_root = PROJECT_ROOT / "data" / "ohlc_data"

    print("\n1) Purging runtime cache + Norgate snapshots...")
    _purge_path(runtime_root)
    _purge_path(norgate_root)

    print("\n2) Fetching full-history Norgate data (adjusted + unadjusted)...")
    fetch_norgate_data_main()

    print("\n3) Rebuilding repository D/W/M candles from Norgate adjusted...")
    rebuild_ohlc_main()

    print("\n4) Rebootstrapping central runtime cache from repository candles...")
    summary = bootstrap_source_candles(
        tickers=list(Ticker),
        timeframes=[TimeFrame.D, TimeFrame.W, TimeFrame.M],
        reset_existing=True,
    )
    print(
        "  Bootstrap summary: "
        f"{summary['success']} success / {summary['failed']} failed / {summary['total']} total"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
