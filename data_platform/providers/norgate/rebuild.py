"""End-to-end Norgate canonical data rebuild.

Step 1  Purge working continuous dirs + runtime cache.
        (Archives in data/norgate/archive/ are NEVER touched.)

Step 2  Fetch adjusted + unadjusted continuous series → working dirs.

Step 3  Archive adjusted continuous (permanent, overwrites latest).
        Archive individual contracts (incremental, skips cached).

Step 4  Migrate working dirs → data/ohlc_data/ (D/W/M + unadj).

Step 5  Bootstrap central runtime cache from ohlc_data.

Run directly:
    python -m data_platform.providers.norgate.rebuild
    python -m data_platform.providers.norgate.rebuild --skip-archive   # skip step 3 (faster)
    python -m data_platform.providers.norgate.rebuild --skip-cache     # skip step 5
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ._paths import norgate_root, working_adjusted_dir, working_unadjusted_dir
from .fetch_continuous import ensure_norgate_running, fetch_all
from .fetch_contracts import archive_continuous, archive_contracts
from .fetch_specs import fetch_specs, save as save_specs
from .migrate import migrate_all

from utils.cache.runtime.bootstrap_source_candles import bootstrap_source_candles
from utils.cache.runtime.cache_paths import default_runtime_root, project_root
from utils.core.enums import Ticker, TimeFrame


def _purge(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
        print(f"  Removed: {path}")


def rebuild(*, skip_archive: bool = False, skip_cache: bool = False) -> None:
    from ._constants import TICKER_TO_CCB, TICKER_TO_RAW
    from ._paths import working_adjusted_dir, working_unadjusted_dir

    ensure_norgate_running()

    print("=" * 60)
    print("Norgate Canonical Rebuild")
    print("=" * 60)

    # ── step 1: purge working dirs + runtime cache ────────────────────────
    print("\n1) Purging working continuous dirs + runtime cache...")
    _purge(working_adjusted_dir())
    _purge(working_unadjusted_dir())
    _purge(default_runtime_root())

    # ── step 2: fetch working continuous series ───────────────────────────
    print("\n2) Fetching continuous series (adjusted + unadjusted)...")
    from .fetch_continuous import fetch_all
    fetch_all(TICKER_TO_CCB, working_adjusted_dir(),   "back-adjusted (CCB)")
    fetch_all(TICKER_TO_RAW, working_unadjusted_dir(), "unadjusted")

    # ── step 3: update permanent archives ────────────────────────────────
    if not skip_archive:
        print("\n3) Updating permanent archives...")
        archive_continuous()
        archive_contracts()
        print("\n3b) Fetching contract specs (point values, tick sizes, margins)...")
        save_specs(fetch_specs())
    else:
        print("\n3) Skipping permanent archive update (--skip-archive).")

    # ── step 4: migrate to ohlc_data ─────────────────────────────────────
    print("\n4) Migrating to ohlc_data (D/W/M + unadj)...")
    migrate_all()

    # ── step 5: bootstrap runtime cache ──────────────────────────────────
    if not skip_cache:
        print("\n5) Bootstrapping central runtime cache...")
        summary = bootstrap_source_candles(
            tickers=list(Ticker),
            timeframes=[TimeFrame.D, TimeFrame.W, TimeFrame.M],
            reset_existing=True,
        )
        print(
            f"  Bootstrap: {summary['success']} ok / "
            f"{summary['failed']} failed / {summary['total']} total"
        )
    else:
        print("\n5) Skipping runtime cache bootstrap (--skip-cache).")

    print("\nDone.")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="End-to-end Norgate canonical data rebuild.")
    p.add_argument("--skip-archive", action="store_true",
                   help="Skip updating the permanent archive (faster dev rebuild).")
    p.add_argument("--skip-cache", action="store_true",
                   help="Skip the runtime cache bootstrap step.")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    rebuild(skip_archive=args.skip_archive, skip_cache=args.skip_cache)


if __name__ == "__main__":
    main()
