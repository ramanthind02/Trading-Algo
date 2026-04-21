#!/usr/bin/env python3
"""
Enigma Prop Forecast Entrypoint
================================

Thin wrapper that invokes the shared forecast pipeline with the prop-firm
profile: reads ``vault/``, sizes micro futures contracts (MES, MNQ, MGC,
M2K, MYM, ZN proxy for TLT), and posts to the prop-firm Telegram channel.

Designed to run once per day around 6:00 PM ET -- after the CME settlement
window ends and the official daily candle has closed at 5:00 PM ET.

Usage (delegates everything else to ``enigma_live_forecast.main``)::

    python scripts/enigma_prop_forecast.py [--dry-run] [--capital N] [--port N]
"""
from __future__ import annotations

import sys

try:
    from scripts._bootstrap import ensure_project_root_on_path
except ImportError:
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from scripts.enigma_live_forecast import main as _run_forecast


def main() -> None:
    # Insert the profile flag so argparse in _run_forecast picks it up without
    # us needing to duplicate argument handling here.
    sys.argv.insert(1, "--profile")
    sys.argv.insert(2, "prop")
    _run_forecast()


if __name__ == "__main__":
    main()
