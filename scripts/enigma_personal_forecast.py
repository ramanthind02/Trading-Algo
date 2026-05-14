#!/usr/bin/env python3
"""
Enigma Personal Forecast Entrypoint
====================================

Thin wrapper that invokes the shared forecast pipeline with the personal
profile: reads ``vault_personal/``, sizes ETF fractional shares (SPY, QQQ,
GLD, IWM, DIA, TLT), builds a **session-only** partial daily row from
15-minute bars (``use_rth=0`` so extended and overnight data are included;
**not** written to the central cache), and posts to the personal-account
Telegram channel.

Designed to run once per day around 3:45 PM ET -- 15 minutes before the US
equity market close at 4:00 PM ET so orders can be placed while fractional
share trading is still enabled on IB.

Usage (delegates everything else to ``enigma_live_forecast.main``)::

    python scripts/enigma_personal_forecast.py [--dry-run] [--capital N] [--port N]
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
    sys.argv.insert(1, "--profile")
    sys.argv.insert(2, "personal")
    _run_forecast()


if __name__ == "__main__":
    main()
