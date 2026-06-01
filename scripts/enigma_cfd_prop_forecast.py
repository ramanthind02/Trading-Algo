#!/usr/bin/env python3
"""
Enigma CFD Prop Forecast Entrypoint
====================================

Thin wrapper that invokes the shared forecast pipeline with the CFD
prop-firm profile: reads ``vault_cfd_prop/``, posts target-allocation
preview to the CFD Telegram channel, then fans out per-MT5-account
lot sizing + batch approval + execution via
:func:`execution.run_mt5_execution.run_cfd_prop_execution`.

Designed to run once per day in alignment with the futures prop schedule
(around 6:00 PM ET, after the CME settlement close). The script:

1. Builds the global forecast (same vault math, but using
   ``vault_cfd_prop/``).
2. Posts a per-ticker target-% preview to Telegram.
3. For each enabled MT5 account in
   ``configs/live_forecast_config_cfd_prop.json``: connects, reads live
   balance/equity, computes lot-level rebalancing intents, and surfaces
   them in a single batch approval message before placing orders.

Usage::

    python scripts/enigma_cfd_prop_forecast.py [--dry-run] [--port N]
    python scripts/enigma_cfd_prop_forecast.py --execute [--dry-run-execute]
    python scripts/enigma_cfd_prop_forecast.py --execute --live  # arms live trading
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
    # Inject the profile flag so argparse in _run_forecast picks it up
    # without us needing to duplicate argument handling here.
    sys.argv.insert(1, "--profile")
    sys.argv.insert(2, "cfd_prop")
    _run_forecast()


if __name__ == "__main__":
    main()
