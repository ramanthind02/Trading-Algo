#!/usr/bin/env python3
"""
Enigma Prop Forecast Entrypoint (DEPRECATED -- futures prop alias)
==================================================================

This script has been renamed to ``enigma_futures_prop_forecast.py`` now
that a separate CFD prop-firm flow exists (``enigma_cfd_prop_forecast.py``).

This file is kept as a thin shim so existing cron jobs / shortcuts continue
to work; it prints a deprecation notice and delegates to the futures
script. Please update your scheduler to call ``enigma_futures_prop_forecast.py``
directly.
"""
from __future__ import annotations

import sys
import warnings

try:
    from scripts._bootstrap import ensure_project_root_on_path
except ImportError:
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from scripts.enigma_futures_prop_forecast import main as _run_futures_prop


def main() -> None:
    warnings.warn(
        "scripts/enigma_prop_forecast.py is deprecated; use "
        "scripts/enigma_futures_prop_forecast.py instead. This shim will be "
        "removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    print(
        "[deprecation] enigma_prop_forecast.py is deprecated; please use "
        "enigma_futures_prop_forecast.py.",
        file=sys.stderr,
    )
    _run_futures_prop()


if __name__ == "__main__":
    main()

