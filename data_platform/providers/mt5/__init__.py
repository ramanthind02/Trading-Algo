"""MT5 (Darwinex) data adapter.

Modules (imported directly, not re-exported here — they import ``MetaTrader5``
at module top level, which requires the terminal, so we keep the package import
side-effect-free):

  scraper.py        incremental M1 bars + ticks (the scheduled-task job)
  daily_scraper.py  full-history daily (D1) bars for the whole symbol universe
  probes/           dev/diagnostic scripts incl. infer_sessions (session hours)

See README.md and docs/library/Data/mt5_data_scraper.md.
"""
