"""Rollover-cost execution study.

Quantifies the cost of the daily swap-avoidance overlay on Darwinex index/commodity
CFDs: flatten before the 00:00-broker-time rollover, re-enter after the 01:00 reopen.
Measures optimal exit timing, market-vs-limit execution, entry limit placement,
aggressive-chase-vs-skip, and net savings vs holding through the rollover.

ALL timestamps in data/mt5_data are broker time (EET) mislabelled UTC — the rollover
is at 00:00 broker time. See docs/library/Data/mt5_timezones.md.
"""
