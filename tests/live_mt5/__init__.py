"""Live broker end-to-end tests for the vendored MT5 ⇄ Nautilus adapter.

These are NOT unit tests. They drive the REAL adapter classes
(``mt5connect.connection.MT5Connection``, ``MT5InstrumentProvider``,
``MT5DataClient``, ``MT5LiveExecutionClient``, and a real
``nautilus_trader`` ``TradingNode``) against a **prop-firm DEMO** MT5
terminal (FTMO-Demo).

They are gated behind environment flags and a hard DEMO/login/server guard
(see ``conftest.py``) so they:
  * never run by default (a plain ``pytest tests/`` skips them),
  * bind only to the FTMO terminal (never the live Darwinex tick-scraper),
  * can never place an order on a non-demo account.

Run them with ``deployment\ops\\run_mt5_ftmo_tests.bat`` or by setting
``MT5_LIVE_TESTS=1`` (and ``MT5_LIVE_ORDERS=1`` for the order tiers).
"""
