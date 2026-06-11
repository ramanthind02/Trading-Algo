# `tests/live_mt5/` — live MT5 ⇄ Nautilus adapter tests (FTMO demo)

End-to-end tests that drive the **real** vendored adapter classes
(`mt5connect.connection.MT5Connection`, `MT5InstrumentProvider`, `MT5DataClient`,
`MT5LiveExecutionClient`) and a real `nautilus_trader` `TradingNode` against a
**prop-firm DEMO** terminal (FTMO-Demo). These are *not* unit tests — for the
mock-based unit suite see `deployment/nautilus_mt5/vendor/mt5-connect/tests/`.

Full operator runbook + reference results:
[`deployment/nautilus_mt5/CONNECTION_TEST.md`](../../deployment/nautilus_mt5/CONNECTION_TEST.md).

## Safety model (enforced in `conftest.py`)

- **One terminal per process** — every fixture binds via `config.path` to the
  FTMO terminal, so this process talks only to FTMO and runs **in parallel with
  the Darwinex tick-scraper**, never touching it.
- **Hard guard** — the session aborts (`pytest.exit`) unless the bound account is
  the FTMO demo login, on the FTMO server, with `trade_mode == DEMO`. Nothing
  downstream can place an order on the wrong account.
- **Dedicated magic** — order tiers tag everything with `TEST_MAGIC = 990510`
  (≠ the adapter's production 510) and flatten that magic on teardown.
- **Opt-in** — skipped unless `MT5_LIVE_TESTS=1`; order tiers also need
  `MT5_LIVE_ORDERS=1`. A plain `pytest tests/` skips this whole directory.
- Order tiers self-skip when the market is closed (tick-freshness gate —
  `order_check` returns "Done" even on a closed weekend market, so it's unreliable).

## Tiers

| File | Tier | Orders? | Covers |
|---|---|---|---|
| `test_connection_live.py` | 1 | no | `MT5Connection.connect()` → CONNECTED via `path`, account snapshot, terminal info, DEMO guard |
| `test_instruments_live.py` | 1 | no | `MT5InstrumentProvider` parses real FTMO symbols (EURUSD→CurrencyPair, US100.cash→Cfd); filtered `load_all_async` |
| `test_data_live.py` | 1 | no | live tick + H1 bar parsing on real data; `MT5DataClient` construction |
| `test_execution_live.py` | 2 | **yes** | EURUSD round-trip through `_submit_order`/`_poll_exec_once`/`_cancel_order`; live reject path |
| `test_node_live.py` | 3 | conn: no / order: yes | full `TradingNode` build/run/stop via `build_mt5_node_config`; gated node order |

## Running

Prereqs: FTMO MT5 terminal open + logged into the demo + **Algo Trading enabled**;
`FTMO_DEMO_{SERVER,LOGIN,PASSWORD}` in gitignored `.env`; optional
`FTMO_DEMO_TERMINAL_PATH` override.

```powershell
# Read-only tiers (no orders)
$env:MT5_LIVE_TESTS = "1"
.\.venv\Scripts\python.exe -m pytest tests\live_mt5 -v -rs

# Also place guarded demo orders (round-trip + node order; needs an OPEN market)
$env:MT5_LIVE_ORDERS = "1"
.\.venv\Scripts\python.exe -m pytest tests\live_mt5 -v -rs
```

Or use the runner (binds only FTMO, parallel to the Darwinex scraper):

```
deployment\ops\run_mt5_ftmo_tests.bat            # read-only tiers
deployment\ops\run_mt5_ftmo_tests.bat orders     # + demo orders
```
