# `deployment/nautilus_mt5/` — vendored MT5 adapter for NautilusTrader

This directory vendors the community **MT5 ⇄ NautilusTrader** adapter
(`aulekator/mt5-connect`, packaged as `mt5connect`). Nautilus ships an
Interactive Brokers adapter but **no MT5 adapter**, so the CFD/MT5 venue in
WP-4 depends on this fork. The adapter is hobby-grade (~6 commits, Windows-only)
— per the WP-4 decision log it **must be vendored, pinned, audited, and
paper-validated before any funded account**.

> **SAFETY:** vendoring is source-only. Nothing here connects to a terminal or
> places an order. The live Darwinex terminal must NOT be used for any
> sandbox/demo work — those rungs require a SEPARATE broker terminal instance.

## Vendor pin

| Field | Value |
|---|---|
| Upstream repo | `https://github.com/aulekator/mt5-connect.git` |
| Vendored path | `deployment/nautilus_mt5/vendor/mt5-connect/` |
| **Pinned commit SHA** | **`c6767d4b4bd7f1396898abf67121ddcf4183653c`** |
| Pinned commit subject | `Fix repo URLs to match mt5-connect repository name` |
| Vendored on | 2026-06-05 |
| License | MIT (`vendor/mt5-connect/LICENSE`, © 2025 nautilus-mt5 contributors) — preserved verbatim |
| Package | `mt5connect/` (config, connection, data, execution, factories, providers, downloader, parsing, errors, constants) |

### Re-pin / refresh procedure

```powershell
# from repo root
cd deployment\nautilus_mt5\vendor\mt5-connect
git fetch origin
git checkout <new-sha>          # then update the SHA table above + re-run the audit
git rev-parse HEAD              # record the new pin
```

To update the pin, change the SHA above, re-run the **audit checklist**, and
re-run the import smoke test. Never bump the pin without re-auditing.

## Why vendored (not `pip install`)

The WP-4 decision log forbids depending on the PyPI dev package for funded
accounts. Vendoring gives us a frozen, audited, reviewable copy whose exact
bytes we control, independent of upstream force-pushes or yanks.

## Audit checklist (MUST be completed before any demo → funded step)

The audit verifies the adapter reproduces our current `mt5_trade_executor`
semantics exactly. A missing item is a release blocker.

- [ ] **License**: `vendor/mt5-connect/LICENSE` is MIT and preserved unmodified.
- [ ] **Dependency surface**: review `vendor/mt5-connect/pyproject.toml`; confirm
      no surprising/transitive deps; confirm `MetaTrader5` + `nautilus_trader`
      version pins are compatible with our env.
- [ ] **Order lifecycle — open** (`mt5connect/execution.py`): market order build
      maps to `TRADE_ACTION_DEAL` with `ORDER_FILLING_IOC` / `ORDER_TIME_GTC`,
      a deviation guard, and our **magic-number** isolation. Compare against
      `execution/mt5_trade_executor.py::_market_order` (`:288`).
- [ ] **Order lifecycle — close (CRITICAL)**: confirm closes send an
      **opposite-side IOC deal with `position=<ticket>`** so MT5 treats it as a
      partial/full close of that *specific* ticket (NOT a new hedge). This is the
      single most important mechanic — a NETTING-style close would open hedges on
      hedging-mode accounts. Compare against `mt5_trade_executor.py::close_ticket`
      (`:373`) and the oldest-first partial-close summing in
      `mt5_rebalancer.py::_build_partial_close_actions` (`:148`).
- [ ] **Retry/retcodes**: retry set + success retcode (`TRADE_RETCODE_DONE`)
      match our `RETRYABLE_RETCODES` / `max_retries` behaviour.
- [ ] **Reconnection** (`mt5connect/connection.py`): exponential backoff
      (`reconnect_initial_delay_s`/`max_delay_s`/`max_attempts`) — verify it does
      not silently drop orders/fills across a reconnect; verify the advertised
      startup + continuous **account reconciliation** populates Nautilus
      `OrderStatusReport` / `FillReport` / `PositionStatusReport`.
- [ ] **Symbol suffix handling**: the adapter advertises automatic suffix
      normalisation. Confirm it does NOT silently remap a symbol we did not
      intend — our broker-aware mapping (`data_platform/providers/mt5/brokers.py`)
      is the authority for canonical ⇄ broker symbol; the adapter must receive the
      already-resolved broker symbol.
- [ ] **Hedging vs netting OMS**: confirm the adapter/instrument config we use
      matches FTMO's account mode; reproduce per-ticket closes in the sandbox
      before any demo order.
- [x] **Adapter unit tests**: `vendor/mt5-connect/tests/` pass under our venv —
      **604 pass** (2026-06-06, incl. the new `TestTerminalPath` coverage for the
      `path` fix). Run: `python -m pytest deployment/nautilus_mt5/vendor/mt5-connect/tests -q`.
- [x] **Import smoke** (offline, no terminal):
      `python -c "import sys; sys.path.insert(0, r'deployment/nautilus_mt5/vendor/mt5-connect'); import mt5connect"`
      — note: importing pulls in `MetaTrader5`, which is Windows-only and may
      require the package installed; this only confirms the source is importable,
      it does NOT connect.

### Validation status (2026-06-06)

The **`MT5Config.path` gap is CLOSED** — `MT5Connection` now binds deterministically
to a named terminal, so the adapter cannot attach to the wrong broker on this
multi-terminal machine (unit-covered by `TestTerminalPath`; live-proven by the
suite below).

The live pytest suite [`tests/live_mt5/`](../../tests/live_mt5/) (see its
[`README`](../../tests/live_mt5/README.md)) drives the real adapter against the
FTMO demo. As of 2026-06-06 it **PASSES**: connection lifecycle, instrument
parsing, live tick/bar parsing, data-client construction, the live order **reject**
path, and a full `TradingNode` build/connect/stop. **Still pending an open-market
run** (the order tiers self-skip on weekends): the open→fill→close round-trip
through `_submit_order`/`_cancel_order` — i.e. the **CRITICAL per-ticket close**
and **hedging-vs-netting** audit items above are not yet live-validated. Re-run
`deployment\ops\run_mt5_ftmo_tests.bat orders` during market hours to complete them.

## Connecting + testing against a broker terminal

See **[`CONNECTION_TEST.md`](CONNECTION_TEST.md)** — the operator runbook for connecting the
adapter to a prop-firm **demo** and validating it end-to-end. Covers the
**one-terminal-install-per-broker** rule (and the `-10005 IPC timeout` you hit otherwise), the
`path` terminal binding, Algo-Trading / market-execution gotchas, the hard DEMO guard, and the
confirmed broker symbol mappings.

Two ways to validate:

- **Automated (preferred):** the live pytest suite **[`tests/live_mt5/`](../../tests/live_mt5/)**
  ([README](../../tests/live_mt5/README.md)) drives the *real* adapter classes + a real
  `TradingNode` against the FTMO demo, gated behind `MT5_LIVE_TESTS=1` /`MT5_LIVE_ORDERS=1` and a
  hard DEMO guard. Runner: `deployment\ops\run_mt5_ftmo_tests.bat`.
- **One-shot smoke:** `scripts/dev/mt5_adapter_test.py` (print-based). First validated on
  FTMO-Demo 2026-06-05 (round-trip order passed).

## Per-broker configuration

Broker-specific facts — symbol names, timezone, asset classes, **market hours/sessions**, and
**rules** (filling/hedging/lots; prop-firm daily/total-loss + profit-target) — are declared in
[`configs/mt5_brokers.yaml`](../../configs/mt5_brokers.yaml) and read via
[`data_platform/providers/mt5/brokers.py`](../../data_platform/providers/mt5/brokers.py). The
adapter receives the already-resolved native symbol and `terminal_path` from there. Full reference:
[`docs/library/Data/mt5_broker_config.md`](../../docs/library/Data/mt5_broker_config.md).

## How the sandbox config consumes this

`deployment/live/config_mt5_sandbox.py` builds a Nautilus `TradingNodeConfig`
that pairs the vendored MT5 **live `DataClient`** (real quotes) with Nautilus's
built-in **`SandboxExecutionClient`** (local virtual fills — orders never leave
the machine). See that file's docstring for the connection prerequisite (a
SEPARATE FTMO terminal; the live Darwinex terminal must NOT be used) and the
explicit "do NOT run yet" gate.

> _Verified against current code via CodeGraph on 2026-06-07._
