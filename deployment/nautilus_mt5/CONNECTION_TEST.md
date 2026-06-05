# MT5 ⇄ Nautilus adapter — connection test runbook

How to connect the vendored `mt5connect` adapter to a **prop-firm demo** account
and validate it end-to-end. This is **ladder rung-2 evidence** (real broker demo,
real order lifecycle) for WP-4 — see [`docs/refactor/nautilus/04_live_execution_trading_node.md`](../../docs/refactor/nautilus/04_live_execution_trading_node.md)
and the ledger `docs/refactor/nautilus/_workflow_state/WP-4.md` §8.

> **🔴 SAFETY**
> - The test driver has a **hard DEMO guard**: it aborts before any order unless
>   `account_info().trade_mode == DEMO`. It is structurally unable to trade a live account.
> - Credentials live ONLY in gitignored `.env` (`<BROKER>_DEMO_*`), never in code or docs.
> - Use the prop-firm's OWN terminal install — **never** the live Darwinex terminal
>   (real account `4000093084`). With separate installs they never interfere.

First validated: **FTMO-Demo, 2026-06-05** — all layers + a EURUSD round-trip order PASSED.

---

## The core finding: one MT5 terminal install **per broker**

The `MetaTrader5` Python package does not connect to a broker directly — it drives a
**running MT5 terminal process** over local IPC, and **each terminal install ships only
its own broker's server list**. Consequences:

- You **cannot** log a Darwinex terminal into `FTMO-Demo` or `FundedNext-Server3` — the
  server name is unresolvable, so `mt5.login(...)` fails with **`-10005 IPC timeout`**.
  (Confirmed for both FTMO and FundedNext against the Darwinex terminal.)
- To use a broker you must install **that broker's** MT5 terminal (download it from the
  firm's client area), open it once, and log into the demo manually.
- Bind the Python bridge to a specific terminal with
  **`mt5.initialize(path=<that terminal's terminal64.exe>)`**.
- Nautilus is **one-node-per-process**, so two brokers (e.g. live Darwinex + demo FTMO) =
  **two terminal installs = two processes**.

### Terminal registry (this machine)

| Broker | `terminal64.exe` | Data folder | Account |
|---|---|---|---|
| Darwinex (live) | `C:\Program Files\MetaTrader 5\terminal64.exe` | `…\Terminal\D0E8…075` (base `Darwinex-Live`) | `4000093084` (REAL — do not disturb) |
| FTMO (demo) | `C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe` | `…\Terminal\81A933…3850` (base `FTMO-Demo`) | `1513568029` (DEMO) |

> To find a terminal's exe from its data-folder GUID, read `<data-folder>\origin.txt`
> (it points at the install dir). The broker server lists it knows are the subfolders
> under `<data-folder>\bases\`.

---

## Prerequisites (one-time per broker)

1. **Install the broker's MT5 terminal** into its **own folder** (do NOT overwrite another
   broker's install).
2. **Open it and log into the demo account** so it is running and connected.
3. **Enable Algo Trading** — toolbar "Algo Trading" button must be **green**
   (or Tools → Options → Expert Advisors → ✓ *Allow algorithmic trading*). Otherwise
   `order_send` returns **`10027` "Autotrading disabled by client"**.
4. Put the demo creds in gitignored `.env`:
   ```
   FTMO_DEMO_SERVER=FTMO-Demo
   FTMO_DEMO_LOGIN=1513568029
   FTMO_DEMO_PASSWORD=…
   ```
   (Use the `<BROKER>_DEMO_*` naming; the driver's `--broker` flag selects the prefix.)

---

## Running the test

Driver: [`scripts/dev/mt5_adapter_test.py`](../../scripts/dev/mt5_adapter_test.py)
(broker-agnostic; reads `<BROKER>_DEMO_*` from `.env`).

```powershell
# Full test incl. a guarded demo round-trip order:
.\.venv\Scripts\python.exe scripts\dev\mt5_adapter_test.py `
    --broker FTMO --path "C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"

# Connectivity only (no order):
... --no-order

# Force the test-order symbol:
... --order-symbol EURUSD
```

### What it validates (mirrors the adapter's real layers)

| Layer | Adapter surface exercised |
|---|---|
| Bind | `mt5.initialize(path=…, login, server, password)` selects the correct terminal |
| 1–3 | `MT5Connection` initialize → login → `get_account_info()` + **DEMO guard** |
| 4 | `MT5InstrumentProvider.load_symbol` → parses to a real Nautilus instrument |
| 5–6 | live tick (`symbol_info_tick`) + historical bars (`copy_rates_range`) |
| 7–8 | `MT5DataClient` + `MT5LiveExecutionClient` construct against real NT msgbus/cache |
| 9 | round-trip **open → confirm position → close → flat**, using the adapter's exact `order_send` request shape; fills verified from `history_deals_get` |

### Reference result (FTMO-Demo, 2026-06-05)

```
account: FTMO-Demo  bal=100000.00 USD  lev=1:30  trade_mode=DEMO
load_symbol('US100.cash') -> Nautilus Cfd          price_precision=2
load_symbol('EURUSD')     -> Nautilus CurrencyPair price_precision=5
order round-trip:
  BUY  0.01 EURUSD @ 1.16150 (IN)  commission=-0.03
  SELL 0.01 EURUSD @ 1.16145 (OUT) commission=-0.03  profit=-0.05
  -> account flat, 0 residual    (net ≈ -$0.11 = ½-spread + 2× commission)
```

---

## Gotchas (all handled by the driver)

| Symptom | Cause | Handling |
|---|---|---|
| `-10005 IPC timeout` on login | terminal doesn't know that broker's server | install the broker's own terminal; bind via `--path` |
| `order_send` → `10027` | Algo Trading disabled in the terminal | enable the Algo Trading toolbar button |
| `order_send` result has `deal=0 price=0` despite `retcode=10009` | MT5 **market-execution** populates the deal asynchronously | treat `history_deals_get` as authoritative; settle-delay before reading |
| "1 residual position" right after close | positions table lags the fill | driver retries the flat-check with a settle delay |
| order placed on a non-demo account | — | **impossible**: hard DEMO guard aborts first |

If a test ever leaves a position open, flatten it: query `positions_get()` filtered by
`magic == 510` (the adapter's `MT5_MAGIC_NUMBER`) and send an opposite-side
`TRADE_ACTION_DEAL` with `position=<ticket>`.

---

## Known adapter gap — `MT5Config` has no `path`

`mt5connect.config.MT5Config` exposes no terminal-`path` field, and
`MT5Connection._initialize()` calls bare `mt5.initialize()`. On a machine with **multiple
terminal installs** that binds **non-deterministically** (whichever terminal is registered).

- **Workaround (what the driver does):** call `mt5.initialize(path=…, login, server, password)`
  **before** `MT5Connection.connect()`. A subsequent bare `mt5.initialize()` then attaches to
  that same already-bound terminal.
- **Planned fix (WP-4 edit unit 1, when vendoring):** add a `path: str | None` field to
  `MT5Config` and thread it into `_initialize()`. Track in `deployment/nautilus_mt5/README.md`
  audit notes.

---

## Confirmed broker symbol mappings

Native symbol names differ per broker and are the single source of truth in
[`configs/mt5_brokers.yaml`](../../configs/mt5_brokers.yaml) (resolve via
`data_platform.providers.mt5.brokers.resolve(broker, canonical)`). FTMO was confirmed by
enumerating all 166 symbols (each canonical matched exactly one native symbol — **no guessing**):

| Canonical | FTMO (`.cash` = index/energy CFD) | Darwinex |
|---|---|---|
| NQ | `US100.cash` | `NDX` |
| ES | `US500.cash` | `SP500` |
| YM | `US30.cash` | `WS30` |
| DAX | `GER40.cash` | `GDAXI` |
| FTSE | `UK100.cash` | `UK100` |
| GC | `XAUUSD` | `XAUUSD` |
| SI | `XAGUSD` | `XAGUSD` |
| CL | `USOIL.cash` (WTI; Brent = `UKOIL.cash`) | `XTIUSD` |
| NG | `NATGAS.cash` | `XNGUSD` |
| FX (EURUSD, …) | 1:1 (unsuffixed) | 1:1 |

---

## After testing

- If you switched the Darwinex terminal away from its live account at any point (only needed
  before separate installs existed), **re-login it manually**.
- The test is read-mostly + a single self-closing micro-order; it leaves the demo account flat.
