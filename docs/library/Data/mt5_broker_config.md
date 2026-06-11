# MT5 broker config — symbols, sessions, rules, asset classes

> **One-line rule:** every per-broker difference (symbol names, timezone,
> asset-class membership, market hours, order mechanics, prop-firm risk limits)
> lives in **`configs/mt5_brokers.yaml`** and is read through typed, terminal-free
> accessors in **`data_platform/providers/mt5/brokers.py`**. Never hardcode a
> broker quirk anywhere else.

We trade multiple MT5 brokers and **each is slightly different**: Darwinex calls
the Nasdaq `NDX`, FTMO calls it `US100.cash`; FTMO forbids weekend holding and
enforces a 5%/10% loss limit, Darwinex (live retail) has no such rules; index
sessions differ; some brokers stream crypto on weekends, others don't. This page
documents the config schema and the accessor API that makes those differences
data, not code.

Related: [[mt5_timezones]] (the EET/EEST timestamp trap this builds on),
[[mt5_data_scraper]] (per-broker data namespace), and the operator runbook
[`deployment/nautilus_mt5/CONNECTION_TEST.md`](../../../deployment/nautilus_mt5/CONNECTION_TEST.md).

---

## Where things live

| Thing | Path |
|---|---|
| Config (single source of truth) | [`configs/mt5_brokers.yaml`](../../../configs/mt5_brokers.yaml) |
| Accessors (pure, no terminal) | [`data_platform/providers/mt5/brokers.py`](../../../data_platform/providers/mt5/brokers.py) |
| Unit tests | [`tests/data_platform/test_broker_config.py`](../../../tests/data_platform/test_broker_config.py) |
| Validate + summarize | `python -m data_platform.providers.mt5.brokers` |

> **SAFETY:** importing or calling anything in `brokers.py` **never** touches a
> terminal. Symbol *discovery* (`mt5.symbols_get()`) is a separate, explicit step
> run against a SEPARATE broker terminal — never the live Darwinex one.

---

## Config schema (`configs/mt5_brokers.yaml`)

Each broker is one block under `brokers:`. Only `symbols` is required for the
legacy resolver; the rest are additive.

```yaml
brokers:
  ftmo:
    display_name: "FTMO"
    server: "FTMO-Demo"            # informational; creds live in .env, never here
    confirmed: true                # symbols verified on that broker's own terminal
    confirmed_on: "2026-06-05"
    terminal_path: 'C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe'
    timezone: "EET"                # broker server tz label (EET winter / EEST summer)

    symbols:                       # canonical -> native (resolve / canonical_for)
      NQ: US100.cash
      EURUSD: EURUSD
      # …

    candidates:                    # plausible native names to probe during discovery
      NQ: [US100, NAS100, NDX, USTEC]

    asset_classes:                 # canonical -> fx|index|metal|energy|crypto
      NQ: index
      GC: metal
      CL: energy
      EURUSD: fx

    sessions:                      # per asset-class market hours, in the broker tz
      fx:     {weekdays: "0-4", open: "00:00", close: "24:00", daily_break: "23:58-00:02"}
      index:  {weekdays: "0-4", open: "01:00", close: "23:15"}   # APPROX
      crypto: {weekdays: "0-6", open: "00:00", close: "24:00"}   # 24/7

    rules:
      execution:                   # how orders are placed on this broker
        account_mode: hedging      # hedging | netting
        filling_mode: IOC          # IOC | FOK | RETURN (maps to mt5.ORDER_FILLING_*)
        max_deviation_points: 20
        default_lot_min: 0.01
        default_lot_step: 0.01
        weekend_holding: false     # prop firms often forbid holding over the weekend
        magic_number: 510
      risk:                        # prop-firm limits; OMIT or set null for retail
        max_daily_loss_pct: 5.0
        max_total_loss_pct: 10.0
        profit_target_pct: 10.0
        min_trading_days: 4
        max_leverage: 30
        news_trading_restricted: true
```

### Field notes

- **`timezone`** — a *label* (`EET`). The accessor maps it to the IANA zone
  `Europe/Athens`, which observes the exact EET↔EEST DST schedule MT5 servers
  use. See [[mt5_timezones]].
- **`asset_classes`** — broker-independent classification, keyed by **canonical**
  ticker. `asset_class()` also accepts the broker's **native** symbol (it maps
  back via `canonical_for`).
- **`sessions`** — `weekdays` is `Mon=0 … Sun=6` (matches Python `weekday()`),
  written as a range (`"0-4"`), list (`"0,2,4"`), or both. `open`/`close` are
  `HH:MM` in the **broker tz**; `"24:00"` means end-of-day. `daily_break` is the
  short financing-rollover gap; it may **wrap midnight** (`"23:58-00:02"`) and is
  handled correctly. Intraday bounds marked `# APPROX` are best-known starting
  values — calibrate later from the inferred `_symbol_sessions.json` (below).
- **`rules.execution`** — drives order placement. `account_mode: hedging` is the
  critical one: closes must be sent as an opposite-side deal with
  `position=<ticket>` (see the adapter audit in the README), not a netting close.
- **`rules.risk`** — prop-firm evaluation limits. Set to `null` (or omit) for a
  live-retail broker like Darwinex; `risk_rules()` then returns `None`.

---

## Accessor API (`data_platform/providers/mt5/brokers.py`)

All pure functions over the YAML — no terminal contact, results are immutable
(`@dataclass(frozen=True)`) and typed with enums (`AssetClass`, `AccountMode`,
`FillingMode`).

| Call | Returns | Notes |
|---|---|---|
| `resolve(broker, canonical)` | `str` | canonical → native symbol (raises on `UNKNOWN`) |
| `canonical_for(broker, native)` | `str` | native → canonical |
| `broker_timezone(broker)` | `str` | tz label, e.g. `"EET"` |
| `broker_tzinfo(broker)` | `ZoneInfo` | `EET`/`EEST` → `Europe/Athens` |
| `asset_class(broker, canonical_or_native)` | `AssetClass` | `OTHER` if untagged |
| `session(broker, asset_class)` | `Session` | raises if not declared |
| `sessions(broker)` | `dict[AssetClass, Session]` | all declared sessions |
| `is_market_open(broker, canonical, at=None)` | `bool` | schedule check at a UTC instant |
| `execution_rules(broker)` | `ExecutionRules` | order mechanics (with defaults) |
| `risk_rules(broker)` | `RiskRules \| None` | `None` for retail brokers |
| `rules(broker)` | `BrokerRules` | bundles execution + risk |
| `terminal_path(broker)` | `str \| None` | feeds `MT5Config.path` |
| `inferred_session_hours(symbol)` | `dict \| None` | empirical refinement (see below) |
| `validate_config()` | `None` | raises on any malformed block |

### `is_market_open` — a *schedule* check, not ground truth

```python
from data_platform.providers.mt5 import brokers
brokers.is_market_open("ftmo", "EURUSD")                 # now (UTC) — False on weekends
brokers.is_market_open("ftmo", "NQ", at=some_utc_dt)     # at a specific instant
```

It converts `at` (default now-UTC; naive treated as UTC) to the broker tz, then
checks `weekday ∈ session.weekdays` and `open ≤ minute-of-day < close` minus the
`daily_break`. It knows the **weekly schedule** but **not** ad-hoc holidays or
early closes — so for "can I actually trade right now?" a **live tick-freshness
probe is the ground truth** (see `tests/live_mt5/conftest.py::LiveKit.market_open`,
which is what gates the live order tests). Use `is_market_open` for planning,
risk windows, and scheduling; use a fresh tick for go/no-go.

### Empirical session refinement

[`build_symbol_sessions.py`](../../../data_platform/providers/mt5/build_symbol_sessions.py)
infers each symbol's real open/close (NY tz) from M1 bars into
`data/mt5_data/_symbol_sessions.json`. `inferred_session_hours(symbol)` exposes
that for reference/calibration. It is **not** auto-wired into `is_market_open`
(that JSON is NY-based and currently Darwinex-only); the declarative `sessions`
block stays authoritative until per-broker calibration is done.

---

## Brokers currently configured

| Broker | `confirmed` | Symbols | Risk rules | Notes |
|---|---|---|---|---|
| `darwinex` | ✅ | 16 | `null` (live retail) | live tick-scraper terminal — **do not disturb** |
| `ftmo` | ✅ | 16 | standard Challenge (5%/10%/10%) | demo; no weekend holding |
| `fundednext` | ❌ stub | `UNKNOWN` | placeholder | needs its own terminal + discovery |

> FTMO risk numbers are the **standard Challenge defaults** and vary by program /
> phase — confirm against the actual account before relying on them for a gate.

---

## Adding / confirming a broker

1. Add a block under `brokers:` with `display_name`, `server`, `terminal_path`,
   `timezone`, and `symbols:` placeholders (`UNKNOWN`) + `candidates:`.
2. Install **that broker's own** MT5 terminal, log into its demo, and confirm the
   native symbol names against it (`mt5.symbols_get()` — the
   `discover_symbols(broker)` stub documents this; never run it against the live
   Darwinex terminal). Paste confirmed names in, set `confirmed: true`.
3. Fill `asset_classes`, `sessions`, and `rules` (use `# APPROX` on anything not
   yet verified — never assert precision you don't have).
4. Run `python -m data_platform.providers.mt5.brokers` — it validates every block
   and prints a summary. Add assertions to
   [`tests/data_platform/test_broker_config.py`](../../../tests/data_platform/test_broker_config.py).

---

## Consumers & integration

- **Live adapter** — `terminal_path(broker)` feeds the `MT5Config.path` field, so
  `MT5Connection` binds deterministically to the right terminal on a
  multi-terminal machine (see CONNECTION_TEST.md → "Adapter path binding").
- **Live tests** — `tests/live_mt5/` exercises the adapter against the FTMO demo;
  its market gate is tick-freshness (ground truth) and can cross-check against
  `is_market_open`.
- **Resolver** — `resolve` / `canonical_for` / `broker_data_dir` are unchanged
  (34 callers across the catalog, ingest, and research layers); the schema above
  is purely additive.

> _Verified against current code via CodeGraph on 2026-06-07._
