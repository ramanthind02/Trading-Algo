# broker_cache/\<broker\>/live_state — file contract

The `data/broker_cache/<broker>/live_state/` directory is the IPC channel
between the running `TradingNode` (writer) and the monitoring dashboard API
(reader). The node is the sole owner of its MT5 terminal; the dashboard
must never open a second connection. Instead the node publishes a JSON
snapshot on a timer and consumes a command file.

All path helpers are in `deployment/live/monitoring/live_state.py`:

```
live_state_dir(broker)   → data/broker_cache/<broker>/live_state/
snapshot_path(dir)       → .../snapshot.json
command_path(dir)        → .../command.json
halt_path(dir)           → .../halt.json
equity_path(dir)         → .../equity_history.jsonl   (capped dashboard series)
equity_durable_path(dir) → .../equity.jsonl            (unbounded durable series)
forecasts_jsonl_path(b)  → .../forecasts.jsonl
submits_jsonl_path(b)    → .../slippage/submits.jsonl  (one level up from live_state/)
```

---

## IPC files — disposable, overwritten

These files exist only to communicate current state. They are **never
archived** on account rotation. A crash or stale read is self-healing:
the node regenerates them on the next publish cycle.

### snapshot.json

Written atomically every ~5 s by the node via `write_snapshot`
(`fsync=False` — atomic rename but no forced disk flush to avoid blocking
the event loop). Contains account equity/balance, open positions,
target-vs-actual fractions, warmup status, risk-baseline gauges, and
`schema_version`.

**NUL-snapshot crash artifact:** if the node crashes mid-write before the
rename completes, the OS may leave a NUL-filled file. `live_ingest.py`
handles this: if `snapshot.json` is unreadable and the broker has exactly
one active account in the registry, it falls back to that single account.
If there are multiple active accounts the ingest is skipped with a loud
message rather than guessing.

The dashboard API reads `snapshot.json` with `read_snapshot`
(`read_or_quarantine` — quarantines corrupt files, returns `None` instead
of raising).

### command.json

Written atomically by the dashboard API; read once by the node on each
event-loop tick. After the node executes the command it deletes the file
(`clear_command`) so it can never be re-executed after a restart.

Shape: `{"id": "<uuid>", "action": "flatten", "issued_at": "<ISO>", "confirm": null}`.

### halt.json

Durable kill-switch state. Persisted so a crash/auto-restart does NOT
silently resume trading — a halted node reloads this on start and stays
halted until an operator explicitly clears it (delete the file and
restart, or re-flatten to clean up).

Shape: `{"halted": true, "by_command_id": "<uuid>", "at": "<ISO>"}`.

---

## Durable capture files — append-only, retained forever

These files accumulate the historical record of live trading activity.
They are **archived** (moved to `_archive/account_<id>/`) on account
rotation and must never be pruned. The last line of any JSONL may be a
torn record if the node crashed mid-write; all readers skip torn lines
(ADR-3: tolerate torn last line, never raise on it).

### forecasts.jsonl (live_state/)

One row per decision×canonical appended at the ENTRY decision point by
`deployment/live/monitoring/forecast_log.py::record_forecasts`. Contains
`as_of`, `canonical`, `forecast_score`, `target_fraction`, `target_qty`,
`warmup_ready`, `vault_root`, `engine_config_hash`, `schema_version`.

Ingested by `registry ingest live` → `forecast_history` table.

### deals.jsonl (live_state/)

MT5 deal records for the account's magic number, captured by the node
after each fill. Contains the raw MT5 fields (`ticket`, `order`,
`position_id`, `symbol`, `type`, `entry`, `volume`, `price`,
`commission`, `swap`, `fee`, `profit`, `magic`, `time`, `comment`).

Ingested by `registry ingest live` → `deals` table.

### submit_results.jsonl (live_state/)

MT5 order-send result records, keyed by `client_order_id`. Contains
`retcode`, `result_price`, `result_deal`. Joined to `submits.jsonl` by
`client_order_id` during registry ingest.

### equity.jsonl (live_state/)

Durable equity log — every sample kept forever, never compacted. Shape:
`{"ts": "<ISO>", "equity": <float>, "balance": <float>, "floating_pnl": <float>, ...}`.
Optional fields (`balance`, `floating_pnl`, `gross_notional`,
`marks_fresh`) are omitted when not available; the base `{ts, equity}`
shape is always valid. Written by `append_equity_sample_durable`.

Compare to `equity_history.jsonl` (the capped dashboard series, compacted
at 1 MB / 5,000 samples): the durable log is authoritative; the
dashboard series is a bounded working copy for the equity chart.

Ingested by `registry ingest live` → `equity_snapshots` table (deduped
on `UNIQUE(account_id, ts)`).

### slippage/submits.jsonl (one level up from live_state/)

Live bid/ask quotes captured at order submit by
`deployment/live/monitoring/slippage.py::record_submit`. This is the
**only** source of spread data for these CFDs — the brokers serve no
historical ticks, so reconstruction after the fact is impossible.

Shape: `{"client_order_id": "<str>", "canonical": "<str>", "side": "BUY|SELL", "qty": <float>, "bid": <float>, "ask": <float>, "broker_time": "<ISO>", ["intent": "EXIT|ENTRY", "target_fraction": <float>]}`.

Ingested by `registry ingest live` → `orders` table (joined with
`submit_results.jsonl`).

---

## Ingest semantics

`registry ingest live` reads every durable file in full on each run
(ADR-3: idempotent full-file re-read — never tail by byte offset).
Existing rows in the DB are skipped via `INSERT OR IGNORE` (unique
constraints on ticket/client_order_id/ts). Torn last lines are counted
and warned but never raise.

Account resolution: the ingest reads `snapshot.json` to find `account.login`,
then looks up the `accounts` table. No accounts row → loud skip. Single
active account + unreadable snapshot → single-account fallback. Multiple
active accounts + unreadable snapshot → hard skip (ambiguous).

All `broker_time` values are stored as-is from the JSONL (broker
wall-clock ISO string) with no timezone conversion — see
[[mt5_timezones]] for the EET/EEST encoding.

---

## Related

- `deployment/live/monitoring/live_state.py` — path helpers, snapshot/command/halt IO
- `deployment/live/monitoring/slippage.py` — submit capture + fill reader
- `deployment/live/monitoring/forecast_log.py` — forecast capture
- `data_platform/registry/live_ingest.py` — ingest logic (ADR-3, account resolution)
- [[registry]] — registry.db operational reference (CLI, schema, rebuild)
- [[mt5_timezones]] — broker EET/EEST timestamp handling

> _Verified against the working tree on 2026-06-10._
