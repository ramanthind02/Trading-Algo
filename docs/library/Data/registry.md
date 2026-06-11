# data/registry.db — operational reference

`data/registry.db` is a SQLite **metadata index** built on top of the repo's
parquet stores. It records *what* data exists and *how* it was produced — it
never stores price rows itself. Because it is derived from files on disk it
can always be rebuilt from scratch.

---

## What it is (and what it is not)

| | registry.db |
|---|---|
| **Stores** | Instrument metadata, source-symbol maps, blob manifest, research run lineage, vault feature index, live-trading account registry, deals/orders/forecasts/equity from live nodes |
| **Does NOT store** | OHLCV price bars, tick data, parquet payloads of any kind |
| **Why SQLite** | Relational joins (spec → run → vault feature), sub-millisecond indexed reads, zero deployment overhead, hot-backup via the SQLite backup API |
| **Schema** | `data_platform/registry/schema.sql` |

---

## The rebuildable invariant

All tables except the live-trading tables (`accounts`, `deals`, `orders`,
`forecasts`, `equity_snapshots`, `job_runs`) are **fully rebuildable from disk
files** by running `rebuild`. The live tables are append-only durable captures
and cannot be reconstructed after the fact — they are the only "live" data in
the DB.

If `registry.db` is lost or corrupt:

1. Run `python -m data_platform.registry rebuild` to restore all derived tables.
2. Re-register live-trading accounts with `accounts add` (see below).
3. Re-ingest live JSONL files with `ingest live`.

---

## CLI reference

All subcommands: `python -m data_platform.registry <subcommand> [options]`

### Read-only queries

```powershell
# Per-store key counts + coverage span
python -m data_platform.registry coverage [--store <name>]

# Newest coverage_end per store (or per key with --store)
python -m data_platform.registry freshness [--store <name>]

# List research runs (run_id, kind, status, spec, created_at, headline sharpe)
python -m data_platform.registry runs

# Spec→run→vault join for a feature_name or run_id
python -m data_platform.registry lineage <feature_name_or_run_id>

# Tabulate v_slippage rows joined on broker
python -m data_platform.registry fills [--broker <name>]

# Markdown store/key/count inventory (live equivalent of data_store_inventory.md)
python -m data_platform.registry report

# Cost observations; --refresh recomputes from M1 data
python -m data_platform.registry costs [--refresh]
```

### Write operations

```powershell
# Rebuild all derived/rebuildable tables from disk files (instruments, specs,
# runs, vault, calendar, manifest, ticks)
python -m data_platform.registry rebuild [--domains d1,d2,...] [--include-stocks]

# Hot-backup via SQLite backup API → data/backups/
python -m data_platform.registry backup [--out-dir DIR]

# Ingest live broker data (deals, orders, forecasts, equity) from live_state/ dirs
python -m data_platform.registry ingest live [--broker <name>]

# Backfill legacy slippage CSV + cfd_prop audit JSON (migration plan §7.7)
python -m data_platform.registry ingest backfill
```

### Account management

```powershell
# Register a new live-trading account
python -m data_platform.registry accounts add \
    --broker ftmo --login 12345 --server FTMOServer2 \
    --exec-tier demo --phase challenge \
    [--currency USD] [--initial-balance 100000] [--magic 510] [--notes "..."]

# List all registered accounts
python -m data_platform.registry accounts list

# Retire current account, insert successor, archive JSONLs into _archive/account_<old_id>/
python -m data_platform.registry accounts rotate \
    --broker ftmo --new-login 99999 --new-server FTMOServer3 [--phase funded]

# Mark a broker's active account as retired (without inserting a successor)
python -m data_platform.registry accounts retire --broker ftmo
```

---

## Reader / writer rules

**Single writer** — only one process at a time may write to `registry.db`.
Reads are always safe from any process (SQLite WAL mode).

All writers go through `data_platform/registry/writer.py` (typed dataclass
helpers) and mutate only inside `db.transaction()`. Never write raw SQL
outside of `writer.py`.

---

## Test idiom: `_REGISTRY_DB_PATH`

Tests that need an isolated DB override the path via the
`_REGISTRY_DB_PATH` environment variable (read by `data_platform.registry.db.connect`
before defaulting to `data/registry.db`). Example:

```python
import os, tempfile, pathlib
from data_platform.registry import db

with tempfile.TemporaryDirectory() as tmp:
    os.environ["_REGISTRY_DB_PATH"] = str(pathlib.Path(tmp) / "test_registry.db")
    conn = db.connect()
    # ... test code ...
    conn.close()
```

---

## Related

- [[data_platform_migration_plan]] — full plan (storage + registry + live ingest)
- [[live_state_contract]] — file contract for the live_state/ IPC and durable captures
- [[data_store_inventory]] — on-disk store survey (2026-06-08 snapshot;
  live equivalent: `python -m data_platform.registry report`)
- `data_platform/registry/schema.sql` — authoritative schema DDL

> _Verified against the working tree on 2026-06-10._
