"""CLI for data_platform.registry.

Usage
-----
    python -m data_platform.registry [subcommand] [options]

Subcommands
-----------
Read-only (M6.1, ADR-9):
  coverage  Per-store key counts + coverage span (or per-key detail with --store).
  freshness Newest coverage_end per store (or per key with --store).
  runs      List runs: run_id, kind, status, spec, created_at, headline sharpe.
  lineage   spec→run→vault join for a feature_name or run_id.
  fills     Tabulate v_slippage rows joined on broker.
  report    Markdown store/key/count inventory (replaces data_store_inventory.md).
  costs     Read cost_observations; --refresh recomputes from M1 data.

Write:
  rebuild   Rebuild all derived/rebuildable registry tables from disk files.
  backup    Copy registry.db to data/backups/ using the SQLite backup API (ADR-10).
  ingest    Ingest live broker data (deals, orders, forecasts, equity) into the DB.
  accounts  Manage live-trading account registry rows.

rebuild options
---------------
  --domains d1,d2,...   Comma-separated subset of:
                        instruments,specs,runs,vault,calendar,manifest,ticks
                        (default: all)
  --include-stocks      Also rebuild stock_data manifest rows (329k files; slow).
  --db PATH             Override the registry DB path (default: data/registry.db).

ingest live options
-------------------
  --broker BROKER       Ingest only this broker (default: all).
  --db PATH             Override the registry DB path.

ingest backfill options
-----------------------
  --db PATH             Override the registry DB path.

accounts add options
--------------------
  --broker B            Broker name (required).
  --login L             Account login (required).
  --server S            Broker server (required).
  --exec-tier TIER      sandbox | demo | live (required).
  --phase PHASE         challenge | verification | funded | retail | personal | scraper.
  --currency CUR        Default USD.
  --initial-balance N   Starting equity for prop gauges.
  --magic N             MT5 magic number.
  --notes TEXT          Free-text notes.

accounts rotate options
-----------------------
  --broker B            Broker name (required).
  --new-login L         New account login (required).
  --new-server S        New account server (required).
  --phase PHASE         New program phase (default: inherits from old account).
"""
from __future__ import annotations

import sys
from pathlib import Path


_SUBCOMMANDS: tuple[str, ...] = (
    "rebuild", "backup",
    # read-only queries (M6.1, ADR-9)
    "coverage", "freshness", "runs", "lineage", "fills", "report", "costs",
    # write subcommands
    "ingest", "accounts",
)


def _rebuild(args: list[str]) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m data_platform.registry rebuild",
        description="Rebuild rebuildable registry domains from disk files.",
    )
    parser.add_argument(
        "--domains",
        default=None,
        help=(
            "Comma-separated domain list (default: all). "
            "Valid: instruments,specs,runs,vault,calendar,manifest,ticks"
        ),
    )
    parser.add_argument(
        "--include-stocks",
        action="store_true",
        default=False,
        help="Include stock_data manifest rows (slow).",
    )
    parser.add_argument(
        "--db",
        default=None,
        metavar="PATH",
        help="Override registry DB path (default: data/registry.db).",
    )
    ns = parser.parse_args(args)

    from data_platform.registry import db
    from data_platform.registry.rebuild import ALL_DOMAINS, orchestrate

    db_path = Path(ns.db) if ns.db else None

    domains: list[str] | None = None
    if ns.domains:
        raw_domains = [d.strip() for d in ns.domains.split(",") if d.strip()]
        unknown = [d for d in raw_domains if d not in ALL_DOMAINS]
        if unknown:
            print(
                f"Unknown domain(s): {', '.join(unknown)}. "
                f"Valid: {', '.join(ALL_DOMAINS)}",
                file=sys.stderr,
            )
            sys.exit(1)
        domains = raw_domains

    conn = db.connect(db_path)
    try:
        counts = orchestrate(conn, domains=domains, include_stocks=ns.include_stocks)
    finally:
        conn.close()

    print("Registry rebuild complete.")
    for domain, n in counts.items():
        print(f"  {domain:<14} {n:>8} rows")


def _backup(args: list[str]) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m data_platform.registry backup",
        description="Hot-backup registry.db via the SQLite3 backup API.",
    )
    parser.add_argument(
        "--db",
        default=None,
        metavar="PATH",
        help="Override registry DB path (default: data/registry.db).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        metavar="DIR",
        help="Directory to write backups to (default: data/backups/).",
    )
    ns = parser.parse_args(args)

    from data_platform.registry.backup import backup
    from pathlib import Path as _Path

    db_path = _Path(ns.db) if ns.db else None
    out_dir = _Path(ns.out_dir) if ns.out_dir else None

    dest = backup(db_path=db_path, out_dir=out_dir)
    print(f"Registry backup complete: {dest}")


def _ingest(args: list[str]) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m data_platform.registry ingest",
        description="Ingest live broker data (deals/orders/forecasts/equity) into the registry.",
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    live_parser = subparsers.add_parser("live", help="Ingest from broker live_state/ dirs.")
    live_parser.add_argument(
        "--broker", default=None,
        help="Broker name to ingest (default: all brokers with live_state/ dirs).",
    )
    live_parser.add_argument(
        "--db", default=None, metavar="PATH",
        help="Override registry DB path (default: data/registry.db).",
    )

    backfill_parser = subparsers.add_parser(
        "backfill",
        help="Backfill legacy slippage CSV + cfd_prop audit JSON records (migration plan §7.7).",
    )
    backfill_parser.add_argument(
        "--db", default=None, metavar="PATH",
        help="Override registry DB path (default: data/registry.db).",
    )

    ns = parser.parse_args(args)
    if ns.subcommand not in ("live", "backfill"):
        parser.print_help()
        sys.exit(1)

    from data_platform.registry import db

    if ns.subcommand == "live":
        from data_platform.registry.live_ingest import ingest_all

        db_path = Path(ns.db) if ns.db else None
        conn = db.connect(db_path)
        try:
            all_cov = ingest_all(conn, broker=ns.broker)
        finally:
            conn.close()

        if not all_cov:
            print("Nothing ingested.")
            return
        total = sum(v["rows"] for cov in all_cov.values() for v in cov.values())
        print(f"Ingest complete: {total} total rows across {len(all_cov)} broker(s).")

    elif ns.subcommand == "backfill":
        from data_platform.registry.backfill import run_backfill

        db_path = Path(ns.db) if ns.db else None
        conn = db.connect(db_path)
        try:
            coverage = run_backfill(conn)
        finally:
            conn.close()

        slip = coverage.get("slippage_csv_deals", {})
        audit = coverage.get("audit", {})
        slip_total = sum(slip.values())
        audit_deals = audit.get("deals", 0)
        audit_equity = audit.get("equity_snapshots", 0)
        print(
            f"Backfill complete: "
            f"slippage_csv_deals={slip_total} "
            f"(brokers: {', '.join(f'{b}={n}' for b, n in slip.items()) or 'none'}), "
            f"audit_deals={audit_deals}, "
            f"audit_equity_snapshots={audit_equity}"
        )


def _accounts(args: list[str]) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m data_platform.registry accounts",
        description="Manage live-trading account registry rows.",
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    # ── accounts add ──────────────────────────────────────────────────────────
    add_p = subparsers.add_parser("add", help="Register a new account.")
    add_p.add_argument("--broker", required=True)
    add_p.add_argument("--login", required=True)
    add_p.add_argument("--server", required=True)
    add_p.add_argument("--exec-tier", required=True, dest="exec_tier",
                       choices=["sandbox", "demo", "live"])
    add_p.add_argument("--phase", required=True,
                       choices=["challenge", "verification", "funded",
                                "retail", "personal", "scraper"])
    add_p.add_argument("--currency", default="USD")
    add_p.add_argument("--initial-balance", type=float, default=None,
                       dest="initial_balance")
    add_p.add_argument("--magic", type=int, default=None)
    add_p.add_argument("--notes", default=None)
    add_p.add_argument("--db", default=None, metavar="PATH")

    # ── accounts list ─────────────────────────────────────────────────────────
    list_p = subparsers.add_parser("list", help="Print all accounts.")
    list_p.add_argument("--db", default=None, metavar="PATH")

    # ── accounts rotate ───────────────────────────────────────────────────────
    rot_p = subparsers.add_parser("rotate",
                                  help="Retire current account, insert successor, archive JSONLs.")
    rot_p.add_argument("--broker", required=True)
    rot_p.add_argument("--new-login", required=True, dest="new_login")
    rot_p.add_argument("--new-server", required=True, dest="new_server")
    rot_p.add_argument("--phase", default=None,
                       choices=[None, "challenge", "verification", "funded",
                                "retail", "personal", "scraper"])
    rot_p.add_argument("--db", default=None, metavar="PATH")

    # ── accounts retire ───────────────────────────────────────────────────────
    ret_p = subparsers.add_parser("retire",
                                  help="Set active account for a broker to retired.")
    ret_p.add_argument("--broker", required=True)
    ret_p.add_argument("--db", default=None, metavar="PATH")

    ns = parser.parse_args(args)
    if not ns.subcommand:
        parser.print_help()
        sys.exit(1)

    from data_platform.registry import db
    from data_platform.registry.live_ingest import (
        accounts_add, accounts_retire, accounts_rotate,
    )

    db_path = Path(ns.db) if ns.db else None
    conn = db.connect(db_path)
    try:
        if ns.subcommand == "add":
            account_id = accounts_add(
                conn,
                broker=ns.broker,
                login=ns.login,
                server=ns.server,
                exec_tier=ns.exec_tier,
                phase=ns.phase,
                currency=ns.currency,
                initial_balance=ns.initial_balance,
                magic=ns.magic,
                notes=ns.notes,
            )
            print(
                f"Account added: id={account_id} broker={ns.broker!r} "
                f"login={ns.login!r} server={ns.server!r} "
                f"tier={ns.exec_tier} phase={ns.phase}"
            )

        elif ns.subcommand == "list":
            rows = conn.execute(
                "SELECT account_id, broker, login, exec_tier, program_phase, "
                "status, valid_from, valid_to FROM accounts ORDER BY broker, valid_from"
            ).fetchall()
            if not rows:
                print("No accounts registered.")
            else:
                hdr = f"{'id':>5}  {'broker':<12}  {'login':<12}  {'tier':<8}  "
                hdr += f"{'phase':<14}  {'status':<8}  {'valid_from':<26}  valid_to"
                print(hdr)
                print("-" * len(hdr))
                for r in rows:
                    print(
                        f"{r['account_id']:>5}  {r['broker']:<12}  {r['login']:<12}  "
                        f"{r['exec_tier']:<8}  {r['program_phase']:<14}  "
                        f"{r['status']:<8}  {str(r['valid_from']):<26}  "
                        f"{r['valid_to'] or ''}"
                    )

        elif ns.subcommand == "rotate":
            old_id, new_id = accounts_rotate(
                conn,
                ns.broker,
                new_login=ns.new_login,
                new_server=ns.new_server,
                phase=ns.phase,
            )
            print(
                f"Rotated: broker={ns.broker!r} "
                f"old_account_id={old_id} → new_account_id={new_id}"
            )

        elif ns.subcommand == "retire":
            account_id = accounts_retire(conn, ns.broker)
            print(f"Retired: broker={ns.broker!r} account_id={account_id}")

    finally:
        conn.close()


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print("data_platform.registry — available subcommands:")
        for cmd in _SUBCOMMANDS:
            print(f"  {cmd}")
        print("\nUsage: python -m data_platform.registry <subcommand> [options]")
        return

    cmd = args[0]
    rest = args[1:]

    if cmd == "rebuild":
        _rebuild(rest)
        return

    if cmd == "backup":
        _backup(rest)
        return

    if cmd == "ingest":
        _ingest(rest)
        return

    if cmd == "accounts":
        _accounts(rest)
        return

    # ── read-only subcommands (M6.1, ADR-9) ──────────────────────────────────
    _READ_CMDS = ("coverage", "freshness", "runs", "lineage", "fills", "report", "costs")
    if cmd in _READ_CMDS:
        from data_platform.registry.cli_read import (
            _costs,
            _coverage,
            _fills,
            _freshness,
            _lineage,
            _report,
            _runs,
        )

        _handlers = {
            "coverage": _coverage,
            "freshness": _freshness,
            "runs": _runs,
            "lineage": _lineage,
            "fills": _fills,
            "report": _report,
            "costs": _costs,
        }
        _handlers[cmd](rest)
        return

    if cmd not in _SUBCOMMANDS:
        print(
            f"Unknown subcommand {cmd!r}. Available: {', '.join(_SUBCOMMANDS)}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"{cmd}: not yet implemented")


if __name__ == "__main__":
    main()
