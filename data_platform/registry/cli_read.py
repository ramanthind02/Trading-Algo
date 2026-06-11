"""Read-only CLI subcommands for data_platform.registry (M6.1, ADR-9).

All subcommands open the DB via db.connect_readonly() except `costs --refresh`,
which uses db.connect() to write computed cost observations.

Public entry points (called from __main__.py):
    _coverage   -- per-store key counts + coverage span
    _freshness  -- newest coverage_end per store / per key
    _runs       -- run_id, kind, status, spec, created_at, headline sharpe
    _lineage    -- spec→run→vault join for a feature name or run_id
    _fills      -- v_slippage rows tabulated (joined on broker)
    _report     -- store/key/count inventory table (markdown to stdout)
    _costs      -- read / --refresh recompute cost_observations from M1 data
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────


def _open_ro(db_path: Path | None):
    from data_platform.registry import db as _db

    try:
        return _db.connect_readonly(db_path)
    except FileNotFoundError as exc:
        print(f"Registry DB not found: {exc}", file=sys.stderr)
        sys.exit(1)


def _db_path_from_ns(ns) -> Path | None:
    return Path(ns.db) if getattr(ns, "db", None) else None


# ── coverage ──────────────────────────────────────────────────────────────────


def _coverage(args: list[str]) -> None:
    """Per-store key counts + coverage span summary (or per-key listing with --store)."""
    parser = argparse.ArgumentParser(
        prog="python -m data_platform.registry coverage",
        description="Show blob_manifest coverage summary.",
    )
    parser.add_argument("--store", default=None, help="Show per-key detail for a single store.")
    parser.add_argument("--db", default=None, metavar="PATH")
    ns = parser.parse_args(args)

    conn = _open_ro(_db_path_from_ns(ns))
    try:
        if ns.store:
            from data_platform.registry import reader

            rows = reader.coverage(conn, ns.store)
            if not rows:
                print(f"No coverage rows for store {ns.store!r}.")
                return
            hdr = f"{'key_json':<60}  {'coverage_start':<14}  {'coverage_end':<14}  {'rows':>10}"
            print(hdr)
            print("-" * len(hdr))
            for r in rows:
                kj = str(r["key_json"])[:58]
                print(
                    f"{kj:<60}  "
                    f"{str(r['coverage_start'] or ''):<14}  "
                    f"{str(r['coverage_end'] or ''):<14}  "
                    f"{(r['rows'] or 0):>10}"
                )
        else:
            rows = conn.execute(
                """
                SELECT store,
                       count(*)            AS key_count,
                       MIN(coverage_start) AS earliest,
                       MAX(coverage_end)   AS latest,
                       SUM(rows)           AS total_rows
                FROM blob_manifest
                WHERE valid = 1
                GROUP BY store
                ORDER BY store
                """
            ).fetchall()
            if not rows:
                print("No coverage data in registry.")
                return
            hdr = (
                f"{'store':<25}  {'keys':>8}  "
                f"{'earliest':<14}  {'latest':<14}  {'total_rows':>12}"
            )
            print(hdr)
            print("-" * len(hdr))
            for r in rows:
                print(
                    f"{r['store']:<25}  {r['key_count']:>8}  "
                    f"{str(r['earliest'] or ''):<14}  "
                    f"{str(r['latest'] or ''):<14}  "
                    f"{(r['total_rows'] or 0):>12}"
                )
    finally:
        conn.close()


# ── freshness ─────────────────────────────────────────────────────────────────


def _freshness(args: list[str]) -> None:
    """Newest coverage_end per store (default) or per key with --store."""
    parser = argparse.ArgumentParser(
        prog="python -m data_platform.registry freshness",
        description="Show newest coverage_end per store / per key.",
    )
    parser.add_argument(
        "--store", default=None,
        help="Show per-key freshness for a single store.",
    )
    parser.add_argument("--db", default=None, metavar="PATH")
    ns = parser.parse_args(args)

    conn = _open_ro(_db_path_from_ns(ns))
    try:
        if ns.store:
            from data_platform.registry import reader

            rows = reader.freshness(conn, ns.store)
            if not rows:
                print(f"No rows for store {ns.store!r}.")
                return
            hdr = f"{'key_json':<60}  {'coverage_end':<14}"
            print(hdr)
            print("-" * len(hdr))
            for r in rows:
                kj = str(r["key_json"])[:58]
                print(f"{kj:<60}  {str(r['coverage_end'] or ''):<14}")
        else:
            rows = conn.execute(
                """
                SELECT store, MAX(coverage_end) AS coverage_end
                FROM blob_manifest
                WHERE valid = 1
                GROUP BY store
                ORDER BY store
                """
            ).fetchall()
            if not rows:
                print("No freshness data in registry.")
                return
            hdr = f"{'store':<30}  {'newest_coverage_end':<14}"
            print(hdr)
            print("-" * len(hdr))
            for r in rows:
                print(f"{r['store']:<30}  {str(r['coverage_end'] or ''):<14}")
    finally:
        conn.close()


# ── runs ──────────────────────────────────────────────────────────────────────


def _runs(args: list[str]) -> None:
    """List runs: run_id, kind, status, spec name, created_at, headline sharpe."""
    parser = argparse.ArgumentParser(
        prog="python -m data_platform.registry runs",
        description="List research runs from the registry.",
    )
    parser.add_argument("--kind", default=None, help="Filter by run kind.")
    parser.add_argument("--limit", type=int, default=20, help="Max rows to show (default 20).")
    parser.add_argument("--db", default=None, metavar="PATH")
    ns = parser.parse_args(args)

    conn = _open_ro(_db_path_from_ns(ns))
    try:
        q_params: list = []
        where = ""
        if ns.kind:
            where = "WHERE r.kind = ?"
            q_params.append(ns.kind)
        q_params.append(ns.limit)

        rows = conn.execute(
            f"""
            SELECT r.run_id, r.kind, r.status,
                   s.name      AS spec_name,
                   r.created_at,
                   r.headline_metrics_json
            FROM runs r
            LEFT JOIN specs s ON s.id = r.spec_id
            {where}
            ORDER BY r.created_at DESC
            LIMIT ?
            """,
            q_params,
        ).fetchall()

        if not rows:
            print("No runs found.")
            return

        hdr = (
            f"{'run_id':<38}  {'kind':<18}  {'status':<12}  "
            f"{'spec':<28}  {'created':<26}  {'sharpe':>8}"
        )
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            sharpe_str = ""
            if r["headline_metrics_json"]:
                try:
                    m = json.loads(r["headline_metrics_json"])
                    val = m.get("sharpe") or m.get("sharpe_is")
                    if val is not None:
                        sharpe_str = f"{float(val):.3f}"
                except Exception:
                    pass
            print(
                f"{str(r['run_id']):<38}  "
                f"{str(r['kind']):<18}  "
                f"{str(r['status']):<12}  "
                f"{str(r['spec_name'] or '')[:26]:<28}  "
                f"{str(r['created_at']):<26}  "
                f"{sharpe_str:>8}"
            )
    finally:
        conn.close()


# ── lineage ───────────────────────────────────────────────────────────────────


def _lineage(args: list[str]) -> None:
    """Print spec→run→vault join for a feature_name or run_id."""
    parser = argparse.ArgumentParser(
        prog="python -m data_platform.registry lineage",
        description="Show vault or run lineage.",
    )
    parser.add_argument("name", help="Feature name or run_id.")
    parser.add_argument("--db", default=None, metavar="PATH")
    ns = parser.parse_args(args)

    conn = _open_ro(_db_path_from_ns(ns))
    try:
        from data_platform.registry import reader

        # Try vault lineage first (feature_name lookup)
        row = reader.vault_lineage(conn, ns.name)
        if row:
            print(f"Vault lineage for {ns.name!r}:")
            for field in (
                "vault_profile", "timeframe", "ensemble_leaf",
                "feature_name", "producing_run_id",
                "run_kind", "run_status", "spec_name", "updated_at",
            ):
                print(f"  {field:<22}: {row[field]}")
            if row["headline_metrics_json"]:
                try:
                    m = json.loads(row["headline_metrics_json"])
                    print(f"  {'headline':<22}: {json.dumps(m)}")
                except Exception:
                    pass
            return

        # Try as run_id
        run_row = reader.run(conn, ns.name)
        if run_row:
            print(f"Run lineage for {ns.name!r}:")
            for field in ("kind", "status", "spec_id", "created_at", "started_at", "finished_at"):
                print(f"  {field:<22}: {run_row[field]}")
            if run_row["headline_metrics_json"]:
                try:
                    m = json.loads(run_row["headline_metrics_json"])
                    print(f"  {'headline':<22}: {json.dumps(m)}")
                except Exception:
                    pass
            if run_row["spec_hash"]:
                related = reader.runs_for_spec_hash(conn, run_row["spec_hash"])
                print(f"  {'related_runs':<22}: {len(related)} for same spec_hash")
            return

        print(f"No vault entry or run found for {ns.name!r}.", file=sys.stderr)
        sys.exit(1)
    finally:
        conn.close()


# ── fills ─────────────────────────────────────────────────────────────────────


def _fills(args: list[str]) -> None:
    """Tabulate v_slippage rows, joined to broker via accounts."""
    parser = argparse.ArgumentParser(
        prog="python -m data_platform.registry fills",
        description="Show fill / slippage records from the registry.",
    )
    parser.add_argument("--broker", default=None, help="Filter by broker name.")
    parser.add_argument("--limit", type=int, default=50, help="Max rows (default 50).")
    parser.add_argument("--db", default=None, metavar="PATH")
    ns = parser.parse_args(args)

    conn = _open_ro(_db_path_from_ns(ns))
    try:
        q_params: list = []
        where = ""
        if ns.broker:
            where = "WHERE a.broker = ?"
            q_params.append(ns.broker)
        q_params.append(ns.limit)

        rows = conn.execute(
            f"""
            SELECT v.ticket, v.canonical, v.entry, v.volume,
                   v.fill_px, v.spread_bps, v.slip_vs_touch_bps,
                   v.broker_time, a.broker
            FROM v_slippage v
            JOIN accounts a ON a.account_id = v.account_id
            {where}
            ORDER BY v.broker_time DESC
            LIMIT ?
            """,
            q_params,
        ).fetchall()

        if not rows:
            print("No fills found.")
            return

        hdr = (
            f"{'broker':<12}  {'ticket':>8}  {'canonical':<12}  "
            f"{'entry':<6}  {'vol':>6}  {'fill_px':>10}  "
            f"{'spread_bps':>10}  {'slip_touch':>10}  {'broker_time':<26}"
        )
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            print(
                f"{str(r['broker']):<12}  "
                f"{r['ticket']:>8}  "
                f"{str(r['canonical'] or ''):<12}  "
                f"{str(r['entry'] or ''):<6}  "
                f"{r['volume']:>6.2f}  "
                f"{r['fill_px']:>10.4f}  "
                f"{(r['spread_bps'] or 0):>10.2f}  "
                f"{(r['slip_vs_touch_bps'] or 0):>10.2f}  "
                f"{str(r['broker_time']):<26}"
            )
    finally:
        conn.close()


# ── report ────────────────────────────────────────────────────────────────────


def _report(args: list[str]) -> None:
    """Markdown store/key/count inventory table (replaces hand-written data_store_inventory.md)."""
    parser = argparse.ArgumentParser(
        prog="python -m data_platform.registry report",
        description="Print the blob_manifest inventory as a Markdown table.",
    )
    parser.add_argument("--db", default=None, metavar="PATH")
    ns = parser.parse_args(args)

    conn = _open_ro(_db_path_from_ns(ns))
    try:
        # Per-store summary rows
        summaries = conn.execute(
            """
            SELECT store,
                   count(*)            AS key_count,
                   MIN(coverage_start) AS earliest,
                   MAX(coverage_end)   AS latest,
                   SUM(rows)           AS total_rows,
                   SUM(CASE WHEN valid=0 THEN 1 ELSE 0 END) AS invalid_count
            FROM blob_manifest
            GROUP BY store
            ORDER BY store
            """
        ).fetchall()

        if not summaries:
            print("| store | keys | earliest | latest | rows | invalid |")
            print("|-------|------|----------|--------|------|---------|")
            print("*(empty — run `registry rebuild` to populate)*")
            return

        print("| store | keys | earliest | latest | rows | invalid |")
        print("|-------|------|----------|--------|------|---------|")
        for r in summaries:
            total = r["total_rows"] or 0
            print(
                f"| {r['store']} "
                f"| {r['key_count']} "
                f"| {r['earliest'] or ''} "
                f"| {r['latest'] or ''} "
                f"| {total:,} "
                f"| {r['invalid_count']} |"
            )

        # Footer totals
        totals = conn.execute(
            """
            SELECT count(DISTINCT store) AS n_stores,
                   count(*)              AS n_keys,
                   SUM(rows)             AS total_rows
            FROM blob_manifest
            WHERE valid = 1
            """
        ).fetchone()
        print()
        print(
            f"> {totals['n_stores']} stores  "
            f"· {totals['n_keys']} keys  "
            f"· {(totals['total_rows'] or 0):,} rows (valid=1)"
        )
    finally:
        conn.close()


# ── costs ─────────────────────────────────────────────────────────────────────


def _costs(args: list[str]) -> None:
    """Read cost_observations; with --refresh recompute from M1 data and upsert."""
    parser = argparse.ArgumentParser(
        prog="python -m data_platform.registry costs",
        description=(
            "Show or recompute cost observations.\n\n"
            "Without --refresh: reads cost_observations table (read-only).\n"
            "With --refresh: samples M1 spread data for all registered MT5 instruments, "
            "computes p25/p50/p75 spread + spread_bps per 1-hour broker-time bucket "
            "(trailing 365d window from the most-recent year partition), and upserts into "
            "cost_observations. Records a job_runs row on completion."
        ),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        default=False,
        help="Recompute from M1 data and upsert (requires read-write connection).",
    )
    parser.add_argument(
        "--instrument",
        default=None,
        metavar="I",
        help="Restrict to one instrument_id or MT5 native_symbol.",
    )
    parser.add_argument("--db", default=None, metavar="PATH")
    ns = parser.parse_args(args)

    db_path = _db_path_from_ns(ns)

    if not ns.refresh:
        # ── read-only path ─────────────────────────────────────────────────────
        conn = _open_ro(db_path)
        try:
            q_params: list = []
            where = ""
            if ns.instrument:
                where = "WHERE instrument_id = ?"
                q_params.append(ns.instrument)

            rows = conn.execute(
                f"""
                SELECT instrument_id, broker, tod_bucket,
                       window_start, window_end,
                       spread_points_p25, spread_points_p50, spread_points_p75,
                       spread_bps_p50,
                       slip_vs_touch_bps_p50,
                       n_spread_obs, n_fill_obs, computed_at
                FROM cost_observations
                {where}
                ORDER BY instrument_id, broker, tod_bucket
                """,
                q_params,
            ).fetchall()

            if not rows:
                print("No cost observations. Run with --refresh to compute.")
                return

            hdr = (
                f"{'instrument_id':<20}  {'broker':<10}  {'tod_bucket':<10}  "
                f"{'p25':>8}  {'p50':>8}  {'p75':>8}  "
                f"{'bps_p50':>9}  {'slip_p50':>9}  {'n_obs':>6}  {'computed_at':<26}"
            )
            print(hdr)
            print("-" * len(hdr))
            for r in rows:
                print(
                    f"{str(r['instrument_id']):<20}  "
                    f"{str(r['broker']):<10}  "
                    f"{str(r['tod_bucket']):<10}  "
                    f"{(r['spread_points_p25'] or 0):>8.4f}  "
                    f"{(r['spread_points_p50'] or 0):>8.4f}  "
                    f"{(r['spread_points_p75'] or 0):>8.4f}  "
                    f"{(r['spread_bps_p50'] or 0):>9.4f}  "
                    f"{(r['slip_vs_touch_bps_p50'] or 0):>9.4f}  "
                    f"{(r['n_spread_obs'] or 0):>6}  "
                    f"{str(r['computed_at']):<26}"
                )
        finally:
            conn.close()
        return

    # ── --refresh path ─────────────────────────────────────────────────────────
    _costs_refresh(db_path, ns.instrument)


def _costs_refresh(db_path: Path | None, instrument_filter: str | None) -> None:
    """Recompute cost_observations from MT5 M1 data; upsert and record job_run."""
    from datetime import datetime, timezone

    from data_platform.registry import db as _db
    from data_platform.registry import writer

    try:
        import numpy as np
        import pandas as pd
        import pyarrow.parquet as pq
    except ImportError as exc:
        print(f"costs --refresh requires numpy, pandas, pyarrow: {exc}", file=sys.stderr)
        sys.exit(1)

    repo_root = _db._repo_root()
    mt5_root = repo_root / "data" / "mt5_data"

    conn = _db.connect(db_path)
    try:
        started_at = datetime.now(timezone.utc).isoformat()
        with _db.transaction(conn):
            job_run_id = writer.record_job_run(
                conn,
                writer.JobRun(job_name="costs_refresh", started_at=started_at),
            )

        if not mt5_root.exists():
            err = f"mt5_data directory not found: {mt5_root}"
            with _db.transaction(conn):
                writer.finish_job_run(conn, job_run_id, 1, 0, None, err)
            print(err, file=sys.stderr)
            sys.exit(1)

        # All registered MT5 instruments: id, raw_symbol, native_symbol (MT5 broker symbol)
        inst_rows = conn.execute(
            """
            SELECT i.id, i.raw_symbol, iss.native_symbol
            FROM instruments i
            JOIN instrument_source_symbols iss ON iss.instrument_id = i.id
            WHERE iss.source = 'mt5'
            ORDER BY i.id
            """
        ).fetchall()

        if instrument_filter:
            inst_rows = [
                r for r in inst_rows
                if r["id"] == instrument_filter or r["native_symbol"] == instrument_filter
            ]

        total_rows = 0
        errors: list[str] = []

        for inst in inst_rows:
            instrument_id: str = inst["id"]
            native_sym: str = inst["native_symbol"]
            raw_sym: str = inst["raw_symbol"]

            bars_m1_dir = mt5_root / native_sym / "bars_M1"
            if not bars_m1_dir.exists():
                continue

            year_dirs = sorted(
                d for d in bars_m1_dir.iterdir()
                if d.is_dir() and d.name.startswith("year=")
            )
            if not year_dirs:
                continue

            parquet_file = year_dirs[-1] / "part.parquet"
            if not parquet_file.exists():
                continue

            try:
                table = pq.read_table(
                    str(parquet_file), columns=["time", "close", "spread"]
                )
                df = table.to_pandas()

                if df.empty:
                    continue

                # time column is tz-aware (UTC label but actually broker EET).
                # dt.hour extracts the stored-as-UTC hour, which equals broker EET hour.
                if hasattr(df["time"].dt, "tz") and df["time"].dt.tz is not None:
                    # Strip tz info so pandas operations are faster; values unchanged.
                    df["time"] = df["time"].dt.tz_convert(None)

                window_start = str(df["time"].min().date())
                window_end   = str(df["time"].max().date())

                df["tod_bucket"] = df["time"].dt.hour.map(lambda h: f"{h:02d}:00")
                computed_at = datetime.now(timezone.utc).isoformat()

                # Slip data for this instrument (JOIN on accounts → no per-bucket filtering
                # needed here; we split by hour after fetching).
                slip_df: pd.DataFrame | None = None
                try:
                    slip_raw = conn.execute(
                        """
                        SELECT v.slip_vs_touch_bps,
                               strftime('%H', v.broker_time) AS h
                        FROM v_slippage v
                        JOIN accounts a ON a.account_id = v.account_id
                        WHERE v.canonical = ?
                          AND v.slip_vs_touch_bps IS NOT NULL
                        """,
                        (raw_sym,),
                    ).fetchall()
                    if slip_raw:
                        slip_df = pd.DataFrame(
                            [{"h": r["h"], "slip": r["slip_vs_touch_bps"]} for r in slip_raw]
                        )
                except Exception:
                    pass

                instrument_upserts: list[tuple] = []
                for bucket, grp in df.groupby("tod_bucket", sort=True):
                    spreads = grp["spread"].dropna().to_numpy(dtype=float)
                    closes  = grp["close"].dropna().to_numpy(dtype=float)

                    if len(spreads) == 0:
                        continue

                    p25 = float(np.percentile(spreads, 25))
                    p50 = float(np.percentile(spreads, 50))
                    p75 = float(np.percentile(spreads, 75))

                    median_close = float(np.median(closes)) if len(closes) > 0 else None
                    spread_bps_p50 = (
                        p50 / median_close * 10_000
                        if median_close and median_close > 0
                        else None
                    )

                    slip_p50 = slip_p90 = None
                    n_fill_obs = 0
                    if slip_df is not None and not slip_df.empty:
                        # bucket is "HH:00"; h in slip_df is "HH" (zero-padded)
                        bucket_h = str(bucket).split(":")[0]
                        bucket_slips = (
                            slip_df[slip_df["h"] == bucket_h]["slip"]
                            .dropna()
                            .to_numpy(dtype=float)
                        )
                        if len(bucket_slips) > 0:
                            slip_p50 = float(np.percentile(bucket_slips, 50))
                            slip_p90 = float(np.percentile(bucket_slips, 90))
                            n_fill_obs = int(len(bucket_slips))

                    instrument_upserts.append((
                        instrument_id, "darwinex", str(bucket),
                        window_start, window_end,
                        p25, p50, p75, spread_bps_p50,
                        slip_p50, slip_p90,
                        int(len(spreads)), n_fill_obs, computed_at,
                    ))

                if instrument_upserts:
                    with _db.transaction(conn):
                        conn.executemany(
                            """
                            INSERT INTO cost_observations
                              (instrument_id, broker, tod_bucket,
                               window_start, window_end,
                               spread_points_p25, spread_points_p50, spread_points_p75,
                               spread_bps_p50,
                               slip_vs_touch_bps_p50, slip_vs_touch_bps_p90,
                               n_spread_obs, n_fill_obs, computed_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(instrument_id, broker, tod_bucket,
                                        window_start, window_end)
                            DO UPDATE SET
                              spread_points_p25     = excluded.spread_points_p25,
                              spread_points_p50     = excluded.spread_points_p50,
                              spread_points_p75     = excluded.spread_points_p75,
                              spread_bps_p50        = excluded.spread_bps_p50,
                              slip_vs_touch_bps_p50 = excluded.slip_vs_touch_bps_p50,
                              slip_vs_touch_bps_p90 = excluded.slip_vs_touch_bps_p90,
                              n_spread_obs          = excluded.n_spread_obs,
                              n_fill_obs            = excluded.n_fill_obs,
                              computed_at           = excluded.computed_at
                            """,
                            instrument_upserts,
                        )
                    total_rows += len(instrument_upserts)
                    print(f"  {native_sym}: {len(instrument_upserts)} bucket rows")

            except Exception as exc:  # noqa: BLE001
                errors.append(f"{native_sym}: {exc}")

        error_text = "; ".join(errors) if errors else None
        with _db.transaction(conn):
            writer.finish_job_run(
                conn, job_run_id,
                exit_code=0 if not errors else 1,
                rows_written=total_rows,
                coverage_json=None,
                error_text=error_text,
            )

        print(f"Cost refresh complete: {total_rows} rows written, {len(errors)} errors.")
        if errors:
            for e in errors[:5]:
                print(f"  ERROR: {e}", file=sys.stderr)

    finally:
        conn.close()
