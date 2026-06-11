"""Registry rebuild — repopulates all rebuildable domains from disk files.

Usage (via __main__.py CLI):
    python -m data_platform.registry rebuild
    python -m data_platform.registry rebuild --domains instruments,specs,vault
    python -m data_platform.registry rebuild --include-stocks
    python -m data_platform.registry rebuild --db /tmp/test.db

Domain builders (each runs within its own atomic transaction):
  instruments  catalog.parquet → instruments + instrument_source_symbols
  specs        research/specs/*.json → specs
  runs         feature_research/shared_results/_runs_index.json → runs
  vault        vault/, vault_personal/, vault_cfd_prop/ → vault_entries + sleeves
  calendar     data/events/calendar/*.json → fomc_decision + nyse_holiday
  manifest     data/ohlc_data/, data/mt5_data/ → blob_manifest
  ticks        data/mt5_data/<SYM>/ticks_cache/ → tick_chunks

Rebuild invariant: each selected domain is DELETE-then-repopulate in one
transaction; unselected domains are untouched.  A job_runs row records the
rebuild itself (name='registry_rebuild', args, per-domain counts, duration).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq

from data_platform.registry import writer
from data_platform.registry.db import transaction

log = logging.getLogger(__name__)

ALL_DOMAINS: tuple[str, ...] = (
    "instruments",
    "specs",
    "runs",
    "vault",
    "calendar",
    "manifest",
    "ticks",
)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    return next(
        (p for p in here.parents if (p / ".git").exists() or (p / "AGENTS.md").exists()),
        here.parents[2],
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _name_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


# ── Coverage extraction ──────────────────────────────────────────────────────


def _extract_coverage(
    path: Path, time_col: str = "date"
) -> tuple[str | None, str | None, int | None]:
    """Extract (coverage_start, coverage_end, row_count) from parquet footer stats.

    Returns (None, None, None) on any error so callers can still insert a row.
    """
    try:
        pf = pq.ParquetFile(path)
        meta = pf.metadata
        row_count: int = meta.num_rows

        min_val = None
        max_val = None
        for rg_idx in range(meta.num_row_groups):
            rg = meta.row_group(rg_idx)
            for col_idx in range(rg.num_columns):
                col = rg.column(col_idx)
                if col.path_in_schema != time_col:
                    continue
                s = col.statistics
                if s and s.has_min_max:
                    if min_val is None or s.min < min_val:
                        min_val = s.min
                    if max_val is None or s.max > max_val:
                        max_val = s.max

        if min_val is None:
            return None, None, row_count

        # min_val may be datetime.datetime, pandas Timestamp, or plain date
        cov_start = min_val.date().isoformat() if hasattr(min_val, "date") else str(min_val)[:10]
        cov_end = max_val.date().isoformat() if hasattr(max_val, "date") else str(max_val)[:10]
        return cov_start, cov_end, row_count
    except Exception:
        return None, None, None


# ── Builder: instruments ─────────────────────────────────────────────────────


def build_instruments(
    conn: sqlite3.Connection,
    repo_root: Path,
    *,
    catalog_path: Path | None = None,
) -> int:
    """Rebuild instruments + instrument_source_symbols from catalog.parquet."""
    from data_platform.core.catalog import InstrumentCatalog, catalog_parquet_path

    path = catalog_path or catalog_parquet_path()
    if not path.exists():
        log.info("instruments: catalog not found at %s — skipped", path)
        return 0

    # Load via the standard loader (reads parquet at path)
    # Monkeypatch catalog_parquet_path in tests if you need to redirect.
    catalog = InstrumentCatalog.load()
    instruments = catalog.all()
    count = 0

    with transaction(conn):
        conn.execute("DELETE FROM instrument_source_symbols")
        conn.execute("DELETE FROM instruments")

        for inst in instruments:
            rec = writer.Instrument(
                id=str(inst.id),
                raw_symbol=inst.raw_symbol,
                asset_class=inst.asset_class.value,
                instrument_class=inst.instrument_class.value,
                price_precision=inst.price_precision,
                price_increment=inst.price_increment,
                multiplier=getattr(inst, "multiplier", 1.0),
                quote_currency=getattr(inst, "quote_currency", "USD"),
                activation=inst.activation.isoformat() if inst.activation else None,
                expiration=inst.expiration.isoformat() if inst.expiration else None,
                data_source=getattr(inst, "data_source", None),
                info_json=json.dumps(inst.info, default=str),
            )
            writer.upsert_instrument(conn, rec)

            # Normalise source_symbols: skip any key starting with 'ib' (IB retired).
            # The catalog has a handful of cases where two instruments share the same
            # (source, native_symbol) pair (e.g. both ES.XCME and SP500.XNAS map to
            # mt5='SP500').  The DB enforces uniqueness per-pair; on collision we log
            # a warning and keep the first mapping (first-wins, alphabetical by id).
            ss = inst.info.get("source_symbols", {})
            if isinstance(ss, dict):
                filtered = {
                    k: v
                    for k, v in ss.items()
                    if not k.startswith("ib") and v is not None
                }
                if filtered:
                    try:
                        writer.replace_source_symbols(conn, str(inst.id), filtered)
                    except sqlite3.IntegrityError:
                        # Fall back: insert each mapping individually, skipping conflicts
                        conn.execute(
                            "DELETE FROM instrument_source_symbols WHERE instrument_id = ?",
                            (str(inst.id),),
                        )
                        for source, sym in filtered.items():
                            try:
                                conn.execute(
                                    "INSERT INTO instrument_source_symbols"
                                    " (instrument_id, source, native_symbol) VALUES (?, ?, ?)",
                                    (str(inst.id), source, sym),
                                )
                            except sqlite3.IntegrityError:
                                log.warning(
                                    "instruments: skipping duplicate source_symbol"
                                    " (%s, %r) for %s — already held by another instrument",
                                    source,
                                    sym,
                                    str(inst.id),
                                )

            count += 1

    return count


# ── Builder: specs ───────────────────────────────────────────────────────────


def build_specs(conn: sqlite3.Connection, repo_root: Path) -> int:
    """Rebuild specs from research/specs/*.json."""
    specs_dir = repo_root / "research" / "specs"
    if not specs_dir.exists():
        return 0

    now = _now_iso()
    count = 0
    generated_slugs: set[str] = set()

    with transaction(conn):
        # NULL out runs.spec_id before deleting specs to satisfy the FK constraint.
        # build_runs (if selected) will restore the links during its own pass.
        conn.execute("UPDATE runs SET spec_id = NULL WHERE spec_id IS NOT NULL")
        conn.execute("DELETE FROM specs")

        for json_file in sorted(specs_dir.glob("*.json")):
            spec_id = json_file.stem
            raw = ""
            try:
                raw = json_file.read_text(encoding="utf-8")
                parsed = json.loads(raw)
                canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
                content_hash = hashlib.sha256(canonical.encode()).hexdigest()
                name = parsed.get("name") or spec_id
                spec_version = str(parsed.get("spec_version", "1.0"))
                valid = 1
                error = None
                if "name" not in parsed:
                    valid = 0
                    error = "missing required key: name"
            except (json.JSONDecodeError, OSError) as exc:
                if not raw:
                    try:
                        raw = json_file.read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        raw = ""
                content_hash = hashlib.sha256(raw.encode()).hexdigest()
                name = spec_id
                spec_version = "1.0"
                valid = 0
                error = str(exc)

            # Deduplicate slug within this batch
            base_slug = _name_slug(name)
            slug = base_slug
            suffix = 0
            while slug in generated_slugs:
                suffix += 1
                slug = f"{base_slug}-{suffix}"
            generated_slugs.add(slug)

            # Use file mtime for created_at/updated_at (best-effort)
            try:
                mtime = datetime.fromtimestamp(
                    json_file.stat().st_mtime, tz=timezone.utc
                ).isoformat()
            except OSError:
                mtime = now

            file_path = str(json_file.relative_to(repo_root))

            writer.upsert_spec(
                conn,
                writer.SpecRecord(
                    id=spec_id,
                    name=name,
                    name_slug=slug,
                    content_hash=content_hash,
                    spec_version=spec_version,
                    valid=valid,
                    error=error,
                    spec_json=raw,
                    file_path=file_path,
                    created_at=mtime,
                    updated_at=mtime,
                ),
            )
            count += 1

    return count


# ── Builder: runs ────────────────────────────────────────────────────────────


_PHASE_TO_KIND: dict[str, str] = {
    "exploration": "exploration",
    "validation": "validation",
    "portfolio": "portfolio",
    "final_validation": "final_validation",
    "experiment": "experiment",
}

_STATUS_REMAP: dict[str, str] = {
    "queued": "failed",
    "running": "failed",
    "completed": "completed",
    "failed": "failed",
}


def build_runs(
    conn: sqlite3.Connection,
    repo_root: Path,
    log_dir: Path,
) -> int:
    """Rebuild runs from feature_research/shared_results/_runs_index.json."""
    idx_path = repo_root / "feature_research" / "shared_results" / "_runs_index.json"
    if not idx_path.exists():
        return 0

    try:
        raw_list = json.loads(idx_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0

    if isinstance(raw_list, dict):
        raw_list = list(raw_list.values())
    if not isinstance(raw_list, list):
        return 0

    # Known spec IDs already in DB (from build_specs run before us, or pre-existing)
    known_specs: set[str] = {
        r[0] for r in conn.execute("SELECT id FROM specs").fetchall()
    }

    count = 0
    with transaction(conn):
        conn.execute("DELETE FROM results")
        conn.execute("DELETE FROM runs")

        for entry in raw_list:
            if not isinstance(entry, dict):
                continue
            run_id = entry.get("run_id")
            if not run_id:
                continue

            # phase → kind (tolerant: unknown maps to 'exploration')
            phase = entry.get("phase") or entry.get("kind") or "exploration"
            kind = _PHASE_TO_KIND.get(phase, "exploration")

            # status mapping
            raw_status = entry.get("status", "completed")
            status = _STATUS_REMAP.get(raw_status, "completed")
            error_text = entry.get("error_text")
            if raw_status in ("queued", "running"):
                error_text = "Interrupted (registry backfill)"

            # spec_id: keep only if the spec exists in the DB
            spec_id: str | None = entry.get("spec_id")
            if spec_id not in known_specs:
                spec_id = None

            # log_text → write to file, store path in DB
            log_path: str | None = None
            log_text: str | None = entry.get("log_text")
            if log_text:
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = log_dir / f"{run_id}.log"
                log_file.write_text(log_text, encoding="utf-8")
                log_path = str(log_file)

            writer.upsert_run(
                conn,
                writer.RunRecord(
                    run_id=run_id,
                    kind=kind,
                    status=status,
                    spec_id=spec_id,
                    spec_hash=entry.get("spec_hash"),
                    spec_snapshot_json=entry.get("spec_snapshot_json"),
                    research_feed_used=entry.get("research_feed_used"),
                    ewsd_blend=entry.get("ewsd_blend"),
                    realistic_phases=entry.get("realistic_phases"),
                    git_sha=entry.get("git_sha"),
                    num_combos=entry.get("num_combos"),
                    reports_dir=entry.get("reports_dir"),
                    viz_dir=entry.get("viz_dir") or entry.get("viz_parent_dir"),
                    log_path=log_path,
                    headline_metrics_json=None,  # NULL for backfilled rows
                    created_at=entry.get("created_at", _now_iso()),
                    started_at=entry.get("started_at"),
                    finished_at=entry.get("finished_at"),
                    error_text=error_text,
                ),
            )
            count += 1

    return count


# ── Builder: vault ───────────────────────────────────────────────────────────


def _iter_vault_feature_jsons(
    vault_root: Path,
) -> Iterator[tuple[Path, str, str | None, str]]:
    """Yield (json_path, timeframe, weight_hierarchy_group, ensemble_leaf).

    Supports two layouts:
      nested: <vault_root>/<TF>/<group>/<ensemble>/features/*.json
      flat:   <vault_root>/<TF>/<ensemble>/features/*.json
    """
    from ensemble.vault.constants import VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES

    for tf_dir in sorted(vault_root.iterdir()):
        if not tf_dir.is_dir() or tf_dir.name not in ("D", "W", "M"):
            continue
        tf = tf_dir.name

        for level2 in sorted(tf_dir.iterdir()):
            if not level2.is_dir():
                continue

            if level2.name in VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES:
                # Nested layout: level2 is a weight_hierarchy_group directory
                group = level2.name
                for ensemble_dir in sorted(level2.iterdir()):
                    if not ensemble_dir.is_dir():
                        continue
                    ensemble_leaf = ensemble_dir.name
                    features_dir = ensemble_dir / "features"
                    if features_dir.is_dir():
                        yield from (
                            (f, tf, group, ensemble_leaf)
                            for f in sorted(features_dir.glob("*.json"))
                        )
            else:
                # Flat / legacy layout: level2 is the ensemble_leaf itself
                ensemble_leaf = level2.name
                features_dir = level2 / "features"
                if features_dir.is_dir():
                    yield from (
                        (f, tf, None, ensemble_leaf)
                        for f in sorted(features_dir.glob("*.json"))
                    )


def build_vault(
    conn: sqlite3.Connection,
    repo_root: Path,
    *,
    vault_root_map: dict[Path, str] | None = None,
) -> int:
    """Rebuild vault_entries and sleeves from vault directories.

    vault_root_map: {vault_root_path: profile_name}.  If None, uses
    lib.core.vault_paths to resolve the standard roots.
    """
    from ensemble.vault.constants import VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES

    if vault_root_map is None:
        from lib.core.vault_paths import (
            resolve_vault_cfd_prop,
            resolve_vault_personal,
            resolve_vault_prop,
        )

        vault_root_map = {
            resolve_vault_prop(): "prop",
            resolve_vault_personal(): "personal",
            resolve_vault_cfd_prop(): "cfd_prop",
        }

    now = _now_iso()
    count = 0

    with transaction(conn):
        conn.execute("DELETE FROM vault_entries")
        conn.execute("DELETE FROM sleeves")

        # Seed all known sleeves
        for sleeve in sorted(VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES):
            conn.execute("INSERT OR IGNORE INTO sleeves (name) VALUES (?)", (sleeve,))

        for vault_root, profile in vault_root_map.items():
            if not vault_root.exists():
                continue

            for json_path, tf, group, ensemble_leaf in _iter_vault_feature_jsons(vault_root):
                _insert_vault_entry(
                    conn, json_path, tf, group, ensemble_leaf, profile, repo_root, now
                )
                count += 1

    return count


def _insert_vault_entry(
    conn: sqlite3.Connection,
    json_path: Path,
    tf: str,
    group: str | None,
    ensemble_leaf: str,
    profile: str,
    repo_root: Path,
    now: str,
) -> None:
    """Parse one feature JSON and upsert into vault_entries."""
    raw = ""
    try:
        raw = json_path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (json.JSONDecodeError, OSError) as exc:
        try:
            raw = json_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            raw = ""
        _upsert_vault_entry_invalid(
            conn, json_path.stem, tf, group, ensemble_leaf, profile, repo_root,
            json_path, raw, str(exc), now,
        )
        return

    try:
        feature_name = data.get("feature_name") or json_path.stem

        bm_list = data.get("base_models") or [{}]
        bm0 = bm_list[0] if bm_list else {}
        feature_column = data.get("feature_column") or bm0.get("feature_column") or ""

        bm0_bn = bm0.get("bias_node_spec") or {}
        top_bn = data.get("bias_node_spec") or {}
        module_name = bm0_bn.get("module_name") or top_bn.get("module_name") or ""

        # Prefer explicit JSON field; fall back to directory-derived group
        wh_group = data.get("weight_hierarchy_group") or group

        # Ensure custom sleeve exists
        if wh_group:
            conn.execute("INSERT OR IGNORE INTO sleeves (name) VALUES (?)", (wh_group,))

        direction = data.get("direction")
        created_at = data.get("created_at") or now
        updated_at = data.get("updated_at") or now

        valid = 1
        validation_error = None
        if not feature_name:
            valid = 0
            validation_error = "missing feature_name"

        try:
            rel_path = str(json_path.relative_to(repo_root))
        except ValueError:
            rel_path = str(json_path)

        writer.upsert_vault_entry(
            conn,
            writer.VaultEntry(
                vault_profile=profile,
                timeframe=tf,
                weight_hierarchy_group=wh_group,
                ensemble_leaf=ensemble_leaf,
                feature_name=feature_name,
                feature_column=feature_column,
                module_name=module_name,
                direction=direction,
                config_json=raw,
                file_path=rel_path,
                valid=valid,
                validation_error=validation_error,
                created_at=created_at,
                updated_at=updated_at,
            ),
        )
    except Exception as exc:
        _upsert_vault_entry_invalid(
            conn, json_path.stem, tf, group, ensemble_leaf, profile, repo_root,
            json_path, raw, str(exc), now,
        )


def _upsert_vault_entry_invalid(
    conn: sqlite3.Connection,
    stem: str,
    tf: str,
    group: str | None,
    ensemble_leaf: str,
    profile: str,
    repo_root: Path,
    json_path: Path,
    raw: str,
    error: str,
    now: str,
) -> None:
    try:
        rel_path = str(json_path.relative_to(repo_root))
    except ValueError:
        rel_path = str(json_path)

    writer.upsert_vault_entry(
        conn,
        writer.VaultEntry(
            vault_profile=profile,
            timeframe=tf,
            weight_hierarchy_group=group,
            ensemble_leaf=ensemble_leaf,
            feature_name=stem,
            feature_column="",
            module_name="",
            config_json=raw,
            file_path=rel_path,
            valid=0,
            validation_error=error,
            created_at=now,
            updated_at=now,
        ),
    )


# ── Builder: calendar ────────────────────────────────────────────────────────


def build_calendar(conn: sqlite3.Connection, repo_root: Path) -> int:
    """Rebuild fomc_decision and nyse_holiday from calendar JSON files."""
    cal_dir = repo_root / "data" / "events" / "calendar"
    count = 0

    with transaction(conn):
        conn.execute("DELETE FROM nyse_holiday")
        conn.execute("DELETE FROM fomc_decision")

        fomc_path = cal_dir / "fomc_decision_dates.json"
        if fomc_path.exists():
            try:
                data = json.loads(fomc_path.read_text(encoding="utf-8"))
                scraped_at = data.get("scraped_at")
                for date_str in data.get("dates", []):
                    conn.execute(
                        "INSERT OR IGNORE INTO fomc_decision"
                        " (decision_date, scraped_at) VALUES (?, ?)",
                        (date_str, scraped_at),
                    )
                    count += 1
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("calendar: could not parse fomc_decision_dates.json: %s", exc)

        nyse_path = cal_dir / "nyse_holiday_events.json"
        if nyse_path.exists():
            try:
                data = json.loads(nyse_path.read_text(encoding="utf-8"))
                for event in data.get("events", []):
                    conn.execute(
                        "INSERT OR IGNORE INTO nyse_holiday"
                        " (holiday_id, closure_date, d0, asset_bucket) VALUES (?, ?, ?, ?)",
                        (
                            event["holiday_id"],
                            event["closure_date"],
                            event["d0"],
                            event["asset_bucket"],
                        ),
                    )
                    count += 1
            except (json.JSONDecodeError, OSError, KeyError) as exc:
                log.warning("calendar: could not parse nyse_holiday_events.json: %s", exc)

    return count


# ── Builder: manifest ────────────────────────────────────────────────────────


def build_manifest(
    conn: sqlite3.Connection,
    repo_root: Path,
    *,
    include_stocks: bool = False,
) -> int:
    """Rebuild blob_manifest from data directory trees.

    Stores covered: ohlc_data, mt5_d1, mt5_m1, mt5_ticks, rollover_ticks.
    With --include-stocks: also stock_data (329 k files — slow).
    """
    now = _now_iso()
    managed_stores = ["ohlc_data", "mt5_d1", "mt5_m1", "mt5_ticks", "rollover_ticks"]
    if include_stocks:
        managed_stores.append("stock_data")

    count = 0

    with transaction(conn):
        for store in managed_stores:
            conn.execute("DELETE FROM blob_manifest WHERE store = ?", (store,))

        # ── ohlc_data ──────────────────────────────────────────────────────
        ohlc_root = repo_root / "data" / "ohlc_data"
        if ohlc_root.exists():
            for ticker_dir in sorted(ohlc_root.iterdir()):
                if not ticker_dir.is_dir():
                    continue
                ticker = ticker_dir.name
                for pq_file in sorted(ticker_dir.glob("*.parquet")):
                    parts = pq_file.stem.split("_")  # e.g. D_CL, D_CL_ratio, D_CL_unadj
                    tf = parts[0] if parts else "D"
                    adj = parts[-1] if len(parts) >= 3 else "none"
                    key = json.dumps({"ticker": ticker, "tf": tf, "adjustment": adj}, sort_keys=True)
                    cov_start, cov_end, rows = _extract_coverage(pq_file, "date")
                    rel = str(pq_file.relative_to(repo_root))
                    mtime = _mtime_iso(pq_file, now)
                    writer.record_blob(
                        conn,
                        writer.BlobRecord(
                            store="ohlc_data",
                            key_json=key,
                            relative_path=rel,
                            written_at=mtime,
                            engine="pyarrow",
                            rows=rows,
                            coverage_start=cov_start,
                            coverage_end=cov_end,
                        ),
                    )
                    count += 1

        # ── mt5_data ───────────────────────────────────────────────────────
        mt5_root = repo_root / "data" / "mt5_data"
        if mt5_root.exists():
            for sym_dir in sorted(mt5_root.iterdir()):
                if not sym_dir.is_dir() or sym_dir.name.startswith("_"):
                    continue
                symbol = sym_dir.name

                # bars_D1 / part.parquet
                d1_file = sym_dir / "bars_D1" / "part.parquet"
                if d1_file.exists():
                    key = json.dumps({"symbol": symbol}, sort_keys=True)
                    cov_start, cov_end, rows = _extract_coverage(d1_file, "time")
                    writer.record_blob(
                        conn,
                        writer.BlobRecord(
                            store="mt5_d1",
                            key_json=key,
                            relative_path=str(d1_file.relative_to(repo_root)),
                            written_at=_mtime_iso(d1_file, now),
                            broker="darwinex",
                            timezone="broker_eet_as_utc",
                            engine="pyarrow",
                            rows=rows,
                            coverage_start=cov_start,
                            coverage_end=cov_end,
                        ),
                    )
                    count += 1

                # bars_M1 / year=* / part.parquet
                m1_dir = sym_dir / "bars_M1"
                if m1_dir.exists():
                    for year_dir in sorted(m1_dir.iterdir()):
                        if not year_dir.is_dir() or not year_dir.name.startswith("year="):
                            continue
                        year = year_dir.name.split("=", 1)[1]
                        for pq_file in sorted(year_dir.glob("*.parquet")):
                            key = json.dumps({"symbol": symbol, "year": year}, sort_keys=True)
                            cov_start, cov_end, rows = _extract_coverage(pq_file, "time")
                            writer.record_blob(
                                conn,
                                writer.BlobRecord(
                                    store="mt5_m1",
                                    key_json=key,
                                    relative_path=str(pq_file.relative_to(repo_root)),
                                    written_at=_mtime_iso(pq_file, now),
                                    broker="darwinex",
                                    timezone="broker_eet_as_utc",
                                    engine="pyarrow",
                                    rows=rows,
                                    coverage_start=cov_start,
                                    coverage_end=cov_end,
                                ),
                            )
                            count += 1

                # ticks / year=* / part.parquet
                ticks_dir = sym_dir / "ticks"
                if ticks_dir.exists():
                    for year_dir in sorted(ticks_dir.iterdir()):
                        if not year_dir.is_dir() or not year_dir.name.startswith("year="):
                            continue
                        year = year_dir.name.split("=", 1)[1]
                        for pq_file in sorted(year_dir.glob("*.parquet")):
                            key = json.dumps({"symbol": symbol, "year": year}, sort_keys=True)
                            cov_start, cov_end, rows = _extract_coverage(pq_file, "time")
                            writer.record_blob(
                                conn,
                                writer.BlobRecord(
                                    store="mt5_ticks",
                                    key_json=key,
                                    relative_path=str(pq_file.relative_to(repo_root)),
                                    written_at=_mtime_iso(pq_file, now),
                                    broker="darwinex",
                                    timezone="broker_eet_as_utc",
                                    engine="pyarrow",
                                    rows=rows,
                                    coverage_start=cov_start,
                                    coverage_end=cov_end,
                                ),
                            )
                            count += 1

                # ticks_rollover_exit / ticks_rollover_entry
                for window_name in ("ticks_rollover_exit", "ticks_rollover_entry"):
                    rollover_dir = sym_dir / window_name
                    if not rollover_dir.exists():
                        continue
                    win = "exit" if "exit" in window_name else "entry"
                    for year_dir in sorted(rollover_dir.iterdir()):
                        if not year_dir.is_dir() or not year_dir.name.startswith("year="):
                            # flat layout (part.parquet directly under rollover_dir)
                            continue
                        year = year_dir.name.split("=", 1)[1]
                        for pq_file in sorted(year_dir.glob("*.parquet")):
                            key = json.dumps(
                                {"symbol": symbol, "year": year, "window": win},
                                sort_keys=True,
                            )
                            cov_start, cov_end, rows = _extract_coverage(pq_file, "time")
                            writer.record_blob(
                                conn,
                                writer.BlobRecord(
                                    store="rollover_ticks",
                                    key_json=key,
                                    relative_path=str(pq_file.relative_to(repo_root)),
                                    written_at=_mtime_iso(pq_file, now),
                                    broker="darwinex",
                                    timezone="broker_eet_as_utc",
                                    engine="pyarrow",
                                    rows=rows,
                                    coverage_start=cov_start,
                                    coverage_end=cov_end,
                                ),
                            )
                            count += 1

        # ── stock_data (optional) ──────────────────────────────────────────
        if include_stocks:
            stock_root = repo_root / "data" / "stock_data"
            if stock_root.exists():
                for sym_dir in sorted(stock_root.iterdir()):
                    if not sym_dir.is_dir():
                        continue
                    symbol = sym_dir.name
                    for pq_file in sorted(sym_dir.glob("*.parquet")):
                        key = json.dumps({"symbol": symbol}, sort_keys=True)
                        cov_start, cov_end, rows = _extract_coverage(pq_file, "date")
                        writer.record_blob(
                            conn,
                            writer.BlobRecord(
                                store="stock_data",
                                key_json=key,
                                relative_path=str(pq_file.relative_to(repo_root)),
                                written_at=_mtime_iso(pq_file, now),
                                engine="pyarrow",
                                rows=rows,
                                coverage_start=cov_start,
                                coverage_end=cov_end,
                            ),
                        )
                        count += 1

    return count


def _mtime_iso(path: Path, fallback: str) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    except OSError:
        return fallback


# ── Builder: ticks ───────────────────────────────────────────────────────────


def build_ticks(conn: sqlite3.Connection, repo_root: Path) -> int:
    """Rebuild tick_chunks from ticks_cache/<start_ns>-<end_ns>.parquet filenames."""
    mt5_root = repo_root / "data" / "mt5_data"
    if not mt5_root.exists():
        return 0

    count = 0

    with transaction(conn):
        conn.execute("DELETE FROM tick_chunks")

        for sym_dir in sorted(mt5_root.iterdir()):
            if not sym_dir.is_dir() or sym_dir.name.startswith("_"):
                continue
            symbol = sym_dir.name

            cache_dir = sym_dir / "ticks_cache"
            if not cache_dir.exists():
                continue

            for chunk_file in sorted(cache_dir.glob("*.parquet")):
                stem = chunk_file.stem  # e.g. 1749509700000000000-1749521100000000000
                parts = stem.split("-")
                if len(parts) != 2:
                    continue
                try:
                    start_ns = int(parts[0])
                    end_ns = int(parts[1])
                except ValueError:
                    continue

                try:
                    n_ticks = pq.ParquetFile(chunk_file).metadata.num_rows
                except Exception:
                    n_ticks = 0

                rel = str(chunk_file.relative_to(repo_root))
                conn.execute(
                    "INSERT OR IGNORE INTO tick_chunks"
                    " (symbol, start_ns, end_ns, file_path, n_ticks) VALUES (?, ?, ?, ?, ?)",
                    (symbol, start_ns, end_ns, rel, n_ticks),
                )
                count += 1

    return count


# ── Orchestrator ─────────────────────────────────────────────────────────────


def orchestrate(
    conn: sqlite3.Connection,
    *,
    repo_root: Path | None = None,
    domains: list[str] | None = None,
    include_stocks: bool = False,
    log_dir: Path | None = None,
    vault_root_map: dict[Path, str] | None = None,
) -> dict[str, int]:
    """Run selected domain builders; return per-domain row counts.

    Also records a job_runs row for the rebuild job (name='registry_rebuild').
    """
    repo_root = repo_root or _repo_root()
    selected = list(domains) if domains is not None else list(ALL_DOMAINS)
    log_dir = log_dir or (repo_root / "logs" / "registry_backfill")

    started_at = _now_iso()
    args_json = json.dumps({"domains": selected, "include_stocks": include_stocks})

    # Record the job start (committed immediately so it survives partial failure)
    with transaction(conn):
        job_run_id = writer.record_job_run(
            conn,
            writer.JobRun(job_name="registry_rebuild", started_at=started_at, args_json=args_json),
        )

    _builders: dict[str, object] = {
        "instruments": lambda: build_instruments(conn, repo_root),
        "specs": lambda: build_specs(conn, repo_root),
        "runs": lambda: build_runs(conn, repo_root, log_dir),
        "vault": lambda: build_vault(conn, repo_root, vault_root_map=vault_root_map),
        "calendar": lambda: build_calendar(conn, repo_root),
        "manifest": lambda: build_manifest(conn, repo_root, include_stocks=include_stocks),
        "ticks": lambda: build_ticks(conn, repo_root),
    }

    counts: dict[str, int] = {}
    t0 = time.monotonic()
    error_text: str | None = None
    total = 0

    try:
        for domain in selected:
            if domain not in _builders:
                log.warning("Unknown domain %r — skipped", domain)
                continue
            n: int = _builders[domain]()  # type: ignore[operator]
            counts[domain] = n
            total += n
            log.info("%-12s  %d rows", domain, n)
    except Exception as exc:
        error_text = str(exc)
        raise
    finally:
        duration = time.monotonic() - t0
        coverage_json = json.dumps({"duration_seconds": round(duration, 2), "counts": counts})
        with transaction(conn):
            writer.finish_job_run(
                conn,
                job_run_id,
                exit_code=1 if error_text else 0,
                rows_written=total,
                coverage_json=coverage_json,
                error_text=error_text,
            )

    return counts
