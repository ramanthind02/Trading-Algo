"""Registry hot-backup via the SQLite3 backup API (ADR-10).

Usage (via __main__.py CLI):
    python -m data_platform.registry backup
    python -m data_platform.registry backup --out-dir /mnt/nas/registry_backups
    python -m data_platform.registry backup --db /path/to/registry.db

The backup file is a point-in-time snapshot named::

    registry-YYYYMMDD-HHMMSS.db

Stored under ``data/backups/`` by default (one directory level above
``data/registry.db``).  After each successful backup, any files beyond the
newest **14** are pruned so the backup directory does not grow unboundedly.

A ``job_runs`` row (job_name='registry_backup') is written to the *live* DB
around each run.  The backup itself includes the job-start row (captured
before the snapshot) but not the job-finish row — this is expected and
harmless.

Resilience rule: any failure in the job_run recording must not abort the
backup.  The backup always completes or raises on its own error.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    return next(
        (p for p in here.parents if (p / ".git").exists() or (p / "AGENTS.md").exists()),
        here.parents[2],
    )


def backup(
    db_path: Path | None = None,
    out_dir: Path | None = None,
) -> Path:
    """Copy the registry DB to *out_dir* using ``sqlite3.Connection.backup()``.

    Parameters
    ----------
    db_path:
        Source registry DB.  Defaults to ``data/registry.db``.
    out_dir:
        Destination directory.  Defaults to ``data/backups/``.

    Returns
    -------
    Path
        Absolute path to the newly created backup file.

    Raises
    ------
    FileNotFoundError
        When the source DB does not exist.
    """
    from data_platform.registry.db import registry_path, connect

    src_path = Path(db_path) if db_path is not None else registry_path()
    if not src_path.exists():
        raise FileNotFoundError(f"Registry DB not found at {src_path!r}")

    repo = _repo_root()
    dest_dir = Path(out_dir) if out_dir is not None else repo / "data" / "backups"
    dest_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    dest_path = dest_dir / f"registry-{ts}.db"

    # ── Record job start in the live DB (resilience-wrapped) ─────────────────
    _src_conn = None
    _job_run_id: int | None = None
    try:
        from data_platform.registry import writer as _w, db as _db
        _src_conn = connect(src_path)
        with _db.transaction(_src_conn):
            _job_run_id = _w.record_job_run(
                _src_conn,
                _w.JobRun(
                    job_name="registry_backup",
                    started_at=datetime.now(timezone.utc).isoformat(),
                    args_json=None,
                ),
            )
    except Exception as _exc:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).warning(
            "backup: could not record job_run start: %s", _exc
        )
        # If we failed to open a connection, open a fresh read-write one just for backup.
        if _src_conn is None:
            try:
                _src_conn = sqlite3.connect(str(src_path))
                _src_conn.row_factory = sqlite3.Row
            except Exception:  # noqa: BLE001
                _src_conn = None

    # ── Perform the backup ────────────────────────────────────────────────────
    # If we already have a connection from above, reuse it.  Otherwise open a
    # plain read-only connection just for the backup call.
    _close_src = _src_conn is None
    if _src_conn is None:
        _src_conn = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True)

    try:
        dest_conn = sqlite3.connect(str(dest_path))
        try:
            _src_conn.backup(dest_conn)
        finally:
            dest_conn.close()
    except Exception:
        # Cleanup a partially-written file on failure
        if dest_path.exists():
            try:
                dest_path.unlink()
            except OSError:
                pass
        raise

    # ── Record job finish in the live DB (resilience-wrapped) ─────────────────
    if _src_conn is not None and _job_run_id is not None:
        try:
            from data_platform.registry import writer as _w2, db as _db2
            with _db2.transaction(_src_conn):
                _w2.finish_job_run(
                    _src_conn, _job_run_id,
                    exit_code=0,
                    rows_written=0,
                    coverage_json=None,
                    error_text=None,
                )
        except Exception as _exc2:  # noqa: BLE001
            import logging
            logging.getLogger(__name__).warning(
                "backup: could not record job_run finish: %s", _exc2
            )

    if _close_src and _src_conn is not None:
        try:
            _src_conn.close()
        except Exception:  # noqa: BLE001
            pass
    elif _src_conn is not None:
        try:
            _src_conn.close()
        except Exception:  # noqa: BLE001
            pass

    # ── Prune old backups ─────────────────────────────────────────────────────
    _prune(dest_dir, keep=14)

    return dest_path


def _prune(backup_dir: Path, *, keep: int = 14) -> None:
    """Delete the oldest backup files, retaining the ``keep`` newest."""
    files = sorted(backup_dir.glob("registry-*.db"))
    to_delete = files[:-keep] if len(files) > keep else []
    for f in to_delete:
        try:
            f.unlink()
        except OSError:
            pass
