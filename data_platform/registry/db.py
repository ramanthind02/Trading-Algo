"""Registry database connection management.

WAL, synchronous=NORMAL, foreign_keys=ON applied per-connection.
Single-writer discipline: only writer.py mutates; reader.py and
connect_readonly() enforce the read-only boundary (ADR-3, ADR-9).
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


SCHEMA_VERSION = 1


class RegistryVersionError(RuntimeError):
    """Raised when an existing DB has a user_version != SCHEMA_VERSION."""


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    return next(
        (p for p in here.parents if (p / ".git").exists() or (p / "AGENTS.md").exists()),
        here.parents[2],
    )


def registry_path() -> Path:
    """Canonical path to data/registry.db under the repo root."""
    return _repo_root() / "data" / "registry.db"


def _apply_pragmas(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")


def connect(path: Path | None = None) -> sqlite3.Connection:
    """Open a read-write connection to the registry DB.

    - Creates the DB file and initialises the schema when the DB is new/empty.
    - Raises RegistryVersionError when the DB exists but user_version != SCHEMA_VERSION.
    """
    db_path = Path(path) if path is not None else registry_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    _apply_pragmas(conn)

    table_count: int = conn.execute(
        "SELECT count(*) FROM sqlite_master WHERE type='table'"
    ).fetchone()[0]

    if table_count == 0:
        schema_sql = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")
        conn.executescript(schema_sql)
        # Re-apply per-connection pragmas; executescript issues an implicit COMMIT
        # which does not reset them, but we set them again for clarity.
        _apply_pragmas(conn)
    else:
        user_version: int = conn.execute("PRAGMA user_version").fetchone()[0]
        if user_version != SCHEMA_VERSION:
            conn.close()
            raise RegistryVersionError(
                f"Registry DB schema version is {user_version!r}, expected {SCHEMA_VERSION}. "
                "Apply forward-only migration scripts — see ADR-10 in "
                "docs/library/Data/data_platform_migration_plan.md."
            )

    return conn


def connect_readonly(path: Path | None = None) -> sqlite3.Connection:
    """Open a read-only connection to an existing registry DB (ADR-9).

    Raises FileNotFoundError if the DB file does not exist.
    """
    db_path = Path(path) if path is not None else registry_path()
    if not db_path.exists():
        raise FileNotFoundError(f"Registry DB not found at {db_path!r}")

    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[None]:
    """Commit on success, roll back on any exception.

    Writer functions do NOT auto-commit; callers must wrap mutations here::

        with transaction(conn):
            writer.upsert_instrument(conn, ...)
            writer.replace_source_symbols(conn, ...)
    """
    try:
        yield
        conn.commit()
    except Exception:
        conn.rollback()
        raise
