"""Open/locate the Nautilus ``ParquetDataCatalog`` for the repo.

The canonical store lives at ``data/nautilus_catalog/`` under the repo root.
``get_catalog`` returns a ready-to-use ``ParquetDataCatalog`` (creating the
directory if needed); pass an explicit ``path`` to root it elsewhere (e.g. a
``tmp_path`` in tests).
"""
from __future__ import annotations

from pathlib import Path

from nautilus_trader.persistence.catalog import ParquetDataCatalog


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    return next(
        (p for p in here.parents if (p / ".git").exists() or (p / "AGENTS.md").exists()),
        here.parents[2],
    )


def default_catalog_path() -> Path:
    """The canonical on-disk Nautilus catalog root: ``data/nautilus_catalog/``."""
    return _repo_root() / "data" / "nautilus_catalog"


def get_catalog(path: str | Path | None = None) -> ParquetDataCatalog:
    """Open (or create) a ``ParquetDataCatalog`` rooted at *path*.

    Defaults to :func:`default_catalog_path`. The directory is created if it
    does not yet exist so a fresh catalog can be written immediately.
    """
    root = Path(path) if path is not None else default_catalog_path()
    root.mkdir(parents=True, exist_ok=True)
    return ParquetDataCatalog(path=str(root))
