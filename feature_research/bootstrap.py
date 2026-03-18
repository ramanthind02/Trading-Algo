from __future__ import annotations

import sys
from pathlib import Path


def find_repo_root(start: Path) -> Path | None:
    """Search up from start path to find repository root."""
    search_root = start if start.is_dir() else start.parent
    for parent in (search_root, *search_root.parents):
        if (parent / "pyproject.toml").exists():
            return parent
        if (parent / ".git").exists():
            return parent
    return None


def ensure_repo_root_on_syspath(start: Path) -> None:
    """Ensure repository root is on sys.path for direct script execution."""
    repo_root = find_repo_root(start)
    if repo_root is not None and str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
