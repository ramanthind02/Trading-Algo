from __future__ import annotations

import sys
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path | None:
    """Search upward from ``start`` (or cwd) for the repository root."""
    candidate = (start or Path.cwd()).resolve()
    search_root = candidate if candidate.is_dir() else candidate.parent
    for parent in (search_root, *search_root.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return None


def require_repo_root(start: Path | None = None) -> Path:
    """Return the repository root or raise when it cannot be located."""
    repo_root = find_repo_root(start)
    if repo_root is None:
        raise RuntimeError("Could not locate repository root from the provided start path.")
    return repo_root


def ensure_repo_root_on_syspath(start: Path | None = None) -> Path:
    """Ensure the repository root is importable for direct script execution."""
    repo_root = require_repo_root(start)
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


def ensure_project_root_on_path(start: Path | None = None) -> Path:
    """Backward-compatible alias used by older scripts."""
    return ensure_repo_root_on_syspath(start)


def resolve_repo_path(path: Path | str, *, repo_root: Path | None = None) -> Path:
    """Resolve ``path`` against the repository root when it is not absolute."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    root = repo_root or require_repo_root()
    return root / candidate


def project_root() -> Path:
    """Repository root anchored at this module's location (stable regardless of cwd).

    Walks upward from this file for a repo marker (``.git`` / ``pyproject.toml`` /
    ``AGENTS.md``); falls back to the known depth of this module in the package tree.
    Lives in ``lib.core`` so foundation modules (e.g. ``vault_paths``) and the cache
    layer can both depend on it without a layering cycle.
    """
    resolved = Path(__file__).resolve()
    return next(
        (
            parent
            for parent in resolved.parents
            if (parent / ".git").exists()
            or (parent / "pyproject.toml").exists()
            or (parent / "AGENTS.md").exists()
        ),
        resolved.parents[2],
    )
