from __future__ import annotations

from pathlib import Path

from utils.repo_bootstrap import ensure_repo_root_on_syspath, find_repo_root

__all__ = ["ensure_repo_root_on_syspath", "find_repo_root"]


def ensure_feature_research_repo_root(start: Path) -> Path:
    """Compatibility wrapper for callers expecting a returned root path."""
    return ensure_repo_root_on_syspath(start)
