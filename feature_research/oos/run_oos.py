"""Unified out-of-sample entrypoint for both continuous and rule-based research.

Run with:
    source /home/raman/repos/Trading-Algo/venv/bin/activate
    python feature_research/oos/run_oos.py

This is a stub implementation. OOS pipeline for research not yet fully implemented.
When ready, this will dispatch on feature_type (from config) to run the appropriate
out-of-sample validation for continuous or rule-based features.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path | None:
    """Search up from start path to find repo root (pyproject.toml or .git)."""
    search_root = start if start.is_dir() else start.parent

    for parent in (search_root, *search_root.parents):
        if (parent / "pyproject.toml").exists():
            return parent
        if (parent / ".git").exists():
            return parent

    return None


_repo_root = _find_repo_root(Path(__file__).resolve())
if _repo_root is not None and str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def main() -> None:
    raise NotImplementedError(
        "OOS pipeline for research not implemented yet. "
        "When implemented, this will dispatch on config.feature_type "
        "to run continuous or rule-based OOS validation."
    )


if __name__ == "__main__":
    main()
