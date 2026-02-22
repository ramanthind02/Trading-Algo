"""Out-of-sample entrypoint for rule-based research.

Run with:
    source /home/raman/repos/Trading-Algo/venv/bin/activate
    python feature_research/oos/rule_based/run_oos.py
"""
from __future__ import annotations

import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path | None:
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
    raise NotImplementedError("OOS pipeline for rule-based research not implemented yet.")


if __name__ == "__main__":
    main()
