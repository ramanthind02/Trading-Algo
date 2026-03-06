"""Portfolio test entrypoint.

Loads portfolio research config and runs the test pipeline.
See portfolio_research/pipelines/portfolio_test.py for implementation.

Usage
-----
    source /home/raman/repos/Trading-Algo/venv/bin/activate
    python portfolio_research/run_portfolio_test.py

Artifacts are written to config.output_root.
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

from portfolio_research.config import load_config
from portfolio_research.pipelines.portfolio_test import run_portfolio_test_pipeline


if __name__ == "__main__":
    config = load_config()
    run_portfolio_test_pipeline(config)
