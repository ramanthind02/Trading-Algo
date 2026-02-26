"""Unified out-of-sample entrypoint for both continuous and rule-based research.

Run with:
    source /home/raman/repos/Trading-Algo/venv/bin/activate
    python feature_research/oos/run_oos.py

Uses config.oos_window from feature_research.config.load_config() for the single
OOS fold (train/test dates). Reuses the same evaluation and selection logic as
walkforward; artifacts are written under output_root/.../oos/.
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

from feature_research.in_sample.config import load_config
from feature_research.pipeline import run_oos_pipeline


def main() -> None:
    config = load_config()
    report = run_oos_pipeline(config)
    print(
        f"OOS complete. {len(report.folds_df)} fold(s). "
        f"Artifacts written to {config.walkforward.output_root}"
    )


if __name__ == "__main__":
    main()
