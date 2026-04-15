"""Unified out-of-sample entrypoint for the signed-signal trading pipeline.

Run with (repo root as cwd):

    Linux/macOS:  python -m feature_research oos
                  # or:  python feature_research/oos/run_oos.py

    Windows:      .\\.venv\\Scripts\\python.exe -m feature_research oos
                  # or:  .\\.venv\\Scripts\\python.exe feature_research\\oos\\run_oos.py

Uses config.oos_window from feature_research.config.load_config() for the single
OOS fold (train/test dates). Artifacts are written under output_root/.../oos/.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _prepend_repo_root_to_syspath() -> None:
    """Allow ``python path/to/run_oos.py`` without PYTHONPATH."""
    start = Path(__file__).resolve()
    for parent in (start.parent, *start.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            root = str(parent)
            if root not in sys.path:
                sys.path.insert(0, root)
            return
    raise RuntimeError(
        "Could not locate repository root (no pyproject.toml or .git above this file)."
    )


_prepend_repo_root_to_syspath()

from feature_research.bootstrap import ensure_repo_root_on_syspath

ensure_repo_root_on_syspath(Path(__file__).resolve())

from feature_research.config import load_config
from feature_research.pipeline import run_oos_pipeline


def main() -> None:
    config = load_config()
    report = run_oos_pipeline(config)
    print(
        f"OOS complete. {len(report.folds_df)} fold(s). "
        f"Artifacts written to {config.output_root}"
    )


if __name__ == "__main__":
    main()
