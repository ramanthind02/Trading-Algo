"""Unified validation entrypoint for the signed-signal trading pipeline.

Run with (repo root as cwd):

    Linux/macOS:  python -m feature_research validation
                  # or:  python feature_research/validation/run_validation.py

    Windows:      .\\.venv\\Scripts\\python.exe -m feature_research validation
                  # or:  .\\.venv\\Scripts\\python.exe feature_research\\validation\\run_validation.py

Direct script execution prepends the repo root to ``sys.path`` (stdlib only) so
``feature_research`` imports work without ``PYTHONPATH``.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _prepend_repo_root_to_syspath() -> None:
    """Allow ``python path/to/run_validation.py`` without PYTHONPATH."""
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
from feature_research.pipeline import run_validation_pipeline


def main() -> None:
    config = load_config()
    # Use default output path: output_root/feature_type/module_name/validation
    report = run_validation_pipeline(config, output_dir=None)
    print(
        f"Validation complete. {len(report.folds_df)} fold(s). "
        f"Artifacts written to {config.output_root}"
    )


if __name__ == "__main__":
    main()
