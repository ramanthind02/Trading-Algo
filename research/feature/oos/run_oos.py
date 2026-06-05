"""Unified out-of-sample entrypoint for the signed-signal trading pipeline.

Run with (repo root as cwd):

    Linux/macOS:  python -m feature_research portfolio_addition
                  # alias: python -m feature_research oos
                  # or:  python feature_research/oos/run_oos.py

    Windows:      .\\.venv\\Scripts\\python.exe -m feature_research portfolio_addition
                  # alias: .\\.venv\\Scripts\\python.exe -m feature_research oos
                  # or:  .\\.venv\\Scripts\\python.exe feature_research\\oos\\run_oos.py

Uses config.research_window from feature_research.config.load_config() for the single
train/validation fold. Artifacts are written under output_root/.../oos/.
"""
from __future__ import annotations

from pathlib import Path

from lib.core.repo_bootstrap import ensure_repo_root_on_syspath

ensure_repo_root_on_syspath(Path(__file__).resolve())

from research.feature.config import load_config
from research.feature.portfolio_addition import run_portfolio_addition_pipeline


def main() -> None:
    config = load_config()
    report = run_portfolio_addition_pipeline(config)
    print(
        f"OOS complete. {len(report.folds_df)} fold(s). "
        f"Artifacts written to {config.output_root}"
    )


if __name__ == "__main__":
    main()
