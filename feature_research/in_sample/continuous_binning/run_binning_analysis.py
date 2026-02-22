"""Entry point for the continuous binning analysis pipeline.

Usage
-----
    source /home/raman/repos/Trading-Algo/venv/bin/activate
    python feature_research/in_sample/continuous_binning/run_binning_analysis.py
"""
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

from feature_research.in_sample.continuous_binning.binning_analysis import run_binning_analysis_pipeline
from feature_research.in_sample.continuous_binning.config import load_config


if __name__ == "__main__":
    config = load_config()
    results = run_binning_analysis_pipeline(config, config.reports_dir)
    print(f"Binning analysis complete. {len(results)} reports written to {config.reports_dir}")
