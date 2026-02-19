"""Entry point for the continuous binning analysis pipeline.

Usage
-----
    source /home/raman/repos/Trading-Algo/venv/bin/activate
    python feature_research/continuous_binning/run_binning_analysis.py
"""
import sys
from pathlib import Path

# Ensure project root is on path when run as script (e.g. python feature_research/.../run_binning_analysis.py)
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from feature_research.continuous_binning.binning_analysis import run_binning_analysis_pipeline
from feature_research.continuous_binning.config import load_config


if __name__ == "__main__":
    config = load_config()
    results = run_binning_analysis_pipeline(config, config.reports_dir)
    print(f"Binning analysis complete. {len(results)} reports written to {config.reports_dir}")
