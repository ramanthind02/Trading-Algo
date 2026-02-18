"""Entry point for the continuous binning analysis pipeline.

Usage
-----
    source /home/raman/repos/Trading-Algo/venv/bin/activate
    python feature_research/continuous_binning/run_binning_analysis.py
"""
from feature_research.continuous_binning.binning_analysis import run_binning_analysis_pipeline
from feature_research.continuous_binning.config import load_config


if __name__ == "__main__":
    config = load_config()
    results = run_binning_analysis_pipeline(config, config.reports_dir)
    print(f"Binning analysis complete. {len(results)} reports written to {config.reports_dir}")
