"""Entry point for the continuous binning EDA research pipeline.

Usage
-----
    source /home/raman/repos/Trading-Algo/venv/bin/activate
    python feature_research/continuous_binning/run_eda.py

Results are written to feature_research/continuous_binning/results/{module_name}/.
Edit feature_research/continuous_binning/config.py to change tickers, dates, or bias specs.
"""
from feature_research.continuous_binning.config import load_config
from feature_research.continuous_binning.pipeline import run_continuous_eda_pipeline

if __name__ == "__main__":
    config = load_config()
    results = run_continuous_eda_pipeline(config, config.reports_dir)
    print(f"EDA complete. {len(results)} param combos written to {config.reports_dir}")
