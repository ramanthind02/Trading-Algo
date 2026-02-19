"""Entry point for the rule-based feature EDA research pipeline.

Usage
-----
    source /home/raman/repos/Trading-Algo/venv/bin/activate
    python feature_research/rule_based/run_eda.py

Results are written to feature_research/rule_based/results/{module_name}/.
Edit feature_research/rule_based/config.py to change tickers, dates, or bias specs.
"""
import sys
from pathlib import Path

# Ensure project root is on path when run as script (e.g. python feature_research/.../run_eda.py)
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from feature_research.rule_based.config import load_config
from feature_research.rule_based.pipeline import run_rule_based_eda_pipeline

if __name__ == "__main__":
    config = load_config()
    results = run_rule_based_eda_pipeline(config, config.reports_dir)
    print(f"EDA complete. {len(results)} param combos written to {config.reports_dir}")
