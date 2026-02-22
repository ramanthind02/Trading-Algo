"""Entry point for the rule-based feature EDA research pipeline.

Usage
-----
    source /home/raman/repos/Trading-Algo/venv/bin/activate
    python feature_research/in_sample/rule_based/run_eda.py

Results are written to feature_research/in_sample/rule_based/results/{module_name}/.
Edit feature_research/in_sample/rule_based/config.py to change tickers, dates, or bias specs.
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

from feature_research.in_sample.rule_based.config import load_config
from feature_research.in_sample.rule_based.pipeline import run_rule_based_eda_pipeline

if __name__ == "__main__":
    config = load_config()
    results = run_rule_based_eda_pipeline(config, config.reports_dir)
    print(f"EDA complete. {len(results)} param combos written to {config.reports_dir}")
