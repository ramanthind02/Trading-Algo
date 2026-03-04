"""Unified walkforward research entry point for both continuous and rule-based features.

Runs walkforward-only analysis (no EDA). Selection method is controlled by config:
default is top_k (see WalkforwardResearchConfig); set
``walkforward_defaults.selection_method`` in ``feature_research/config.py`` to TOP_K
or ENHANCED if needed.

This script dispatches on feature_type to use the appropriate evaluation strategy:
  - CONTINUOUS: uses continuous binning with bin-count expansion
  - RULE_BASED: uses simple returns multiplier approach

Usage
-----
    source /home/raman/repos/Trading-Algo/venv/bin/activate
    python feature_research/walkforward/run_walkforward.py

Artifacts are written to the ``output_root`` defined in ``config.walkforward``
(default: ``feature_research/shared_results/{feature_type}/{module_name}/walkforward/``).

To customize tickers, dates, bias_spec, or walkforward parameters, edit
``feature_research/config.py`` (single source of truth for shared defaults and phase presets).
"""
import dataclasses
import sys
from pathlib import Path

_repo_hint = Path(__file__).resolve().parents[2]
if str(_repo_hint) not in sys.path:
    sys.path.insert(0, str(_repo_hint))

from feature_research.bootstrap import ensure_repo_root_on_syspath

ensure_repo_root_on_syspath(Path(__file__).resolve())

from feature_research.in_sample.config import load_config
from feature_research.pipeline import run_walkforward_pipeline


if __name__ == "__main__":
    config = load_config()

    walkforward = dataclasses.replace(
        config.walkforward,
        enabled=True,
    )
    config = dataclasses.replace(config, walkforward=walkforward)

    report = run_walkforward_pipeline(config, config.reports_dir / "walkforward")
    n_folds = len(report.folds_df)
    print(
        f"Walkforward complete. {n_folds} folds. "
        f"Artifacts written to {config.walkforward.output_root}"
    )
