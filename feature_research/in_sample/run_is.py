"""Unified in-sample research entry point for both continuous and rule-based features.

This single script runs the full in-sample research pipeline:
  1. EDA (exploratory data analysis for all param combos)
  2. Optional Phase 2 binning analysis (continuous only)
  3. Optional walkforward validation

Usage
-----
    source /home/raman/repos/Trading-Algo/venv/bin/activate
    python feature_research/in_sample/run_is.py

To customize tickers, dates, bias_spec, or phase settings, edit
``feature_research/config.py`` and ``feature_research/in_sample/config.py``.

To disable walkforward, set ``enabled=False`` in the walkforward config.

Output
------
All reports and artifacts are written to the configured ``reports_dir`` from the
in-sample config (e.g., ``feature_research/in_sample/results/continuous/`` or
``feature_research/in_sample/results/rule_based/``).
"""
import dataclasses
import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path | None:
    """Search up from start path to find repo root (pyproject.toml or .git)."""
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

from feature_research.in_sample.config import load_config
from feature_research.in_sample.pipeline import run_eda_pipeline, run_walkforward_pipeline


if __name__ == "__main__":
    config = load_config()

    # Run EDA for all param combos
    print(f"\n{'*'*70}")
    print(f"PHASE 1: In-Sample EDA ({config.feature_type.value.upper()})")
    print(f"{'*'*70}")
    eda_results = run_eda_pipeline(config, config.reports_dir)

    # Run optional walkforward validation
    if config.walkforward.enabled:
        print(f"\n{'*'*70}")
        print(f"PHASE 2: Walkforward Validation ({config.feature_type.value.upper()})")
        print(f"{'*'*70}")
        walkforward_report = run_walkforward_pipeline(config, config.reports_dir / "walkforward")
        print(
            f"\nIn-sample research complete. "
            f"{len(eda_results)} EDA combos, "
            f"{len(walkforward_report.folds_df)} walkforward folds. "
            f"Artifacts in {config.reports_dir}"
        )
    else:
        print(f"\n{'*'*70}")
        print("Walkforward validation disabled (set config.walkforward.enabled=True to enable)")
        print(f"{'*'*70}")
        print(f"\nIn-sample research complete. {len(eda_results)} EDA combos. "
              f"Artifacts in {config.reports_dir}")
