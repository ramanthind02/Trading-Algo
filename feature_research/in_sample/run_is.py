"""Unified in-sample research entry point for both continuous and rule-based features.

This single script runs the full in-sample research pipeline:
  1. EDA (exploratory data analysis for all param combos)
  2. Optional in-sample permutation (Stage 1 vector shuffle → Stage 2 candle shuffle)
  3. Optional Phase 2 binning analysis (continuous only)
  4. Optional walkforward validation

Usage
-----
    source /home/raman/repos/Trading-Algo/venv/bin/activate
    python feature_research/in_sample/run_is.py

To customize tickers, dates, bias_spec, walkforward defaults, or phase presets, edit
``feature_research/config.py`` (single source of truth). ``feature_research/in_sample/config.py``
mainly defines the runtime dataclasses and assembles shared defaults.

To enable permutation: set ``in_sample_permutation.enabled=True`` in ``feature_research/config.py``.
To speed up Stage 2: set ``in_sample_permutation.n_jobs_stage2_reps`` to the number of
CPU cores to use (e.g. 4); reps run in separate processes (real parallelism, no GIL).
To disable walkforward, set ``enabled=False`` in the walkforward config.

To profile Stage 2 (find bottlenecks): run
``python feature_research/in_sample/profile_stage2.py --reps 2 --out profile.stats``
then ``snakeviz profile.stats``.

Optional: for faster bias-node computation (ATR, EMA, RSI, etc.), build the Cython
extensions: ``python utils/compute/cython/setup_cython.py build_ext --inplace``.
Nodes that use ``utils.compute.fast_nodes`` then use the compiled path when available.

Output
------
All reports and artifacts are written to the configured ``reports_dir`` from the
in-sample config (e.g., ``feature_research/in_sample/results/continuous/`` or
``feature_research/in_sample/results/rule_based/``). When permutation is enabled,
``permutation_summary.csv`` and ``permutation_summary.md`` are written there.
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
from feature_research.pipeline import (
    run_eda_pipeline,
    run_permutation_pipeline,
    run_walkforward_pipeline,
    write_permutation_summary,
)


if __name__ == "__main__":
    config = load_config()

    print(f"\nPermutation suite: enabled={config.in_sample_permutation.enabled}")
    print(f"Reports dir: {config.reports_dir.resolve()}")

    # Run EDA for all param combos
    print(f"\n{'*'*70}")
    print(f"PHASE 1: In-Sample EDA ({config.feature_type.value.upper()})")
    print(f"{'*'*70}")
    eda_results = run_eda_pipeline(config, config.reports_dir)

    # Run optional in-sample permutation test (Stage 1 vector shuffle → Stage 2 candle shuffle)
    if config.in_sample_permutation.enabled:
        print(f"\n{'*'*70}")
        print(f"PHASE 2: In-Sample Permutation ({config.feature_type.value.upper()})")
        print(f"{'*'*70}")
        try:
            permutation_suite = run_permutation_pipeline(config, config.reports_dir)
            csv_path, md_path = write_permutation_summary(permutation_suite, config.reports_dir)
            print(f"\nPermutation summary written:")
            print(f"  CSV: {csv_path.resolve()}")
            print(f"  MD:  {md_path.resolve()}")
        except Exception as e:
            print(f"\nPermutation pipeline failed: {e}")
            raise

    # Run optional walkforward validation
    if config.walkforward.enabled:
        print(f"\n{'*'*70}")
        print(f"Walkforward: Validation ({config.feature_type.value.upper()})")
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
