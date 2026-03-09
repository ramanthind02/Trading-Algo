"""Unified in-sample research entry point for both continuous and rule-based features.

This single script runs the in-sample research pipeline:
  1. EDA (exploratory data analysis for all param combos)
  2. Optional in-sample permutation (Stage 1 vector shuffle → Stage 2 candle shuffle)
  3. Optional Phase 2 binning analysis (continuous only)

Validation is a separate phase; run ``feature_research/validation/run_validation.py`` for that.

Usage
-----
    source /home/raman/repos/Trading-Algo/venv/bin/activate
    python feature_research/in_sample/run_is.py

To customize tickers, dates, bias_spec, or phase presets, edit
``feature_research/config.py`` (single source of truth). ``feature_research/in_sample/config.py``
mainly defines the runtime dataclasses and assembles shared defaults.

To enable permutation: set ``permutation.enabled=True`` in ``feature_research/config.py``.
To speed up Stage 2: set ``permutation.n_jobs_stage2_reps`` to the number of
CPU cores to use (e.g. 4); reps run in separate processes (real parallelism, no GIL).

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

_repo_hint = Path(__file__).resolve().parents[2]
if str(_repo_hint) not in sys.path:
    sys.path.insert(0, str(_repo_hint))

from feature_research.bootstrap import ensure_repo_root_on_syspath

ensure_repo_root_on_syspath(Path(__file__).resolve())

from feature_research.config import FeatureType
from feature_research.in_sample.binning_analysis import run_binning_analysis_pipeline
from feature_research.in_sample.config import load_config
from feature_research.pipeline import (
    run_eda_pipeline,
    run_permutation_pipeline,
    write_permutation_summary,
)


if __name__ == "__main__":
    config = load_config()

    print(f"\nPermutation enabled: {config.permutation.enabled}")
    print(f"Reports dir: {config.reports_dir.resolve()}")

    # Run EDA for all param combos
    print(f"\n{'*'*70}")
    print(f"PHASE 1: In-Sample EDA ({config.feature_type.value.upper()})")
    print(f"{'*'*70}")
    eda_results = run_eda_pipeline(config, config.reports_dir)

    # Run optional continuous binning analysis (Phase 2, continuous features only)
    if config.feature_type == FeatureType.CONTINUOUS:
        print(f"\n{'*'*70}")
        print(f"PHASE 2: Continuous Binning Analysis")
        print(f"{'*'*70}")
        run_binning_analysis_pipeline(config, config.reports_dir)

    # Run optional in-sample permutation test (Stage 1 vector shuffle → Stage 2 candle shuffle)
    if config.permutation.enabled:
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

    msg = f"\nIn-sample research complete. {len(eda_results)} EDA combos. Artifacts in {config.reports_dir}"
    if config.permutation.enabled:
        msg += " Permutation summary CSV/MD in reports_dir."
    print(msg)
