"""Unified in-sample research entry point for the signed-signal trading pipeline.

This single script runs the in-sample research pipeline:
  1. EDA (exploratory data analysis for all param combos)
  2. Optional in-sample permutation (vector shuffle only)
Validation is a separate phase; run ``feature_research/validation/run_validation.py`` for that.

Usage
-----
    Linux/macOS (after ``activate``):

        source /path/to/Trading-Algo/venv/bin/activate
        python -m feature_research.in_sample.run_is

    Windows PowerShell from repo root (no PATH activation required)::

        ./.venv/Scripts/python.exe -m feature_research.in_sample.run_is

    Or from the repo root: ``python feature_research/in_sample/run_is.py``
    (the script prepends the repo root to ``sys.path`` when run by path).
    If your venv folder is named ``venv`` instead of ``.venv``, use
    ``./venv/Scripts/python.exe`` instead.

To customize tickers, dates, bias specs, or phase presets, edit
``feature_research/config.py`` for the signed-signal pipeline.
Continuous-node binning / EDA research is configured separately in
``feature_research/binning/config.py``.

To enable permutation: set ``permutation.enabled=True`` in ``feature_research/config.py``.
Signals are materialized once per combo, then Stage 1 runs ``run_vector_shuffle_test``:
each null replicate **permutes the feature vector** on the timeline (fixed target), one
combo at a time. (Passing pre-aligned batches to the orchestrator would select a different
null that permutes the target instead—avoided here.)

Optional: for faster bias-node computation (ATR, EMA, RSI, etc.), build the Cython
extensions: ``python utils/compute/cython/setup_cython.py build_ext --inplace``.
Nodes that use ``utils.compute.fast_nodes`` then use the compiled path when available.

Output
------
All reports and artifacts are written to the configured ``reports_dir`` from the
in-sample config. EDA analysis and permutation loads use ``ResearchConfig.training_window_bounds``
(OOS or validation train slice when set, else global ``start``/``end``); bias/EWSD cache population
uses the full ``start``/``end`` span so later phases need not rebuild artifacts. Param-sensitivity and permutation Power BI tables live under
``feature_research/in_sample/results/powerbi/`` (CSV tables: pooled param sensitivity, **per-ticker** param sensitivity, ``param_combo_long``, ``equity_curve.csv``; path is fixed for Power BI imports). In Power BI, set the primary key on ``param_sensitivity_by_ticker`` to ``param_sensitivity_by_ticker_key`` only—``param_combo_label`` repeats per ticker and must not be a key. When
permutation is enabled, ``permutation_summary.csv`` and ``permutation_summary.md``
are also written at the reports root.
"""
import sys
from pathlib import Path


def _prepend_repo_root_to_syspath() -> None:
    """Allow ``python path/to/run_is.py`` without PYTHONPATH (stdlib only)."""
    start = Path(__file__).resolve()
    for parent in (start.parent, *start.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            root = str(parent)
            if root not in sys.path:
                sys.path.insert(0, root)
            return
    raise RuntimeError(
        "Could not locate repository root (no pyproject.toml or .git above this file)."
    )


_prepend_repo_root_to_syspath()

from feature_research.bootstrap import ensure_repo_root_on_syspath

ensure_repo_root_on_syspath(Path(__file__).resolve())

from feature_research.config import load_config
from feature_research.research_table_exports import canonical_in_sample_power_bi_dir
from feature_research.pipeline import (
    run_eda_pipeline,
    run_permutation_pipeline,
    write_permutation_summary,
)


def main() -> None:
    config = load_config()

    print(f"\nPermutation enabled: {config.permutation.enabled}")
    print(f"Reports dir: {config.reports_dir.resolve()}")

    # Run EDA for all param combos
    print(f"\n{'*'*70}")
    print(f"PHASE 1: In-Sample EDA ({config.feature_type.name})")
    print(f"{'*'*70}")
    eda_results = run_eda_pipeline(config, config.reports_dir)

    # Run optional in-sample permutation (vector shuffle only)
    if config.permutation.enabled:
        print(f"\n{'*'*70}")
        print(
            "PHASE 2: In-Sample Permutation — vector shuffle (same expanded grid as EDA)"
        )
        print(f"{'*'*70}")
        try:
            permutation_suite, perm_param_grid = run_permutation_pipeline(
                config, config.reports_dir
            )
            csv_path, md_path = write_permutation_summary(
                permutation_suite,
                config.reports_dir,
                objective_metric=config.permutation.objective_metric,
                param_grid=perm_param_grid,
            )
            print(f"\nPermutation summary written:")
            print(f"  CSV: {csv_path.resolve()}")
            print(f"  MD:  {md_path.resolve()}")
            print(f"  Power BI folder: {canonical_in_sample_power_bi_dir().resolve()}")
        except Exception as e:
            print(f"\nPermutation pipeline failed: {e}")
            raise

    msg = f"\nIn-sample research complete. {len(eda_results)} EDA combos. Artifacts in {config.reports_dir}"
    if config.permutation.enabled:
        msg += " Permutation summary CSV/MD in reports_dir."
    print(msg)


if __name__ == "__main__":
    main()
