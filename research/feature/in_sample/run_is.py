"""Unified in-sample research entry point for the signed-signal trading pipeline.

This single script runs the exploration phase in canonical order:
  1. EDA (parameter sweep for all param combos)
  2. In-sample robustness (when ``config.robustness.enabled``)
  3. Vector-shuffle permutation (when ``config.permutation.enabled`` and
     ``config.permutation.run_vector_shuffle``)
Validation is a separate phase; run ``feature_research/validation/run_validation.py`` for that.

Usage
-----
    Linux/macOS (after ``activate``):

        source /path/to/Trading-Algo/venv/bin/activate
        python -m feature_research exploration

    Windows PowerShell from repo root (no PATH activation required)::

        ./.venv/Scripts/python.exe -m feature_research exploration

    Or from the repo root: ``python feature_research/in_sample/run_is.py``
    (the script prepends the repo root to ``sys.path`` when run by path).
    If your venv folder is named ``venv`` instead of ``.venv``, use
    ``./venv/Scripts/python.exe`` instead.

To customize tickers, dates, bias specs, or phase presets, edit
``feature_research/config.py`` for the signed-signal pipeline.
Continuous-node binning / EDA research is configured separately in
``feature_research/binning/config.py``.

Automatic robustness checks are driven by ``config.robustness``. Vector-shuffle
permutation is driven by ``config.permutation`` (see
``docs/library/Feature_selection/permutation_testing.md``).

Optional: for faster bias-node computation (ATR, EMA, RSI, etc.), build the Cython
extensions: ``python utils/compute/cython/setup_cython.py build_ext --inplace``.
Nodes that use ``utils.compute.fast_nodes`` then use the compiled path when available.

Output
------
All reports and artifacts are written to the configured ``reports_dir`` from the
in-sample config. EDA analysis and robustness loads use ``ResearchConfig.training_window_bounds``
(OOS or validation train slice when set, else global ``start``/``end``); bias/EWSD cache population
uses the full common OHLC history in ``data/ohlc_data`` (not the analysis window) so indicator
warmup happens once at data inception and later phases need not rebuild artifacts.
Param-sensitivity and visualization CSVs live under
``feature_research/in_sample/results/visualization/`` (CSV tables: pooled param sensitivity,
**per-ticker** param sensitivity, ``param_combo_long``, ``equity_curve.csv`` with per-ticker rows
plus one ``ALL`` combined row per param combo). When
robustness is enabled, ``robustness_summary.csv``, ``robustness_summary.md``,
``robustness_combinations.csv``, and ``robustness_report.json`` are also written at the
reports root. When permutation is enabled, ``permutation_summary.csv``, ``permutation_summary.md``,
and ``visualization/permutation_vector_shuffle.csv`` are written as documented in
``docs/library/Feature_selection/permutation_testing.md``.
"""
from pathlib import Path

from lib.core.repo_bootstrap import ensure_repo_root_on_syspath

ensure_repo_root_on_syspath(Path(__file__).resolve())

from lib.core.research_feed import set_research_feed
from research.feature.config import load_config
from research.feature.exploration import execute_exploration_phase, exploration_permutation_enabled
from research.feature.research_table_exports import canonical_in_sample_visualization_dir


def main() -> None:
    config = load_config()

    # Exploration stays frictionless/vectorized (no cost lane). It may use the
    # longer FUTURES history for stock-index tickers via the per-ticker override
    # while everything else uses the configured feed (CFD by default).
    set_research_feed(
        config.data_feed,
        futures_tickers=config.exploration_futures_index_tickers,
    )

    print(f"\nRobustness enabled: {config.robustness.enabled}")
    print(f"Full-grid permutation: {config.robustness.run_full_grid_permutation}")
    print(f"Vector-shuffle permutation: {exploration_permutation_enabled(config)}")
    print(f"Reports dir: {config.reports_dir.resolve()}")

    print(f"\n{'*'*70}")
    print(f"EXPLORATION ({config.feature_type.name})")
    print(f"{'*'*70}")

    result = execute_exploration_phase(config, config.reports_dir)

    print(f"\nExploration complete. {len(result.eda_results)} EDA combo(s).")
    print(f"Artifacts in {config.reports_dir.resolve()}")
    if result.pass1_binning is not None:
        decile_chart = (
            canonical_in_sample_visualization_dir() / "matplotlib" / "atr_pct_decile_chart.png"
        )
        print(f"  Pass 1 ATR% decile chart: {decile_chart.resolve()}")
    if result.robustness_artifacts is not None:
        print(f"  Robustness CSV:  {result.robustness_artifacts['summary_csv'].resolve()}")
        print(f"  Robustness MD:   {result.robustness_artifacts['markdown'].resolve()}")
        print(f"  Robustness JSON: {result.robustness_artifacts['json'].resolve()}")
    if result.permutation_summary_csv is not None:
        print(f"  Permutation CSV: {result.permutation_summary_csv.resolve()}")
        print(f"  Permutation MD:  {result.permutation_summary_md.resolve()}")
    print(
        f"  Visualization CSV folder: {canonical_in_sample_visualization_dir().resolve()}"
    )
    if result.matplotlib_plots:
        print(
            f"  Matplotlib PNGs ({len(result.matplotlib_plots)}): "
            f"{result.matplotlib_plots[0].parent.resolve()}"
        )


if __name__ == "__main__":
    main()
