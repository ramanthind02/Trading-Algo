"""Parity test for the feature_research single-feature IS+OOS pipeline.

Case (pinned in ``feature_research.config.load_config()``):
    SI daily SMA(252) regime signal, signed-signal LONG_SHORT, train 2000–2018 /
    validation 2019–2022, permutation/robustness seed=42. We run the deterministic
    OOS evaluation (``run_oos_pipeline``) and snapshot the resulting
    ``WalkforwardRunReport``:

      * folds_df, fold_scores_df, selection_summary_df, portfolio_results_df,
        fold_signal_metrics_df  (the report DataFrames)
      * aggregate_oos_returns   (the OOS returns series)
      * headline metrics        (Sharpe/Sortino/maxDD/Calmar/total return from the
                                 OOS returns series, via the project metrics layer)

Regenerate:  .\\.venv\\Scripts\\python.exe -m pytest tests/parity -m regen -k feature_research
Verify:      .\\.venv\\Scripts\\python.exe -m pytest tests/parity -k feature_research
"""
from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from tests.parity import _diff
from tests.parity._metrics import headline_metrics, normalize_returns_series
from tests.parity.conftest import (
    PipelineImports,
    read_frame_snapshot,
    read_scalars_snapshot,
    read_series_snapshot,
    snapshot_exists,
    write_frame_snapshot,
    write_scalars_snapshot,
    write_series_snapshot,
)

pytestmark = pytest.mark.parity

CASE = "feature_research/oos_si_sma_regime"

# DataFrames carried by WalkforwardRunReport that are deterministic and worth
# snapshotting. Each becomes a parquet snapshot under snapshots/.
_REPORT_FRAMES = (
    "folds_df",
    "fold_scores_df",
    "selection_summary_df",
    "portfolio_results_df",
    "fold_signal_metrics_df",
)

_TOL = _diff.Tolerance()  # default rtol=1e-8, atol=1e-10


def _run_report(pipe: PipelineImports):
    """Run the OOS pipeline on a temp output_root; return the report."""
    assert pipe.feature_load_config is not None
    assert pipe.run_oos_pipeline is not None
    config = pipe.feature_load_config()
    tmp_root = Path(tempfile.mkdtemp(prefix="parity_feature_research_"))
    config = replace(
        config,
        output_root=tmp_root,
        # Pin the Norgate futures feed: this gate guards byte-identical refactor
        # parity against the pre-migration snapshot. The CFD feed is a deliberate
        # numbers change verified by the separate similarity gate, not here.
        data_feed="futures",
        # Force the vectorized lane on the OOS phase: the snapshot is the
        # frictionless OOS result. The realistic Nautilus lane is checked
        # separately. Mirrors tests/parity/test_portfolio_research_parity.py.
        realistic_phases=(),
    )
    return pipe.run_oos_pipeline(config)


def _frame_for(report: object, attr: str) -> pd.DataFrame:
    df = getattr(report, attr)
    if df is None:
        return pd.DataFrame()
    return df.reset_index(drop=False)


def _aggregate_returns(report: object) -> pd.Series:
    series = getattr(report, "aggregate_oos_returns", None)
    if series is None or len(series) == 0:
        return pd.Series(dtype=float, name="oos_return")
    return normalize_returns_series(series, name="oos_return")


@pytest.mark.regen
def test_regen_feature_research_snapshots(require_pipeline, require_data) -> None:
    """Write golden snapshots from current behaviour (``-m regen`` only)."""
    report = _run_report(require_pipeline)

    for attr in _REPORT_FRAMES:
        write_frame_snapshot(f"{CASE}__{attr}", _frame_for(report, attr))

    returns = _aggregate_returns(report)
    write_series_snapshot(f"{CASE}__aggregate_oos_returns", returns)
    write_scalars_snapshot(f"{CASE}__metrics", headline_metrics(returns))

    # Sanity: snapshots are now on disk.
    assert snapshot_exists(f"{CASE}__metrics", "scalars")


def test_feature_research_parity(require_pipeline, require_data) -> None:
    """Verify current OOS outputs match the committed golden snapshots."""
    if not snapshot_exists(f"{CASE}__metrics", "scalars"):
        pytest.skip(
            f"No golden snapshot for {CASE}. Generate it on a known-good "
            "build:  python -m pytest tests/parity -m regen -k feature_research"
        )

    report = _run_report(require_pipeline)
    results: list[_diff.DiffResult] = []

    for attr in _REPORT_FRAMES:
        expected = read_frame_snapshot(f"{CASE}__{attr}")
        actual = _frame_for(report, attr)
        results.append(_diff.diff_dataframe(f"{CASE}/{attr}", expected, actual, _TOL))

    returns = _aggregate_returns(report)
    expected_ret = read_series_snapshot(f"{CASE}__aggregate_oos_returns")
    results.append(
        _diff.diff_series(f"{CASE}/aggregate_oos_returns", expected_ret, returns, _TOL)
    )

    expected_metrics = read_scalars_snapshot(f"{CASE}__metrics")
    actual_metrics = headline_metrics(returns)
    results.append(
        _diff.diff_scalars(f"{CASE}/metrics", expected_metrics, actual_metrics, _TOL)
    )

    _diff.assert_all_pass(results)
