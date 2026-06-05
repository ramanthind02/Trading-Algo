"""Parity test for the portfolio_research portfolio_test pipeline.

Case (pinned in ``portfolio_research.config.load_config()``):
    ES/NQ/GC/CL, prop-vault ensembles (D/W/M), train 2000–2018 /
    validation 2019–2022 / test 2023–2026-05-13, target_vol=0.07,
    hierarchy_equal weight layer, max_position_pct=3.5.

We evaluate the **test** phase (fit on train+validation, score on the test window)
via ``run_single_phase_for_prop_firm(config, 'test', emit_tearsheets=False,
run_purpose='metrics_only')`` so no HTML tearsheets, prop-firm reports, or vault
writes are produced — only the structured ``PhaseResult``. From it we snapshot:

  * combined_positions        per-(ticker, datetime) position_fraction
  * combined_strategy_returns the daily strategy returns series
  * combined_baseline_returns the daily baseline (buy & hold) returns series
  * headline metrics          Sharpe/Sortino/maxDD/Calmar/total return (strategy
                              and baseline), via the project metrics layer

Regenerate:  .\\.venv\\Scripts\\python.exe -m pytest tests/parity -m regen -k portfolio_research
Verify:      .\\.venv\\Scripts\\python.exe -m pytest tests/parity -k portfolio_research

Note: the test phase runs through the latest data and includes a vault cache
preflight; expect a multi-minute first run while bias-node caches build.
"""
from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from tests.parity import _diff
from tests.parity._metrics import (
    headline_metrics,
    normalize_positions_frame,
    normalize_returns_series,
)
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

CASE = "portfolio_research/portfolio_test_default"
PHASE = "test"

_TOL = _diff.Tolerance()  # default rtol=1e-8, atol=1e-10


def _run_phase(pipe: PipelineImports):
    """Run the test phase on a temp output_root; return the PhaseResult."""
    assert pipe.portfolio_load_config is not None
    assert pipe.run_single_phase_for_prop_firm is not None
    config = pipe.portfolio_load_config()
    tmp_root = Path(tempfile.mkdtemp(prefix="parity_portfolio_research_"))
    # Redirect all artifact writes; disable optional reports to keep the run lean
    # and behaviour-neutral for the numbers we snapshot.
    config = replace(
        config,
        output_root=tmp_root,
        export_per_timeframe_tearsheets=False,
        export_per_ensemble_tearsheets=False,
        prop_firm_report=replace(config.prop_firm_report, enabled=False),
        feature_vault_correlation=replace(config.feature_vault_correlation, enabled=False),
    )
    return pipe.run_single_phase_for_prop_firm(
        config,
        PHASE,
        emit_tearsheets=False,
        run_purpose="metrics_only",
    )


def _strategy_returns(result: object) -> pd.Series:
    return normalize_returns_series(
        getattr(result, "combined_strategy_returns"), name="strategy_return"
    )


def _baseline_returns(result: object) -> pd.Series:
    return normalize_returns_series(
        getattr(result, "combined_baseline_returns"), name="baseline_return"
    )


def _positions(result: object) -> pd.DataFrame:
    return normalize_positions_frame(getattr(result, "combined_positions"))


def _all_metrics(result: object) -> dict[str, float]:
    strat = headline_metrics(_strategy_returns(result))
    base = headline_metrics(_baseline_returns(result))
    merged = {f"strategy.{k}": v for k, v in strat.items()}
    merged.update({f"baseline.{k}": v for k, v in base.items()})
    return merged


@pytest.mark.regen
def test_regen_portfolio_research_snapshots(require_pipeline, require_data) -> None:
    """Write golden snapshots from current behaviour (``-m regen`` only)."""
    result = _run_phase(require_pipeline)

    write_frame_snapshot(f"{CASE}__{PHASE}__positions", _positions(result))
    write_series_snapshot(
        f"{CASE}__{PHASE}__strategy_returns", _strategy_returns(result)
    )
    write_series_snapshot(
        f"{CASE}__{PHASE}__baseline_returns", _baseline_returns(result)
    )
    write_scalars_snapshot(f"{CASE}__{PHASE}__metrics", _all_metrics(result))

    assert snapshot_exists(f"{CASE}__{PHASE}__metrics", "scalars")


def test_portfolio_research_parity(require_pipeline, require_data) -> None:
    """Verify current test-phase outputs match the committed golden snapshots."""
    if not snapshot_exists(f"{CASE}__{PHASE}__metrics", "scalars"):
        pytest.skip(
            f"No golden snapshot for {CASE}. Generate it on a known-good "
            "build:  python -m pytest tests/parity -m regen -k portfolio_research"
        )

    result = _run_phase(require_pipeline)
    results: list[_diff.DiffResult] = []

    expected_pos = read_frame_snapshot(f"{CASE}__{PHASE}__positions")
    actual_pos = _positions(result)
    results.append(
        _diff.diff_dataframe(f"{CASE}/{PHASE}/positions", expected_pos, actual_pos, _TOL)
    )

    expected_strat = read_series_snapshot(f"{CASE}__{PHASE}__strategy_returns")
    results.append(
        _diff.diff_series(
            f"{CASE}/{PHASE}/strategy_returns",
            expected_strat,
            _strategy_returns(result),
            _TOL,
        )
    )

    expected_base = read_series_snapshot(f"{CASE}__{PHASE}__baseline_returns")
    results.append(
        _diff.diff_series(
            f"{CASE}/{PHASE}/baseline_returns",
            expected_base,
            _baseline_returns(result),
            _TOL,
        )
    )

    expected_metrics = read_scalars_snapshot(f"{CASE}__{PHASE}__metrics")
    results.append(
        _diff.diff_scalars(
            f"{CASE}/{PHASE}/metrics", expected_metrics, _all_metrics(result), _TOL
        )
    )

    _diff.assert_all_pass(results)
