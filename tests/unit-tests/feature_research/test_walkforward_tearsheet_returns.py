from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.feature.inclusion_gates import write_candidate_strategy_tearsheets
from research.portfolio.pipelines.portfolio_test import PhaseResult
from research.evaluation.walkforward.tearsheet_returns import (
    load_wf_tearsheet_returns_csv,
    write_wf_tearsheet_returns_csv,
)


def test_write_and_load_wf_tearsheet_returns_csv(tmp_path: Path) -> None:
    index = pd.date_range("2020-01-01", periods=30, freq="B")
    strategy = pd.Series(np.linspace(0.001, 0.002, len(index)), index=index)
    baseline = pd.Series(np.linspace(0.0005, 0.001, len(index)), index=index)
    path = tmp_path / "returns.csv"
    write_wf_tearsheet_returns_csv(
        strategy_returns=strategy,
        baseline_returns=baseline,
        output_path=path,
    )
    loaded_strategy, loaded_baseline = load_wf_tearsheet_returns_csv(path)
    pd.testing.assert_series_equal(loaded_strategy, strategy, check_names=False)
    pd.testing.assert_series_equal(loaded_baseline, baseline, check_names=False)


def test_candidate_always_written_even_when_identical_to_sleeve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Standalone candidate tearsheets are always emitted (canonical set of 12)."""
    index_tr = pd.date_range("2000-01-03", periods=4700, freq="B")
    index_val = pd.date_range("2019-01-02", periods=750, freq="B")
    rng = np.random.default_rng(0)
    phase_tr = PhaseResult(
        name="tr",
        output_dir=tmp_path / "tr",
        combined_strategy_returns=pd.Series(
            rng.normal(0.001, 0.01, len(index_tr)), index=index_tr
        ),
        combined_baseline_returns=pd.Series(0.0, index=index_tr),
    )
    phase_val = PhaseResult(
        name="val",
        output_dir=tmp_path / "val",
        combined_strategy_returns=pd.Series(
            rng.normal(0.001, 0.01, len(index_val)), index=index_val
        ),
        combined_baseline_returns=pd.Series(0.0, index=index_val),
    )

    written: list[str] = []
    monkeypatch.setattr(
        "research.feature.inclusion_gates.generate_tearsheet",
        lambda **kw: written.append(str(kw.get("feature_name", ""))),
    )

    paths = write_candidate_strategy_tearsheets(
        output_root=tmp_path / "portfolio_gate_tearsheets",
        phase_candidate_tr=phase_tr,
        phase_candidate_val=phase_val,
        candidate_key="cl_breakout",
    )

    assert len(paths) == 3
    assert written == [
        "Candidate strategy — train",
        "Candidate strategy — validation",
        "Candidate strategy — train+validation",
    ]
