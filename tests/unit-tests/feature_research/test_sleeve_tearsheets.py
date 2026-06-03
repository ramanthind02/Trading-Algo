from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from feature_research.inclusion_gates import write_sleeve_level_tearsheets
from portfolio_research.pipelines.portfolio_test import PhaseResult


def _phase_result(name: str, *, n: int = 120, seed: int = 0) -> PhaseResult:
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(seed)
    strategy = pd.Series(rng.normal(0.001, 0.002, size=n), index=index, dtype=float)
    baseline = pd.Series(rng.normal(0.0005, 0.001, size=n), index=index, dtype=float)
    return PhaseResult(
        name=name,
        output_dir=Path("/tmp") / name,
        combined_strategy_returns=strategy,
        combined_baseline_returns=baseline,
    )


def test_write_sleeve_level_tearsheets_full_comparison(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    written: list[str] = []

    def _fake_generate(**kwargs: object) -> None:
        written.append(str(kwargs["output_file"]))

    monkeypatch.setattr(
        "feature_research.inclusion_gates.generate_tearsheet",
        _fake_generate,
    )
    paths = write_sleeve_level_tearsheets(
        output_dir=tmp_path,
        sleeve_label="equity_indices/mean_reversion_indices",
        phase_without_tr=_phase_result("w_tr", seed=1),
        phase_without_val=_phase_result("w_val", seed=2),
        phase_with_tr=_phase_result("i_tr", seed=3),
        phase_with_val=_phase_result("i_val", seed=4),
    )
    assert len(paths) == 6
    assert (
        tmp_path / "portfolio_gate_tearsheets" / "sleeve_equity_indices__mean_reversion_indices"
    ).is_dir()
    assert len(written) == 6


def test_write_sleeve_level_tearsheets_with_only_first_in_sleeve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    written: list[str] = []

    def _fake_generate(**kwargs: object) -> None:
        written.append(str(kwargs["output_file"]))

    monkeypatch.setattr(
        "feature_research.inclusion_gates.generate_tearsheet",
        _fake_generate,
    )
    paths = write_sleeve_level_tearsheets(
        output_dir=tmp_path,
        sleeve_label="equity_indices/momentum",
        phase_with_tr=_phase_result("c_tr", seed=5),
        phase_with_val=_phase_result("c_val", seed=6),
    )
    assert len(paths) == 3
    assert (
        tmp_path / "portfolio_gate_tearsheets" / "sleeve_equity_indices__momentum" / "with_candidate"
    ).is_dir()


def test_write_sleeve_level_tearsheets_skips_flat_without(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    written: list[str] = []

    def _fake_generate(**kwargs: object) -> None:
        written.append(str(kwargs["output_file"]))

    monkeypatch.setattr(
        "feature_research.inclusion_gates.generate_tearsheet",
        _fake_generate,
    )
    base_flat = _phase_result("flat")
    flat = replace(
        base_flat,
        combined_strategy_returns=pd.Series(
            0.0,
            index=base_flat.combined_strategy_returns.index,
            dtype=float,
        ),
    )
    paths = write_sleeve_level_tearsheets(
        output_dir=tmp_path,
        sleeve_label="commodities/crude_oil_mr",
        phase_without_tr=flat,
        phase_without_val=flat,
        phase_with_tr=_phase_result("with_tr", seed=7),
        phase_with_val=_phase_result("with_val", seed=8),
    )
    assert len(paths) == 3
    assert all("with_candidate" in str(p) for p in paths)
    assert len(written) == 3
