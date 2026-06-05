from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.feature.pipelines.permutation import write_permutation_summary
from features.validation.objective_metrics import ObjectiveMetricSpec


@dataclass(frozen=True)
class _Combo:
    label: str | None
    params: dict[str, object]


@dataclass(frozen=True)
class _Config:
    n_permutations: int


@dataclass(frozen=True)
class _FullGrid:
    n_combinations: int
    observed_score: float
    p_value: float
    observed_best_combination: _Combo
    config: _Config


def test_write_permutation_summary_includes_full_grid_section(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    suite = SimpleNamespace(
        feature_name="signal",
        feature_type="signed_signal",
        funnel_stats=SimpleNamespace(total_params=3, stage1_pass=3),
        stage1_reports={},
    )
    monkeypatch.setattr(
        "research.feature.pipelines.permutation.write_permutation_vector_shuffle_exports",
        lambda *_args, **_kwargs: {
            "permutation_vector_shuffle_csv": tmp_path / "permutation_vector_shuffle.csv"
        },
    )
    monkeypatch.setattr(
        "research.feature.pipelines.permutation.permutation_vector_shuffle_records",
        lambda *_args, **_kwargs: [
            {
                "param_combo": "period_126",
                "param_combo_label": "period_126",
                "observed_metric": 1.0,
                "p_value": 0.05,
                "passed": True,
                "alpha": 0.1,
                "n_reps": 100,
            }
        ],
    )
    full_grid = _FullGrid(
        n_combinations=3,
        observed_score=4.58,
        p_value=0.02,
        observed_best_combination=_Combo(label="period_378", params={"period": 378}),
        config=_Config(n_permutations=100),
    )

    _, md_path = write_permutation_summary(
        suite,
        tmp_path,
        objective_metric=ObjectiveMetricSpec(builtin="t_stat"),
        param_grid=[{"period": 126}, {"period": 252}, {"period": 378}],
        full_grid_permutation=full_grid,  # type: ignore[arg-type]
        selection_metric_label="newey_west_t_stat",
    )

    text = md_path.read_text(encoding="utf-8")
    assert "## Full-grid search-bias permutation" in text
    assert "period_378" in text
    assert "0.0200" in text
    assert "## Vector shuffle (per combo)" in text
