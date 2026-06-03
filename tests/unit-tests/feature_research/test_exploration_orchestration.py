from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from feature_research.config import FeatureType, ParamSensitivityConfig
from feature_research.exploration.filter_gate_catalog import ExplorationFilterGatesConfig
from feature_research.exploration.orchestrate import (
    execute_exploration_phase,
    exploration_permutation_enabled,
)


@dataclass
class _PermutationConfig:
    enabled: bool = False
    run_vector_shuffle: bool = True
    objective_metric: object = field(default_factory=object)


@dataclass
class _RobustnessConfig:
    enabled: bool = False


@dataclass
class _Config:
    robustness: _RobustnessConfig
    permutation: _PermutationConfig
    feature_type: FeatureType = FeatureType.SIGNED_SIGNAL
    param_sensitivity: ParamSensitivityConfig = field(
        default_factory=lambda: ParamSensitivityConfig(perturbation_enabled=False)
    )
    exploration_filter_gates: ExplorationFilterGatesConfig = field(
        default_factory=lambda: ExplorationFilterGatesConfig(enabled=False)
    )


@pytest.fixture(autouse=True)
def _disable_pass1_vol_regime_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "feature_research.exploration.orchestrate.exploration_includes_filter_gate",
        lambda _config: False,
    )
    monkeypatch.setattr(
        "feature_research.exploration.orchestrate.ensure_phase_visualization_reports",
        lambda *_args, **_kwargs: [],
    )


def test_exploration_permutation_enabled_requires_both_flags() -> None:
    assert exploration_permutation_enabled(
        _Config(_RobustnessConfig(), _PermutationConfig(enabled=True, run_vector_shuffle=True))
    )
    assert not exploration_permutation_enabled(
        _Config(_RobustnessConfig(), _PermutationConfig(enabled=False, run_vector_shuffle=True))
    )
    assert not exploration_permutation_enabled(
        _Config(_RobustnessConfig(), _PermutationConfig(enabled=True, run_vector_shuffle=False))
    )


def test_execute_exploration_phase_runs_eda_only_when_optional_steps_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _Config(_RobustnessConfig(enabled=False), _PermutationConfig(enabled=False))
    eda_paths = {"combo_a": tmp_path / "combo_a"}
    call_order: list[str] = []

    monkeypatch.setattr(
        "feature_research.exploration.orchestrate._run_eda_pipeline",
        lambda _config, output_dir: (
            call_order.append("eda") or eda_paths
            if output_dir == tmp_path
            else {}
        ),
    )
    monkeypatch.setattr(
        "feature_research.exploration.orchestrate._run_robustness_pipeline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("robustness")),
    )
    monkeypatch.setattr(
        "feature_research.exploration.orchestrate._run_permutation_pipeline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("permutation")),
    )

    result = execute_exploration_phase(config, tmp_path)

    assert call_order == ["eda"]
    assert result.pass1_binning is None
    assert result.eda_results == eda_paths
    assert result.robustness_artifacts is None
    assert result.permutation_summary_csv is None


def test_execute_exploration_phase_runs_pass1_when_auto_filter_gates_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from datetime import datetime
    from pathlib import Path as PathType

    from feature_research.config import (
        BinningAnalysisConfig,
        ExplorationFilterGatesConfig,
        FeatureType,
        InSampleDefaultsCatalog,
        InSamplePhaseDefaultsConfig,
        PermutationResearchConfig,
        ResearchConfig,
    )
    from feature_selection.validation.objective_metrics import ObjectiveMetricSpec
    from utils.core.enums import Direction, Ticker, TimeFrame

    spec = {
        "module_name": "donchian_long_only",
        "timeframes": [TimeFrame.D],
        "params": {"entry_lookback": 20, "exit_lookback": 10, "sma_period": 0},
    }
    in_sample = InSampleDefaultsCatalog(
        continuous=InSamplePhaseDefaultsConfig(
            bias_spec=spec,
            target_col="log_return_ewsd",
            strategy=Direction.LONG,
            reports_dir=PathType("feature_research/in_sample/results/signed_signal/test"),
        ),
        signed_signal=InSamplePhaseDefaultsConfig(
            bias_spec=spec,
            target_col="log_return_ewsd",
            strategy=Direction.LONG,
            reports_dir=PathType("feature_research/in_sample/results/signed_signal/test"),
        ),
    )
    config = ResearchConfig(
        tickers=[Ticker.GC],
        start=datetime(2000, 1, 1),
        end=datetime(2018, 12, 31),
        permutation=PermutationResearchConfig(
            objective_metric=ObjectiveMetricSpec(builtin="t_stat"),
            enabled=False,
        ),
        objective_metric_presets={},
        binning_params=BinningAnalysisConfig(),
        in_sample_defaults=in_sample,
        feature_type=FeatureType.SIGNED_SIGNAL,
        exploration_filter_gates=ExplorationFilterGatesConfig(),
    )
    call_order: list[str] = []
    pass1_summary = {"n_bins": 10}

    monkeypatch.setattr(
        "feature_research.exploration.orchestrate.exploration_includes_filter_gate",
        lambda _config: True,
    )
    monkeypatch.setattr(
        "feature_research.exploration.orchestrate.run_pass1_vol_regime_binning",
        lambda _config: call_order.append("pass1") or pass1_summary,
    )
    monkeypatch.setattr(
        "feature_research.exploration.orchestrate._run_eda_pipeline",
        lambda _config, output_dir: call_order.append("eda") or {},
    )

    result = execute_exploration_phase(config, tmp_path)

    assert call_order == ["pass1", "eda"]
    assert result.pass1_binning == pass1_summary


def test_execute_exploration_phase_runs_pass1_before_eda_for_filter_exploration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _Config(_RobustnessConfig(enabled=False), _PermutationConfig(enabled=False))
    call_order: list[str] = []
    pass1_summary = {"n_bins": 10, "feature": "atrPct"}

    monkeypatch.setattr(
        "feature_research.exploration.orchestrate.exploration_includes_filter_gate",
        lambda _config: True,
    )
    monkeypatch.setattr(
        "feature_research.exploration.orchestrate.run_pass1_vol_regime_binning",
        lambda _config: call_order.append("pass1") or pass1_summary,
    )
    monkeypatch.setattr(
        "feature_research.exploration.orchestrate._run_eda_pipeline",
        lambda _config, output_dir: call_order.append("eda") or {"combo": output_dir},
    )

    result = execute_exploration_phase(config, tmp_path)

    assert call_order == ["pass1", "eda"]
    assert result.pass1_binning == pass1_summary


def test_execute_exploration_phase_runs_robustness_before_vector_shuffle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Robustness runs before vector shuffle on the selected combo."""

    config = _Config(_RobustnessConfig(enabled=True), _PermutationConfig(enabled=True))
    call_order: list[str] = []

    monkeypatch.setattr(
        "feature_research.exploration.orchestrate._run_eda_pipeline",
        lambda _config, output_dir: call_order.append("eda") or {"combo": output_dir},
    )
    robustness_report = SimpleNamespace(
        full_grid_permutation=None,
        selection_metric="t_stat",
        best_param_combo="period_126",
    )
    monkeypatch.setattr(
        "feature_research.exploration.orchestrate._run_robustness_pipeline",
        lambda _config, output_dir: (
            call_order.append("robustness") or (robustness_report, [])
        ),
    )
    monkeypatch.setattr(
        "feature_research.exploration.orchestrate._write_robustness_summary",
        lambda _report, output_dir: (
            call_order.append("robustness_write")
            or {"summary_csv": output_dir / "robustness_summary.csv"}
        ),
    )
    monkeypatch.setattr(
        "feature_research.exploration.orchestrate._run_permutation_pipeline",
        lambda _config, output_dir, *, selected_combo_name=None: (
            call_order.append("permutation") or (MagicMock(), [])
            if selected_combo_name == "period_126"
            else (_ for _ in ()).throw(AssertionError("selected_combo_name"))
        ),
    )
    monkeypatch.setattr(
        "feature_research.exploration.orchestrate._write_permutation_summary",
        lambda *_args, **_kwargs: (
            call_order.append("permutation_write")
            or (tmp_path / "permutation_summary.csv", tmp_path / "permutation_summary.md")
        ),
    )

    execute_exploration_phase(config, tmp_path)

    assert call_order == [
        "eda",
        "robustness",
        "robustness_write",
        "permutation",
        "permutation_write",
    ]


def test_execute_exploration_phase_runs_perturbation_between_robustness_and_permutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from quantfoundry_core.robustness import ParamPerturbationSpec

    config = _Config(
        _RobustnessConfig(enabled=True),
        _PermutationConfig(enabled=True),
        param_sensitivity=ParamSensitivityConfig(
            perturbation_enabled=True,
            perturbation_specs={"period": ParamPerturbationSpec(min_step=1, valid_min=1)},
        ),
    )
    call_order: list[str] = []
    robustness_report = SimpleNamespace(
        full_grid_permutation=None,
        selection_metric="t_stat",
        best_param_combo="period_126",
        best_combination=SimpleNamespace(params={"period": 126}),
    )
    perturbation_result = SimpleNamespace(
        report_json=tmp_path / "perturbation_report.json",
        result=SimpleNamespace(stability_ratio=0.95, passed=True),
    )

    monkeypatch.setattr(
        "feature_research.exploration.orchestrate._run_eda_pipeline",
        lambda _config, output_dir: call_order.append("eda") or {"combo": output_dir},
    )
    monkeypatch.setattr(
        "feature_research.exploration.orchestrate._run_robustness_pipeline",
        lambda _config, output_dir: call_order.append("robustness") or (robustness_report, []),
    )
    monkeypatch.setattr(
        "feature_research.exploration.orchestrate._write_robustness_summary",
        lambda _report, output_dir: call_order.append("robustness_write")
        or {"summary_csv": output_dir / "robustness_summary.csv"},
    )
    monkeypatch.setattr(
        "feature_research.exploration.orchestrate.run_min_step_perturbation_pipeline",
        lambda _config, output_dir, *, chosen_params: (
            call_order.append("perturbation") or perturbation_result
        ),
    )
    monkeypatch.setattr(
        "feature_research.exploration.orchestrate._run_permutation_pipeline",
        lambda _config, output_dir, *, selected_combo_name=None: (
            call_order.append("permutation") or (MagicMock(), [])
        ),
    )
    monkeypatch.setattr(
        "feature_research.exploration.orchestrate._write_permutation_summary",
        lambda *_args, **_kwargs: (
            call_order.append("permutation_write")
            or (tmp_path / "permutation_summary.csv", tmp_path / "permutation_summary.md")
        ),
    )

    result = execute_exploration_phase(config, tmp_path)

    assert call_order == [
        "eda",
        "robustness",
        "robustness_write",
        "perturbation",
        "permutation",
        "permutation_write",
    ]
    assert result.perturbation_result is perturbation_result


def test_execute_exploration_phase_runs_robustness_then_permutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _Config(_RobustnessConfig(enabled=True), _PermutationConfig(enabled=True))
    robustness_report = SimpleNamespace(
        full_grid_permutation=None,
        selection_metric="t_stat",
        best_param_combo="period_126",
    )
    permutation_suite = MagicMock()
    param_grid = [{"period": 126}]
    call_order: list[str] = []

    monkeypatch.setattr(
        "feature_research.exploration.orchestrate._run_eda_pipeline",
        lambda _config, output_dir: call_order.append("eda") or {"combo": output_dir},
    )
    monkeypatch.setattr(
        "feature_research.exploration.orchestrate._run_robustness_pipeline",
        lambda _config, output_dir: (
            call_order.append("robustness") or (robustness_report, param_grid)
        ),
    )
    monkeypatch.setattr(
        "feature_research.exploration.orchestrate._write_robustness_summary",
        lambda report, output_dir: (
            call_order.append("robustness_write")
            or {"summary_csv": output_dir / "robustness_summary.csv"}
            if report is robustness_report
            else {}
        ),
    )
    monkeypatch.setattr(
        "feature_research.exploration.orchestrate._run_permutation_pipeline",
        lambda _config, output_dir, *, selected_combo_name=None: (
            call_order.append("permutation") or (permutation_suite, param_grid)
            if selected_combo_name == "period_126"
            else (_ for _ in ()).throw(AssertionError("selected_combo_name"))
        ),
    )
    monkeypatch.setattr(
        "feature_research.exploration.orchestrate._write_permutation_summary",
        lambda suite, output_dir, *, objective_metric=None, param_grid=None, **kwargs: (
            call_order.append("permutation_write")
            or (
                output_dir / "permutation_summary.csv",
                output_dir / "permutation_summary.md",
            )
        ),
    )

    result = execute_exploration_phase(config, tmp_path)

    assert call_order == [
        "eda",
        "robustness",
        "robustness_write",
        "permutation",
        "permutation_write",
    ]
    assert result.robustness_artifacts is not None
    assert result.permutation_summary_csv == tmp_path / "permutation_summary.csv"
    assert result.permutation_summary_md == tmp_path / "permutation_summary.md"
