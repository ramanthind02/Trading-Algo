from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest
from quantfoundry_core.robustness import ParamPerturbationSpec

from feature_research.config import FeatureType, ParamSensitivityConfig
from feature_research.pipelines.param_perturbation import (
    _merge_perturbation_param_view,
    _perturbation_param_view,
    exploration_perturbation_enabled,
    run_min_step_perturbation_pipeline,
    write_perturbation_artifacts,
)
from quantfoundry_core.robustness import build_perturbation_neighbors


@dataclass
class _Config:
    feature_type: FeatureType = FeatureType.SIGNED_SIGNAL
    timeframe: object = field(default_factory=lambda: SimpleNamespace(bars_per_year=252))
    param_sensitivity: ParamSensitivityConfig = field(
        default_factory=lambda: ParamSensitivityConfig(
            perturbation_specs={
                "momentum_lookback": ParamPerturbationSpec(min_step=1, valid_min=1),
                "rsi_period": ParamPerturbationSpec(min_step=1, valid_min=1),
                "rsi_max": ParamPerturbationSpec(min_step=1.0, valid_min=0.0, valid_max=100.0),
                "exit_bars": ParamPerturbationSpec(min_step=1, valid_min=1),
            },
        )
    )

    @property
    def bias_spec(self) -> dict[str, object]:
        return {
            "module_name": "algomatic_momentum_signal",
            "timeframes": ["D"],
            "params": {},
        }


def test_perturbation_param_view_unwraps_filter_gate_signal_params() -> None:
    gate_params = {
        "filter_module": "sma_above_filter",
        "filter_params": {"period": 252},
        "signal_module": "percent_b_signal",
        "signal_params": {"period": 10, "std_dev": 2.5},
    }
    specs = {
        "period": ParamPerturbationSpec(min_step=1, valid_min=2),
        "std_dev": ParamPerturbationSpec(min_step=0.25, valid_min=0.25),
    }
    view = _perturbation_param_view(gate_params, specs)
    assert view == {"period": 10, "std_dev": 2.5}
    neighbors = build_perturbation_neighbors(view, specs)
    merged = _merge_perturbation_param_view(gate_params, neighbors[0])
    assert merged["signal_params"]["period"] != 10
    assert merged["filter_params"]["period"] == 252


def test_run_min_step_perturbation_pipeline_inflates_robustness_flat_params(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from feature_research.binning.transforms import flatten_params_for_combo_long_table

    config = _Config(
        param_sensitivity=ParamSensitivityConfig(
            perturbation_specs={
                "rsi_period": ParamPerturbationSpec(min_step=1, valid_min=2),
                "oversold": ParamPerturbationSpec(min_step=1.0, valid_min=0.0),
            },
        )
    )
    nested = {
        "filter_module": "sma_above_filter",
        "filter_params": {"period": 252},
        "signal_module": "rsi_signal",
        "signal_params": {"rsi_period": 2, "oversold": 25.0},
        "_bias_module": "filter_gate",
    }
    flat = flatten_params_for_combo_long_table(nested)
    flat["_bias_module"] = "filter_gate"

    monkeypatch.setattr(
        "feature_research.pipelines.param_perturbation._metric_for_params",
        lambda _config, _params: 2.0,
    )

    result = run_min_step_perturbation_pipeline(
        config,
        tmp_path,
        chosen_params=flat,
    )
    assert result.result.n_perturbed == 3


def test_exploration_perturbation_enabled_requires_signed_signal_and_specs() -> None:
    enabled = _Config()
    assert exploration_perturbation_enabled(enabled)

    disabled = _Config(
        param_sensitivity=ParamSensitivityConfig(perturbation_enabled=False),
    )
    assert not exploration_perturbation_enabled(disabled)

    continuous = _Config(feature_type=FeatureType.CONTINUOUS)
    assert not exploration_perturbation_enabled(continuous)


def test_run_min_step_perturbation_pipeline_filters_specs_to_chosen_params(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Multi-module catalogs may list Turtle + Donchian axes; only winner keys perturb."""
    config = _Config(
        param_sensitivity=ParamSensitivityConfig(
            perturbation_specs={
                "lookback": ParamPerturbationSpec(min_step=1, valid_min=2),
                "entry_lookback": ParamPerturbationSpec(min_step=1, valid_min=1),
                "stop_lookback": ParamPerturbationSpec(min_step=1, valid_min=1),
            },
        )
    )
    chosen = {"lookback": 40, "_bias_module": "donchian_channel"}

    monkeypatch.setattr(
        "feature_research.pipelines.param_perturbation._metric_for_params",
        lambda _config, _params: 2.0,
    )

    result = run_min_step_perturbation_pipeline(
        config,
        tmp_path,
        chosen_params=chosen,
    )
    assert result.result.n_perturbed == 2
    neighbor_params = {run.param_changed for run in result.runs if run.role == "neighbor"}
    assert neighbor_params == {"lookback"}


def test_run_min_step_perturbation_pipeline_writes_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _Config()
    chosen = {
        "exit_bars": 5,
        "momentum_lookback": 10,
        "rsi_max": 85.0,
        "rsi_period": 2,
    }
    metric_by_combo = {
        tuple(sorted(chosen.items())): 3.0,
    }

    def _fake_metric(_config: _Config, params: dict[str, object]) -> float:
        key = tuple(sorted(params.items()))
        if key not in metric_by_combo:
            metric_by_combo[key] = 2.5
        return metric_by_combo[key]

    monkeypatch.setattr(
        "feature_research.pipelines.param_perturbation._metric_for_params",
        _fake_metric,
    )

    result = run_min_step_perturbation_pipeline(
        config,
        tmp_path,
        chosen_params=chosen,
    )

    assert result.result.n_perturbed == 8
    assert result.result.peak_metric == 3.0
    assert len(result.runs) == 9
    assert result.report_json.exists()
    assert result.runs_csv.exists()
    assert result.summary_md.exists()

    payload = json.loads(result.report_json.read_text(encoding="utf-8"))
    assert payload["stability_ratio"] == pytest.approx(result.result.stability_ratio)
    assert payload["perturbation_metric"] == "t_stat"
    assert len(payload["neighbor_runs"]) == 9

    runs_df = pd.read_csv(result.runs_csv)
    assert len(runs_df) == 9
    assert set(runs_df["role"]) == {"center", "neighbor"}


def test_write_perturbation_artifacts_round_trip(tmp_path: Path) -> None:
    from quantfoundry_core.robustness import aggregate_perturbation_results

    from feature_research.pipelines.param_perturbation import PerturbationRunRecord

    chosen = {"lookback": 10}
    result = aggregate_perturbation_results(
        chosen_params=chosen,
        chosen_metric=3.0,
        perturbed_metrics=[2.8, 2.9, 2.7, 2.85],
    )
    runs = (
        PerturbationRunRecord(
            role="center",
            param_changed="",
            direction="",
            metric=3.0,
            params=chosen,
        ),
    )
    paths = write_perturbation_artifacts(
        result,
        runs,
        perturbation_metric="t_stat",
        visualization_parent_dir=tmp_path,
    )
    assert paths["report_json"].exists()
    payload = json.loads(paths["report_json"].read_text(encoding="utf-8"))
    assert payload["passed"] == result.passed
