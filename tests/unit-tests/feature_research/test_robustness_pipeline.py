from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from math import isfinite
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from research.feature.config import ResearchWindowConfig, load_config
from research.feature.core_helpers import combo_key
from research.feature.pipelines.permutation import _filter_param_grid_to_selected_combo
from research.feature.pipelines.robustness import (
    _PreparedRobustnessCombo,
    _build_core_permutation_grid,
    _build_metric_adapter,
    _core_json_mapping,
    _plain_grid_score,
    _resolve_full_grid_permutation_metric,
    _selection_score,
    run_robustness_pipeline,
    write_robustness_summary,
)
from features.validation.objective_metrics import ObjectiveMetricSpec
from features.validation.stability_analysis import _param_combo_name
from research.feature.in_sample.data_loader import enrich_param_combo_with_module
from lib.core.enums import Direction


def _build_combo_frame(
    index: pd.DatetimeIndex,
    target: pd.Series,
    signal_level: float,
    *,
    signal_name: str,
) -> pd.DataFrame:
    signal = pd.Series(signal_level, index=index, name=signal_name, dtype=float)
    returns = signal.mul(target).rename("returns")
    return pd.DataFrame(
        {
            "signal": signal,
            "target": target,
            "returns": returns,
        }
    )


def test_run_robustness_pipeline_builds_core_report(monkeypatch, tmp_path: Path) -> None:
    index = pd.date_range("2020-01-01", periods=12, freq="D")
    target = pd.Series(
        [0.02, 0.01, -0.01, 0.03, 0.02, -0.01, 0.04, 0.01, 0.02, -0.01, 0.03, 0.02],
        index=index,
        name="target",
        dtype=float,
    )
    combo_params = [{"lookback": 2}, {"lookback": 8}]
    signal_frames = {
        combo_key(params): _build_combo_frame(
            index,
            target,
            1.0 if params["lookback"] == 2 else 0.25,
            signal_name=f"sig_{params['lookback']}",
        )
        for params in combo_params
    }

    monkeypatch.setattr(
        "research.feature.pipelines.robustness.populate_cache_if_needed",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "research.feature.pipelines.robustness.expand_bias_specs",
        lambda _bias_spec: [
            {"module_name": "rsisignal", "timeframes": [], "params": params}
            for params in combo_params
        ],
    )
    monkeypatch.setattr(
        "research.feature.pipelines.robustness.load_signed_signal_research_data",
        lambda _config, _expanded, print_loaded=False: SimpleNamespace(
            combo_signal_target=signal_frames,
            successful_param_grid=combo_params,
            reference_index=index,
            reference_target_series=target,
        ),
    )

    base = load_config()
    config = replace(
        base,
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 12),
        research_window=ResearchWindowConfig(
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2020, 1, 12),
            val_start=datetime(2020, 1, 13),
            val_end=datetime(2020, 1, 20),
        ),
        robustness=replace(
            base.robustness,
            enabled=True,
            run_full_grid_permutation=False,
            run_individual_permutation=False,
            rolling_window=4,
            n_eff_min_overlap=2,
        ),
    )

    report, grid = run_robustness_pipeline(config, tmp_path)

    assert grid == combo_params
    assert report.n_combinations == 2
    assert report.best_param_combo == "lookback_2"
    assert report.best_param_combo_label == "lb2"
    assert report.best_combination.label == "lookback_2"
    assert report.full_grid_permutation is None
    assert report.individual_permutation is None
    assert report.n_effective.n_combinations == 2
    assert len(report.combo_scores) == 2
    assert sum(int(combo.is_best) for combo in report.combo_scores) == 1
    assert "DSR=" in report.interpretation
    assert report.stability_chart.window <= 126
    assert len(report.stability_chart.cusum_series) == len(index)


def test_run_robustness_pipeline_writes_artifacts(monkeypatch, tmp_path: Path) -> None:
    index = pd.date_range("2021-01-01", periods=10, freq="D")
    target = pd.Series(
        [0.01, 0.02, -0.01, 0.02, 0.01, -0.01, 0.03, 0.01, 0.02, 0.01],
        index=index,
        name="target",
        dtype=float,
    )
    combo_params = [{"lookback": 3}, {"lookback": 5}]
    signal_frames = {
        combo_key(params): _build_combo_frame(
            index,
            target,
            1.0 if params["lookback"] == 3 else 0.5,
            signal_name=f"sig_{params['lookback']}",
        )
        for params in combo_params
    }

    monkeypatch.setattr(
        "research.feature.pipelines.robustness.populate_cache_if_needed",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "research.feature.pipelines.robustness.expand_bias_specs",
        lambda _bias_spec: [
            {"module_name": "rsisignal", "timeframes": [], "params": params}
            for params in combo_params
        ],
    )
    monkeypatch.setattr(
        "research.feature.pipelines.robustness.load_signed_signal_research_data",
        lambda _config, _expanded, print_loaded=False: SimpleNamespace(
            combo_signal_target=signal_frames,
            successful_param_grid=combo_params,
            reference_index=index,
            reference_target_series=target,
        ),
    )

    base = load_config()
    config = replace(
        base,
        start=datetime(2021, 1, 1),
        end=datetime(2021, 1, 10),
        research_window=ResearchWindowConfig(
            train_start=datetime(2021, 1, 1),
            train_end=datetime(2021, 1, 10),
            val_start=datetime(2021, 1, 11),
            val_end=datetime(2021, 1, 20),
        ),
        robustness=replace(
            base.robustness,
            enabled=True,
            n_permutations=5,
            random_seed=7,
            run_full_grid_permutation=True,
            run_individual_permutation=True,
            rolling_window=4,
            n_eff_min_overlap=2,
        ),
    )

    report, _ = run_robustness_pipeline(config, tmp_path)
    paths = write_robustness_summary(report, tmp_path)

    assert report.full_grid_permutation is not None
    assert report.individual_permutation is not None
    assert report.full_grid_permutation.metric.metric_name == "sharpe_annualized"
    assert report.individual_permutation.metric.metric_name == "sharpe_annualized"
    assert paths["summary_csv"].exists()
    assert paths["combos_csv"].exists()
    assert paths["json"].exists()
    assert paths["markdown"].exists()
    assert paths["rolling_cusum_csv"].exists()
    assert paths["cusum_plot_png"].exists()

    summary_df = pd.read_csv(paths["summary_csv"])
    combos_df = pd.read_csv(paths["combos_csv"])

    assert "dsr_probability" in summary_df.columns
    assert "full_grid_p_value" in summary_df.columns
    assert "param_combo_label" in combos_df.columns
    assert combos_df["is_best"].sum() == 1


def test_run_robustness_pipeline_normalizes_enum_params_for_core(
    monkeypatch,
    tmp_path: Path,
) -> None:
    index = pd.date_range("2022-01-01", periods=8, freq="D")
    target = pd.Series(
        [0.01, 0.02, -0.01, 0.03, 0.01, -0.02, 0.02, 0.01],
        index=index,
        name="target",
        dtype=float,
    )
    combo_params = [
        {"lookback": 2, "direction": Direction.LONG_SHORT},
        {"lookback": 4, "direction": Direction.LONG_SHORT},
    ]
    signal_frames = {
        combo_key(params): _build_combo_frame(
            index,
            target,
            1.0 if params["lookback"] == 2 else 0.5,
            signal_name=f"sig_{params['lookback']}",
        )
        for params in combo_params
    }

    monkeypatch.setattr(
        "research.feature.pipelines.robustness.populate_cache_if_needed",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "research.feature.pipelines.robustness.expand_bias_specs",
        lambda _bias_spec: [
            {"module_name": "rsisignal", "timeframes": [], "params": params}
            for params in combo_params
        ],
    )
    monkeypatch.setattr(
        "research.feature.pipelines.robustness.load_signed_signal_research_data",
        lambda _config, _expanded, print_loaded=False: SimpleNamespace(
            combo_signal_target=signal_frames,
            successful_param_grid=combo_params,
            reference_index=index,
            reference_target_series=target,
        ),
    )

    base = load_config()
    config = replace(
        base,
        start=datetime(2022, 1, 1),
        end=datetime(2022, 1, 8),
        research_window=ResearchWindowConfig(
            train_start=datetime(2022, 1, 1),
            train_end=datetime(2022, 1, 8),
            val_start=datetime(2022, 1, 9),
            val_end=datetime(2022, 1, 16),
        ),
        robustness=replace(
            base.robustness,
            enabled=True,
            run_full_grid_permutation=False,
            run_individual_permutation=False,
            rolling_window=4,
            n_eff_min_overlap=2,
        ),
    )

    report, _ = run_robustness_pipeline(config, tmp_path)

    assert report.best_combination.params["direction"] == Direction.LONG_SHORT.value


def test_resolve_full_grid_permutation_metric_defaults_to_plain_annualized_sharpe() -> None:
    config = load_config()
    metric = _resolve_full_grid_permutation_metric(config, periods_per_year=252)
    assert metric.builtin == "sharpe"
    assert metric.kwargs["annualization_factor"] == 252.0


def test_plain_grid_score_skips_newey_west_for_t_stat() -> None:
    import numpy as np

    noise = np.random.default_rng(7).normal(0.0, 0.01, 320)
    ar = np.zeros(320, dtype=float)
    ar[0] = noise[0]
    ar[1:] = 0.75 * ar[:-1] + noise[1:]
    returns = pd.Series(ar)
    metric = ObjectiveMetricSpec(builtin="t_stat")
    plain = _plain_grid_score(returns, metric)
    nw = _selection_score(returns, metric)
    assert plain != nw
    assert abs(plain) > abs(nw)


def test_build_metric_adapter_uses_plain_metric_for_grid_permutation() -> None:
    from research.feature.pipelines.robustness import _PreparedRobustnessCombo
    from quantfoundry_core.robustness import PermutationGrid

    index = pd.date_range("2020-01-01", periods=8, freq="D")
    target = pd.Series([0.01, 0.02, -0.01, 0.02, 0.01, -0.01, 0.03, 0.01], index=index)
    combo = _PreparedRobustnessCombo(
        param_combo="lookback_2",
        param_combo_label="lookback_2",
        params={"lookback": 2},
        signal=pd.Series(1.0, index=index, name="signal"),
        returns=target.rename("returns"),
    )
    grid = PermutationGrid.from_params_list([{"lookback": 2}])
    metric = ObjectiveMetricSpec(builtin="t_stat")
    nw_adapter = _build_metric_adapter((combo,), metric, nw_adjusted=True)
    plain_adapter = _build_metric_adapter((combo,), metric, nw_adjusted=False)
    assert nw_adapter.spec.metric_name == "newey_west_t_stat"
    assert plain_adapter.spec.metric_name == "t_stat"
    observed_nw = nw_adapter.score_grid(target, grid)[0]
    observed_plain = plain_adapter.score_grid(target, grid)[0]
    assert observed_plain != observed_nw


def test_core_json_mapping_flattens_filter_gate_nested_params() -> None:
    params = {
        "filter_module": "atr_percentile_filter",
        "filter_params": {
            "atr_period": 14,
            "lookback": 252,
            "max_rank_fraction": 0.3,
            "percentile_tail": "high",
        },
        "signal_module": "cumulative_rsi_signal",
        "signal_params": {"lookback": 3, "avg_period": 3},
    }
    flat = _core_json_mapping(params)
    assert "filter_params" not in flat
    assert "signal_params" not in flat
    assert flat["f_atr_period"] == 14
    assert flat["s_lookback"] == 3

    index = pd.date_range("2020-01-01", periods=8, freq="D")
    combo = _PreparedRobustnessCombo(
        param_combo="gate_combo",
        param_combo_label="gate_combo",
        params=params,
        signal=pd.Series(1.0, index=index, name="signal"),
        returns=pd.Series(0.01, index=index, name="returns"),
    )
    grid = _build_core_permutation_grid([combo])
    assert len(grid) == 1

    metric = ObjectiveMetricSpec(builtin="t_stat")
    adapter = _build_metric_adapter((combo,), metric, nw_adjusted=True)
    target = pd.Series(0.01, index=index, name="target")
    scores = adapter.score_grid(target, grid)
    assert len(scores) == 1
    assert isfinite(scores[0])


def test_filter_gate_variants_disambiguated_for_permutation_selection() -> None:
    shared = {
        "filter_module": "sma_above_filter",
        "filter_params": {"period": 252},
        "signal_module": "cumulative_rsi_signal",
        "signal_params": {"lookback": 3},
    }
    entry_only = enrich_param_combo_with_module(shared, "filter_gate_entry_only")
    full_gate = enrich_param_combo_with_module(shared, "filter_gate")
    grid = [entry_only, full_gate]
    winner = _param_combo_name(entry_only)
    matched = _filter_param_grid_to_selected_combo(grid, selected_combo_name=winner)
    assert matched == [entry_only]
    assert _param_combo_name(full_gate) != winner
